# compaction/compaction_methods/kvzip.py
"""
KVzip (Kim et al., NeurIPS 2025, arXiv 2505.23416), ported from the authors' code.

Source: github.com/snu-mllab/KVzip @ 5d84729 (MIT License, Jang-Hyun Kim).
The variant is the paper's main method: context-dependent, pair-level eviction with a
non-uniform (global) budget. The KVzip parts below are copied verbatim, with only the
`self.` state renamed:

  * chunk_fn                                         model/wrapper.py
  * the reconstruction inputs (_self_task)           model/wrapper.py  ModelKVzip.self_task
  * the scoring loop (_scoring)                      model/wrapper.py  ModelKVzip.scoring
  * _get_score, _make_mask, _mask_causal, _threshold attention/score.py KVScore

Glue that is ours: a forward hook on each attention layer stands in for KVzip's patched
attention forward. It rebuilds the post-RoPE query states exactly as
attention/attn.py:llama_qwen_attn_forward does and reads the layer's keys from the cache
after the update, then calls _get_score. Also ours: converting the kept pairs into this
repo's (C1, beta, C2) cache, with beta = 0 and the original values, and every head padded
to its layer's longest with beta = -inf, as GlobalHighestAttentionKeysCompaction does.

Mapping onto this repo:
  * KVzip's system prompt (evict_range start, also the `sink`) = every token before the
    article. Like KVzip, these are never evicted. The tokens after the article
    (the chat template's closing tags) are also kept; KVzip has none there.
  * Scoring runs on the prefill of [prefix + article] only, as KVzip's prefill is
    [system prompt + context]; the reconstruction prompt follows the article directly.
  * The assistant-turn suffix between the reconstruction prompt and the chunk is taken
    from the tokenizer's chat template, as NVIDIA KVPress (the version the LCLM paper ran)
    does. KVzip's own template table would add "<think>\n\n</think>\n\n" for any
    "qwen3-" name, which Qwen3-4B-Instruct-2507's template does not have.
  * Run with --chunking none: KVzip chunks the context itself (2,000 tokens) and selects
    globally over the whole context.
"""
import math
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .base import FullCacheCompactionAlgorithm


# ---------------------------------------------------------------------------------------
# Verbatim from snu-mllab/KVzip model/wrapper.py
# ---------------------------------------------------------------------------------------
def chunk_fn(ctx_ids: torch.Tensor, chunk_size: int) -> List[torch.Tensor]:
    """ Chunk tokens
    """
    ctx_len = ctx_ids.shape[1]
    if ctx_len > chunk_size:
        chunk_num = (ctx_len - 1) // chunk_size + 1
        print(f"chunk inputs, size: {chunk_size} (num {chunk_num})")

        input_ids = []
        for i in range(chunk_num):
            start = i * chunk_size
            end = (i + 1) * chunk_size
            a_ids = ctx_ids[:, start:end]
            if a_ids.shape[1] == 0:
                continue
            input_ids.append(a_ids)
    else:
        input_ids = [ctx_ids]

    return input_ids


class _KVzipScore:
    """ State and scoring functions of KVzip's KVScore (attention/score.py), verbatim. """

    def __init__(self, n_layers: int, n_heads_kv: int, dtype, device, sink: int):
        self.n_layers = n_layers
        self.n_heads_kv = n_heads_kv
        self.dtype = dtype
        self.device = device
        self.sink = sink  # retain initial KV pairs for system prompts
        self.start_idx, self.end_idx = None, None
        self.get_score = True
        self.causal_mask_score = None
        self.score = None

    def init_score(self):
        self.get_score = True
        self.causal_mask_score = None
        self.score = [
            torch.zeros((1, self.n_heads_kv, 0), dtype=self.dtype, device=self.device)
            for _ in range(self.n_layers)
        ]

    def _update_score(self, layer_idx: int, score: torch.Tensor):
        self.score[layer_idx] = torch.cat([self.score[layer_idx], score], dim=-1)

    def _get_score(self, query_states: torch.Tensor, key_states: torch.Tensor, layer_idx: int):
        """ Compute KV importance scores.
            # key_states: bsz x head_kv x k x dim, query_states: bsz x head x q x dim
        """

        bsz, num_heads, q_len, head_dim = query_states.shape
        num_kv = key_states.size(1)

        query_states = query_states.view(bsz, num_kv, -1, q_len, head_dim)
        key_states = torch.cat(
            [
                key_states[:, :, :self.sink],  # sink tokens (generally system prompt)
                key_states[:, :, self.start_idx:self.end_idx],  # KV chunk in the cache
                key_states[:, :, -q_len:],  # KV repeat chunk
            ],
            dim=2)

        # bsz, head, 1, dim, k
        key_states = key_states.unsqueeze(2).transpose(-2, -1).contiguous()
        ctx_len = self.end_idx - self.start_idx

        attn_weights = torch.matmul(query_states, key_states) / math.sqrt(head_dim)
        self._mask_causal(attn_weights, q_len)

        # bsz, head, group, q, ctx_len
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)  # not fp32
        attn_weights = attn_weights[..., self.sink:self.sink + ctx_len]
        score = attn_weights.amax(dim=(-3, -2))  # max over group, q

        self._update_score(layer_idx, score)

    def _make_mask(self, attn_weights: torch.Tensor, window_size: int):
        """ Define causal mask shared across layers
        """
        mask = torch.full((window_size, window_size),
                          torch.finfo(attn_weights.dtype).min,
                          device=attn_weights.device)
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        self.causal_mask_score = mask[None, None, None, :, :]

    def _mask_causal(self, attn_weights: torch.Tensor, window_size: int):
        """ Apply causal maksing
        """
        if self.causal_mask_score is None:
            self._make_mask(attn_weights, window_size)
        elif self.causal_mask_score.size(-1) != window_size:
            self._make_mask(attn_weights, window_size)

        attn_weights[..., -window_size:, -window_size:] += self.causal_mask_score

    def _threshold(self, score: Union[torch.Tensor, List[torch.Tensor]], ratio: float):
        """ Apply thresholding to KV importance scores
        """
        if type(score) == list:
            score = torch.stack(score, dim=0)
        if ratio < 1:
            score_sort = torch.sort(score.reshape(-1), descending=True).values
            n = max(int(len(score_sort) * ratio) - 1, 0)
            thres = score_sort[n].item()
            valids = torch.where(score > thres, True, False).bool()
        else:
            valids = torch.ones_like(score, dtype=bool)
            thres = 0.

        return valids, thres


# ---------------------------------------------------------------------------------------
# The compaction method
# ---------------------------------------------------------------------------------------
class KVzipCompaction(FullCacheCompactionAlgorithm):
    """KVzip context-dependent eviction: score by context reconstruction, keep the global top."""

    def __init__(self, chunk_size: int = 2000, prev_postfix_size: int = 8,
                 config_name: Optional[str] = None, **unused):
        # KVzip defaults: ModelKVzip.self_task(chunk_size=2000, prev_postfix_size=8)
        self.chunk_size = chunk_size
        self.prev_postfix_size = prev_postfix_size
        self.config_name = config_name
        if unused:
            print(f"KVzip: ignoring kwargs {sorted(unused)}")

    def name(self) -> str:
        return self.config_name or "kvzip"

    # -- KVzip model/wrapper.py ModelKVzip.self_task, verbatim except `self.encode` -------
    def _self_task(self, ctx_ids: torch.Tensor, encode, postfix_ids: torch.Tensor):
        """ Prepare chunked inputs for KV importance scoring with context reconstruction
            return: List[torch.Tensor]
        """
        chunked_inputs = chunk_fn(ctx_ids, self.chunk_size)

        input_ids = []
        for i, a_ids in enumerate(chunked_inputs):
            if i == 0:
                prompt = f"\n\nRepeat the previous context exactly."
                q_ids = encode(prompt)
            else:
                prompt = f"\n\nRepeat the part of the previous context exactly, starting with "
                q_ids = encode(prompt)
                postfix_prev = chunked_inputs[i - 1][:, -self.prev_postfix_size:]
                q_ids = torch.cat([q_ids, postfix_prev], dim=1)

            input_ids.append((a_ids, torch.cat([q_ids, postfix_ids, a_ids], dim=1)))

        return input_ids

    @staticmethod
    def _assistant_suffix(tokenizer) -> str:
        """The chat template's text between a user message and the assistant's reply
        (KVPress KVzipPress.__call__)."""
        dummy_context = "dummy context"
        separator = "\n" + "#" * len(dummy_context)
        temp_context = tokenizer.apply_chat_template(
            [{"role": "user", "content": dummy_context + separator}],
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
        _, suffix_text = temp_context.split(separator)
        return suffix_text

    @staticmethod
    def _layer_kv(cache, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if hasattr(cache, 'layers'):
            return cache.layers[layer_idx].keys, cache.layers[layer_idx].values
        if hasattr(cache, 'key_cache'):
            return cache.key_cache[layer_idx], cache.value_cache[layer_idx]
        return cache[layer_idx][0], cache[layer_idx][1]

    def kvzip_scores(self, past_key_values, article_start: int, article_end: int,
                     model: Any, tokenizer: Any, formatted_context: str):
        """KVzip importance scores of the article pairs: list over layers of
        (1, heads_kv, article_len), the `kv.score` of KVzip's ModelKVzip.scoring."""
        from transformers.cache_utils import DynamicCache
        from models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

        num_layers = len(past_key_values.layers) if hasattr(past_key_values, 'layers') \
            else len(past_key_values)
        K0, _ = self._layer_kv(past_key_values, 0)
        _, n_heads_kv, seq_len, _ = K0.shape
        device, dtype = K0.device, K0.dtype
        ctx_len = article_end - article_start

        def encode(text):
            return tokenizer.encode(text, add_special_tokens=False, return_tensors="pt").to(device)

        postfix_ids = encode(self._assistant_suffix(tokenizer))
        ctx_ids = encode(formatted_context)
        full_ids_len = ctx_ids.shape[1]
        if full_ids_len != seq_len:
            raise ValueError(f"KVzip: formatted context is {full_ids_len} tokens, cache has {seq_len}")
        ctx_ids = ctx_ids[:, article_start:article_end]

        # KVzip's prefill is [system prompt + context]: score on the cache up to the article end.
        kv = DynamicCache()
        for layer_idx in range(num_layers):
            k, v = self._layer_kv(past_key_values, layer_idx)
            kv.update(k[:, :, :article_end].clone(), v[:, :, :article_end].clone(), layer_idx)

        scorer = _KVzipScore(num_layers, n_heads_kv, dtype, device, sink=article_start)

        # Stand-in for KVzip's patched attention (attention/attn.py llama_qwen_attn_forward):
        # same post-RoPE query states; keys = the layer's cache after this step's update.
        def make_hook(layer_idx):
            def hook(module, args, kwargs, output):
                if not scorer.get_score:
                    return output
                hidden_states = kwargs['hidden_states'] if 'hidden_states' in kwargs else args[0]
                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, module.head_dim)
                query_states = module.q_norm(module.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
                cos, sin = kwargs['position_embeddings']
                query_states, _ = apply_rotary_pos_emb(query_states, query_states, cos, sin)
                key_states, _ = self._layer_kv(kwargs['past_key_values'], layer_idx)
                scorer._get_score(query_states, key_states, layer_idx)
                return output
            return hook

        t0 = time.time()
        hooks = [model.model.layers[i].self_attn.register_forward_hook(make_hook(i), with_kwargs=True)
                 for i in range(num_layers)]
        try:
            # -- KVzip model/wrapper.py ModelKVzip.scoring (load_score=False) ------------
            with torch.inference_mode():
                scorer.init_score()
                scorer.start_idx = article_start
                input_ids = self._self_task(ctx_ids, encode, postfix_ids)
                for i, (prefill_ids_p, repeat_ids_p) in enumerate(input_ids):
                    scorer.end_idx = scorer.start_idx + prefill_ids_p.shape[1]  # indices for a chunk
                    model.model(repeat_ids_p, past_key_values=kv, use_cache=True)  # get score
                    kv.crop(article_end)  # kv.slice(seen_token_prev): drop the repeat KV
                    scorer.start_idx = scorer.end_idx
                assert scorer.score[0].shape[-1] == ctx_len
                scorer.get_score = False
        finally:
            for h in hooks:
                h.remove()
        del kv
        scoring_time = time.time() - t0

        return scorer.score, len(input_ids), scoring_time

    def compact_kv_cache(
        self,
        past_key_values,
        target_size: int,
        indices: Optional[range],
        query_config: Any,
        model: Any,
        tokenizer: Any,
        formatted_context: str,
        compute_stats: bool = False,
        verbose_logging: bool = False,
        vllm_model: Optional[Any] = None,
        sliding_layer_indices: Optional[set] = None,
        past_key_values_for_queries: Optional[Any] = None,
    ) -> Tuple[Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...], Dict]:
        if indices is None:
            raise ValueError("KVzip needs the article indices (the system prompt is never evicted)")
        if sliding_layer_indices:
            raise NotImplementedError("KVzip port: sliding-window models are not supported")
        if past_key_values_for_queries is not None:
            raise NotImplementedError("KVzip port: run with --chunking none")

        num_layers = len(past_key_values) if not hasattr(past_key_values, 'layers') \
            else len(past_key_values.layers)
        K0, _ = self._layer_kv(past_key_values, 0)
        bsz, n_heads_kv, seq_len, head_dim = K0.shape
        if bsz != 1:
            raise NotImplementedError("KVzip port supports batch_size=1")
        device, dtype = K0.device, K0.dtype

        article_start, article_end = indices.start, indices.stop
        ctx_len = article_end - article_start
        num_to_keep = seq_len - ctx_len
        ratio = (target_size - num_to_keep) / ctx_len  # KVzip `ratio` = fraction of pairs kept
        print(f"KVzip: article {article_start}-{article_end} ({ctx_len} tokens), "
              f"keep ratio {ratio:.4f}, {num_to_keep} non-article tokens kept")

        scores, n_chunks, scoring_time = self.kvzip_scores(
            past_key_values, article_start, article_end, model, tokenizer, formatted_context)

        # -- KVzip EvictCache.prune(ratio, level="pair") -----------------------------------
        valid, thres = _KVzipScore._threshold(None, scores, ratio)  # (layers, 1, heads_kv, ctx_len)
        r_ = valid.float().mean().item()
        print(f"KVzip: kept {r_:.4f} of article pairs (threshold {thres:.6f}), "
              f"scoring {scoring_time:.1f}s for {n_chunks} chunk(s)")

        compacted_layers = []
        total_selected = 0
        for layer_idx in range(num_layers):
            K, V = self._layer_kv(past_key_values, layer_idx)
            C1_heads, beta_heads, C2_heads = [], [], []
            for h in range(n_heads_kv):
                sel = valid[layer_idx, 0, h].nonzero(as_tuple=True)[0] + article_start
                total_selected += sel.numel()
                C1 = torch.cat([K[0, h, :article_start], K[0, h, sel], K[0, h, article_end:]], dim=0)
                C2 = torch.cat([V[0, h, :article_start], V[0, h, sel], V[0, h, article_end:]], dim=0)
                C1_heads.append(C1)
                C2_heads.append(C2)
                beta_heads.append(C1.new_zeros(C1.shape[0]))
            L = max(c.shape[0] for c in C1_heads)
            for i in range(n_heads_kv):
                pad = L - C1_heads[i].shape[0]
                if pad:
                    C1_heads[i] = torch.cat([C1_heads[i], C1_heads[i].new_zeros(pad, head_dim)])
                    C2_heads[i] = torch.cat([C2_heads[i], C2_heads[i].new_zeros(pad, head_dim)])
                    beta_heads[i] = torch.cat([beta_heads[i], beta_heads[i].new_full((pad,), float('-inf'))])
            compacted_layers.append((torch.stack(C1_heads)[None], torch.stack(beta_heads)[None],
                                     torch.stack(C2_heads)[None]))

        effective_article_tokens = total_selected / (num_layers * n_heads_kv)
        tensor_len = sum(c[0].shape[2] for c in compacted_layers) / num_layers
        stats = {
            'is_partial_compaction': True,
            'kvzip': {'keep_ratio_requested': ratio, 'keep_ratio_actual': r_, 'threshold': thres,
                      'n_score_chunks': n_chunks, 'chunk_size': self.chunk_size,
                      'source': 'snu-mllab/KVzip@5d84729'},
            'query_generation': {'query_generation_time': scoring_time},
            'effective_article_tokens': effective_article_tokens,
            'tensor_article_tokens': tensor_len - num_to_keep,
            'tensor_compacted_seq_len': tensor_len,
            'effective_compacted_seq_len': effective_article_tokens + num_to_keep,
            'compaction_indices': {'start': article_start, 'end': article_end, 'num_positions': ctx_len},
        }
        return tuple(compacted_layers), stats
