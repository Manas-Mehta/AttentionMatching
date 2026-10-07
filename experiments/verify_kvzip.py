"""Check the KVzip port (official/compaction/compaction_methods/kvzip.py) on GPU.

The KVzip scoring functions in the port are copied verbatim from snu-mllab/KVzip (an AST
comparison confirms it). What this checks is the glue: that the hook rebuilds the right
post-RoPE queries and reads the right keys inside this repo's Qwen3.

Reference, built independently of the port: stock transformers Qwen3 with eager attention
and output_attentions=True. KVzip's score is a softmax over [sink, chunk, repeat] only;
renormalising the full eager softmax over those columns gives exactly that softmax. Then
max over the query group and over queries, as in KVzip's _get_score.

Reports, per document: the largest score difference, and the overlap (Jaccard) of the pairs
kept by KVzip's own threshold rule at 4x..64x. bf16 noise between sdpa and eager can flip
pairs that sit at the threshold, so overlap a little under 1.0 is expected.

Run from official/:  python ../experiments/verify_kvzip.py [n_docs]
"""
import sys

import torch

sys.path.insert(0, '.')
from compaction.compaction_methods.kvzip import KVzipCompaction, _KVzipScore  # noqa: E402
from evaluation.datasets import load_mixed_data  # noqa: E402
from evaluation.utils import extract_full_kv_cache, load_model_and_tokenizer  # noqa: E402

MODEL = 'Qwen/Qwen3-4B-Instruct-2507'
N_DOCS = int(sys.argv[1]) if len(sys.argv) > 1 else 3


@torch.inference_mode()
def reference_scores(ref_model, tokenizer, full_ids, a0, a1, port):
    from transformers.cache_utils import DynamicCache
    n_kv = ref_model.config.num_key_value_heads
    kv = DynamicCache()
    ref_model.model(full_ids[:, :a1], past_key_values=kv, use_cache=True)

    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False, return_tensors='pt').to(full_ids.device)

    postfix_ids = encode(port._assistant_suffix(tokenizer))
    pairs = port._self_task(full_ids[:, a0:a1], encode, postfix_ids)
    out, start = None, a0
    for prefill_ids_p, repeat_ids_p in pairs:
        end = start + prefill_ids_p.shape[1]
        q = repeat_ids_p.shape[1]
        o = ref_model.model(repeat_ids_p, past_key_values=kv, use_cache=True, output_attentions=True)
        kv.crop(a1)
        chunk = []
        for att in o.attentions:  # (1, heads, q, a1 + q), already causal and softmaxed
            att = att.float()
            cols = torch.cat([att[..., :a0], att[..., start:end], att[..., -q:]], dim=-1)
            cols = cols / cols.sum(-1, keepdim=True)
            s = cols[..., a0:a0 + (end - start)]
            s = s.view(1, n_kv, -1, q, end - start).amax(dim=(-3, -2))
            chunk.append(s)
        out = chunk if out is None else [torch.cat([x, y], -1) for x, y in zip(out, chunk)]
        start = end
    return out


def main():
    from transformers import AutoModelForCausalLM
    model, tokenizer = load_model_and_tokenizer(MODEL, device='cuda')
    model.eval()
    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation='eager', device_map='cuda').eval()
    port = KVzipCompaction()
    docs = load_mixed_data('mixed_v2_4k_nologic')[:N_DOCS]
    worst = 1.0
    for d in docs:
        seq_len, pkv, idx, fctx, _ = extract_full_kv_cache(model, tokenizer, d['article'], 'cuda',
                                                           model_name=MODEL)
        a0, a1 = idx.start, idx.stop
        ours, n_chunks, t = port.kvzip_scores(pkv, a0, a1, model, tokenizer, fctx)
        full_ids = tokenizer(fctx, return_tensors='pt', add_special_tokens=False)['input_ids'].cuda()
        ref = reference_scores(ref_model, tokenizer, full_ids, a0, a1, port)
        diff = max((o.float() - r).abs().max().item() for o, r in zip(ours, ref))
        line = [f"{d['article_id']}: {a1 - a0} article tokens, {n_chunks} chunks, "
                f"max |score diff| {diff:.2e}"]
        for ratio in (4, 8, 16, 32, 64):
            vo, _ = _KVzipScore._threshold(None, [o.float() for o in ours], 1 / ratio)
            vr, _ = _KVzipScore._threshold(None, ref, 1 / ratio)
            jac = (vo & vr).sum().item() / max(1, (vo | vr).sum().item())
            worst = min(worst, jac)
            line.append(f"{ratio}x overlap {jac:.4f}")
        print(' | '.join(line), flush=True)
        del pkv
        torch.cuda.empty_cache()
    print(f"worst overlap {worst:.4f}  ->  {'PASS' if worst >= 0.95 else 'CHECK'}")


if __name__ == '__main__':
    main()
