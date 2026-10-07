"""Check the KVzip port (official/compaction/compaction_methods/kvzip.py) on GPU.

The KVzip scoring functions in the port are copied verbatim from snu-mllab/KVzip (an AST
comparison confirms it). What this checks is the glue: that the hook rebuilds the right
post-RoPE queries and reads the right keys inside this repo's Qwen3.

Reference, built independently of the port: stock transformers Qwen3 with eager attention.
KVzip's score is a softmax over [sink, chunk, repeat] only; renormalising the full eager
softmax over those columns gives exactly that softmax. Then max over the query group and
over queries, as in KVzip's _get_score.

Three comparisons, each as the overlap (Jaccard) of the pairs kept by KVzip's own threshold
rule at 4x..64x, plus the largest score difference:
  port bf16 vs reference bf16   what the runs use
  port fp32 vs reference fp32   the glue check: should be ~1.0
  reference bf16 vs fp32        noise floor: how much bf16 alone moves KVzip's choice

Run from official/:  python ../experiments/verify_kvzip.py [n_docs]
"""
import gc
import sys

import torch

sys.path.insert(0, '.')
# evaluation first: importing compaction first hits a circular import inside the repo.
from evaluation.datasets import load_mixed_data  # noqa: E402
from evaluation.utils import extract_full_kv_cache, load_model_and_tokenizer  # noqa: E402
from compaction.compaction_methods.kvzip import KVzipCompaction, _KVzipScore  # noqa: E402

MODEL = 'Qwen/Qwen3-4B-Instruct-2507'
N_DOCS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
RATIOS = (4, 8, 16, 32, 64)


@torch.inference_mode()
def reference_scores(ref_model, tokenizer, full_ids, a0, a1, port):
    """KVzip scores from eager attention weights, reduced per layer inside a hook."""
    from transformers.cache_utils import DynamicCache
    n_kv = ref_model.config.num_key_value_heads
    kv = DynamicCache()
    ref_model.model(full_ids[:, :a1], past_key_values=kv, use_cache=True)

    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False, return_tensors='pt').to(full_ids.device)

    postfix_ids = encode(port._assistant_suffix(tokenizer))
    pairs = port._self_task(full_ids[:, a0:a1], encode, postfix_ids)
    span = {}
    chunk = {}

    def make_hook(i):
        def hook(module, args, kwargs, output):
            att = output[1].float()  # eager: (1, heads, q, a1 + q), causal, softmaxed
            q, start, end = span['q'], span['start'], span['end']
            cols = torch.cat([att[..., :a0], att[..., start:end], att[..., -q:]], dim=-1)
            cols = cols / cols.sum(-1, keepdim=True)
            s = cols[..., a0:a0 + (end - start)]
            chunk[i] = s.view(1, n_kv, -1, q, end - start).amax(dim=(-3, -2)).cpu()
            return output
        return hook

    layers = ref_model.model.layers
    hooks = [layers[i].self_attn.register_forward_hook(make_hook(i), with_kwargs=True)
             for i in range(len(layers))]
    out, start = None, a0
    try:
        for prefill_ids_p, repeat_ids_p in pairs:
            end = start + prefill_ids_p.shape[1]
            span.update(q=repeat_ids_p.shape[1], start=start, end=end)
            ref_model.model(repeat_ids_p, past_key_values=kv, use_cache=True)
            kv.crop(a1)
            cur = [chunk[i] for i in range(len(layers))]
            out = cur if out is None else [torch.cat([x, y], -1) for x, y in zip(out, cur)]
            start = end
    finally:
        for h in hooks:
            h.remove()
    return out


def compare(a, b):
    a = [x.float().cpu() for x in a]
    b = [x.float().cpu() for x in b]
    diff = max((x - y).abs().max().item() for x, y in zip(a, b))
    jac = {}
    for r in RATIOS:
        va, _ = _KVzipScore._threshold(None, a, 1 / r)
        vb, _ = _KVzipScore._threshold(None, b, 1 / r)
        jac[r] = (va & vb).sum().item() / max(1, (va | vb).sum().item())
    return diff, jac


def port_scores(dtype, docs):
    model, tokenizer = load_model_and_tokenizer(MODEL, device='cuda', dtype=dtype)
    model.eval()
    port = KVzipCompaction()
    out = []
    for d in docs:
        _, pkv, idx, fctx, _ = extract_full_kv_cache(model, tokenizer, d['article'], 'cuda', model_name=MODEL)
        s, _, _ = port.kvzip_scores(pkv, idx.start, idx.stop, model, tokenizer, fctx)
        ids = tokenizer(fctx, return_tensors='pt', add_special_tokens=False)['input_ids']
        out.append(dict(scores=[x.float().cpu() for x in s], a0=idx.start, a1=idx.stop, ids=ids, id=d['article_id']))
        del pkv
    del model
    gc.collect(); torch.cuda.empty_cache()
    return out, tokenizer, port


def ref_scores(dtype, docs_info, tokenizer, port):
    from transformers import AutoModelForCausalLM
    ref = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype, attn_implementation='eager',
                                               device_map='cuda').eval()
    out = [reference_scores(ref, tokenizer, d['ids'].cuda(), d['a0'], d['a1'], port) for d in docs_info]
    del ref
    gc.collect(); torch.cuda.empty_cache()
    return out


def main():
    docs = load_mixed_data('mixed_v2_4k_nologic')[:N_DOCS]
    p16, tok, port = port_scores(torch.bfloat16, docs)
    r16 = ref_scores(torch.bfloat16, p16, tok, port)
    p32, _, _ = port_scores(torch.float32, docs)
    r32 = ref_scores(torch.float32, p32, tok, port)
    worst = {}
    for name, A, B in (('port bf16 vs ref bf16', [d['scores'] for d in p16], r16),
                       ('port fp32 vs ref fp32', [d['scores'] for d in p32], r32),
                       ('ref bf16 vs ref fp32 (noise floor)', r16, r32)):
        print(f'== {name}')
        for d, a, b in zip(p16, A, B):
            diff, jac = compare(a, b)
            worst[name] = min([worst.get(name, 1.0)] + list(jac.values()))
            print(f"  {d['id']}: max |score diff| {diff:.2e} | "
                  + ' | '.join(f'{r}x {j:.4f}' for r, j in jac.items()), flush=True)
    for name, w in worst.items():
        print(f'worst overlap  {name}: {w:.4f}')
    ok = worst['port fp32 vs ref fp32'] >= 0.99
    print('glue check (fp32):', 'PASS' if ok else 'FAIL')


if __name__ == '__main__':
    main()
