"""Which KV pairs does KVzip keep, by part of the document? (combined task, mixed_v2_4k)

KVzip keeps (layer, KV head, token) pairs, ranked globally. A token is only fully gone when it
is dropped in all 288 (36 layers x 8 KV heads) of its pairs. Per document and ratio this
recomputes the port's own keep/drop decision (same scores, same threshold rule as
compaction_methods/kvzip.py) and reports, per token class:
  share_of_tokens   class tokens / article tokens
  share_of_budget   kept pairs on the class / all kept pairs
  kept_frac         mean share of a token's 288 pairs that are kept
  zero_frac         share of the class's tokens with no pair kept at all
  by_layer          kept_frac per layer (cwe words and filler only)

Token classes: cwe_word (the words of the numbered list), cwe_num (its numbers, dots, newlines,
header), uuid, multikey, chain (the five VAR lines), logic, filler (essay prose).

Run from official/:  python ../experiments/kvzip_retention.py [n_docs] [out.json]
"""
import json
import re
import sys

import torch

sys.path.insert(0, '.')
from evaluation.datasets import load_mixed_data  # noqa: E402
from evaluation.utils import extract_full_kv_cache, load_model_and_tokenizer  # noqa: E402
from compaction.compaction_methods.kvzip import KVzipCompaction, _KVzipScore  # noqa: E402

MODEL = 'Qwen/Qwen3-4B-Instruct-2507'
N_DOCS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
OUT = sys.argv[2] if len(sys.argv) > 2 else 'kvzip_retention.json'
RATIOS = (4, 8, 16)
CLASSES = ('cwe_word', 'cwe_num', 'uuid', 'multikey', 'chain', 'logic', 'filler')


def char_labels(row):
    """One class label per character of the document."""
    ctx, b = row['context'], row['blocks']
    lab = ['filler'] * len(ctx)

    def paint(s, e, name):
        for i in range(s, e):
            lab[i] = name

    paint(b['uuid']['char_start'], b['uuid']['char_end'], 'uuid')
    for k, v in b.items():
        if k.startswith('multikey'):
            paint(v['char_start'], v['char_end'], 'multikey')
    if 'logic' in b:
        paint(b['logic']['char_start'], b['logic']['char_end'], 'logic')
    for m in re.finditer(r'VAR \w+ = (?:VAR \w+|\d+)', ctx):
        paint(m.start(), m.end(), 'chain')
    cs, ce = b['cwe']['char_start'], b['cwe']['char_end']
    paint(cs, ce, 'cwe_num')
    for m in re.finditer(r'^\d+\. (\w+)', ctx[cs:ce], re.M):
        paint(cs + m.start(1), cs + m.end(1), 'cwe_word')
    return lab


@torch.inference_mode()
def main():
    rows = [json.loads(l) for l in open('data/mixed/mixed_v2_4k.jsonl') if l.strip()]
    docs = load_mixed_data('mixed_v2_4k_nologic')[:N_DOCS]
    model, tok = load_model_and_tokenizer(MODEL, device='cuda', dtype=torch.bfloat16)
    model.eval()
    port = KVzipCompaction()
    out = []
    for d, row in zip(docs, rows):
        assert d['article'] == row['context'], 'article text differs from the dataset context'
        _, pkv, idx, fctx, _ = extract_full_kv_cache(model, tok, d['article'], 'cuda', model_name=MODEL)
        scores, _, _ = port.kvzip_scores(pkv, idx.start, idx.stop, model, tok, fctx)
        del pkv
        a0, a1 = idx.start, idx.stop
        off = tok(fctx, return_offsets_mapping=True, add_special_tokens=False)['offset_mapping']
        base = fctx.index(row['context'])
        lab = char_labels(row)
        tlab = []
        for t in range(a0, a1):
            s, e = off[t]
            chars = [lab[c - base] for c in range(s, e) if 0 <= c - base < len(lab)]
            # a token belongs to the most specific class among its characters
            tlab.append(next((c for c in CLASSES if c in chars), 'filler'))
        T = a1 - a0
        S = torch.stack([x.float() for x in scores], 0)  # (layers, 1, heads, T)
        L, H = S.shape[0], S.shape[2]
        rec = {'doc': row['doc_index'], 'article_tokens': T, 'pairs_per_token': L * H, 'ratios': {}}
        cls_idx = {c: torch.tensor([i for i, x in enumerate(tlab) if x == c], dtype=torch.long) for c in CLASSES}
        for r in RATIOS:
            valid, _ = _KVzipScore._threshold(None, S, 1 / r)
            v = valid[:, 0].float().cpu()            # (layers, heads, T)
            per_tok = v.mean(dim=(0, 1))             # kept share of each token's pairs
            total = v.sum().item()
            res = {'kept_pairs_frac': total / (L * H * T)}
            for c, ix in cls_idx.items():
                if len(ix) == 0:
                    continue
                res[c] = {'n_tokens': len(ix), 'share_of_tokens': len(ix) / T,
                          'share_of_budget': v[:, :, ix].sum().item() / total,
                          'kept_frac': per_tok[ix].mean().item(),
                          'zero_frac': (per_tok[ix] == 0).float().mean().item()}
                if c in ('cwe_word', 'filler'):
                    res[c]['by_layer'] = v[:, :, ix].mean(dim=(1, 2)).tolist()
            rec['ratios'][str(r)] = res
        out.append(rec)
        print(f"doc {row['doc_index']}: " + ' | '.join(
            f"{r}x " + ' '.join(f"{c}={rec['ratios'][str(r)][c]['kept_frac']:.2f}/{rec['ratios'][str(r)][c]['zero_frac']:.2f}"
                                for c in CLASSES if c in rec['ratios'][str(r)]) for r in RATIOS), flush=True)
        torch.cuda.empty_cache()
    json.dump(out, open(OUT, 'w'))
    # summary over docs: kept_frac / zero_frac / share_of_budget vs share_of_tokens
    print('\nmean over docs  (kept share of pairs | tokens with nothing kept | share of budget vs share of tokens)')
    for r in RATIOS:
        for c in CLASSES:
            xs = [d['ratios'][str(r)][c] for d in out if c in d['ratios'][str(r)]]
            if not xs:
                continue
            m = lambda k: sum(x[k] for x in xs) / len(xs)
            print(f"  {r:>2}x {c:9s} kept {m('kept_frac'):.3f} | none kept {m('zero_frac'):.3f} | "
                  f"budget {m('share_of_budget'):.3f} vs tokens {m('share_of_tokens'):.3f}")
    print(f'-> {OUT}')


if __name__ == '__main__':
    main()
