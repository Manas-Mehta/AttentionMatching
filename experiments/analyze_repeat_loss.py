#!/usr/bin/env python3
"""
Analyze the repeat-prefill damage map produced by --compute-repeat-loss.

Reads the per-article npz sidecars (nll_am / nll_full / nll_none / needle_mask)
and answers the question the TODO actually asks: over the repeat-prefill target,
WHICH tokens does the AM-compressed cache lose, and is the needle one of them?

Three quantities per token i:
    delta(i)  = NLL_AM(i) - NLL_full(i)                       nats of damage
    norm(i)   = delta(i) / (NLL_none(i) - NLL_full(i))        0 = intact, 1 = as if unread
    info(i)   = NLL_none(i) - NLL_full(i)                     how much the context was worth

Articles differ in length and needle depth, so raw index-wise averaging is
meaningless. Three alignments are reported instead:
    1. needle-centred    -- profile in a +/-W window around the needle
    2. info-bucketed     -- delta as a function of how much the context was worth
                            at that token (the main hypothesis: damage should
                            concentrate on high-information tokens)
    3. normalized depth  -- delta vs relative position in the article

Usage:
    python experiments/analyze_repeat_loss.py results/repeat_loss/4k/16x
    python experiments/analyze_repeat_loss.py results/repeat_loss/4k/16x --window 150
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_cell(results_dir: Path):
    """Load every article npz under <results_dir>/repeat_loss/<method>/."""
    rl_root = results_dir / 'repeat_loss'
    if not rl_root.is_dir():
        sys.exit(f"no repeat_loss/ under {results_dir} -- was --compute-repeat-loss 1 set?")

    # layout: <results_dir>/repeat_loss/<task>/<method>/article*.npz
    cells = {}
    method_dirs = [p for p in rl_root.rglob('*') if p.is_dir() and any(p.glob('article*.npz'))]
    for method_dir in sorted(method_dirs):
        arts = []
        for f in sorted(method_dir.glob('article*.npz')):
            d = np.load(f)
            arts.append({
                'file': f,
                'am': d['nll_am'].astype(np.float64),
                'full': d['nll_full'].astype(np.float64),
                'none': d['nll_none'].astype(np.float64),
                'needle': d['needle_mask'].astype(bool),
                'article_idx': int(d['article_idx']),
            })
        if arts:
            cells[str(method_dir.relative_to(rl_root))] = arts
    if not cells:
        sys.exit(f"found {rl_root} but no article*.npz inside")
    return cells


def summarize(arts, window: int, n_buckets: int):
    out = {}

    # ---------- headline: needle vs background ----------
    per_article = []
    for a in arts:
        delta = a['am'] - a['full']
        m = a['needle']
        if not m.any():
            continue
        per_article.append({
            'idx': a['article_idx'],
            'n_tok': len(delta),
            'needle_delta': float(delta[m].mean()),
            'needle_delta_max': float(delta[m].max()),
            'bg_delta': float(delta[~m].mean()),
            'bg_p99': float(np.percentile(delta[~m], 99)),
            'am_ppl': float(np.exp(a['am'].mean())),
            'full_ppl': float(np.exp(a['full'].mean())),
            'none_ppl': float(np.exp(a['none'].mean())),
            'needle_norm': (float(delta[m].sum() / (a['none'][m] - a['full'][m]).sum())
                            if abs((a['none'][m] - a['full'][m]).sum()) > 1e-6 else float('nan')),
        })
    out['per_article'] = per_article

    if per_article:
        nd = np.array([p['needle_delta'] for p in per_article])
        bg = np.array([p['bg_delta'] for p in per_article])
        out['needle_delta_mean'] = float(nd.mean())
        out['bg_delta_mean'] = float(bg.mean())
        out['ratio'] = float(nd.mean() / bg.mean()) if bg.mean() > 1e-9 else float('nan')
        # paired across articles: how often is the needle worse than its own background?
        out['frac_articles_needle_worse'] = float((nd > bg).mean())
        # how extreme is the needle within its own article's damage distribution?
        pcts = []
        for a in arts:
            delta = a['am'] - a['full']
            m = a['needle']
            if m.any():
                pcts.append(float((delta < delta[m].mean()).mean() * 100))
        out['needle_percentile_within_article'] = float(np.mean(pcts)) if pcts else float('nan')

    # ---------- alignment 1: needle-centred profile ----------
    prof = np.full((len(arts), 2 * window + 1), np.nan)
    for r, a in enumerate(arts):
        m = a['needle']
        if not m.any():
            continue
        c = int(np.flatnonzero(m)[0])
        delta = a['am'] - a['full']
        lo, hi = c - window, c + window + 1
        src_lo, src_hi = max(lo, 0), min(hi, len(delta))
        prof[r, (src_lo - lo):(src_hi - lo)] = delta[src_lo:src_hi]
    out['needle_profile'] = np.nanmean(prof, axis=0)
    out['needle_profile_n'] = np.sum(~np.isnan(prof), axis=0)

    # ---------- alignment 2: information-bucketed ----------
    info_all = np.concatenate([a['none'] - a['full'] for a in arts])
    delta_all = np.concatenate([a['am'] - a['full'] for a in arts])
    needle_all = np.concatenate([a['needle'] for a in arts])
    qs = np.linspace(0, 100, n_buckets + 1)
    edges = np.percentile(info_all, qs)
    edges[-1] += 1e-6
    rows = []
    for b in range(n_buckets):
        sel = (info_all >= edges[b]) & (info_all < edges[b + 1])
        if sel.sum() == 0:
            continue
        rows.append({
            'bucket': b,
            'info_lo': float(edges[b]), 'info_hi': float(edges[b + 1]),
            'n': int(sel.sum()),
            'delta_mean': float(delta_all[sel].mean()),
            # ratio of sums, not mean of ratios -- tokens with info ~0 (punctuation,
            # fixed UUID hyphen positions) otherwise blow the average up
            'norm_mean': (float(delta_all[sel].sum() / info_all[sel].sum())
                          if abs(info_all[sel].sum()) > 1e-6 else float('nan')),
            'pct_needle': float(needle_all[sel].mean() * 100),
        })
    out['info_buckets'] = rows
    out['n_tokens_total'] = int(len(delta_all))
    # correlation between "how much the context was worth" and "how much AM lost"
    if len(delta_all) > 2:
        out['corr_info_delta'] = float(np.corrcoef(info_all, delta_all)[0, 1])

    # ---------- alignment 3: normalized depth ----------
    depth_rows = []
    n_dec = 10
    dec_sum = np.zeros(n_dec)
    dec_cnt = np.zeros(n_dec)
    for a in arts:
        delta = a['am'] - a['full']
        pos = np.arange(len(delta)) / max(len(delta) - 1, 1)
        b = np.clip((pos * n_dec).astype(int), 0, n_dec - 1)
        for d in range(n_dec):
            s = b == d
            dec_sum[d] += delta[s].sum()
            dec_cnt[d] += s.sum()
    for d in range(n_dec):
        depth_rows.append({
            'decile': d,
            'delta_mean': float(dec_sum[d] / dec_cnt[d]) if dec_cnt[d] else float('nan'),
            'n': int(dec_cnt[d]),
        })
    out['depth_deciles'] = depth_rows
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('results_dir', type=str)
    ap.add_argument('--window', type=int, default=200)
    ap.add_argument('--buckets', type=int, default=10)
    ap.add_argument('--save', type=str, default=None, help='write aggregated arrays to this .npz')
    args = ap.parse_args()

    cells = load_cell(Path(args.results_dir))

    for method, arts in cells.items():
        s = summarize(arts, args.window, args.buckets)
        print(f"\n{'='*74}")
        print(f"{method}   ({len(arts)} articles, {s['n_tokens_total']:,} target tokens)")
        print(f"{'='*74}")

        pa = s['per_article']
        if pa:
            print(f"\n  aggregate perplexity over the repeat-prefill target")
            print(f"    AM   {np.mean([p['am_ppl'] for p in pa]):9.3f}")
            print(f"    full {np.mean([p['full_ppl'] for p in pa]):9.3f}   (ceiling)")
            print(f"    none {np.mean([p['none_ppl'] for p in pa]):9.3f}   (floor)")

            print(f"\n  DAMAGE  delta = NLL_AM - NLL_full   (nats)")
            print(f"    needle tokens      {s['needle_delta_mean']:9.4f}")
            print(f"    background tokens  {s['bg_delta_mean']:9.4f}")
            print(f"    ratio              {s['ratio']:9.1f}x")
            print(f"    needle sits at the {s['needle_percentile_within_article']:.1f}th percentile "
                  f"of its own article's damage")
            print(f"    needle worse than background in "
                  f"{s['frac_articles_needle_worse']*100:.0f}% of articles")

        if 'corr_info_delta' in s:
            print(f"\n  corr(context value, AM damage) = {s['corr_info_delta']:+.4f}")

        print(f"\n  damage vs how much the context was worth at that token")
        print(f"    {'bucket':>6} {'info range (nats)':>22} {'n':>9} {'delta':>9} {'norm':>7} {'%needle':>8}")
        for r in s['info_buckets']:
            print(f"    {r['bucket']:>6} {r['info_lo']:>10.3f}..{r['info_hi']:<10.3f} "
                  f"{r['n']:>9,} {r['delta_mean']:>9.4f} {r['norm_mean']:>7.3f} {r['pct_needle']:>7.2f}%")

        print(f"\n  damage vs depth in article (deciles)")
        line = "    " + "  ".join(f"{r['delta_mean']:.3f}" for r in s['depth_deciles'])
        print(line)

        prof = s['needle_profile']
        w = args.window
        print(f"\n  needle-centred profile (delta, +/-{w} tokens around needle start)")
        for off in (-w, -50, -10, -3, -1, 0, 1, 2, 3, 5, 10, 50, w):
            j = off + w
            if 0 <= j < len(prof):
                tag = "  <-- NEEDLE" if off == 0 else ""
                print(f"    offset {off:>+5}   delta={prof[j]:8.4f}{tag}")

        if args.save:
            np.savez_compressed(
                args.save,
                needle_profile=prof,
                needle_profile_n=s['needle_profile_n'],
                **{f'bucket_{k}': np.array([r[k] for r in s['info_buckets']])
                   for k in ('info_lo', 'info_hi', 'delta_mean', 'norm_mean', 'pct_needle')},
            )
            print(f"\n  wrote arrays -> {args.save}")


if __name__ == '__main__':
    main()
