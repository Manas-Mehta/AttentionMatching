"""Analysis of the §B8 mixed-dataset grid ("Trading Memory for Compute", Idea 1).

Three steps, so the heavy JSONs never leave the cluster:

  extract  (torch, repo root)  results/mixed_{1x,R,SS}/... JSONs -> two small CSVs
           results/mixed_rows.csv   one row per question per cell (no CoT text)
           results/mixed_docs.csv   one row per document per cell (ratios, build times)
  table    (anywhere)          per-type accuracy, CoT length, recovery vs 1x, stored / CoT /
           peak memory -> results/mixed_summary.csv + markdown (per-cell CIs are reference
           only: comparisons are made with `sig`, never by overlapping CIs)
  sig      (anywhere)          Greg's paired bootstrap for every A-vs-B comparison, per type
           -> results/mixed_sig.csv + markdown
  plot     (local, matplotlib) figures into Notes/figures/mixed/

  python experiments/analyze_mixed.py extract [--partial]
  python experiments/analyze_mixed.py table   [--rows results/mixed_rows.csv]
  python experiments/analyze_mixed.py plot    [--rows ...] [--out ../Notes/figures/mixed]
  python experiments/analyze_mixed.py sig     [--rows ...] [--kinds] [--n 10000]

Scores are exact-set (`ruler_score`); `set_overlap` is the partial score (used for cwe).
Accuracy is reported per question type only, never averaged across types (§B8b).
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict

TYPES = ['uuid', 'multikey', 'chain_last', 'chain_hop', 'cwe', 'logic']
LOGIC_KINDS = ['stated_base', 'stated_derivable', 'derived', 'false']
MODES = ['immediate', 'moderate', 'long']
FILLER_MODES = ['moderate_filler', 'long_filler']          # matched-length twins (phase 2)
MODE_ORDER = MODES + FILLER_MODES
MODE_DIR = {'modeimmediate': 'immediate', 'modemoderate_reason': 'moderate',
            'modelong_reason': 'long', 'modemoderate_filler': 'moderate_filler',
            'modelong_filler': 'long_filler'}
RATIO_DIR = {'1x': 1, '4x': 4, '8x': 8, '16x': 16, '32x': 32, '64x': 64,
             'ts0.0078125': 128, 'ts0.00390625': 256}

ROW_FIELDS = ['qset', 'ratio', 'mode', 'partial', 'doc_index', 'qid', 'qtype', 'logic_kind',
              'depth', 'gold', 'score', 'overlap', 'reasoning_tokens', 'hit_ceiling',
              'answer_tokens', 'reasoning_time_batch', 'answer_time_batch', 'batch_n',
              'prefix_cache_bytes', 'cot_kv_bytes', 'peak_total_bytes',
              'gpu_peak_alloc_batch_bytes']
DOC_FIELDS = ['qset', 'ratio', 'mode', 'partial', 'doc_index', 'original_article_tokens',
              'effective_article_tokens', 'achieved_ratio', 'tensor_ratio',
              'query_generation_time', 'compaction_time', 'extraction_time']


# ----------------------------------------------------------------------------- extract
def _cells(root, partial):
    """Yield (qset, ratio, mode, path, is_partial) for every result cell under root."""
    pats = [('1x', f'{root}/mixed_1x/mixed_v2_4k/*/*'),
            ('floor', f'{root}/mixed_1x/mixed_v2_4k_noctx/*/*'),
            ('R', f'{root}/mixed_R/mixed_v2_4k/*/*'),
            ('SS', f'{root}/mixed_SS/mixed_v2_4k/*/*')]
    for qset, pat in pats:
        for d in sorted(glob.glob(pat)):
            ratio = RATIO_DIR.get(os.path.basename(os.path.dirname(d)))
            mode = MODE_DIR.get(os.path.basename(d))
            if ratio is None or mode is None:
                continue
            finals = sorted(f for f in glob.glob(f'{d}/*.json') if 'summary' not in f)
            if finals:
                yield qset, ratio, mode, finals[-1], False
            elif partial:
                parts = glob.glob(f'{d}/*.json.partial')
                if parts:
                    yield qset, ratio, mode, parts[0], True


def extract(args):
    rows_path = os.path.join(args.root, 'mixed_rows.csv')
    docs_path = os.path.join(args.root, 'mixed_docs.csv')
    n_cells = 0
    with open(rows_path, 'w', newline='') as fr, open(docs_path, 'w', newline='') as fd:
        wr = csv.DictWriter(fr, ROW_FIELDS)
        wd = csv.DictWriter(fd, DOC_FIELDS)
        wr.writeheader()
        wd.writeheader()
        for qset, ratio, mode, path, is_partial in _cells(args.root, args.partial):
            d = json.load(open(path))
            n_cells += 1
            for a in d['results']:
                rs = a['qa_results']['results_per_question']
                doc = rs[0].get('doc_index') if rs else None
                base = dict(qset=qset, ratio=ratio, mode=mode, partial=int(is_partial))
                wd.writerow(dict(base, doc_index=doc,
                                 original_article_tokens=a.get('original_article_tokens'),
                                 effective_article_tokens=a.get('effective_article_tokens'),
                                 achieved_ratio=a.get('article_compaction_ratio'),
                                 tensor_ratio=a.get('article_tensor_compaction_ratio'),
                                 query_generation_time=a.get('query_generation_time'),
                                 compaction_time=a.get('compaction_time'),
                                 extraction_time=a.get('extraction_time')))
                for r in rs:
                    wr.writerow(dict(
                        base, doc_index=r.get('doc_index'), qid=r.get('question_id'),
                        qtype=r.get('qtype'), logic_kind=r.get('logic_kind') or '',
                        depth=r.get('depth'), gold='|'.join(r.get('ruler_outputs', [])),
                        score=r.get('ruler_score'), overlap=r.get('set_overlap'),
                        reasoning_tokens=r.get('reasoning_tokens'),
                        hit_ceiling=int(bool(r.get('hit_ceiling'))),
                        answer_tokens=r.get('num_generated_tokens'),
                        **{k: r.get(k) for k in (
                            'reasoning_time_batch', 'answer_time_batch', 'batch_n',
                            'prefix_cache_bytes', 'cot_kv_bytes', 'peak_total_bytes',
                            'gpu_peak_alloc_batch_bytes')}))
            print(f"  {qset:5s} {ratio:>3}x {mode:9s} {'PARTIAL ' if is_partial else ''}"
                  f"{len(d['results'])} docs  <- {path}")
    print(f"{n_cells} cells -> {rows_path}, {docs_path}")


# ----------------------------------------------------------------------------- table
def _load(rows_path):
    rows = list(csv.DictReader(open(rows_path)))
    for r in rows:
        r['ratio'] = int(r['ratio'])
        r['partial'] = int(r['partial'])
        for k in ('score', 'overlap', 'reasoning_tokens', 'hit_ceiling', 'prefix_cache_bytes',
                  'cot_kv_bytes', 'peak_total_bytes', 'reasoning_time_batch',
                  'answer_time_batch', 'batch_n', 'gpu_peak_alloc_batch_bytes'):
            v = r.get(k)
            r[k] = float(v) if v not in (None, '', 'None') else None
    return rows


def _groups(rows):
    """(qset, ratio, mode, group) -> rows, group = question type or logic/<kind>."""
    g = defaultdict(list)
    for r in rows:
        key = (r['qset'], r['ratio'], r['mode'])
        g[key + (r['qtype'],)].append(r)
        if r['qtype'] == 'logic' and r['logic_kind']:
            g[key + (f"logic/{r['logic_kind']}",)].append(r)
    return g


def _boot_ci(rows, field='score', n=1000, seed=0):
    """95% CI of the mean by resampling documents (questions in a doc are not independent)."""
    import numpy as np
    by_doc = defaultdict(list)
    for r in rows:
        by_doc[r['doc_index']].append(r[field])
    docs = list(by_doc)
    sums = np.array([sum(by_doc[d]) for d in docs])
    cnts = np.array([len(by_doc[d]) for d in docs])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(docs), size=(n, len(docs)))
    means = sums[idx].sum(1) / cnts[idx].sum(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def summarize(rows):
    g = _groups(rows)
    out = []
    for (qset, ratio, mode, grp), rs in sorted(g.items(), key=lambda kv: (
            ['1x', 'floor', 'R', 'SS'].index(kv[0][0]), kv[0][1], MODE_ORDER.index(kv[0][2]),
            kv[0][3])):
        acc = _mean([r['score'] for r in rs])
        lo, hi = _boot_ci(rs)
        base = g.get(('1x', 1, mode, grp))
        base_acc = _mean([r['score'] for r in base]) if base else None
        out.append(dict(
            qset=qset, ratio=ratio, mode=mode, group=grp, n=len(rs),
            n_docs=len({r['doc_index'] for r in rs}),
            partial=max(r['partial'] for r in rs),
            acc=acc, ci_lo=lo, ci_hi=hi,
            overlap=_mean([r['overlap'] for r in rs]),
            recovery_vs_1x=(acc / base_acc) if base_acc else None,
            cot_tokens=_mean([r['reasoning_tokens'] for r in rs]),
            hit_ceiling=_mean([r['hit_ceiling'] for r in rs]),
            stored_mb=_mean([r['prefix_cache_bytes'] for r in rs]) / 1e6
            if _mean([r['prefix_cache_bytes'] for r in rs]) else None,
            cot_kv_mb=(_mean([r['cot_kv_bytes'] for r in rs]) or 0) / 1e6,
            peak_mb=(_mean([r['peak_total_bytes'] for r in rs]) or 0) / 1e6,
            reason_s_per_q=_mean([r['reasoning_time_batch'] / r['batch_n'] for r in rs
                                  if r['reasoning_time_batch'] is not None and r['batch_n']]),
        ))
    return out


def table(args):
    rows = _load(args.rows)
    summ = summarize(rows)
    path = os.path.join(os.path.dirname(args.rows) or '.', 'mixed_summary.csv')
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, list(summ[0].keys()))
        w.writeheader()
        w.writerows(summ)
    idx = {(s['qset'], s['ratio'], s['mode'], s['group']): s for s in summ}
    qsets = [q for q in ('R', 'SS') if any(s['qset'] == q for s in summ)]
    ratios = sorted({s['ratio'] for s in summ if s['qset'] in ('R', 'SS')})
    cols = [(q, m) for q in qsets for m in MODES]
    groups = TYPES + [f'logic/{k}' for k in LOGIC_KINDS]
    fmt = lambda s, k: '' if s is None or s.get(k) is None else (
        f"{s[k]:.2f}" + ('*' if s.get('partial') else ''))
    for metric, label in (('acc', 'accuracy (exact)'), ('cot_tokens', 'mean CoT tokens')):
        print(f"\n## {label}   (* = partial cell)\n")
        for grp in groups:
            if not any(s['group'] == grp for s in summ):
                continue
            print(f"### {grp}\n")
            print('| ratio | ' + ' | '.join(f'{q} {m[:3]}' for q, m in cols) + ' |')
            print('|---|' + '---|' * len(cols))
            base = [idx.get(('1x', 1, m, grp)) for m in MODES]
            print('| 1× | ' + ' | '.join(fmt(base[MODES.index(m)], metric) if metric != 'cot_tokens'
                                          else (f"{base[MODES.index(m)][metric]:.0f}" if base[MODES.index(m)] else '')
                                          for q, m in cols) + ' |')
            for ra in ratios:
                cells = []
                for q, m in cols:
                    s = idx.get((q, ra, m, grp))
                    if metric == 'cot_tokens':
                        cells.append('' if not s else f"{s[metric]:.0f}" + ('*' if s['partial'] else ''))
                    else:
                        cells.append(fmt(s, metric))
                print(f'| {ra}× | ' + ' | '.join(cells) + ' |')
            fl = idx.get(('floor', 1, 'immediate', grp))
            if fl and metric == 'acc':
                print(f"| floor | {fl['acc']:.2f} (immediate, no document) |" + ' |' * (len(cols) - 1))
            print()
    print(f"-> {path}")


# ----------------------------------------------------------------------------- plot
def plot(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = _load(args.rows)
    summ = summarize(rows)
    os.makedirs(args.out, exist_ok=True)
    idx = {(s['qset'], s['ratio'], s['mode'], s['group']): s for s in summ}
    qsets = [q for q in ('R', 'SS') if any(s['qset'] == q for s in summ)]
    ratios = [1] + sorted({s['ratio'] for s in summ if s['qset'] in ('R', 'SS')})
    cmap = plt.get_cmap('viridis')
    color = {ra: cmap(i / max(1, len(ratios) - 1)) for i, ra in enumerate(ratios)}
    title = {'R': 'Repeat-prefill (R)', 'SS': 'Self-study (SS)'}

    # 1) per type: accuracy vs measured mean CoT tokens, one line per ratio (modes = points)
    for grp in TYPES + [f'logic/{k}' for k in LOGIC_KINDS]:
        if not any(s['group'] == grp for s in summ):
            continue
        fig, axes = plt.subplots(1, len(qsets), figsize=(5.2 * len(qsets), 4.2), sharey=True,
                                 squeeze=False)
        xmax = max((s['cot_tokens'] or 0) for s in summ if s['group'] == grp) * 1.05 + 1
        for ax, q in zip(axes[0], qsets):
            for ra in ratios:
                src = '1x' if ra == 1 else q
                pts = [idx.get((src, ra, m, grp)) for m in MODES]
                pts = [p for p in pts if p]
                if not pts:
                    continue
                ax.plot([p['cot_tokens'] for p in pts], [p['acc'] for p in pts], marker='o',
                        color=color[ra], label=f'{ra}x', lw=2, ms=5,
                        linestyle='--' if any(p['partial'] for p in pts) else '-')
            fl = idx.get(('floor', 1, 'immediate', grp))
            if fl:
                ax.plot([0], [fl['acc']], 'x', color='0.4', ms=8, label='no document')
            ax.set_title(title[q])
            ax.set_xlim(-0.02 * xmax, xmax)
            ax.set_ylim(-0.03, 1.03)
            ax.set_xlabel('measured mean CoT tokens (points: immediate, moderate, long)')
            ax.grid(alpha=0.3)
        axes[0][0].set_ylabel(f'accuracy, exact ({grp})')
        handles = {}
        for ax in axes[0]:
            for h, l in zip(*ax.get_legend_handles_labels()):
                handles.setdefault(l, h)
        order = [f'{ra}x' for ra in ratios if f'{ra}x' in handles] + \
                [l for l in handles if not l.endswith('x') or l == 'no document']
        order = list(dict.fromkeys(order))
        axes[0][-1].legend([handles[l] for l in order], order, title='compression',
                           loc='best', fontsize=8)
        fig.suptitle(f'Mixed dataset, {grp}: accuracy vs CoT length '
                     f'(100 docs; dashed = cell still running)')
        fig.tight_layout()
        fn = os.path.join(args.out, f"acc_vs_cot_{grp.replace('/', '_')}.png")
        fig.savefig(fn, dpi=150)
        plt.close(fig)
        print('  ', fn)

    # 2) overview: accuracy vs compression ratio, rows = query set, cols = type, lines = mode
    mcol = {'immediate': '#d95f02', 'moderate': '#7570b3', 'long': '#1b9e77'}
    fig, axes = plt.subplots(len(qsets), len(TYPES), figsize=(2.6 * len(TYPES), 2.6 * len(qsets)),
                             sharex=True, sharey=True, squeeze=False)
    for i, q in enumerate(qsets):
        for j, t in enumerate(TYPES):
            ax = axes[i][j]
            for m in MODES:
                # NaN for a missing cell breaks the line instead of bridging the gap
                xs, ys, part = [], [], False
                for ra in ratios:
                    s = idx.get(('1x' if ra == 1 else q, ra, m, t))
                    xs.append(ra)
                    ys.append(s['acc'] if s else float('nan'))
                    part |= bool(s and s['partial'])
                if any(y == y for y in ys):
                    ax.plot(xs, ys, marker='o', color=mcol[m], ms=3, lw=1.5, label=m,
                            linestyle='--' if part else '-')
            fl = idx.get(('floor', 1, 'immediate', t))
            if fl:
                ax.plot([1], [fl['acc']], 'x', color='0.4', ms=6, label='no document')
            ax.set_xscale('log', base=2)
            ax.set_xticks(ratios)
            ax.set_xticklabels([f'{r}' for r in ratios], fontsize=7)
            ax.set_ylim(-0.03, 1.03)
            ax.grid(alpha=0.3)
            if i == 0:
                ax.set_title(t, fontsize=10)
            if j == 0:
                ax.set_ylabel(f'{title[q]}\naccuracy', fontsize=9)
            if i == len(qsets) - 1:
                ax.set_xlabel('compression (x)', fontsize=8)
    axes[0][-1].legend(fontsize=7, loc='best')
    fig.suptitle('Mixed dataset: accuracy vs compression, per question type (dashed = cell still running)')
    fig.tight_layout()
    fn = os.path.join(args.out, 'acc_vs_ratio_overview.png')
    fig.savefig(fn, dpi=150)
    plt.close(fig)
    print('  ', fn)


# ----------------------------------------------------------------------------- sig
def paired_bootstrap(rows_a, rows_b, n=10000, seed=0):
    """Greg's paired bootstrap: p = share of resampled test sets on which B fails to beat A.

    "Fails to beat" counts ties (B - A <= 0), following his reading of the result ("you
    continued to beat system A 95% of the time"). With strict "< 0", a comparison where B
    never loses a document but wins only a couple reads p = 0 from the resamples that tie.

    Pairs A and B by question id (same document, same question, two systems). Resamples
    documents, not questions: every question in a document is answered from the same
    cache, so they are not independent. For one-question-per-document types this is the
    same as resampling questions. No new inference; predictions are reused.
    Returns dict(acc_a, acc_b, delta, p, n_q, n_docs).
    """
    import numpy as np
    a = {r['qid']: r for r in rows_a}
    b = {r['qid']: r for r in rows_b}
    per_doc = defaultdict(lambda: [0.0, 0.0, 0])
    for q, ra in a.items():
        rb = b.get(q)
        if rb is None:
            continue
        d = per_doc[ra['doc_index']]
        d[0] += ra['score']
        d[1] += rb['score']
        d[2] += 1
    if not per_doc:
        return None
    docs = list(per_doc)
    sa = np.array([per_doc[d][0] for d in docs])
    sb = np.array([per_doc[d][1] for d in docs])
    cnt = np.array([per_doc[d][2] for d in docs])
    rng = np.random.default_rng(seed)
    p_counter = p_rev = 0
    for start in range(0, n, 2000):                   # chunks keep memory flat
        idx = rng.integers(0, len(docs), size=(min(2000, n - start), len(docs)))
        delta = (sb[idx].sum(1) - sa[idx].sum(1)) / cnt[idx].sum(1)
        p_counter += int((delta <= 0).sum())
        p_rev += int((delta >= 0).sum())              # the same test with A and B swapped
    acc_a, acc_b = sa.sum() / cnt.sum(), sb.sum() / cnt.sum()
    return dict(acc_a=float(acc_a), acc_b=float(acc_b), delta=float(acc_b - acc_a),
                p=p_counter / n, p_rev=p_rev / n, n_q=int(cnt.sum()), n_docs=len(docs))


def _comparisons(cells):
    """(family, label, A-cell, B-cell) for every comparison whose two cells both exist."""
    out = []
    grid = sorted({(q, r) for q, r, _ in cells if q in ('1x', 'R', 'SS')},
                  key=lambda x: (['1x', 'R', 'SS'].index(x[0]), x[1]))
    for q, r in grid:                                              # CoT vs no CoT
        for m in ('moderate', 'long'):
            if (q, r, 'immediate') in cells and (q, r, m) in cells:
                name = '1x' if q == '1x' else f'{q} {r}x'
                out.append(('cot', f'{name}: {m} vs immediate', (q, r, 'immediate'), (q, r, m)))
    for r in sorted({r for q, r, _ in cells if q in ('R', 'SS')}):  # query set
        for m in MODES:
            if ('R', r, m) in cells and ('SS', r, m) in cells:
                out.append(('qset', f'{r}x {m}: SS vs R', ('R', r, m), ('SS', r, m)))
    for q, r in grid:                                              # reasoning vs filler twin
        for m in ('moderate', 'long'):
            if (q, r, m) in cells and (q, r, f'{m}_filler') in cells:
                out.append(('filler', f"{'1x' if q == '1x' else f'{q} {r}x'}: {m} vs its filler twin",
                            (q, r, f'{m}_filler'), (q, r, m)))
    return out


def sig(args):
    rows = _load(args.rows)
    by_cell_grp = defaultdict(list)
    partial = defaultdict(int)
    for r in rows:
        key = (r['qset'], r['ratio'], r['mode'])
        partial[key] |= r['partial']
        by_cell_grp[key + (r['qtype'],)].append(r)
        if args.kinds and r['qtype'] == 'logic' and r['logic_kind']:
            by_cell_grp[key + (f"logic/{r['logic_kind']}",)].append(r)
    cells = {k[:3] for k in by_cell_grp}
    groups = TYPES + ([f'logic/{k}' for k in LOGIC_KINDS] if args.kinds else [])
    results = []
    for fam, label, ca, cb in _comparisons(cells):
        for grp in groups:
            ra, rb = by_cell_grp.get(ca + (grp,)), by_cell_grp.get(cb + (grp,))
            if not ra or not rb:
                continue
            res = paired_bootstrap(ra, rb, n=args.n)
            if res is None:
                continue
            # +1 = B significantly ahead, -1 = A significantly ahead (swapped test), 0 = neither.
            # Ties count against the leader, so identical systems give p = 1, not 0.
            sig_dir = (1 if res['delta'] > 0 and res['p'] < 0.05 else
                       -1 if res['delta'] < 0 and res['p_rev'] < 0.05 else 0)
            res.update(family=fam, comparison=label, group=grp, significant=sig_dir,
                       partial=int(partial[ca] or partial[cb]))
            results.append(res)
    path = os.path.join(os.path.dirname(args.rows) or '.', 'mixed_sig.csv')
    with open(path, 'w', newline='') as f:
        cols = ['family', 'comparison', 'group', 'acc_a', 'acc_b', 'delta', 'p', 'p_rev',
                'significant', 'n_q', 'n_docs', 'partial']
        w = csv.DictWriter(f, cols)
        w.writeheader()
        w.writerows({k: r[k] for k in cols} for r in results)

    title = {'cot': 'CoT effect (B = mode with CoT, A = immediate, same cache)',
             'qset': 'Query set (B = SS, A = R, same ratio and mode)',
             'filler': 'Reasoning vs matched-length filler (B = reasoning, A = filler)'}
    print(f"Paired bootstrap, N={args.n}, documents resampled. Cell = B - A in accuracy points; "
          f"* = significant (p < 0.05) in the direction of the sign; ~ = a cell still running.")
    for fam in ('cot', 'qset', 'filler'):
        fr = [r for r in results if r['family'] == fam]
        if not fr:
            continue
        print(f"\n### {title[fam]}\n")
        print('| comparison | ' + ' | '.join(groups) + ' |')
        print('|---|' + '---|' * len(groups))
        for label in dict.fromkeys(r['comparison'] for r in fr):
            cells_out = []
            for grp in groups:
                r = next((x for x in fr if x['comparison'] == label and x['group'] == grp), None)
                if r is None:
                    cells_out.append('')
                    continue
                pv = r['p'] if r['delta'] >= 0 else r['p_rev']
                cells_out.append(f"{100 * r['delta']:+.0f}{'*' if r['significant'] else ''}"
                                 f"{'~' if r['partial'] else ''} (p={pv:.3f})")
            print(f'| {label} | ' + ' | '.join(cells_out) + ' |')
    print(f"\n-> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    e = sub.add_parser('extract')
    e.add_argument('--root', default='results')
    e.add_argument('--partial', action='store_true', help='include running cells (.json.partial)')
    t = sub.add_parser('table')
    t.add_argument('--rows', default='results/mixed_rows.csv')
    p = sub.add_parser('plot')
    p.add_argument('--rows', default='results/mixed_rows.csv')
    p.add_argument('--out', default='../Notes/figures/mixed')
    s = sub.add_parser('sig')
    s.add_argument('--rows', default='results/mixed_rows.csv')
    s.add_argument('--kinds', action='store_true', help='also test logic kinds separately')
    s.add_argument('--n', type=int, default=10000)
    args = ap.parse_args()
    {'extract': extract, 'table': table, 'plot': plot, 'sig': sig}[args.cmd](args)


if __name__ == '__main__':
    main()
