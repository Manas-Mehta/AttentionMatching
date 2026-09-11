#!/usr/bin/env python3
"""
Read the Phase 1 grid and produce the surface ("Trading Memory for Compute", §5).

Input:  results/<outroot>/<task>/<ratio>/<cell>/*.json, one directory per cell.
Output: the accuracy table, the retained-but-latent labels, the footprint table,
        and a contour plot of accuracy over (compression, CoT budget).

    python experiments/analyze_chain_grid.py --root results/chain_phase1

Section 5 asks for three readings off this surface:
  * isolines with negative slope   -- CoT substituting for cache
  * a visible vertical asymptote   -- the compression floor, past which no budget helps
  * a non-trivial silent-failure cell at moderate compression

and one control: real CoT against matched filler. If filler matches real CoT, the
effect was cache capacity and not reasoning.
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

RATIO_ORDER = ["1x", "4x", "8x", "16x", "32x", "64x"]
BUDGETS = [0, 64, 256, 1024]


def walk_questions(obj):
    """Every per-question record, wherever the harness nested it."""
    if isinstance(obj, dict):
        if "ruler_score" in obj and "question_id" in obj:
            yield obj
        for v in obj.values():
            yield from walk_questions(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from walk_questions(v)


def article_stats(payload):
    """Per-article compaction numbers, which live beside the questions, not in them."""
    out = {}
    for r in payload.get("results", []) or []:
        idx = r.get("article_index", r.get("article_idx"))
        out[idx] = {
            "effective_ratio": r.get("article_compaction_ratio"),
            "tensor_ratio": r.get("article_tensor_compaction_ratio"),
            "compaction_time": r.get("compaction_time"),
        }
    return out


def load_cells(root: Path):
    """{(ratio, budget, mode): [question records]} plus the per-cell metadata."""
    cells, meta = defaultdict(list), {}
    for fp in sorted(root.glob("**/*.json")):
        try:
            payload = json.load(open(fp))
        except json.JSONDecodeError:
            print(f"  skipping unreadable {fp}")
            continue
        qs = list(walk_questions(payload))
        if not qs:
            continue
        ratio = fp.parent.parent.name          # .../<ratio>/<cell>/file.json
        budget = qs[0].get("cot_budget", 0) or 0
        # a probe pins its own budget to 0, so read the mode off a non-probe row
        main = next((q for q in qs if not q.get("is_probe", False)), qs[0])
        budget = main.get("cot_budget", 0) or 0
        mode = main.get("cot_mode", "none")
        key = (ratio, budget, "none" if budget == 0 else mode)
        cells[key] += qs
        meta.setdefault(key, {}).update(article_stats(payload))
    return cells, meta


def label_instances(questions):
    """Section 5's three buckets, per document.

    every lookup recoverable + composed correct -> no failure
    every lookup recoverable + composed wrong   -> silent failure
    a lookup not recoverable                    -> storage failure

    The sharp form of the hypothesis is that CoT gain sits in the silent bucket.
    """
    by_doc = defaultdict(lambda: {"probes": [], "main": None})
    for q in questions:
        d = q.get("doc_index")
        if d is None:
            continue
        if q.get("is_probe"):
            by_doc[d]["probes"].append(q["ruler_score"] == 1.0)
        else:
            by_doc[d]["main"] = q["ruler_score"] == 1.0

    counts = {"no_failure": 0, "silent": 0, "storage": 0, "unlabelled": 0}
    for d, v in by_doc.items():
        if v["main"] is None or not v["probes"]:
            counts["unlabelled"] += 1
        elif not all(v["probes"]):
            counts["storage"] += 1
        elif v["main"]:
            counts["no_failure"] += 1
        else:
            counts["silent"] += 1
    return counts, len(by_doc)


def accuracy(questions, probes=False):
    sel = [q for q in questions if bool(q.get("is_probe", False)) == probes]
    if not sel:
        return float("nan"), 0
    return sum(q["ruler_score"] for q in sel) / len(sel), len(sel)


def fmt(x, w=6):
    return " " * w if x != x else f"{x:{w}.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/chain_phase1")
    ap.add_argument("--figdir", default=None, help="write the contour plot here")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"no such directory: {root}")
    cells, meta = load_cells(root)
    if not cells:
        raise SystemExit(f"no result JSON under {root}")

    ratios = [r for r in RATIO_ORDER if any(k[0] == r for k in cells)]
    ratios += sorted({k[0] for k in cells} - set(RATIO_ORDER))

    # ---------------------------------------------------------------- accuracy
    for mode, title in (("reason", "REAL CoT"), ("filler", "FILLER (control)")):
        print(f"\n=== composed-task accuracy, {title} ===")
        print(f"{'ratio':>7} " + "".join(f"{b:>8}" for b in BUDGETS) + "     n")
        for r in ratios:
            row, n = [], 0
            for b in BUDGETS:
                key = (r, b, "none" if b == 0 else mode)
                acc, n_q = (accuracy(cells[key]) if key in cells else (float("nan"), 0))
                row.append(acc)
                n = max(n, n_q)
            print(f"{r:>7} " + "".join(fmt(a, 8) for a in row) + f"  {n:>4}")

    # ------------------------------------------------------- probes and labels
    print("\n=== per-hop probes, and the retained-but-latent split ===")
    print(f"{'ratio':>7} {'budget':>7} {'mode':>7} {'probe':>7} "
          f"{'ok':>5} {'silent':>7} {'storage':>8}")
    for r in ratios:
        for b in BUDGETS:
            for mode in ("none", "reason", "filler"):
                key = (r, b, mode)
                if key not in cells:
                    continue
                pacc, _ = accuracy(cells[key], probes=True)
                lab, ndoc = label_instances(cells[key])
                print(f"{r:>7} {b:>7} {mode:>7} {fmt(pacc, 7)} "
                      f"{lab['no_failure']:>5} {lab['silent']:>7} {lab['storage']:>8}")

    # ------------------------------------------------------------- footprint
    print("\n=== footprint (millions of KV elements) ===")
    print(f"{'ratio':>7} {'budget':>7} {'mode':>7} {'prefix':>9} {'peak':>9} "
          f"{'eff.ratio':>10} {'tensor':>8}")
    for r in ratios:
        for b in BUDGETS:
            for mode in ("none", "reason", "filler"):
                key = (r, b, mode)
                if key not in cells:
                    continue
                mains = [q for q in cells[key] if not q.get("is_probe")]
                pre = [q.get("prefix_cache_elems") for q in mains if q.get("prefix_cache_elems")]
                pk = [q.get("peak_total_elems") for q in mains if q.get("peak_total_elems")]
                st = [v for v in meta.get(key, {}).values()]
                eff = [s["effective_ratio"] for s in st if s.get("effective_ratio")]
                ten = [s["tensor_ratio"] for s in st if s.get("tensor_ratio")]
                avg = lambda xs: sum(xs) / len(xs) if xs else float("nan")
                print(f"{r:>7} {b:>7} {mode:>7} {fmt(avg(pre)/1e6, 9)} {fmt(avg(pk)/1e6, 9)} "
                      f"{fmt(avg(eff), 10)} {fmt(avg(ten), 8)}")

    # ------------------------------------------------------ reasoning length
    print("\n=== reasoning tokens actually used (mean) ===")
    for r in ratios:
        parts = []
        for b in BUDGETS:
            key = (r, b, "none" if b == 0 else "reason")
            if key in cells:
                mains = [q for q in cells[key] if not q.get("is_probe")]
                used = [q.get("reasoning_tokens", 0) for q in mains]
                parts.append(f"{b}:{sum(used)/len(used):.0f}" if used else f"{b}:-")
        print(f"{r:>7}  " + "  ".join(parts))

    # ------------------------------------------------------------------ plot
    if args.figdir:
        try:
            plot(cells, ratios, Path(args.figdir))
        except Exception as e:
            print(f"\n(plot skipped: {e.__class__.__name__}: {e})")


def plot(cells, ratios, figdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figdir.mkdir(parents=True, exist_ok=True)
    # x is the CoT budget, y is compression. 0 is plotted at 32 so a log axis works.
    xs = [32 if b == 0 else b for b in BUDGETS]
    ys = [float(r.rstrip("x")) for r in ratios]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, mode, title in zip(axes, ("reason", "filler"),
                               ("real chain of thought", "matched filler")):
        Z = np.full((len(ys), len(xs)), np.nan)
        for i, r in enumerate(ratios):
            for j, b in enumerate(BUDGETS):
                key = (r, b, "none" if b == 0 else mode)
                if key in cells:
                    Z[i, j] = accuracy(cells[key])[0]
        m = np.ma.masked_invalid(Z)
        im = ax.contourf(xs, ys, m, levels=np.linspace(0, 1, 11), cmap="viridis")
        if np.isfinite(Z).sum() > 3:
            cs = ax.contour(xs, ys, m, levels=[.2, .4, .6, .8], colors="white", linewidths=1)
            ax.clabel(cs, inline=True, fontsize=8, fmt="%.1f")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.set_xticks(xs)
        ax.set_xticklabels(["0"] + [str(b) for b in BUDGETS[1:]])
        ax.set_yticks(ys)
        ax.set_yticklabels(ratios)
        ax.set_xlabel("decoded tokens before answering")
        ax.set_title(title)
    axes[0].set_ylabel("cache compression")
    fig.colorbar(im, ax=axes, label="fraction of documents answered correctly")
    out = figdir / "phase1_surface.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
