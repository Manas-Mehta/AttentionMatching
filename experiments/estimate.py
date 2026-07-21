#!/usr/bin/env python3
"""
Project the cost of a full sweep from smoke-test timings.

Reads the per-article timings the evaluator writes into each results JSON
(compaction, query generation, generation) and extrapolates to a target preset.

    python experiments/estimate.py --from smoke --to full

Context-length scaling is measured from the smoke run itself (it includes 4k and
16k), so we fit cost ~ ctx^alpha rather than assuming a scaling exponent.
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import cells as cellmod

CTX_TOKENS = {"4k": 4096, "8k": 8192, "16k": 16384}


def load_runs(results_dir: Path, preset: str):
    """-> {(ctx, ratio, task, kind): seconds_per_sample}"""
    out = {}
    root = results_dir / preset
    if not root.is_dir():
        raise SystemExit(f"no results at {root} — run the smoke preset first")
    for f in sorted(root.rglob("*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        cfg = d.get("config", {})
        ctx_ratio = f.relative_to(root).parts  # (ctx, ratio, file)
        if len(ctx_ratio) < 2:
            continue
        ctx, ratio = ctx_ratio[0], ctx_ratio[1]
        n = cfg.get("n_articles") or 1
        for res in d.get("results", []):
            method = res.get("method", "?")
            kind = "baseline" if method in ("original", "no_context") else "am"
            per_art = res.get("avg_compaction_time_per_article")
            total = res.get("total_compaction_time")
            qa = res.get("qa_results", {})
            gen = sum(q.get("generation_time", 0.0)
                      for q in qa.get("results_per_question", []))
            comp = total if total is not None else (per_art or 0.0) * n
            secs = (comp + gen) / max(n, 1)
            key = (ctx, ratio, res.get("task_name") or f.stem, kind)
            out[key] = max(out.get(key, 0.0), secs)
    return out


def fit_alpha(per_sample, kind):
    """Fit seconds ~ tokens^alpha from cells that differ only in context length."""
    by_task = defaultdict(dict)
    for (ctx, ratio, task, k), s in per_sample.items():
        if k == kind and ctx in CTX_TOKENS:
            by_task[(task, ratio)][ctx] = s
    pts = []
    for _, d in by_task.items():
        if "4k" in d and "16k" in d and d["4k"] > 0:
            pts.append(math.log(d["16k"] / d["4k"]) / math.log(4.0))
    if not pts:
        return None
    return sum(pts) / len(pts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", default="smoke")
    ap.add_argument("--to", dest="dst", default="full")
    ap.add_argument("--results", default="results")
    ap.add_argument("--concurrency", type=int, default=8,
                    help="how many array jobs run at once (for wall-clock estimate)")
    a = ap.parse_args()

    per_sample = load_runs(Path(a.results), a.src)
    if not per_sample:
        raise SystemExit("no timing data found")

    print(f"=== measured, preset '{a.src}' (seconds per sample) ===")
    for (ctx, ratio, task, kind) in sorted(per_sample):
        print(f"  {kind:8s} {ctx:>3s} {ratio:>4s} {task:18s} {per_sample[(ctx,ratio,task,kind)]:8.2f}s")

    est = {}
    for kind in ("am", "baseline"):
        alpha = fit_alpha(per_sample, kind)
        base = [s for (c, _, _, k), s in per_sample.items() if k == kind and c == "4k"]
        if not base:
            continue
        b = sum(base) / len(base)
        est[kind] = (b, alpha)
        a_txt = f"{alpha:.2f}" if alpha is not None else "n/a (need 4k+16k)"
        print(f"\n{kind}: {b:.2f}s/sample at 4k, scaling exponent alpha={a_txt}")

    print(f"\n=== projected, preset '{a.dst}' ===")
    target = cellmod.build_cells(a.dst)
    total = 0.0
    per_ctx = defaultdict(float)
    for c in target:
        kind = c["kind"]
        if kind not in est:
            continue
        b, alpha = est[kind]
        alpha = 1.0 if alpha is None else alpha
        scale = (CTX_TOKENS[c["ctx"]] / 4096.0) ** alpha
        secs = b * scale * c["samples"]
        if kind == "baseline":
            secs *= 2  # baseline cells run two methods
        total += secs
        per_ctx[c["ctx"]] += secs

    for ctx in ("4k", "8k", "16k"):
        if per_ctx[ctx]:
            print(f"  {ctx:>3s}: {per_ctx[ctx]/3600:7.1f} GPU-hours")
    print(f"  {'TOTAL':>3s}: {total/3600:7.1f} GPU-hours over {len(target)} cells")
    print(f"  longest single cell: ~{max((est['am'][0]*(CTX_TOKENS['16k']/4096)**(est['am'][1] or 1)*c['samples'])/60 for c in target if c['kind']=='am'):.0f} min")
    print(f"  wall-clock at concurrency {a.concurrency}: ~{total/3600/a.concurrency:.1f} h")


if __name__ == "__main__":
    main()
