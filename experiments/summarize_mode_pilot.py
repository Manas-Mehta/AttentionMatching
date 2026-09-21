#!/usr/bin/env python3
"""Read a method-A pilot and report, per prompt mode, the length the model chose.

Answers the go/no-go question directly: do the four modes produce distinct, spaced
average lengths? Also flags a mode that keeps hitting the safety ceiling (a runaway,
which means the "no hard cap" assumption is being violated for that mode).

    python experiments/summarize_mode_pilot.py --root results/chain_pilot/<task>
"""
import argparse, json, statistics as st
from collections import defaultdict
from pathlib import Path


def walk_questions(obj):
    if isinstance(obj, dict):
        if "ruler_score" in obj and "question_id" in obj:
            yield obj
        for v in obj.values():
            yield from walk_questions(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from walk_questions(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="results/<outroot>/<task>")
    ap.add_argument("--ceiling", type=int, default=2048)
    args = ap.parse_args()
    root = Path(args.root)

    # {(ratio, label): {"tok": [...], "ok": [...]}}
    cells = defaultdict(lambda: {"tok": [], "ok": []})
    for fp in sorted(root.glob("**/*.json")):
        try:
            payload = json.load(open(fp))
        except json.JSONDecodeError:
            continue
        ratio = fp.parent.parent.name
        for q in walk_questions(payload):
            if q.get("is_probe"):
                continue
            label = q.get("cot_label") or q.get("cot_prompt_mode") or (
                "cot0" if not q.get("cot_budget") else
                f"cot{q['cot_budget']}_{q.get('cot_mode','reason')}")
            c = cells[(ratio, label)]
            c["tok"].append(int(q.get("reasoning_tokens", 0) or 0))
            c["ok"].append(1.0 if q.get("ruler_score") == 1.0 else 0.0)

    if not cells:
        print(f"no cells under {root}")
        return

    hdr = f'{"ratio":>6} {"mode/cell":22} {"n":>3} {"mean":>7} {"median":>7} ' \
          f'{"min":>5} {"max":>6} {"std":>7} {"%ceil":>6} {"acc":>5}'
    print(hdr); print("-" * len(hdr))
    for (ratio, label) in sorted(cells):
        c = cells[(ratio, label)]
        tok, ok = c["tok"], c["ok"]
        n = len(tok)
        mean = sum(tok) / n
        med = st.median(tok)
        sd = st.pstdev(tok) if n > 1 else 0.0
        pceil = 100.0 * sum(1 for t in tok if t >= 0.98 * args.ceiling) / n
        acc = sum(ok) / n
        print(f'{ratio:>6} {label:22} {n:>3} {mean:>7.1f} {med:>7.1f} '
              f'{min(tok):>5} {max(tok):>6} {sd:>7.1f} {pceil:>5.0f}% {acc:>5.2f}')

    print("\nGo/no-go: means should be distinct and spaced across modes, and %ceil")
    print("should be ~0. A mode near the ceiling is a runaway -> method B for it.")


if __name__ == "__main__":
    main()
