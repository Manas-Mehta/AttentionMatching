#!/usr/bin/env python3
"""
Defines the RULER sweep cell matrix and emits the command for a single cell.

Matrix: 13 tasks x {4k,8k,16k} x {16x,8x,4x} = 117 AM cells,
plus 13 x 3 = 39 baseline cells (original + no_context, ratio-irrelevant) = 156 total.

Usage:
    python experiments/cells.py --count
    python experiments/cells.py --list
    python experiments/cells.py --index 42          # prints the command
    python experiments/cells.py --index 42 --run    # runs it

Run from the repo root (the parent of official/).
"""
import argparse
import shlex
import subprocess
import sys
from pathlib import Path

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
BUDGET = "head_budget_optimization/head_budgets/Qwen3-4B-Instruct-2507/optimized_agnostic.json"
N_SAMPLES = 100

TASKS = [
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery",
    "vt", "cwe", "fwe", "qa_1", "qa_2",
]
CONTEXTS = ["4k", "8k", "16k"]
# LCLM's compression ratios -> retained fraction
RATIOS = {"16x": 0.0625, "8x": 0.125, "4x": 0.25}

# AM-Fast = AM-HighestAttnKeys-fast: repeat-prefill queries, no self-study, no OMP.
# This is the config both the AM paper (Table 11) and LCLM used for RULER.
AM_METHOD = "highest_attn_keys_rms_nnls2_-3_3_lsq_on-policy"
AM_ALGO_CONFIG = "best"
AM_QUERY_CONFIG = "repeat"


def build_cells():
    cells = []
    # Baselines first so the ceiling/floor exist before we interpret AM.
    for ctx in CONTEXTS:
        for task in TASKS:
            cells.append({
                "kind": "baseline", "ctx": ctx, "ratio": "full", "task": task,
            })
    for ctx in CONTEXTS:
        for ratio in RATIOS:
            for task in TASKS:
                cells.append({
                    "kind": "am", "ctx": ctx, "ratio": ratio, "task": task,
                })
    return cells


def cell_command(cell):
    ctx, ratio, task = cell["ctx"], cell["ratio"], cell["task"]
    log_dir = f"../results/ruler/{ctx}/{ratio}"
    args = [
        sys.executable, "-u", "-m", "evaluation.run_qa_evaluation",
        "--model-name", MODEL,
        "--dataset-name", f"ruler_{ctx}_{task}",
        "--n-articles", str(N_SAMPLES),
        "--start-article", "0",
        "--log-dir", log_dir,
        "--name", task,
        "--compute-perplexity", "0",   # separate experiment; keeps the sweep fast
        "--compute-stats", "0",        # can OOM on long contexts
    ]
    if cell["kind"] == "baseline":
        args += ["--methods", "original", "no_context", "--target-size", "0.99"]
    else:
        args += [
            "--methods", AM_METHOD,
            "--target-size", str(RATIOS[ratio]),
            "--query-config", AM_QUERY_CONFIG,
            "--algorithm-config", AM_ALGO_CONFIG,
            "--precomputed-budget-path", BUDGET,
            "--max-ratio-per-head", "0.95",
        ]
    return args


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--count", action="store_true")
    p.add_argument("--list", action="store_true")
    p.add_argument("--index", type=int)
    p.add_argument("--run", action="store_true")
    a = p.parse_args()

    cells = build_cells()

    if a.count:
        print(len(cells))
        return
    if a.list:
        for i, c in enumerate(cells):
            print(f"{i:4d}  {c['kind']:8s} {c['ctx']:>3s} {c['ratio']:>4s}  {c['task']}")
        return
    if a.index is None:
        p.error("need --count, --list, or --index")

    if not (0 <= a.index < len(cells)):
        p.error(f"index {a.index} out of range 0..{len(cells)-1}")

    cell = cells[a.index]
    cmd = cell_command(cell)

    if not a.run:
        print(" ".join(shlex.quote(x) for x in cmd))
        return

    official = Path(__file__).resolve().parent.parent / "official"
    if not official.is_dir():
        sys.exit(f"official/ not found at {official}")
    print(f"[cell {a.index}] {cell['kind']} ctx={cell['ctx']} ratio={cell['ratio']} task={cell['task']}",
          flush=True)
    raise SystemExit(subprocess.call(cmd, cwd=official))


if __name__ == "__main__":
    main()
