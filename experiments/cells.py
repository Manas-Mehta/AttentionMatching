#!/usr/bin/env python3
"""
Defines the RULER sweep cell matrix and emits the command for a single cell.

Presets control the size of the sweep. Cell indices are PRESET-SPECIFIC —
always pass the same --preset to sbatch and to any analysis.

    smoke : 4 diagnostic tasks, 4k only, 16x,          20 samples ->   8 cells
    small : 4 diagnostic tasks, 4k+16k, 16x+4x,        50 samples ->  24 cells
    repro : all 13 tasks, 4k only, 16x+8x,             50 samples ->  39 cells
            (closest match to the paper's own released RULER script)
    full  : all 13 tasks, 4k+8k+16k, 16x+8x+4x,       100 samples -> 156 cells

Usage:
    python experiments/cells.py --preset smoke --count
    python experiments/cells.py --preset smoke --list
    python experiments/cells.py --preset smoke --index 3
    python experiments/cells.py --preset smoke --index 3 --run

Run from the repo root (the parent of official/).
"""
import argparse
import shlex
import subprocess
import sys
from pathlib import Path

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
BUDGET = "head_budget_optimization/head_budgets/Qwen3-4B-Instruct-2507/optimized_agnostic.json"

ALL_TASKS = [
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery",
    "vt", "cwe", "fwe", "qa_1", "qa_2",
]
# ns1 = control (no UUID), ns3/nm3 = the UUID collapse, cwe = where AM wins.
DIAG_TASKS = ["niah_single_1", "niah_single_3", "niah_multikey_3", "cwe"]

# LCLM compression ratio -> retained fraction
RATIO_FRAC = {"16x": 0.0625, "8x": 0.125, "4x": 0.25}

# vllm max_model_len per context. Must cover repeat-prefill, which builds a
# "{C} Repeat the previous context. {C}" sequence ~2x the RULER context length.
# All values are < the model's native 262144, so no YaRN is triggered. Without
# this, vllm defaults to 262144 and OOMs its KV cache on smaller GPUs (e.g. L40S).
MAX_MODEL_LEN = {"4k": 16384, "8k": 24576, "16k": 40960}

# Baseline methods (original, no_context) are registered in the `summarize` config,
# not `fast`/`best`. This matches the paper's own qwen-ruler.sh baseline line.
BASELINE_ALGO_CONFIG = "summarize"

# smoke is deliberately minimal: just enough to prove the pipeline runs and to
# anchor the 4k-vs-16k cost ratio for extrapolation.
#   ns1 -> AM should score ~100, so it validates that AM *works*
#   nm3 -> AM should score ~0, which is the expected result and therefore CANNOT
#          validate anything on its own; ns1 is the real check.
PRESETS = {
    "smoke": dict(tasks=["niah_single_1", "niah_multikey_3"],
                  contexts=["4k", "16k"], ratios=["16x"], samples=4),
    "repro": dict(tasks=ALL_TASKS,  contexts=["4k"],              ratios=["16x", "8x"],       samples=50),
    "full":  dict(tasks=ALL_TASKS,  contexts=["4k", "8k", "16k"], ratios=["16x", "8x", "4x"], samples=50),
}

# AM-Fast = AM-HighestAttnKeys-fast: repeat-prefill queries, no self-study, no OMP.
# Same config the AM paper (Table 11) and LCLM used for RULER.
AM_METHOD = "highest_attn_keys_rms_nnls2_-3_3_lsq_on-policy"
AM_ALGO_CONFIG = "best"
AM_QUERY_CONFIG = "repeat"


def build_cells(preset):
    spec = PRESETS[preset]
    cells = []
    # Baselines first: they are cheap and establish the ceiling/floor.
    for ctx in spec["contexts"]:
        for task in spec["tasks"]:
            cells.append(dict(kind="baseline", ctx=ctx, ratio="full", task=task,
                              samples=spec["samples"], preset=preset))
    for ctx in spec["contexts"]:
        for ratio in spec["ratios"]:
            for task in spec["tasks"]:
                cells.append(dict(kind="am", ctx=ctx, ratio=ratio, task=task,
                                  samples=spec["samples"], preset=preset))
    return cells


def cell_command(cell):
    ctx, ratio, task = cell["ctx"], cell["ratio"], cell["task"]
    log_dir = f"../results/{cell['preset']}/{ctx}/{ratio}"
    args = [
        sys.executable, "-u", "-m", "evaluation.run_qa_evaluation",
        "--model-name", MODEL,
        "--dataset-name", f"ruler_{ctx}_{task}",
        "--n-articles", str(cell["samples"]),
        "--start-article", "0",
        "--log-dir", log_dir,
        "--name", task,
        "--max-model-len", str(MAX_MODEL_LEN[ctx]),  # bound vllm KV cache; covers repeat-prefill
        "--compute-perplexity", "0",   # separate experiment; keeps the sweep fast
        "--compute-stats", "0",        # can OOM on long contexts
    ]
    if cell["kind"] == "baseline":
        args += ["--methods", "original", "no_context",
                 "--algorithm-config", BASELINE_ALGO_CONFIG, "--target-size", "0.99"]
    else:
        args += [
            "--methods", AM_METHOD,
            "--target-size", str(RATIO_FRAC[ratio]),
            "--query-config", AM_QUERY_CONFIG,
            "--algorithm-config", AM_ALGO_CONFIG,
            "--precomputed-budget-path", BUDGET,
            "--max-ratio-per-head", "0.95",
        ]
    return args


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", default="smoke", choices=sorted(PRESETS))
    p.add_argument("--count", action="store_true")
    p.add_argument("--list", action="store_true")
    p.add_argument("--summary", action="store_true")
    p.add_argument("--index", type=int)
    p.add_argument("--run", action="store_true")
    a = p.parse_args()

    if a.summary:
        print(f"{'preset':8s} {'cells':>6s} {'samples':>8s} {'evals':>8s}  scope")
        for name, s in PRESETS.items():
            cells = build_cells(name)
            n_am = sum(1 for c in cells if c["kind"] == "am")
            n_bl = len(cells) - n_am
            evals = (n_am + 2 * n_bl) * s["samples"]
            print(f"{name:8s} {len(cells):6d} {s['samples']:8d} {evals:8d}  "
                  f"{len(s['tasks'])} tasks x {'+'.join(s['contexts'])} x {'+'.join(s['ratios'])}")
        return

    cells = build_cells(a.preset)

    if a.count:
        print(len(cells)); return
    if a.list:
        for i, c in enumerate(cells):
            print(f"{i:4d}  {c['kind']:8s} {c['ctx']:>3s} {c['ratio']:>4s}  {c['task']}")
        return
    if a.index is None:
        p.error("need --count, --list, --summary, or --index")
    if not (0 <= a.index < len(cells)):
        p.error(f"index {a.index} out of range 0..{len(cells)-1} for preset '{a.preset}'")

    cell = cells[a.index]
    cmd = cell_command(cell)

    if not a.run:
        print(" ".join(shlex.quote(x) for x in cmd)); return

    official = Path(__file__).resolve().parent.parent / "official"
    if not official.is_dir():
        sys.exit(f"official/ not found at {official}")
    print(f"[{a.preset} cell {a.index}] {cell['kind']} ctx={cell['ctx']} "
          f"ratio={cell['ratio']} task={cell['task']} n={cell['samples']}", flush=True)
    raise SystemExit(subprocess.call(cmd, cwd=official))


if __name__ == "__main__":
    main()
