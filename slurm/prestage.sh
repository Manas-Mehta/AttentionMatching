#!/bin/bash
# Run this ON THE LOGIN NODE (it needs internet). Compute nodes run with
# HF_HUB_OFFLINE=1, so everything must be in the cache before any sbatch.
#
#   bash slurm/prestage.sh
#
set -euo pipefail

NETID="mm14444"
SCRATCH="/scratch/${NETID}"

eval "$(conda shell.bash hook)"
conda activate "${SCRATCH}/conda_envs/am"

export HF_HOME="${SCRATCH}/hf_cache"
unset HF_HUB_OFFLINE

echo "HF_HOME=${HF_HOME}"
echo

python - <<'PY'
import os
from huggingface_hub import snapshot_download
from datasets import load_dataset

MODEL = "Qwen/Qwen3-4B-Instruct-2507"

print(f"--- model: {MODEL} ---", flush=True)
p = snapshot_download(MODEL)
print(f"    cached at {p}\n", flush=True)

for cfg in ["4096", "8192", "16384"]:
    print(f"--- ruler config {cfg} ---", flush=True)
    ds = load_dataset("simonjegou/ruler", cfg, split="test")
    tasks = sorted(set(ds["task"]))
    print(f"    {len(ds)} rows, {len(tasks)} tasks", flush=True)
    assert len(ds) == 6500, f"expected 6500 rows, got {len(ds)}"
    assert len(tasks) == 13, f"expected 13 tasks, got {len(tasks)}"
print("\nAll assets cached.")
PY

echo
echo "Verifying offline load works (simulates a compute node)..."
HF_HUB_OFFLINE=1 python - <<'PY'
from datasets import load_dataset
from transformers import AutoConfig
AutoConfig.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
for cfg in ["4096", "8192", "16384"]:
    d = load_dataset("simonjegou/ruler", cfg, split="test")
    print(f"  ruler/{cfg}: {len(d)} rows  OK")
print("Offline load OK — safe to sbatch.")
PY
