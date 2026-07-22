#!/bin/bash
# Run this ON THE LOGIN NODE (it needs internet). Compute nodes run with
# HF_HUB_OFFLINE=1, so everything must be cached first. Resumable — safe to
# re-run if the login node kills it partway.
#
#   bash slurm/prestage.sh
#
set -eo pipefail

SCRATCH="/scratch/${USER}"
ENV="${SCRATCH}/conda_envs/am"

export HF_HOME="${SCRATCH}/hf_cache"
export TMPDIR="${SCRATCH}/tmp"; mkdir -p "${TMPDIR}"
unset HF_HUB_OFFLINE
echo "HF_HOME=${HF_HOME}"
echo

# Download FILES only (snapshot_download), not load_dataset — building an in-memory
# arrow table for the 16k config is what likely tripped the login-node memory cap.
# snapshot_download streams to disk and resumes from partial cache.
"${ENV}/bin/python" - <<'PY'
from huggingface_hub import snapshot_download

print("--- ruler dataset (all configs, parquet) ---", flush=True)
p = snapshot_download("simonjegou/ruler", repo_type="dataset")
print("    ->", p, "\n", flush=True)

print("--- model Qwen/Qwen3-4B-Instruct-2507 ---", flush=True)
p = snapshot_download("Qwen/Qwen3-4B-Instruct-2507")
print("    ->", p, flush=True)

print("\nDownload complete. Verify with: python experiments/verify_assets.py", flush=True)
PY
