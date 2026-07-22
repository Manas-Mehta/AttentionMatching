#!/bin/bash
# Run this ON THE LOGIN NODE (needs internet). Compute nodes are offline, so all
# assets must be cached first. The login node kills big downloads partway, so we
# retry in a loop — snapshot_download resumes from the partial cache each time.
#
#   bash slurm/prestage.sh
#
set -eo pipefail

SCRATCH="/scratch/${USER}"
PY="${SCRATCH}/conda_envs/am/bin/python"     # use the am env explicitly

export HF_HOME="${SCRATCH}/hf_cache"
export TMPDIR="${SCRATCH}/tmp"; mkdir -p "${TMPDIR}"
export HF_HUB_ENABLE_HF_TRANSFER=0
unset HF_HUB_OFFLINE
echo "HF_HOME=${HF_HOME}"
echo "python: ${PY}"
echo

# --- datasets (small, one shot) ---
"${PY}" - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download("simonjegou/ruler", repo_type="dataset")
print("ruler dataset ->", p)
PY

# --- model (retry loop; resumes each time until complete) ---
echo
echo "Downloading model (will retry through login-node kills)..."
for attempt in $(seq 1 20); do
  if "${PY}" - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download("Qwen/Qwen3-4B-Instruct-2507",
                      max_workers=1)      # one file at a time -> lower memory
print("model ->", p)
PY
  then
    echo "model download complete on attempt ${attempt}"
    break
  else
    echo "attempt ${attempt} killed/failed — resuming..."
    sleep 3
  fi
done

echo
echo "Now verify:  ${PY} experiments/verify_assets.py"
