#!/bin/bash
# One-shot environment setup for a Lightning AI Studio (or any single-GPU cloud box).
# Mirrors the proven Empire env (python 3.12 + official/requirements.txt) but with
# local paths and no SLURM. Idempotent: safe to re-run.
#
#   bash lightning_setup.sh
#
# Then run a cell with:
#   source .venv/bin/activate
#   CTX=8k TASK=niah_multikey_3 TARGET_SIZE=0.0625 QCFG=ss-plus-repeat \
#     OUTROOT=grid_smoke N=3 bash experiments/run_repeat_loss.sh
set -eo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"
echo "=== repo: $REPO_DIR ($(git rev-parse --abbrev-ref HEAD 2>/dev/null)@$(git rev-parse --short HEAD 2>/dev/null)) ==="

echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || { echo "NO GPU VISIBLE"; exit 1; }

# --- Python 3.12 venv (fall back to system python3 if 3.12 absent) ---
PYBIN="$(command -v python3.12 || command -v python3)"
echo "=== using $($PYBIN --version) at $PYBIN ==="
if [ ! -d .venv ]; then
  "$PYBIN" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip wheel

echo "=== installing pinned requirements (torch 2.8 / vllm 0.11 / transformers 4.57) ==="
pip install -r official/requirements.txt

# --- HF cache local to the studio; model + dataset are public, download on first run ---
export HF_HOME="${HF_HOME:-$REPO_DIR/hf_cache}"
mkdir -p "$HF_HOME" results
echo "HF_HOME=$HF_HOME"

echo "=== import + CUDA check ==="
python - <<'PY'
import torch, transformers, vllm
print("torch       ", torch.__version__, "| cuda avail:", torch.cuda.is_available(),
      "| device:", (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"))
print("transformers", transformers.__version__)
print("vllm        ", vllm.__version__)
assert torch.cuda.is_available(), "CUDA not available -- attach a GPU to the Studio"
PY

echo
echo "=== SETUP OK ==="
echo "Next (smoke test, ~N=3):"
echo "  source .venv/bin/activate && export HF_HOME=$HF_HOME"
echo "  CTX=8k TASK=niah_multikey_3 TARGET_SIZE=0.0625 QCFG=ss-plus-repeat OUTROOT=grid_smoke N=3 \\"
echo "    GOLD_DIR=../results/ss_gold CACHE_STORE=../results/cache_store \\"
echo "    bash experiments/run_repeat_loss.sh"
