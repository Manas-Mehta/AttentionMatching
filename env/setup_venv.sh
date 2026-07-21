#!/bin/bash
# Fallback env build: plain venv instead of conda.
# Use when `conda create` hangs on "Executing transaction" (slow scratch linking).
# Much faster: no conda solver, no per-file linking of a full python install.
#
#   bash env/setup_venv.sh
#
set -eo pipefail

NETID="mm14444"
SCRATCH="/scratch/${NETID}"
VENV="${SCRATCH}/venvs/am"

# Base python must be 3.12ish. Check what's available:
#   module avail python     (if the cluster uses modules)
PY="${PY:-python3}"
echo "base python: $($PY --version 2>&1) at $(command -v $PY)"

mkdir -p "$(dirname "${VENV}")"
if [ -d "${VENV}" ]; then
  echo "venv already exists at ${VENV}"
else
  "$PY" -m venv "${VENV}"
fi

set +u
source "${VENV}/bin/activate"
set -u

export PIP_CACHE_DIR="${SCRATCH}/pip_cache"
export TMPDIR="${SCRATCH}/tmp"
mkdir -p "${TMPDIR}"

pip install --upgrade pip

# Pins from official/requirements.txt.
# vllm is only needed for self-study query generation; we use --query-config repeat,
# so if it fights with torch you can drop it (see NOTE below).
pip install \
  "torch==2.8.0" \
  "transformers==4.57.1" \
  "accelerate==1.12.0" \
  "datasets==4.4.1"

# NOTE: install vllm separately so a failure here doesn't cost you the rest.
pip install "vllm==0.11.0" || {
  echo ""
  echo "!! vllm install failed — continuing without it."
  echo "   This is fine for the RULER sweep (--query-config repeat needs no self-study)."
}

python - <<'PY'
import torch, transformers
print(f"torch        {torch.__version__}")
print(f"transformers {transformers.__version__}")
print(f"cuda build   {torch.version.cuda}")
print(f"cuda avail   {torch.cuda.is_available()}  (False on a login node is expected)")
try:
    import vllm; print(f"vllm         {vllm.__version__}")
except Exception as e:
    print(f"vllm         not installed ({type(e).__name__}) — OK for this sweep")
PY

echo
echo "Env ready: ${VENV}"
echo "Activate with: source ${VENV}/bin/activate"
echo "Next: bash slurm/prestage.sh"
