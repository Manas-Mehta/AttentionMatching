#!/bin/bash
# One-time environment build on the Torch login node.
#   bash env/setup_torch.sh
set -euo pipefail

NETID="mm14444"
SCRATCH="/scratch/${NETID}"
ENV_PATH="${SCRATCH}/conda_envs/am"

eval "$(conda shell.bash hook)"

if [ -d "${ENV_PATH}" ]; then
  echo "env already exists at ${ENV_PATH}; activating"
else
  conda create -y -p "${ENV_PATH}" python=3.12
fi
conda activate "${ENV_PATH}"

export PIP_CACHE_DIR="${SCRATCH}/pip_cache"

# Pins from official/requirements.txt
pip install \
  "torch==2.8.0" \
  "transformers==4.57.1" \
  "vllm==0.11.0" \
  "accelerate==1.12.0" \
  "datasets==4.4.1"

python - <<'PY'
import torch, transformers
print(f"torch        {torch.__version__}")
print(f"transformers {transformers.__version__}")
print(f"cuda build   {torch.version.cuda}")
print(f"cuda avail   {torch.cuda.is_available()}  (False on a login node is expected)")
PY

echo
echo "Env ready: ${ENV_PATH}"
echo "Next: bash slurm/prestage.sh"
