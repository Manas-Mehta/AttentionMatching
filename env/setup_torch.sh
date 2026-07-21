#!/bin/bash
# Build the env exactly as the upstream repo specifies: python 3.12 + official/requirements.txt
#   bash env/setup_torch.sh
set -eo pipefail

ENV_PATH="/scratch/${USER}/conda_envs/am"

set +u
eval "$(conda shell.bash hook)"
set -u

[ -d "${ENV_PATH}" ] || conda create -y -p "${ENV_PATH}" python=3.12

export PIP_CACHE_DIR="/scratch/${USER}/pip_cache"
export TMPDIR="/scratch/${USER}/tmp"
mkdir -p "${TMPDIR}"

"${ENV_PATH}/bin/pip" install --upgrade pip
"${ENV_PATH}/bin/pip" install -r official/requirements.txt

"${ENV_PATH}/bin/python" - <<'PY'
import torch, transformers
print(f"torch        {torch.__version__}")
print(f"transformers {transformers.__version__}")
print(f"cuda build   {torch.version.cuda}")
PY

echo
echo "Done: ${ENV_PATH}"
