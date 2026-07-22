#!/bin/bash
#SBATCH --job-name=ruler_am
#SBATCH --account=torch_pr_219_courant
#SBATCH --partition=h200_courant   # override: sbatch --partition=l40s_courant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/ruler_%x_%A_%a.out
#SBATCH --error=slurm_logs/ruler_%x_%A_%a.err
#
# RULER sweep — one array element per cell. Size is set by PRESET.
#
#   smoke   8 cells   ns1+nm3, 4k+16k, 16x,          4 samples  (pipeline check + timing anchor)
#   repro  39 cells  13 tasks, 4k, 16x+8x,            50 samples  (~the paper's own script)
#   full  156 cells  13 tasks, 3 lengths, 3 ratios,   50 samples
#
# Baselines always come first in the index order; AM cells follow.
# Check the layout before submitting:  python experiments/cells.py --preset X --list
#
# Usage:
#   sbatch --export=PRESET=smoke --array=0-7 --partition=l40s_courant slurm/ruler_sweep.sh
#   sbatch --export=PRESET=full  --array=0-155%8 slurm/ruler_sweep.sh   # throttle to 8 at once
#
# NOTE: cell indices are preset-specific. Always pass the same PRESET.

set -eo pipefail

NETID="mm14444"
SCRATCH="/scratch/${NETID}"
PROJECT_DIR="${SCRATCH}/AttentionMatching"

# Activate whichever env exists (conda preferred, venv fallback)
set +u
if [ -d "${SCRATCH}/conda_envs/am" ]; then
  eval "$(conda shell.bash hook)"
  conda activate "${SCRATCH}/conda_envs/am"
elif [ -d "${SCRATCH}/venvs/am" ]; then
  source "${SCRATCH}/venvs/am/bin/activate"
else
  echo "no env found: build one with env/setup_torch.sh or env/setup_venv.sh" >&2
  exit 1
fi
set -u

export HF_HOME="${SCRATCH}/hf_cache"
export HF_HUB_OFFLINE=1              # compute nodes have no internet — pre-stage first
export TOKENIZERS_PARALLELISM=false

cd "${PROJECT_DIR}"
mkdir -p slurm_logs results

PRESET="${PRESET:-smoke}"
IDX="${SLURM_ARRAY_TASK_ID:-${1:-}}"
if [ -z "$IDX" ]; then
  echo "usage: sbatch --export=PRESET=<p> --array=<n> $0   |   PRESET=<p> $0 <n>" >&2
  exit 2
fi

DESC="$(python experiments/cells.py --preset "$PRESET" --list | awk -v i="$IDX" '$1==i {$1=""; print substr($0,2)}')"

echo "=== RULER sweep [${PRESET}] cell ${IDX} ==="
echo "  cell:      ${DESC}"
echo "  job:       ${SLURM_JOB_ID:-local}"
echo "  node:      $(hostname)"
echo "  GPU:       $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo N/A)"
echo "  timestamp: $(date)"
echo ""
python experiments/cells.py --preset "$PRESET" --index "$IDX" | sed 's/^/  cmd: /'
echo ""

start=$(date +%s)
set +e
python experiments/cells.py --preset "$PRESET" --index "$IDX" --run
rc=$?
set -e
end=$(date +%s)
elapsed=$((end - start))

printf '\n=== cell %s done: rc=%d, %dh %dm %ds ===\n' \
  "$IDX" "$rc" $((elapsed/3600)) $(((elapsed%3600)/60)) $((elapsed%60))

# Central timing table for the benchmarking TODO
GPUNAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr ' ' '_')"
printf '%s\t%s\t%s\t%s\t%d\t%d\t%s\n' "$PRESET" "$IDX" "${DESC// /_}" "${GPUNAME:-unknown}" "$elapsed" "$rc" "$(date -Iseconds)" \
  >> results/timings.tsv

exit $rc
