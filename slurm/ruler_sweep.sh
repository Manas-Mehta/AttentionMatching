#!/bin/bash
#SBATCH --job-name=ruler_am
#SBATCH --account=torch_pr_219_courant
#SBATCH --partition=l40s_courant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/ruler_%x_%A_%a.out
#SBATCH --error=slurm_logs/ruler_%x_%A_%a.err
#
# RULER sweep — one array element per cell. Size is set by PRESET.
#
#   smoke   8 cells   4 tasks, 4k, 16x,          20 samples
#   small  24 cells   4 tasks, 4k+16k, 16x+4x,   50 samples
#   repro  39 cells  13 tasks, 4k, 16x+8x,       50 samples  (~the paper's own script)
#   full  156 cells  13 tasks, 3 lengths, 3 ratios, 100 samples
#
# Baselines always come first in the index order; AM cells follow.
# Check the layout before submitting:  python experiments/cells.py --preset X --list
#
# Usage:
#   sbatch --export=PRESET=smoke --array=0-7  slurm/ruler_sweep.sh
#   sbatch --export=PRESET=small --array=0-23 slurm/ruler_sweep.sh
#   sbatch --export=PRESET=small --array=0-23%4 slurm/ruler_sweep.sh   # throttle to 4 at once
#
# NOTE: cell indices are preset-specific. Always pass the same PRESET.

set -eo pipefail

NETID="mm14444"
SCRATCH="/scratch/${NETID}"
PROJECT_DIR="${SCRATCH}/AttentionMatching"

eval "$(conda shell.bash hook)"
conda activate "${SCRATCH}/conda_envs/am"

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
printf '%s\t%s\t%s\t%d\t%d\t%s\n' "$PRESET" "$IDX" "${DESC// /_}" "$elapsed" "$rc" "$(date -Iseconds)" \
  >> results/timings.tsv

exit $rc
