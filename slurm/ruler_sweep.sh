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
# RULER sweep — one array element per cell.
#
#   cells   0-38   baselines (original + no_context), 13 tasks x 3 lengths
#   cells  39-155  AM-Fast, 13 tasks x 3 lengths x 3 ratios
#
# Usage:
#   sbatch --array=0-38   slurm/ruler_sweep.sh     # baselines first (cheap)
#   sbatch --array=39-155 slurm/ruler_sweep.sh     # then AM
#   sbatch --array=44     slurm/ruler_sweep.sh     # single cell (nm3 4k 16x)
#
# Throttle concurrent jobs with e.g. --array=39-155%8

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

IDX="${SLURM_ARRAY_TASK_ID:-${1:-}}"
if [ -z "$IDX" ]; then
  echo "usage: sbatch --array=<n> $0   |   $0 <n>" >&2
  exit 2
fi

DESC="$(python experiments/cells.py --list | awk -v i="$IDX" '$1==i {$1=""; print substr($0,2)}')"

echo "=== RULER sweep cell ${IDX} ==="
echo "  cell:      ${DESC}"
echo "  job:       ${SLURM_JOB_ID:-local}"
echo "  node:      $(hostname)"
echo "  GPU:       $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo N/A)"
echo "  timestamp: $(date)"
echo ""
python experiments/cells.py --index "$IDX" | sed 's/^/  cmd: /'
echo ""

start=$(date +%s)
set +e
python experiments/cells.py --index "$IDX" --run
rc=$?
set -e
end=$(date +%s)
elapsed=$((end - start))

printf '\n=== cell %s done: rc=%d, %dh %dm %ds ===\n' \
  "$IDX" "$rc" $((elapsed/3600)) $(((elapsed%3600)/60)) $((elapsed%60))

# Central timing table for the benchmarking TODO
printf '%s\t%s\t%d\t%d\t%s\n' "$IDX" "${DESC// /_}" "$elapsed" "$rc" "$(date -Iseconds)" \
  >> results/timings.tsv

exit $rc
