#!/bin/bash
#SBATCH --job-name=ruler-am
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --array=0-155
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#
# PLACEHOLDERS — fill in once we know the Torch cluster:
#   --partition=<...>
#   --account=<...>
#   --gres=gpu:<type>:1
#
# Cells 0-38   : baselines (original + no_context), 13 tasks x 3 lengths
# Cells 39-155 : AM-Fast, 13 tasks x 3 lengths x 3 ratios
#
# Run a subset with e.g.:  sbatch --array=0-38 slurm/ruler_sweep.sh   (baselines only)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs results

# --- environment (adjust for the cluster) ---
# module load cuda
# source ~/.bashrc
# conda activate am

export PYTHONHASHSEED=0          # RULER loader groups contexts by hash()
export TOKENIZERS_PARALLELISM=false
# export HF_HOME=/scratch/$USER/hf        # keep caches off the home quota
# export HF_HUB_OFFLINE=1                 # if compute nodes have no internet

IDX="${SLURM_ARRAY_TASK_ID:-${1:-}}"
if [ -z "$IDX" ]; then
  echo "usage: SLURM_ARRAY_TASK_ID=<n> $0   |   $0 <n>" >&2
  exit 2
fi

echo "=== cell $IDX ==="
echo "host: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python experiments/cells.py --index "$IDX" | sed 's/^/cmd: /'

start=$(date +%s)
python experiments/cells.py --index "$IDX" --run
rc=$?
end=$(date +%s)

elapsed=$((end - start))
printf 'cell %s finished rc=%d in %dh %dm %ds\n' \
  "$IDX" "$rc" $((elapsed/3600)) $(((elapsed%3600)/60)) $((elapsed%60))

# Append wall-clock to a central timing file for the benchmarking TODO
mkdir -p results
printf '%s\t%s\t%d\t%d\n' "$IDX" "$(python experiments/cells.py --index "$IDX" | md5sum | cut -c1-8)" "$elapsed" "$rc" \
  >> results/timings.tsv

exit $rc
