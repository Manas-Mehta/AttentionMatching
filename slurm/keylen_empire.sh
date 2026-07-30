#!/bin/bash
#SBATCH --job-name=keylen
#SBATCH --account=ny_gdurrett_training
#SBATCH --partition=nyu               # nyu | alpha  -> same nodes (H100 alphagpu01-18)
#SBATCH --exclude=alphagpu01,alphagpu02,alphagpu06,alphagpu08,alphagpu10,alphagpu11,alphagpu20,alphagpu24  # known-bad nodes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1                  # 1 GPU is plenty: 4B model + 4k ctx
#SBATCH --time=01:30:00               # 4k n=50 finishes in ~25 min; ~3x headroom guards a slow node
                                      # from a kill (job exits when done, so the window has no runtime cost)
#SBATCH --output=slurm_logs/keylen_%j.out
#SBATCH --error=slurm_logs/keylen_%j.err
#
# Key-length sweep ("UUID Bit" experiment) on Empire AI (alpha1).
#
# One cell = one (BITS, TARGET_SIZE) pair at 4k, n=50. Submit the whole 12-cell grid
# in parallel from a login node:
#
#   for B in 16 32 64 96; do
#     for TS in 0.25 0.125 0.0625; do
#       sbatch --export=ALL,BITS=$B,TARGET_SIZE=$TS slurm/keylen_empire.sh
#     done
#   done
#
# NOTE: run `python experiments/make_keylen_data.py` ONCE first if data/keylen/ is
# not present (it is committed, so normally you can skip this).
#
# >>> VERIFY THESE TWO LINES match your Empire setup before first submit <<<
SCRATCH="/mnt/home/DDN_Copy/nyu/mmehta"  # miniforge, conda_envs, hf_cache live here
CONDA_ENV="${SCRATCH}/conda_envs/am"     # path OR name of the 'am' env
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$HOME/AttentionMatching}"

set -eo pipefail
set +u
source "${SCRATCH}/miniforge3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
set -u

export HF_HOME="${SCRATCH}/hf_cache"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CACHE_STORE="${CACHE_STORE:-${SCRATCH}/compacted_cache_store}"

cd "${PROJECT_DIR}"
mkdir -p slurm_logs results

echo "=== keylen  job=${SLURM_JOB_ID:-local}  node=$(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no GPU?"
echo "  BITS=${BITS:-16}  TARGET_SIZE=${TARGET_SIZE:-0.25}  N=${N:-50}  (4k)"

bash experiments/run_keylen.sh
