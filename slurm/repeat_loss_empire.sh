#!/bin/bash
#SBATCH --job-name=repeat_loss
#SBATCH --account=ny_gdurrett_training
#SBATCH --partition=nyu               # nyu | alpha  -> same nodes (H100 alphagpu01-18)
#SBATCH --qos=priority             # factor 1000 vs standard 500, max wall 1d; every run before 2026-08-12 used standard and queued for hours
#SBATCH --exclude=alphagpu01,alphagpu02,alphagpu06,alphagpu08,alphagpu10,alphagpu11,alphagpu20,alphagpu24  # known-bad nodes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00               # generous: 3 extra full forwards per article
#SBATCH --output=slurm_logs/repeat_loss_%j.out
#SBATCH --error=slurm_logs/repeat_loss_%j.err
#
# Repeat-prefill damage map on Empire AI (alpha1). See experiments/run_repeat_loss.sh
# for what is actually measured.
#
# Submit (defaults = ns3 @ 4k, 16x, n=50):
#   sbatch slurm/repeat_loss_empire.sh
#
# Smoke test first:
#   sbatch --export=ALL,N=2 slurm/repeat_loss_empire.sh
#
# Other cells:
#   sbatch --export=ALL,TASK=cwe slurm/repeat_loss_empire.sh                  # aggregate-task control
#   sbatch --export=ALL,TARGET_SIZE=0.25 slurm/repeat_loss_empire.sh          # 4x, AM works -> flat curve
#   sbatch --export=ALL,CTX=16k,MAXLEN=40960 slurm/repeat_loss_empire.sh
#
SCRATCH="/mnt/home/DDN_Copy/nyu/mmehta"
CONDA_ENV="${SCRATCH}/conda_envs/am"
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
export CACHE_STORE="${CACHE_STORE:-}"   # empty = do NOT save compacted caches (59-180 MB/article); set a path to opt in

cd "${PROJECT_DIR}"
mkdir -p slurm_logs results

echo "=== repeat_loss  job=${SLURM_JOB_ID:-local}  node=$(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no GPU?"
echo "  CTX=${CTX:-4k}  TASK=${TASK:-niah_single_3}  TARGET_SIZE=${TARGET_SIZE:-0.0625}  N=${N:-50}"

bash experiments/run_repeat_loss.sh
