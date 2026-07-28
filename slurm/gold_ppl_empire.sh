#!/bin/bash
#SBATCH --job-name=gold_ppl
#SBATCH --account=ny_gdurrett_training
#SBATCH --partition=nyu               # nyu | alpha  -> same nodes (H100 alphagpu01-18)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1                  # 1 GPU is plenty: 4B model + 4k ctx
#SBATCH --time=00:45:00               # short: n=8 @ 4k finishes in a few min
#SBATCH --output=slurm_logs/gold_ppl_%j.out
#SBATCH --error=slurm_logs/gold_ppl_%j.err
#
# Gold-answer perplexity experiment on Empire AI (alpha1).
#
# Submit (defaults = ns3 @ 4k, 4x, n=8):
#   sbatch slurm/gold_ppl_empire.sh
#
# Other cells (override env):
#   sbatch --export=ALL,TARGET_SIZE=0.0625 slurm/gold_ppl_empire.sh          # ns3 4k 16x (collapse)
#   sbatch --export=ALL,CTX=16k,TARGET_SIZE=0.0625,MAXLEN=40960 slurm/gold_ppl_empire.sh
#   sbatch --export=ALL,TASK=niah_multikey_3,CTX=16k,TARGET_SIZE=0.0625,MAXLEN=40960 slurm/gold_ppl_empire.sh
#
# >>> VERIFY THESE TWO LINES match your Empire setup before first submit <<<
SCRATCH="/mnt/home/DDN_Copy/nyu/mmehta"  # miniforge, conda_envs, hf_cache live here
CONDA_ENV="${SCRATCH}/conda_envs/am"     # path OR name of the 'am' env
# the repo lives in HOME (/mnt/home/mmehta/AttentionMatching), not scratch. Use the
# dir sbatch was submitted from, falling back to HOME/AttentionMatching.
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$HOME/AttentionMatching}"

set -eo pipefail
set +u
# batch jobs don't source ~/.bashrc, so `conda` isn't on PATH — source it directly
source "${SCRATCH}/miniforge3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
set -u

export HF_HOME="${SCRATCH}/hf_cache"
export HF_HUB_OFFLINE=1               # compute nodes have no internet — pre-stage the model first
export HF_DATASETS_OFFLINE=1          # same for the RULER dataset load
export TOKENIZERS_PARALLELISM=false

cd "${PROJECT_DIR}"
mkdir -p slurm_logs results

echo "=== gold_ppl  job=${SLURM_JOB_ID:-local}  node=$(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no GPU?"
echo "  CTX=${CTX:-4k}  TASK=${TASK:-niah_single_3}  TARGET_SIZE=${TARGET_SIZE:-0.25}  N=${N:-8}"

bash experiments/run_gold_ppl.sh
