#!/bin/bash
#SBATCH --job-name=chain
#SBATCH --account=ny_gdurrett_training
#SBATCH --partition=nyu,alpha          # both map to alphagpu nodes; alpha adds 51-54
                                       # (RTX PRO 6000, 96GB). Slurm starts the job in
                                       # whichever partition frees a GPU first.
#SBATCH --qos=priority
#SBATCH --exclude=alphagpu01,alphagpu02,alphagpu06,alphagpu08,alphagpu10,alphagpu11,alphagpu20,alphagpu24
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00                # smoke default; override for the real grid
#SBATCH --output=slurm_logs/chain_%x_%j.out
#SBATCH --error=slurm_logs/chain_%x_%j.err
#
# Phase 1 cell on Empire AI ("Trading Memory for Compute", section 5).
#
# One job = one (dataset, compression, reasoning budget, reasoning mode) cell.
# Parameters come in as env vars through --export:
#
#   sbatch --export=ALL,TASK=chain_d4_b1_word_prose_4k_last,TARGET_SIZE=0.0625,\
#          COT_BUDGET=256,COT_MODE=reason,N=4 slurm/chain_empire.sh
#
# A short --time is deliberate: it keeps the job eligible for backfill, which on a
# fully-allocated cluster matters as much as the QOS.
set -eo pipefail

SCRATCH="/mnt/home/DDN_Copy/nyu/mmehta"
CONDA_ENV="${SCRATCH}/conda_envs/am"
PROJECT_DIR="${SLURM_SUBMIT_DIR:-/mnt/home/mmehta/AttentionMatching}"

set +u
source "${SCRATCH}/miniforge3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
set -u

export HF_HOME="${SCRATCH}/hf_cache"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CACHE_STORE="${CACHE_STORE:-}"

# Everything scratch-side. The login and compute nodes share a small /tmp ramdisk
# that other users fill, and torch/vllm/triton all spill compile caches there by
# default, which is a silent way to lose a long job to ENOSPC.
export TMPDIR="${SCRATCH}/tmp"
export TRITON_CACHE_DIR="${SCRATCH}/tmp/triton"
export VLLM_CACHE_ROOT="${SCRATCH}/tmp/vllm"
export TORCHINDUCTOR_CACHE_DIR="${SCRATCH}/tmp/inductor"
export XDG_CACHE_HOME="${SCRATCH}/.cache"
export MPLCONFIGDIR="${SCRATCH}/tmp/mpl"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" \
         "$XDG_CACHE_HOME" "$MPLCONFIGDIR"


cd "${PROJECT_DIR}"
mkdir -p slurm_logs results

echo "=== chain  job=${SLURM_JOB_ID:-local}  node=$(hostname)  part=${SLURM_JOB_PARTITION:-?} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no GPU?"
echo "  TASK=${TASK:-default}  TARGET_SIZE=${TARGET_SIZE:-0.0625}  METHOD=${METHOD:-am}"
echo "  COT_BUDGET=${COT_BUDGET:-0}  COT_MODE=${COT_MODE:-reason}  N=${N:-50}"
echo ""

start=$(date +%s)
set +e
bash experiments/run_chain.sh
rc=$?
set -e
elapsed=$(( $(date +%s) - start ))

printf '\n=== done: rc=%d, %dh %dm %ds ===\n' "$rc" \
  $((elapsed/3600)) $(((elapsed%3600)/60)) $((elapsed%60))

GPUNAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr ' ' '_')"
printf 'chain\t%s\t%s_%s_cot%s_%s\t%s\t%d\t%d\t%s\n' \
  "${SLURM_JOB_ID:-local}" "${TASK:-d}" "${TARGET_SIZE:-d}" "${COT_BUDGET:-0}" \
  "${COT_MODE:-reason}" "${GPUNAME:-unknown}" "$elapsed" "$rc" "$(date -Iseconds)" \
  >> results/timings.tsv

exit $rc
