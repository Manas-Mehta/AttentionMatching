#!/bin/bash
#SBATCH --job-name=rploss
#SBATCH --account=torch_pr_219_courant
#SBATCH --partition=h200_courant   # override: sbatch --partition=h200_courant,l40s_courant,rtx6000_courant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/rploss_%x_%j.out
#SBATCH --error=slurm_logs/rploss_%x_%j.err
#
# Per-token reproduction loss + key selection, on the NYU torch cluster.
# Same job as slurm/repeat_loss_empire.sh; only account, partition and conda paths differ.
#
# Answers the professor's "different sloping lines" question: plot each token's
# loss of being reproduced at 4x / 8x / 16x, split by how many of the 288 heads
# kept that token's key.
#
#   sbatch --export=ALL,CTX=4k,TASK=niah_single_3,TARGET_SIZE=0.0625,N=20,\
#          VERBOSE=1,SSLOSS=0,OUTROOT=tokenloss slurm/repeat_loss_torch.sh
#
# SSLOSS=0 matters: left at its default of 1 the runner regenerates self-study gold
# answers with greedy HF decoding, four specs per article, because torch has no
# cached gold directory. That is minutes per article for output we do not use here.
set -eo pipefail

NETID="mm14444"
SCRATCH="/scratch/${NETID}"
PROJECT_DIR="${SLURM_SUBMIT_DIR:-${SCRATCH}/AttentionMatching}"

set +u                      # conda activate.d touches unbound vars
eval "$(conda shell.bash hook)"
conda activate "${SCRATCH}/conda_envs/am"
set -u

export HF_HOME="${SCRATCH}/hf_cache"
export HF_HUB_OFFLINE=1              # compute nodes have no internet — pre-stage first
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

echo "=== rploss  job=${SLURM_JOB_ID:-local}  node=$(hostname)  part=${SLURM_JOB_PARTITION:-?} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no GPU?"
echo "  CTX=${CTX:-4k}  TASK=${TASK:-niah_single_3}  TARGET_SIZE=${TARGET_SIZE:-0.0625}  N=${N:-50}"
echo "  VERBOSE=${VERBOSE:-0}  RLOSS=${RLOSS:-1}  SSLOSS=${SSLOSS:-1}  OUTROOT=${OUTROOT:-repeat_loss}"
echo ""

start=$(date +%s)
set +e
bash experiments/run_repeat_loss.sh
rc=$?
set -e
elapsed=$(( $(date +%s) - start ))

printf '\n=== done: rc=%d, %dh %dm %ds ===\n' "$rc" \
  $((elapsed/3600)) $(((elapsed%3600)/60)) $((elapsed%60))

GPUNAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr ' ' '_')"
printf 'rploss\t%s\t%s_%s_%s\t%s\t%d\t%d\t%s\n' \
  "${SLURM_JOB_ID:-local}" "${TASK:-d}" "${CTX:-4k}" "${TARGET_SIZE:-d}" \
  "${GPUNAME:-unknown}" "$elapsed" "$rc" "$(date -Iseconds)" \
  >> results/timings.tsv

exit $rc
