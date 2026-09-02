#!/bin/bash
# Run TWO self-study cells in one 2-GPU allocation (one cell per GPU).
# Requesting gpu:2 lands on H100/H200 (dodges the single-GPU -> RTX6000 governance)
# and both GPUs stay busy. Cell params come via --export:
#   CTX1 TASK1 TS1  (GPU 0)   and   CTX2 TASK2 TS2 (GPU 1, optional)
# Shared: QCFG OUTROOT VERBOSE STATS N
#SBATCH --job-name=ss_pair
#SBATCH --account=ny_gdurrett_training
#SBATCH --partition=nyu
#SBATCH --qos=priority
#SBATCH --exclude=alphagpu01,alphagpu02,alphagpu06,alphagpu08,alphagpu10,alphagpu11,alphagpu20,alphagpu24
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --gres=gpu:2
#SBATCH --time=03:30:00
#SBATCH --output=slurm_logs/sspair_%x_%j.out
#SBATCH --error=slurm_logs/sspair_%x_%j.err
set -uo pipefail

SCRATCH="/mnt/home/DDN_Copy/nyu/mmehta"
PROJECT_DIR="/mnt/home/mmehta/AttentionMatching"
source "${SCRATCH}/miniforge3/etc/profile.d/conda.sh"
conda activate "${SCRATCH}/conda_envs/am"
export HF_HOME="${SCRATCH}/hf_cache" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CACHE_STORE=""
cd "${PROJECT_DIR}"
mkdir -p slurm_logs results

echo "=== ss_pair job=${SLURM_JOB_ID} node=$(hostname) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || echo "no GPU?"

QCFG="${QCFG:-ss-plus-repeat}"; OUTROOT="${OUTROOT:-ss_sweep}"
VERBOSE="${VERBOSE:-1}"; STATS="${STATS:-1}"; N="${N:-50}"

echo ">>> GPU0: CTX=${CTX1} TASK=${TASK1} TS=${TS1}"
CUDA_VISIBLE_DEVICES=0 env CTX="$CTX1" TASK="$TASK1" TARGET_SIZE="$TS1" \
  QCFG="$QCFG" OUTROOT="$OUTROOT" VERBOSE="$VERBOSE" STATS="$STATS" N="$N" \
  bash experiments/run_repeat_loss.sh > "slurm_logs/cellA_${SLURM_JOB_ID}.log" 2>&1 &
PIDA=$!

PIDB=""
if [ -n "${TASK2:-}" ]; then
  echo ">>> GPU1: CTX=${CTX2} TASK=${TASK2} TS=${TS2}"
  CUDA_VISIBLE_DEVICES=1 env CTX="$CTX2" TASK="$TASK2" TARGET_SIZE="$TS2" \
    QCFG="$QCFG" OUTROOT="$OUTROOT" VERBOSE="$VERBOSE" STATS="$STATS" N="$N" \
    bash experiments/run_repeat_loss.sh > "slurm_logs/cellB_${SLURM_JOB_ID}.log" 2>&1 &
  PIDB=$!
fi

wait "$PIDA"; RA=$?
RB=0; [ -n "$PIDB" ] && { wait "$PIDB"; RB=$?; }
echo "=== done: cellA(${TASK1}/${TS1}) rc=$RA  cellB(${TASK2:-none}/${TS2:-}) rc=$RB ==="
[ "$RA" -gt "$RB" ] && exit "$RA" || exit "$RB"
