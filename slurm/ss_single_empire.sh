#!/bin/bash
# Single-GPU runner: ONE cell on ONE GPU. gpu:1 gets force-routed to the RTX6000
# pool (separate from the H100/H200 nodes), and only needs a single GPU to free,
# so it backfills sooner than a gpu:2 job when the cluster is saturated. RTX PRO
# 6000 has 96GB -- ample for Qwen3-4B at 16k. Cell params via --export:
#   CTX TASK TS  [QCFG OUTROOT]
# Shares the same grid defaults as the pair job (chunked, ss-loss, cache-store, ...).
#SBATCH --job-name=ss_single
#SBATCH --account=ny_gdurrett_training
#SBATCH --partition=nyu
#SBATCH --qos=priority
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --output=slurm_logs/sssingle_%x_%j.out
#SBATCH --error=slurm_logs/sssingle_%x_%j.err
set -uo pipefail

SCRATCH="/mnt/home/DDN_Copy/nyu/mmehta"
PROJECT_DIR="/mnt/home/mmehta/AttentionMatching"
source "${SCRATCH}/miniforge3/etc/profile.d/conda.sh"
conda activate "${SCRATCH}/conda_envs/am"
export HF_HOME="${SCRATCH}/hf_cache" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "${PROJECT_DIR}"
mkdir -p slurm_logs results

echo "=== ss_single job=${SLURM_JOB_ID} node=$(hostname) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || echo "no GPU?"

CTX="${CTX:-8k}"; TASK="${TASK:-niah_multikey_3}"; TS="${TS:-0.0625}"
QCFG="${QCFG:-ss-plus-repeat}"; OUTROOT="${OUTROOT:-grid_smoke}"
VERBOSE="${VERBOSE:-1}"; STATS="${STATS:-1}"; N="${N:-3}"
SSLOSS="${SSLOSS:-1}"; GOLD_DIR="${GOLD_DIR:-../results/ss_gold}"
CHUNKING="${CHUNKING:-fixed}"; CHUNK_SIZE="${CHUNK_SIZE:-4096}"
NSAMP="${NSAMP:-5}"; GREEDY="${GREEDY:-1}"
CACHE_STORE="${CACHE_STORE:-../results/cache_store}"

echo ">>> CTX=${CTX} TASK=${TASK} TS=${TS} QCFG=${QCFG} OUTROOT=${OUTROOT}"
env CTX="$CTX" TASK="$TASK" TARGET_SIZE="$TS" QCFG="$QCFG" OUTROOT="$OUTROOT" \
  VERBOSE="$VERBOSE" STATS="$STATS" N="$N" SSLOSS="$SSLOSS" GOLD_DIR="$GOLD_DIR" \
  CHUNKING="$CHUNKING" CHUNK_SIZE="$CHUNK_SIZE" NSAMP="$NSAMP" GREEDY="$GREEDY" \
  CACHE_STORE="$CACHE_STORE" \
  bash experiments/run_repeat_loss.sh
echo "=== done rc=$? ==="
