#!/bin/bash
#SBATCH --job-name=chain
#SBATCH --account=ny_gdurrett_training
#SBATCH --partition=beta                # all B200; every QOS here forces a 4-GPU (whole-node) min
#SBATCH --qos=priority                  # 1000 (highest) + can preempt; beta is often 100+ deep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/chain_%x_%j.out
#SBATCH --error=slurm_logs/chain_%x_%j.err
#
# Phase 1 cell on Empire AI **beta** (B200 / aarch64), "Trading Memory for Compute".
# One job = one (dataset, compression, reasoning mode) cell. Params via --export.
# All the beta-specific env below was validated by the GB200 vLLM smoke (2026-09-21):
#   ARM venv am_arm, the complete DDN hf_cache, and FLASH_ATTN (no CUDA toolkit on nodes).
set -eo pipefail

DDN="/mnt/home/DDN_Copy/nyu/mmehta"
PROJECT_DIR="${SLURM_SUBMIT_DIR:-/mnt/home/mmehta/AttentionMatching}"

set +u                                  # venv activate touches unbound vars
source "${DDN}/venvs/am_arm/bin/activate"
set -u

export HF_HOME="${DDN}/hf_cache"        # the COMPLETE 7.6G snapshot (default ~/.cache is a broken partial)
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_ATTENTION_BACKEND=FLASH_ATTN   # B200 nodes have no nvcc -> FlashInfer JIT crashes; use precompiled
export CACHE_STORE="${CACHE_STORE:-}"

# Per-job compile caches (shared caches race on concurrent vLLM cold-starts -> corrupt .cubin).
JOBTMP="/mnt/home/mmehta/tmp/job_${SLURM_JOB_ID:-$$}"
export TMPDIR="${JOBTMP}"
export TRITON_CACHE_DIR="${JOBTMP}/triton"
export VLLM_CACHE_ROOT="${JOBTMP}/vllm"
export TORCHINDUCTOR_CACHE_DIR="${JOBTMP}/inductor"
export XDG_CACHE_HOME="${JOBTMP}/.cache"
export MPLCONFIGDIR="${JOBTMP}/mpl"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" \
         "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
trap 'rm -rf "${JOBTMP}" 2>/dev/null || true' EXIT

cd "${PROJECT_DIR}"
mkdir -p slurm_logs results

echo "=== chain  job=${SLURM_JOB_ID:-local}  node=$(hostname)  part=${SLURM_JOB_PARTITION:-?} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no GPU?"
echo "  TASK=${TASK:-default}  TARGET_SIZE=${TARGET_SIZE:-0.0625}  METHOD=${METHOD:-am}  QUERY_CONFIG=${QUERY_CONFIG:-repeat}"
echo "  COT_PROMPT=${COT_PROMPT:-none}  COT_MODE=${COT_MODE:-reason}  N=${N:-50}"
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
