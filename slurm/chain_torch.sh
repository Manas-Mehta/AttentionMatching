#!/bin/bash
#SBATCH --job-name=chain
#SBATCH --account=torch_pr_219_courant
#SBATCH --partition=h200_courant   # override: sbatch --partition=l40s_courant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/chain_%x_%j.out
#SBATCH --error=slurm_logs/chain_%x_%j.err
#
# Phase 1 cell on the NYU torch cluster ("Trading Memory for Compute", section 5).
# Same job as slurm/chain_empire.sh; only the account, partition and conda paths differ.
#
# One job = one (dataset, compression, reasoning budget, reasoning mode) cell.
#
#   sbatch --export=ALL,TASK=chain_d4_b1_word_prose_4k_last,TARGET_SIZE=0.0625,\
#          COT_BUDGET=256,COT_MODE=reason,N=50 slurm/chain_torch.sh
#
# The 1x cells run the uncompressed baseline through vLLM and need more room:
#   sbatch --mem=120G --cpus-per-task=12 --export=ALL,...,METHOD=original,TARGET_SIZE=1.0 ...
#
# Generate the datasets once on a login node before submitting:
#   python experiments/make_chain_data.py --hops 4 --chains 1 --names word \
#          --haystack prose --ctx 4096 --n 50
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
