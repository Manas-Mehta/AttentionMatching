#!/bin/bash
#SBATCH --job-name=pack
#SBATCH --account=ny_gdurrett_training
#SBATCH --partition=beta                # all B200; every QOS here forces a 4-GPU (whole-node) min
#SBATCH --qos=priority
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --gres=gpu:4
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/pack_%x_%j.out
#SBATCH --error=slurm_logs/pack_%x_%j.err
#
# Up to 4 independent cells on one beta node, one GPU each. Beta only hands out whole 4-GPU
# nodes, so one cell per job (slurm/chain_beta.sh) leaves 3 GPUs idle.
#
#   CELLS   cells separated by ';', each a space-separated list of VAR=value pairs passed to
#           experiments/run_chain.sh, e.g.
#           "TASK=logic_pw_4k TARGET_SIZE=0.0625 METHOD=am COT_PROMPT=long COT_CEILING=2048 N=50 OUTROOT=logic_R QUERY_CONFIG=repeat"
#           A cell "RUN=verify_kvzip N=2" runs experiments/verify_kvzip.py instead.
# Each cell logs to slurm_logs/pack_<job>_<i>.log; the job fails if any cell fails.
# Environment as in slurm/chain_beta.sh (validated by the GB200 vLLM smoke, 2026-09-21).
set -eo pipefail

DDN="/mnt/home/DDN_Copy/nyu/mmehta"
PROJECT_DIR="${SLURM_SUBMIT_DIR:-/mnt/home/mmehta/AttentionMatching}"

set +u
source "${DDN}/venvs/am_arm/bin/activate"
set -u

export HF_HOME="${DDN}/hf_cache"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

cd "${PROJECT_DIR}"
mkdir -p slurm_logs results

echo "=== pack  job=${SLURM_JOB_ID:-local}  node=$(hostname) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || echo "no GPU?"

IFS=';' read -r -a cells <<< "${CELLS:?CELLS is required}"
if [ "${#cells[@]}" -gt 4 ]; then echo "at most 4 cells per node" >&2; exit 2; fi

pids=()
for i in "${!cells[@]}"; do
  cell="$(echo "${cells[$i]}" | xargs)"
  log="slurm_logs/pack_${SLURM_JOB_ID:-local}_${i}.log"
  echo "  gpu ${i}: ${cell}  -> ${log}"
  (
    # per-cell compile caches: shared caches race on concurrent cold starts
    JOBTMP="/mnt/home/mmehta/tmp/job_${SLURM_JOB_ID:-$$}_${i}"
    export TMPDIR="${JOBTMP}" TRITON_CACHE_DIR="${JOBTMP}/triton" VLLM_CACHE_ROOT="${JOBTMP}/vllm" \
           TORCHINDUCTOR_CACHE_DIR="${JOBTMP}/inductor" XDG_CACHE_HOME="${JOBTMP}/.cache" \
           MPLCONFIGDIR="${JOBTMP}/mpl"
    mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" \
             "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
    export CUDA_VISIBLE_DEVICES="${i}"
    for kv in ${cell}; do export "${kv?}"; done
    start=$(date +%s)
    set +e
    if [ "${RUN:-}" = "verify_kvzip" ]; then
      (cd official && python -u ../experiments/verify_kvzip.py "${N:-2}")
    else
      bash experiments/run_chain.sh
    fi
    rc=$?
    set -e
    elapsed=$(( $(date +%s) - start ))
    printf '\n=== cell %d done: rc=%d, %dh %dm %ds ===\n' "$i" "$rc" \
      $((elapsed/3600)) $(((elapsed%3600)/60)) $((elapsed%60))
    rm -rf "${JOBTMP}" 2>/dev/null || true
    exit $rc
  ) > "${log}" 2>&1 &
  pids+=($!)
done

fail=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then echo "  cell ${i}: ok"; else echo "  cell ${i}: FAILED (see slurm_logs/pack_${SLURM_JOB_ID:-local}_${i}.log)"; fail=1; fi
done
exit $fail
