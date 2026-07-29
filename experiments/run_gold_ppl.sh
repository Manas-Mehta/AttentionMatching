#!/bin/bash
# Gold-answer perplexity experiment.
#
# Teacher-forces the GOLD needle answer (the true UUID) and measures its
# perplexity under three memories:
#   full context  (ceiling, expect ~1)   |  AM-compressed cache  |  no context (floor, ~16/hex)
# plus a per-gold-token NLL profile that localizes where the key gets corrupted.
#
# Default cell = niah_single_3 (ns3) @ 4k context, 4x compression. This is the
# EASY cell (AM ~98% accuracy here), so AM perplexity should come out ~1 — a
# sanity check that the teacher-forcing is correct before we point it at the
# collapse cells (ns3/nm3 @ 16x).
#
# Env overrides:  CTX=4k|8k|16k  TASK=niah_single_3  TARGET_SIZE=0.25(4x)/0.125(8x)/0.0625(16x)
#                 N=<#articles>  MAXLEN=<vllm max_model_len>
set -eo pipefail

CTX="${CTX:-4k}"
TASK="${TASK:-niah_single_3}"
TARGET_SIZE="${TARGET_SIZE:-0.25}"     # 0.25 = 4x, 0.125 = 8x, 0.0625 = 16x
N="${N:-10}"
MAXLEN="${MAXLEN:-16384}"
CACHE_STORE="${CACHE_STORE:-}"         # if set, save each compacted cache here for reuse

# ratio label for the output dir (0.25->4x etc.)
case "$TARGET_SIZE" in
  0.25)   RATIO=4x ;;
  0.125)  RATIO=8x ;;
  0.0625) RATIO=16x ;;
  *)      RATIO="ts${TARGET_SIZE}" ;;
esac

cd "$(dirname "$0")/../official"

echo "=== gold-perplexity: ${TASK} ${CTX} ${RATIO}  (n=${N}) ==="
python -u -m evaluation.run_qa_evaluation \
  --model-name Qwen/Qwen3-4B-Instruct-2507 \
  --dataset-name "ruler_${CTX}_${TASK}" \
  --n-articles "${N}" --start-article 0 \
  --log-dir "../results/gold_ppl/${CTX}/${RATIO}" --name "${TASK}" \
  --max-model-len "${MAXLEN}" \
  --max-new-tokens 128 \
  --methods highest_attn_keys_rms_nnls2_-3_3_lsq_on-policy \
  --target-size "${TARGET_SIZE}" \
  --query-config repeat \
  --algorithm-config best \
  --precomputed-budget-path head_budget_optimization/head_budgets/Qwen3-4B-Instruct-2507/optimized_agnostic.json \
  --max-ratio-per-head 0.95 \
  --compute-perplexity 0 \
  --compute-gold-perplexity 1 \
  --compute-stats 0 \
  --cache-store-dir "${CACHE_STORE}"
