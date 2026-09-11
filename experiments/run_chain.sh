#!/bin/bash
# Phase 1 runner — "Trading Memory for Compute", section 5.
#
# One invocation = one cell of the surface: one dataset, one compression ratio, one
# reasoning budget, one reasoning mode. Every cell also evaluates the per-hop probes
# that ship with each document, against the same compacted cache, which is what
# produces the retained-but-latent labels.
#
# Env:
#   TASK          chain dataset name, i.e. data/chain/${TASK}.jsonl
#   TARGET_SIZE   1.0 | 0.25 | 0.125 | 0.0625 | 0.03125 | 0.015625  (1x..64x)
#   METHOD        am (attention matching) | original (uncompressed baseline)
#   COT_BUDGET    0 | 64 | 256 | 1024   decode cap for the reasoning stage
#   COT_MODE      reason | filler       filler = padding of the same token count
#   N             documents (default 50)
#   OUTROOT       results subdir (default chain)
#   STATS         1 to log attention-reconstruction loss (needed later for §6b)
set -eo pipefail

TASK="${TASK:-chain_d4_b1_word_prose_4k_last}"
TARGET_SIZE="${TARGET_SIZE:-0.0625}"
METHOD="${METHOD:-am}"
COT_BUDGET="${COT_BUDGET:-0}"
COT_MODE="${COT_MODE:-reason}"
COT_FILLER="${COT_FILLER:- ...}"
N="${N:-50}"
MAXLEN="${MAXLEN:-16384}"
OUTROOT="${OUTROOT:-chain}"
STATS="${STATS:-1}"
VERBOSE="${VERBOSE:-0}"
CHUNKING="${CHUNKING:-fixed}"
CHUNK_SIZE="${CHUNK_SIZE:-4096}"
CACHE_STORE="${CACHE_STORE:-}"

case "$TARGET_SIZE" in
  1.0|1)      RATIO=1x ;;
  0.25)       RATIO=4x ;;
  0.125)      RATIO=8x ;;
  0.0625)     RATIO=16x ;;
  0.03125)    RATIO=32x ;;
  0.015625)   RATIO=64x ;;
  *)          RATIO="ts${TARGET_SIZE}" ;;
esac

if [ "$METHOD" = "original" ]; then
  METHODS="original"
  RATIO=1x                      # no compaction happens; target-size is ignored
else
  METHODS="highest_attn_keys_rms_nnls2_-3_3_lsq_on-policy"
fi

# Cell tag: budget and mode are part of the identity, so nothing clobbers anything.
if [ "$COT_BUDGET" -eq 0 ]; then CELL="cot0"; else CELL="cot${COT_BUDGET}_${COT_MODE}"; fi

cd "$(dirname "$0")/../official"

# Read by evaluation/qa_evaluator.py. Section 5: decode cap + forced-answer suffix,
# and the filler control that separates re-derivation from raw added cache capacity.
export AM_COT_BUDGET="${COT_BUDGET}"
export AM_COT_MODE="${COT_MODE}"
export AM_COT_FILLER="${COT_FILLER}"

echo "=== chain cell: ${TASK} | ${RATIO} | ${CELL} | n=${N} ==="
echo "    method=${METHODS}"
echo "    AM_COT_BUDGET=${AM_COT_BUDGET} AM_COT_MODE=${AM_COT_MODE}"

python -u -m evaluation.run_qa_evaluation \
  --model-name Qwen/Qwen3-4B-Instruct-2507 \
  --dataset-name "${TASK}" \
  --n-articles "${N}" --start-article 0 \
  --log-dir "../results/${OUTROOT}/${TASK}/${RATIO}/${CELL}" --name "${TASK}" \
  --max-model-len "${MAXLEN}" \
  --max-new-tokens 128 \
  --methods ${METHODS} \
  --target-size "${TARGET_SIZE}" \
  --query-config repeat \
  --algorithm-config best \
  --precomputed-budget-path head_budget_optimization/head_budgets/Qwen3-4B-Instruct-2507/optimized_agnostic.json \
  --max-ratio-per-head 0.95 \
  --compute-perplexity 0 \
  --compute-gold-perplexity 0 \
  --chunking "${CHUNKING}" \
  --chunk-size "${CHUNK_SIZE}" \
  --compute-stats "${STATS}" \
  --verbose-logging "${VERBOSE}" \
  --cache-store-dir "${CACHE_STORE}"
