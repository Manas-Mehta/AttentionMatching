#!/bin/bash
# Repeat-prefill damage map.
#
# THE TODO: "Evaluate the loss over the RULER repeat-prefill queries using the AM
# compressed cache. Some tokens will have high loss, some low loss. What does that
# look like?"
#
# AM fits its compacted cache to queries harvested from ONE sequence:
#     formatted_context + "Repeat the previous context verbatim." + article
# This script scores that third segment token by token under three memories:
#     A: AM compacted cache   B: full context (ceiling)   C: no context (floor)
# and writes the three NLL curves per article to
#     <log-dir>/repeat_loss/<method>/article#####.npz
#
# Delta(i) = NLL_AM(i) - NLL_full(i)  isolates damage CAUSED by compaction from
# tokens that were merely unpredictable to begin with. The gold needle span is
# located in the same token indexing, so "is the needle a damage spike?" is a
# direct read-off (needle_delta_ratio in the JSON summary).
#
# Default cell = niah_single_3 @ 4k @ 16x -- the COLLAPSE cell (AM ~0% accuracy).
# For a sanity contrast run TARGET_SIZE=0.25 (4x), where AM is ~98% accurate and
# the damage curve should be nearly flat.
#
# Env overrides:  CTX=4k|8k|16k  TASK=niah_single_3|cwe|...  N=<#articles>
#                 TARGET_SIZE=0.25(4x)/0.125(8x)/0.0625(16x)  MAXLEN=<vllm max_model_len>
set -eo pipefail

CTX="${CTX:-4k}"
TASK="${TASK:-niah_single_3}"
TARGET_SIZE="${TARGET_SIZE:-0.0625}"   # 0.25 = 4x, 0.125 = 8x, 0.0625 = 16x
N="${N:-50}"
MAXLEN="${MAXLEN:-16384}"
CACHE_STORE="${CACHE_STORE:-}"

case "$TARGET_SIZE" in
  0.25)   RATIO=4x ;;
  0.125)  RATIO=8x ;;
  0.0625) RATIO=16x ;;
  *)      RATIO="ts${TARGET_SIZE}" ;;
esac

cd "$(dirname "$0")/../official"

echo "=== repeat-prefill damage map: ${TASK} ${CTX} ${RATIO}  (n=${N}) ==="
python -u -m evaluation.run_qa_evaluation \
  --model-name Qwen/Qwen3-4B-Instruct-2507 \
  --dataset-name "ruler_${CTX}_${TASK}" \
  --n-articles "${N}" --start-article 0 \
  --log-dir "../results/repeat_loss/${CTX}/${RATIO}" --name "${TASK}" \
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
  --compute-repeat-loss 1 \
  --compute-stats 0 \
  --cache-store-dir "${CACHE_STORE}"
