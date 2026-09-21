#!/bin/bash
# Method-A pilot: run the four prompt modes on one cell, no hard cap, and read off
# the average tokens each mode actually generated. This is the professor's go/no-go
# gate before the full grid:
#   distinct, spaced averages  -> prompt modes work, run the grid
#   bunched averages / a short mode that runs long -> fall back to method B ("Wait")
#
# One cell = one (dataset, compression). The four modes run back to back.
#
#   TASK         chain dataset (data/chain/${TASK}.jsonl)
#   TARGET_SIZE  compression (1.0 | 0.25 | 0.125 | 0.0625 | 0.03125 | 0.015625)
#   METHOD       am | original    (original = uncompressed 1x baseline)
#   N            documents (default 20 for a pilot)
#   MODES        space list of modes (default "immediate brief moderate long")
#   COT_CEILING  runaway guard on the reasoning stage (default 2048)
#   OUTROOT      results subdir (default chain_pilot)
#
# Example (16x, 20 docs):
#   TASK=chain_d4_b1_word_prose_4k_last TARGET_SIZE=0.0625 N=20 \
#     bash experiments/run_mode_pilot.sh
set -eo pipefail

TASK="${TASK:-chain_d4_b1_word_prose_4k_last}"
TARGET_SIZE="${TARGET_SIZE:-0.0625}"
METHOD="${METHOD:-am}"
N="${N:-20}"
MODES="${MODES:-immediate brief moderate long}"
COT_CEILING="${COT_CEILING:-2048}"
OUTROOT="${OUTROOT:-chain_pilot}"

HERE="$(cd "$(dirname "$0")" && pwd)"

for mode in $MODES; do
  echo ""
  echo "############ pilot mode=${mode}  task=${TASK}  ts=${TARGET_SIZE} ############"
  TASK="$TASK" TARGET_SIZE="$TARGET_SIZE" METHOD="$METHOD" N="$N" \
    COT_PROMPT="$mode" COT_MODE=reason COT_CEILING="$COT_CEILING" \
    OUTROOT="$OUTROOT" \
    bash "${HERE}/run_chain.sh"
done

echo ""
echo "############ pilot summary ############"
python3 "${HERE}/summarize_mode_pilot.py" \
  --root "${HERE}/../results/${OUTROOT}/${TASK}" || true
