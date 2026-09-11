#!/bin/bash
# Submit the Phase 1 surface grid ("Trading Memory for Compute", section 5).
#
#   Compression  {1x, 4x, 8x, 16x, 32x, 64x}
#   CoT budget   {0, 64, 256, 1024}, decode cap + forced-answer suffix
#   Mode         real reasoning, plus matched filler at every non-zero budget
#
# 6 x 4 = 24 reasoning cells + 6 x 3 = 18 filler cells = 42 jobs.
# Filler at budget 0 is skipped: it is the same run as reasoning at budget 0.
#
# Each cell also evaluates the per-hop probes that ship with every document,
# against that document's own compacted cache. Those are what label an instance
# as a silent failure (every lookup recoverable, composed task fails) or a
# storage failure (a lookup not recoverable).
#
# Usage:
#   bash experiments/submit_chain_grid.sh torch          # NYU torch cluster
#   bash experiments/submit_chain_grid.sh empire         # Empire AI alpha1
#   DRYRUN=1 bash experiments/submit_chain_grid.sh torch # print, do not submit
#
# Generate the dataset first, on a login node (needs the tokenizer):
#   python experiments/make_chain_data.py --hops 4 --chains 1 \
#          --names word --haystack prose --ctx 4096 --n 50
set -eo pipefail

SITE="${1:-torch}"
TASK="${TASK:-chain_d4_b1_word_prose_4k_last}"
N="${N:-50}"
OUTROOT="${OUTROOT:-chain_phase1}"
DRYRUN="${DRYRUN:-0}"

case "$SITE" in
  torch)  SCRIPT=slurm/chain_torch.sh;  EXTRA="" ;;
  empire) SCRIPT=slurm/chain_empire.sh; EXTRA="" ;;
  *) echo "usage: $0 [torch|empire]" >&2; exit 2 ;;
esac

# The 1x row is the uncompressed baseline: no compaction, generation through vLLM,
# which needs more host memory than the compacted cells.
BIG="--mem=120G --cpus-per-task=12"

submit() {   # $1 ratio-tag  $2 target-size  $3 method  $4 budget  $5 mode  $6 extra-sbatch-args
  local tag=$1 ts=$2 method=$3 budget=$4 mode=$5 extra=$6
  local name
  if [ "$budget" -eq 0 ]; then name="g_${tag}_cot0"; else name="g_${tag}_${budget}${mode:0:1}"; fi
  local cmd="sbatch --job-name=${name} ${extra} --export=ALL,TASK=${TASK},TARGET_SIZE=${ts},METHOD=${method},COT_BUDGET=${budget},COT_MODE=${mode},N=${N},OUTROOT=${OUTROOT} ${SCRIPT}"
  if [ "$DRYRUN" = "1" ]; then echo "$cmd"; else eval "$cmd"; fi
}

n=0
for cell in "1x 1.0 original ${BIG}" "4x 0.25 am" "8x 0.125 am" \
            "16x 0.0625 am" "32x 0.03125 am" "64x 0.015625 am"; do
  set -- $cell
  tag=$1; ts=$2; method=$3; shift 3; extra="$*"
  for budget in 0 64 256 1024; do
    submit "$tag" "$ts" "$method" "$budget" reason "$extra"; n=$((n+1))
    if [ "$budget" -ne 0 ]; then
      submit "$tag" "$ts" "$method" "$budget" filler "$extra"; n=$((n+1))
    fi
  done
done

echo ""
echo "${n} cells (expect 42)  task=${TASK}  n=${N}  results -> results/${OUTROOT}/"
