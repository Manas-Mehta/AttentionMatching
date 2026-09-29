#!/bin/bash
# Submit the Phase 1 method-A surface grid ("Trading Memory for Compute", section 5,
# prompt-mode variant). Replaces the fixed CoT-budget axis {0,64,256,1024} with four
# prompt modes; there is NO hard cap. Each cell's x-value is the MEASURED mean tokens
# it actually generated (read off afterwards), COT_CEILING is only a runaway guard.
#
#   Compression  {1x, 4x, 8x, 16x, 32x, 64x}
#   Prompt mode  {immediate, brief, moderate, long}
#
# 6 x 4 = 24 reasoning cells. This is PHASE 1. Filler is PHASE 2: once these finish,
# read the measured mean tokens per cell and submit matched-length filler twins with
# AM_COT_FILLER_TARGET set to each reason cell's own mean (immediate is ~0 = its own
# twin, so it is skipped). Pinning to this run's measured lengths is what makes the
# filler a real length-twin rather than an approximate one.
#
# Each cell also evaluates the per-hop probes that ship with every document, against
# that document's compacted cache (single-pass, pinned budget 0).
#
# Usage:
#   bash experiments/submit_mode_grid.sh torch           # NYU torch cluster
#   bash experiments/submit_mode_grid.sh empire          # Empire AI alpha1
#   DRYRUN=1 bash experiments/submit_mode_grid.sh torch  # print, do not submit
#
# Generate the dataset first, on a login node (needs the tokenizer):
#   python experiments/make_chain_data.py --hops 4 --chains 1 \
#          --names word --haystack prose --ctx 4096 --n 50
set -eo pipefail

SITE="${1:-torch}"
TASK="${TASK:-chain_d4_b1_word_prose_4k_last}"
N="${N:-50}"
OUTROOT="${OUTROOT:-chain_modeA}"
QUERY_CONFIG="${QUERY_CONFIG:-repeat}"   # repeat | self-study (self-study => set OUTROOT=chain_modeA_ss)
COT_CEILING="${COT_CEILING:-2048}"
MODES="${MODES:-immediate brief moderate long}"
DRYRUN="${DRYRUN:-0}"

# Torch reroutes nothing at submit time, so a partition list lets Slurm start the job
# wherever a GPU frees first. Empire routes a 1-GPU job to the slow RTX6000 pool, so
# request 2 to land on the free H100/H200 pool.
PARTS="${PARTS:-h200_courant,l40s_courant}"

case "$SITE" in
  torch)  SCRIPT=slurm/chain_torch.sh;  EXTRA="--partition=${PARTS}" ;;
  empire) SCRIPT=slurm/chain_empire.sh; EXTRA="--gres=gpu:2" ;;
  beta)   SCRIPT=slurm/chain_beta.sh;   EXTRA="--gres=gpu:4" ;;   # beta forces 4-GPU whole-node min
  *) echo "usage: $0 [torch|empire|beta]" >&2; exit 2 ;;
esac

# The 1x row is the uncompressed baseline: no compaction, generation through vLLM,
# which needs more host memory than the compacted cells.
BIG="--mem=120G --cpus-per-task=12"

submit() {   # $1 ratio-tag  $2 target-size  $3 method  $4 mode  $5 extra-sbatch-args
  local tag=$1 ts=$2 method=$3 mode=$4 extra=$5
  local name="m_${tag}_${mode:0:3}"
  local cmd="sbatch --job-name=${name} ${extra} --export=ALL,TASK=${TASK},TARGET_SIZE=${ts},METHOD=${method},COT_PROMPT=${mode},COT_MODE=reason,COT_CEILING=${COT_CEILING},N=${N},OUTROOT=${OUTROOT},QUERY_CONFIG=${QUERY_CONFIG} ${SCRIPT}"
  if [ "$DRYRUN" = "1" ]; then echo "$cmd"; else eval "$cmd"; fi
}

n=0
for cell in "1x 1.0 original ${BIG}" "4x 0.25 am" "8x 0.125 am" \
            "16x 0.0625 am" "32x 0.03125 am" "64x 0.015625 am"; do
  set -- $cell
  tag=$1; ts=$2; method=$3; shift 3; extra="$EXTRA $*"
  for mode in $MODES; do
    submit "$tag" "$ts" "$method" "$mode" "$extra"; n=$((n+1))
  done
done

echo ""
echo "${n} cells (expect 24)  task=${TASK}  n=${N}  modes=[${MODES}]  results -> results/${OUTROOT}/"
