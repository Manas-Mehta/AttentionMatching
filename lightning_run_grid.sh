#!/bin/bash
# Run the query-generalization grid on a 2-GPU Lightning Studio (no SLURM).
# 12 runs = {R, S, RS} x {ns3, nm3} x {4k, 8k}, 16x, chunked-4096, all capture on.
#
# Scheduled in 6 phases of 2 (one run per GPU). R rows run FIRST so the shared
# greedy self-study gold (GOLD_DIR, keyed by article) is minted once, then the
# S/RS rows of the same cell reuse it -- no duplicate gold generation.
#
#   bash lightning_run_grid.sh            # N=50 (full)
#   N=20 bash lightning_run_grid.sh       # smaller, cheaper
#
# Per-cache outputs: results/grid_R|grid_S|grid_RS / <ctx>/16x/<task>_*.json
set -uo pipefail
cd "$(dirname "$0")"

# shellcheck disable=SC1091
[ -d .venv ] && source .venv/bin/activate
export HF_HOME="${HF_HOME:-$(pwd)/hf_cache}"

N="${N:-50}"
TS=0.0625                    # 16x
export SSLOSS=1 STATS=1 VERBOSE=1 RLOSS=1
export CHUNKING=fixed CHUNK_SIZE=4096
export NSAMP=5 GREEDY=1
export GOLD_DIR="../results/ss_gold"
export CACHE_STORE="../results/cache_store"
mkdir -p results/logs

# one cell on one GPU: gpu qcfg outroot ctx task
run_cell () {
  local gpu="$1" qcfg="$2" outroot="$3" ctx="$4" task="$5"
  local log="results/logs/${outroot}_${ctx}_${task}.log"
  echo "  [GPU$gpu] $outroot $ctx $task -> $log"
  CUDA_VISIBLE_DEVICES="$gpu" env \
    CTX="$ctx" TASK="$task" TARGET_SIZE="$TS" QCFG="$qcfg" OUTROOT="$outroot" \
    N="$N" SSLOSS="$SSLOSS" STATS="$STATS" VERBOSE="$VERBOSE" RLOSS="$RLOSS" \
    CHUNKING="$CHUNKING" CHUNK_SIZE="$CHUNK_SIZE" NSAMP="$NSAMP" GREEDY="$GREEDY" \
    GOLD_DIR="$GOLD_DIR" CACHE_STORE="$CACHE_STORE" \
    bash experiments/run_repeat_loss.sh > "$log" 2>&1
}

# a phase = two cells, one per GPU, run concurrently
phase () {
  echo "=== phase: [$1 $4 $5] | [$6 $9 ${10}] ==="
  run_cell 0 "$2" "$3" "$4" "$5" & local p0=$!
  run_cell 1 "$7" "$8" "$9" "${10}" & local p1=$!
  wait $p0; local r0=$?
  wait $p1; local r1=$?
  echo "=== phase done: rc0=$r0 rc1=$r1 ==="
}

t0=$(date +%s)
#      GPU0: qcfg outroot ctx task   |  GPU1: qcfg outroot ctx task
phase repeat     grid_R  4k ns3        repeat     grid_R  4k nm3      # mint 4k golds
phase repeat     grid_R  8k ns3        repeat     grid_R  8k nm3      # mint 8k golds
phase self-study grid_S  4k ns3        ss-plus-repeat grid_RS 4k ns3
phase self-study grid_S  4k nm3        ss-plus-repeat grid_RS 4k nm3
phase self-study grid_S  8k ns3        ss-plus-repeat grid_RS 8k ns3
phase self-study grid_S  8k nm3        ss-plus-repeat grid_RS 8k nm3
echo "=== GRID DONE in $(( ($(date +%s)-t0)/60 )) min  (N=$N) ==="
