#!/bin/bash
# Compressed ProofWriter cells (logic_pw_4k) on Empire beta, 4 cells per node
# (slurm/chain_beta_pack.sh). Same cells as `ONLY=logic_grid` in submit_mixed_grid.sh:
# {AM-R, AM-SS, KVzip} x {4, 8, 16}x x {immediate, brief, moderate (cap 512), long (cap 2048)}.
#
#   DRYRUN=1 bash experiments/submit_logic_beta.sh            # print, do not submit
#   ONLY=smoke bash experiments/submit_logic_beta.sh          # 1 node: R/SS/KVZ 16x long on 2 docs
#                                                             #   + the KVzip fp32 glue check on B200
#   ONLY=grid bash experiments/submit_logic_beta.sh           # 9 nodes, 36 cells, 50 docs
set -eo pipefail

ONLY="${ONLY:-grid}"
N="${N:-50}"
TASK="${TASK:-logic_pw_4k}"
DRYRUN="${DRYRUN:-0}"
MODES="immediate brief moderate long"

tsize() { case $1 in 4x) echo 0.25 ;; 8x) echo 0.125 ;; 16x) echo 0.0625 ;; esac; }
cap() { case $1 in moderate) echo 512 ;; *) echo 2048 ;; esac; }
# Node time = the slowest cell on it; override after the smoke with T_IMM / T_BRI / T_MOD / T_LONG.
tmode() { case $1 in immediate) echo "${T_IMM:-01:30:00}" ;; brief) echo "${T_BRI:-02:00:00}" ;;
  moderate) echo "${T_MOD:-04:00:00}" ;; long) echo "${T_LONG:-08:00:00}" ;; esac; }

cell() {   # method ratio mode ndocs outroot-prefix
  local m=$1 r=$2 mode=$3 nd=$4 root=$5 meth qc
  case $m in R) meth=am; qc=repeat ;; SS) meth=am; qc=self-study ;; KVZ) meth=kvzip; qc=repeat ;; esac
  echo "TASK=${TASK} TARGET_SIZE=$(tsize $r) METHOD=${meth} COT_PROMPT=${mode} COT_MODE=reason COT_CEILING=$(cap $mode) N=${nd} OUTROOT=${root}_${m} QUERY_CONFIG=${qc}"
}

n=0
node() {   # name time cells...
  local name=$1 time=$2; shift 2
  local cells; cells=$(IFS=';'; echo "$*")
  # CELLS goes through the environment (--export=ALL), not the --export list: sbatch splits that
  # list on commas and the cell strings carry '=' signs.
  local cmd=(sbatch --job-name="$name" --time="$time" --export=ALL slurm/chain_beta_pack.sh)
  if [ "$DRYRUN" = "1" ]; then printf 'CELLS=%q %s\n' "$cells" "${cmd[*]}"; else CELLS="$cells" "${cmd[@]}"; fi
  n=$((n+1))
}

case "$ONLY" in
  smoke)
    node lgb_smoke 01:30:00 "$(cell R 16x long 2 logic_smoke)" "$(cell SS 16x long 2 logic_smoke)" \
         "$(cell KVZ 16x long 2 logic_smoke)" "RUN=verify_kvzip N=2"
    ;;
  grid)
    # Per mode: 9 cells (3 methods x 3 ratios) -> two full nodes + one leftover; the four
    # leftovers (one per mode) share a node at the long-mode time limit.
    left=()
    for mode in $MODES; do
      cs=()
      for r in 4x 8x 16x; do for m in R SS KVZ; do cs+=("$(cell $m $r $mode $N logic)"); done; done
      node "lgb_${mode:0:3}_a" "$(tmode $mode)" "${cs[@]:0:4}"
      node "lgb_${mode:0:3}_b" "$(tmode $mode)" "${cs[@]:4:4}"
      left+=("${cs[8]}")
    done
    node lgb_mix "$(tmode long)" "${left[@]}"
    ;;
  *) echo "ONLY must be smoke|grid" >&2; exit 2 ;;
esac
echo ""
echo "${n} node jobs  only=${ONLY}  task=${TASK}  docs=${N}"
