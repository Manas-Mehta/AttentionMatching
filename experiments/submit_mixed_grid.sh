#!/bin/bash
# Submit the §B8 mixed-dataset grid ("Trading Memory for Compute", Idea 1) on torch.
#
#   1x row      no compaction (METHOD=original, vLLM), shared by every query set
#   floor       1x, document replaced by a one-line stand-in (mixed_v2_4k_noctx), immediate
#   compressed  query set {R, SS} x ratio {4, 8, 16, 32, 64}x
#   modes       {immediate, moderate, long} (prompt modes, no hard cap; ceiling 2048);
#               brief added afterwards: MODES=brief ONLY=1x|compressed
#
# Dataset: mixed_v2_4k, all 100 docs, all 2,831 questions (logic included).
#
# Usage (from the repo root on torch):
#   DRYRUN=1 bash experiments/submit_mixed_grid.sh      # print, do not submit
#   ONLY=smoke bash experiments/submit_mixed_grid.sh    # 3 jobs, 2 docs: 1x / R 16x / SS 16x, long
#   bash experiments/submit_mixed_grid.sh               # the grid (34 jobs)
#   ONLY=1x | floor | compressed                        # one part only
#   ONLY=filler DRYRUN=1 ...                            # phase 2: matched-length filler twins (22)
#   ONLY=kvzip_smoke | kvzip                            # KVzip as published (METHOD=kvzip), on
#                                                       # ${TASK}_nologic: same docs, logic not asked;
#                                                       # 4 modes, 4/8/16x, moderate capped at 512
#   ONLY=logic_gate                                     # standalone logic datasets at 1x (10 jobs):
#                                                       # {logic_pw_4k, logic_sl_4k} x 4 modes, plus a
#                                                       # no-document floor each; 50 docs; moderate cap 512
#   ONLY=logic_smoke | logic_grid                       # ProofWriter (logic_pw_4k) compressed: smoke = 2 docs,
#                                                       # 16x long, R/SS/KVZ (3 jobs); grid = {R, SS, KVZ} x
#                                                       # LOGIC_RATIOS x 4 modes (36 jobs), 50 docs, moderate cap 512
set -eo pipefail

TASK="${TASK:-mixed_v2_4k}"
N="${N:-100}"
ONLY="${ONLY:-all}"                     # all | smoke | 1x | floor | compressed
QSETS="${QSETS:-R SS}"
RATIOS="${RATIOS:-4x 8x 16x 32x 64x}"
MODES="${MODES:-immediate moderate long}"
COT_CEILING="${COT_CEILING:-2048}"
DRYRUN="${DRYRUN:-0}"
# Torch has no priority QOS for our account; the queue levers are listing every GPU
# partition (courant first) and not over-asking --time. H200 only: on L40S (44 GB) the
# answer-stage prefill of a 16-question logic batch carrying ~2k-token CoTs OOMs
# (smoke job 18749736, "Tried to allocate 11.05 GiB"). Smaller batches would fit but add
# ~2 batches x 2,048 decode steps per doc.
PARTS="${PARTS:-h200_courant,h200_public}"
BIG="--mem=120G --cpus-per-task=12"     # the vLLM (1x) path needs more host memory

qconfig() { case $1 in
  R) echo repeat ;; SS) echo self-study ;; rj) echo repeat-json ;;
  ssnojson) echo leave-one-out/no-structure-json ;;
  *) echo "unknown query set $1" >&2; exit 2 ;; esac; }
tsize() { case $1 in
  4x) echo 0.25 ;; 8x) echo 0.125 ;; 16x) echo 0.0625 ;; 32x) echo 0.03125 ;;
  64x) echo 0.015625 ;; *) echo "unknown ratio $1" >&2; exit 2 ;; esac; }
# Limits from the H200 smoke (2 docs, 16x long): ~220-240 s/doc generation, i.e. ~7-9 h
# for 100 docs in long mode; moderate batches also hit the 2,048 cap (§B8g).
# Override from the smoke-run timings: T1X_*, TC_* env vars.
t1x() { case $1 in immediate) echo "${T1X_IMM:-01:00:00}" ;; brief) echo "${T1X_BRI:-01:30:00}" ;;
  moderate) echo "${T1X_MOD:-02:00:00}" ;; long) echo "${T1X_LONG:-03:00:00}" ;; esac; }
tc()  { case $1 in immediate) echo "${TC_IMM:-03:00:00}" ;; brief) echo "${TC_BRI:-04:00:00}" ;;
  moderate) echo "${TC_MOD:-12:00:00}" ;; long) echo "${TC_LONG:-14:00:00}" ;; esac; }

# KVzip cells (ONLY=kvzip_smoke|kvzip): all four modes, and moderate capped below long.
# In the AM grid both shared the 2,048 ceiling and cwe ran to it in both (findings §B8,
# 2026-09-29), so the modes did not differ in length there. Moderate's p90 outside cwe is
# 250-420 tokens (1x-16x), so 512 rarely binds except on cwe. Long keeps 2,048.
KVZ_RATIOS="${KVZ_RATIOS:-4x 8x 16x}"
# Time limits for compressed ProofWriter cells (50 docs, ~20 questions per doc in one batch);
# override from the logic smoke timings with TLG_* env vars.
tlg() { case $1 in immediate) echo "${TLG_IMM:-01:30:00}" ;; brief) echo "${TLG_BRI:-02:00:00}" ;;
  moderate) echo "${TLG_MOD:-04:00:00}" ;; long) echo "${TLG_LONG:-08:00:00}" ;; esac; }
KVZ_MODES="${KVZ_MODES:-immediate brief moderate long}"
kvz_cap() { case $1 in moderate) echo "${CAP_MOD:-512}" ;; *) echo "${COT_CEILING}" ;; esac; }

n=0
submit() {   # name  task  outroot  target-size  method  mode  query-config  time  extra  ndocs  [ceiling]
  local cmd="sbatch --job-name=$1 --partition=${PARTS} --time=$8 $9 --export=ALL,TASK=$2,TARGET_SIZE=$4,METHOD=$5,COT_PROMPT=$6,COT_MODE=reason,COT_CEILING=${11:-${COT_CEILING}},N=${10},OUTROOT=$3,QUERY_CONFIG=$7 slurm/chain_torch.sh"
  if [ "$DRYRUN" = "1" ]; then echo "$cmd"; else eval "$cmd"; fi
  n=$((n+1))
}

case "$ONLY" in
  smoke)
    submit mx_smk2_1x  "$TASK" mixed_smoke2    1.0    original long repeat     00:45:00 "$BIG" 2
    submit mx_smk2_R16 "$TASK" mixed_smoke2_R  0.0625 am       long repeat     01:00:00 ""     2
    submit mx_smk2_S16 "$TASK" mixed_smoke2_SS 0.0625 am       long self-study 01:00:00 ""     2
    ;;
  all|1x|floor|compressed)
    if [ "$ONLY" = "all" ] || [ "$ONLY" = "1x" ]; then
      for mode in $MODES; do
        submit "mx_1x_${mode:0:3}" "$TASK" mixed_1x 1.0 original "$mode" repeat "$(t1x $mode)" "$BIG" "$N"
      done
    fi
    if [ "$ONLY" = "all" ] || [ "$ONLY" = "floor" ]; then
      submit mx_floor_imm "${TASK}_noctx" mixed_1x 1.0 original immediate repeat 01:00:00 "$BIG" "$N"
    fi
    if [ "$ONLY" = "all" ] || [ "$ONLY" = "compressed" ]; then
      for qs in $QSETS; do
        for r in $RATIOS; do
          for mode in $MODES; do
            submit "mx_${qs}_${r}_${mode:0:3}" "$TASK" "mixed_${qs}" "$(tsize $r)" am "$mode" \
                   "$(qconfig $qs)" "$(tc $mode)" "" "$N"
          done
        done
      done
    fi
    ;;
  filler)
    # Matched-length filler twins (proposal §5 control), phase 2 of the grid. Each twin
    # replaces the CoT of its paired reason run with ' ...' padding of exactly that
    # question's measured CoT length (AM_COT_FILLER_FROM -> load_mixed_data), same cache
    # recipe, same prompt. immediate has no CoT, so it has no twin. Cells whose paired
    # run has no final JSON yet are skipped. Output: .../mode<mode>_filler/.
    # Guard: an older loader ignores AM_COT_FILLER_FROM and would pad with 0 tokens silently.
    if ! grep -q AM_COT_FILLER_FROM official/evaluation/datasets.py; then
      echo "official/evaluation/datasets.py lacks per-question filler (AM_COT_FILLER_FROM)" >&2
      [ "$DRYRUN" = "1" ] || exit 2
    fi
    fill() {   # name  outroot  target-size  method  mode  query-config  paired-dir  extra
      if ! ls "$7"/*.json >/dev/null 2>&1; then echo "skip $1: paired run not finished ($7)"; return; fi
      local cmd="sbatch --job-name=$1 --partition=${PARTS} --time=${TF:-03:00:00} $8 --export=ALL,TASK=${TASK},TARGET_SIZE=$3,METHOD=$4,COT_PROMPT=$5,COT_MODE=filler,COT_CEILING=${COT_CEILING},N=${N},OUTROOT=$2,QUERY_CONFIG=$6,AM_COT_FILLER_FROM=$7 slurm/chain_torch.sh"
      if [ "$DRYRUN" = "1" ]; then echo "$cmd"; else eval "$cmd"; fi
      n=$((n+1))
    }
    for mode in moderate long; do
      fill "mx_fill_1x_${mode:0:3}" mixed_1x 1.0 original "$mode" repeat \
           "$(pwd)/results/mixed_1x/${TASK}/1x/mode${mode}_reason" "$BIG"
    done
    for qs in $QSETS; do
      for r in $RATIOS; do
        for mode in moderate long; do
          fill "mx_fill_${qs}_${r}_${mode:0:3}" "mixed_${qs}" "$(tsize $r)" am "$mode" \
               "$(qconfig $qs)" "$(pwd)/results/mixed_${qs}/${TASK}/${r}/mode${mode}_reason" ""
        done
      done
    done
    ;;
  kvzip_smoke)
    # 2 docs, 16x, every mode: the port runs end to end, caps apply, result JSONs parse.
    for mode in $KVZ_MODES; do
      submit "mx_kvz_smk_${mode:0:3}" "${TASK}_nologic" mixed_KVZ_smoke 0.0625 kvzip "$mode" repeat \
             01:00:00 "" 2 "$(kvz_cap $mode)"
    done
    ;;
  kvzip)
    # KVzip (official/compaction/compaction_methods/kvzip.py), 4 modes x KVZ_RATIOS.
    # Logic questions are not asked; the logic block stays in every document.
    for r in $KVZ_RATIOS; do
      for mode in $KVZ_MODES; do
        submit "mx_KVZ_${r}_${mode:0:3}" "${TASK}_nologic" mixed_KVZ "$(tsize $r)" kvzip "$mode" \
               repeat "${TKVZ:-$(tc $mode)}" "" "$N" "$(kvz_cap $mode)"
      done
    done
    ;;
  logic_gate)
    # Standalone logic datasets (experiments/make_logic_data.py), uncompressed: can the model
    # do them at all, per mode, and does a no-document floor sit at the label share (~50%)?
    for lt in ${LOGIC_TASKS:-logic_pw_4k logic_sl_4k}; do
      tag=${lt#logic_}; tag=${tag%_4k}
      for mode in $KVZ_MODES; do
        submit "lg_${tag}_1x_${mode:0:3}" "$lt" logic_1x 1.0 original "$mode" repeat \
               "$(t1x $mode)" "$BIG" "${N_LOGIC:-50}" "$(kvz_cap $mode)"
      done
      submit "lg_${tag}_floor" "${lt}_noctx" logic_1x 1.0 original immediate repeat 01:00:00 "$BIG" "${N_LOGIC:-50}"
    done
    ;;
  logic_smoke|logic_grid)
    # Compressed ProofWriter cells. AM (R, SS) and KVzip on the same documents, all four modes,
    # moderate capped at 512 for every method (as in the 1x gate), so the modes are comparable.
    lt=${LOGIC_TASK:-logic_pw_4k}
    if [ "$ONLY" = "logic_smoke" ]; then lr="16x"; lm="long"; nd=2; root=logic_smoke
    else lr="${LOGIC_RATIOS:-4x 8x 16x}"; lm="$KVZ_MODES"; nd="${N_LOGIC:-50}"; root=logic; fi
    for r in $lr; do
      for mode in $lm; do
        for qs in R SS; do
          submit "lg_${qs}_${r}_${mode:0:3}" "$lt" "${root}_${qs}" "$(tsize $r)" am "$mode" \
                 "$(qconfig $qs)" "${TLG:-$(tlg $mode)}" "" "$nd" "$(kvz_cap $mode)"
        done
        submit "lg_KVZ_${r}_${mode:0:3}" "$lt" "${root}_KVZ" "$(tsize $r)" kvzip "$mode" repeat \
               "${TLG:-$(tlg $mode)}" "" "$nd" "$(kvz_cap $mode)"
      done
    done
    ;;
  *) echo "ONLY must be all|smoke|1x|floor|compressed|filler|kvzip_smoke|kvzip|logic_gate|logic_smoke|logic_grid" >&2; exit 2 ;;
esac

echo ""
case "$ONLY" in
  logic_smoke|logic_grid) echo "${n} jobs  only=${ONLY}  task=${LOGIC_TASK:-logic_pw_4k}  methods=[R SS KVZ]  moderate cap=$(kvz_cap moderate)" ;;
  logic_gate) echo "${n} jobs  only=logic_gate  tasks=[${LOGIC_TASKS:-logic_pw_4k logic_sl_4k}]  1x  modes=[${KVZ_MODES}]+floor  moderate cap=$(kvz_cap moderate)  docs=${N_LOGIC:-50}" ;;
  kvzip*) echo "${n} jobs  only=${ONLY}  task=${TASK}_nologic  ratios=[${KVZ_RATIOS}]  modes=[${KVZ_MODES}]  moderate cap=$(kvz_cap moderate)" ;;
  *) echo "${n} jobs  only=${ONLY}  task=${TASK}  qsets=[${QSETS}]  ratios=[${RATIOS}]  modes=[${MODES}]" ;;
esac
