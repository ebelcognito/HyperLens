#!/bin/bash
# Triplet APS-C re-run — fixes the sensor-class mismatch from §12 of the
# experiment log. Replaces the small-r_sensor cooke triplet
# (cooke_smaller_ap_1mm.json, r_sensor=5.175mm) with an APS-C cooke triplet
# (cooke_apsc_85mm_f4.json, r_sensor=17mm, foclen=85mm, F/4).
#
# Runs HL + SA+GA back-to-back on a single allocation, seed=42 only by default.
# Pass --seeds "42 43" to do both seeds (~150 min total).
#
# Usage:
#   bash run_triplet_apsc_redo.sh <ALLOC_JOBID> [seeds]
#
# Default: bash run_triplet_apsc_redo.sh 12345        # seed 42 only, ~75 min
#   Both: bash run_triplet_apsc_redo.sh 12345 "42 43" # ~150 min

set -uo pipefail
if [[ $# -lt 1 ]]; then
    echo "usage: $0 <ALLOC> [seeds]" >&2; exit 1
fi
ALLOC=$1
SEEDS=${2:-42}

cd /nas/longleaf/home/hisham26/Desktop/hyperlens
HPY=/nas/longleaf/home/hisham26/miniforge3/envs/hyperlens/bin/python
TS=$(date +%Y%m%d_%H%M%S)
ROOT="results/triplet_apsc_redo_${TS}"
LOGDIR="${ROOT}/logs"
CFGDIR="${ROOT}/configs"
mkdir -p "$LOGDIR" "$CFGDIR"

PROBLEM_CFG=optical_rl/qd/lensmega/config/examples/cooke_triplet_apsc_85mm_modernstack.yaml
SAGA_CFG=configs/sa_ga_baseline_norefine_100k.yaml   # use same SA+GA hyperparams
                                                     # as overnight (div=0.05) to
                                                     # match §12's protocol

log() { echo "[$(date +'%H:%M:%S')] $*" | tee -a "$LOGDIR/master.log"; }

log "===== TRIPLET APS-C RE-RUN on alloc=${ALLOC}  seeds=${SEEDS} ====="
log "  Problem cfg: ${PROBLEM_CFG}"
log "  SA+GA cfg:   ${SAGA_CFG}"

# ---- Phase 1: search (HL then SA+GA, sequential) ----
for SEED in $SEEDS; do
    # HL: clone problem cfg with seed override
    HL_CFG="${CFGDIR}/triplet_apsc_seed${SEED}.yaml"
    sed -e "s/cma_seed: 42/cma_seed: ${SEED}/" \
        -e "s/run_name: cooke_triplet_apsc_85mm_modernstack/run_name: cooke_triplet_apsc_85mm_modernstack_seed${SEED}/" \
        "$PROBLEM_CFG" > "$HL_CFG"

    log "----- HL search seed=${SEED} -----"
    srun --jobid="${ALLOC}" --overlap --exact \
         --cpus-per-task=32 --mem=48G \
         "$HPY" -m optical_rl.qd.lensmega.run \
            --config "$HL_CFG" --iterations 100000 \
         > "${LOGDIR}/triplet_seed${SEED}_hl.log" 2>&1
    HL_EXIT=$?
    if [[ $HL_EXIT -ne 0 ]]; then
        log "  HL exit=${HL_EXIT} (may be benign torch shutdown crash; check artifacts)"
    fi
    HL_DIR=$(ls -dt results/lensmega/cooke_triplet_apsc_85mm_modernstack_seed${SEED}_* 2>/dev/null | head -1)
    log "  HL dir: ${HL_DIR}"

    # SA+GA: same problem cfg, default baseline
    SAGA_OUT="${ROOT}/seed${SEED}/sa_ga"
    mkdir -p "$SAGA_OUT"
    log "----- SA+GA search seed=${SEED} -----"
    # patch SAGA seed in baseline cfg
    SAGA_SEED_CFG="${CFGDIR}/saga_seed${SEED}.yaml"
    sed -e "s/^seed: .*/seed: ${SEED}/" "$SAGA_CFG" > "$SAGA_SEED_CFG"
    srun --jobid="${ALLOC}" --overlap --exact \
         --cpus-per-task=32 --mem=48G \
         "$HPY" sa_ga_refine_baseline.py \
            --problem-config "$PROBLEM_CFG" \
            --baseline-config "$SAGA_SEED_CFG" \
            --budget 100000 \
            --workers 30 \
            --no-refine \
            --out "$SAGA_OUT" \
            > "${LOGDIR}/triplet_seed${SEED}_saga.log" 2>&1
    log "  SAGA done"

    # ---- Phase 2: refinement ----
    log "----- HL refine seed=${SEED} (top-500) -----"
    srun --jobid="${ALLOC}" --overlap --exact \
         --cpus-per-task=32 --mem=48G \
         "$HPY" tools/refine_all_parallel.py \
            --pool-from hl \
            --hl-run "$HL_DIR" \
            --problem-config "$HL_CFG" \
            --threshold -1.0 \
            --top-n 500 \
            --out "${HL_DIR}/refined_top500_seed${SEED}_hl.json" \
            --workers 30 \
            > "${LOGDIR}/triplet_seed${SEED}_refine_hl.log" 2>&1
    log "  HL refine done"

    log "----- SA+GA refine seed=${SEED} (all above -1.0) -----"
    srun --jobid="${ALLOC}" --overlap --exact \
         --cpus-per-task=32 --mem=48G \
         "$HPY" tools/refine_all_parallel.py \
            --pool-from sa_ga \
            --sa-ga-run "$SAGA_OUT" \
            --problem-config "$PROBLEM_CFG" \
            --threshold -1.0 \
            --out "${SAGA_OUT}/refined_above_-1.0_seed${SEED}_sa_ga.json" \
            --workers 30 \
            > "${LOGDIR}/triplet_seed${SEED}_refine_saga.log" 2>&1
    log "  SAGA refine done"
done

log "===== TRIPLET APS-C RE-RUN COMPLETE ====="
log "Run root: ${ROOT}"
log "HL run dirs: results/lensmega/cooke_triplet_apsc_85mm_modernstack_seed*_*"
log "SAGA refined JSONs: ${ROOT}/seed*/sa_ga/refined_above_-1.0_seed*_sa_ga.json"
