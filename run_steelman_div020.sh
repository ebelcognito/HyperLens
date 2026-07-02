#!/bin/bash
# Steelman SA+GA replication with QGSO-default div_rate=0.20.
# Reuses overnight HL pools (no HL re-run needed).
#
# Usage: bash run_steelman_div020.sh <ALLOC_JOBID>
#
# Time: ~75 min for all 3 systems × seed 42 (search + refinement).

set -uo pipefail
if [[ $# -lt 1 ]]; then echo "usage: $0 <ALLOC>" >&2; exit 1; fi
ALLOC=$1
cd /nas/longleaf/home/hisham26/Desktop/hyperlens
HPY=/nas/longleaf/home/hisham26/miniforge3/envs/hyperlens/bin/python
TS=$(date +%Y%m%d_%H%M%S)
ROOT="results/steelman_div020_${TS}"
LOGDIR="${ROOT}/logs"
mkdir -p "$LOGDIR"

CFG_BASE=configs/sa_ga_baseline_norefine_100k_div020.yaml
declare -A SYS_CFG
SYS_CFG[triplet]=optical_rl/qd/lensmega/config/examples/cooke_triplet_materials_modernstack.yaml
SYS_CFG[sixlet]=optical_rl/qd/lensmega/config/examples/sixlet_apsc_discovery_v2.yaml
SYS_CFG[octuplet]=optical_rl/qd/lensmega/config/examples/octuplet_apsc_2cement_v2.yaml

log() { echo "[$(date +'%H:%M:%S')] $*" | tee -a "$LOGDIR/master.log"; }

log "===== STEELMAN SA+GA (div_rate=0.20) on alloc ${ALLOC} ====="

# ---- Phase 1: search (sequential, single allocation) ----
for SYS in triplet sixlet octuplet; do
    OUTDIR="${ROOT}/${SYS}/seed42/sa_ga"
    mkdir -p "$OUTDIR"
    log "----- search ${SYS} seed=42 -----"
    srun --jobid="${ALLOC}" --overlap --exact \
         --cpus-per-task=32 --mem=48G \
         "$HPY" sa_ga_refine_baseline.py \
            --problem-config "${SYS_CFG[$SYS]}" \
            --baseline-config "$CFG_BASE" \
            --budget 100000 \
            --workers 30 \
            --no-refine \
            --out "$OUTDIR" \
            > "${LOGDIR}/${SYS}_seed42_saga.log" 2>&1
    if [[ $? -eq 0 ]]; then log "  ${SYS} search done"; else log "  ${SYS} search FAILED"; fi
done

# ---- Phase 2: refinement (sequential) ----
for SYS in triplet sixlet octuplet; do
    OUTDIR="${ROOT}/${SYS}/seed42/sa_ga"
    log "----- refine ${SYS} seed=42 -----"
    srun --jobid="${ALLOC}" --overlap --exact \
         --cpus-per-task=32 --mem=48G \
         "$HPY" tools/refine_all_parallel.py \
            --pool-from sa_ga \
            --sa-ga-run "$OUTDIR" \
            --problem-config "${SYS_CFG[$SYS]}" \
            --threshold -1.0 \
            --out "${OUTDIR}/refined_above_-1.0_seed42_sa_ga.json" \
            --workers 30 \
            > "${LOGDIR}/${SYS}_seed42_refine_saga.log" 2>&1
    if [[ $? -eq 0 ]]; then log "  ${SYS} refine done"; else log "  ${SYS} refine FAILED"; fi
done

log "===== STEELMAN COMPLETE ====="
log "Run root: ${ROOT}"
log "Logs: ${LOGDIR}/"
