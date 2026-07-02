#!/bin/bash
# Phase E: cleanup pass on alloc 49195985 (32 cores).
# Step 1 — refine 4 already-saved archives (parallel pairs)
# Step 2 — run 3 missing searches (S_sigma10, F_patience30, F_patience300)
# Step 3 — refine those 3 new archives
set -uo pipefail
cd /nas/longleaf/home/hisham26/Desktop/hyperlens
HPY=/nas/longleaf/home/hisham26/miniforge3/envs/hyperlens/bin/python
ALLOC=${1:?ALLOC required}
LOGDIR=results/phase_e_logs_$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOGDIR"
echo "log dir: $LOGDIR"

refine_existing() {
    local TAG=$1
    local RUN_DIR=$2
    local OUT=$RUN_DIR/refined_top500_seed42_${TAG}.json
    local LOG=$LOGDIR/${TAG}_refine.log
    echo "[$(date +%H:%M:%S)] [refine] $TAG -> $OUT"
    srun --jobid=$ALLOC --overlap --exact --ntasks=1 --cpus-per-task=16 --mem=20G \
        --chdir=$(pwd) \
        $HPY tools/refine_all_parallel.py \
        --pool-from hl --hl-run "$RUN_DIR" \
        --threshold -1.0 --top-n 500 --workers 16 --out "$OUT" \
        > "$LOG" 2>&1
    echo "[$(date +%H:%M:%S)] [refine] $TAG done"
}

run_chain() {
    local TAG=$1
    local CFG=configs/ablations/sixlet_REFSWEEP2_${TAG}.yaml
    local SEARCH_LOG=$LOGDIR/${TAG}_search.log
    echo "[$(date +%H:%M:%S)] [search] $TAG"
    srun --jobid=$ALLOC --overlap --exact --ntasks=1 --cpus-per-task=16 --mem=20G \
        --chdir=$(pwd) \
        $HPY -m optical_rl.qd.lensmega.run --config $CFG --iterations 100000 \
        > "$SEARCH_LOG" 2>&1
    echo "[$(date +%H:%M:%S)] [search] $TAG done"
    local RUN_DIR=$(ls -td results/lensmega/sixlet_REFSWEEP2_${TAG}_seed42_* 2>/dev/null | head -1)
    if [[ -z "$RUN_DIR" ]]; then echo "  no run dir; skip refine"; return; fi
    refine_existing "$TAG" "$RUN_DIR"
}

# Step 1: refine 4 existing archives in parallel pairs
refine_existing G_grid1024 results/lensmega/sixlet_REFSWEEP2_G_grid1024_seed42_20260509_012705 &
P1=$!
refine_existing W_wider_bd  results/lensmega/sixlet_REFSWEEP2_W_wider_bd_seed42_20260509_015423 &
P2=$!
wait $P1 $P2

refine_existing T_tighter_bd results/lensmega/sixlet_REFSWEEP2_T_tighter_bd_seed42_20260509_015423 &
P3=$!
refine_existing P_pop36 results/lensmega/sixlet_REFSWEEP2_P_pop36_seed42_20260508_215842 &
P4=$!
wait $P3 $P4

# Step 2: searches (sigma10, patience30, patience300) — run in pairs
run_chain S_sigma10 &
P5=$!
run_chain F_patience30 &
P6=$!
wait $P5 $P6

run_chain F_patience300

echo "phase E complete"
