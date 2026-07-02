#!/bin/bash
# Phase C: fitness_patience sweep on a separate alloc.
# Runs F_patience30 and F_patience300 in PARALLEL using 24 workers each
# on a 48-core alloc.
set -uo pipefail
cd /nas/longleaf/home/hisham26/Desktop/hyperlens
HPY=/nas/longleaf/home/hisham26/miniforge3/envs/hyperlens/bin/python
ALLOC=${1:?ALLOC required}
LOGDIR=results/phase_c_logs_$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOGDIR"
echo "log dir: $LOGDIR"

run_search_then_refine() {
    local TAG=$1
    local CFG=configs/ablations/sixlet_REFSWEEP2_${TAG}.yaml
    local SEARCH_LOG=$LOGDIR/${TAG}_search.log
    local REFINE_LOG=$LOGDIR/${TAG}_refine.log
    echo "[$(date +%H:%M:%S)] [search] $TAG"
    srun --jobid=$ALLOC --overlap --exact --ntasks=1 --cpus-per-task=24 --mem=24G \
        --chdir=$(pwd) \
        $HPY -m optical_rl.qd.lensmega.run --config $CFG --iterations 100000 \
        > "$SEARCH_LOG" 2>&1
    echo "[$(date +%H:%M:%S)] [search] $TAG done"

    local RUN_DIR=$(ls -td results/lensmega/sixlet_REFSWEEP2_${TAG}_seed42_* 2>/dev/null | head -1)
    if [[ -z "$RUN_DIR" ]]; then
        echo "  no run dir; skipping refine"
        return
    fi
    local OUT=$RUN_DIR/refined_top500_seed42_${TAG}.json
    echo "[$(date +%H:%M:%S)] [refine] $TAG -> $OUT"
    srun --jobid=$ALLOC --overlap --exact --ntasks=1 --cpus-per-task=24 --mem=24G \
        --chdir=$(pwd) \
        $HPY tools/refine_all_parallel.py \
        --pool-from hl --hl-run "$RUN_DIR" \
        --threshold -1.0 --top-n 500 \
        --workers 24 --out "$OUT" \
        > "$REFINE_LOG" 2>&1
    echo "[$(date +%H:%M:%S)] [refine] $TAG done"
}

run_search_then_refine F_patience30 &
PID1=$!
run_search_then_refine F_patience300 &
PID2=$!
wait $PID1; echo "patience30 chain exit=$?"
wait $PID2; echo "patience300 chain exit=$?"
echo "phase C complete"
