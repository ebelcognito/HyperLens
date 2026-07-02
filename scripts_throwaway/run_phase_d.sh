#!/bin/bash
# Phase D: remaining variants on 64-core alloc 49196003.
# Runs pairs in parallel (30 workers each = 60 cores) to fit alloc budget.
set -uo pipefail
cd /nas/longleaf/home/hisham26/Desktop/hyperlens
HPY=/nas/longleaf/home/hisham26/miniforge3/envs/hyperlens/bin/python
ALLOC=${1:?ALLOC required}
LOGDIR=$(ls -td results/phase_d_logs_* 2>/dev/null | head -1)
if [[ -z "$LOGDIR" ]]; then
    LOGDIR=results/phase_d_logs_$(date +%Y%m%d_%H%M%S)
    mkdir -p "$LOGDIR"
fi
echo "log dir: $LOGDIR"

run_chain() {
    local TAG=$1
    local CFG=configs/ablations/sixlet_REFSWEEP2_${TAG}.yaml
    local SEARCH_LOG=$LOGDIR/${TAG}_search.log
    local REFINE_LOG=$LOGDIR/${TAG}_refine.log
    echo "[$(date +%H:%M:%S)] [search] $TAG"
    srun --jobid=$ALLOC --overlap --exact --ntasks=1 --cpus-per-task=30 --mem=24G \
        --chdir=$(pwd) \
        $HPY -m optical_rl.qd.lensmega.run --config $CFG --iterations 100000 \
        > "$SEARCH_LOG" 2>&1
    echo "[$(date +%H:%M:%S)] [search] $TAG done"
    local RUN_DIR=$(ls -td results/lensmega/sixlet_REFSWEEP2_${TAG}_seed42_* 2>/dev/null | head -1)
    if [[ -z "$RUN_DIR" ]]; then echo "  no run dir; skip refine"; return; fi
    local OUT=$RUN_DIR/refined_top500_seed42_${TAG}.json
    echo "[$(date +%H:%M:%S)] [refine] $TAG"
    srun --jobid=$ALLOC --overlap --exact --ntasks=1 --cpus-per-task=30 --mem=24G \
        --chdir=$(pwd) \
        $HPY tools/refine_all_parallel.py \
        --pool-from hl --hl-run "$RUN_DIR" \
        --threshold -1.0 --top-n 500 --workers 30 --out "$OUT" \
        > "$REFINE_LOG" 2>&1
    echo "[$(date +%H:%M:%S)] [refine] $TAG done"
}

# Batch 1: W_wider_bd ‖ T_tighter_bd
run_chain W_wider_bd &  P1=$!
run_chain T_tighter_bd & P2=$!
wait $P1 $P2

# Batch 2: P_pop36 ‖ S_sigma10
run_chain P_pop36 &  P3=$!
run_chain S_sigma10 & P4=$!
wait $P3 $P4

# Batch 3: F_patience30 ‖ F_patience300
run_chain F_patience30 &  P5=$!
run_chain F_patience300 & P6=$!
wait $P5 $P6

echo "phase D complete"
