#!/bin/bash
# Phase F: 3 missing variants (S_sigma10, F_patience30, F_patience300) in
# parallel on 64-core alloc. Each gets 20 workers = 60 cores total.
set -uo pipefail
cd /nas/longleaf/home/hisham26/Desktop/hyperlens
HPY=/nas/longleaf/home/hisham26/miniforge3/envs/hyperlens/bin/python
ALLOC=${1:?ALLOC required}
LOGDIR=results/phase_f_logs_$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOGDIR"
echo "log dir: $LOGDIR"

run_chain() {
    local TAG=$1
    local CFG=configs/ablations/sixlet_REFSWEEP2_${TAG}.yaml
    local SEARCH_LOG=$LOGDIR/${TAG}_search.log
    local REFINE_LOG=$LOGDIR/${TAG}_refine.log
    echo "[$(date +%H:%M:%S)] [search] $TAG"
    srun --jobid=$ALLOC --overlap --exact --ntasks=1 --cpus-per-task=20 --mem=20G \
        --chdir=$(pwd) \
        $HPY -m optical_rl.qd.lensmega.run --config $CFG --iterations 100000 \
        > "$SEARCH_LOG" 2>&1
    echo "[$(date +%H:%M:%S)] [search] $TAG done"
    local RUN_DIR=$(ls -td results/lensmega/sixlet_REFSWEEP2_${TAG}_seed42_* 2>/dev/null | head -1)
    if [[ -z "$RUN_DIR" ]]; then echo "  no run dir; skip refine"; return; fi
    local OUT=$RUN_DIR/refined_top500_seed42_${TAG}.json
    echo "[$(date +%H:%M:%S)] [refine] $TAG"
    srun --jobid=$ALLOC --overlap --exact --ntasks=1 --cpus-per-task=20 --mem=20G \
        --chdir=$(pwd) \
        $HPY tools/refine_all_parallel.py \
        --pool-from hl --hl-run "$RUN_DIR" \
        --threshold -1.0 --top-n 500 --workers 20 --out "$OUT" \
        > "$REFINE_LOG" 2>&1
    echo "[$(date +%H:%M:%S)] [refine] $TAG done"
}

run_chain S_sigma10 &     P1=$!
run_chain F_patience30 &  P2=$!
run_chain F_patience300 & P3=$!
wait $P1 $P2 $P3
echo "phase F complete"
