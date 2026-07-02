#!/bin/bash
# Phase B: search-side sweeps. For each variant, run HL search at 100k iter,
# then refine top-500 with the original (matched) refine config.
set -uo pipefail
cd /nas/longleaf/home/hisham26/Desktop/hyperlens
HPY=/nas/longleaf/home/hisham26/miniforge3/envs/hyperlens/bin/python
ALLOC=${1:-49179086}
LOGDIR=results/phase_b_logs_$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOGDIR"
echo "log dir: $LOGDIR"

for TAG in G_grid128 G_grid1024 W_wider_bd T_tighter_bd P_pop36 S_sigma10; do
    CFG=configs/ablations/sixlet_REFSWEEP2_${TAG}.yaml
    SEARCH_LOG=$LOGDIR/${TAG}_search.log
    REFINE_LOG=$LOGDIR/${TAG}_refine.log
    echo "[$(date +%H:%M:%S)] [search] $TAG"
    srun --jobid=$ALLOC --overlap --exact --ntasks=1 --cpus-per-task=30 --mem=40G \
        --chdir=$(pwd) \
        $HPY -m optical_rl.qd.lensmega.run --config $CFG --iterations 100000 \
        > "$SEARCH_LOG" 2>&1
    echo "[$(date +%H:%M:%S)] [search] $TAG done"

    RUN_DIR=$(ls -td results/lensmega/sixlet_REFSWEEP2_${TAG}_seed42_* 2>/dev/null | head -1)
    if [[ -z "$RUN_DIR" ]]; then
        echo "  no run dir found, skipping refine"
        continue
    fi
    OUT=$RUN_DIR/refined_top500_seed42_${TAG}.json
    echo "[$(date +%H:%M:%S)] [refine] $TAG -> $OUT"
    srun --jobid=$ALLOC --overlap --exact --ntasks=1 --cpus-per-task=30 --mem=40G \
        --chdir=$(pwd) \
        $HPY tools/refine_all_parallel.py \
        --pool-from hl --hl-run "$RUN_DIR" \
        --threshold -1.0 --top-n 500 \
        --workers 30 --out "$OUT" \
        > "$REFINE_LOG" 2>&1
    echo "[$(date +%H:%M:%S)] [refine] $TAG done"
done
echo "phase B complete"
