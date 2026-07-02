#!/usr/bin/env bash
# Recovery launcher for the short alloc 49216793 (~30 min left).
# Refines T_tighter_bd from its existing archive.pkl.
set -uo pipefail

ALLOC=49216793
HPY=/nas/longleaf/home/hisham26/miniforge3/envs/hyperlens/bin/python
WORKDIR=/nas/longleaf/home/hisham26/Desktop/hyperlens
LOGDIR="$WORKDIR/results/octuplet_refsweep_logs_alloc${ALLOC}"
mkdir -p "$LOGDIR"

TAG=T_tighter_bd
RUN_DIR=$(ls -td "$WORKDIR"/results/lensmega/octuplet_REFSWEEP_${TAG}_seed42_* 2>/dev/null | head -1)
if [[ -z "$RUN_DIR" || ! -f "$RUN_DIR/archive.pkl" ]]; then
    echo "[$(date +%H:%M:%S)] ABORT: no archive.pkl for $TAG"
    exit 1
fi

OUT="$RUN_DIR/refined_top500_seed42_${TAG}.json"
echo "[$(date +%H:%M:%S)] [refine] $TAG (recovery from $RUN_DIR)"
srun --jobid="$ALLOC" --overlap --exact --ntasks=1 --cpus-per-task=30 --mem=30G \
    --chdir="$WORKDIR" \
    "$HPY" tools/refine_all_parallel.py \
    --pool-from hl --hl-run "$RUN_DIR" \
    --threshold -1.0 --top-n 500 --workers 30 \
    --out "$OUT" \
    > "$LOGDIR/${TAG}_refine.log" 2>&1
echo "[$(date +%H:%M:%S)] [refine] $TAG done rc=$?"
echo "[$(date +%H:%M:%S)] alloc=$ALLOC FAST DONE"
