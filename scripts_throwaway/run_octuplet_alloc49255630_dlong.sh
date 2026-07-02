#!/usr/bin/env bash
# D_long_lbfgs search+refine on fresh alloc 49255630 (c130501).
# Runs in parallel with main recovery's W_wider_bd to save ~42 min total.
set -uo pipefail

ALLOC=49255630
HPY=/nas/longleaf/home/hisham26/miniforge3/envs/hyperlens/bin/python
WORKDIR=/nas/longleaf/home/hisham26/Desktop/hyperlens
LOGDIR="$WORKDIR/results/octuplet_refsweep_logs_alloc${ALLOC}"
mkdir -p "$LOGDIR"

TAG=D_long_lbfgs
CFG=configs/ablations/octuplet_REFSWEEP_${TAG}.yaml

echo "[$(date +%H:%M:%S)] alloc=$ALLOC [search] $TAG"
srun --jobid="$ALLOC" --overlap --exact --ntasks=1 --cpus-per-task=30 --mem=30G \
    --chdir="$WORKDIR" \
    "$HPY" -m optical_rl.qd.lensmega.run \
    --config "$CFG" --iterations 100000 \
    > "$LOGDIR/${TAG}_search.log" 2>&1
echo "[$(date +%H:%M:%S)] [search] $TAG done rc=$?"

RUN_DIR=$(ls -td "$WORKDIR"/results/lensmega/octuplet_REFSWEEP_${TAG}_seed42_* 2>/dev/null | head -1)
if [[ -z "$RUN_DIR" ]]; then
    echo "[$(date +%H:%M:%S)] no run dir for $TAG"; exit 1
fi
OUT="$RUN_DIR/refined_top500_seed42_${TAG}.json"
echo "[$(date +%H:%M:%S)] [refine] $TAG -> $OUT"
srun --jobid="$ALLOC" --overlap --exact --ntasks=1 --cpus-per-task=30 --mem=30G \
    --chdir="$WORKDIR" \
    "$HPY" tools/refine_all_parallel.py \
    --pool-from hl --hl-run "$RUN_DIR" \
    --threshold -1.0 --top-n 500 --workers 30 \
    --out "$OUT" \
    > "$LOGDIR/${TAG}_refine.log" 2>&1
echo "[$(date +%H:%M:%S)] [refine] $TAG done rc=$?"
echo "[$(date +%H:%M:%S)] alloc=$ALLOC DONE"
