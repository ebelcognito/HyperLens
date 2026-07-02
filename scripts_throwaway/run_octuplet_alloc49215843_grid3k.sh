#!/usr/bin/env bash
# Follow-up launcher for alloc 49215843: waits for the recheck+BLIND_COMBO
# launcher to finish, then runs G_grid3000 (search + refine) on the same alloc.
set -uo pipefail

ALLOC=49215843
MAIN_PID=1673948
MAIN_LOG=/tmp/octuplet_alloc49215843.log
HPY=/nas/longleaf/home/hisham26/miniforge3/envs/hyperlens/bin/python
WORKDIR=/nas/longleaf/home/hisham26/Desktop/hyperlens
LOGDIR="$WORKDIR/results/octuplet_refsweep_logs_alloc${ALLOC}"
mkdir -p "$LOGDIR"

echo "[$(date +%H:%M:%S)] [grid3k] waiting for main launcher PID=$MAIN_PID to finish"
while kill -0 "$MAIN_PID" 2>/dev/null; do
    sleep 30
done
echo "[$(date +%H:%M:%S)] [grid3k] main launcher exited"

if ! grep -q "ALL DONE" "$MAIN_LOG"; then
    echo "[$(date +%H:%M:%S)] [grid3k] WARNING: main launcher exited without ALL DONE marker — check $MAIN_LOG"
    # Continue anyway: if the alloc is still alive, G_grid3000 can still run.
fi

# Confirm alloc still alive before launching
if ! squeue -j "$ALLOC" -h 2>/dev/null | grep -q "$ALLOC"; then
    echo "[$(date +%H:%M:%S)] [grid3k] ABORT: alloc $ALLOC no longer in squeue"
    exit 1
fi

TAG=G_grid3000
CFG=configs/ablations/octuplet_REFSWEEP_${TAG}.yaml

echo "[$(date +%H:%M:%S)] [search] $TAG"
srun --jobid="$ALLOC" --overlap --exact --ntasks=1 --cpus-per-task=30 --mem=35G \
    --chdir="$WORKDIR" \
    "$HPY" -m optical_rl.qd.lensmega.run \
    --config "$CFG" --iterations 100000 \
    > "$LOGDIR/${TAG}_search.log" 2>&1
SEARCH_RC=$?
echo "[$(date +%H:%M:%S)] [search] $TAG done rc=$SEARCH_RC"

RUN_DIR=$(ls -td "$WORKDIR"/results/lensmega/octuplet_REFSWEEP_${TAG}_seed42_* 2>/dev/null | head -1)
if [[ -z "$RUN_DIR" ]]; then
    echo "[$(date +%H:%M:%S)] [grid3k] no run dir for $TAG"
    exit 1
fi

OUT="$RUN_DIR/refined_top500_seed42_${TAG}.json"
echo "[$(date +%H:%M:%S)] [refine] $TAG -> $OUT"
srun --jobid="$ALLOC" --overlap --exact --ntasks=1 --cpus-per-task=30 --mem=35G \
    --chdir="$WORKDIR" \
    "$HPY" tools/refine_all_parallel.py \
    --pool-from hl --hl-run "$RUN_DIR" \
    --threshold -1.0 --top-n 500 --workers 30 \
    --out "$OUT" \
    > "$LOGDIR/${TAG}_refine.log" 2>&1
REFINE_RC=$?
echo "[$(date +%H:%M:%S)] [refine] $TAG done rc=$REFINE_RC"

echo "[$(date +%H:%M:%S)] alloc=$ALLOC GRID3K DONE"
