#!/usr/bin/env bash
# Custom launcher for alloc 49215843 (64-CPU, 48G, 4hr).
# Runs: (1) baseline-recheck (refine fig4 archive), (2) BLIND_COMBO variant.
set -uo pipefail

ALLOC=49215843
HPY=/nas/longleaf/home/hisham26/miniforge3/envs/hyperlens/bin/python
WORKDIR=/nas/longleaf/home/hisham26/Desktop/hyperlens
LOGDIR="$WORKDIR/results/octuplet_refsweep_logs_alloc${ALLOC}"
mkdir -p "$LOGDIR"

echo "[$(date +%H:%M:%S)] alloc=$ALLOC starting"

# ---------- 1. Baseline recheck (Step 1 of runbook) ----------
BASE_RUN_DIR=results/lensmega/octuplet_apsc_2cement_v2_seed42_20260506_040025
BASE_OUT=/tmp/octuplet_baseline_recheck.json
echo "[$(date +%H:%M:%S)] [recheck] refining fig4 baseline archive"
srun --jobid="$ALLOC" --overlap --exact --ntasks=1 --cpus-per-task=30 --mem=22G \
    --chdir="$WORKDIR" \
    "$HPY" tools/refine_all_parallel.py \
    --pool-from hl --hl-run "$BASE_RUN_DIR" \
    --threshold -1.0 --top-n 500 --workers 30 \
    --out "$BASE_OUT" \
    > "$LOGDIR/baseline_recheck.log" 2>&1
RECHECK_RC=$?
echo "[$(date +%H:%M:%S)] [recheck] done rc=$RECHECK_RC -> $BASE_OUT"

# ---------- 2. BLIND_COMBO (search + refine) ----------
TAG=BLIND_COMBO
CFG=configs/ablations/octuplet_REFSWEEP_${TAG}.yaml

echo "[$(date +%H:%M:%S)] [search] $TAG"
srun --jobid="$ALLOC" --overlap --exact --ntasks=1 --cpus-per-task=30 --mem=30G \
    --chdir="$WORKDIR" \
    "$HPY" -m optical_rl.qd.lensmega.run \
    --config "$CFG" --iterations 100000 \
    > "$LOGDIR/${TAG}_search.log" 2>&1
SEARCH_RC=$?
echo "[$(date +%H:%M:%S)] [search] $TAG done rc=$SEARCH_RC"

RUN_DIR=$(ls -td "$WORKDIR"/results/lensmega/octuplet_REFSWEEP_${TAG}_seed42_* 2>/dev/null | head -1)
if [[ -z "$RUN_DIR" ]]; then
    echo "[$(date +%H:%M:%S)] [skip-refine] no run dir for $TAG"
    exit 1
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
REFINE_RC=$?
echo "[$(date +%H:%M:%S)] [refine] $TAG done rc=$REFINE_RC"

echo "[$(date +%H:%M:%S)] alloc=$ALLOC ALL DONE"
