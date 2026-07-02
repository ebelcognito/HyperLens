#!/usr/bin/env bash
# Sixlet method-ablation: refine batch 2 (randbc_seed43 + randbc_seed44) in parallel.
# Designed to run on a *separate* alloc from batch 1.
#
# Usage: refine_sixlet_methodabl_batch2.sh <ALLOC_JOBID>
set -uo pipefail

ALLOC=${1:?missing alloc id}
HPY=/nas/longleaf/home/hisham26/miniforge3/envs/hyperlens/bin/python
WORKDIR=/nas/longleaf/home/hisham26/Desktop/hyperlens
LOGDIR="$WORKDIR/results/sixlet_methodabl_refine_logs_alloc${ALLOC}"
mkdir -p "$LOGDIR"

TAGS=(randbc_seed43 randbc_seed44)

refine_one() {
    local TAG=$1
    local CFG="configs/ablations/sixlet_methodabl_${TAG}.yaml"
    local RUN_DIR
    RUN_DIR=$(ls -td "$WORKDIR"/results/lensmega/sixlet_methodabl_${TAG}_* 2>/dev/null | head -1)
    if [[ -z "$RUN_DIR" ]]; then
        echo "[$(date +%H:%M:%S)] [skip-refine] no run dir for $TAG" >&2
        return 1
    fi
    local OUT="$RUN_DIR/refined_top500_${TAG}.json"
    if [[ -f "$OUT" ]]; then
        echo "[$(date +%H:%M:%S)] [skip-refine] $TAG already refined: $OUT"
        return 0
    fi
    echo "[$(date +%H:%M:%S)] [refine] $TAG -> $(basename "$OUT")"
    srun --jobid="$ALLOC" --overlap --exact --ntasks=1 --cpus-per-task=30 --mem=22G \
        --chdir="$WORKDIR" \
        "$HPY" tools/refine_all_parallel.py \
        --pool-from hl --hl-run "$RUN_DIR" \
        --problem-config "$CFG" \
        --threshold -1.0 --top-n 500 --workers 30 \
        --out "$OUT" \
        > "$LOGDIR/${TAG}_refine.log" 2>&1
    local RC=$?
    echo "[$(date +%H:%M:%S)] [refine] $TAG done rc=$RC"
    return $RC
}

echo "[$(date +%H:%M:%S)] alloc=$ALLOC starting BATCH 2 (parallel): ${TAGS[*]}"
pids=()
for TAG in "${TAGS[@]}"; do
    refine_one "$TAG" &
    pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
echo "[$(date +%H:%M:%S)] alloc=$ALLOC BATCH 2 DONE"
