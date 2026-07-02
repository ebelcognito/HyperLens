#!/usr/bin/env bash
# Refine the 4 remaining sixlet method-ablation runs on a fresh alloc.
# Runs 2 refines in parallel per batch to fit a 1-hour alloc.
#
# Usage: refine_sixlet_method_ablation.sh <ALLOC_JOBID>
set -uo pipefail

ALLOC=${1:?missing alloc id}
HPY=/nas/longleaf/home/hisham26/miniforge3/envs/hyperlens/bin/python
WORKDIR=/nas/longleaf/home/hisham26/Desktop/hyperlens
LOGDIR="$WORKDIR/results/sixlet_methodabl_refine_logs_alloc${ALLOC}"
mkdir -p "$LOGDIR"

# 4 variants to refine; arranged so each batch pairs one seidel + one randbc
# (no dependency between them, just keeps batches symmetric).
BATCH1=(seidel_seed44 randbc_seed42)
BATCH2=(randbc_seed43 randbc_seed44)

refine_one() {
    local TAG=$1
    local CFG="configs/ablations/sixlet_methodabl_${TAG}.yaml"
    local RUN_DIR
    RUN_DIR=$(ls -td "$WORKDIR"/results/lensmega/sixlet_methodabl_${TAG}_* 2>/dev/null | head -1)
    if [[ -z "$RUN_DIR" ]]; then
        echo "[$(date +%H:%M:%S)] [skip-refine] no run dir for $TAG" >&2
        return 1
    fi
    if [[ ! -f "$RUN_DIR/archive.pkl" && ! -f "$RUN_DIR/checkpoint.pkl" ]]; then
        echo "[$(date +%H:%M:%S)] [skip-refine] no archive/ckpt for $TAG" >&2
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

run_batch() {
    local -n BATCH=$1
    local pids=()
    for TAG in "${BATCH[@]}"; do
        refine_one "$TAG" &
        pids+=($!)
    done
    for p in "${pids[@]}"; do
        wait "$p"
    done
}

echo "[$(date +%H:%M:%S)] alloc=$ALLOC starting 4-variant refinement (2 parallel × 2 batches)"

echo "[$(date +%H:%M:%S)] BATCH 1: ${BATCH1[*]}"
run_batch BATCH1

echo "[$(date +%H:%M:%S)] BATCH 2: ${BATCH2[*]}"
run_batch BATCH2

echo "[$(date +%H:%M:%S)] alloc=$ALLOC ALL REFINES DONE"
