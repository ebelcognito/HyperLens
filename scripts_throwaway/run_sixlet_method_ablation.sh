#!/usr/bin/env bash
# Sixlet method-ablation: Seidel CHAMPION vs Random-projection BC, 3 seeds each.
#
# Usage: run_sixlet_method_ablation.sh <ALLOC_JOBID> [TAG1 TAG2 ...]
#   default TAGS: all 5 new runs (seed42 seidel reuses existing CHAMPION).
#
# Each variant runs sequentially (search -> refine) on the same alloc.
# We do NOT run variants in parallel (RUNBOOK pitfall: OOM/contention).
set -uo pipefail

ALLOC=${1:?missing alloc id}
shift || true
TAGS=("$@")
if [[ ${#TAGS[@]} -eq 0 ]]; then
    # seed42 seidel = the existing CHAMPION; we skip re-running it
    TAGS=(seidel_seed43 seidel_seed44 randbc_seed42 randbc_seed43 randbc_seed44)
fi

HPY=/nas/longleaf/home/hisham26/miniforge3/envs/hyperlens/bin/python
WORKDIR=/nas/longleaf/home/hisham26/Desktop/hyperlens
LOGDIR="$WORKDIR/results/sixlet_methodabl_logs_alloc${ALLOC}"
mkdir -p "$LOGDIR"

echo "[$(date +%H:%M:%S)] alloc=$ALLOC variants: ${TAGS[*]}"
echo "[$(date +%H:%M:%S)] logs -> $LOGDIR"

for TAG in "${TAGS[@]}"; do
    CFG=configs/ablations/sixlet_methodabl_${TAG}.yaml
    if [[ ! -f "$WORKDIR/$CFG" ]]; then
        echo "[$(date +%H:%M:%S)] [skip] missing config: $CFG"
        continue
    fi

    echo "[$(date +%H:%M:%S)] [search] $TAG"
    srun --jobid="$ALLOC" --overlap --exact --ntasks=1 --cpus-per-task=32 --mem=40G \
        --chdir="$WORKDIR" \
        "$HPY" -m optical_rl.qd.lensmega.run \
        --config "$CFG" \
        > "$LOGDIR/${TAG}_search.log" 2>&1
    SEARCH_RC=$?
    echo "[$(date +%H:%M:%S)] [search] $TAG done rc=$SEARCH_RC"

    RUN_DIR=$(ls -td "$WORKDIR"/results/lensmega/sixlet_methodabl_${TAG}_* 2>/dev/null | head -1)
    if [[ -z "$RUN_DIR" ]]; then
        echo "[$(date +%H:%M:%S)] [skip-refine] no run dir for $TAG"
        continue
    fi
    if [[ ! -f "$RUN_DIR/archive.pkl" && ! -f "$RUN_DIR/checkpoint.pkl" ]]; then
        echo "[$(date +%H:%M:%S)] [skip-refine] no archive/ckpt for $TAG in $RUN_DIR"
        continue
    fi

    OUT="$RUN_DIR/refined_top500_${TAG}.json"
    echo "[$(date +%H:%M:%S)] [refine] $TAG -> $OUT"
    srun --jobid="$ALLOC" --overlap --exact --ntasks=1 --cpus-per-task=30 --mem=40G \
        --chdir="$WORKDIR" \
        "$HPY" tools/refine_all_parallel.py \
        --pool-from hl --hl-run "$RUN_DIR" \
        --problem-config "$CFG" \
        --threshold -1.0 --top-n 500 --workers 30 \
        --out "$OUT" \
        > "$LOGDIR/${TAG}_refine.log" 2>&1
    REFINE_RC=$?
    echo "[$(date +%H:%M:%S)] [refine] $TAG done rc=$REFINE_RC"
done

echo "[$(date +%H:%M:%S)] alloc=$ALLOC ALL DONE: ${TAGS[*]}"
