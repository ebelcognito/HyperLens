#!/usr/bin/env bash
# Run a subset of octuplet ablation variants on a single SLURM allocation.
#
# Usage: run_octuplet_ablation_alloc.sh <ALLOC_JOBID> <SRUN_MEM> <TAG1> [TAG2 ...]
#
# Each variant is run sequentially: search -> refine. Variants do NOT run in
# parallel within an allocation (per runbook: causes OOM / contention).
set -uo pipefail

ALLOC=${1:?missing alloc id}
SRUN_MEM=${2:?missing srun mem (e.g. 22G or 30G)}
shift 2
TAGS=("$@")
if [[ ${#TAGS[@]} -eq 0 ]]; then
    echo "no tags provided" >&2
    exit 1
fi

HPY=/nas/longleaf/home/hisham26/miniforge3/envs/hyperlens/bin/python
WORKDIR=/nas/longleaf/home/hisham26/Desktop/hyperlens
LOGDIR="$WORKDIR/results/octuplet_refsweep_logs_alloc${ALLOC}"
mkdir -p "$LOGDIR"

echo "[$(date +%H:%M:%S)] alloc=$ALLOC mem=$SRUN_MEM variants: ${TAGS[*]}"

for TAG in "${TAGS[@]}"; do
    CFG=configs/ablations/octuplet_REFSWEEP_${TAG}.yaml
    if [[ ! -f "$WORKDIR/$CFG" ]]; then
        echo "[$(date +%H:%M:%S)] [skip] missing config: $CFG"
        continue
    fi

    echo "[$(date +%H:%M:%S)] [search] $TAG"
    srun --jobid="$ALLOC" --overlap --exact --ntasks=1 --cpus-per-task=30 --mem="$SRUN_MEM" \
        --chdir="$WORKDIR" \
        "$HPY" -m optical_rl.qd.lensmega.run \
        --config "$CFG" --iterations 100000 \
        > "$LOGDIR/${TAG}_search.log" 2>&1
    SEARCH_RC=$?
    echo "[$(date +%H:%M:%S)] [search] $TAG done rc=$SEARCH_RC"

    RUN_DIR=$(ls -td "$WORKDIR"/results/lensmega/octuplet_REFSWEEP_${TAG}_seed42_* 2>/dev/null | head -1)
    if [[ -z "$RUN_DIR" ]]; then
        echo "[$(date +%H:%M:%S)] [skip-refine] no run dir for $TAG"
        continue
    fi
    if [[ ! -f "$RUN_DIR/archive.pkl" && ! -f "$RUN_DIR/checkpoint.pkl" ]]; then
        echo "[$(date +%H:%M:%S)] [skip-refine] no archive/ckpt for $TAG in $RUN_DIR"
        continue
    fi

    OUT="$RUN_DIR/refined_top500_seed42_${TAG}.json"
    echo "[$(date +%H:%M:%S)] [refine] $TAG -> $OUT"
    srun --jobid="$ALLOC" --overlap --exact --ntasks=1 --cpus-per-task=30 --mem="$SRUN_MEM" \
        --chdir="$WORKDIR" \
        "$HPY" tools/refine_all_parallel.py \
        --pool-from hl --hl-run "$RUN_DIR" \
        --threshold -1.0 --top-n 500 --workers 30 \
        --out "$OUT" \
        > "$LOGDIR/${TAG}_refine.log" 2>&1
    REFINE_RC=$?
    echo "[$(date +%H:%M:%S)] [refine] $TAG done rc=$REFINE_RC"
done

echo "[$(date +%H:%M:%S)] alloc=$ALLOC ALL DONE: ${TAGS[*]}"
