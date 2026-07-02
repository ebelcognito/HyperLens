#!/usr/bin/env bash
# Recovery on alloc 49246789 (c130407, 64 CPU, 48G).
# Three extract+refine tasks for the variants whose searches completed (or partially)
# but whose final archive.pkl write was killed by alloc expiry / OOM / hangup.
set -uo pipefail

ALLOC=49246789
HPY=/nas/longleaf/home/hisham26/miniforge3/envs/hyperlens/bin/python
WORKDIR=/nas/longleaf/home/hisham26/Desktop/hyperlens
LOGDIR="$WORKDIR/results/octuplet_refsweep_logs_alloc${ALLOC}"
mkdir -p "$LOGDIR"

echo "[$(date +%H:%M:%S)] alloc=$ALLOC starting recovery"

refine_only () {
    local TAG=$1
    local RUN_DIR=$(ls -td "$WORKDIR"/results/lensmega/octuplet_REFSWEEP_${TAG}_seed42_* 2>/dev/null | head -1)
    if [[ -z "$RUN_DIR" ]]; then
        echo "[$(date +%H:%M:%S)] [skip] no run dir for $TAG"; return 1
    fi
    if [[ ! -f "$RUN_DIR/archive.pkl" ]]; then
        echo "[$(date +%H:%M:%S)] [extract] $TAG checkpoint -> archive"
        "$HPY" "$WORKDIR/scripts_throwaway/extract_archive_from_checkpoint.py" "$RUN_DIR" \
            > "$LOGDIR/${TAG}_extract.log" 2>&1
        if [[ ! -f "$RUN_DIR/archive.pkl" ]]; then
            echo "[$(date +%H:%M:%S)] [fail-extract] $TAG"; cat "$LOGDIR/${TAG}_extract.log"; return 1
        fi
    fi
    local OUT="$RUN_DIR/refined_top500_seed42_${TAG}.json"
    echo "[$(date +%H:%M:%S)] [refine] $TAG -> $OUT"
    srun --jobid="$ALLOC" --overlap --exact --ntasks=1 --cpus-per-task=30 --mem=30G \
        --chdir="$WORKDIR" \
        "$HPY" tools/refine_all_parallel.py \
        --pool-from hl --hl-run "$RUN_DIR" \
        --threshold -1.0 --top-n 500 --workers 30 \
        --out "$OUT" \
        > "$LOGDIR/${TAG}_refine.log" 2>&1
    echo "[$(date +%H:%M:%S)] [refine] $TAG done rc=$?"
}

refine_only W_wider_bd
refine_only G_grid3000
refine_only BLIND_COMBO

echo "[$(date +%H:%M:%S)] alloc=$ALLOC RECOVERY DONE"
