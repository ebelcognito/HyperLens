#!/usr/bin/env bash
# Finish the APS-C cooke triplet ablation:
#   1. Recover G_grid512 + P_pop40 archives from checkpoint, then refine
#   2. Run E_cma_eta01 + G_grid128 (full search + refine)
#   3. Run CHAMPION combo (search + refine)
#
# Usage: run_apsc_triplet_finish.sh <ALLOC_JOBID>
set -uo pipefail

ALLOC=${1:?missing alloc id}
HPY=/nas/longleaf/home/hisham26/miniforge3/envs/hyperlens/bin/python
WORKDIR=/nas/longleaf/home/hisham26/Desktop/hyperlens
LOGDIR="$WORKDIR/results/apsc_triplet_refsweep_logs_alloc${ALLOC}"
mkdir -p "$LOGDIR"
cd "$WORKDIR"

run_refine() {
    local TAG=$1 RUN_DIR=$2 CFG=$3 OUT=$4
    echo "[$(date +%H:%M:%S)] [refine] $TAG -> $OUT"
    srun --jobid="$ALLOC" --overlap --exact --ntasks=1 --cpus-per-task=30 --mem=40G \
        --chdir="$WORKDIR" \
        "$HPY" tools/refine_all_parallel.py \
        --pool-from hl --hl-run "$RUN_DIR" \
        --problem-config "$CFG" \
        --threshold -1.0 --top-n 500 --workers 30 \
        --out "$OUT" \
        > "$LOGDIR/${TAG}_refine.log" 2>&1
    echo "[$(date +%H:%M:%S)] [refine] $TAG done rc=$?"
}

run_search_then_refine() {
    local TAG=$1
    local CFG=configs/ablations/apsc_triplet_REFSWEEP_${TAG}.yaml
    if [[ ! -f "$WORKDIR/$CFG" ]]; then
        echo "[$(date +%H:%M:%S)] [skip] missing config: $CFG"; return
    fi
    echo "[$(date +%H:%M:%S)] [search] $TAG"
    srun --jobid="$ALLOC" --overlap --exact --ntasks=1 --cpus-per-task=30 --mem=40G \
        --chdir="$WORKDIR" \
        "$HPY" -m optical_rl.qd.lensmega.run \
        --config "$CFG" --iterations 100000 \
        > "$LOGDIR/${TAG}_search.log" 2>&1
    echo "[$(date +%H:%M:%S)] [search] $TAG done rc=$?"

    local RUN_DIR
    RUN_DIR=$(ls -td "$WORKDIR"/results/lensmega/apsc_triplet_REFSWEEP_${TAG}_seed42_* 2>/dev/null | head -1)
    if [[ -z "$RUN_DIR" || ( ! -f "$RUN_DIR/archive.pkl" && ! -f "$RUN_DIR/checkpoint.pkl" ) ]]; then
        echo "[$(date +%H:%M:%S)] [skip-refine] no archive/ckpt for $TAG"; return
    fi
    if [[ ! -f "$RUN_DIR/archive.pkl" && -f "$RUN_DIR/checkpoint.pkl" ]]; then
        echo "[$(date +%H:%M:%S)] [recover] $TAG archive.pkl from checkpoint"
        "$HPY" "$WORKDIR/scripts_throwaway/recover_archive_from_ckpt.py" "$RUN_DIR" \
            >> "$LOGDIR/${TAG}_refine.log" 2>&1
    fi
    run_refine "$TAG" "$RUN_DIR" "$CFG" "$RUN_DIR/refined_top500_seed42_${TAG}.json"
}

echo "[$(date +%H:%M:%S)] === FINISH PIPELINE START alloc=$ALLOC ==="

# ---- Step 1: recover + refine the two killed runs ----
for TAG_DIR in \
    "G_grid512:results/lensmega/apsc_triplet_REFSWEEP_G_grid512_seed42_20260509_154741" \
    "P_pop40:results/lensmega/apsc_triplet_REFSWEEP_P_pop40_seed42_20260509_155957"; do
    TAG=${TAG_DIR%%:*}
    RUN_DIR="$WORKDIR/${TAG_DIR##*:}"
    CFG=configs/ablations/apsc_triplet_REFSWEEP_${TAG}.yaml
    OUT="$RUN_DIR/refined_top500_seed42_${TAG}.json"

    if [[ ! -d "$RUN_DIR" ]]; then
        echo "[$(date +%H:%M:%S)] [skip] $TAG: missing run dir"; continue
    fi
    echo "[$(date +%H:%M:%S)] [recover] $TAG archive.pkl from checkpoint"
    "$HPY" scripts_throwaway/recover_archive_from_ckpt.py "$RUN_DIR" \
        > "$LOGDIR/${TAG}_recover.log" 2>&1
    echo "[$(date +%H:%M:%S)] [recover] $TAG rc=$? $(cat $LOGDIR/${TAG}_recover.log | tail -1)"
    run_refine "$TAG" "$RUN_DIR" "$CFG" "$OUT"
done

# ---- Step 2: run the two unstarted variants ----
for TAG in G_grid128 E_cma_eta01; do
    run_search_then_refine "$TAG"
done

# ---- Step 3: CHAMPION combo ----
run_search_then_refine CHAMPION

echo "[$(date +%H:%M:%S)] === FINISH PIPELINE END alloc=$ALLOC ==="
