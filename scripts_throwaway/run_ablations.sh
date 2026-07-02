#!/bin/bash
# ============================================================================
# Run NoGrad and RandomBC ablations at matched compute to fig4 HL.
# Mirrors run_diversity_experiments.sh phase 1 (search) + phase 2 (refine top-500).
#
# Usage:  bash scripts_throwaway/run_ablations.sh <ALLOC_JOBID> [budget=100000]
# Or invoke from within an `salloc`-shell: bash scripts_throwaway/run_ablations.sh "" [budget]
# ============================================================================
set -uo pipefail

ALLOC=${1-""}
BUDGET=${2:-100000}

cd /nas/longleaf/home/hisham26/Desktop/hyperlens
HPY=/nas/longleaf/home/hisham26/miniforge3/envs/hyperlens/bin/python
LOGDIR=results/ablations_logs_$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOGDIR"
echo "log dir: $LOGDIR"

# srun wrapper — uses --jobid if ALLOC was given, else assumes we're already inside an allocation
if [[ -n "$ALLOC" ]]; then
    SRUN="srun --jobid=$ALLOC --overlap --exact --ntasks=1 --cpus-per-task=32 --mem=48G --chdir=$(pwd)"
else
    SRUN=""   # already in allocation, just run directly
fi

run_search() {
    local tag=$1
    local cfg=configs/ablations/cooke_triplet_modernstack_${tag}_seed42.yaml
    local log=$LOGDIR/${tag}_search.log
    echo "[search] $tag -> $log"
    $SRUN $HPY -m optical_rl.qd.lensmega.run --config $cfg --iterations $BUDGET > "$log" 2>&1
    echo "[search] $tag done (exit $?)"
}

# Phase 1: search (run sequentially if single alloc; in parallel if needed)
run_search NOGRAD &
PID1=$!
run_search RANDBC &
PID2=$!
wait $PID1; echo "NOGRAD search exit=$?"
wait $PID2; echo "RANDBC search exit=$?"

# Phase 2: refine top-500 from each ablation archive
for TAG in NOGRAD RANDBC; do
    RUN_DIR=$(ls -td results/lensmega/cooke_triplet_modernstack_${TAG}_seed42_* 2>/dev/null | head -1)
    if [[ -z "$RUN_DIR" ]]; then
        echo "[refine] $TAG: no run dir found, skipping"
        continue
    fi
    OUT=$RUN_DIR/refined_top500_seed42_${TAG}.json
    LOG=$LOGDIR/${TAG}_refine.log
    echo "[refine] $TAG -> $OUT"
    $SRUN $HPY tools/refine_all_parallel.py \
        --pool-from hl --hl-run "$RUN_DIR" \
        --threshold -1.0 --top-n 500 \
        --workers 30 --out "$OUT" > "$LOG" 2>&1 &
done
wait
echo "all refine jobs done"
echo "results in: $LOGDIR"
