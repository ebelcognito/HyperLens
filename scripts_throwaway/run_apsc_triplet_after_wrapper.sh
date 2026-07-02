#!/usr/bin/env bash
# After the main wrapper finishes, refine G_grid512 + P_pop40 (both archives
# already recovered from checkpoint on login node — see recover_archive_from_ckpt.py).
#
# Usage: run_apsc_triplet_after_wrapper.sh <ALLOC_JOBID> <WRAPPER_PID>
set -uo pipefail

ALLOC=${1:?missing alloc id}
WRAPPER_PID=${2:?missing wrapper pid}
HPY=/nas/longleaf/home/hisham26/miniforge3/envs/hyperlens/bin/python
WORKDIR=/nas/longleaf/home/hisham26/Desktop/hyperlens
LOGDIR="$WORKDIR/results/apsc_triplet_refsweep_logs_alloc${ALLOC}"
mkdir -p "$LOGDIR"
cd "$WORKDIR"

echo "[$(date +%H:%M:%S)] [after-wrapper] waiting for PID $WRAPPER_PID to exit..."
# Poll for wrapper exit. /proc/<pid> disappears when the process is gone.
while [[ -d "/proc/$WRAPPER_PID" ]]; do
    sleep 30
done
echo "[$(date +%H:%M:%S)] [after-wrapper] wrapper $WRAPPER_PID is done; refining recovered archives"

# Verify alloc still alive before submitting
if ! squeue -u "$(whoami)" -h -o "%i" 2>/dev/null | grep -q "^${ALLOC}$"; then
    echo "[$(date +%H:%M:%S)] [after-wrapper] alloc $ALLOC no longer in queue — aborting"
    exit 1
fi

run_refine() {
    local TAG=$1 RUN_DIR=$2
    local CFG=configs/ablations/apsc_triplet_REFSWEEP_${TAG}.yaml
    local OUT="$RUN_DIR/refined_top500_seed42_${TAG}.json"
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

run_refine G_grid512 results/lensmega/apsc_triplet_REFSWEEP_G_grid512_seed42_20260509_154741
run_refine P_pop40   results/lensmega/apsc_triplet_REFSWEEP_P_pop40_seed42_20260509_155957

echo "[$(date +%H:%M:%S)] [after-wrapper] DONE"
