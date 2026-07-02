#!/bin/bash
# The 4 enforced SEARCHES are already running (orphaned from the killed driver).
# This watcher waits for each search to finalize (archive_cells.json appears),
# then runs a TOP-100 refine on it. Independent per run, 4 concurrent.
#
# Usage: bash experiments/fnum_enforcement/wait_refine_top100.sh <ALLOC_JOBID>
set -uo pipefail
cd /nas/longleaf/home/hisham26/Desktop/hyperlens
HPY=/nas/longleaf/home/hisham26/miniforge3/envs/hyperlens/bin/python
ALLOC=${1:?usage: $0 <ALLOC_JOBID>}
CFGDIR=experiments/fnum_enforcement/configs
LOGDIR=experiments/fnum_enforcement/logs
mkdir -p "$LOGDIR"

wait_and_refine() {
    local CFG=$1
    local RUN_NAME
    RUN_NAME=$("$HPY" -c "import yaml;print(yaml.safe_load(open('$CFG'))['io']['run_name'])")
    local RUNDIR
    RUNDIR=$(ls -dt results/lensmega/${RUN_NAME}_* 2>/dev/null | head -1)
    if [[ -z "$RUNDIR" ]]; then echo "[$(date +%H:%M:%S)] ERR no run dir for $RUN_NAME"; return 1; fi
    echo "[$(date +%H:%M:%S)] watching $RUN_NAME ($RUNDIR)"

    # search is done when archive_cells.json is finalized
    until [[ -f "$RUNDIR/archive_cells.json" ]]; do sleep 15; done
    sleep 5   # let the finalize flush fully
    echo "[$(date +%H:%M:%S)] search DONE -> top-100 refine: $RUN_NAME"

    srun --jobid="$ALLOC" --overlap --exact --ntasks=1 --cpus-per-task=16 --mem=11G \
        --chdir="$(pwd)" \
        "$HPY" tools/refine_all_parallel.py \
        --pool-from hl --hl-run "$RUNDIR" \
        --problem-config "$CFG" \
        --threshold -1.0 --top-n 100 --workers 15 \
        --out "$RUNDIR/refined_top100_${RUN_NAME}.json" \
        > "$LOGDIR/${RUN_NAME}_refine100.log" 2>&1
    echo "[$(date +%H:%M:%S)] refine exit=$? : $RUN_NAME -> refined_top100_${RUN_NAME}.json"
}

echo "===== wait+top100-refine on alloc=$ALLOC ($(date)) ====="
PIDS=()
for CFG in "$CFGDIR"/sixlet_ENF_W1.yaml "$CFGDIR"/sixlet_ENF_W10.yaml \
           "$CFGDIR"/octuplet_ENF_W1.yaml "$CFGDIR"/octuplet_ENF_W10.yaml; do
    wait_and_refine "$CFG" &
    PIDS+=($!)
done
for pid in "${PIDS[@]}"; do wait "$pid" || echo "  watcher pid=$pid returned $?"; done
echo "===== all top-100 refines finished ($(date)) ====="
ls -la results/lensmega/*_ENF_W*/refined_top100_*.json 2>/dev/null
