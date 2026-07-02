#!/bin/bash
# Enforced-f/# A/B experiment: rerun the sixlet & octuplet soft baselines with
# enforce_fnum:true at two penalty weights (1, 10), matched on everything else
# (same template, bins/bounds, 100k iters, seed). Search + top-500 refine for
# each, 4 pipelines concurrently against an existing salloc.
#
# Usage:  bash experiments/fnum_enforcement/run_enforce_comparison.sh <ALLOC_JOBID>
set -uo pipefail
cd /nas/longleaf/home/hisham26/Desktop/hyperlens
HPY=/nas/longleaf/home/hisham26/miniforge3/envs/hyperlens/bin/python
ALLOC=${1:?usage: $0 <ALLOC_JOBID>}

CFGDIR=experiments/fnum_enforcement/configs
LOGDIR=experiments/fnum_enforcement/logs
mkdir -p "$LOGDIR"

ITERS=100000          # match the soft baselines (trajectory capped at 100k)
CPT=16                # cpus per step; 4 x 16 = 64
MEM=11G               # 4 x 11 = 44G <= 48G
WORKERS=15            # refine workers (search uses parallel_workers:15 from cfg)

run_pipeline() {
    local CFG=$1
    local NAME
    NAME=$(basename "$CFG" .yaml)
    local SLOG=$LOGDIR/${NAME}_search.log
    local RLOG=$LOGDIR/${NAME}_refine.log
    echo "[$(date +%H:%M:%S)] START search: $NAME"

    srun --jobid="$ALLOC" --overlap --exact --ntasks=1 --cpus-per-task=$CPT --mem=$MEM \
        --chdir="$(pwd)" \
        "$HPY" -m optical_rl.qd.lensmega.run --config "$CFG" --iterations $ITERS \
        > "$SLOG" 2>&1
    echo "[$(date +%H:%M:%S)] search exit=$? : $NAME (torch-shutdown nonzero is benign; check artifacts)"

    # discover the run dir this config just created (run_name is in the cfg io block)
    local RUN_NAME
    RUN_NAME=$("$HPY" -c "import yaml,sys; print(yaml.safe_load(open('$CFG'))['io']['run_name'])")
    local RUNDIR
    RUNDIR=$(ls -dt results/lensmega/${RUN_NAME}_* 2>/dev/null | head -1)
    if [[ -z "$RUNDIR" || ! -d "$RUNDIR" ]]; then
        echo "[$(date +%H:%M:%S)] ERROR: no run dir for $RUN_NAME — skipping refine"; return 1
    fi
    echo "[$(date +%H:%M:%S)] START refine: $NAME -> $RUNDIR"

    srun --jobid="$ALLOC" --overlap --exact --ntasks=1 --cpus-per-task=$CPT --mem=$MEM \
        --chdir="$(pwd)" \
        "$HPY" tools/refine_all_parallel.py \
        --pool-from hl --hl-run "$RUNDIR" \
        --problem-config "$CFG" \
        --threshold -1.0 --top-n 500 --workers $WORKERS \
        --out "$RUNDIR/refined_top500_${RUN_NAME}.json" \
        > "$RLOG" 2>&1
    echo "[$(date +%H:%M:%S)] refine exit=$? : $NAME ; out=$RUNDIR/refined_top500_${RUN_NAME}.json"
}

echo "===== enforce-f/# A/B on alloc=$ALLOC  ($(date)) ====="
PIDS=()
for CFG in "$CFGDIR"/sixlet_ENF_W1.yaml "$CFGDIR"/sixlet_ENF_W10.yaml \
           "$CFGDIR"/octuplet_ENF_W1.yaml "$CFGDIR"/octuplet_ENF_W10.yaml; do
    run_pipeline "$CFG" &
    PIDS+=($!)
    sleep 2
done

FAIL=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || { echo "  pipeline pid=$pid returned $?"; FAIL=1; }
done
echo "===== all pipelines finished ($(date)) ; FAIL=$FAIL ====="
ls -la results/lensmega/*_ENF_W*/refined_top500_*.json 2>/dev/null
