#!/bin/bash
# Re-refine sixlet HL top-500 archive under 5 refinement configs (A..E),
# in parallel on alloc 49085059. ~3-5 min per refine job.
set -uo pipefail

cd /nas/longleaf/home/hisham26/Desktop/hyperlens
HPY=/nas/longleaf/home/hisham26/miniforge3/envs/hyperlens/bin/python
ALLOC=${1:-49085059}
SIXLET_HL_RUN=results/lensmega/sixlet_apsc_discovery_v2_seed42_20260506_034601
LOGDIR=results/refsweep_logs_$(date +%Y%m%d_%H%M%S)
OUTDIR=results/refsweep_outputs
mkdir -p "$LOGDIR" "$OUTDIR"
echo "log dir: $LOGDIR  output dir: $OUTDIR"

PIDS=()
for TAG in A_baseline B_efl_tight C_uniform_fw D_long_lbfgs E_periphery; do
    CFG=configs/ablations/sixlet_REFSWEEP_${TAG}.yaml
    OUT=$OUTDIR/refined_top500_sixlet_seed42_${TAG}.json
    LOG=$LOGDIR/${TAG}.log
    echo "[refine] $TAG -> $OUT"
    srun --jobid=$ALLOC --overlap --exact --ntasks=1 --cpus-per-task=6 --mem=8G \
        --chdir=$(pwd) \
        $HPY tools/refine_all_parallel.py \
        --pool-from hl --hl-run "$SIXLET_HL_RUN" \
        --problem-config "$CFG" \
        --threshold -1.0 --top-n 500 \
        --workers 6 --out "$OUT" \
        > "$LOG" 2>&1 &
    PIDS+=($!)
done

# wait all
for pid in "${PIDS[@]}"; do
    wait $pid && echo "  pid=$pid done" || echo "  pid=$pid FAILED ($?)"
done
echo "all 5 refsweep jobs done"
ls -la $OUTDIR/
