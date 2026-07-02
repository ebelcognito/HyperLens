#!/bin/bash
# Sequential refsweep: one job at a time, 30 workers each. Mirrors fig4's
# refine config exactly. Per-job ~12 min, total ~60 min for 5 variants.
set -uo pipefail

cd /nas/longleaf/home/hisham26/Desktop/hyperlens
HPY=/nas/longleaf/home/hisham26/miniforge3/envs/hyperlens/bin/python
ALLOC=${1:-49085059}
SIXLET_HL_RUN=results/lensmega/sixlet_apsc_discovery_v2_seed42_20260506_034601
LOGDIR=results/refsweep_logs_seq_$(date +%Y%m%d_%H%M%S)
OUTDIR=results/refsweep_outputs
mkdir -p "$LOGDIR" "$OUTDIR"
echo "log dir: $LOGDIR  output dir: $OUTDIR"

for TAG in A_baseline B_efl_tight C_uniform_fw E_periphery D_long_lbfgs; do
    CFG=configs/ablations/sixlet_REFSWEEP_${TAG}.yaml
    OUT=$OUTDIR/refined_top500_sixlet_seed42_${TAG}.json
    LOG=$LOGDIR/${TAG}.log
    echo "[$(date +%H:%M:%S)] [refine] $TAG -> $OUT"
    srun --jobid=$ALLOC --overlap --exact --ntasks=1 --cpus-per-task=30 --mem=40G \
        --chdir=$(pwd) \
        $HPY tools/refine_all_parallel.py \
        --pool-from hl --hl-run "$SIXLET_HL_RUN" \
        --problem-config "$CFG" \
        --threshold -1.0 --top-n 500 \
        --workers 30 --out "$OUT" \
        > "$LOG" 2>&1
    echo "[$(date +%H:%M:%S)] [refine] $TAG done (exit $?)"
done
echo "all 5 refsweep jobs done"
ls -la $OUTDIR/
