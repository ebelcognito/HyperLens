#!/usr/bin/env bash
# Octuplet-tuned CHAMPION: bins=2048² + σ=1.0 + patience=20.
# srun --mem=40G (vs BLIND_COMBO's 30G that OOM'd at gen 1700).
set -uo pipefail

ALLOC=${1:?missing alloc id}
HPY=/nas/longleaf/home/hisham26/miniforge3/envs/hyperlens/bin/python
WORKDIR=/nas/longleaf/home/hisham26/Desktop/hyperlens
LOGDIR="$WORKDIR/results/octuplet_refsweep_logs_alloc${ALLOC}"
mkdir -p "$LOGDIR"

TAG=CHAMPION
CFG=configs/ablations/octuplet_REFSWEEP_${TAG}.yaml

echo "[$(date +%H:%M:%S)] alloc=$ALLOC [search] $TAG"
srun --jobid="$ALLOC" --overlap --exact --ntasks=1 --cpus-per-task=30 --mem=40G \
    --chdir="$WORKDIR" \
    "$HPY" -m optical_rl.qd.lensmega.run \
    --config "$CFG" --iterations 100000 \
    > "$LOGDIR/${TAG}_search.log" 2>&1
echo "[$(date +%H:%M:%S)] [search] $TAG done rc=$?"

RUN_DIR=$(ls -td "$WORKDIR"/results/lensmega/octuplet_REFSWEEP_${TAG}_seed42_* 2>/dev/null | head -1)
if [[ -z "$RUN_DIR" ]]; then
    echo "[$(date +%H:%M:%S)] no run dir for $TAG"; exit 1
fi
if [[ ! -f "$RUN_DIR/archive.pkl" && -f "$RUN_DIR/checkpoint.pkl" ]]; then
    echo "[$(date +%H:%M:%S)] no archive.pkl, extracting from checkpoint"
    "$HPY" "$WORKDIR/scripts_throwaway/extract_archive_from_checkpoint.py" "$RUN_DIR" \
        > "$LOGDIR/${TAG}_extract.log" 2>&1
fi

OUT="$RUN_DIR/refined_top500_seed42_${TAG}.json"
echo "[$(date +%H:%M:%S)] [refine] $TAG -> $OUT"
srun --jobid="$ALLOC" --overlap --exact --ntasks=1 --cpus-per-task=30 --mem=40G \
    --chdir="$WORKDIR" \
    "$HPY" tools/refine_all_parallel.py \
    --pool-from hl --hl-run "$RUN_DIR" \
    --threshold -1.0 --top-n 500 --workers 30 \
    --out "$OUT" \
    > "$LOGDIR/${TAG}_refine.log" 2>&1
echo "[$(date +%H:%M:%S)] [refine] $TAG done rc=$?"
echo "[$(date +%H:%M:%S)] alloc=$ALLOC CHAMPION DONE"
