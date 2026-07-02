#!/usr/bin/env bash
# Re-refine the sixlet CHAMPION archive WITHOUT the top-500 cap.
# Polishes every archive cell with raw fitness >= -1 (currently 894 cells).
# Output replaces the existing refined_top500_*.json with a new file under a
# different name so the old data is preserved.
#
# Usage:  ./refine_sixlet_champion_uncapped.sh <ALLOC_JOBID>
#
# Compute estimate: ~22 min wall on 30 workers (894 designs × ~1.5 sec/design,
# parallel).
set -uo pipefail

ALLOC=${1:?missing alloc id}
HPY=/nas/longleaf/home/hisham26/miniforge3/envs/hyperlens/bin/python
WORKDIR=/nas/longleaf/home/hisham26/Desktop/hyperlens
RUN_DIR="$WORKDIR/results/lensmega/sixlet_REFSWEEP2_CHAMPION_seed42_20260509_080537"
LOGDIR="$RUN_DIR/refine_uncapped_log"
mkdir -p "$LOGDIR"

OUT="$RUN_DIR/refined_above_-1.0_seed42_CHAMPION.json"
echo "[$(date +%H:%M:%S)] alloc=$ALLOC  refining ALL cells in $RUN_DIR with fit ≥ -1 (no top-n cap)"
echo "  output: $OUT"

srun --jobid="$ALLOC" --overlap --exact --ntasks=1 --cpus-per-task=30 --mem=30G \
    --chdir="$WORKDIR" \
    "$HPY" tools/refine_all_parallel.py \
    --pool-from hl --hl-run "$RUN_DIR" \
    --threshold -1.0 --workers 30 \
    --out "$OUT" \
    > "$LOGDIR/refine.log" 2>&1
RC=$?
echo "[$(date +%H:%M:%S)] done rc=$RC"

if [[ -f "$OUT" ]]; then
    sz=$(stat -c %s "$OUT")
    n=$($HPY -c "import json; d=json.load(open('$OUT')); print(len(d.get('designs', [])))")
    echo "  $OUT  $sz B  n_designs=$n"
fi
