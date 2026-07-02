#!/usr/bin/env bash
# Idempotent refine pass for sixlet method-ablation variants.
# Refines any (arm, seed) whose run_dir exists, archive.pkl exists, and
# refined_top500_*.json does NOT yet exist.
#
# Usage: refine_sixlet_methodabl_pending.sh <ALLOC_JOBID>
#
# Runs sequentially (one variant at a time on the alloc). Each refine ~22 min.
set -uo pipefail

ALLOC=${1:?missing alloc id}

HPY=/nas/longleaf/home/hisham26/miniforge3/envs/hyperlens/bin/python
WORKDIR=/nas/longleaf/home/hisham26/Desktop/hyperlens
LOGDIR="$WORKDIR/results/sixlet_methodabl_logs_alloc${ALLOC}"
mkdir -p "$LOGDIR"

ALL_TAGS=(seidel_seed43 seidel_seed44 randbc_seed42 randbc_seed43 randbc_seed44)

echo "[$(date +%H:%M:%S)] alloc=$ALLOC idempotent refine pass"

PENDING=()
for TAG in "${ALL_TAGS[@]}"; do
    RD=$(ls -td "$WORKDIR"/results/lensmega/sixlet_methodabl_${TAG}_* 2>/dev/null | head -1)
    if [[ -z "$RD" ]]; then
        echo "[$(date +%H:%M:%S)] [skip] $TAG: no run dir"
        continue
    fi
    if [[ ! -f "$RD/archive.pkl" ]]; then
        echo "[$(date +%H:%M:%S)] [skip] $TAG: no archive.pkl in $RD"
        continue
    fi
    OUT="$RD/refined_top500_${TAG}.json"
    if [[ -f "$OUT" ]]; then
        echo "[$(date +%H:%M:%S)] [done] $TAG: refined JSON exists at $OUT"
        continue
    fi
    PENDING+=("$TAG")
done

echo "[$(date +%H:%M:%S)] pending refines: ${PENDING[*]:-NONE}"
[[ ${#PENDING[@]} -eq 0 ]] && { echo "nothing to do"; exit 0; }

for TAG in "${PENDING[@]}"; do
    CFG=configs/ablations/sixlet_methodabl_${TAG}.yaml
    RD=$(ls -td "$WORKDIR"/results/lensmega/sixlet_methodabl_${TAG}_* | head -1)
    OUT="$RD/refined_top500_${TAG}.json"

    echo "[$(date +%H:%M:%S)] [refine] $TAG -> $(basename $OUT)"
    srun --jobid="$ALLOC" --overlap --exact --ntasks=1 --cpus-per-task=30 --mem=40G \
        --chdir="$WORKDIR" \
        "$HPY" tools/refine_all_parallel.py \
        --pool-from hl --hl-run "$RD" \
        --problem-config "$CFG" \
        --threshold -1.0 --top-n 500 --workers 30 \
        --out "$OUT" \
        > "$LOGDIR/${TAG}_refine.log" 2>&1
    REFINE_RC=$?
    echo "[$(date +%H:%M:%S)] [refine] $TAG done rc=$REFINE_RC"
done

echo "[$(date +%H:%M:%S)] all refines complete"
