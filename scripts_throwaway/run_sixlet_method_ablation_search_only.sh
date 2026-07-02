#!/usr/bin/env bash
# Sixlet method-ablation: SEARCH-ONLY pass for remaining variants.
# Refines are deferred to a follow-up alloc (each refine is ~22 min serial).
#
# Usage: run_sixlet_method_ablation_search_only.sh <ALLOC_JOBID> [TAG1 TAG2 ...]
#   default TAGS: seidel_seed44 randbc_seed42 randbc_seed43 randbc_seed44
#
# (seed42 seidel = existing CHAMPION; seed43 seidel was completed by the
#  previous launch and is already refined.)
set -uo pipefail

ALLOC=${1:?missing alloc id}
shift || true
TAGS=("$@")
if [[ ${#TAGS[@]} -eq 0 ]]; then
    TAGS=(seidel_seed44 randbc_seed42 randbc_seed43 randbc_seed44)
fi

HPY=/nas/longleaf/home/hisham26/miniforge3/envs/hyperlens/bin/python
WORKDIR=/nas/longleaf/home/hisham26/Desktop/hyperlens
LOGDIR="$WORKDIR/results/sixlet_methodabl_logs_alloc${ALLOC}"
mkdir -p "$LOGDIR"

echo "[$(date +%H:%M:%S)] alloc=$ALLOC SEARCH-ONLY variants: ${TAGS[*]}"
echo "[$(date +%H:%M:%S)] logs -> $LOGDIR"

for TAG in "${TAGS[@]}"; do
    CFG=configs/ablations/sixlet_methodabl_${TAG}.yaml
    if [[ ! -f "$WORKDIR/$CFG" ]]; then
        echo "[$(date +%H:%M:%S)] [skip] missing config: $CFG"
        continue
    fi

    echo "[$(date +%H:%M:%S)] [search] $TAG"
    srun --jobid="$ALLOC" --overlap --exact --ntasks=1 --cpus-per-task=32 --mem=40G \
        --chdir="$WORKDIR" \
        "$HPY" -m optical_rl.qd.lensmega.run \
        --config "$CFG" \
        > "$LOGDIR/${TAG}_search.log" 2>&1
    SEARCH_RC=$?
    echo "[$(date +%H:%M:%S)] [search] $TAG done rc=$SEARCH_RC"
done

echo "[$(date +%H:%M:%S)] alloc=$ALLOC ALL SEARCHES DONE: ${TAGS[*]}"
