#!/usr/bin/env bash
# Full APS-C cooke triplet HL ablation pipeline:
#   1. baseline refine of existing best archive (cooke_apsc_85mm_f4_20260504_225808)
#   2. 13-variant sweep via run_apsc_triplet_ablation.sh
#
# Usage: run_apsc_triplet_full_pipeline.sh <ALLOC_JOBID>
set -uo pipefail

ALLOC=${1:?missing alloc id}
HPY=/nas/longleaf/home/hisham26/miniforge3/envs/hyperlens/bin/python
WORKDIR=/nas/longleaf/home/hisham26/Desktop/hyperlens
LOGDIR="$WORKDIR/results/apsc_triplet_refsweep_logs_alloc${ALLOC}"
BASELINE_RUN_DIR="$WORKDIR/results/lensmega/cooke_apsc_85mm_f4_20260504_225808"
BASELINE_CFG=optical_rl/qd/lensmega/config/examples/cooke_apsc_85mm_f4_discovery.yaml
mkdir -p "$LOGDIR"

cd "$WORKDIR"

echo "[$(date +%H:%M:%S)] === PIPELINE START alloc=$ALLOC ==="

# ---- Step 1: baseline refine (uses unmodified APS-C config) ----
BASELINE_OUT="$BASELINE_RUN_DIR/refined_top500_seed42_apsc_baseline.json"
echo "[$(date +%H:%M:%S)] [baseline-refine] -> $BASELINE_OUT"
srun --jobid="$ALLOC" --overlap --exact --ntasks=1 --cpus-per-task=30 --mem=40G \
    --chdir="$WORKDIR" \
    "$HPY" tools/refine_all_parallel.py \
    --pool-from hl --hl-run "$BASELINE_RUN_DIR" \
    --problem-config "$BASELINE_CFG" \
    --threshold -1.0 --top-n 500 --workers 30 \
    --out "$BASELINE_OUT" \
    > "$LOGDIR/_baseline_refine.log" 2>&1
RC=$?
echo "[$(date +%H:%M:%S)] [baseline-refine] done rc=$RC"

# ---- Step 2: 13-variant sweep ----
# Order: most-informative-first so an early alloc death still yields signal.
# F_patience30 is the headline "can we break the 135 µm floor" knob (per RUNBOOK).
TAGS=(
    F_patience30          # breakthrough candidate #1
    G_grid2048            # sixlet's biggest alpha; aggressive
    G_grid1024            # safer grid bump
    S_sigma10             # sixlet winner #2
    F_patience300         # diversity-side check
    D_long_lbfgs          # refine-only saturation check, very fast
    G_grid512             # one-step finer grid
    W_wider_bd            # bounds widen
    T_tighter_bd          # bounds tighten
    B_autoscale_bd        # let calibration auto-pick
    P_pop40               # 2x population
    E_cma_eta01           # gentler CMA step
    G_grid128             # coarser grid (robustness footnote)
)

echo "[$(date +%H:%M:%S)] [sweep] launching ${#TAGS[@]} variants: ${TAGS[*]}"
bash "$WORKDIR/scripts_throwaway/run_apsc_triplet_ablation.sh" "$ALLOC" 40G "${TAGS[@]}"
SWEEP_RC=$?
echo "[$(date +%H:%M:%S)] [sweep] done rc=$SWEEP_RC"

echo "[$(date +%H:%M:%S)] === PIPELINE END alloc=$ALLOC ==="
