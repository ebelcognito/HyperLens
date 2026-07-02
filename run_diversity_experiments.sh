#!/bin/bash
# ============================================================================
# Comprehensive diversity experiments — multi-system, multi-seed
#
# 3 systems × 3 seeds × 2 methods × matched-budget search
#   + top-500 refinement on each pool
#   + manufacturability Monte-Carlo yield on sub-30μm subsets
#   + cross-system / cross-seed analysis
#
# Estimated wall: ~3.5–4.5 hours on 2 allocations (HL + SA+GA in parallel
# per system×seed pair; pairs run sequentially).
#
# Usage:
#   bash run_diversity_experiments.sh <ALLOC_HL> <ALLOC_SAGA> [budget=100000]
#
# Each allocation needs at least 5 hr of remaining time.
# ============================================================================

set -uo pipefail

if [[ $# -lt 2 ]]; then
    echo "ERROR: usage: $0 <ALLOC_HL> <ALLOC_SAGA> [budget=100000]" >&2
    exit 1
fi
ALLOC_HL=$1
ALLOC_SAGA=$2
BUDGET=${3:-100000}

cd /nas/longleaf/home/hisham26/Desktop/hyperlens

HPY=/nas/longleaf/home/hisham26/miniforge3/envs/hyperlens/bin/python
TS=$(date +%Y%m%d_%H%M%S)
ROOT="results/diversity_run_${TS}"
LOGDIR="${ROOT}/logs"
CFGDIR="${ROOT}/configs"
mkdir -p "$LOGDIR" "$CFGDIR"

SAGA_BASE_CFG=configs/sa_ga_baseline_norefine_100k.yaml
declare -A SYS_CFG
SYS_CFG[triplet]=optical_rl/qd/lensmega/config/examples/cooke_triplet_materials_modernstack.yaml
SYS_CFG[sixlet]=optical_rl/qd/lensmega/config/examples/sixlet_apsc_discovery_v2.yaml
SYS_CFG[octuplet]=optical_rl/qd/lensmega/config/examples/octuplet_apsc_2cement_v2.yaml

declare -A SYS_RUN_NAME
SYS_RUN_NAME[triplet]=cooke_triplet_materials_modernstack
SYS_RUN_NAME[sixlet]=sixlet_apsc_discovery_v2
SYS_RUN_NAME[octuplet]=octuplet_apsc_2cement_v2

SYSTEMS=(triplet sixlet octuplet)
SEEDS=(42 43)

log() { echo "[$(date +'%H:%M:%S')] $*" | tee -a "$LOGDIR/master.log"; }

# ============================================================================
# Helper: clone a YAML config with cma_seed and run_name overridden
# ============================================================================
clone_hl_cfg() {
    local sys=$1 seed=$2
    local base_cfg="${SYS_CFG[$sys]}"
    local base_run="${SYS_RUN_NAME[$sys]}"
    local out_cfg="$CFGDIR/hl_${sys}_seed${seed}.yaml"
    local out_run="${base_run}_seed${seed}"
    $HPY -c "
import yaml
with open('$base_cfg') as f: c = yaml.safe_load(f)
c['algorithm']['cma_seed'] = $seed
c['io']['run_name'] = '${out_run}'
with open('$out_cfg', 'w') as f: yaml.safe_dump(c, f, sort_keys=False)
"
    echo "$out_cfg"
}

clone_saga_cfg() {
    local seed=$1
    local out_cfg="$CFGDIR/saga_seed${seed}.yaml"
    $HPY -c "
import yaml
with open('$SAGA_BASE_CFG') as f: c = yaml.safe_load(f)
c['seed'] = $seed
with open('$out_cfg', 'w') as f: yaml.safe_dump(c, f, sort_keys=False)
"
    echo "$out_cfg"
}

# ============================================================================
# Phase 1: Search runs (HL ‖ SA+GA per system×seed pair, pairs sequential)
# ============================================================================
log "===== PHASE 1: SEARCH (3 systems × 3 seeds × 2 methods) ====="

declare -A HL_DIR SAGA_DIR

for SEED in "${SEEDS[@]}"; do
    SAGA_CFG=$(clone_saga_cfg $SEED)
    for SYS in "${SYSTEMS[@]}"; do
        log "----- search: ${SYS} seed=${SEED} -----"
        HL_CFG=$(clone_hl_cfg $SYS $SEED)
        SAGA_OUT="${ROOT}/${SYS}/seed${SEED}/sa_ga"
        HL_LOG="$LOGDIR/${SYS}_seed${SEED}_hl.log"
        SAGA_LOG="$LOGDIR/${SYS}_seed${SEED}_saga.log"
        mkdir -p "$SAGA_OUT"

        # Launch HL
        srun --jobid=$ALLOC_HL --overlap --exact --ntasks=1 \
            --cpus-per-task=32 --mem=48G \
            --chdir=/nas/longleaf/home/hisham26/Desktop/hyperlens \
            $HPY -m optical_rl.qd.lensmega.run \
            --config $HL_CFG --iterations $BUDGET \
            > "$HL_LOG" 2>&1 &
        HL_PID=$!

        # Launch SA+GA --no-refine
        srun --jobid=$ALLOC_SAGA --overlap --exact --ntasks=1 \
            --cpus-per-task=32 --mem=48G \
            --chdir=/nas/longleaf/home/hisham26/Desktop/hyperlens \
            $HPY sa_ga_refine_baseline.py \
            --problem-config ${SYS_CFG[$SYS]} \
            --baseline-config $SAGA_CFG \
            --budget $BUDGET --out "$SAGA_OUT" \
            --torch-threads 1 --workers 30 --no-refine \
            > "$SAGA_LOG" 2>&1 &
        SAGA_PID=$!

        log "  HL pid=$HL_PID  SAGA pid=$SAGA_PID  waiting..."
        wait $HL_PID && log "  HL finished" || log "  HL FAILED ($?)"
        wait $SAGA_PID && log "  SAGA finished" || log "  SAGA FAILED ($?)"

        # Capture latest matching HL output dir
        HL_DIR_VAL=$(ls -td results/lensmega/${SYS_RUN_NAME[$SYS]}_seed${SEED}_* 2>/dev/null | head -1)
        if [[ -z "$HL_DIR_VAL" ]]; then
            HL_DIR_VAL=$(ls -td results/lensmega/${SYS_RUN_NAME[$SYS]}_* 2>/dev/null | head -1)
        fi
        HL_DIR["${SYS}_${SEED}"]="$HL_DIR_VAL"
        SAGA_DIR["${SYS}_${SEED}"]="$SAGA_OUT"
        log "  HL dir:   $HL_DIR_VAL"
        log "  SAGA dir: $SAGA_OUT"
    done
done

log "===== PHASE 1 complete ====="

# ============================================================================
# Phase 2: Refinement (top-500 HL + all SA+GA, parallel across allocs)
# ============================================================================
log "===== PHASE 2: REFINEMENT (top-500 HL + all SA+GA per pool) ====="

for SEED in "${SEEDS[@]}"; do
    for SYS in "${SYSTEMS[@]}"; do
        log "----- refine: ${SYS} seed=${SEED} -----"
        HL_RUN="${HL_DIR[${SYS}_${SEED}]}"
        SG_RUN="${SAGA_DIR[${SYS}_${SEED}]}"
        HL_REFINED="${HL_RUN}/refined_top500_seed${SEED}_hl.json"
        SG_REFINED="${SG_RUN}/refined_above_-1.0_seed${SEED}_sa_ga.json"
        HL_REF_LOG="$LOGDIR/${SYS}_seed${SEED}_refine_hl.log"
        SG_REF_LOG="$LOGDIR/${SYS}_seed${SEED}_refine_saga.log"

        # HL on ALLOC_HL
        srun --jobid=$ALLOC_HL --overlap --exact --ntasks=1 \
            --cpus-per-task=32 --mem=48G \
            --chdir=/nas/longleaf/home/hisham26/Desktop/hyperlens \
            $HPY tools/refine_all_parallel.py \
            --pool-from hl --hl-run "$HL_RUN" \
            --threshold -1.0 --top-n 500 \
            --workers 30 --out "$HL_REFINED" \
            > "$HL_REF_LOG" 2>&1 &
        HL_REF_PID=$!

        # SA+GA on ALLOC_SAGA
        srun --jobid=$ALLOC_SAGA --overlap --exact --ntasks=1 \
            --cpus-per-task=32 --mem=48G \
            --chdir=/nas/longleaf/home/hisham26/Desktop/hyperlens \
            $HPY tools/refine_all_parallel.py \
            --pool-from sa_ga --sa-ga-run "$SG_RUN" \
            --problem-config ${SYS_CFG[$SYS]} \
            --threshold -1.0 \
            --workers 30 --out "$SG_REFINED" \
            > "$SG_REF_LOG" 2>&1 &
        SG_REF_PID=$!

        log "  HL_REF pid=$HL_REF_PID  SAGA_REF pid=$SG_REF_PID  waiting..."
        wait $HL_REF_PID && log "  HL refine done" || log "  HL refine FAILED ($?)"
        wait $SG_REF_PID && log "  SAGA refine done" || log "  SAGA refine FAILED ($?)"
    done
done

log "===== PHASE 2 complete ====="

# ============================================================================
# Phase 3: Cross-system / cross-seed analysis
#   (manufacturability yield deferred — keeping pipeline simple per user choice)
# ============================================================================
log "===== PHASE 3: ANALYSIS ====="

# Build a JSON index of all run paths so the analyzer can load everything.
INDEX="${ROOT}/run_index.json"
$HPY <<EOF > /dev/null
import json
idx = {"systems": {}}
for sys in ${SYSTEMS[@]@A}:
    pass
EOF

# Use a python heredoc to do the analysis
$HPY <<EOF | tee "$LOGDIR/analysis.log"
import json, numpy as np
import sys, os
sys.path.insert(0, '.')
from optical_rl.qd.lensmega.config.loader import load_config
from itertools import combinations

SYSTEMS = ["triplet", "sixlet", "octuplet"]
SEEDS = [42, 43, 44]
SYS_CFG = {
    "triplet": "optical_rl/qd/lensmega/config/examples/cooke_triplet_materials_modernstack.yaml",
    "sixlet": "optical_rl/qd/lensmega/config/examples/sixlet_apsc_discovery_v2.yaml",
    "octuplet": "optical_rl/qd/lensmega/config/examples/octuplet_apsc_2cement_v2.yaml",
}
SYS_RUN = {
    "triplet": "cooke_triplet_materials_modernstack",
    "sixlet": "sixlet_apsc_discovery_v2",
    "octuplet": "octuplet_apsc_2cement_v2",
}
ROOT = "$ROOT"

def epsilon_net(designs, eps, lo, hi):
    rng = hi - lo
    rng = np.where(rng < 1e-12, 1e-12, rng)
    sqrtD = np.sqrt(len(lo))
    sd = sorted(designs, key=lambda x: -x[0])
    accepted = []
    for fit, theta in sd:
        u = (np.asarray(theta) - lo) / rng
        if not accepted: accepted.append(u); continue
        if min(np.linalg.norm(u-a)/sqrtD for a in accepted) > eps:
            accepted.append(u)
    return len(accepted)

print(f"\\n{'='*100}")
print(f"DIVERSITY HEAD-TO-HEAD across {len(SYSTEMS)} systems × {len(SEEDS)} seeds")
print(f"{'='*100}\\n")

summary_rows = []
for sys in SYSTEMS:
    cfg = load_config(SYS_CFG[sys])
    bounds = np.asarray(cfg.problem.theta_bounds, dtype=np.float64)
    lo = bounds[:,0]; hi = bounds[:,1]

    print(f"\\n##### {sys.upper()} #####")
    for seed in SEEDS:
        # locate dirs
        import glob
        hl_candidates = sorted(glob.glob(f"results/lensmega/{SYS_RUN[sys]}_seed{seed}_*"))
        if not hl_candidates:
            hl_candidates = sorted(glob.glob(f"results/lensmega/{SYS_RUN[sys]}_*"))
        if not hl_candidates:
            print(f"  seed={seed}: NO HL DIR FOUND, skipping")
            continue
        hl_dir = hl_candidates[-1]
        sg_dir = f"{ROOT}/{sys}/seed{seed}/sa_ga"
        hl_refined = f"{hl_dir}/refined_top500_seed{seed}_hl.json"
        sg_refined = f"{sg_dir}/refined_above_-1.0_seed{seed}_sa_ga.json"
        hl_yield = f"{hl_dir}/yield_seed{seed}_hl.json"
        sg_yield = f"{sg_dir}/yield_seed{seed}_sa_ga.json"

        if not os.path.exists(hl_refined) or not os.path.exists(sg_refined):
            print(f"  seed={seed}: refined JSON missing, skipping")
            continue

        hl = json.loads(open(hl_refined).read())
        sg = json.loads(open(sg_refined).read())

        hl_designs = hl.get("designs", [])
        sg_designs = sg.get("designs", [])

        for thresh, label in [(-0.030, "<30μm"), (-0.100, "<100μm"), (-1.000, "<1mm")]:
            hp = [(d['refined_fitness'], d['theta']) for d in hl_designs
                  if d['refined_fitness'] >= thresh]
            sp = [(d['refined_fitness'], d['theta']) for d in sg_designs
                  if d['refined_fitness'] >= thresh]
            hl_eps = epsilon_net(hp, 0.20, lo, hi) if hp else 0
            sg_eps = epsilon_net(sp, 0.20, lo, hi) if sp else 0
            summary_rows.append({
                'sys': sys, 'seed': seed, 'rms_thresh': label,
                'hl_n': len(hp), 'hl_eps': hl_eps,
                'sg_n': len(sp), 'sg_eps': sg_eps,
                'hl_eps_per_sg_eps': hl_eps / max(sg_eps, 1),
            })

        # Yield numbers
        hl_y = sg_y = None
        if os.path.exists(hl_yield):
            hl_y = json.loads(open(hl_yield).read())
        if os.path.exists(sg_yield):
            sg_y = json.loads(open(sg_yield).read())

        print(f"\\n  seed={seed}:")
        print(f"    HL pool:   {len(hl_designs):>4d} refined, "
              f"{sum(1 for d in hl_designs if d['rms_um']<30):>3d} sub-30μm")
        print(f"    SAGA pool: {len(sg_designs):>4d} refined, "
              f"{sum(1 for d in sg_designs if d['rms_um']<30):>3d} sub-30μm")
        for thresh, label in [(-0.030, "<30μm"), (-0.100, "<100μm"), (-1.000, "<1mm")]:
            row = next((r for r in summary_rows if r['sys']==sys and r['seed']==seed and r['rms_thresh']==label), None)
            if row:
                print(f"    {label:>8} ε=0.20:  HL={row['hl_eps']:>3d}  SAGA={row['sg_eps']:>3d}  ratio={row['hl_eps_per_sg_eps']:.2f}x")
        if hl_y and sg_y:
            print(f"    Manufacturability yield (1.0% tolerance, sub-30μm subset):")
            print(f"      HL:    {hl_y['yield_count']}/{hl_y['n_evaluated']} = {hl_y['yield_pct']:.1f}%")
            print(f"      SAGA:  {sg_y['yield_count']}/{sg_y['n_evaluated']} = {sg_y['yield_pct']:.1f}%")

print(f"\\n{'='*100}")
print("AGGREGATED ACROSS SEEDS (mean ± std per system×rms_threshold)")
print(f"{'='*100}\\n")
print(f"{'sys':<10} {'rms_thresh':>10} {'HL_eps mean±std':>20} {'SAGA_eps mean±std':>20} {'ratio mean':>12}")
for sys in SYSTEMS:
    for label in ["<30μm", "<100μm", "<1mm"]:
        rows = [r for r in summary_rows if r['sys']==sys and r['rms_thresh']==label]
        if not rows: continue
        hl_eps = np.array([r['hl_eps'] for r in rows])
        sg_eps = np.array([r['sg_eps'] for r in rows])
        ratios = np.array([r['hl_eps_per_sg_eps'] for r in rows])
        print(f"{sys:<10} {label:>10} "
              f"{hl_eps.mean():>10.1f}±{hl_eps.std():.1f}    "
              f"{sg_eps.mean():>10.1f}±{sg_eps.std():.1f}    "
              f"{ratios.mean():>10.2f}x")
EOF

log "===== ALL PHASES COMPLETE ====="
log "Results root:    $ROOT"
log "Master log:      $LOGDIR/master.log"
log "Analysis log:    $LOGDIR/analysis.log"
log "All run dirs:    look in $ROOT or results/lensmega/${SYS_RUN_NAME[*]}_seed*_*/"
