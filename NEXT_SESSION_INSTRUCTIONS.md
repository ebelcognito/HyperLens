# Next session — triplet + sixlet + octuplet diversity comparison

## Agent quick-start (do this right away)

1. **Read `EXPERIMENT_LOG_sa_ga_vs_hyperlens_sixlet.md`** in the same dir for sixlet-system context (what was done in the prior session).
2. **Get the two SLURM JOBIDs** from the user (they should have salloc'd 2 × 32-CPU `hov` allocations, ≥4 hr time, before going to sleep).
3. **Launch in a fresh tmux session:**
   ```bash
   tmux new -s diversity-runner
   cd /nas/longleaf/home/hisham26/Desktop/hyperlens
   bash run_diversity_experiments.sh <JOBID_HL> <JOBID_SAGA> 100000
   # C-b d to detach
   ```
4. **Monitor** by tailing `results/diversity_run_<TS>/logs/master.log`. Expected wall **~2.5 hours**.
5. **When complete**, run the post-launch tasks listed in §"After the launcher finishes" below.
6. If anything errors mid-run, see §"Failure modes / fixes" below.

Python env: `/nas/longleaf/home/hisham26/miniforge3/envs/hyperlens/bin/python`
Working dir for everything: `/nas/longleaf/home/hisham26/Desktop/hyperlens`

---

This is the action handoff for the next chat. **Read [EXPERIMENT_LOG_sa_ga_vs_hyperlens_sixlet.md](hyperlens/EXPERIMENT_LOG_sa_ga_vs_hyperlens_sixlet.md) first** for full sixlet-system context (already done in the prior session).

## Goal

Replicate the sixlet head-to-head (HyperLens vs SA+GA+Refine, ε-net diversity at matched 100k compute budget) on **triplet** and **octuplet** systems. Two seeds per (system, method) for variance evidence.

## Pipeline (already coded)

3 systems × 2 seeds × 2 methods = **12 search runs + 12 refinement passes + cross-system analysis**.

| Phase | Content | Wall |
|---|---|---|
| 1. Search | 3 sys × 2 seeds × 2 methods at 100k iters; HL ‖ SA+GA per pair, pairs sequential | ~110 min |
| 2. Refinement | Top-500 HL + all SA+GA above -1.0 per pool, parallel | ~30 min |
| 3. Analysis | ε-net diversity + RMS distribution, cross-system & cross-seed table | ~10 min |
| **Total** | | **~2.5 hours** |

## What is prepared

| Artifact | Path |
|---|---|
| Master launcher | [`run_diversity_experiments.sh`](hyperlens/run_diversity_experiments.sh) |
| SA+GA driver (with `--no-refine`) | [`sa_ga_refine_baseline.py`](hyperlens/sa_ga_refine_baseline.py) |
| SA+GA hyperparam config (broad threshold, max_tracked=2000) | [`configs/sa_ga_baseline_norefine_100k.yaml`](hyperlens/configs/sa_ga_baseline_norefine_100k.yaml) |
| Triplet config | [`optical_rl/qd/lensmega/config/examples/cooke_triplet_materials_modernstack.yaml`](hyperlens/optical_rl/qd/lensmega/config/examples/cooke_triplet_materials_modernstack.yaml) |
| Sixlet config | [`optical_rl/qd/lensmega/config/examples/sixlet_apsc_discovery_v2.yaml`](hyperlens/optical_rl/qd/lensmega/config/examples/sixlet_apsc_discovery_v2.yaml) |
| **Octuplet v2 config** (fixed bounds vs v1's `null` → wasted cells) | [`optical_rl/qd/lensmega/config/examples/octuplet_apsc_2cement_v2.yaml`](hyperlens/optical_rl/qd/lensmega/config/examples/octuplet_apsc_2cement_v2.yaml) |
| Parallel refiner (with `--top-n` cap) | [`tools/refine_all_parallel.py`](hyperlens/tools/refine_all_parallel.py) |

## Pre-flight: Hisham does this before sleep

Two CPU allocations on `hov` partition with at least 3 hours of remaining time:

```bash
# Window 1
tmux new -s alloc-hl
salloc --partition=hov --nodes=1 --ntasks=1 --cpus-per-task=32 \
       --mem=48G --time=4:00:00
# note JOBID, C-b d to detach

# Window 2
tmux new -s alloc-saga
salloc --partition=hov --nodes=1 --ntasks=1 --cpus-per-task=32 \
       --mem=48G --time=4:00:00
# note JOBID, C-b d to detach
```

Then tell next-chat agent: `JOBID_HL=<...>  JOBID_SAGA=<...>`.

## Launching (next-chat agent does this)

From a third tmux session so the launch survives disconnect:

```bash
tmux new -s diversity-runner
cd /nas/longleaf/home/hisham26/Desktop/hyperlens
bash run_diversity_experiments.sh <JOBID_HL> <JOBID_SAGA> 100000
# C-b d to detach
```

Output rooted at `results/diversity_run_<TS>/`. HL searches additionally write into `results/lensmega/<runname>_seed<N>_<TS>/`. The launcher logs everything to `results/diversity_run_<TS>/logs/`.

## What the launcher does

For each `(system, seed)` pair, in order:

1. Generates a per-seed YAML clone of the HL config (overrides `cma_seed` and `run_name`)
2. Generates a per-seed clone of the SA+GA baseline config
3. Launches HL on `JOBID_HL` and SA+GA --no-refine on `JOBID_SAGA`, both in parallel
4. Waits for both to finish

Then for each `(system, seed)` pair:

5. Refines top-500 HL designs (raw fit ≥ -1.0) on `JOBID_HL`
6. Refines all SA+GA designs above -1.0 on `JOBID_SAGA`
7. Both in parallel, waits for both

Then runs a Python analysis block:

8. Loads all 12 refined-pool JSONs
9. ε-net diversity at refined RMS thresholds {<30μm, <100μm, <1mm} with ε=0.20
10. Aggregates mean ± std across seeds per (system, threshold)
11. Prints a cross-system summary table

## Estimated time / sanity

- Triplet HL @ 100k on a clean 32-CPU node: ~10 min (D=19, fast)
- Sixlet HL @ 100k: ~15 min (D=37)
- Octuplet HL @ 100k: ~30 min (D=45, slower per-eval)
- SA+GA --no-refine has comparable wall on parallel allocation
- Refinement ~5 min per pool × 12 pools / 2 allocs = ~30 min
- Analysis ~10 min

**Total ~2.5–3 hours wall.** Allocations need 4 hours minimum to be safe.

## Failure modes / fixes

**HL output dir not found.** The launcher captures the latest matching `results/lensmega/<runname>_seed<N>_*` dir. If HL failed in seeding, check `<dir>/trajectory.csv`. If empty, re-launch that specific HL run with the same allocation.

**SLURM allocation expired mid-run.** Check `squeue -u hisham26`. Re-request and re-launch from whichever phase failed. Phase 1 search produces archive.pkl + unrefined_elites_sa_ga.json — both must exist before Phase 2.

**Octuplet bounds wasted again.** Check `<octuplet_run>/archive_map.png`. If cells bunch in one corner, the v2 bounds (`[-0.005, 0.005] × [-0.008, 0.008]`) are still wrong. Inspect actual filled cells via `python -c "import pickle, numpy as np; a=pickle.load(open('<dir>/archive.pkl','rb')); m=np.array([(v.measures[0],v.measures[1]) for v in a.cells.values() if v]); print(m.min(0),m.max(0))"` and adjust accordingly.

**Co-tenant CPU contention.** Earlier session had a co-tenant on c130602 cutting HL throughput ~30%. If suspected, check `srun --jobid=$ALLOC --overlap --exact -c2 -m4G ps -ef | grep -v hisham26 | grep python` on the compute node.

**Refinement worker crash with NaN.** `_worker_refine` in `refine_all_parallel.py` catches exceptions and returns `{"error": ...}`. Check `n_refined_err` field.

## After the launcher finishes

The agent should:

1. Read `results/diversity_run_<TS>/logs/analysis.log` and confirm:
   - HL has higher ε=0.20-distinct count than SA+GA at sub-100μm threshold across all 3 systems
   - Variance across seeds is reasonable (std < mean for each method)
2. Append a section "12. Triplet + Octuplet replication (multi-seed)" to the experiment log with the headline cross-system table.
3. Render top-9 ε=0.20-distinct grids per system × method (6 grids total) using the same matplotlib + `_apply_theta_generic` pattern from the previous sixlet renders. Output to `results/diversity_run_<TS>/grids/`.
4. Tell Hisham (in chat) the headline numbers when he wakes up.

## Open work (NOT for tonight)

The user has explicitly deferred these:
- Manufacturability yield (Monte Carlo tolerance perturbation) — too incremental as standalone
- Glass-snap survival — promising 0-to-1, save for next session
- Archetype discovery clustering — promising 0-to-1, save for next session
- Spec-adaptation analysis — abstract, save for next session
- Higher-budget octuplet at 500k — reserve for final paper figures

If the simple comparison succeeds tonight, those become next-session tasks.

## Files NOT to touch

- `EXPERIMENT_LOG_sa_ga_vs_hyperlens_sixlet.md` (append-only — don't overwrite)
- The 2026-05-05 sixlet matched-100k results dirs (canonical reference data)
- `results/lensmega/sixlet_apsc_discovery_v2_20260505_140231` (the 400k sixlet HL reference)

End of instructions.
