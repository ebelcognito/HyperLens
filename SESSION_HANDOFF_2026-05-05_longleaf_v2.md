# Longleaf session handoff — sixlet APS-C v2 lockin

Read this entire file before doing anything. It's short.

## Step 0 — User: request a fresh compute node (paste in tmux on login node)

```bash
salloc --partition=hov --nodes=1 --ntasks=1 --cpus-per-task=32 \
       --mem=48G --time=4:00:00
# Output will say: "salloc: Granted job allocation <JOBID>"
# Note the JOBID, leave the shell open, then tell the agent:
#   "JOBID=<JOBID>  NODE=<nodename>"  (also printed by salloc)
```

The `hov` partition is the patron partition tied to the user's `rc_pranee+` account — typically same-day scheduling (vs days of queue on `general --exclusive`). Drop `--exclusive` (not in the command above): the v2 config doesn't need exclusive, and dropping it is what unlocks the fast scheduling.

## Step 1 — Agent: how to run commands on the user's allocation

The user holds the allocation. You (running on the login node) reach the compute node via `srun --jobid=... --overlap`. Three quirks you must respect:

### Quirk A — `conda activate` does NOT work in `bash -c` subshells

Use the env's python directly:
```bash
HPY=/nas/longleaf/home/hisham26/miniforge3/envs/hyperlens/bin/python
```

`conda activate hyperlens` from inside a `bash -c` will silently no-op and you'll get system Python 3.9, not the env's 3.12 — which then fails at `import torch` (missing mpmath).

### Quirk B — `/tmp/` is node-local, NOT shared

The compute node and login node have different `/tmp/`. **Write all temp configs/scripts to a NAS path the compute node can read**:
```
/nas/longleaf/home/hisham26/Desktop/hyperlens/        ← shared, use this
/tmp/anything                                           ← won't exist on compute node
```

### Quirk C — `srun --overlap` defaults are broken

You need ALL of these flags or you get 1 CPU, no memory, parallel-tasked output, etc:
```
--jobid=<JOBID> --overlap --exact --ntasks=1 --cpus-per-task=32 --mem=48G
```

### The canonical pattern — one-liner template

```bash
JOBID=<from-user>
HPY=/nas/longleaf/home/hisham26/miniforge3/envs/hyperlens/bin/python
HYP=/nas/longleaf/home/hisham26/Desktop/hyperlens

srun --jobid=$JOBID --overlap --exact --ntasks=1 --cpus-per-task=32 --mem=48G bash -c "
cd $HYP
$HPY -m optical_rl.qd.lensmega.run --config <some-yaml-on-nas>
" 2>&1 | tee logs/<runname>.log
```

For background runs (let it cook while you do other things):
```bash
srun --jobid=$JOBID --overlap --exact --ntasks=1 --cpus-per-task=32 --mem=48G bash -c "..." \
  > logs/run.log 2>&1 &
# Check progress with: tail -20 logs/run.log
# Or with squeue -j $JOBID
```

For quick diagnostic queries on the compute node (low CPU/mem so you don't fight the main run):
```bash
srun --jobid=$JOBID --overlap --exact --ntasks=1 --cpus-per-task=2 --mem=4G bash -c "<cmd>"
```

## Current state of the work (as of 2026-05-05)

### v2 sixlet APS-C is the lockin — stop tuning it

- Config:  `optical_rl/qd/lensmega/config/examples/sixlet_apsc_discovery_v2.yaml`
- Lens:    `DeepLens/datasets/lenses/sixlet_50mm_f4_apsc_v2.json`
- Best run: `results/lensmega/sixlet_apsc_discovery_v2_20260505_140231/`

What v2 fixed (vs v1 `sixlet_apsc_discovery.yaml` + `sixlet_50mm_f4_apsc.json`):
- v1 had `c_max=0.08` and `r_CA_max=33mm` → max SAG = 43.6 mm → catastrophic
- v2 has `c_max=0.05` and `r_CA_max=22mm` → max SAG = 12.1 mm → workable
- Seed rejection: 97% → **0%**.
- Best refined RMS: never below ~70μm in v1 → **25.9μm at gen ~6000 in v2**.
- Refinement gave clean 2.7× RMS improvement (69.9 → 25.9 μm), L-BFGS converged in 29 steps, EFL nailed to 50.014mm, F/4.93.
- Yellow flag: best elite has `pct_survived=20%` (sharp basin). Need to refine multiple elites and pick the high-stability one before claiming a manufacturable design.

### Current top-5 / four-stage tools

- `refine_prune_render_top5_diverse.py` — refines top-5 diverse elites and renders. Modes:
  - default: refine then prune (cosmetic)
  - `--prune-first --prune-shrink-fraction 0.3`: shrink CAs first, then refine
  - `--four-stage [--stage1-shrink 0.20]`: shrink → refine → real prune → refine again. **This is the move for the v2 elites.**

### Things explicitly decided AGAINST

- **Don't implement Fix 3 (ray-aware intersect penalty in `penalties.py`).** v2's structural fix is already producing 25.9μm refined designs in <200k evals. Fix 3 was on the table when v2 hadn't been validated yet — now it's not needed.
- **Don't `enforce_fnum: true` for the sixlet APS-C.** User explicitly chose `enforce_fnum: false`. The lens drifted to F/4.93 (target F/4.0); that's accepted.
- **Don't bump `cma_eta` above 0.10** for APS-C templates. v1 hit sigma blow-ups even at 0.10; modern anti-runaway stack saves it but bigger eta would amplify the problem.

## Useful runs / artifacts on disk

| Path | What |
|---|---|
| `results/lensmega/sixlet_apsc_discovery_v2_20260505_140231/` | The good v2 run — 197k evals, 25.9μm best refined RMS |
| `results/lensmega/sixlet_apsc_discovery_20260505_022335/` | v1 reference (broken, kept for comparison) |
| `results/lensmega/cooke_triplet_materials_baseline_20260504_233239/` | A/B baseline (legacy stack) for cooke triplet |
| `results/lensmega/cooke_triplet_materials_modernstack_20260504_234450/` | A/B modern stack — 31% better best fit, 69% more high-quality cells |
| `optical_rl/qd/lensmega/config/examples/cooke_triplet_materials_modern.yaml` | A/B baseline-emulating config |
| `optical_rl/qd/lensmega/config/examples/cooke_triplet_materials_modernstack.yaml` | A/B modern stack config |
| `sixlet_v2_bench.yaml` | Bench variant of v2 (pop=32, iterations=30000) used while validating |

## Outstanding work the next agent could pick up

- **Refine 5 diverse elites from the good v2 run** with `--four-stage`. Pick the one with highest `pct_survived` for actual lens. Worth ~10 min.
- **Apply the modern anti-runaway stack to other configs.** Many `*.yaml` examples still have `new_cell_bonus: 1.0e6` from the legacy era — see the cooke A/B for the pattern. Especially: `cooke_triplet_seidel.yaml`, `cooke_triplet_full_dof.yaml`, `quadruplet_*.yaml`, `septlet_F3_discovery.yaml`.
- **Fan-out campaign.** With v2 working, kick off 5-10 ablations concurrently as separate sbatch jobs (each on a 32-cpu node). The session export from 2026-05-04 has the launcher template. ~5-15 min wall per ablation, can finish a campaign in one lunch.
- **Match-glass + manufacturability pass.** None of the elites have been snapped to a real glass catalog yet (`match_materials` exists in DeepLens). Stability survives that step? open question.

## Quick command cheat sheet for the next agent

```bash
# Where am I?
hostname     # if longleaf-loginN — you're on login. Use srun --jobid for compute.

# Is the user's allocation alive?
squeue -j $JOBID

# Tail the latest lensmega run live
ls -td /nas/longleaf/home/hisham26/Desktop/hyperlens/results/lensmega/*/ | head -1
tail -f <that_dir>/../logs/<runname>.log

# Cancel a stuck job
scancel <JOBID>

# Quick "is the run actually progressing" check
srun --jobid=$JOBID --overlap --exact --ntasks=1 --cpus-per-task=2 --mem=4G bash -c "
  cd /nas/longleaf/home/hisham26/Desktop/hyperlens
  ps -ef | grep hisham26 | grep python | wc -l    # should be ~33 (1 main + 32 workers)
  ls -lh results/lensmega/<runname>/lineage.jsonl  # should be growing
"
```

## Things NOT to do

- Don't run anything heavy on the login node (`longleaf-login{1,2,3,4}`). Workers will be killed by Longleaf admins.
- Don't `git push` or modify the `.git/config`. The PAT episode from 2026-05-04 was bad enough; the user should be on SSH remotes by now (verify `git remote -v`).
- Don't write large output files to `/nas/longleaf/home/hisham26/` — home is at 51GB quota and was at 60GB earlier (over). Use `results/lensmega/` (which is on the same NAS but the user has more headroom).
- Don't kill the user's allocation. They hold it; you ride along.
