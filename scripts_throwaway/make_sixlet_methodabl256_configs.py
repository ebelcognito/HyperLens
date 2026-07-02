#!/usr/bin/env python3
"""Generate sixlet method-ablation configs at bins=256² (vs CHAMPION's 2048²).

The hypothesis: at 2048² the archive operates as a sparse novelty-counter,
washing out the BC's structural information. At 256² the archive fills more
densely (like the triplet OLD ablation at 40-63% coverage), so BC structure
should matter more. If randbc shows the 3× ε-net diversity advantage seen on
the triplet, the grid-resolution mechanism is real.

Everything else matches CHAMPION verbatim: pop=24, σ=1.0, η=0.1, patience=30,
iters=100000, refine pipeline, etc.

Output: configs/ablations/sixlet_methodabl256_{seidel,randbc}_seed{42,43,44}.yaml
"""
import copy
from pathlib import Path
import yaml

REPO = Path('/nas/longleaf/home/hisham26/Desktop/hyperlens')
BASE = REPO / 'configs/ablations/sixlet_REFSWEEP2_CHAMPION.yaml'
OUT_DIR = REPO / 'configs/ablations'

SEEDS = [42, 43, 44]
RANDBC_DESC_SEEDS = {42: 12345, 43: 12346, 44: 12347}
ITER_CAP = 100000
BINS = [256, 256]

base = yaml.safe_load(BASE.read_text())
assert base['descriptor']['kind'] == 'seidel'
assert base['archive']['bins'] == [2048, 2048], 'expected CHAMPION 2048 base'

written = []
for arm in ('seidel', 'randbc'):
    for seed in SEEDS:
        cfg = copy.deepcopy(base)
        cfg['algorithm']['cma_seed'] = seed
        cfg['algorithm']['iterations'] = ITER_CAP

        # KEY CHANGE: 2048² → 256²; cell area grows 64×
        cfg['archive']['bins'] = BINS

        if arm == 'randbc':
            cfg['descriptor'] = {
                'kind': 'random',
                'params': {'dim': 2, 'seed': RANDBC_DESC_SEEDS[seed]},
            }
            cfg['archive']['bounds'] = None  # autocalibrate per seed

        run_name = f'sixlet_methodabl256_{arm}_seed{seed}'
        cfg['io']['run_name'] = run_name

        out_path = OUT_DIR / f'{run_name}.yaml'
        header = (
            f"# METHOD ABLATION @ 256² grid: arm={arm} seed={seed}\n"
            f"# Cloned from sixlet_REFSWEEP2_CHAMPION.yaml; bins=2048²→256², iters={ITER_CAP}\n"
            f"# Hypothesis: lower grid resolution lets BC structure matter more.\n"
        )
        out_path.write_text(header + yaml.dump(cfg, sort_keys=False, default_flow_style=None))
        written.append(out_path.name)
        print(f"wrote {out_path}")

print(f"\nGenerated {len(written)} configs.")
