#!/usr/bin/env python3
"""Generate sixlet method-ablation configs (Seidel vs Random-projection BC).

Clones configs/ablations/sixlet_REFSWEEP2_CHAMPION.yaml verbatim and writes
6 sibling configs differing ONLY in:
  - cma_seed (42 / 43 / 44)
  - descriptor.kind (seidel / random)
  - for random: descriptor.params + archive.bounds=null (autocalibrate)
  - io.run_name

The seed-42 Seidel variant is generated for completeness/reproducibility,
but the existing CHAMPION run (results/lensmega/sixlet_REFSWEEP2_CHAMPION_seed42_20260509_080537)
already covers that arm and will be reused.

Output goes to configs/ablations/sixlet_methodabl_*.yaml.
"""
import copy
from pathlib import Path
import yaml

REPO = Path('/nas/longleaf/home/hisham26/Desktop/hyperlens')
BASE = REPO / 'configs/ablations/sixlet_REFSWEEP2_CHAMPION.yaml'
OUT_DIR = REPO / 'configs/ablations'

SEEDS = [42, 43, 44]
RANDBC_DESC_SEEDS = {42: 12345, 43: 12346, 44: 12347}

base = yaml.safe_load(BASE.read_text())
assert base['descriptor']['kind'] == 'seidel', 'expected seidel base'
assert base['archive']['bins'] == [2048, 2048], 'expected CHAMPION 2048 bins'

ITER_CAP = 100000  # bounds wall to ~45 min; CHAMPION naturally stopped at ~81k

written = []
for arm in ('seidel', 'randbc'):
    for seed in SEEDS:
        cfg = copy.deepcopy(base)
        cfg['algorithm']['cma_seed'] = seed
        cfg['algorithm']['iterations'] = ITER_CAP

        if arm == 'seidel':
            # keep descriptor + bounds verbatim
            pass
        else:
            cfg['descriptor'] = {
                'kind': 'random',
                'params': {'dim': 2, 'seed': RANDBC_DESC_SEEDS[seed]},
            }
            # autocalibrate bounds: random projection lives in different units
            cfg['archive']['bounds'] = None

        run_name = f'sixlet_methodabl_{arm}_seed{seed}'
        cfg['io']['run_name'] = run_name

        out_path = OUT_DIR / f'{run_name}.yaml'
        header = (
            f"# METHOD ABLATION: arm={arm} seed={seed}\n"
            f"# Cloned from sixlet_REFSWEEP2_CHAMPION.yaml; iterations capped to {ITER_CAP}\n"
        )
        out_path.write_text(header + yaml.dump(cfg, sort_keys=False, default_flow_style=None))
        written.append(out_path.name)
        print(f"wrote {out_path}")

print(f"\nGenerated {len(written)} configs.")
