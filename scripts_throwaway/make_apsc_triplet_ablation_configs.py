#!/usr/bin/env python3
"""Generate APS-C cooke triplet HL ablation configs (mirrors sixlet/octuplet refsweep).

Each config = clone of cooke_apsc_85mm_f4_discovery.yaml with ONE knob changed
(except CHAMPION, written separately after Phase B results).

13 variants per RUNBOOK_triplet_hl_ablation.md.
"""
import os
import yaml

BASE = 'optical_rl/qd/lensmega/config/examples/cooke_apsc_85mm_f4_discovery.yaml'
OUT_DIR = 'configs/ablations'
PREFIX = 'apsc_triplet_REFSWEEP'

SWEEPS = {
    # Phase A — refinement saturation check.
    # Baseline already at max_steps=200 (sixlet was 100), so push to 500 + tighter tol.
    'D_long_lbfgs':    {'io.refine_lbfgs_max_steps': 500, 'io.refine_converge_tol': 1.0e-6},

    # Phase B — search-side knobs.
    'G_grid128':       {'archive.bins': [128, 128]},
    'G_grid512':       {'archive.bins': [512, 512]},
    'G_grid1024':      {'archive.bins': [1024, 1024]},
    'G_grid2048':      {'archive.bins': [2048, 2048]},  # aggressive — testing if sixlet's biggest alpha generalizes
    'W_wider_bd':      {'archive.bounds': [[-0.0016, 0.0016], [-0.0016, 0.0016]]},
    'T_tighter_bd':    {'archive.bounds': [[-0.0004, 0.0004], [-0.0004, 0.0004]]},
    'B_autoscale_bd':  {'archive.bounds': None},
    'P_pop40':         {'algorithm.population_size': 40},
    'S_sigma10':       {'algorithm.initial_sigma': 1.0},
    'F_patience30':    {'algorithm.fitness_patience': 30},
    'F_patience300':   {'algorithm.fitness_patience': 300},
    'E_cma_eta01':     {'algorithm.cma_eta': 0.1},
}


def set_nested(d, dotted, val):
    keys = dotted.split('.')
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = val


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for tag, mut in SWEEPS.items():
        with open(BASE) as f:
            cfg = yaml.safe_load(f)
        for k, v in mut.items():
            set_nested(cfg, k, v)
        cfg.setdefault('algorithm', {})['cma_seed'] = 42
        cfg.setdefault('io', {})['run_name'] = f'{PREFIX}_{tag}_seed42'
        out_path = os.path.join(OUT_DIR, f'{PREFIX}_{tag}.yaml')
        with open(out_path, 'w') as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
