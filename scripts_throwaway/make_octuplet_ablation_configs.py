#!/usr/bin/env python3
"""Generate octuplet HL ablation configs (mirrors sixlet refsweep).

Each config = clone of octuplet_apsc_2cement_v2.yaml with ONE knob changed.
"""
import os
import yaml

BASE = 'optical_rl/qd/lensmega/config/examples/octuplet_apsc_2cement_v2.yaml'
OUT_DIR = 'configs/ablations'

SWEEPS = {
    # Phase A — refinement (just one to confirm saturation)
    'D_long_lbfgs':     {'io.refine_lbfgs_max_steps': 300, 'io.refine_converge_tol': 1e-5},

    # Phase B — search-side
    'G_grid128':        {'archive.bins': [128, 128]},
    'G_grid1024':       {'archive.bins': [1024, 1024]},
    'G_grid2048':       {'archive.bins': [2048, 2048]},
    'G_grid3000':       {'archive.bins': [3000, 3000]},
    'W_wider_bd':       {'archive.bounds': [[-0.001, 0.001], [-0.001, 0.001]]},
    'T_tighter_bd':     {'archive.bounds': [[-0.0003, 0.0003], [-0.0005, 0.0005]]},
    'P_pop64':          {'algorithm.population_size': 64},
    'S_sigma10':        {'algorithm.initial_sigma': 1.0},
    'F_patience20':     {'algorithm.fitness_patience': 20},
    'F_patience150':    {'algorithm.fitness_patience': 150},

    # Bonus — sixlet's winning combo applied blindly to octuplet
    # (sixlet CHAMPION = bins 2048² + σ 1.0 + patience 30).
    # Compares against an octuplet-tuned CHAMPION later: do alphas transfer?
    'BLIND_COMBO': {
        'archive.bins': [2048, 2048],
        'algorithm.initial_sigma': 1.0,
        'algorithm.fitness_patience': 30,
    },

    # Octuplet-tuned CHAMPION: layer the knobs that won individually on
    # octuplet (G_grid2048 + S_sigma10 + F_patience20). Differs from
    # BLIND_COMBO only in patience (20 vs 30). Direct test: per-topology
    # tuning vs sixlet-blind transfer.
    'CHAMPION': {
        'archive.bins': [2048, 2048],
        'algorithm.initial_sigma': 1.0,
        'algorithm.fitness_patience': 20,
    },
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
        cfg.setdefault('io', {})['run_name'] = f'octuplet_REFSWEEP_{tag}_seed42'
        out_path = os.path.join(OUT_DIR, f'octuplet_REFSWEEP_{tag}.yaml')
        with open(out_path, 'w') as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
