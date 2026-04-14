# Optical RL

Reinforcement learning and quality-diversity (QD) optimization for automated optical lens design.

## Overview

This project combines RL-based lens optimization with CMA-MEGA (Covariance Matrix Adaptation MAP-Elites via a Gradient Arborescence) to discover diverse, high-performing optical lens configurations. The system can optimize singlet through sixlet lens systems across curvature, gap, and glass parameters while maintaining diversity in aberration space (Seidel coefficients, Zernike descriptors).

### Key Components

- **`optical_rl/envs/`** — Gymnasium environments for RL-based lens optimization (single/dual surface curvature, quad curve glass, randomized slab)
- **`optical_rl/qd/lensmega/`** — LensMEGA: a CMA-MEGA engine for quality-diversity lens discovery
  - `problem/` — Lens parameterization, ray tracing, fitness evaluation, physical constraints
  - `descriptors/` — Behavior characterization via Seidel aberrations and Zernike coefficients
  - `archive/` — Grid-based MAP-Elites archive with auto-calibration
  - `core/` — Parallel CPU worker pool, output management, seeding
  - `config/` — YAML-driven configuration with validated schemas
- **`optical_rl/utils/`** — Ray-optics bridge, scene building, metrics, visualization
- **`optical_rl/experiments/`** — Focal length experiments (mid-solve, soft penalty, power budget, image height)
- **`optical_rl/configs/`** — Training configurations for various lens topologies

## Dependencies

This project depends on the following external repositories (not included):

- [DeepLens](https://github.com/singer-yang/DeepLens) — Differentiable ray tracing engine
- [ray-optics](https://github.com/mjhoptics/ray-optics) — Sequential ray tracing library

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Additional dependencies (install separately):
- PyTorch (with CUDA support recommended)
- DeepLens (clone and install from source)

## Usage

### LensMEGA — Quality-Diversity Lens Discovery

```bash
python -m optical_rl.qd.lensmega.run --config optical_rl/qd/lensmega/config/examples/cooke_triplet_seidel.yaml
```

Example configs in `optical_rl/qd/lensmega/config/examples/`:
- `cooke_triplet_seidel.yaml` — Cooke triplet with Seidel behavior descriptors
- `ef50_discovery.yaml` — Double Gauss EF50 discovery
- `sixlet_discovery.yaml` — 6-element full DOF discovery
- `quadruplet_discovery.yaml` — 4-element system

### RL Training

```bash
python optical_rl/main_stage1.py
```

## Project Structure

```
optical_rl/
├── envs/                  # Gymnasium RL environments
├── qd/
│   ├── lensmega/          # CMA-MEGA quality-diversity engine
│   ├── cmamega_ext/       # Extended CMA-MEGA variants
│   └── cmamega_triplet/   # Triplet-specific QD
├── experiments/           # Controlled experiments
├── configs/               # Training YAML configs
└── utils/                 # Shared utilities
tests/                     # Test suite
benchmarks/                # Performance benchmarks
```
