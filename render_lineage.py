"""Render lens layouts and spot diagrams along a lineage chain.

For each ancestor in a lineage, reconstructs the lens from its theta
vector, draws the 2D lens layout with fan rays, and produces a spot
diagram at a chosen field angle.  The per-exemplar output is a strip
composite with one column per step.

This is a standalone *illustration* tool — it does not modify the run
directory, and it is intentionally decoupled from ``lineage_analysis.py``
which owns the trajectory-math and clustering.

Typical usage
-------------

    # Render all exemplars found in <run_dir>/lineage_analysis/
    python render_lineage.py <run_dir>

    # Pick one exemplar and override field angle / wavelength
    python render_lineage.py <run_dir> --exemplar 2 --field-deg 7 \
                                       --wavelength-um 0.587

    # Stride > 1 (only useful if your chain is long; most are 3-5 deep)
    python render_lineage.py <run_dir> --stride 2

Outputs go to ``<run_dir>/lineage_analysis/render_<id>/``.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

# Lazy imports of torch / deeplens are deferred inside functions so the
# module can be inspected or unit-tested without the full optics stack.


# ======================================================================
# Config + lens reconstruction
# ======================================================================

def _load_run_config(run_dir: Path):
    """Load the run's config (prefers config.yaml, falls back to resolved)."""
    from optical_rl.qd.lensmega.config.loader import load_config
    yaml_path = run_dir / "config.yaml"
    if yaml_path.exists():
        return load_config(yaml_path)
    json_path = run_dir / "config_resolved.json"
    if json_path.exists():
        # Round-trip via RunConfig.model_validate.
        from optical_rl.qd.lensmega.config.schema import RunConfig
        with json_path.open() as f:
            return RunConfig.model_validate(json.load(f))
    raise FileNotFoundError(f"no config found in {run_dir}")


def _build_fresh_lens(cfg):
    """Fresh (template, lens, solver) triple bound to CPU."""
    from optical_rl.qd.lensmega.problem.lens_template import load_lens_template
    from optical_rl.qd.lensmega.problem.paraxial import GeneralizedParaxialSolver
    p = cfg.problem
    template, lens = load_lens_template(
        p.lens_path,
        wavelength_um=p.wavelength_um,
        stop_surface_index=p.stop_surface_index,
        free_curvature_indices=tuple(p.free_curvature_indices),
        derived_curvature_index=p.derived_curvature_index,
        device="cpu",
    )
    solver = GeneralizedParaxialSolver(template)
    return template, lens, solver


def _apply_theta(lens, theta: np.ndarray, cfg, template, solver) -> None:
    from optical_rl.qd.render_lensmega_archive import _apply_theta_generic
    p = cfg.problem
    _apply_theta_generic(
        lens, theta,
        template=template, solver=solver,
        focal_length_target_mm=p.focal_length_target_mm,
        device="cpu",
        parameterization=p.parameterization,
        free_curvature_indices=p.free_curvature_indices,
        free_gap_indices=p.free_gap_indices,
        variable_flange=p.variable_flange,
        variable_materials=p.variable_materials,
        free_material_indices=p.free_material_indices,
    )


def _estimate_hfov_deg(lens, template) -> float:
    try:
        if hasattr(lens, "rfov") and float(lens.rfov) > 0:
            return float(np.rad2deg(float(lens.rfov)))
    except Exception:
        pass
    try:
        efl = 50.0
        half_sensor = template.sensor_size_mm[0] / 2.0
        return math.degrees(math.atan(half_sensor / max(abs(efl), 1.0)))
    except Exception:
        return 14.0


# ======================================================================
# Rendering primitives
# ======================================================================

def _draw_lens_panel(
    ax, lens, *, hfov_deg: float, wavelength_um: float, title: str
) -> None:
    """Draw a 2D lens layout with ray fans at 3 field fractions."""
    lens.draw_lens_2d(ax=ax, fig=ax.figure, color="black")
    for fov_frac, color in [(0.0, "#2196F3"), (0.5, "#4CAF50"),
                            (1.0, "#F44336")]:
        try:
            ray = lens.sample_point_source_2D(
                fov=fov_frac * hfov_deg,
                depth=-20000.0,
                num_rays=11,
                wvln=wavelength_um,
            )
            ray.prop_to(-1.0)
            _, rec = lens.trace2sensor(ray=ray, record=True)
            lens.draw_ray_2d(rec, ax=ax, fig=ax.figure, color=color)
        except Exception:
            pass
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("z (mm)", fontsize=8)
    ax.set_ylabel("y (mm)", fontsize=8)
    ax.tick_params(labelsize=7)


def _spot_intercepts(
    lens, *, fov_x_deg: float, fov_y_deg: float,
    wavelength_um: float, num_rays: int = 256,
) -> Optional[np.ndarray]:
    """Trace a point-source ray bundle and return sensor-plane (x, y)."""
    import torch
    try:
        ray = lens.sample_point_source(
            fov_x=[float(fov_x_deg)],
            fov_y=[float(fov_y_deg)],
            depth=-20000.0,
            num_rays=int(num_rays),
            wvln=float(wavelength_um),
        )
        out = lens.trace2sensor(ray=ray, record=False)
        with torch.no_grad():
            o = out.o.detach().cpu().numpy().reshape(-1, 3)
            try:
                valid_t = out.is_valid() if callable(out.is_valid) else out.is_valid
                valid = valid_t.detach().cpu().numpy().reshape(-1).astype(bool)
                if valid.shape[0] == o.shape[0]:
                    o = o[valid]
            except Exception:
                pass
        if o.shape[0] < 2:
            return None
        return o[:, :2]
    except Exception:
        return None


def _draw_spot_panel(
    ax, lens, *, fov_x_deg: float, fov_y_deg: float,
    wavelength_um: float, num_rays: int = 256,
) -> None:
    xy = _spot_intercepts(
        lens,
        fov_x_deg=fov_x_deg, fov_y_deg=fov_y_deg,
        wavelength_um=wavelength_um, num_rays=num_rays,
    )
    if xy is None or xy.shape[0] < 2:
        ax.text(0.5, 0.5, "no valid rays", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="red")
        ax.set_xticks([])
        ax.set_yticks([])
        return
    cx, cy = xy[:, 0].mean(), xy[:, 1].mean()
    dx = xy[:, 0] - cx
    dy = xy[:, 1] - cy
    rms_um = float(np.sqrt((dx ** 2 + dy ** 2).mean())) * 1000.0
    ax.scatter(dx * 1000.0, dy * 1000.0, s=2, color="#444", alpha=0.6)
    ax.set_aspect("equal")
    lim = float(max(np.abs(dx).max(), np.abs(dy).max()) * 1000.0 * 1.1 + 1e-9)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.grid(alpha=0.3)
    ax.axhline(0, color="#888", lw=0.5)
    ax.axvline(0, color="#888", lw=0.5)
    ax.set_title(f"spot RMS={rms_um:.1f}μm", fontsize=9)
    ax.set_xlabel("x (μm)", fontsize=8)
    ax.set_ylabel("y (μm)", fontsize=8)
    ax.tick_params(labelsize=7)


def _draw_seidel_bars(ax, seidel_all: Optional[Sequence[float]]) -> None:
    labels = ("S1", "S2", "S3", "S4", "S5")
    if seidel_all is None or len(seidel_all) != 5:
        ax.text(0.5, 0.5, "no Seidel", ha="center", va="center",
                transform=ax.transAxes, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        return
    vals = np.abs(np.asarray(seidel_all, dtype=np.float64))
    # Log scale to handle wide dynamic range.
    vals_safe = np.clip(vals, 1e-12, None)
    colors = ["#D62728", "#1F77B4", "#2CA02C", "#FF7F0E", "#9467BD"]
    ax.bar(labels, vals_safe, color=colors, log=True)
    ax.set_ylabel("|Sₖ|", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(which="both", axis="y", alpha=0.3)


# ======================================================================
# Per-exemplar strip
# ======================================================================

def render_exemplar_strip(
    chain: List[dict],
    cfg,
    *,
    field_deg: Optional[float],
    wavelength_um: Optional[float],
    num_rays: int,
    stride: int,
    out_path: Path,
    label: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Stride sub-sampling (always keep first and last).
    if stride > 1 and len(chain) > 2:
        idxs = list(range(0, len(chain), stride))
        if (len(chain) - 1) not in idxs:
            idxs.append(len(chain) - 1)
        chain = [chain[i] for i in idxs]

    n = len(chain)
    fig, axes = plt.subplots(
        3, n, figsize=(4.2 * n, 10), squeeze=False,
        gridspec_kw={"height_ratios": [3, 2.5, 1.3]},
    )

    template, lens, solver = _build_fresh_lens(cfg)
    wvln = wavelength_um or float(cfg.problem.wavelength_um)

    for col, entry in enumerate(chain):
        theta = np.asarray(entry["theta"] if "theta" in entry
                           else entry.get("theta_vec", []), dtype=np.float64)
        if theta.size == 0:
            # The JSON produced by lineage_analysis.py omits theta — we need
            # the full theta, so pull it from the lineage.jsonl by hash.
            # (Handled by the caller; see collect_chains_with_theta.)
            raise RuntimeError("chain entry missing 'theta'")
        try:
            _apply_theta(lens, theta, cfg, template, solver)
        except Exception as e:
            for ax in axes[:, col]:
                ax.text(0.5, 0.5, f"apply failed\n{type(e).__name__}",
                        ha="center", va="center", transform=ax.transAxes,
                        fontsize=8, color="red")
            continue

        hfov = _estimate_hfov_deg(lens, template)
        fld = field_deg if field_deg is not None else 0.5 * hfov

        fit = entry.get("fitness")
        gen = entry.get("gen")
        title = (
            f"step {col} | gen={gen}\n"
            f"fit={fit:+.4f}" if fit is not None else f"step {col} | gen={gen}"
        )
        _draw_lens_panel(
            axes[0, col], lens,
            hfov_deg=hfov, wavelength_um=wvln, title=title,
        )
        _draw_spot_panel(
            axes[1, col], lens,
            fov_x_deg=0.0, fov_y_deg=fld,
            wavelength_um=wvln, num_rays=num_rays,
        )
        _draw_seidel_bars(axes[2, col], entry.get("seidel_all"))

    fig.suptitle(f"{label}  (field={('auto' if field_deg is None else f'{field_deg:g}°')}, λ={wvln:g}μm)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ======================================================================
# Chain collection (attach theta from lineage.jsonl)
# ======================================================================

def _load_theta_by_hash(lineage_path: Path) -> dict:
    """Return {theta_hash: theta_list} from the per-emission log."""
    by_hash: dict = {}
    with lineage_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            h = d.get("theta_hash")
            th = d.get("theta")
            if h and th and h not in by_hash:
                by_hash[h] = th
    return by_hash


def _attach_theta(chain_rows: List[dict], theta_by_hash: dict) -> List[dict]:
    out = []
    for entry in chain_rows:
        h = entry.get("theta_hash")
        if h and h in theta_by_hash and "theta" not in entry:
            entry = dict(entry)
            entry["theta"] = theta_by_hash[h]
        out.append(entry)
    return out


def _discover_exemplars(run_dir: Path, which: Optional[int]) -> List[Path]:
    analysis_dir = run_dir / "lineage_analysis"
    if not analysis_dir.exists():
        sys.exit(f"no lineage_analysis/ in {run_dir}; run lineage_analysis.py first")
    if which is not None:
        matches = sorted(glob.glob(str(analysis_dir / f"exemplar_{which}_*.json")))
        if not matches:
            sys.exit(f"no exemplar_{which}_*.json in {analysis_dir}")
        return [Path(m) for m in matches]
    matches = sorted(glob.glob(str(analysis_dir / "exemplar_*.json")))
    if not matches:
        sys.exit(f"no exemplar JSONs found in {analysis_dir}")
    return [Path(m) for m in matches]


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--exemplar", type=int, default=None,
                    help="render only exemplar N (1-indexed)")
    ap.add_argument("--stride", type=int, default=1,
                    help="sub-sample chain every Nth step (default 1)")
    ap.add_argument("--field-deg", type=float, default=None,
                    help="spot-diagram field angle in degrees (default: 0.5 × HFOV)")
    ap.add_argument("--wavelength-um", type=float, default=None,
                    help="wavelength for spot/fan; defaults to problem's")
    ap.add_argument("--num-rays", type=int, default=2048,
                    help="rays per spot diagram (default 2048)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="output dir (default: <run_dir>/lineage_analysis/renders)")
    args = ap.parse_args()

    run_dir: Path = args.run_dir
    if not run_dir.exists():
        sys.exit(f"no such run dir: {run_dir}")

    cfg = _load_run_config(run_dir)
    theta_by_hash = _load_theta_by_hash(run_dir / "lineage.jsonl")

    out_dir: Path = args.out_dir or (run_dir / "lineage_analysis" / "renders")
    out_dir.mkdir(parents=True, exist_ok=True)

    exemplar_paths = _discover_exemplars(run_dir, args.exemplar)
    for ep in exemplar_paths:
        with ep.open() as f:
            chain_rows = json.load(f)
        chain_rows = _attach_theta(chain_rows, theta_by_hash)
        missing = [i for i, e in enumerate(chain_rows) if "theta" not in e]
        if missing:
            print(f"[render] {ep.name}: missing theta at steps {missing}; skipping them",
                  flush=True)
            chain_rows = [e for e in chain_rows if "theta" in e]
            if not chain_rows:
                continue

        stem = ep.stem  # e.g. exemplar_1_000acd30
        out_png = out_dir / f"{stem}_strip.png"
        print(f"[render] {stem}: {len(chain_rows)} steps → {out_png}", flush=True)
        render_exemplar_strip(
            chain_rows, cfg,
            field_deg=args.field_deg,
            wavelength_um=args.wavelength_um,
            num_rays=args.num_rays,
            stride=args.stride,
            out_path=out_png,
            label=stem,
        )


if __name__ == "__main__":
    main()
