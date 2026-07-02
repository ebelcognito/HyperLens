#!/usr/bin/env python
"""Render kodim23 through the top-4 sixlet designs (by RMS, from
sixlet_top9_eps030_spotlights.json) at 512x512 on GPU via lens.render() with
the PSF-map method.

Output:
  figures/sixlet_top4_navy_kodim/<design_NN>_rms_<x.xx>.png       — individual
  figures/sixlet_top4_navy_kodim/kodim23_top4_composite.png       — original + 4
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
for _k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

ROOT = Path("/nas/longleaf/home/hisham26/Desktop/hyperlens")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "DeepLens"))

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import torch

FRAME_COLOR = "#7BAFD4"

def psnr_db(rendered: np.ndarray, reference: np.ndarray) -> float:
    """PSNR between two float arrays in [0, 1] with the same shape."""
    mse = float(np.mean((rendered.astype(np.float64) - reference.astype(np.float64)) ** 2))
    if mse < 1e-12:
        return 99.99
    return 10.0 * np.log10(1.0 / mse)


_FONT_CANDIDATES = [
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]
def _load_font(size: int):
    for p in _FONT_CANDIDATES:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def add_psnr_badge(img: Image.Image, psnr_value: float, size_px: int = 34) -> Image.Image:
    """Bottom-right semi-transparent black box with white bold PSNR text."""
    img = img.convert("RGB").copy()
    draw = ImageDraw.Draw(img, "RGBA")
    text = f"PSNR  {psnr_value:.2f} dB"
    font = _load_font(size_px)
    W, H = img.size
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad_x, pad_y, margin = 18, 12, 16
    x1, y1 = W - margin, H - margin
    x0, y0 = x1 - tw - 2 * pad_x, y1 - th - 2 * pad_y
    draw.rectangle([x0, y0, x1, y1], fill=(0, 0, 0, 170))
    draw.text((x0 + pad_x - bbox[0], y0 + pad_y - bbox[1]),
              text, fill=(255, 255, 255, 255), font=font)
    return img


from scripts_throwaway.fig5_extract import (
    SYSTEMS, build_lens_template_and_solver, make_geolens,
)

SPOTS_JSON = ROOT / "figures/sixlet_top9_eps030_spotlights.json"
CHART_PATH = ROOT / "results/figure_archive_demo/test_images/kodim23.png"
OUT_DIR    = ROOT / "figures/sixlet_top4_blue_kodim"
N_PICK     = 4
SENSOR_RES = (512, 512)
DEPTH      = -20000.0
PSF_GRID   = (12, 12)
PSF_KS     = 51

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_chart(path: Path, target_hw):
    img = Image.open(path).convert("RGB").resize((target_hw[1], target_hw[0]), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    spots = json.loads(SPOTS_JSON.read_text())
    def _rms(s): return float(s.get("rms_um", s.get("max_per_wvln_rms_um", 9e9)))
    top = sorted(spots, key=_rms)[:N_PICK]
    print(f"device={DEVICE}  cuda available={torch.cuda.is_available()}", flush=True)
    if DEVICE == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"\nTop {N_PICK} sixlets by RMS:")
    for i, s in enumerate(top, 1):
        print(f"  #{i}  RMS={_rms(s):.3f}µm", flush=True)

    t0 = time.time()
    print(f"\nbuilding sixlet template+solver on {DEVICE} ...", flush=True)
    template, solver, prob_cfg, lens_path, p_problem = build_lens_template_and_solver(
        SYSTEMS["sixlet"]["cfg"])
    print(f"  ready in {time.time()-t0:.1f}s", flush=True)

    chart = load_chart(CHART_PATH, SENSOR_RES).to(DEVICE)
    print(f"\nchart loaded {chart.shape} from {CHART_PATH.name}", flush=True)

    results = []
    for i, sp in enumerate(top, 1):
        ts = time.time()
        for k in ("theta","refined_theta","theta_refined"):
            if k in sp: theta = np.asarray(sp[k], dtype=np.float64); break
        else:
            raise KeyError(f"no theta in entry #{i}")
        lens = make_geolens(prob_cfg, p_problem, lens_path, theta, template, solver)
        # Try to move lens to GPU
        try:
            lens = lens.to(DEVICE)
        except Exception:
            pass
        # Match sensor resolution to chart
        try:
            lens.sensor_res = SENSOR_RES
            lens.set_sensor(sensor_res=SENSOR_RES, sensor_size=lens.sensor_size)
        except Exception as e:
            print(f"  #{i} set_sensor warning: {e}", flush=True)

        print(f"\n  #{i} (RMS {_rms(sp):.2f}µm) rendering...", flush=True)
        try:
            with torch.no_grad():
                rendered = lens.render(
                    chart, depth=DEPTH, method="psf_map",
                    psf_grid=PSF_GRID, psf_ks=PSF_KS,
                )
            arr = rendered.squeeze(0).permute(1, 2, 0).cpu().numpy()
            arr = np.clip(arr, 0.0, 1.0)
        except Exception as e:
            print(f"    PSF-map render failed: {e}", flush=True)
            arr = None
        results.append((i, sp, arr))

        if arr is not None:
            ref = chart.squeeze(0).permute(1, 2, 0).cpu().numpy()
            psnr_v = psnr_db(arr, ref)
            stem = f"design_{i:02d}_rms_{_rms(sp):.2f}"
            np.save(OUT_DIR / f"{stem}.raw.npy", arr.astype(np.float32))
            (OUT_DIR / f"{stem}.psnr.txt").write_text(f"{psnr_v:.4f}\n")
            badged = add_psnr_badge(Image.fromarray((arr * 255).astype(np.uint8)), psnr_v)
            badged.save(OUT_DIR / f"{stem}.png")
            results[-1] = (i, sp, arr, psnr_v)
            print(f"    -> {stem}.png  PSNR={psnr_v:.2f} dB  ({time.time()-ts:.1f}s)", flush=True)
        else:
            results[-1] = (i, sp, None, float("nan"))

    # Side-by-side composite: original + 4
    fig, axes = plt.subplots(1, 5, figsize=(22, 5.0),
                             gridspec_kw=dict(wspace=0.04))
    chart_np = chart.squeeze(0).permute(1, 2, 0).cpu().numpy()
    axes[0].imshow(chart_np)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    axes[0].set_title("kodim23 (original)", fontsize=11, fontweight="bold")
    for sp in axes[0].spines.values():
        sp.set_edgecolor("black"); sp.set_linewidth(2)

    for col, (i, s, arr, psnr_v) in enumerate(results, 1):
        ax = axes[col]
        if arr is not None:
            ax.imshow(arr)
            ax.text(0.97, 0.04, f"PSNR  {psnr_v:.2f} dB",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=16, fontweight="bold", color="white",
                    bbox=dict(facecolor=(0, 0, 0, 0.6), edgecolor="none",
                              boxstyle="round,pad=0.50"))
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(FRAME_COLOR); sp.set_linewidth(3)
        ax.set_title(f"design #{i}  ·  RMS {_rms(s):.2f} µm",
                     fontsize=11, fontweight="bold", color=FRAME_COLOR)

    fig.suptitle(
        "kodim23 rendered through the top-4 sixlet designs "
        f"(PSF-map, {SENSOR_RES[0]}×{SENSOR_RES[1]}, depth=∞)",
        fontsize=13, fontweight="bold")
    composite = OUT_DIR / "kodim23_top4_composite.png"
    fig.savefig(composite, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\ncomposite -> {composite.name}")
    print(f"total {time.time()-t0:.1f}s   ->  {OUT_DIR}")


if __name__ == "__main__":
    main()
