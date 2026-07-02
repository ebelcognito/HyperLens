"""Render lens / seidel / stability-ribbon panels for the 8 curated designs in
designs_json/. Output: designs_json/panels/<TAG>_{lens,seidel,robust}.png — 24
PNGs. All in BLUE (per user request — not the red shared-basin variant).

Run on a compute node. Driver only — primitives come from
scripts_throwaway/render_spotlight_panels_split.
"""
from __future__ import annotations
import json, os, sys, tempfile, yaml
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import numpy as np
import torch
torch.set_num_threads(1)

ROOT = Path("/nas/longleaf/home/hisham26/Desktop/hyperlens")
sys.path.insert(0, str(ROOT))

from scripts_throwaway.fig5_extract import (
    build_lens_template_and_solver, make_geolens,
)
from scripts_throwaway.render_spotlight_panels_split import (
    render_lens, render_seidel_s2_s3, render_robust,
    per_field_per_wvln_rms_um, BLUE_FILL,
)

DESIGNS_DIR = ROOT / "designs_json"
OUT_DIR     = DESIGNS_DIR / "panels"
OUT_DIR.mkdir(exist_ok=True)

EXAMPLES = ROOT / "optical_rl/qd/lensmega/config/examples"
LOCAL_LENSES = {
    "septlet":    ROOT / "DeepLens/datasets/lenses/septlet_50mm_f3.json",
    "quadruplet": ROOT / "DeepLens/datasets/lenses/quadruplet_50mm_f4.json",
}

TAG_TO_CFG = {
    "V1": (EXAMPLES / "septlet_F3_discovery.yaml",            "septlet"),
    "V2": (EXAMPLES / "septlet_F3_discovery.yaml",            "septlet"),
    "V3": (EXAMPLES / "septlet_F3_discovery.yaml",            "septlet"),
    "V4": (EXAMPLES / "septlet_F3_discovery.yaml",            "septlet"),
    "Q1": (EXAMPLES / "quadruplet_s2s4.yaml",                 "quadruplet"),
    "Q4": (EXAMPLES / "quadruplet_discovery_materials.yaml",  "quadruplet"),
    "Q5": (EXAMPLES / "quadruplet_discovery_materials.yaml",  "quadruplet"),
    "Q6": (EXAMPLES / "quadruplet_discovery.yaml",            "quadruplet"),
}


def patched_cfg(orig_cfg: Path, local_lens: Path) -> Path:
    """Write a temp YAML that mirrors orig_cfg but with lens_path repointed."""
    d = yaml.safe_load(orig_cfg.read_text())
    d["problem"]["lens_path"] = str(local_lens)
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.safe_dump(d, f)
    f.close()
    return Path(f.name)


def render_one(tag: str, design_path: Path):
    cfg_orig, topo = TAG_TO_CFG[tag]
    local_lens = LOCAL_LENSES[topo]
    cfg_path = patched_cfg(cfg_orig, local_lens)

    J = json.load(design_path.open())
    ref = J["refined"]
    theta = np.asarray(ref["theta"], dtype=np.float64)
    # Canonical (coma, astigmatism) from refined.metrics — falling through the
    # two key conventions used in this codebase. Top-level `measures` is the
    # discovery-descriptor (which is NOT always (S2,S3) — Q1 used (S2,S4)).
    m = ref["metrics"]
    S2 = m.get("seidel_S2", m.get("m0_S2_coma"))
    S3 = m.get("seidel_S3", m.get("m1_S3_astigmatism"))
    if S2 is None or S3 is None:
        raise RuntimeError(f"{design_path.name}: cannot find canonical S2/S3 in refined.metrics; keys={list(m.keys())}")
    rms = float(ref.get("rms_um", J.get("rms_um", float("nan"))))

    template, solver, prob_cfg, lens_path, p_problem = (
        build_lens_template_and_solver(cfg_path)
    )
    lens = make_geolens(prob_cfg, p_problem, lens_path, theta, template, solver)

    prefix = OUT_DIR / tag
    render_lens(lens, BLUE_FILL, str(prefix) + "_lens.png")
    render_seidel_s2_s3(S2, S3, BLUE_FILL, str(prefix) + "_seidel.png")

    field_fracs, wvlns, rms_arr = per_field_per_wvln_rms_um(
        lens, p_problem, n_fields=7, num_rays=256,
    )
    render_robust(field_fracs, wvlns, rms_arr, BLUE_FILL, str(prefix) + "_robust.png")
    print(f"[{tag}] rms={rms:.2f}um  S2={S2:+.3e}  S3={S3:+.3e}  -> {prefix}_{{lens,seidel,robust}}.png")


def main():
    paths = sorted(DESIGNS_DIR.glob("*.json"))
    if not paths:
        sys.exit(f"no JSONs found under {DESIGNS_DIR}")
    for p in paths:
        if p.name == "MANIFEST.md":
            continue
        tag = p.name.split("_", 1)[0]
        if tag not in TAG_TO_CFG:
            print(f"[skip] {p.name} — unknown tag '{tag}'")
            continue
        render_one(tag, p)


if __name__ == "__main__":
    main()
