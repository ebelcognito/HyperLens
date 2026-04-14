from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ProblemCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["lens"] = "lens"
    lens_path: str
    wavelength_um: float = 0.587
    fitness_wavelengths_um: List[float] = Field(
        default_factory=lambda: [0.656, 0.587, 0.486]
    )
    object_depth_mm: float = -20000.0
    stop_surface_index: int
    free_curvature_indices: List[int]
    derived_curvature_index: Optional[int] = None
    focal_length_target_mm: Optional[float] = None
    theta_bounds: List[Tuple[float, float]]
    stage: Literal["stage1", "stage2"] = "stage1"
    num_rays: int = 512
    invalid_tol: float = 0.15
    invalid_penalty_weight: float = 5.0
    intersect_penalty_weight: float = 1.0
    air_min_mm: float = 0.1
    thick_min_mm: float = 1.0
    strict_entrance_pupil: bool = True
    descriptor_field_index: int = 0
    parameterization: Literal["curvature", "power_budget", "full_dof"] = "curvature"
    efl_penalty_weight: float = 5.0
    free_gap_indices: List[int] = Field(default_factory=list)
    variable_flange: bool = False
    flange_bounds: Optional[Tuple[float, float]] = None

    @model_validator(mode="after")
    def _check_bounds_match_free(self):
        expected = len(self.free_curvature_indices) + len(self.free_gap_indices)
        if self.variable_flange:
            expected += 1
        if len(self.theta_bounds) != expected:
            raise ValueError(
                f"theta_bounds length ({len(self.theta_bounds)}) must equal "
                f"free_curvature_indices ({len(self.free_curvature_indices)}) + "
                f"free_gap_indices ({len(self.free_gap_indices)}) + "
                f"variable_flange ({1 if self.variable_flange else 0}) = {expected}"
            )
        return self


class DescriptorCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["zernike", "seidel"]
    params: Dict[str, Any]


class AlgorithmCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")

    population_size: int = 16
    initial_sigma: float = 0.75
    cma_eta: float = 0.05
    cma_seed: int = 0
    seed_count: int = 128
    calibration_count: int = 512
    iterations: int = 1000
    new_cell_bonus: float = 1.0e6
    log_every: int = 50
    restart_patience: int = 1


class ArchiveCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bins: List[int]
    bounds: Optional[List[Tuple[float, float]]] = None
    bound_calibration_pad: float = 0.05
    bound_calibration_pct: Tuple[float, float] = (2.0, 98.0)


class IOCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")

    results_dir: str = "results/lensmega"
    run_name: str
    checkpoint_every: int = 50
    parallel_workers: Optional[int] = None
    device: str = "auto"
    log_seed_accepts: bool = True
    elite_fitness_threshold: Optional[float] = None
    # L-BFGS refiner settings (used when elite_fitness_threshold is set)
    refine_efl_penalty_weight: Optional[float] = None
    refine_invalid_penalty_weight: Optional[float] = None
    refine_intersect_penalty_weight: Optional[float] = None
    refine_lbfgs_max_steps: int = 100
    refine_lbfgs_lr: float = 0.5
    refine_converge_tol: float = 1e-4
    # Basin stability analysis
    refine_stability_samples: int = 20
    refine_stability_perturbation: float = 0.01


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    problem: ProblemCfg
    descriptor: DescriptorCfg
    algorithm: AlgorithmCfg
    archive: ArchiveCfg
    io: IOCfg

    @model_validator(mode="after")
    def _crosscheck(self):
        if self.descriptor.kind == "zernike":
            indices = self.descriptor.params.get("noll_indices")
            if indices is None:
                raise ValueError(
                    "descriptor.params.noll_indices is required for zernike"
                )
            k = len(indices)
        elif self.descriptor.kind == "seidel":
            terms = self.descriptor.params.get("terms")
            if terms is None:
                raise ValueError("descriptor.params.terms is required for seidel")
            k = len(terms)
        else:
            raise ValueError(f"unknown descriptor kind {self.descriptor.kind!r}")

        if len(self.archive.bins) != k:
            raise ValueError(
                f"archive.bins has length {len(self.archive.bins)} but descriptor "
                f"has dim {k}"
            )
        if self.archive.bounds is not None and len(self.archive.bounds) != k:
            raise ValueError(
                f"archive.bounds has length {len(self.archive.bounds)} but descriptor "
                f"has dim {k}"
            )
        return self
