import argparse
import base64
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from stable_baselines3 import PPO

# Allow running from repo root as: python visualize_policy_randomized_sag_v3.py
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from optical_rl.utils.config import load_config
from optical_rl.utils.ray_optics_bridge_persistent import PersistentRayOpticsRunner
from optical_rl.utils.scene_builder import scene_randomized_ray_with_sag_slab_at_y


def _str2bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"1", "true", "t", "yes", "y"}:
        return True
    if lowered in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {value}")


def _norm(value: float, minimum: float, maximum: float) -> float:
    return (value - minimum) / (maximum - minimum)


def parse_args() -> argparse.Namespace:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(
        description="Visualize trained policy rollouts for randomized sag slab v3 environment"
    )
    parser.add_argument("--model", default="results/models/best_model.zip")
    parser.add_argument("--config", default="optical_rl/configs/stage1_randomized_sag_v3.yaml")
    parser.add_argument(
        "--output-dir",
        default=f"results/visualizations/randomized_sag_rollout_v3_{timestamp}",
    )
    parser.add_argument("--render-interval", type=int, default=30)
    parser.add_argument("--rollout-steps", type=int, default=600)
    parser.add_argument("--n-episodes", type=int, default=1)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--deterministic", type=_str2bool, default=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--source-y", type=float, default=None)
    parser.add_argument("--detector-y", type=float, default=None)
    parser.add_argument("--slope", type=float, default=None)
    parser.add_argument("--sag-depth-frac", type=float, default=None)
    parser.add_argument("--initial-quad-y", type=float, default=None)
    parser.add_argument("--initial-sag-depth-frac", type=float, default=None)
    parser.add_argument("--cropbox-width", type=int, default=1000)
    return parser.parse_args()


def _obs_from_state(
    quad_y: float,
    current_power: float,
    step_count: int,
    source_y: float,
    detector_y: float,
    slope: float,
    sag_depth_frac: float,
    was_clipped_y: float,
    was_clipped_sag: float,
    env_cfg: Dict[str, Any],
) -> np.ndarray:
    quad_y_norm = _norm(quad_y, env_cfg["quad_y_min"], env_cfg["quad_y_max"])
    source_y_norm = _norm(source_y, env_cfg["source_y_min"], env_cfg["source_y_max"])
    detector_y_norm = _norm(
        detector_y, env_cfg["detector_y_min"], env_cfg["detector_y_max"]
    )
    slope_norm = _norm(slope, env_cfg["slope_min"], env_cfg["slope_max"])
    sag_depth_norm = _norm(
        sag_depth_frac,
        env_cfg["sag_depth_frac_min"],
        env_cfg["sag_depth_frac_max"],
    )
    dist_y_lower_norm = _norm(quad_y, env_cfg["quad_y_min"], env_cfg["quad_y_max"])
    dist_y_upper_norm = _norm(
        env_cfg["quad_y_max"] - quad_y, 0.0, env_cfg["quad_y_max"] - env_cfg["quad_y_min"]
    )
    dist_sag_lower_norm = _norm(
        sag_depth_frac, env_cfg["sag_depth_frac_min"], env_cfg["sag_depth_frac_max"]
    )
    dist_sag_upper_norm = _norm(
        env_cfg["sag_depth_frac_max"] - sag_depth_frac,
        0.0,
        env_cfg["sag_depth_frac_max"] - env_cfg["sag_depth_frac_min"],
    )

    # Keep observation in training range [0, 1] even when rollout_steps > env max_steps.
    training_max_steps = int(env_cfg["max_steps"])
    progress = min(step_count / training_max_steps, 1.0)

    return np.array(
        [
            quad_y_norm,
            current_power,
            progress,
            source_y_norm,
            detector_y_norm,
            slope_norm,
            sag_depth_norm,
            dist_y_lower_norm,
            dist_y_upper_norm,
            dist_sag_lower_norm,
            dist_sag_upper_norm,
            was_clipped_y,
            was_clipped_sag,
        ],
        dtype=np.float32,
    )


def _build_scene_for_state(
    quad_y: float,
    env_cfg: Dict[str, Any],
    source_y: float,
    detector_y: float,
    slope: float,
    sag_depth_frac: float,
    cropbox_width: int,
) -> Dict[str, Any]:
    scene = scene_randomized_ray_with_sag_slab_at_y(
        source_x=env_cfg["source_x"],
        source_y=source_y,
        slope=slope,
        detector_x=env_cfg["detector_x"],
        detector_center_y=detector_y,
        detector_half_height=env_cfg["detector_height"] / 2.0,
        quad_center_x=env_cfg["quad_center_x"],
        quad_center_y=quad_y,
        quad_size=env_cfg["quad_size"],
        sag_depth_frac=sag_depth_frac,
        ref_index=env_cfg["ref_index"],
        ray_dx=env_cfg.get("ray_dx", 30),
    )
    scene["objs"].append(
        {
            "type": "CropBox",
            "p1": {"x": 0, "y": 0},
            "p4": {"x": env_cfg["world_width"] + 50, "y": env_cfg["world_height"] + 50},
            "width": cropbox_width + 100,
        }
    )
    return scene


def _sample_geometry(
    rng: np.random.Generator,
    env_cfg: Dict[str, Any],
    fixed_source_y: Optional[float],
    fixed_detector_y: Optional[float],
    fixed_slope: Optional[float],
    fixed_sag_depth_frac: Optional[float],
) -> Tuple[float, float, float, float]:
    source_y = (
        float(fixed_source_y)
        if fixed_source_y is not None
        else float(rng.uniform(env_cfg["source_y_min"], env_cfg["source_y_max"]))
    )
    detector_y = (
        float(fixed_detector_y)
        if fixed_detector_y is not None
        else float(rng.uniform(env_cfg["detector_y_min"], env_cfg["detector_y_max"]))
    )
    slope = (
        float(fixed_slope)
        if fixed_slope is not None
        else float(rng.uniform(env_cfg["slope_min"], env_cfg["slope_max"]))
    )
    sag_depth_frac = (
        float(fixed_sag_depth_frac)
        if fixed_sag_depth_frac is not None
        else float(
            rng.uniform(
                env_cfg["sag_depth_frac_min"],
                env_cfg["sag_depth_frac_max"],
            )
        )
    )
    return source_y, detector_y, slope, sag_depth_frac


def _save_frame_from_result(result: Dict[str, Any], frame_path: Path) -> None:
    images = result.get("images") or []
    if not images:
        raise RuntimeError("No images returned. Ensure node-canvas and CropBox are present.")
    data_url = images[0].get("dataUrl", "")
    if "," not in data_url:
        raise RuntimeError("Malformed dataUrl in runner output")
    png_data = data_url.split(",", 1)[1]
    frame_path.write_bytes(base64.b64decode(png_data))


def _write_trajectory_json(
    output_path: Path,
    run_config: Dict[str, Any],
    metadata: Dict[str, Any],
    trajectory: List[Dict[str, Any]],
) -> None:
    payload = {"config": run_config, "metadata": metadata, "trajectory": trajectory}
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _generate_video(frames_dir: Path, video_path: Path, fps: int) -> Tuple[bool, str]:
    base_cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frames_dir / "frame_%04d.png"),
    ]
    codec_options = [
        ["-c:v", "libx264", "-pix_fmt", "yuv420p"],
        ["-c:v", "mpeg4", "-q:v", "3"],
    ]
    try:
        for codec_args in codec_options:
            cmd = [*base_cmd, *codec_args, str(video_path)]
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                return True, ""
            except subprocess.CalledProcessError as exc:
                stderr = exc.stderr.strip()
                if "Unknown encoder" in stderr and "libx264" in codec_args:
                    continue
                return False, f"ffmpeg failed: {stderr}"
        return False, "ffmpeg failed: no supported encoder found"
    except FileNotFoundError:
        return False, "ffmpeg not found on PATH"


def run_episode(
    model: PPO,
    runner: PersistentRayOpticsRunner,
    env_cfg: Dict[str, Any],
    rollout_steps: int,
    render_interval: int,
    fps: int,
    deterministic: bool,
    episode_idx: int,
    episode_dir: Path,
    rng: np.random.Generator,
    source_y: float,
    detector_y: float,
    slope: float,
    sag_depth_frac: float,
    initial_quad_y: Optional[float],
    initial_sag_depth_frac: Optional[float],
    cropbox_width: int,
) -> Dict[str, Any]:
    frames_dir = episode_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    quad_y = (
        float(initial_quad_y)
        if initial_quad_y is not None
        else float(rng.uniform(env_cfg["quad_y_min"], env_cfg["quad_y_max"]))
    )
    quad_y = float(np.clip(quad_y, env_cfg["quad_y_min"], env_cfg["quad_y_max"]))
    initial_quad_y_value = quad_y
    if initial_sag_depth_frac is not None:
        sag_depth_frac = float(initial_sag_depth_frac)
    sag_depth_frac = float(
        np.clip(
            sag_depth_frac,
            env_cfg["sag_depth_frac_min"],
            env_cfg["sag_depth_frac_max"],
        )
    )
    initial_sag_depth_frac_value = sag_depth_frac

    trajectory: List[Dict[str, Any]] = []
    total_reward = 0.0
    frame_count = 0
    step_count = 0
    last_result: Optional[Dict[str, Any]] = None

    initial_scene = _build_scene_for_state(
        quad_y=quad_y,
        env_cfg=env_cfg,
        source_y=source_y,
        detector_y=detector_y,
        slope=slope,
        sag_depth_frac=sag_depth_frac,
        cropbox_width=cropbox_width,
    )
    initial_result, _ = runner.simulate(initial_scene, episode=episode_idx, step=0)
    last_result = initial_result
    current_power = float(initial_result["detectors"][0]["power"])
    _save_frame_from_result(initial_result, frames_dir / f"frame_{frame_count:04d}.png")
    frame_count += 1

    obs = _obs_from_state(
        quad_y=quad_y,
        current_power=current_power,
        step_count=step_count,
        source_y=source_y,
        detector_y=detector_y,
        slope=slope,
        sag_depth_frac=sag_depth_frac,
        was_clipped_y=0.0,
        was_clipped_sag=0.0,
        env_cfg=env_cfg,
    )

    while step_count < rollout_steps:
        action, _ = model.predict(obs, deterministic=deterministic)
        proposed_y = quad_y + float(action[0]) * env_cfg["action_scale_y"]
        new_y = float(np.clip(proposed_y, env_cfg["quad_y_min"], env_cfg["quad_y_max"]))
        was_clipped_y = 1.0 if abs(new_y - proposed_y) > 1e-12 else 0.0
        quad_y = new_y

        proposed_sag = sag_depth_frac + float(action[1]) * env_cfg["action_scale_sag"]
        new_sag = float(
            np.clip(
                proposed_sag,
                env_cfg["sag_depth_frac_min"],
                env_cfg["sag_depth_frac_max"],
            )
        )
        was_clipped_sag = 1.0 if abs(new_sag - proposed_sag) > 1e-12 else 0.0
        sag_depth_frac = new_sag

        scene = _build_scene_for_state(
            quad_y=quad_y,
            env_cfg=env_cfg,
            source_y=source_y,
            detector_y=detector_y,
            slope=slope,
            sag_depth_frac=sag_depth_frac,
            cropbox_width=cropbox_width,
        )
        result, _ = runner.simulate(scene, episode=episode_idx, step=step_count + 1)
        last_result = result
        current_power = float(result["detectors"][0]["power"])
        reward = current_power
        total_reward += reward
        step_count += 1

        if step_count % render_interval == 0:
            _save_frame_from_result(result, frames_dir / f"frame_{frame_count:04d}.png")
            frame_count += 1

        trajectory.append(
            {
                "step": step_count,
                "quad_y": quad_y,
                "action": np.asarray(action, dtype=float).tolist(),
                "reward": reward,
                "power": current_power,
                "was_clipped_y": was_clipped_y,
                "was_clipped_sag": was_clipped_sag,
                "source_y": source_y,
                "detector_y": detector_y,
                "slope": slope,
                "sag_depth_frac": sag_depth_frac,
            }
        )

        obs = _obs_from_state(
            quad_y=quad_y,
            current_power=current_power,
            step_count=step_count,
            source_y=source_y,
            detector_y=detector_y,
            slope=slope,
            sag_depth_frac=sag_depth_frac,
            was_clipped_y=was_clipped_y,
            was_clipped_sag=was_clipped_sag,
            env_cfg=env_cfg,
        )

    if step_count % render_interval != 0 and last_result is not None:
        _save_frame_from_result(last_result, frames_dir / f"frame_{frame_count:04d}.png")
        frame_count += 1

    run_config = {
        "episode": episode_idx,
        "rollout_steps": rollout_steps,
        "training_max_steps": int(env_cfg["max_steps"]),
        "render_interval": render_interval,
        "frame_count": frame_count,
        "source_y": source_y,
        "detector_y": detector_y,
        "slope": slope,
        "sag_depth_frac": sag_depth_frac,
        "initial_quad_y": initial_quad_y_value,
        "initial_sag_depth_frac": initial_sag_depth_frac_value,
    }
    metadata = {
        "total_reward": total_reward,
        "episode_length": step_count,
        "timestamp": datetime.now().isoformat(),
    }
    _write_trajectory_json(episode_dir / "trajectory.json", run_config, metadata, trajectory)
    video_ok, video_message = _generate_video(frames_dir, episode_dir / "video.mp4", fps=fps)
    return {
        "episode": episode_idx,
        "total_reward": total_reward,
        "episode_length": step_count,
        "frames": frame_count,
        "video_ok": video_ok,
        "video_message": video_message,
        "source_y": source_y,
        "detector_y": detector_y,
        "slope": slope,
        "sag_depth_frac": sag_depth_frac,
    }


def main() -> None:
    args = parse_args()
    if args.render_interval <= 0:
        raise ValueError("--render-interval must be > 0")
    if args.rollout_steps <= 0:
        raise ValueError("--rollout-steps must be > 0")
    if args.n_episodes <= 0:
        raise ValueError("--n-episodes must be > 0")
    if args.fps <= 0:
        raise ValueError("--fps must be > 0")

    config = load_config(args.config)
    env_cfg = dict(config["env"])

    if args.sag_depth_frac is not None:
        if args.sag_depth_frac < env_cfg["sag_depth_frac_min"] or args.sag_depth_frac > env_cfg["sag_depth_frac_max"]:
            raise ValueError(
                f"--sag-depth-frac must be within [{env_cfg['sag_depth_frac_min']}, {env_cfg['sag_depth_frac_max']}]"
            )

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    model = PPO.load(args.model)

    seed = args.seed if args.seed is not None else int(datetime.now().timestamp())
    rng = np.random.default_rng(seed)

    summary: List[Dict[str, Any]] = []
    with PersistentRayOpticsRunner(verbose=False, log_every_n=500) as runner:
        for episode_idx in range(args.n_episodes):
            source_y, detector_y, slope, sag_depth_frac = _sample_geometry(
                rng=rng,
                env_cfg=env_cfg,
                fixed_source_y=args.source_y,
                fixed_detector_y=args.detector_y,
                fixed_slope=args.slope,
                fixed_sag_depth_frac=args.sag_depth_frac,
            )

            episode_dir = output_root / f"episode_{episode_idx:03d}"
            episode_dir.mkdir(parents=True, exist_ok=True)
            episode_summary = run_episode(
                model=model,
                runner=runner,
                env_cfg=env_cfg,
                rollout_steps=args.rollout_steps,
                render_interval=args.render_interval,
                fps=args.fps,
                deterministic=args.deterministic,
                episode_idx=episode_idx,
                episode_dir=episode_dir,
                rng=rng,
                source_y=source_y,
                detector_y=detector_y,
                slope=slope,
                sag_depth_frac=sag_depth_frac,
                initial_quad_y=args.initial_quad_y,
                initial_sag_depth_frac=args.initial_sag_depth_frac,
                cropbox_width=args.cropbox_width,
            )
            summary.append(episode_summary)
            print(
                f"episode={episode_summary['episode']} "
                f"reward={episode_summary['total_reward']:.3f} "
                f"steps={episode_summary['episode_length']} "
                f"frames={episode_summary['frames']} "
                f"source_y={episode_summary['source_y']:.2f} "
                f"detector_y={episode_summary['detector_y']:.2f} "
                f"slope={episode_summary['slope']:.4f} "
                f"sag_depth_frac={episode_summary['sag_depth_frac']:.3f} "
                f"video={'ok' if episode_summary['video_ok'] else 'failed'}"
            )
            if not episode_summary["video_ok"]:
                print(f"  note: {episode_summary['video_message']}")

    (output_root / "summary.json").write_text(
        json.dumps(
            {
                "timestamp": datetime.now().isoformat(),
                "model_path": args.model,
                "config_path": args.config,
                "output_dir": str(output_root),
                "seed": seed,
                "rollout_steps": args.rollout_steps,
                "render_interval": args.render_interval,
                "episodes": summary,
                "fixed_overrides": {
                    "source_y": args.source_y,
                    "detector_y": args.detector_y,
                    "slope": args.slope,
                    "sag_depth_frac": args.sag_depth_frac,
                    "initial_quad_y": args.initial_quad_y,
                    "initial_sag_depth_frac": args.initial_sag_depth_frac,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved outputs under {output_root}")


if __name__ == "__main__":
    main()
