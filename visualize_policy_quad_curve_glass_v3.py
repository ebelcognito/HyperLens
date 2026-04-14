import argparse
import base64
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from stable_baselines3 import PPO

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from optical_rl.envs.quad_curve_glass_env_v3 import (
    QuadCurveGlassConfigV3,
    QuadCurveGlassEnvV3,
)
from optical_rl.utils.config import load_config
from optical_rl.utils.ray_optics_bridge_persistent import PersistentRayOpticsRunner


def _str2bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"1", "true", "t", "yes", "y"}:
        return True
    if lowered in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {value}")


def parse_args() -> argparse.Namespace:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(
        description="Visualize trained policy rollouts for quad curve glass environment"
    )
    parser.add_argument("--model", default="/home/hisham26/Desktop/RL/results/models/quad_curve_glass_v3_20260304_130007/best_model.zip")
    parser.add_argument("--config", default="optical_rl/configs/stage1_quad_curve_glass_v3.yaml")
    parser.add_argument(
        "--output-dir",
        default=f"results/visualizations/quad_curve_glass_rollout_{timestamp}",
    )
    parser.add_argument("--render-interval", type=int, default=10)
    parser.add_argument("--rollout-steps", type=int, default=0)
    parser.add_argument("--n-episodes", type=int, default=1)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--deterministic", type=_str2bool, default=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cropbox-width", type=int, default=1000)
    return parser.parse_args()


def _resolve_model_path(model_arg: str) -> Path:
    provided = Path(model_arg)
    if provided.exists():
        return provided
    if provided.parent != Path("."):
        raise FileNotFoundError(f"Model not found: {provided}")

    models_root = Path("results/models")
    matches = sorted(models_root.rglob(provided.name))
    if not matches:
        raise FileNotFoundError(
            f"Model '{provided.name}' not found under {models_root}. "
            f"Pass a full path via --model."
        )
    if len(matches) > 1:
        listed = "\n".join(str(p) for p in matches[:10])
        raise ValueError(
            f"Multiple model files named '{provided.name}' found. "
            f"Use a full path with --model.\n{listed}"
        )
    return matches[0]


def _scene_with_cropbox(env: QuadCurveGlassEnvV3, cropbox_width: int) -> Dict[str, Any]:
    scene = env._build_scene()
    scene["objs"].append(
        {
            "type": "CropBox",
            "p1": {"x": 0, "y": 0},
            "p4": {
                "x": env.config.world_width + 50,
                "y": env.config.world_height + 50,
            },
            "width": cropbox_width + 100,
        }
    )
    return scene


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
    seed: int,
    cropbox_width: int,
) -> Dict[str, Any]:
    frames_dir = episode_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    cfg = QuadCurveGlassConfigV3(**env_cfg)
    env = QuadCurveGlassEnvV3(cfg)

    trajectory: List[Dict[str, Any]] = []
    frame_count = 0
    total_reward = 0.0
    last_render_result: Dict[str, Any] = {}

    try:
        obs, _ = env.reset(seed=seed)
        initial_power = float(env.current_power)

        initial_scene = _scene_with_cropbox(env, cropbox_width)
        render_result, _ = runner.simulate(initial_scene, episode=episode_idx, step=0)
        _save_frame_from_result(render_result, frames_dir / f"frame_{frame_count:04d}.png")
        frame_count += 1
        last_render_result = render_result

        step_count = 0
        done = False
        while step_count < rollout_steps and not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            done = bool(terminated or truncated)
            total_reward += float(reward)

            if step_count % render_interval == 0:
                scene = _scene_with_cropbox(env, cropbox_width)
                render_result, _ = runner.simulate(scene, episode=episode_idx, step=step_count)
                _save_frame_from_result(
                    render_result, frames_dir / f"frame_{frame_count:04d}.png"
                )
                frame_count += 1
                last_render_result = render_result

            trajectory.append(
                {
                    "step": step_count,
                    "action": np.asarray(action, dtype=float).reshape(-1).tolist(),
                    "reward": float(reward),
                    "power": float(env.current_power),
                    "simulation_ok": bool(info.get("simulation_ok", True)),
                    "corners": [[float(x), float(y)] for x, y in env.corners],
                    "sags": [float(s) for s in env.sags],
                    "path_points": [[float(x), float(y)] for x, y in env.points],
                }
            )

        if step_count % render_interval != 0 and last_render_result:
            _save_frame_from_result(
                last_render_result, frames_dir / f"frame_{frame_count:04d}.png"
            )
            frame_count += 1

        run_config = {
            "episode": episode_idx,
            "seed": seed,
            "rollout_steps": rollout_steps,
            "training_max_steps": int(env.config.max_steps),
            "render_interval": render_interval,
            "frame_count": frame_count,
            "initial_power": initial_power,
            "initial_corners": [[float(x), float(y)] for x, y in trajectory[0]["corners"]] if trajectory else [[float(x), float(y)] for x, y in env.corners],
            "initial_sags": trajectory[0]["sags"] if trajectory else [float(s) for s in env.sags],
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
            "initial_power": initial_power,
            "final_power": float(env.current_power),
        }
    finally:
        env.close()


def main() -> None:
    args = parse_args()
    if args.render_interval <= 0:
        raise ValueError("--render-interval must be > 0")
    if args.rollout_steps < 0:
        raise ValueError("--rollout-steps must be >= 0")
    if args.n_episodes <= 0:
        raise ValueError("--n-episodes must be > 0")
    if args.fps <= 0:
        raise ValueError("--fps must be > 0")

    config = load_config(args.config)
    env_cfg = dict(config["env"])
    QuadCurveGlassConfigV3(**env_cfg)

    rollout_steps = int(args.rollout_steps) if args.rollout_steps > 0 else int(env_cfg["max_steps"])

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    model_path = _resolve_model_path(args.model)
    model = PPO.load(str(model_path))
    tmp_env = QuadCurveGlassEnvV3(QuadCurveGlassConfigV3(**env_cfg))
    expected_action_shape = tuple(tmp_env.action_space.shape)
    tmp_env.close()
    model_action_shape = tuple(model.action_space.shape)
    if model_action_shape != expected_action_shape:
        raise ValueError(
            f"Model action space {model_action_shape} does not match environment action space "
            f"{expected_action_shape}. Retrain the model with the current constrained v3 env."
        )

    base_seed = args.seed if args.seed is not None else int(datetime.now().timestamp())
    summary: List[Dict[str, Any]] = []
    with PersistentRayOpticsRunner(
        timeout_s=float(env_cfg.get("timeout_s", 10.0)),
        verbose=False,
        log_every_n=500,
    ) as runner:
        for episode_idx in range(args.n_episodes):
            episode_dir = output_root / f"episode_{episode_idx:03d}"
            episode_dir.mkdir(parents=True, exist_ok=True)
            episode_summary = run_episode(
                model=model,
                runner=runner,
                env_cfg=env_cfg,
                rollout_steps=rollout_steps,
                render_interval=args.render_interval,
                fps=args.fps,
                deterministic=args.deterministic,
                episode_idx=episode_idx,
                episode_dir=episode_dir,
                seed=base_seed + episode_idx,
                cropbox_width=args.cropbox_width,
            )
            summary.append(episode_summary)
            print(
                f"episode={episode_summary['episode']} "
                f"reward={episode_summary['total_reward']:.3f} "
                f"steps={episode_summary['episode_length']} "
                f"frames={episode_summary['frames']} "
                f"initial_power={episode_summary['initial_power']:.3f} "
                f"final_power={episode_summary['final_power']:.3f} "
                f"video={'ok' if episode_summary['video_ok'] else 'failed'}"
            )
            if not episode_summary["video_ok"]:
                print(f"  note: {episode_summary['video_message']}")

    (output_root / "summary.json").write_text(
        json.dumps(
            {
                "timestamp": datetime.now().isoformat(),
                "model_path": str(model_path),
                "config_path": args.config,
                "output_dir": str(output_root),
                "seed": base_seed,
                "rollout_steps": rollout_steps,
                "render_interval": args.render_interval,
                "episodes": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved outputs under {output_root}")


if __name__ == "__main__":
    main()
