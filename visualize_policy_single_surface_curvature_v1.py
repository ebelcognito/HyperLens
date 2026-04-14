import argparse
import base64
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from stable_baselines3 import PPO

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from optical_rl.envs.single_surface_curvature_env_v1 import (
    SingleSurfaceCurvatureConfigV1,
    SingleSurfaceCurvatureEnvV1,
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
        description="Visualize trained policy rollout for single-surface-curvature-v1"
    )
    parser.add_argument("--model", default="results/models/final_model_single_surface_curvature_v1.zip")
    parser.add_argument(
        "--config",
        default="optical_rl/configs/stage1_single_surface_curvature_v1.yaml",
    )
    parser.add_argument(
        "--output-dir",
        default=f"results/visualizations/single_surface_curvature_v1_rollout_{timestamp}",
    )
    parser.add_argument("--rollout-steps", type=int, default=600)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--deterministic", type=_str2bool, default=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--cropbox-width", type=int, default=1000)
    return parser.parse_args()


def _save_frame_from_result(result: Dict[str, Any], frame_path: Path) -> None:
    images = result.get("images") or []
    if not images:
        raise RuntimeError("No images returned. Ensure node-canvas and CropBox are present.")
    data_url = images[0].get("dataUrl", "")
    if "," not in data_url:
        raise RuntimeError("Malformed dataUrl in runner output")
    png_data = data_url.split(",", 1)[1]
    frame_path.write_bytes(base64.b64decode(png_data))


def _scene_with_cropbox(scene: Dict[str, Any], env_cfg: Dict[str, Any], cropbox_width: int) -> Dict[str, Any]:
    scene_copy = json.loads(json.dumps(scene))
    scene_copy["objs"].append(
        {
            "type": "CropBox",
            "p1": {"x": 0, "y": 0},
            "p4": {
                "x": float(env_cfg["world_width"]) + 20.0,
                "y": float(env_cfg["world_height"]) + 20.0,
            },
            "width": int(cropbox_width),
        }
    )
    return scene_copy


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


def main() -> None:
    args = parse_args()
    if args.rollout_steps <= 0:
        raise ValueError("--rollout-steps must be > 0")
    if args.fps <= 0:
        raise ValueError("--fps must be > 0")

    cfg = load_config(args.config)
    env_cfg = dict(cfg["env"])

    env_cfg_dict = dict(env_cfg)
    env_cfg_dict["use_persistent_runner"] = False
    env_cfg_dict["verbose_timing"] = False

    env = SingleSurfaceCurvatureEnvV1(SingleSurfaceCurvatureConfigV1(**env_cfg_dict))
    model = PPO.load(args.model)

    output_root = Path(args.output_dir)
    frames_dir = output_root / "episode_000" / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    render_interval = max(1, args.rollout_steps // 60)

    trajectory: List[Dict[str, Any]] = []
    total_reward = 0.0
    frame_count = 0

    obs, info = env.reset(seed=args.seed)

    with PersistentRayOpticsRunner(verbose=False, log_every_n=5000) as runner:
        initial_scene = _scene_with_cropbox(env._build_scene(), env_cfg, args.cropbox_width)
        initial_result, _ = runner.simulate(initial_scene, episode=0, step=0)
        _save_frame_from_result(initial_result, frames_dir / f"frame_{frame_count:04d}.png")
        frame_count += 1

        done = False
        step = 0
        while step < args.rollout_steps and not done:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            step += 1
            total_reward += float(reward)

            if step % render_interval == 0 or done or step == args.rollout_steps:
                scene = _scene_with_cropbox(env._build_scene(), env_cfg, args.cropbox_width)
                result, _ = runner.simulate(scene, episode=0, step=step)
                _save_frame_from_result(result, frames_dir / f"frame_{frame_count:04d}.png")
                frame_count += 1

            trajectory.append(
                {
                    "step": step,
                    "action": [float(action[0])],
                    "reward": float(reward),
                    "power": float(env.current_power),
                    "left_sag_mm": float(env.left_sag_mm),
                    "was_clipped_sag": float(env.was_clipped_sag),
                    "episode_target_y": float(env.episode_target_y),
                    "episode_origin_y": float(env.episode_origin_y),
                    "done": bool(done),
                }
            )

    run_config = {
        "rollout_steps": int(args.rollout_steps),
        "render_interval": int(render_interval),
        "approx_frames_target": 60,
        "seed": int(args.seed),
        "deterministic": bool(args.deterministic),
        "fps": int(args.fps),
    }
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "total_reward": float(total_reward),
        "episode_length": int(step),
        "frame_count": int(frame_count),
        "model_path": args.model,
        "config_path": args.config,
    }

    episode_dir = output_root / "episode_000"
    (episode_dir / "trajectory.json").write_text(
        json.dumps({"config": run_config, "metadata": metadata, "trajectory": trajectory}, indent=2),
        encoding="utf-8",
    )

    video_ok, video_message = _generate_video(frames_dir, episode_dir / "video.mp4", fps=args.fps)

    summary = {
        "metadata": metadata,
        "video_ok": bool(video_ok),
        "video_message": video_message,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    env.close()

    print(
        f"saved outputs under {output_root} | "
        f"steps={step} frames={frame_count} render_interval={render_interval} "
        f"video={'ok' if video_ok else 'failed'}"
    )
    if not video_ok:
        print(f"note: {video_message}")


if __name__ == "__main__":
    main()
