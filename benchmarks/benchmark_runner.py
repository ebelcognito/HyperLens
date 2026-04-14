import argparse
import statistics
import sys
from pathlib import Path

# Add project root for optical_rl imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optical_rl.utils.ray_optics_bridge import run_simulation
from optical_rl.utils.ray_optics_bridge_persistent import PersistentRayOpticsRunner
from optical_rl.utils.scene_builder import scene_direct


def _format_stats(samples):
    avg = statistics.mean(samples)
    p50 = statistics.median(samples)
    p95 = sorted(samples)[max(int(len(samples) * 0.95) - 1, 0)]
    return avg, p50, p95


def benchmark(num_calls: int):
    scene = scene_direct()

    one_shot_times = []
    for _ in range(num_calls):
        _, timing = run_simulation(scene)
        one_shot_times.append(timing["total_ms"])

    persistent_times = []
    with PersistentRayOpticsRunner() as runner:
        for i in range(num_calls):
            _, timing = runner.simulate(scene, episode=0, step=i)
            persistent_times.append(timing["total_ms"])

    one_avg, one_p50, one_p95 = _format_stats(one_shot_times)
    per_avg, per_p50, per_p95 = _format_stats(persistent_times)
    speedup = one_avg / per_avg if per_avg > 0 else float("inf")

    print(f"calls={num_calls}")
    print(
        f"one-shot avg={one_avg:.2f}ms p50={one_p50:.2f}ms p95={one_p95:.2f}ms"
    )
    print(
        f"persistent avg={per_avg:.2f}ms p50={per_p50:.2f}ms p95={per_p95:.2f}ms"
    )
    print(f"speedup={speedup:.2f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark ray-optics one-shot vs persistent runner")
    parser.add_argument("--calls", type=int, default=100, help="Number of simulation calls")
    args = parser.parse_args()

    benchmark(num_calls=args.calls)
