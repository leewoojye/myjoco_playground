from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from transition_core import DEFAULT_XML, Segment, run_transition


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "transition_dataset.npz")
    parser.add_argument("--seed", type=int, default=11)
    return parser.parse_args()


def random_schedule(rng: np.random.Generator) -> tuple[Segment, ...]:
    start = float(rng.uniform(0.8, 1.4))
    ramp = float(rng.uniform(1.2, 2.2))
    walk = float(rng.uniform(2.5, 4.5))
    recover = float(rng.uniform(0.7, 1.2))
    stop = float(rng.uniform(1.0, 1.8))
    t0 = 0.0
    t1 = start
    t2 = t1 + ramp
    t3 = t2 + walk
    t4 = t3 + recover
    t5 = t4 + stop
    return (
        Segment("stand", t0, t1),
        Segment("start_walk", t1, t2),
        Segment("walk", t2, t3),
        Segment("recover", t3, t4),
        Segment("stop", t4, t5),
        Segment("stand", t5, t5 + 0.8),
    )


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    states = []
    targets = []
    mode_ids = []
    episode_ids = []
    metrics_rows = []

    for episode in range(args.episodes):
        schedule = random_schedule(rng)
        push_time = schedule[3].start + 0.05
        push_vel = float(rng.uniform(-1.5, -0.6))
        metrics, arrays = run_transition(
            xml_path=args.xml,
            schedule=schedule,
            push_time=push_time,
            push_root_pitch_velocity=push_vel,
        )
        states.append(arrays["state"])
        targets.append(arrays["target"])
        mode_ids.append(arrays["mode_id"])
        episode_ids.append(np.full(len(arrays["time"]), episode, dtype=np.int64))
        metrics_rows.append([metrics.distance, float(metrics.alive), metrics.energy, metrics.snap_cost])
        print(f"episode={episode:03d} distance={metrics.distance:.3f} alive={metrics.alive}")

    np.savez_compressed(
        args.out,
        states=np.concatenate(states, axis=0),
        targets=np.concatenate(targets, axis=0),
        mode_ids=np.concatenate(mode_ids, axis=0),
        episode_ids=np.concatenate(episode_ids, axis=0),
        metrics=np.asarray(metrics_rows, dtype=np.float64),
    )
    print(f"saved={args.out}")


if __name__ == "__main__":
    main()
