from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from berkeley_cpg_core import BerkeleyResidual, berkeley_rollout
from cpg_residual_core import DEFAULT_XML, ResidualParams, rollout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", choices=("berkeley", "toy"), default="berkeley")
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--elite-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def score_berkeley(vector: np.ndarray, args: argparse.Namespace) -> tuple[float, float, bool, object]:
    metrics = berkeley_rollout(
        BerkeleyResidual.from_vector(vector),
        duration=args.duration,
        stabilize=True,
    )
    score = (
        5.0 * metrics.distance
        + (2.0 if metrics.alive else -2.0)
        - 0.002 * metrics.energy
        - 2.0 * metrics.final_tilt
        - 0.5 * abs(metrics.final_height - 0.515)
    )
    return score, metrics.distance, metrics.alive, metrics


def score_toy(vector: np.ndarray, args: argparse.Namespace) -> tuple[float, float, bool, object]:
    metrics = rollout(
        ResidualParams.from_vector(vector),
        xml_path=args.xml,
        duration=args.duration,
        assist=True,
    )
    return metrics.reward, metrics.distance, metrics.alive, metrics


def main() -> None:
    args = parse_args()
    if args.out is None:
        name = "berkeley_policy_residual.json" if args.robot == "berkeley" else "policy_residual.json"
        args.out = Path(__file__).resolve().parent / name

    rng = np.random.default_rng(args.seed)
    mean = np.zeros(7)
    if args.robot == "berkeley":
        std = np.asarray([0.10, 0.08, 0.12, 0.08, 0.05, 0.18, 0.03], dtype=float)
        scorer = score_berkeley
    else:
        std = np.asarray([0.18, 0.12, 0.16, 0.08, 0.08, 0.25, 0.10], dtype=float)
        scorer = score_toy

    elite_count = max(2, int(args.population * args.elite_fraction))
    best_reward = -np.inf
    best_vector = mean.copy()
    history = []

    for iteration in range(args.iterations):
        samples = rng.normal(mean, std, size=(args.population, len(mean)))
        scored = []
        for vector in samples:
            reward, distance, alive, metrics = scorer(vector, args)
            scored.append((reward, vector, distance, alive, metrics))

        scored.sort(key=lambda item: item[0], reverse=True)
        elites = np.asarray([item[1] for item in scored[:elite_count]])
        mean = elites.mean(axis=0)
        std = np.maximum(elites.std(axis=0), 0.015)

        if scored[0][0] > best_reward:
            best_reward = float(scored[0][0])
            best_vector = scored[0][1].copy()

        history.append({"iteration": iteration, "best_reward": float(scored[0][0])})
        print(
            f"iter={iteration:03d} best_reward={scored[0][0]:.3f} "
            f"distance={scored[0][2]:.3f} alive={scored[0][3]}"
        )

    payload = {
        "robot": args.robot,
        "berkeley_residual" if args.robot == "berkeley" else "residual": best_vector.tolist(),
        "best_reward": best_reward,
        "history": history,
        "note": "CEM residual for CPG-Residual Natural Walker demo.",
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved={args.out}")


if __name__ == "__main__":
    main()
