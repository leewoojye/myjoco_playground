from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np


THIS_DIR = Path(__file__).resolve().parent
CPG_DIR = THIS_DIR.parent / "01_cpg_residual_natural_walker"
sys.path.insert(0, str(CPG_DIR))

from cpg_residual_core import (  # noqa: E402
    ASSIST_ACTUATORS,
    DEFAULT_XML,
    WALKER_ACTUATORS,
    CPGParams,
    ResidualParams,
    cpg_targets,
    named_actuator_ids,
    reset_model,
    walker_state,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--control-skip", type=int, default=15)
    parser.add_argument("--out", type=Path, default=THIS_DIR / "motion_rollouts.npz")
    parser.add_argument("--seed", type=int, default=23)
    return parser.parse_args()


def sample_residual(rng: np.random.Generator) -> ResidualParams:
    vector = rng.normal(
        loc=np.zeros(7),
        scale=np.asarray([0.12, 0.08, 0.12, 0.06, 0.05, 0.18, 0.08]),
    )
    return ResidualParams.from_vector(vector)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    model = mujoco.MjModel.from_xml_path(str(args.xml))
    leg_ids = named_actuator_ids(model, WALKER_ACTUATORS)
    assist_ids = named_actuator_ids(model, ASSIST_ACTUATORS)
    dt = float(model.opt.timestep)

    all_states = []
    all_actions = []
    all_conditions = []
    all_scores = []

    for episode in range(args.episodes):
        data = mujoco.MjData(model)
        reset_model(model, data)
        residual = sample_residual(rng)
        target_speed = float(rng.uniform(0.15, 0.45))
        base = CPGParams(target_speed=target_speed)
        states = []
        actions = []
        energy = 0.0
        start_x = float(data.qpos[0])

        for _ in range(args.horizon):
            state = walker_state(data)
            target = cpg_targets(float(data.time), state, base, residual)
            states.append(np.r_[data.qpos.copy(), data.qvel.copy()])
            actions.append(target.copy())
            for _ in range(args.control_skip):
                data.ctrl[leg_ids] = target
                data.ctrl[assist_ids[0]] = 0.08
                data.ctrl[assist_ids[1]] = 0.86
                mujoco.mj_step(model, data)
                joint_ids = model.actuator_trnid[leg_ids, 0]
                dof_ids = model.jnt_dofadr[joint_ids]
                energy += float(np.sum(np.abs(data.actuator_force[leg_ids] * data.qvel[dof_ids]))) * dt

        distance = float(data.qpos[0] - start_x)
        alive = 0.55 < data.qpos[1] < 1.25 and abs(data.qpos[2]) < 0.55
        score = distance - 0.001 * energy + (0.5 if alive else -0.5)
        all_states.append(np.asarray(states, dtype=np.float32))
        all_actions.append(np.asarray(actions, dtype=np.float32))
        all_conditions.append(np.asarray([target_speed, float(alive), distance, energy], dtype=np.float32))
        all_scores.append(score)
        print(f"episode={episode:03d} score={score:.3f} distance={distance:.3f} alive={alive}")

    np.savez_compressed(
        args.out,
        states=np.asarray(all_states, dtype=np.float32),
        actions=np.asarray(all_actions, dtype=np.float32),
        conditions=np.asarray(all_conditions, dtype=np.float32),
        scores=np.asarray(all_scores, dtype=np.float32),
        dt=np.asarray(args.control_skip * dt, dtype=np.float32),
    )
    print(f"saved={args.out}")


if __name__ == "__main__":
    main()
