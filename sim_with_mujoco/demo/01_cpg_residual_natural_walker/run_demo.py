from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import glfw
import mujoco
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[2]
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from sim_with_mujoco.environment.env import Environment

from berkeley_cpg_core import (
    DEFAULT_BERKELEY_XML,
    BERKELEY_ACTUATORS,
    BerkeleyCPGParams,
    BerkeleyMetrics,
    BerkeleyResidual,
    apply_pelvis_stabilizer,
    berkeley_rollout,
    berkeley_targets,
    is_berkeley_alive,
    named_actuator_ids as berkeley_actuator_ids,
)
from cpg_residual_core import (
    ASSIST_ACTUATORS,
    DEFAULT_XML,
    WALKER_ACTUATORS,
    CPGParams,
    ResidualParams,
    RolloutMetrics,
    cpg_targets,
    is_alive,
    named_actuator_ids as toy_actuator_ids,
    rollout,
    walker_state,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", choices=("berkeley", "toy"), default="berkeley")
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--policy-json", type=Path, default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--no-assist", action="store_true")
    return parser.parse_args()


def resolve_xml(robot: str, xml_path: Path) -> Path:
    if xml_path != DEFAULT_XML:
        return xml_path
    if robot == "berkeley":
        return DEFAULT_BERKELEY_XML
    return DEFAULT_XML


def load_residual(policy_json: Path | None, robot: str) -> ResidualParams | BerkeleyResidual:
    if policy_json is None:
        return BerkeleyResidual() if robot == "berkeley" else ResidualParams()
    with policy_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    vector = payload.get("berkeley_residual", payload.get("residual"))
    if vector is None:
        raise ValueError(f"Missing residual vector in {policy_json}")
    if robot == "berkeley":
        return BerkeleyResidual.from_vector(vector)
    return ResidualParams.from_vector(vector)


def run_toy_viewer(
    residual: ResidualParams,
    *,
    xml_path: Path,
    duration: float,
    assist: bool,
    realtime: bool,
) -> RolloutMetrics:
    env = Environment(str(xml_path), "torso")
    mujoco.mj_resetDataKeyframe(env.model, env.data, 0)
    mujoco.mj_forward(env.model, env.data)

    leg_ids = toy_actuator_ids(env.model, WALKER_ACTUATORS)
    assist_ids = toy_actuator_ids(env.model, ASSIST_ACTUATORS)
    if not assist:
        env.model.actuator_forcerange[assist_ids, :] = 0.0

    torso_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    env.viewer.init_viewer(env.data.xpos[torso_id])
    env.viewer.cam.distance = 3.2
    env.viewer.cam.azimuth = 90
    env.viewer.cam.elevation = -13
    env.viewer.cam.lookat[:] = (0.6, 0.0, 0.55)

    start_x = float(env.data.qpos[0])
    start_time = float(env.data.time)
    wall_start = time.time()
    dt = float(env.model.opt.timestep)
    render_interval = 1.0 / 60.0
    next_render_time = 0.0
    energy = 0.0
    smoothness = 0.0
    pitch_cost = 0.0
    alive_time = 0.0
    prev_target = None
    prev_delta = np.zeros(len(leg_ids))

    try:
        while env.data.time - start_time < duration and not glfw.window_should_close(env.viewer.window):
            glfw.poll_events()
            env.viewer.poll_target()

            target = cpg_targets(float(env.data.time), walker_state(env.data), CPGParams(), residual)
            env.data.ctrl[leg_ids] = target
            if assist:
                env.data.ctrl[assist_ids[0]] = 0.08
                env.data.ctrl[assist_ids[1]] = 0.86

            if prev_target is not None:
                delta = target - prev_target
                smoothness += float(np.sum((delta - prev_delta) ** 2))
                prev_delta = delta
            prev_target = target.copy()

            env.step(1)
            joint_ids = env.model.actuator_trnid[leg_ids, 0]
            dof_ids = env.model.jnt_dofadr[joint_ids]
            energy += float(np.sum(np.abs(env.data.actuator_force[leg_ids] * env.data.qvel[dof_ids]))) * dt
            pitch_cost += float(env.data.qpos[2] ** 2 + 0.04 * env.data.qvel[2] ** 2) * dt
            if is_alive(env.data):
                alive_time = float(env.data.time - start_time)

            sim_elapsed = float(env.data.time - start_time)
            if sim_elapsed >= next_render_time:
                env.viewer.cam.lookat[0] = env.data.qpos[0] + 0.35
                env.viewer.render()
                next_render_time += render_interval

            if realtime:
                elapsed = time.time() - wall_start
                if sim_elapsed > elapsed:
                    time.sleep(sim_elapsed - elapsed)
    finally:
        env.viewer.terminate_viewer()

    elapsed = max(float(env.data.time - start_time), 1e-6)
    distance = float(env.data.qpos[0] - start_x)
    mean_speed = distance / elapsed
    alive = bool(is_alive(env.data))
    reward = (
        8.0 * distance
        + 2.0 * min(alive_time / max(duration, 1e-6), 1.0)
        - 2.0 * abs(mean_speed - CPGParams().target_speed)
        - 0.0015 * energy
        - 0.06 * smoothness
        - 2.0 * pitch_cost
    )
    if not alive:
        reward -= 2.0

    return RolloutMetrics(
        reward=reward,
        distance=distance,
        mean_speed=mean_speed,
        alive_time=alive_time,
        alive=alive,
        energy=energy,
        smoothness=smoothness,
        pitch_cost=pitch_cost,
        final_height=float(env.data.qpos[1]),
        final_pitch=float(env.data.qpos[2]),
    )


def run_berkeley_viewer(
    residual: BerkeleyResidual,
    *,
    xml_path: Path,
    duration: float,
    stabilize: bool,
    realtime: bool,
) -> BerkeleyMetrics:
    env = Environment(str(xml_path), "torso")
    mujoco.mj_resetDataKeyframe(env.model, env.data, 0)
    mujoco.mj_forward(env.model, env.data)

    ids = berkeley_actuator_ids(env.model, BERKELEY_ACTUATORS)
    home = env.data.ctrl.copy()
    params = BerkeleyCPGParams()
    torso_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    env.viewer.init_viewer(env.data.xpos[torso_id])
    env.viewer.cam.distance = 2.0
    env.viewer.cam.azimuth = 90
    env.viewer.cam.elevation = -12
    env.viewer.cam.lookat[:] = (0.4, 0.0, 0.42)

    start_x = float(env.data.qpos[0])
    start_time = float(env.data.time)
    wall_start = time.time()
    dt = float(env.model.opt.timestep)
    render_interval = 1.0 / 60.0
    next_render_time = 0.0
    energy = 0.0

    try:
        while env.data.time - start_time < duration and not glfw.window_should_close(env.viewer.window):
            glfw.poll_events()
            env.viewer.poll_target()

            env.data.ctrl[ids] = berkeley_targets(float(env.data.time), home, params, residual)
            if stabilize:
                apply_pelvis_stabilizer(env.model, env.data, target_height=params.target_height)
            else:
                env.data.qfrc_applied[:] = 0.0

            env.step(1)
            joint_ids = env.model.actuator_trnid[ids, 0]
            dof_ids = env.model.jnt_dofadr[joint_ids]
            energy += float(np.sum(np.abs(env.data.actuator_force[ids] * env.data.qvel[dof_ids]))) * dt

            sim_elapsed = float(env.data.time - start_time)
            if sim_elapsed >= next_render_time:
                env.viewer.cam.lookat[0] = env.data.qpos[0] + 0.25
                env.viewer.render()
                next_render_time += render_interval

            if realtime:
                elapsed = time.time() - wall_start
                if sim_elapsed > elapsed:
                    time.sleep(sim_elapsed - elapsed)
    finally:
        env.viewer.terminate_viewer()

    elapsed = max(float(env.data.time - start_time), 1e-6)
    distance = float(env.data.qpos[0] - start_x)
    return BerkeleyMetrics(
        distance=distance,
        mean_speed=distance / elapsed,
        alive=is_berkeley_alive(env.data),
        final_height=float(env.data.qpos[2]),
        final_tilt=float(np.linalg.norm(env.data.qpos[4:7])),
        energy=energy,
    )


def main() -> None:
    args = parse_args()
    xml_path = resolve_xml(args.robot, args.xml)
    residual = load_residual(args.policy_json, args.robot)

    if args.robot == "berkeley":
        if args.headless:
            metrics = berkeley_rollout(
                residual,
                xml_path=xml_path,
                duration=args.duration,
                stabilize=not args.no_assist,
            )
        else:
            metrics = run_berkeley_viewer(
                residual,
                xml_path=xml_path,
                duration=args.duration,
                stabilize=not args.no_assist,
                realtime=not args.fast,
            )
    else:
        if args.headless:
            metrics = rollout(
                residual,
                xml_path=xml_path,
                duration=args.duration,
                assist=not args.no_assist,
            )
        else:
            metrics = run_toy_viewer(
                residual,
                xml_path=xml_path,
                duration=args.duration,
                assist=not args.no_assist,
                realtime=not args.fast,
            )
    print(json.dumps(metrics.__dict__, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
