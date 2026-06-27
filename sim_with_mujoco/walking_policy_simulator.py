from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import glfw
import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sim_with_mujoco.environment.env import Environment
from sim_with_mujoco.rl.walking_policy import (
    CPGWalkingPolicy,
    StandingPolicy,
    WalkingMetrics,
    actuator_ids,
    compute_energy,
    is_alive,
    read_walker_state,
)


DEFAULT_XML = PROJECT_ROOT / "assets" / "robots" / "myjoco_planar_walker" / "scene.xml"
ASSIST_ACTUATORS = ("torso_pitch_assist", "torso_height_assist")
VIEWER_BODY = "torso"


def load_model(xml_path: Path) -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return model, data


def load_environment(xml_path: Path) -> Environment:
    env = Environment(str(xml_path), VIEWER_BODY)
    mujoco.mj_resetDataKeyframe(env.model, env.data, 0)
    mujoco.mj_forward(env.model, env.data)
    return env


def run_episode(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    policy_name: str,
    duration: float,
    realtime: bool,
    render: bool,
    env: Environment | None,
    assist: bool,
    pitch_target: float,
    height_target: float,
) -> WalkingMetrics:
    if policy_name == "stand":
        policy = StandingPolicy()
    elif policy_name == "cpg":
        policy = CPGWalkingPolicy()
    else:
        raise ValueError(f"Unknown policy: {policy_name}")

    ids = actuator_ids(model)
    assist_ids = _assist_actuator_ids(model)
    if not assist:
        model.actuator_forcerange[assist_ids, :] = 0.0

    start_x = float(data.qpos[0])
    energy = 0.0
    dt = float(model.opt.timestep)

    if render:
        if env is None:
            raise ValueError("Custom MyJoCo viewer requires an Environment instance.")
        torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, VIEWER_BODY)
        env.viewer.init_viewer(data.xpos[torso_id])
        env.viewer.cam.distance = 3.2
        env.viewer.cam.azimuth = 90
        env.viewer.cam.elevation = -13
        env.viewer.cam.lookat[:] = (0.6, 0.0, 0.55)

    try:
        wall_start = time.time()
        while data.time < duration:
            if render and glfw.window_should_close(env.viewer.window):
                break
            if render:
                glfw.poll_events()
                env.viewer.poll_target()

            state = read_walker_state(data)
            data.ctrl[ids] = policy.target(float(data.time), state)
            if assist:
                data.ctrl[assist_ids[0]] = pitch_target
                data.ctrl[assist_ids[1]] = height_target
            if env is not None:
                env.step(1)
            else:
                mujoco.mj_step(model, data)
            energy += compute_energy(model, data, ids) * dt

            if render:
                env.viewer.cam.lookat[0] = data.qpos[0] + 0.35
                env.viewer.render()

            if realtime:
                elapsed = time.time() - wall_start
                target_elapsed = data.time
                if target_elapsed > elapsed:
                    time.sleep(target_elapsed - elapsed)
    finally:
        if render and env is not None:
            env.viewer.terminate_viewer()

    final_state = read_walker_state(data)
    distance = float(final_state.root_x - start_x)
    return WalkingMetrics(
        distance=distance,
        mean_speed=distance / max(duration, 1e-6),
        alive=is_alive(final_state),
        final_height=final_state.root_z,
        final_pitch=final_state.root_pitch,
        energy=energy,
    )


def _assist_actuator_ids(model: mujoco.MjModel) -> np.ndarray:
    ids = []
    for name in ASSIST_ACTUATORS:
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if actuator_id < 0:
            raise ValueError(f"Missing assist actuator: {name}")
        ids.append(actuator_id)
    return np.asarray(ids, dtype=int)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a toy MyJoCo planar walking policy demo.")
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML, help="MJCF scene path.")
    parser.add_argument("--policy", choices=("cpg", "stand"), default="cpg")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--headless", action="store_true", help="Run without opening the MuJoCo viewer.")
    parser.add_argument("--fast", action="store_true", help="Disable realtime sleep.")
    parser.add_argument("--no-assist", action="store_true", help="Disable virtual pitch/height stabilizers.")
    parser.add_argument("--pitch-target", type=float, default=0.08)
    parser.add_argument("--height-target", type=float, default=0.86)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = None
    if args.headless:
        model, data = load_model(args.xml)
    else:
        env = load_environment(args.xml)
        model, data = env.model, env.data

    metrics = run_episode(
        model,
        data,
        policy_name=args.policy,
        duration=args.duration,
        realtime=not args.fast,
        render=not args.headless,
        env=env,
        assist=not args.no_assist,
        pitch_target=args.pitch_target,
        height_target=args.height_target,
    )

    print(f"policy={args.policy}")
    print(f"assist={not args.no_assist}")
    print(f"distance={metrics.distance:.3f} m")
    print(f"mean_speed={metrics.mean_speed:.3f} m/s")
    print(f"alive={metrics.alive}")
    print(f"final_height={metrics.final_height:.3f} m")
    print(f"final_pitch={metrics.final_pitch:.3f} rad")
    print(f"energy={metrics.energy:.3f}")


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    main()
