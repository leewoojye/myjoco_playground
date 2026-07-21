from pathlib import Path

import glfw
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from sim_with_mujoco.environment.dvrk_obstacle_env import DvrkNeedleObstacleEnv
from sim_with_mujoco.rl.models.dynamics_dvrk import RBFEKFDynamics
from sim_with_mujoco.rl.planners.mppi import dVRKMPPIPlanner
from sim_with_mujoco.utils.dvrk_ik import get_site_transform, solve_rcm_ik
from sim_with_mujoco.utils.math3d import get_body_T
from sim_with_mujoco.viewer.surrol_keyboard_viewer import SurrolKeyboardViewer


ROOT_DIR = Path(__file__).resolve().parents[2]
XML_PATH = ROOT_DIR / "assets" / "robots" / "dvrk" / "scene_psm_surrol_needle_obstacle.xml"
ACTION_LIMIT = np.array([0.004, 0.004, 0.004, 0.05, 0.05, 0.05, 0.05], dtype=np.float32)


def get_state(env, ecm_id, jaw_qpos_id):
    world_T_ecm = get_body_T(env.data, ecm_id)
    ecm_T_tip = np.linalg.inv(world_T_ecm) @ get_site_transform(env.data, env.tip_site_id)
    state = np.r_[
        ecm_T_tip[:3, 3],
        Rotation.from_matrix(ecm_T_tip[:3, :3].T).as_rotvec(),
        env.data.qpos[jaw_qpos_id],
    ].astype(np.float32)
    return state, world_T_ecm, ecm_T_tip


def main():
    env = DvrkNeedleObstacleEnv(XML_PATH, control_steps=10)
    env.reset()
    ecm_id = env.get_id(mujoco.mjtObj.mjOBJ_BODY, "ECM_tool_roll_link")
    jaw_joint_id = env.get_id(mujoco.mjtObj.mjOBJ_JOINT, "PSM1_jaw")
    jaw_actuator_id = env.get_id(mujoco.mjtObj.mjOBJ_ACTUATOR, "PSM1_jaw")
    jaw_qpos_id = env.model.jnt_qposadr[jaw_joint_id]

    state, world_T_ecm, _ = get_state(env, ecm_id, jaw_qpos_id)
    ecm_T_world = np.linalg.inv(world_T_ecm)
    target_position = (ecm_T_world @ np.r_[env.data.site_xpos[env.target_site_id], 1.0])[:3]
    obstacle_position = (ecm_T_world @ np.r_[env.obstacle_position, 1.0])[:3]
    goal = state.copy()
    goal[:3] = target_position

    num_basis = 10
    rng = np.random.default_rng(0)
    base_width = 2.0 * np.maximum(np.abs(goal - state), ACTION_LIMIT)
    path = np.linspace(0.0, 1.0, num_basis, dtype=np.float32)
    centers = np.empty((7, num_basis, 2), dtype=np.float32)
    centers[:, :, 0] = state[:, None] + (goal - state)[:, None] * path
    centers[:, :, 0] += rng.normal(size=(7, num_basis)) * 0.1 * base_width[:, None]
    centers[:, 0, 0], centers[:, -1, 0] = state, goal
    centers[:, :, 1] = rng.uniform(-ACTION_LIMIT[:, None], ACTION_LIMIT[:, None], size=(7, num_basis))
    centers[:, 0, 1] = 0.0
    widths = (base_width[:, None] * rng.uniform(0.75, 1.25, size=(7, num_basis))).astype(np.float32)
    dynamics = RBFEKFDynamics(centers, widths, weights=np.ones((7, num_basis), dtype=np.float32))
    planner = dVRKMPPIPlanner(
        dynamics,
        obstacle_position,
        env.obstacle_radius,
        tip_radius=env.TIP_RADIUS,
    )
    planner.set_goal(goal)

    viewer = SurrolKeyboardViewer(env.model, env.data)
    viewer.init_viewer(
        window_title="dVRK MPPI",
        initial_camera=(180, -20, 0.55),
        focus_position=env.data.site_xpos[env.target_site_id],
    )
    tip_error = np.linalg.norm(goal[:3] - state[:3])

    try:
        for step in range(env.max_steps):
            glfw.poll_events()
            if glfw.window_should_close(viewer.window):
                break

            state, world_T_ecm, ecm_T_tip = get_state(env, ecm_id, jaw_qpos_id)
            action = planner.command(state)
            target_T_ecm = ecm_T_tip.copy()
            target_T_ecm[:3, 3] += action[:3]
            target_T_ecm[:3, :3] = ecm_T_tip[:3, :3] @ Rotation.from_rotvec(action[3:6]).as_matrix()

            q_des = solve_rcm_ik(
                env.model,
                env.data,
                world_T_ecm @ target_T_ecm,
                env.tip_site_id,
                env.rcm_pos,
                env.joint_ids,
                dq_limit=0.035,
                rcm_site_id=env.rcm_site_id,
            )
            for joint_id, actuator_id in zip(env.joint_ids, env.arm_actuator_ids):
                target_qpos = q_des[env.model.jnt_qposadr[joint_id]]
                env.data.ctrl[actuator_id] = np.clip(
                    target_qpos, *env.model.actuator_ctrlrange[actuator_id]
                )
            env.data.ctrl[jaw_actuator_id] = np.clip(
                state[6] + action[6],
                *env.model.actuator_ctrlrange[jaw_actuator_id],
            )
            env.plant.step(env.control_steps)

            next_state, _, _ = get_state(env, ecm_id, jaw_qpos_id)
            dynamics.update(state, action, next_state)
            tip_error = np.linalg.norm(goal[:3] - next_state[:3])
            viewer.set_overlay([f"Tip error: {tip_error * 1000.0:5.1f} mm"])
            viewer.render()
            print(f"step={step:03d} tip_error={tip_error * 1000.0:6.2f} mm")
            if tip_error <= env.tolerance:
                break
    finally:
        viewer.terminate_viewer()

    success = tip_error <= env.tolerance
    print(f"{'success' if success else 'failed'}")


if __name__ == "__main__":
    main()
