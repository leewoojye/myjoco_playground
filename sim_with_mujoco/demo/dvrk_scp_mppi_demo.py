from pathlib import Path

import glfw
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from sim_with_mujoco.environment.dvrk_needle_reach_env import DvrkNeedleReachEnv
from sim_with_mujoco.rl.models.dynamics_dvrk import KinematicDynamics
from sim_with_mujoco.rl.planners.csvto_mppi import DvrkCSVTOPlanner
from sim_with_mujoco.utils.dvrk_ik import get_site_transform, solve_rcm_ik
from sim_with_mujoco.utils.math3d import get_body_T
from sim_with_mujoco.viewer.surrol_keyboard_viewer import SurrolKeyboardViewer

ROOT_DIR = Path(__file__).resolve().parents[2]
XML_PATH = ROOT_DIR / "assets" / "robots" / "dvrk" / "scene_psm_surrol_needle_reach_offset.xml"


def get_state(env, ecm_id, jaw_qpos_id):
    world_T_ecm = get_body_T(env.data, ecm_id)
    ecm_T_tip = np.linalg.inv(world_T_ecm) @ get_site_transform(env.data, env.tip_site_id)
    state = np.r_[
        ecm_T_tip[:3, 3],
        Rotation.from_matrix(ecm_T_tip[:3, :3].T).as_rotvec(),
        env.data.qpos[jaw_qpos_id],
    ].astype(np.float32)
    return state, world_T_ecm, ecm_T_tip


def get_rcm_linearization(env, ecm_T_world, ecm_T_tip, wrist_id, dof_map):
    tip_jacp = np.zeros((3, env.model.nv))
    tip_jacr = np.zeros((3, env.model.nv))
    wrist_jacp = np.zeros((3, env.model.nv))
    wrist_jacr = np.zeros((3, env.model.nv))
    mujoco.mj_jacSite(env.model, env.data, tip_jacp, tip_jacr, env.tip_site_id)
    mujoco.mj_jacBody(env.model, env.data, wrist_jacp, wrist_jacr, wrist_id)

    ecm_R_world = ecm_T_world[:3, :3]
    wrist_position = (ecm_T_world @ np.r_[env.data.xpos[wrist_id], 1.0])[:3]
    shaft_direction = ecm_R_world @ env.data.xmat[wrist_id].reshape(3, 3)[:, 2]
    tip_jacobian = np.vstack((ecm_R_world @ tip_jacp, ecm_R_world @ tip_jacr)) @ dof_map
    return (
        wrist_position,
        shaft_direction,
        ecm_T_tip[:3, :3],
        tip_jacobian,
        ecm_R_world @ wrist_jacp @ dof_map,
        ecm_R_world @ wrist_jacr @ dof_map,
    )


def main():
    env = DvrkNeedleReachEnv(XML_PATH, control_steps=10)
    env.reset()
    ecm_id = env.get_id(mujoco.mjtObj.mjOBJ_BODY, "ECM_tool_roll_link")
    jaw_joint_id = env.get_id(mujoco.mjtObj.mjOBJ_JOINT, "PSM1_jaw")
    jaw_actuator_id = env.get_id(mujoco.mjtObj.mjOBJ_ACTUATOR, "PSM1_jaw")
    jaw_qpos_id = env.model.jnt_qposadr[jaw_joint_id]
    wrist_id = env.get_id(mujoco.mjtObj.mjOBJ_BODY, "PSM1_tool_wrist_link")
    dof_map = np.zeros((env.model.nv, len(env.joint_ids)))
    for column, joint_id in enumerate(env.joint_ids):
        dof_map[env.model.jnt_dofadr[joint_id], column] = 1.0
    pitch_2_id = env.get_id(mujoco.mjtObj.mjOBJ_JOINT, "PSM1_pitch_2")
    pitch_3_id = env.get_id(mujoco.mjtObj.mjOBJ_JOINT, "PSM1_pitch_3")
    dof_map[env.model.jnt_dofadr[pitch_2_id], 1] = -1.0
    dof_map[env.model.jnt_dofadr[pitch_3_id], 1] = 1.0

    state, world_T_ecm, _ = get_state(env, ecm_id, jaw_qpos_id)
    ecm_T_world = np.linalg.inv(world_T_ecm)
    goal = state.copy()
    goal[:3] = (ecm_T_world @ np.r_[env.data.site_xpos[env.target_site_id], 1.0])[:3]
    rcm_position = (ecm_T_world @ np.r_[env.rcm_pos, 1.0])[:3]

    planner = DvrkCSVTOPlanner(
        KinematicDynamics(),
        rcm_position,
        num_samples=128,
        horizon=24,
        num_control_points=4,
        svgd_iterations=3,
    )
    planner.set_goal(goal)

    viewer = SurrolKeyboardViewer(env.model, env.data)
    viewer.init_viewer(
        window_title="dVRK CSVTO-MPPI DEMO",
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
            planner.set_rcm_linearization(
                *get_rcm_linearization(env, np.linalg.inv(world_T_ecm), ecm_T_tip, wrist_id, dof_map)
            )
            action = planner.command(state)
            target_T_ecm = ecm_T_tip.copy()
            target_T_ecm[:3, 3] = action[:3]
            target_T_ecm[:3, :3] = ecm_T_tip[:3, :3] @ Rotation.from_rotvec(action[3:6]).as_matrix()

            q_des = solve_rcm_ik(
                env.model,
                env.data,
                world_T_ecm @ target_T_ecm,
                env.tip_site_id,
                env.rcm_pos,
                env.joint_ids,
                dq_limit=0.045,
                rcm_site_id=env.rcm_site_id,
            )
            for joint_id, actuator_id in zip(env.joint_ids, env.arm_actuator_ids):
                target_qpos = q_des[env.model.jnt_qposadr[joint_id]]
                env.data.ctrl[actuator_id] = np.clip(
                    target_qpos,
                    *env.model.actuator_ctrlrange[actuator_id],
                )
            env.data.ctrl[jaw_actuator_id] = np.clip(
                state[6] + action[6],
                *env.model.actuator_ctrlrange[jaw_actuator_id],
            )
            env.plant.step(env.control_steps)

            next_state, next_world_T_ecm, _ = get_state(env, ecm_id, jaw_qpos_id)
            tip_error = np.linalg.norm(goal[:3] - next_state[:3])
            next_ecm_T_world = np.linalg.inv(next_world_T_ecm)
            wrist_position = (next_ecm_T_world @ np.r_[env.data.xpos[wrist_id], 1.0])[:3]
            shaft_direction = next_ecm_T_world[:3, :3] @ env.data.xmat[wrist_id].reshape(3, 3)[:, 2]
            rcm_error = planner.rcm_deviation(wrist_position, shaft_direction).item()
            feasible = (
                (planner.candidate_rcm_deviation.max(dim=1).values <= planner.rcm_tolerance).float().mean().item()
            )
            viewer.set_overlay([
                f"Tip error: {tip_error * 1000.0:5.1f} mm",
                f"RCM error: {rcm_error * 1000.0:4.1f} mm",
                f"Feasible samples: {feasible * 100.0:4.0f}%",
            ])
            viewer.render()
            print(
                f"step={step:03d} tip_error={tip_error * 1000.0:6.2f} mm "
                f"rcm_error={rcm_error * 1000.0:5.2f} mm feasible={feasible:4.2f}"
            )
            if tip_error <= env.tolerance:
                break
    finally:
        viewer.terminate_viewer()

    print("success" if tip_error <= env.tolerance else "failed")


if __name__ == "__main__":
    main()
