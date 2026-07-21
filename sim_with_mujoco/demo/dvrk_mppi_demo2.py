from pathlib import Path

import glfw
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from sim_with_mujoco.environment.env import DvrkEnv
from sim_with_mujoco.rl.planners.mppi_pytorch import SRTMPPIPlanner
from sim_with_mujoco.utils.dvrk_ik import get_site_transform, solve_rcm_ik
from sim_with_mujoco.utils.math3d import get_body_T
from sim_with_mujoco.viewer.surrol_keyboard_viewer import SurrolKeyboardViewer


ROOT_DIR = Path(__file__).resolve().parents[2]
XML_PATH = ROOT_DIR / "assets" / "robots" / "dvrk" / "scene_psm_surrol_needle_reach.xml"


def main():
    env = DvrkEnv(XML_PATH, control_steps=10)
    env.reset()
    ecm_id = env.get_id(mujoco.mjtObj.mjOBJ_BODY, "ECM_tool_roll_link")
    jaw_joint_id = env.get_id(mujoco.mjtObj.mjOBJ_JOINT, "PSM1_jaw")
    jaw_actuator_id = env.get_id(mujoco.mjtObj.mjOBJ_ACTUATOR, "PSM1_jaw")
    jaw_qpos_id = env.model.jnt_qposadr[jaw_joint_id]

    world_T_ecm = get_body_T(env.data, ecm_id)
    ecm_T_tip = np.linalg.inv(world_T_ecm) @ get_site_transform(env.data, env.tip_site_id)
    target_position = (np.linalg.inv(world_T_ecm) @ np.append(env.data.site_xpos[env.target_site_id], 1.0))[:3]

    planner = SRTMPPIPlanner(dt=env.control_steps * env.model.opt.timestep)
    planner.set_goal(target_position, ecm_T_tip[:3, :3], env.data.qpos[jaw_qpos_id])

    viewer = SurrolKeyboardViewer(env.model, env.data)
    viewer.init_viewer(
        window_title="MyJoCo dVRK MPPI",
        initial_camera=(180, -20, 0.55),
        focus_position=env.data.site_xpos[env.target_site_id],
    )
    tip_error = np.linalg.norm(target_position - ecm_T_tip[:3, 3])
    viewer.render()

    for step in range(env.max_steps):
        glfw.poll_events()
        if glfw.window_should_close(viewer.window):
            break

        world_T_ecm = get_body_T(env.data, ecm_id)
        ecm_T_tip = np.linalg.inv(world_T_ecm) @ get_site_transform(env.data, env.tip_site_id)
        jacp = np.zeros((3, env.model.nv))
        jacr = np.zeros((3, env.model.nv))
        mujoco.mj_jacSite(env.model, env.data, jacp, jacr, env.tip_site_id)
        world_R_ecm = world_T_ecm[:3, :3]
        state = planner.make_state(
            ecm_T_tip[:3, 3],
            ecm_T_tip[:3, :3],
            env.data.qpos[jaw_qpos_id],
            world_R_ecm.T @ (jacp @ env.data.qvel),
            world_R_ecm.T @ (jacr @ env.data.qvel),
        )
        action = planner.command(state)

        target_T_ecm = ecm_T_tip.copy()
        target_T_ecm[:3, 3] += action[:3]
        target_T_ecm[:3, :3] @= Rotation.from_rotvec(action[3:6]).as_matrix()
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
            if env.model.actuator_ctrllimited[actuator_id]:
                target_qpos = np.clip(target_qpos, *env.model.actuator_ctrlrange[actuator_id])
            env.data.ctrl[actuator_id] = target_qpos
        env.data.ctrl[jaw_actuator_id] = np.clip(
            env.data.qpos[jaw_qpos_id] + action[6],
            *env.model.actuator_ctrlrange[jaw_actuator_id],
        )
        env.plant.step(env.control_steps)

        world_T_ecm = get_body_T(env.data, ecm_id)
        ecm_T_tip = np.linalg.inv(world_T_ecm) @ get_site_transform(env.data, env.tip_site_id)
        tip_error = np.linalg.norm(target_position - ecm_T_tip[:3, 3])
        viewer.set_overlay([f"Tip error: {tip_error * 1000.0:5.1f} mm"])
        viewer.render()
        print(f"step={step:03d} tip_error={tip_error * 1000.0:6.2f} mm")
        if tip_error <= env.tolerance:
            break

    viewer.terminate_viewer()
    print("success" if tip_error <= env.tolerance else "failed")


if __name__ == "__main__":
    main()
