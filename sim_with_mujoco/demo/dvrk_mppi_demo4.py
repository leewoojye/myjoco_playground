from pathlib import Path

import glfw
import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from sim_with_mujoco.environment.dvrk_needle_reach_env import DvrkNeedleReachEnv
from sim_with_mujoco.rl.models.dynamics_dvrk import KinematicDynamics
from sim_with_mujoco.rl.planners.mppi import SMPPIDVRKPlanner
from sim_with_mujoco.utils.dvrk_ik import get_site_transform, solve_rcm_ik
from sim_with_mujoco.utils.math3d import get_body_T
from sim_with_mujoco.viewer.surrol_keyboard_viewer import SurrolKeyboardViewer

ROOT_DIR = Path(__file__).resolve().parents[2]
XML_PATH = ROOT_DIR / "assets" / "robots" / "dvrk" / "scene_psm_surrol_needle_reach_offset.xml"
TRACE_PATH = ROOT_DIR / "temp" / "dvrk_mppi_demo4_trace.pt"
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
    env = DvrkNeedleReachEnv(XML_PATH, control_steps=10)
    env.reset()
    ecm_id = env.get_id(mujoco.mjtObj.mjOBJ_BODY, "ECM_tool_roll_link")
    jaw_joint_id = env.get_id(mujoco.mjtObj.mjOBJ_JOINT, "PSM1_jaw")
    jaw_actuator_id = env.get_id(mujoco.mjtObj.mjOBJ_ACTUATOR, "PSM1_jaw")
    jaw_qpos_id = env.model.jnt_qposadr[jaw_joint_id]

    state, world_T_ecm, _ = get_state(env, ecm_id, jaw_qpos_id)
    ecm_T_world = np.linalg.inv(world_T_ecm)
    target_position = (ecm_T_world @ np.r_[env.data.site_xpos[env.target_site_id], 1.0])[:3]
    goal = state.copy()
    goal[:3] = target_position

    dynamics = KinematicDynamics()
    planner = SMPPIDVRKPlanner(dynamics)
    planner.set_goal(goal)
    trace = []

    viewer = SurrolKeyboardViewer(env.model, env.data)
    viewer.init_viewer(
        window_title="dVRK MPPI DEMO",
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
            nominal_before = torch.roll(planner.mppi.action_sequence.detach().cpu().clone(), -1, dims=0)
            nominal_before[-1] = planner.mppi.action_sequence[-1].detach().cpu()
            action = planner.command(state)
            mppi = planner.mppi
            candidate_states = torch.cat(
                (mppi.state.expand(mppi.K, -1).unsqueeze(1), mppi.states[0, :, :-1]),
                dim=1,
            )
            candidate_actions = mppi.actions[0].clone()
            candidate_actions[..., :3] += candidate_states[..., :3]
            step_trace = {
                "state": torch.from_numpy(state.copy()),
                "rbf_candidate_states": candidate_states.detach().cpu(),
                "rbf_candidate_actions": candidate_actions.detach().cpu(),
                "rollout_position": mppi.states[0, ..., :3].detach().cpu().clone(),
                "rollout_cost": mppi.cost_total.detach().cpu().clone(),
                "rollout_weight": mppi.omega.detach().cpu().clone(),
                "nominal_before": nominal_before,
                "nominal_after": mppi.action_sequence.detach().cpu().clone(),
            }
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
                env.data.ctrl[actuator_id] = np.clip(target_qpos, *env.model.actuator_ctrlrange[actuator_id])
            env.data.ctrl[jaw_actuator_id] = np.clip(
                state[6] + action[6],
                *env.model.actuator_ctrlrange[jaw_actuator_id],
            )
            env.plant.step(env.control_steps)

            next_state, _, _ = get_state(env, ecm_id, jaw_qpos_id)
            step_trace.update({
                "rbf_action": torch.from_numpy(action.copy()),
                "next_state": torch.from_numpy(next_state.copy()),
            })
            trace.append(step_trace)
            tip_error = np.linalg.norm(goal[:3] - next_state[:3])
            viewer.set_overlay([f"Tip error: {tip_error * 1000.0:5.1f} mm"])
            viewer.render()
            print(f"step={step:03d} tip_error={tip_error * 1000.0:6.2f} mm")
            if tip_error <= env.tolerance:
                break
    finally:
        viewer.terminate_viewer()
        TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "goal": torch.from_numpy(goal.copy()),
                "rbf_position_action": "absolute_ecm",
                "steps": trace,
            },
            TRACE_PATH,
        )
        print(f"trace saved: {TRACE_PATH}")

    success = tip_error <= env.tolerance
    print(f"{'success' if success else 'failed'}")


if __name__ == "__main__":
    main()
