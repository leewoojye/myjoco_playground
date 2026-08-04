import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import glfw

if os.environ.get("MUJOCO_GL") == "osmesa" and "--headless" in sys.argv:
    os.environ.pop("MUJOCO_GL")

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from sim_with_mujoco.environment.dvrk_needle_reach_env import DvrkNeedleReachEnv
from sim_with_mujoco.rl.models.dynamics_dvrk import KinematicDynamics
from sim_with_mujoco.rl.planners.scp_smppi import DvrkSCPSMPPIPlanner
from sim_with_mujoco.utils.dvrk_ik import get_site_transform
from sim_with_mujoco.utils.ik_qp import solve_differential_ik
from sim_with_mujoco.utils.math3d import get_body_T
from sim_with_mujoco.viewer.surrol_keyboard_viewer import SurrolKeyboardViewer

ROOT_DIR = Path(__file__).resolve().parents[2]
XML_PATH = ROOT_DIR / "assets" / "robots" / "dvrk" / "scene_psm_surrol_needle_reach_offset.xml"
TRACE_PATH = ROOT_DIR / "temp" / "dvrk_scp_mppi_trace.pt"
INITIAL_CAMERA = (180, -20, 0.55)


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default="cuda",
        help="PyTorch device for batched MPPI rollouts and SVGD updates (default: cuda).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible SCP-SMPPI particle sampling (default: 0).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without the interactive GLFW viewer.",
    )
    parser.add_argument(
        "--record",
        type=Path,
        metavar="MP4_PATH",
        help="Write a headless offscreen render to this MP4 path.",
    )
    parser.add_argument(
        "--video-fps",
        type=int,
        default=30,
        help="Frames per second for --record (default: 30).",
    )
    parser.add_argument(
        "--render-width",
        type=int,
        default=1200,
        help="Video width in pixels for --record (default: 1200).",
    )
    parser.add_argument(
        "--render-height",
        type=int,
        default=900,
        help="Video height in pixels for --record (default: 900).",
    )
    args = parser.parse_args()
    if args.record is not None and not args.headless:
        parser.error("--record requires --headless")
    if args.video_fps <= 0 or args.render_width <= 0 or args.render_height <= 0:
        parser.error("--video-fps, --render-width, and --render-height must be positive")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    return args


def resolve_device(requested_device):
    device = torch.device(requested_device)
    if device.type != "cuda":
        return device
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot access a CUDA device.")
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise ValueError(
            f"Requested CUDA device index {device.index}, "
            f"but only {torch.cuda.device_count()} visible device(s) are available."
        )
    return device


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def render_headless_video(args, qpos_trajectory):
    """Render CUDA-planned states in an isolated CPU OSMesa process."""
    if not qpos_trajectory:
        return

    with tempfile.TemporaryDirectory(prefix=".dvrk_recording_", dir=args.record.parent) as temporary_dir:
        trajectory_path = Path(temporary_dir) / "qpos_trajectory.npz"
        np.savez_compressed(trajectory_path, qpos=np.asarray(qpos_trajectory))
        renderer_environment = os.environ.copy()
        renderer_environment["MUJOCO_GL"] = "osmesa"
        renderer_environment["PYOPENGL_PLATFORM"] = "osmesa"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "sim_with_mujoco.demo.dvrk_headless_video_renderer",
                "--trajectory",
                str(trajectory_path),
                "--output",
                str(args.record),
                "--video-fps",
                str(args.video_fps),
                "--render-width",
                str(args.render_width),
                "--render-height",
                str(args.render_height),
            ],
            check=True,
            env=renderer_environment,
        )


def main():
    args = parse_args()
    set_random_seed(args.seed)
    device = resolve_device(args.device)
    env = DvrkNeedleReachEnv(XML_PATH, control_steps=10, max_steps=200)
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
    control_dt = env.control_steps * env.model.opt.timestep

    state, world_T_ecm, _ = get_state(env, ecm_id, jaw_qpos_id)
    ecm_T_world = np.linalg.inv(world_T_ecm)
    goal = state.copy()
    goal[:3] = (ecm_T_world @ np.r_[env.data.site_xpos[env.target_site_id], 1.0])[:3]
    rcm_position = (ecm_T_world @ np.r_[env.rcm_pos, 1.0])[:3]

    planner = DvrkSCPSMPPIPlanner(
        KinematicDynamics().to(device),
        rcm_position,
        num_samples=512,
        horizon=8,
        num_control_points=4,
        svgd_iterations=0,
        lambda_=0.1,  # 가중합을 하고 난 뒤 SVGD를 수행해 가중합 결과에 덜 민감함
        action_smoothness_weight=3.0,
    )
    planner.set_goal(goal)
    trace = []
    physics_tip_time = [float(env.data.time)]
    physics_tip_position = [env.data.site_xpos[env.tip_site_id].copy()]
    print(f"SCP-SMPPI rollout device: {device}")
    print(f"SCP-SMPPI random seed: {args.seed}")

    viewer = None
    qpos_trajectory = [] if args.record is not None else None
    if args.headless:
        if args.record is not None:
            args.record.parent.mkdir(parents=True, exist_ok=True)
            print(f"Recording headless MP4 to: {args.record} (OSMesa CPU renderer)")
    else:
        viewer = SurrolKeyboardViewer(env.model, env.data)
        viewer.init_viewer(
            window_title="dVRK SCP-SMPPI DEMO",
            initial_camera=INITIAL_CAMERA,
            focus_position=env.data.site_xpos[env.target_site_id],
        )
    tip_error = np.linalg.norm(goal[:3] - state[:3])

    try:
        for step in range(env.max_steps):
            if viewer is not None:
                glfw.poll_events()
                if glfw.window_should_close(viewer.window):
                    break

            state, world_T_ecm, ecm_T_tip = get_state(env, ecm_id, jaw_qpos_id)
            planner.set_rcm_linearization(
                *get_rcm_linearization(env, np.linalg.inv(world_T_ecm), ecm_T_tip, wrist_id, dof_map)
            )
            action = planner.command(state)
            step_trace = {
                "state": torch.from_numpy(state.copy()),
                "rollout_position": planner.candidate_states[..., :3].detach().cpu().clone(),
                "rollout_cost": planner.candidate_costs.detach().cpu().clone(),
                "rollout_weight": planner.candidate_weights.detach().cpu().clone(),
                "nominal_after": planner.nominal_action_sequence.detach().cpu().clone(),
                "spline_nominal_after": (planner.nominal_derivative_sequence * planner.dt).detach().cpu().clone(),
                "derivative_after": planner.nominal_derivative_sequence.detach().cpu().clone(),
            }
            target_T_ecm = ecm_T_tip.copy()
            target_T_ecm[:3, 3] = action[:3]
            target_T_ecm[:3, :3] = ecm_T_tip[:3, :3] @ Rotation.from_rotvec(action[3:6]).as_matrix()

            q_des, _, ik_success = solve_differential_ik(
                env.model,
                env.data,
                (
                    env.tip_site_id,
                    world_T_ecm @ target_T_ecm,
                    True,
                    1.0,
                    mujoco.mjtObj.mjOBJ_SITE,
                ),
                env.joint_ids,
                dt=control_dt,
                gain=20.0,
                damping=1e-3,
                dq_limit=0.045,
                dof_map=dof_map,
            )
            if not ik_success:
                raise RuntimeError("Differential IK least-squares solve failed.")
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
            for _ in range(env.control_steps):
                env.plant.step()
                physics_tip_time.append(float(env.data.time))
                physics_tip_position.append(env.data.site_xpos[env.tip_site_id].copy())

            next_state, next_world_T_ecm, _ = get_state(env, ecm_id, jaw_qpos_id)
            step_trace.update({
                "action": torch.from_numpy(action.copy()),
                "next_state": torch.from_numpy(next_state.copy()),
            })
            trace.append(step_trace)
            tip_error = np.linalg.norm(goal[:3] - next_state[:3])
            next_ecm_T_world = np.linalg.inv(next_world_T_ecm)
            wrist_position = (next_ecm_T_world @ np.r_[env.data.xpos[wrist_id], 1.0])[:3]
            shaft_direction = next_ecm_T_world[:3, :3] @ env.data.xmat[wrist_id].reshape(3, 3)[:, 2]
            rcm_error = planner.rcm_deviation(wrist_position, shaft_direction).item()
            feasible = (
                (planner.candidate_rcm_deviation.max(dim=1).values <= planner.rcm_tolerance).float().mean().item()
            )
            if viewer is not None:
                viewer.set_overlay([
                    f"Tip error: {tip_error * 1000.0:5.1f} mm",
                    f"RCM error: {rcm_error * 1000.0:4.1f} mm",
                    f"Feasible samples: {feasible * 100.0:4.0f}%",
                ])
                viewer.render()
            if qpos_trajectory is not None:
                qpos_trajectory.append(env.data.qpos.copy())
            print(
                f"step={step:03d} tip_error={tip_error * 1000.0:6.2f} mm "
                f"rcm_error={rcm_error * 1000.0:5.2f} mm feasible={feasible:4.2f}"
            )
            if tip_error <= env.tolerance:
                break
    finally:
        if viewer is not None:
            viewer.terminate_viewer()
        TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "goal": torch.from_numpy(goal.copy()),
                "position_action": "absolute_ecm",
                "planner": "scp_smppi",
                "seed": args.seed,
                "steps": trace,
                "physics_tip_time": torch.as_tensor(physics_tip_time, dtype=torch.float64),
                "physics_tip_position": torch.as_tensor(np.asarray(physics_tip_position), dtype=torch.float64),
                "physics_samples_per_control": env.control_steps,
            },
            TRACE_PATH,
        )
        print(f"trace saved: {TRACE_PATH}")

    if args.record is not None:
        print(f"SCP-SMPPI finished; rendering {len(qpos_trajectory)} recorded frame(s) with OSMesa.")
        render_headless_video(args, qpos_trajectory)
        print(f"Saved {len(qpos_trajectory)} frame(s) to: {args.record}")
    print("success" if tip_error <= env.tolerance else "failed")


if __name__ == "__main__":
    main()
