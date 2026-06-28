import argparse
import os
from pathlib import Path
import time

import glfw
import mujoco
import numpy as np

from sim.model.math3d.rotation import rpy2rotation_matrix
from sim_with_mujoco.environment.env import Environment
from sim_with_mujoco.tasks.surgical.safety_metrics import compute_surgical_metrics
from sim_with_mujoco.tasks.surgical.target_sequence import SurgicalReachTarget, SurgicalTargetSequence
from sim_with_mujoco.utils.dvrk_ik import get_site_transform, solve_dvrk_rcm_ik
from sim_with_mujoco.utils.mj import joint_ids_from_names

ROOT_DIR = Path(__file__).resolve().parents[2]
XML_PATH = ROOT_DIR / "assets" / "robots" / "dvrk" / "scene_psm_peg_needle.xml"

PSM_JOINT_NAMES = [
    "psm_yaw",
    "psm_pitch",
    "psm_insertion",
    "psm_roll",
    "psm_wrist_pitch",
    "psm_wrist_yaw",
]

INITIAL_QPOS = {
    "psm_yaw": 0.18,
    "psm_pitch": 0.08,
    "psm_insertion": 0.07,
    "psm_roll": 0.0,
    "psm_wrist_pitch": 0.0,
    "psm_wrist_yaw": 0.0,
    "psm_jaw": 0.15,
}


def _site_id(model, name):
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if site_id == -1:
        raise ValueError(f"Unknown site: {name}")
    return site_id


def _camera_id(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)


def _set_position_ctrl(model, data, joint_ids, q_des):
    for joint_id in joint_ids:
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)
        if actuator_id == -1:
            continue

        qadr = model.jnt_qposadr[joint_id]
        ctrl = q_des[qadr]
        if model.actuator_ctrllimited[actuator_id]:
            lo, hi = model.actuator_ctrlrange[actuator_id]
            ctrl = np.clip(ctrl, lo, hi)
        data.ctrl[actuator_id] = ctrl


def _set_jaw(model, data, jaw_target):
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "psm_jaw")
    if actuator_id != -1:
        lo, hi = model.actuator_ctrlrange[actuator_id]
        data.ctrl[actuator_id] = np.clip(jaw_target, lo, hi)

    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "psm_jaw")
    if joint_id != -1:
        qadr = model.jnt_qposadr[joint_id]
        lower, upper = model.jnt_range[joint_id]
        data.qpos[qadr] = np.clip(jaw_target, lower, upper)


def _apply_kinematic_qpos(model, data, joint_ids, q_des):
    for joint_id in joint_ids:
        qadr = model.jnt_qposadr[joint_id]
        data.qpos[qadr] = q_des[qadr]
    data.qvel[:] = 0.0
    _set_position_ctrl(model, data, joint_ids, q_des)


def _build_sequence(model, data):
    target_specs = [
        ("Peg 1", "peg_target_1", 0.006),
        ("Peg 2", "peg_target_2", 0.006),
        ("Peg 3", "peg_target_3", 0.006),
        ("Needle Approach", "needle_approach_target", 0.008),
    ]
    targets = [
        SurgicalReachTarget(name, data.site_xpos[_site_id(model, site_name)].copy(), tolerance)
        for name, site_name, tolerance in target_specs
    ]
    return SurgicalTargetSequence(targets)


def _set_mocap_target(model, data, position):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "active_target_marker")
    mocap_id = model.body_mocapid[body_id]
    if mocap_id != -1:
        data.mocap_pos[mocap_id] = position
        data.mocap_quat[mocap_id] = np.array([1.0, 0.0, 0.0, 0.0])


def _has_display():
    if os.name != "posix":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _require_display():
    if _has_display():
        return
    raise RuntimeError(
        "GUI demo requested, but DISPLAY/WAYLAND_DISPLAY is not set. "
        "Run this from a desktop/remote-desktop session or enable SSH X11 forwarding. "
        "Use --headless only for the numeric smoke test."
    )


def _step_rcm_ik(model, data, ref_data, q_des, q_home, target_T, tip_site_id, shaft_start_site_id, rcm_pos, joint_ids):
    mujoco.mj_copyData(ref_data, model, data)
    ref_data.qpos[:] = q_des
    ref_data.qvel[:] = 0.0
    mujoco.mj_forward(model, ref_data)

    ik_result = solve_dvrk_rcm_ik(
        model,
        ref_data,
        target_T,
        tip_site_id,
        shaft_start_site_id,
        tip_site_id,
        rcm_pos,
        joint_ids,
        model.opt.timestep,
        q_home=q_home,
        dq_limit=0.012,
        pose_weight=(0.02, 1.0),
        posture_weight=0.0,
    )
    q_des = ik_result.q_next
    _apply_kinematic_qpos(model, data, joint_ids, q_des)
    _set_jaw(model, data, INITIAL_QPOS["psm_jaw"])
    mujoco.mj_forward(model, data)
    data.time += model.opt.timestep
    return q_des


def run_headless_smoke_test(seconds=2.0):
    env = Environment(XML_PATH, "psm_tool_tip")
    env.initial_qpos(INITIAL_QPOS)

    model = env.model
    data = env.data
    ref_data = mujoco.MjData(model)

    tip_site_id = _site_id(model, "psm_tool_tip_site")
    shaft_start_site_id = _site_id(model, "psm_shaft_base_site")
    shaft_end_site_id = tip_site_id
    rcm_site_id = _site_id(model, "psm_rcm_site")
    rcm_pos = data.site_xpos[rcm_site_id].copy()
    joint_ids = joint_ids_from_names(model, PSM_JOINT_NAMES)

    sequence = _build_sequence(model, data)
    _set_mocap_target(model, data, sequence.current.position)

    target_T = get_site_transform(data, tip_site_id)
    q_des = data.qpos.copy()
    q_home = data.qpos.copy()
    steps = int(seconds / model.opt.timestep)

    for _ in range(steps):
        target_T[:3, 3] = sequence.current.position
        q_des = _step_rcm_ik(
            model,
            data,
            ref_data,
            q_des,
            q_home,
            target_T,
            tip_site_id,
            shaft_start_site_id,
            rcm_pos,
            joint_ids,
        )
        sequence.update(data.site_xpos[tip_site_id].copy(), data.time)
        _set_mocap_target(model, data, sequence.current.position)

    metrics = compute_surgical_metrics(
        model,
        data,
        tip_site_id,
        shaft_start_site_id,
        shaft_end_site_id,
        rcm_pos,
        sequence.current.position,
        joint_ids,
    )
    print("DISPLAY/WAYLAND_DISPLAY is missing; ran headless dVRK RCM reach smoke test instead.")
    print(f"XML: {XML_PATH}")
    print(f"Task: {sequence.current.name} ({sequence.progress_text()})")
    print(f"Tip error: {metrics.tip_error * 1000.0:.2f} mm")
    print(f"RCM error: {metrics.rcm_error * 1000.0:.2f} mm")
    print(f"Safety: {metrics.status()}")


def run_gui():
    _require_display()

    env = Environment(XML_PATH, "psm_tool_tip")
    env.initial_qpos(INITIAL_QPOS)

    model = env.model
    data = env.data
    ref_data = mujoco.MjData(model)

    tip_site_id = _site_id(model, "psm_tool_tip_site")
    shaft_start_site_id = _site_id(model, "psm_shaft_base_site")
    shaft_end_site_id = tip_site_id
    rcm_site_id = _site_id(model, "psm_rcm_site")
    rcm_pos = data.site_xpos[rcm_site_id].copy()
    joint_ids = joint_ids_from_names(model, PSM_JOINT_NAMES)

    sequence = _build_sequence(model, data)
    _set_mocap_target(model, data, sequence.current.position)

    env.viewer.init_viewer(
        env.initial_target_pos,
        slider_range=(-0.22, 0.22),
        rotation_slider_range=(-0.8, 0.8),
        target_axes=("dX", "dY", "dZ", "Roll", "Pitch", "Yaw", "Jaw", "Scale"),
        target_ranges=[
            (-0.22, 0.22),
            (-0.22, 0.22),
            (-0.12, 0.12),
            (-0.8, 0.8),
            (-0.8, 0.8),
            (-0.8, 0.8),
            (0.0, 0.8),
            (0.15, 1.0),
        ],
        window_title="MyJoCo dVRK PSM Teleoperation",
        initial_camera=(145, -24, 1.0, 0.28),
    )

    overview_camera_id = _camera_id(model, "overview_camera")
    ecm_camera_id = _camera_id(model, "ecm_camera")
    if overview_camera_id != -1:
        env.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        env.viewer.cam.fixedcamid = overview_camera_id

    initial_tip_T = get_site_transform(data, tip_site_id)
    target_T = initial_tip_T.copy()
    q_des = data.qpos.copy()
    q_home = data.qpos.copy()
    jaw_target = INITIAL_QPOS["psm_jaw"]
    motion_scale = 0.35

    poll_interval = 1.0 / 60.0
    render_interval = 1.0 / 60.0
    last_poll_time = 0.0
    last_render_time = 0.0
    steps_per_frame = 8

    try:
        while not glfw.window_should_close(env.viewer.window):
            glfw.poll_events()
            now = time.time()

            if ecm_camera_id != -1 and glfw.get_key(env.viewer.window, glfw.KEY_C) == glfw.PRESS:
                env.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                env.viewer.cam.fixedcamid = ecm_camera_id
            if overview_camera_id != -1 and glfw.get_key(env.viewer.window, glfw.KEY_V) == glfw.PRESS:
                env.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                env.viewer.cam.fixedcamid = overview_camera_id

            if now - last_poll_time >= poll_interval:
                last_poll_time = now
                polled_target, _ = env.viewer.poll_target()

                if polled_target is not None:
                    target_T = initial_tip_T.copy()
                    target_T[:3, 3] = polled_target[:3]
                    target_rpy = polled_target[3:6]
                    target_rot = rpy2rotation_matrix(target_rpy[0], target_rpy[1], target_rpy[2])
                    target_T[:3, :3] = initial_tip_T[:3, :3] @ target_rot
                    jaw_target = polled_target[6]
                    motion_scale = max(0.15, polled_target[7])

            for _ in range(steps_per_frame):
                mujoco.mj_copyData(ref_data, model, data)
                ref_data.qpos[:] = q_des
                ref_data.qvel[:] = 0.0
                mujoco.mj_forward(model, ref_data)

                ik_result = solve_dvrk_rcm_ik(
                    model,
                    ref_data,
                    target_T,
                    tip_site_id,
                    shaft_start_site_id,
                    shaft_end_site_id,
                    rcm_pos,
                    joint_ids,
                    model.opt.timestep,
                    q_home=q_home,
                    dq_limit=0.035 * motion_scale,
                    pose_weight=(0.02, 1.0),
                    posture_weight=0.0,
                )
                q_des = ik_result.q_next

                _apply_kinematic_qpos(model, data, joint_ids, q_des)
                _set_jaw(model, data, jaw_target)
                mujoco.mj_forward(model, data)
                data.time += model.opt.timestep

            tip_pos = data.site_xpos[tip_site_id].copy()
            sequence.update(tip_pos, data.time)
            _set_mocap_target(model, data, sequence.current.position)
            metrics = compute_surgical_metrics(
                model,
                data,
                tip_site_id,
                shaft_start_site_id,
                shaft_end_site_id,
                rcm_pos,
                sequence.current.position,
                joint_ids,
            )
            env.viewer.set_overlay(
                [
                    f"Task: {sequence.current.name} ({sequence.progress_text()})",
                    f"Tip error: {metrics.tip_error * 1000.0:5.1f} mm",
                    f"RCM error: {metrics.rcm_error * 1000.0:5.2f} mm",
                    f"Joint margin: {metrics.joint_margin:5.3f} rad/m",
                    f"Safety: {metrics.status()}",
                ]
            )

            if now - last_render_time >= render_interval:
                env.viewer.render()
                last_render_time = now

    finally:
        env.viewer.terminate_viewer()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="dVRK PSM teleoperation demo with RCM-constrained IK.")
    parser.add_argument("--headless", action="store_true", help="Run a non-interactive numeric reach smoke test.")
    parser.add_argument("--headless-seconds", type=float, default=2.0, help="Duration for the headless smoke test.")
    args = parser.parse_args()

    if args.headless:
        run_headless_smoke_test(args.headless_seconds)
    else:
        run_gui()
