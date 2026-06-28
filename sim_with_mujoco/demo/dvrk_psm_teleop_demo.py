import argparse
import os
from pathlib import Path
import sys
import time

import glfw
import mujoco
import numpy as np

from sim_with_mujoco.environment.env import Environment
from sim_with_mujoco.tasks.surgical.safety_metrics import compute_surgical_metrics
from sim_with_mujoco.tasks.surgical.target_sequence import SurgicalReachTarget, SurgicalTargetSequence
from sim_with_mujoco.utils.dvrk_ik import get_site_transform, solve_dvrk_rcm_ik
from sim_with_mujoco.utils.mj import joint_ids_from_names
from sim_with_mujoco.viewer.surrol_keyboard_viewer import SurrolKeyboardViewer

ROOT_DIR = Path(__file__).resolve().parents[2]
XML_PATH = ROOT_DIR / "assets" / "robots" / "dvrk" / "scene_psm_surrol_needle_reach.xml"

SURROL_SCALING = 5.0
SURROL_POSITION_SCALE = 0.01 * SURROL_SCALING
WORKSPACE_MARGIN = np.array([0.04, 0.06, 0.02])
TARGET_SMOOTHING_TAU = 0.12

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


def _available_joint_mimics(model):
    mimic_names = [
        ("psm_pitch_2", "psm_pitch", -1.0, 0.0),
        ("psm_pitch_3", "psm_pitch", 1.0, 0.0),
        ("psm_pitch_back", "psm_pitch", 1.0, 0.0),
        ("psm_pitch_bottom", "psm_pitch", -1.0, 0.0),
        ("psm_pitch_top", "psm_pitch", -1.0, 0.0),
        ("psm_pitch_front", "psm_pitch", 1.0, 0.0),
        ("psm_jaw_1", "psm_jaw", 0.5, 0.0),
        ("psm_jaw_2", "psm_jaw", 0.5, 0.0),
    ]
    mimics = []
    for passive_name, driver_name, multiplier, offset in mimic_names:
        passive_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, passive_name)
        driver_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, driver_name)
        if passive_id != -1 and driver_id != -1:
            mimics.append((passive_id, driver_id, multiplier, offset))
    return mimics


def _sync_mimic_qpos(model, qpos, joint_mimics):
    for passive_id, driver_id, multiplier, offset in joint_mimics:
        qpos[model.jnt_qposadr[passive_id]] = offset + multiplier * qpos[model.jnt_qposadr[driver_id]]


def _sync_mimic_state(model, data, joint_mimics):
    for passive_id, driver_id, multiplier, offset in joint_mimics:
        passive_qadr = model.jnt_qposadr[passive_id]
        driver_qadr = model.jnt_qposadr[driver_id]
        passive_dadr = model.jnt_dofadr[passive_id]
        driver_dadr = model.jnt_dofadr[driver_id]
        data.qpos[passive_qadr] = offset + multiplier * data.qpos[driver_qadr]
        data.qvel[passive_dadr] = multiplier * data.qvel[driver_dadr]


def _apply_kinematic_servo(model, data, joint_ids, q_des, joint_mimics, jaw_target):
    for joint_id in joint_ids:
        qadr = model.jnt_qposadr[joint_id]
        data.qpos[qadr] = q_des[qadr]

    jaw_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "psm_jaw")
    if jaw_id != -1:
        jaw_qadr = model.jnt_qposadr[jaw_id]
        if model.jnt_limited[jaw_id]:
            lo, hi = model.jnt_range[jaw_id]
            jaw_target = np.clip(jaw_target, lo, hi)
        data.qpos[jaw_qadr] = jaw_target

    _sync_mimic_qpos(model, data.qpos, joint_mimics)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    data.time += model.opt.timestep


def _build_sequence(model, data):
    target_specs = [
        ("Needle Reach", "needle_reach_target", 0.008),
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


def _needle_reach_workspace(initial_tip_pos, goal_pos):
    lower = np.minimum(initial_tip_pos, goal_pos) - WORKSPACE_MARGIN
    upper = np.maximum(initial_tip_pos, goal_pos) + WORKSPACE_MARGIN
    upper[2] += 0.04
    return np.column_stack((lower, upper))


def _smooth_position(current, target, dt, tau):
    if tau <= 0.0:
        return target.copy()
    alpha = 1.0 - np.exp(-dt / tau)
    return current + alpha * (target - current)


def _is_reached(metrics, target):
    return metrics.tip_error <= target.tolerance


def _has_display():
    if sys.platform == "darwin":
        return True
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


def _step_rcm_ik(
    model,
    data,
    ref_data,
    q_home,
    target_T,
    tip_site_id,
    rcm_pos,
    joint_ids,
    joint_mimics,
):
    mujoco.mj_copyData(ref_data, model, data)
    _sync_mimic_state(model, ref_data, joint_mimics)
    mujoco.mj_forward(model, ref_data)

    q_des = solve_dvrk_rcm_ik(
        model,
        ref_data,
        target_T,
        tip_site_id,
        rcm_pos,
        joint_ids,
        q_home=q_home,
        dq_limit=0.012,
        joint_mimics=joint_mimics,
    )
    _apply_kinematic_servo(model, data, joint_ids, q_des, joint_mimics, INITIAL_QPOS["psm_jaw"])


def run_headless_smoke_test(seconds=2.0, xml_path=XML_PATH):
    env = Environment(xml_path, "psm_tool_tip")
    env.initial_qpos(INITIAL_QPOS)

    model = env.model
    data = env.data
    ref_data = mujoco.MjData(model)
    joint_mimics = _available_joint_mimics(model)
    _sync_mimic_qpos(model, data.qpos, joint_mimics)
    mujoco.mj_forward(model, data)

    tip_site_id = _site_id(model, "psm_tool_tip_site")
    shaft_start_site_id = _site_id(model, "psm_shaft_base_site")
    shaft_end_site_id = tip_site_id
    rcm_site_id = _site_id(model, "psm_rcm_site")
    rcm_pos = data.site_xpos[rcm_site_id].copy()
    joint_ids = joint_ids_from_names(model, PSM_JOINT_NAMES)

    sequence = _build_sequence(model, data)
    _set_mocap_target(model, data, sequence.current.position)

    target_T = get_site_transform(data, tip_site_id)
    q_home = data.qpos.copy()
    steps = int(seconds / model.opt.timestep)

    for _ in range(steps):
        target_T[:3, 3] = sequence.current.position
        _step_rcm_ik(
            model,
            data,
            ref_data,
            q_home,
            target_T,
            tip_site_id,
            rcm_pos,
            joint_ids,
            joint_mimics,
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
    print("Ran headless dVRK RCM reach smoke test.")
    print(f"XML: {xml_path}")
    print(f"Task: {sequence.current.name} ({sequence.progress_text()})")
    print(f"Tip error: {metrics.tip_error * 1000.0:.2f} mm")
    print(f"RCM error: {metrics.rcm_error * 1000.0:.2f} mm")
    print(f"Safety: {metrics.status()}")


def run_gui(xml_path=XML_PATH):
    _require_display()

    env = Environment(xml_path, "psm_tool_tip")
    env.initial_qpos(INITIAL_QPOS)

    model = env.model
    data = env.data
    env.viewer = SurrolKeyboardViewer(model, data)
    ref_data = mujoco.MjData(model)
    joint_mimics = _available_joint_mimics(model)
    _sync_mimic_qpos(model, data.qpos, joint_mimics)
    mujoco.mj_forward(model, data)

    tip_site_id = _site_id(model, "psm_tool_tip_site")
    shaft_start_site_id = _site_id(model, "psm_shaft_base_site")
    shaft_end_site_id = tip_site_id
    rcm_site_id = _site_id(model, "psm_rcm_site")
    rcm_pos = data.site_xpos[rcm_site_id].copy()
    joint_ids = joint_ids_from_names(model, PSM_JOINT_NAMES)

    sequence = _build_sequence(model, data)
    _set_mocap_target(model, data, sequence.current.position)

    env.viewer.init_viewer(
        window_title="MyJoCo dVRK PSM Teleoperation",
        initial_camera=(180, -20, 0.55),
        focus_position=sequence.current.position,
    )

    overview_camera_id = _camera_id(model, "overview_camera")

    q_home = data.qpos.copy()
    target_T = get_site_transform(data, tip_site_id)
    servo_T = target_T.copy()
    workspace_limits = _needle_reach_workspace(target_T[:3, 3], sequence.current.position)
    last_action = np.zeros(3, dtype=float)

    poll_interval = SurrolKeyboardViewer.KEY_REPEAT_INTERVAL
    render_interval = 1.0 / 60.0
    last_poll_time = 0.0
    last_render_time = 0.0
    steps_per_frame = 8

    try:
        while not glfw.window_should_close(env.viewer.window):
            glfw.poll_events()
            now = time.time()

            if overview_camera_id != -1 and (
                glfw.get_key(env.viewer.window, glfw.KEY_V) == glfw.PRESS
                or glfw.get_key(env.viewer.window, glfw.KEY_M) == glfw.PRESS
            ):
                env.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                env.viewer.cam.fixedcamid = overview_camera_id
            if glfw.get_key(env.viewer.window, glfw.KEY_F) == glfw.PRESS:
                env.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE

            if now - last_poll_time >= poll_interval:
                last_poll_time = now
                last_action = env.viewer.poll_action()[:3]
                target_T[:3, 3] = target_T[:3, 3] + last_action[:3] * SURROL_POSITION_SCALE
                target_T[:3, 3] = np.clip(
                    target_T[:3, 3],
                    workspace_limits[:, 0],
                    workspace_limits[:, 1],
                )

            for _ in range(steps_per_frame):
                mujoco.mj_copyData(ref_data, model, data)
                _sync_mimic_state(model, ref_data, joint_mimics)
                mujoco.mj_forward(model, ref_data)

                servo_T[:3, 3] = _smooth_position(
                    servo_T[:3, 3],
                    target_T[:3, 3],
                    model.opt.timestep,
                    TARGET_SMOOTHING_TAU,
                )
                q_des = solve_dvrk_rcm_ik(
                    model,
                    ref_data,
                    servo_T,
                    tip_site_id,
                    rcm_pos,
                    joint_ids,
                    q_home=q_home,
                    dq_limit=0.035,
                    joint_mimics=joint_mimics,
                )
                _apply_kinematic_servo(model, data, joint_ids, q_des, joint_mimics, 0.0)

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
                    "Input: SurRoL keyboard preview",
                    f"Task: {sequence.current.name} ({sequence.progress_text()})",
                    f"Action world x/y/z: {last_action[0]:+.3f}, {last_action[1]:+.3f}, {last_action[2]:+.3f}",
                    f"Tip error: {metrics.tip_error * 1000.0:5.1f} mm",
                    f"Reached: {'YES' if _is_reached(metrics, sequence.current) else 'NO'}",
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
    parser.add_argument("--xml", type=Path, help="Override the scene XML path.")
    parser.add_argument("--headless", action="store_true", help="Run a non-interactive numeric reach smoke test.")
    parser.add_argument("--headless-seconds", type=float, default=2.0, help="Duration for the headless smoke test.")
    args = parser.parse_args()
    xml_path = args.xml.expanduser().resolve() if args.xml else XML_PATH

    if args.headless:
        run_headless_smoke_test(args.headless_seconds, xml_path)
    else:
        run_gui(xml_path)
