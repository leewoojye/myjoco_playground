import sys
from pathlib import Path

import manim as mn
import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation

ROOT_DIR = Path(__file__).resolve().parents[3]
TRACE_PATH = ROOT_DIR / "temp" / "dvrk_mppi_demo4_trace.pt"
XML_PATH = ROOT_DIR / "assets" / "robots" / "dvrk" / "scene_psm_surrol_needle_obstacle.xml"
DT = 0.02

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def trace_states(trace_path=TRACE_PATH):
    trace = torch.load(trace_path, map_location="cpu", weights_only=True)
    steps = trace["steps"]
    states = torch.stack([steps[0]["state"], *[step["next_state"] for step in steps]])
    return trace, states


def tip_trajectory(trace_path=TRACE_PATH):
    trace, states = trace_states(trace_path)
    return trace, states[:, :3]


def trajectory_kinematics(positions, dt=DT):
    positions = torch.as_tensor(positions)
    velocity = torch.diff(positions, dim=-2) / dt
    jerk = torch.diff(velocity, n=2, dim=-2) / dt**2
    return velocity, jerk


def log_dimensionless_jerk(positions, dt=DT):
    positions = torch.as_tensor(positions)
    velocity, jerk = trajectory_kinematics(positions, dt)
    duration = (positions.shape[-2] - 1) * dt
    peak_speed_sq = torch.linalg.vector_norm(velocity, dim=-1).amax(dim=-1).square()
    squared_jerk = jerk.square().sum(dim=(-2, -1)) * dt
    dimensionless_jerk = duration**5 * squared_jerk / peak_speed_sq.clamp_min(torch.finfo(positions.dtype).eps)
    return -torch.log(dimensionless_jerk.clamp_min(torch.finfo(positions.dtype).eps))


def rcm_deviation(trace_path=TRACE_PATH):
    from sim_with_mujoco.utils.dvrk_ik import get_site_transform, solve_rcm_ik
    from sim_with_mujoco.utils.math3d import get_body_T

    trace = torch.load(trace_path, map_location="cpu", weights_only=True)
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    def get_id(object_type, name):
        return mujoco.mj_name2id(model, object_type, name)

    key_id = get_id(mujoco.mjtObj.mjOBJ_KEY, "dvrk_home")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    ecm_id = get_id(mujoco.mjtObj.mjOBJ_BODY, "ECM_tool_roll_link")
    tip_id = get_id(mujoco.mjtObj.mjOBJ_SITE, "PSM1_tool_tip_site")
    rcm_id = get_id(mujoco.mjtObj.mjOBJ_SITE, "PSM1_rcm_site")
    shaft_id = get_id(mujoco.mjtObj.mjOBJ_BODY, "PSM1_tool_wrist_link")
    joint_names = (
        "PSM1_yaw",
        "PSM1_pitch",
        "PSM1_insertion",
        "PSM1_roll",
        "PSM1_wrist_pitch",
        "PSM1_wrist_yaw",
    )
    joint_ids = [get_id(mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names]
    actuator_ids = [get_id(mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in joint_names]
    jaw_joint_id = get_id(mujoco.mjtObj.mjOBJ_JOINT, "PSM1_jaw")
    jaw_id = get_id(mujoco.mjtObj.mjOBJ_ACTUATOR, "PSM1_jaw")
    jaw_qpos_id = model.jnt_qposadr[jaw_joint_id]
    world_T_ecm = get_body_T(data, ecm_id)
    ecm_T_world = np.linalg.inv(world_T_ecm)
    rcm_position = data.site_xpos[rcm_id].copy()
    shaft_position = data.xpos[shaft_id]
    shaft_direction = data.xmat[shaft_id].reshape(3, 3)[:, 2]
    errors = [np.linalg.norm(np.cross(rcm_position - shaft_position, shaft_direction))]

    for step in trace["steps"]:
        action = step["rbf_action"].numpy()
        ecm_T_tip = ecm_T_world @ get_site_transform(data, tip_id)
        target_T_ecm = ecm_T_tip.copy()
        target_T_ecm[:3, 3] = action[:3]
        target_T_ecm[:3, :3] = ecm_T_tip[:3, :3] @ Rotation.from_rotvec(action[3:6]).as_matrix()
        q_des = solve_rcm_ik(
            model,
            data,
            world_T_ecm @ target_T_ecm,
            tip_id,
            rcm_position,
            joint_ids,
            dq_limit=0.035,
            rcm_site_id=rcm_id,
        )
        for joint_id, actuator_id in zip(joint_ids, actuator_ids):
            target_qpos = q_des[model.jnt_qposadr[joint_id]]
            data.ctrl[actuator_id] = np.clip(target_qpos, *model.actuator_ctrlrange[actuator_id])
        data.ctrl[jaw_id] = np.clip(data.qpos[jaw_qpos_id] + action[6], *model.actuator_ctrlrange[jaw_id])
        mujoco.mj_step(model, data, nstep=10)

        shaft_position = data.xpos[shaft_id]
        shaft_direction = data.xmat[shaft_id].reshape(3, 3)[:, 2]
        errors.append(np.linalg.norm(np.cross(rcm_position - shaft_position, shaft_direction)))

    return torch.tensor(errors)


def evaluate_trace(trace_path=TRACE_PATH, dt=DT):
    _, states = trace_states(trace_path)
    positions = states[:, :3]
    velocity, jerk = trajectory_kinematics(positions, dt)
    rcm_error = rcm_deviation(trace_path) * 1000.0
    return {
        "ldlj": log_dimensionless_jerk(positions, dt).item(),
        "duration": (positions.shape[0] - 1) * dt,
        "peak_speed": torch.linalg.vector_norm(velocity, dim=-1).max().item(),
        "integrated_squared_jerk": (jerk.square().sum() * dt).item(),
        "rcm_mean_mm": rcm_error.mean().item(),
        "rcm_max_mm": rcm_error.max().item(),
    }


class LogDimensionlessJerkScene(mn.Scene):
    def construct(self):
        trace, states = trace_states()
        positions = states[:, :3]
        velocity, jerk = trajectory_kinematics(positions)
        positions = positions.numpy()
        speed = torch.linalg.vector_norm(velocity, dim=-1).numpy() * 1000.0
        jerk_norm = torch.linalg.vector_norm(jerk, dim=-1).numpy()
        rcm_error = (rcm_deviation() * 1000.0).numpy()
        time = np.arange(len(positions)) * DT
        speed_time = time[1:]
        jerk_time = time[3:]

        self.camera.background_color = "#101318"
        title = mn.Text("dVRK Trajectory Metrics", font_size=34, weight="BOLD").to_edge(mn.UP)

        path_axes = mn.Axes(
            x_range=self._range(positions[:, 0] * 1000.0, 10.0),
            y_range=self._range(positions[:, 2] * 1000.0, 10.0),
            x_length=6.2,
            y_length=5.5,
            tips=False,
            axis_config={"color": mn.GREY_B, "stroke_width": 2},
        ).shift(3.55 * mn.LEFT + 0.15 * mn.DOWN)
        path_label = mn.Text("ECM X-Z tip path [mm]", font_size=21, color=mn.GREY_A)
        path_label.next_to(path_axes, mn.UP, buff=0.2)

        path_points = [path_axes.c2p(position[0] * 1000.0, position[2] * 1000.0) for position in positions]
        path = mn.VMobject().set_points_as_corners(path_points).set_stroke(mn.WHITE, width=3)
        goal = trace["goal"].numpy()
        goal_dot = mn.Dot(path_axes.c2p(goal[0] * 1000.0, goal[2] * 1000.0), color=mn.GREEN, radius=0.07)

        obstacle = trace["obstacle_position"].numpy() * 1000.0
        radius = float(trace["obstacle_radius"]) * 1000.0
        center = path_axes.c2p(obstacle[0], obstacle[2])
        obstacle_shape = mn.Ellipse(
            width=2 * abs(path_axes.c2p(obstacle[0] + radius, obstacle[2])[0] - center[0]),
            height=2 * abs(path_axes.c2p(obstacle[0], obstacle[2] + radius)[1] - center[1]),
            color=mn.RED,
            fill_opacity=0.18,
        ).move_to(center)

        high_jerk = np.flatnonzero(jerk_norm >= np.quantile(jerk_norm, 0.9)) + 3
        high_jerk_dots = mn.VGroup(*[
            mn.Dot(path_points[index], color=mn.RED, radius=0.035) for index in high_jerk
        ])

        duration = time[-1]
        speed_axes = self._time_axes(duration, speed, 1.0, 20.0).shift(3.4 * mn.RIGHT + 1.9 * mn.UP)
        jerk_axes = self._time_axes(duration, jerk_norm, 1.0, 20.0).shift(3.4 * mn.RIGHT + 0.05 * mn.UP)
        rcm_axes = self._time_axes(duration, rcm_error, 1.0, 0.25).shift(3.4 * mn.RIGHT + 1.8 * mn.DOWN)
        speed_graph = speed_axes.plot_line_graph(
            speed_time,
            speed,
            add_vertex_dots=False,
            line_color=mn.BLUE_B,
            stroke_width=3,
        )
        jerk_graph = jerk_axes.plot_line_graph(
            jerk_time,
            jerk_norm,
            add_vertex_dots=False,
            line_color=mn.RED,
            stroke_width=3,
        )
        rcm_graph = rcm_axes.plot_line_graph(
            time,
            rcm_error,
            add_vertex_dots=False,
            line_color=mn.ORANGE,
            stroke_width=3,
        )
        speed_label = mn.Text("tip speed [mm/s]", font_size=19, color=mn.BLUE_B).next_to(
            speed_axes, mn.UP, buff=0.12
        )
        jerk_label = mn.Text("jerk [10^3 mm/s^3]", font_size=19, color=mn.RED).next_to(
            jerk_axes, mn.UP, buff=0.12
        )
        rcm_label = mn.Text("RCM deviation [mm]", font_size=19, color=mn.ORANGE).next_to(
            rcm_axes, mn.UP, buff=0.12
        )

        ldlj = log_dimensionless_jerk(torch.from_numpy(positions)).item()
        metric = mn.VGroup(
            mn.VGroup(
                mn.Text("LDLJ", font_size=20, color=mn.GREY_A),
                mn.DecimalNumber(ldlj, num_decimal_places=3, font_size=24, color=mn.YELLOW),
            ).arrange(mn.RIGHT, buff=0.12),
            mn.VGroup(
                mn.Text("RCM mean", font_size=20, color=mn.GREY_A),
                mn.DecimalNumber(rcm_error.mean(), num_decimal_places=2, font_size=24, color=mn.ORANGE),
                mn.Text("mm", font_size=18, color=mn.GREY_A),
            ).arrange(mn.RIGHT, buff=0.1),
            mn.VGroup(
                mn.Text("max", font_size=20, color=mn.GREY_A),
                mn.DecimalNumber(rcm_error.max(), num_decimal_places=2, font_size=24, color=mn.ORANGE),
                mn.Text("mm", font_size=18, color=mn.GREY_A),
            ).arrange(mn.RIGHT, buff=0.1),
        ).arrange(mn.RIGHT, buff=0.45).to_edge(mn.DOWN)

        tracker = mn.ValueTracker(0.0)
        tip_dot = mn.always_redraw(
            lambda: mn.Dot(path_points[self._index(tracker.get_value(), DT, len(path_points))], color=mn.YELLOW)
        )
        speed_dot = mn.always_redraw(
            lambda: mn.Dot(
                speed_axes.c2p(
                    speed_time[self._index(tracker.get_value() - DT, DT, len(speed))],
                    speed[self._index(tracker.get_value() - DT, DT, len(speed))],
                ),
                color=mn.YELLOW,
                radius=0.055,
            )
        )
        jerk_dot = mn.always_redraw(
            lambda: mn.Dot(
                jerk_axes.c2p(
                    jerk_time[self._index(tracker.get_value() - 3 * DT, DT, len(jerk_norm))],
                    jerk_norm[self._index(tracker.get_value() - 3 * DT, DT, len(jerk_norm))],
                ),
                color=mn.YELLOW,
                radius=0.055,
            )
        )
        rcm_dot = mn.always_redraw(
            lambda: mn.Dot(
                rcm_axes.c2p(
                    time[self._index(tracker.get_value(), DT, len(rcm_error))],
                    rcm_error[self._index(tracker.get_value(), DT, len(rcm_error))],
                ),
                color=mn.YELLOW,
                radius=0.055,
            )
        )

        self.play(
            mn.FadeIn(title, path_label, speed_label, jerk_label, rcm_label, goal_dot, obstacle_shape, metric),
            mn.Create(path_axes),
            mn.Create(speed_axes),
            mn.Create(jerk_axes),
            mn.Create(rcm_axes),
            mn.Create(path),
            mn.Create(speed_graph),
            mn.Create(jerk_graph),
            mn.Create(rcm_graph),
            mn.FadeIn(high_jerk_dots, tip_dot, speed_dot, jerk_dot, rcm_dot),
            run_time=1.2,
        )
        self.play(tracker.animate.set_value(duration), run_time=8.0, rate_func=mn.linear)
        self.wait(0.8)

    @staticmethod
    def _range(values, step):
        return [
            step * np.floor((values.min() - step) / step),
            step * np.ceil((values.max() + step) / step),
            step,
        ]

    @staticmethod
    def _time_axes(duration, values, height, y_step):
        y_max = max(y_step, y_step * np.ceil(values.max() / y_step))
        return mn.Axes(
            x_range=[0.0, duration, 1.0],
            y_range=[0.0, y_max, y_max / 2.0],
            x_length=5.5,
            y_length=height,
            tips=False,
            axis_config={"color": mn.GREY_B, "stroke_width": 2, "include_numbers": False},
        )

    @staticmethod
    def _index(time_value, dt, length):
        return int(np.clip(round(time_value / dt), 0, length - 1))
