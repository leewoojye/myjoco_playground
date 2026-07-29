import sys
from pathlib import Path

import manim as mn
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[3]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sim_with_mujoco.rl.utils.metrics import (  # noqa: E402
    DT,
    log_dimensionless_jerk,
    rcm_deviation,
    trace_states,
    trajectory_kinematics,
)


class LogDimensionlessJerkScene(mn.Scene):
    def construct(self):
        trace, states = trace_states()
        positions = states[:, :3]
        _, jerk = trajectory_kinematics(positions)
        positions = positions.numpy()
        goal = trace["goal"].numpy()
        tip_error = np.linalg.norm(positions - goal[:3], axis=-1) * 1000.0
        jerk_norm = torch.linalg.vector_norm(jerk, dim=-1).numpy()
        integrated_squared_jerk = np.cumsum(np.square(jerk_norm)) * DT
        rcm_error = (rcm_deviation() * 1000.0).numpy()
        time = np.arange(len(positions)) * DT
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

        path_points = [
            path_axes.c2p(position[0] * 1000.0, position[2] * 1000.0) for position in positions
        ]
        path = mn.VMobject().set_points_as_corners(path_points).set_stroke(mn.WHITE, width=3)
        goal_dot = mn.Dot(path_axes.c2p(goal[0] * 1000.0, goal[2] * 1000.0), color=mn.GREEN, radius=0.07)

        high_jerk = np.flatnonzero(jerk_norm >= np.quantile(jerk_norm, 0.9)) + 3
        high_jerk_dots = mn.VGroup(*[
            mn.Dot(path_points[index], color=mn.RED, radius=0.035) for index in high_jerk
        ])

        duration = time[-1]
        error_axes = self._time_axes(duration, tip_error, 1.0, 20.0).shift(3.4 * mn.RIGHT + 1.9 * mn.UP)
        integrated_jerk_axes = self._time_axes(
            duration,
            integrated_squared_jerk,
            1.0,
            max(float(integrated_squared_jerk[-1]) / 4.0, np.finfo(np.float32).eps),
        ).shift(3.4 * mn.RIGHT + 0.05 * mn.UP)
        rcm_axes = self._time_axes(duration, rcm_error, 1.0, 0.25).shift(3.4 * mn.RIGHT + 1.8 * mn.DOWN)
        error_graph = error_axes.plot_line_graph(
            time,
            tip_error,
            add_vertex_dots=False,
            line_color=mn.BLUE_B,
            stroke_width=3,
        )
        integrated_jerk_graph = integrated_jerk_axes.plot_line_graph(
            jerk_time,
            integrated_squared_jerk,
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
        error_label = mn.Text("tip error [mm]", font_size=19, color=mn.BLUE_B).next_to(
            error_axes, mn.UP, buff=0.12
        )
        integrated_jerk_label = mn.Text("integrated squared jerk", font_size=19, color=mn.RED).next_to(
            integrated_jerk_axes, mn.UP, buff=0.12
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
        error_dot = mn.always_redraw(
            lambda: mn.Dot(
                error_axes.c2p(
                    time[self._index(tracker.get_value(), DT, len(tip_error))],
                    tip_error[self._index(tracker.get_value(), DT, len(tip_error))],
                ),
                color=mn.YELLOW,
                radius=0.055,
            )
        )
        integrated_jerk_dot = mn.always_redraw(
            lambda: mn.Dot(
                integrated_jerk_axes.c2p(
                    jerk_time[self._index(tracker.get_value() - 3 * DT, DT, len(jerk_norm))],
                    integrated_squared_jerk[self._index(tracker.get_value() - 3 * DT, DT, len(jerk_norm))],
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
            mn.FadeIn(title, path_label, error_label, integrated_jerk_label, rcm_label, goal_dot, metric),
            mn.Create(path_axes),
            mn.Create(error_axes),
            mn.Create(integrated_jerk_axes),
            mn.Create(rcm_axes),
            mn.Create(path),
            mn.Create(error_graph),
            mn.Create(integrated_jerk_graph),
            mn.Create(rcm_graph),
            mn.FadeIn(high_jerk_dots, tip_dot, error_dot, integrated_jerk_dot, rcm_dot),
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
