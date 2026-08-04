from datetime import datetime
from pathlib import Path

import manim as mn
import numpy as np
import torch
from scipy.interpolate import CubicSpline

ROOT_DIR = Path(__file__).resolve().parents[2]
TRACE_PATH = ROOT_DIR / "temp" / "dvrk_scp_mppi_trace.pt"
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
RUN_DIR = ROOT_DIR / "media" / "dvrk_scp_mppi_runs" / RUN_ID
mn.config.media_dir = str(RUN_DIR)
mn.config.video_dir = str(RUN_DIR)
mn.config.output_file = "dvrk_scp_mppi_trace.mp4"
NUM_ROLLOUTS = 256
FRAME_STRIDE = 4
DT = 0.02
NUM_CONTROL_POINTS = 4
CONTROL_LIMIT = torch.tensor([0.004, 0.004, 0.004, 0.06, 0.06, 0.06, 0.05])


class DvrkSMPPITraceScene(mn.ThreeDScene):
    def construct(self):
        trace = torch.load(TRACE_PATH, map_location="cpu", weights_only=True)
        steps = trace["steps"]
        goal = trace["goal"].numpy()
        actual = torch.stack([steps[0]["state"][:3], *[step["next_state"][:3] for step in steps]]).numpy()
        x_min, x_max, y_min, y_max, z_min, z_max = self._plot_range(steps, goal)

        axes = mn.ThreeDAxes(
            x_range=[x_min, x_max, 10],
            y_range=[y_min, y_max, 10],
            z_range=[z_min, z_max, 10],
            x_length=6.5,
            y_length=5.0,
            z_length=6.5,
            tips=False,
            axis_config={"color": mn.GREY_B, "stroke_width": 2},
        ).shift(2.7 * mn.LEFT + 0.3 * mn.DOWN)
        self.set_camera_orientation(phi=68 * mn.DEGREES, theta=-55 * mn.DEGREES, zoom=1.05)

        title = mn.Text("dVRK Constrained SCP-MPPI", font_size=34, weight="BOLD").to_edge(mn.UP)
        projection = mn.Text("ECM X-Y-Z trajectory [mm]", font_size=20, color=mn.GREY_A).move_to([-3.5, 2.95, 0])
        goal_dot = mn.Dot3D(axes.c2p(*(goal[:3] * 1000)), color=mn.GREEN, radius=0.08)
        rollouts = self._rollout_group(axes, steps[0])
        actual_path = self._path(axes, actual[:2], mn.WHITE, 4.0, 1.0)
        tip_dot = mn.Dot3D(axes.c2p(*(actual[1] * 1000)), color=mn.WHITE, radius=0.065)

        legend = (
            mn
            .VGroup(
                self._legend_item(mn.BLUE_B, "sampled rollouts"),
                self._legend_item(mn.YELLOW, "lowest cost"),
                self._legend_item(mn.WHITE, "executed tip"),
                self._legend_item(mn.GREEN, "goal"),
            )
            .arrange(mn.RIGHT, buff=0.28)
            .scale(0.72)
            .move_to([-3.2, -3.5, 0])
        )

        c1_y_max = self._continuity_y_max(steps, derivative_order=1)
        c2_y_max = self._continuity_y_max(steps, derivative_order=2)
        action_title = mn.Text("integrated action A", font_size=18).move_to([4.45, 2.7, 0])
        rate_title = mn.Text("control rate U = dA / dt", font_size=18).move_to([4.45, 1.45, 0])
        c1_title = mn.Text("C¹ spline continuity: 1st-derivative knot jump", font_size=14).move_to([4.45, 0.2, 0])
        c2_title = mn.Text("C² spline continuity: 2nd-derivative knot jump", font_size=14).move_to([4.45, -1.05, 0])
        action_axes = self._panel_axes([2.75, 1.85, 0], [6.15, 2.32, 0], "|dp| [mm]", 4.5)
        rate_axes = self._panel_axes([2.75, 0.6, 0], [6.15, 1.07, 0], "|Uₚ| [mm/s]", 220)
        c1_axes = self._panel_axes(
            [2.75, -0.65, 0],
            [6.15, -0.18, 0],
            "C¹ jump [mm/s]",
            c1_y_max,
            x_max=1,
            y_min=-0.1 * c1_y_max,
        )
        c2_axes = self._panel_axes(
            [2.75, -1.9, 0],
            [6.15, -1.43, 0],
            "C² jump [mm/s²]",
            c2_y_max,
            x_max=1,
            y_min=-0.1 * c2_y_max,
        )
        action_curve = self._action_curve(action_axes, steps[0])
        rate_curve = self._rate_curve(rate_axes, steps[0])
        c1_curve = self._continuity_curve(c1_axes, steps[0], derivative_order=1, color=mn.PURPLE_C)
        c2_curve = self._continuity_curve(c2_axes, steps[0], derivative_order=2, color=mn.RED_C)

        step_number = mn.Integer(0, font_size=22, mob_class=mn.Text)
        error_number = mn.DecimalNumber(
            self._tip_error(steps[0], goal),
            num_decimal_places=1,
            font_size=22,
            mob_class=mn.Text,
        )
        omega_number = mn.DecimalNumber(
            self._omega(steps[0]),
            num_decimal_places=2,
            font_size=22,
            mob_class=mn.Text,
        )
        stats = (
            mn
            .VGroup(
                mn.VGroup(mn.Text("step", font_size=18, color=mn.GREY_B), step_number).arrange(mn.RIGHT, buff=0.12),
                mn.VGroup(
                    mn.Text("tip error", font_size=18, color=mn.GREY_B), error_number, mn.Text("mm", font_size=18)
                ).arrange(mn.RIGHT, buff=0.1),
                mn.VGroup(mn.Text("Omega(A)", font_size=18, color=mn.GREY_B), omega_number).arrange(
                    mn.RIGHT, buff=0.12
                ),
            )
            .arrange(mn.DOWN, aligned_edge=mn.LEFT, buff=0.12)
            .move_to([4.15, -3.0, 0], aligned_edge=mn.LEFT)
        )

        self.camera.background_color = "#101318"
        self.add_fixed_in_frame_mobjects(
            title,
            projection,
            legend,
            action_title,
            rate_title,
            c1_title,
            c2_title,
            action_axes,
            rate_axes,
            c1_axes,
            c2_axes,
            action_curve,
            rate_curve,
            c1_curve,
            c2_curve,
            stats,
        )
        self.play(
            mn.FadeIn(title, projection, legend, goal_dot),
            mn.Create(axes),
            mn.FadeIn(
                rollouts,
                actual_path,
                tip_dot,
                action_title,
                rate_title,
                c1_title,
                c2_title,
                action_axes,
                rate_axes,
                c1_axes,
                c2_axes,
                action_curve,
                rate_curve,
                c1_curve,
                c2_curve,
                stats,
            ),
            run_time=0.8,
        )

        frame_ids = list(range(0, len(steps), FRAME_STRIDE))
        if frame_ids[-1] != len(steps) - 1:
            frame_ids.append(len(steps) - 1)
        for frame_id in frame_ids[1:]:
            step = steps[frame_id]
            step_number.set_value(frame_id)
            error_number.set_value(self._tip_error(step, goal))
            omega_number.set_value(self._omega(step))
            self.camera.add_fixed_in_frame_mobjects(step_number, error_number, omega_number)
            self.play(
                mn.Transform(rollouts, self._rollout_group(axes, step)),
                mn.Transform(actual_path, self._path(axes, actual[: frame_id + 2], mn.WHITE, 4.0, 1.0)),
                tip_dot.animate.move_to(axes.c2p(*(actual[frame_id + 1] * 1000))),
                mn.Transform(action_curve, self._action_curve(action_axes, step)),
                mn.Transform(rate_curve, self._rate_curve(rate_axes, step)),
                mn.Transform(c1_curve, self._continuity_curve(c1_axes, step, derivative_order=1, color=mn.PURPLE_C)),
                mn.Transform(c2_curve, self._continuity_curve(c2_axes, step, derivative_order=2, color=mn.RED_C)),
                run_time=0.12,
                rate_func=mn.linear,
            )
        self.wait(0.8)

    @staticmethod
    def _plot_range(steps, goal):
        positions = torch.cat([step["rollout_position"].reshape(-1, 3) for step in steps]).numpy() * 1000
        return tuple(
            bound
            for axis in range(3)
            for bound in (
                10 * np.floor((np.r_[positions[:, axis], goal[axis] * 1000].min() - 5) / 10),
                10 * np.ceil((np.r_[positions[:, axis], goal[axis] * 1000].max() + 5) / 10),
            )
        )

    @staticmethod
    def _path(axes, positions, color, width, opacity):
        points = [axes.c2p(*(position[:3] * 1000)) for position in positions]
        path = mn.VMobject()
        path.set_points_as_corners(points if len(points) > 1 else points * 2)
        return path.set_stroke(color, width=width, opacity=opacity)

    def _rollout_group(self, axes, step):
        indices = torch.topk(
            step["rollout_cost"], min(NUM_ROLLOUTS, step["rollout_cost"].numel()), largest=False
        ).indices
        return mn.VGroup(*[
            self._path(
                axes,
                step["rollout_position"][index].numpy(),
                mn.YELLOW if rank == 0 else mn.BLUE_B,
                3.0 if rank == 0 else 1.1,
                0.95 if rank == 0 else 0.38 * (1.0 - rank / len(indices)) + 0.08,
            )
            for rank, index in enumerate(indices)
        ])

    @staticmethod
    def _panel_axes(bottom_left, top_right, label, y_max, x_max=7, y_min=0.0):
        axes = mn.Axes(
            x_range=[0, x_max, 1],
            y_range=[y_min, y_max, (y_max - y_min) / 2],
            x_length=top_right[0] - bottom_left[0],
            y_length=top_right[1] - bottom_left[1],
            tips=False,
            axis_config={"color": mn.GREY_C, "stroke_width": 1.5, "include_ticks": False},
        ).move_to([(bottom_left[0] + top_right[0]) / 2, (bottom_left[1] + top_right[1]) / 2, 0])
        label_mobject = mn.Text(label, font_size=14, color=mn.GREY_B).next_to(axes, mn.LEFT, buff=0.08)
        return mn.VGroup(axes, label_mobject)

    @staticmethod
    def _action_curve(panel, step):
        values = torch.linalg.vector_norm(step["nominal_after"][:, :3], dim=-1).numpy() * 1000
        axes = panel[0]
        return axes.plot_line_graph(
            range(len(values)), np.clip(values, 0, 4.5), add_vertex_dots=False, line_color=mn.TEAL
        ).set_stroke(width=3)

    @staticmethod
    def _rate_curve(panel, step):
        values = torch.linalg.vector_norm(torch.diff(step["nominal_after"][:, :3], dim=0), dim=-1).numpy() * 1000 / DT
        axes = panel[0]
        if len(values) == 1:
            values = np.repeat(values, 2)
        return axes.plot_line_graph(
            range(len(values)), np.clip(values, 0, 220), add_vertex_dots=False, line_color=mn.ORANGE
        ).set_stroke(width=3)

    @staticmethod
    def _continuity_jumps(step, derivative_order):
        if "spline_nominal_after" not in step:
            raise ValueError(
                "This trace lacks the unclamped spline control sequence. "
                "Run dvrk_scp_mppi_demo.py again to create a continuity-compatible trace."
            )
        values = step["spline_nominal_after"][:, :3].numpy().astype(np.float64)
        support_indices = np.linspace(0, len(values) - 1, NUM_CONTROL_POINTS).round().astype(int)
        spline = CubicSpline(support_indices * DT, values[support_indices], axis=0)
        knots = support_indices[1:-1] * DT
        jumps = []
        for knot in knots:
            left = spline(np.nextafter(knot, -np.inf), nu=derivative_order)
            right = spline(np.nextafter(knot, np.inf), nu=derivative_order)
            jumps.append(np.linalg.norm(right - left) * 1000.0)
        return np.asarray(jumps)

    @classmethod
    def _continuity_y_max(cls, steps, derivative_order):
        maximum = max(
            (cls._continuity_jumps(step, derivative_order).max(initial=0.0) for step in steps),
            default=0.0,
        )
        return max(1.25 * maximum, 1e-9)

    @classmethod
    def _continuity_curve(cls, panel, step, derivative_order, color):
        values = cls._continuity_jumps(step, derivative_order)
        return (
            panel[0]
            .plot_line_graph(
                range(len(values)),
                values,
                add_vertex_dots=True,
                line_color=color,
            )
            .set_stroke(width=2.5)
        )

    @staticmethod
    def _omega(step):
        action_difference = torch.diff(step["nominal_after"], dim=0)
        return 0.1 * (action_difference / CONTROL_LIMIT).square().sum().item()

    @staticmethod
    def _tip_error(step, goal):
        return np.linalg.norm(step["next_state"][:3].numpy() - goal[:3]) * 1000

    @staticmethod
    def _legend_item(color, label):
        line = mn.Line(mn.LEFT * 0.15, mn.RIGHT * 0.15, color=color, stroke_width=4)
        return mn.VGroup(line, mn.Text(label, font_size=17, color=mn.GREY_A)).arrange(mn.RIGHT, buff=0.08)
