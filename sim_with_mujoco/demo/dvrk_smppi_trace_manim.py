from pathlib import Path

import manim as mn
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
TRACE_PATH = ROOT_DIR / "temp" / "dvrk_mppi_demo4_trace.pt"
NUM_ROLLOUTS = 20
FRAME_STRIDE = 4
DT = 0.02
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

        title = mn.Text("dVRK SMPPI", font_size=34, weight="BOLD").to_edge(mn.UP)
        projection = mn.Text("ECM X-Y-Z trajectory [mm]", font_size=20, color=mn.GREY_A).move_to([-3.5, 2.95, 0])
        goal_dot = mn.Dot3D(axes.c2p(*(goal[:3] * 1000)), color=mn.GREEN, radius=0.08)
        rollouts = self._rollout_group(axes, steps[0])
        actual_path = self._path(axes, actual[:2], mn.WHITE, 4.0, 1.0)
        tip_dot = mn.Dot3D(axes.c2p(*(actual[1] * 1000)), color=mn.WHITE, radius=0.065)

        legend = mn.VGroup(
            self._legend_item(mn.BLUE_B, "sampled rollouts"),
            self._legend_item(mn.YELLOW, "lowest cost"),
            self._legend_item(mn.WHITE, "executed tip"),
            self._legend_item(mn.GREEN, "goal"),
        ).arrange(mn.RIGHT, buff=0.28).scale(0.72).move_to([-3.2, -3.5, 0])

        action_title = mn.Text("integrated action A", font_size=23).move_to([4.4, 2.75, 0])
        rate_title = mn.Text("control rate U = dA / dt", font_size=23).move_to([4.4, 1.15, 0])
        action_axes = self._panel_axes([2.55, 1.5, 0], [6.3, 2.35, 0], "|dp| [mm]", 4.5)
        rate_axes = self._panel_axes([2.55, -0.1, 0], [6.3, 0.75, 0], "|U_p| [mm/s]", 220)
        action_curve = self._action_curve(action_axes, steps[0])
        rate_curve = self._rate_curve(rate_axes, steps[0])

        step_number = mn.Integer(0, font_size=22)
        error_number = mn.DecimalNumber(self._tip_error(steps[0], goal), num_decimal_places=1, font_size=22)
        omega_number = mn.DecimalNumber(self._omega(steps[0]), num_decimal_places=2, font_size=22)
        stats = mn.VGroup(
            mn.VGroup(mn.Text("step", font_size=18, color=mn.GREY_B), step_number).arrange(mn.RIGHT, buff=0.12),
            mn.VGroup(mn.Text("tip error", font_size=18, color=mn.GREY_B), error_number, mn.Text("mm", font_size=18)).arrange(mn.RIGHT, buff=0.1),
            mn.VGroup(mn.Text("Omega(A)", font_size=18, color=mn.GREY_B), omega_number).arrange(mn.RIGHT, buff=0.12),
        ).arrange(mn.DOWN, aligned_edge=mn.LEFT, buff=0.12).move_to([4.15, -2.45, 0], aligned_edge=mn.LEFT)

        self.camera.background_color = "#101318"
        self.add_fixed_in_frame_mobjects(
            title,
            projection,
            legend,
            action_title,
            rate_title,
            action_axes,
            rate_axes,
            action_curve,
            rate_curve,
            stats,
        )
        self.play(
            mn.FadeIn(title, projection, legend, goal_dot),
            mn.Create(axes),
            mn.FadeIn(rollouts, actual_path, tip_dot, action_title, rate_title, action_axes, rate_axes, action_curve, rate_curve, stats),
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
        indices = torch.topk(step["rollout_cost"], min(NUM_ROLLOUTS, step["rollout_cost"].numel()), largest=False).indices
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
    def _panel_axes(bottom_left, top_right, label, y_max):
        axes = mn.Axes(
            x_range=[0, 7, 1],
            y_range=[0, y_max, y_max / 2],
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
        return axes.plot_line_graph(range(len(values)), np.clip(values, 0, 4.5), add_vertex_dots=False, line_color=mn.TEAL).set_stroke(width=3)

    @staticmethod
    def _rate_curve(panel, step):
        values = torch.linalg.vector_norm(torch.diff(step["nominal_after"][:, :3], dim=0), dim=-1).numpy() * 1000 / DT
        axes = panel[0]
        if len(values) == 1:
            values = np.repeat(values, 2)
        return axes.plot_line_graph(range(len(values)), np.clip(values, 0, 220), add_vertex_dots=False, line_color=mn.ORANGE).set_stroke(width=3)

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
