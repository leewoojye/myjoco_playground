from pathlib import Path

import manim as mn
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
TRACE_PATH = ROOT_DIR / "temp" / "dvrk_mppi_demo4_trace.pt"
NUM_ROLLOUTS = 48
FRAME_STRIDE = 2


class DvrkMPPITraceScene(mn.Scene):
    def construct(self):
        trace = torch.load(TRACE_PATH, map_location="cpu", weights_only=True)
        steps = trace["steps"]
        goal = trace["goal"].numpy()
        obstacle = trace["obstacle_position"].numpy()
        obstacle_radius = float(trace["obstacle_radius"])

        actual = torch.stack([steps[0]["state"][:3], *[step["next_state"][:3] for step in steps]]).numpy()
        x_min, x_max, z_min, z_max = self._plot_range(steps, goal, obstacle, obstacle_radius)
        axes = mn.Axes(
            x_range=[x_min, x_max, 10],
            y_range=[z_min, z_max, 10],
            x_length=7.8,
            y_length=5.8,
            tips=False,
            axis_config={"color": mn.GREY_B, "stroke_width": 2},
        ).shift(2.7 * mn.LEFT + 0.3 * mn.DOWN)

        title = mn.Text("dVRK MPPI / RBF-EKF", font_size=34, weight="BOLD").to_edge(mn.UP)
        projection = mn.Text("ECM X-Z projection [mm]", font_size=20, color=mn.GREY_A)
        projection.next_to(axes, mn.UP, buff=0.15)

        goal_dot = mn.Dot(axes.c2p(goal[0] * 1000, goal[2] * 1000), color=mn.GREEN, radius=0.08)
        obstacle_shape = self._obstacle_shape(axes, obstacle, obstacle_radius)
        rollouts = self._rollout_group(axes, steps[0])
        actual_path = self._path(axes, actual[:2], mn.WHITE, 4.0, 1.0)
        tip_dot = mn.Dot(axes.c2p(actual[1, 0] * 1000, actual[1, 2] * 1000), color=mn.WHITE, radius=0.065)

        legend = (
            mn
            .VGroup(
                self._legend_item(mn.BLUE_B, "sampled rollouts"),
                self._legend_item(mn.YELLOW, "lowest cost"),
                self._legend_item(mn.WHITE, "executed tip"),
                self._legend_item(mn.RED, "obstacle"),
                self._legend_item(mn.GREEN, "goal"),
            )
            .arrange(mn.RIGHT, buff=0.28)
            .scale(0.72)
        )
        legend.next_to(axes, mn.DOWN, buff=0.25)

        basis_title = mn.Text("RBF activation", font_size=23).move_to([4.45, 2.75, 0])
        basis_bar_bottom = 1.25
        basis_bars = self._basis_bars(steps[0], basis_bar_bottom)
        basis_labels = mn.VGroup(*[
            mn.Text(str(i), font_size=14, color=mn.GREY_B).move_to([2.75 + 0.38 * i, basis_bar_bottom - 0.18, 0])
            for i in range(10)
        ])
        basis_axis = mn.Line([2.55, basis_bar_bottom, 0], [6.35, basis_bar_bottom, 0], color=mn.GREY_C)

        heatmap_title = mn.Text("EKF weight update  |delta W|", font_size=23).move_to([4.45, 0.62, 0])
        max_update = max(
            torch.linalg.vector_norm(step["pose_weights_after"] - step["pose_weights_before"], dim=-1).max().item()
            for step in steps
        )
        heatmap = self._weight_heatmap(steps[0], max_update)
        row_labels = mn.VGroup(*[
            mn.Text(label, font_size=14, color=mn.GREY_B).move_to([2.4, 0.15 - 0.34 * row, 0])
            for row, label in enumerate(("x", "y", "z", "rx", "ry", "rz"))
        ])

        step_number = mn.Integer(0, font_size=22)
        error_number = mn.DecimalNumber(self._tip_error(steps[0], goal), num_decimal_places=1, font_size=22)
        cost_number = mn.DecimalNumber(steps[0]["rollout_cost"].min().item(), num_decimal_places=1, font_size=22)
        stats = mn.VGroup(
            mn.VGroup(mn.Text("step", font_size=18, color=mn.GREY_B), step_number).arrange(mn.RIGHT, buff=0.12),
            mn.VGroup(
                mn.Text("tip error", font_size=18, color=mn.GREY_B), error_number, mn.Text("mm", font_size=18)
            ).arrange(mn.RIGHT, buff=0.1),
            mn.VGroup(mn.Text("min cost", font_size=18, color=mn.GREY_B), cost_number).arrange(mn.RIGHT, buff=0.12),
        ).arrange(mn.DOWN, aligned_edge=mn.LEFT, buff=0.12)
        stats.move_to([4.15, -2.85, 0], aligned_edge=mn.LEFT)

        self.camera.background_color = "#101318"
        self.play(
            mn.FadeIn(title, projection, legend, goal_dot, obstacle_shape),
            mn.Create(axes),
            mn.FadeIn(
                rollouts,
                actual_path,
                tip_dot,
                basis_title,
                basis_axis,
                basis_bars,
                basis_labels,
                heatmap_title,
                heatmap,
                row_labels,
                stats,
            ),
            run_time=0.8,
        )

        frame_ids = list(range(0, len(steps), FRAME_STRIDE))
        if frame_ids[-1] != len(steps) - 1:
            frame_ids.append(len(steps) - 1)

        for frame_id in frame_ids[1:]:
            step = steps[frame_id]
            self.play(
                mn.Transform(rollouts, self._rollout_group(axes, step)),
                mn.Transform(actual_path, self._path(axes, actual[: frame_id + 2], mn.WHITE, 4.0, 1.0)),
                tip_dot.animate.move_to(axes.c2p(actual[frame_id + 1, 0] * 1000, actual[frame_id + 1, 2] * 1000)),
                mn.Transform(basis_bars, self._basis_bars(step, basis_bar_bottom)),
                mn.Transform(heatmap, self._weight_heatmap(step, max_update)),
                mn.ChangeDecimalToValue(step_number, frame_id),
                mn.ChangeDecimalToValue(error_number, self._tip_error(step, goal)),
                mn.ChangeDecimalToValue(cost_number, step["rollout_cost"].min().item()),
                run_time=0.12,
                rate_func=mn.linear,
            )

        self.wait(0.8)

    @staticmethod
    def _plot_range(steps, goal, obstacle, obstacle_radius):
        positions = torch.cat([step["rollout_position"][..., (0, 2)].reshape(-1, 2) for step in steps]).numpy() * 1000
        obstacle_xz = obstacle[[0, 2]] * 1000
        goal_xz = goal[[0, 2]] * 1000
        radius = obstacle_radius * 1000
        x_values = np.r_[positions[:, 0], obstacle_xz[0] - radius, obstacle_xz[0] + radius, goal_xz[0]]
        z_values = np.r_[positions[:, 1], obstacle_xz[1] - radius, obstacle_xz[1] + radius, goal_xz[1]]
        x_min = 10 * np.floor((x_values.min() - 5) / 10)
        x_max = 10 * np.ceil((x_values.max() + 5) / 10)
        z_min = 10 * np.floor((z_values.min() - 5) / 10)
        z_max = 10 * np.ceil((z_values.max() + 5) / 10)
        return x_min, x_max, z_min, z_max

    @staticmethod
    def _path(axes, positions, color, width, opacity):
        points = [axes.c2p(position[0] * 1000, position[2] * 1000) for position in positions]
        if len(points) == 1:
            points.append(points[0])
        path = mn.VMobject()
        path.set_points_as_corners(points)
        path.set_stroke(color, width=width, opacity=opacity)
        return path

    def _rollout_group(self, axes, step):
        costs = step["rollout_cost"]
        indices = torch.topk(costs, min(NUM_ROLLOUTS, costs.numel()), largest=False).indices
        paths = []
        for rank, index in enumerate(indices):
            color = mn.YELLOW if rank == 0 else mn.BLUE_B
            width = 3.0 if rank == 0 else 1.1
            opacity = 0.95 if rank == 0 else 0.38 * (1.0 - rank / len(indices)) + 0.08
            paths.append(self._path(axes, step["rollout_position"][index].numpy(), color, width, opacity))
        return mn.VGroup(*paths)

    @staticmethod
    def _obstacle_shape(axes, obstacle, radius):
        center = axes.c2p(obstacle[0] * 1000, obstacle[2] * 1000)
        x_radius = np.linalg.norm(axes.c2p((obstacle[0] + radius) * 1000, obstacle[2] * 1000) - center)
        z_radius = np.linalg.norm(axes.c2p(obstacle[0] * 1000, (obstacle[2] + radius) * 1000) - center)
        return mn.Ellipse(
            width=2 * x_radius,
            height=2 * z_radius,
            color=mn.RED,
            fill_color=mn.RED,
            fill_opacity=0.2,
            stroke_width=2,
        ).move_to(center)

    @staticmethod
    def _basis_bars(step, bottom):
        bars = []
        for index, value in enumerate(step["pose_basis"]):
            height = max(0.02, 1.2 * value.item())
            bar = mn.Rectangle(width=0.25, height=height, stroke_width=0, fill_color=mn.TEAL, fill_opacity=0.9)
            bar.move_to([2.75 + 0.38 * index, bottom + height / 2, 0])
            bars.append(bar)
        return mn.VGroup(*bars)

    @staticmethod
    def _weight_heatmap(step, max_update):
        update = torch.linalg.vector_norm(step["pose_weights_after"] - step["pose_weights_before"], dim=-1)
        cells = []
        for row in range(6):
            for column in range(10):
                alpha = min(1.0, update[row, column].item() / max(max_update, 1e-8))
                cell = mn.Square(side_length=0.28, stroke_color=mn.GREY_D, stroke_width=0.5)
                cell.set_fill(mn.interpolate_color(mn.GREY_E, mn.ORANGE, alpha), opacity=1.0)
                cell.move_to([2.75 + 0.34 * column, 0.15 - 0.34 * row, 0])
                cells.append(cell)
        return mn.VGroup(*cells)

    @staticmethod
    def _legend_item(color, label):
        line = mn.Line(mn.LEFT * 0.15, mn.RIGHT * 0.15, color=color, stroke_width=4)
        text = mn.Text(label, font_size=17, color=mn.GREY_A)
        return mn.VGroup(line, text).arrange(mn.RIGHT, buff=0.08)

    @staticmethod
    def _tip_error(step, goal):
        return np.linalg.norm(step["next_state"][:3].numpy() - goal[:3]) * 1000
