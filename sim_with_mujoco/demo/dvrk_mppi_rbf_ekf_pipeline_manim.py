from pathlib import Path

import manim as mn
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
TRACE_PATH = ROOT_DIR / "temp" / "dvrk_mppi_demo4_trace.pt"
TRACE_STEP = 34
NUM_ROLLOUTS = 42

BG = "#101318"
PANEL = "#171C23"
CYAN = "#58D6D6"
BLUE = "#4F8EC9"
YELLOW = "#F2CC60"
ORANGE = "#E6904E"
GREEN = "#73C991"
RED = "#E05A5A"
MUTED = "#8B949E"


class DvrkMPPIRBFEKFPipelineScene(mn.Scene):
    def construct(self):
        self.camera.background_color = BG
        trace = torch.load(TRACE_PATH, map_location="cpu", weights_only=True)
        step = trace["steps"][min(TRACE_STEP, len(trace["steps"]) - 1)]
        goal = trace["goal"].numpy()
        obstacle = trace["obstacle_position"].numpy()
        best_index = int(step["rollout_cost"].argmin())
        indices = self._rollout_indices(step["rollout_cost"])

        layout = self._layout()
        flows = self._flow_paths(layout)
        mppi = self._mppi_contents(step, goal, obstacle, indices, best_index)
        rbf = self._rbf_contents(step)
        plant = self._plant_contents()
        ekf = self._ekf_contents(step)

        self.play(
            mn.FadeIn(layout["title"]),
            mn.LaggedStart(
                *[mn.FadeIn(layout[name]) for name in ("mppi", "rbf", "plant", "ekf")],
                lag_ratio=0.1,
            ),
            mn.LaggedStart(*[mn.Create(path) for path in flows.values()], lag_ratio=0.08),
            run_time=1.4,
        )
        self.play(
            mn.FadeIn(mppi["static"]),
            mn.FadeIn(rbf["static"]),
            mn.FadeIn(plant["static"]),
            mn.FadeIn(ekf["static"]),
            run_time=0.8,
        )

        self._animate_state_input(layout, flows, mppi, plant)
        self._animate_sampling(layout, mppi)
        self._animate_rbf_rollout(layout, flows, mppi, rbf)
        self._animate_cost_and_action(layout, flows, mppi, plant)
        self._animate_plant_observation(layout, flows, plant, ekf)
        self._animate_ekf_update(layout, flows, rbf, ekf)
        self._animate_next_cycle(layout, flows, mppi, plant)
        self.wait(1.0)

    def _layout(self):
        title = mn.Text("dVRK MPPI / RBF-EKF", font_size=28, weight="BOLD")
        title.to_edge(mn.UP, buff=0.16)

        mppi = self._panel("MPPI", YELLOW, 6.7, 6.6).move_to([-3.55, -0.25, 0])
        rbf = self._panel("RBF", ORANGE, 6.15, 3.72).move_to([3.55, 1.2, 0])
        plant = self._panel("PLANT", CYAN, 3.25, 2.45).move_to([2.1, -2.28, 0])
        ekf = self._panel("EKF", GREEN, 2.7, 2.45).move_to([5.28, -2.28, 0])
        return {"title": title, "mppi": mppi, "rbf": rbf, "plant": plant, "ekf": ekf}

    def _flow_paths(self, layout):
        mppi_box, rbf_box = layout["mppi"][0], layout["rbf"][0]
        plant_box, ekf_box = layout["plant"][0], layout["ekf"][0]
        return {
            "candidate": self._arrow(
                mppi_box.get_right() + 2.05 * mn.UP,
                rbf_box.get_left() + 1.0 * mn.UP,
                ORANGE,
            ),
            "rollout": self._arrow(
                rbf_box.get_left() + 0.15 * mn.UP,
                mppi_box.get_right() + 1.1 * mn.UP,
                CYAN,
            ),
            "action": self._arrow(
                mppi_box.get_right() + 1.55 * mn.DOWN,
                plant_box.get_left() + 0.5 * mn.UP,
                YELLOW,
            ),
            "observe": self._arrow(
                plant_box.get_right(),
                ekf_box.get_left(),
                RED,
            ),
            "update": mn.CurvedArrow(
                ekf_box.get_top() + 0.2 * mn.LEFT,
                rbf_box.get_bottom() + 1.75 * mn.RIGHT,
                angle=0.25,
                color=GREEN,
                stroke_width=1.5,
                tip_length=0.12,
            ).set_opacity(0.42),
            "state": mn.CurvedArrow(
                plant_box.get_left() + 0.55 * mn.DOWN,
                mppi_box.get_right() + 2.4 * mn.DOWN,
                angle=-0.35,
                color=CYAN,
                stroke_width=1.5,
                tip_length=0.12,
            ).set_opacity(0.42),
        }

    def _mppi_contents(self, step, goal, obstacle, indices, best_index):
        panel_center = np.array([-3.55, -0.25, 0])
        projected, project = self._project_rollouts(
            step["rollout_position"][indices].numpy(),
            center=np.array([-3.75, 0.35, 0]),
            width=5.65,
            height=3.75,
        )
        paths = mn.VGroup(*[
            self._screen_path(points, CYAN, 1.05, 0.2)
            for points in projected
        ])
        chosen_rank = int(np.flatnonzero(indices == best_index)[0])
        chosen = self._screen_path(projected[chosen_rank], YELLOW, 3.6, 1.0)

        start = mn.Dot(projected[0, 0], color=CYAN, radius=0.055)
        goal_dot = mn.Dot(project(goal[:3]), color=GREEN, radius=0.07)
        goal_ring = mn.Circle(radius=0.15, color=GREEN, stroke_width=1.2).move_to(goal_dot)
        obstacle_dot = project(obstacle)
        obstacle_shape = mn.Circle(
            radius=0.38,
            stroke_color=RED,
            stroke_width=1.2,
            fill_color=RED,
            fill_opacity=0.12,
        ).move_to(obstacle_dot)
        obstacle_cross = mn.VGroup(
            mn.Line(obstacle_dot + [-0.25, -0.25, 0], obstacle_dot + [0.25, 0.25, 0], color=RED),
            mn.Line(obstacle_dot + [-0.25, 0.25, 0], obstacle_dot + [0.25, -0.25, 0], color=RED),
        ).set_stroke(width=1.1, opacity=0.55)

        costs = step["rollout_cost"][indices].numpy()
        bars, bars_selected = self._cost_bars(costs, best_rank=chosen_rank)
        bars.move_to([-3.65, -2.75, 0])
        bars_selected.move_to(bars)
        cost_label = self._label("J_k", YELLOW, 14).next_to(bars, mn.LEFT, buff=0.16)
        weight_label = self._label("w_k", MUTED, 12).next_to(bars, mn.RIGHT, buff=0.16)
        shape = self._label("U [512,25,7]", MUTED, 13).move_to(panel_center + [-2.15, 2.55, 0])
        static = mn.VGroup(shape, start, goal_dot, goal_ring, obstacle_shape, obstacle_cross, cost_label, weight_label)
        return {
            "static": static,
            "paths": paths,
            "chosen": chosen,
            "bars": bars,
            "bars_selected": bars_selected,
            "start": start,
            "goal": goal_dot,
            "chosen_points": projected[chosen_rank],
        }

    def _rbf_contents(self, step):
        basis = step["pose_basis"].numpy()
        active = set(np.argsort(basis)[-5:])
        axes = mn.Axes(
            x_range=[-3.2, 3.2, 1],
            y_range=[0, 1.05, 0.5],
            x_length=5.05,
            y_length=1.45,
            tips=False,
            axis_config={"color": mn.GREY_C, "stroke_width": 1.0},
        ).move_to([3.2, 1.48, 0])
        curves = mn.VGroup()
        activation_dots = mn.VGroup()
        sigma = 0.72
        for index, value in enumerate(np.clip(basis, 1e-5, 1.0)):
            distance = min(3.1, sigma * np.sqrt(-2.0 * np.log(value)))
            center = distance * (-1 if index % 2 else 1)
            color = YELLOW if index == int(np.argmax(basis)) else (CYAN if index in active else BLUE)
            curve = axes.plot(
                lambda x, c=center: np.exp(-0.5 * ((x - c) / sigma) ** 2),
                x_range=[-3.2, 3.2, 0.08],
                color=color,
                stroke_width=2.0 if index in active else 0.8,
            )
            curve.set_opacity(0.9 if index in active else 0.13)
            curves.add(curve)
            activation_dots.add(
                mn.Dot(
                    axes.c2p(0, value),
                    radius=0.036 if index in active else 0.022,
                    color=color,
                ).set_opacity(1.0 if index in active else 0.25)
            )
        probe = mn.DashedLine(
            axes.c2p(0, 0),
            axes.c2p(0, 1.02),
            color=mn.WHITE,
            stroke_width=1.2,
            dash_length=0.05,
        )
        input_label = self._label("z=[x_pose,u_pose] [12]", MUTED, 12).move_to([1.7, 2.3, 0])
        basis_label = self._label("phi_1 ... phi_20", CYAN, 12).move_to([4.55, 2.3, 0])

        nodes = mn.VGroup()
        for index, value in enumerate(basis):
            position = np.array([1.05 + 0.42 * (index % 10), 0.2 - 0.38 * (index // 10), 0])
            nodes.add(
                mn.Circle(
                    radius=0.038 + 0.055 * float(value),
                    stroke_color=CYAN if index in active else mn.GREY_C,
                    stroke_width=0.8,
                    fill_color=YELLOW if index == int(np.argmax(basis)) else CYAN,
                    fill_opacity=0.15 + 0.8 * float(value),
                ).move_to(position)
            )
        sum_node = mn.Circle(radius=0.22, color=ORANGE, stroke_width=1.5).move_to([5.75, 0.02, 0])
        sum_text = self._label("SUM", ORANGE, 10).move_to(sum_node)
        before_strength = self._basis_weight_strength(step["pose_weights_before"], basis)
        after_strength = self._updated_strength(step, basis, before_strength)
        connections = self._weighted_connections(nodes, sum_node, before_strength, active, ORANGE)
        updated_connections = self._weighted_connections(nodes, sum_node, after_strength, active, GREEN)
        output = self._arrow(sum_node.get_right(), np.array([6.45, 0.02, 0]), ORANGE)
        output_label = self._label("delta x_hat", ORANGE, 11).move_to([6.0, -0.34, 0])
        weighted_label = self._label("W_i phi_i", ORANGE, 12).move_to([4.7, 0.44, 0])
        static = mn.VGroup(axes, input_label, basis_label, nodes, sum_node, sum_text, output, output_label, weighted_label)
        return {
            "static": static,
            "curves": curves,
            "probe": probe,
            "activation_dots": activation_dots,
            "connections": connections,
            "updated_connections": updated_connections,
            "sum": sum_node,
            "output": output,
        }

    def _plant_contents(self):
        base = np.array([2.0, -2.5, 0])
        current = self._instrument(base, CYAN)
        predicted = mn.Dot(base + [0.62, 0.22, 0], radius=0.055, color=ORANGE).set_opacity(0.65)
        observed_position = base + [0.72, -0.02, 0]
        observed = mn.Dot(observed_position, radius=0.065, color=mn.WHITE)
        target = mn.Dot(base + [0.9, 0.04, 0], radius=0.035, color=GREEN)
        pred_label = self._label("x_hat", ORANGE, 10).next_to(predicted, mn.UP, buff=0.05)
        obs_label = self._label("x", CYAN, 10).next_to(observed, mn.DOWN, buff=0.05)
        static = mn.VGroup(current, target)
        return {
            "static": static,
            "current": current,
            "predicted": predicted,
            "observed": observed,
            "pred_label": pred_label,
            "obs_label": obs_label,
            "observed_position": observed_position,
        }

    def _ekf_contents(self, step):
        before = self._weight_matrix(step, after=False).move_to([5.28, -2.35, 0])
        after = self._weight_matrix(step, after=True).move_to(before)
        label = self._label("W", MUTED, 12).move_to([4.25, -1.55, 0])
        label_after = self._label("W+", GREEN, 12).move_to(label)
        innovation = self._label("e [7]", RED, 11).move_to([4.2, -2.9, 0])
        static = mn.VGroup(before, label, innovation)
        return {
            "static": static,
            "before": before,
            "after": after,
            "label": label,
            "label_after": label_after,
            "innovation": innovation,
        }

    def _animate_state_input(self, layout, flows, mppi, plant):
        self.play(mn.Indicate(layout["plant"][0], color=CYAN, scale_factor=1.01), run_time=0.45)
        token = self._packet("x_t [7]", CYAN).move_to(flows["state"].get_start())
        self.play(mn.FadeIn(token), run_time=0.2)
        self.play(mn.MoveAlongPath(token, flows["state"]), run_time=0.85, rate_func=mn.linear)
        self.play(
            mn.FadeOut(token),
            mn.Indicate(mppi["start"], color=CYAN, scale_factor=1.8),
            mn.Indicate(layout["mppi"][0], color=YELLOW, scale_factor=1.005),
            run_time=0.5,
        )

    def _animate_sampling(self, layout, mppi):
        self.play(
            mn.LaggedStart(*[mn.Create(path) for path in mppi["paths"]], lag_ratio=0.018),
            run_time=1.65,
        )
        fan = mn.VGroup(*[
            mn.Line(
                mppi["start"].get_center(),
                mppi["start"].get_center() + [0.28 + 0.04 * index, -0.23 + 0.055 * index, 0],
                color=CYAN,
                stroke_width=1.0,
            ).set_opacity(0.35)
            for index in range(9)
        ])
        self.play(mn.LaggedStart(*[mn.Create(line) for line in fan], lag_ratio=0.04), run_time=0.55)
        self.play(mn.FadeOut(fan), run_time=0.25)
        self.play(mn.Indicate(layout["mppi"][1], color=YELLOW, scale_factor=1.08), run_time=0.35)

    def _animate_rbf_rollout(self, layout, flows, mppi, rbf):
        candidate = self._packet("U [512,25,7]", ORANGE).move_to(flows["candidate"].get_start())
        self.play(mn.FadeIn(candidate), run_time=0.2)
        self.play(mn.MoveAlongPath(candidate, flows["candidate"]), run_time=0.9, rate_func=mn.linear)
        self.play(
            mn.FadeOut(candidate),
            mn.Indicate(layout["rbf"][0], color=ORANGE, scale_factor=1.005),
            run_time=0.35,
        )
        self.play(
            mn.LaggedStart(*[mn.Create(curve) for curve in rbf["curves"]], lag_ratio=0.02),
            mn.Create(rbf["probe"]),
            run_time=1.35,
        )
        self.play(
            mn.LaggedStart(*[mn.GrowFromCenter(dot) for dot in rbf["activation_dots"]], lag_ratio=0.018),
            run_time=0.55,
        )
        self.play(
            mn.LaggedStart(*[mn.Create(line) for line in rbf["connections"]], lag_ratio=0.012),
            mn.Indicate(rbf["sum"], color=YELLOW, scale_factor=1.18),
            run_time=0.85,
        )
        rollout = self._packet("x_hat [512,26,7]", CYAN).move_to(flows["rollout"].get_start())
        self.play(mn.FadeIn(rollout), run_time=0.2)
        self.play(
            mn.MoveAlongPath(rollout, flows["rollout"]),
            mn.ShowPassingFlash(mppi["chosen"].copy().set_stroke(CYAN, width=5), time_width=0.35),
            run_time=0.9,
            rate_func=mn.linear,
        )
        self.play(mn.FadeOut(rollout), run_time=0.2)

    def _animate_cost_and_action(self, layout, flows, mppi, plant):
        self.play(mn.FadeIn(mppi["bars"], shift=0.12 * mn.UP), run_time=0.55)
        self.play(
            mn.Transform(mppi["bars"], mppi["bars_selected"]),
            *[path.animate.set_opacity(0.035) for path in mppi["paths"]],
            mn.Create(mppi["chosen"]),
            run_time=1.0,
        )
        pulse = mn.Dot(mppi["chosen_points"][0], radius=0.05, color=YELLOW)
        self.play(mn.MoveAlongPath(pulse, mppi["chosen"]), run_time=0.75, rate_func=mn.linear)
        first = self._screen_path(mppi["chosen_points"][:2], YELLOW, 6.0, 1.0)
        self.play(mn.ShowPassingFlash(first, time_width=0.7), run_time=0.45)
        action = self._packet("u_0 [7]", YELLOW).move_to(flows["action"].get_start())
        self.play(mn.FadeIn(action), run_time=0.2)
        self.play(mn.MoveAlongPath(action, flows["action"]), run_time=0.8, rate_func=mn.linear)
        self.play(
            mn.FadeOut(action, pulse),
            mn.Indicate(layout["plant"][0], color=CYAN, scale_factor=1.01),
            run_time=0.4,
        )

    def _animate_plant_observation(self, layout, flows, plant, ekf):
        moved_tool = self._instrument(plant["observed_position"], CYAN)
        self.play(
            mn.Transform(plant["current"], moved_tool),
            mn.FadeIn(plant["predicted"], plant["pred_label"]),
            run_time=0.7,
        )
        self.play(
            mn.GrowFromCenter(plant["observed"]),
            mn.FadeIn(plant["obs_label"]),
            run_time=0.4,
        )
        error = mn.Arrow(
            plant["predicted"].get_center(),
            plant["observed"].get_center(),
            buff=0.03,
            color=RED,
            stroke_width=1.6,
            max_tip_length_to_length_ratio=0.18,
        )
        self.play(mn.Create(error), run_time=0.45)
        observation = self._packet("e [7]", RED).move_to(flows["observe"].get_start())
        self.play(mn.FadeIn(observation), run_time=0.2)
        self.play(mn.MoveAlongPath(observation, flows["observe"]), run_time=0.65, rate_func=mn.linear)
        self.play(
            mn.FadeOut(observation),
            mn.Indicate(layout["ekf"][0], color=GREEN, scale_factor=1.01),
            run_time=0.35,
        )
        plant["error"] = error

    def _animate_ekf_update(self, layout, flows, rbf, ekf):
        self.play(
            mn.Transform(ekf["before"], ekf["after"]),
            mn.Transform(ekf["label"], ekf["label_after"]),
            run_time=0.95,
        )
        update = self._packet("W+ [6,20,6]", GREEN).move_to(flows["update"].get_start())
        self.play(mn.FadeIn(update), run_time=0.2)
        self.play(mn.MoveAlongPath(update, flows["update"]), run_time=0.8, rate_func=mn.linear)
        self.play(
            mn.FadeOut(update),
            mn.Transform(rbf["connections"], rbf["updated_connections"]),
            mn.Indicate(layout["rbf"][0], color=GREEN, scale_factor=1.005),
            run_time=0.85,
        )
        self.play(
            mn.ShowPassingFlash(rbf["output"].copy().set_color(GREEN), time_width=0.7),
            run_time=0.55,
        )

    def _animate_next_cycle(self, layout, flows, mppi, plant):
        state = self._packet("x_(t+1) [7]", CYAN).move_to(flows["state"].get_start())
        self.play(mn.FadeIn(state), run_time=0.2)
        self.play(mn.MoveAlongPath(state, flows["state"]), run_time=0.85, rate_func=mn.linear)
        self.play(
            mn.FadeOut(state),
            mn.Indicate(mppi["start"], color=CYAN, scale_factor=1.7),
            mn.Indicate(layout["mppi"][0], color=YELLOW, scale_factor=1.005),
            run_time=0.5,
        )
        loop = mn.VGroup(*flows.values())
        self.play(mn.ShowPassingFlash(loop.copy().set_stroke(width=3.0, opacity=1.0), time_width=0.22), run_time=1.1)

    @staticmethod
    def _panel(label, color, width, height):
        box = mn.RoundedRectangle(
            width=width,
            height=height,
            corner_radius=0.08,
            stroke_color=color,
            stroke_width=1.5,
            fill_color=PANEL,
            fill_opacity=0.3,
        )
        text = mn.Text(label, font_size=20, weight="BOLD", color=color)
        text.move_to(box.get_top() + 0.28 * mn.DOWN + (width / 2 - text.width / 2 - 0.25) * mn.LEFT)
        rule = mn.Line(
            box.get_left() + [0.18, height / 2 - 0.55, 0],
            box.get_right() + [-0.18, height / 2 - 0.55, 0],
            color=color,
            stroke_width=0.8,
        ).set_opacity(0.35)
        return mn.VGroup(box, text, rule)

    @staticmethod
    def _label(text, color, size):
        return mn.Text(text, font_size=size, color=color)

    @staticmethod
    def _packet(text, color):
        label = mn.Text(text, font_size=11, color=color)
        box = mn.RoundedRectangle(
            width=label.width + 0.24,
            height=label.height + 0.16,
            corner_radius=0.04,
            stroke_color=color,
            stroke_width=1.0,
            fill_color=BG,
            fill_opacity=0.98,
        )
        return mn.VGroup(box, label)

    @staticmethod
    def _arrow(start, end, color):
        return mn.Arrow(
            start,
            end,
            buff=0.05,
            color=color,
            stroke_width=1.5,
            max_tip_length_to_length_ratio=0.08,
        ).set_opacity(0.42)

    @staticmethod
    def _instrument(center, color):
        shaft = mn.Line(center + [-0.32, -0.18, 0], center, color=color, stroke_width=2.6)
        jaw_1 = mn.Line(center, center + [0.18, 0.08, 0], color=color, stroke_width=1.8)
        jaw_2 = mn.Line(center, center + [0.18, -0.08, 0], color=color, stroke_width=1.8)
        pivot = mn.Dot(center, radius=0.035, color=color)
        return mn.VGroup(shaft, jaw_1, jaw_2, pivot)

    @staticmethod
    def _rollout_indices(costs):
        order = torch.argsort(costs)
        ranks = torch.linspace(0, min(costs.numel() - 1, 260), NUM_ROLLOUTS).long()
        indices = order[ranks].numpy()
        best = int(order[0])
        if best not in indices:
            indices[0] = best
        return indices

    @staticmethod
    def _project_rollouts(rollouts, center, width, height):
        points = rollouts.reshape(-1, 3)
        basis = np.array([[1.0, -0.52, 0.0], [0.28, 0.2, 1.0]])
        projected = points @ basis.T
        low, high = projected.min(axis=0), projected.max(axis=0)
        span = np.maximum(high - low, 1e-8)
        scale = min(width / span[0], height / span[1])
        midpoint = 0.5 * (low + high)

        def project(position):
            point = np.asarray(position) @ basis.T
            screen = (point - midpoint) * scale
            return center + [screen[0], screen[1], 0]

        projected_rollouts = np.array([
            [project(position) for position in rollout]
            for rollout in rollouts
        ])
        return projected_rollouts, project

    @staticmethod
    def _screen_path(points, color, width, opacity):
        path = mn.VMobject()
        path.set_points_as_corners(points)
        path.set_stroke(color, width=width, opacity=opacity)
        return path

    @staticmethod
    def _cost_bars(costs, best_rank):
        log_cost = np.log1p(costs - costs.min())
        normalized = (log_cost - log_cost.min()) / max(np.ptp(log_cost), 1e-8)
        bars, selected = [], []
        for index, value in enumerate(normalized):
            height = 0.08 + 0.62 * float(value)
            bar = mn.Rectangle(
                width=0.08,
                height=height,
                stroke_width=0,
                fill_color=MUTED,
                fill_opacity=0.55,
            )
            bar.move_to([0.105 * index, height / 2, 0])
            bars.append(bar)
            selected_bar = bar.copy()
            if index == best_rank:
                selected_bar.set_fill(YELLOW, opacity=1.0)
            else:
                selected_bar.set_fill(RED, opacity=0.14 + 0.35 * float(value))
            selected.append(selected_bar)
        return mn.VGroup(*bars), mn.VGroup(*selected)

    @staticmethod
    def _basis_weight_strength(weights, basis):
        strength = torch.linalg.vector_norm(weights, dim=(0, 2)).numpy() * basis
        return strength / max(strength.max(), 1e-8)

    @staticmethod
    def _updated_strength(step, basis, before_strength):
        delta = torch.linalg.vector_norm(
            step["pose_weights_after"] - step["pose_weights_before"],
            dim=(0, 2),
        ).numpy()
        delta = delta / max(delta.max(), 1e-8)
        return np.clip(0.55 * before_strength + 0.75 * delta * basis, 0.0, 1.0)

    @staticmethod
    def _weighted_connections(nodes, sum_node, strength, active, color):
        return mn.VGroup(*[
            mn.Line(
                node.get_right(),
                sum_node.get_left(),
                color=color if index in active else mn.GREY_D,
                stroke_width=0.35 + 2.3 * float(strength[index]),
            ).set_opacity(0.12 + 0.75 * float(strength[index]))
            for index, node in enumerate(nodes)
        ])

    @staticmethod
    def _weight_matrix(step, after):
        before = step["pose_weights_before"]
        weights = step["pose_weights_after"] if after else before
        delta = torch.linalg.vector_norm(step["pose_weights_after"] - before, dim=-1)
        magnitude = torch.linalg.vector_norm(weights, dim=-1)
        magnitude = magnitude / max(magnitude.max().item(), 1e-8)
        delta = delta / max(delta.max().item(), 1e-8)
        cells = []
        for row in range(6):
            for column in range(20):
                change = float(delta[row, column])
                value = float(magnitude[row, column])
                color = (
                    mn.interpolate_color(mn.ManimColor(CYAN), mn.ManimColor(GREEN), change)
                    if after
                    else mn.ManimColor(CYAN)
                )
                opacity = 0.12 + 0.75 * max(value, change if after else 0.0)
                cell = mn.Square(
                    side_length=0.082,
                    stroke_color=mn.GREY_D,
                    stroke_width=0.25,
                    fill_color=color,
                    fill_opacity=opacity,
                )
                cell.move_to([0.092 * column, -0.092 * row, 0])
                cells.append(cell)
        return mn.VGroup(*cells).move_to(mn.ORIGIN)
