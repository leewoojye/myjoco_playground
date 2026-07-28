import numpy as np
import torch

from sim_with_mujoco.rl.planners.mppi import LowFrequencyMPPI, log_dimensionless_jerk


class LFMPPIPlanner:
    DIM = 7

    def __init__(
        self,
        dynamics,
        num_samples=512,
        horizon=8,
        ldj_weight=1.0,
        dt=0.02,
        gamma=2.0,
    ):
        self.dynamics = dynamics
        self.goal = torch.zeros(
            self.DIM,
            dtype=dynamics.centers.dtype,
            device=dynamics.centers.device,
        )
        control_limit = torch.tensor(
            [0.004, 0.004, 0.004, 0.06, 0.06, 0.06, 0.05],
            dtype=self.goal.dtype,
            device=self.goal.device,
        )
        self.previous_control = None
        rollout_positions = []

        def rollout_dynamics(state, control, t):
            if t == 0:
                rollout_positions.clear()
                rollout_positions.append(state[..., :3])
            action = control.clone()
            action[..., :3] = state[..., :3] + control[..., :3]
            next_state = self.dynamics(state, action)
            rollout_positions.append(next_state[..., :3])
            return next_state

        def running_cost(state, control, t):
            position_error = state[..., :3] - self.goal[:3]
            ldj_cost = 0.0
            if ldj_weight and t == horizon - 1:
                ldj_cost = -ldj_weight * log_dimensionless_jerk(torch.stack(rollout_positions, dim=-2), dt)
            return (
                2000.0 * position_error.square().sum(dim=-1)
                + ldj_cost
                + 0.01
                * (
                    10.0 * (control[..., :3] / control_limit[:3]).square().sum(dim=-1)
                    + (control[..., 3:] / control_limit[3:]).square().sum(dim=-1)
                )
            )

        # I2RIS 기준 terminal cost: T-1 timestep에서 비용
        # 단, T-1 timestep에서 비용은 우회 궤적을 선호할 수도 있을 우려
        def terminal_cost(states, actions):
            # control_delta = torch.diff(actions, dim=-2) / control_limit
            # smoothness_cost = control_delta.square().sum(dim=(-2, -1))

            control_second_difference = torch.diff(actions, n=2, dim=-2) / control_limit
            second_difference_cost = control_second_difference.square().sum(dim=(-2, -1))
            boundary_cost = torch.zeros_like(second_difference_cost)
            if self.previous_control is not None:
                boundary_delta = (actions[..., 0, :] - self.previous_control) / control_limit
                boundary_cost = boundary_delta.square().sum(dim=-1)
            # 비용항 가중치 동적 조절?
            return (
                4.0 * running_cost(states[..., -1, :], actions[..., -1, :], 0)
                + 1.0 * second_difference_cost
                + 1.0 * boundary_cost
            )  # second_difference_cost

        self.mppi = LowFrequencyMPPI(
            dynamics=rollout_dynamics,
            running_cost=running_cost,
            terminal_state_cost=terminal_cost,
            nx=self.DIM,
            noise_sigma=torch.diag((0.5 * control_limit).square()),
            num_samples=num_samples,
            horizon=horizon,
            lambda_=0.01,  # 0.01/1.0/2.0
            # lambda_=0.01, # 0.01/1.0/2.0
            u_min=-control_limit,
            u_max=control_limit,
            step_dependent_dynamics=True,
            gamma=gamma,
        )

    def set_goal(self, goal):
        self.goal = torch.as_tensor(goal, dtype=self.goal.dtype, device=self.goal.device)

    def command(self, state):
        state = torch.as_tensor(state, dtype=self.goal.dtype, device=self.goal.device)
        control = self.mppi.command(state)
        self.previous_control = control.detach().clone()
        action = control.clone()
        action[:3] = state[:3] + control[:3]
        return action.cpu().numpy().astype(np.float32)
