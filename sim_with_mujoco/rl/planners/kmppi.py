import numpy as np
import torch
from pytorch_mppi import KMPPI

from sim_with_mujoco.rl.planners.mppi import log_dimensionless_jerk


class dVRKKMPPIPlanner:
    DIM = 7

    def __init__(
        self,
        dynamics,
        num_samples=512,
        horizon=20,
        num_support_pts=5,
        ldj_weight=1.0,
        dt=0.02,
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
                    4.0 * (control[..., :3] / control_limit[:3]).square().sum(dim=-1)
                    + (control[..., 3:] / control_limit[3:]).square().sum(dim=-1)
                )
            )

        def terminal_cost(states, actions):
            return 10.0 * running_cost(states[..., -1, :], actions[..., -1, :], 0)

        self.mppi = KMPPI(
            dynamics=rollout_dynamics,
            running_cost=running_cost,
            terminal_state_cost=terminal_cost,
            nx=self.DIM,
            noise_sigma=torch.diag((0.5 * control_limit).square()),
            num_samples=num_samples,
            horizon=horizon,
            num_support_pts=num_support_pts,
            lambda_=0.1,
            u_min=-control_limit,
            u_max=control_limit,
            step_dependent_dynamics=True,
        )

    def set_goal(self, goal):
        goal = torch.as_tensor(goal, dtype=self.goal.dtype, device=self.goal.device)
        self.goal = goal

    def command(self, state):
        state = torch.as_tensor(state, dtype=self.goal.dtype, device=self.goal.device)
        control = self.mppi.command(state)
        action = control.clone()
        action[:3] = state[:3] + control[:3]
        return action.cpu().numpy().astype(np.float32)
