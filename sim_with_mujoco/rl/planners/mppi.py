import numpy as np
import torch
from pytorch_mppi import MPPI


class dVRKMPPIPlanner:
    DIM = 7

    def __init__(
        self,
        dynamics,
        obstacle_position,
        obstacle_radius,
        num_samples=512,
        horizon=25,
        tip_radius=0.006,
        collision_weight=1000.0,
    ):
        self.dynamics = dynamics
        self.goal = torch.zeros(
            self.DIM,
            dtype=dynamics.centers.dtype,
            device=dynamics.centers.device,
        )
        self.obstacle_position = torch.as_tensor(
            obstacle_position,
            dtype=self.goal.dtype,
            device=self.goal.device,
        )
        self.collision_radius = float(obstacle_radius + tip_radius)
        action_limit = torch.tensor(
            [0.004, 0.004, 0.004, 0.05, 0.05, 0.05, 0.05],
            dtype=self.goal.dtype,
            device=self.goal.device,
        )

        def running_cost(state, action):
            position_error = state[..., :3] - self.goal[:3]
            rotation_vector = -state[..., 3:6]
            angle = torch.linalg.vector_norm(rotation_vector, dim=-1)
            local_z = torch.zeros_like(rotation_vector)
            local_z[..., 2] = 1.0
            cross_once = torch.linalg.cross(rotation_vector, local_z, dim=-1)
            cross_twice = torch.linalg.cross(rotation_vector, cross_once, dim=-1)
            shaft_direction = (
                local_z
                + torch.sinc(angle / torch.pi)[..., None] * cross_once
                + 0.5 * torch.sinc(angle / (2.0 * torch.pi)).square()[..., None] * cross_twice
            )
            obstacle_offset = self.obstacle_position - state[..., :3]
            perpendicular = (
                obstacle_offset - (obstacle_offset * shaft_direction).sum(dim=-1, keepdim=True) * shaft_direction
            )
            line_distance = torch.linalg.vector_norm(perpendicular, dim=-1)
            collision_cost = collision_weight * torch.relu(1.0 - line_distance / self.collision_radius).square()
            return (
                2000.0 * position_error.square().sum(dim=-1)
                + collision_cost
                + 0.01 * (action / action_limit).square().sum(dim=-1)
            )

        def terminal_cost(states, actions):
            final_state = states[..., -1, :]
            position_error = final_state[..., :3] - self.goal[:3]
            return 17.0 * 2000.0 * position_error.square().sum(dim=-1)

        self.mppi = MPPI(
            dynamics=self.dynamics,
            running_cost=running_cost,
            terminal_state_cost=terminal_cost,
            nx=self.DIM,
            noise_sigma=torch.diag((0.5 * action_limit).square()),
            num_samples=num_samples,
            horizon=horizon,
            lambda_=0.01,
            u_min=-action_limit,
            u_max=action_limit,
        )

    def set_goal(self, goal):
        goal = torch.as_tensor(goal, dtype=self.goal.dtype, device=self.goal.device)
        self.goal = goal

    def command(self, state):
        state = torch.as_tensor(state, dtype=self.goal.dtype, device=self.goal.device)
        return self.mppi.command(state).cpu().numpy().astype(np.float32)
