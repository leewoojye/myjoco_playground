import numpy as np
import torch
from pytorch_mppi import KMPPI


class dVRKKMPPIPlanner:
    DIM = 7

    def __init__(
        self,
        dynamics,
        obstacle_position,
        obstacle_radius,
        rcm_position,
        num_samples=512,
        horizon=25,
        num_support_pts=None,
        tip_radius=0.006,
        collision_weight=1000.0,
        rcm_weight=20000.0,
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
        self.rcm_position = torch.as_tensor(
            rcm_position,
            dtype=self.goal.dtype,
            device=self.goal.device,
        )
        self.collision_radius = float(obstacle_radius + tip_radius)
        control_limit = torch.tensor(
            [0.004, 0.004, 0.004, 0.05, 0.05, 0.05, 0.05],
            dtype=self.goal.dtype,
            device=self.goal.device,
        )

        def rollout_dynamics(state, control):
            action = control.clone()
            action[..., :3] = state[..., :3] + control[..., :3]
            return self.dynamics(state, action)

        def running_cost(state, control):
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
            rcm_offset = self.rcm_position - state[..., :3]
            rcm_deviation = torch.linalg.vector_norm(
                torch.linalg.cross(rcm_offset, shaft_direction, dim=-1),
                dim=-1,
            )
            obstacle_offset = self.obstacle_position - state[..., :3]
            perpendicular = (
                obstacle_offset - (obstacle_offset * shaft_direction).sum(dim=-1, keepdim=True) * shaft_direction
            )
            line_distance = torch.linalg.vector_norm(perpendicular, dim=-1)
            collision_cost = collision_weight * torch.relu(1.0 - line_distance / self.collision_radius).square()
            return (
                2000.0 * position_error.square().sum(dim=-1)
                + rcm_weight * rcm_deviation.square()
                + collision_cost
                + 0.01 * (control / control_limit).square().sum(dim=-1) # 행동 크기에 대한 패널티를 부여하기 위해 mppi action은 증분으로 표현됨
            )

        def terminal_cost(states, actions):
            final_state = states[..., -1, :]
            position_error = final_state[..., :3] - self.goal[:3]
            return 17.0 * 2000.0 * position_error.square().sum(dim=-1)

        self.mppi = KMPPI(
            dynamics=rollout_dynamics,
            running_cost=running_cost,
            terminal_state_cost=terminal_cost,
            nx=self.DIM,
            noise_sigma=torch.diag((0.5 * control_limit).square()),
            num_samples=num_samples,
            horizon=horizon,
            num_support_pts=num_support_pts,
            lambda_=0.01,
            u_min=-control_limit,
            u_max=control_limit,
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
