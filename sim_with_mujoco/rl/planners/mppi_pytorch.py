import numpy as np
import torch
from pytorch_mppi import MPPI


class MPPIPlanner:
    def __init__(self, action_scale=0.004, num_samples=512, horizon=25):
        self.action_scale = float(action_scale)

        def dynamics(state, action):  # rollout용 dynamic model
            return state - self.action_scale * action

        def running_cost(state, action):  # 각 후보 궤적의 매 시점마다 최소화하는 비용 함수
            return 1000.0 * state.square().sum(dim=-1) + 0.01 * action.square().sum(dim=-1)

        # specific_action_sampler: typing.Optional[SpecificActionSampler] = None
        self.mppi = MPPI(
            dynamics=dynamics,
            running_cost=running_cost,
            nx=3,
            noise_sigma=0.5 * torch.eye(3),
            num_samples=num_samples,
            horizon=horizon,
            lambda_=0.1,
            u_min=-torch.ones(3),
            u_max=torch.ones(3),
        )

    def command(self, observation):  # 관측값을 받은 MPPI가 계산한 다음 한 스텝의 최적 행동을 반환 (추후 수정)
        state = torch.as_tensor(observation, dtype=torch.float32)
        return self.mppi.command(state).cpu().numpy().astype(np.float32)


class SRTMPPIPlanner:
    STATE_DIM = 16
    ACTION_DIM = 7

    def __init__(self, dt=0.02, num_samples=512, horizon=25):
        self.dt = float(dt)
        self.goal = torch.zeros(10, dtype=torch.float32)
        self.goal[3:9] = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])

        action_limit = torch.tensor(
            [0.004, 0.004, 0.004, 0.05, 0.05, 0.05, 0.05],
            dtype=torch.float32,
        )
        noise_std = 0.5 * action_limit

        def dynamics(state, action):
            rotation = self._rotation_6d_to_matrix(state[..., 3:9])
            response = 0.8
            linear_velocity = (1.0 - response) * state[..., 10:13] + response * action[..., :3] / self.dt
            angular_velocity_tool = rotation.transpose(-1, -2) @ state[..., 13:16, None]
            angular_velocity_tool = angular_velocity_tool.squeeze(-1)
            angular_velocity_tool = (1.0 - response) * angular_velocity_tool + response * action[..., 3:6] / self.dt
            position = state[..., :3] + self.dt * linear_velocity
            delta_rotation = self._rotation_vector_to_matrix(self.dt * angular_velocity_tool)
            rotation = rotation @ delta_rotation
            jaw = torch.clamp(state[..., 9:10] + action[..., 6:7], 0.0, 1.5707)
            angular_velocity = (rotation @ angular_velocity_tool[..., None]).squeeze(-1)
            velocity = torch.cat((linear_velocity, angular_velocity), dim=-1)
            return torch.cat(
                (position, self._matrix_to_rotation_6d(rotation), jaw, velocity),
                dim=-1,
            )

        def running_cost(state, action):
            position_error = state[..., :3] - self.goal[:3]
            rotation = self._rotation_6d_to_matrix(state[..., 3:9])
            goal_rotation = self._rotation_6d_to_matrix(self.goal[3:9])
            relative_rotation = goal_rotation.transpose(-1, -2) @ rotation
            cosine = torch.clamp(
                (relative_rotation.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0,
                -1.0 + 1e-6,
                1.0 - 1e-6,
            )
            rotation_error = torch.acos(cosine)
            jaw_error = state[..., 9] - self.goal[9]
            scaled_action = action / action_limit
            return (
                2000.0 * position_error.square().sum(dim=-1)
                + 4.0 * rotation_error.square()
                + 2.0 * jaw_error.square()
                + 0.01 * state[..., 10:16].square().sum(dim=-1)
                + 0.01 * scaled_action.square().sum(dim=-1)
            )

        self.mppi = MPPI(
            dynamics=dynamics,
            running_cost=running_cost,
            nx=self.STATE_DIM,
            noise_sigma=torch.diag(noise_std.square()),
            num_samples=num_samples,
            horizon=horizon,
            lambda_=1.0,
            u_min=-action_limit,
            u_max=action_limit,
        )

    def set_goal(self, position, rotation, jaw):
        position = torch.as_tensor(position, dtype=torch.float32)
        rotation = torch.as_tensor(rotation, dtype=torch.float32)
        if rotation.shape == (3, 3):
            rotation = self._matrix_to_rotation_6d(rotation)
        if position.shape != (3,) or rotation.shape != (6,):
            raise ValueError("position and rotation must have shapes (3,) and (3, 3) or (6,)")
        self.goal = torch.cat((position, rotation, torch.tensor([jaw], dtype=torch.float32)))

    def command(self, state):
        state = torch.as_tensor(state, dtype=torch.float32)
        if state.shape != (self.STATE_DIM,):
            raise ValueError(f"state must have shape ({self.STATE_DIM},)")
        return self.mppi.command(state).cpu().numpy().astype(np.float32)

    @classmethod
    def make_state(cls, position, rotation, jaw, linear_velocity, angular_velocity):
        rotation = torch.as_tensor(rotation, dtype=torch.float32)
        state = torch.cat((
            torch.as_tensor(position, dtype=torch.float32),
            cls._matrix_to_rotation_6d(rotation),
            torch.tensor([jaw], dtype=torch.float32),
            torch.as_tensor(linear_velocity, dtype=torch.float32),
            torch.as_tensor(angular_velocity, dtype=torch.float32),
        ))
        return state.numpy().astype(np.float32)

    @staticmethod
    def _matrix_to_rotation_6d(rotation):
        return rotation[..., :, :2].transpose(-1, -2).reshape(*rotation.shape[:-2], 6)

    @staticmethod
    def _rotation_6d_to_matrix(rotation):
        first = torch.nn.functional.normalize(rotation[..., :3], dim=-1)
        second = rotation[..., 3:6]
        second = second - (first * second).sum(dim=-1, keepdim=True) * first
        second = torch.nn.functional.normalize(second, dim=-1)
        third = torch.linalg.cross(first, second, dim=-1)
        return torch.stack((first, second, third), dim=-1)

    @staticmethod
    def _rotation_vector_to_matrix(rotation_vector):
        x, y, z = rotation_vector.unbind(dim=-1)
        zeros = torch.zeros_like(x)
        skew = torch.stack(
            (zeros, -z, y, z, zeros, -x, -y, x, zeros),
            dim=-1,
        ).reshape(*rotation_vector.shape[:-1], 3, 3)
        angle = torch.linalg.vector_norm(rotation_vector, dim=-1)
        a = torch.sinc(angle / torch.pi)[..., None, None]
        b = (0.5 * torch.sinc(angle / (2.0 * torch.pi)).square())[..., None, None]
        identity = torch.eye(3, dtype=rotation_vector.dtype, device=rotation_vector.device)
        return identity + a * skew + b * (skew @ skew)


class DvrkRBFMPPIPlanner:
    DIM = 7

    def __init__(self, dynamics, num_samples=512, horizon=25):
        if getattr(dynamics, "dof", None) != self.DIM:
            raise ValueError("dynamics must use seven state/action channels")

        self.dynamics = dynamics
        self.goal = torch.zeros(
            self.DIM,
            dtype=dynamics.centers.dtype,
            device=dynamics.centers.device,
        )
        action_limit = torch.tensor(
            [0.004, 0.004, 0.004, 0.05, 0.05, 0.05, 0.05],
            dtype=self.goal.dtype,
            device=self.goal.device,
        )

        def running_cost(state, action):
            position_error = state[..., :3] - self.goal[:3]
            return (
                2000.0 * position_error.square().sum(dim=-1)
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
            lambda_=0.01, # 지수 가중합 temperature
            u_min=-action_limit,
            u_max=action_limit,
        )

    def set_goal(self, goal):
        goal = torch.as_tensor(goal, dtype=self.goal.dtype, device=self.goal.device)
        if goal.shape != (self.DIM,):
            raise ValueError("goal must have shape (7,)")
        self.goal = goal

    def command(self, state):
        state = torch.as_tensor(state, dtype=self.goal.dtype, device=self.goal.device)
        if state.shape != (self.DIM,):
            raise ValueError("state must have shape (7,)")
        return self.mppi.command(state).cpu().numpy().astype(np.float32)
