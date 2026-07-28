import numpy as np
import torch
from pytorch_mppi import MPPI


class LowFrequencyMPPI(MPPI):
    def __init__(self, *args, gamma=2.0, **kwargs):
        self.gamma = float(gamma)
        super().__init__(*args, **kwargs)

    def _sample_noise(self, shape):
        horizon = shape[-1]
        num_frequencies = horizon // 2 + 1
        frequency = torch.arange(num_frequencies, dtype=self.dtype, device=self.d)
        frequency = torch.clamp(frequency / num_frequencies, min=1.0 / num_frequencies)
        frequency_variance = frequency.pow(-self.gamma)
        normalization = (
            horizon**-2
            * num_frequencies**self.gamma
            * (
                1.0
                + 4.0
                * torch
                .arange(
                    1,
                    num_frequencies,
                    dtype=self.dtype,
                    device=self.d,
                )
                .pow(-self.gamma)
                .sum()
            )
        )
        frequency_std = torch.sqrt(frequency_variance / normalization)
        frequency_std = frequency_std.reshape((1,) * (len(shape) - 1) + (num_frequencies, 1))
        spectrum_shape = shape[:-1] + (num_frequencies, self.nu)
        real = torch.randn(*spectrum_shape, dtype=self.dtype, device=self.d) * frequency_std
        imag = torch.randn(*spectrum_shape, dtype=self.dtype, device=self.d) * frequency_std
        imag[..., 0, :] = 0.0
        if horizon % 2 == 0:
            imag[..., -1, :] = 0.0

        noise = torch.fft.irfft(torch.complex(real, imag), n=horizon, dim=-2)
        if self._diagonal_sigma:
            noise = noise * self._noise_sigma_sqrt_diag
        else:
            noise = noise @ self._noise_sigma_chol.T
        return noise + self.noise_mu


def log_dimensionless_jerk(positions, dt):
    velocity = torch.diff(positions, dim=-2) / dt
    jerk = torch.diff(velocity, n=2, dim=-2) / dt**2
    duration = (positions.shape[-2] - 1) * dt
    peak_speed_sq = torch.linalg.vector_norm(velocity, dim=-1).amax(dim=-1).square()
    squared_jerk = jerk.square().sum(dim=(-2, -1)) * dt
    dimensionless_jerk = duration**5 * squared_jerk / peak_speed_sq.clamp_min(torch.finfo(positions.dtype).eps)
    return -torch.log(dimensionless_jerk.clamp_min(torch.finfo(positions.dtype).eps))


class dVRKMPPIPlanner:
    DIM = 7

    def __init__(
        self,
        dynamics,
        obstacle_position,
        obstacle_radius,
        rcm_position,
        num_samples=512,
        horizon=40,
        tip_radius=0.006,
        collision_weight=10000.0,
        ldj_weight=1.0,
        dt=0.02,
        mppi_class=MPPI,
        mppi_kwargs=None,
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
            rotation_vector = -state[..., 3:6]
            angle = torch.linalg.vector_norm(rotation_vector, dim=-1)
            local_z = torch.zeros_like(rotation_vector)
            local_z[..., 2] = 1.0
            cross_once = torch.linalg.cross(rotation_vector, local_z, dim=-1)
            cross_twice = torch.linalg.cross(rotation_vector, cross_once, dim=-1)
            tip_direction = (
                local_z
                + torch.sinc(angle / torch.pi)[..., None] * cross_once
                + 0.5 * torch.sinc(angle / (2.0 * torch.pi)).square()[..., None] * cross_twice
            )
            wrist_position = state[..., :3] - 0.010 * tip_direction

            shaft = wrist_position - self.rcm_position
            shaft_t = ((self.obstacle_position - self.rcm_position) * shaft).sum(dim=-1) / shaft.square().sum(dim=-1)
            shaft_closest = self.rcm_position + shaft_t.clamp(0.0, 1.0)[..., None] * shaft

            distal = state[..., :3] - wrist_position
            distal_t = ((self.obstacle_position - wrist_position) * distal).sum(dim=-1) / distal.square().sum(dim=-1)
            distal_closest = wrist_position + distal_t.clamp(0.0, 1.0)[..., None] * distal

            collision_distance = torch.minimum(
                torch.linalg.vector_norm(self.obstacle_position - shaft_closest, dim=-1),
                torch.linalg.vector_norm(self.obstacle_position - distal_closest, dim=-1),
            )
            collision_cost = collision_weight * torch.relu(1.0 - collision_distance / self.collision_radius).square()
            ldj_cost = 0.0
            if ldj_weight and t == horizon - 1:
                ldj_cost = -ldj_weight * log_dimensionless_jerk(torch.stack(rollout_positions, dim=-2), dt)
            return (
                2000.0 * position_error.square().sum(dim=-1)
                + collision_cost
                + ldj_cost
                + 0.01
                * (control / control_limit)
                .square()
                .sum(dim=-1)  # 행동 크기에 대한 패널티를 부여하기 위해 mppi action은 증분으로 표현됨
            )

        def terminal_cost(states, actions):
            # terminal scale: 5/8/15
            return 5.0 * running_cost(states[..., -1, :], actions[..., -1, :], 0)

        # def terminal_cost(states, actions):
        #     final_state = states[..., -1, :]
        #     position_error = final_state[..., :3] - self.goal[:3]
        #     return 17.0 * 2000.0 * position_error.square().sum(dim=-1)

        self.mppi = mppi_class(
            dynamics=rollout_dynamics,
            running_cost=running_cost,
            terminal_state_cost=terminal_cost,
            nx=self.DIM,
            noise_sigma=torch.diag((0.5 * control_limit).square()),
            num_samples=num_samples,
            horizon=horizon,
            lambda_=0.01,
            u_min=-control_limit,
            u_max=control_limit,
            step_dependent_dynamics=True,
            **(mppi_kwargs or {}),
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


class LFMPPIPlanner(dVRKMPPIPlanner):
    def __init__(self, *args, gamma=2.0, **kwargs):
        super().__init__(
            *args,
            mppi_class=LowFrequencyMPPI,
            mppi_kwargs={"gamma": gamma},
            **kwargs,
        )
