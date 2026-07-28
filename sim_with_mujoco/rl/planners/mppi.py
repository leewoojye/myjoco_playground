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
        num_samples=512,
        horizon=20,
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
            # lambda_=1.0,
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
