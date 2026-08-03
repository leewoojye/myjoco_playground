import numpy as np
import torch
from pytorch_mppi import MPPI, SMPPI


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


class DvrkSMPPI(SMPPI):
    def __init__(self, *args, action_cost_weight, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_cost_weight = torch.as_tensor(
            action_cost_weight,
            dtype=self.dtype,
            device=self.d,
        )

    def _compute_total_cost_batch(self):
        self._compute_perturbed_action_and_noise()
        action_cost = self._compute_action_cost(self.noise)
        action_difference = self.u_scale * torch.diff(self.perturbed_action, dim=-2)
        action_smoothness_cost = (action_difference.square() * self.action_cost_weight).sum(dim=(1, 2))

        rollout_cost, self.states, actions = self._compute_rollout_costs(self.perturbed_action)
        self.actions = actions / self.u_scale if actions is not None else None
        perturbation_cost = torch.sum(self.U * action_cost, dim=(1, 2))
        self.cost_total = rollout_cost + perturbation_cost + action_smoothness_cost
        return self.cost_total


class dVRKMPPIPlanner:
    DIM = 7
    CONTROL_LIMIT = (0.004, 0.004, 0.004, 0.06, 0.06, 0.06, 0.05)

    def __init__(
        self,
        dynamics,
        num_samples=512,
        horizon=8,
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
            self.CONTROL_LIMIT,
            dtype=self.goal.dtype,
            device=self.goal.device,
        )
        use_smppi = mppi_class is DvrkSMPPI
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
            state_cost = 2000.0 * position_error.square().sum(dim=-1)
            if use_smppi:
                return state_cost
            return state_cost + 0.1 * (
                4.0 * (control[..., :3] / control_limit[:3]).square().sum(dim=-1)
                + (control[..., 3:] / control_limit[3:]).square().sum(dim=-1)
            )

        def terminal_cost(states, actions):
            terminal = 10.0 * running_cost(states[..., -1, :], actions[..., -1, :], 0)
            if use_smppi:
                return terminal
            smoothness_cost = (torch.diff(actions, dim=-2) / control_limit).square().sum(dim=(-2, -1))
            second_difference_cost = (
                (  # noqa: F841
                    torch.diff(actions, n=2, dim=-2) / control_limit
                )
                .square()
                .sum(dim=(-2, -1))
            )
            return (
                terminal
                # + smoothness_cost * 0.1
                # + second_difference_cost * 0.1
            )

        mppi_config = {
            "noise_sigma": torch.diag((0.5 * control_limit).square()),
            "u_min": -control_limit,
            "u_max": control_limit,
        }
        mppi_config.update(mppi_kwargs or {})
        self.mppi = mppi_class(
            dynamics=rollout_dynamics,
            running_cost=running_cost,
            terminal_state_cost=terminal_cost,
            nx=self.DIM,
            num_samples=num_samples,
            horizon=horizon,
            # lambda_=0.1,
            lambda_=0.01,
            step_dependent_dynamics=True,
            **mppi_config,
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


class SMPPIDVRKPlanner(dVRKMPPIPlanner):
    def __init__(self, *args, action_cost_weight=0.1, dt=0.02, **kwargs):
        control_limit = torch.tensor(self.CONTROL_LIMIT)
        rate_limit = control_limit / dt
        super().__init__(
            *args,
            dt=dt,
            mppi_class=DvrkSMPPI,
            mppi_kwargs={
                "noise_sigma": torch.diag((0.5 * rate_limit).square()),
                "u_min": -rate_limit,
                "u_max": rate_limit,
                "delta_t": dt,
                "action_min": -control_limit,
                "action_max": control_limit,
                "action_cost_weight": action_cost_weight / control_limit.square(),
            },
            **kwargs,
        )
