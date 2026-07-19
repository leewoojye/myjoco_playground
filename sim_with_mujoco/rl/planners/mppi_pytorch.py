import numpy as np
import torch
from pytorch_mppi import MPPI


class DvrkMPPIController:
    """MPPI controller for Cartesian dVRK tip-error dynamics."""

    def __init__(self, action_scale=0.004, num_samples=512, horizon=25):
        self.action_scale = float(action_scale)

        def dynamics(state, action):
            return state - self.action_scale * action

        def running_cost(state, action):
            return 1000.0 * state.square().sum(dim=-1) + 0.01 * action.square().sum(dim=-1)

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

    def command(self, observation):
        state = torch.as_tensor(observation, dtype=torch.float32)
        return self.mppi.command(state).cpu().numpy().astype(np.float32)
