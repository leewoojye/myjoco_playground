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
