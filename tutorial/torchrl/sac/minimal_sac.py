"""Minimal SAC training example using TorchRL and MuJoCo InvertedPendulum.

Install:
    pip install torch torchrl "gymnasium[mujoco]"

Run:
    python tutorial/torchrl/sac/minimal_sac.py
"""

import torch
from torch import nn
from tensordict.nn import TensorDictModule
from torchrl.collectors import Collector
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.envs import (
    DoubleToFloat,
    ExplorationType,
    GymEnv,
    TransformedEnv,
    set_exploration_type,
)
from torchrl.modules import (
    NormalParamExtractor,
    ProbabilisticActor,
    TanhNormal,
    ValueOperator,
)
from torchrl.objectives import SACLoss, SoftUpdate


def make_env() -> TransformedEnv:
    return TransformedEnv(GymEnv("InvertedPendulum-v5"), DoubleToFloat())


class QNetwork(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self, observation: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        return self.network(torch.cat((observation, action), dim=-1))


def main() -> None:
    torch.manual_seed(0)
    env = make_env()
    observation_dim = env.observation_spec["observation"].shape[-1]
    action_dim = env.action_spec.shape[-1]

    actor_parameters = TensorDictModule(
        nn.Sequential(
            nn.Linear(observation_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2 * action_dim),
            NormalParamExtractor(),
        ),
        in_keys=["observation"],
        out_keys=["loc", "scale"],
    )
    actor = ProbabilisticActor(
        module=actor_parameters,
        in_keys=["loc", "scale"],
        spec=env.action_spec,
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": env.action_spec.space.low,
            "high": env.action_spec.space.high,
        },
    )
    q_network = ValueOperator(
        QNetwork(observation_dim, action_dim),
        in_keys=["observation", "action"],
    )

    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=20_000),
        batch_size=64,
    )
    loss_module = SACLoss(
        actor,
        q_network,
        action_spec=env.action_spec,
        delay_qvalue=True,
    )
    loss_module.make_value_estimator(gamma=0.99)
    target_updater = SoftUpdate(loss_module, eps=0.995)
    optimizer = torch.optim.Adam(loss_module.parameters(), lr=3e-4)

    collector = Collector(
        create_env_fn=make_env,
        policy=actor,
        frames_per_batch=100,
        total_frames=20_000,
        init_random_frames=1_000,
        auto_register_policy_transforms=True,
    )

    try:
        for batch_index, batch in enumerate(collector):
            replay_buffer.extend(batch)

            for _ in range(10):
                sample = replay_buffer.sample()
                losses = loss_module(sample)
                total_loss = (
                    losses["loss_actor"]
                    + losses["loss_qvalue"]
                    + losses["loss_alpha"]
                )

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                target_updater.step()

            collector.update_policy_weights_()

            if batch_index % 10 == 0:
                print(
                    f"frames={len(replay_buffer):4d}  "
                    f"actor_loss={losses['loss_actor'].item():.3f}  "
                    f"q_loss={losses['loss_qvalue'].item():.3f}"
                )
    finally:
        collector.shutdown()

    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        rollout = env.rollout(max_steps=200, policy=actor)
    episode_return = rollout["next", "reward"].sum().item()
    print(f"evaluation return: {episode_return:.1f}")
    env.close()


if __name__ == "__main__":
    main()
