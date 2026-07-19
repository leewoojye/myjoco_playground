"""Minimal DQN training example using TorchRL and CartPole-v1.

Install:
    pip install torch torchrl gymnasium

Run:
    python tutorial/torchrl/dqn/minimal_dqn.py
"""

import torch
from torch import nn
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.collectors import Collector
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.envs import ExplorationType, GymEnv, set_exploration_type
from torchrl.modules import EGreedyModule, QValueActor
from torchrl.objectives import DQNLoss, SoftUpdate


def make_env() -> GymEnv:
    return GymEnv("CartPole-v1")


def main() -> None:
    torch.manual_seed(0)
    env = make_env()

    q_network = TensorDictModule(
        nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        ),
        in_keys=["observation"],
        out_keys=["action_value"],
    )
    greedy_policy = QValueActor(q_network, spec=env.action_spec)
    epsilon_greedy = EGreedyModule(
        spec=env.action_spec,
        eps_init=1.0,
        eps_end=0.05,
        annealing_num_steps=5_000,
    )
    collection_policy = TensorDictSequential(greedy_policy, epsilon_greedy)

    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=20_000),
        batch_size=64,
    )
    loss_module = DQNLoss(
        greedy_policy,
        action_space=env.action_spec,
        delay_value=True,
    )
    loss_module.make_value_estimator(gamma=0.99)
    target_updater = SoftUpdate(loss_module, eps=0.995)
    optimizer = torch.optim.Adam(loss_module.parameters(), lr=1e-3)

    collector = Collector(
        create_env_fn=make_env,
        policy=collection_policy,
        frames_per_batch=100,
        total_frames=10_000,
        init_random_frames=1_000,
        auto_register_policy_transforms=True,
    )

    try:
        for batch_index, batch in enumerate(collector):
            replay_buffer.extend(batch)

            for _ in range(10):
                sample = replay_buffer.sample()
                loss = loss_module(sample)["loss"]

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                target_updater.step()

            epsilon_greedy.step(batch.numel())
            collector.update_policy_weights_()

            if batch_index % 10 == 0:
                print(
                    f"frames={len(replay_buffer):4d}  "
                    f"loss={loss.item():.3f}  epsilon={epsilon_greedy.eps.item():.3f}"
                )
    finally:
        collector.shutdown()

    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        rollout = env.rollout(max_steps=500, policy=greedy_policy)
    episode_return = rollout["next", "reward"].sum().item()
    print(f"evaluation return: {episode_return:.1f}")
    env.close()


if __name__ == "__main__":
    main()
