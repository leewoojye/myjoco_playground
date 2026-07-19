"""Minimal vector models for DQN and SAC.

The model patterns are adapted from TorchRL's MIT-licensed models module:
https://github.com/pytorch/rl/blob/main/torchrl/modules/models/models.py
"""

from collections.abc import Sequence

import torch
from torch import nn


class MLP(nn.Sequential):
    """MLP that concatenates multiple inputs along the last dimension."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_cells: Sequence[int] = (256, 256),
        activation_class: type[nn.Module] = nn.ReLU,
    ) -> None:
        sizes = [in_features, *num_cells, out_features]
        layers = []
        for index, (input_size, output_size) in enumerate(zip(sizes, sizes[1:])):
            layers.append(nn.Linear(input_size, output_size))
            if index < len(sizes) - 2:
                layers.append(activation_class())
        super().__init__(*layers)

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        value = inputs[0] if len(inputs) == 1 else torch.cat(inputs, dim=-1)
        return super().forward(value)


class DuelingMlpDQNet(nn.Module):
    """Discrete Q-network with separate state-value and advantage heads."""

    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.features = MLP(observation_dim, hidden_dim, (hidden_dim,))
        self.advantage = nn.Linear(hidden_dim, action_dim)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        features = self.features(observation)
        advantage = self.advantage(features)
        value = self.value(features)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)


class GaussianPolicyNetwork(nn.Module):
    """SAC policy network that returns Gaussian location and scale."""

    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.network = MLP(observation_dim, 2 * action_dim, (hidden_dim, hidden_dim))

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        location, log_scale = self.network(observation).chunk(2, dim=-1)
        scale = log_scale.clamp(-20.0, 2.0).exp()
        return location, scale


class ContinuousQNetwork(nn.Module):
    """State-action Q-network for SAC and other continuous-control methods."""

    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.network = MLP(
            observation_dim + action_dim,
            1,
            (hidden_dim, hidden_dim),
        )

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.network(observation, action)


__all__ = [
    "ContinuousQNetwork",
    "DuelingMlpDQNet",
    "GaussianPolicyNetwork",
    "MLP",
]
