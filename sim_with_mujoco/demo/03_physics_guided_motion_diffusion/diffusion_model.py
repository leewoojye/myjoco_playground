from __future__ import annotations


def require_torch():
    try:
        import torch
        import torch.nn as nn
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "This diffusion demo requires PyTorch. Install torch in the active venv "
            "or use collect_rollouts.py first, which only needs MuJoCo and NumPy."
        ) from exc
    return torch, nn


def make_beta_schedule(torch, steps: int):
    return torch.linspace(1e-4, 0.02, steps)


class TinyDenoiserFactory:
    @staticmethod
    def build(sequence_dim: int, condition_dim: int, hidden_dim: int = 256):
        torch, nn = require_torch()

        class TinyDenoiser(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(sequence_dim + condition_dim + 1, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, sequence_dim),
                )

            def forward(self, noisy_sequence, condition, timestep):
                if timestep.ndim == 1:
                    timestep_feature = timestep[:, None].float() / 1000.0
                else:
                    timestep_feature = timestep.float() / 1000.0
                x = torch.cat([noisy_sequence, condition, timestep_feature], dim=-1)
                return self.net(x)

        return TinyDenoiser()
