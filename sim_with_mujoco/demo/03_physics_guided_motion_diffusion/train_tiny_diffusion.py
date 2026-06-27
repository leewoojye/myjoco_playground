from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from diffusion_model import TinyDenoiserFactory, make_beta_schedule, require_torch


THIS_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=THIS_DIR / "motion_rollouts.npz")
    parser.add_argument("--out", type=Path, default=THIS_DIR / "tiny_diffusion_policy.pt")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--diffusion-steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=23)
    return parser.parse_args()


def main() -> None:
    torch, _ = require_torch()
    args = parse_args()
    data = np.load(args.dataset)
    actions = data["actions"].astype(np.float32)
    conditions = data["conditions"].astype(np.float32)
    flat_actions = actions.reshape(actions.shape[0], -1)

    action_mean = flat_actions.mean(axis=0, keepdims=True)
    action_std = flat_actions.std(axis=0, keepdims=True) + 1e-6
    cond_mean = conditions.mean(axis=0, keepdims=True)
    cond_std = conditions.std(axis=0, keepdims=True) + 1e-6
    x0 = torch.from_numpy((flat_actions - action_mean) / action_std)
    cond = torch.from_numpy((conditions - cond_mean) / cond_std)

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    model = TinyDenoiserFactory.build(x0.shape[1], cond.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    betas = make_beta_schedule(torch, args.diffusion_steps)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    for step in range(args.steps):
        batch_idx = torch.as_tensor(rng.integers(0, len(x0), size=args.batch_size), dtype=torch.long)
        t = torch.as_tensor(rng.integers(0, args.diffusion_steps, size=args.batch_size), dtype=torch.long)
        clean = x0[batch_idx]
        batch_cond = cond[batch_idx]
        noise = torch.randn_like(clean)
        alpha_bar = alpha_bars[t][:, None]
        noisy = torch.sqrt(alpha_bar) * clean + torch.sqrt(1.0 - alpha_bar) * noise
        pred = model(noisy, batch_cond, t)
        loss = torch.mean((pred - noise) ** 2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"step={step:05d} loss={float(loss):.6f}")

    torch.save(
        {
            "model": model.state_dict(),
            "horizon": actions.shape[1],
            "action_dim": actions.shape[2],
            "condition_dim": conditions.shape[1],
            "diffusion_steps": args.diffusion_steps,
            "action_mean": action_mean,
            "action_std": action_std,
            "cond_mean": cond_mean,
            "cond_std": cond_std,
        },
        args.out,
    )
    print(f"saved={args.out}")


if __name__ == "__main__":
    main()
