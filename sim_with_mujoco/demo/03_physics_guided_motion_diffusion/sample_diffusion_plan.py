from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from diffusion_model import TinyDenoiserFactory, make_beta_schedule, require_torch


THIS_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=THIS_DIR / "tiny_diffusion_policy.pt")
    parser.add_argument("--out", type=Path, default=THIS_DIR / "sampled_plan.npy")
    parser.add_argument("--target-speed", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=31)
    return parser.parse_args()


def main() -> None:
    torch, _ = require_torch()
    args = parse_args()
    torch.manual_seed(args.seed)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    sequence_dim = ckpt["horizon"] * ckpt["action_dim"]
    model = TinyDenoiserFactory.build(sequence_dim, ckpt["condition_dim"])
    model.load_state_dict(ckpt["model"])
    model.eval()

    betas = make_beta_schedule(torch, ckpt["diffusion_steps"])
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    condition = np.asarray([[args.target_speed, 1.0, 0.0, 0.0]], dtype=np.float32)
    condition = (condition - ckpt["cond_mean"]) / ckpt["cond_std"]
    cond = torch.from_numpy(condition)
    x = torch.randn(1, sequence_dim)

    for t_int in reversed(range(ckpt["diffusion_steps"])):
        t = torch.full((1,), t_int, dtype=torch.long)
        pred_noise = model(x, cond, t)
        alpha = alphas[t_int]
        alpha_bar = alpha_bars[t_int]
        x = (x - (1 - alpha) / torch.sqrt(1 - alpha_bar) * pred_noise) / torch.sqrt(alpha)
        if t_int > 0:
            x = x + torch.sqrt(betas[t_int]) * torch.randn_like(x)

    plan = x.detach().numpy() * ckpt["action_std"] + ckpt["action_mean"]
    plan = plan.reshape(ckpt["horizon"], ckpt["action_dim"])
    np.save(args.out, plan)
    print(f"saved={args.out}")
    print(plan[:5])


if __name__ == "__main__":
    main()
