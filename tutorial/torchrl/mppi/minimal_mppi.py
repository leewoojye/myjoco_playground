"""1D point-mass control with pytorch-mppi.

Install:
    pip install torch pytorch-mppi

Run:
    python tutorial/torchrl/mppi/minimal_mppi.py
"""

import torch
from pytorch_mppi import MPPI


DT = 0.05
GOAL_POSITION = 2.0


def dynamics(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """Advance [position, velocity] by one step using acceleration input."""
    position = state[..., 0]
    velocity = state[..., 1]
    acceleration = action[..., 0]

    next_position = position + DT * velocity
    next_velocity = velocity + DT * acceleration
    return torch.stack((next_position, next_velocity), dim=-1)


def running_cost(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """Penalize position error, speed, and control effort."""
    position_error = state[..., 0] - GOAL_POSITION
    return (
        5.0 * position_error.square()
        + 0.2 * state[..., 1].square()
        + 0.02 * action[..., 0].square()
    )


def main() -> None:
    torch.manual_seed(0)

    horizon = 30
    controller = MPPI(
        dynamics=dynamics,
        running_cost=running_cost,
        nx=2,
        noise_sigma=torch.tensor([[1.0]]),
        num_samples=500,
        horizon=horizon,
        lambda_=0.2,
        u_min=torch.tensor([-2.0]),
        u_max=torch.tensor([2.0]),
        u_init=torch.zeros(1),
        U_init=torch.zeros(horizon, 1),
    )

    state = torch.tensor([0.0, 0.0])

    print("step | position | velocity | action")
    for step in range(80):
        action = controller.command(state)
        state = dynamics(state, action)

        if step % 10 == 0 or step == 79:
            print(
                f"{step:4d} | {state[0]:8.3f} | "
                f"{state[1]:8.3f} | {action[0]:7.3f}"
            )

    print(f"\ngoal={GOAL_POSITION:.3f}, final_position={state[0]:.3f}")


if __name__ == "__main__":
    main()
