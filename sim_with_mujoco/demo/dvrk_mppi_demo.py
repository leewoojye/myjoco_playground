from pathlib import Path

from sim_with_mujoco.rl import MPPIPlanner
from sim_with_mujoco.environment.env import DvrkEnv


ROOT_DIR = Path(__file__).resolve().parents[2]
XML_PATH = ROOT_DIR / "assets" / "robots" / "dvrk" / "scene_psm_surrol_needle_reach.xml"


def main():
    env = DvrkEnv(XML_PATH, control_steps=10)
    controller = MPPIPlanner(action_scale=env.action_scale)
    observation, _ = env.reset()

    for step in range(env.max_steps):
        action = controller.command(observation)
        observation, _, terminated, truncated, info = env.step(action)
        print(f"step={step:03d} tip_error={info['tip_error'] * 1000.0:6.2f} mm")
        if terminated or truncated:
            break

    print("success" if terminated else "failed")


if __name__ == "__main__":
    main()
