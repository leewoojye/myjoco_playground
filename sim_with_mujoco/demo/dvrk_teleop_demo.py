from pathlib import Path
import time
import glfw
import mujoco
import numpy as np

from sim_with_mujoco.environment.env import DvrkEnv
from sim_with_mujoco.viewer.surrol_keyboard_viewer import SurrolKeyboardViewer


ROOT_DIR = Path(__file__).resolve().parents[2]
XML_PATH = ROOT_DIR / "assets" / "robots" / "dvrk" / "scene_psm_surrol_needle_reach.xml"

CONTROL_PERIOD = 1.0 / 20.0
POSITION_STEP = 0.001
JAW_OPEN = 0.30
JAW_CLOSED = 0.0


def run_gui():
    env = DvrkEnv(XML_PATH, action_scale=POSITION_STEP, control_steps=25)
    observation, _ = env.reset()

    model, data = env.model, env.data
    viewer = SurrolKeyboardViewer(model, data)
    viewer.init_viewer(
        window_title="MyJoCo dVRK PSM Teleoperation",
        initial_camera=(180, -20, 0.55),
        focus_position=data.site_xpos[env.target_site_id],
    )

    jaw_actuator_id = env.get_id(
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        "PSM1_jaw",
    )
    last_control_time = time.monotonic() - CONTROL_PERIOD
    jaw_target = JAW_OPEN

    try:
        while not glfw.window_should_close(viewer.window):
            glfw.poll_events()
            keyboard_action = viewer.poll_action()
            now = time.monotonic()

            if now - last_control_time >= CONTROL_PERIOD:
                action = np.sign(keyboard_action[:3])
                jaw_target = JAW_CLOSED if keyboard_action[4] < 0.0 else JAW_OPEN
                data.ctrl[jaw_actuator_id] = jaw_target

                if np.any(action):
                    observation, _, _, _, _ = env.step(action)
                else:
                    env.plant.step(env.control_steps)
                    observation = env.get_observation()
                last_control_time = now

            viewer.set_overlay(
                [
                    f"Tip error: {np.linalg.norm(observation) * 1000.0:5.1f} mm",
                    f"Jaw target: {jaw_target:+.2f} rad",
                ]
            )
            viewer.render()
    finally:
        viewer.terminate_viewer()


if __name__ == "__main__":
    run_gui()
