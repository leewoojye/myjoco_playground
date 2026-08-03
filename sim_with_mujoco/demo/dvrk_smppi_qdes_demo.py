from pathlib import Path

import glfw
import mujoco
import numpy as np

from sim_with_mujoco.environment.dvrk_needle_reach_env import DvrkNeedleReachEnv
from sim_with_mujoco.rl.planners.smppi_qdes import DvrkQDesSMPPIPlanner
from sim_with_mujoco.viewer.surrol_keyboard_viewer import SurrolKeyboardViewer

ROOT_DIR = Path(__file__).resolve().parents[2]
XML_PATH = ROOT_DIR / "assets" / "robots" / "dvrk" / "scene_psm_surrol_needle_reach_offset.xml"


def main():
    env = DvrkNeedleReachEnv(XML_PATH, control_steps=10)
    env.reset()
    jaw_actuator_id = env.get_id(mujoco.mjtObj.mjOBJ_ACTUATOR, "PSM1_jaw")
    planner = DvrkQDesSMPPIPlanner(
        env,
        env.data.site_xpos[env.target_site_id],
        dt=env.model.opt.timestep * env.control_steps,
    )
    viewer = SurrolKeyboardViewer(env.model, env.data)
    viewer.init_viewer(
        window_title="dVRK q_des SMPPI DEMO",
        initial_camera=(180, -20, 0.55),
        focus_position=env.data.site_xpos[env.target_site_id],
    )

    tip_error = np.inf
    try:
        for step in range(env.max_steps):
            glfw.poll_events()
            if glfw.window_should_close(viewer.window):
                break

            q_des = planner.command()
            for joint_id, actuator_id, target_qpos in zip(env.joint_ids, env.arm_actuator_ids, q_des[:6]):
                env.data.ctrl[actuator_id] = np.clip(target_qpos, *env.model.actuator_ctrlrange[actuator_id])
            env.data.ctrl[jaw_actuator_id] = np.clip(q_des[6], *env.model.actuator_ctrlrange[jaw_actuator_id])
            env.plant.step(env.control_steps)

            tip_error = np.linalg.norm(env.data.site_xpos[env.target_site_id] - env.data.site_xpos[env.tip_site_id])
            viewer.set_overlay([f"Tip error: {tip_error * 1000.0:5.1f} mm"])
            viewer.render()
            print(f"step={step:03d} tip_error={tip_error * 1000.0:6.2f} mm")
            if tip_error <= env.tolerance:
                break
    finally:
        viewer.terminate_viewer()

    print("success" if tip_error <= env.tolerance else "failed")


if __name__ == "__main__":
    main()
