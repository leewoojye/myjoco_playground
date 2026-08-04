"""Render a recorded dVRK qpos trajectory with MuJoCo's CPU OSMesa backend."""

import argparse
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
XML_PATH = ROOT_DIR / "assets" / "robots" / "dvrk" / "scene_psm_surrol_needle_reach_offset.xml"
INITIAL_CAMERA = (180, -20, 0.55)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", required=True, type=Path, help="NPZ file containing qpos frames.")
    parser.add_argument("--output", required=True, type=Path, help="Output MP4 path.")
    parser.add_argument("--video-fps", required=True, type=int)
    parser.add_argument("--render-width", required=True, type=int)
    parser.add_argument("--render-height", required=True, type=int)
    return parser.parse_args()


def make_camera(model, focus_position):
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = focus_position
    camera.azimuth, camera.elevation, camera.distance = INITIAL_CAMERA
    return camera


def main():
    args = parse_args()
    with np.load(args.trajectory) as trajectory_data:
        qpos_trajectory = trajectory_data["qpos"]
    if qpos_trajectory.ndim != 2 or not len(qpos_trajectory):
        raise ValueError("The trajectory must contain at least one two-dimensional qpos frame.")

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    if qpos_trajectory.shape[1] != model.nq:
        raise ValueError(f"Trajectory nq={qpos_trajectory.shape[1]} does not match model nq={model.nq}.")
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.render_width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, args.render_height)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    camera = make_camera(model, data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "target")])
    render_option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(render_option)
    render_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True

    args.output.parent.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(model, height=args.render_height, width=args.render_width)
    video_writer = imageio.get_writer(
        args.output,
        fps=args.video_fps,
        codec="libx264",
        macro_block_size=1,
    )
    try:
        for qpos in qpos_trajectory:
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera, render_option)
            video_writer.append_data(renderer.render())
    finally:
        video_writer.close()
        renderer.close()


if __name__ == "__main__":
    main()
