import glfw
import mujoco
import numpy as np

from sim_with_mujoco.viewer.glfw_panel import GlfwTargetPanel
from sim_with_mujoco.viewer.gui_panel import GUIPanel


# mujoco viewer + target panel 통합
class Viewer:
    def __init__(self, model, data):  # 참조 전달
        self.model = model
        self.data = data
        self.overlay_lines = []

    def set_overlay(self, lines):
        self.overlay_lines = list(lines)

    def init_viewer(
        self,
        initial_target_pos,
        slider_range=(-0.2, 0.2),
        rotation_slider_range=(-0.25, 0.25),
        target_axes=None,
        target_ranges=None,
        window_title="MyJoCo",
        initial_camera=(180, -20, 3, 1),
    ):
        self.hand_pose_panel = GlfwTargetPanel(
            initial_target_pos,
            axes=target_axes,
            slider_range=slider_range,
            rotation_slider_range=rotation_slider_range,
            ranges=target_ranges,
        )
        self.gui_panel = GUIPanel(initial_camera=initial_camera)

        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        self.window = glfw.create_window(1200, 900, window_title, None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)  # 창 갱신을 모니터 refresh 주기에 맞춤

        # force = np.zeros(6)
        # mujoco.mj_contactForce(self.model, self.data, i, force)

        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()  # 시각화 대상 설정 옵션
        mujoco.mjv_defaultOption(self.opt)

        # self.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
        self.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        self.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
        # self.opt.flags[mujoco.mjtVisFlag.mjVIS_BODYBVH] = True # self-collision 탐지용 바운딩박스 시각화

        self.model.vis.scale.contactwidth = 0.2
        self.model.vis.scale.contactheight = 0.03
        self.model.vis.rgba.contactpoint[:] = [1.0, 0.15, 0.05, 1.0]

        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
        self.context = mujoco.MjrContext(
            self.model,
            mujoco.mjtFontScale.mjFONTSCALE_150,
        )

        mujoco.mjv_defaultCamera(self.cam)
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.lookat[:] = [0, 0, 1.0]
        self.cam.distance = 3.0
        self.cam.azimuth = 180
        self.cam.elevation = -20

    # polling wrapper
    # panel 입력을 polling으로 가져와서 viewer option을 갱신
    def poll_target(self):
        polled_target = self.hand_pose_panel.poll_target(self.window)
        polled_camera = self.gui_panel.poll_camera(self.window)

        if polled_camera is not None:
            self.cam.azimuth = polled_camera[0]
            self.cam.elevation = polled_camera[1]
            self.cam.distance = polled_camera[2]
            self.cam.lookat[2] = polled_camera[3]

        return polled_target, polled_camera

    def render(self):
        self.width, self.height = glfw.get_framebuffer_size(self.window)
        self.viewport = mujoco.MjrRect(0, 0, self.width, self.height)

        mujoco.mjv_updateScene(
            self.model,
            self.data,
            self.opt,
            None,
            self.cam,
            mujoco.mjtCatBit.mjCAT_ALL,
            self.scene,
        )

        # 다음 화면을 그려 버퍼에 저장
        mujoco.mjr_render(self.viewport, self.scene, self.context)
        self.hand_pose_panel.render(self.window, self.context)
        self.gui_panel.render(self.window, self.context)
        if self.overlay_lines:
            mujoco.mjr_overlay(
                mujoco.mjtFont.mjFONT_NORMAL,
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                self.viewport,
                "\n".join(self.overlay_lines),
                "",
                self.context,
            )

        # glfw/opengl - double buffering
        # render()가 그린 화면을 비로소 창에 띄움
        # step -> render -> swap -> poll
        glfw.swap_buffers(self.window)
        # glfw.poll_events()

    def terminate_viewer(self):
        glfw.terminate()
