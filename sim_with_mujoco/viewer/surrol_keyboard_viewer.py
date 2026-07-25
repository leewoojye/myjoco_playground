import time

import glfw
import mujoco
import numpy as np


class SurrolKeyboardViewer:
    KEY_INITIAL_DELAY = 0.12
    KEY_REPEAT_INTERVAL = 1.0 / 20.0
    PSM_INCREMENT = 0.006

    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.overlay_lines = []
        self.psm_action = np.zeros(5, dtype=float)
        self.psm_action[4] = 0.5
        self._pressed_since = {}
        self._last_repeat = {}
        self._mouse_button_left = False
        self._mouse_button_middle = False
        self._mouse_button_right = False
        self._last_cursor = None

    def init_viewer(
        self,
        window_title="MyJoCo dVRK PSM Teleoperation",
        initial_camera=(145, -24, 1.0, 0.28),
        focus_position=None,
    ):
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        self.window = glfw.create_window(1200, 900, window_title, None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        glfw.set_mouse_button_callback(self.window, self._on_mouse_button)
        glfw.set_cursor_pos_callback(self.window, self._on_cursor_pos)
        glfw.set_scroll_callback(self.window, self._on_scroll)

        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        mujoco.mjv_defaultCamera(self.cam)
        mujoco.mjv_defaultOption(self.opt)
        self.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True

        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        if focus_position is None:
            self.cam.lookat[:] = self.model.stat.center
        else:
            self.cam.lookat[:] = focus_position
        self.cam.azimuth = initial_camera[0]
        self.cam.elevation = initial_camera[1]
        self.cam.distance = initial_camera[2]
        if focus_position is None and len(initial_camera) > 3:
            self.cam.lookat[2] = initial_camera[3]

        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
        self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)

    def set_overlay(self, lines):
        self.overlay_lines = list(lines)

    def poll_action(self):
        now = time.time()
        pressed_by_dim = {0: False, 1: False, 2: False}

        for key, dim, increment in (
            (glfw.KEY_W, 2, self.PSM_INCREMENT),
            (glfw.KEY_S, 2, -self.PSM_INCREMENT),
            (glfw.KEY_D, 1, self.PSM_INCREMENT),
            (glfw.KEY_A, 1, -self.PSM_INCREMENT),
            (glfw.KEY_Q, 0, self.PSM_INCREMENT),
            (glfw.KEY_E, 0, -self.PSM_INCREMENT),
        ):
            if glfw.get_key(self.window, key) == glfw.PRESS:
                pressed_by_dim[dim] = True
                if key not in self._pressed_since:
                    self._pressed_since[key] = now
                    self._last_repeat[key] = now
                    continue
                if now - self._pressed_since[key] < self.KEY_INITIAL_DELAY:
                    continue
                if now - self._last_repeat.get(key, 0.0) >= self.KEY_REPEAT_INTERVAL:
                    self.psm_action[dim] += increment
                    self._last_repeat[key] = now
            else:
                self._pressed_since.pop(key, None)
                self._last_repeat.pop(key, None)

        for dim, pressed in pressed_by_dim.items():
            if not pressed:
                self.psm_action[dim] = 0.0

        if glfw.get_key(self.window, glfw.KEY_C) == glfw.PRESS:
            self.psm_action[4] = -0.5
        else:
            self.psm_action[4] = 1.0

        return self.psm_action.copy()

    def render(self):
        width, height = glfw.get_framebuffer_size(self.window)
        viewport = mujoco.MjrRect(0, 0, width, height)

        mujoco.mjv_updateScene(
            self.model,
            self.data,
            self.opt,
            None,
            self.cam,
            mujoco.mjtCatBit.mjCAT_ALL,
            self.scene,
        )
        mujoco.mjr_render(viewport, self.scene, self.context)

        if self.overlay_lines:
            mujoco.mjr_overlay(
                mujoco.mjtFont.mjFONT_NORMAL,
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                viewport,
                "\n".join(self.overlay_lines),
                "",
                self.context,
            )

        glfw.swap_buffers(self.window)

    def terminate_viewer(self):
        glfw.terminate()

    def _on_mouse_button(self, window, button, action, mods):
        pressed = action == glfw.PRESS
        if button == glfw.MOUSE_BUTTON_LEFT:
            self._mouse_button_left = pressed
        elif button == glfw.MOUSE_BUTTON_MIDDLE:
            self._mouse_button_middle = pressed
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            self._mouse_button_right = pressed
        self._last_cursor = glfw.get_cursor_pos(window)

    def _on_cursor_pos(self, window, xpos, ypos):
        if self._last_cursor is None:
            self._last_cursor = (xpos, ypos)
            return

        dx = xpos - self._last_cursor[0]
        dy = ypos - self._last_cursor[1]
        self._last_cursor = (xpos, ypos)

        if not (self._mouse_button_left or self._mouse_button_middle or self._mouse_button_right):
            return

        width, height = glfw.get_window_size(window)
        if height == 0:
            return

        if self._mouse_button_right:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_H
        elif self._mouse_button_middle:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_V
        else:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_H

        mujoco.mjv_moveCamera(
            self.model,
            action,
            dx / height,
            dy / height,
            self.scene,
            self.cam,
        )

    def _on_scroll(self, window, xoffset, yoffset):
        mujoco.mjv_moveCamera(
            self.model,
            mujoco.mjtMouse.mjMOUSE_ZOOM,
            0.0,
            -0.05 * yoffset,
            self.scene,
            self.cam,
        )
