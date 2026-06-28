import glfw
import mujoco
import numpy as np


class GlfwTargetPanel:
    DEFAULT_AXES = ("dX", "dY", "dZ", "Roll", "Pitch", "Yaw", "thumb", "finger")

    def __init__(
        self,
        initial_target,
        axes=None,
        slider_range=(-0.2, 0.2),
        rotation_slider_range=(-0.4, 0.4),
        grasp_slider_range=(0.0, 1.0),
        ranges=None,
    ):
        self.axes = tuple(axes or self.DEFAULT_AXES)
        self.base_target = np.r_[np.asarray(initial_target, dtype=float).reshape(3), np.zeros(len(self.axes) - 3)]
        self.offset = np.zeros(len(self.axes))
        self.target = self.base_target.copy()
        self.ranges = list(ranges or ([slider_range] * 3 + [rotation_slider_range] * 3 + [grasp_slider_range] * 2))
        self.drag_index = None
        self.changed = False

    def poll_target(self, window):
        mouse_x, mouse_y = glfw.get_cursor_pos(window)
        win_w, win_h = glfw.get_window_size(window)
        fb_w, fb_h = glfw.get_framebuffer_size(window)
        mouse_x = mouse_x * fb_w / win_w
        mouse_y = (win_h - mouse_y) * fb_h / win_h

        pressed = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS

        if pressed:
            if self.drag_index is None:
                self.drag_index = self._hit_slider(mouse_x, mouse_y)
            if self.drag_index is not None:
                self._set_value(self.drag_index, mouse_x)
        else:
            self.drag_index = None

        if not self.changed:
            return None

        self.changed = False
        return self.target.copy()

    def render(self, window, context):
        width, height = glfw.get_framebuffer_size(window)

        for i, axis in enumerate(self.axes):
            y = height - 50 - i * 35
            x = width - 300

            mujoco.mjr_rectangle(
                mujoco.MjrRect(x, y, 95, 28),
                0.1,
                0.1,
                0.1,
                0.8,
            )
            mujoco.mjr_text(
                mujoco.mjtFont.mjFONT_NORMAL,
                axis,
                context,
                0.1,
                0.15,
                1.0,
                1.0,
                1.0,
            )

            mujoco.mjr_rectangle(
                mujoco.MjrRect(x + 110, y + 10, 160, 5),
                0.4,
                0.4,
                0.4,
                1.0,
            )

            lo, hi = self.ranges[i]
            t = (self.offset[i] - lo) / (hi - lo)
            knob_x = int(x + 110 + t * 160)

            mujoco.mjr_rectangle(
                mujoco.MjrRect(knob_x - 4, y + 2, 8, 20),
                1.0,
                0.6,
                0.2,
                1.0,
            )

    def _hit_slider(self, mouse_x, mouse_y):
        width, height = glfw.get_framebuffer_size(glfw.get_current_context())

        for i in range(len(self.axes)):
            y = height - 50 - i * 35
            x = width - 300
            if x + 110 <= mouse_x <= x + 270 and y <= mouse_y <= y + 28:
                return i

        return None

    def _set_value(self, index, mouse_x):
        width, _ = glfw.get_framebuffer_size(glfw.get_current_context())
        x = width - 300

        lo, hi = self.ranges[index]
        t = np.clip((mouse_x - (x + 110)) / 160, 0.0, 1.0)
        value = lo + t * (hi - lo)

        if np.isclose(self.offset[index], value):
            return

        self.offset[index] = value
        self.target = self.base_target + self.offset
        self.changed = True
