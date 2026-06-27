from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


WALKER_ACTUATORS = (
    "left_hip_pos",
    "left_knee_pos",
    "left_ankle_pos",
    "right_hip_pos",
    "right_knee_pos",
    "right_ankle_pos",
)


@dataclass(frozen=True)
class WalkerState:
    root_x: float
    root_z: float
    root_pitch: float
    root_xvel: float
    root_zvel: float
    root_pitch_vel: float


@dataclass
class WalkingMetrics:
    distance: float
    mean_speed: float
    alive: bool
    final_height: float
    final_pitch: float
    energy: float


def actuator_ids(model: mujoco.MjModel, names: tuple[str, ...] = WALKER_ACTUATORS) -> np.ndarray:
    ids = []
    for name in names:
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if actuator_id < 0:
            raise ValueError(f"Missing actuator: {name}")
        ids.append(actuator_id)
    return np.asarray(ids, dtype=int)


def read_walker_state(data: mujoco.MjData) -> WalkerState:
    return WalkerState(
        root_x=float(data.qpos[0]),
        root_z=float(data.qpos[1]),
        root_pitch=float(data.qpos[2]),
        root_xvel=float(data.qvel[0]),
        root_zvel=float(data.qvel[1]),
        root_pitch_vel=float(data.qvel[2]),
    )


def is_alive(state: WalkerState) -> bool:
    return 0.55 < state.root_z < 1.25 and abs(state.root_pitch) < 0.55


class CPGWalkingPolicy:
    """Small central-pattern-generator policy for the toy planar walker.

    The action is a vector of joint position targets. This is intentionally not a
    trained controller; it is a deterministic policy baseline that can later be
    replaced by an RL policy with the same six-dimensional action interface.
    """

    def __init__(
        self,
        step_frequency: float = 1.5,
        hip_amplitude: float = 0.52,
        knee_amplitude: float = 0.60,
        ankle_amplitude: float = 0.20,
        forward_lean: float = 0.20,
        target_speed: float = 0.55,
    ) -> None:
        self.step_frequency = step_frequency
        self.hip_amplitude = hip_amplitude
        self.knee_amplitude = knee_amplitude
        self.ankle_amplitude = ankle_amplitude
        self.forward_lean = forward_lean
        self.target_speed = target_speed

    def target(self, time: float, state: WalkerState) -> np.ndarray:
        phase = 2.0 * np.pi * self.step_frequency * time
        left_phase = np.sin(phase)
        right_phase = -left_phase

        pitch_feedback = -0.75 * (state.root_pitch - self.forward_lean) - 0.12 * state.root_pitch_vel
        speed_feedback = 0.16 * (self.target_speed - state.root_xvel)
        stance_bias = np.clip(pitch_feedback + speed_feedback, -0.22, 0.22)

        left = self._leg_targets(left_phase, stance_bias)
        right = self._leg_targets(right_phase, stance_bias)
        return np.asarray((*left, *right), dtype=np.float64)

    def _leg_targets(self, phase_signal: float, stance_bias: float) -> tuple[float, float, float]:
        swing = max(phase_signal, 0.0)
        stance = max(-phase_signal, 0.0)

        hip = -self.hip_amplitude * phase_signal + stance_bias
        knee = 0.16 + self.knee_amplitude * swing + 0.10 * stance
        ankle = -0.10 - self.ankle_amplitude * phase_signal - 0.35 * hip - 0.16 * knee

        return (
            float(np.clip(hip, -0.75, 0.65)),
            float(np.clip(knee, 0.05, 1.05)),
            float(np.clip(ankle, -0.55, 0.45)),
        )


class StandingPolicy:
    def target(self, time: float, state: WalkerState) -> np.ndarray:
        del time, state
        return np.asarray((0.08, 0.18, -0.12, 0.08, 0.18, -0.12), dtype=np.float64)


def compute_energy(model: mujoco.MjModel, data: mujoco.MjData, ids: np.ndarray) -> float:
    joint_ids = model.actuator_trnid[ids, 0]
    dof_ids = model.jnt_dofadr[joint_ids]
    return float(np.sum(np.abs(data.actuator_force[ids] * data.qvel[dof_ids])))
