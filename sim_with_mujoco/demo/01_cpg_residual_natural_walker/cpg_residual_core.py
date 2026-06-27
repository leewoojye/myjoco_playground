from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_XML = THIS_DIR / "toy_walker.xml"
WALKER_ACTUATORS = (
    "left_hip_pos",
    "left_knee_pos",
    "left_ankle_pos",
    "right_hip_pos",
    "right_knee_pos",
    "right_ankle_pos",
)
ASSIST_ACTUATORS = ("torso_pitch_assist", "torso_height_assist")


@dataclass
class CPGParams:
    step_frequency: float = 1.5
    hip_amplitude: float = 0.52
    knee_amplitude: float = 0.60
    ankle_amplitude: float = 0.20
    forward_lean: float = 0.20
    target_speed: float = 0.35


@dataclass
class ResidualParams:
    freq_delta: float = 0.0
    hip_delta: float = 0.0
    knee_delta: float = 0.0
    ankle_delta: float = 0.0
    lean_delta: float = 0.0
    phase_delta: float = 0.0
    crouch_delta: float = 0.0

    @classmethod
    def from_vector(cls, vector: np.ndarray) -> "ResidualParams":
        values = np.asarray(vector, dtype=float).reshape(7)
        return cls(*values.tolist())

    def to_vector(self) -> np.ndarray:
        return np.asarray(list(asdict(self).values()), dtype=float)


@dataclass
class RolloutMetrics:
    reward: float
    distance: float
    mean_speed: float
    alive_time: float
    alive: bool
    energy: float
    smoothness: float
    pitch_cost: float
    final_height: float
    final_pitch: float


def load_model(xml_path: Path = DEFAULT_XML) -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    reset_model(model, data)
    return model, data


def reset_model(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)


def named_actuator_ids(model: mujoco.MjModel, names: tuple[str, ...]) -> np.ndarray:
    ids = []
    for name in names:
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if actuator_id < 0:
            raise ValueError(f"Missing actuator: {name}")
        ids.append(actuator_id)
    return np.asarray(ids, dtype=int)


def walker_state(data: mujoco.MjData) -> np.ndarray:
    return np.asarray(
        [
            data.qpos[0],
            data.qpos[1],
            data.qpos[2],
            data.qvel[0],
            data.qvel[1],
            data.qvel[2],
        ],
        dtype=np.float64,
    )


def is_alive(data: mujoco.MjData) -> bool:
    return 0.55 < data.qpos[1] < 1.25 and abs(data.qpos[2]) < 0.55


def cpg_targets(time_s: float, state: np.ndarray, base: CPGParams, residual: ResidualParams) -> np.ndarray:
    freq = np.clip(base.step_frequency + residual.freq_delta, 0.5, 2.4)
    hip_amp = np.clip(base.hip_amplitude + residual.hip_delta, 0.10, 0.85)
    knee_amp = np.clip(base.knee_amplitude + residual.knee_delta, 0.10, 1.00)
    ankle_amp = np.clip(base.ankle_amplitude + residual.ankle_delta, 0.02, 0.45)
    lean = np.clip(base.forward_lean + residual.lean_delta, -0.05, 0.30)

    phase = 2.0 * np.pi * freq * time_s
    left_phase = np.sin(phase + residual.phase_delta)
    right_phase = -np.sin(phase - residual.phase_delta)

    pitch = state[2]
    pitch_vel = state[5]
    xvel = state[3]
    pitch_feedback = -0.75 * (pitch - lean) - 0.12 * pitch_vel
    speed_feedback = 0.12 * (base.target_speed - xvel)
    stance_bias = float(np.clip(pitch_feedback + speed_feedback, -0.22, 0.22))

    left = _leg_targets(left_phase, hip_amp, knee_amp, ankle_amp, stance_bias, residual.crouch_delta)
    right = _leg_targets(right_phase, hip_amp, knee_amp, ankle_amp, stance_bias, residual.crouch_delta)
    return np.asarray((*left, *right), dtype=np.float64)


def _leg_targets(
    phase_signal: float,
    hip_amp: float,
    knee_amp: float,
    ankle_amp: float,
    stance_bias: float,
    crouch_delta: float,
) -> tuple[float, float, float]:
    swing = max(phase_signal, 0.0)
    stance = max(-phase_signal, 0.0)
    hip = -hip_amp * phase_signal + stance_bias
    knee = 0.16 + crouch_delta + knee_amp * swing + 0.10 * stance
    ankle = -0.10 - ankle_amp * phase_signal - 0.35 * hip - 0.16 * knee
    return (
        float(np.clip(hip, -0.75, 0.65)),
        float(np.clip(knee, 0.05, 1.15)),
        float(np.clip(ankle, -0.55, 0.45)),
    )


def rollout(
    residual: ResidualParams | None = None,
    *,
    xml_path: Path = DEFAULT_XML,
    duration: float = 8.0,
    base: CPGParams | None = None,
    assist: bool = True,
    pitch_target: float = 0.08,
    height_target: float = 0.86,
) -> RolloutMetrics:
    base = base or CPGParams()
    residual = residual or ResidualParams()
    model, data = load_model(xml_path)
    leg_ids = named_actuator_ids(model, WALKER_ACTUATORS)
    assist_ids = named_actuator_ids(model, ASSIST_ACTUATORS)
    if not assist:
        model.actuator_forcerange[assist_ids, :] = 0.0

    start_x = float(data.qpos[0])
    prev_target = None
    prev_delta = np.zeros(len(leg_ids))
    energy = 0.0
    smoothness = 0.0
    pitch_cost = 0.0
    alive_time = 0.0
    dt = float(model.opt.timestep)

    while data.time < duration:
        target = cpg_targets(float(data.time), walker_state(data), base, residual)
        data.ctrl[leg_ids] = target
        if assist:
            data.ctrl[assist_ids[0]] = pitch_target
            data.ctrl[assist_ids[1]] = height_target

        if prev_target is not None:
            delta = target - prev_target
            smoothness += float(np.sum((delta - prev_delta) ** 2))
            prev_delta = delta
        prev_target = target.copy()

        mujoco.mj_step(model, data)
        joint_ids = model.actuator_trnid[leg_ids, 0]
        dof_ids = model.jnt_dofadr[joint_ids]
        energy += float(np.sum(np.abs(data.actuator_force[leg_ids] * data.qvel[dof_ids]))) * dt
        pitch_cost += float(data.qpos[2] ** 2 + 0.04 * data.qvel[2] ** 2) * dt

        if is_alive(data):
            alive_time = float(data.time)

    distance = float(data.qpos[0] - start_x)
    mean_speed = distance / max(duration, 1e-6)
    alive = bool(is_alive(data))
    reward = (
        8.0 * distance
        + 2.0 * min(alive_time / duration, 1.0)
        - 2.0 * abs(mean_speed - base.target_speed)
        - 0.0015 * energy
        - 0.06 * smoothness
        - 2.0 * pitch_cost
    )
    if not alive:
        reward -= 2.0

    return RolloutMetrics(
        reward=reward,
        distance=distance,
        mean_speed=mean_speed,
        alive_time=alive_time,
        alive=alive,
        energy=energy,
        smoothness=smoothness,
        pitch_cost=pitch_cost,
        final_height=float(data.qpos[1]),
        final_pitch=float(data.qpos[2]),
    )
