from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_BERKELEY_XML = THIS_DIR / "assets" / "mujoco_menagerie" / "berkeley_humanoid" / "scene.xml"
BERKELEY_ACTUATORS = (
    "LL_HR",
    "LL_HAA",
    "LL_HFE",
    "LL_KFE",
    "LL_FFE",
    "LL_FAA",
    "LR_HR",
    "LR_HAA",
    "LR_HFE",
    "LR_KFE",
    "LR_FFE",
    "LR_FAA",
)


@dataclass
class BerkeleyCPGParams:
    step_frequency: float = 1.4
    hip_amplitude: float = 0.42
    knee_amplitude: float = 0.15
    ankle_amplitude: float = 0.14
    lateral_amplitude: float = 0.035
    forward_lean: float = 0.0
    target_height: float = 0.515


@dataclass
class BerkeleyResidual:
    freq_delta: float = 0.0
    hip_delta: float = 0.0
    knee_delta: float = 0.0
    ankle_delta: float = 0.0
    lean_delta: float = 0.0
    phase_delta: float = 0.0
    lateral_delta: float = 0.0

    @classmethod
    def from_vector(cls, vector: np.ndarray) -> "BerkeleyResidual":
        values = np.asarray(vector, dtype=float).reshape(7)
        return cls(*values.tolist())


@dataclass
class BerkeleyMetrics:
    distance: float
    mean_speed: float
    alive: bool
    final_height: float
    final_tilt: float
    energy: float


def load_berkeley(xml_path: Path = DEFAULT_BERKELEY_XML) -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return model, data


def named_actuator_ids(model: mujoco.MjModel, names: tuple[str, ...] = BERKELEY_ACTUATORS) -> np.ndarray:
    ids = []
    for name in names:
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if actuator_id < 0:
            raise ValueError(f"Missing actuator: {name}")
        ids.append(actuator_id)
    return np.asarray(ids, dtype=int)


def berkeley_targets(
    time_s: float,
    home_ctrl: np.ndarray,
    params: BerkeleyCPGParams | None = None,
    residual: BerkeleyResidual | None = None,
) -> np.ndarray:
    params = params or BerkeleyCPGParams()
    residual = residual or BerkeleyResidual()
    freq = np.clip(params.step_frequency + residual.freq_delta, 0.5, 2.2)
    hip_amp = np.clip(params.hip_amplitude + residual.hip_delta, 0.08, 0.55)
    knee_amp = np.clip(params.knee_amplitude + residual.knee_delta, 0.02, 0.55)
    ankle_amp = np.clip(params.ankle_amplitude + residual.ankle_delta, 0.02, 0.35)
    lateral_amp = np.clip(params.lateral_amplitude + residual.lateral_delta, 0.0, 0.12)
    lean = np.clip(params.forward_lean + residual.lean_delta, -0.15, 0.15)

    phase = 2.0 * np.pi * freq * time_s
    signal = np.sin(phase + residual.phase_delta)
    left_swing = max(signal, 0.0)
    right_swing = max(-signal, 0.0)

    target = home_ctrl.copy()
    target[1] = home_ctrl[1] + lateral_amp * signal
    target[7] = home_ctrl[7] + lateral_amp * signal

    target[2] = home_ctrl[2] + hip_amp * signal + lean
    target[3] = home_ctrl[3] + knee_amp * left_swing
    target[4] = home_ctrl[4] - ankle_amp * signal

    target[8] = home_ctrl[8] - hip_amp * signal + lean
    target[9] = home_ctrl[9] + knee_amp * right_swing
    target[10] = home_ctrl[10] + ankle_amp * signal
    return np.clip(target, -2.5, 2.5)


def apply_pelvis_stabilizer(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    target_height: float,
    height_kp: float = 1000.0,
    height_kd: float = 80.0,
    tilt_kp: float = 120.0,
    tilt_kd: float = 12.0,
) -> None:
    # Demo scaffold: the Menagerie Berkeley model needs a trained balance policy
    # for real locomotion. This virtual root assist keeps the CPG preview upright.
    data.qfrc_applied[:] = 0.0
    data.qfrc_applied[:6] = data.qfrc_bias[:6]
    data.qfrc_applied[2] += height_kp * (target_height - data.qpos[2]) - height_kd * data.qvel[2]
    data.qfrc_applied[3:6] += -tilt_kp * data.qpos[4:7] - tilt_kd * data.qvel[3:6]


def is_berkeley_alive(data: mujoco.MjData) -> bool:
    return bool(0.35 < data.qpos[2] < 0.75 and np.linalg.norm(data.qpos[4:7]) < 0.35)


def berkeley_rollout(
    residual: BerkeleyResidual | None = None,
    *,
    xml_path: Path = DEFAULT_BERKELEY_XML,
    duration: float = 8.0,
    params: BerkeleyCPGParams | None = None,
    stabilize: bool = True,
) -> BerkeleyMetrics:
    params = params or BerkeleyCPGParams()
    residual = residual or BerkeleyResidual()
    model, data = load_berkeley(xml_path)
    ids = named_actuator_ids(model)
    home = data.ctrl.copy()
    start_x = float(data.qpos[0])
    energy = 0.0
    dt = float(model.opt.timestep)

    while data.time < duration:
        data.ctrl[ids] = berkeley_targets(float(data.time), home, params, residual)
        if stabilize:
            apply_pelvis_stabilizer(model, data, target_height=params.target_height)
        mujoco.mj_step(model, data)
        joint_ids = model.actuator_trnid[ids, 0]
        dof_ids = model.jnt_dofadr[joint_ids]
        energy += float(np.sum(np.abs(data.actuator_force[ids] * data.qvel[dof_ids]))) * dt

    distance = float(data.qpos[0] - start_x)
    return BerkeleyMetrics(
        distance=distance,
        mean_speed=distance / max(duration, 1e-6),
        alive=is_berkeley_alive(data),
        final_height=float(data.qpos[2]),
        final_tilt=float(np.linalg.norm(data.qpos[4:7])),
        energy=energy,
    )
