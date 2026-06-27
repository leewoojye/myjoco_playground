from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import mujoco
import numpy as np


THIS_DIR = Path(__file__).resolve().parent
CPG_DIR = THIS_DIR.parent / "01_cpg_residual_natural_walker"
sys.path.insert(0, str(CPG_DIR))

from cpg_residual_core import (  # noqa: E402
    ASSIST_ACTUATORS,
    DEFAULT_XML,
    WALKER_ACTUATORS,
    CPGParams,
    ResidualParams,
    cpg_targets,
    named_actuator_ids,
    reset_model,
    walker_state,
)


STAND_TARGET = np.asarray((0.08, 0.18, -0.12, 0.08, 0.18, -0.12), dtype=np.float64)


@dataclass(frozen=True)
class Segment:
    mode: str
    start: float
    end: float


@dataclass
class TransitionMetrics:
    distance: float
    alive: bool
    final_height: float
    final_pitch: float
    energy: float
    snap_cost: float


DEFAULT_SCHEDULE = (
    Segment("stand", 0.0, 1.0),
    Segment("start_walk", 1.0, 3.0),
    Segment("walk", 3.0, 6.5),
    Segment("recover", 6.5, 7.5),
    Segment("stop", 7.5, 9.0),
    Segment("stand", 9.0, 10.0),
)


def smoothstep(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def segment_at(time_s: float, schedule: tuple[Segment, ...]) -> Segment:
    for segment in schedule:
        if segment.start <= time_s < segment.end:
            return segment
    return schedule[-1]


def transition_target(time_s: float, state: np.ndarray, schedule: tuple[Segment, ...]) -> np.ndarray:
    segment = segment_at(time_s, schedule)
    phase = (time_s - segment.start) / max(segment.end - segment.start, 1e-6)
    cpg = cpg_targets(time_s, state, CPGParams(target_speed=0.32), ResidualParams())

    if segment.mode == "stand":
        blend = 0.0
    elif segment.mode == "start_walk":
        blend = smoothstep(phase)
    elif segment.mode == "walk":
        blend = 1.0
    elif segment.mode == "stop":
        blend = 1.0 - smoothstep(phase)
    elif segment.mode == "recover":
        # Slightly crouched, slower CPG target to absorb disturbance.
        recover = cpg_targets(
            time_s,
            state,
            CPGParams(step_frequency=1.2, target_speed=0.18),
            ResidualParams(crouch_delta=0.12, lean_delta=-0.08),
        )
        return 0.25 * STAND_TARGET + 0.75 * recover
    else:
        raise ValueError(f"Unknown segment mode: {segment.mode}")

    return (1.0 - blend) * STAND_TARGET + blend * cpg


def run_transition(
    *,
    xml_path: Path = DEFAULT_XML,
    schedule: tuple[Segment, ...] = DEFAULT_SCHEDULE,
    push_time: float | None = 6.55,
    push_root_pitch_velocity: float = -1.25,
    sample_dt: float = 0.03,
) -> tuple[TransitionMetrics, dict[str, np.ndarray]]:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    reset_model(model, data)

    leg_ids = named_actuator_ids(model, WALKER_ACTUATORS)
    assist_ids = named_actuator_ids(model, ASSIST_ACTUATORS)
    start_x = float(data.qpos[0])
    duration = schedule[-1].end
    next_sample = 0.0
    pushed = False
    prev_target = STAND_TARGET.copy()
    energy = 0.0
    snap_cost = 0.0
    dt = float(model.opt.timestep)

    samples = {"time": [], "state": [], "target": [], "mode_id": []}
    mode_to_id = {mode: idx for idx, mode in enumerate(sorted({seg.mode for seg in schedule}))}

    while data.time < duration:
        if push_time is not None and not pushed and data.time >= push_time:
            data.qvel[2] += push_root_pitch_velocity
            pushed = True

        state = walker_state(data)
        target = transition_target(float(data.time), state, schedule)
        data.ctrl[leg_ids] = target
        data.ctrl[assist_ids[0]] = 0.08
        data.ctrl[assist_ids[1]] = 0.86
        mujoco.mj_step(model, data)

        joint_ids = model.actuator_trnid[leg_ids, 0]
        dof_ids = model.jnt_dofadr[joint_ids]
        energy += float(np.sum(np.abs(data.actuator_force[leg_ids] * data.qvel[dof_ids]))) * dt
        snap_cost += float(np.sum((target - prev_target) ** 2))
        prev_target = target.copy()

        if data.time >= next_sample:
            segment = segment_at(float(data.time), schedule)
            samples["time"].append(float(data.time))
            samples["state"].append(walker_state(data).copy())
            samples["target"].append(target.copy())
            samples["mode_id"].append(mode_to_id[segment.mode])
            next_sample += sample_dt

    alive = bool(0.55 < data.qpos[1] < 1.25 and abs(data.qpos[2]) < 0.55)
    metrics = TransitionMetrics(
        distance=float(data.qpos[0] - start_x),
        alive=alive,
        final_height=float(data.qpos[1]),
        final_pitch=float(data.qpos[2]),
        energy=energy,
        snap_cost=snap_cost,
    )
    arrays = {key: np.asarray(value) for key, value in samples.items()}
    return metrics, arrays
