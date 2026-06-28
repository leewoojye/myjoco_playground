from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.optimize import lsq_linear

from sim.model.kinematics.ik import calculate_twist_error
from sim.model.math3d.lie import Adjoint


@dataclass
class DvrkIKResult:
    q_next: np.ndarray
    qvel_cmd: np.ndarray
    success: bool
    tip_error: float
    rcm_error: float
    rcm_closest: np.ndarray


def get_site_transform(data, site_id):
    transform = np.eye(4)
    transform[:3, 3] = data.site_xpos[site_id]
    transform[:3, :3] = data.site_xmat[site_id].reshape(3, 3)
    return transform


def shaft_rcm_error(model, data, shaft_start_site_id, shaft_end_site_id, rcm_pos, dof_ids=None):
    start = data.site_xpos[shaft_start_site_id].copy()
    end = data.site_xpos[shaft_end_site_id].copy()
    shaft = end - start
    denom = float(np.dot(shaft, shaft))

    if denom < 1e-12:
        closest = start
        ratio = 0.0
    else:
        ratio = float(np.dot(rcm_pos - start, shaft) / denom)
        closest = start + ratio * shaft

    error = rcm_pos - closest

    if dof_ids is None:
        return error, closest, None

    start_jacp = np.zeros((3, model.nv))
    start_jacr = np.zeros((3, model.nv))
    end_jacp = np.zeros((3, model.nv))
    end_jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, start_jacp, start_jacr, shaft_start_site_id)
    mujoco.mj_jacSite(model, data, end_jacp, end_jacr, shaft_end_site_id)

    # This is a local linearization of the closest point on the shaft line.
    # It ignores d(ratio)/dq, which is enough for the small resolved-rate steps used here.
    closest_jac = (1.0 - ratio) * start_jacp[:, dof_ids] + ratio * end_jacp[:, dof_ids]
    return error, closest, closest_jac


def solve_dvrk_rcm_ik(
    model,
    data,
    target_T,
    tip_site_id,
    shaft_start_site_id,
    shaft_end_site_id,
    rcm_pos,
    joint_ids,
    dt,
    q_home=None,
    pose_gain=18.0,
    rcm_gain=28.0,
    posture_gain=1.2,
    pose_weight=(0.25, 1.0),
    rcm_weight=8.0,
    posture_weight=0.04,
    damping=1e-3,
    dq_limit=0.025,
    qvel_limit=None,
):
    joint_ids = np.asarray(joint_ids, dtype=int)
    dof_ids = np.array([model.jnt_dofadr[jid] for jid in joint_ids], dtype=int)
    qpos_ids = model.jnt_qposadr[joint_ids]
    q_current = data.qpos[qpos_ids]

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, tip_site_id)

    orientation_weight, position_weight = pose_weight
    current_T = get_site_transform(data, tip_site_id)
    tip_position_error = target_T[:3, 3] - data.site_xpos[tip_site_id]

    J_list = [np.sqrt(position_weight) * jacp[:, dof_ids]]
    vel_list = [np.sqrt(position_weight) * pose_gain * tip_position_error]

    if orientation_weight > 0.0:
        _, tip_twist_error = calculate_twist_error(current_T, target_T)
        site_jac_body = Adjoint(np.linalg.inv(current_T)) @ np.vstack([jacr, jacp])
        J_list.append(np.sqrt(orientation_weight) * site_jac_body[:3, dof_ids])
        vel_list.append(np.sqrt(orientation_weight) * pose_gain * tip_twist_error[:3])

    rcm_error, rcm_closest, rcm_jac = shaft_rcm_error(
        model,
        data,
        shaft_start_site_id,
        shaft_end_site_id,
        rcm_pos,
        dof_ids,
    )
    rcm_scale = np.sqrt(rcm_weight)
    J_list.append(rcm_scale * rcm_jac)
    vel_list.append(rcm_scale * rcm_gain * rcm_error)

    if q_home is not None and posture_weight > 0.0:
        q_home = np.asarray(q_home)[qpos_ids]
        posture_scale = np.sqrt(posture_weight)
        J_list.append(posture_scale * np.eye(len(dof_ids)))
        vel_list.append(posture_scale * posture_gain * (q_home - q_current))

    A = np.vstack(J_list)
    b = np.hstack(vel_list)

    if damping > 0.0:
        A = np.vstack([A, damping * np.eye(len(dof_ids))])
        b = np.hstack([b, np.zeros(len(dof_ids))])

    lower = np.full(len(dof_ids), -np.inf)
    upper = np.full(len(dof_ids), np.inf)

    if dq_limit is not None:
        lower = np.maximum(lower, -dq_limit / dt)
        upper = np.minimum(upper, dq_limit / dt)

    if qvel_limit is not None:
        qvel_limit = np.asarray(qvel_limit)
        lower = np.maximum(lower, -qvel_limit)
        upper = np.minimum(upper, qvel_limit)

    for i, joint_id in enumerate(joint_ids):
        if model.jnt_limited[joint_id]:
            q_lower, q_upper = model.jnt_range[joint_id]
            lower[i] = max(lower[i], (q_lower - q_current[i]) / dt)
            upper[i] = min(upper[i], (q_upper - q_current[i]) / dt)

    result = lsq_linear(A, b, bounds=(lower, upper))
    qvel_cmd = result.x

    q_next = data.qpos.copy()
    q_next[qpos_ids] = q_current + qvel_cmd * dt

    return DvrkIKResult(
        q_next=q_next,
        qvel_cmd=qvel_cmd,
        success=bool(result.success),
        tip_error=float(np.linalg.norm(target_T[:3, 3] - current_T[:3, 3])),
        rcm_error=float(np.linalg.norm(rcm_error)),
        rcm_closest=rcm_closest,
    )
