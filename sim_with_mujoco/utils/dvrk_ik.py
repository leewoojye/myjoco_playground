import mujoco
import numpy as np


def get_site_transform(data, site_id):
    transform = np.eye(4)
    transform[:3, 3] = data.site_xpos[site_id]
    transform[:3, :3] = data.site_xmat[site_id].reshape(3, 3)
    return transform


def _joint_id(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)


def _qpos_id(model, joint_id):
    return model.jnt_qposadr[joint_id]


def _joint_value(model, qpos, name):
    joint_id = _joint_id(model, name)
    if joint_id == -1:
        raise ValueError(f"Unknown joint: {name}")
    return qpos[_qpos_id(model, joint_id)]


def _set_joint_value(model, qpos, name, value):
    joint_id = _joint_id(model, name)
    if joint_id == -1:
        raise ValueError(f"Unknown joint: {name}")
    qpos[_qpos_id(model, joint_id)] = value


def _clip_joint_value(model, joint_id, value):
    if not model.jnt_limited[joint_id]:
        return float(value), False
    lower, upper = model.jnt_range[joint_id]
    clipped = float(np.clip(value, lower, upper))
    return clipped, not np.isclose(clipped, value)


def _rcm_frame(model, data, rcm_pos):
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "psm_rcm_site")
    transform = np.eye(4)
    transform[:3, 3] = rcm_pos
    if site_id != -1:
        transform[:3, :3] = data.site_xmat[site_id].reshape(3, 3)
    return transform


def shaft_rcm_error(model, data, shaft_start_site_id, shaft_end_site_id, rcm_pos):
    start = data.site_xpos[shaft_start_site_id].copy()
    end = data.site_xpos[shaft_end_site_id].copy()
    shaft = end - start
    denom = float(np.dot(shaft, shaft))

    if denom < 1e-12:
        closest = start
    else:
        ratio = float(np.dot(rcm_pos - start, shaft) / denom)
        closest = start + ratio * shaft

    error = rcm_pos - closest
    return error, closest, None


def _mimic_specs(joint_mimics):
    specs = []
    for mimic in joint_mimics or []:
        if len(mimic) == 3:
            passive_joint_id, driver_joint_id, multiplier = mimic
            offset = 0.0
        else:
            passive_joint_id, driver_joint_id, multiplier, offset = mimic
        specs.append((int(passive_joint_id), int(driver_joint_id), float(multiplier), float(offset)))
    return specs


def _sync_mimic_qpos(model, qpos, mimic_specs):
    for passive_joint_id, driver_joint_id, multiplier, offset in mimic_specs:
        passive_qpos_id = model.jnt_qposadr[passive_joint_id]
        driver_qpos_id = model.jnt_qposadr[driver_joint_id]
        qpos[passive_qpos_id] = offset + multiplier * qpos[driver_qpos_id]


def solve_dvrk_rcm_ik(
    model,
    data,
    target_T,
    tip_site_id,
    rcm_pos,
    joint_ids,
    q_home=None,
    dq_limit=0.025,
    joint_mimics=None,
):
    joint_ids = np.asarray(joint_ids, dtype=int)
    qpos_ids = model.jnt_qposadr[joint_ids]

    q_initial = data.qpos.copy()
    posture_qpos = q_initial if q_home is None else np.asarray(q_home)
    q_next = q_initial.copy()

    mujoco.mj_forward(model, data)
    world_T_rcm = _rcm_frame(model, data, rcm_pos)
    rcm_R_world = world_T_rcm[:3, :3].T
    target_tip_rcm = rcm_R_world @ (target_T[:3, 3] - world_T_rcm[:3, 3])
    current_tip_rcm = rcm_R_world @ (data.site_xpos[tip_site_id] - world_T_rcm[:3, 3])

    insertion = _joint_value(model, q_initial, "psm_insertion")
    tip_offset = float(np.linalg.norm(current_tip_rcm) - insertion)
    if not np.isfinite(tip_offset) or tip_offset < 0.0 or tip_offset > 0.04:
        tip_offset = 0.0037

    target_length = float(np.linalg.norm(target_tip_rcm))
    if target_length < max(tip_offset, 1e-8):
        target_length = max(tip_offset, 1e-8)

    yaw_des = np.arctan2(target_tip_rcm[0], -target_tip_rcm[2])
    pitch_arg = np.clip(-target_tip_rcm[1] / target_length, -1.0, 1.0)
    pitch_des = np.arcsin(pitch_arg)
    insertion_des = target_length - tip_offset

    psm_solution = {
        "psm_yaw": yaw_des,
        "psm_pitch": pitch_des,
        "psm_insertion": insertion_des,
    }

    for name, value in psm_solution.items():
        joint_id = _joint_id(model, name)
        if joint_id == -1:
            continue
        clipped, _ = _clip_joint_value(model, joint_id, value)
        _set_joint_value(model, q_next, name, clipped)

    for name in ("psm_roll", "psm_wrist_pitch", "psm_wrist_yaw"):
        joint_id = _joint_id(model, name)
        if joint_id == -1:
            continue
        desired = posture_qpos[_qpos_id(model, joint_id)]
        clipped, _ = _clip_joint_value(model, joint_id, desired)
        _set_joint_value(model, q_next, name, clipped)

    if dq_limit is not None:
        for i, joint_id in enumerate(joint_ids):
            if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_SLIDE:
                continue
            qpos_id = qpos_ids[i]
            q_next[qpos_id] = q_initial[qpos_id] + np.clip(
                q_next[qpos_id] - q_initial[qpos_id],
                -float(dq_limit),
                float(dq_limit),
            )

    _sync_mimic_qpos(model, q_next, _mimic_specs(joint_mimics))
    return q_next
