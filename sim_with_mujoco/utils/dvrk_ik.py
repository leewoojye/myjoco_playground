import mujoco
import numpy as np

from sim_with_mujoco.utils.ik import solve_ik
from sim_with_mujoco.utils.math3d import get_body_T


DOF = 6

TOOL_T_TIP = np.array([
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [-1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])


def get_site_transform(data, site_id):
    T = np.eye(4)
    T[:3, 3] = data.site_xpos[site_id]
    T[:3, :3] = data.site_xmat[site_id].reshape(3, 3)
    return T


def _pose_transform(pose, mat, premultiply=True):
    T = pose.copy()
    if premultiply:
        return mat @ T
    return T @ mat


def _get_world_T_rcm(data, rcm_pos, rcm_site_id=None):
    T = np.eye(4)
    T[:3, 3] = rcm_pos
    if rcm_site_id is not None:
        T[:3, 3] = data.site_xpos[rcm_site_id]
        T[:3, :3] = data.site_xmat[rcm_site_id].reshape(3, 3)
    return T


def _pose_rcm2world(pose, world_T_rcm, scaling=1.0):
    pose_rcm = _pose_transform(pose, np.linalg.inv(TOOL_T_TIP), premultiply=False)
    pose_rcm[:3, 3] *= scaling
    return _pose_transform(pose_rcm, world_T_rcm)


def _pose_world2rcm(pose, world_T_rcm, scaling=1.0):
    pose_rcm = _pose_transform(pose, np.linalg.inv(world_T_rcm), premultiply=True)
    pose_rcm = pose_rcm @ TOOL_T_TIP
    pose_rcm[:3, 3] /= scaling
    return pose_rcm


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


def solve_rcm_ik(
    model,
    data,
    target_T,
    tip_site_id,
    rcm_pos,
    joint_ids,
    q_home=None,
    dq_limit=0.025,
    rcm_site_id=None,
):
    del q_home
    joint_ids = np.asarray(joint_ids, dtype=int)
    body_id = int(model.site_bodyid[tip_site_id])

    ik_data = mujoco.MjData(model)
    mujoco.mj_copyData(ik_data, model, data)
    mujoco.mj_forward(model, ik_data)

    world_T_eef = get_body_T(ik_data, body_id)
    world_T_tip = get_site_transform(ik_data, tip_site_id)
    eef_T_tip = np.linalg.inv(world_T_eef) @ world_T_tip
    tip_T_eef = np.linalg.inv(eef_T_tip)
    world_T_rcm = _get_world_T_rcm(ik_data, rcm_pos, rcm_site_id)

    # target tip pose를 RCM frame action으로 변환한 뒤 IK 계산 (SurRoL 방식)
    pose_eef = _pose_transform(target_T, tip_T_eef, premultiply=False)
    action_rcm = _pose_world2rcm(pose_eef, world_T_rcm)
    pose_world = _pose_rcm2world(action_rcm, world_T_rcm)

    joint_names = []
    for joint_id in joint_ids[:DOF]:
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(joint_id))
        if joint_name is None:
            raise ValueError(f"Unknown joint id: {joint_id}")
        joint_names.append(joint_name)

    q_next, _ = solve_ik(
        model,
        ik_data,
        body_id,
        target_T=pose_world,
        is_pose=True,
        joint_names=joint_names,
        check_collision=False,
    )

    if dq_limit is not None:
        for joint_id in joint_ids[:DOF]:
            qadr = model.jnt_qposadr[int(joint_id)]
            delta = np.clip(q_next[qadr] - data.qpos[qadr], -float(dq_limit), float(dq_limit))
            q_next[qadr] = data.qpos[qadr] + delta

    return q_next
