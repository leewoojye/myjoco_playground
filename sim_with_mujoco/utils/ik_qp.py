import mujoco
import numpy as np
from scipy.optimize import lsq_linear

from sim.model.kinematics.ik import calculate_twist_error
from sim_with_mujoco.utils.math3d import get_body_T


def _get_site_T(data, site_id):
    T = np.eye(4)
    T[:3, 3] = data.site_xpos[site_id]
    T[:3, :3] = data.site_xmat[site_id].reshape(3, 3)
    return T


# QP 기반 differential IK 속도 명령과 다음 관절각 반환
# input target에 대응하는 q가 아닌 시뮬레이션 시간 동안 이동한 뒤의 q를 반환
def solve_differential_ik(
    model,
    data,
    targets,
    joint_ids,
    dt,
    gain=20.0,
    damping=1e-3,
    dq_limit=0.05,
    qvel_limit=None,
    dof_map=None,
):
    if isinstance(targets, tuple):
        targets = [targets]

    joint_ids = np.asarray(joint_ids, dtype=int)
    dof_ids = np.array([model.jnt_dofadr[jid] for jid in joint_ids], dtype=int)
    qpos_ids = model.jnt_qposadr[joint_ids]
    q_current = data.qpos[qpos_ids]

    if dof_map is not None:
        dof_map = np.asarray(dof_map, dtype=float)
        expected_shape = (model.nv, len(joint_ids))
        if dof_map.shape != expected_shape:
            raise ValueError(f"dof_map must have shape {expected_shape}, got {dof_map.shape}")

    J_list = []
    vel_list = []  # 현재 오차를 줄이기 위해 e.e가 내야 할 속도 리스트

    for target in targets:
        if len(target) == 4:
            object_id, target_T, is_pose, weight = target
            object_type = mujoco.mjtObj.mjOBJ_BODY
        elif len(target) == 5:
            object_id, target_T, is_pose, weight, object_type = target
        else:
            raise ValueError(
                "Each target must be (object_id, target_T, is_pose, weight) "
                "or (object_id, target_T, is_pose, weight, object_type)."
            )

        object_id = int(object_id)
        object_type = int(object_type)
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))

        if object_type == int(mujoco.mjtObj.mjOBJ_BODY):
            mujoco.mj_jacBody(model, data, jacp, jacr, object_id)
            T_current = get_body_T(data, object_id)
        elif object_type == int(mujoco.mjtObj.mjOBJ_SITE):
            mujoco.mj_jacSite(model, data, jacp, jacr, object_id)
            T_current = _get_site_T(data, object_id)
        else:
            raise ValueError("solve_differential_ik supports body and site targets only")

        if is_pose:
            _, err = calculate_twist_error(T_current, target_T)
            # mj_jacBody/mj_jacSite return angular and point-linear velocity
            # Jacobians in the world frame.  Rotate both blocks into the
            # current body/site frame to match Log(T_current^-1 T_target).
            R_world_to_local = T_current[:3, :3].T
            J = np.block([
                [R_world_to_local, np.zeros((3, 3))],
                [np.zeros((3, 3)), R_world_to_local],
            ]) @ np.vstack([jacr, jacp])
        else:
            target_pos = target_T[:3, 3]
            err = target_pos - T_current[:3, 3]
            J = jacp

        # For a closed-chain model, dof_map projects the full MuJoCo Jacobian
        # onto the independent coordinates.  For the dVRK PSM this includes
        # the pitch_2/pitch_3 motion induced by the independent pitch joint.
        J_controlled = J[:, dof_ids] if dof_map is None else J @ dof_map

        scale = np.sqrt(weight)
        J_list.append(scale * J_controlled)
        vel_list.append(scale * gain * err)

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

    return q_next, qvel_cmd, result.success
