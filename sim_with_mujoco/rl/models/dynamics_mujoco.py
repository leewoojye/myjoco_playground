import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch import nn

from sim_with_mujoco.utils.dvrk_ik import get_site_transform, solve_rcm_ik
from sim_with_mujoco.utils.math3d import get_body_T


class DvrkMujocoDynamics(nn.Module):
    DIM = 7

    def __init__(self, env, dq_limit=0.045, control_steps=None):
        super().__init__()
        self.env = env
        self.model = env.model
        self.dq_limit = float(dq_limit)
        self.control_steps = env.control_steps if control_steps is None else int(control_steps)
        self.ecm_id = env.get_id(mujoco.mjtObj.mjOBJ_BODY, "ECM_tool_roll_link")
        self.jaw_joint_id = env.get_id(mujoco.mjtObj.mjOBJ_JOINT, "PSM1_jaw")
        self.jaw_actuator_id = env.get_id(mujoco.mjtObj.mjOBJ_ACTUATOR, "PSM1_jaw")
        self.jaw_qpos_id = self.model.jnt_qposadr[self.jaw_joint_id]
        self.register_buffer("centers", torch.empty(0))
        self.register_buffer("pose_weights", torch.empty(0))
        self.register_buffer("jaw_weights", torch.empty(0))

    @staticmethod
    def _ecm_T_tip(state):
        transform = np.eye(4)
        transform[:3, 3] = state[:3]
        transform[:3, :3] = Rotation.from_rotvec(state[3:6]).as_matrix().T
        return transform

    def _set_state(self, data, state):
        world_T_ecm = get_body_T(data, self.ecm_id)
        qpos = solve_rcm_ik(
            self.model,
            data,
            world_T_ecm @ self._ecm_T_tip(state),
            self.env.tip_site_id,
            self.env.rcm_pos,
            self.env.joint_ids,
            dq_limit=None,
            rcm_site_id=self.env.rcm_site_id,
        )
        for joint_id, actuator_id in zip(self.env.joint_ids, self.env.arm_actuator_ids):
            qpos_id = self.model.jnt_qposadr[joint_id]
            data.qpos[qpos_id] = qpos[qpos_id]
            data.ctrl[actuator_id] = qpos[qpos_id]
        data.qpos[self.jaw_qpos_id] = state[6]
        data.ctrl[self.jaw_actuator_id] = state[6]
        data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, data)

    def _step(self, state, action):
        data = mujoco.MjData(self.model)
        mujoco.mj_copyData(data, self.model, self.env.data)
        self._set_state(data, state)

        target_T_ecm = self._ecm_T_tip(state)
        target_T_ecm[:3, 3] = action[:3]
        target_T_ecm[:3, :3] = target_T_ecm[:3, :3] @ Rotation.from_rotvec(action[3:6]).as_matrix()
        q_des = solve_rcm_ik(
            self.model,
            data,
            get_body_T(data, self.ecm_id) @ target_T_ecm,
            self.env.tip_site_id,
            self.env.rcm_pos,
            self.env.joint_ids,
            dq_limit=self.dq_limit,
            rcm_site_id=self.env.rcm_site_id,
        )
        for joint_id, actuator_id in zip(self.env.joint_ids, self.env.arm_actuator_ids):
            qpos_id = self.model.jnt_qposadr[joint_id]
            data.ctrl[actuator_id] = np.clip(
                q_des[qpos_id],
                *self.model.actuator_ctrlrange[actuator_id],
            )
        data.ctrl[self.jaw_actuator_id] = np.clip(
            state[6] + action[6],
            *self.model.actuator_ctrlrange[self.jaw_actuator_id],
        )
        for _ in range(self.control_steps):
            mujoco.mj_step(self.model, data)

        ecm_T_tip = np.linalg.inv(get_body_T(data, self.ecm_id)) @ get_site_transform(data, self.env.tip_site_id)
        return np.r_[
            ecm_T_tip[:3, 3],
            Rotation.from_matrix(ecm_T_tip[:3, :3].T).as_rotvec(),
            data.qpos[self.jaw_qpos_id],
        ]

    @torch.no_grad()
    def forward(self, state, action):
        state, action = torch.broadcast_tensors(state, action)
        batch_shape = state.shape[:-1]
        state_np = state.detach().cpu().numpy().reshape(-1, self.DIM)
        action_np = action.detach().cpu().numpy().reshape(-1, self.DIM)
        next_state = np.stack([self._step(x, u) for x, u in zip(state_np, action_np)])
        return torch.as_tensor(next_state, dtype=state.dtype, device=state.device).reshape(*batch_shape, self.DIM)

    def update(self, state, action, next_state):
        return {}
