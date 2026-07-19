# dm_control Physics / robosuite MujocoEnv

import mujoco
import numpy as np
from sim.model.kinematics.ik import calculate_twist_error
from sim_with_mujoco.mjcf.parser import parser
from sim_with_mujoco.utils.ik import damped_pseudoinverse
from sim_with_mujoco.utils.math3d import get_body_T
from sim_with_mujoco.utils.mj import dof_ids_from_joints
from sim_with_mujoco.viewer.viewer import Viewer


class Environment:
    # MjModel, MjData
    # body/joint/geom/site/sensor name
    # qpos/qvel/ctrl (named access 기능 추가)
    # getter, setter: tick/data.time
    # forward/step/reset/render etc. wrapper
    # IK solver (보류)
    # contact points 가져오기 (렌더링용)
    # e.e의 site/body id field

    # viewer, environment(simulator state management) 분리
    # main.py는 렌더링 루프, 시뮬레이션 루프 이중 반복문 구조
    # main.py에서 view(mujoco viewer, glfw panel), env 인스턴스 생성 -> 매 렌더링마다 panel state polling -> polled target으로 ik solver 호출 -> 목표 관절각 env.forward() -> ...

    def __init__(self, xml_path, end_effector, secondary_body=None):
        self.model, self.data = parser(xml_path)
        self.ee_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            end_effector,
        )
        if self.ee_body_id == -1:
            raise ValueError(f"Unknown end effector body: {end_effector}")
        self.ee_body_name = end_effector
        self.secondary_body_name = secondary_body
        self.left_hand_id = -1
        self.left_initial_T = None
        self.viewer = Viewer(self.model, self.data)
        # self.left_hand_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "arm_l_link7")  # 추후 수정
        # self.left_initial_T = get_body_T(self.data, self.left_hand_id)
        self.tick = 0

    def get_ctrl(self):
        return self.data.ctrl

    # env.set_control() -> env.step() 흐름
    def set_ctrl(self, ctrl):
        np.copyto(self.data.ctrl, ctrl)
        return

    # position/motor 액추에이터 ctrl 설정
    def set_joint(self, joint_name, q_des, is_kinematic=False, kp=2.0, kd=0.2):
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)

        qadr = self.model.jnt_qposadr[joint_id]
        dadr = self.model.jnt_dofadr[joint_id]

        if is_kinematic: # 키네마틱 모드면 data.qpos만 처리
            self.data.qpos[qadr] = q_des
            return

        if self.model.actuator_biastype[actuator_id] != mujoco.mjtBias.mjBIAS_NONE:
            ctrl = q_des
        else: # 추후 수정
            ctrl = kp * (q_des - self.data.qpos[qadr]) - kd * self.data.qvel[dadr]

        if self.model.actuator_ctrllimited[actuator_id]:
            lo, hi = self.model.actuator_ctrlrange[actuator_id]
            ctrl = np.clip(ctrl, lo, hi)
        self.data.ctrl[actuator_id] = ctrl

    def initial_qpos(self, q_des: dict):  # qpos 초기화, qpos/ctrl 모두 고려
        # self.data.qpos[:] = qpos
        # mujoco.mj_forward(self.model, self.data)

        for name, value in q_des.items():
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id == -1:
                raise ValueError(f"Unknown joint in initial_qpos: {name}")
            actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            qadr = self.model.jnt_qposadr[joint_id]
            self.data.qpos[qadr] = value
            if actuator_id != -1:
                self.data.ctrl[actuator_id] = self.data.qpos[qadr]

        # qpos/qvel/ctrl 기준으로 kinematics + velocity, force, qacc 등등 계산
        mujoco.mj_forward(self.model, self.data)

        # 초기 목표 위치 및 자세 저장
        self.initial_target_pos = self.data.xpos[self.ee_body_id].copy()
        self.initial_pose = get_body_T(self.data, self.ee_body_id)
        self.initial_q = self.data.qpos.copy()

        secondary_body = self.secondary_body_name or "arm_l_link7"
        self.left_hand_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, secondary_body)
        if self.left_hand_id != -1:
            self.left_initial_T = get_body_T(self.data, self.left_hand_id)

    # step() wrapper
    def step(self, nstep=1):
        mujoco.mj_step(self.model, self.data, nstep)
        return

    def render(self):  # rendering wrapper
        # 렌더링 주기 - 시뮬레이션 주기 분리
        # viewer 인스턴스의 render api 호출해서 window buffer 업데이트
        # 렌더링 로직은 viewer 인스턴스에서 전담하고 env.render()는 wrapper 용도
        self.viewer.render()
        return

    def get_state():
        return

    def set_state():
        return

    # data.time
    def get_time(self):
        return self.data.time

    # def set_time(): # mj_step()에서 관리
    #     return

    # def get_tick(self):
    #     return self.tick

    # def set_tick():
    #     return

    # def inc_tick(self):
    #     self.tick = self.tick + 1

    #     return self.tick

    # transformation matrix getter: data.xpos + data.xmat
    # def get_space_T(self, site_id):
    #     T = np.eye(4)  # 4x4 단위행렬 생성
    #     T[:3, :3] = self.data.xpos[site_id]
    #     T[:3, 3] = self.data.xmat[site_id].reshape(3, 3)
    #     return T

    # def get_body_T(self, body_id):
    #     T = np.eye(4)  # 4x4 단위행렬 생성
    #     T[:3, :3] = self.data.xpos[body_id]
    #     T[:3, 3] = self.data.xmat[body_id].reshape(3, 3)
    #     return T

    # rotation matrix getter
    def get_rotation():
        R = np.eye(3)
        return

    def reset():
        return

    def forward(self, qpos):
        # ik solver 호출 (보류)
        self.data.qpos = qpos
        mujoco.mj_forward(self.model, self.data)

        return

    def ik_wrapper():
        return

    # pose(6D) 기준 task-space x를 joint space q로 변환
    # 함수명 수정 예정
    def task_to_joint_space(self, twist_des, twistdot_des, joint_ids):
        dof_ids = dof_ids_from_joints(self.model, joint_ids)

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        jacp_dot = np.zeros((3, self.model.nv))
        jacr_dot = np.zeros((3, self.model.nv))

        ee_pos = self.data.xpos[self.ee_body_id].copy()

        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.ee_body_id)
        mujoco.mj_jacDot(self.model, self.data, jacp_dot, jacr_dot, ee_pos, self.ee_body_id)

        J = np.vstack([jacr, jacp])[:, dof_ids]
        J_dot = np.vstack([jacr_dot, jacp_dot])[:, dof_ids]

        qvel = self.data.qvel[dof_ids]  # qdot

        J_inv = damped_pseudoinverse(J)

        qdot_des = J_inv @ twist_des
        qddot_des = J_inv @ (twistdot_des - J_dot @ qvel)  # qdot_des를 미분해서 전개한 식

        return qdot_des, qddot_des

    def get_twist_error(self, target_T):
        T = np.eye(4)
        T[:3, 3] = self.data.xpos[self.ee_body_id]
        T[:3, :3] = self.data.xmat[self.ee_body_id].reshape(3, 3)
        _, twist_error = calculate_twist_error(T, target_T)

        return twist_error


import gymnasium as gym
from gymnasium import spaces

from sim_with_mujoco.utils.dvrk_ik import get_site_transform, solve_dvrk_rcm_ik
from sim_with_mujoco.utils.mj import joint_ids_from_names


class DvrkEnv(gym.Env):
    """Minimal Gymnasium environment for dVRK PSM needle reaching."""

    JOINT_NAMES = (
        "psm_yaw",
        "psm_pitch",
        "psm_insertion",
        "psm_roll",
        "psm_wrist_pitch",
        "psm_wrist_yaw",
    )
    INITIAL_QPOS = {
        "psm_yaw": 0.18,
        "psm_pitch": 0.08,
        "psm_insertion": 0.07,
        "psm_roll": 0.0,
        "psm_wrist_pitch": 0.0,
        "psm_wrist_yaw": 0.0,
        "psm_jaw": 0.15,
    }
    MIMICS = (
        ("psm_pitch_2", "psm_pitch", -1.0),
        ("psm_pitch_3", "psm_pitch", 1.0),
        ("psm_pitch_back", "psm_pitch", 1.0),
        ("psm_pitch_bottom", "psm_pitch", -1.0),
        ("psm_pitch_top", "psm_pitch", -1.0),
        ("psm_pitch_front", "psm_pitch", 1.0),
        ("psm_jaw_1", "psm_jaw", 0.5),
        ("psm_jaw_2", "psm_jaw", 0.5),
    )

    def __init__(self, xml_path, action_scale=0.004, max_steps=100, tolerance=0.008):
        super().__init__()
        self.plant = Environment(xml_path, "psm_tool_tip")
        self.model, self.data = self.plant.model, self.plant.data
        self.action_scale = float(action_scale)
        self.max_steps = int(max_steps)
        self.tolerance = float(tolerance)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32)

        self.tip_site_id = self._id(mujoco.mjtObj.mjOBJ_SITE, "psm_tool_tip_site")
        self.target_site_id = self._id(mujoco.mjtObj.mjOBJ_SITE, "needle_reach_target")
        self.rcm_site_id = self._id(mujoco.mjtObj.mjOBJ_SITE, "psm_rcm_site")
        self.joint_ids = joint_ids_from_names(self.model, self.JOINT_NAMES)
        self.mimics = []
        for passive, driver, multiplier in self.MIMICS:
            passive_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, passive)
            driver_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, driver)
            if passive_id != -1 and driver_id != -1:
                self.mimics.append((passive_id, driver_id, multiplier, 0.0))
        self.step_count = 0

    def _id(self, object_type, name):
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id == -1:
            raise ValueError(f"Unknown MuJoCo object: {name}")
        return object_id

    def _sync_mimics(self):
        for passive_id, driver_id, multiplier, offset in self.mimics:
            self.data.qpos[self.model.jnt_qposadr[passive_id]] = (
                offset + multiplier * self.data.qpos[self.model.jnt_qposadr[driver_id]]
            )

    def _observation(self):
        error = self.data.site_xpos[self.target_site_id] - self.data.site_xpos[self.tip_site_id]
        return error.astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.plant.initial_qpos(self.INITIAL_QPOS)
        self._sync_mimics()
        mujoco.mj_forward(self.model, self.data)
        self.rcm_pos = self.data.site_xpos[self.rcm_site_id].copy()
        self.step_count = 0
        return self._observation(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        target_T = get_site_transform(self.data, self.tip_site_id)
        target_T[:3, 3] += self.action_scale * action
        q_des = solve_dvrk_rcm_ik(
            self.model,
            self.data,
            target_T,
            self.tip_site_id,
            self.rcm_pos,
            self.joint_ids,
            dq_limit=0.035,
            joint_mimics=self.mimics,
        )
        for joint_id in self.joint_ids:
            qpos_id = self.model.jnt_qposadr[joint_id]
            self.data.qpos[qpos_id] = q_des[qpos_id]
        self._sync_mimics()
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.data.time += self.model.opt.timestep

        self.step_count += 1
        observation = self._observation()
        tip_error = float(np.linalg.norm(observation))
        terminated = tip_error <= self.tolerance
        truncated = self.step_count >= self.max_steps
        return observation, -tip_error, terminated, truncated, {"tip_error": tip_error}
