import mujoco
import numpy as np
import torch
from pytorch_mppi import SMPPI


class DvrkQDesSMPPI(SMPPI):
    def __init__(self, *args, action_cost_weight, action_step_limit, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_cost_weight = torch.as_tensor(action_cost_weight, dtype=self.dtype, device=self.d)
        self.action_step_limit = torch.as_tensor(action_step_limit, dtype=self.dtype, device=self.d)

    def _bound_action_sequence(self, actions, initial_action):
        bounded = actions.clone()
        previous = initial_action
        for time in range(self.T):
            bounded[..., time, :] = torch.clamp(
                bounded[..., time, :],
                previous - self.action_step_limit,
                previous + self.action_step_limit,
            )
            previous = bounded[..., time, :]
        return bounded

    def _compute_perturbed_action_and_noise(self):
        noise = self._sample_noise((self.K, self.T))
        perturbed_control = self._bound_d_action(self.U + noise)
        self.perturbed_action = self._bound_action(self.action_sequence + perturbed_control * self.delta_t)
        self.perturbed_action = self._bound_action_sequence(self.perturbed_action, self.state)
        self.noise = (self.perturbed_action - self.action_sequence) / self.delta_t - self.U

    def _command(self, state):
        self.state = torch.as_tensor(state, dtype=self.dtype, device=self.d)
        self._compute_weighting(self._compute_total_cost_batch())
        self.U += torch.einsum("k,ktn->tn", self.omega, self.noise)
        self.action_sequence = self._bound_action(self.action_sequence + self.U * self.delta_t)
        self.action_sequence = self._bound_action_sequence(self.action_sequence, self.state)
        return self.action_sequence[0]

    def _compute_total_cost_batch(self):
        self._compute_perturbed_action_and_noise()
        action_cost = self._compute_action_cost(self.noise)
        action_difference = torch.diff(self.perturbed_action, dim=-2)
        omega_cost = (action_difference.square() * self.action_cost_weight).sum(dim=(1, 2))
        rollout_cost, self.states, actions = self._compute_rollout_costs(self.perturbed_action)
        self.actions = actions
        perturbation_cost = torch.sum(self.U * action_cost, dim=(1, 2))
        self.cost_total = rollout_cost + perturbation_cost + omega_cost
        return self.cost_total


class DvrkQDesSMPPIPlanner:
    DIM = 7

    def __init__(self, env, goal_position, num_samples=200, horizon=8, dt=0.02, dq_limit=0.045):
        self.env = env
        self.model = env.model
        self.goal_position = torch.as_tensor(goal_position, dtype=torch.float32)
        self.qpos_ids = np.array([self.model.jnt_qposadr[joint_id] for joint_id in env.joint_ids], dtype=int)
        jaw_joint_id = env.get_id(mujoco.mjtObj.mjOBJ_JOINT, "PSM1_jaw")
        self.qpos_ids = np.r_[self.qpos_ids, self.model.jnt_qposadr[jaw_joint_id]]
        self.actuator_ids = np.array([*env.arm_actuator_ids, env.get_id(mujoco.mjtObj.mjOBJ_ACTUATOR, "PSM1_jaw")])
        self.action_min = torch.as_tensor(self.model.actuator_ctrlrange[self.actuator_ids, 0], dtype=torch.float32)
        self.action_max = torch.as_tensor(self.model.actuator_ctrlrange[self.actuator_ids, 1], dtype=torch.float32)
        self.action_step_limit = self._tip_step_limits(dq_limit)
        self.rate_limit = self.action_step_limit / dt
        self.reference_qpos = env.data.qpos.copy()
        self.fk_data = mujoco.MjData(self.model)

        def dynamics(state, action, t):
            return action

        def running_cost(state, action, t):
            position_error = self._tip_positions(state) - self.goal_position.to(state)
            return 2000.0 * position_error.square().sum(dim=-1)

        def terminal_cost(states, actions):
            return 10.0 * running_cost(states[..., -1, :], actions[..., -1, :], 0)

        q_current = self.current_qpos()
        self.mppi = DvrkQDesSMPPI(
            dynamics=dynamics,
            running_cost=running_cost,
            terminal_state_cost=terminal_cost,
            nx=self.DIM,
            noise_sigma=torch.diag((0.5 * self.rate_limit).square()),
            num_samples=num_samples,
            horizon=horizon,
            lambda_=0.01,
            u_min=-self.rate_limit,
            u_max=self.rate_limit,
            U_init=q_current.repeat(horizon, 1),
            action_min=self.action_min,
            action_max=self.action_max,
            delta_t=dt,
            action_cost_weight=0.1 / self.action_step_limit.square(),
            action_step_limit=self.action_step_limit,
            step_dependent_dynamics=True,
        )

    def current_qpos(self):
        return torch.as_tensor(self.env.data.qpos[self.qpos_ids].copy(), dtype=torch.float32)

    def command(self):
        return self.mppi.command(self.current_qpos()).detach().cpu().numpy()

    def _tip_step_limits(self, dq_limit):
        jacobian = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.env.data, jacobian, None, self.env.tip_site_id)
        dof_ids = [self.model.jnt_dofadr[joint_id] for joint_id in self.env.joint_ids]
        joint_limits = np.minimum(
            float(dq_limit), 0.004 / np.maximum(np.linalg.norm(jacobian[:, dof_ids], axis=0), 1e-8)
        )
        return torch.as_tensor(np.r_[joint_limits, 0.05], dtype=torch.float32)

    def _tip_positions(self, q):
        q_np = q.detach().cpu().numpy().reshape(-1, self.DIM)
        positions = np.empty((len(q_np), 3), dtype=np.float32)
        for index, target_qpos in enumerate(q_np):
            self.fk_data.qpos[:] = self.reference_qpos
            self.fk_data.qpos[self.qpos_ids] = target_qpos
            self._apply_mimics()
            mujoco.mj_forward(self.model, self.fk_data)
            positions[index] = self.fk_data.site_xpos[self.env.tip_site_id]
        return torch.as_tensor(positions, dtype=q.dtype, device=q.device).reshape(*q.shape[:-1], 3)

    def _apply_mimics(self):
        for equality_id in range(self.model.neq):
            if self.model.eq_type[equality_id] != mujoco.mjtEq.mjEQ_JOINT:
                continue
            joint1, joint2 = self.model.eq_obj1id[equality_id], self.model.eq_obj2id[equality_id]
            if joint2 < 0:
                continue
            qpos1 = self.model.jnt_qposadr[joint1]
            qpos2 = self.model.jnt_qposadr[joint2]
            if qpos2 not in self.qpos_ids:
                continue
            coefficients = self.model.eq_data[equality_id, :5]
            value = self.fk_data.qpos[qpos2]
            self.fk_data.qpos[qpos1] = np.polynomial.polynomial.polyval(value, coefficients)
