"""SCP-SMPPI with sparse derivative controls and an SE(3) error-state cost.

The MPPI/SVGD/RCM structure is kept from the existing planner.  Only the pose
rollout and running cost are extended to use the Lie-group error formulation
from Arefinia et al. (2026):

    e = g_d^{-1} g,
    omega = Log(R_e),
    chi = J_l(omega)^{-1} p_e,

with a finite-difference body-twist velocity error.  No EWMA uncertainty term
is included.
"""

import math

import numpy as np
import torch
from scipy.interpolate import CubicSpline
from torch.func import jacrev, vmap


class DvrkSCPSMPPIPlanner:
    """Sparse-control-point SMPPI with RCM-projected SVGD and SE(3) costs.

    The public state layout remains compact and backward compatible::

        state = [p_x, p_y, p_z, r_x, r_y, r_z, jaw]

    where ``r`` is a rotation vector for the actual tip rotation matrix in the
    ECM frame.  During cost evaluation the pose is lifted to SE(3), so the
    controller does not subtract rotation-vector coordinates directly.

    When ``use_se3_kinematic_rollout`` is true, rollout orientation is updated
    by right/body composition ``R_next = R_current Exp(delta_r^)``.  This lets
    the current dVRK demo use the existing ``KinematicDynamics`` object without
    modifying ``dynamics_dvrk.py``.
    """

    DIM = 7
    CONTROL_LIMIT = (0.004, 0.004, 0.004, 0.06, 0.06, 0.06, 0.05)
    RATE_LIMIT = (0.1, 0.1, 0.1, 0.06, 0.06, 0.06, 0.05)
    COST_COMPONENT_NAMES = (
        "orientation_running",
        "position_running",
        "velocity_running",
        "control_running",
        "action_smoothness",
        "orientation_terminal",
        "position_terminal",
    )

    def __init__(
        self,
        dynamics,
        rcm_position,
        num_samples=64,
        horizon=24,
        num_control_points=4,
        svgd_iterations=3,
        svgd_step_size=0.05,
        constraint_step_size=1.0,
        lambda_=0.01,
        rcm_tolerance=0.015,
        dt=0.02,
        action_smoothness_weight=0.1,
        orientation_weight=0.0,  # 100
        position_weight=2000.0,
        velocity_weight=1.0,
        control_weight=0.01,
        terminal_orientation_weight=0.0,  # 1000
        terminal_position_weight=20000.0,
        use_se3_kinematic_rollout=False,
    ):
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if horizon < 2:
            raise ValueError("horizon must be at least 2")
        if num_control_points < 2:
            raise ValueError("num_control_points must be at least 2")

        self.dynamics = dynamics
        self.num_samples = num_samples
        self.horizon = horizon
        self.svgd_iterations = svgd_iterations
        self.svgd_step_size = svgd_step_size
        self.constraint_step_size = constraint_step_size
        self.lambda_ = lambda_
        self.rcm_tolerance = rcm_tolerance
        self.dt = dt
        self.action_smoothness_weight = action_smoothness_weight
        self.orientation_weight = orientation_weight
        self.position_weight = position_weight
        self.velocity_weight = velocity_weight
        self.control_weight = control_weight
        self.terminal_orientation_weight = terminal_orientation_weight
        self.terminal_position_weight = terminal_position_weight
        self.use_se3_kinematic_rollout = use_se3_kinematic_rollout

        dtype = dynamics.centers.dtype
        device = dynamics.centers.device
        self.goal = torch.zeros(self.DIM, dtype=dtype, device=device)
        self.rcm_position = torch.as_tensor(rcm_position, dtype=dtype, device=device)
        self.control_limit = torch.tensor(self.CONTROL_LIMIT, dtype=dtype, device=device)
        self.speed_limit = torch.tensor(self.RATE_LIMIT, dtype=dtype, device=device)
        self.rate_limit = self.speed_limit
        self.rate_noise_std = 0.5 * self.rate_limit

        support_indices = np.linspace(0, horizon - 1, num_control_points).round().astype(int)
        if np.unique(support_indices).size != num_control_points:
            raise ValueError("num_control_points is too large for the selected horizon")
        spline = CubicSpline(support_indices, np.eye(num_control_points), axis=0)
        spline_basis = spline(np.arange(horizon)).astype(np.float32)
        self.support_indices = torch.as_tensor(support_indices, dtype=torch.long, device=device)
        self.spline_basis = torch.as_tensor(spline_basis, dtype=dtype, device=device)
        self.derivative_control_points = torch.zeros(
            num_control_points,
            self.DIM,
            dtype=dtype,
            device=device,
        )
        self.action_sequence = torch.zeros(horizon, self.DIM, dtype=dtype, device=device)

        self.candidate_states = None
        self.candidate_controls = None
        self.candidate_derivatives = None
        self.candidate_costs = None
        self.candidate_weights = None
        self.candidate_rcm_deviation = None
        self.candidate_pose_errors = None
        self.candidate_velocity_errors = None
        self.candidate_cost_components = None
        self.nominal_action_sequence = None
        self.nominal_derivative_sequence = None

    def set_rcm_linearization(
        self,
        wrist_position,
        shaft_direction,
        tip_rotation,
        tip_jacobian,
        wrist_position_jacobian,
        wrist_rotation_jacobian,
    ):
        def convert(value):
            return torch.as_tensor(value, dtype=self.goal.dtype, device=self.goal.device)

        self.wrist_position = convert(wrist_position)
        self.shaft_direction = convert(shaft_direction)
        self.tip_rotation = convert(tip_rotation)
        self.tip_jacobian_inverse = torch.linalg.pinv(convert(tip_jacobian))
        self.wrist_position_jacobian = convert(wrist_position_jacobian)
        self.wrist_rotation_jacobian = convert(wrist_rotation_jacobian)

    def set_goal(self, goal):
        goal = torch.as_tensor(goal, dtype=self.goal.dtype, device=self.goal.device)
        if goal.shape != (self.DIM,):
            raise ValueError(f"goal must have shape ({self.DIM},), got {tuple(goal.shape)}")
        self.goal = goal

    def pose_error(self, state):
        """Return the paper-style pose error ``[omega, chi]``.

        ``omega`` is the logarithm of ``R_goal.T @ R`` and ``chi`` is the
        translation of ``goal^{-1} current`` corrected by the inverse SO(3)
        left Jacobian.
        """
        state = torch.as_tensor(state, dtype=self.goal.dtype, device=self.goal.device)
        return self._se3_pose_error(state)

    def command(self, state):
        state = torch.as_tensor(state, dtype=self.goal.dtype, device=self.goal.device)
        particles = torch.randn(
            self.num_samples,
            self.derivative_control_points.shape[0],
            self.DIM,
            dtype=state.dtype,
            device=state.device,
        )
        particles[0].zero_()

        for _ in range(self.svgd_iterations):
            particles.requires_grad_(True)
            actions = self._candidate_actions(particles)
            _, costs, rcm_dev, _, _, _ = self._evaluate(state, actions)
            costs = costs + rcm_dev.amax(dim=-1) * 10000.0
            beta = costs.min().detach()
            log_likelihood = -torch.log(costs - beta + 10.0)
            score = torch.autograd.grad(log_likelihood.sum(), particles)[0]

            particles = particles.detach()
            projection, correction = self._constraint_geometry(particles)
            stein = self._stein_direction(particles, score.detach()).reshape(self.num_samples, -1)
            tangent_stein = torch.einsum("kij,kj->ki", projection, stein).reshape_as(particles)
            particles = (
                particles
                + self.svgd_step_size * tangent_stein
                + self.constraint_step_size * correction.reshape_as(particles)
            )

        with torch.no_grad():
            sparse_derivatives = self._sparse_derivatives(particles)
            derivatives = self._interpolate(sparse_derivatives).clamp(-self.rate_limit, self.rate_limit)
            actions = self._integrate_actions(derivatives)
            states, costs, rcm_deviation, pose_errors, velocity_errors, cost_components = self._evaluate(
                state,
                actions,
            )
            weights = torch.softmax(-(costs - costs.min()) / self.lambda_, dim=0)

            perturbations = sparse_derivatives - self.derivative_control_points
            self.derivative_control_points.add_(torch.einsum("k,kmd->md", weights, perturbations))
            self.derivative_control_points.clamp_(-self.rate_limit, self.rate_limit)

            nominal_derivatives = self._interpolate(self.derivative_control_points).clamp(
                -self.rate_limit,
                self.rate_limit,
            )
            self.action_sequence.add_(nominal_derivatives * self.dt)
            self.action_sequence.clamp_(-self.control_limit, self.control_limit)

            self.nominal_action_sequence = self.action_sequence.clone()
            self.nominal_derivative_sequence = nominal_derivatives.clone()
            control = self.action_sequence[0].clone()

            shifted_actions = torch.cat((self.action_sequence[1:], self.action_sequence[-1:]), dim=0)
            self.action_sequence.copy_(shifted_actions)
            shifted_derivatives = torch.cat((nominal_derivatives[1:], nominal_derivatives[-1:]), dim=0)
            self.derivative_control_points.copy_(shifted_derivatives[self.support_indices])

            self.candidate_states = states
            self.candidate_controls = actions
            self.candidate_derivatives = derivatives
            self.candidate_costs = costs
            self.candidate_weights = weights
            self.candidate_rcm_deviation = rcm_deviation
            self.candidate_pose_errors = pose_errors
            self.candidate_velocity_errors = velocity_errors
            self.candidate_cost_components = cost_components

        action = control.clone()
        action[:3] = state[:3] + control[:3]
        return action.cpu().numpy().astype(np.float32)

    def rcm_deviation(self, wrist_position, shaft_direction):
        wrist_position = torch.as_tensor(wrist_position, dtype=self.goal.dtype, device=self.goal.device)
        shaft_direction = torch.as_tensor(shaft_direction, dtype=self.goal.dtype, device=self.goal.device)
        return torch.linalg.vector_norm(
            torch.linalg.cross(self.rcm_position - wrist_position, shaft_direction, dim=-1),
            dim=-1,
        )

    def _candidate_actions(self, particles):
        derivatives = self._interpolate(self._sparse_derivatives(particles)).clamp(
            -self.rate_limit,
            self.rate_limit,
        )
        return self._integrate_actions(derivatives)

    def _sparse_derivatives(self, particles):
        return (self.derivative_control_points + particles * self.rate_noise_std).clamp(
            -self.rate_limit,
            self.rate_limit,
        )

    def _integrate_actions(self, derivatives):
        return (self.action_sequence + derivatives * self.dt).clamp(
            -self.control_limit,
            self.control_limit,
        )

    def _interpolate(self, sparse_derivatives):
        return torch.einsum("tm,...md->...td", self.spline_basis, sparse_derivatives)

    def _evaluate(self, initial_state, actions):
        state = initial_state.expand(actions.shape[0], -1)
        states = []
        for time in range(self.horizon):
            action = actions[:, time]
            dynamics_action = action.clone()
            dynamics_action[:, :3] = state[:, :3] + action[:, :3]
            if self.use_se3_kinematic_rollout:
                state = self._se3_kinematic_step(state, dynamics_action, action)
            else:
                state = self.dynamics(state, dynamics_action)
            states.append(state)

        states = torch.stack(states, dim=1)
        pose_errors = self._se3_pose_error(states)
        velocity_errors = self._finite_difference_velocity_error(initial_state, states, pose_errors[..., :3])

        orientation_running = self.orientation_weight * pose_errors[..., :3].square().sum(dim=(-2, -1))
        position_running = self.position_weight * pose_errors[..., 3:6].square().sum(dim=(-2, -1))
        velocity_running = self.velocity_weight * velocity_errors.square().sum(dim=(-2, -1))
        control_running = 0.5 * self.control_weight * (actions / self.control_limit).square().sum(dim=(-2, -1))

        action_difference = actions[:, 1:] - actions[:, :-1]
        action_smoothness = self.action_smoothness_weight * (action_difference / self.control_limit).square().sum(
            dim=(-2, -1)
        )

        orientation_terminal = self.terminal_orientation_weight * pose_errors[:, -1, :3].square().sum(dim=-1)
        position_terminal = self.terminal_position_weight * pose_errors[:, -1, 3:6].square().sum(dim=-1)

        cost_components = torch.stack(
            (
                orientation_running,
                position_running,
                velocity_running,
                control_running,
                action_smoothness,
                orientation_terminal,
                position_terminal,
            ),
            dim=-1,
        )
        total_cost = cost_components.sum(dim=-1)
        rcm_deviation = torch.linalg.vector_norm(self._rollout_rcm_residual(actions), dim=-1)
        return states, total_cost, rcm_deviation, pose_errors, velocity_errors, cost_components

    def _se3_kinematic_step(self, state, dynamics_action, incremental_action):
        """Kinematic dVRK rollout with group-consistent body rotation."""
        position = dynamics_action[..., :3]
        rotation = self._compose_body_rotation(state[..., 3:6], incremental_action[..., 3:6])
        jaw = state[..., 6:7] + incremental_action[..., 6:7]
        return torch.cat((position, rotation, jaw), dim=-1)

    def _se3_pose_error(self, state):
        current_position = state[..., :3]
        current_rotation = state[..., 3:6]
        goal_position = self.goal[:3]
        goal_rotation = self.goal[3:6]

        omega = self._relative_rotation_vector(goal_rotation, current_rotation)
        goal_rotation_matrix = self._rotation_matrix(goal_rotation)
        position_error_goal_frame = torch.einsum(
            "ji,...j->...i",
            goal_rotation_matrix,
            current_position - goal_position,
        )
        chi = torch.einsum(
            "...ij,...j->...i",
            self._left_jacobian_inverse(omega),
            position_error_goal_frame,
        )
        return torch.cat((omega, chi), dim=-1)

    def _finite_difference_velocity_error(self, initial_state, states, pose_rotation_error):
        """Approximate the paper's tangent-space body-twist error.

        The current demo has a pose-only kinematic state and a static target.
        Thus desired twist is zero and the actual body twist is estimated from
        consecutive rollout poses using ``Log(T_prev^{-1} T_next) / dt``.
        """
        batch_initial = initial_state.expand(states.shape[0], -1).unsqueeze(1)
        all_states = torch.cat((batch_initial, states), dim=1)
        previous = all_states[:, :-1]
        current = all_states[:, 1:]

        delta_omega = self._relative_rotation_vector(previous[..., 3:6], current[..., 3:6])
        previous_rotation = self._rotation_matrix(previous[..., 3:6])
        delta_position_body = torch.einsum(
            "...ji,...j->...i",
            previous_rotation,
            current[..., :3] - previous[..., :3],
        )
        delta_linear = torch.einsum(
            "...ij,...j->...i",
            self._left_jacobian_inverse(delta_omega),
            delta_position_body,
        )
        body_twist = torch.cat((delta_omega / self.dt, delta_linear / self.dt), dim=-1)

        raw_velocity_error = -body_twist
        corrected_linear_error = torch.einsum(
            "...ij,...j->...i",
            self._left_jacobian_inverse(pose_rotation_error),
            raw_velocity_error[..., 3:6],
        )
        return torch.cat((raw_velocity_error[..., :3], corrected_linear_error), dim=-1)

    def _constraint_geometry(self, particles):
        def residual_with_aux(particle):
            residual = self._particle_rcm_residual(particle)
            return residual, residual

        jacobian, residual = vmap(jacrev(residual_with_aux, has_aux=True))(particles)
        jacobian = jacobian.reshape(particles.shape[0], self.horizon, 3, particles[0].numel())
        residual = residual.reshape(particles.shape[0], self.horizon, 3)

        deviation = torch.linalg.vector_norm(residual, dim=-1)
        normal = residual / deviation.unsqueeze(-1).clamp_min(torch.finfo(residual.dtype).eps)
        constraint_jacobian = torch.einsum("kti,ktip->ktp", normal, jacobian)
        active = deviation > self.rcm_tolerance
        constraint_jacobian = torch.where(active.unsqueeze(-1), constraint_jacobian, 0.0)
        violation = (deviation - self.rcm_tolerance).clamp_min(0.0)

        inverse = torch.linalg.pinv(constraint_jacobian)
        eye = torch.eye(constraint_jacobian.shape[-1], dtype=particles.dtype, device=particles.device)
        projection = eye - inverse @ constraint_jacobian
        correction = -torch.einsum("kij,kj->ki", inverse, violation)
        return projection.detach(), correction.detach()

    def _particle_rcm_residual(self, particle):
        derivatives = self._interpolate(self._sparse_derivatives(particle)).clamp(
            -self.rate_limit,
            self.rate_limit,
        )
        actions = self._integrate_actions(derivatives)
        return self._rollout_rcm_residual(actions.unsqueeze(0))[0]

    def _rollout_rcm_residual(self, actions):
        wrist_position = self.wrist_position.expand(actions.shape[0], -1)
        shaft_direction = self.shaft_direction.expand(actions.shape[0], -1)
        residuals = []

        for time in range(self.horizon):
            body_rotation = actions[:, time, 3:6]
            angular_displacement = torch.einsum("ij,kj->ki", self.tip_rotation, body_rotation)
            tip_displacement = torch.cat((actions[:, time, :3], angular_displacement), dim=-1)
            joint_displacement = torch.einsum("ij,kj->ki", self.tip_jacobian_inverse, tip_displacement)
            wrist_position = wrist_position + torch.einsum(
                "ij,kj->ki",
                self.wrist_position_jacobian,
                joint_displacement,
            )
            wrist_rotation = torch.einsum("ij,kj->ki", self.wrist_rotation_jacobian, joint_displacement)
            shaft_direction = torch.einsum(
                "kij,kj->ki",
                self._rotation_matrix(wrist_rotation),
                shaft_direction,
            )
            offset = self.rcm_position - wrist_position
            residuals.append(offset - (offset * shaft_direction).sum(dim=-1, keepdim=True) * shaft_direction)

        return torch.stack(residuals, dim=1) - 0.002

    def _stein_direction(self, particles, score):
        flat_particles = particles.reshape(self.num_samples, -1)
        flat_score = score.reshape(self.num_samples, -1)
        difference = flat_particles[:, None, :] - flat_particles[None, :, :]
        distance_squared = difference.square().sum(dim=-1)
        bandwidth = flat_particles.square().sum(dim=-1).median() / math.log(self.num_samples)
        bandwidth = bandwidth.detach().clamp_min(torch.finfo(flat_particles.dtype).eps)
        kernel = torch.exp(-distance_squared / (2.0 * bandwidth))
        attraction = kernel.T @ flat_score / self.num_samples
        repulsion = -(kernel[..., None] * difference).sum(dim=0) / (self.num_samples * bandwidth)
        return (attraction + repulsion).reshape_as(particles)

    @staticmethod
    def _rotation_vector_to_quaternion(rotation_vector):
        angle = torch.linalg.vector_norm(rotation_vector, dim=-1)
        half_angle = 0.5 * angle
        vector_scale = 0.5 * torch.sinc(half_angle / torch.pi)
        return torch.cat(
            (
                torch.cos(half_angle).unsqueeze(-1),
                vector_scale.unsqueeze(-1) * rotation_vector,
            ),
            dim=-1,
        )

    @staticmethod
    def _quaternion_multiply(left, right):
        left, right = torch.broadcast_tensors(left, right)
        left_scalar, left_vector = left[..., :1], left[..., 1:]
        right_scalar, right_vector = right[..., :1], right[..., 1:]
        scalar = left_scalar * right_scalar - (left_vector * right_vector).sum(dim=-1, keepdim=True)
        vector = (
            left_scalar * right_vector
            + right_scalar * left_vector
            + torch.linalg.cross(left_vector, right_vector, dim=-1)
        )
        return torch.cat((scalar, vector), dim=-1)

    @staticmethod
    def _quaternion_to_rotation_vector(quaternion):
        quaternion = quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True).clamp_min(
            torch.finfo(quaternion.dtype).eps
        )
        quaternion = torch.where(quaternion[..., :1] < 0.0, -quaternion, quaternion)
        vector_norm = torch.linalg.vector_norm(quaternion[..., 1:], dim=-1)
        angle = 2.0 * torch.atan2(vector_norm, quaternion[..., 0].clamp_min(0.0))
        scale = angle / vector_norm.clamp_min(torch.finfo(quaternion.dtype).eps)
        scale = torch.where(vector_norm > 1e-7, scale, torch.full_like(scale, 2.0))
        return scale.unsqueeze(-1) * quaternion[..., 1:]

    @classmethod
    def _compose_body_rotation(cls, current_rotation, body_increment):
        current_quaternion = cls._rotation_vector_to_quaternion(current_rotation)
        increment_quaternion = cls._rotation_vector_to_quaternion(body_increment)
        return cls._quaternion_to_rotation_vector(cls._quaternion_multiply(current_quaternion, increment_quaternion))

    @classmethod
    def _relative_rotation_vector(cls, reference_rotation, current_rotation):
        reference_quaternion = cls._rotation_vector_to_quaternion(reference_rotation)
        current_quaternion = cls._rotation_vector_to_quaternion(current_rotation)
        reference_inverse = torch.cat(
            (reference_quaternion[..., :1], -reference_quaternion[..., 1:]),
            dim=-1,
        )
        return cls._quaternion_to_rotation_vector(cls._quaternion_multiply(reference_inverse, current_quaternion))

    @staticmethod
    def _left_jacobian_inverse(rotation_vector):
        """Inverse SO(3) left Jacobian with a small-angle series."""
        x, y, z = rotation_vector.unbind(dim=-1)
        zero = torch.zeros_like(x)
        skew = torch.stack(
            (zero, -z, y, z, zero, -x, -y, x, zero),
            dim=-1,
        ).reshape(*rotation_vector.shape[:-1], 3, 3)
        skew_squared = skew @ skew
        theta = torch.linalg.vector_norm(rotation_vector, dim=-1)
        safe_theta = theta.clamp_min(1e-7)
        half_theta = 0.5 * safe_theta
        coefficient_regular = (1.0 - half_theta / torch.tan(half_theta)) / safe_theta.square()
        coefficient_series = 1.0 / 12.0 + theta.square() / 720.0 + theta.pow(4) / 30240.0
        coefficient = torch.where(theta < 1e-4, coefficient_series, coefficient_regular)
        eye = torch.eye(3, dtype=rotation_vector.dtype, device=rotation_vector.device)
        return eye - 0.5 * skew + coefficient[..., None, None] * skew_squared

    @staticmethod
    def _rotation_matrix(rotation_vector):
        x, y, z = rotation_vector.unbind(dim=-1)
        zero = torch.zeros_like(x)
        skew = torch.stack(
            (zero, -z, y, z, zero, -x, -y, x, zero),
            dim=-1,
        ).reshape(*rotation_vector.shape[:-1], 3, 3)
        angle = torch.linalg.vector_norm(rotation_vector, dim=-1)
        a = torch.sinc(angle / torch.pi)
        b = 0.5 * torch.sinc(angle / (2.0 * torch.pi)).square()
        eye = torch.eye(3, dtype=rotation_vector.dtype, device=rotation_vector.device)
        return eye + a[..., None, None] * skew + b[..., None, None] * (skew @ skew)
