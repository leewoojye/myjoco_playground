"""Sparse-control-point SMPPI with cubic interpolation and RCM-projected SVGD."""

import math

import numpy as np
import torch
from scipy.interpolate import CubicSpline
from torch.func import jacrev, vmap


class DvrkSCPSMPPIPlanner:
    """SCP-MPPI whose sparse Cartesian control points follow SMPPI lifting.

    The planner keeps sparse derivative-action control points ``U_tilde`` and
    sparse Cartesian action control points ``A_tilde``.  Gaussian perturbations,
    the SMPPI i-axis update, and the SMPPI importance-sampling term are all
    evaluated in this sparse control-point space.  Cubic interpolation then
    expands the sparse candidates to the full rollout horizon.

    Sparse lifted update:

        U_tilde^{i+1} = U_tilde^i + sum_k w_k epsilon_tilde^k
        A_tilde^{i+1} = A_tilde^i + U_tilde^{i+1} * dt

    Full rollout sequence:

        U = CubicSpline(U_tilde)
        A = CubicSpline(A_tilde)

    The t-axis action-difference cost and the dynamics/RCM rollout are evaluated
    on the full interpolated action sequence.  The state cost intentionally uses
    the original Euclidean tip-position error; no SE(3)/se(3) error-state cost is
    used.
    """

    DIM = 7
    CONTROL_LIMIT = (0.004, 0.004, 0.004, 0.06, 0.06, 0.06, 0.05)
    RATE_LIMIT = (0.1, 0.1, 0.1, 0.06, 0.06, 0.06, 0.05)

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
        rcm_cost_weight=10000.0,
        action_limit_cost_weight=1.0e6,
    ):
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if horizon < 2:
            raise ValueError("horizon must be at least 2")
        if not 2 <= num_control_points <= horizon:
            raise ValueError("num_control_points must satisfy 2 <= num_control_points <= horizon")

        self.dynamics = dynamics
        self.num_samples = num_samples
        self.horizon = horizon
        self.num_control_points = num_control_points
        self.svgd_iterations = svgd_iterations
        self.svgd_step_size = svgd_step_size
        self.constraint_step_size = constraint_step_size
        self.lambda_ = lambda_
        self.rcm_tolerance = rcm_tolerance
        self.dt = dt
        self.action_smoothness_weight = action_smoothness_weight
        self.rcm_cost_weight = rcm_cost_weight
        self.action_limit_cost_weight = action_limit_cost_weight

        dtype = dynamics.centers.dtype
        device = dynamics.centers.device
        self.goal = torch.zeros(self.DIM, dtype=dtype, device=device)
        self.rcm_position = torch.as_tensor(
            rcm_position,
            dtype=dtype,
            device=device,
        )
        self.control_limit = torch.tensor(
            self.CONTROL_LIMIT,
            dtype=dtype,
            device=device,
        )
        self.rate_limit = torch.tensor(
            self.RATE_LIMIT,
            dtype=dtype,
            device=device,
        )
        self.control_point_noise_std = 0.5 * self.rate_limit
        self.control_point_noise_variance_inv = (
            self.control_point_noise_std.square().clamp_min(torch.finfo(dtype).eps).reciprocal()
        )

        support_indices = (
            np
            .linspace(
                0,
                horizon - 1,
                num_control_points,
            )
            .round()
            .astype(int)
        )
        if np.unique(support_indices).size != num_control_points:
            raise ValueError("spline support indices must be unique")

        # CubicSpline is linear in its knot values.  Interpolating the identity
        # matrix therefore gives a reusable basis B such that full = B @ sparse.
        spline = CubicSpline(
            support_indices,
            np.eye(num_control_points),
            axis=0,
        )
        spline_basis = spline(np.arange(horizon)).astype(np.float32)

        self.support_indices = torch.as_tensor(
            support_indices,
            dtype=torch.long,
            device=device,
        )
        self.spline_basis = torch.as_tensor(
            spline_basis,
            dtype=dtype,
            device=device,
        )

        # Sparse lifted SMPPI variables.
        self.derivative_control_points = torch.zeros(
            num_control_points,
            self.DIM,
            dtype=dtype,
            device=device,
        )
        self.action_control_points = torch.zeros(
            num_control_points,
            self.DIM,
            dtype=dtype,
            device=device,
        )

        # Full-horizon caches kept for tracing and receding-horizon warm starts.
        self.action_sequence = torch.zeros(
            horizon,
            self.DIM,
            dtype=dtype,
            device=device,
        )
        self.derivative_sequence = torch.zeros_like(self.action_sequence)

        self.candidate_states = None
        self.candidate_controls = None
        self.candidate_derivatives = None
        self.candidate_control_point_noise = None
        self.candidate_costs = None
        self.candidate_weights = None
        self.candidate_rcm_deviation = None
        self.candidate_importance_cost = None
        self.candidate_t_axis_cost = None
        self.candidate_action_limit_cost = None
        self.nominal_action_sequence = None
        self.nominal_derivative_sequence = None
        self.nominal_action_control_points = None
        self.nominal_derivative_control_points = None

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
            return torch.as_tensor(
                value,
                dtype=self.goal.dtype,
                device=self.goal.device,
            )

        self.wrist_position = convert(wrist_position)
        self.shaft_direction = convert(shaft_direction)
        self.tip_rotation = convert(tip_rotation)
        self.tip_jacobian_inverse = torch.linalg.pinv(convert(tip_jacobian))
        self.wrist_position_jacobian = convert(wrist_position_jacobian)
        self.wrist_rotation_jacobian = convert(wrist_rotation_jacobian)

    def set_goal(self, goal):
        self.goal = torch.as_tensor(
            goal,
            dtype=self.goal.dtype,
            device=self.goal.device,
        )

    def command(self, state):
        state = torch.as_tensor(
            state,
            dtype=self.goal.dtype,
            device=self.goal.device,
        )

        # Shift the previous full solution by one rollout step, then extract the
        # sparse values at the fixed support indices for the next inference.
        self._shift_nominal_sequences()

        particles = torch.randn(
            self.num_samples,
            self.num_control_points,
            self.DIM,
            dtype=state.dtype,
            device=state.device,
        )
        particles[0].zero_()

        for _ in range(self.svgd_iterations):
            particles.requires_grad_(True)
            (
                derivatives,
                actions,
                corrected_control_point_noise,
            ) = self._candidate_sequences(particles)
            (
                _,
                state_cost,
                rcm_dev,
                t_axis_cost,
                action_limit_cost,
            ) = self._evaluate(state, actions)
            importance_cost = self._importance_sampling_cost(corrected_control_point_noise)
            costs = (
                state_cost + t_axis_cost + action_limit_cost + importance_cost
                # + self.rcm_cost_weight * rcm_dev.amax(dim=-1)
            )

            beta = costs.min().detach()
            log_likelihood = -torch.log(costs - beta + 10.0)
            score = torch.autograd.grad(log_likelihood.sum(), particles)[0]

            particles = particles.detach()
            projection, correction = self._constraint_geometry(particles)
            stein = self._stein_direction(
                particles,
                score.detach(),
            ).reshape(self.num_samples, -1)
            tangent_stein = torch.einsum(
                "kij,kj->ki",
                projection,
                stein,
            ).reshape_as(particles)
            particles = (
                particles
                + self.svgd_step_size * tangent_stein
                + self.constraint_step_size * correction.reshape_as(particles)
            )

        with torch.no_grad():
            (
                derivatives,
                actions,
                corrected_control_point_noise,
            ) = self._candidate_sequences(particles)
            (
                states,
                state_cost,
                rcm_deviation,
                t_axis_cost,
                action_limit_cost,
            ) = self._evaluate(state, actions)
            importance_cost = self._importance_sampling_cost(corrected_control_point_noise)
            costs = (
                state_cost + t_axis_cost + action_limit_cost + importance_cost
                # + self.rcm_cost_weight * rcm_deviation.amax(dim=-1)
            )
            weights = torch.softmax(
                -(costs - costs.min()) / self.lambda_,
                dim=0,
            )

            # SMPPI Eq. (16), applied only to sparse derivative control points.
            weighted_control_point_noise = torch.einsum(
                "k,kmd->md",
                weights,
                corrected_control_point_noise,
            )
            self.derivative_control_points.add_(weighted_control_point_noise)
            self.derivative_control_points.clamp_(
                -self.rate_limit,
                self.rate_limit,
            )

            # SMPPI Eq. (17), applied at the same sparse control-point indices.
            self.action_control_points.add_(self.derivative_control_points * self.dt)
            self.action_control_points.clamp_(
                -self.control_limit,
                self.control_limit,
            )

            # SCP-MPPI: only the sparse variables are optimized.  The full
            # horizon used for rollout/execution is obtained by cubic spline.
            nominal_derivatives = self._interpolate(self.derivative_control_points)
            nominal_actions = self._interpolate(self.action_control_points)

            self.derivative_sequence.copy_(nominal_derivatives)
            self.action_sequence.copy_(nominal_actions)
            self.nominal_action_sequence = nominal_actions.clone()
            self.nominal_derivative_sequence = nominal_derivatives.clone()
            self.nominal_action_control_points = self.action_control_points.clone()
            self.nominal_derivative_control_points = self.derivative_control_points.clone()
            control = nominal_actions[0].clone()

            self.candidate_states = states
            self.candidate_controls = actions
            self.candidate_derivatives = derivatives
            self.candidate_control_point_noise = corrected_control_point_noise
            self.candidate_costs = costs
            self.candidate_weights = weights
            self.candidate_rcm_deviation = rcm_deviation
            self.candidate_importance_cost = importance_cost
            self.candidate_t_axis_cost = t_axis_cost
            self.candidate_action_limit_cost = action_limit_cost

        action = control.clone()
        # Position action remains an absolute ECM-frame target for the demo.
        action[:3] = state[:3] + control[:3]
        return action.cpu().numpy().astype(np.float32)

    def _shift_nominal_sequences(self):
        full_derivatives = self._interpolate(self.derivative_control_points)
        shifted_derivatives = torch.cat(
            (
                full_derivatives[1:],
                torch.zeros_like(full_derivatives[-1:]),
            ),
            dim=0,
        )
        self.derivative_control_points.copy_(shifted_derivatives[self.support_indices])

        full_actions = self._interpolate(self.action_control_points)
        shifted_actions = torch.cat(
            (
                full_actions[1:],
                full_actions[-1:],
            ),
            dim=0,
        )
        self.action_control_points.copy_(shifted_actions[self.support_indices])

        self.derivative_sequence.copy_(self._interpolate(self.derivative_control_points))
        self.action_sequence.copy_(self._interpolate(self.action_control_points))

    def rcm_deviation(self, wrist_position, shaft_direction):
        wrist_position = torch.as_tensor(
            wrist_position,
            dtype=self.goal.dtype,
            device=self.goal.device,
        )
        shaft_direction = torch.as_tensor(
            shaft_direction,
            dtype=self.goal.dtype,
            device=self.goal.device,
        )
        return torch.linalg.vector_norm(
            torch.linalg.cross(
                self.rcm_position - wrist_position,
                shaft_direction,
                dim=-1,
            ),
            dim=-1,
        )

    def _candidate_sequences(self, particles):
        # Noise is injected only at the sparse control points, as in SCP-MPPI.
        raw_control_point_noise = particles * self.control_point_noise_std
        candidate_derivative_control_points = (
            self.derivative_control_points.unsqueeze(0) + raw_control_point_noise
        ).clamp(
            -self.rate_limit,
            self.rate_limit,
        )

        # Lift derivative control points into action control points (SMPPI).
        candidate_action_control_points = (
            self.action_control_points.unsqueeze(0) + candidate_derivative_control_points * self.dt
        ).clamp(
            -self.control_limit,
            self.control_limit,
        )

        # Recompute the effective derivative/noise after both sparse-space bounds.
        effective_derivative_control_points = (
            candidate_action_control_points - self.action_control_points.unsqueeze(0)
        ) / self.dt
        corrected_control_point_noise = effective_derivative_control_points - self.derivative_control_points.unsqueeze(
            0
        )

        derivatives = self._interpolate(effective_derivative_control_points)
        actions = self._interpolate(candidate_action_control_points)
        return derivatives, actions, corrected_control_point_noise

    def _importance_sampling_cost(self, corrected_control_point_noise):
        # SMPPI Eq. (18), consistently evaluated in the sampled sparse space.
        return self.lambda_ * (
            self.derivative_control_points.unsqueeze(0)
            * corrected_control_point_noise
            * self.control_point_noise_variance_inv
        ).sum(dim=(-2, -1))

    def _interpolate(self, control_points):
        return torch.einsum(
            "tm,...md->...td",
            self.spline_basis,
            control_points,
        )

    def _evaluate(self, initial_state, actions):
        state = initial_state.expand(actions.shape[0], -1)
        states = []
        for time in range(self.horizon):
            action = actions[:, time]
            dynamics_action = action.clone()
            dynamics_action[:, :3] = state[:, :3] + action[:, :3]
            state = self.dynamics(state, dynamics_action)
            states.append(state)

        states = torch.stack(states, dim=1)

        # Original simple Cartesian position error only.
        position_error = states[..., :3] - self.goal[:3]
        running_position_cost = 2000.0 * position_error.square().sum(dim=(-2, -1))
        terminal_position_cost = 20000.0 * (position_error[:, -1].square().sum(dim=-1))
        state_cost = running_position_cost + terminal_position_cost

        # SMPPI t-axis cost is evaluated on the full spline-interpolated actions.
        action_difference = actions[:, 1:] - actions[:, :-1]
        t_axis_cost = self.action_smoothness_weight * (action_difference / self.control_limit).square().sum(
            dim=(-2, -1)
        )

        # Cubic interpolation can overshoot bounded knot values.  Penalize that
        # overshoot rather than clipping the full sequence and breaking the
        # interpolation's differentiability/continuity inside the rollout.
        action_limit_violation = (actions.abs() - self.control_limit).clamp_min(0.0)
        action_limit_cost = self.action_limit_cost_weight * (action_limit_violation / self.control_limit).square().sum(
            dim=(-2, -1)
        )

        rcm_deviation = torch.linalg.vector_norm(
            self._rollout_rcm_residual(actions),
            dim=-1,
        )
        return (
            states,
            state_cost,
            rcm_deviation,
            t_axis_cost,
            action_limit_cost,
        )

    def _constraint_geometry(self, particles):
        def residual_with_aux(particle):
            residual = self._particle_rcm_residual(particle)
            return residual, residual

        jacobian, residual = vmap(jacrev(residual_with_aux, has_aux=True))(particles)
        jacobian = jacobian.reshape(
            particles.shape[0],
            self.horizon,
            3,
            particles[0].numel(),
        )
        residual = residual.reshape(
            particles.shape[0],
            self.horizon,
            3,
        )

        deviation = torch.linalg.vector_norm(residual, dim=-1)
        normal = residual / deviation.unsqueeze(-1).clamp_min(torch.finfo(residual.dtype).eps)
        constraint_jacobian = torch.einsum(
            "kti,ktip->ktp",
            normal,
            jacobian,
        )
        active = deviation > self.rcm_tolerance
        constraint_jacobian = torch.where(
            active.unsqueeze(-1),
            constraint_jacobian,
            0.0,
        )
        violation = (deviation - self.rcm_tolerance).clamp_min(0.0)

        inverse = torch.linalg.pinv(constraint_jacobian)
        eye = torch.eye(
            constraint_jacobian.shape[-1],
            dtype=particles.dtype,
            device=particles.device,
        )
        projection = eye - inverse @ constraint_jacobian
        correction = -torch.einsum(
            "kij,kj->ki",
            inverse,
            violation,
        )
        return projection.detach(), correction.detach()

    def _particle_rcm_residual(self, particle):
        _, actions, _ = self._candidate_sequences(particle.unsqueeze(0))
        return self._rollout_rcm_residual(actions)[0]

    def _rollout_rcm_residual(self, actions):
        wrist_position = self.wrist_position.expand(
            actions.shape[0],
            -1,
        )
        shaft_direction = self.shaft_direction.expand(
            actions.shape[0],
            -1,
        )
        residuals = []

        for time in range(self.horizon):
            body_rotation = actions[:, time, 3:6]
            angular_displacement = torch.einsum(
                "ij,kj->ki",
                self.tip_rotation,
                body_rotation,
            )
            tip_displacement = torch.cat(
                (
                    actions[:, time, :3],
                    angular_displacement,
                ),
                dim=-1,
            )
            joint_displacement = torch.einsum(
                "ij,kj->ki",
                self.tip_jacobian_inverse,
                tip_displacement,
            )
            wrist_position = wrist_position + torch.einsum(
                "ij,kj->ki",
                self.wrist_position_jacobian,
                joint_displacement,
            )
            wrist_rotation = torch.einsum(
                "ij,kj->ki",
                self.wrist_rotation_jacobian,
                joint_displacement,
            )
            shaft_direction = torch.einsum(
                "kij,kj->ki",
                self._rotation_matrix(wrist_rotation),
                shaft_direction,
            )
            offset = self.rcm_position - wrist_position
            residuals.append(
                offset
                - (offset * shaft_direction).sum(
                    dim=-1,
                    keepdim=True,
                )
                * shaft_direction
            )

        return torch.stack(residuals, dim=1) - 0.002

    def _stein_direction(self, particles, score):
        flat_particles = particles.reshape(
            self.num_samples,
            -1,
        )
        flat_score = score.reshape(
            self.num_samples,
            -1,
        )
        difference = flat_particles[:, None, :] - flat_particles[None, :, :]
        distance_squared = difference.square().sum(dim=-1)
        bandwidth = flat_particles.square().sum(dim=-1).median() / math.log(self.num_samples)
        bandwidth = bandwidth.detach().clamp_min(torch.finfo(flat_particles.dtype).eps)
        kernel = torch.exp(-distance_squared / (2.0 * bandwidth))
        attraction = kernel.T @ flat_score / self.num_samples
        repulsion = -(kernel[..., None] * difference).sum(dim=0) / (self.num_samples * bandwidth)
        return (attraction + repulsion).reshape_as(particles)

    @staticmethod
    def _rotation_matrix(rotation_vector):
        x, y, z = rotation_vector.unbind(dim=-1)
        zero = torch.zeros_like(x)
        skew = torch.stack(
            (
                zero,
                -z,
                y,
                z,
                zero,
                -x,
                -y,
                x,
                zero,
            ),
            dim=-1,
        ).reshape(
            *rotation_vector.shape[:-1],
            3,
            3,
        )
        angle = torch.linalg.vector_norm(
            rotation_vector,
            dim=-1,
        )
        a = torch.sinc(angle / torch.pi)
        b = 0.5 * torch.sinc(angle / (2.0 * torch.pi)).square()
        eye = torch.eye(
            3,
            dtype=rotation_vector.dtype,
            device=rotation_vector.device,
        )
        return eye + a[..., None, None] * skew + b[..., None, None] * (skew @ skew)
