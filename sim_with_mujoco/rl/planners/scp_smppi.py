"""SCP-SMPPI with derivative controls, action integration, and RCM-projected SVGD."""

import math

import numpy as np
import torch
from scipy.interpolate import CubicSpline
from torch.func import jacrev, vmap


class DvrkSCPSMPPIPlanner:
    """Sparse-control-point SMPPI with the existing RCM SVGD projection.

    Particles parameterize sparse derivative controls ``U``.  Cubic spline
    interpolation creates a derivative sequence, which is integrated over one
    MPC update into the persistent action sequence ``A``.  The existing
    position/control costs are retained, and ``Omega(A)`` penalizes action
    variation along the prediction horizon.  RCM projection is applied only
    to the SVGD transport of the sparse ``U`` particles.
    """

    DIM = 7
    # CONTROL_LIMIT = (0.002, 0.002, 0.002, 0.06, 0.06, 0.06, 0.05)
    CONTROL_LIMIT = (0.004, 0.004, 0.004, 0.06, 0.06, 0.06, 0.05)
    RATE_LIMIT = (0.1, 0.1, 0.1, 0.06, 0.06, 0.06, 0.05)  # 위치단위: m/s

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
    ):
        if dt <= 0.0:
            raise ValueError("dt must be positive")

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

        dtype = dynamics.centers.dtype
        device = dynamics.centers.device
        self.goal = torch.zeros(self.DIM, dtype=dtype, device=device)
        self.rcm_position = torch.as_tensor(rcm_position, dtype=dtype, device=device)
        self.control_limit = torch.tensor(self.CONTROL_LIMIT, dtype=dtype, device=device)
        self.speed_limit = torch.tensor(self.RATE_LIMIT, dtype=dtype, device=device)
        # self.rate_limit = self.control_limit / dt
        # self.rate_limit = 0.002  # dt 동안 낼 수 있는 속도 상한선
        # self.rate_limit = self.speed_limit / dt
        self.rate_limit = self.speed_limit
        self.rate_noise_std = 0.5 * self.rate_limit
        # self.rate_noise_std = self.rate_limit

        support_indices = np.linspace(0, horizon - 1, num_control_points).round().astype(int)
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
        self.goal = torch.as_tensor(goal, dtype=self.goal.dtype, device=self.goal.device)

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
            _, costs, rcm_dev = self._evaluate(state, actions)
            costs = costs + rcm_dev.amax(dim=-1) * 10000
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
            states, costs, rcm_deviation = self._evaluate(state, actions)
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
            state = self.dynamics(state, dynamics_action)
            states.append(state)

        states = torch.stack(states, dim=1)
        position_error = states[..., :3] - self.goal[:3]
        per_time_cost = position_error.square().sum(dim=-1)  # [K, T]
        min_cost, min_t = per_time_cost.min(dim=-1)  # min_cost: [K], min_t: [K]

        goal_cost = 2000.0 * position_error.square().sum(dim=(-2, -1))
        # path_vec = torch.linalg.vector_norm(position_error, dim=-1).sum(dim=-1)  # 누적 L2 거리 (시간 합)
        # 한 번에 스칼라(전체 평균+거리)로
        # path_scalar = torch.linalg.vector_norm(position_error.reshape(position_error.shape[0], -1), dim=-1)
        terminal_cost = 20000.0 * position_error[:, -1].square().sum(dim=-1)
        # terminal_cost = 20000.0 * min_cost

        control_cost = 0.01 * (actions / self.control_limit).square().sum(dim=(-2, -1))
        action_difference = actions[:, 1:] - actions[:, :-1]
        smoothness_cost = self.action_smoothness_weight * (action_difference / self.control_limit).square().sum(
            dim=(-2, -1)
        )
        rcm_deviation = torch.linalg.vector_norm(self._rollout_rcm_residual(actions), dim=-1)
        # rcm_dev_scalar = torch.linalg.vector_norm(self._rollout_rcm_residual(actions), dim=-1).amax(dim=-1)
        return states, goal_cost + terminal_cost + smoothness_cost, rcm_deviation
        # return states, goal_cost + terminal_cost + control_cost + smoothness_cost, rcm_deviation

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

        # return torch.stack(residuals, dim=1)
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
