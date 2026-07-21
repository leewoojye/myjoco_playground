import torch
from torch import nn


class RBFEKFDynamics(nn.Module):
    DIM = 7
    POSE_DIM = 6

    def __init__(
        self,
        centers,
        widths,
        weights,
        process_noise=0.07,
        measurement_noise=0.1,
    ):
        super().__init__()
        centers = torch.as_tensor(centers, dtype=torch.get_default_dtype())
        self.num_basis = centers.shape[1]
        widths = torch.as_tensor(widths, dtype=centers.dtype, device=centers.device)
        weights = torch.as_tensor(weights, dtype=centers.dtype, device=centers.device)

        pose_weights = torch.zeros(
            self.POSE_DIM,
            self.num_basis,
            self.POSE_DIM,
            dtype=centers.dtype,
            device=centers.device,
        )
        for axis in range(self.POSE_DIM):
            pose_weights[axis, :, axis] = weights[axis] - 1.0

        pose_parameter_dim = self.POSE_DIM * self.num_basis
        pose_eye = torch.eye(pose_parameter_dim, dtype=centers.dtype, device=centers.device)
        jaw_eye = torch.eye(self.num_basis, dtype=centers.dtype, device=centers.device)
        self.register_buffer("centers", centers)
        self.register_buffer("widths", widths)
        self.register_buffer(
            "pose_scale",
            torch.tensor(
                [0.004, 0.004, 0.004, 0.05, 0.05, 0.05],
                dtype=centers.dtype,
                device=centers.device,
            ),
        )
        self.register_buffer(
            "jaw_scale",
            torch.tensor(0.05, dtype=centers.dtype, device=centers.device),
        )
        self.register_buffer("pose_weights", pose_weights)
        self.register_buffer("jaw_weights", weights[6].clone() - 1.0)
        self.register_buffer("pose_covariance", pose_eye.repeat(self.POSE_DIM, 1, 1))
        self.register_buffer("jaw_covariance", jaw_eye)
        self.register_buffer(
            "pose_process_covariance",
            process_noise * pose_eye.repeat(self.POSE_DIM, 1, 1),
        )
        self.register_buffer("jaw_process_covariance", process_noise * jaw_eye)
        self.register_buffer(
            "measurement_variance",
            torch.full(
                (self.DIM,),
                measurement_noise,
                dtype=centers.dtype,
                device=centers.device,
            ),
        )

    def basis(self, state, action):
        center_state = self.centers[: self.POSE_DIM, :, 0].transpose(0, 1)
        center_action = self.centers[: self.POSE_DIM, :, 1].transpose(0, 1)
        pose_width = self.widths[: self.POSE_DIM].transpose(0, 1)
        state_position_error = state[..., None, :3] - center_state[:, :3]
        state_rotation_error = self.relative_rotation_vector(center_state[:, 3:6], state[..., None, 3:6])
        action_position_error = action[..., None, :3] - center_action[:, :3]
        action_rotation_error = self.body_rotation_vector_difference(center_action[:, 3:6], action[..., None, 3:6])
        pose_error = torch.cat(
            (
                state_position_error,
                state_rotation_error,
                action_position_error,
                action_rotation_error,
            ),
            dim=-1,
        )
        pose_width = torch.cat((pose_width, pose_width), dim=-1)
        pose_basis = torch.exp(-0.5 * (pose_error / pose_width).square().sum(dim=-1))

        jaw_pair = torch.stack((state[..., 6], action[..., 6]), dim=-1)
        jaw_error = jaw_pair.unsqueeze(-2) - self.centers[6]
        jaw_basis = torch.exp(-jaw_error.square().sum(dim=-1) / (2.0 * self.widths[6].square()))
        return pose_basis, jaw_basis

    def forward(self, state, action):
        state = torch.as_tensor(state, dtype=self.centers.dtype, device=self.centers.device)
        action = torch.as_tensor(action, dtype=self.centers.dtype, device=self.centers.device)
        pose_basis, jaw_basis = self.basis(state, action)
        normalized_action = action[..., : self.POSE_DIM] / self.pose_scale
        residual_gain = torch.einsum("...n,jnl->...jl", pose_basis, self.pose_weights)
        normalized_increment = normalized_action + torch.einsum("...jl,...l->...j", residual_gain, normalized_action)
        pose_increment = self.pose_scale * normalized_increment
        jaw_increment = ((1.0 + (jaw_basis * self.jaw_weights).sum(dim=-1)) * action[..., 6]).unsqueeze(-1)
        position = state[..., :3] + pose_increment[..., :3]
        rotation = self.compose_rotation_vectors(-pose_increment[..., 3:6], state[..., 3:6])
        jaw = state[..., 6:7] + jaw_increment
        return torch.cat((position, rotation, jaw), dim=-1)

    @torch.no_grad()
    def update(self, state, action, next_state):
        state = torch.as_tensor(state, dtype=self.centers.dtype, device=self.centers.device)
        action = torch.as_tensor(action, dtype=self.centers.dtype, device=self.centers.device)
        next_state = torch.as_tensor(next_state, dtype=self.centers.dtype, device=self.centers.device)

        pose_basis, jaw_basis = self.basis(state, action)
        observed_pose_increment = torch.cat((
            next_state[:3] - state[:3],
            self.relative_rotation_vector(state[3:6], next_state[3:6]),
        ))
        observed_pose_increment = observed_pose_increment / self.pose_scale
        normalized_action = action[: self.POSE_DIM] / self.pose_scale
        pose_feature = (pose_basis[:, None] * normalized_action[None, :]).reshape(-1)
        pose_weights = self.pose_weights.reshape(self.POSE_DIM, -1)

        pose_covariance_prior = self.pose_covariance + self.pose_process_covariance
        pose_denominator = (
            torch.einsum("p,opq,q->o", pose_feature, pose_covariance_prior, pose_feature)
            + self.measurement_variance[: self.POSE_DIM]
        )
        pose_gain = torch.einsum("opq,q->op", pose_covariance_prior, pose_feature) / pose_denominator[:, None]
        pose_innovation = observed_pose_increment - normalized_action - pose_weights @ pose_feature
        pose_weights.add_(pose_gain * pose_innovation[:, None])

        pose_eye = torch.eye(pose_feature.numel(), dtype=state.dtype, device=state.device)
        pose_correction = pose_eye - pose_gain.unsqueeze(-1) * pose_feature[None, None, :]
        self.pose_covariance.copy_(pose_correction @ pose_covariance_prior)

        jaw_feature = jaw_basis * (action[6] / self.jaw_scale)
        jaw_covariance_prior = self.jaw_covariance + self.jaw_process_covariance
        jaw_denominator = jaw_feature @ jaw_covariance_prior @ jaw_feature + self.measurement_variance[6]
        jaw_gain = (jaw_covariance_prior @ jaw_feature) / jaw_denominator
        observed_jaw_increment = (next_state[6] - state[6]) / self.jaw_scale
        jaw_innovation = observed_jaw_increment - action[6] / self.jaw_scale - self.jaw_weights @ jaw_feature
        self.jaw_weights.add_(jaw_gain * jaw_innovation)
        jaw_eye = torch.eye(self.num_basis, dtype=state.dtype, device=state.device)
        jaw_correction = jaw_eye - jaw_gain.unsqueeze(-1) * jaw_feature.unsqueeze(0)
        self.jaw_covariance.copy_(jaw_correction @ jaw_covariance_prior)

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
        quaternion = torch.where(quaternion[..., :1] < 0, -quaternion, quaternion)
        vector_norm = torch.linalg.vector_norm(quaternion[..., 1:], dim=-1)
        angle = 2.0 * torch.atan2(vector_norm, quaternion[..., 0].clamp_min(0.0))
        scale = angle / vector_norm.clamp_min(torch.finfo(quaternion.dtype).eps)
        scale = torch.where(vector_norm > 1e-7, scale, torch.full_like(scale, 2.0))
        return scale.unsqueeze(-1) * quaternion[..., 1:]

    @classmethod
    def compose_rotation_vectors(cls, left, right):
        left_quaternion = cls._rotation_vector_to_quaternion(left)
        right_quaternion = cls._rotation_vector_to_quaternion(right)
        return cls._quaternion_to_rotation_vector(cls._quaternion_multiply(left_quaternion, right_quaternion))

    @classmethod
    def body_rotation_vector_difference(cls, current, target):
        current_quaternion = cls._rotation_vector_to_quaternion(current)
        target_quaternion = cls._rotation_vector_to_quaternion(target)
        current_inverse = torch.cat((current_quaternion[..., :1], -current_quaternion[..., 1:]), dim=-1)
        return cls._quaternion_to_rotation_vector(cls._quaternion_multiply(current_inverse, target_quaternion))

    @classmethod
    def relative_rotation_vector(cls, current, target):
        current_quaternion = cls._rotation_vector_to_quaternion(current)
        target_quaternion = cls._rotation_vector_to_quaternion(target)
        current_inverse = torch.cat((current_quaternion[..., :1], -current_quaternion[..., 1:]), dim=-1)
        return -cls._quaternion_to_rotation_vector(cls._quaternion_multiply(target_quaternion, current_inverse))
