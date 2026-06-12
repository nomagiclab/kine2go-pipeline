import torch
from genesis.utils.geom import transform_by_quat, xyz_to_quat

from go2_genesis.utils import XYZ_DIM


def calc_heading_from_quat(quat: torch.Tensor) -> torch.Tensor:
    """Return yaw heading for batched real-first quaternions with shape (N, 4)."""
    forward_vector = torch.zeros((quat.shape[0], XYZ_DIM), device=quat.device)
    forward_vector[:, 0] = 1.0
    rotated_vector = transform_by_quat(forward_vector, quat)
    heading = torch.atan2(rotated_vector[:, 1], rotated_vector[:, 0])
    return heading


def calc_quat_from_heading(heading: torch.Tensor) -> torch.Tensor:
    """Build real-first yaw-only quaternions from a heading tensor with shape (N,)."""
    heading = heading.unsqueeze(-1)
    heading = torch.cat([torch.zeros_like(heading), torch.zeros_like(heading), heading], dim=-1)
    return xyz_to_quat(heading, rpy=True, degrees=False)


# Taken from pytorch3d
def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to axis/angle.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    sin_half_angles_over_angles = 0.5 * torch.sinc(half_angles / torch.pi)
    # angles/2 are between [-pi/2, pi/2], thus sin_half_angles_over_angles
    # can't be zero
    return quaternions[..., 1:] / sin_half_angles_over_angles
