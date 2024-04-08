import torch
import torch.nn.functional as F
import transformations
import numpy as np
import timeit

def quat_mul(a, b):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = torch.stack([x, y, z, w], dim=-1).view(shape)

    return quat


def quat_conjugate(a):
    shape = a.shape
    a = a.reshape(-1, 4)
    return torch.cat((-a[:, :3], a[:, -1:]), dim=-1).view(shape)


def quaternion_error(desired, current, square=False, numpy=False, flatten=False):
    q_c = quat_conjugate(current)
    q_r = quat_mul(desired, q_c)
    error = q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)
    if square:
        error = error**2
    if numpy:
        error = error.numpy()
    if flatten:
        error = error.flatten()
    return error


def position_error(desired, current, square=False, numpy=False, flatten=False):
    error = desired - current
    if square:
        error = error**2
    if numpy:
        error = error.numpy()
    if flatten:
        error = error.flatten()
    return error

def quaternion_to_matrix_batch(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def matrix_to_quaternion_batch(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    
def construct_homo_mat(base_pose, link_pose, fix_frame_type):
    """ 
    Construct 4x4 homo matrix between the base frame and a specific link's frame (e.g. end-effector frame).
    
    1. Input:     
    base_pose (7,): pose of the base frame (x, y, z, qx, qy, qz, qw) 
    link_pose (7,): pose of the link frame (x, y, z, qx, qy, qz, qw) 
    fix_frame_type: Whether to align Isaacgym's coordinate frame with PyKDL frame. 
            Perform some simple swap of quaternion elements. Only applicable to a few links, not all.
            
    2. Output:
    T (4,4): 4x4 homo matrix
    """  
    B = int(base_pose.shape[0]/(7))
    New_base_pose = base_pose.view(B,7,)
    New_link_pose = link_pose.view(B,7,)
    p_base = New_base_pose[:,:3].clone().detach()
    p_end_effector = New_link_pose[:,:3].clone().detach()

    R_base = torch.zeros(B,3,3)
    R_end_effector = torch.zeros(B,3,3)
    R_base = quaternion_to_matrix_batch(New_base_pose[:,3:])
    R_end_effector = quaternion_to_matrix_batch(New_link_pose[:,3:])

    # Compute the translation vector
    t = torch.sub(p_end_effector , p_base)

    R = torch.bmm(R_end_effector, R_base.transpose(1,2))

    # Assemble the homogeneous transformation matrix
    T = torch.zeros(B, 4, 4)
    T[:B, :3, :3] = R
    T[:B, :3, 3] = t
    T[:B, 3, 3] = 1
    
    quat = matrix_to_quaternion_batch(T[:,:3,:3])
    
    new_quat = quat.view(int(B/8),8,4)

    new_quat[:,:7,1], new_quat[:,:7,3] = -new_quat[:,:7,3], -new_quat[:,:7,1]

    temp = new_quat[:, 7, 1].clone()
    new_quat[:, 7, 1] = new_quat[:, 7, 3]
    new_quat[:, 7, 3] = temp
   
    new_quat[:,7:8,2] *= -1
    
    T[:,:3,:3] = quaternion_to_matrix_batch(new_quat.view(B,4))
    
    T[:B,:2,3] *= -1

    return T





############################################ old math before batch ###############################################
# import torch
# def quat_mul(a, b):
#     assert a.shape == b.shape
#     shape = a.shape
#     a = a.reshape(-1, 4)
#     b = b.reshape(-1, 4)

#     x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
#     x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
#     ww = (z1 + x1) * (x2 + y2)
#     yy = (w1 - y1) * (w2 + z2)
#     zz = (w1 + y1) * (w2 - z2)
#     xx = ww + yy + zz
#     qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
#     w = qq - ww + (z1 - y1) * (y2 - z2)
#     x = qq - xx + (x1 + w1) * (x2 + w2)
#     y = qq - yy + (w1 - x1) * (y2 + z2)
#     z = qq - zz + (z1 + y1) * (w2 - x2)

#     quat = torch.stack([x, y, z, w], dim=-1).view(shape)

#     return quat


# def quat_conjugate(a):
#     shape = a.shape
#     a = a.reshape(-1, 4)
#     return torch.cat((-a[:, :3], a[:, -1:]), dim=-1).view(shape)


# def quaternion_error(desired, current, square=False, numpy=False, flatten=False):
#     q_c = quat_conjugate(current)
#     q_r = quat_mul(desired, q_c)
#     error = q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)
#     if square:
#         error = error**2
#     if numpy:
#         error = error.numpy()
#     if flatten:
#         error = error.flatten()
#     return error


# def position_error(desired, current, square=False, numpy=False, flatten=False):
#     error = desired - current
#     if square:
#         error = error**2
#     if numpy:
#         error = error.numpy()
#     if flatten:
#         error = error.flatten()
#     return error