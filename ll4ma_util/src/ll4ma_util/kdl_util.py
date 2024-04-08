import PyKDL as kdl
import torch
import numpy as np

def rotation_log_matrix(R, R2=None):
    """
    Takes a Rotation matrix (or 2) and returns the axis-angle representation
      - R : (3,3) torch.tensor representation of rotation
      - R2 : (3,3) torch.tensor representation of rotation
    Returns:
      - angle : (,) float of same type as R
      - axis : (3,) array of same type as input R
    """
    if R2 is not None:
        R = R @ R2.T
    angle, axis = kdl.Rotation(*R.flatten().cpu()).GetRotAngle()
    if torch.is_tensor(R):
        angle = torch.tensor([angle], dtype=R.dtype, device=R.device).squeeze()
        axis = torch.tensor([*axis], dtype=R.dtype, device=R.device)
    else:
        angle = np.array([angle])[0]
        axis = np.array([*axis])
    return angle, axis

def orientation_error(R,R2):
    """
    Takes two Rotation matrices and returns the angle between them as the error
      - R : (3,3) torch.tensor representation of rotation
      - R2 : (3,3) torch.tensor representation of rotation
    Returns:
      - error : (,) torch.tensor
    """
    angle, axis = rotation_log_matrix(R,R2)
    return angle

def orientation_error_grad(R,R2):
    """
    Takes two Rotation matrices and returns the axis of the angle between them as the gradient of the error
      - R : (3,3) torch.tensor representation of rotation
      - R2 : (3,3) torch.tensor representation of rotation
    Returns:
      - gradient : (3,) torch.tensor
    """
    angle, axis = rotation_log_matrix(R,R2)
    return axis*angle
