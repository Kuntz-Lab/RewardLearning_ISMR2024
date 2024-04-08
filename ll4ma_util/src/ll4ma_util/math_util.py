import torch
import numpy as np
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation


def rotation_to_quat(R, normalize=False):
    istorch = torch.is_tensor(R)
    if istorch:
        device = R.device
        R = R.cpu().numpy()
    if normalize:
        # If R is not exactly a rotation matrix, we can leverage scipy to make it so
        R = Rotation.from_matrix(R).as_matrix()
    T = np.eye(4)
    T[:3,:3] = R
    quat = homogeneous_to_quat(T)
    if istorch:
        return torch.tensor(quat, device=device)
    return quat


def rotation_to_homogeneous(R):
    T = np.eye(4)
    T[:3,:3] = R
    return T


def homogeneous_to_position(T):
    if torch.is_tensor(T) and T.dim() == 3:
        p = T[:,:3,3]  # Accounts for batch dim
    else:
        p = T[:3,3]
    return p


def homogeneous_to_rotation(T):
    if torch.is_tensor(T) and T.dim() == 3:
        R = T[:,:3,:3]  # Accounts for batch dim
    else:
        R = T[:3,:3]
    return R


def homogeneous_to_quat(T):
    """
    Converts rotation matrix from homogeneous TF matrix to quaternion.
    """
    istorch = torch.is_tensor(T)
    if istorch:
        device = T.device
        T = T.cpu().numpy()
    q = Quaternion(matrix=T) # w, x, y, z
    # Need to switch to x, y, z, w
    q = torch.tensor([q.x, q.y, q.z, q.w], device=device) if istorch else np.array([q.x, q.y, q.z, q.w])
    return q


def homogeneous_to_pose(T):
    """
    Converts homogeneous TF matrix to a 3D position and quaternion.
    """
    return homogeneous_to_position(T), homogeneous_to_quat(T)


def quat_to_homogeneous(q):
    """
    Converts quaternion to homogeneous TF matrix. Assumes (x, y, z, w) quaternion input.
    """
    istorch = torch.is_tensor(q)
    if istorch:
        device = q.device
        q = q.cpu().numpy()
    T = Quaternion(q[3], q[0], q[1], q[2]).transformation_matrix  # Quaternion is (w, x, y, z)
    if istorch:
        T = torch.tensor(T, device=device)
    return T


def quat_to_rotation(q):
    """
    Converts quaternion to rotation matrix. Assumes (x, y, z, w) quaternion input.
    """
    return quat_to_homogeneous(q)[:3,:3]


def pose_to_homogeneous(p, q=None, R=None):
    """
    Converts position and orientation to a homogeneous TF matrix.
    Supports multiple orientation representations.

    Args:
        p: 3D Position of shape (3,)
        q: Quaternion (xyzw) orientation of shape (4,)
        R: Rotation matrix of shape (3,3)

    Note only specify one orientation representation.
    """
    if q is not None:
        T = quat_to_homogeneous(q)
    elif R is not None:
        T = rotation_to_homogeneous(R)
    else:
        raise ValueError("Must specify one of the orientation representations")
    T[:3,3] = p
    return T


def homogeneous_inverse(T):
    """
    Computes inverse of homogeneous TF matrix.
    """
    if torch.is_tensor(T) and T.dim() == 3:
        # Handle batch
        new_T = torch.eye(4).unsqueeze(0).repeat(T.size(0), 1, 1)
        new_T[:,:3,:3] = T[:,:3,:3].transpose(1, 2)
        new_T[:,:3,3] = -torch.bmm(new_T[:,:3,:3], T[:,:3,3].unsqueeze(-1)).squeeze(-1)
    else:
        new_T = torch.eye(4) if torch.is_tensor(T) else np.eye(4)
        new_T[:3,:3] = T[:3,:3].T
        new_T[:3,3] = -(new_T[:3,:3] @ T[:3,3])
    return new_T
    

def random_quat():
    q = Quaternion.random()
    return np.array([q.x, q.y, q.z, q.w])


def random_rotation():
    return quat_to_rotation(random_quat())

def uniform_random_planar_pose(
        min_pos,
        max_pos,
        sample_axis=[0,0,1],
        min_angle=-np.pi,
        max_angle=np.pi
):
    pos_ranges = zip(min_pos, max_pos)
    pos = np.array([np.random.uniform(*pos_range) for pos_range in pos_ranges])
    angle = np.random.uniform(min_angle, max_angle)
    q = Quaternion(axis=sample_axis, angle=angle)
    quat = np.array([q.x, q.y, q.z, q.w])
    return pos, quat


def gaussian_random_planar_poses(
        pos_mean=[0., 0., 0.],
        pos_std=[0.1, 0.1, 0.1],
        angle_std=1.,
        sample_axis=[0,0,1],
        n_samples=1,
        rot_return='matrix'
):
    """
    Computes random 3D poses where positions are sampled from a Gaussian, and
    an orientation angle about a sample axis is sampled from a Gaussian.

    Args:
        pos_mean: Mean 3D position
        pos_std: Standard deviation for 3D position
        angle_std: Standard deviation of angle about sample axis
        sample_axis: Sample axis for randomizing orientation
        n_samples: Number of samples to compute
        rot_return: Orientation representation to return, one of 'quat', 'matrix',
                    'euler', or 'rotvec'.
    """
    pos_samples = np.random.normal(pos_mean, pos_std, size=(n_samples, 3))
    angle_samples = np.expand_dims(np.random.normal(0., angle_std, size=(n_samples,)), -1)
    axes = np.expand_dims(np.array(sample_axis), 0).repeat(n_samples, axis=0)
    rots = Rotation.from_rotvec(angle_samples * axes)
    if rot_return == 'quat':
        rot_samples = rots.as_quat()
    elif rot_return == 'matrix':
        rot_samples = rots.as_matrix()
    elif rot_return == 'euler':
        rot_samples = rots.as_euler()
    elif rot_return == 'rotvec':
        rot_samples = rots.as_rotvec()
    else:
        raise ValueError(f"Unknown rotation return type: {rot_return}")
    return pos_samples, rot_samples

def get_x_axis_rotation(theta):
    return np.array([[1,             0,              0],
                     [0, np.cos(theta), -np.sin(theta)],
                     [0, np.sin(theta),  np.cos(theta)]])


def get_y_axis_rotation(theta):
    return np.array([[ np.cos(theta), 0, np.sin(theta)],
                     [             0, 1,             0],
                     [-np.sin(theta), 0, np.cos(theta)]])


def get_z_axis_rotation(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta),  np.cos(theta), 0],
                     [            0,              0, 1]])


def construct_rotation_matrix(x=None, y=None, z=None, normalize=True):
    """
    Constructs a (right-handed) rotation matrix given the individual
    axes. Must specify at least two axes, and the third will be computed
    as a cross product of the others.

    Args:
        x (ndarray): X-axis of shape (3,)
        y (ndarray): Y-axis of shape (3,)
        z (ndarray): Z-axis of shape (3,)
        normalize (bool): Inputs are normalized to unit length if True. If you know
                          your vectors are unit already you can set False.
    """
    if normalize:
        x = x if x is None else x / np.linalg.norm(x)
        y = y if y is None else y / np.linalg.norm(y)
        z = z if z is None else z / np.linalg.norm(z)

    if x is None:
        if y is None or z is None:
            raise ValueError("Must specify at least two axes to construct rotation matrix")
        x = np.cross(y, z)
        x /= np.linalg.norm(x)
    elif y is None:
        if x is None or z is None:
            raise ValueError("Must specify at least two axes to construct rotation matrix")
        y = np.cross(z, x)
        y /= np.linalg.norm(y)
    elif z is None:
        if x is None or y is None:
            raise ValueError("Must specify at least two axes to construct rotation matrix")
        z = np.cross(x, y)
        z /= np.linalg.norm(z)

    R = np.eye(3)
    R[:,0] = x
    R[:,1] = y
    R[:,2] = z

    # Sometimes it's close but apparently not perfect which is problematic when
    # you try to do quaternion conversions, so leveraging scipy's normalization
    R = Rotation.from_matrix(R).as_matrix()

    return R


def quat_mul(a, b):
    x1, y1, z1, w1 = a
    x2, y2, z2, w2 = b
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    if torch.is_tensor(a):
        quat = torch.stack([x, y, z, w], dim=-1).view(shape)
    else:
        quat = np.array([x, y, z, w])

    return quat


def geodesic_error(R1, R2):
    return np.arccos(np.clip((np.trace(np.dot(R1, R2.T)) - 1.) / 2., -1, 1))


def rotation_matrix_log(T):
    """
    Lynch And Park Equation 3.53 / 3.54
    with minor check to ensure the matrix is sufficienty close to valid rotation matrix
    if not, row normalize
    """
    if torch.abs(torch.trace(T) - 1) > 2.0:
        T = torch.nn.functional.normalize(T, p=2, d=0)
    # make the value just tiny bit smaller to avoid acos limits
    phi = torch.arccos(0.5 * (torch.trace(T) - 1) * (1 - 1e-10))
    skew_symmetric_unit_rotation = (1 / (2 * torch.sin(phi))) * (T - T.T)

    # cleary this variable name is ridiculous...
    return skew_symmetric_unit_rotation, phi


def adjoint_matrix(T):
    """
    Takes a list of 4x4 transformation matrix and converts it to a 6x6 adjoint matrix
    See Lynch and Park 3.83 (assumes the angular and positional velocities are switched)
    """
    sshape = T.shape
    if len(sshape) == 2:
        T = T.unsqueeze(0)

    adj = torch.zeros((T.shape[0],6,6), dtype=T.dtype)
    R = T[:,:3,:3]
    adj[:,3:,:3] = R
    adj[:,:3,3:] = R
    adj[:,3:,3:] = skew_symmetric(T[:,3,:3]) @ R

    if len(sshape) == 2:
        adj = adj.squeeze(0)
    return adj


def skew_symmetric(pos):
    """
    Takes a position vector and coverts it to a 3x3 skew-symmetric matrix
    See Lynch and Park 3.30
    """
    sshape = pos.shape
    if len(sshape) == 1:
        pos = pos.unsqueeze(0)
    skew = torch.zeros((pos.shape[0],3,3), dtype=pos.dtype)
    skew[:,0,1] = -pos[:,2]
    skew[:,0,2] = pos[:,1]
    skew[:,1,0] = pos[:,2]
    skew[:,1,2] = pos[:,0]
    skew[:,2,0] = -pos[:,1]
    skew[:,2,1] = pos[:,0]
    if len(sshape) == 1:
        skew = skew.squeeze(0)
    return skew


if __name__ == '__main__':
    R1 = random_rotation()
    R2 = random_rotation()

    print(geodesic_error(R1, R2))
