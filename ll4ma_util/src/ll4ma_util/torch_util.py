from math import pi

import torch
import torch.nn.functional as F

from ll4ma_util import ui_util


def convert_to_float_tensors(tensor_dict, keys=[]):
    keys = keys if keys else tensor_dict.keys()
    for k in keys:
        if torch.is_tensor(tensor_dict[k]):
            tensor_dict[k] = tensor_dict[k].float()
        else:
            tensor_dict[k] = torch.FloatTensor(tensor_dict[k])


def convert_to_long_tensors(tensor_dict, keys=[]):
    keys = keys if keys else tensor_dict.keys()
    for k in keys:
        if torch.is_tensor(tensor_dict[k]):
            tensor_dict[k] = tensor_dict[k].long()
        else:
            tensor_dict[k] = torch.LongTensor(tensor_dict[k])


def make_batch(x, n_batch=1):
    """
    Batchifies a tensor by adding a batch dim and repeating it over that
    dimension n_batch times.
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    ndim = x.dim()
    x = x.unsqueeze(0)
    if n_batch > 1:
        x = x.repeat(n_batch, *[1]*ndim)  # unrolls list of 1's of length ndim
    return x


def move_batch_to_device(batch_dict, device):
    """
    Recursive function that moves a (nested) dictionary of tensors to the specified device.
    """
    for k, v in batch_dict.items():
        if isinstance(v, torch.Tensor):
            batch_dict[k] = v.to(device)
        elif isinstance(v, dict):
            move_batch_to_device(v, device)


def move_models_to_device(models_dict, device):
    """
    Assuming flat dictionary where values are all type torch.nn.Module.
    """
    for k, v in models_dict.items():
        models_dict[k] = v.to(device)


def set_models_to_train(models_dict):
    for v in models_dict.values():
        v.train()


def set_models_to_eval(models_dict):
    for v in models_dict.values():
        v.eval()


def load_state_dicts(models_dict, state_dicts):
    for k, v in models_dict.items():
        if k not in state_dicts:
            ui_util.print_warning(f"Model {k} does not have state to load")
            continue
        v.load_state_dict(state_dicts[k])


def accumulate_parameters(models_dict):
    params = []
    for v in models_dict.values():
        params += list(v.parameters())
    return params



# torch quaternion functions from NVIDIA:


@torch.jit.script
def quat_mul(a, b):
    # type: (Tensor,Tensor) -> Tensor
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


@torch.jit.script
def quat_apply(a, b):
    # type: (Tensor,Tensor) -> Tensor
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    xyz = a[:, :3]
    t = xyz.cross(b, dim=-1) * 2
    return (b + a[:, 3:] * t + xyz.cross(t, dim=-1)).view(shape)


@torch.jit.script
def qpose_mul(a, b):
    # type: (Tensor,Tensor) -> Tensor
    c = torch.zeros_like(a)
    c[:,:3] = a[:,:3] + quat_apply(a[:,3:7], b[:,:3])
    c[:,3:7] = quat_mul(a[:,3:7], b[:,3:7])
    return c


@torch.jit.script
def tf_combine(q1, t1, q2, t2):
    return quat_mul(q1, q2), quat_apply(q1, t2) + t1


def position_error(desired, current, square=False, numpy=False, flatten=False):
    error = desired - current
    if square:
        error = error**2
    if numpy:
        error = error.cpu().numpy()
    if flatten:
        error = error.flatten()
    return error


@torch.jit.script
def ortho6d_to_rotation(ortho6d):
    """
    Computes rotation matrix from 6D orthorgraphic representation.

    From https://github.com/papagina/RotationContinuity/blob/master/sanity_test/code/tools.py#L47
    """
    if ortho6d.dim() == 1:
        ortho6d = ortho6d.unsqueeze(0)
    x_raw = ortho6d[:,0:3]
    y_raw = ortho6d[:,3:6]

    x = F.normalize(x_raw)
    z = torch.cross(x,y_raw)
    z = F.normalize(z)
    y = torch.cross(z,x)

    x = x.view(-1,3,1)
    y = y.view(-1,3,1)
    z = z.view(-1,3,1)
    matrix = torch.cat((x,y,z), 2)
    return matrix.squeeze()


# @torch.jit.script
def random_quat(batch_size=1, device='cuda'):
    """
    Computes a random quaternion sampled uniformly from the unit sphere.

    Note this is primarily implemented so that it uses torch random generator
    instead of numpy to avoid issues with how torch/np random generators
    interact when training with randomization:
        https://tanelp.github.io/posts/a-bug-that-plagues-thousands-of-open-source-ml-projects/

    See: https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L261
    """
    r1 = torch.rand(batch_size).to(device).view(-1, 1)
    r2 = torch.rand(batch_size).to(device).view(-1, 1)
    r3 = torch.rand(batch_size).to(device).view(-1, 1)

    w = torch.sqrt(1.0 - r1) * (torch.sin(2.0 * pi * r2))
    x = torch.sqrt(1.0 - r1) * (torch.cos(2.0 * pi * r2))
    y = torch.sqrt(r1)       * (torch.sin(2.0 * pi * r3))
    z = torch.sqrt(r1)       * (torch.cos(2.0 * pi * r3))
    # Normalize just to be sure since there can be slight numerical differences from pure unit
    return F.normalize(torch.cat([x, y, z, w], dim=-1))


@torch.jit.script
def quat_conjugate(a):
    # type: (Tensor) -> Tensor
    shape = a.shape
    a = a.reshape(-1, 4)
    return torch.cat((-a[:, :3], a[:, -1:]), dim=-1).view(shape)


@torch.jit.script
def quaternion_error(desired, current, square=False):
    # type: (Tensor, Tensor, bool) -> Tensor
    q_c = quat_conjugate(current)
    q_r = quat_mul(desired, q_c)
    error = q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)
    if square:
        error = error**2
    return error


def random_rotation(batch_size=1, device='cuda'):
    return quat_to_rotation(random_quat(batch_size, device))


@torch.jit.script
def quat_to_rotation(q):
    batch = q.size(0)
    qx = q[:,0].view(batch, 1)
    qy = q[:,1].view(batch, 1)
    qz = q[:,2].view(batch, 1)
    qw = q[:,3].view(batch, 1)

    # Unit quaternion rotation matrices computatation
    xx = qx*qx
    yy = qy*qy
    zz = qz*qz
    xy = qx*qy
    xz = qx*qz
    yz = qy*qz
    xw = qx*qw
    yw = qy*qw
    zw = qz*qw

    row0 = torch.cat((1-2*yy-2*zz, 2*xy - 2*zw, 2*xz + 2*yw), 1)
    row1 = torch.cat((2*xy+ 2*zw,  1-2*xx-2*zz, 2*yz-2*xw  ), 1)
    row2 = torch.cat((2*xz-2*yw,   2*yz+2*xw,   1-2*xx-2*yy), 1)

    matrix = torch.cat((row0.view(batch,1,3), row1.view(batch,1,3), row2.view(batch,1,3)),1)
    return matrix


@torch.jit.script
def quat_to_axisangle(q):
    # type: (Tensor) -> Tuple[Tensor, Tensor]
    vnorm = torch.norm(q[:,:3], p=2, dim=-1)
    axis = q[:,:3] / vnorm.unsqueeze(1)
    angle = torch.atan2(vnorm, q[:,3])
    return axis, angle


def pose_to_homogeneous(p, q=None, R=None):
    # TODO need to integrate this into math_util and add support for batch there
    if q is not None:
        R = quat_to_rotation(q)
    T = make_batch(torch.eye(4), p.size(0))
    T[:,:3,:3] = R
    T[:,:3,3] = p
    return T


@torch.jit.script
def get_pinv(mat, inv_threshold):
    # type: (Tensor, float) -> Tensor
    umat,smat,vmat = torch.svd(mat)
    smat = 1 / torch.clamp(smat, min=inv_threshold)
    return vmat @ torch.diag_embed(smat) @ umat.transpose(1,2)


def construct_rotation_matrix(x=None, y=None, z=None, normalize=True):
    """
    Constructs a (right-handed) rotation matrix given the individual
    axes. Must specify at least two axes, and the third will be computed
    as a cross product of the others.

    TODO want to integrate this better with math_util version

    Args:
        x (ndarray): X-axis of shape (3,)
        y (ndarray): Y-axis of shape (3,)
        z (ndarray): Z-axis of shape (3,)
        normalize (bool): Inputs are normalized to unit length if True. If you know
                          your vectors are unit already you can set False.
    """
    if normalize:
        x = x if x is None else F.normalize(x, dim=-1)
        y = y if y is None else F.normalize(y, dim=-1)
        z = z if z is None else F.normalize(z, dim=-1)

    if x is None:
        if y is None or z is None:
            raise ValueError("Must specify at least two axes to construct rotation matrix")
        x = F.normalize(y.cross(z, dim=-1), dim=-1)
    elif y is None:
        if x is None or z is None:
            raise ValueError("Must specify at least two axes to construct rotation matrix")
        y = F.normalize(z.cross(x, dim=-1), dim=-1)
    elif z is None:
        if x is None or y is None:
            raise ValueError("Must specify at least two axes to construct rotation matrix")
        z = F.normalize(x.cross(y, dim=-1), dim=-1)

    R = torch.eye(3).unsqueeze(0).repeat(x.size(0), 1, 1)
    R[:,:,0] = x
    R[:,:,1] = y
    R[:,:,2] = z

    return R


def batch_cov(points):
    """
    Computes sample covariance in batch, where points are of
    size (B, N, D) for batch size B, N samples, and vector size D.

    See: https://stackoverflow.com/q/71357619/3711266
    """
    B, N, D = points.size()
    mean = points.mean(dim=1).unsqueeze(1)
    diffs = (points - mean).reshape(B * N, D)
    prods = torch.bmm(diffs.unsqueeze(2), diffs.unsqueeze(1)).reshape(B, N, D, D)
    bcov = prods.sum(dim=1) / (N - 1)
    return bcov

@torch.jit.script
def orientation_error(R, R2):
    # type: (Tensor, Tensor) -> Tensor
    """
    Takes two Rotation matrices and returns the angle between them as the error
      - R : (3,3) torch.tensor representation of rotation
      - R2 : (3,3) torch.tensor representation of rotation
    Returns:
      - error : (,) torch.tensor
    """
    T = R @ R2.T
    if torch.abs(torch.trace(T) - 1) > 2.0:
        T = torch.nn.functional.normalize(T, p=2., dim=0)
    # make the value just tiny bit smaller to avoid acos limits
    theta = torch.arccos(0.5 * (torch.trace(T) - 1) * (1 - 1e-10))
    return theta


@torch.jit.script
def orientation_error_grad(R, R2):
    # type: (Tensor, Tensor) -> Tensor
    """
    Takes two Rotation matrices and returns the angle between them as the error
      - R : (3,3) torch.tensor representation of rotation
      - R2 : (3,3) torch.tensor representation of rotation
    Returns:
      - error : (,) torch.tensor
    """
    T = R @ R2.T
    if torch.abs(torch.trace(T) - 1) > 2.0:
        T = torch.nn.functional.normalize(T, p=2., dim=0)
    axis = torch.zeros_like(T[0])
    if torch.trace(T) == -1:
        theta = torch.pi
        axis[0] = T[0,2]
        axis[1] = T[1,2]
        axis[2] = T[2,2]+1
        axis = axis / (2*axis[2]).sqrt()
    else:
        # make the value just tiny bit smaller to avoid acos limits
        theta = torch.arccos(0.5 * (torch.trace(T) - 1) * (1 - 1e-10)).item()
        rot_skew = (T - T.T) / (2 * torch.sin(theta))
        axis[0] = rot_skew[2,1]
        axis[1] = rot_skew[0,2]
        axis[2] = rot_skew[1,0]

    return axis * theta

if __name__ == '__main__':
    # print(random_rotation(1))

    a = torch.tensor([[1,2,3],
                      [4,5,6],
                      [7,8,9]])
    print(make_batch(a, 11).shape)
