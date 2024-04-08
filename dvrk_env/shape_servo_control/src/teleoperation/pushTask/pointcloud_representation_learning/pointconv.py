"""
Code taken from PointConv [Wu et al. 2019]:
https://github.com/DylanWusee/pointconv_pytorch/blob/master/utils/pointconv_util.py
Modified by Wesley Khademi
"""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# from utils import torch_utils, KNN


def masked_knn(pts1, valid1, pts2, valid2, knn):
    with torch.no_grad():
        nn_dists, nn_idx = knn(pts1, valid1, pts2, valid2)
        nn_dists = nn_dists.permute(0, 2, 1).contiguous()
        nn_idx = nn_idx.permute(0, 2, 1). contiguous()

    return nn_dists, nn_idx


def masked_group(points, idx, valid):
    B, N = valid.shape

    new_points = grouping_operation(points, idx)
    new_points = new_points #* valid.view(B, 1, N, 1)

    return new_points


def group(xyz, points, valid_xyz, new_xyz, valid_new_xyz, nn_idx=None, knn=KNN(k=8)):
    """
    Input:
        xyz: input points position data, [B, S, C]
        points: input points data, [B, S, D]
        valid_xyz: tensor indicating valid 'xyz' points (non-filler points)
        new_xyz: query points position data, [B, N, C]
        nsample: number of nearest neighbors to sample
        valid_new_xyz: tensor indicating valid 'new_xyz' points (non-filler points)
    Return:
        grouped_xyz_norm: relative point positions [B, 3, nsample, N]
        new_points: sampled points data, [B,  C+D, nsample, N]
        valid_knn: valid nearest neighbors, [B, nsample, N]
    """
    device =  xyz.device
    B, C, N = new_xyz.shape
    _, D, _ = points.shape

    if nn_idx is None:
        _, idx = masked_knn(xyz, valid_xyz, new_xyz, valid_new_xyz, knn)  # [B, npoint, nsample]
    else:
        idx = nn_idx

    grouped_xyz = masked_group(xyz, idx, valid_new_xyz)  # [B, C, npoint, nsample]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, C, N, 1)
    grouped_points = masked_group(points, idx, valid_new_xyz)
    new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=1)  # [B, C+D, npoint, nsample]
    new_points = new_points.permute(0, 1, 3, 2)
    grouped_xyz_norm = grouped_xyz_norm.permute(0, 1, 3, 2)

    return new_points, grouped_xyz_norm


class WeightNet(nn.Module):
    def __init__(self, in_channel, out_channel, hidden_unit=[8, 8]):
        super(WeightNet, self).__init__()

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        if hidden_unit is None or len(hidden_unit) == 0:
            self.mlp_convs.append(nn.Conv2d(in_channel, out_channel, 1, bias=False))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
        else:
            self.mlp_convs.append(nn.Conv2d(in_channel, hidden_unit[0], 1, bias=False))
            self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[0]))
            for i in range(1, len(hidden_unit)):
                self.mlp_convs.append(nn.Conv2d(hidden_unit[i - 1], hidden_unit[i], 1, bias=False))
                self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[i]))
            self.mlp_convs.append(nn.Conv2d(hidden_unit[-1], out_channel, 1, bias=False))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))

    def forward(self, localized_xyz, valid):
        weights = localized_xyz
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            weights = F.leaky_relu(bn(conv(weights)) #* valid, negative_slope=0.2)

        return weights


class PointConv(nn.Module):
    def __init__(self, nsample, point_dim, in_channel, out_channel, mlp=[],
                 final_norm='batch', final_act='leaky_relu'):
        super(PointConv, self).__init__()
        self.nsample = nsample
        self.final_norm = final_norm
        self.final_act = final_act

        self.knn = KNN(k=nsample)

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_ch = point_dim + in_channel
        for out_ch in mlp:
            self.mlp_convs.append(nn.Conv2d(last_ch, out_ch, 1, bias=False))
            self.mlp_bns.append(nn.BatchNorm2d(out_ch))
            last_ch = out_ch

        self.weightnet = WeightNet(point_dim, 4)
        self.linear = nn.Linear(4 * last_ch, out_channel, bias=False)
        if self.final_norm == 'batch':
            self.norm_linear = nn.BatchNorm1d(out_channel)
        else:
            self.norm_linear = nn.Identity()

    def forward(self, xyz, points, valid_xyz, new_xyz, valid_new_xyz, nn_idx=None):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
            valid_xyz: indicates valid xyz points [B, N]
            new_xyz: sampled points position data [B, C, S]
            valid_new_xyz: indicates valid new_xyz poins [B, S]
        Return:
            new_points: sample points feature data, [B, D', S]
        """
        B, _, N = xyz.shape
        _, _, S = new_xyz.shape

        new_points, grouped_xyz_norm = group(xyz, points, valid_xyz, new_xyz,
                                             valid_new_xyz, nn_idx, self.knn)

        valid = valid_new_xyz.view(B, 1, 1, S).expand(B, 1, self.nsample, S)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.leaky_relu(bn(conv(new_points)) #* valid, negative_slope=0.2)

        weights = self.weightnet(grouped_xyz_norm, valid)
        new_points = torch.matmul(input=new_points.permute(0, 3, 1, 2), other=weights.permute(0, 3, 2, 1)).view(B, S, -1)
        new_points = self.linear(new_points)
        new_points = new_points.permute(0, 2, 1).contiguous()

        if self.final_norm is not None:
            new_points = self.norm_linear(new_points) #* valid_new_xyz.view(B, 1, S)

        if self.final_act == 'leaky_relu':
            new_points = F.leaky_relu(new_points, negative_slope=0.2)

        return new_points


class BottleneckPointConv(nn.Module):
    def __init__(self, nsample, point_dim, in_channel, out_channel, mlp=[],
                 bottleneck=4, residual=False):
        super(BottleneckPointConv, self).__init__()
        self.residual = residual

        self.reduce = nn.Conv1d(in_channel, in_channel//bottleneck, 1, bias=False)
        self.reduce_norm = nn.BatchNorm1d(in_channel//bottleneck)

        self.pointconv = PointConv(nsample=nsample, point_dim=point_dim,
                                   in_channel=in_channel//bottleneck,
                                   out_channel=out_channel//bottleneck, mlp=mlp)

        self.expand = nn.Conv1d(out_channel//bottleneck, out_channel, 1, bias=False)
        self.expand_norm = nn.BatchNorm1d(out_channel)

    def forward(self, xyz, points, valid_xyz, new_xyz, valid_new_xyz, nn_idx=None):
        reduced_points = self.reduce(points)
        reduced_points = self.reduce_norm(reduced_points) ##* valid_xyz.unsqueeze(dim=1)
        reduced_points = F.leaky_relu(reduced_points, negative_slope=0.2)

        new_points = self.pointconv(xyz, reduced_points, valid_xyz, new_xyz,
                                    valid_new_xyz, nn_idx=nn_idx)

        new_points = self.expand(new_points)
        new_points = self.expand_norm(new_points) ##* valid_new_xyz.unsqueeze(dim=1)
        if self.residual:
            new_points = F.leaky_relu(points + new_points, negative_slope=0.2)
        else:
            new_points = F.leaky_relu(new_points, negative_slope=0.2)

        return new_points
