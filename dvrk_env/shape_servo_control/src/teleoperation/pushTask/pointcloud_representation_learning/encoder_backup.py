import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import torch_utils

from pointconv import BottleneckPointConv, PointConv


class PCFE(nn.Module):
    '''
        PointConv Feature Extraction module
    '''

    def __init__(self, n, point_dim, channel_in, channel_out):
        super(PCFE, self).__init__()

        self.downsample_layer = BottleneckPointConv(nsample=16, point_dim=point_dim,
                                                    in_channel=channel_in, out_channel=channel_out,
                                                    bottleneck=4, residual=False)

        self.pointconv_resblocks = nn.ModuleList()
        for _ in range(3):
            self.pointconv_resblocks.append(BottleneckPointConv(nsample=16, point_dim=point_dim,
                                                                in_channel=channel_out, out_channel=channel_out,
                                                                bottleneck=4, residual=True))


    def forward(self, xyz, features, valid_xyz, downsampled_xyz, downsampled_valid_xyz, nn_idx, downsampled_nn_idx):
        '''
            Args:
                xyz: input points' 3D location (+ normal) [B, C, N]
                features: input points' features [B, D, N]
        '''
        downsampled_features = self.downsample_layer(xyz, features, valid_xyz,
                                                     downsampled_xyz, downsampled_valid_xyz, nn_idx)

        for pointconv_resblock in self.pointconv_resblocks:
            downsampled_features = pointconv_resblock(downsampled_xyz, downsampled_features,
                                                      downsampled_valid_xyz, downsampled_xyz,
                                                      downsampled_valid_xyz, downsampled_nn_idx)

        return downsampled_features


class Encoder(nn.Module):
    '''
        Encode partial 3D point cloud into a n-D latent space.
    '''

    def __init__(self, n, point_dim, latent_dim, base_feature_dim=16,
                 num_layers=5, feature_dims=[32, 48, 64, 96, 128], downsample_ratio=2):
        super(Encoder, self).__init__()

        # point wise encoding
        self.pw_conv1 = nn.Conv1d(point_dim, base_feature_dim, 1, bias=False)
        self.pw_bn1 = nn.BatchNorm1d(base_feature_dim)
        self.pw_conv2 = nn.Conv1d(base_feature_dim, base_feature_dim, 1, bias=False)
        self.pw_bn2 = nn.BatchNorm1d(base_feature_dim)
        self.pw_conv3 = nn.Conv1d(base_feature_dim, base_feature_dim, 1, bias=False)
        self.pw_bn3 = nn.BatchNorm1d(base_feature_dim)

        # local feature extraction
        self.fe_layers = nn.ModuleList()
        num_points = n
        channel_in = base_feature_dim
        for channel_out in feature_dims:
            num_points = num_points // downsample_ratio
            self.fe_layers.append(PCFE(num_points, point_dim, channel_in, channel_out))
            channel_in = channel_out

        # global feature extractor
        self.global_conv1 = nn.Conv1d(channel_out, 256, 1, bias=False)
        self.global_bn1 = nn.BatchNorm1d(256)
        self.global_conv2 = nn.Conv1d(256, 512, 1)

        # interpolate local features for partial input
        self.interpolate = BottleneckPointConv(nsample=16, point_dim=point_dim,
                                               in_channel=channel_out, out_channel=channel_out,
                                               bottleneck=4, residual=False)

    def forward(self, xyz_sets, nn_ids, valid_partials):
        '''
            Args:
                xyz_sets: list of input points' 3D location [[B, C, N1], ..., [B, C, N5]]
            Returns:
                positions: output points' n-D location [B, C', N]
                features: output points' features [B, D', N]
        '''
        # point wise encoding
        B, _, N = xyz_sets[0].shape
        features = self.pw_conv1(xyz_sets[0])
        features = F.leaky_relu(self.pw_bn1(features) * valid_partials[0].view(B, 1, N), negative_slope=0.2)
        features = self.pw_conv2(features)
        features = F.leaky_relu(self.pw_bn2(features) * valid_partials[0].view(B, 1, N), negative_slope=0.2)
        features = self.pw_conv3(features)
        features = F.leaky_relu(self.pw_bn3(features) * valid_partials[0].view(B, 1, N), negative_slope=0.2)

        # extract local shape information from partial input
        for idx, fe_layer in enumerate(self.fe_layers):
            features = fe_layer(xyz_sets[idx], features, valid_partials[idx],
                                xyz_sets[idx+1], valid_partials[idx+1], nn_ids[2*idx+1], nn_ids[2*idx+2])

        # extract global feature vector
        B, _, S = xyz_sets[-1].shape
        global_feature = self.global_conv1(features)
        global_feature = F.leaky_relu(self.global_bn1(global_feature) * valid_partials[-1].view(B, 1, S), negative_slope=0.2)
        global_feature = self.global_conv2(global_feature) * valid_partials[-1].view(B, 1, S)
        global_feature = torch.max(global_feature, dim=-1, keepdim=True)[0]

        # interpolate partial input's local features
        xyz = xyz_sets[0]
        valid_xyz = valid_partials[0]
        features = self.interpolate(xyz_sets[-1], features, valid_partials[-1],
                                    xyz, valid_xyz, nn_ids[-1])

        return features, global_feature
