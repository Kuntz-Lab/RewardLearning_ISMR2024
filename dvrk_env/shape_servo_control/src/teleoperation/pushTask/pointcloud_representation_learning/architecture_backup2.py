import torch.nn as nn
import torch
import torch.nn.functional as F
# from pointnet2_utils_groupnorm import PointNetSetAbstraction,PointNetFeaturePropagation
from pointconv_util_groupnorm_2 import PointConvDensitySetAbstraction,PointConvFeaturePropagation
from pointnet2_utils import PointNetSetAbstraction,PointNetFeaturePropagation

class AutoEncoder(nn.Module):
    def __init__(self, normal_channel=False):
        super(AutoEncoder, self).__init__()
        self.loss = ChamferLoss()
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = normal_channel

        
        # self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=6+additional_channel, mlp=[64], group_all=False)
        # self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=64 + 3, mlp=[128], group_all=False)
        # self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=128 + 3, mlp=[256], group_all=True)
        
        # self.fp3 = PointNetFeaturePropagation(in_channel=128+256, mlp=[128])
        # self.fp2 = PointNetFeaturePropagation(in_channel=64+128, mlp=[64])
        # self.fp1 = PointNetFeaturePropagation(in_channel=64+additional_channel, mlp=[64, 64, 3+additional_channel])
        # self.conv1 = nn.Conv1d(64, 64, 1)
        # self.bn1 = nn.GroupNorm(1, 64)
        # self.drop1 = nn.Dropout(0.5)
        # self.conv2 = nn.Conv1d(64, 3, 1)


        self.sa1 = PointNetSetAbstraction(npoint=128, radius=0.2, nsample=8, in_channel=6+additional_channel, mlp=[32], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=64, radius=0.4, nsample=16, in_channel=32 + 3, mlp=[64], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=64 + 3, mlp=[128], group_all=True)
        
        # self.fp3 = PointNetFeaturePropagation(in_channel=64+128, mlp=[64])
        # self.fp2 = PointNetFeaturePropagation(in_channel=32+64, mlp=[32])
        # self.fp1 = PointNetFeaturePropagation(in_channel=32+additional_channel, mlp=[32, 32, 3+additional_channel])

        # self.conv1 = nn.Conv1d(32, 32, 1)
        # self.bn1 = nn.GroupNorm(1, 32)
        # self.drop1 = nn.Dropout(0.5)
        # self.conv2 = nn.Conv1d(32, 3, 1)
        
        # self.sa1 = PointNetSetAbstraction(npoint=256, radius=0.2, nsample=32, in_channel=6+additional_channel, mlp=[64], group_all=False)
        # self.sa2 = PointNetSetAbstraction(npoint=32, radius=0.4, nsample=64, in_channel=64 + 3, mlp=[64, 64, 128], group_all=False)
        # self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=128 + 3, mlp=[128, 128, 128], group_all=True)
        # self.fp3 = PointNetFeaturePropagation(in_channel=384, mlp=[128, 128])
        # self.fp2 = PointNetFeaturePropagation(in_channel=192, mlp=[128, 64])
        # # self.fp1 = PointNetFeaturePropagation(in_channel=64+additional_channel, mlp=[64, 32])
        # self.fp1 = PointNetFeaturePropagation(in_channel=64+additional_channel, mlp=[64, 32, 3+additional_channel])

        # self.sa1 = PointConvDensitySetAbstraction(npoint=128, nsample=8, in_channel=6+additional_channel, mlp=[32], bandwidth = 0.1, group_all=False)
        # self.sa2 = PointConvDensitySetAbstraction(npoint=64, nsample=16, in_channel=32 + 3, mlp=[64], bandwidth = 0.2, group_all=False)
        # self.sa3 = PointConvDensitySetAbstraction(npoint=1, nsample=None, in_channel=64 + 3, mlp=[128], bandwidth = 0.4, group_all=True)
        
        # self.fp3 = PointConvFeaturePropagation(in_channel=64+128, mlp=[64], bandwidth = 0.4, linear_shape=64+3)
        # self.fp2 = PointConvFeaturePropagation(in_channel=32+64, mlp=[32], bandwidth = 0.2, linear_shape=32+3)
        # self.fp1 = PointConvFeaturePropagation(in_channel=32+additional_channel, mlp=[32, 32, 3+additional_channel], bandwidth = 0.1, linear_shape=3)

        # self.sa1 = PointConvDensitySetAbstraction(npoint=512, nsample=32, in_channel=6+additional_channel, mlp=[64], bandwidth = 0.1, group_all=False)
        # self.sa2 = PointConvDensitySetAbstraction(npoint=128, nsample=64, in_channel=64 + 3, mlp=[128], bandwidth = 0.2, group_all=False)
        # self.sa3 = PointConvDensitySetAbstraction(npoint=1, nsample=None, in_channel=128 + 3, mlp=[256], bandwidth = 0.4, group_all=True)
        
        # self.fp3 = PointConvFeaturePropagation(in_channel=128+256, mlp=[128], bandwidth = 0.4, linear_shape=128+3)
        # self.fp2 = PointConvFeaturePropagation(in_channel=64+128, mlp=[64], bandwidth = 0.2, linear_shape=64+3)
        # self.fp1 = PointConvFeaturePropagation(in_channel=64+additional_channel, mlp=[64, 64], bandwidth = 0.1, linear_shape=3)

        # self.latent_dim = 256
        self.fc1 = nn.Linear(128, 256)
        self.bn1 = nn.GroupNorm(1, 256)
        # self.bn1 = nn.BatchNorm1d(256)

        self.fc2 = nn.Linear(256, 512)
        self.bn2 = nn.GroupNorm(1, 512)
        # self.bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, 256*3)
        self.bn3 = nn.GroupNorm(1, 256*3)
        # self.bn3 = nn.BatchNorm1d(256*3)

    def forward(self, xyz):
        # Set Abstraction layers
        B,C,N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:,:3,:]
        else:
            l0_points = xyz
            l0_xyz = xyz
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        # l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        # #print(l2_points.shape)
        # l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        # l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)


        # return l0_points

        x = l3_points.view(B, 128)
        x = F.relu(self.bn1(self.fc1(x)))
        # x = self.drop3(F.relu(self.bn3(self.fc3(x))))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        x = x.view(B, 3, 256)

        return x

        # #print(l0_points.shape)
        x = l0_points
        #print(x.shape)



        # #print(l2_points.shape)
        # #print(l1_points.shape)
        # #print(l0_points.shape)
        
        
        # FC layers
        feat =  F.relu(self.bn1(self.conv1(x)))
        x = feat #x = self.drop1(feat)
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        # x = x.permute(0, 2, 1)
        return x
        # return l3_points
        return x, l3_points

    def get_loss(self, input, output):
        # input shape  (batch_size, 2048, 3)
        # output shape (batch_size, 2025, 3)
        return self.loss(input, output)

class ChamferLoss(nn.Module):
    def __init__(self):
        super(ChamferLoss, self).__init__()
        self.use_cuda = torch.cuda.is_available()

    def batch_pairwise_dist(self, x, y):
        bs, num_points_x, points_dim = x.size()
        _, num_points_y, _ = y.size()
        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))
        diag_ind_x = torch.arange(0, num_points_x)
        diag_ind_y = torch.arange(0, num_points_y)
        if x.get_device() != -1:
            diag_ind_x = diag_ind_x.cuda(x.get_device())
            diag_ind_y = diag_ind_y.cuda(x.get_device())
        rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2, 1))
        ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
        P = (rx.transpose(2, 1) + ry - 2 * zz)
        return P

    def forward(self, preds, gts):
        P = self.batch_pairwise_dist(gts, preds)
        mins, _ = torch.min(P, 1)
        loss_1 = torch.sum(mins)
        mins, _ = torch.min(P, 2)
        loss_2 = torch.sum(mins)
        return loss_1 + loss_2

if __name__ == '__main__':

    num_classes = 2
    device = torch.device("cuda") # cuda
    pc = torch.randn((8,3,1024)).float().to(device)
    pc_goal = torch.randn((8,3,1024)).float().to(device)
    # labels = torch.randn((8,16)).float().to(device)
    model = AutoEncoder().to(device)
    # out = model(pc)
    out = model(pc)
    # #print(out[0].shape)   
    # #print(out[1].shape)      
    print(out.shape)
    # #print(out)

    # pc2 = pc[:,torch.randperm(pc.size()[1])]
    # out2 = model(pc, pc_goal)
    # #print(out - out2)