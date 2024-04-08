import torch.nn as nn
import torch
import torch.nn.functional as F
# from pointnet2_utils_groupnorm import PointNetSetAbstraction,PointNetFeaturePropagation
from pointconv_util_groupnorm_2 import PointConvDensitySetAbstraction,PointConvFeaturePropagation
from pointnet2_utils import PointNetSetAbstraction,PointNetFeaturePropagation

class AutoEncoder(nn.Module):
    def __init__(self, normal_channel=False):
        super(AutoEncoder, self).__init__()
        self.chamfer_loss = ChamferLoss()
        self.huber_loss = HuberLoss(reduction="sum")

        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = normal_channel
        self.num_pts = 256#*3
        
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

        # ### Architecture 1
        # self.sa1 = PointNetSetAbstraction(npoint=128, radius=0.2, nsample=8, in_channel=6+additional_channel, mlp=[32], group_all=False)
        # self.sa2 = PointNetSetAbstraction(npoint=64, radius=0.4, nsample=16, in_channel=32 + 3, mlp=[64], group_all=False)
        # self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=64 + 3, mlp=[128], group_all=True)

        # self.fc1 = nn.Linear(128, 256)
        # self.bn1 = nn.GroupNorm(1, 256)

        # self.fc2 = nn.Linear(256, 512)
        # self.bn2 = nn.GroupNorm(1, 512)

        # self.fc3 = nn.Linear(512, self.num_pts*3)
        # self.bn3 = nn.GroupNorm(1, self.num_pts*3)


        ### Architecture 2
        self.sa1 = PointNetSetAbstraction(npoint=256, radius=0.2, nsample=32, in_channel=6+additional_channel, mlp=[64, 64], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=64 + 3, mlp=[128, 128], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=128 + 3, mlp=[256, 256], group_all=True)

        self.fc1 = nn.Linear(256, 512)
        self.bn1 = nn.GroupNorm(1, 512)

        self.fc2 = nn.Linear(512, 512)
        self.bn2 = nn.GroupNorm(1, 512)

        self.fc3 = nn.Linear(512, self.num_pts*3)
        self.bn3 = nn.GroupNorm(1, self.num_pts*3)






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

        # x = l3_points.view(B, 128)
        x = l3_points.view(B, 256)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        x = x.view(B, 3, self.num_pts)

        return x



    def get_chamfer_loss(self, input, output):
        # input shape  (batch_size, num_pts, 3)
        # output shape (batch_size, num_pts, 3)
        return self.chamfer_loss(input, output)

    def get_huber_loss(self, input, output):
        # shape  (batch_size, 3, num_pts)

        return self.huber_loss(input, output)


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

class HuberLoss(nn.Module):
    def __init__(self, reduction, delta=1.0):
        super(HuberLoss, self).__init__()
        self.reduction = reduction
        self.delta = delta

    def forward(self, y_pred, y_gt):
        B, N, C = y_pred.shape

        abs_error = torch.abs(y_pred - y_gt)

        error = torch.where(abs_error < self.delta,
                            0.5*(y_pred - y_gt)**2,
                            self.delta*(abs_error - 0.5*self.delta))

        if self.reduction == 'mean':
            loss = torch.mean(error)
        elif self.reduction == 'sum':
            loss = torch.sum(error)

        return loss

if __name__ == '__main__':

    device = torch.device("cuda") # "cpu"
    pc = torch.randn((8,3,1024)).float().to(device)
    model = AutoEncoder().to(device)
    out = model(pc)
 
    print(out.shape)
