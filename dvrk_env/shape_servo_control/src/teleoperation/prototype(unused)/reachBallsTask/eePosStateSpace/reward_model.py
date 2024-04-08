import torch.nn as nn
import torch
import torch.nn.functional as F
import sys
sys.path.append("../")
import numpy as np

class RewardNet(nn.Module):
    def __init__(self):
        super().__init__()

        # self.conv1 = nn.Conv2d(4, 16, 7, stride=3)
        # self.conv2 = nn.Conv2d(16, 16, 5, stride=2)
        # self.conv3 = nn.Conv2d(16, 16, 3, stride=1)
        # self.conv4 = nn.Conv2d(16, 16, 3, stride=1)
        self.fc1 = nn.Linear(3, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 1)



    def cum_return(self, traj):
        '''calculate cumulative return of trajectory'''
        sum_rewards = 0
        sum_abs_rewards = 0
        
        x = F.leaky_relu(self.fc1(traj))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = F.leaky_relu(self.fc4(x))
        r = self.fc5(x)
        sum_rewards += torch.sum(r)
        sum_abs_rewards += torch.sum(torch.abs(r))
        
        return sum_rewards, sum_abs_rewards



    def forward(self, traj_i, traj_j):
        '''compute cumulative return for each trajectory and return logits'''
        cum_r_i, abs_r_i = self.cum_return(traj_i)
        cum_r_j, abs_r_j = self.cum_return(traj_j)
        return torch.cat((cum_r_i.unsqueeze(0), cum_r_j.unsqueeze(0)),0), abs_r_i + abs_r_j


class RewardNet2(nn.Module):
    def __init__(self):
        super().__init__()

        # self.conv1 = nn.Conv2d(4, 16, 7, stride=3)
        # self.conv2 = nn.Conv2d(16, 16, 5, stride=2)
        # self.conv3 = nn.Conv2d(16, 16, 3, stride=1)
        # self.conv4 = nn.Conv2d(16, 16, 3, stride=1)
        # self.fc1 = nn.Linear(9, 256)
        # self.fc2 = nn.Linear(256, 256)
        # self.fc3 = nn.Linear(256, 128)
        # self.fc4 = nn.Linear(128, 64)
        # self.fc5 = nn.Linear(64, 1)

        
        self.fc1 = nn.Linear(6, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 1)


    def cum_return(self, traj):
        '''calculate cumulative return of trajectory'''
        sum_rewards = 0
        sum_abs_rewards = 0
        
        x = F.leaky_relu(self.fc1(traj))
        # x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = F.leaky_relu(self.fc4(x))
        r = self.fc5(x)
        sum_rewards += torch.sum(r)
        sum_abs_rewards += torch.sum(torch.abs(r))
        
        return sum_rewards, sum_abs_rewards



    def forward(self, traj_i, traj_j):
        '''compute cumulative return for each trajectory and return logits'''
        cum_r_i, abs_r_i = self.cum_return(traj_i)
        cum_r_j, abs_r_j = self.cum_return(traj_j)
        return torch.cat((cum_r_i.unsqueeze(0), cum_r_j.unsqueeze(0)),0), abs_r_i + abs_r_j


class RewardNetPointCloud(nn.Module):
    def __init__(self):
        super().__init__()


        self.obj_fc1 = nn.Linear(256, 128)
        self.obj_fc2 = nn.Linear(128, 128)

        self.eef_fc1 = nn.Linear(3, 128)
        self.eef_fc2 = nn.Linear(128, 128)


        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

        



    def cum_return(self, eef_pose, obj_emb):
        '''calculate cumulative return of trajectory'''
        sum_rewards = 0
        sum_abs_rewards = 0
        # print("+++++:", eef_pose.shape, obj_emb.shape)
        obj = F.leaky_relu(self.obj_fc1(obj_emb))
        obj = F.leaky_relu(self.obj_fc2(obj))

        eef = F.leaky_relu(self.eef_fc1(eef_pose))
        eef = F.leaky_relu(self.eef_fc2(eef))   

        x = torch.cat((eef, obj),dim=-1)
        # print("+++++:", x.shape)

        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        
        r = self.fc3(x)

        sum_rewards += torch.sum(r)
        sum_abs_rewards += torch.sum(torch.abs(r))
        
        return sum_rewards, sum_abs_rewards



    def forward(self, traj_i, traj_j, obj_emb_i, obj_emb_j):
        '''compute cumulative return for each trajectory and return logits'''
        cum_r_i, abs_r_i = self.cum_return(traj_i, obj_emb_i)
        cum_r_j, abs_r_j = self.cum_return(traj_j, obj_emb_j)
        return torch.cat((cum_r_i.unsqueeze(0), cum_r_j.unsqueeze(0)),0), abs_r_i + abs_r_j

if __name__ == '__main__':


    device = torch.device("cuda") # cuda
#     traj_1 = torch.randn((21,3)).float().to(device)
#     traj_2 = torch.randn((21,3)).float().to(device)
#     model = RewardNet().to(device)
#     out = model(traj_1, traj_2)
#     print(out[0].shape)
#     print(out[1])
#     # print(out)
#     # print(out.type())


#     traj_1 = torch.randn((21,3)).float().to(device)
#     traj_2 = torch.randn((21,3)).float().to(device)
#     model = RewardNet().to(device)
#     out = model(traj_1, traj_2)
#     print(out[0].shape)
#     print(out[1])
#     # print(out)
#     # print(out.type())