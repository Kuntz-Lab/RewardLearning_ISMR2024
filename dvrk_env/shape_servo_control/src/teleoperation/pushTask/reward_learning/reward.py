import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

class RewardNetPointCloud(nn.Module):
    '''
    only 256-dimensional object embedding as state
    '''
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 128)

        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)


    def cum_return(self, emb_traj):
        '''calculate cumulative return of trajectory'''
        sum_rewards = 0
        sum_abs_rewards = 0
        
        x = F.leaky_relu(self.fc1(emb_traj))
        x = F.leaky_relu(self.fc2(x))

        x = F.leaky_relu(self.fc3(x))
        r = F.leaky_relu(self.fc4(x))
        
        sum_rewards += torch.sum(r, dim=1)
        sum_abs_rewards += torch.sum(torch.abs(r), dim=1)
        
        return sum_rewards, sum_abs_rewards

    def forward(self, traj_i, traj_j):
        '''compute cumulative return for each trajectory and return logits'''
        cum_r_i, abs_r_i = self.cum_return(traj_i)
        cum_r_j, abs_r_j = self.cum_return(traj_j)
        # [r_i, r_j], abs_r_i + abs_r_j
        return torch.cat((cum_r_i, cum_r_j),dim=1), abs_r_i + abs_r_j

    def single_return(self, emb_traj):

        x = F.leaky_relu(self.fc1(emb_traj))
        x = F.leaky_relu(self.fc2(x))

        x = F.leaky_relu(self.fc3(x))
        r = F.leaky_relu(self.fc4(x))
        
        return r


if __name__ == '__main__':
    device = torch.device("cuda") # cuda
    # traj_1 = torch.randn((4,21,3)).float().to(device)
    # traj_2 = torch.randn((4,21,3)).float().to(device)
    obj_emb_1 = torch.randn((4, 21,256)).float().to(device)
    obj_emb_2 = torch.randn((4, 21,256)).float().to(device)

    model = RewardNetPointCloud().to(device)
    out = model(obj_emb_1,obj_emb_2)
    print("reward pair: ", out[0].shape)
    print("abs reward: ", out[1].shape)