import torch.nn as nn
import torch
import torch.nn.functional as F
import sys
sys.path.append("../")
import numpy as np

class RewardNet1(nn.Module):
    '''
    Conditioned on eef pose and single goal pose
    '''
    def __init__(self):
        super().__init__()
        
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

    def single_return(self, traj):
        x = F.leaky_relu(self.fc1(traj))
        # x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = F.leaky_relu(self.fc4(x))
        r = self.fc5(x)
        return r

    def forward(self, traj_i, traj_j):
        '''compute cumulative return for each trajectory and return logits'''
        cum_r_i, abs_r_i = self.cum_return(traj_i)
        cum_r_j, abs_r_j = self.cum_return(traj_j)
        return torch.cat((cum_r_i.unsqueeze(0), cum_r_j.unsqueeze(0)),0), abs_r_i + abs_r_j



class RewardNet2(nn.Module):
    '''
    Conditioned on eef pose, mid goal pose, final goal pose and goal indicator
    '''
    def __init__(self):
        super().__init__()

        self.goal_fc1 = nn.Linear(1,128)
        self.goal_fc2 = nn.Linear(128,128)

        self.pos_fc1 = nn.Linear(9, 128)
        self.pos_fc2 = nn.Linear(128,128)
        
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)


    def cum_return(self, traj):
        '''calculate cumulative return of trajectory'''
        #traj shape = (traj_length, 10)
        sum_rewards = 0
        sum_abs_rewards = 0

        g = traj[:, 0:1]
        g = F.leaky_relu(self.goal_fc1(g))
        g = F.leaky_relu(self.goal_fc2(g))

        pos = traj[:, 1:10]
        pos = F.leaky_relu(self.pos_fc1(pos))
        pos = F.leaky_relu(self.pos_fc2(pos))

        x = torch.cat((g, pos), dim=1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        r = F.leaky_relu(self.fc3(x))
        
        sum_rewards += torch.sum(r)
        sum_abs_rewards += torch.sum(torch.abs(r))
        
        return sum_rewards, sum_abs_rewards

    def single_return(self, traj):
        g = traj[:, 0:1]
        g = F.leaky_relu(self.goal_fc1(g))
        g = F.leaky_relu(self.goal_fc2(g))

        pos = traj[:, 1:10]
        pos = F.leaky_relu(self.pos_fc1(pos))
        pos = F.leaky_relu(self.pos_fc2(pos))

        x = torch.cat((g, pos), dim=1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        r = F.leaky_relu(self.fc3(x))
        
        return r

    def forward(self, traj_i, traj_j):
        '''compute cumulative return for each trajectory and return logits'''
        cum_r_i, abs_r_i = self.cum_return(traj_i)
        cum_r_j, abs_r_j = self.cum_return(traj_j)
        # [r_i, r_j], abs_r_i + abs_r_j
        return torch.cat((cum_r_i.unsqueeze(0), cum_r_j.unsqueeze(0)),0), abs_r_i + abs_r_j


class RewardNetPointCloud(nn.Module):
    def __init__(self):
        super().__init__()
        #print("########################################## HERE DUMBASS ##################################")

        self.obj_fc1 = nn.Linear(256, 128)
        self.obj_fc2 = nn.Linear(128, 128)

        self.state_fc1 = nn.Linear(6, 128)
        self.state_fc2 = nn.Linear(128, 128)


        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)


    def cum_return(self, state_traj, obj_emb):
        '''calculate cumulative return of trajectory'''
        sum_rewards = 0
        sum_abs_rewards = 0
        # print("+++++:", eef_pose.shape, obj_emb.shape)
        obj = F.leaky_relu(self.obj_fc1(obj_emb))
        obj = F.leaky_relu(self.obj_fc2(obj))

        state = F.leaky_relu(self.state_fc1(state_traj))
        state = F.leaky_relu(self.state_fc2(state))   

        x = torch.cat((state, obj),dim=-1)

        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        
        r = self.fc3(x)

        sum_rewards += torch.sum(r)
        sum_abs_rewards += torch.sum(torch.abs(r))
        
        return sum_rewards, sum_abs_rewards

    def single_return(self, state_traj, obj_emb):

        
        obj = F.leaky_relu(self.obj_fc1(obj_emb))
        obj = F.leaky_relu(self.obj_fc2(obj))

        state = F.leaky_relu(self.eef_fc1(state_traj))
        state = F.leaky_relu(self.eef_fc2(state))   

        x = torch.cat((state, obj),dim=-1)

        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        
        r = self.fc3(x)
        
        return r


if __name__ == '__main__':


    device = torch.device("cuda") # cuda
    # traj_1 = torch.randn((21,6)).float().to(device)
    # traj_2 = torch.randn((21,6)).float().to(device)
    # obj_emb_i = torch.randn((21,256)).float().to(device)
    # obj_emb_j = torch.randn((21,256)).float().to(device)
    # model = RewardNetPointCloud().to(device)
    # out = model(traj_1, traj_2, obj_emb_i, obj_emb_j)
    # print(out[0].shape)
    # print(out[1])
    # print(out)
    # print(out.type())


    traj_1 = torch.randn((21,10)).float().to(device)
    traj_2 = torch.randn((21,10)).float().to(device)
    model = RewardNet2().to(device)
    out = model(traj_1, traj_2)
    print(len(out))
    print("rewardPair: ", out[0], '\n')
    print("absReward: ", out[1], '\n')