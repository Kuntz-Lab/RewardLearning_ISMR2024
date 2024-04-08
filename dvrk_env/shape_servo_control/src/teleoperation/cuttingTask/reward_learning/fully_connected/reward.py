import torch
import torch.nn as nn
import torch.nn.functional as F
#import sys
#sys.path.append("../")
import numpy as np

class RewardNetPointCloudEEFOld(nn.Module):
    '''
    256-dimensional object embedding and 3-dimensional end-effector pose as state
    '''
    def __init__(self):
        super().__init__()
        # self.obj_dropout1 = nn.Dropout(p=0.5)
        # self.obj_dropout2 = nn.Dropout(p=0.5)
        # self.eef_dropout1 = nn.Dropout(p=0.5)
        # self.eef_dropout2 = nn.Dropout(p=0.5)

        self.obj_fc1 = nn.Linear(256, 128)
        self.obj_fc2 = nn.Linear(128, 128)

        self.eef_fc1 = nn.Linear(3, 128)
        self.eef_fc2 = nn.Linear(128, 128)

        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)
        self.dropout3 = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)


    def cum_return(self, eef_traj, obj_emb):
        '''calculate cumulative return of trajectory'''
        sum_rewards = 0
        sum_abs_rewards = 0
        # print("+++++:", eef_pose.shape, obj_emb.shape)
        obj = F.leaky_relu(self.obj_fc1(obj_emb))
        #obj = self.obj_dropout1(obj)
        obj = F.leaky_relu(self.obj_fc2(obj))
        #obj = self.obj_dropout2(obj)

        eef = F.leaky_relu(self.eef_fc1(eef_traj))
        #eef = self.eef_dropout1(eef)
        eef = F.leaky_relu(self.eef_fc2(eef))   
        #eef = self.eef_dropout2(eef)

        x = torch.cat((eef, obj),dim=-1)
        x = self.dropout1(x)

        x = F.leaky_relu(self.fc1(x))
        x = self.dropout2(x)
        
        x = F.leaky_relu(self.fc2(x))
        x = self.dropout3(x)
        
        r = self.fc3(x)
        

        #print(torch.sum(r, dim=1).shape)

        sum_rewards += torch.sum(r, dim=1)
        sum_abs_rewards += torch.sum(torch.abs(r), dim=1)
        
        return sum_rewards, sum_abs_rewards

    def forward(self, ee_traj_i, obj_emb_i, ee_traj_j, obj_emb_j):
        '''compute cumulative return for each trajectory and return logits'''
        cum_r_i, abs_r_i = self.cum_return(ee_traj_i, obj_emb_i)
        cum_r_j, abs_r_j = self.cum_return(ee_traj_j, obj_emb_j)
        # [r_i, r_j], abs_r_i + abs_r_j
        #return torch.cat((cum_r_i.unsqueeze(0), cum_r_j.unsqueeze(0)),0), abs_r_i + abs_r_j
        return torch.cat((cum_r_i, cum_r_j),dim=1), abs_r_i + abs_r_j

    def single_return(self, eef_traj, obj_emb):

       obj = F.leaky_relu(self.obj_fc1(obj_emb))
       #obj = self.obj_dropout1(obj)
       obj = F.leaky_relu(self.obj_fc2(obj))
       #obj = self.obj_dropout2(obj)

       eef = F.leaky_relu(self.eef_fc1(eef_traj))
       #eef = self.eef_dropout1(eef)
       eef = F.leaky_relu(self.eef_fc2(eef))  
       #eef = self.eef_dropout2(eef) 

       x = torch.cat((eef, obj),dim=-1)
       x = self.dropout1(x)

       x = F.leaky_relu(self.fc1(x))
       x = self.dropout2(x)
       
       x = F.leaky_relu(self.fc2(x))
       x = self.dropout3(x)

       r = self.fc3(x)
        
       return r

class RewardNetPointCloudEEF(nn.Module):
    '''
    256-dimensional object embedding and 3-dimensional end-effector pose as state
    '''
    def __init__(self):
        super().__init__()
        self.obj_dropout1 = nn.Dropout(p=0.3)
        self.obj_dropout2 = nn.Dropout(p=0.3)
        self.eef_dropout1 = nn.Dropout(p=0.3)
        self.eef_dropout2 = nn.Dropout(p=0.3)

        self.obj_fc1 = nn.Linear(256, 128) # 256, 128
        self.obj_fc2 = nn.Linear(128, 128) # 128, 128
        #self.obj_fc3 = nn.Linear(128, 128)


        self.eef_fc1 = nn.Linear(3, 128)
        self.eef_fc2 = nn.Linear(128, 128)

        # 0.5
        self.dropout1 = nn.Dropout(p=0)
        self.dropout2 = nn.Dropout(p=0)
        self.dropout3 = nn.Dropout(p=0)

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)


    def cum_return(self, eef_traj, obj_emb):
        '''calculate cumulative return of trajectory'''
        sum_rewards = 0
        sum_abs_rewards = 0
        # print("+++++:", eef_pose.shape, obj_emb.shape)
        obj = F.leaky_relu(self.obj_fc1(obj_emb))
        obj = self.obj_dropout1(obj)
        obj = F.leaky_relu(self.obj_fc2(obj))
        obj = self.obj_dropout2(obj)
        # obj = F.leaky_relu(self.obj_fc3(obj))

        eef = F.leaky_relu(self.eef_fc1(eef_traj))
        eef = self.eef_dropout1(eef)
        eef = F.leaky_relu(self.eef_fc2(eef))   
        eef = self.eef_dropout2(eef)

        x = torch.cat((eef, obj),dim=-1)
        x = self.dropout1(x)

        x = F.leaky_relu(self.fc1(x))
        x = self.dropout2(x)
        
        x = F.leaky_relu(self.fc2(x))
        x = self.dropout3(x)
        
        r = self.fc3(x)
        

        #print(torch.sum(r, dim=1).shape)

        sum_rewards += torch.sum(r, dim=1)
        sum_abs_rewards += torch.sum(torch.abs(r), dim=1)
        
        return sum_rewards, sum_abs_rewards

    def forward(self, ee_traj_i, obj_emb_i, ee_traj_j, obj_emb_j):
        '''compute cumulative return for each trajectory and return logits'''
        cum_r_i, abs_r_i = self.cum_return(ee_traj_i, obj_emb_i)
        cum_r_j, abs_r_j = self.cum_return(ee_traj_j, obj_emb_j)
        # [r_i, r_j], abs_r_i + abs_r_j
        #return torch.cat((cum_r_i.unsqueeze(0), cum_r_j.unsqueeze(0)),0), abs_r_i + abs_r_j
        return torch.cat((cum_r_i, cum_r_j),dim=1), abs_r_i + abs_r_j

    def single_return(self, eef_traj, obj_emb):

       obj = F.leaky_relu(self.obj_fc1(obj_emb))
       #obj = self.obj_dropout1(obj)
       obj = F.leaky_relu(self.obj_fc2(obj))
       #obj = self.obj_dropout2(obj)

       eef = F.leaky_relu(self.eef_fc1(eef_traj))
       #eef = self.eef_dropout1(eef)
       eef = F.leaky_relu(self.eef_fc2(eef))  
       #eef = self.eef_dropout2(eef) 

       x = torch.cat((eef, obj),dim=-1)
       x = self.dropout1(x)

       x = F.leaky_relu(self.fc1(x))
       x = self.dropout2(x)
       
       x = F.leaky_relu(self.fc2(x))
       x = self.dropout3(x)

       r = self.fc3(x)
        
       return r





if __name__ == '__main__':


    device = torch.device("cuda") # cuda
    traj_1 = torch.randn((4,21,3)).float().to(device)
    traj_2 = torch.randn((4,21,3)).float().to(device)
    obj_emb_1 = torch.randn((4, 21,256)).float().to(device)
    obj_emb_2 = torch.randn((4, 21,256)).float().to(device)
    model = RewardNetPointCloudEEF().to(device)
    out = model(traj_1, obj_emb_1, traj_2, obj_emb_2)
    print("reward pair: ", out[0].shape)
    print("abs reward: ", out[1].shape)


    # traj_1 = torch.randn((21,256)).float().to(device)
    # traj_2 = torch.randn((21,256)).float().to(device)
    # model = RewardNetPointCloud().to(device)
    # out = model(traj_1, traj_2)
    # print(len(out))
    # print("rewardPair: ", out[0], '\n')
    # print("absReward: ", out[1], '\n')