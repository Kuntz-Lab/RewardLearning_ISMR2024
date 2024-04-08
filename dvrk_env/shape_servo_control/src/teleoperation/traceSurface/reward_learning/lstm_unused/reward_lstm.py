import torch
import torch.nn as nn
import torch.nn.functional as F
#import sys
#sys.path.append("../")
import numpy as np

'''
LSTM that outputs the reward of a trajectory
'''


class TrajEmbedding(nn.Module):
    def __init__(self, obj_emb_dim=256, eef_dim=3, out_emb_dim=128):
        super().__init__()

        self.obj_fc1 = nn.Linear(obj_emb_dim, out_emb_dim) # 256, 128
        self.obj_fc2 = nn.Linear(out_emb_dim, out_emb_dim//2) # 128, 64

        self.eef_fc1 = nn.Linear(eef_dim, out_emb_dim) # 3, 128
        self.eef_fc2 = nn.Linear(out_emb_dim, out_emb_dim//2) # 128, 64

    def forward(self, eef_traj, obj_emb):
        obj = F.leaky_relu(self.obj_fc1(obj_emb))
        obj = F.leaky_relu(self.obj_fc2(obj))

        eef = F.leaky_relu(self.eef_fc1(eef_traj))
        eef = F.leaky_relu(self.eef_fc2(eef))   

        traj_emb = torch.cat((eef, obj),dim=-1) # 128 dim

        return traj_emb

class HiddenToReward(nn.Module):
    def __init__(self, hidden_dim=128, output_dim=1):
        super().__init__()

        self.fc1 = nn.Linear(hidden_dim, hidden_dim//2)
        self.fc2 = nn.Linear(hidden_dim//2, output_dim)

    def forward(self, hidden_state):
       x = F.leaky_relu(self.fc1(hidden_state))
       r = self.fc2(x) # 1 dim
       return r



class RewardLSTM(nn.Module):
    def __init__(self, input_dim=(256, 3), embedding_dim=128,  hidden_dim=128, output_dim=1,  n_layers=1, drop_prob=0):
        super(RewardLSTM, self).__init__()

        obj_emb_dim, eef_dim = input_dim
        self.embedding = TrajEmbedding(obj_emb_dim=obj_emb_dim, eef_dim=eef_dim, out_emb_dim=embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
        #self.dropout = nn.Dropout(drop_prob)
        self.hidden_2_reward = HiddenToReward(hidden_dim, output_dim)
        
    def forward(self, eef_traj, obj_emb):
        embeds = self.embedding(eef_traj, obj_emb) #(N,,L,emb)
        lstm_out, hiddens = self.lstm(embeds) #(N,L,D)
        
        #out = self.dropout(lstm_out)
        out = self.hidden_2_reward(lstm_out) #(N,L,1)
        reward = torch.sum(out, dim=1)

        return reward, hiddens

    def single_return(self, eef_traj, obj_emb):
        embeds = self.embedding(eef_traj, obj_emb) #(N,,L,emb)
        lstm_out, hiddens = self.lstm(embeds) #(N,L,D)
        
        #out = self.dropout(lstm_out)
        out = self.hidden_2_reward(lstm_out) #(N,L,1)

        return out, hiddens


if __name__ == '__main__':


    device = torch.device("cuda") # cuda
    eef_traj= torch.randn((4,21,3)).float().to(device)
    obj_emb = torch.randn((4,21,256)).float().to(device)
    model = RewardLSTM(input_dim=(256, 3), embedding_dim=128,  hidden_dim=128, output_dim=1,  n_layers=1, drop_prob=0).to(device)
    reward, hiddens = model(eef_traj, obj_emb)
    print("reward: ", reward.shape)
    print("hidden: ", hiddens[0].shape)
    print("cell: ", hiddens[1].shape)