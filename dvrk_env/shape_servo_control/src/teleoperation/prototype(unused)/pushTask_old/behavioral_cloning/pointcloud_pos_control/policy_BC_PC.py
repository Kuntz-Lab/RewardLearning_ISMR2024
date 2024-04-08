import torch.nn as nn
import torch
import torch.nn.functional as F
import sys
import numpy as np
from torch.distributions import MultivariateNormal



class BCPolicyPC(nn.Module):
    '''
    Learn the action by behavioral cloning using L2 loss
    '''
    def __init__(self, act_dim):
        super().__init__()

        self.obj_fc1 = nn.Linear(256, 128)
        self.obj_fc2 = nn.Linear(128, 128)

        self.eef_fc1 = nn.Linear(3, 128)
        self.eef_fc2 = nn.Linear(128, 128)


        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, act_dim)

    def forward(self, obs):
        eef_pose = obs[:, 0:3]
        obj_emb = obs[:, 3:]
        obj = F.leaky_relu(self.obj_fc1(obj_emb))
        obj = F.leaky_relu(self.obj_fc2(obj))

        eef = F.leaky_relu(self.eef_fc1(eef_pose))
        eef = F.leaky_relu(self.eef_fc2(eef))   

        x = torch.cat((eef, obj),dim=-1)

        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        pred_action = self.fc3(x)

        return pred_action


class ActorPC(nn.Module):
    '''
    Learn the optimal mean and log_std that maximize teh log probability of expert action given state of a multivariate normal distribution from which actions are sampled.
    '''
    def __init__(self, act_dim, initial_std):
        super().__init__()

        self.obj_fc1 = nn.Linear(256, 128)
        self.obj_fc2 = nn.Linear(128, 128)

        self.eef_fc1 = nn.Linear(3, 128)
        self.eef_fc2 = nn.Linear(128, 128)


        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, act_dim)

        self.log_std = nn.Parameter(np.log(initial_std) * torch.ones(act_dim))

    def forward(self, obs):
        eef_pose = obs[:, 0:3]
        obj_emb = obs[:, 3:]
        obj = F.leaky_relu(self.obj_fc1(obj_emb))
        obj = F.leaky_relu(self.obj_fc2(obj))

        eef = F.leaky_relu(self.eef_fc1(eef_pose))
        eef = F.leaky_relu(self.eef_fc2(eef))   

        x = torch.cat((eef, obj),dim=-1)

        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        action_mean = self.fc3(x)

        return action_mean

    def act(self, obs):
        action_mean = self.forward(obs)
        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(action_mean, scale_tril=covariance)
        action = distribution.sample()
        return action


if __name__ == "__main__":
    device = torch.device("cuda") # cuda
    eef_states = torch.randn((4,3)).float().to(device)
    obj_states = torch.randn((4,256)).float().to(device)
    states = torch.cat((eef_states, obj_states), dim=-1)
    print("states shape: ", states.shape)
    model = ActorPC(act_dim=3, initial_std=1.0).to(device)
    out = model(states)
    print("action mean: ", out.shape)
    print("sample shape", model.act(states).shape)

