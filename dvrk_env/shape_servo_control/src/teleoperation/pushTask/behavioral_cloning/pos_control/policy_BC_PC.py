import torch.nn as nn
import torch
import torch.nn.functional as F
import sys
import numpy as np
from torch.distributions import MultivariateNormal

'''
Policy learned by behavioral cloning:
input: robot state, embedding of the environment
output: action 
'''

class BCPolicyPC(nn.Module):
    '''
    Learn the action by behavioral cloning using L2 loss
    '''
    def __init__(self, act_dim, robot_state_dim, emb_dim):
        super().__init__()
        self.robot_state_dim = robot_state_dim
        self.emb_dim = emb_dim
        self.act_dim = act_dim

        self.obj_fc1 = nn.Linear(emb_dim, 128)
        self.obj_fc2 = nn.Linear(128, 128)

        self.robot_fc1 = nn.Linear(robot_state_dim, 128)
        self.robot_fc2 = nn.Linear(128, 128)


        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, act_dim)

    def forward(self, obs):
        robot_state = obs[:, 0:self.robot_state_dim]
        obj_emb = obs[:, self.robot_state_dim:]
        obj = F.leaky_relu(self.obj_fc1(obj_emb))
        obj = F.leaky_relu(self.obj_fc2(obj))

        robot = F.leaky_relu(self.robot_fc1(robot_state))
        robot = F.leaky_relu(self.robot_fc2(robot))   

        x = torch.cat((robot, obj),dim=-1)

        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        pred_action = self.fc3(x)

        return pred_action


class ActorPC(nn.Module):
    '''
    Learn the optimal mean and log_std that maximize the log probability of expert action given state using a multivariate normal distribution from which actions are sampled.
    '''
    def __init__(self, act_dim, robot_state_dim, emb_dim, initial_std):
        super().__init__()
        self.robot_state_dim = robot_state_dim
        self.emb_dim = emb_dim
        self.act_dim = act_dim

        self.obj_fc1 = nn.Linear(emb_dim, 128)
        self.obj_fc2 = nn.Linear(128, 128)

        self.robot_fc1 = nn.Linear(robot_state_dim, 128)
        self.robot_fc2 = nn.Linear(128, 128)


        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, act_dim)

        self.log_std = nn.Parameter(np.log(initial_std) * torch.ones(act_dim))

    def forward(self, obs):
        robot_state = obs[:, 0:self.robot_state_dim]
        obj_emb = obs[:, self.robot_state_dim:]
        obj = F.leaky_relu(self.obj_fc1(obj_emb))
        obj = F.leaky_relu(self.obj_fc2(obj))

        robot = F.leaky_relu(self.robot_fc1(robot_state))
        robot = F.leaky_relu(self.robot_fc2(robot))   

        x = torch.cat((robot, obj),dim=-1)

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
    eef_states = torch.randn((4,2)).float().to(device)
    obj_states = torch.randn((4,256)).float().to(device)
    states = torch.cat((eef_states, obj_states), dim=-1)
    print("states shape: ", states.shape)
    model = ActorPC(act_dim=2, robot_state_dim=2, emb_dim=256, initial_std=1.0).to(device)
    out = model(states)
    print("action mean: ", out.shape)
    print("sample shape", model.act(states).shape)

