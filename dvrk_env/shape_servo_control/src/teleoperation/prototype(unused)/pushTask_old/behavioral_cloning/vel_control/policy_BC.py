import torch.nn as nn
import torch
import torch.nn.functional as F
import sys
import numpy as np
from torch.distributions import MultivariateNormal



class BCPolicy(nn.Module):
    '''
    Learn the action by behavioral cloning using L2 loss
    '''
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.actor_mlp = nn.Sequential(
            nn.Linear(obs_dim, 32),
            nn.ELU(1.0), 
            nn.Linear(32, 32),
            nn.ELU(1.0), 
            nn.Linear(32,act_dim)
        )

        # self.actor_mlp = nn.Sequential(
        #     nn.Linear(obs_dim, 64),
        #     nn.ELU(1.0), 
        #     nn.Linear(64, 64),
        #     nn.ELU(1.0), 
        #     nn.Linear(64, 32),
        #     nn.ELU(1.0), 
        #     nn.Linear(32, act_dim)
        # )

    def forward(self, x):
        pred_action = self.actor_mlp(x)
        return pred_action


class Actor(nn.Module):
    '''
    Learn the optimal mean and log_std that maximize teh log probability of expert action given state of a multivariate normal distribution from which actions are sampled.
    '''
    def __init__(self, obs_dim, hidden_dims, action_dim, activation_name, initial_std):
        super().__init__()

        activation = self.get_activation(activation_name)

        actor_layers = []
        actor_layers.append(nn.Linear(obs_dim, hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                actor_layers.append(nn.Linear(hidden_dims[l], action_dim))
            else:
                actor_layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                actor_layers.append(activation)

        self.actor_mlp = nn.Sequential(*actor_layers)

        self.log_std = nn.Parameter(np.log(initial_std) * torch.ones(action_dim))

    def forward(self, obs):
        action_mean = self.actor_mlp(obs)
        return action_mean

    def act(self, obs):
        action_mean = self.actor_mlp(obs)
        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(action_mean, scale_tril=covariance)
        action = distribution.sample()
        return action

    def get_activation(self, act_name):
        if act_name == "elu":
            return nn.ELU()
        elif act_name == "selu":
            return nn.SELU()
        elif act_name == "relu":
            return nn.ReLU()
        elif act_name == "crelu":
            return nn.ReLU()
        elif act_name == "lrelu":
            return nn.LeakyReLU()
        elif act_name == "tanh":
            return nn.Tanh()
        elif act_name == "sigmoid":
            return nn.Sigmoid()
        else:
            print("invalid activation function!")
            return None


if __name__ == "__main__":
    device = torch.device("cuda") # cuda
    states = torch.randn((4,9)).float().to(device)
    model = Actor(obs_dim=9, hidden_dims=[32,32], action_dim=10, activation_name="elu", initial_std=1.0).to(device)
    out = model(states)
    print("action mean: ", out.shape)
    print("sample shape", model.act(states).shape)

