import numpy as np

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import os
import sys
sys.path.append('/home/dvrk/catkin_ws/src/dvrk_env/shape_servo_control/src/teleoperation/pushTask/behavioral_cloning/pos_control')
from policy_BC_PC import ActorPC


class CustomActorCritic(nn.Module):

    def __init__(self, obs_shape, states_shape, actions_shape, initial_std, model_cfg, asymmetric=False):
        super(CustomActorCritic, self).__init__()

        self.asymmetric = asymmetric

        if model_cfg is None:
            actor_hidden_dim = [256, 256, 256]
            critic_hidden_dim = [256, 256, 256]
            activation = get_activation("selu")
        else:
            actor_hidden_dim = model_cfg['pi_hid_sizes']
            critic_hidden_dim = model_cfg['vf_hid_sizes']
            activation = get_activation(model_cfg['activation'])

        #print(f"--------obs_shape: {*obs_shape}   actions_shape: {*actions_shape}")
        # Policy
        self.actor = ActorPC(act_dim=2, robot_state_dim=2, emb_dim=4, initial_std=1.0) #Actor(obs_dim=obs_shape[0], hidden_dims=[256, 128, 64], action_dim=actions_shape[0], activation_name="elu", initial_std=initial_std)
        saved_model_path = "/home/dvrk/LfD_data/ex_push/BC/pos_control/weights/weights_1" #"/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_push_BC_corrected/weights/weights_max_prob_regularized"
        self.actor.load_state_dict(torch.load(os.path.join(saved_model_path, "epoch_200")))

        # self.actor = BCPolicy(obs_dim=9, act_dim=10)
        # saved_model_path = "/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_push_BC_corrected/weights/weights_L2"
        # self.actor.load_state_dict(torch.load(os.path.join(saved_model_path, "epoch 100")))
        # Value function
        critic_layers = []
        if self.asymmetric:
            critic_layers.append(nn.Linear(*states_shape, critic_hidden_dim[0]))
        else:
            critic_layers.append(nn.Linear(*obs_shape, critic_hidden_dim[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dim)):
            if l == len(critic_hidden_dim) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dim[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dim[l], critic_hidden_dim[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(self.actor)
        print(self.critic)

        # Action noise
        self.log_std = self.actor.log_std
        # initial_std = torch.ones(*actions_shape)
        # initial_std *= 1e-4
        # initial_std[0] /= (1e-4/2)
        # initial_std[3] /= 1e-4
        self.log_std = nn.Parameter(np.log(initial_std) * torch.ones(*actions_shape))

        # Initialize the weights like in stable baselines
        # actor_weights = [np.sqrt(2)] * len(actor_hidden_dim)
        # actor_weights.append(0.01)
        critic_weights = [np.sqrt(2)] * len(critic_hidden_dim)
        critic_weights.append(1.0)
        #self.init_weights(self.actor, actor_weights)
        self.init_weights(self.critic, critic_weights)

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def forward(self):
        raise NotImplementedError

    def act(self, observations, states):
        actions_mean = self.actor(observations)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions = distribution.sample()
        actions_log_prob = distribution.log_prob(actions)

        if self.asymmetric:
            value = self.critic(states)
        else:
            value = self.critic(observations)

        return actions.detach(), actions_log_prob.detach(), value.detach(), actions_mean.detach(), self.log_std.repeat(actions_mean.shape[0], 1).detach()

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, observations, states, actions):
        actions_mean = self.actor(observations)
        # print("observation size:", observations.shape)
        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions_log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        if self.asymmetric:
            value = self.critic(states)
        else:
            value = self.critic(observations)

        return actions_log_prob, entropy, value, actions_mean, self.log_std.repeat(actions_mean.shape[0], 1)


def get_activation(act_name):
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
