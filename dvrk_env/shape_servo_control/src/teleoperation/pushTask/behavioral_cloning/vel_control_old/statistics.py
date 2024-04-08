import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset_loader import BCDataset
from policy_BC import Actor
import os
import argparse
import matplotlib.pyplot as plt
import pickle

from torch.distributions import MultivariateNormal

def plot_curve(losses, title):
    epochs = [i for i in range(len(losses))]
    fig, ax = plt.subplots()
    ax.plot(epochs, losses)
    ax.set_title(title)
    ax.set_xlabel("frame")
    ax.set_ylabel("losses")
    os.makedirs(f"./figures_stats", exist_ok=True)
    fig.savefig(f"./figures_stats/{title}.png")

def plot_dist(action_values, dim):
    max = torch.max(action_values).item()
    #print("max", max)
    min = torch.min(action_values).item()
    #print("min", min)
    plt.hist(action_values, color = 'blue', edgecolor = 'black',
         bins = int((max-min)*10.0))
    plt.title(f'Histogram of action value at dim {dim}')
    plt.xlabel('scaled (x1000) action value')
    plt.ylabel('frequency')
    plt.show()
    
def plot_dist_and_save(action_values, dim, title):
    fig, ax = plt.subplots()
    max = torch.max(action_values).item()
    #print("max", max)
    min = torch.min(action_values).item()
    #print("min", min)
    ax.hist(action_values, color = 'blue', edgecolor = 'black',
         bins = int((max-min)*10.0))
    ax.set_title(title)
    ax.set_xlabel(f"scaled (x1000) value of action at dim {dim}")
    ax.set_ylabel("scaled frequency")
    os.makedirs(f"./figures_stats", exist_ok=True)
    fig.savefig(f"./figures_stats/{title}.png")
    plt.cla()
    plt.close(fig)

if __name__ == "__main__":
    data_processed_path = f"/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_push_BC_corrected/data_processed_train_400"
    model_path = f"/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_push_BC_corrected/weights/weights_max_prob_400"
    num_data = len(os.listdir(data_processed_path))
    print(f"num_data: {num_data}")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Actor(obs_dim=9, hidden_dims=[256, 128, 64], action_dim=10, activation_name="elu", initial_std=1.0)
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(model_path, "epoch 200")))
    model.eval()

    states = []
    actions = []
    for i in range(num_data):
        if i%1000==0:
            print(f"finished {(i+1)/num_data*100} %")
        
        with open(os.path.join(data_processed_path, "processed sample " + str(i) + ".pickle"), 'rb') as handle:
            sample = pickle.load(handle)
        
        state, action = sample["state"].to(device), sample["action"].to(device)
        states.append(state)
        actions.append(action)
    
    states = torch.cat(states, dim=0)
    actions = torch.cat(actions, dim=0)
    print(f"states shape: {states.shape}")
    print(f"actions shape: {actions.shape}")


    ################################ plot sampled action given state #####################################################
    # num_samples = 10000
    # for data_idx in range(0, 71, 10):
    #     print("==========================================================")
    #     print(f"for data {data_idx}: ")
    #     state = states[data_idx].unsqueeze(0)
    #     expert_action = actions[data_idx]
    #     sampled_actions = []
    #     for i in range(num_samples):
    #         action_mean = model(state)
    #         covariance = torch.diag(model.log_std.exp() * model.log_std.exp())
    #         distribution = MultivariateNormal(action_mean, scale_tril=covariance)
    #         sampled_action = distribution.sample()
    #         sampled_actions.append(sampled_action)
    #         #print(sampled_action)

    #     sampled_actions = torch.cat(sampled_actions, dim=0)
    #     print("expert action: ", expert_action)
    #     print("now plotting ...")
    #     for i in range(10):
    #         plot_dist_and_save((sampled_actions[:,i]*1000).cpu(), dim=i, title=f"data_idx_{data_idx}_action_dist_at_dim_{i}")
    #     print("==========================================================")

    ################################ plot L2 distance #####################################################
    losses = []
    for data_idx in range(0, 71, 1):
        print("==========================================================")
        print(f"for data {data_idx}: ")
        state = states[data_idx].unsqueeze(0)
        expert_action = actions[data_idx]
        #sampled_actions = []
        action_mean = model(state)
        covariance = torch.diag(model.log_std.exp() * model.log_std.exp())
        distribution = MultivariateNormal(action_mean, scale_tril=covariance)
        sampled_action = distribution.sample()
        #sampled_actions.append(sampled_action)
        print("sampled action: ", sampled_action)
        print("expert action: ", expert_action)

        loss = torch.sum((sampled_action - expert_action)**2).cpu()
        losses.append(loss)
        print("==========================================================")

    plot_curve(losses, title="actor L2 loss over 70 frames")
    ################################ estimate action mean (not conditioned on states) ######################################
    
    # # (action_dim, )
    # action_mean_estimate = torch.sum(actions, dim=0).unsqueeze(0)/num_data
    # # (num_data, action_dim)
    # diff = actions - action_mean_estimate
    # # (action_dim, num_data) * (num_data, action_dim)
    # covariance_estimate =diff.permute((1,0)).matmul(diff)/(num_data-1) 
    
    # model_means = model(states)
    # model_covariance = torch.diag(model.log_std.exp() * model.log_std.exp())
    # dist = MultivariateNormal(model_means, scale_tril=model_covariance)
    # print(f"log_prob shape: {len(-dist.log_prob(actions))}")
    
    # print(f"sample action mean: {action_mean_estimate}")
    # print(f"model mean avg: {torch.sum(model_means, dim=0)/num_data}")
    # print(f"sample covariance: {covariance_estimate}")
    # print(f"model covariance: {model_covariance.matmul(model_covariance.permute((1,0)))}")
    
    # sampled_actions = model.act(states)
    
    # # for i in range(10):
    # #     plot_dist_and_save((actions[:,i]*1000).cpu(), dim=i, title=f"expert_action_distribution_at_dim_{i}")
    # #     plot_dist_and_save((sampled_actions[:,i]*1000).cpu(), dim=i, title=f"sampled_action_distribution_at_dim_{i}")