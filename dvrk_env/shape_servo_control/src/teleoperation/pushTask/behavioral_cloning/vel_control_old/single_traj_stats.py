import sys
# import roslib.packages as rp
# pkg_path = rp.get_pkg_dir('shape_servo_control')
# sys.path.append(pkg_path + '/src')

#import open3d

#from util.isaac_utils import *

import torch
import math
import argparse
import os
import numpy as np
import pickle
import timeit
from random import sample
import open3d
#from utils import *
import matplotlib.pyplot as plt

from policy_BC import Actor
from torch.distributions import MultivariateNormal

def get_trajectory(data_recording_path, group, idx=0):
    '''
    get a trajectory randomly from a group.
    Returns the sample_idx, data
    '''
    # idx = np.random.randint(low=0, high=num_samples_per_group)
    file = os.path.join(data_recording_path, f"group {group} sample {idx}.pickle")
    with open(file, 'rb') as handle:
        data = pickle.load(handle)
    
    return idx, data


def get_states(data):

    eef_poses = []
    for i, eef_state in enumerate(data["eef_states"]):
        eef_pose = np.array(list(eef_state["pose"]["p"]))
        eef_poses.append(eef_pose)
    eef_poses = np.array(eef_poses)

    box_poses = []
    for i, box_state in enumerate(data["box_states"]):
        box_pose = np.array(list(box_state["pose"]["p"]))
        box_pose = [box_pose[0][0], box_pose[0][1], box_pose[0][2]]
        box_poses.append(box_pose)

    cone_pose = list(data["cone_pose"])
    cone_poses = [cone_pose for _ in range(len(box_poses))]

    dof_vels = list(data["dof_vels"])
    
    assert(len(cone_poses)==len(box_poses)==len(eef_poses)==len(dof_vels))
   
    return eef_poses, box_poses, cone_poses, dof_vels

def plot_curve(vals, title, y_label="L2 difference of action (dof_vel)"):
    time = [i for i in range(len(vals))]
    fig, ax = plt.subplots()
    ax.plot(time, vals)
    ax.set_title(title)
    ax.set_xlabel("time")
    ax.set_ylabel(y_label)
    os.makedirs(f"./figures_stats", exist_ok=True)
    fig.savefig(f"./figures_stats/{title}.png")
    
def plot_log_probs(expert_probs, policy_probs, title):
    time = [i for i in range(len(expert_probs))]
    fig, ax = plt.subplots()
    ax.plot(time, expert_probs, label="expert action")
    ax.plot(time, policy_probs, label="policy action")
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel("time")
    ax.set_ylabel("model's log prob of action at time t")
    os.makedirs(f"./figures_stats", exist_ok=True)
    fig.savefig(f"./figures_stats/{title}.png")




if __name__ == "__main__":
    ### CHANGE ####
    demo_idx = 1
    policy_demo_path = f"/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_push_BC_corrected/policy_demos_{demo_idx}_L2"
    expert_demo_path = f"/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_push_BC_corrected/expert_demos_{demo_idx}"

    device = torch.device("cuda")

    _, policy_data = get_trajectory(policy_demo_path, 0)  
    _, expert_data = get_trajectory(expert_demo_path, 0)  
    pi_eef_poses, pi_box_poses, pi_cone_poses, pi_dof_vels = get_states(policy_data)
    best_eef_poses, best_box_poses, best_cone_poses, best_dof_vels = get_states(expert_data)
    assert(len(pi_eef_poses)==len(best_eef_poses))
    traj_len = len(best_eef_poses)

    model_path = f"/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_push_BC_corrected/weights/weights_max_prob_400"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Actor(obs_dim=9, hidden_dims=[256, 128, 64], action_dim=10, activation_name="elu", initial_std=1.0)
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(model_path, "epoch 200")))
    model.eval()
    
    L2_diff_actions = []
    expert_log_probs = []
    policy_log_probs = []
    for j in range(traj_len):
        pi_eef_pose = torch.tensor(pi_eef_poses[j], device=device).unsqueeze(0)
        pi_box_pose = torch.tensor(pi_box_poses[j], device=device).unsqueeze(0)
        pi_cone_pose = torch.tensor(pi_cone_poses[j], device=device).unsqueeze(0)
        pi_dof_vel = torch.tensor(pi_dof_vels[j], device=device).unsqueeze(0)
        pi_state = torch.cat((pi_eef_pose, pi_box_pose, pi_cone_pose), dim=-1)
        
        best_eef_pose = torch.tensor(pi_eef_poses[j], device=device).unsqueeze(0)
        best_box_pose = torch.tensor(pi_box_poses[j], device=device).unsqueeze(0)
        best_cone_pose = torch.tensor(pi_cone_poses[j], device=device).unsqueeze(0)
        best_dof_vel = torch.tensor(best_dof_vels[j], device=device).unsqueeze(0)
        best_state = torch.cat((best_eef_pose, best_box_pose, best_cone_pose), dim=-1)
        
        L2_diff_action = torch.sum((pi_dof_vel - best_dof_vel)**2).cpu()
        L2_diff_actions.append(L2_diff_action)
        
        policy_mean = model(best_state)
        policy_covariance = torch.diag(model.log_std.exp() * model.log_std.exp())
        dist = MultivariateNormal(policy_mean, scale_tril=policy_covariance)
        policy_log_prob = dist.log_prob(pi_dof_vel)[0]
        policy_log_probs.append(policy_log_prob.item())

        expert_mean = model(best_state)
        expert_covariance = torch.diag(model.log_std.exp() * model.log_std.exp())
        dist = MultivariateNormal(expert_mean, scale_tril=expert_covariance)

        expert_log_prob = dist.log_prob(best_dof_vel)[0]
        expert_log_probs.append(expert_log_prob.item())
        
        # print(f"expert_log_prob: {expert_log_prob}")
        # print(f"policy_log_prob: {policy_log_prob}")
        
    plot_curve(L2_diff_actions, "L2 difference of expert and policy actions (L2)", y_label="L2 difference of action (dof_vel)")
    plot_log_probs(expert_log_probs, policy_log_probs, "log prob of expert's and policy's actions (L2)")
       


    