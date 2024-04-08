import sys

from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym import gymutil
import torch
import math
import argparse
import os
import numpy as np
import pickle
import timeit
import open3d

sys.path.append("../../pointcloud_representation_learning")
sys.path.append("../../../pc_utils")
sys.path.append("../../process_data")
from architecture import AutoEncoder
from compute_partial_pc import farthest_point_sample_batched
from process_traj_w_reward import to_obj_emb_batched, process_state_for_gt, gt_reward_function_inv_dist, gt_reward_function_neg_dist, gt_reward_function_gaussian, compute_traj_reward


import matplotlib.pyplot as plt
from reward import RewardNetPointCloudEEF as RewardNet

def get_trajectory(data_recording_path, group, idx):
    '''
    get a trajectory
    '''
    file = os.path.join(data_recording_path, f"group {group} sample {idx}.pickle")
    with open(file, 'rb') as handle:
        data = pickle.load(handle)
    
    return idx, data

def compute_traj_reward(data):
    eef_states = data["eef_states"]
    balls_poses_list = data["balls_poses_list"]
    last_ball_poses = data["last_ball_poses"]
    num_balls_reached = data["num_balls_reached"]

    traj_len = len(eef_states)
    assert(len(eef_states)==len(balls_poses_list)==len(last_ball_poses))
    cum_reward = 0
    rewards_no_bonus = []
    for t in range(traj_len):
        eef_xyz, reduced_balls_xyz, last_ball_xyz = process_state_for_gt(eef_states[t], balls_poses_list[t], last_ball_poses[t])
        reward = gt_reward_function_inv_dist(eef_xyz, reduced_balls_xyz, last_ball_xyz)
        cum_reward += reward
        rewards_no_bonus.append(reward)
    
    bonus = 1000000 #800 #0 #1000000 #0
    cum_reward += num_balls_reached * bonus

    return cum_reward, rewards_no_bonus


def get_states(data, model, device):
    """Return point clouds,eef poses of the trajectory"""

    pcds = data["pcds"]
    obj_embs = torch.ones([len(pcds), model.embedding_size], dtype=torch.float64, device=device)
    
    ################## for disappear balls (faster in batch) ################
    processed_pcds = []
    max_num_points = max([len(pcd) for pcd in pcds])
    pcd_is_empty = torch.zeros((len(pcds), )).to(device)
    first_empty_idx = -1
    is_first_empty = True
    for i, pcd in enumerate(pcds):
        # print("pcd: ", pcd.shape)
        processed_pcd = np.zeros((max_num_points, 3))
        if len(pcd)==0:
            #print(f"first pcd empty at {i}")
            pcd_is_empty[i] = 1
            if is_first_empty:
                first_empty_idx = i
                is_first_empty = False
        else:
            pad_point = pcd[-1, :]
            processed_pcd[:len(pcd), :] = pcd
            processed_pcd[len(pcd):, :] = np.expand_dims(pad_point, axis=0)
        processed_pcds.append(np.expand_dims(processed_pcd, axis=0))

    pcds = np.concatenate(processed_pcds, axis=0)
    
    pcds = np.array(farthest_point_sample_batched(pcds, npoint=256))

    obj_embs = to_obj_emb_batched(model, device, pcds).float()
    emb_before_empty = obj_embs[first_empty_idx-1]

    pcd_is_empty = pcd_is_empty.unsqueeze(1).expand(-1, 256)
    obj_embs = torch.where(pcd_is_empty==1, emb_before_empty, obj_embs)

    eef_poses = []
    for i, eef_state in enumerate(data["eef_states"]):
        eef_pose = np.array(list(eef_state["pose"]["p"]))
        eef_poses.append(eef_pose)
    eef_poses = np.array(eef_poses)

    assert(len(eef_poses)==len(pcds))
   
    return obj_embs, eef_poses



def plot_bar_chart(xs, rewards, title):
    fig, ax = plt.subplots()
    ax.bar(xs, rewards, width=0.8,align='center', tick_label=["both", "reach 1 near 1", "near 1", "no reach"])
    #ax.legend()
    ax.set_title(title)
    ax.set_xlabel("traj")
    ax.set_ylabel("cumulative reward")
    os.makedirs(f"./figures/vis_preference", exist_ok=True)
    fig.savefig(f"./figures/vis_preference/{title}.png")
    plt.cla()
    plt.close(fig)

def plot_traj_learned_reward(traj_reward, data, group_idx, sample_idx):
    max_reward = torch.max(traj_reward).item()
    min_reward = torch.min(traj_reward).item()

    balls_xyz = data["balls_xyz"]
    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")
    for ball_pose in balls_xyz:
        ax.plot(ball_pose[0], ball_pose[1], ball_pose[2], "o", markersize=40)

    xs = [eef_poses[t,0] for t in range(len(eef_poses))]
    ys = [eef_poses[t,1] for t in range(len(eef_poses))]
    zs = [eef_poses[t,2] for t in range(len(eef_poses))]
    heats = [[(traj_reward[t].item() - min_reward) / (max_reward - min_reward), 0, 0] for t in range(len(eef_poses))]
    #heats = [[1, 0, 0] for t in range(len(eef_poses))]
    ax.scatter(xs, ys, zs, c=heats, s=[30+2*i for i in range(len(eef_poses))]) 
    ax.plot(xs, ys, zs, color='red')

    traj_plot_title = f"group {group_idx} sample {sample_idx}"
    plt.title(traj_plot_title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xticks([-0.1+0.02*i for i in range(11)])
    ax.set_yticks([-0.6+0.02*i for i in range(11)])
    ax.set_zticks([0+0.02*i for i in range(11)])
    os.makedirs(f"./figures/vis_preference/group{group_idx}", exist_ok=True)
    fig.savefig(f"./figures/vis_preference/group{group_idx}/{traj_plot_title}.png")
    plt.cla()
    plt.close(fig)   


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--data_recording_path', default=f"/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_trace_curve_corrected/ex_traj", type=str, help="path to recorded data")
    parser.add_argument('--AE_model_path', default="/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_AE_balls_corrected/weights_straight_flat_2ball/weights_1/epoch_150", type=str, help="path to pre-trained autoencoder weights")
    parser.add_argument('--rmp', default='/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_trace_curve_corrected/weights_straight_flat_2ball/weights_1/epoch_200', help="reward_model_path")
    args = parser.parse_args()

    data_recording_path = args.data_recording_path
    AE_model_path = args.AE_model_path
    reward_model_path = args.rmp 
    num_samples_per_group = 4
    NUM_GROUP = 1

    start_time = timeit.default_timer() 

    device = torch.device("cuda")
    model = AutoEncoder(num_points=256, embedding_size=256).to(device)
    model.load_state_dict(torch.load(AE_model_path))
    model.eval()


    ######################## gt reward ############################
    traj_gt_rewards = {i:[] for i in range(NUM_GROUP)}
    for group_idx in range(NUM_GROUP):
        if group_idx % 50 == 0:
                print("========================================")
                print("current count:", group_idx, " , time passed:", timeit.default_timer() - start_time)

        for sample_idx in range(num_samples_per_group):

            _, data = get_trajectory(data_recording_path, group_idx, sample_idx)  

            gt_rew, _ = compute_traj_reward(data)
            traj_gt_rewards[group_idx].append(gt_rew)

        plot_bar_chart([0,1,2,3], traj_gt_rewards[group_idx], f"group {group_idx} gt reward preference inv dist w bonus")


    ######################## learned reward #####################
    reward_net = RewardNet()
    reward_net.to(device)
    reward_net.eval()

    reward_net.load_state_dict(torch.load(reward_model_path))
    traj_learned_rewards = {i:[] for i in range(NUM_GROUP)}
    for group_idx in range(NUM_GROUP):

        bar_chart_title = f"group {group_idx} learned reward (inv dist) preference"

        if group_idx % 50 == 0:
                print("========================================")
                print("current count:", group_idx, " , time passed:", timeit.default_timer() - start_time)

        for sample_idx in range(num_samples_per_group):

            _, data = get_trajectory(data_recording_path, group_idx, sample_idx)  

            print(data["num_balls_reached"])

            obj_embs, eef_poses = get_states(data, model, device)
            eef_traj = torch.tensor(eef_poses).to(device).float().unsqueeze(0)
            obj_embs = obj_embs.unsqueeze(0).float()
            with torch.no_grad():
                rew, _ = reward_net.cum_return(eef_traj, obj_embs)
                traj_reward = reward_net.single_return(eef_traj, obj_embs).squeeze(0)
            traj_learned_rewards[group_idx].append(rew[0].item())

            ########## vis traj ##########
            plot_traj_learned_reward(traj_reward, data, group_idx, sample_idx)

        plot_bar_chart([0,1,2,3], traj_learned_rewards[group_idx], bar_chart_title)

    

        