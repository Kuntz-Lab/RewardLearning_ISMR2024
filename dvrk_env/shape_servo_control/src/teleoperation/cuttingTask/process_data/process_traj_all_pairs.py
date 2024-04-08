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

sys.path.append("../pointcloud_representation_learning")
sys.path.append("../../pc_utils")
from architecture import AutoEncoder
from compute_partial_pc import farthest_point_sample_batched

import random
import copy



def get_trajectory_pairs(data_recording_path, group, pair):
    idx1, idx2 = pair

    file = os.path.join(data_recording_path, f"group {group} sample {idx1}.pickle")
    with open(file, 'rb') as handle:
        data1 = pickle.load(handle)

    file = os.path.join(data_recording_path, f"group {group} sample {idx2}.pickle")
    with open(file, 'rb') as handle:
        data2 = pickle.load(handle)
    
    return idx1, idx2, data1, data2

def to_obj_emb(model, device, pcd, visualize=False):
    pcd_tensor = torch.from_numpy(pcd.transpose(1,0)).unsqueeze(0).float().to(device)
    emb = model(pcd_tensor, get_global_embedding=True)
    
    if visualize:
        points = pcd
        print(points.shape)

        points = points[np.random.permutation(points.shape[0])]
    
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(np.array(points))  
        pcd.paint_uniform_color([0, 1, 0])

        points_tensor = torch.from_numpy(points.transpose(1,0)).unsqueeze(0).float().to(device)
        reconstructed_points = model(points_tensor)
        
        reconstructed_points = np.swapaxes(reconstructed_points.squeeze().cpu().detach().numpy(),0,1)
        reconstructed_points = reconstructed_points[:,:3]
        print(reconstructed_points.shape)

        pcd2 = open3d.geometry.PointCloud()
        pcd2.points = open3d.utility.Vector3dVector(np.array(reconstructed_points)) 
        pcd2.paint_uniform_color([1, 0, 0])
        open3d.visualization.draw_geometries([pcd, pcd2.translate((0,0,0.1))]) 
    
    return emb

def to_obj_emb_batched(model, device, pcds, visualize=False):
    pcd_tensor = torch.from_numpy(pcds.transpose(0,2,1)).float().to(device)
    emb = model(pcd_tensor, get_global_embedding=True)
    
    if visualize:
        points = pcds[0]
        print(points.shape)

        points = points[np.random.permutation(points.shape[0])]
    
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(np.array(points))  
        pcd.paint_uniform_color([0, 1, 0])

        points_tensor = torch.from_numpy(points.transpose(1,0)).unsqueeze(0).float().to(device)
        reconstructed_points = model(points_tensor)
        
        reconstructed_points = np.swapaxes(reconstructed_points.squeeze().cpu().detach().numpy(),0,1)
        reconstructed_points = reconstructed_points[:,:3]
        print(reconstructed_points.shape)

        pcd2 = open3d.geometry.PointCloud()
        pcd2.points = open3d.utility.Vector3dVector(np.array(reconstructed_points)) 
        pcd2.paint_uniform_color([1, 0, 0])
        open3d.visualization.draw_geometries([pcd, pcd2.translate((0,0,0.1))]) 
    
    return emb

def process_state_for_gt(eef_state, all_balls_xyz, last_ball_xyz):
    eef_xyz = np.array([eef_state["pose"]["p"]["x"], eef_state["pose"]["p"]["y"], eef_state["pose"]["p"]["z"]])
    reduced_balls_xyz = []
    for i, ball_xyz in enumerate(all_balls_xyz):
        if ball_xyz[0] != 100 or ball_xyz[1] !=100 or ball_xyz[2] !=-100:
            ball_xyz = np.array([ball_xyz[0], ball_xyz[1], ball_xyz[2]])
            reduced_balls_xyz.append(ball_xyz)
    
    return eef_xyz, reduced_balls_xyz, last_ball_xyz

def gt_reward_function_inv_dist(eef_xyz, balls_xyz, last_ball_xyz):
    reward = -math.inf
    if len(balls_xyz)==0:
        reward = 1/(np.sum((eef_xyz - last_ball_xyz)**2)+1e-4)
        return reward
    for i, ball_xyz in enumerate(balls_xyz):
        reward = max(1/(np.sum((eef_xyz - ball_xyz)**2)+1e-4), reward)
    return reward

def gt_reward_function_neg_dist(eef_xyz, balls_xyz, last_ball_xyz):
    reward = -math.inf
    if len(balls_xyz)==0:
        reward = -np.sum((eef_xyz - last_ball_xyz)**2)
        return reward
    for i, ball_xyz in enumerate(balls_xyz):
        reward = max(-(np.sum((eef_xyz - ball_xyz)**2)), reward)
    return reward

def gt_reward_function_inv_dist_normalized(eef_xyz, balls_xyz, last_ball_xyz):
    reward = -math.inf
    if len(balls_xyz)==0:
        reward = 1/(np.sum((eef_xyz - last_ball_xyz)**2)+1e-4)
        return reward
    for i, ball_xyz in enumerate(balls_xyz):
        reward = max(1/(np.sum((eef_xyz - ball_xyz)**2)+1e-4), reward)
    return reward/1e4


def compute_traj_reward(data):
    #print(data.keys())
    eef_states = data["eef_states"]
    balls_xyzs_list = data["balls_xyzs_list"]
    last_ball_xyzs = data["last_ball_xyzs"]
    num_balls_reached = data["num_balls_reached"]

    traj_len = len(eef_states)
    assert(len(eef_states)==len(balls_xyzs_list)==len(last_ball_xyzs))
    cum_reward = 0
    rewards_no_bonus = []
    for t in range(traj_len):
        eef_xyz, all_balls_xyz, last_ball_xyz = process_state_for_gt(eef_states[t], balls_xyzs_list[t], last_ball_xyzs[t])
        reward = gt_reward_function_inv_dist(eef_xyz, all_balls_xyz, last_ball_xyz)
        cum_reward += reward
        rewards_no_bonus.append(reward)
    
    bonus = 1000000 #800 #0 #1000000 #0
    cum_reward += num_balls_reached * bonus

    return cum_reward, rewards_no_bonus


def get_states(data, model, device):
    """Return point clouds,eef poses of the trajectory"""

    pcds = np.array(data["pcds"])
    obj_embs = torch.ones([len(pcds), model.embedding_size], dtype=torch.float64, device=device)

    obj_embs = to_obj_emb_batched(model, device, pcds).float()

    eef_xyzs = []
    for i, eef_state in enumerate(data["eef_states"]):
        eef_pose = np.array(list(eef_state["pose"]["p"]))
        eef_xyzs.append(eef_pose)
    eef_xyzs = np.array(eef_xyzs)

    assert(len(eef_xyzs)==len(pcds))
   
    return obj_embs, eef_xyzs


def compare_two_trajectories_w_reward(data_1, data_2, model, device):
    datas = [data_1, data_2]
    success_counts = []
    emb_states = []
    eef_states = []
    rewards  = []
    rewards_no_bonus_trajs = []

    for i, data in enumerate(datas):
        obj_embs, eef_xyzs = get_states(data, model, device)
        emb_states.append(obj_embs)
        eef_states.append(torch.from_numpy(eef_xyzs).float().to(device))

        success_counts.append(int(data["num_balls_reached"]))
        traj_reward, rewards_no_bonus = compute_traj_reward(data)
        rewards.append(traj_reward)
        rewards_no_bonus_trajs.append(rewards_no_bonus)

    if rewards[0] > rewards[1]:
        label=False
    else:
        label=True
  

    return emb_states, eef_states, label, rewards, rewards_no_bonus_trajs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--data_recording_path', type=str, help="path to recorded data")
    parser.add_argument('--data_processed_path', type=str, help="path to save processed data")
    parser.add_argument('--AE_model_path', type=str, help="path to pre-trained autoencoder weights")
    parser.add_argument('--num_group', default=30, type=int, help="num groups to process")
    parser.add_argument('--num_samples_per_group', default=30, type=int, help="num samples per group")
    parser.add_argument('--rand_seed', default=2021, type=int, help="random seed")


    args = parser.parse_args()

    np.random.seed(args.rand_seed)
    random.seed(args.rand_seed)

    data_recording_path = args.data_recording_path
    data_processed_path = args.data_processed_path
    AE_model_path = args.AE_model_path
    NUM_GROUP = args.num_group
    num_samples_per_group = args.num_samples_per_group
    num_data_pt = NUM_GROUP * (num_samples_per_group*(num_samples_per_group-1)/2) 
    os.makedirs(data_processed_path, exist_ok=True)
    print(f"data processed path: {data_processed_path}")
    print(f"num data pt : {num_data_pt}")

    start_time = timeit.default_timer() 

    device = torch.device("cuda")
    model = AutoEncoder(num_points=256, embedding_size=256).to(device)
    model.load_state_dict(torch.load(AE_model_path))
    model.eval()

   
    all_sample_pairs = []
    for idx1 in range(num_samples_per_group):
        for idx2 in range(num_samples_per_group):
            if (idx1 != idx2) and (idx2, idx1) not in all_sample_pairs:
                    all_sample_pairs.append((idx1, idx2))

    assert(len(all_sample_pairs)==num_samples_per_group*(num_samples_per_group-1)/2)

    data_idx = 0
    for group_idx in range(NUM_GROUP):
        if group_idx % 2 == 0:
            print("========================================")
            print("current group:", group_idx, " , time passed:", timeit.default_timer() - start_time)

        for pair_idx in range(len(all_sample_pairs)):
            idx_1, idx_2, data_1, data_2 = get_trajectory_pairs(data_recording_path, group_idx, all_sample_pairs[pair_idx])
            
        
            emb_states, eef_states, label, gt_rewards, rewards_no_bonus_trajs = compare_two_trajectories_w_reward(data_1, data_2, model, device)
            processed_data = {"emb_traj_1": emb_states[0], "emb_traj_2": emb_states[1], \
                                "eef_traj_1": eef_states[0], "eef_traj_2": eef_states[1], "indices":[idx_1, idx_2], "group":group_idx, \
                                "gt_reward_1": gt_rewards[0], "gt_reward_2": gt_rewards[1], \
                                "label": label, "balls_xyz": data_1["balls_xyz"],  \
                                "rewards_no_bonus_1": rewards_no_bonus_trajs[0], "rewards_no_bonus_2": rewards_no_bonus_trajs[1], \
                                "num_balls_reached_1": data_1["num_balls_reached"], "num_balls_reached_2": data_2["num_balls_reached"]}
        
            
            with open(os.path.join(data_processed_path, "processed sample " + str(data_idx) + ".pickle"), 'wb') as handle:
                pickle.dump(processed_data, handle, protocol=3)    

            data_idx+=1 




        