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

def get_trajectory_pairs(data_recording_path, group, unused_pairs):
    '''
    get a random trajectory pairs that has not been collected yet from a group.
    '''
    sample = random.sample(unused_pairs[group])
    idx1, idx2 = sample[0]
    unused_pairs[group].remove((idx1, idx2))

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

def process_state_for_gt(eef_state, all_balls_poses, last_ball_pose):
    eef_xyz = np.array([eef_state["pose"]["p"]["x"], eef_state["pose"]["p"]["y"], eef_state["pose"]["p"]["z"]])
    reduced_balls_xyz = []
    for i, ball_pose in enumerate(all_balls_poses):
        if ball_pose.p.x != 100 or ball_pose.p.y !=100 or ball_pose.p.z !=-100:
            ball_xyz = np.array([ball_pose.p.x, ball_pose.p.y, ball_pose.p.z])
            reduced_balls_xyz.append(ball_xyz)
    
    if last_ball_pose != None:
        last_ball_xyz = np.array([last_ball_pose.p.x, last_ball_pose.p.y, last_ball_pose.p.z])
    else:
        last_ball_xyz = None
    
    return eef_xyz, reduced_balls_xyz, last_ball_xyz

def gt_reward_function_inv_dist(eef_xyz, balls_xyz, last_ball_xyz):
    reward = -math.inf
    if len(balls_xyz)==0:
        reward = 1/(np.sum((eef_xyz - last_ball_xyz)**2)+1e-4)
        return reward
    for i, ball_xyz in enumerate(balls_xyz):
        reward = max(1/(np.sum((eef_xyz - ball_xyz)**2)+1e-4), reward)
    return reward


def gt_reward_function_inv_dist_sum(eef_xyz, balls_xyz, last_ball_xyz):
    reward = 0
    if len(balls_xyz)==0:
        reward = 1/(np.sum((eef_xyz - last_ball_xyz)**2)+1e-2)
        return reward
    for i, ball_xyz in enumerate(balls_xyz):
        reward += 1/(np.sum((eef_xyz - ball_xyz)**2)+1e-2)
    return reward


def gt_reward_function_neg_dist(eef_xyz, balls_xyz, last_ball_xyz):
    reward = -math.inf
    if len(balls_xyz)==0:
        reward = -np.sum((eef_xyz - last_ball_xyz)**2)
        return reward
    for i, ball_xyz in enumerate(balls_xyz):
        reward = max(-(np.sum((eef_xyz - ball_xyz)**2)), reward)
    return reward


def gt_reward_function_gaussian(eef_xyz, balls_xyz, last_ball_xyz):
    reward = 0
    var = 0.03 #smooth
    if len(balls_xyz)==0:
        reward = 1/(var*(2*np.pi)**0.5)*np.exp(-np.sum((eef_xyz - last_ball_xyz)**2)/(2*var**2))
        return reward
    reward = -math.inf
    for i, ball_xyz in enumerate(balls_xyz):
        gauss_reward = 1/(var*(2*np.pi)**0.5)*np.exp(-np.sum((eef_xyz - ball_xyz)**2)/(2*var**2))
        reward = max(gauss_reward, reward)
    return reward

def compute_traj_reward(data):
    #print(data.keys())
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
    
    bonus = 0 #1000000 #800 #0 #1000000 #0
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


def compare_two_trajectories_w_reward(data_1, data_2, model, device):
    datas = [data_1, data_2]
    success_counts = []
    emb_states = []
    eef_states = []
    rewards  = []
    rewards_no_bonus_trajs = []

    for i, data in enumerate(datas):
        obj_embs, eef_poses = get_states(data, model, device)
        emb_states.append(obj_embs)
        eef_states.append(torch.from_numpy(eef_poses).float().to(device))

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
    is_train = True
    suffix = "train" if is_train else "test"
    parser.add_argument('--data_recording_path', default=f"/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_trace_curve_corrected/demos_{suffix}_straight_flat_2ball", type=str, help="path to recorded data")
    parser.add_argument('--data_processed_path', default=f"/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_trace_curve_corrected/processed_data_{suffix}_straight_flat_2ball", type=str, help="path data to be processed")
    parser.add_argument('--AE_model_path', default="/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_AE_balls_corrected/weights_straight_flat_2ball/weights_1/epoch_150", type=str, help="path to pre-trained autoencoder weights")
    parser.add_argument('--num_group', default=30, type=int, help="num groups to process")
    parser.add_argument('--num_samples_per_group', default=30, type=int, help="num samples per group")
    parser.add_argument('--num_data_pt', default=14000, type=int, help="num datapoints to create and save")
    parser.add_argument('--rand_seed', default=2021, type=int, help="random seed")


    args = parser.parse_args()

    np.random.seed(args.rand_seed)
    random.seed(args.rand_seed)

    data_recording_path = args.data_recording_path
    data_processed_path = args.data_processed_path
    AE_model_path = args.AE_model_path
    NUM_GROUP = args.num_group
    num_samples_per_group = args.num_samples_per_group
    num_data_pt = args.num_data_pt
    os.makedirs(data_processed_path, exist_ok=True)
    print(f"data processed path: {data_processed_path}")
    print(f"numdata pt : {num_data_pt}")


    start_time = timeit.default_timer() 

    device = torch.device("cuda")
    model = AutoEncoder(num_points=256, embedding_size=256).to(device)
    model.load_state_dict(torch.load(AE_model_path))
    model.eval()

    all_sample_pairs = set()
    for idx1 in range(num_samples_per_group):
        for idx2 in range(num_samples_per_group):
            if (idx1 != idx2) and (idx2, idx1) not in all_sample_pairs:
                    all_sample_pairs.add((idx1, idx2))

    assert(len(all_sample_pairs)==num_samples_per_group*(num_samples_per_group-1)/2)

     # make sure no pairs of trajectories are repeated
    unused_pairs = {group:copy.deepcopy(all_sample_pairs) for group in range(NUM_GROUP)}
    unused_pairs[-1] = set()


    # make sure num_data_pt is not larger than (num_group*(num_sample_per_group C 2))
    for i in range(num_data_pt):

        if i % 50 == 0:
            print("========================================")
            print("current count:", i, " , time passed:", timeit.default_timer() - start_time)

        group_idx = -1
        while len(unused_pairs[group_idx])==0:
            group_idx = random.choise(list(unused_pairs.keys()))#np.random.randint(low=0, high=NUM_GROUP)

        idx_1, idx_2, data_1, data_2 = get_trajectory_pairs(data_recording_path, group_idx, unused_pairs)

        emb_states, eef_states, label, gt_rewards, rewards_no_bonus_trajs = compare_two_trajectories_w_reward(data_1, data_2, model, device)
        processed_data = {"emb_traj_1": emb_states[0], "emb_traj_2": emb_states[1], \
                        "eef_traj_1": eef_states[0], "eef_traj_2": eef_states[1], "indices":[idx_1, idx_2], "group":group_idx, \
                        "gt_reward_1": gt_rewards[0], "gt_reward_2": gt_rewards[1], \
                        "label": label, "balls_xyz": data_1["balls_xyz"],  \
                        "rewards_no_bonus_1": rewards_no_bonus_trajs[0], "rewards_no_bonus_2": rewards_no_bonus_trajs[1], \
                        "num_balls_reached_1": data_1["num_balls_reached"], "num_balls_reached_2": data_2["num_balls_reached"]}
        
        with open(os.path.join(data_processed_path, "processed sample " + str(i) + ".pickle"), 'wb') as handle:
            pickle.dump(processed_data, handle, protocol=3)     




    