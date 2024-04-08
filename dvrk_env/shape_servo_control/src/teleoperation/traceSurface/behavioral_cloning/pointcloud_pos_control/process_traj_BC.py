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
from architecture import AutoEncoder
from compute_partial_pc import farthest_point_sample_batched

'''
Extract and save states and rewards from each trajectory
'''

def get_trajectory(data_recording_path, group, sample_idx):
    '''
    get a trajectory from a group and a specific sample index.
    Returns data
    '''
    file = os.path.join(data_recording_path, f"group {group} sample {sample_idx}.pickle")
    with open(file, 'rb') as handle:
        data = pickle.load(handle)
    
    return data

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

def compute_traj_states_and_reward(data, model, device):
    obj_embs, eef_poses = get_states(data, model, device)
    eef_poses = torch.from_numpy(eef_poses).float().to(device)
    traj_reward, rewards_no_bonus = compute_traj_reward(data)
    return obj_embs, eef_poses, traj_reward, rewards_no_bonus

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--data_recording_path', type=str, help="path to recorded data")
    parser.add_argument('--data_processed_path', type=str, help="path to data to be processed")
    parser.add_argument('--AE_model_path', type=str, help="path to pre-trained autoencoder weights")
    parser.add_argument('--num_groups', default=30, type=int, help="num groups to process")
    parser.add_argument('--num_samples_per_group', default=30, type=int, help="num samples per group")
    parser.add_argument('--rand_seed', default=2021, type=int, help="random seed")


    args = parser.parse_args()

    np.random.seed(args.rand_seed)

    data_recording_path = args.data_recording_path
    data_processed_path = args.data_processed_path
    AE_model_path = args.AE_model_path
    NUM_GROUP = args.num_groups
    num_samples_per_group = args.num_samples_per_group
    os.makedirs(data_processed_path, exist_ok=True)
    print(f"data processed path: {data_processed_path}")

    start_time = timeit.default_timer() 

    device = torch.device("cuda")
    model = AutoEncoder(num_points=256, embedding_size=256).to(device)
    model.load_state_dict(torch.load(AE_model_path))
    model.eval()


    for group_idx in range(NUM_GROUP):
        for sample_idx in range(num_samples_per_group):

            if group_idx % 2 == 0:
                print("========================================")
                print("current group:", group_idx, " , time passed:", timeit.default_timer() - start_time)

            data = get_trajectory(data_recording_path, group_idx, sample_idx)  
            emb_states, eef_states, traj_reward, rewards_no_bonus = compute_traj_states_and_reward(data, model, device)
            processed_data = {"emb_traj": emb_states, "eef_traj": eef_states, "gt_reward": traj_reward, "rewards_no_bonus": rewards_no_bonus, 
                            "balls_xyz": data["balls_xyz"], "num_balls_reached": data["num_balls_reached"]}
            
            with open(os.path.join(data_processed_path, f"group {group_idx} sample {sample_idx}.pickle"), 'wb') as handle:
                pickle.dump(processed_data, handle, protocol=3)     