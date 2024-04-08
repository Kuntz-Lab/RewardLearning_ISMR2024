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
from random import sample
import open3d

sys.path.append("../pointcloud_representation_learning")
sys.path.append("../../pc_utils")
from architecture import AutoEncoder
from compute_partial_pc import farthest_point_sample_batched
#from utils import *

def get_random_trajectory(data_recording_path, group, num_samples_per_group):
    '''
    get a trajectory randomly from a group.
    Returns the sample_idx, data
    '''
    idx = np.random.randint(low=0, high=num_samples_per_group)
    file = os.path.join(data_recording_path, f"group {group} sample {idx}.pickle")
    with open(file, 'rb') as handle:
        data = pickle.load(handle)
    
    return idx, data

def get_trajectory(data_recording_path, group, sample_idx):
    '''
    get a specific trajectory
    '''
    file = os.path.join(data_recording_path, f"group {group} sample {sample_idx}.pickle")
    with open(file, 'rb') as handle:
        data = pickle.load(handle)
    
    return data

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

def get_states(data, model, device):
    """Return point clouds, box poses, eef poses and cone pose of the trajectory"""

    pcds = data["pcds"]
    obj_embs = torch.ones([len(pcds), model.embedding_size], dtype=torch.float64, device=device)
    
    processed_pcds = []
    max_num_points = max([len(pcd) for pcd in pcds])
    #print(f"max num points in pointcloud: {max_num_points}")
    for i, pcd in enumerate(pcds):
        processed_pcd = np.zeros((max_num_points, 3))
        pad_point = pcd[-1, :]
        processed_pcd[:len(pcd), :] = pcd
        processed_pcd[len(pcd):, :] = np.expand_dims(pad_point, axis=0)
        processed_pcds.append(np.expand_dims(processed_pcd, axis=0))

    pcds = np.concatenate(processed_pcds, axis=0)
    pcds = np.array(farthest_point_sample_batched(pcds, npoint=256))
    obj_embs = to_obj_emb_batched(model, device, pcds).float()

    eef_poses = []
    for i, eef_state in enumerate(data["eef_states"]):
        eef_pose = list(eef_state["pose"]["p"])
        eef_pose = np.array([eef_pose[0][0], eef_pose[0][1], eef_pose[0][2]])
        eef_poses.append(eef_pose)
    eef_poses = np.array(eef_poses)
    
    eef_poses = torch.from_numpy(eef_poses).float().to(device)

    box_poses = []
    for i, box_state in enumerate(data["box_states"]):
        box_pose = np.array(list(box_state["pose"]["p"]))
        box_pose = [box_pose[0][0], box_pose[0][1], box_pose[0][2]]
        box_poses.append(box_pose)

    box_poses = np.array(box_poses)
    cone_pose = np.array(data["cone_pose"])
    # print("++++++++++++++++++", cone_pose)

    #print("++++++++++++++++ box_pose z:", box_poses[0:10])

    assert(len(eef_poses)==len(pcds)==len(box_poses))
   
    return obj_embs, eef_poses, box_poses, cone_pose


def sqr_dist(box_pose, cone_pose):
    return np.sum((box_pose[:2] - cone_pose[:2])**2)#(box_pose[0]-cone_pose[0])**2 + (box_pose[1]-cone_pose[1])**2

def sum_dist_to_goal(box_poses, cone_pose):
    dist = 0
    for box_pose in box_poses:
        dist += sqr_dist(box_pose, cone_pose)
    return dist


def compare_two_trajectories_w_reward(data_1, data_2, model, device):
    datas = [data_1, data_2]
    success_counts = []
    emb_states = []
    eef_states = []
    sum_dist = []
    rewards  = []

    for i, data in enumerate(datas):
        obj_embs, eef_poses, box_poses, cone_pose = get_states(data, model, device)
        emb_states.append(obj_embs)
        eef_states.append(eef_poses)

        success_counts.append(int(data["success_goal"]))
        sum_dist.append(sum_dist_to_goal(box_poses, cone_pose))
        rewards.append(success_counts[i]*10 - sum_dist[i])

    if success_counts[0] > success_counts[1]:
        label = False
    elif success_counts[0] < success_counts[1]:
        label = True
    else:        
        if sum_dist[0] > sum_dist[1]:
            label = True
        elif sum_dist[0] < sum_dist[1]:
            label = False   

    return emb_states, eef_states, label, rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Options')
    is_train = True
    suffix = "train" if is_train else "test"
    parser.add_argument('--data_recording_path', default=f"/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_push/demos_{suffix}", type=str, help="path to recorded data")
    parser.add_argument('--data_processed_path', default=f"/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_push/data_processed_{suffix}", type=str, help="path data to be processed")
    parser.add_argument('--AE_model_path', default="/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_AE_boxCone/weights/weights_1/epoch_150", type=str, help="path to pre-trained autoencoder weights")
    parser.add_argument('--num_group', default=30, type=int, help="num groups to process")
    parser.add_argument('--num_samples_per_group', default=30, type=int, help="num samples per group")
    parser.add_argument('--num_data_pt', default=14000, type=int, help="num datapoints to create and save")
    parser.add_argument('--rand_seed', default=2021, type=int, help="random seed")

    args = parser.parse_args()

    np.random.seed(args.rand_seed)

    data_recording_path = args.data_recording_path
    data_processed_path = args.data_processed_path
    AE_model_path = args.AE_model_path
    os.makedirs(data_processed_path, exist_ok=True)

    NUM_GROUP = args.num_group
    num_samples_per_group = args.num_samples_per_group
    num_data_pt = args.num_data_pt

    print(f"data processed path: {data_processed_path}")
    print(f"numdata pt : {num_data_pt}")

    get_gt_rewards = True

    start_time = timeit.default_timer() 

    device = torch.device("cuda")
    model = AutoEncoder(num_points=256, embedding_size=256).to(device)
    model.load_state_dict(torch.load(AE_model_path))
    model.eval()

    ######################## statistics #####################
    success_count = 0
    for group in range(NUM_GROUP):
        for sample in range(num_samples_per_group):
            data = get_trajectory(data_recording_path, group, sample)
            success_count += int(data["success_goal"])
    print(f"success rate of demos: {success_count/(NUM_GROUP*num_samples_per_group)}")

    for i in range(num_data_pt):

        if i % 50 == 0:
            print("========================================")
            print("current count:", i, " , time passed:", timeit.default_timer() - start_time)

        group_idx = np.random.randint(low=0, high=NUM_GROUP)

        idx_1 = 0
        idx_2 = 0
        while (idx_1 == idx_2):
            idx_1, data_1 = get_random_trajectory(data_recording_path, group_idx, num_samples_per_group)  
            idx_2, data_2 = get_random_trajectory(data_recording_path, group_idx, num_samples_per_group)

        if get_gt_rewards:
            emb_states, eef_states, label, gt_rewards = compare_two_trajectories_w_reward(data_1, data_2, model, device)
            processed_data = {"emb_traj_1": emb_states[0], "emb_traj_2": emb_states[1], \
                              "eef_traj_1": eef_states[0], "eef_traj_2": eef_states[1], "indices":[idx_1, idx_2],"group":group_idx, \
                              "gt_reward_1": gt_rewards[0], "gt_reward_2": gt_rewards[1], \
                              "label": label}
        else:
            emb_states, eef_states, label, _ = compare_two_trajectories_w_reward(data_1, data_2, model, device)    

            processed_data = {"emb_traj_1": emb_states[0], "emb_traj_2": emb_states[1], \
                              "eef_traj_1": eef_states[0], "eef_traj_2": eef_states[1], "indices":[idx_1, idx_2], "group":group_idx, \
                              "label": label}
                        
        
        with open(os.path.join(data_processed_path, "processed sample " + str(i) + ".pickle"), 'wb') as handle:
            pickle.dump(processed_data, handle, protocol=3)     




    