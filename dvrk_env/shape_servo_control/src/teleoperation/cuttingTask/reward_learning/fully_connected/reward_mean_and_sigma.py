


import pickle
from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym import gymutil
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import open3d

import sys
sys.path.append("../../pointcloud_representation_learning")
#sys.path.append("../../../pc_utils")
from architecture import AutoEncoder
#from compute_partial_pc import farthest_point_sample_batched
from reward import RewardNetPointCloudEEF as RewardNet
import argparse

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

def compute_predicted_reward(partial_pc, eef_pose, encoder, reward_net, device): 
    # partial_pc = np.expand_dims(partial_pc, axis=0)
    # partial_pc = np.squeeze(farthest_point_sample_batched(partial_pc, npoint=256), axis=0)
    traj = torch.Tensor(partial_pc.transpose(1,0)).unsqueeze(0).float().to(device) 
    embedding = encoder(traj, get_global_embedding=True)
    eef_pose = torch.Tensor(eef_pose).unsqueeze(0).float().to(device)
    reward = reward_net.cum_return(eef_pose.unsqueeze(0), embedding.unsqueeze(0))[0][0]
    # print(reward.cpu().detach().numpy())
    return reward.item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--reward_model_path', type=str, help="path to reward model weights")
    parser.add_argument('--AE_model_path', type=str, help="path to pre-trained autoencoder weights")
    parser.add_argument('--demo_data_root_path', type=str, help="path to demo to compute sample mean and standard deviation")
    parser.add_argument('--num_group', type=int, help="number of groups")
    #parser.add_argument('--num_sample_per_group', type=int, help="number samples per group")

    args = parser.parse_args()
    AE_model_path = args.AE_model_path
    reward_model_path = args.reward_model_path
    demo_data_root_path = args.demo_data_root_path
    num_group = args.num_group
    #num_sample_per_group = args.num_sample_per_group

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    reward_net = RewardNet().to(device)
    reward_net.load_state_dict(torch.load(reward_model_path))
    reward_net.eval()
    encoder =  AutoEncoder(num_points=256, embedding_size=256).to(device)
    encoder.load_state_dict(torch.load(AE_model_path))


    rewards= []
    for group_idx in range(num_group):
        if group_idx%10 == 0:
            print("curr group: ", group_idx)
        for sample_idx in range(1):

            with open(os.path.join(demo_data_root_path, f"group {group_idx} sample {sample_idx}.pickle"), 'rb') as handle:
                data = pickle.load(handle)

    
            partial_pcs = data["pcds"]
            print("num_balls_total: ", len(data["balls_xyz"]))
            print("reach cps: ", data["reach_cps"])
            print("num balls reached: ", data["num_balls_reached"])
            print("len traj: ", len(partial_pcs))


            balls_xyzs_list = []
            for i, balls_xyz in enumerate(data["balls_xyzs_list"]):
                arr = []
                for ball_xyz in balls_xyz:
                    if abs(ball_xyz[0])<100 and abs(ball_xyz[1])<100 and abs( ball_xyz[2])<100:
                        arr.append(ball_xyz)
                balls_xyzs_list.append(arr)

            frame=0
            balls_xyz = balls_xyzs_list[frame]
            partial_pc = partial_pcs[frame]

            eef_ground_z_offset = data["eef_ground_z_offset"]

            num_samples = 1000

            for sample in range(num_samples):
                # x = np.random.uniform(low=-0.1, high=0.1)
                # y = np.random.uniform(low=-0.45, high=-0.35)
                x = np.random.uniform(low=-0.025, high=0.025)
                y = np.random.uniform(low=-0.4, high=-0.375)
                z = np.random.uniform(low=eef_ground_z_offset, high=0.1)
                eef_pose = np.array([x,y,z])
                rew = compute_predicted_reward(partial_pc, eef_pose, encoder, reward_net, device)
                rewards.append(rew)

    rewards = np.array(rewards)
    max_reward = np.max(rewards)
    print("max reward", max_reward)
    min_reward = np.min(rewards)
    print("min reward", min_reward)
    mean_reward = np.sum(rewards)/len(rewards)
    print("mean reward", mean_reward)
    variance_reward = np.var(rewards)
    print("variance reward", variance_reward)
    std_reward = np.std(rewards)
    print("std reward", std_reward)

    print("after normalizing ...")
    normalized_max_reward = (max_reward - mean_reward)/std_reward
    print("normalized max reward", normalized_max_reward)
    normalized_min_reward = (min_reward - mean_reward)/std_reward
    print("normalized_min reward", normalized_min_reward)
    











