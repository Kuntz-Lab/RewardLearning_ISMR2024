


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
    parser.add_argument('--vis_data_path', type=str, help="path to data for visualization (pickle file of specific group and sample)")

    args = parser.parse_args()
    AE_model_path = args.AE_model_path
    reward_model_path = args.reward_model_path
    vis_data_path = args.vis_data_path

    with open(vis_data_path, 'rb') as handle:
        data = pickle.load(handle)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    reward_net = RewardNet().to(device)
    reward_net.load_state_dict(torch.load(reward_model_path))
    reward_net.eval()
    encoder =  AutoEncoder(num_points=256, embedding_size=256).to(device)
    encoder.load_state_dict(torch.load(AE_model_path))

    partial_pcs = data["pcds"]
    print("num_balls_total: ", len(data["balls_xyz"]))
    print("reach cps: ", data["reach_cps"])
    print("num balls reached: ", data["num_balls_reached"])
    print("len traj: ", len(partial_pcs))

    # for pc in partial_pcs:
    #     pcd = open3d.geometry.PointCloud()
    #     pcd.points = open3d.utility.Vector3dVector(pc)
    #     open3d.visualization.draw_geometries([pcd])    
    #     break

    to_obj_emb(encoder, device, partial_pcs[0], visualize=False)

    balls_xyzs_list = []
    for i, balls_xyz in enumerate(data["balls_xyzs_list"]):
        arr = []
        for ball_xyz in balls_xyz:
            if abs(ball_xyz[0])<100 and abs(ball_xyz[1])<100 and abs( ball_xyz[2])<100:
                arr.append(ball_xyz)
        balls_xyzs_list.append(arr)

    frame=0#36#20#10#28#4#70
    balls_xyz = balls_xyzs_list[frame]
    partial_pc = partial_pcs[frame]
    print(partial_pc.shape)


    ############################### 3D with function plot (assuming all eef poses have the same z-value) ##########################################
    ####################################### for sampling different eef poses #################################
    rewards = []
    num_samples = 1000 #1000
    eef_poses = []

    z = balls_xyz[0][2]#+0.02#data["eef_ground_z_offset"]+0.0005
    for sample in range(num_samples):
        print(sample)
        x = np.random.uniform(low=-0.1, high=0.1)
        y = np.random.uniform(low=-0.5, high=-0.3)
        # x = np.random.uniform(low=-0.025, high=0.025)
        # y = np.random.uniform(low=-0.425, high=-0.375)
        eef_pose = np.array([x,y,z])
        eef_poses.append(eef_pose)
        # rew = 100*(compute_predicted_reward(partial_pc, eef_pose, encoder, reward_net, device)+62)
        rew = compute_predicted_reward(partial_pc, eef_pose, encoder, reward_net, device)
        # if np.abs(x-balls_xyz[0][0])<=0.01 and np.abs(y-balls_xyz[0][1])<=0.01:
        #     print(rew)
        rewards.append(rew)

    max_reward = max(rewards)
    print("max reward", max_reward)
    min_reward = min(rewards)
    print("min reward", min_reward)
    ##########################################################################################################


    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")

    for ball_xyz in balls_xyz:
        ax.plot(ball_xyz[0], ball_xyz[1], ball_xyz[2], "o", markersize=20)

    xs = [eef_poses[t][0] for t in range(num_samples)]
    ys = [eef_poses[t][1] for t in range(num_samples)]
    heats = [[(rewards[t] - min_reward) / (max_reward - min_reward), 0, 0] for t in range(num_samples)]
    zs = [(heats[t][0]) for t in range(num_samples)] # height of each point is the reward
    #zs = [(rewards[t]) for t in range(num_samples)] # height of each point is the reward
    ax.scatter(xs, ys, zs, c=heats) 

    plt.title(f"function of predicted rewards (heat) with different eef xy poses at frame {frame} (fixed z)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()

    ############################### 3D with function plot (assuming all eef poses have the same y-value) ##########################################
     ####################################### for sampling different eef poses #################################
    rewards = []
    num_samples = 1000
    eef_poses = []

    y = balls_xyz[0][1]
    for sample in range(num_samples):
        x = np.random.uniform(low=-0.1, high=0.1)
        z = np.random.uniform(low=0.03, high=0.2)
        eef_pose = np.array([x,y,z])
        eef_poses.append(eef_pose)
        rew = compute_predicted_reward(partial_pc, eef_pose, encoder, reward_net, device)
        rewards.append(rew)

    max_reward = max(rewards)
    min_reward = min(rewards)
    ##########################################################################################################



    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")

    for ball_xyz in balls_xyz:
        ax.plot(ball_xyz[0], ball_xyz[2], ball_xyz[1], "o", markersize=20)

    xs = [eef_poses[t][0] for t in range(num_samples)]
    ys = [eef_poses[t][2] for t in range(num_samples)]
    heats = [[(rewards[t] - min_reward) / (max_reward - min_reward), 0, 0] for t in range(num_samples)]
    zs = [(heats[t][0]) for t in range(num_samples)] # height of each point is the reward
    ax.scatter(xs, ys, zs, c=heats) 

    plt.title(f"function of predicted rewards (heat) with different eef xz poses at frame {frame} (fixed y)")
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("reward")
    plt.show()

    ############################### 3D with function plot (assuming all eef poses have the same x-value) ##########################################
     ####################################### for sampling different eef poses #################################
    rewards = []
    num_samples = 1000
    eef_poses = []

    x = balls_xyz[0][0]
    for sample in range(num_samples):
        y = np.random.uniform(low=-0.45, high=-0.35)
        z = np.random.uniform(low=0.03, high=0.2)
        eef_pose = np.array([x,y,z])
        eef_poses.append(eef_pose)
        rew = compute_predicted_reward(partial_pc, eef_pose, encoder, reward_net, device)
        rewards.append(rew)

    max_reward = max(rewards)
    min_reward = min(rewards)
    ##########################################################################################################


    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")

    for ball_xyz in balls_xyz:
        ax.plot(ball_xyz[1], ball_xyz[2], ball_xyz[0], "o", markersize=20)

    xs = [eef_poses[t][1] for t in range(num_samples)]
    ys = [eef_poses[t][2] for t in range(num_samples)]
    heats = [[(rewards[t] - min_reward) / (max_reward - min_reward), 0, 0] for t in range(num_samples)]
    zs = [(heats[t][0]) for t in range(num_samples)] # height of each point is the reward
    ax.scatter(xs, ys, zs, c=heats) 

    plt.title(f"function of predicted rewards (heat) with different eef yz poses at frame {frame} (fixed x)")
    ax.set_xlabel("y")
    ax.set_ylabel("z")
    ax.set_zlabel("reward")
    plt.show()








































####################################################### old version ##################################################
# import pickle

# #vis_data_path = "/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_trace_curve/demos_train_5balls/group 0 sample 47.pickle" #7
# # vis_data_path = "/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_trace_curve/demos_train_5balls_varied/group 0 sample 10.pickle" #7
# #vis_data_path = "/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_trace_curve/demos_train_5balls/group 0 sample 10.pickle"
# #vis_data_path = "/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_trace_curve/demos_train_1ball/group 0 sample 10.pickle" 
# vis_data_path = "/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_trace_curve_corrected/demos_train_straight_3D_flat_2ball_varied/group 2 sample 0.pickle"# group 2 sample 0 or 8# group 15 uneven peaks#group20 good sample5# group 26 sample 3 #0 #7
# #vis_data_path = "/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_trace_curve_corrected/ex_traj/group 0 sample 0.pickle"
# with open(vis_data_path, 'rb') as handle:
#     data = pickle.load(handle)

# import torch
# import os
# import numpy as np
# import timeit
# import roslib.packages as rp
# import sys
# sys.path.append("./pointcloud_representation_learning")
# pkg_path = rp.get_pkg_dir('shape_servo_control')
# sys.path.append(pkg_path + '/src')

# #import open3d

# from util.isaac_utils import *
# #from reward_model import RewardNetPointCloud as RewardNet
# from architecture import AutoEncoder
# #from utils import *
# from reward import RewardNetPointCloudEEF as RewardNet
# import matplotlib.pyplot as plt
# from curve import *

# def compute_predicted_reward(partial_pc, eef_pose, encoder, reward_net, device): 
#     partial_pc = np.array(down_sampling(partial_pc, num_pts=256))
#     traj = torch.Tensor(partial_pc.transpose(1,0)).unsqueeze(0).float().to(device) 
#     embedding = encoder(traj, get_global_embedding=True)
#     eef_pose = torch.Tensor(eef_pose).unsqueeze(0).float().to(device)
#     reward = reward_net.cum_return(eef_pose.unsqueeze(0), embedding.unsqueeze(0))[0][0]
#     # print(reward.cpu().detach().numpy())
#     return reward.item()


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# reward_net = RewardNet()
# reward_net.to(device)

# encoder =  AutoEncoder(num_points=256, embedding_size=256).to(device)
# # encoder.load_state_dict(torch.load("/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_AE_balls_corrected/weights_straight3D_partial_flat_spread/weights_2/epoch 150"))
# encoder.load_state_dict(torch.load("/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_AE_balls_corrected/weights_straight3D_partial_flat_2ball_varied/weights_1/epoch 150"))

# reward_model_path = '/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_trace_curve_corrected/weights_straight_3D_flat_2ball_varied_gaussian_bonus/weights_2'
# reward_net.load_state_dict(torch.load(os.path.join(reward_model_path, "epoch 200")))
# reward_net.eval()

# partial_pcs = data["pcds"]
# print("num_balls_total: ", len(data["balls_xyz"]))
# print("reach cps: ", data["reach_cps"])
# #print("num_balls: ", data["rand_num_balls"])
# print("num balls reached: ", data["num_balls_reached"])
# print("len traj: ", len(partial_pcs))

# # for pc in partial_pcs:
# #     pcd = open3d.geometry.PointCloud()
# #     pcd.points = open3d.utility.Vector3dVector(pc)
# #     open3d.visualization.draw_geometries([pcd])    


# balls_poses_list = []
# for i, balls_poses in enumerate(data["balls_poses_list"]):
#     arr = []
#     for ball_pose in balls_poses:
#         if abs(ball_pose.p.x)<100 and abs(ball_pose.p.y)<100 and abs( ball_pose.p.z)<100:
#             arr.append([ball_pose.p.x, ball_pose.p.y, ball_pose.p.z])
#             # if i==0:
#             #     print([ball_pose.p.x, ball_pose.p.y, ball_pose.p.z])
#     balls_poses_list.append(arr)

# frame=0#36#20#10#28#4#70
# balls_poses = balls_poses_list[frame]
# partial_pc = partial_pcs[frame]
# print(partial_pc.shape)#gaussian_bonus_1ball_2.png

# pcd = open3d.geometry.PointCloud()
# pcd.points = open3d.utility.Vector3dVector(partial_pc)
# open3d.visualization.draw_geometries([pcd])    

# ####################################### for sampling different eef poses #################################
# weights_list = data["weights_list"]
# rewards = []
# num_samples = 1000
# eef_poses = []

# #z=0.1
# for sample in range(num_samples):
#     x = np.random.uniform(low=-0.1, high=0.1)
#     y = np.random.uniform(low=-0.6, high=-0.4)
#     #eef_pose = np.random.uniform(low=[-0.11, -0.5, height], high=[-0.05, -0.38, height], size=3)
#     z = abs(poly3D(x, y, weights_list))
#     eef_pose = np.array([x,y,z])
#     eef_poses.append(eef_pose)
#     rew = compute_predicted_reward(partial_pc, eef_pose, encoder, reward_net, device)
#     rewards.append(rew)

# ##########################################################################################################

# max_reward = max(rewards)
# min_reward = min(rewards)

# ################################ 2D with heatmap ####################################
# ##plt.figure(figsize=(10,7))

# # for i in range(num_samples):
# #     heat = (rewards[i] - min_reward) / (max_reward - min_reward)
# #     plt.plot(eef_poses[i][0], eef_poses[i][1], '.', color=(heat, 0, 0), markersize=20) 

# # for ball_pose in balls_poses:
# #     plt.plot(ball_pose[0],ball_pose[1], "o", markersize=30)

# # plt.title(f"Heat map of predicted rewards with different eef xy poses at frame {frame}")
# # plt.xlabel("x", fontsize=10)
# # plt.ylabel("y", fontsize=10)
# # plt.xticks(fontsize=8)
# # plt.yticks(fontsize=8)

# # plt.show()

# ############################### 3D with function plot (assuming all eef poses have the same z-value ##########################################

# fig = plt.figure(figsize = (10, 7))
# ax = plt.axes(projection ="3d")

# for ball_pose in balls_poses:
#     ax.plot(ball_pose[0], ball_pose[1], ball_pose[2], "o", markersize=20)

# xs = [eef_poses[t][0] for t in range(num_samples)]
# ys = [eef_poses[t][1] for t in range(num_samples)]
# zs = [(rewards[t]) for t in range(num_samples)] # height of each point is the reward
# heats = [[(rewards[t] - min_reward) / (max_reward - min_reward), 0, 0] for t in range(num_samples)]
# ax.scatter(xs, ys, zs, c=heats) 

# plt.title(f"function of predicted rewards with different eef xy poses at frame {frame}")
# ax.set_xlabel("x")
# ax.set_ylabel("y")
# ax.set_zlabel("z")
# plt.show()


