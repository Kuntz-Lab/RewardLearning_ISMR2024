


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
sys.path.append("../../../pc_utils")
sys.path.append("../../config_utils")
from architecture import AutoEncoder
from compute_partial_pc import farthest_point_sample_batched
from curve import poly3D
from reward import RewardNetPointCloudEEF as RewardNet


def compute_predicted_reward(partial_pc, eef_pose, encoder, reward_net, device): 
    partial_pc = np.expand_dims(partial_pc, axis=0)
    partial_pc = np.squeeze(farthest_point_sample_batched(partial_pc, npoint=256), axis=0)
    traj = torch.Tensor(partial_pc.transpose(1,0)).unsqueeze(0).float().to(device) 
    embedding = encoder(traj, get_global_embedding=True)
    eef_pose = torch.Tensor(eef_pose).unsqueeze(0).float().to(device)
    reward = reward_net.cum_return(eef_pose.unsqueeze(0), embedding.unsqueeze(0))[0][0]
    # print(reward.cpu().detach().numpy())
    return reward.item()

if __name__ == "__main__":
    AE_model_path = "/home/dvrk/LfD_data/ex_AE_balls_corrected/weights_straight_flat_2ball/weights_1/epoch_150"
    reward_model_path = '/home/dvrk/LfD_data/ex_trace_curve_corrected/weights_straight_flat_2ball/weights_1/epoch_200'
    vis_data_path = "/home/dvrk/LfD_data/ex_trace_curve_corrected/demos_test_straight_flat_2ball/group 2 sample 0.pickle"# group 2 sample 0 or 8# group 15 uneven peaks#group20 good sample5# group 26 sample 3 #0 #7
    # vis_data_path = "/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_trace_curve_corrected/demos_train_straight_3D_flat_2ball_varied/group 2 sample 0.pickle"# group 2 sample 0 or 8# group 15 uneven peaks#group20 good sample5# group 26 sample 3 #0 #7
    #vis_data_path = "/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_trace_curve_corrected/ex_traj/group 0 sample 0.pickle"

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

    balls_poses_list = []
    for i, balls_poses in enumerate(data["balls_poses_list"]):
        arr = []
        for ball_pose in balls_poses:
            if abs(ball_pose.p.x)<100 and abs(ball_pose.p.y)<100 and abs( ball_pose.p.z)<100:
                arr.append([ball_pose.p.x, ball_pose.p.y, ball_pose.p.z])
        balls_poses_list.append(arr)

    frame=50#36#20#10#28#4#70
    balls_poses = balls_poses_list[frame]
    partial_pc = partial_pcs[frame]
    print(partial_pc.shape)#gaussian_bonus_1ball_2.png

    # pcd = open3d.geometry.PointCloud()
    # pcd.points = open3d.utility.Vector3dVector(partial_pc)
    # open3d.visualization.draw_geometries([pcd])    

    ####################################### for sampling different eef poses #################################
    weights_list = data["weights_list"]
    rewards = []
    num_samples = 1000
    eef_poses = []

    #z=0.1
    for sample in range(num_samples):
        x = np.random.uniform(low=-0.1, high=0.1)
        y = np.random.uniform(low=-0.6, high=-0.4)
        #eef_pose = np.random.uniform(low=[-0.11, -0.5, height], high=[-0.05, -0.38, height], size=3)
        z = abs(poly3D(x, y, weights_list))
        eef_pose = np.array([x,y,z])
        eef_poses.append(eef_pose)
        rew = compute_predicted_reward(partial_pc, eef_pose, encoder, reward_net, device)
        rewards.append(rew)

    ##########################################################################################################

    max_reward = max(rewards)
    min_reward = min(rewards)

    ################################ 2D with heatmap ####################################
    ##plt.figure(figsize=(10,7))

    # for i in range(num_samples):
    #     heat = (rewards[i] - min_reward) / (max_reward - min_reward)
    #     plt.plot(eef_poses[i][0], eef_poses[i][1], '.', color=(heat, 0, 0), markersize=20) 

    # for ball_pose in balls_poses:
    #     plt.plot(ball_pose[0],ball_pose[1], "o", markersize=30)

    # plt.title(f"Heat map of predicted rewards with different eef xy poses at frame {frame}")
    # plt.xlabel("x", fontsize=10)
    # plt.ylabel("y", fontsize=10)
    # plt.xticks(fontsize=8)
    # plt.yticks(fontsize=8)

    # plt.show()

    ############################### 3D with function plot (assuming all eef poses have the same z-value ##########################################

    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")

    colors = ['y', 'b']
    for i, ball_pose in enumerate(balls_poses):
        ax.plot(ball_pose[0], ball_pose[1], ball_pose[2], f"{colors[i]}o", markersize=20)

    xs = [eef_poses[t][0] for t in range(num_samples)]
    ys = [eef_poses[t][1] for t in range(num_samples)]
    heats = [[(rewards[t] - min_reward) / (max_reward - min_reward), 0, 0] for t in range(num_samples)]
    zs = [(rewards[t]) for t in range(num_samples)] # height of each point is the reward
    
    ax.scatter(xs, ys, zs, c=heats) 

    plt.title(f"function of predicted rewards with different eef xy poses at frame {frame}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
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


