import os
import numpy as np
import pickle
import timeit
from random import sample
import open3d
import torch

def open3d_ize(traj):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(traj[:,:3])  # only get eef_pose
    return pcd

def compute_perfect_trajectory(group, data_path, num_points=10):
    file = os.path.join(data_path, f"group {group} sample 0.pickle")
    with open(file, 'rb') as handle:
        data = pickle.load(handle)
    init_pose = np.array(list(data["traj"][0]["pose"]["p"]))
    mid_pose = np.array(data["mid pose"])
    goal_pose = np.array(data["goal pose"])

    segment_1 = np.linspace(start=init_pose, stop=mid_pose, num=num_points)
    segment_2 = np.linspace(start=mid_pose, stop=goal_pose, num=num_points)

    perfect_trajectory = np.concatenate((segment_1, segment_2), axis=0)
    # print("perfect_trajectory.shape:", perfect_trajectory.shape)

    return open3d_ize(perfect_trajectory)

def compute_deviation(traj, perfect_traj):
    return np.linalg.norm(np.asarray(perfect_traj.compute_point_cloud_distance(traj))) \
          + np.linalg.norm(np.asarray(traj.compute_point_cloud_distance(perfect_traj)))

def compute_predicted_reward(state, reward_net, device):

    state = torch.Tensor(state).unsqueeze(0).float().to(device) 
    reward = reward_net.cum_return(state)[0]
    return reward 

def test_reward_model(reward_net, object_poses, eef_pose):
    """
    object_poses: np.array shape (6,)
    eef_pose: np.array shape (3,)
    Return: reward of current state.
    """
    # reward_net: input torch tensor shape (1,9), output a scalar
    state = np.concatenate((eef_pose, object_poses), axis=None) # shape (9,)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    predicted_reward = compute_predicted_reward(state, reward_net, device)

    return predicted_reward