import sys
import roslib.packages as rp
pkg_path = rp.get_pkg_dir('shape_servo_control')
sys.path.append(pkg_path + '/src')
sys.path.append("./pointcloud_representation_learning")

#import open3d

from util.isaac_utils import *

import torch
import math
import argparse
import os
import numpy as np
import pickle
import timeit
from random import sample
import open3d
from architecture import AutoEncoder
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

def to_obj_emb(model, device, pcd):
    pcd_tensor = torch.from_numpy(pcd.transpose(1,0)).unsqueeze(0).float().to(device)
    emb = model(pcd_tensor, get_global_embedding=True)
    return emb

def get_states(data, model, device):
    """Return point clouds, box poses, eef poses and cone pose of the trajectory"""

    pcds = data["pcds"]
    obj_embs = torch.ones([len(pcds), model.embedding_size], dtype=torch.float64, device=device)
    for i, pcd in enumerate(pcds):
        pcd = np.array(down_sampling(pcd, num_pts=256))
        emb = to_obj_emb(model, device, pcd)
        obj_embs[i] = emb
    #print(obj_embs.shape)

    eef_poses = []
    for i, eef_state in enumerate(data["eef_states"]):
        eef_pose = np.array(list(eef_state["pose"]["p"]))
        eef_poses.append(eef_pose)
    eef_poses = np.array(eef_poses)
    eef_poses = torch.from_numpy(eef_poses).float().to(device)

    assert(len(eef_poses)==len(pcds))

    box_poses = []
    for i, box_state in enumerate(data["box_states"]):
        box_pose = np.array(list(box_state["pose"]["p"]))
        box_pose = [box_pose[0][0], box_pose[0][1], box_pose[0][2]]
        box_poses.append(box_pose)

    cone_pose = list(data["cone_pose"])
    # print("++++++++++++++++++", cone_pose)

    #print("++++++++++++++++ box_pose z:", box_poses[0:10])
   
    return obj_embs, eef_poses, box_poses, cone_pose


def sqr_dist(box_pose, cone_pose):
    return (box_pose[0]-cone_pose[0])**2 + (box_pose[1]-cone_pose[1])**2 + (box_pose[2]-cone_pose[2])**2

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
    delta_dist = []
    rewards  = []

    for i, data in enumerate(datas):
        obj_embs, eef_poses, box_poses, cone_pose = get_states(data, model, device)
        emb_states.append(obj_embs)
        eef_states.append(eef_poses)

        success_counts.append(int(data["success_goal"]))
        sum_dist.append(sum_dist_to_goal(box_poses, cone_pose))
        delta_dist.append(sqr_dist(box_poses[-1], cone_pose) - sqr_dist(box_poses[0], cone_pose))
        rewards.append(success_counts[i]*5 - delta_dist[i] - sum_dist[i])

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
    ### CHANGE ####
    is_train = False
    suffix = "train" if is_train else "test"

    data_recording_path = f"/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_push/demos_{suffix}"
    data_processed_path = f"/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_push/data_processed_{suffix}"
    os.makedirs(data_processed_path, exist_ok=True)
    AE_model_path = "/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_AE_boxConeCloser/weights/weights_1/epoch 140"
    num_samples_per_group = 10 #Change
    NUM_GROUP = 10 # Change
    num_data_pt = 4000 #Change
    get_gt_rewards = True #Change

    start_time = timeit.default_timer() 

    device = torch.device("cuda")
    model = AutoEncoder(num_points=256, embedding_size=256).to(device)
    model.load_state_dict(torch.load(AE_model_path))
    model.eval()


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
                              "eef_traj_1": eef_states[0], "eef_traj_2": eef_states[1], "indices":[idx_1, idx_2], \
                              "gt_reward_1": gt_rewards[0], "gt_reward_2": gt_rewards[1], \
                              "label": label}
        else:
            emb_states, eef_states, label, _ = compare_two_trajectories_w_reward(data_1, data_2, model, device)    

            processed_data = {"emb_traj_1": emb_states[0], "emb_traj_2": emb_states[1], \
                              "eef_traj_1": eef_states[0], "eef_traj_2": eef_states[1], "indices":[idx_1, idx_2], \
                              "label": label}
                        
        
        with open(os.path.join(data_processed_path, "processed sample " + str(i) + ".pickle"), 'wb') as handle:
            pickle.dump(processed_data, handle, protocol=3)     




    