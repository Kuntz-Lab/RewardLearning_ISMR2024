import sys
import roslib.packages as rp
pkg_path = rp.get_pkg_dir('shape_servo_control')
sys.path.append(pkg_path + '/src')
sys.path.append("../../pointcloud_representation_learning")

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

def get_trajectory(data_recording_path, group, idx=0):
    '''
    get a trajectory randomly from a group.
    Returns the sample_idx, data
    '''
    # idx = np.random.randint(low=0, high=num_samples_per_group)
    file = os.path.join(data_recording_path, f"group {group} sample {idx}.pickle")
    with open(file, 'rb') as handle:
        data = pickle.load(handle)
    
    return idx, data

def to_obj_emb(model, device, pcd):
    pcd_tensor = torch.from_numpy(pcd.transpose(1,0)).unsqueeze(0).float().to(device)
    emb = model(pcd_tensor, get_global_embedding=True)
    return emb


def get_states(data, model, device):
    pcds = data["pcds"]
    obj_embs = torch.ones([len(pcds), model.embedding_size], dtype=torch.float64, device=device)
    for i, pcd in enumerate(pcds):
        pcd = np.array(down_sampling(pcd, num_pts=256))
        emb = to_obj_emb(model, device, pcd)
        obj_embs[i] = emb

    eef_poses = []
    for i, eef_state in enumerate(data["eef_states"]):
        eef_pose = np.array(list(eef_state["pose"]["p"]))
        eef_poses.append(eef_pose)
    eef_poses = np.array(eef_poses)

    box_poses = []
    for i, box_state in enumerate(data["box_states"]):
        box_pose = np.array(list(box_state["pose"]["p"]))
        box_pose = [box_pose[0][0], box_pose[0][1], box_pose[0][2]]
        box_poses.append(box_pose)

    cone_pose = list(data["cone_pose"])
    cone_poses = [cone_pose for _ in range(len(box_poses))]

    dof_vels = list(data["dof_vels"])
    
    assert(len(cone_poses)==len(box_poses)==len(eef_poses)==len(dof_vels))
   
    return obj_embs, eef_poses, box_poses, cone_poses, dof_vels


if __name__ == "__main__":
    ### CHANGE ####
    is_train = False
    suffix = "train" if is_train else "test"

    data_recording_path = f"/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_push_BC_corrected/demos_{suffix}"
    data_processed_path = f"/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_push_BC_corrected/pointcloud_pos_control/data_processed_{suffix}"
    AE_model_path = "/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_AE_boxCone_corrected/weights/weights_40000/epoch 150"
    os.makedirs(data_processed_path, exist_ok=True)
    num_samples_per_group = 1 #Change
    NUM_GROUP = 50 # Change
    data_idx = 0

    start_time = timeit.default_timer() 

    device = torch.device("cuda")
    model = AutoEncoder(num_points=256, embedding_size=256).to(device)
    model.load_state_dict(torch.load(AE_model_path))
    model.eval()

    for i in range(NUM_GROUP):
        if i % 50 == 0:
            print("========================================")
            print("current count:", i, " , time passed:", timeit.default_timer() - start_time)

        idx, data = get_trajectory(data_recording_path, i)  
        obj_embs, eef_poses, box_poses, cone_poses, dof_vels = get_states(data, model, device)
        # print("box init pose: ", box_poses[0])
        # print("cone init pose: ", cone_poses[0])
        assert(data["success_goal"]==True)

        offset = 50#20#4

        for j in range(0, len(eef_poses)-offset, 1):
            eef_pose = torch.tensor(eef_poses[j], device=device)
            obj_emb = obj_embs[j]
            #dof_vel = torch.tensor(dof_vels[j], device=device).unsqueeze(0)
            state = torch.cat((eef_pose, obj_emb), dim=0)
            action = torch.tensor(eef_poses[j+offset], device=device)
            # print(state.shape)
            # print(action.shape)
            
            processed_data = {"state": state, "action": action}
            with open(os.path.join(data_processed_path, "processed sample " + str(data_idx) + ".pickle"), 'wb') as handle:
                pickle.dump(processed_data, handle, protocol=3)   

            data_idx += 1



    