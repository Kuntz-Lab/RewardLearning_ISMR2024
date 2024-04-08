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

sys.path.append("../../pointcloud_representation_learning")
sys.path.append("../../../pc_utils")
from architecture import AutoEncoder
from compute_partial_pc import farthest_point_sample_batched


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


def get_states(data):
    # get the x-y positions of eef, box and cone
    eef_poses = []
    for i, eef_state in enumerate(data["eef_states"]):
        eef_pose = list(eef_state["pose"]["p"])
        eef_pose = np.array([eef_pose[0][0], eef_pose[0][1]])
        eef_poses.append(eef_pose)
    eef_poses = np.array(eef_poses)

    box_poses = []
    for i, box_state in enumerate(data["box_states"]):
        box_pose = np.array(list(box_state["pose"]["p"]))
        box_pose = [box_pose[0][0], box_pose[0][1]]
        box_poses.append(box_pose)

    cone_pose = list(data["cone_pose"])
    cone_pose = [cone_pose[0], cone_pose[1]]
    cone_poses = [cone_pose for _ in range(len(box_poses))]

    assert(len(cone_poses)==len(box_poses)==len(eef_poses))
   
    return eef_poses, box_poses, cone_poses


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Options')
    is_train = True
    suffix = "train" if is_train else "test"
    parser.add_argument('--data_recording_path', default=f"/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_push/BC/demos_{suffix}", type=str, help="path to recorded data")
    parser.add_argument('--data_processed_path', default=f"/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_push/BC/data_processed_{suffix}", type=str, help="path data to be processed")
    parser.add_argument('--num_group', default=30, type=int, help="num groups to process")
    parser.add_argument('--rand_seed', default=2021, type=int, help="random seed")

    args = parser.parse_args()

    np.random.seed(args.rand_seed)

    data_recording_path = args.data_recording_path
    data_processed_path = args.data_processed_path
    os.makedirs(data_processed_path, exist_ok=True)

    NUM_GROUP = args.num_group
    print(f"data processed path: {data_processed_path}")

    data_idx = 0

    start_time = timeit.default_timer() 

    device = torch.device("cuda")

    for i in range(NUM_GROUP):
        if i % 50 == 0:
            print("========================================")
            print("current count:", i, " , time passed:", timeit.default_timer() - start_time)

        idx, data = get_trajectory(data_recording_path, i)  
        eef_poses, box_poses, cone_poses = get_states(data)
        assert(data["success_goal"]==True)

        offset = 5

        for j in range(0, len(eef_poses)-offset, 1):
            eef_pose = torch.tensor(eef_poses[j], device=device)
            box_pose = torch.tensor(box_poses[j], device=device)
            cone_pose = torch.tensor(cone_poses[j], device=device)
            state = torch.cat((eef_pose, box_pose, cone_pose), dim=-1)
            action = torch.tensor(eef_poses[j+offset], device=device)
            
            processed_data = {"state": state, "action": action}
            with open(os.path.join(data_processed_path, "processed sample " + str(data_idx) + ".pickle"), 'wb') as handle:
                pickle.dump(processed_data, handle, protocol=3)   

            data_idx += 1



    