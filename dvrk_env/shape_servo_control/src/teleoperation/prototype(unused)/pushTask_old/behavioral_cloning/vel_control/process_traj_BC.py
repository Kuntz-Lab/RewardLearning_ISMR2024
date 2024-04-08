import sys
# import roslib.packages as rp
# pkg_path = rp.get_pkg_dir('shape_servo_control')
# sys.path.append(pkg_path + '/src')

#import open3d

#from util.isaac_utils import *

import torch
import math
import argparse
import os
import numpy as np
import pickle
import timeit
from random import sample
import open3d
#from utils import *

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
   
    return eef_poses, box_poses, cone_poses, dof_vels


if __name__ == "__main__":
    ### CHANGE ####
    is_train = True
    suffix = "train" if is_train else "test"

    data_recording_path = f"/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_push_BC_corrected/demos_{suffix}"
    data_processed_path = f"/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_push_BC_corrected/data_processed_{suffix}_1"
    os.makedirs(data_processed_path, exist_ok=True)
    num_samples_per_group = 1 #Change
    NUM_GROUP = 1 # Change
    data_idx = 0

    start_time = timeit.default_timer() 

    device = torch.device("cuda")

    for i in range(NUM_GROUP):
        if i % 50 == 0:
            print("========================================")
            print("current count:", i, " , time passed:", timeit.default_timer() - start_time)

        idx, data = get_trajectory(data_recording_path, i)  
        eef_poses, box_poses, cone_poses, dof_vels = get_states(data)
        print("box init pose: ", box_poses[0])
        print("cone init pose: ", cone_poses[0])
        assert(data["success_goal"]==True)

        for j in range(len(eef_poses)):
            eef_pose = torch.tensor(eef_poses[j], device=device).unsqueeze(0)
            box_pose = torch.tensor(box_poses[j], device=device).unsqueeze(0)
            cone_pose = torch.tensor(cone_poses[j], device=device).unsqueeze(0)
            dof_vel = torch.tensor(dof_vels[j], device=device).unsqueeze(0)
            pose = torch.cat((eef_pose, box_pose, cone_pose), dim=-1)
            
            processed_data = {"state": pose, "action": dof_vel}

            with open(os.path.join(data_processed_path, "processed sample " + str(data_idx) + ".pickle"), 'wb') as handle:
                pickle.dump(processed_data, handle, protocol=3)   

            data_idx += 1



    