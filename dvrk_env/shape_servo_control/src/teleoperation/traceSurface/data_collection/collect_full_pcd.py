#!/usr/bin/env python3
from __future__ import print_function, division, absolute_import


import sys
import roslib.packages as rp
pkg_path = rp.get_pkg_dir('shape_servo_control')
sys.path.append(pkg_path + '/src')
import os
import math
import numpy as np
import random
import trimesh


from copy import deepcopy
import pickle

from util.isaac_utils import *
from curve import *
import argparse

# final_pc = []
# xy = np.random.uniform(low=[-0.05, -0.45], high=[0.05, -0.5])
# for i in range(num_balls):
#     pc = get_ball_pc(np.array([x, poly(x, -0.5, -1), 0.1]), num_pts_per_ball, radius)
#     final_pc.extend(pc)
#     x += 2*radius
# return np.array(final_pc)

DEGREE = 1
MIN_NUM_BALLS = 4
MAX_NUM_BALLS = 10

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--save_data', default="False", type=str, help="True: save recorded data to pickles files")

    is_train = True
    suffix = "train" if is_train else "test"
    parser.add_argument('--data_recording_path', default=f"/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_AE_balls/demos_{suffix}_straight3D_full", type=str, help="where you want to record data")

    args = parser.parse_args()
    args.save_data = args.save_data == "True"
    data_recording_path = args.data_recording_path
    os.makedirs(data_recording_path, exist_ok=True)

    print(f"num existing files: {len(os.listdir(data_recording_path))}")

    max_sample_count = 20000+len(os.listdir(data_recording_path))
    init_sample_count = len(os.listdir(data_recording_path))
    print(f"max_sample_count: {max_sample_count} init_sample_count: {init_sample_count} num_sample: {max_sample_count-init_sample_count}")
    for sample_count in range(init_sample_count, max_sample_count, 1):
        num_balls = random.randint(MIN_NUM_BALLS, MAX_NUM_BALLS)
        print("++++++++++++++++++++ num_balls: ", num_balls)
        print("++++++++++++++++++++ sample_count: ", sample_count, "/", max_sample_count)    
        ball_pc, poly3D_weights_list, xy_curve_weights, balls_xyz = get_ball_pcs_curve(num_balls, degree=DEGREE)
        print("pc shape: ", ball_pc.shape)
        #print(f"polynomial: {weights_list_to_string(poly3D_weights_list)}")

        vis=False
        if vis:
                pcd = open3d.geometry.PointCloud()
                pcd.points = open3d.utility.Vector3dVector(ball_pc)
                open3d.visualization.draw_geometries([pcd])         

        if args.save_data:
            data = {
                    "partial_pc": ball_pc, "poly_weights_list": poly3D_weights_list, "xy_curve_weights": xy_curve_weights, "balls_xyz": balls_xyz
                    }

            with open(os.path.join(data_recording_path, f"sample {sample_count}.pickle"), 'wb') as handle:
                pickle.dump(data, handle, protocol=3)

# with open(os.path.join(data_recording_path, f"sample {0}.pickle"), 'wb') as handle:
#     pickle.dump(data, handle, protocol=3)
#     print(f"writing {0}")   

