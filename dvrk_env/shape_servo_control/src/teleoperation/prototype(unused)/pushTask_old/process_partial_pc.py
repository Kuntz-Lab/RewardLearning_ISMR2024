
import sys
import roslib.packages as rp
pkg_path = rp.get_pkg_dir('shape_servo_control')
sys.path.append(pkg_path + '/src')

import os
import math
import numpy as np
import pickle
import timeit
#import open3d
import argparse

from util.isaac_utils import *
import random

def process_and_save(data, data_processed_path, index):
    pcd = data["partial_pc"]
    pcd = down_sampling(pcd, num_pts=256)
    data["partial_pc"] = pcd
     
    with open(os.path.join(data_processed_path, "processed sample " + str(index) + ".pickle"), 'wb') as handle:
        pickle.dump(data, handle, protocol=3)   
    print(f"sample {index}\tshape: {pcd.shape}")  


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Options')
    ### CHANGE ####
    is_train = True
    suffix = "train" if is_train else "test"
    # parser.add_argument('--data_recording_path', default= f"/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_AE_boxCone/demos_{suffix}" , type=str, help="location of existing raw data")
    # parser.add_argument('--data_processed_path', default= f"/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_AE_boxCone/processed_data_{suffix}", type=int, help="the path to save processed data")
    ### CHANGE ####
    #args = parser.parse_args()
    #data_processed_path = args.data_processed_path
    data_processed_path = f"/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_AE_boxCone_corrected/processed_data_{suffix}"
    os.makedirs(data_processed_path, exist_ok=True)
    #data_recording_path = args.data_recording_path
    data_recording_path = f"/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_AE_boxCone_corrected/demos_{suffix}"
    
    num_data_pt = len(os.listdir(data_recording_path))
    print("num_data_pt", num_data_pt)
    assert(num_data_pt==50000)

    ############################### process and save ######################################
    # for i in range(num_data_pt):
    #     #print("gh")
    #     with open(os.path.join(data_recording_path, f"sample {i}.pickle"), 'rb') as handle:
    #         data = pickle.load(handle)
    #     process_and_save(data, data_processed_path, i)

    # assert(len(os.listdir(data_processed_path))==len(os.listdir(data_recording_path)))

    ############################### visualize processed point clouds ############################
    for i in range(50000):
        with open(os.path.join(data_processed_path, "processed sample " + str(i) + ".pickle"), 'rb') as handle:
            print(os.path.join(data_processed_path, "processed sample " + str(i) + ".pickle"))
            data = pickle.load(handle)   
        print(f"sample {i}\tshape:", data["partial_pc"].shape)  
        pcd = data["partial_pc"]
        pcd2 = open3d.geometry.PointCloud()
        pcd2.points = open3d.utility.Vector3dVector(np.array(pcd)) 
        pcd2.paint_uniform_color([1, 0, 0])
        open3d.visualization.draw_geometries([pcd2])  