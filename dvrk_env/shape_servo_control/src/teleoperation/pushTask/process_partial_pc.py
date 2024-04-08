
import sys
import os
import math
import numpy as np
import pickle
import timeit
import open3d
import argparse

sys.path.append("../pc_utils")
from compute_partial_pc import farthest_point_sample_batched
import random

'''
downsample each partial pointcloud to 256 points
'''
def process_and_save(data, data_processed_path, index):
    pcd = np.expand_dims(data["partial_pc"], axis=0) # shape (1, n, d)
    down_sampled_pcd = farthest_point_sample_batched(point=pcd, npoint=256)
    down_sampled_pcd = np.squeeze(down_sampled_pcd, axis=0)
    data["partial_pc"] = down_sampled_pcd
     
    with open(os.path.join(data_processed_path, "processed sample " + str(index) + ".pickle"), 'wb') as handle:
        pickle.dump(data, handle, protocol=3)   
    assert(down_sampled_pcd.shape[0]==256 and down_sampled_pcd.shape[1]==3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Options')
    is_train = True
    suffix = "train" if is_train else "test"
    parser.add_argument('--data_recording_path', default=f"/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_AE_boxCone/demos_{suffix}", type=str, help="where you want to record data")
    parser.add_argument('--data_processed_path', default=f"/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_AE_boxCone/processed_data_{suffix}", type=str, help="where you want to save processed data")
    parser.add_argument('--vis', default="False", type=str, help="if False: visualize processed data instead of saving processed data")

    args = parser.parse_args()

    data_recording_path = args.data_recording_path
    data_processed_path = args.data_processed_path
    vis = args.vis=="True"
    os.makedirs(data_processed_path, exist_ok=True)
    
    num_data_pt = len(os.listdir(data_recording_path))
    print("num_data_pt", num_data_pt)

    if not vis:
        ############################### process and save ######################################
        for i in range(num_data_pt):
            with open(os.path.join(data_recording_path, f"sample {i}.pickle"), 'rb') as handle:
                data = pickle.load(handle)
            process_and_save(data, data_processed_path, i)

            if i%1000==0:
                print(f"finished processing sample[{i}]")

        assert(len(os.listdir(data_processed_path))==len(os.listdir(data_recording_path)))
    else:
        ############################### visualize processed point clouds ############################
        for i in range(10):
            with open(os.path.join(data_processed_path, "processed sample " + str(i) + ".pickle"), 'rb') as handle:
                print(os.path.join(data_processed_path, "processed sample " + str(i) + ".pickle"))
                data = pickle.load(handle)   
            print(f"sample {i}\tshape:", data["partial_pc"].shape)  
            pcd = data["partial_pc"]
            pcd2 = open3d.geometry.PointCloud()
            pcd2.points = open3d.utility.Vector3dVector(np.array(pcd)) 
            pcd2.paint_uniform_color([1, 0, 0])
            open3d.visualization.draw_geometries([pcd2])  