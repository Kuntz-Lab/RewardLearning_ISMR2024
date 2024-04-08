import sys
# import roslib.packages as rp
# pkg_path = rp.get_pkg_dir('shape_servo_control')
# sys.path.append(pkg_path + '/src')

import os
import numpy as np
import pickle
import open3d
import argparse

#from util.isaac_utils import *
sys.path.append("../../pc_utils")
from compute_partial_pc import farthest_point_sample_batched

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

    assert(down_sampled_pcd.shape[0]==256)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Options')
    is_train = True
    suffix = "train" if is_train else "test"
    parser.add_argument('--data_recording_path', default=f"/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_AE_balls_corrected/demos_{suffix}_straight_flat_2ball", type=str, help="where you want to record data")
    parser.add_argument('--data_processed_path', default=f"/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_AE_balls_corrected/processed_data_{suffix}_straight_flat_2ball", type=str, help="where you want to record data")
    parser.add_argument('--vis', default="False", type=str, help="if False: visualize processed data instead of saving processed data")

    args = parser.parse_args()

    data_recording_path = args.data_recording_path
    data_processed_path = args.data_processed_path
    vis = args.vis=="True"
    os.makedirs(data_processed_path, exist_ok=True)

    with open(os.path.join(data_recording_path, f"group {0} sample {0}.pickle"), 'rb') as handle:
        data = pickle.load(handle)
    
    max_num_balls = len(data["balls_xyz"])
    num_samples_per_group = 2**max_num_balls - 1
    num_groups = len(os.listdir(data_recording_path))//num_samples_per_group
    print("num_groups to process: ", num_groups)
    print("num_samples_per_group: ", num_samples_per_group)

    if not vis:
    # ############################### process and save ######################################
        processed_sample_idx = 0
        for group_idx in range(num_groups):
            for sample_idx in range(num_samples_per_group):
                with open(os.path.join(data_recording_path, f"group {group_idx} sample {sample_idx}.pickle"), 'rb') as handle:
                    data = pickle.load(handle)
                process_and_save(data, data_processed_path, processed_sample_idx)
                print(f"processed_sample_idx: {processed_sample_idx}")
                processed_sample_idx += 1
            if group_idx%1000==0:
                print(f"finished processing group[{group_idx}]")
    else:
        ############################### visualize part of the processed point clouds ############################
        for i in range(100):
            with open(os.path.join(data_processed_path, "processed sample " + str(i) + ".pickle"), 'rb') as handle:
                print(os.path.join(data_processed_path, "processed sample " + str(i) + ".pickle"))
                data = pickle.load(handle)   
            print(f"sample {i}\tshape:", data["partial_pc"].shape) 
            print(f"num_balls: ", len(data["balls_xyz"])) 
            pcd = data["partial_pc"]
            pcd2 = open3d.geometry.PointCloud()
            pcd2.points = open3d.utility.Vector3dVector(np.array(pcd)) 
            pcd2.paint_uniform_color([1, 0, 0])
            open3d.visualization.draw_geometries([pcd2])  


































##################################old version#################################################
# import sys
# import roslib.packages as rp
# pkg_path = rp.get_pkg_dir('shape_servo_control')
# sys.path.append(pkg_path + '/src')

# import os
# import math
# import numpy as np
# import pickle
# import timeit
# #import open3d
# import argparse

# from util.isaac_utils import *
# import random

# def process_and_save(data, data_processed_path, index):
#     pcd = data["partial_pc"]
#     pcd = down_sampling(pcd, num_pts=256)
#     data["partial_pc"] = pcd
     
#     with open(os.path.join(data_processed_path, "processed sample " + str(index) + ".pickle"), 'wb') as handle:
#         pickle.dump(data, handle, protocol=3)   
#     assert(pcd.shape[0]==256)


# if __name__ == "__main__":
#     ### CHANGE ####
#     is_train = True
#     suffix = "train" if is_train else "test"
#     data_processed_path = f"/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_AE_balls_corrected/processed_data_{suffix}_straight3D_partial_flat_2ball_varied"
#     os.makedirs(data_processed_path, exist_ok=True)
#     data_recording_path = f"/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_AE_balls_corrected/demos_{suffix}_straight3D_partial_flat_2ball_varied"
    
#     num_data_pt = len(os.listdir(data_recording_path))
#     print("num_data_pt", num_data_pt)
#     #assert(num_data_pt==50000)

#     # ############################### process and save ######################################
#     for i in range(0, num_data_pt, 1):
#         #print("gh")
#         with open(os.path.join(data_recording_path, f"sample {i}.pickle"), 'rb') as handle:
#             data = pickle.load(handle)
#         process_and_save(data, data_processed_path, i)
#         if i%1000==0:
#             print(f"finished processing samples[{i}]")

#     assert(len(os.listdir(data_processed_path))==len(os.listdir(data_recording_path)))

#     ############################### visualize processed point clouds ############################
#     # for i in range(100):
#     #     with open(os.path.join(data_processed_path, "processed sample " + str(i) + ".pickle"), 'rb') as handle:
#     #         print(os.path.join(data_processed_path, "processed sample " + str(i) + ".pickle"))
#     #         data = pickle.load(handle)   
#     #     print(f"sample {i}\tshape:", data["partial_pc"].shape) 
#     #     print(f"num_balls: ", len(data["balls_xyz"])) 
#     #     pcd = data["partial_pc"]
#     #     pcd2 = open3d.geometry.PointCloud()
#     #     pcd2.points = open3d.utility.Vector3dVector(np.array(pcd)) 
#     #     pcd2.paint_uniform_color([1, 0, 0])
#     #     open3d.visualization.draw_geometries([pcd2])  