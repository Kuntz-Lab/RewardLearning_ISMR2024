import sys
#sys.path.append("/home/dvrk/catkin_ws/src/dvrk_env/shape_servo_control/src/teleoperation/traceSurface")
sys.path.append("../pointcloud_representation_learning")
sys.path.append("../../pc_utils")

import os
import numpy as np
import pickle
import timeit
import matplotlib.pyplot as plt
from compute_partial_pc import get_all_bin_seq_driver


from architecture import AutoEncoder

import torch
import open3d


def get_episode_data(data_recording_path, episode_idx):
    file = os.path.join(data_recording_path, f"episode {episode_idx}.pickle")
    with open(file, 'rb') as handle:
        data = pickle.load(handle)
    
    return data

def to_obj_emb(model, device, pcds):
    '''
    pcds has shape (num_batch, num_points, point_dim)
    '''
    pcd_tensor = torch.from_numpy(pcds.transpose(0,2,1)).float().to(device)
    with torch.no_grad():
        emb = model(pcd_tensor, get_global_embedding=True)
    return emb


if __name__ == "__main__":
    device = "cuda"
    encoder =  AutoEncoder(num_points=256, embedding_size=256).to(device)
    encoder.load_state_dict(torch.load("/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_AE_balls_corrected_old/weights_straight3D_partial_flat_2ball_varied/weights_1/epoch 150"))

    all_bin_seqs = get_all_bin_seq_driver(2)
    all_bin_seqs.remove(tuple([1 for i in range(2)]))

    root_folder = "/home/dvrk/RL_data/traceCurve/debug_2ball"
    reward_folder = "pc2"
    data_recording_path = f"{root_folder}/{reward_folder}/data"
            
    data1 = get_episode_data(data_recording_path, episode_idx=20)
    pc_dict = data1["pc_dict"]

    for bin_seq in all_bin_seqs:
        print(f"dropped mask: {bin_seq}")
        pcs = pc_dict[bin_seq]

        with torch.no_grad():
            points0 = pcs[0]
            points1 = pcs[1]
            print(points1.shape)

            points1 = points1[np.random.permutation(points1.shape[0])]
            points0 = points0[np.random.permutation(points0.shape[0])]
        
            pcd1 = open3d.geometry.PointCloud()
            pcd1.points = open3d.utility.Vector3dVector(np.array(points1))
            pcd1.paint_uniform_color([0, 0, 1])

            pcd0 = open3d.geometry.PointCloud()
            pcd0.points = open3d.utility.Vector3dVector(np.array(points0))
            pcd0.paint_uniform_color([0, 1, 0])

            points_tensor1 = torch.from_numpy(points1.transpose(1,0)).unsqueeze(0).float().to(device)
            reconstructed_points1 = encoder(points_tensor1)

            points_tensor0 = torch.from_numpy(points0.transpose(1,0)).unsqueeze(0).float().to(device)
            reconstructed_points0 = encoder(points_tensor0)
            
            reconstructed_points0 = np.swapaxes(reconstructed_points0.squeeze().cpu().detach().numpy(),0,1)
            reconstructed_points0 = reconstructed_points0[:,:3]
            print(reconstructed_points0.shape)

            reconstructed_points1 = np.swapaxes(reconstructed_points1.squeeze().cpu().detach().numpy(),0,1)
            reconstructed_points1 = reconstructed_points1[:,:3]
            print(reconstructed_points1.shape)

            pcd2 = open3d.geometry.PointCloud()
            pcd2.points = open3d.utility.Vector3dVector(np.array(reconstructed_points0))
            pcd2.paint_uniform_color([1, 0, 1])
            pcd3 = open3d.geometry.PointCloud()
            pcd3.points = open3d.utility.Vector3dVector(np.array(reconstructed_points1))
            pcd3.paint_uniform_color([1, 0, 0])
            open3d.visualization.draw_geometries([pcd0, pcd1, pcd2.translate((0,0,0.1)), pcd3.translate((0,0,0.1))])