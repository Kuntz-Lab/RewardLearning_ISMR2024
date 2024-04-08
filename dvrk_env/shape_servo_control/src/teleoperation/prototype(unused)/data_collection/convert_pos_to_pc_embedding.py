#!/usr/bin/env python3

import os
import numpy as np
import pickle
import timeit
import torch
import trimesh
import open3d
import sys
sys.path.append("/home/dvrk/shape_servo_DNN/teleoperation")
from architecture import AutoEncoder

def open3d_ize(pc):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(pc)  # only get eef_pose
    return pcd

# def transform_and_convert_mesh_to_pc(mesh, pose, num_pts):
#     T = trimesh.transformations.translation_matrix(pose)
#     mesh.apply_transform(T)
#     pc = trimesh.sample.sample_surface_even(mesh, count=num_pts)[0]
    
#     return pc

def get_pc_embedding(pc, partial=True):
    
    if partial:
        model = model_partial
    else:
        model = model_full 
    

    pc_tensor = torch.from_numpy(pc.transpose(1,0)).unsqueeze(0).float().to(device)
    pc_embedding =  model(pc_tensor, get_global_embedding=True)

    return pc_embedding.cpu().detach().numpy()


def pose_to_embedding(obj_pose, vis_pc=False):
    pose_1 = obj_pose[:3]
    
    radius = 0.02
    num_pts = 256

    mesh_mid = trimesh.creation.icosphere(radius=radius)

    pc_mid = transform_and_convert_mesh_to_pc(mesh_mid, pose_1, num_pts=num_pts)


    if vis_pc:
        pcd_mid = open3d_ize(pc_mid)
        open3d.visualization.draw_geometries([pcd_mid]) 

    emb_mid = get_pc_embedding(pc_mid)


    # print("embedding shape:", np.array(emb_mid)[0].shape)
    return np.array(emb_mid)[0]   # shape (256,)




def get_states(data, partial=True, vis_pc=False):
    """Return all states of the trajectory"""

    states = []

    for eef_state in data["traj"]:
        states.append(list(eef_state["pose"]["p"]))   # states: shape (traj_length, 3). 

    
    if partial:
        obj_embedding = get_pc_embedding(data["partial_pc"], partial=partial)
        if vis_pc:
            pcd_mid = open3d_ize(data["partial_pc"])
            open3d.visualization.draw_geometries([pcd_mid]) 

    else:
        obj_embedding = pose_to_embedding(obj_pose, vis_pc=vis_pc)

        

    return np.array(states), obj_embedding


start_time = timeit.default_timer() 
num_samples_per_group = 20000 #20
NUM_GROUP = 1
data_recording_path = "/home/dvrk/LfD_data/group_meeting/demos"
data_processed_path = "/home/dvrk/LfD_data/group_meeting/demos_w_embedding"
os.makedirs(data_processed_path, exist_ok=True)

device = torch.device("cuda")

model_partial = AutoEncoder(num_points=256, embedding_size=256).to(device)
weight_path_partial = "/home/dvrk/LfD_data/weights/autoencoder/single_ball/partial_pc/epoch 150"
model_partial.load_state_dict(torch.load(weight_path_partial))
model_partial.eval()  

model_full = AutoEncoder(num_points=256, embedding_size=256).to(device)
weight_path_full = "/home/dvrk/LfD_data/weights/autoencoder/single_ball/full_pc/epoch 150" 
model_full.load_state_dict(torch.load(weight_path_full))
model_full.eval()  

current_sample_count = len(data_recording_path)    

for group_idx in range(NUM_GROUP):    
    for sample_idx in range(0, num_samples_per_group): 
        file = os.path.join(data_recording_path, f"group {group_idx} sample {sample_idx}.pickle")
        with open(file, 'rb') as handle:
            data = pickle.load(handle)

        states, obj_embedding = get_states(data, partial=True, vis_pc=False)

        processed_data = {"states": states, "obj_embedding": obj_embedding, \
                        "traj": data["traj"], 
                        "partial_pc": data["partial_pc"]}  #"goal pose": data["goal pose"], \
                        

        with open(os.path.join(data_processed_path, f"group {group_idx} sample {sample_idx}.pickle"), 'wb') as handle:
            pickle.dump(processed_data, handle, protocol=3)     



        print("========================================")
        print("current group:", group_idx, " , time passed:", timeit.default_timer() - start_time)
