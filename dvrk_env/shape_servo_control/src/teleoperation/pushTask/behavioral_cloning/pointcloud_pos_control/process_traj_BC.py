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

def to_obj_emb_batched(model, device, pcds, visualize=False):
    pcd_tensor = torch.from_numpy(pcds.transpose(0,2,1)).float().to(device)
    emb = model(pcd_tensor, get_global_embedding=True)
    
    if visualize:
        points = pcds[0]
        print(points.shape)

        points = points[np.random.permutation(points.shape[0])]
    
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(np.array(points))  
        pcd.paint_uniform_color([0, 1, 0])

        points_tensor = torch.from_numpy(points.transpose(1,0)).unsqueeze(0).float().to(device)
        reconstructed_points = model(points_tensor)
        
        reconstructed_points = np.swapaxes(reconstructed_points.squeeze().cpu().detach().numpy(),0,1)
        reconstructed_points = reconstructed_points[:,:3]
        print(reconstructed_points.shape)

        pcd2 = open3d.geometry.PointCloud()
        pcd2.points = open3d.utility.Vector3dVector(np.array(reconstructed_points)) 
        pcd2.paint_uniform_color([1, 0, 0])
        open3d.visualization.draw_geometries([pcd, pcd2.translate((0,0,0.1))]) 
    
    return emb


def get_states(data, model, device):
    pcds = data["pcds"]
    obj_embs = torch.ones([len(pcds), model.embedding_size], dtype=torch.float64, device=device)
    
    processed_pcds = []
    max_num_points = max([len(pcd) for pcd in pcds])
    #print(f"max num points in pointcloud: {max_num_points}")
    for i, pcd in enumerate(pcds):
        processed_pcd = np.zeros((max_num_points, 3))
        pad_point = pcd[-1, :]
        processed_pcd[:len(pcd), :] = pcd
        processed_pcd[len(pcd):, :] = np.expand_dims(pad_point, axis=0)
        processed_pcds.append(np.expand_dims(processed_pcd, axis=0))

    pcds = np.concatenate(processed_pcds, axis=0)
    pcds = np.array(farthest_point_sample_batched(pcds, npoint=256))
    obj_embs = to_obj_emb_batched(model, device, pcds).float()

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
   
    return obj_embs, eef_poses, box_poses, cone_poses


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Options')
    is_train = True
    suffix = "train" if is_train else "test"
    parser.add_argument('--data_recording_path', default=f"/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_push/BC/demos_{suffix}", type=str, help="path to recorded data")
    parser.add_argument('--data_processed_path', default=f"/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_push/BC/data_processed_{suffix}", type=str, help="path data to be processed")
    parser.add_argument('--AE_model_path', default="/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_AE_boxCone/weights/weights_1/epoch_150", type=str, help="path to pre-trained autoencoder weights")
    parser.add_argument('--num_group', default=30, type=int, help="num groups to process")
    parser.add_argument('--rand_seed', default=2021, type=int, help="random seed")

    args = parser.parse_args()

    np.random.seed(args.rand_seed)

    data_recording_path = args.data_recording_path
    data_processed_path = args.data_processed_path
    AE_model_path = args.AE_model_path
    os.makedirs(data_processed_path, exist_ok=True)

    NUM_GROUP = args.num_group
    print(f"data processed path: {data_processed_path}")

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
        obj_embs, eef_poses, box_poses, cone_poses = get_states(data, model, device)
        assert(data["success_goal"]==True)

        offset = 10#20#4

        for j in range(0, len(eef_poses)-offset, 1):
            eef_pose = torch.tensor(eef_poses[j], device=device)
            obj_emb = obj_embs[j]
            state = torch.cat((eef_pose, obj_emb), dim=0) # size = 2 + 256
            action = torch.tensor(eef_poses[j+offset], device=device) # size = 2
            
            processed_data = {"state": state, "action": action}
            with open(os.path.join(data_processed_path, "processed sample " + str(data_idx) + ".pickle"), 'wb') as handle:
                pickle.dump(processed_data, handle, protocol=3)   

            data_idx += 1



    