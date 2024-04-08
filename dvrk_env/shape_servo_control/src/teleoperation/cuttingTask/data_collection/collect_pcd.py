#!/usr/bin/env python3
from __future__ import print_function, division, absolute_import
import sys
import roslib.packages as rp
pkg_path = rp.get_pkg_dir('shape_servo_control')
sys.path.append(pkg_path + '/src')
import os
import math
import numpy as np
from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym import gymutil
from copy import deepcopy
import rospy
import pickle5 as pickle
import timeit
# import open3d

from geometry_msgs.msg import PoseStamped, Pose

from util.isaac_utils import *
from util.grasp_utils import GraspClient

from core import Robot
import argparse
import random
import trimesh
import time

sys.path.append("../../pc_utils")
from get_isaac_partial_pc import get_partial_pointcloud_vectorized
from compute_partial_pc import farthest_point_sample_batched


import torch
import torchvision
from torchvision.utils import make_grid




'''
collecting partial point clouds of rigid, fixed tissues
'''

def grid_layout_images(images, num_columns=4, output_file=None, display_on_screen=False):

    """
    Display N images in a grid layout of size num_columns x np.ceil(N/num_columns) using pytorch.
    
    1.Input: 
    images: a list of torch tensor images, shape (3,H,W).
    
    """

    import cv2
    
    if not isinstance(images[0], torch.Tensor):
        # Convert the images to a PyTorch tensor
        torch_images = [torch.from_numpy(image).permute(2,0,1) for image in images]
        images = torch_images

    # num_images = len(images)   
    Grid = make_grid(images, nrow=num_columns, padding=0)
    
    # display result
    img = torchvision.transforms.ToPILImage()(Grid)

    if display_on_screen:
        # Display figure to screen
        cv2.imshow('Images', cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if output_file is not None:
        # Save the figure to the specified output file
        img.save(output_file)
    else:
        img_np = np.array(img)
        # Return the grid image as a NumPy array
        return img_np


def visualize_camera_views(gym, sim, envs, cam_handles, resolution=[600,600], output_file=None, num_columns=4):
    images = []
    gym.render_all_camera_sensors(sim)

    for i, cam_handle in enumerate(cam_handles):
        image = gym.get_camera_image(sim, envs[i], cam_handle, gymapi.IMAGE_COLOR).reshape((resolution[0],resolution[1],4))[:,:,:3]
        # print(image.shape)
        images.append(torch.from_numpy(image).permute(2,0,1) )

    grid_layout_images(images, num_columns, output_file=output_file, display_on_screen=True)



def configure_isaacgym(gym, args):
    # configure sim
    sim, sim_params = default_sim_config(gym, args)

    # add ground plane
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up ground
    gym.add_ground(sim, plane_params)

    # create viewer
    if not args.headless:
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        if viewer is None:
            print("*** Failed to create viewer")
            quit()

    rigid_urdf_path = os.path.join(args.rigid_state_path, "urdf")
    asset_root = rigid_urdf_path
    soft_asset_file = f"{args.object_name}.urdf"

    soft_pose =  gymapi.Transform()
    soft_pose.p = gymapi.Vec3(0,0,0)
    soft_thickness = 0

    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.disable_gravity = True
    asset_options.thickness = soft_thickness

    soft_asset = gym.load_asset(sim, asset_root, soft_asset_file, asset_options) 

    # set up the env grid
   
    num_envs = 1
    spacing = 1.0
    env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
    env_upper = gymapi.Vec3(spacing, spacing, spacing)
    num_per_row = int(math.sqrt(num_envs))
  
    # cache some common handles for later use
    envs = []
    soft_handles = []
    for i in range(num_envs):
        # create env
        env = gym.create_env(sim, env_lower, env_upper, num_per_row)
        envs.append(env)

        # add soft obj       
        soft_handle = gym.create_actor(env, soft_asset, soft_pose, f"soft", i+1, 0)
        soft_handles.append(soft_handle)


    # Viewer camera setup
    if not args.headless:
        cam_target = gymapi.Vec3(0.0, -0.4, 0.05)
        cam_pos = gymapi.Vec3(0.3, -0.8, 0.5)

        middle_env = envs[num_envs // 2 + num_per_row // 2]
        gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)

    
     # Camera for point cloud setup
    cam_handles = []
    cam_width = 300#400
    cam_height = 300#400
    cam_targets = gymapi.Vec3(0.0, -0.4, 0.01)
    cam_positions = gymapi.Vec3(0.0, -0.0, 0.1)

    
    for i, env in enumerate(envs):
        cam_handle, cam_prop = setup_cam(gym, envs[i], cam_width, cam_height, cam_positions, cam_targets)
        cam_handles.append(cam_handle)

    if not args.headless:
        return envs, sim, soft_handles, cam_handles, cam_prop, viewer
    else:
        return envs, sim, soft_handles, cam_handles, cam_prop, None


def step_physics(sim_cache):
    gym = sim_cache["gym"]
    sim = sim_cache["sim"]
    # viewer = sim_cache["viewer"]
    # if gym.query_viewer_has_closed(viewer):
    #     return True  
    gym.simulate(sim)
    gym.fetch_results(sim, True)

def step_rendering(sim_cache, args):
    gym = sim_cache["gym"]
    sim = sim_cache["sim"]
    viewer = sim_cache["viewer"]
    gym.step_graphics(sim)
    if not args.headless:
        gym.draw_viewer(viewer, sim, False)



def record_state(sim_cache, data_config, args, record_pc=True):
    state = "record state"
    rospy.loginfo("**Current state: " + state) 
    envs = sim_cache["envs"]
    gym = sim_cache["gym"]
     # for recording point clouds or images
    cam_handles = sim_cache["cam_handles"]
    cam_prop = sim_cache["cam_prop"]
    # for saving data
    data_recording_path = data_config["data_recording_path"]

    #visualize_camera_views(gym, sim, envs, cam_handles, resolution=[300,300], output_file=None, num_columns=4)

    vis=True
    if record_pc:
        gym.render_all_camera_sensors(sim)
        pcd = get_partial_pointcloud_vectorized(gym, sim, envs[0], cam_handles[0], cam_prop, color=[1,0,0], min_z = 0.005, visualization=vis, min_depth=-1)
        pcd = np.expand_dims(pcd, axis=0) # shape (1, n, d)
        down_sampled_pcd = farthest_point_sample_batched(point=pcd, npoint=256)
        down_sampled_pcd = np.squeeze(down_sampled_pcd, axis=0)
        data_config["partial_pc"] = np.array(down_sampled_pcd)
        #data_config["partial_pc"] = np.array(pcd)
        print("pc shape: ", pcd.shape) 
        print("downsampled pc shape: ", np.array(data_config["partial_pc"]).shape) 
    

    if args.save_data and len(pcd)!=0:
        print(f"FINISHED COLLECTING POINT CLOUDS {args.group_count} ---- saving -----")
        data = {
                "partial_pc": data_config["partial_pc"]
                }

        # with open(os.path.join(data_recording_path, f"group {args.group_count}.pickle"), 'wb') as handle:
        #     pickle.dump(data, handle, protocol=3)   
        with open(os.path.join(data_recording_path, f"processed sample {args.group_count}.pickle"), 'wb') as handle:
            pickle.dump(data, handle, protocol=3)   

    state = "reset"
    return state
                

def reset_state(sim_cache, data_config):
    sim = sim_cache["sim"]
    envs = sim_cache["envs"]
    gym = sim_cache["gym"]

    rospy.logwarn("==== RESETTING ====")
    data_config["group_count"] += 1
   
    state = "record state"

    return state

if __name__ == "__main__":
     # initialize gym
    gym = gymapi.acquire_gym()
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--data_recording_path', type=str, help="where you want to record data")
    parser.add_argument('--rigid_state_path', type=str, help="root path of the rigid state")
    parser.add_argument('--group_count', default=0, type=int, help="the current group the data is collecting for")
    parser.add_argument('--headless', default="False", type=str, help="Index of grasp candidate to test")
    parser.add_argument('--save_data', default="False", type=str, help="True: save recorded data to pickles files")
    parser.add_argument('--rand_seed', default=2021, type=int, help="random seed")
    parser.add_argument('--object_name', type=str, help="name of the saved deformable object in rigid state")
    

    args = parser.parse_args()

    
    args.headless = args.headless == "True"
    args.save_data = args.save_data == "True"

    data_recording_path = args.data_recording_path
    os.makedirs(data_recording_path, exist_ok=True)

    rand_seed = args.rand_seed
    np.random.seed(rand_seed)
    random.seed(rand_seed)

    envs, sim, soft_handles, cam_handles, cam_prop, viewer = configure_isaacgym(gym, args)

    sim_cache = {"gym":gym, "sim":sim, "envs":envs, \
                "soft_handles":soft_handles, "cam_handles":cam_handles, "cam_prop": cam_prop, "viewer":viewer,\
                }
    
    data_config = {"group_count":0, "max_group_count":1, "data_recording_path": data_recording_path, "partial_pc":None}

    state = "record state"
    start_time = timeit.default_timer()   

    while (True): 
        step_physics(sim_cache)

        if state == "record state":
            state = record_state(sim_cache, data_config, args)
        if state == "reset":
            state = reset_state(sim_cache, data_config)
        if data_config["group_count"] >= data_config["max_group_count"]:
            break
        step_rendering(sim_cache, args)


    print("All done !")
    print("Elapsed time", timeit.default_timer() - start_time)
    if not args.headless:
        gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
