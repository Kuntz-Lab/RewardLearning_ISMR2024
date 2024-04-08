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
import pickle
import timeit
# import open3d
from curve import *

from geometry_msgs.msg import PoseStamped, Pose

from util.isaac_utils import *
from util.grasp_utils import GraspClient

from core import Robot
from behaviors import MoveToPose, TaskVelocityControl
import argparse
from PIL import Image
import random



'''
collecting data of point clouds of balls arranged along different curves autonomously
'''

DIAMETER = 0.01

def sample_balls_poses():
    balls_poses = []

    balls_xyz = get_balls_xyz_around_workspace(LOW_X=-0.1, UP_X=0.1, LOW_Y=-0.65, UP_Y=-0.35)

    for xyz in balls_xyz:
        ball_pose = gymapi.Transform()
        pose = [xyz[0], xyz[1], xyz[2]]
        ball_pose.p = gymapi.Vec3(*pose)
        balls_poses.append(ball_pose)
        
    return balls_xyz, balls_poses

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


    #################### assume all on the same x-y plane ######################
    balls_xyz, all_balls_poses = sample_balls_poses()

    asset_root = '/home/dvrk/catkin_ws/src/dvrk_env/shape_servo_control/src/teleoperation/random_stuff'
    ball_asset_file = "new_ball.urdf"
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = False
    asset_options.disable_gravity = True
    asset_options.thickness = 0.000001

    ball_asset = gym.load_asset(sim, asset_root, ball_asset_file, asset_options)   

    # set up the env grid
    num_envs = 1
    spacing = 0.0
    env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
    env_upper = gymapi.Vec3(spacing, spacing, spacing)
    num_per_row = int(math.sqrt(num_envs))
  
    # cache some common handles for later use
    envs = []
    ball_handles = []

    for i in range(num_envs):
        # create env
        env = gym.create_env(sim, env_lower, env_upper, num_per_row)
        envs.append(env)

        # add ball obj       
        for j in range(len(balls_xyz)):     
            ball_actor = gym.create_actor(env, ball_asset, all_balls_poses[j])
            color = gymapi.Vec3(*[j+1/len(balls_xyz), 0, 0])
            gym.set_rigid_body_color(env, ball_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
            ball_handles.append(ball_actor)


    # Viewer camera setup
    if not args.headless:
        cam_target = gymapi.Vec3(0.0, -0.44, 0.1)
        cam_pos = gymapi.Vec3(0.0, -0.54, 0.9)

        middle_env = envs[num_envs // 2 + num_per_row // 2]
        gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)

    
    # Camera for point cloud setup
    cam_handles = []
    cam_width = 600
    cam_height = 600
    cam_targets = gymapi.Vec3(0.0, -0.4, 0.05)
    cam_positions = gymapi.Vec3(0.2, -0.7, 0.2)
    # cam_targets = gymapi.Vec3(0.0, -0.44, 0.1)
    # cam_positions = gymapi.Vec3(0, -0.7, 0.5)


    cam_handle, cam_prop = setup_cam(gym, envs[0], cam_width, cam_height, cam_positions, cam_targets)
    for i, env in enumerate(envs):
        cam_handles.append(cam_handle)

    if not args.headless:
        return envs, sim, ball_handles, cam_handles, cam_prop, viewer, balls_xyz, all_balls_poses
    else:
        return envs, sim, ball_handles, cam_handles, cam_prop, None, balls_xyz, all_balls_poses

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
    sample_count = data_config["sample_count"]

    print("++++++++++++++++++++ num_balls: ", len(sim_cache["balls_xyz"]))
    print("++++++++++++++++++++ sample_count: ", sample_count, "/", data_config["max_sample_count"])    

    if record_pc:
        pcd = get_partial_point_cloud(gym, sim, envs[0], cam_handles[0], cam_prop, color=[1,0,0], min_z = 0.05, visualization=False)
        data_config["partial_pc"] = np.array(pcd)

    print("pc shape: ", np.array(data_config["partial_pc"]).shape)

    vis=True
    if vis:
            pcd = open3d.geometry.PointCloud()
            pcd.points = open3d.utility.Vector3dVector(data_config["partial_pc"])
            open3d.visualization.draw_geometries([pcd])         

    if args.save_data and len(pcd)!=0:
        data = {
                "partial_pc": data_config["partial_pc"], "balls_xyz":sim_cache["balls_xyz"]
                }

        with open(os.path.join(data_recording_path, f"sample {sample_count}.pickle"), 'wb') as handle:
            pickle.dump(data, handle, protocol=3)   

        data_config["sample_count"] += 1 

    state = "record state"
    return state
                

if __name__ == "__main__":
     # initialize gym
    gym = gymapi.acquire_gym()

    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--headless', default="False", type=str, help="Index of grasp candidate to test")
    parser.add_argument('--save_data', default="False", type=str, help="True: save recorded data to pickles files")

    is_train = True
    suffix = "train" if is_train else "test"
    parser.add_argument('--data_recording_path', default=f"/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_AE_balls/demos_{suffix}_straight", type=str, help="where you want to record data")

    args = parser.parse_args()
    args.headless = args.headless == "True"
    args.save_data = args.save_data == "True"
    data_recording_path = args.data_recording_path
    #os.makedirs(data_recording_path, exist_ok=True)


    envs, sim, ball_handles, cam_handles, cam_prop, viewer, balls_xyz, all_balls_poses = configure_isaacgym(gym, args)
   
    '''
    Main simulation stuff starts from here
    '''
    rospy.init_node('shape_servo_control')
    rospy.logerr(f"Save data: {args.save_data}")


    state = "record state"

    sim_cache = {"gym":gym, "sim":sim, "envs":envs, \
                "ball_handles":ball_handles, "cam_handles":cam_handles, "cam_prop": cam_prop, "viewer":viewer,\
               "balls_xyz":balls_xyz, "all_balls_poses": all_balls_poses}

    data_config = {"sample_count":0, "max_sample_count":50000, "data_recording_path": data_recording_path, "partial_pc":[]}


    start_time = timeit.default_timer()   
    vis_count=0

    while (True): 
        step_physics(sim_cache)

        if state == "record state":
            state = record_state(sim_cache, data_config, args)
        if data_config["sample_count"] >= data_config["max_sample_count"]:
            break
        step_rendering(sim_cache, args)


    print("All done !")
    print("Elapsed time", timeit.default_timer() - start_time)
    if not args.headless:
        gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)











