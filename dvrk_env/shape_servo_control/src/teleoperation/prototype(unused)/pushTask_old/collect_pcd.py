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

from util.isaac_utils import *

import argparse


'''
collecting data of point clouds of box and cone autonomously
'''

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


    #################### assume box and cone on the x-y plane ######################
    # Load box object
    init_pose = [0.0, -0.0, 0.022]
    asset_root = "/home/dvrk/catkin_ws/src/dvrk_env/shape_servo_control/src/teleoperation/random_stuff"
    box_asset_file = "push_box.urdf"    
    box_pose = gymapi.Transform()
    pose = list(np.array(init_pose) + np.random.uniform(low=[-0.1,-0.6,0], high=[0.1,-0.4,0], size=3))
    box_pose.p = gymapi.Vec3(*pose)
    
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = False
    asset_options.disable_gravity = False
    asset_options.thickness = 0.00025

    box_asset = gym.load_asset(sim, asset_root, box_asset_file, asset_options)     

    # load cone object
    cone_asset_file = "cone.urdf"  
    cone_pose = gymapi.Transform()
    pose = list(np.array(init_pose) + np.random.uniform(low=[-0.1,-0.6,0], high=[0.1,-0.4,0], size=3))
    cone_pose.p = gymapi.Vec3(*pose)
    
    cone_asset = gym.load_asset(sim, asset_root, cone_asset_file, asset_options) 

    # set up the env grid
    num_envs = 1
    spacing = 0.0
    env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
    env_upper = gymapi.Vec3(spacing, spacing, spacing)
    num_per_row = int(math.sqrt(num_envs))
  
    # cache some common handles for later use
    envs = []
    object_handles = []

    for i in range(num_envs):
        # create env
        env = gym.create_env(sim, env_lower, env_upper, num_per_row)
        envs.append(env)

        # add box and cone (goal) obj            
        box_actor = gym.create_actor(env, box_asset, box_pose, "box", i, 0)
        color = gymapi.Vec3(0,1,0)
        gym.set_rigid_body_color(env, box_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
        object_handles.append(box_actor)

        cone_actor = gym.create_actor(env, cone_asset, cone_pose, "cone", i+1, 0)
        object_handles.append(cone_actor)


    # Viewer camera setup
    if not args.headless:
        cam_pos = gymapi.Vec3(0.5, -0.7, 0.4)
        cam_target = gymapi.Vec3(0.0, -0.4, 0.1)

        middle_env = envs[num_envs // 2 + num_per_row // 2]
        gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)

    
    # Camera for point cloud setup
    cam_handles = []
    cam_width = 300
    cam_height = 300
    # cam_positions = gymapi.Vec3(0.0, -0.4400001, 0.7)
    cam_targets = gymapi.Vec3(0.0, -0.44, 0.1)
    cam_positions = gymapi.Vec3(0.7, -0.54, 0.76)
    # cam_targets = gymapi.Vec3(0.0, -0.44, 0.16+0.015)


    cam_handle, cam_prop = setup_cam(gym, envs[0], cam_width, cam_height, cam_positions, cam_targets)
    for i, env in enumerate(envs):
        cam_handles.append(cam_handle)

    if not args.headless:
        return envs, sim, object_handles, cam_handles, cam_prop, viewer, box_pose, cone_pose, init_pose
    else:
        return envs, sim, object_handles, cam_handles, cam_prop, None, box_pose, cone_pose, init_pose

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

def record_state(sim_cache, data_config, args):
    state = "record state"
    rospy.loginfo("**Current state: " + state) 
    envs = sim_cache["envs"]
    gym = sim_cache["gym"]
     # for recording point clouds or images
    cam_handles = sim_cache["cam_handles"]
    cam_prop = sim_cache["cam_prop"]
    data_recording_path = data_config["data_recording_path"]
    sample_count = data_config["sample_count"]
    print("++++++++++++++++++++ sample_count: ", sample_count, "/", data_config["max_sample_count"])    

    pcd = get_partial_point_cloud(gym, sim, envs[0], cam_handles[0], cam_prop, color=[1,0,0], min_z = 0.009, visualization=False)
    box_state = deepcopy(gym.get_actor_rigid_body_states(envs[0], object_handles[0], gymapi.STATE_POS))
    cone_state = deepcopy(gym.get_actor_rigid_body_states(envs[0], object_handles[1], gymapi.STATE_POS))
    sim_cache["box_pose"].p.x  = box_state['pose']['p']['x']
    sim_cache["box_pose"].p.y  = box_state['pose']['p']['y'] 
    sim_cache["box_pose"].p.z  = box_state['pose']['p']['z'] 
    sim_cache["cone_pose"].p.x  = cone_state['pose']['p']['x']
    sim_cache["cone_pose"].p.y  = cone_state['pose']['p']['y'] 
    sim_cache["cone_pose"].p.z  = cone_state['pose']['p']['z'] 
    #data_config["partial_pc"].append(pcd)
    data_config["partial_pc"] = pcd
    data_config["partial_pc"] = np.array(data_config["partial_pc"])
    #data_config["partial_pc"] = np.transpose(data_config["partial_pc"], (0, 2, 1))
    print("pc shape: ", np.array(data_config["partial_pc"]).shape)
    print("box_pose: ", sim_cache["box_pose"].p.x, ",", sim_cache["box_pose"].p.y, ",", sim_cache["box_pose"].p.z)
    print("cone_pose: ", sim_cache["cone_pose"].p.x, ",", sim_cache["cone_pose"].p.y, ",", sim_cache["cone_pose"].p.z)
    vis=False
    if vis:
            pcd = open3d.geometry.PointCloud()
            pcd.points = open3d.utility.Vector3dVector(data_config["partial_pc"])
            open3d.visualization.draw_geometries([pcd])         

    if args.save_data:
        data = { "cone_pose":[sim_cache["cone_pose"].p.x, sim_cache["cone_pose"].p.y, sim_cache["cone_pose"].p.z],\
                "box_pose": [sim_cache["box_pose"].p.x, sim_cache["box_pose"].p.y, sim_cache["box_pose"].p.z], \
                "partial_pc": data_config["partial_pc"]
                }

        with open(os.path.join(data_recording_path, f"sample {sample_count}.pickle"), 'wb') as handle:
            pickle.dump(data, handle, protocol=3)   

        data_config["sample_count"] += 1 

    state = "reset"
    return state
                

def reset_state(sim_cache, data_congfig):
    envs = sim_cache["envs"]
    gym = sim_cache["gym"]
    object_handles = sim_cache["object_handles"]
    init_pose = sim_cache["init_pose"] 

    rospy.logwarn("==== RESETTING ====")

    box_state = gym.get_actor_rigid_body_states(envs[0], object_handles[0], gymapi.STATE_POS)
    new_pose = np.array(init_pose) + np.random.uniform(low=[-0.1,-0.6,0], high=[0.1,-0.4,0], size=3)
    box_state['pose']['p']['x'] = new_pose[0]    
    box_state['pose']['p']['y'] = new_pose[1]
    box_state['pose']['p']['z'] = new_pose[2]    
    sim_cache["box_pose"].p.x = new_pose[0] 
    sim_cache["box_pose"].p.y = new_pose[1] 
    sim_cache["box_pose"].p.z = new_pose[2] 
    sim_cache["init_box_pose"].p.x = new_pose[0] 
    sim_cache["init_box_pose"].p.y = new_pose[1] 
    sim_cache["init_box_pose"].p.z = new_pose[2]       
    gym.set_actor_rigid_body_states(envs[0], object_handles[0], box_state, gymapi.STATE_ALL)

    cone_state = gym.get_actor_rigid_body_states(envs[0], object_handles[1], gymapi.STATE_POS)
    new_pose = np.array(init_pose) + np.random.uniform(low=[-0.1,-0.6,0], high=[0.1,-0.4,0], size=3)
    cone_state['pose']['p']['x'] = new_pose[0]    
    cone_state['pose']['p']['y'] = new_pose[1]
    cone_state['pose']['p']['z'] = new_pose[2]    
    sim_cache["cone_pose"].p.x = new_pose[0] 
    sim_cache["cone_pose"].p.y = new_pose[1] 
    sim_cache["cone_pose"].p.z = new_pose[2]    
    gym.set_actor_rigid_body_states(envs[0], object_handles[1], cone_state, gymapi.STATE_ALL)  

    data_config["partial_pc"] = []
   
    state = "record state"

    return state

if __name__ == "__main__":
    # train
    np.random.seed(1000)

     # initialize gym
    gym = gymapi.acquire_gym()

    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--headless', default="False", type=str, help="Index of grasp candidate to test")
    parser.add_argument('--save_data', default="False", type=str, help="True: save recorded data to pickles files")

    is_train = True
    suffix = "train" if is_train else "test"
    parser.add_argument('--data_recording_path', default=f"/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_AE_boxCone_corrected/demos_{suffix}", type=str, help="where you want to record data")

    args = parser.parse_args()
    args.headless = args.headless == "True"
    args.save_data = args.save_data == "True"
    data_recording_path = args.data_recording_path
    os.makedirs(data_recording_path, exist_ok=True)


    envs, sim, object_handles, cam_handles, cam_prop, viewer, box_pose, cone_pose, init_pose = configure_isaacgym(gym, args)
   
    '''
    Main simulation stuff starts from here
    '''
    rospy.init_node('shape_servo_control')
    rospy.logerr(f"Save data: {args.save_data}")


    state = "record state"

    init_box_pose = gymapi.Transform()
    init_box_pose.p.x = deepcopy(box_pose.p.x)
    init_box_pose.p.y = deepcopy(box_pose.p.y)
    init_box_pose.p.z = deepcopy(box_pose.p.z)

    sim_cache = {"gym":gym, "sim":sim, "envs":envs, \
                "object_handles":object_handles, "cam_handles":cam_handles, "cam_prop": cam_prop, "viewer":viewer,\
                "box_pose": box_pose, "init_box_pose":init_box_pose, "cone_pose":cone_pose, "init_pose": init_pose}

    data_config = {"sample_count":0, "max_sample_count":50000, "data_recording_path": data_recording_path, "partial_pc":[]}


    start_time = timeit.default_timer()   
    vis_count=0
    step_physics(sim_cache)
    step_physics(sim_cache)
    while (True): 
        step_physics(sim_cache)

        if state == "record state":
            state = record_state(sim_cache, data_config, args)
        if state == "reset":
            state = reset_state(sim_cache, data_config)
        if data_config["sample_count"] >= data_config["max_sample_count"]:
            break
        step_rendering(sim_cache, args)


    print("All done !")
    print("Elapsed time", timeit.default_timer() - start_time)
    if not args.headless:
        gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)











