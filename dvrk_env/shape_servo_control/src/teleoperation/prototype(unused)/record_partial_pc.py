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

from geometry_msgs.msg import PoseStamped, Pose

from util.isaac_utils import *
from util.grasp_utils import GraspClient

from core import Robot
from behaviors import MoveToPose, TaskVelocityControl
import argparse
from PIL import Image
import random

ROBOT_Z_OFFSET = 0.25




if __name__ == "__main__":

    # initialize gym
    gym = gymapi.acquire_gym()

    # parse arguments
    # args = gymutil.parse_arguments(
    #     description="dvrk",
    #     custom_parameters=[
    #         {"name": "--obj_name", "type": str, "default": 'box_0', "help": "select variations of a primitive shape"},
    #         {"name": "--headless", "type": bool, "default": False, "help": "headless mode"}])
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--obj_name', default='box_0', type=str, help="select variations of a primitive shape")
    parser.add_argument('--headless', default="False", type=str, help="Index of grasp candidate to test")
    parser.add_argument('--data_recording_path', default="/home/baothach/shape_servo_data/generalization/multi_boxes_1000Pa/data", type=str, help="path to save the recorded data")
    parser.add_argument('--object_meshes_path', default="/home/baothach/sim_data/Custom/Custom_mesh/multi_boxes_1000Pa", type=str, help="path to the objects' tet meshe files")
    parser.add_argument('--max_data_point_count', default=30000, type=int, help="path to the objects' tet meshe files")
    parser.add_argument('--save_data', default="False", type=str, help="True: save recorded data to pickles files")
    args = parser.parse_args()
    args.headless = args.headless == "True"
    args.save_data = args.save_data == "True"
    

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


    # load robot asset
    dvrk_asset = default_dvrk_asset(gym, sim)
    dvrk_pose = gymapi.Transform()
    dvrk_pose.p = gymapi.Vec3(0.0, 0.0, ROBOT_Z_OFFSET)
    dvrk_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)    


    # Load ball object
    init_pose = [0.0, -0.44, 0.16+0.015]
    asset_root = "/home/dvrk/Documents/IsaacGym_Preview_3_Package/isaacgym/assets/urdf"
    soft_asset_file = "new_ball.urdf"    
    soft_pose = gymapi.Transform()

    # pose = list(np.array(init_pose) + np.random.uniform(low=[-0.1,-0.03,-0.05], high=[0.1,0.02,0.05], size=3))
    # pose = list(np.array(init_pose) + np.array([0.1,-0.03,-0.05]))

    # soft_pose.p = gymapi.Vec3(*pose)
    # soft_pose.p = gymapi.Vec3(-0.08, -0.47, 0.16+0.015)
    soft_pose.p = gymapi.Vec3(0.05, -0.44, 0.16+0.015-0.02)

    soft_thickness = 0.0005    # important to add some thickness to the soft body to avoid interpenetrations

    goal_pose = gymapi.Transform()
    # goal_pose.p = gymapi.Vec3(0.0, -0.13-0.38686955, 0.2507219)    
    goal_pose.p = gymapi.Vec3(0.0, -0.44, 0.16+0.015)
    #gymapi.Vec3(*list(np.array(init_pose) + np.random.uniform(low=[-0.1,-0.03,-0.05], high=[0.1,0.02,0.05], size=3))) 


    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.thickness = soft_thickness
    asset_options.disable_gravity = True

    soft_asset = gym.load_asset(sim, asset_root, soft_asset_file, asset_options)       

    # wall_asset_root = "/home/baothach/dvrk_ws/src/dvrk_env/shape_servo_control/src/teleoperation/random_stuff"
    # wall_asset_file = "wall.urdf"  
    # wall_asset = gym.load_asset(sim, wall_asset_root, wall_asset_file, asset_options)  


    # set up the env grid
    num_envs = 1
    spacing = 0.0
    env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
    env_upper = gymapi.Vec3(spacing, spacing, spacing)
    num_per_row = int(math.sqrt(num_envs))
  

    # cache some common handles for later use
    envs = []
    dvrk_handles = []
    object_handles = []


    for i in range(num_envs):
        # create env
        env = gym.create_env(sim, env_lower, env_upper, num_per_row)
        envs.append(env)

        # # add dvrk
        # dvrk_handle = gym.create_actor(env, dvrk_asset, dvrk_pose, "dvrk", i, 1, segmentationId=11)    
        # dvrk_handles.append(dvrk_handle)    
        

        # add soft obj            
        # ball_actor = gym.create_actor(env, soft_asset, soft_pose, "soft", i, 0)
        ball_actor = gym.create_actor(env, soft_asset, soft_pose, "soft", i+1, 1)
        color = gymapi.Vec3(0,1,0)
        gym.set_rigid_body_color(env, ball_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
        object_handles.append(ball_actor)

        marker_env = gym.create_env(sim, env_lower, env_upper, num_per_row)
        goal_actor = gym.create_actor(marker_env, soft_asset, goal_pose, "soft", i+1, 1)
        color = gymapi.Vec3(1,0,0)
        gym.set_rigid_body_color(marker_env, goal_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
        # object_handles.append(goal_actor)

        # wall_actor = gym.create_actor(env, wall_asset, soft_pose, "wall", i+1, 1)
        # color = gymapi.Vec3(0,1,0)
        # gym.set_rigid_body_color(env, wall_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)





    
    # Camera for point cloud setup
    cam_handles = []
    cam_width = 300 #256
    cam_height = 300 #256 
    # cam_positions = gymapi.Vec3(0.17, -0.62, 0.2)
    # cam_targets = gymapi.Vec3(0.0, 0.40-0.86, 0.01)
    # low=[-0.1,-0.03,-0.05], high=[0.1,0.02,0.05]
    cam_positions = gymapi.Vec3(0.2, -0.54, 0.16+0.015 + 0.1)
    cam_targets = gymapi.Vec3(0.0, -0.44, 0.16+0.015)

    cam_handle, cam_prop = setup_cam(gym, envs[0], cam_width, cam_height, cam_positions, cam_targets)
    for i, env in enumerate(envs):
        cam_handles.append(cam_handle)

    # Viewer camera setup
    if not args.headless:
        # cam_pos = gymapi.Vec3(0.4, -0.7, 0.4)
        # cam_target = gymapi.Vec3(0.0, -0.36, 0.1)

        cam_pos = cam_positions
        cam_target = cam_targets

        # cam_pos = gymapi.Vec3(0.0, -0.440001, 1)
        # cam_target = gymapi.Vec3(0.0, -0.44, 0.1)

        middle_env = envs[num_envs // 2 + num_per_row // 2]
        gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)
       

    '''
    Main simulation stuff starts from here
    '''
    rospy.init_node('shape_servo_control')
    rospy.logerr(f"Save data: {args.save_data}")

    # Some important paramters


    all_done = False
    state = "record"
    frame_count = 0
    
    # data_recording_path = "/home/baothach"
    data_recording_path = "/home/dvrk/LfD_data/random"
    
    data_point_count = len(os.listdir(data_recording_path))
    max_data_point_count = data_point_count+1 #50000

    
    start_time = timeit.default_timer()   
    close_viewer = False



    while (not close_viewer) and (not all_done): 

        if not args.headless:
            close_viewer = gym.query_viewer_has_closed(viewer)  

        # step the physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)


        if state == "record" :   
            frame_count += 1
           
            
            if frame_count == 5:
                # rospy.loginfo("**Current state: " + state)
                frame_count = 0

                # Get current point cloud
                saved_partial_pc = get_partial_point_cloud(gym, sim, envs[0], cam_handle, cam_prop, color=[1,0,0], min_z = 0.1, visualization=False)
                rospy.logerr(f"pc shape: {saved_partial_pc.shape}")

                if args.save_data:

                    data = {"gt_pose": [soft_pose.p.x, soft_pose.p.y, soft_pose.p.z], \
                            "partial_pc": down_sampling(saved_partial_pc, num_pts=256)}   

                    with open(os.path.join(data_recording_path, f"sample {data_point_count}.pickle"), 'wb') as handle:
                        pickle.dump(data, handle, protocol=3)   
            

                data_point_count += 1   

                
                state = "reset"
                    
            
        if state == "reset":   
            # rospy.logwarn("==== RESETTING ====")

            ball_state = gym.get_actor_rigid_body_states(envs[i], object_handles[0], gymapi.STATE_POS)   
            
            
            new_pose = np.array(init_pose) + np.random.uniform(low=[-0.1,-0.03,-0.05], high=[0.1,0.02,0.05], size=3)
            soft_pose.p.x, soft_pose.p.y, soft_pose.p.z = new_pose[0], new_pose[1], new_pose[2] 
            ball_state['pose']['p']['x'] = new_pose[0]    
            ball_state['pose']['p']['y'] = new_pose[1]
            ball_state['pose']['p']['z'] = new_pose[2]         


            gym.set_actor_rigid_body_states(envs[i], object_handles[0], ball_state, gymapi.STATE_ALL)         


            rospy.logwarn("Succesfully reset ball")        
            state = "record"

            
        if data_point_count >= max_data_point_count:
            all_done = True



        # step rendering
        gym.step_graphics(sim)
        if not args.headless:
            gym.draw_viewer(viewer, sim, False)
            # gym.sync_frame_time(sim)


  
   



    print("All done !")
    print("Elapsed time", timeit.default_timer() - start_time)
    if not args.headless:
        gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

