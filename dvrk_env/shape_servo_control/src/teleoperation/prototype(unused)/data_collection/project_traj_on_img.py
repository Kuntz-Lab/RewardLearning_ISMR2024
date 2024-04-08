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

from geometry_msgs.msg import PoseStamped, Pose, TransformStamped
from sensor_msgs.msg import Joy
from util.isaac_utils import * #fix_object_frame, get_pykdl_client, down_sampling, get_partial_point_cloud
from util.grasp_utils import GraspClient

from core import Robot
from behaviors import TaskVelocityControl2
import argparse
from PIL import Image
import random
import transformations
import cv2


sys.path.append(os.path.join(rp.get_pkg_dir('shape_servo_control'), "src/teleoperation/tele_utils"))
from teleop_utils import * #console, reset_mtm, wait_for_message_custom

ROBOT_Z_OFFSET = 0.25

def get_states(data):
    """Return all states of the trajectory"""

    states = []

    for eef_state in data["traj"]:
        states.append(list(eef_state["pose"]["p"]))   # states: shape (traj_length, 3). 

    return np.array(states)

def get_goal_projected_on_image(goal_pc, i, thickness = 0):
    # proj = gym.get_camera_proj_matrix(sim, envs_obj[i], cam_handles[i])
    # fu = 2/proj[0, 0]
    # fv = 2/proj[1, 1]
    

    u_s =[]
    v_s = []
    for point in goal_pc:
        point = list(point) + [1]

        point = np.expand_dims(np.array(point), axis=0)

        point_cam_frame = point * np.matrix(gym.get_camera_view_matrix(sim, envs[i], vis_cam_handles[0]))
        # print("point_cam_frame:", point_cam_frame)
        # image_coordinates = (gym.get_camera_proj_matrix(sim, envs_obj[i], cam_handles[0]) * point_cam_frame)
        # print("image_coordinates:",image_coordinates)
        # u_s.append(image_coordinates[1, 0]/image_coordinates[2, 0]*2)
        # v_s.append(image_coordinates[0, 0]/image_coordinates[2, 0]*2)
        # print("fu fv:", fu, fv)
        u_s.append(1/2 * point_cam_frame[0, 0]/point_cam_frame[0, 2])
        v_s.append(1/2 * point_cam_frame[0, 1]/point_cam_frame[0, 2])      
          
    centerU = vis_cam_width/2
    centerV = vis_cam_height/2    
    # print(centerU - np.array(u_s)*cam_width)
    # y_s = (np.array(u_s)*cam_width).astype(int)
    # x_s = (np.array(v_s)*cam_height).astype(int)
    y_s = (centerU - np.array(u_s)*vis_cam_width).astype(int)
    x_s = (centerV + np.array(v_s)*vis_cam_height).astype(int)    

    if thickness != 0:
        new_y_s = deepcopy(list(y_s))
        new_x_s = deepcopy(list(x_s))
        for y, x in zip(y_s, x_s):
            for t in range(1, thickness+1):
                new_y_s.append(max(y-t,0))
                new_x_s.append(max(x-t,0))
                new_y_s.append(max(y-t,0))
                new_x_s.append(min(x+t, vis_cam_height-1))                
                new_y_s.append(min(y+t, vis_cam_width-1))
                new_x_s.append(max(x-t,0))                    
                new_y_s.append(min(y+t, vis_cam_width-1))                
                new_x_s.append(min(x+t, vis_cam_height-1))
        y_s = new_y_s
        x_s = new_x_s
    # print(x_s)
    return x_s, y_s

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
    parser.add_argument('--data_recording_path', default="/home/dvrk/shape_servo_data/generalization/multi_boxes_1000Pa/data", type=str, help="path to save the recorded data")
    parser.add_argument('--object_meshes_path', default="/home/dvrk/sim_data/Custom/Custom_mesh/multi_boxes_1000Pa", type=str, help="path to the objects' tet meshe files")
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
    # soft_pose.p = gymapi.Vec3(0.0, -0.44, 0.16+0.015)
    # pose = list(np.array(init_pose) + np.random.uniform(low=[-0.1,-0.03,-0.05], high=[0.1,0.02,0.05], size=3))
    pose = list(np.array(init_pose) + np.random.uniform(low=[-0.1,-0.03,-0.05], high=[0.1,0.02,0.05], size=3))
    # soft_pose.p = gymapi.Vec3(*pose)
    soft_pose.p = gymapi.Vec3(0.05, -0.44, 0.16+0.015-0.02)

    soft_thickness = 0.0005    # important to add some thickness to the soft body to avoid interpenetrations

    goal_pose = gymapi.Transform()
    goal_pose.p = gymapi.Vec3(*list(np.array(init_pose) + np.random.uniform(low=[-0.1,-0.03,-0.05], high=[0.1,0.02,0.05], size=3))) 


    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.thickness = soft_thickness
    asset_options.disable_gravity = True

    soft_asset = gym.load_asset(sim, asset_root, soft_asset_file, asset_options)       

    # wall_asset_root = "/home/dvrk/dvrk_ws/src/dvrk_env/shape_servo_control/src/teleoperation/random_stuff"
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

        # add dvrk
        dvrk_handle = gym.create_actor(env, dvrk_asset, dvrk_pose, "dvrk", i, 1, segmentationId=11)    
        dvrk_handles.append(dvrk_handle)    
        

        # add soft obj            
        # ball_actor = gym.create_actor(env, soft_asset, soft_pose, "soft", i, 0)
        ball_actor = gym.create_actor(env, soft_asset, soft_pose, "soft", i+1, 1)
        color = gymapi.Vec3(0,1,0)
        gym.set_rigid_body_color(env, ball_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
        object_handles.append(ball_actor)

        # goal_actor = gym.create_actor(env, soft_asset, goal_pose, "soft", i+1, 1)
        # color = gymapi.Vec3(1,0,0)
        # gym.set_rigid_body_color(env, goal_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
        # object_handles.append(goal_actor)

        # wall_actor = gym.create_actor(env, wall_asset, soft_pose, "wall", i+1, 1)
        # color = gymapi.Vec3(0,1,0)
        # gym.set_rigid_body_color(env, wall_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)




    # Viewer camera setup
    if not args.headless:

        cam_pos = gymapi.Vec3(0.2, -0.34, 0.4)
        cam_target = gymapi.Vec3(0.0, -0.44, 0.1)

        middle_env = envs[num_envs // 2 + num_per_row // 2]
        gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)

    
    # # Camera for point cloud setup
    # cam_handles = []
    # cam_width = 300 #256
    # cam_height = 300 #256 
    # # cam_positions = gymapi.Vec3(0.17, -0.62, 0.2)
    # # cam_targets = gymapi.Vec3(0.0, 0.40-0.86, 0.01)
    # # low=[-0.1,-0.03,-0.05], high=[0.1,0.02,0.05]
    # cam_positions = gymapi.Vec3(0.2, -0.54, 0.16+0.015 + 0.1)
    # cam_targets = gymapi.Vec3(0.0, -0.44, 0.16+0.015)

    # cam_handle, cam_prop = setup_cam(gym, envs[0], cam_width, cam_height, cam_positions, cam_targets)
    # for i, env in enumerate(envs):
    #     cam_handles.append(cam_handle)

    # Camera for vis
    vis_cam_handles = []
    vis_cam_width = 1000
    vis_cam_height = 1000  
    vis_cam_positions = gymapi.Vec3(0.25, -0.45, 0.16+0.015 + 0.15)
    vis_cam_targets = gymapi.Vec3(0.0, -0.35, 0.16+0.015)

    vis_cam_handle, vis_cam_prop = setup_cam(gym, envs[0], vis_cam_width, vis_cam_height, vis_cam_positions, vis_cam_targets)
    for i, env in enumerate(envs):
        vis_cam_handles.append(vis_cam_handle)
       

    '''
    Main simulation stuff starts from here
    '''
    rospy.init_node('shape_servo_controla')
    rospy.logerr(f"Save data: {args.save_data}")

    # Some important paramters
    init_dvrk_joints(gym, envs[i], dvrk_handles[i], joint_angles=[0,0,0,0,0.20,0,0,0,0.35,-0.35])  # Initilize robot's joints    

    all_done = False
    state = "home"
    
    demos_path = "/home/dvrk/LfD_data/group_meeting/demos"
    image_main_path = "/home/dvrk/LfD_data/group_meeting/images/processed_images"    
    os.makedirs(image_main_path, exist_ok=True)
    
    frame_count = 0
    sample_count = 0 #len(os.listdir(demos_path)) #0
    max_sample_count = 20000#10 #20#3
    group_count = 0
    max_group_count = 20#10 #20
    vis_frame_count = 0
    # data_point_count = len(os.listdir(data_recording_path))
    # rospy.logerr(f"data_point_count: {data_point_count}")
    # max_data_point_count = 200 #data_point_count+10
    # eef_states_1 = []
    # eef_states_2 = []
    eef_states = []
    num_image = 0
    record_video = True #True
    num_cp = 0
    # max_cp = 1#2    # number of checkpoints between start and goal (not oncluding start and goal) 3 means start -> cp -> green -> cp -> goal

    first_time = True

    start_time = timeit.default_timer()
    close_viewer = False

    while (not close_viewer) and (not all_done): 

        if not args.headless:
            close_viewer = gym.query_viewer_has_closed(viewer)  

        # step the physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)
 
        # Record videos

        radius = 6 #1        
        # Red color in BGR
        color = (0, 0, 255)
        thickness = 7 

        gym.render_all_camera_sensors(sim)
        im = gym.get_camera_image(sim, envs[i], vis_cam_handles[0], gymapi.IMAGE_COLOR).reshape((vis_cam_height,vis_cam_width,4))[:,:,:3]
        
        # im = Image.fromarray(im)     

        for group_idx in range(0,1):    
            for sample_idx in range(sample_count, max_sample_count): 
                file = os.path.join(demos_path, f"group {group_idx} sample {sample_idx}.pickle")
                with open(file, 'rb') as handle:
                    data = pickle.load(handle)   
                    
                traj = get_states(data) 

                goal_xs, goal_ys = get_goal_projected_on_image(traj, 0, thickness = 0)
                points = np.column_stack((np.array(goal_ys), np.array(goal_xs)))            

                image = im.astype(np.uint8)

                for idx, point in enumerate(points):
                    if idx == 0:
                        image = cv2.circle(image, tuple(point), round(radius*1.5), (0,255,255), round(thickness*1.5))
                    elif idx == points.shape[0]-1:
                        image = cv2.circle(image, tuple(point), round(radius*1.5), (0,0,255), round(thickness*1.5))
                    else:
                        image = cv2.circle(image, tuple(point), radius, (255, 0, 0), thickness)   

                img_path =  os.path.join(image_main_path, f"group {group_idx} sample {sample_idx}.png")            
                cv2.imwrite(img_path, image)

        
        all_done = True
            






  
   



    print("All done !")
    print("Elapsed time", timeit.default_timer() - start_time)
    if not args.headless:
        gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

