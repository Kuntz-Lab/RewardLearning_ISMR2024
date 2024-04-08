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

import torch
import torchvision
from torchvision.utils import make_grid


'''
Saving the triangular mesh of the retracted tissue and the urdf linking to it. 
'''

ROBOT_Z_OFFSET = 0.25

def default_sim_config_deformable(gym, args):
    sim_type = gymapi.SIM_FLEX
    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

    sim_params.substeps = 4
    sim_params.dt = 1./60.
    sim_params.flex.solver_type = 5
    sim_params.flex.num_outer_iterations = 10
    sim_params.flex.num_inner_iterations = 50
    sim_params.flex.relaxation = 0.7
    sim_params.flex.warm_start = 0.1
    sim_params.flex.shape_collision_distance = 5e-4
    sim_params.flex.contact_regularization = 1.0e-6
    sim_params.flex.shape_collision_margin = 1.0e-4
    sim_params.flex.deterministic_mode = True    
    # return gym.create_sim(args.compute_device_id, args.graphics_device_id, sim_type, sim_params), sim_params

    gpu_physics = 0
    gpu_render = 0
    # if args.headless:
    #     gpu_render = -1
    return gym.create_sim(gpu_physics, gpu_render, sim_type,
                          sim_params), sim_params


def default_dvrk_asset(gym, sim):
    '''
    load the dvrk asset
    '''
    # dvrk asset
    asset_options = gymapi.AssetOptions()
    asset_options.armature = 0.001
    asset_options.fix_base_link = True
    asset_options.thickness = 0.001#0.0001

    asset_options.flip_visual_attachments = False
    asset_options.collapse_fixed_joints = True
    asset_options.disable_gravity = True
    asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
    asset_options.max_angular_velocity = 40000.

    asset_root = "/home/dvrk/catkin_ws/src/dvrk_env"
    dvrk_asset_file = "dvrk_description/psm/psm_for_issacgym.urdf"
    print("Loading asset '%s' from '%s'" % (dvrk_asset_file, asset_root))
    return gym.load_asset(sim, asset_root, dvrk_asset_file, asset_options)

def configure_isaacgym(gym, args):
    # configure sim
    sim, sim_params = default_sim_config_deformable(gym, args)

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


    # load robot helpers asset
    dvrk_asset = default_dvrk_asset(gym, sim)

    dvrk_pose = gymapi.Transform()
    dvrk_pose.p = gymapi.Vec3(0.0, 0.0, ROBOT_Z_OFFSET)
    dvrk_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0) 

    dvrk_1_pose = gymapi.Transform()
    dvrk_1_pose.p = gymapi.Vec3(0.3, 0.0, ROBOT_Z_OFFSET)
    dvrk_1_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0) 

    dvrk_2_pose = gymapi.Transform()
    dvrk_2_pose.p = gymapi.Vec3(-0.3, 0.0, ROBOT_Z_OFFSET)
    dvrk_2_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)       

   
    asset_root = args.object_urdf_path
    soft_asset_file = f"{args.object_name}.urdf"

    mesh_root = args.object_mesh_path
    with open(os.path.join(mesh_root, "primitive_dict.pickle"), 'rb') as handle:
        data = pickle.load(handle)  
    h = data["box"]["height"]
    w = data["box"]["width"]
    thickness = data["box"]["thickness"]
    base_thickness = data["base"]["thickness"]

    soft_pose =  gymapi.Transform()
    z_coor = base_thickness + thickness*0.5 #+ 0.01
    soft_pose.p = gymapi.Vec3(0.0, -0.4, z_coor)
    #soft_pose.r = gymapi.Quat(0.0, 0.0, 0.707107, 0.707107)
    soft_thickness = 0.001

    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.disable_gravity = True
    asset_options.thickness = soft_thickness

    soft_asset = gym.load_asset(sim, asset_root, soft_asset_file, asset_options) 

    with open(os.path.join(args.object_urdf_path, f"{args.object_name}_balls_relative_xyz.pickle"), 'rb') as handle:
        balls_xyz_data = pickle.load(handle)

    balls_relative_xyz = np.array(balls_xyz_data["balls_relative_xyz"])

    print(f"@@@@@@@@@@@@@@@@@@ balls relative xyz: {balls_relative_xyz} @@@@@@@@@@@@@@@@@@@@@@@@@@@")

    # set up the env grid
    num_envs = 1
    spacing = 0.0
    env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
    env_upper = gymapi.Vec3(spacing, spacing, spacing)
    num_per_row = int(math.sqrt(num_envs))

    # asset_root = '/home/dvrk/catkin_ws/src/dvrk_env/shape_servo_control/src/teleoperation/random_stuff'
    # ball_asset_file = "new_ball.urdf"
    # asset_options = gymapi.AssetOptions()
    # asset_options.fix_base_link = True
    # asset_options.disable_gravity = True
    # asset_options.thickness = 0.000001

    # ball_asset = gym.load_asset(sim, asset_root, ball_asset_file, asset_options)   

  
    # cache some common handles for later use
    envs = []
    dvrk_handles = []
    dvrk_1_handles = []
    dvrk_2_handles = []
    soft_handles = []
    for i in range(num_envs):
        # create env
        env = gym.create_env(sim, env_lower, env_upper, num_per_row)
        envs.append(env)

        # add dvrk
        # dvrk_handle = gym.create_actor(env, dvrk_asset, dvrk_pose, "dvrk", i+1, 1, segmentationId=11)    
        # dvrk_handles.append(dvrk_handle) 

        # ball1pose = gymapi.Transform()
        # ball1pose.p = gymapi.Vec3(-0.1, -0.5, 0.01)
        # ball2pose = gymapi.Transform()
        # ball2pose.p = gymapi.Vec3(0.1, -0.3, 0.01)
        # ball3pose = gymapi.Transform()
        # ball3pose.p = gymapi.Vec3(-0.1, -0.3, 0.01)
        # ball4pose = gymapi.Transform()
        # ball4pose.p = gymapi.Vec3(0.1, -0.5, 0.01)
        # gym.create_actor(env, ball_asset, ball1pose, "ball1", i, 1, segmentationId=11)    
        # gym.create_actor(env, ball_asset, ball2pose, "ball2", i, 1, segmentationId=11)    
        # gym.create_actor(env, ball_asset, ball3pose, "ball3", i, 1, segmentationId=11)    
        # gym.create_actor(env, ball_asset, ball4pose, "ball4", i, 1, segmentationId=11)    


        # add dvrk 1
        dvrk_1_handle = gym.create_actor(env, dvrk_asset, dvrk_1_pose, "dvrk1", i+1, 1, segmentationId=11)    
        dvrk_1_handles.append(dvrk_1_handle) 

        # add dvrk 2
        dvrk_2_handle = gym.create_actor(env, dvrk_asset, dvrk_2_pose, "dvrk2", i+1, 1, segmentationId=11)    
        dvrk_2_handles.append(dvrk_2_handle)    
        
        # add soft obj       
        soft_handle = gym.create_actor(env, soft_asset, soft_pose, f"soft", i+1, 0)
        soft_handles.append(soft_handle)


    # DOF Properties and Drive Modes 
    dof_props = gym.get_actor_dof_properties(envs[0], dvrk_1_handles[0])
    dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
    dof_props["stiffness"].fill(200.0)
    dof_props["damping"].fill(40.0)
    dof_props["stiffness"][8:].fill(1)
    dof_props["damping"][8:].fill(2)  
    
    for i, env in enumerate(envs):
        #gym.set_actor_dof_properties(env, dvrk_handles[i], dof_props)
        gym.set_actor_dof_properties(env, dvrk_1_handles[i], dof_props)
        gym.set_actor_dof_properties(env, dvrk_2_handles[i], dof_props)


    # Viewer camera setup
    if not args.headless:
        cam_target = gymapi.Vec3(0.0, -0.4, 0.001) #gymapi.Vec3(0.0, -0.4, 0.05)
        cam_pos = gymapi.Vec3(0.0, -0.0, 0.1) #gymapi.Vec3(0.3, -0.8, 0.5)
        middle_env = envs[num_envs // 2 + num_per_row // 2]
        gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)

    
     # Camera for point cloud setup
    cam_handles = []
    cam_width = 300#400
    cam_height = 300#400
   
    cam_targets = gymapi.Vec3(0.0, -0.4, 0.01)
    cam_positions = gymapi.Vec3(0.0, -0.0, 0.1)
    
    cam_handle, cam_prop = setup_cam(gym, envs[0], cam_width, cam_height, cam_positions, cam_targets)
    for i, env in enumerate(envs):
        cam_handles.append(cam_handle)

    if not args.headless:
        return envs, sim, dvrk_handles, dvrk_1_handles,  dvrk_2_handles, dvrk_1_pose, dvrk_2_pose, soft_handles, cam_handles, cam_prop, viewer, balls_relative_xyz, soft_pose
    else:
        return envs, sim, dvrk_handles, dvrk_1_handles,  dvrk_2_handles, dvrk_1_pose, dvrk_2_pose, soft_handles, cam_handles, cam_prop, None, balls_relative_xyz, soft_pose


def step_physics(sim_cache):
    gym = sim_cache["gym"]
    sim = sim_cache["sim"]
    gym.simulate(sim)
    gym.fetch_results(sim, True)

def step_rendering(sim_cache, args):
    gym = sim_cache["gym"]
    sim = sim_cache["sim"]
    viewer = sim_cache["viewer"]
    gym.step_graphics(sim)
    if not args.headless:
        gym.draw_viewer(viewer, sim, False)

def home_state(sim_cache, args):
    state = "home"
    rospy.loginfo("**Current state: " + state)
    
    envs = sim_cache["envs"]
    gym = sim_cache["gym"]
    dvrk_1_handles = sim_cache["dvrk_1_handles"]
    dvrk_2_handles = sim_cache["dvrk_2_handles"]

    # for frame_count in range(50):
    #     step_physics(sim_cache)
    #     gym.set_joint_target_position(envs[0], gym.get_joint_handle(envs[0], "dvrk1", "psm_main_insertion_joint"), 0.20)            
    #     if frame_count == 49:
    #         # # Save robot and object states for reset 
    #         init_robot_state = deepcopy(gym.get_actor_rigid_body_states(envs[0], dvrk_1_handles[0], gymapi.STATE_ALL))
    #         #end-effector pose           
    #         current_pose = deepcopy(gym.get_actor_rigid_body_states(envs[0], dvrk_1_handles[0], gymapi.STATE_POS)[-3])

    #     step_rendering(sim_cache, args)

    # for frame_count in range(50):
    #     step_physics(sim_cache)
    #     gym.set_joint_target_position(envs[0], gym.get_joint_handle(envs[0], "dvrk2", "psm_main_insertion_joint"), 0.20)            
    #     if frame_count == 49:
    #         # # Save robot and object states for reset 
    #         init_robot_state = deepcopy(gym.get_actor_rigid_body_states(envs[0], dvrk_2_handles[0], gymapi.STATE_ALL))
    #         #end-effector pose           
    #         current_pose = deepcopy(gym.get_actor_rigid_body_states(envs[0], dvrk_2_handles[0], gymapi.STATE_POS)[-3])
    #     step_rendering(sim_cache, args)

    state = "grasp"

    return state

def init_dvrk_before_grasp(gym, env, dvrk_handle, joint_angles=None):
    dvrk_dof_states = gym.get_actor_dof_states(env, dvrk_handle, gymapi.STATE_NONE)
    if joint_angles is None:
        dvrk_dof_states['pos'][8] = 1.5
        dvrk_dof_states['pos'][9] = 0.8
        dvrk_dof_states['pos'][4] = 0.15#0.24
        gym.set_actor_dof_states(env, dvrk_handle, dvrk_dof_states, gymapi.STATE_POS)
    else:
        dvrk_dof_states['pos'] = joint_angles


def multi_robots_move_to_xyz(sim_cache, robot_list, dc_client, robot_handles_list, robot_pose_list, xyz_list, args, grippers_pos=[1.5,0.8]):  
    envs = sim_cache["envs"]
    gym = sim_cache["gym"]
    
    num_robots = len(robot_list)
    plan_traj_list = [None for _ in range(num_robots)]
    for i in range(num_robots):
        success = False
        plan_traj = []
        while(not success):
            cartesian_pose = Pose()
            cartesian_pose.orientation.x = 0
            cartesian_pose.orientation.y = 0.707107
            cartesian_pose.orientation.z = 0.707107
            cartesian_pose.orientation.w = 0
            cartesian_pose.position.x = robot_pose_list[i].p.x-xyz_list[i][0]
            cartesian_pose.position.y = robot_pose_list[i].p.y-xyz_list[i][1]
            cartesian_pose.position.z = xyz_list[i][2]-ROBOT_Z_OFFSET

            # Set up moveit for the above delta x, y, z
            plan_traj = dc_client.arm_moveit_planner_client(go_home=False, cartesian_goal=cartesian_pose, current_position=robot_list[i].get_full_joint_positions())
            if (plan_traj):
                success = True
        
        plan_traj_list[i] = plan_traj

    rospy.loginfo("++++++++++++++++++ got the motion plan !!!")

    traj_indices = [0 for _ in range(num_robots)]
    plan_traj_len_list = [len(plan_traj) for plan_traj in plan_traj_list]
    print("plan lengths: ", plan_traj_len_list)

    successes = [False for _ in range(num_robots)]
    start_time = timeit.default_timer()
    ten_min = 600
    while not all(successes):
        if timeit.default_timer() - start_time > ten_min:
            print(f"taking too long to move: {timeit.default_timer() - start_time} sec")
            break
             
        # contacts = [contact[4] for contact in gym.get_soft_contacts(sim)]
        # print("20 contact: ", (20 in contacts or 21 in contacts))
        # print("9 contact: ",  (9 in contacts or 10 in contacts))
        step_physics(sim_cache)
        print("successes: ", successes)
        for i in range(num_robots):
            # Set target joint positions
            if not successes[i]:
                dof_states = robot_list[i].get_full_joint_positions()
                plan_traj_with_gripper = [plan+grippers_pos for plan in plan_traj_list[i]] #[plan+[0.35,-0.35] for plan in plan_traj_list[i]]
                pos_targets = np.array(plan_traj_with_gripper[traj_indices[i]], dtype=np.float32)
                gym.set_actor_dof_position_targets(envs[0], robot_handles_list[i][0], pos_targets)                

                if traj_indices[i] <= plan_traj_len_list[i] - 2:
                    if np.allclose(dof_states[:8], pos_targets[:8], rtol=0, atol=0.1):
                        traj_indices[i] += 1 
                else:
                    if np.allclose(dof_states[:8], pos_targets[:8], rtol=0, atol=0.02):
                        traj_indices[i] += 1   
                
                if traj_indices[i] >= plan_traj_len_list[i]:
                    successes[i] = True

        step_rendering(sim_cache, args)

    rospy.loginfo("++++++++++++++++++ Finished moving towards specified xyz")

def get_object_particle_state(gym, sim, data_config, sim_cache, vis=False):
    gym.refresh_particle_state_tensor(sim)
    particle_state_tensor = deepcopy(gymtorch.wrap_tensor(gym.acquire_particle_state_tensor(sim)))
    particles = particle_state_tensor.numpy()[:, :3]  
    
    if vis:
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(np.array(particles))
        pcd.paint_uniform_color([1,0,0]) # color: list of len 3

        corner_z = np.max(particles[:, 2])
        corner_y = np.max(particles[:, 1])
        corner_x_neg = np.min(particles[:, 0])
        corner_x_pos = np.max(particles[:, 0])

        soft_pose = sim_cache["soft_pose"]
        balls_relative_xyz = data_config["balls_relative_xyz"]
        corners_and_attachment = np.array([[corner_x_pos, corner_y, corner_z], [corner_x_neg, corner_y, corner_z], 
                                    [soft_pose.p.x+balls_relative_xyz[0][0], soft_pose.p.y+balls_relative_xyz[0][1], balls_relative_xyz[0][2]]])

        corner_pcd = open3d.geometry.PointCloud()
        corner_pcd.points = open3d.utility.Vector3dVector(np.array(corners_and_attachment))
        corner_pcd.paint_uniform_color([0,0,1]) # color: list of len 3
        
        open3d.visualization.draw_geometries([pcd, corner_pcd]) 
    
    return particles.astype('float32')

def grasp_state(sim_cache, robot_list, dc_client, args):
    state = "grasp"
    rospy.loginfo("**Current state: " + state)

    envs = sim_cache["envs"]
    gym = sim_cache["gym"]
    dvrk_1_handles = sim_cache["dvrk_1_handles"]
    dvrk_2_handles = sim_cache["dvrk_2_handles"]
    dvrk_1_pose = sim_cache["dvrk_1_pose"]
    dvrk_2_pose = sim_cache["dvrk_2_pose"]


    obj_particles = get_object_particle_state(gym, sim, data_config, sim_cache, vis=False)
    eps = 0.005 # in order to grasp at a point more inside the object
    corner_z = np.max(obj_particles[:, 2]) #+ eps
    corner_y = np.max(obj_particles[:, 1]) - eps - 0.02
    corner_x_neg = np.min(obj_particles[:, 0]) + eps + 0.02
    corner_x_pos = np.max(obj_particles[:, 0]) - eps - 0.02

    # move to grasping point
    robot_handles_list = [dvrk_1_handles, dvrk_2_handles]
    xyz_list = [[corner_x_pos, corner_y, corner_z], [corner_x_neg, corner_y, corner_z]]
    robot_pose_list = [dvrk_1_pose, dvrk_2_pose]
    rospy.loginfo(f"--------- step 1: move above")
    rospy.loginfo(f"--------- dvrk 1 move to: {xyz_list[0]}")
    rospy.loginfo(f"--------- dvrk 2 move to: {xyz_list[1]}")
    multi_robots_move_to_xyz(sim_cache, robot_list, dc_client, robot_handles_list, robot_pose_list, xyz_list, args, grippers_pos=[1.5, 0.8])

    robot_handles_list = [dvrk_1_handles, dvrk_2_handles]
    xyz_list = [[corner_x_pos, corner_y, corner_z - eps/2], [corner_x_neg, corner_y, corner_z - eps/2]]
    robot_pose_list = [dvrk_1_pose, dvrk_2_pose]
    rospy.loginfo(f"--------- step 2: press down")
    rospy.loginfo(f"--------- dvrk 1 move to: {xyz_list[0]}")
    rospy.loginfo(f"--------- dvrk 2 move to: {xyz_list[1]}")
    multi_robots_move_to_xyz(sim_cache, robot_list, dc_client, robot_handles_list, robot_pose_list, xyz_list, args, grippers_pos=[1.5, 0.8])

    # grasp
    dof_states_1 = gym.get_actor_dof_states(envs[0], dvrk_1_handles[0], gymapi.STATE_POS)
    dof_states_2 = gym.get_actor_dof_states(envs[0], dvrk_2_handles[0], gymapi.STATE_POS)
    rospy.loginfo(f"--------- grasping object")
    while dof_states_1['pos'][8]>=0.4 or dof_states_2['pos'][8]>=0.4:
         print("dvrk1 gripper pos: ", dof_states_1["pos"][8])
         print("dvrk2 gripper pos: ", dof_states_2["pos"][8])
         step_physics(sim_cache)
         gym.set_joint_target_position(envs[0], gym.get_joint_handle(envs[0], "dvrk1", "psm_tool_gripper1_joint"), -2.5)
         gym.set_joint_target_position(envs[0], gym.get_joint_handle(envs[0], "dvrk1", "psm_tool_gripper2_joint"), -3.0)
         gym.set_joint_target_position(envs[0], gym.get_joint_handle(envs[0], "dvrk2", "psm_tool_gripper1_joint"), -2.5)
         gym.set_joint_target_position(envs[0], gym.get_joint_handle(envs[0], "dvrk2", "psm_tool_gripper2_joint"), -3.0)
         dof_states_1 = gym.get_actor_dof_states(envs[0], dvrk_1_handles[0], gymapi.STATE_POS)
         dof_states_2 = gym.get_actor_dof_states(envs[0], dvrk_2_handles[0], gymapi.STATE_POS)
         step_rendering(sim_cache, args)

    state = "retract"
    return state, xyz_list

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

    grid_layout_images(images, num_columns, output_file=output_file, display_on_screen=False)

def retract_state(sim_cache, data_config, robot_list, dc_client, xyz_list, args):
    state = "retract"
    rospy.loginfo("**Current state: " + state)

    envs = sim_cache["envs"]
    gym = sim_cache["gym"]
    sim = sim_cache["sim"]
    dvrk_1_handles = sim_cache["dvrk_1_handles"]
    dvrk_2_handles = sim_cache["dvrk_2_handles"]
    dvrk_1_pose = sim_cache["dvrk_1_pose"]
    dvrk_2_pose = sim_cache["dvrk_2_pose"]
    soft_pose = sim_cache["soft_pose"]


    x_1, y_1, z_1 = xyz_list[0]
    x_2, y_2, z_2 = xyz_list[1]

    # use balls xyz here ??
    x_b, y_b, z_b = data_config["balls_relative_xyz"][0]
    ##########For the mistake I made in setting the origin_xyz of the attachment point in the URDF ##############
    x_b = x_b/2
    y_b = y_b/2
    # assert(data_config["balls_relative_xyz"][0][0]!=x_b)
    # assert(data_config["balls_relative_xyz"][0][1]!=y_b)
    #############################################################################################################
    x_b = soft_pose.p.x + x_b
    y_b = soft_pose.p.y + y_b
    #z_b = soft_pose.p.z + z_b
    print(f"ball: {x_b},{y_b},{z_b}")
    print(f"dvrk eef 1: {x_1},{y_1},{z_1}")
    print(f"dvrk eef 2: {x_2},{y_2},{z_2}")
    offset_y_1 = y_1 - y_b
    offset_y_2 = y_2 - y_b
    adjust_y_1 = np.abs(x_1 - x_b)
    adjust_y_2 = np.abs(x_2 - x_b)
    print("adjust y 1: ", adjust_y_1)
    print("adjust y 2: ", adjust_y_2)
    assert(offset_y_1 >= 0)
    assert(offset_y_2 >= 0)

    extra_offset = 0.08 #0.1 for larger sampling area of attachment points, otherwise, use the default value
    xyz_list = [[x_1, y_1 - (offset_y_1 + extra_offset), z_1 + (offset_y_1 + extra_offset)], [x_2, y_2 - (offset_y_2 + extra_offset), z_2 + (offset_y_2 + extra_offset)]]
    if adjust_y_1 <=0.039:
        xyz_list[0] = [x_1, y_1 - (adjust_y_1)*(offset_y_1 + extra_offset), z_1 + (offset_y_1 + extra_offset)]
    if adjust_y_2 <=0.039:
        xyz_list[1] = [x_2, y_2 - adjust_y_2*(offset_y_2 + extra_offset), z_2 + (offset_y_2 + extra_offset)]

    robot_handles_list = [dvrk_1_handles, dvrk_2_handles]
    robot_pose_list = [dvrk_1_pose, dvrk_2_pose]
    rospy.loginfo(f"--------- dvrk 1 move to: {xyz_list[0]}")
    rospy.loginfo(f"--------- dvrk 2 move to: {xyz_list[1]}")
    multi_robots_move_to_xyz(sim_cache, robot_list, dc_client, robot_handles_list, robot_pose_list, xyz_list, args, grippers_pos=[0.35, -0.35])


    gym.render_all_camera_sensors(sim)
    #visualize_camera_views(gym, sim, envs, cam_handles, resolution=[300,300], output_file="home/dvrk/Downloads/test_image.png", num_columns=4)
    pcd = get_partial_pointcloud_vectorized(gym, sim, envs[0], cam_handles[0], cam_prop, color=[1,0,0], min_z = 0.005, visualization=False)

    contacts = [contact[4] for contact in gym.get_soft_contacts(sim)]
    if not(20 in contacts or 21 in contacts) or not(9 in contacts or 10 in contacts):
        print("------ Not in contact, discard the current sample ----------")
        state = "reset"
        return state
    

    if args.save_data:
        rospy.loginfo("--saving data--")
        obj_particles = get_object_particle_state(gym, sim, data_config, sim_cache, vis=False)

        (tri_indices, _, _) = gym.get_sim_triangles(sim)
        tri_indices = np.array(tri_indices).reshape(-1,3)
        trimesh_mesh = trimesh.Trimesh(vertices=obj_particles, faces=tri_indices)
       
        data_recording_path = data_config["data_recording_path"]
        full_data_recording_path = os.path.join(data_recording_path, "full_data")
        mesh_data_recording_path = os.path.join(data_recording_path, "tri_mesh")
        urdf_data_recording_path = os.path.join(data_recording_path, "urdf")

        num_data = len(os.listdir(full_data_recording_path))
        assert(len(os.listdir(full_data_recording_path))==len(os.listdir(mesh_data_recording_path))==len(os.listdir(urdf_data_recording_path)))

        # save triangular mesh
        trimesh_mesh.export(os.path.join(mesh_data_recording_path, f"mesh_{num_data}.obj")) 

        # save other data
        data = {"num_vertices": obj_particles.shape[0], "tri_indices": tri_indices, "balls_relative_xyz": data_config["balls_relative_xyz"], "soft_xyz": [soft_pose.p.x, soft_pose.p.y, soft_pose.p.z]}
        with open(os.path.join(full_data_recording_path, f"group {num_data}.pickle"), 'wb') as handle:
                pickle.dump(data, handle, protocol=3)

        # write urdf
        f = open(os.path.join(urdf_data_recording_path, f"tissue_{num_data}.urdf"), 'w')
        urdf_str = f"""<?xml version="1.0"?>
            <robot name="tissue">
                <link name="tissue">
                    <visual>
                        <origin xyz="0.0 0.0 0.0"/>
                            <geometry>
                                <mesh filename="{os.path.join(mesh_data_recording_path, f"mesh_{num_data}.obj")}" scale="1.0 1.0 1.0"/>
                            </geometry>
                    </visual>
                    <collision>
                        <origin xyz="0.0 0.0 0.0"/>
                        <geometry>
                            <mesh filename="{os.path.join(mesh_data_recording_path, f"mesh_{num_data}.obj")}" scale="1.0 1.0 1.0"/>
                        </geometry>
                    </collision>
                     <inertial>
                        <mass value="5000"/>
                        <inertia ixx="0.0001" ixy="0.0" ixz="0.0" iyy="0.0001" iyz="0.0" izz="0.0001"/>
                    </inertial>
                </link>
            </robot>
        """   
        f.write(urdf_str)
        f.close()

    state = "reset"
    return state




   

if __name__ == "__main__":
    gym = gymapi.acquire_gym() # initialize gym

    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--data_recording_path', type=str, help="where you want to record data")
    parser.add_argument('--object_urdf_path', type=str, help="root path of the existing urdf")
    parser.add_argument('--object_mesh_path', type=str, help="root path of the existing meshes and primitive dict")
    parser.add_argument('--headless', default="False", type=str, help="run without viewer?")
    parser.add_argument('--save_data', default="False", type=str, help="True: save recorded data to pickles files")
    parser.add_argument('--rand_seed', default=2021, type=int, help="random seed")
    parser.add_argument('--object_name', type=str, help="name of the deformable object")
    
    args = parser.parse_args()

    args.headless = args.headless == "True"
    args.save_data = args.save_data == "True"
    data_recording_path = args.data_recording_path
    os.makedirs(data_recording_path, exist_ok=True)
    os.makedirs(os.path.join(data_recording_path, "full_data"), exist_ok=True)
    os.makedirs(os.path.join(data_recording_path, "tri_mesh"), exist_ok=True)
    os.makedirs(os.path.join(data_recording_path, "urdf"), exist_ok=True)


    rand_seed = args.rand_seed
    np.random.seed(rand_seed)
    random.seed(rand_seed)

   
    '''
    Main simulation stuff starts from here
    '''
    rospy.init_node('shape_servo_control')
    envs, sim, dvrk_handles, dvrk_1_handles,  dvrk_2_handles, dvrk_1_pose, dvrk_2_pose, soft_handles, cam_handles, cam_prop, viewer, balls_relative_xyz, soft_pose = configure_isaacgym(gym, args)

    state = "home"

    sim_cache = sim_cache = {"gym":gym, "sim":sim, "envs":envs, "dvrk_1_handles":dvrk_1_handles, "dvrk_2_handles":dvrk_2_handles, \
                "soft_handles":soft_handles, "cam_handles":cam_handles, "cam_prop": cam_prop, "viewer":viewer, \
                "dvrk_1_pose": dvrk_1_pose, "dvrk_2_pose": dvrk_2_pose, "soft_pose": soft_pose}


    data_config = {"data_recording_path": data_recording_path, "balls_relative_xyz": balls_relative_xyz}

    init_dvrk_before_grasp(gym, envs[0], dvrk_1_handles[0]) 
    init_dvrk_before_grasp(gym, envs[0], dvrk_2_handles[0])  



    
    start_time = timeit.default_timer()   
    robot_1 = Robot(gym, sim, envs[0], dvrk_1_handles[0])
    robot_2 = Robot(gym, sim, envs[0], dvrk_2_handles[0])
    dc_client = GraspClient()

    robot_list = [robot_1, robot_2]

    while (True): 
        step_physics(sim_cache)
        if state == "home" :   
            state = home_state(sim_cache, args)
        if state == "grasp":
            state, xyz_list = grasp_state(sim_cache, robot_list, dc_client, args)
        if state == "retract":
            state = retract_state(sim_cache, data_config, robot_list, dc_client, xyz_list, args)
        if state == "reset":
            break
        step_rendering(sim_cache, args)


    print("All done !")
    print("Elapsed time", timeit.default_timer() - start_time)
    if not args.headless:
        gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
