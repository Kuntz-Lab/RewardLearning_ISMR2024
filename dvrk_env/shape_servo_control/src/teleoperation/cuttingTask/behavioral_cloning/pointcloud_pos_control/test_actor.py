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

from policy_BC_PC import ActorPC
import torch

sys.path.append("../../pointcloud_representation_learning")
sys.path.append("../../../pc_utils")
from architecture import AutoEncoder
from get_isaac_partial_pc import get_partial_pointcloud_vectorized
from compute_partial_pc import farthest_point_sample_batched

'''
Collect perfect demos of cutting autonomously
'''

ROBOT_Z_OFFSET = 0.25
OVERLAP_TOLERANCE = None
EEF_GROUND_Z_OFFSET = 0.03

def is_overlap(p1, p2, max_dist):
    return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2 <=max_dist

def get_sq_dist_between(p1, p2):
    return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2

def drop_ball(sim_cache, plan_config, eef_state, overlap_tolerance):
    balls_xyz = sim_cache["all_balls_xyz"]
    eef_xyz = np.array([eef_state["pose"]["p"]["x"], eef_state["pose"]["p"]["y"], eef_state["pose"]["p"]["z"]])

    for i, ball_xyz in enumerate(balls_xyz):
        #print("eef_xyz: ", eef_xyz)
        #print(f"sq dist between eef and ball: {get_sq_dist_between(eef_xyz, ball_xyz)}")
        if is_overlap(eef_xyz, ball_xyz, max_dist=overlap_tolerance):
            print(f"++++++++++++++++++++ in contact with ball {i}")
            print("overlap eef: ", eef_xyz)
            print("overlap ball: ", ball_xyz)
            sim_cache["all_balls_xyz"][i] = np.array([100,100,-100])
            plan_config["last_ball_xyz"] = ball_xyz
            plan_config["num_balls_reached"] += 1
            plan_config["which_balls_reached"][i] = 1

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

     # load robot helpers asset
    dvrk_asset = default_dvrk_asset(gym, sim)

    dvrk_pose = gymapi.Transform()
    dvrk_pose.p = gymapi.Vec3(0.0, 0.0, ROBOT_Z_OFFSET)
    dvrk_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0) 

    # load urdf of rigid tissue
    rigid_urdf_path = os.path.join(args.rigid_state_path, "urdf")
    asset_root = rigid_urdf_path

    soft_asset_file = f"{args.object_name}.urdf"

    with open(os.path.join(args.rigid_state_path, "full_data", f"group {args.group_count}.pickle"), 'rb') as handle:
        full_data = pickle.load(handle)
    soft_xyz = full_data["soft_xyz"]
    balls_relative_xyz = np.array(full_data["balls_relative_xyz"])

    print(f"@@@@@@@@@@@@@@@@@@ balls relative xyz: {balls_relative_xyz}")

    soft_pose =  gymapi.Transform()
    soft_pose.p = gymapi.Vec3(0, 0, 0)
    print(f"soft pose: {soft_xyz}")
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
    dvrk_handles = []
    soft_handles = []
    for i in range(num_envs):
        # create env
        env = gym.create_env(sim, env_lower, env_upper, num_per_row)
        envs.append(env)

        # add dvrk
        dvrk_handle = gym.create_actor(env, dvrk_asset, dvrk_pose, "dvrk", i, 1, segmentationId=11)    
        dvrk_handles.append(dvrk_handle) 

        # add soft obj       
        soft_handle = gym.create_actor(env, soft_asset, soft_pose, f"soft", i+1, 0)
        soft_handles.append(soft_handle)

    dof_props = gym.get_actor_dof_properties(envs[0], dvrk_handles[0])
    dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
    dof_props["stiffness"].fill(200.0)
    dof_props["damping"].fill(40.0)
    dof_props["stiffness"][8:].fill(1)
    dof_props["damping"][8:].fill(2)  
    
    for i, env in enumerate(envs):
        #gym.set_actor_dof_properties(env, dvrk_handles[i], dof_props)
        gym.set_actor_dof_properties(env, dvrk_handles[i], dof_props)
        
    # Viewer camera setup
    if not args.headless:
        cam_target = gymapi.Vec3(0.0, -0.4, 0.01)
        cam_pos = gymapi.Vec3(0.0, -0.3, 0.3)

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
        return envs, sim, dvrk_handles, soft_handles, cam_handles, cam_prop, viewer, balls_relative_xyz, soft_xyz
    else:
        return envs, sim, dvrk_handles, soft_handles, cam_handles, cam_prop, None, balls_relative_xyz, soft_xyz

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

def move_near_workspace(robot, dc_client):
    success = False
    plan_traj = []
    while(not success):
        cartesian_pose = Pose()
        cartesian_pose.orientation.x = 0
        cartesian_pose.orientation.y = 0.707107
        cartesian_pose.orientation.z = 0.707107
        cartesian_pose.orientation.w = 0
        cartesian_pose.position.x = 0
        cartesian_pose.position.y = -(-0.3)
        cartesian_pose.position.z = 0.05-ROBOT_Z_OFFSET

        # Set up moveit for the above delta x, y, z
        plan_traj = dc_client.arm_moveit_planner_client(go_home=False, cartesian_goal=cartesian_pose, current_position=robot.get_full_joint_positions())
        if (plan_traj):
           success = True

    
    print("((((((((plan:", [plan+[0.35,-0.35] for plan in plan_traj])

    traj_index = 0
    while traj_index < len(plan_traj):
        step_physics(sim_cache)

        # Set target joint positions
        dof_states = robot.get_full_joint_positions()
        plan_traj_with_gripper = [plan+[0.35,-0.35] for plan in plan_traj]
        pos_targets = np.array(plan_traj_with_gripper[traj_index], dtype=np.float32)
        gym.set_actor_dof_position_targets(envs[0], dvrk_handles[0], pos_targets)                

        if traj_index <= len(plan_traj) - 2:
            if np.allclose(dof_states[:8], pos_targets[:8], rtol=0, atol=0.1):
                traj_index += 1 
        else:
            if np.allclose(dof_states[:8], pos_targets[:8], rtol=0, atol=0.02):
                traj_index += 1   

        step_rendering(sim_cache, args)

    rospy.loginfo("++++++++++++++++++ Finished moving end-effector near workspace")
    #print(deepcopy(gym.get_actor_rigid_body_states(envs[0], dvrk_handles[0], gymapi.POS)[0]))

def vis_attachment(gym, sim, sim_cache):
    gym.render_all_camera_sensors(sim)
    particles = get_partial_pointcloud_vectorized(gym, sim, envs[0], cam_handles[0], cam_prop, color=[1,0,0], min_z = 0.005, visualization=False, min_depth=-1) 
    
    if True:
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(np.array(particles))
        pcd.paint_uniform_color([1,0,0]) # color: list of len 3

        soft_xyz = sim_cache["soft_xyz"]
        balls_relative_xyz = sim_cache["balls_relative_xyz"]
        np_balls_xyz = sim_cache["balls_xyz_init"]
        attachment = np.array([[soft_xyz[0]+balls_relative_xyz[0][0], soft_xyz[1]+balls_relative_xyz[0][1], balls_relative_xyz[0][2]]])

        attachment_pcd = open3d.geometry.PointCloud()
        attachment_pcd.points = open3d.utility.Vector3dVector(attachment)
        attachment_pcd.paint_uniform_color([0,0,1]) # color: list of len 3

        attachment_np_pcd = open3d.geometry.PointCloud()
        attachment_np_pcd.points = open3d.utility.Vector3dVector(np_balls_xyz)
        attachment_pcd.paint_uniform_color([0,1,0]) # color: list of len 3

        
        open3d.visualization.draw_geometries([pcd, attachment_pcd, attachment_np_pcd]) 
    
    return particles.astype('float32')


def home_state(sim_cache, robot, dc_client, args):
    state = "home"
    envs = sim_cache["envs"]
    gym = sim_cache["gym"]
    dvrk_handles = sim_cache["dvrk_handles"]
    
    for frame_count in range(50):
        step_physics(sim_cache)
        #vis_attachment(gym, sim, sim_cache)
        gym.set_joint_target_position(envs[0], gym.get_joint_handle(envs[0], "dvrk", "psm_main_insertion_joint"), 0.20)            
        if frame_count == 49:
            rospy.loginfo("**Current state: " + state)
            # # Save robot and object states for reset 
            init_robot_state = deepcopy(gym.get_actor_rigid_body_states(envs[0], dvrk_handles[0], gymapi.STATE_ALL))
            #end-effector pose           
            current_pose = deepcopy(gym.get_actor_rigid_body_states(envs[0], dvrk_handles[0], gymapi.STATE_POS)[-3])

            print("current pose:", current_pose["pose"]["p"])
            
        step_rendering(sim_cache, args)

    move_near_workspace(robot, dc_client)
    init_robot_state = deepcopy(gym.get_actor_rigid_body_states(envs[0], dvrk_handles[0], gymapi.STATE_ALL))

    # Go to next state
    state = "config"
    return state, init_robot_state

def get_random_indices(num_idxs, arr_len):
    indices = []
    for i in range(num_idxs):
        rand_idx = random.randint(0, arr_len-1)
        while rand_idx in indices:
            rand_idx = random.randint(0, arr_len-1)
        indices.append(rand_idx)
    return indices

def config_state(sim_cache, data_config):
    state = "config state"
    rospy.loginfo("**Current state: " + state) 

    state = "get shape servo plan"

    plan_config = {"which_balls_reached": [0 for i in range(len(balls_xyz))], "last_ball_xyz":None, "num_balls_reached": 0, \
                  "eef_states":[], "pcds":[], "balls_xyzs_list": [], "last_ball_xyzs": [], "frame":0, "max_frame":800, "stay":False}

    return state, plan_config

def to_obj_emb(model, device, pcd):
    pcd_tensor = torch.from_numpy(pcd.transpose(1,0)).unsqueeze(0).float().to(device)
    emb = model(pcd_tensor, get_global_embedding=True)
    return emb

def get_shape_servo_plan_state(sim_cache, plan_config, robot, dc_client, device, AE_model, policy):
    state = "get shape servo plan"
    rospy.loginfo("**Current state: " + state) 

    eef_state = deepcopy(gym.get_actor_rigid_body_states(envs[0], dvrk_handles[0], gymapi.STATE_ALL)[-3])
    gym.render_all_camera_sensors(sim)
    pcd = get_partial_pointcloud_vectorized(gym, sim, envs[0], cam_handles[0], cam_prop, color=[1,0,0], min_z = 0.01, visualization=False)
    pcd = np.expand_dims(pcd, axis=0) # shape (1, n, d)
    print(f"pcd shape: {pcd.shape}")
    down_sampled_pcd = farthest_point_sample_batched(point=pcd, npoint=256)
    down_sampled_pcd = np.squeeze(down_sampled_pcd, axis=0)

    eef_pos = np.array([eef_state["pose"]["p"][0], eef_state["pose"]["p"][1], eef_state["pose"]["p"][2]])
    eef_pos = torch.from_numpy(eef_pos).to(device).float().unsqueeze(0)

    obj_emb = to_obj_emb(AE_model, device, down_sampled_pcd)
    
    pose = gymapi.Transform()
    state = torch.cat((eef_pos, obj_emb), dim=-1)
    print("the shpae of state:" , state.size())
    action = policy.act(state)
    next_eef_pos = [action[0][i].item() for i in range(3)]

    print("=======================================================")
    print(f"#################### eef pos: {eef_pos}")
    print(f"#################### next eef pos (action): {next_eef_pos}")
    print("=======================================================")

    pose.p = gymapi.Vec3(*next_eef_pos)

    cartesian_pose = Pose()
    cartesian_pose.orientation.x = 0
    cartesian_pose.orientation.y = 0.707107
    cartesian_pose.orientation.z = 0.707107
    cartesian_pose.orientation.w = 0
    cartesian_pose.position.x = -pose.p.x
    cartesian_pose.position.y = -pose.p.y
    cartesian_pose.position.z = pose.p.z-ROBOT_Z_OFFSET

    # Set up moveit for the above delta x, y, z
    plan_traj = dc_client.arm_moveit_planner_client(go_home=False, cartesian_goal=cartesian_pose, current_position=robot.get_full_joint_positions())

    if (not plan_traj):
        rospy.logerr('Cannot find moveit plan to shape servo. Ignore this sample.\n')  
        state = "reset"
        plan_config["no_plan"] = True
    else:
        state = "move to goal"
        
    return state, plan_traj

def process_pc(partial_pc):
    pcd = np.expand_dims(partial_pc, axis=0) # shape (1, n, d)
    down_sampled_pcd = farthest_point_sample_batched(point=pcd, npoint=256)
    down_sampled_pcd = np.squeeze(down_sampled_pcd, axis=0)
    
    assert(down_sampled_pcd.shape[0]==256)

    return down_sampled_pcd

def move_to_goal_state(sim_cache, plan_config, data_config, robot, plan_traj, args, record_pc=True):
    state = "move to goal"
    rospy.loginfo("**Current state: " + state) 
    envs = sim_cache["envs"]
    gym = sim_cache["gym"]
    dvrk_handles = sim_cache["dvrk_handles"]
    
    cam_handles = sim_cache["cam_handles"]
    cam_prop = sim_cache["cam_prop"]

    group_count = data_config["group_count"]
    sample_count = data_config["sample_count"]
    data_recording_path = data_config["data_recording_path"]

    
    traj_index = 0
    save_frame = 5
    while traj_index <= len(plan_traj):
        step_physics(sim_cache)

        # save data every save_frame frames
        eef_state = deepcopy(gym.get_actor_rigid_body_states(envs[0], dvrk_handles[0], gymapi.STATE_ALL)[-3])
        if plan_config["frame"]%save_frame==0:
            plan_config["eef_states"].append(eef_state)
            plan_config["balls_xyzs_list"].append(deepcopy(sim_cache["all_balls_xyz"]))
            if record_pc:
                gym.render_all_camera_sensors(sim)
                pcd = get_partial_pointcloud_vectorized(gym, sim, envs[0], cam_handles[0], cam_prop, color=[1,0,0], min_z = 0.005, visualization=False, min_depth=-1) 
                pcd =process_pc(pcd)
                plan_config["pcds"].append(pcd)
                assert(len(plan_config["eef_states"])==len(plan_config["pcds"]))
            
            plan_config["last_ball_xyzs"].append(plan_config["last_ball_xyz"])

            drop_ball(sim_cache, plan_config, eef_state, overlap_tolerance=OVERLAP_TOLERANCE)


        plan_config["frame"]+=1


        # Determine if time is out and if it is the end of trajectory
        if plan_config["frame"] < plan_config["max_frame"] - 1:
            if traj_index == len(plan_traj):
                rospy.loginfo("Succesfully executed moveit arm plan")
                state = "get shape servo plan"
                return state
        else:
            rospy.logwarn(f"TIME'S OUT")  
            rospy.loginfo(f"++++++++++++++++ behavioral cloning test: Done group {group_count} sample {sample_count}")
            rospy.loginfo("Let's record demos!!")

            # save data at the end of the sample
            plan_config["eef_states"].append(eef_state)
            plan_config["balls_xyzs_list"].append(deepcopy(sim_cache["all_balls_xyz"]))
            if record_pc:
                gym.render_all_camera_sensors(sim)
                pcd = get_partial_pointcloud_vectorized(gym, sim, envs[0], cam_handles[0], cam_prop, color=[1,0,0], min_z = 0.005, visualization=False, min_depth=-1) 
                pcd =process_pc(pcd)
                plan_config["pcds"].append(pcd)
                assert(len(plan_config["eef_states"])==len(plan_config["pcds"]))
            
            plan_config["last_ball_xyzs"].append(plan_config["last_ball_xyz"])

            if save_frame != 1:
                assert(len(plan_config["eef_states"])==plan_config["max_frame"]//save_frame+1)
            elif save_frame == 1:
                assert(len(plan_config["eef_states"])==plan_config["max_frame"]//save_frame)
            
            if args.save_data:
                data = {"eef_states": plan_config["eef_states"], "pcds":plan_config["pcds"], "last_ball_xyzs": plan_config["last_ball_xyzs"], \
                    "balls_xyz":sim_cache["balls_xyz_init"], "num_balls_reached":plan_config["num_balls_reached"], "balls_xyzs_list": plan_config["balls_xyzs_list"],\
                    "which_balls_reached": plan_config["which_balls_reached"], "eef_ground_z_offset": EEF_GROUND_Z_OFFSET
                }

                with open(os.path.join(data_recording_path, f"group {group_count} sample {sample_count}.pickle"), 'wb') as handle:
                        pickle.dump(data, handle, protocol=3)   
            
            print("##################### end of traj ########################")
            print("###### num balls reached: ", plan_config["num_balls_reached"]) 
            print("###### Len trajectory: ", len(plan_config["eef_states"])) 
            print("##################### end of traj ########################")

            data_config["sample_count"] += 1 
            
            #print("############## not perfect demo, collect again !")
            state = "reset"
            return state

        # Set target joint positions
        dof_states = robot.get_full_joint_positions()
        plan_traj_with_gripper = [plan+[0.35,-0.35] for plan in plan_traj]
        pos_targets = np.array(plan_traj_with_gripper[traj_index], dtype=np.float32)
        gym.set_actor_dof_position_targets(envs[0], dvrk_handles[0], pos_targets)                

        if traj_index <= len(plan_traj) - 2:
            if np.allclose(dof_states[:8], pos_targets[:8], rtol=0, atol=0.1):
                traj_index += 1 
        else:
            if np.allclose(dof_states[:8], pos_targets[:8], rtol=0, atol=0.02):
                traj_index += 1   
                # drop_ball(sim_cache, plan_config, eef_state)

        step_rendering(sim_cache, args)

def reset_state(sim_cache, data_config, init_robot_state, args):
    envs = sim_cache["envs"]
    gym = sim_cache["gym"]
    dvrk_handles = sim_cache["dvrk_handles"]

    rospy.logwarn("==== RESETTING ====")

    gym.set_actor_rigid_body_states(envs[0], dvrk_handles[0], init_robot_state, gymapi.STATE_ALL) 
    print("Sucessfully reset robot")

    sim_cache["all_balls_xyz"] = np.copy(sim_cache["balls_xyz_init"])

    state = "config"

    return state



if __name__ == "__main__":
    gym = gymapi.acquire_gym() # initialize gym

    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--data_recording_path', type=str, help="where you want to record data")
    parser.add_argument('--rigid_state_path', type=str, help="root path of the rigid state")
    parser.add_argument('--group_count', default=0, type=int, help="the current group the data is collecting for")
    parser.add_argument('--num_samples', default=30, type=int, help="number of samples per group you want to collect")
    parser.add_argument('--overlap_tolerance', default=0.0001, type=float, help="threshold determining overlapping of eef and ball")
    parser.add_argument('--headless', default="False", type=str, help="run without viewer?")
    parser.add_argument('--save_data', default="False", type=str, help="True: save recorded data to pickles files")
    parser.add_argument('--record_pc', default="True", type=str, help="True: save recorded data to pickles files")
    parser.add_argument('--rand_seed', default=2021, type=int, help="random seed")
    parser.add_argument('--object_name', type=str, help="name of the saved deformable object in rigid state")

    parser.add_argument('--BC_model_path', type=str, help="policy")
    parser.add_argument('--AE_model_path', type=str, help="autoencoder")
    
    args = parser.parse_args()

    OVERLAP_TOLERANCE = args.overlap_tolerance

    args.headless = args.headless == "True"
    args.save_data = args.save_data == "True"
    args.record_pc = args.record_pc == "True"
    data_recording_path = args.data_recording_path
    os.makedirs(data_recording_path, exist_ok=True)

    BC_model_path = args.BC_model_path
    AE_model_path = args.AE_model_path

    rand_seed = args.rand_seed
    np.random.seed(rand_seed)
    random.seed(rand_seed)

    max_sample_count = args.num_samples
    group_count = args.group_count

    '''
    Main simulation stuff starts from here
    '''
    rospy.init_node('shape_servo_control')
    envs, sim, dvrk_handles, soft_handles, cam_handles, cam_prop, viewer, balls_relative_xyz, soft_xyz = configure_isaacgym(gym, args)

    # compute the absolute xyz pos of the ball(s)
    print(f"@@@ balls relative xyz: {balls_relative_xyz}")
    balls_xyz = np.copy(balls_relative_xyz)
    soft_xyz = np.array(soft_xyz)
    balls_xyz[:, :2] = balls_xyz[:, :2] + soft_xyz[:2]
    print(f"@@@@@@@@@@@@@@@@@@ balls absolute xyz: {balls_xyz} ")
    balls_xyz[:, 2]+=EEF_GROUND_Z_OFFSET
    print(f"@@@@@@@@balls absolute xyz with eefGroundZOffset: {balls_xyz} ")

    init_dvrk_joints(gym, envs[0], dvrk_handles[0], joint_angles=[0,0,0,0,0.20,0,0,0,0.35,-0.35])  # Initialize robot's joints    

    state = "home"

    sim_cache = sim_cache = {"gym":gym, "sim":sim, "envs":envs, "dvrk_handles":dvrk_handles, "soft_handles":soft_handles, "cam_handles":cam_handles, "cam_prop": cam_prop, "viewer":viewer, \
    "soft_xyz": soft_xyz, "balls_relative_xyz": balls_relative_xyz, "all_balls_xyz": np.copy(balls_xyz), "balls_xyz_init": np.copy(balls_xyz), "eef_ground_z_offset": EEF_GROUND_Z_OFFSET}


    data_config = {"data_recording_path": data_recording_path, "balls_relative_xyz": balls_relative_xyz, \
                  "sample_count":0, "max_sample_count":max_sample_count, "group_count": group_count}

    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    policy = ActorPC(act_dim=3, robot_state_dim=3, emb_dim=256, initial_std=1.0).to(device)
    policy.load_state_dict(torch.load(BC_model_path))
    policy.eval()

    AE_model = AutoEncoder(num_points=256, embedding_size=256).to(device)
    AE_model.load_state_dict(torch.load(AE_model_path))
    AE_model.eval()

    dc_client = GraspClient()
    start_time = timeit.default_timer()   
    robot = Robot(gym, sim, envs[0], dvrk_handles[0])


    while (True): 
        step_physics(sim_cache)
        if state == "home" :   
            state, init_robot_state = home_state(sim_cache, robot, dc_client, args)
        if state == "config":
            state, plan_config = config_state(sim_cache, data_config)
        if state == "get shape servo plan":
            # get plan to go to the next checkpoint
            state, plan_traj = get_shape_servo_plan_state(sim_cache, plan_config, robot, dc_client, device, AE_model, policy)
        if state == "move to goal":
            state = move_to_goal_state(sim_cache, plan_config, data_config, robot, plan_traj, args, record_pc=args.record_pc)
        if state == "reset":
            state = reset_state(sim_cache, data_config, init_robot_state, args)
        if data_config["sample_count"] >= data_config["max_sample_count"]:
            break
        step_rendering(sim_cache, args)


    print("All done !")
    print("Elapsed time", timeit.default_timer() - start_time)
    if not args.headless:
        gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)









