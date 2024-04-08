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
# from behaviors import MoveToPose, TaskVelocityControl
import argparse
# from PIL import Image
import random

sys.path.append("../../pc_utils")
sys.path.append("../config_utils")
from curve import is_overlap, get_balls_xyz_curve
from get_isaac_partial_pc import get_partial_pointcloud_vectorized

'''
collecting data of robot tracing a curve made of small spheres autonomously for visualizing preferences

REMEMBER TO SET:
group
sample
data recording path
record pc
record data
headless
max num balls
checkpoint length
tolerance of overlap e.g. use 0.00025?
init eef pos

very IMPORTANT: 
* how to generate the curve : modify get_balls_xyz_curve
* how far the balls should be apart within the workspace: modify the lower and upper bound of rand_offset in sample_balls_poses
'''

ROBOT_Z_OFFSET = 0.25
MAX_NUM_BALLS = 2#10
OVERLAP_TOLERANCE = 0.0001
CURVE_TYPE = "2ballFlatLinear"


def default_dvrk_asset(gym, sim):
    '''
    load the dvrk asset
    '''
    # dvrk asset
    asset_options = gymapi.AssetOptions()
    asset_options.armature = 0.001
    asset_options.fix_base_link = True
    asset_options.thickness = 0.0005#0.0001

    asset_options.flip_visual_attachments = False
    asset_options.collapse_fixed_joints = True
    asset_options.disable_gravity = True
    asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
    asset_options.max_angular_velocity = 40000.

    asset_root = "/home/dvrk/catkin_ws/src/dvrk_env"
    dvrk_asset_file = "dvrk_description/psm/psm_for_issacgym.urdf"
    print("Loading asset '%s' from '%s'" % (dvrk_asset_file, asset_root))
    return gym.load_asset(sim, asset_root, dvrk_asset_file, asset_options)

def sample_balls_poses(num_balls, curve_type):
    balls_poses = []

    # rand_offset = np.random.uniform(low=0.02, high=0.1)
    # weights_list, xy_curve_weights, balls_xyz = get_balls_xyz_curve(num_balls, degree=degree, offset=rand_offset)
    weights_list, xy_curve_weights, balls_xyz = get_balls_xyz_curve(num_balls, curve_type=curve_type)

    for xyz in balls_xyz:
        ball_pose = gymapi.Transform()
        pose = [xyz[0], xyz[1], xyz[2]]
        ball_pose.p = gymapi.Vec3(*pose)
        balls_poses.append(ball_pose)
        
    return weights_list, xy_curve_weights, balls_xyz, balls_poses

def default_poses():
    poses = []
    for i in range(MAX_NUM_BALLS):
        ball_pose = gymapi.Transform()
        pose = [100, 100, -100]
        ball_pose.p = gymapi.Vec3(*pose)
        poses.append(ball_pose)
    return poses

def set_balls_poses():
    rand_num_balls = MAX_NUM_BALLS
    weights_list, xy_curve_weights, balls_xyz, balls_poses = sample_balls_poses(num_balls=rand_num_balls, curve_type=CURVE_TYPE)
    # Load ball objects with maximum amount
    all_balls_poses = default_poses()
    all_balls_poses[0:rand_num_balls] = balls_poses
    
    return weights_list, xy_curve_weights, balls_xyz, all_balls_poses, rand_num_balls

def recover_ball_poses_from_xyz(sim_cache):
    balls_poses = []
    balls_xyz = sim_cache["balls_xyz"]
    
    for xyz in balls_xyz:
        ball_pose = gymapi.Transform()
        pose = [xyz[0], xyz[1], xyz[2]]
        ball_pose.p = gymapi.Vec3(*pose)
        balls_poses.append(ball_pose)
    return balls_poses

def reset_balls(sim_cache):
    ball_handles = sim_cache["ball_handles"]
    all_balls_poses = default_poses()
    balls_poses = recover_ball_poses_from_xyz(sim_cache)
    num_balls = len(balls_poses)
    all_balls_poses[0:num_balls] = balls_poses

    for i, ball_handle in enumerate(ball_handles):
        ball_state = gym.get_actor_rigid_body_states(envs[0], ball_handle, gymapi.STATE_POS)
        ball_state['pose']['p']['x'] = all_balls_poses[i].p.x
        ball_state['pose']['p']['y'] = all_balls_poses[i].p.y
        ball_state['pose']['p']['z'] = all_balls_poses[i].p.z         
        gym.set_actor_rigid_body_states(envs[0], ball_handle, ball_state, gymapi.STATE_ALL)
        
    sim_cache["all_balls_poses"] = all_balls_poses


def drop_ball(sim_cache, plan_config, eef_state):
    ball_handles = sim_cache["ball_handles"]
    all_balls_poses = sim_cache["all_balls_poses"]
    eef_pose = gymapi.Transform()
    eef_pose.p = gymapi.Vec3(*[eef_state["pose"]["p"]["x"], eef_state["pose"]["p"]["y"], eef_state["pose"]["p"]["z"]])

    for i, ball_handle in enumerate(ball_handles):
        if is_overlap(eef_pose, all_balls_poses[i], max_dist=OVERLAP_TOLERANCE):
            plan_config["last_ball_pose"] = all_balls_poses[i]
            plan_config["num_balls_reached"] += 1
            ball_state = gym.get_actor_rigid_body_states(envs[0], ball_handle, gymapi.STATE_POS)
            ball_state['pose']['p']['x'] = 100
            ball_state['pose']['p']['y'] = 100
            ball_state['pose']['p']['z'] = -100          
            gym.set_actor_rigid_body_states(envs[0], ball_handle, ball_state, gymapi.STATE_ALL)
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(*[100, 100, -100])
            sim_cache["all_balls_poses"][i] = pose
            print(f"++++++++++++++++++++ in contact with ball {i}")


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


    # load robot asset
    dvrk_asset = default_dvrk_asset(gym, sim)
    dvrk_pose = gymapi.Transform()
    dvrk_pose.p = gymapi.Vec3(0.0, 0.0, ROBOT_Z_OFFSET)
    dvrk_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)    

    #################### assume all on the same x-y plane ######################
    weights_list, xy_curve_weights, balls_xyz, all_balls_poses, rand_num_balls = set_balls_poses()

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
    dvrk_handles = []
    ball_handles = []
    ball_name = 5
    for i in range(num_envs):
        # create env
        env = gym.create_env(sim, env_lower, env_upper, num_per_row)
        envs.append(env)

        # add dvrk
        dvrk_handle = gym.create_actor(env, dvrk_asset, dvrk_pose, "dvrk", i+1, 1, segmentationId=11)    
        dvrk_handles.append(dvrk_handle)    
        
        # add ball obj       
        for j in range(MAX_NUM_BALLS):     
            ball_actor = gym.create_actor(env, ball_asset, all_balls_poses[j], f"{ball_name}", ball_name, 1)
            color = gymapi.Vec3(*list(np.random.uniform(low=[0,0,0], high=[1,1,1], size=3)))
            gym.set_rigid_body_color(env, ball_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
            ball_handles.append(ball_actor)
            ball_name += 1


    # DOF Properties and Drive Modes 
    dof_props = gym.get_actor_dof_properties(envs[0], dvrk_handles[0])
    dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
    dof_props["stiffness"].fill(200.0)
    dof_props["damping"].fill(40.0)
    dof_props["stiffness"][8:].fill(1)
    dof_props["damping"][8:].fill(2)  
    
    for i, env in enumerate(envs):
        gym.set_actor_dof_properties(env, dvrk_handles[i], dof_props)    # set dof properties 


    # Viewer camera setup
    if not args.headless:
        cam_target = gymapi.Vec3(0.0, -0.4, 0.05)
        cam_pos = gymapi.Vec3(0.3, -0.8, 0.5)

        middle_env = envs[num_envs // 2 + num_per_row // 2]
        gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)

    
     # Camera for point cloud setup
    cam_handles = []
    cam_width = 600#400
    cam_height = 600#400
    # cam_positions = gymapi.Vec3(0.0, -0.4400001, 0.7)
    cam_targets = gymapi.Vec3(0.0, -0.4, 0.05)
    cam_positions = gymapi.Vec3(0.2, -0.7, 0.2) #0.3, -0.8, 0.5
    # cam_targets = gymapi.Vec3(0.0, -0.44, 0.16+0.015)

    cam_handle, cam_prop = setup_cam(gym, envs[0], cam_width, cam_height, cam_positions, cam_targets)
    for i, env in enumerate(envs):
        cam_handles.append(cam_handle)

    if not args.headless:
        return envs, sim, dvrk_handles, ball_handles, cam_handles, cam_prop, viewer, weights_list, xy_curve_weights, balls_xyz, all_balls_poses, rand_num_balls
    else:
        return envs, sim, dvrk_handles, ball_handles, cam_handles, cam_prop, None, weights_list, xy_curve_weights, balls_xyz, all_balls_poses, rand_num_balls


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

def move_near_workspace(robot, dc_client):
    success = False
    plan_traj = []
    while(not success):
        cartesian_pose = Pose()
        cartesian_pose.orientation.x = 0
        cartesian_pose.orientation.y = 0.707107
        cartesian_pose.orientation.z = 0.707107
        cartesian_pose.orientation.w = 0
        cartesian_pose.position.x = 0.1
        cartesian_pose.position.y = -(-0.48)
        cartesian_pose.position.z = 0.1-ROBOT_Z_OFFSET

        # Set up moveit for the above delta x, y, z
        plan_traj = dc_client.arm_moveit_planner_client(go_home=False, cartesian_goal=cartesian_pose, current_position=robot.get_full_joint_positions())
        if (plan_traj):
           success = True

    
    #print("((((((((plan:", [plan+[0.35,-0.35] for plan in plan_traj])

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
    

def home_state(sim_cache, robot, dc_client, args):
    state = "home"
    envs = sim_cache["envs"]
    gym = sim_cache["gym"]
    dvrk_handles = sim_cache["dvrk_handles"]
    
    for frame_count in range(50):
        step_physics(sim_cache)
        gym.set_joint_target_position(envs[0], gym.get_joint_handle(envs[0], "dvrk", "psm_main_insertion_joint"), 0.20)            
        if frame_count == 49:
            rospy.loginfo("**Current state: " + state)
            # # Save robot and object states for reset 
            init_robot_state = deepcopy(gym.get_actor_rigid_body_states(envs[0], dvrk_handles[0], gymapi.STATE_ALL))
            #end-effector pose           
            current_pose = deepcopy(gym.get_actor_rigid_body_states(envs[0], dvrk_handles[0], gymapi.STATE_POS)[-3])

            print("current pose:", current_pose["pose"]["p"])
            # Go to next state
            state = "config"
        step_rendering(sim_cache, args)

    move_near_workspace(robot, dc_client)
    init_robot_state = deepcopy(gym.get_actor_rigid_body_states(envs[0], dvrk_handles[0], gymapi.STATE_ALL))

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
    
    #############################################################################################
    balls_xyz = sim_cache["balls_xyz"]
    shuffled_balls_xyz = np.copy(balls_xyz)
    np.random.shuffle(shuffled_balls_xyz)
    shuffled_balls_xyz = np.array([balls_xyz[1], balls_xyz[0]])
    print("##### shuffled ball: ", shuffled_balls_xyz)
    checkpoints = []
    epsilon = 0.015
    
    reach_cps = []
    if data_config["sample_count"] == 0:
        ##### reach two balls ####
        checkpoints_length = 2
        num_reach = len(balls_xyz)
        reach_cps = [0,1]
        ball_idx = 0
        for i in range(checkpoints_length):
            if i in reach_cps:
                checkpoints.append(np.array(shuffled_balls_xyz[ball_idx]))
                ball_idx +=1
            else:
                checkpoints.append(np.random.uniform(low=[-0.1, -0.6, 0.011], high=[0.1,-0.4,0.2], size=3))
    elif data_config["sample_count"] == 2:
        ##### stay near a ball ####
        checkpoints_length = 1
        num_reach = 1
        ball_idx = 0
        for i in range(checkpoints_length):
            epsilons = np.array([-epsilon, 0, 0])
            checkpoints.append(np.array(shuffled_balls_xyz[ball_idx])+epsilons)
            ball_idx +=1
    elif data_config["sample_count"] == 1:
        ##### stay near another ball after reaching a ball####
        checkpoints_length = 2
        ball_idx = 0
        reach_cps = [0]
        for i in range(checkpoints_length):
            if i==0:
                checkpoints.append(np.array(shuffled_balls_xyz[ball_idx]))
                ball_idx +=1
            elif i==1:
                epsilons = np.array([epsilon, 0, 0])
                checkpoints.append(np.array(shuffled_balls_xyz[ball_idx])+epsilons)
                ball_idx +=1
    elif data_config["sample_count"] == 3:
         ##### not reaching anything ####
        checkpoints_length = 5
        num_reach = 0
        for i in range(checkpoints_length):
            checkpoints.append(np.random.uniform(low=[-0.1, -0.6, 0.011], high=[0.1,-0.4,0.2], size=3))

    state = "get shape servo plan"

    plan_config = {"checkpoints": checkpoints, "current_cp":0, "reach_cps": reach_cps,
                  "eef_states":[], "pcds":[], "balls_poses_list": [], "last_ball_poses": [], "frame":0, "max_frame":210, "num_balls_reached": 0, "stay":False, "last_ball_pose":None} #max_frame = 180
    

    return state, plan_config


def get_shape_servo_plan_state(sim_cache, plan_config, robot, dc_client):
    state = "get shape servo plan"
    rospy.loginfo("**Current state: " + state) 

    current_cp = plan_config["current_cp"]
    checkpoints = plan_config["checkpoints"]
    pose = gymapi.Transform()

    pose.p = gymapi.Vec3(*checkpoints[current_cp])
    
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

    if data_config["sample_count"] == 3 or data_config["sample_count"] == 2:
        for xyz in sim_cache["balls_xyz"]:
             dc_client.add_obstacle_client(remove_obs=False, x=-xyz[0], y=-xyz[1], z=xyz[2]-ROBOT_Z_OFFSET, rot_x=0, rot_y=0, rot_z=1, rot_w=0)
    else:
        dc_client.add_obstacle_client(remove_obs=True)

    if (not plan_traj):
        rospy.logerr('Cannot find moveit plan to shape servo. Ignore this sample.\n')  
        state = "reset"
    else:
        state = "move to goal"
        
    return state, plan_traj


def move_to_goal_state(sim_cache, plan_config, data_config, robot, plan_traj, args, record_pc=True):
    state = "move to goal"
    rospy.loginfo("**Current state: " + state) 
    envs = sim_cache["envs"]
    gym = sim_cache["gym"]
    dvrk_handles = sim_cache["dvrk_handles"]
    # for recording point clouds or images
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
            #rospy.loginfo("+++++++++++++++++ recording partial pc and poses")
            plan_config["eef_states"].append(eef_state)
            plan_config["balls_poses_list"].append(deepcopy(sim_cache["all_balls_poses"]))
            if record_pc:
                gym.render_all_camera_sensors(sim)
                pcd = get_partial_pointcloud_vectorized(gym, sim, envs[0], cam_handles[0], cam_prop, color=[1,0,0], min_z = 0.01, visualization=False)
                #print(f"pcd shape: {pcd.shape}")
                plan_config["pcds"].append(pcd)
                assert(len(plan_config["eef_states"])==len(plan_config["pcds"]))
            
            plan_config["last_ball_poses"].append(plan_config["last_ball_pose"])
            
            drop_ball(sim_cache, plan_config, eef_state)

        plan_config["frame"]+=1


        # Determine if time is out and if it is the end of trajectory
        if plan_config["frame"] < plan_config["max_frame"] - 1:
            if traj_index == len(plan_traj) and plan_config["current_cp"] <  len(plan_config["checkpoints"]) -1:
                # not the last checkpoints and not time out yet
                 plan_config["current_cp"] += 1
                 state = "get shape servo plan"
                 return state
            if (traj_index == len(plan_traj) and plan_config["current_cp"] == len(plan_config["checkpoints"]) -1) or plan_config["stay"]:
                 # last checkpoint and not time out yet
                plan_config["stay"]=True
                rospy.loginfo("Stay put: NOT TIMEOUT YET")
                state = "move to goal"
                return state
        else:
            rospy.loginfo(f"++++++++++++++++Done group {group_count} sample {sample_count}")
            rospy.loginfo("Let's record demos!!")

            # save data at the end of the sample
            plan_config["eef_states"].append(eef_state)
            plan_config["balls_poses_list"].append(deepcopy(sim_cache["all_balls_poses"]))
            if record_pc:
                gym.render_all_camera_sensors(sim)
                pcd = get_partial_pointcloud_vectorized(gym, sim, envs[0], cam_handles[0], cam_prop, color=[1,0,0], min_z = 0.01, visualization=False)
                plan_config["pcds"].append(pcd)
                assert(len(plan_config["eef_states"])==len(plan_config["pcds"]))
                
            plan_config["last_ball_poses"].append(plan_config["last_ball_pose"])

            if save_frame != 1:
                assert(len(plan_config["eef_states"])==plan_config["max_frame"]//save_frame+1)
            elif save_frame == 1:
                assert(len(plan_config["eef_states"])==plan_config["max_frame"]//save_frame)

            if args.save_data:
                data = {"eef_states": plan_config["eef_states"], "pcds":plan_config["pcds"], "last_ball_poses": plan_config["last_ball_poses"], 
                    "weights_list":sim_cache["weights_list"], "xy_curve_weights":sim_cache["xy_curve_weights"]\
                    , "balls_xyz":sim_cache["balls_xyz"], "num_balls_reached":plan_config["num_balls_reached"], "balls_poses_list": plan_config["balls_poses_list"],
                    "rand_num_balls":sim_cache["rand_num_balls"], "reach_cps": plan_config["reach_cps"]
                }

                with open(os.path.join(data_recording_path, f"group {group_count} sample {sample_count}.pickle"), 'wb') as handle:
                        pickle.dump(data, handle, protocol=3)   
                    
            print("Len trajectory: ", len(plan_config["eef_states"])) 

            data_config["sample_count"] += 1 

            state = "reset"
            return state

        #drop_ball(sim_cache, plan_config, eef_state)

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
                #drop_ball(sim_cache, plan_config, eef_state)

        step_rendering(sim_cache, args)


def reset_state(sim_cache, data_config, init_robot_state):
    envs = sim_cache["envs"]
    gym = sim_cache["gym"]
    dvrk_handles = sim_cache["dvrk_handles"]
    ball_handles = sim_cache["ball_handles"]

    rospy.logwarn("==== RESETTING ====")

    gym.set_actor_rigid_body_states(envs[0], dvrk_handles[0], init_robot_state, gymapi.STATE_ALL) 
    print("Sucessfully reset robot")

    reset_balls(sim_cache)
    print("successfully reset balls") 

    state = "config"

    if data_config["sample_count"] >= data_config["max_sample_count"]:
        data_config["sample_count"] = 0
        data_config["group_count"] += 1
        
        weights_list, xy_curve_weights, balls_xyz, all_balls_poses, rand_num_balls = set_balls_poses()

        for i, ball_handle in enumerate(ball_handles):
            ball_state = gym.get_actor_rigid_body_states(envs[0], ball_handle, gymapi.STATE_POS)
            ball_state['pose']['p']['x'] = all_balls_poses[i].p.x
            ball_state['pose']['p']['y'] = all_balls_poses[i].p.y
            ball_state['pose']['p']['z'] = all_balls_poses[i].p.z         
            gym.set_actor_rigid_body_states(envs[0], ball_handle, ball_state, gymapi.STATE_ALL)
        

        sim_cache["all_balls_poses"] = all_balls_poses
        sim_cache["weights_list"] = weights_list
        sim_cache["xy_curve_weights"] = xy_curve_weights
        sim_cache["balls_xyz"] = balls_xyz
        sim_cache["rand_num_balls"] = rand_num_balls

        gym.set_actor_dof_position_targets(robot.env_handle, robot.robot_handle, [0,0,0,0,0.20,0,0,0,0.35,-0.35])        

        rospy.logwarn("Successfully resample and reset balls")        
        state = "home"

    return state

if __name__ == "__main__":

    gym = gymapi.acquire_gym() # initialize gym

    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--data_recording_path', default=f"/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_trace_curve_corrected/ex_traj", type=str, help="where you want to record data")
    parser.add_argument('--headless', default="False", type=str, help="Index of grasp candidate to test")
    parser.add_argument('--save_data', default="True", type=str, help="True: save recorded data to pickles files")
    parser.add_argument('--record_pc', default="True", type=str, help="True: record partial point cloud")
    parser.add_argument('--rand_seed', default=1945, type=int, help="random seed")
    parser.add_argument('--max_num_balls', default=2, type=int, help="number of balls originally in the workspace")
    parser.add_argument('--overlap_tolerance', default=0.0001, type=float, help="number of balls originally in the workspace")
    parser.add_argument('--curve_type', default="2ballFlatLinear", type=str, help="type of curve on which the balls are situated")
    
    args = parser.parse_args()

    MAX_NUM_BALLS = args.max_num_balls
    OVERLAP_TOLERANCE = args.overlap_tolerance
    CURVE_TYPE = args.curve_type

    np.random.seed(args.rand_seed)
    random.seed(args.rand_seed)

    args.headless = args.headless == "True"
    args.save_data = args.save_data == "True"
    args.record_pc = args.record_pc == "True"
    data_recording_path = args.data_recording_path
    os.makedirs(data_recording_path, exist_ok=True)

    envs, sim, dvrk_handles, ball_handles, cam_handles, cam_prop, viewer, weights_list, xy_curve_weights, balls_xyz, all_balls_poses, rand_num_balls = configure_isaacgym(gym, args)
   
    '''
    Main simulation stuff starts from here
    '''
    rospy.init_node('shape_servo_control')
    rospy.logerr(f"Save data: {args.save_data}")

    # Some important paramters
    init_dvrk_joints(gym, envs[0], dvrk_handles[0], joint_angles=[0,0,0,0,0.20,0,0,0,0.35,-0.35])  # Initialize robot's joints    


    state = "home"

    sim_cache = sim_cache = {"gym":gym, "sim":sim, "envs":envs, "dvrk_handles":dvrk_handles, \
                "ball_handles":ball_handles, "cam_handles":cam_handles, "cam_prop": cam_prop, "viewer":viewer,\
                "weights_list":weights_list, "xy_curve_weights":xy_curve_weights, "balls_xyz":balls_xyz,
                "all_balls_poses": all_balls_poses, "rand_num_balls": rand_num_balls}


    data_config = {"sample_count":0, "max_sample_count":4, "group_count":0, \
                "max_group_count":1, "data_recording_path": data_recording_path}

    
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
            state, plan_traj = get_shape_servo_plan_state(sim_cache, plan_config, robot, dc_client)
        if state == "move to goal":
            state = move_to_goal_state(sim_cache, plan_config, data_config, robot, plan_traj, args, record_pc=args.record_pc)
        if state == "reset":
            state = reset_state(sim_cache, data_config, init_robot_state)
        if data_config["group_count"] >= data_config["max_group_count"]:
            break
        step_rendering(sim_cache, args)


    print("All done !")
    print("Elapsed time", timeit.default_timer() - start_time)
    if not args.headless:
        gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)











