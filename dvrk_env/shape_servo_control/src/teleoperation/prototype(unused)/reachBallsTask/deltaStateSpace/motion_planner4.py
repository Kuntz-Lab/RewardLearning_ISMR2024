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

    asset_root = "/home/dvrk/dvrk_ws/src/dvrk_env"
    dvrk_asset_file = "dvrk_description/psm/psm_for_issacgym.urdf"
    print("Loading asset '%s' from '%s'" % (dvrk_asset_file, asset_root))
    return gym.load_asset(sim, asset_root, dvrk_asset_file, asset_options)

def configure_isaacgym(args):
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
    pose = list(np.array(init_pose) + np.random.uniform(low=[-0.1,-0.03,-0.05], high=[0.1,0.02,0.05], size=3))
    soft_pose.p = gymapi.Vec3(*pose)
    soft_thickness = 0.0005    # important to add some thickness to the soft body to avoid interpenetrations

    goal_pose = gymapi.Transform()   
    goal_pose.p = gymapi.Vec3(*list(np.array(init_pose) + np.random.uniform(low=[-0.1,-0.03,-0.05], high=[0.1,0.02,0.05], size=3))) 


    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.thickness = soft_thickness
    asset_options.disable_gravity = True

    soft_asset = gym.load_asset(sim, asset_root, soft_asset_file, asset_options)       

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
        ball_actor = gym.create_actor(env, soft_asset, soft_pose, "soft", i+1, 1)
        color = gymapi.Vec3(0,1,0)
        gym.set_rigid_body_color(env, ball_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
        object_handles.append(ball_actor)

        goal_actor = gym.create_actor(env, soft_asset, goal_pose, "soft", i+1, 1)
        color = gymapi.Vec3(1,0,0)
        gym.set_rigid_body_color(env, goal_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
        object_handles.append(goal_actor)

    # DOF Properties and Drive Modes 
    dof_props = gym.get_actor_dof_properties(envs[0], dvrk_handles[0])
    dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
    dof_props["stiffness"].fill(200.0)
    dof_props["damping"].fill(40.0)
    dof_props["stiffness"][8:].fill(1)
    dof_props["damping"][8:].fill(2)  
    
    for env in envs:
        gym.set_actor_dof_properties(env, dvrk_handles[i], dof_props)    # set dof properties 


    # Viewer camera setup
    if not args.headless:
        cam_pos = gymapi.Vec3(0.4, -0.7, 0.4)
        cam_target = gymapi.Vec3(0.0, -0.36, 0.1)

        # cam_pos = gymapi.Vec3(0.0, -0.440001, 1)
        # cam_target = gymapi.Vec3(0.0, -0.44, 0.1)

        middle_env = envs[num_envs // 2 + num_per_row // 2]
        gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)

    
    # Camera for point cloud setup
    cam_handles = []
    cam_width = 2000
    cam_height = 2000 
    # cam_positions = gymapi.Vec3(0.17, -0.62, 0.2)
    # cam_targets = gymapi.Vec3(0.0, 0.40-0.86, 0.01)
    cam_positions = gymapi.Vec3(0.0, -0.4400001, 0.7)
    cam_targets = gymapi.Vec3(0.0, -0.44, 0.1)

    cam_handle, cam_prop = setup_cam(gym, envs[0], cam_width, cam_height, cam_positions, cam_targets)
    for i, env in enumerate(envs):
        cam_handles.append(cam_handle)

    if not args.headless:
        return envs, sim, dvrk_handles, object_handles, cam_handles, viewer, soft_pose, goal_pose, init_pose
    else:
        return envs, sim, dvrk_handles, object_handles, cam_handles, None, soft_pose, goal_pose, init_pose

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



def home_state(sim_cache, args):
    state = "home"
    envs = sim_cache["envs"]
    gym = sim_cache["gym"]
    dvrk_handles = sim_cache["dvrk_handles"]
    soft_pose = sim_cache["soft_pose"]
    goal_pose = sim_cache["goal_pose"]

    for frame_count in range(50):
        step_physics(sim_cache)
        gym.set_joint_target_position(envs[0], gym.get_joint_handle(envs[0], "dvrk", "psm_main_insertion_joint"), 0.20)            
        if frame_count == 49:
            rospy.loginfo("**Current state: " + state)
            # # Save robot and object states for reset 
            init_robot_state = deepcopy(gym.get_actor_rigid_body_states(envs[0], dvrk_handles[0], gymapi.STATE_ALL))
            #end-effector pose           
            current_pose = deepcopy(gym.get_actor_rigid_body_states(envs[0], dvrk_handles[0], gymapi.STATE_POS)[-3])

            delta_to_ball = [(current_pose["pose"]["p"]["x"] - soft_pose.p.x),
                            (current_pose["pose"]["p"]["y"] - soft_pose.p.y),
                            -(current_pose["pose"]["p"]["z"] - soft_pose.p.z)]

            delta_to_goal = [(current_pose["pose"]["p"]["x"] - goal_pose.p.x),
                            (current_pose["pose"]["p"]["y"] - goal_pose.p.y),
                            -(current_pose["pose"]["p"]["z"] - goal_pose.p.z)]

            print("current pose:", current_pose["pose"]["p"])
            # Go to next state
            state = "config"
        step_rendering(sim_cache, args)

    return state, delta_to_ball, delta_to_goal, init_robot_state, current_pose



def config_state():
    num_cp = 0
    max_cp = np.random.randint(low=1, high=5) # from 1 to 4 
    reach_goal = random.choice([True, False]) #reach red ball?
    reach_checkpoint = random.choice([True, False]) #reach green ball?
    cp_idx = np.random.randint(low=0, high=max_cp) # from 0 to mp-1            
    rospy.logwarn(f"Num cp {max_cp}; Goal? {reach_goal}; Checkpoint? {reach_checkpoint}")    
    state = "get shape servo plan"

    plan_config = {"reach_goal":reach_goal, "reach_checkpoint":reach_checkpoint, \
                    "max_cp":max_cp, "cp_idx":cp_idx, "num_cp": num_cp}

    return state, plan_config



def get_shape_servo_plan_state(plan_config, robot, delta_to_ball, delta_to_goal, current_pose, dc_client):
    state = "get shape servo plan"
    rospy.loginfo("**Current state: " + state) 
    reach_goal = plan_config["reach_goal"]
    reach_checkpoint = plan_config["reach_checkpoint"]
    max_cp = plan_config["max_cp"]
    cp_idx = plan_config["cp_idx"]
    num_cp = plan_config["num_cp"]

    if num_cp <= max_cp -1:
        if reach_checkpoint and num_cp == cp_idx:
            delta_x, delta_y, delta_z = delta_to_ball[0], delta_to_ball[1], delta_to_ball[2]
        else:
            delta_x = np.random.uniform(low = -0.1 , high = 0.1)
            delta_y = np.random.uniform(low = 0 , high = 0.1)
            delta_z = np.random.uniform(low = -0.15 , high = 0.1)                
    else:
        # go to goal after going to all the checkpoints
        if reach_goal:
            delta_x, delta_y, delta_z = delta_to_goal[0], delta_to_goal[1], delta_to_goal[2]

        else:
            delta_x = np.random.uniform(low = -0.1 , high = 0.1)
            delta_y = np.random.uniform(low = 0 , high = 0.1)
            delta_z = np.random.uniform(low = -0.15 , high = 0.1)                            


    cartesian_pose = Pose()
    cartesian_pose.orientation.x = 0
    cartesian_pose.orientation.y = 0.707107
    cartesian_pose.orientation.z = 0.707107
    cartesian_pose.orientation.w = 0
    cartesian_pose.position.x = -current_pose["pose"]["p"]["x"] + delta_x
    cartesian_pose.position.y = -current_pose["pose"]["p"]["y"] + delta_y
    cartesian_pose.position.z = current_pose["pose"]["p"]["z"] - ROBOT_Z_OFFSET + delta_z

    # Set up moveit for the above delta x, y, z
    plan_traj = dc_client.arm_moveit_planner_client(go_home=False, cartesian_goal=cartesian_pose, current_position=robot.get_full_joint_positions())
    if (not plan_traj):
        rospy.logerr('Can not find moveit plan to shape servo. Ignore this grasp.\n')  
        state = "reset"
    else:
        state = "move to goal"
        rospy.loginfo(f"current cp: {num_cp}, max cp: {max_cp}")
    
    return state, plan_traj



def append_was_cp_reached(plan_config, cp_reached_history, traj_index, plan_traj):
    reach_checkpoint = plan_config["reach_checkpoint"]
    cp_idx = plan_config["cp_idx"]
    num_cp = plan_config["num_cp"]

    if num_cp < cp_idx:
        cp_reached_history.append(False)
    elif num_cp > cp_idx:
        if reach_checkpoint:
            cp_reached_history.append(True)
            print("++++++++++++++++++++++++after cp is reached")
        else:
            cp_reached_history.append(False)
    elif num_cp == cp_idx:
        if traj_index < len(plan_traj)-1:
            cp_reached_history.append(False)
        else:
            if reach_checkpoint:
                cp_reached_history.append(True)
                print("++++++++++++++++++++++++cp is reached")
            else:
                cp_reached_history.append(False)


def move_to_goal_state(sim_cache, plan_config, data_config, robot, plan_traj, eef_states, cp_reached_history, args):
    envs = sim_cache["envs"]
    gym = sim_cache["gym"]
    dvrk_handles = sim_cache["dvrk_handles"]
    soft_pose = sim_cache["soft_pose"]
    goal_pose = sim_cache["goal_pose"]
    data_recording_path = data_config["data_recording_path"]
    group_count = data_config["group_count"]
    sample_count = data_config["sample_count"]

    frame_count = 0
    traj_index = 0
    while traj_index < len(plan_traj):
        step_physics(sim_cache)
        if frame_count % 5 == 0:
            eef_states.append(deepcopy(gym.get_actor_rigid_body_states(envs[0], dvrk_handles[0], gymapi.STATE_ALL)[-3]))
            append_was_cp_reached(plan_config, cp_reached_history, traj_index, plan_traj)
            
        frame_count+=1

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

        if traj_index == len(plan_traj):
            rospy.loginfo("Succesfully executed moveit arm plan. Let's record demos!!") 
            
            eef_states.append(deepcopy(gym.get_actor_rigid_body_states(envs[0], dvrk_handles[0], gymapi.STATE_ALL)[-3]))
            append_was_cp_reached(plan_config, cp_reached_history, traj_index, plan_traj)

            assert(len(cp_reached_history)==len(eef_states))          

            if plan_config["num_cp"] <=  plan_config["max_cp"] -1:
                plan_config["num_cp"] += 1
                state = "get shape servo plan"
                return state
                
            else:
                # plan_config["num_cp"] = 0
                if args.save_data:
                    
                    ee_pose = deepcopy(gym.get_actor_rigid_body_states(envs[0], dvrk_handles[0], gymapi.STATE_ALL)[-3])
                    final_pose = np.array(list(ee_pose["pose"]["p"]))

                    success_goal = np.allclose(final_pose, np.array([goal_pose.p.x, goal_pose.p.y, goal_pose.p.z]), \
                                        rtol=0, atol=0.02) or plan_config["reach_goal"]

                    rospy.loginfo(f"Success reach goal? {success_goal}")

                    data = {"traj": eef_states,\
                            "mid pose": [soft_pose.p.x, soft_pose.p.y, soft_pose.p.z],\
                            "success goal": success_goal, "success mid": plan_config["reach_checkpoint"], \
                            "goal pose": [goal_pose.p.x, goal_pose.p.y, goal_pose.p.z],\
                            "num cp": plan_config["max_cp"], "cp was reached": cp_reached_history}                        
                    
                    with open(os.path.join(data_recording_path, f"group {group_count} sample {sample_count}.pickle"), 'wb') as handle:
                        pickle.dump(data, handle, protocol=3)   
                    
                    print(f"Len trajectory: {len(eef_states)}")              
                
                    rospy.loginfo(f"++++++++++++++++Done group {group_count} sample {sample_count}")
                    data_config["sample_count"] += 1 
                
                state = "reset"

                return state

        step_rendering(sim_cache, args)

def reset_state(sim_cache, plan_config, data_config, eef_states, cp_reached_history, init_robot_state):
    envs = sim_cache["envs"]
    gym = sim_cache["gym"]
    dvrk_handles = sim_cache["dvrk_handles"]
    object_handles = sim_cache["object_handles"]
    soft_pose = sim_cache["soft_pose"]
    goal_pose = sim_cache["goal_pose"]

    rospy.logwarn("==== RESETTING ====")
    eef_states.clear()
    cp_reached_history.clear()
    plan_config["num_cp"] = 0

    gym.set_actor_rigid_body_states(envs[0], dvrk_handles[0], init_robot_state, gymapi.STATE_ALL) 
    print("Sucessfully reset robot")   

    state = "config"         

    if data_config["sample_count"] >= data_config["max_sample_count"]:
        data_config["sample_count"] = 0
        data_config["group_count"] += 1
        mid_state = gym.get_actor_rigid_body_states(envs[0], object_handles[0], gymapi.STATE_POS)   
        goal_state = gym.get_actor_rigid_body_states(envs[0], object_handles[1], gymapi.STATE_POS) 
        init_pose = sim_cache["init_pose"]             

        new_pose = np.array(init_pose) + np.random.uniform(low=[-0.1,-0.03,-0.05], high=[0.1,0.02,0.05], size=3)
        soft_pose.p.x, soft_pose.p.y, soft_pose.p.z = new_pose[0], new_pose[1], new_pose[2] 
        mid_state['pose']['p']['x'] = new_pose[0]    
        mid_state['pose']['p']['y'] = new_pose[1]
        mid_state['pose']['p']['z'] = new_pose[2]    

        sim_cache["soft_pose"].p.x = new_pose[0] 
        sim_cache["soft_pose"].p.y = new_pose[1] 
        sim_cache["soft_pose"].p.z = new_pose[2]      

        new_pose = np.array(init_pose) + np.random.uniform(low=[-0.1,-0.03,-0.05], high=[0.1,0.02,0.05], size=3)
        goal_pose.p.x, goal_pose.p.y, goal_pose.p.z = new_pose[0], new_pose[1], new_pose[2] 
        goal_state['pose']['p']['x'] = new_pose[0]    
        goal_state['pose']['p']['y'] = new_pose[1]
        goal_state['pose']['p']['z'] = new_pose[2]      

        sim_cache["goal_pose"].p.x = new_pose[0] 
        sim_cache["goal_pose"].p.y = new_pose[1] 
        sim_cache["goal_pose"].p.z = new_pose[2] 

        gym.set_actor_rigid_body_states(envs[0], object_handles[0], mid_state, gymapi.STATE_ALL)        
        gym.set_actor_rigid_body_states(envs[0], object_handles[1], goal_state, gymapi.STATE_ALL)  

        gym.set_actor_dof_position_targets(robot.env_handle, robot.robot_handle, [0,0,0,0,0.20,0,0,0,0.35,-0.35]) 

        
        rospy.logwarn("Successfully reset ball")        
        state = "home"

    return state







if __name__ == "__main__":
     # initialize gym
    gym = gymapi.acquire_gym()

    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--headless', default="False", type=str, help="Index of grasp candidate to test")
    parser.add_argument('--save_data', default="False", type=str, help="True: save recorded data to pickles files")
    args = parser.parse_args()
    args.headless = args.headless == "True"
    args.save_data = args.save_data == "True"
    

    envs, sim, dvrk_handles, object_handles, cam_handles, viewer, soft_pose, goal_pose, init_pose = configure_isaacgym(args)
   
    '''
    Main simulation stuff starts from here
    '''
    rospy.init_node('shape_servo_control')
    rospy.logerr(f"Save data: {args.save_data}")

    # Some important paramters
    init_dvrk_joints(gym, envs[0], dvrk_handles[0], joint_angles=[0,0,0,0,0.20,0,0,0,0.35,-0.35])  # Initialize robot's joints    


    state = "home"

    sim_cache = {"gym":gym, "sim":sim, "envs":envs, "dvrk_handles":dvrk_handles, \
                "object_handles":object_handles, "cam_handles":cam_handles, "viewer":viewer,\
                "soft_pose": soft_pose, "goal_pose": goal_pose, "init_pose": init_pose}

    data_recording_path = "/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_3/demos_test" #CHANGE

    data_config = {"sample_count":0, "max_sample_count":20, "group_count":0, \
                "max_group_count":20, "data_recording_path": data_recording_path}
    
    eef_states = []
    cp_reached_history = []

    first_time = True

    dc_client = GraspClient()
    start_time = timeit.default_timer()   
    robot = Robot(gym, sim, envs[0], dvrk_handles[0])

    while (True): 

        if state == "home" :   
            state, delta_to_ball, delta_to_goal, init_robot_state, current_pose = home_state(sim_cache, args)

        if state == "config": 
            # Configure demo:
            eef_states = []
            cp_reached_history = []
            state, plan_config = config_state()

        ############################################################################
        # get shape servo plan: sample random delta x, y, z and set up MoveIt
        ############################################################################
        if state == "get shape servo plan":
            # get plan to go to the next checkpoint
            state, plan_traj = get_shape_servo_plan_state(plan_config, robot, delta_to_ball, delta_to_goal, current_pose, dc_client)

        ############################################################################
        # move to goal: Move robot gripper to the desired delta x, y, z using MoveIt
        ############################################################################
        if state == "move to goal":
            # go to the next checkpoint
            state = move_to_goal_state(sim_cache, plan_config, data_config, robot, plan_traj, eef_states, cp_reached_history, args)
            
        if data_config["sample_count"] >= data_config["max_sample_count"]:
            state = "reset"
            
        ############################################################################
        # grasp object: Reset robot and object to the initial state
        ############################################################################  
        if state == "reset":   
            state = reset_state(sim_cache, plan_config, data_config, eef_states, cp_reached_history, init_robot_state)

        if data_config["group_count"] >= data_config["max_group_count"]:
            break

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











