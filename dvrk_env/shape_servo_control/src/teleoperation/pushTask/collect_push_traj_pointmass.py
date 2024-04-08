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

import argparse
from PIL import Image
import random

sys.path.append("../pc_utils")
from get_isaac_partial_pc import get_partial_pointcloud_vectorized

'''
collecting data of robot pushing box to a goal autonomously
each sample terminates only when time is out
the agent should not move when it reaches the goal or when it has reached the end of the sample but time is not out yet
'''


OVERLAP_TOLERANCE  = 0.001 #0.0008
THICKNESS = 0.001   #0.0000025
BOX_RADIUS = 0.03
CONE_Z_EXTRA_OFFSET = 0.01 #0.005
EEF_Z = 0.04


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
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = False
    asset_options.disable_gravity = True
    asset_options.thickness = THICKNESS
    asset_options.density = 10000000 #10000 #
    asset_options.armature = 20 #10
    asset_options.use_physx_armature = True

    dvrk_asset = gym.create_sphere(sim, 0.01, asset_options)
    dvrk_pose = gymapi.Transform()
    dvrk_pose.p = gymapi.Vec3(0.1, -0.4, EEF_Z)
    #dvrk_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)    

    #################### assume box and cone on the x-y plane ######################
    # Load box object
    init_pose = [0.0, -0.5, 0.03 + THICKNESS]
    box_pose = gymapi.Transform()
    pose = list(np.array(init_pose) + np.random.uniform(low=[-0.1,-0.05,0], high=[0.1,0.05,0], size=3))
    box_pose.p = gymapi.Vec3(*pose)
    
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = False
    asset_options.disable_gravity = False
    asset_options.thickness = THICKNESS
    asset_options.density = 1000 #10000 #
    asset_options.armature = 20 #10
    asset_options.use_physx_armature = True
    # asset_options.linear_damping = 100000
    # asset_options.angular_damping = 100000

    box_asset = gym.create_sphere(sim, BOX_RADIUS, asset_options) #gym.create_box(sim, *[0.04, 0.04, 0.03], asset_options)#

    # load cone object
    cone_pose = gymapi.Transform()
    pose = list(np.array(init_pose) + np.random.uniform(low=[-0.1,-0.05,0], high=[0.1,0.05,0], size=3))
    pose[2] += CONE_Z_EXTRA_OFFSET
    cone_pose.p = gymapi.Vec3(*pose)

    while is_overlap_xy(cone_pose, box_pose, OVERLAP_TOLERANCE):
        pose = list(np.array(init_pose) + np.random.uniform(low=[-0.1,-0.05,0], high=[0.1,0.05,0], size=3))
        pose[2] += CONE_Z_EXTRA_OFFSET
        cone_pose.p = gymapi.Vec3(*pose)

    cone_asset = gym.create_box(sim, *[0.04, 0.04, 0.08], asset_options) #gym.create_sphere(sim, 0.02, asset_options) #gym.create_box(sim, *[0.04, 0.04, 0.08], asset_options)

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
        
        # add box and cone (goal) obj            
        box_actor = gym.create_actor(env, box_asset, box_pose, "box", i, 0)
        color = gymapi.Vec3(0,1,0)
        gym.set_rigid_body_color(env, box_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
        object_handles.append(box_actor)

        cone_actor = gym.create_actor(env, cone_asset, cone_pose, "cone", i+1, 0)
        object_handles.append(cone_actor)


    # Viewer camera setup
    if not args.headless:
        cam_pos = gymapi.Vec3(0.4, -0.7, 0.4)
        cam_target = gymapi.Vec3(0.0, -0.36, 0.1)

        middle_env = envs[num_envs // 2 + num_per_row // 2]
        gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)

    
    # Camera for point cloud setup
    cam_handles = []
    cam_width = 300 #300
    cam_height = 300 #300
    cam_targets = gymapi.Vec3(0.0, -0.4, 0.02)
    cam_positions = gymapi.Vec3(0.2, -0.7, 0.5)

    cam_handle, cam_prop = setup_cam(gym, envs[0], cam_width, cam_height, cam_positions, cam_targets)
    for i, env in enumerate(envs):
        cam_handles.append(cam_handle)

    if not args.headless:
        return envs, sim, dvrk_handles, object_handles, cam_handles, cam_prop, viewer, box_pose, cone_pose, init_pose#, table_handles
    else:
        return envs, sim, dvrk_handles, object_handles, cam_handles, cam_prop, None, box_pose, cone_pose, init_pose#, table_handles

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

def push_straight_point(box_pose, goal_pose, offset=0.06): #0.1
    '''
    returns which point to go to in order to push the box to the goal
    # default: 0.1 offset
    '''
    if box_pose.p.x <= goal_pose.p.x:
        x = box_pose.p.x - offset
    else:
        x = box_pose.p.x + offset
    y = (box_pose.p.y-goal_pose.p.y)/(box_pose.p.x-goal_pose.p.x)*(x-box_pose.p.x)+box_pose.p.y
    return x, y

def avoid_box_point(eef_pose, box_pose, box_radius):
    '''
    return a point where the eef should go in order to avoid the box before pushing it
    '''
    box_xy = np.array([box_pose.p.x, box_pose.p.y])
    eef_xy = np.array([eef_pose.p.x, eef_pose.p.y])
    box_eef_displace = box_xy - eef_xy
    box_eef_dist = np.linalg.norm(box_eef_displace)
    angle = np.arcsin(box_radius/box_eef_dist) + 0.4
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], \
                                [np.sin(angle), np.cos(angle)]])
    avoid_point_displace = 1.12*np.matmul(rotation_matrix, box_eef_displace)
    avoid_point_xy = avoid_point_displace + eef_xy

    return avoid_point_xy[0], avoid_point_xy[1]

def is_box_blocking(eef_pose, next_eef_pose, box_pose):
    '''
    return whether the box may block the path 
    '''
    box_xy = np.array([box_pose.p.x, box_pose.p.y])
    eef_xy = np.array([eef_pose.p.x, eef_pose.p.y])
    next_eef_xy = np.array([next_eef_pose.p.x, next_eef_pose.p.y])
    box_x_between = ((eef_xy[0]<box_xy[0]) and (box_xy[0]<next_eef_xy[0])) or ((eef_xy[0]>box_xy[0]) and (box_xy[0]>next_eef_xy[0]))
    box_y_between = ((eef_xy[1]<box_xy[1]) and (box_xy[1]<next_eef_xy[1])) or ((eef_xy[1]>box_xy[1]) and (box_xy[1]>next_eef_xy[1]))
    
    return box_x_between and box_y_between


def home_state(sim_cache,args):
    state = "home"
    print("**Current state: " + state) 
    envs = sim_cache["envs"]
    gym = sim_cache["gym"]
    dvrk_handles = sim_cache["dvrk_handles"]
    init_robot_state = deepcopy(gym.get_actor_rigid_body_states(envs[0], dvrk_handles[0], gymapi.STATE_ALL))
    state = "config"
    return state, init_robot_state

def config_state(sim_cache, args):
    state = "config"
    print("**Current state: " + state) 
    num_cp = 0
    max_cp = np.random.randint(low=1, high=2) 
    reach_goal = random.choice([True, False]) #reach cone?
         
    rospy.logwarn(f"Number of cp {max_cp}; Goal? {reach_goal}")    
    state = "get shape servo plan"

    plan_config = {"reach_goal":reach_goal, "max_cp":max_cp, "num_cp": num_cp, "sample_cp": True, "is_near_box": True, "avoid_box": False,
                  "cp_pose":gymapi.Transform(), "max_frame":600, "frame":0, \
                  "eef_states":[], "box_states":[], "pcds":[], "goal_was_reached":False, "stay_before_timeout": False, \
                    "no_plan": False} #max frame 400

    ####### for behavioral cloning purpose
    if args.is_behavioral_cloning:
        plan_config["max_frame"] = 400
        plan_config["max_cp"] = 0
        plan_config["reach_goal"] = True

    return state, plan_config

def is_in_workspace(x, y):
    low_x = x>=-0.1
    up_x = x<=0.1
    up_y = y<=-0.4
    low_y = y>=-0.6

    return low_x and up_x and up_y and low_y

def is_in_workspace_slack(x, y, slack=0.03):
    low_x = x>=-0.1-slack
    up_x = x<=0.1+slack
    up_y = y<=-0.4+slack
    low_y = y>=-0.6-slack

    return low_x and up_x and up_y and low_y


def get_shape_servo_plan_state(sim_cache, plan_config):
    state = "get shape servo plan"
    print("**Current state: " + state) 

    #step_physics(sim_cache)

    reach_goal = plan_config["reach_goal"]
    max_cp = plan_config["max_cp"]
    num_cp = plan_config["num_cp"]
    is_near_box = plan_config["is_near_box"]

    box_state = deepcopy(gym.get_actor_rigid_body_states(envs[0], object_handles[0], gymapi.STATE_POS))
    sim_cache["box_pose"].p.x  = box_state['pose']['p']['x']
    sim_cache["box_pose"].p.y  = box_state['pose']['p']['y'] 
    sim_cache["box_pose"].p.z  = box_state['pose']['p']['z'] 
    box_z = box_state['pose']['p']['z'] 

    box_pose = sim_cache["box_pose"]
    cone_pose = sim_cache["cone_pose"]

    eef_state = deepcopy(gym.get_actor_rigid_body_states(envs[0], dvrk_handles[0], gymapi.STATE_ALL))
    eef_pose = gymapi.Transform()
    eef_pose.p = gymapi.Vec3(*[eef_state['pose']['p']['x'], eef_state['pose']['p']['y'], eef_state['pose']['p']['z']])

    pose = gymapi.Transform()
    offset_from_box = 0.06 #0.06
    min_offset_from_box = 0 #0.04
    max_sample_cp_iter = 50

    print("=============================================================")
    print("box pose: ", f"{box_pose.p.x}, {box_pose.p.y}, {box_pose.p.z}")
   
    if num_cp <= max_cp -1:
        if plan_config["sample_cp"]:
            # sample a random checkpoint
            rand_x = box_pose.p.x+np.random.uniform(low = -offset_from_box , high = offset_from_box)
            rand_y = box_pose.p.y+np.random.uniform(low = -offset_from_box , high = offset_from_box)
            cp_pose = gymapi.Transform()
            cp_pose.p = gymapi.Vec3(rand_x, rand_y, box_z)
            push_x, push_y = push_straight_point(box_pose, cp_pose)
            sample_cp_iter = 0
            while((rand_x-box_pose.p.x)**2<=min_offset_from_box**2 or (rand_y-box_pose.p.y)**2<=min_offset_from_box**2) or not is_in_workspace(rand_x, rand_y) or not is_in_workspace_slack(push_x, push_y):
                sample_cp_iter += 1
                if sample_cp_iter >= max_sample_cp_iter:
                    print("XXXXXXXXXXXXXXXXX NO PLAN XXXXXXXXXXXXXXXXXXXXX")
                    state = "reset"
                    plan_config["no_plan"] = True
                    return state, pose
                print("sampling cp : hi, is in workspace?: ", is_in_workspace(rand_x, rand_y))
                print("x diff", rand_x-box_pose.p.x, "| x", rand_x)
                print("y diff", rand_x-box_pose.p.y, "| y", rand_y)
                print("push x: ", push_x)
                print("push y: ", push_y)
                
                rand_x = box_pose.p.x+np.random.uniform(low = -offset_from_box , high = offset_from_box)
                rand_y = box_pose.p.y+np.random.uniform(low = -offset_from_box , high = offset_from_box)
                
                cp_pose = gymapi.Transform()
                cp_pose.p = gymapi.Vec3(rand_x, rand_y, box_z)
                push_x, push_y = push_straight_point(box_pose, cp_pose)

            print("sampling cp : hi, is in workspace?: ", is_in_workspace(rand_x, rand_y))
            print("x diff", rand_x-box_pose.p.x, "| x", rand_x)
            print("y diff", rand_x-box_pose.p.y, "| y", rand_y)
            print("push x: ", push_x)
            print("push y: ", push_y)
        
            plan_config["cp_pose"] = cp_pose

            plan_config["sample_cp"] = False

        if is_near_box:
            cp_pose = plan_config["cp_pose"]
            
            if plan_config["avoid_box"]:
                    plan_config["avoid_box"] = False
                    print("************ PUSH BOX after AVOID************")
                    # move to the a point on the line connecting the box and the checkpoint
                    x,y =push_straight_point(box_pose, cp_pose)
                    pose.p = gymapi.Vec3(x, y, box_z)
            else:
                x,y =push_straight_point(box_pose, cp_pose)
                push_point_pose = gymapi.Transform()
                push_point_pose.p = gymapi.Vec3(x, y, box_z)
                # if box may be between eef and the next point, avoid box first
                is_blocking = is_box_blocking(eef_pose, push_point_pose, box_pose)
                if is_blocking:
                    print("$$$$$$$$$$$$$ AVOID BOX $$$$$$$$$$$$$$$$$")
                    plan_config["avoid_box"] = True
                    x, y = avoid_box_point(eef_pose, box_pose, BOX_RADIUS)
                    pose.p = gymapi.Vec3(x, y, box_z)
                else:
                    # move to the a point on the line connecting the box and the checkpoint
                    print("************ PUSH BOX WITHOUT AVOID************")
                    x,y =push_straight_point(box_pose, cp_pose)
                    pose.p = gymapi.Vec3(x, y, box_z)
        else:
            print("************ PUSH to checkpoint ************")
            # move to the checkpoint
            pose.p = plan_config["cp_pose"].p
    else:
        if reach_goal:
            if is_near_box:
                # move to the a point on the line connecting the box and the cone
                if plan_config["avoid_box"]:
                    plan_config["avoid_box"] = False
                    print("************ PUSH BOX after AVOID************")
                    # move to the a point on the line connecting the box and the checkpoint
                    x,y =push_straight_point(box_pose, cone_pose)
                    pose.p = gymapi.Vec3(x, y, box_z)
                else:
                    x,y =push_straight_point(box_pose, cone_pose)
                    push_point_pose = gymapi.Transform()
                    push_point_pose.p = gymapi.Vec3(x, y, box_z)
                    # if box may be between eef and the next point, avoid box first
                    is_blocking = is_box_blocking(eef_pose, push_point_pose, box_pose)
                    if is_blocking:
                        print("$$$$$$$$$$$$$ AVOID BOX $$$$$$$$$$$$$$$$$")
                        plan_config["avoid_box"] = True
                        x, y = avoid_box_point(eef_pose, box_pose, BOX_RADIUS)
                        pose.p = gymapi.Vec3(x, y, box_z)
                    else:
                        # move to the a point on the line connecting the box and the checkpoint
                        print("************ PUSH BOX WITHOUT AVOID************")
                        x,y =push_straight_point(box_pose, cone_pose)
                        pose.p = gymapi.Vec3(x, y, box_z)
            else:
                print("************ PUSH to GOAL !!!!!!!!!!!!************")
                # move to the cone
                pose.p = cone_pose.p
        else:
            if is_near_box:
                if plan_config["sample_cp"]:
                    # sample a random checkpoint
                    rand_x = box_pose.p.x+np.random.uniform(low = -offset_from_box , high = offset_from_box)
                    rand_y = box_pose.p.y+np.random.uniform(low = -offset_from_box , high = offset_from_box)
                    cp_pose = gymapi.Transform()
                    cp_pose.p = gymapi.Vec3(rand_x, rand_y, box_z)
                    push_x, push_y = push_straight_point(box_pose, cp_pose)
                    sample_cp_iter = 0
                    while((rand_x-box_pose.p.x)**2<=min_offset_from_box**2 or (rand_y-box_pose.p.y)**2<=min_offset_from_box**2) or not is_in_workspace(rand_x, rand_y)  or not is_in_workspace_slack(push_x, push_y):
                        sample_cp_iter += 1
                        if sample_cp_iter >= max_sample_cp_iter:
                             print("XXXXXXXXXXXXXXXXX NO PLAN XXXXXXXXXXXXXXXXXXXXX")
                             state = "reset"
                             plan_config["no_plan"] = True
                             return state, pose
                        print("final point: sampling cp : hi, is in workspace?: ", is_in_workspace(rand_x, rand_y))
                        print("x diff", rand_x-box_pose.p.x, "| x", rand_x)
                        print("y diff", rand_x-box_pose.p.y, "| y", rand_y)
                        print("push x: ", push_x)
                        print("push y: ", push_y)

                        rand_x = box_pose.p.x+np.random.uniform(low = -offset_from_box , high = offset_from_box)
                        rand_y = box_pose.p.y+np.random.uniform(low = -offset_from_box , high = offset_from_box)

                        cp_pose = gymapi.Transform()
                        cp_pose.p = gymapi.Vec3(rand_x, rand_y, box_z)
                        push_x, push_y = push_straight_point(box_pose, cp_pose)
                        
                    print("sampling cp : hi, is in workspace?: ", is_in_workspace(rand_x, rand_y))
                    print("x diff", rand_x-box_pose.p.x, "| x", rand_x)
                    print("y diff", rand_x-box_pose.p.y, "| y", rand_y)
                    print("push x: ", push_x)
                    print("push y: ", push_y)
                    plan_config["cp_pose"] = cp_pose

                    plan_config["sample_cp"] = False
                
                if is_near_box:
                    cp_pose = plan_config["cp_pose"]
                    
                    if plan_config["avoid_box"]:
                        print("************ PUSH BOX after AVOID************")
                        plan_config["avoid_box"] = False
                        # move to the a point on the line connecting the box and the checkpoint
                        x,y =push_straight_point(box_pose, cp_pose)
                        pose.p = gymapi.Vec3(x, y, box_z)
                    else:
                        x,y =push_straight_point(box_pose, cp_pose)
                        push_point_pose = gymapi.Transform()
                        push_point_pose.p = gymapi.Vec3(x, y, box_z)
                        # if box may be between eef and the next point, avoid box first
                        is_blocking = is_box_blocking(eef_pose, push_point_pose, box_pose)
                        if is_blocking:
                            print("$$$$$$$$$$$$$ AVOID BOX $$$$$$$$$$$$$$$$$")
                            plan_config["avoid_box"] = True
                            x, y = avoid_box_point(eef_pose, box_pose, BOX_RADIUS)
                            pose.p = gymapi.Vec3(x, y, box_z)
                        else:
                            print("************ PUSH BOX WITHOUT AVOID************")
                            # move to the a point on the line connecting the box and the checkpoint
                            x,y =push_straight_point(box_pose, cp_pose)
                            pose.p = gymapi.Vec3(x, y, box_z)
                else:
                    print("************ PUSH to last point (not goal) !!!!!!!!!!!!************")
                    # move to the checkpoint
                    pose.p = plan_config["cp_pose"].p
            
    pose.p.z = EEF_Z
    print(f"target xy pos: {pose.p.x}, {pose.p.y}")
    print("=============================================================")
    
    state = "move to goal"
        
    return state, pose

def is_overlap_xy(p1, p2, max_dist):
    return (p1.p.x-p2.p.x)**2 + (p1.p.y-p2.p.y)**2 <=max_dist
    #return np.allclose(np.array([p1.p.x, p1.p.y, p1.p.z]), np.array([p2.p.x, p2.p.y, p2.p.z]), rtol=0, atol=0.058)

def overlap_xy(p1, p2):
    return (p1.p.x-p2.p.x)**2 + (p1.p.y-p2.p.y)**2

def move_to_goal_state(sim_cache, plan_config, data_config, target_pose, args, record_pc=False):
    state = "move to goal"
    print("**Current state: " + state) 
    envs = sim_cache["envs"]
    gym = sim_cache["gym"]
    dvrk_handles = sim_cache["dvrk_handles"]

    group_count = data_config["group_count"]
    sample_count = data_config["sample_count"]
    data_recording_path = data_config["data_recording_path"]

    # for recording point clouds or images
    cam_handles = sim_cache["cam_handles"]
    cam_prop = sim_cache["cam_prop"]

    save_frame = 5
    eef_state = deepcopy(gym.get_actor_rigid_body_states(envs[0], dvrk_handles[0], gymapi.STATE_ALL))
    eef_xy = np.array([eef_state['pose']['p']['x'][0], eef_state['pose']['p']['y'][0]])
    target_xy = np.array([target_pose.p.x, target_pose.p.y])
    error = np.linalg.norm(eef_xy - target_xy)
    # print("eef_xy: ", eef_xy)
    # print("target_xy: ", target_xy)
    # print("off: ", eef_xy - target_xy)
    # print("error: ", error)
    box_state = deepcopy(gym.get_actor_rigid_body_states(envs[0], object_handles[0], gymapi.STATE_POS))
    error_threshold = 0.001
    while error >= error_threshold or plan_config["goal_was_reached"] or plan_config["stay_before_timeout"]:
        step_physics(sim_cache)

        box_state = deepcopy(gym.get_actor_rigid_body_states(envs[0], object_handles[0], gymapi.STATE_POS))
        cone_state = deepcopy(gym.get_actor_rigid_body_states(envs[0], object_handles[1], gymapi.STATE_POS))
        eef_state = deepcopy(gym.get_actor_rigid_body_states(envs[0], dvrk_handles[0], gymapi.STATE_ALL))
        eef_xy = np.array([eef_state['pose']['p']['x'][0], eef_state['pose']['p']['y'][0]])
        error = np.linalg.norm(eef_xy - target_xy)
        # print("++++++++++++++++")
        # # print("eef_xy: ", eef_xy)
        # # print("target_xy: ", target_xy)
        # print("error: ", error)
        # print("frame: ", plan_config["frame"])

        
        # update box pose
        sim_cache["box_pose"].p.x  = box_state['pose']['p']['x']
        sim_cache["box_pose"].p.y  = box_state['pose']['p']['y'] 
        sim_cache["box_pose"].p.z  = box_state['pose']['p']['z'] 


        # save data every save_frame frames
        if plan_config["frame"]%save_frame==0:
            plan_config["eef_states"].append(deepcopy(eef_state))
            plan_config["box_states"].append(deepcopy(box_state))
            if not is_in_workspace_slack(box_state['pose']['p']['x'], box_state['pose']['p']['y']):
                print("[[[[[[[[[[[[[[[[ box (object being pushed) not in workspace ]]]]]]]]]]]]]]]]]]]]")
                print("x: ", box_state['pose']['p']['x'])
                print("y: ", box_state['pose']['p']['y'])
                state = "reset"
                return state
            elif not is_in_workspace_slack(eef_state['pose']['p']['x'], eef_state['pose']['p']['y']):
                print("[[[[[[[[[[[[[[[[ END-EFFECTOR not in workspace ]]]]]]]]]]]]]]]]]]]]")
                print("x: ", eef_state['pose']['p']['x'])
                print("y: ", eef_state['pose']['p']['y'])
                state = "reset"
                return state
            
            if record_pc:
                gym.render_all_camera_sensors(sim)
                pcd = get_partial_pointcloud_vectorized(gym, sim, envs[0], cam_handles[0], cam_prop, color=[1,0,0], min_z = 0.009, visualization=False)
                #print(pcd.shape)
                plan_config["pcds"].append(pcd)
                assert(len(plan_config["eef_states"])==len(plan_config["pcds"]))

        plan_config["frame"]+=1

        if plan_config["frame"] >= plan_config["max_frame"]-1:
            # timeout, save sample
            plan_config["eef_states"].append(deepcopy(gym.get_actor_rigid_body_states(envs[0], dvrk_handles[0], gymapi.STATE_ALL)))
            plan_config["box_states"].append(deepcopy(box_state))
            if record_pc:
                gym.render_all_camera_sensors(sim)
                pcd = get_partial_pointcloud_vectorized(gym, sim, envs[0], cam_handles[0], cam_prop, color=[1,0,0], min_z = 0.009, visualization=False)
                plan_config["pcds"].append(pcd)
                assert(len(plan_config["eef_states"])==len(plan_config["pcds"]))
            assert(len(plan_config["box_states"])==len(plan_config["eef_states"]))
            print("++++++++++++++++++++++++++++++++++++++ traj_len: ", len(plan_config["box_states"]), " last_frame: ", plan_config["frame"])
            
            if save_frame != 1:
                assert(len(plan_config["eef_states"])==plan_config["max_frame"]//save_frame+1)
            elif save_frame == 1:
                assert(len(plan_config["eef_states"])==plan_config["max_frame"]//save_frame)

            print(f"TIME'S OUT")  
            print(f"++++++++++++++++Done group {group_count} sample {sample_count}")
            print("Let's record demos!!")

            # save data
            success_goal = plan_config["goal_was_reached"]
            data = {"eef_states": plan_config["eef_states"],\
                            "box_states": plan_config["box_states"],\
                            "cone_pose":[sim_cache["cone_pose"].p.x, sim_cache["cone_pose"].p.y, sim_cache["cone_pose"].p.z],\
                            "num_cp": plan_config["max_cp"], "success_goal":success_goal, "pcds":plan_config["pcds"]
                    }
            if args.is_behavioral_cloning:
                if success_goal:
                    if args.save_data:

                        with open(os.path.join(data_recording_path, f"group {group_count} sample {sample_count}.pickle"), 'wb') as handle:
                                pickle.dump(data, handle, protocol=3)   
                            
                        print("Len trajectory: ", len(plan_config["eef_states"]))      

                    data_config["sample_count"] += 1 
            else:
                if args.save_data:

                    with open(os.path.join(data_recording_path, f"group {group_count} sample {sample_count}.pickle"), 'wb') as handle:
                            pickle.dump(data, handle, protocol=3)   
                        
                    print("Len trajectory: ", len(plan_config["eef_states"])) 

                data_config["sample_count"] += 1 

            state = "reset"
            return state
        else:
            if plan_config["stay_before_timeout"]:
                # wait until timeout
                print("Stay put: NOT TIMEOUT YET")
                state = "move to goal"
                step_rendering(sim_cache, args)
                return state

            # stay if close to cone
            if is_overlap_xy(sim_cache["box_pose"], sim_cache["cone_pose"], OVERLAP_TOLERANCE) or plan_config["goal_was_reached"]:
                print(f"STAY PUT: box already reached goal") 
                plan_config["goal_was_reached"]=True
                state = "move to goal"
                step_rendering(sim_cache, args)
                return state

            if error <= error_threshold:
                 eef_state = gym.get_actor_rigid_body_states(envs[0], dvrk_handles[0], gymapi.STATE_ALL)
                 eef_state['vel']['linear']['x'] = 0
                 eef_state['vel']['linear']['y'] = 0
                 eef_state['vel']['linear']['z'] = 0    
                 gym.set_actor_rigid_body_states(envs[0], dvrk_handles[0], eef_state, gymapi.STATE_ALL)
                 step_physics(sim_cache)
                 step_rendering(sim_cache, args)

                 #################### after finished current sub-trajectory  ################
                 print("Succesfully executed moveit arm plan")
                 if plan_config["is_near_box"]==True:
                     if not plan_config["avoid_box"]:
                         plan_config["is_near_box"] = False
                     # update box pose
                     sim_cache["box_pose"].p.x  = box_state['pose']['p']['x']
                     sim_cache["box_pose"].p.y  = box_state['pose']['p']['y'] 
                     sim_cache["box_pose"].p.z  = box_state['pose']['p']['z'] 
                     state = "get shape servo plan"
                     # don't increment num_cp
                     return state
                 else:
                    
                     plan_config["is_near_box"] = True
                     plan_config["sample_cp"] = True
                     # update box pose
                     sim_cache["box_pose"].p.x  = box_state['pose']['p']['x']
                     sim_cache["box_pose"].p.y  = box_state['pose']['p']['y'] 
                     sim_cache["box_pose"].p.z  = box_state['pose']['p']['z'] 
                    
                     if plan_config["num_cp"] <=  plan_config["max_cp"] -1:
                         print("try next cp")
                         plan_config["num_cp"] += 1
                         state = "get shape servo plan"
                         return state
                     else:
                         print("stay after finishing all cps")
                         plan_config["stay_before_timeout"]=True
                         print("Stay put: NOT TIMEOUT YET")
                         state = "move to goal"
                         return state
                    

         # set velocity of point mass robot
        vel = target_xy - eef_xy
        speed = np.linalg.norm(vel)
        eef_state = gym.get_actor_rigid_body_states(envs[0], dvrk_handles[0], gymapi.STATE_ALL)
        eef_state['vel']['linear']['x'] =  (target_xy[0] - eef_xy[0])*2#/speed
        eef_state['vel']['linear']['y'] = (target_xy[1] - eef_xy[1])*2#/speed
        eef_state['vel']['linear']['z'] = 0    
        gym.set_actor_rigid_body_states(envs[0], dvrk_handles[0], eef_state, gymapi.STATE_ALL)
        eef_state = gym.get_actor_rigid_body_states(envs[0], dvrk_handles[0], gymapi.STATE_ALL)

        step_rendering(sim_cache, args)
    

def reset_state(sim_cache, data_config, init_robot_state, args):
    envs = sim_cache["envs"]
    gym = sim_cache["gym"]
    dvrk_handles = sim_cache["dvrk_handles"]
    object_handles = sim_cache["object_handles"]
    init_pose = sim_cache["init_pose"] 

    rospy.logwarn("==== RESETTING ====")

    gym.set_actor_rigid_body_states(envs[0], dvrk_handles[0], init_robot_state, gymapi.STATE_ALL) 
    print("Sucessfully reset robot")

    box_state = gym.get_actor_rigid_body_states(envs[0], object_handles[0], gymapi.STATE_POS)
    box_state['pose']['p']['x'] = sim_cache["init_box_pose"].p.x  
    box_state['pose']['p']['y'] = sim_cache["init_box_pose"].p.y
    box_state['pose']['p']['z'] = sim_cache["init_box_pose"].p.z
    sim_cache["box_pose"] = deepcopy(sim_cache["init_box_pose"])
    gym.set_actor_rigid_body_states(envs[0], object_handles[0], box_state, gymapi.STATE_ALL)
    print("successfully reset box") 

    state = "config"

    re_sample = plan_config["no_plan"] or data_config["sample_count"] >= data_config["max_sample_count"] or (args.is_behavioral_cloning and plan_config["goal_was_reached"]==False)

    if re_sample:
        if data_config["sample_count"] >= data_config["max_sample_count"]: 
            data_config["sample_count"] = 0
            data_config["group_count"] += 1
        if plan_config["no_plan"]:
            plan_config["no_plan"] = False

        box_state = gym.get_actor_rigid_body_states(envs[0], object_handles[0], gymapi.STATE_POS)
        new_pose = np.array(init_pose) + np.random.uniform(low=[-0.1,-0.05,0], high=[0.1,0.05,0], size=3)
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
        # don't let the cone overlap with the box
        new_pose = np.array(init_pose) + np.random.uniform(low=[-0.1,-0.05,0], high=[0.1,0.05,0], size=3)
        new_pose[2] += CONE_Z_EXTRA_OFFSET
        cone_state['pose']['p']['x'] = new_pose[0]    
        cone_state['pose']['p']['y'] = new_pose[1]
        cone_state['pose']['p']['z'] = new_pose[2]    
        sim_cache["cone_pose"].p.x = new_pose[0] 
        sim_cache["cone_pose"].p.y = new_pose[1] 
        sim_cache["cone_pose"].p.z = new_pose[2]
        while is_overlap_xy(sim_cache["cone_pose"], sim_cache["box_pose"], OVERLAP_TOLERANCE):
            new_pose = np.array(init_pose) + np.random.uniform(low=[-0.1,-0.05,0], high=[0.1,0.05,0], size=3)
            new_pose[2] += CONE_Z_EXTRA_OFFSET
            cone_state['pose']['p']['x'] = new_pose[0]    
            cone_state['pose']['p']['y'] = new_pose[1]
            cone_state['pose']['p']['z'] = new_pose[2]    
            sim_cache["cone_pose"].p.x = new_pose[0] 
            sim_cache["cone_pose"].p.y = new_pose[1] 
            sim_cache["cone_pose"].p.z = new_pose[2]       
        gym.set_actor_rigid_body_states(envs[0], object_handles[1], cone_state, gymapi.STATE_ALL)      

        rospy.logwarn("Successfully reset box and cone")        
        state = "home"

    return state

if __name__ == "__main__":
     # initialize gym
    gym = gymapi.acquire_gym()

    parser = argparse.ArgumentParser(description='Options')
    is_train = False
    suffix = "train" if is_train else "test"
    parser.add_argument('--data_recording_path', default=f"/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_push/demos_{suffix}", type=str, help="where you want to record data")
    parser.add_argument('--headless', default="False", type=str, help="Index of grasp candidate to test")
    parser.add_argument('--save_data', default="False", type=str, help="True: save recorded data to pickles files")
    parser.add_argument('--record_pc', default="True", type=str, help="True: record partial point cloud")
    parser.add_argument('--rand_seed', default=2021, type=int, help="random seed")
    parser.add_argument('--num_groups', default=30, type=int, help="number of groups you want to collect")
    parser.add_argument('--num_samples', default=30, type=int, help="number of samples per group you want to collect")
    parser.add_argument('--overlap_tolerance', default=0.001, type=float, help="threshold determining overlapping of cone and box")
    parser.add_argument('--is_behavioral_cloning', default='False', type=str, help="is the trajectories collected for behavioral cloning")

    args = parser.parse_args()

    OVERLAP_TOLERANCE = args.overlap_tolerance

    args.headless = args.headless == "True"
    args.save_data = args.save_data == "True"
    args.record_pc = args.record_pc == "True"
    args.is_behavioral_cloning = args.is_behavioral_cloning == 'True'
    data_recording_path = args.data_recording_path
    os.makedirs(data_recording_path, exist_ok=True)

    rand_seed = args.rand_seed
    np.random.seed(rand_seed)
    random.seed(rand_seed)

    max_group_count = args.num_groups
    max_sample_count = args.num_samples

    print("!!!!!!!!!!!! IMPORTANT INFO !!!!!!!!!!!!!!!!!")
    print(f"data_recording_path: {data_recording_path}")
    print(f"headless: {args.headless}")
    print(f"* Save data: {args.save_data}")
    print(f"random seed: {rand_seed}")
    print(f"overlap_tolerance: {OVERLAP_TOLERANCE}")
    print(f"max_group_count: {max_group_count} max_sample_count: {max_sample_count}")
    print(f"is behavioral cloning: {args.is_behavioral_cloning}")


    envs, sim, dvrk_handles, object_handles, cam_handles, cam_prop, viewer, box_pose, cone_pose, init_pose = configure_isaacgym(gym, args)
   
    #rospy.init_node('shape_servo_control')

    state = "home"

    init_box_pose = gymapi.Transform()
    init_box_pose.p.x = deepcopy(box_pose.p.x)
    init_box_pose.p.y = deepcopy(box_pose.p.y)
    init_box_pose.p.z = deepcopy(box_pose.p.z)

    sim_cache = {"gym":gym, "sim":sim, "envs":envs, "dvrk_handles":dvrk_handles, \
                "object_handles":object_handles, "cam_handles":cam_handles, "cam_prop": cam_prop, "viewer":viewer,\
                "box_pose": box_pose, "init_box_pose":init_box_pose, "cone_pose":cone_pose, "init_pose": init_pose}

    data_config = {"sample_count":0, "max_sample_count":max_sample_count, "group_count":0, \
                "max_group_count":max_group_count, "data_recording_path": data_recording_path}

    start_time = timeit.default_timer()   

    while (True): 
        step_physics(sim_cache)
        if state == "home" :   
            state, init_robot_state = home_state(sim_cache, args)
        if state == "config":
            state, plan_config = config_state(sim_cache, args)
        if state == "get shape servo plan":
            # get plan to go to the next checkpoint
            state, target_pose = get_shape_servo_plan_state(sim_cache, plan_config)
        if state == "move to goal":
            state = move_to_goal_state(sim_cache, plan_config, data_config, target_pose, args, record_pc=args.record_pc)
        if state == "reset":
            state = reset_state(sim_cache, data_config, init_robot_state, args)
        if data_config["group_count"] >= data_config["max_group_count"]:
            break
        step_rendering(sim_cache, args)


    print("All done !")
    print("Elapsed time", timeit.default_timer() - start_time)
    if not args.headless:
        gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)









