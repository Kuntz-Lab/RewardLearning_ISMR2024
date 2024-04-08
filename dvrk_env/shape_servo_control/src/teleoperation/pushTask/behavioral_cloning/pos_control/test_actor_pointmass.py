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

from policy_BC_PC import ActorPC
import torch

sys.path.append("../../../pc_utils")
from get_isaac_partial_pc import get_partial_pointcloud_vectorized
from compute_partial_pc import farthest_point_sample_batched

'''
Test trained Gaussian agent(actor) with cartesian (xyz) control
'''

ROBOT_Z_OFFSET = 0.25
THICKNESS = 0.001   #0.0000025
CONE_Z_EXTRA_OFFSET = 0.01
OVERLAP_TOLERANCE  = 0.001
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

    #################### assume box and cone on the x-y plane ######################
    # Load box object
    init_pose = [0.0, -0.5, 0.03 + THICKNESS]
    box_pose = gymapi.Transform()
    pose = list(np.array(init_pose) + np.random.uniform(low=[-0.1,-0.05,0], high=[0.1,0.05,0], size=3))
    box_pose.p = gymapi.Vec3(*pose)

    print(f"################## box pose: {pose}")
    
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = False
    asset_options.disable_gravity = False
    asset_options.thickness = THICKNESS
    asset_options.density = 1000 #10000 #
    asset_options.armature = 20 #10
    asset_options.use_physx_armature = True

    box_asset = gym.create_sphere(sim, 0.03, asset_options)    

    # load cone object
    cone_pose = gymapi.Transform()
    pose = list(np.array(init_pose) + np.random.uniform(low=[-0.1,-0.05,0], high=[0.1,0.05,0], size=3))
    pose[2] += CONE_Z_EXTRA_OFFSET
    cone_pose.p = gymapi.Vec3(*pose)

    print(f"################## cone pose: {pose}")

    while is_overlap_xy(cone_pose, box_pose, OVERLAP_TOLERANCE):
        pose = list(np.array(init_pose) + np.random.uniform(low=[-0.1,-0.05,0], high=[0.1,0.05,0], size=3))
        pose[2] += CONE_Z_EXTRA_OFFSET
        cone_pose.p = gymapi.Vec3(*pose)
        print(f"################## new cone pose: {pose}")

    cone_asset = gym.create_box(sim, *[0.04, 0.04, 0.08], asset_options)

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
    cam_width = 300
    cam_height = 300
    cam_targets = gymapi.Vec3(0.0, -0.4, 0.02)
    cam_positions = gymapi.Vec3(0.2, -0.7, 0.5)


    cam_handle, cam_prop = setup_cam(gym, envs[0], cam_width, cam_height, cam_positions, cam_targets)
    for i, env in enumerate(envs):
        cam_handles.append(cam_handle)

    if not args.headless:
        return envs, sim, dvrk_handles, object_handles, cam_handles, cam_prop, viewer, box_pose, cone_pose, init_pose
    else:
        return envs, sim, dvrk_handles, object_handles, cam_handles, cam_prop, None, box_pose, cone_pose, init_pose

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
    state = "config"
    init_robot_state = deepcopy(gym.get_actor_rigid_body_states(envs[0], dvrk_handles[0], gymapi.STATE_ALL))

    return state, init_robot_state

def is_overlap_xy(p1, p2, max_dist):
    return (p1.p.x-p2.p.x)**2 + (p1.p.y-p2.p.y)**2 <=max_dist
    #return np.allclose(np.array([p1.p.x, p1.p.y, p1.p.z]), np.array([p2.p.x, p2.p.y, p2.p.z]), rtol=0, atol=0.058)


def config_state(): 
    state = "get shape servo plan"

    plan_config = {"cp_pose":gymapi.Transform(), "max_frame":800, "frame":0, \
                  "eef_states":[], "box_states":[], "pcds":[], "dof_vels":[], "no_plan": False, "goal_was_reached":False}


    return state, plan_config

def is_in_workspace(x, y):
    low_x = x>=-0.1
    up_x = x<=0.1
    up_y = y<=-0.4
    low_y = y>=-0.6

    return low_x and up_x and up_y and low_y

def to_obj_emb(model, device, pcd):
    pcd_tensor = torch.from_numpy(pcd.transpose(1,0)).unsqueeze(0).float().to(device)
    emb = model(pcd_tensor, get_global_embedding=True)
    return emb


def get_shape_servo_plan_state(sim_cache, plan_config, policy):
    state = "get shape servo plan"
    rospy.loginfo("**Current state: " + state) 
    step_physics(sim_cache)

    box_pose = sim_cache["box_pose"]
    cone_pose = sim_cache["cone_pose"]

    pose = gymapi.Transform()

    box_state = deepcopy(gym.get_actor_rigid_body_states(envs[0], object_handles[0], gymapi.STATE_POS))
    eef_state = deepcopy(gym.get_actor_rigid_body_states(envs[0], dvrk_handles[0], gymapi.STATE_ALL))
    
    eef_pos = np.array([eef_state["pose"]["p"][0][0], eef_state["pose"]["p"][0][1]])
    box_pos = np.array([box_state["pose"]["p"][0][0], box_state["pose"]["p"][0][1]])
    #print([*box_state["pose"]["p"]])
    cone_pos = np.array([sim_cache["cone_pose"].p.x, sim_cache["cone_pose"].p.y])
    eef_pos = torch.from_numpy(eef_pos).to(device).float().unsqueeze(0)
    box_pos = torch.from_numpy(box_pos).to(device).float().unsqueeze(0)
    cone_pos = torch.from_numpy(cone_pos).to(device).float().unsqueeze(0)
    state = torch.cat((eef_pos, box_pos, cone_pos), dim=1)

    action = policy.act(state)
    next_eef_pos = [action[0][i].item() for i in range(2)]

    print("=======================================================")
    print(f"#################### sim cache box pos: {[box_pose.p.x, box_pose.p.y, box_pose.p.z]}")
    print(f"#################### sim cache cone pos: {[cone_pose.p.x, cone_pose.p.y, cone_pose.p.z]}")
    print(f"#################### box pos: {box_pos}")
    print(f"#################### cone pos: {cone_pos}")
    print(f"#################### eef pos: {eef_pos}")
    print(f"#################### next eef pos (action): {next_eef_pos}")
    print("=======================================================")

    next_eef_pos = [next_eef_pos[0], next_eef_pos[1], EEF_Z]
    pose.p = gymapi.Vec3(*next_eef_pos)
        
    
    state = "move to goal"
        
    return state, pose

def move_to_goal_state(sim_cache, plan_config, data_config, target_pose, args, record_pc=False):
    state = "move to goal"
    rospy.loginfo("**Current state: " + state) 
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
    box_state = deepcopy(gym.get_actor_rigid_body_states(envs[0], object_handles[0], gymapi.STATE_POS))
    error_threshold = 0.001

    while error >= error_threshold or plan_config["goal_was_reached"]:
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


        if is_overlap_xy(sim_cache["box_pose"], sim_cache["cone_pose"], OVERLAP_TOLERANCE):
            print(f"NICE: box already reached goal") 
            plan_config["goal_was_reached"]=True
        
       

        # save data every save_frame frames
        if plan_config["frame"]%save_frame==0:
            plan_config["eef_states"].append(deepcopy(eef_state))
            plan_config["box_states"].append(deepcopy(box_state))
            
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
                            "success_goal":success_goal, "pcds":plan_config["pcds"]
                    }
            if args.save_data:

                with open(os.path.join(data_recording_path, f"group {group_count} sample {sample_count}.pickle"), 'wb') as handle:
                        pickle.dump(data, handle, protocol=3)   
                    
                print("Len trajectory: ", len(plan_config["eef_states"])) 

            data_config["sample_count"] += 1 

            state = "reset"
            return state
        else:
            if error <= error_threshold:
                print("finish action, get another action")
                state = "get shape servo plan"
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
        # print("&&&&&&&&&&&&& eef state: ", eef_state)

        step_rendering(sim_cache, args)

def reset_state(sim_cache, data_config, init_robot_state):
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

    re_sample = plan_config["no_plan"] or data_config["sample_count"] >= data_config["max_sample_count"]

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
    is_train = True
    suffix = "train" if is_train else "test"
    parser.add_argument('--headless', default="False", type=str, help="Index of grasp candidate to test")
    parser.add_argument('--save_data', default="False", type=str, help="True: save recorded data to pickles files")
    parser.add_argument('--record_pc', default="False", type=str, help="True: record partial point cloud")
    parser.add_argument('--data_recording_path', default=f"/home/dvrk/LfD_data/ex_push/BC/debug", type=str, help="where you want to record data")
    parser.add_argument('--model_path', default=f"/home/dvrk/LfD_data/ex_push/BC/pointcloud_pos_control/weights/weights_1/epoch_200", type=str, help="policy")
    parser.add_argument('--rand_seed', default=2021, type=int, help="random seed")
    parser.add_argument('--overlap_tolerance', default=0.001, type=float, help="threshold determining overlapping of cone and box")

    args = parser.parse_args()

    OVERLAP_TOLERANCE = args.overlap_tolerance

    args.headless = args.headless == "True"
    args.save_data = args.save_data == "True"
    args.record_pc = args.record_pc == "True"
    data_recording_path = args.data_recording_path
    model_path = args.model_path
    os.makedirs(data_recording_path, exist_ok=True)

    rand_seed = args.rand_seed
    np.random.seed(rand_seed)
    random.seed(rand_seed)

    envs, sim, dvrk_handles, object_handles, cam_handles, cam_prop, viewer, box_pose, cone_pose, init_pose = configure_isaacgym(gym, args)
   
    '''
    Main simulation stuff starts from here
    '''
    rospy.init_node('shape_servo_control')
    rospy.logerr(f"Save data: {args.save_data}")

    state = "home"

    init_box_pose = gymapi.Transform()
    init_box_pose.p.x = deepcopy(box_pose.p.x)
    init_box_pose.p.y = deepcopy(box_pose.p.y)
    init_box_pose.p.z = deepcopy(box_pose.p.z)

    sim_cache = {"gym":gym, "sim":sim, "envs":envs, "dvrk_handles":dvrk_handles, \
                "object_handles":object_handles, "cam_handles":cam_handles, "cam_prop": cam_prop, "viewer":viewer,\
                "box_pose": box_pose, "init_box_pose":init_box_pose, "cone_pose":cone_pose, "init_pose": init_pose}

    data_config = {"sample_count":0, "max_sample_count":1, "group_count":0, \
                "max_group_count":5, "data_recording_path": data_recording_path}

    # dummy
    plan_config = {}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    policy = ActorPC(act_dim=2, robot_state_dim=2, emb_dim=4, initial_std=1.0)
    policy.to(device)
    policy.load_state_dict(torch.load(model_path))
    policy.eval()

    start_time = timeit.default_timer()   

    while (True): 
        step_physics(sim_cache)
        if state == "home" :   
            state, init_robot_state = home_state(sim_cache, args)
        if state == "config":
            state, plan_config = config_state()
        if state == "get shape servo plan":
            # get plan to go to the next checkpoint
            state, target_pose = get_shape_servo_plan_state(sim_cache, plan_config, policy)
        if state == "move to goal":
            state = move_to_goal_state(sim_cache, plan_config, data_config, target_pose, args, record_pc=args.record_pc)
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









