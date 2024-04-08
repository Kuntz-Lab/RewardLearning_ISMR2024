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

from policy_BC import BCPolicy

import torch
'''
test the learned policy
'''

ROBOT_Z_OFFSET = 0.25

def default_dvrk_asset(gym, sim):
    '''
    load the dvrk asset
    '''
    # dvrk asset
    asset_options = gymapi.AssetOptions()
    asset_options.armature = 0.001
    asset_options.fix_base_link = True
    asset_options.thickness = 0.00025#0.0001

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


    # load robot asset
    dvrk_asset = default_dvrk_asset(gym, sim)
    dvrk_pose = gymapi.Transform()
    dvrk_pose.p = gymapi.Vec3(0.0, 0.0, ROBOT_Z_OFFSET)
    dvrk_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)    

    #################### assume box and cone on the x-y plane ######################
    # Load box object
    init_pose = [0.0, -0.5, 0.022]
    asset_root = "/home/dvrk/catkin_ws/src/dvrk_env/shape_servo_control/src/teleoperation/random_stuff"
    box_asset_file = "push_box.urdf"    
    box_pose = gymapi.Transform()
    pose =   [0.02119565576148094, -0.47666306388065016, 0.022]#[0.021195656, -0.47666305, 0.020250006]
    box_pose.p = gymapi.Vec3(*pose)
    
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = False
    asset_options.disable_gravity = False
    # i have changed this !!!!!!!!!!
    asset_options.thickness = 0.00025

    box_asset = gym.load_asset(sim, asset_root, box_asset_file, asset_options)     

    # load cone object
    cone_asset_file = "cone.urdf"  
    cone_pose = gymapi.Transform()
    pose = [-0.0642013788099624, -0.4747074571223906, 0.022]#[-0.06420137733221054, -0.550000011920929, 0.02199999988079071]
    cone_pose.p = gymapi.Vec3(*pose)

    # while is_overlap(cone_pose, box_pose):
    #     pose = list(np.array(init_pose) + np.random.uniform(low=[-0.3,-0.03,0], high=[-0.1,-0.02,0], size=3))
    #     cone_pose.p = gymapi.Vec3(*pose)

    #asset_options.fix_base_link = True
    
    cone_asset = gym.load_asset(sim, asset_root, cone_asset_file, asset_options) 

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

    # DOF Properties and Drive Modes 
    dof_props = gym.get_actor_dof_properties(envs[0], dvrk_handles[0])
    dof_props["driveMode"].fill(gymapi.DOF_MODE_VEL)
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

        middle_env = envs[num_envs // 2 + num_per_row // 2]
        gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)

    
    # Camera for point cloud setup
    cam_handles = []
    cam_width = 300
    cam_height = 300
    # cam_positions = gymapi.Vec3(0.0, -0.4400001, 0.7)
    cam_targets = gymapi.Vec3(0.0, -0.44, 0.1)
    # cam_positions = gymapi.Vec3(0.3, -0.54, 0.16+0.015 + 0.15)
    cam_positions = gymapi.Vec3(0.7, -0.54, 0.76)
    # cam_targets = gymapi.Vec3(0.0, -0.44, 0.16+0.015)


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
    cone_pose = sim_cache["cone_pose"]

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

    return state, init_robot_state

def config_state():
    state = "move to goal"
    rospy.loginfo("**Current state: " + "config") 

    plan_config = {"max_frame":180, "frame":0, \
                  "eef_states":[], "box_states":[], "pcds":[], "dof_vels":[]}

    return state, plan_config


def is_overlap(p1, p2, max_dist=0.005):
    return (p1.p.x-p2.p.x)**2 + (p1.p.y-p2.p.y)**2 + (p1.p.z-p2.p.z)**2 <=max_dist
    #return np.allclose(np.array([p1.p.x, p1.p.y, p1.p.z]), np.array([p2.p.x, p2.p.y, p2.p.z]), rtol=0, atol=0.058)

def move_to_goal_state(policy, device, sim_cache, plan_config, data_config, args, record_pc):
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
    while (True):
        #print(f"traj_index: {traj_index}:{len(plan_traj)}")
        step_physics(sim_cache)

        box_state = deepcopy(gym.get_actor_rigid_body_states(envs[0], object_handles[0], gymapi.STATE_POS))
        eef_state = deepcopy(gym.get_actor_rigid_body_states(envs[0], dvrk_handles[0], gymapi.STATE_ALL)[-3])
        dof_vel = deepcopy(gym.get_actor_dof_states(envs[0], dvrk_handles[0], gymapi.STATE_VEL)['vel'])
        
        # update box pose
        sim_cache["box_pose"].p.x  = box_state['pose']['p']['x']
        sim_cache["box_pose"].p.y  = box_state['pose']['p']['y'] 
        sim_cache["box_pose"].p.z  = box_state['pose']['p']['z'] 
        
        # save data every save_frame frames
        if plan_config["frame"]%save_frame==0:
            plan_config["eef_states"].append(deepcopy(eef_state))
            plan_config["box_states"].append(deepcopy(box_state))
            plan_config["dof_vels"].append(deepcopy(dof_vel))
            if record_pc:
                pcd = get_partial_point_cloud(gym, sim, envs[0], cam_handles[0], cam_prop, color=[1,0,0], min_z = 0.01, visualization=False)
                plan_config["pcds"].append(pcd)


        #print("########## box z: ", sim_cache["box_pose"].p.z)
        plan_config["frame"]+=1

        if plan_config["frame"] >= plan_config["max_frame"]-1:
            # timeout, save sample
            plan_config["eef_states"].append(deepcopy(gym.get_actor_rigid_body_states(envs[0], dvrk_handles[0], gymapi.STATE_ALL)[-3]))
            plan_config["box_states"].append(deepcopy(box_state))
            plan_config["dof_vels"].append(deepcopy(dof_vel))
            if record_pc:
                 pcd = get_partial_point_cloud(gym, sim, envs[0], cam_handles[0], cam_prop, color=[1,0,0], min_z = 0.01, visualization=False)
                 plan_config["pcds"].append(pcd)
                 assert(len(plan_config["box_states"])==len(plan_config["pcds"]))
            assert(len(plan_config["box_states"])==len(plan_config["eef_states"]))
            print("++++++++++++++++++++++++++++++++++++++ traj_len: ", len(plan_config["box_states"]), " last_frame: ", plan_config["frame"])
            assert(len(plan_config["eef_states"])==plan_config["max_frame"]//save_frame+1)

            rospy.logwarn(f"TIME'S OUT")  
            rospy.loginfo(f"++++++++++++++++Done group {group_count} sample {sample_count}")
            rospy.loginfo("Let's record demos!!")

            # save data
            if args.save_data:
                # success_goal = is_overlap(sim_cache["box_pose"], sim_cache["cone_pose"])
                #success_goal = plan_config["goal_was_reached"]
                data = {"eef_states": plan_config["eef_states"],\
                    "box_states": plan_config["box_states"],\
                    "dof_vels": plan_config["dof_vels"],\
                    "cone_pose":[sim_cache["cone_pose"].p.x, sim_cache["cone_pose"].p.y, sim_cache["cone_pose"].p.z],\
                     "pcds":plan_config["pcds"]
                }

                with open(os.path.join(data_recording_path, f"group {group_count} sample {sample_count}.pickle"), 'wb') as handle:
                        pickle.dump(data, handle, protocol=3)   
                    
                print("Len trajectory: ", len(plan_config["eef_states"]))      


            data_config["sample_count"] += 1 
            state = "reset"
            return state

        # Set target joint positions
        #print(box_state["pose"]["p"])
        eef_pos = np.array([eef_state["pose"]["p"][0], eef_state["pose"]["p"][1], eef_state["pose"]["p"][2]])
        box_pos = np.array([box_state["pose"]["p"][0][0], box_state["pose"]["p"][0][1], box_state["pose"]["p"][0][2]])
        #print([*box_state["pose"]["p"]])
        cone_pos = np.array([sim_cache["cone_pose"].p.x, sim_cache["cone_pose"].p.y, sim_cache["cone_pose"].p.z])
        eef_pos = torch.from_numpy(eef_pos).to(device).float().unsqueeze(0)
        box_pos = torch.from_numpy(box_pos).to(device).float().unsqueeze(0)
        cone_pos = torch.from_numpy(cone_pos).to(device).float().unsqueeze(0)
        state = torch.cat((eef_pos, box_pos, cone_pos), dim=1)
        with torch.no_grad():
            action = policy(state)
        dof_vel = [action[0][i].item() for i in range(10)]
        # print("+++++++++++++++++++++++++", dof_vel)
        gym.set_actor_dof_velocity_targets(envs[0], dvrk_handles[0], dof_vel)                

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

    if data_config["sample_count"] >= data_config["max_sample_count"]:
        data_config["sample_count"] = 0
        data_config["group_count"] += 1

        box_state = gym.get_actor_rigid_body_states(envs[0], object_handles[0], gymapi.STATE_POS)
        new_pose = np.array(init_pose) + np.random.uniform(low=[-0.1,-0.03,0], high=[0.1,-0.02,0], size=3)
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
        new_pose = np.array(init_pose) + np.random.uniform(low=[-0.3,-0.03,0], high=[0.1,-0.02,0], size=3)
        cone_state['pose']['p']['x'] = new_pose[0]    
        cone_state['pose']['p']['y'] = new_pose[1]
        cone_state['pose']['p']['z'] = new_pose[2]    
        sim_cache["cone_pose"].p.x = new_pose[0] 
        sim_cache["cone_pose"].p.y = new_pose[1] 
        sim_cache["cone_pose"].p.z = new_pose[2]
        while is_overlap(sim_cache["cone_pose"], sim_cache["box_pose"]):
            new_pose = np.array(init_pose) + np.random.uniform(low=[-0.3,-0.03,0], high=[0.1,-0.02,0], size=3)
            cone_state['pose']['p']['x'] = new_pose[0]    
            cone_state['pose']['p']['y'] = new_pose[1]
            cone_state['pose']['p']['z'] = new_pose[2]    
            sim_cache["cone_pose"].p.x = new_pose[0] 
            sim_cache["cone_pose"].p.y = new_pose[1] 
            sim_cache["cone_pose"].p.z = new_pose[2]       
        gym.set_actor_rigid_body_states(envs[0], object_handles[1], cone_state, gymapi.STATE_ALL)  

        gym.set_actor_dof_position_targets(robot.env_handle, robot.robot_handle, [0,0,0,0,0.20,0,0,0,0.35,-0.35])        

        rospy.logwarn("Successfully reset box and cone")        
        state = "home"

    return state

if __name__ == "__main__":
     # initialize gym
    gym = gymapi.acquire_gym()

    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--headless', default="False", type=str, help="Index of grasp candidate to test")
    parser.add_argument('--save_data', default="True", type=str, help="True: save recorded data to pickles files")
    parser.add_argument('--record_pc', default="True", type=str, help="True: record partial point cloud")

    ########## CHANGE ############
    is_train = True
    suffix = "train" if is_train else "test"
    parser.add_argument('--data_recording_path', default=f"/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_push_BC_corrected/policy_demos_1_L2", type=str, help="where you want to record data")
    parser.add_argument('--model_path', default=f"/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_push_BC_corrected/weights/weights_L2_1", type=str, help="model path")

    args = parser.parse_args()
    args.headless = args.headless == "True"
    args.save_data = args.save_data == "True"
    args.record_pc = args.record_pc == "True"
    data_recording_path = args.data_recording_path
    model_path = args.model_path
    os.makedirs(data_recording_path, exist_ok=True)


    envs, sim, dvrk_handles, object_handles, cam_handles, cam_prop, viewer, box_pose, cone_pose, init_pose = configure_isaacgym(gym, args)
   
    '''
    Main simulation stuff starts from here
    '''
    rospy.init_node('shape_servo_control')
    rospy.logerr(f"Save data: {args.save_data}")

    # Some important paramters
    init_dvrk_joints(gym, envs[0], dvrk_handles[0], joint_angles=[0,0,0,0,0.20,0,0,0,0.35,-0.35])  # Initialize robot's joints    


    state = "home"

    init_box_pose = gymapi.Transform()
    init_box_pose.p.x = deepcopy(box_pose.p.x)
    init_box_pose.p.y = deepcopy(box_pose.p.y)
    init_box_pose.p.z = deepcopy(box_pose.p.z)

    sim_cache = {"gym":gym, "sim":sim, "envs":envs, "dvrk_handles":dvrk_handles, \
                "object_handles":object_handles, "cam_handles":cam_handles, "cam_prop": cam_prop, "viewer":viewer,\
                "box_pose": box_pose, "init_box_pose":init_box_pose, "cone_pose":cone_pose, "init_pose": init_pose}

    data_config = {"sample_count":0, "max_sample_count":1, "group_count":0, \
                "max_group_count":1, "data_recording_path": data_recording_path}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BCPolicy(obs_dim=9, act_dim=10) 
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(model_path, "epoch 100")))
    model.eval()

    start_time = timeit.default_timer()   
    robot = Robot(gym, sim, envs[0], dvrk_handles[0])
    vis_count=0

    while (True): 
        step_physics(sim_cache)
        if state == "home" :   
            state, init_robot_state = home_state(sim_cache, args)
        if state == "config":
            state, plan_config = config_state()
        if state == "move to goal":
            state = move_to_goal_state(model, device, sim_cache, plan_config, data_config, args, args.record_pc)
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









