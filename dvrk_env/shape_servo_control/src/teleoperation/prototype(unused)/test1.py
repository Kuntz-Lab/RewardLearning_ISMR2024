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
import open3d

from geometry_msgs.msg import PoseStamped, Pose, TransformStamped

from util.isaac_utils import *
from util.grasp_utils import GraspClient

from core import Robot
from behaviors import MoveToPose, TaskVelocityControl
import argparse


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
    parser.add_argument('--data_recording_path', default="/home/dvrk/shape_servo_data/generalization/multi_boxes_1000Pa/data", type=str, help="path to save the recorded data")
    parser.add_argument('--object_meshes_path', default="/home/dvrk/sim_data/Custom/Custom_mesh/multi_boxes_1000Pa", type=str, help="path to the objects' tet meshe files")
    parser.add_argument('--max_data_point_count', default=30000, type=int, help="path to the objects' tet meshe files")
    parser.add_argument('--save_data', default=False, type=bool, help="True: save recorded data to pickles files")
    args = parser.parse_args()
    args.headless = args.headless == "True"

    # configure sim
    sim, sim_params = default_sim_config(gym, args)


    # # Get primitive shape dictionary to know the dimension of the object   
    # object_meshes_path = args.object_meshes_path  
    # with open(os.path.join(object_meshes_path, "primitive_dict_box.pickle"), 'rb') as handle:
    #     data = pickle.load(handle)    
    # h = data[args.obj_name]["height"]
    # w = data[args.obj_name]["width"]
    # thickness = data[args.obj_name]["thickness"]


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


    # # Load deformable object
    # asset_root = "/home/dvrk/Documents/IsaacGym_Preview_2_Package/isaacgym/assets/urdf"
    # soft_asset_file = "new_ball.urdf"    
    # soft_pose = gymapi.Transform()
    # soft_pose.p = gymapi.Vec3(0.0, -0.42, 0.025)
    # soft_pose.r = gymapi.Quat(0.0, 0.0, 0.707107, 0.707107)
    # soft_thickness = 0.0005    # important to add some thickness to the soft body to avoid interpenetrations

    # asset_options = gymapi.AssetOptions()
    # asset_options.fix_base_link = True
    # asset_options.thickness = soft_thickness
    # asset_options.disable_gravity = True

    # soft_asset = gym.load_asset(sim, asset_root, soft_asset_file, asset_options)       


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
        

        # # add soft obj            
        # soft_actor = gym.create_actor(env, soft_asset, soft_pose, "soft", i, 0)
        # color = gymapi.Vec3(1,0,0)
        # gym.set_rigid_body_color(env, soft_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
        # object_handles.append(soft_actor)



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
        cam_pos = gymapi.Vec3(0.5, -0.5, 0.25)
        cam_target = gymapi.Vec3(0.0, -0.36, 0.1)
        middle_env = envs[num_envs // 2 + num_per_row // 2]
        gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)

    
    # Camera for point cloud setup
    cam_handles = []
    cam_width = 256
    cam_height = 256 
    cam_positions = gymapi.Vec3(0.17, -0.62, 0.2)
    cam_targets = gymapi.Vec3(0.0, 0.40-0.86, 0.01)
    cam_handle, cam_prop = setup_cam(gym, envs[0], cam_width, cam_height, cam_positions, cam_targets)
    # for i, env in enumerate(envs):
    #     cam_handles.append(cam_handle)


       

    '''
    Main simulation stuff starts from here
    '''
    rospy.init_node('shape_servo_control')
    rospy.logerr("======Loading object ... " + str(args.obj_name))  

    # Some important paramters
    init_dvrk_joints(gym, envs[i], dvrk_handles[i])  # Initilize robot's joints    

    all_done = False
    state = "home"
    
    
    # data_recording_path = args.data_recording_path #"/home/dvrk/shape_servo_data/generalization/multi_boxes_1000Pa/data"
    terminate_count = 0 
    sample_count = 0
    frame_count = 0
    group_count = 0
    # data_point_count = len(os.listdir(data_recording_path))
    # max_sample_count = 10   # number of trajectories per manipulation point
    # max_data_point_count = args.max_data_point_count    # total num of data points
    # max_data_point_per_variation = data_point_count + 300   # limit to only 300 data points per shape variation
    # rospy.logwarn("max data point per shape variation:" + str(max_data_point_per_variation))
    # new_init_config = False#True  # get a new initial pose for the object 

    dc_client = GraspClient()


    recorded_particle_states = []
    recorded_pcs = []
    recorded_poses = []
    
    start_time = timeit.default_timer()   
    close_viewer = False
    robot = Robot(gym, sim, envs[0], dvrk_handles[0])

    # # Load multiple initial object poses:
    # if new_init_config:
    #     with open('/home/dvrk/shape_servo_data/generalization/multi_object_poses/box1k.pickle', 'rb') as handle:
    #         saved_object_states = pickle.load(handle) 

    while (not close_viewer) and (not all_done): 

        if not args.headless:
            close_viewer = gym.query_viewer_has_closed(viewer)  

        # step the physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)
 

        if state == "home" :   
            frame_count += 1
            gym.set_joint_target_position(envs[0], gym.get_joint_handle(envs[0], "dvrk", "psm_main_insertion_joint"), 0.203)            
            
            if frame_count == 10:
                rospy.loginfo("**Current state: " + state)
                frame_count = 0

                # # Save robot and object states for reset 
                # gym.refresh_particle_state_tensor(sim)
                # saved_object_state = deepcopy(gymtorch.wrap_tensor(gym.acquire_particle_state_tensor(sim))) 
                init_robot_state = deepcopy(gym.get_actor_rigid_body_states(envs[i], dvrk_handles[i], gymapi.STATE_ALL))           
                

                # Go to next state
                # state = "get shape servo plan"
                state = "get shape servo plan"



        # ############################################################################
        # # grasp object: close gripper
        # ############################################################################        
        # if state == "grasp object":             
        #     rospy.loginfo("**Current state: " + state)
        #     gym.set_joint_target_position(envs[i], gym.get_joint_handle(envs[i], "dvrk", "psm_tool_gripper1_joint"), -2.5)
        #     gym.set_joint_target_position(envs[i], gym.get_joint_handle(envs[i], "dvrk", "psm_tool_gripper2_joint"), -3.0)         

        #     g_1_pos = 0.35
        #     g_2_pos = -0.35
        #     dof_states = gym.get_actor_dof_states(envs[i], dvrk_handles[i], gymapi.STATE_POS)
        #     if dof_states['pos'][8] < 0.35:                                     
                                    
        #         gym.set_joint_target_position(envs[i], gym.get_joint_handle(envs[i], "dvrk", "psm_tool_gripper1_joint"), g_1_pos)
        #         gym.set_joint_target_position(envs[i], gym.get_joint_handle(envs[i], "dvrk", "psm_tool_gripper2_joint"), g_2_pos)         
        
                
        #         # print("***Current x, y, z: ", current_pose["pose"]["p"]["x"], current_pose["pose"]["p"]["y"], current_pose["pose"]["p"]["z"] )
                
        #         # Save robot and object states for reset
        #         gym.refresh_particle_state_tensor(sim)
        #         saved_object_state = deepcopy(gymtorch.wrap_tensor(gym.acquire_particle_state_tensor(sim))) 
        #         init_robot_state = deepcopy(gym.get_actor_rigid_body_states(envs[i], dvrk_handles[i], gymapi.STATE_ALL))
        #         shapesrv_start_time = timeit.default_timer()

        #         state = "get shape servo plan"



        ############################################################################
        # get shape servo plan: sample random delta x, y, z and set up MoveIt
        ############################################################################
        if state == "get shape servo plan":
            rospy.loginfo("**Current state: " + state) 

            current_pose = deepcopy(gym.get_actor_rigid_body_states(envs[i], dvrk_handles[i], gymapi.STATE_POS)[-3])
            print("current_pose:", current_pose["pose"]["p"])

            # delta_pose = rospy.wait_for_message('/ee_xyz', Pose)   
            pose =  rospy.wait_for_message('/PSM1/measured_cp', TransformStamped)

            print("desired pose:", pose)
            cartesian_pose = Pose()
            cartesian_pose.orientation.x = pose.transform.rotation.z
            cartesian_pose.orientation.y = pose.transform.rotation.x
            cartesian_pose.orientation.z = pose.transform.rotation.y
            cartesian_pose.orientation.w = pose.transform.rotation.w
            # cartesian_pose.orientation.x = 0
            # cartesian_pose.orientation.y = 0.707107
            # cartesian_pose.orientation.z = 0.707107
            # cartesian_pose.orientation.w = 0
            # cartesian_pose.position.x = pose.transform.translation.x
            # cartesian_pose.position.y = pose.transform.translation.y
            # cartesian_pose.position.z = 0 #pose.transform.translation.z
            cartesian_pose.position.x = -current_pose["pose"]["p"]["x"] 
            cartesian_pose.position.y = -current_pose["pose"]["p"]["y"] 
            cartesian_pose.position.z = current_pose["pose"]["p"]["z"] - ROBOT_Z_OFFSET    
            # Set up moveit for the above delta x, y, z
            plan_traj = dc_client.arm_moveit_planner_client(go_home=False, cartesian_goal=cartesian_pose, current_position=robot.get_full_joint_positions())
            if (not plan_traj):
                rospy.logerr('Can not find moveit plan to shape servo. \n')  
                cartesian_pose = Pose()
                cartesian_pose.orientation.x = 0
                cartesian_pose.orientation.y = 0.707107
                cartesian_pose.orientation.z = 0.707107
                cartesian_pose.orientation.w = 0
                cartesian_pose.position.x = -current_pose["pose"]["p"]["x"] 
                cartesian_pose.position.y = -current_pose["pose"]["p"]["y"] 
                cartesian_pose.position.z = current_pose["pose"]["p"]["z"] - ROBOT_Z_OFFSET         
                plan_traj = dc_client.arm_moveit_planner_client(go_home=False, cartesian_goal=cartesian_pose, current_position=robot.get_full_joint_positions())

            traj_index = 0
            frame_count = 0
            state = "move to goal"


        ############################################################################
        # move to goal: Move robot gripper to the desired delta x, y, z using MoveIt
        ############################################################################
        if state == "move to goal":      
            g_1_pos = 0.35
            g_2_pos = -0.35
            # Set target joint positions
            dof_states = robot.get_full_joint_positions()
            plan_traj_with_gripper = [plan+[g_1_pos,g_2_pos] for plan in plan_traj]
            pos_targets = np.array(plan_traj_with_gripper[traj_index], dtype=np.float32)
            gym.set_actor_dof_position_targets(envs[0], dvrk_handles[0], pos_targets)                
            

            if traj_index <= len(plan_traj) - 2:
                if np.allclose(dof_states[:8], pos_targets[:8], rtol=0, atol=0.1):
                    traj_index += 1 
            else:
                if np.allclose(dof_states[:8], pos_targets[:8], rtol=0, atol=0.02):
                    traj_index += 1   

            if traj_index == len(plan_traj):
                traj_index = 0  
                rospy.loginfo("Succesfully executed moveit arm plan. Let's record point cloud!!")                       
                state = "get shape servo plan"
                
            frame_count += 1
            if frame_count % 10 == 0:
                state = "get shape servo plan"
            
            
        ############################################################################
        # grasp object: Reset robot and object to the initial state
        ############################################################################  
        if state == "reset":   
            rospy.logwarn("==== RESETTING ====")
            frame_count = 0
            sample_count = 0

            gym.set_actor_rigid_body_states(envs[i], dvrk_handles[i], init_robot_state, gymapi.STATE_ALL) 
            # gym.set_particle_state_tensor(sim, gymtorch.unwrap_tensor(saved_object_state))
            print("Sucessfully reset robot and object")                

            # Reset recorded data
            recorded_particle_states = []
            recorded_pcs = []
            recorded_poses = []            

            state = "get shape servo plan"
            # Failed move to preshape -> back to home
            if fail_mtp:
                state = "home"  
                fail_mtp = False
        


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

