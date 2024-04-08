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

from geometry_msgs.msg import PoseStamped, Pose

from util.isaac_utils import *
from util.grasp_utils import GraspClient

from core import Robot
from behaviors import MoveToPose, TaskVelocityControl
import argparse
from PIL import Image


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
    parser.add_argument('--save_data', default=True, type=bool, help="True: save recorded data to pickles files")
    args = parser.parse_args()
    args.headless = args.headless == "True"

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


    # Load deformable object
    asset_root = "/home/dvrk/Documents/IsaacGym_Preview_3_Package/isaacgym/assets/urdf"
    soft_asset_file = "new_ball.urdf"    
    soft_pose = gymapi.Transform()
    soft_pose.p = gymapi.Vec3(0.0, -0.44, 0.16+0.015)
    # soft_pose.r = gymapi.Quat(0.0, 0.0, 0.707107, 0.707107)
    soft_thickness = 0.0005    # important to add some thickness to the soft body to avoid interpenetrations

    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.thickness = soft_thickness
    asset_options.disable_gravity = True

    soft_asset = gym.load_asset(sim, asset_root, soft_asset_file, asset_options)       

    wall_asset_root = "/home/dvrk/dvrk_ws/src/dvrk_env/shape_servo_control/src/teleoperation/random_stuff"
    wall_asset_file = "wall.urdf"  
    wall_asset = gym.load_asset(sim, wall_asset_root, wall_asset_file, asset_options)  


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
        #soft_actor = gym.create_actor(env, soft_asset, soft_pose, "soft", i, 0)
        soft_actor = gym.create_actor(env, soft_asset, soft_pose, "soft", i+1, 1)
        color = gymapi.Vec3(1,0,0)
        gym.set_rigid_body_color(env, soft_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
        object_handles.append(soft_actor)

        wall_actor = gym.create_actor(env, wall_asset, soft_pose, "wall", i+1, 1)
        color = gymapi.Vec3(0,1,0)
        gym.set_rigid_body_color(env, wall_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

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
        # cam_pos = gymapi.Vec3(0.5, -0.5, 0.25)
        # cam_target = gymapi.Vec3(0.0, -0.36, 0.1)

        cam_pos = gymapi.Vec3(0.0, -0.440001, 1)
        cam_target = gymapi.Vec3(0.0, -0.44, 0.1)

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


       

    '''
    Main simulation stuff starts from here
    '''
    rospy.init_node('shape_servo_control')
    rospy.logerr("======Loading object ... " + str(args.obj_name))  

    # Some important paramters
    init_dvrk_joints(gym, envs[i], dvrk_handles[i])  # Initilize robot's joints    

    all_done = False
    state = "home"
    
    
    data_recording_path = "/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_2/test_demos"
    video_main_path = "/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_2/test_videos"

    frame_count = 0
    vis_frame_count = 0
    data_point_count = 11+len(os.listdir(data_recording_path))
    max_data_point_count = data_point_count+10
    eef_states = []
    num_image = 0
    record_video = True


    dc_client = GraspClient()

    
    start_time = timeit.default_timer()   
    close_viewer = False
    robot = Robot(gym, sim, envs[0], dvrk_handles[0])


    while (not close_viewer) and (not all_done): 

        if not args.headless:
            close_viewer = gym.query_viewer_has_closed(viewer)  

        # step the physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)
 
        # Record videos
        if record_video and vis_frame_count % 10 == 0:
            video_path = os.path.join(video_main_path, f"demo_{data_point_count}")
            if not os.path.isdir(video_path):
                os.mkdir(video_path)

            gym.render_all_camera_sensors(sim)
            im = gym.get_camera_image(sim, envs[i], cam_handles[0], gymapi.IMAGE_COLOR).reshape((cam_height,cam_width,4))
            im = Image.fromarray(im)            
            img_path =  os.path.join(video_path, "image" + f'{num_image:03}' + ".png")            
            im.save(img_path)
            num_image += 1
        vis_frame_count += 1


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
                state = "grasp object"



        ############################################################################
        # grasp object: close gripper
        ############################################################################        
        if state == "grasp object":             
            rospy.loginfo("**Current state: " + state)
            gym.set_joint_target_position(envs[i], gym.get_joint_handle(envs[i], "dvrk", "psm_tool_gripper1_joint"), -2.5)
            gym.set_joint_target_position(envs[i], gym.get_joint_handle(envs[i], "dvrk", "psm_tool_gripper2_joint"), -3.0)         

            g_1_pos = 0.35
            g_2_pos = -0.35
            dof_states = gym.get_actor_dof_states(envs[i], dvrk_handles[i], gymapi.STATE_POS)
            if dof_states['pos'][8] < 0.35:                                     
                                    
                gym.set_joint_target_position(envs[i], gym.get_joint_handle(envs[i], "dvrk", "psm_tool_gripper1_joint"), g_1_pos)
                gym.set_joint_target_position(envs[i], gym.get_joint_handle(envs[i], "dvrk", "psm_tool_gripper2_joint"), g_2_pos)         
        
                
                # print("***Current x, y, z: ", current_pose["pose"]["p"]["x"], current_pose["pose"]["p"]["y"], current_pose["pose"]["p"]["z"] )
                
                # Save robot and object states for reset
                gym.refresh_particle_state_tensor(sim)
                saved_object_state = deepcopy(gymtorch.wrap_tensor(gym.acquire_particle_state_tensor(sim))) 
                init_robot_state = deepcopy(gym.get_actor_rigid_body_states(envs[i], dvrk_handles[i], gymapi.STATE_ALL))
                shapesrv_start_time = timeit.default_timer()

                state = "get shape servo plan"



        ############################################################################
        # get shape servo plan: sample random delta x, y, z and set up MoveIt
        ############################################################################
        if state == "get shape servo plan":
            rospy.loginfo("**Current state: " + state) 

            current_pose = deepcopy(gym.get_actor_rigid_body_states(envs[i], dvrk_handles[i], gymapi.STATE_POS)[-3])
            print("+++++++++++++++", gym.get_actor_rigid_body_states(envs[i], dvrk_handles[i], gymapi.STATE_ALL).shape)

            # delta_pose = rospy.wait_for_message('/ee_xyz', Pose)    
            delta_x = 0.00
            delta_y = 0.13
            delta_z = -0.00


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
                traj_index = 0
            
            
            frame_count = 0

        ############################################################################
        # move to goal: Move robot gripper to the desired delta x, y, z using MoveIt
        ############################################################################
        if state == "move to goal":      

            if frame_count % 5 == 0:
                eef_states.append(deepcopy(gym.get_actor_rigid_body_states(envs[i], dvrk_handles[i], gymapi.STATE_ALL)[-3]))
            frame_count += 1  

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
                rospy.loginfo("Succesfully executed moveit arm plan. Let's record demos!!")                       
                # state = "get shape servo plan"
                # state = "test"

                # Save each data point to a pickle file
                
                eef_states.append(deepcopy(gym.get_actor_rigid_body_states(envs[i], dvrk_handles[i], gymapi.STATE_ALL)[-3]))

                if args.save_data:
                    data = eef_states
                    with open(os.path.join(data_recording_path, "sample " + str(data_point_count) + ".pickle"), 'wb') as handle:
                        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)                    
                

                print("+++++++++++data_point_count:", data_point_count)
                data_point_count += 1            

                frame_count = 0
                eef_states = []
                state = "reset"


        # if state == "test":
        #     final_pose = deepcopy(gym.get_actor_rigid_body_states(envs[i], dvrk_handles[i], gymapi.STATE_POS)[-3])["pose"]["p"]
        #     init_pose = current_pose["pose"]["p"]
            # print("*********************************")
            # print("delta x:", final_pose["x"] - init_pose["x"])
            # print("delta y:", final_pose["y"] - init_pose["y"])
            # print("delta z:", final_pose["z"] - init_pose["z"])

            # frame_count += 1
            # if frame_count % 10 == 0:
            #     state = "get shape servo plan"
            
            
        ############################################################################
        # grasp object: Reset robot and object to the initial state
        ############################################################################  
        if state == "reset":   
            rospy.logwarn("==== RESETTING ====")
            frame_count = 0
            num_image = 0

            gym.set_actor_rigid_body_states(envs[i], dvrk_handles[i], init_robot_state, gymapi.STATE_ALL) 
            print("Sucessfully reset robot")                

            state = "get shape servo plan"


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

