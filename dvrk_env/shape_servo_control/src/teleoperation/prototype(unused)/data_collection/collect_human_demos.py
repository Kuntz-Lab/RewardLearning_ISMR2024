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


sys.path.append(os.path.join(rp.get_pkg_dir('shape_servo_control'), "src/teleoperation/tele_utils"))
from teleop_utils import * #console, reset_mtm, wait_for_message_custom

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
    # pose = list(np.array(init_pose) + np.random.uniform(low=[-0.1,-0.03,-0.05], high=[0.1,0.02,0.05], size=3))
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

    # DOF Properties and Drive Modes 
    dof_props = gym.get_actor_dof_properties(envs[0], dvrk_handles[0])
    dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
    dof_props["stiffness"].fill(200.0)
    dof_props["damping"].fill(40.0)
    dof_props["stiffness"][8:].fill(1)
    dof_props["damping"][8:].fill(2)  
    vel_limits = dof_props['velocity'] 
    
    for env in envs:
        gym.set_actor_dof_properties(env, dvrk_handles[i], dof_props)    # set dof properties 


    # Viewer camera setup
    if not args.headless:

        cam_pos = gymapi.Vec3(0.2, -0.34, 0.4)
        cam_target = gymapi.Vec3(0.0, -0.44, 0.1)

        middle_env = envs[num_envs // 2 + num_per_row // 2]
        gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)

    
    # Camera for point cloud setup
    cam_handles = []
    cam_width = 300 #256
    cam_height = 300 #256 
    # cam_positions = gymapi.Vec3(0.17, -0.62, 0.2)
    # cam_targets = gymapi.Vec3(0.0, 0.40-0.86, 0.01)
    # low=[-0.1,-0.03,-0.05], high=[0.1,0.02,0.05]
    cam_positions = gymapi.Vec3(0.2, -0.54, 0.16+0.015 + 0.1)
    cam_targets = gymapi.Vec3(0.0, -0.44, 0.16+0.015)

    cam_handle, cam_prop = setup_cam(gym, envs[0], cam_width, cam_height, cam_positions, cam_targets)
    for i, env in enumerate(envs):
        cam_handles.append(cam_handle)

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
    
    data_recording_path = "/home/dvrk/LfD_data/group_meeting/demos"
    video_main_path = "/home/dvrk/LfD_data/group_meeting/images"    
    os.makedirs(data_recording_path, exist_ok=True)
    
    frame_count = 0
    sample_count = len(os.listdir(data_recording_path)) #Change #0
    max_sample_count = 20000#10 #20#3
    group_count = 0
    max_group_count = 10000 #10 #20
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


    dc_client = GraspClient()

    
    start_time = timeit.default_timer()   
    close_viewer = False
    robot = Robot(gym, sim, envs[0], dvrk_handles[0])

    record_partial_pc = True
    rospy.logerr(f"record_partial_pc: {record_partial_pc}")
    
    cs = console()


    while (not close_viewer) and (not all_done): 

        if not args.headless:
            close_viewer = gym.query_viewer_has_closed(viewer)  

        # step the physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)
 
        # # Record videos
        # if record_video and vis_frame_count == 0: #vis_frame_count % 10 == 0:
        #     video_path = os.path.join(video_main_path, f"scene_raw_image")
        #     os.makedirs(video_path, exist_ok=True)

        #     gym.render_all_camera_sensors(sim)
        #     im = gym.get_camera_image(sim, envs[i], vis_cam_handles[0], gymapi.IMAGE_COLOR).reshape((vis_cam_height,vis_cam_width,4))
        #     im = Image.fromarray(im)            
        #     img_path =  os.path.join(video_path, "scene_raw_image.png")            
        #     im.save(img_path)
        #     num_image += 1
            
        #     all_done = True
            
        # vis_frame_count += 1


        if state == "home" :   
            frame_count += 1
            gym.set_joint_target_position(envs[0], gym.get_joint_handle(envs[0], "dvrk", "psm_main_insertion_joint"), 0.103)            
            
            if frame_count == 10:
                rospy.loginfo("**Current state: " + state)
                frame_count = 0
                
                if first_time:
                    init_robot_state = deepcopy(gym.get_actor_rigid_body_states(envs[i], dvrk_handles[i], gymapi.STATE_ALL))
                    current_pose = deepcopy(gym.get_actor_rigid_body_states(envs[i], dvrk_handles[i], gymapi.STATE_POS)[-3])
                    # print("current_pose:", current_pose["pose"]["p"])
      
            
                    _,init_pose = get_pykdl_client(robot.get_arm_joint_positions())
                    init_eulers = transformations.euler_from_matrix(init_pose)

                    dof_props['driveMode'][:8].fill(gymapi.DOF_MODE_VEL)
                    dof_props["stiffness"][:8].fill(0.0)
                    dof_props["damping"][:8].fill(200.0)
                    gym.set_actor_dof_properties(env, dvrk_handles[i], dof_props) 

                if record_partial_pc:
                    saved_partial_pc = get_partial_point_cloud(gym, sim, envs[0], cam_handle, cam_prop, color=[1,0,0], min_z = 0.1, visualization=False)

                state = "get shape servo plan"
        ############################################################################
        # get shape servo plan: sample random delta x, y, z and set up MoveIt
        ############################################################################
        if state == "get shape servo plan":
            rospy.loginfo("**Current state: " + state) 



            # delta_pose = rospy.wait_for_message('/ee_xyz', Pose)   
            desired_pose =  rospy.wait_for_message('/PSM1/measured_cp', TransformStamped)
            desired_eulers = transformations.euler_from_quaternion([desired_pose.transform.rotation.y, desired_pose.transform.rotation.x,desired_pose.transform.rotation.z,desired_pose.transform.rotation.w])
            # print("desired pose:", desired_pose)
            print("desired eulers:", desired_eulers)

            delta_x, delta_y, delta_z = desired_pose.transform.translation.x, desired_pose.transform.translation.y, desired_pose.transform.translation.z
            # delta_x *= 3
            # delta_y *= 3
            # delta_z *= 3
            
            delta_alpha, delta_beta, delta_gamma = -desired_eulers[1], desired_eulers[2], -(desired_eulers[0]  - np.pi/2)
            x = delta_x + init_pose[0,3]
            y = delta_y + init_pose[1,3]
            z = delta_z + init_pose[2,3]
            alpha = delta_alpha + init_eulers[0]
            beta = delta_beta + init_eulers[1]
            gamma = delta_gamma + init_eulers[2]

            print("alpha, beta, gamma:", alpha, beta, gamma)

            tvc_behavior = TaskVelocityControl2([x,y,z,alpha,beta,gamma], robot, sim_params.dt, 3, vel_limits=vel_limits, pos_threshold = 1e-3, ori_threshold=5e-2)
            

            # frame_count = 0
            state = "move to goal"

        ############################################################################
        # move to goal: Move robot gripper to the desired delta x, y, z using MoveIt
        ############################################################################
        if state == "move to goal":      
            gym.set_joint_target_position(envs[i], gym.get_joint_handle(envs[i], "dvrk", "psm_tool_gripper1_joint"), 0.4)
            gym.set_joint_target_position(envs[i], gym.get_joint_handle(envs[i], "dvrk", "psm_tool_gripper2_joint"), -0.4) 

            action = tvc_behavior.get_action() 

            if action is not None: 
                gym.set_actor_dof_velocity_targets(robot.env_handle, robot.robot_handle, action.get_joint_position())  
            else:
                state = "get shape servo plan"    

            if frame_count % 2 == 0:
                state = "get shape servo plan"
                
            if frame_count % 5 == 0:
                eef_states.append(deepcopy(gym.get_actor_rigid_body_states(envs[i], dvrk_handles[i], gymapi.STATE_ALL)[-3]))
                

            # if frame_count == 400:
            if frame_count % 10 == 0:
                pedal_reset_message = get_pedal_message(topic_type=Joy, pedal_name="clutch")
                if pedal_reset_message is not None and pedal_reset_message.buttons[0] != 0:
                    rospy.logerr("IGNORE this demonstration. Resetting ...")
                    reset_mtm(cs)
                    state = "reset"                    
                
                else:
                    pedal_stop_message = get_pedal_message(topic_type=Joy, pedal_name="camera")
                    # rospy.logerr(pedal_stop_message)                
                    # pedal_stop_message = wait_for_message_custom('/footpedals/camera', Joy, timeout=0.01)               
                    # if pedal_stop_message is not None and pedal_stop_message.buttons[0] == 1:
                    
                    if pedal_stop_message is not None and pedal_stop_message.buttons[0] != 0:
                    
                        if args.save_data:
                            
                            ee_pose = deepcopy(gym.get_actor_rigid_body_states(envs[i], dvrk_handles[i], gymapi.STATE_ALL)[-3])


                            if record_partial_pc:
                                
                                rospy.logerr(f"pc shape: {saved_partial_pc.shape}")

                                data = {"traj": eef_states,\
                                        "goal pose": [soft_pose.p.x, soft_pose.p.y, soft_pose.p.z],\
                                        "partial_pc": down_sampling(saved_partial_pc, num_pts=256)} 

                            else:
                                data = {"traj": eef_states,\
                                        "goal pose": [soft_pose.p.x, soft_pose.p.y, soft_pose.p.z]}                        
                            
                            with open(os.path.join(data_recording_path, f"group {group_count} sample {sample_count}.pickle"), 'wb') as handle:
                                pickle.dump(data, handle, protocol=3)   
                    
                            print(f"Len trajectory: {len(eef_states)}")  
                                        
                            rospy.logerr("Sucessful demonstration. Resetting ...")
                            rospy.logerr(f"SAVED group {group_count} sample {sample_count}") 
                            sample_count += 1 
                        
                        reset_mtm(cs)
                        state = "reset"

            frame_count += 1

                    


        if sample_count >= max_sample_count:
            state = "reset"
            
            
        ############################################################################
        # grasp object: Reset robot and object to the initial state
        ############################################################################  
        if state == "reset":   
            rospy.logwarn("==== RESETTING ====")
            frame_count = 0
            num_image = 0
            eef_states = []

            # dof_props['driveMode'][:8].fill(gymapi.DOF_MODE_POS)
            # dof_props["stiffness"][:8].fill(200.0)
            # dof_props["damping"][:8].fill(40.0)
            gym.set_actor_rigid_body_states(robot.env_handle, robot.robot_handle, init_robot_state, gymapi.STATE_ALL) 
            # gym.set_actor_dof_velocity_targets(robot.env_handle, robot.robot_handle, [0]*8)
            # gym.set_actor_dof_properties(robot.env_handle, robot.robot_handle, dof_props)  
            # gym.set_actor_dof_position_targets(robot.env_handle, robot.robot_handle, [0,0,0,0,0.103,0,0,0,1.5,0.8]) 
            print("Sucessfully reset robot")   
            
            
            state = "get shape servo plan"    
      

            if sample_count >= max_sample_count:
                sample_count = 0
                group_count += 1
   
                state = "home"



        if group_count >= max_group_count:
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

