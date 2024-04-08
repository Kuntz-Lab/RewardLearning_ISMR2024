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

def create_sim(gym):
    sim_type = gymapi.SIM_FLEX
    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

    sim_params.substeps = 4
    sim_params.dt = 1./60.
    sim_params.flex.solver_type = 5
    sim_params.flex.num_outer_iterations = 4
    sim_params.flex.num_inner_iterations = 50
    sim_params.flex.relaxation = 0.7
    sim_params.flex.warm_start = 0.1
    sim_params.flex.shape_collision_distance = 5e-4
    sim_params.flex.contact_regularization = 1.0e-6
    sim_params.flex.shape_collision_margin = 1.0e-4
    sim_params.flex.deterministic_mode = True    

    gpu_physics = 0
    gpu_render = 0
    return gym.create_sim(gpu_physics, gpu_render, sim_type,
                          sim_params), sim_params


def create_ground_plane(gym, sim, up_axis='z'):
    # add ground plane
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1) if up_axis == 'z' else gymapi.Vec3(0.0, 1.0, 0.0) # z-up ground
    gym.add_ground(sim, plane_params)

def bounded_rand_float(number, small_offset=0.0015):
    return random.uniform(number-small_offset, number+small_offset)

def create_envs(gym, sim):
    num_envs = 9
    spacing = 1.0
    num_per_row = int(np.sqrt(num_envs))
    # define plane on which environments are initialized
    lower = gymapi.Vec3(-spacing, 0.0, -spacing)
    upper = gymapi.Vec3(spacing, spacing, spacing)

    # load robot asset
    dvrk_asset = default_dvrk_asset(gym, sim)
    dvrk_pose = gymapi.Transform()
    dvrk_pose.p = gymapi.Vec3(0.0, 0.0, ROBOT_Z_OFFSET)
    dvrk_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)

    # Load deformable objects
    num_soft_assets = 3
    asset_root = "/home/dvrk/Documents/IsaacGym_Preview_3_Package/isaacgym/assets/urdf"
    soft_asset_files = ["new_ball.urdf", "cube.urdf", "long_box.urdf"]


    # cache some common handles for later use
    envs = []
    dvrk_handles = []
    object_handles = []
    object0_handles = []
    object1_handles = []
    object2_handles = []
    soft_poses_handles = []

    for i in range(num_envs):
        rand_num = random.random()
        # create a set of 3 soft assets for envs[i] with random positions
        if rand_num < 0.3:
            soft_poses_p = [gymapi.Vec3(bounded_rand_float(-0.18715141713619232), bounded_rand_float(-0.19574125111103058), 0.16+0.015), 
                            gymapi.Vec3(bounded_rand_float(-0.0), bounded_rand_float(-0.19574125111103058), 0.16+0.015), 
                            gymapi.Vec3(bounded_rand_float(0.18715141713619232), bounded_rand_float(-0.19574125111103058), 0.16+0.015)]
        else:
            if rand_num < 0.6:
                soft_poses_p = [gymapi.Vec3(bounded_rand_float(-0.0), bounded_rand_float(-0.19574125111103058), 0.16+0.015), 
                                gymapi.Vec3(bounded_rand_float(-0.18715141713619232), bounded_rand_float(-0.19574125111103058), 0.16+0.015), 
                                gymapi.Vec3(bounded_rand_float(0.18715141713619232), bounded_rand_float(-0.19574125111103058), 0.16+0.015)]
            else:
                soft_poses_p = [gymapi.Vec3(bounded_rand_float(-0.0), bounded_rand_float(-0.19574125111103058), 0.16+0.015), 
                                gymapi.Vec3(bounded_rand_float(0.18715141713619232), bounded_rand_float(-0.19574125111103058), 0.16+0.015), 
                                gymapi.Vec3(bounded_rand_float(-0.18715141713619232), bounded_rand_float(-0.19574125111103058), 0.16+0.015)]
        
        # soft_poses[a][b] is the pose of the bth soft asset of the ath environment
        soft_poses = []
        # soft_assets[b] is the bth soft asset of the current environment
        soft_assets = []
        for j in range(num_soft_assets):
            soft_pose = gymapi.Transform()
            soft_pose.p = soft_poses_p[j]
            soft_poses.append(soft_pose)
            soft_thickness = 0.0005    # important to add some thickness to the soft body to avoid interpenetrations   
            asset_options = gymapi.AssetOptions()
            asset_options.fix_base_link = True
            asset_options.thickness = soft_thickness
            asset_options.disable_gravity = True

            soft_asset = gym.load_asset(sim, asset_root, soft_asset_files[j], asset_options)
            soft_assets.append(soft_asset)

        soft_poses_handles.append(soft_poses)
        
        # create env
        env = gym.create_env(sim, lower, upper, num_per_row)
        envs.append(env)

        # add dvrk
        dvrk_handle = gym.create_actor(env, dvrk_asset, dvrk_pose, "dvrk", i, 1, segmentationId=11)    
        dvrk_handles.append(dvrk_handle)    

        # add soft obj 0           
        soft_actor = gym.create_actor(env, soft_assets[0], soft_poses[0], "soft0", i+1, 1)
        color = gymapi.Vec3(1,0,0)
        gym.set_rigid_body_color(env, soft_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
        object0_handles.append(soft_actor)

         # add soft obj 1            
        soft_actor = gym.create_actor(env, soft_assets[1], soft_poses[1], "soft1", i+1, 1)
        color = gymapi.Vec3(0,1,0)
        gym.set_rigid_body_color(env, soft_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
        object1_handles.append(soft_actor)

         # add soft obj 2            
        soft_actor = gym.create_actor(env, soft_assets[2], soft_poses[2], "soft2", i+1, 1)
        color = gymapi.Vec3(0,0,1)
        gym.set_rigid_body_color(env, soft_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
        object2_handles.append(soft_actor)

        init_dvrk_joints(gym, envs[i], dvrk_handles[i])

    # DOF Properties and Drive Modes 
    dof_props = gym.get_actor_dof_properties(envs[0], dvrk_handles[0])
    dof_props["driveMode"].fill(gymapi.DOF_MODE_POS) #original: gymapi.DOF_MODE_EFFORT
    dof_props["stiffness"].fill(200.0)
    dof_props["damping"].fill(40.0)
    dof_props["stiffness"][8:].fill(1)
    dof_props["damping"][8:].fill(2)  
        
    for i, env in enumerate(envs):
        gym.set_actor_dof_properties(envs[i], dvrk_handles[i], dof_props)    # set dof properties 

    object_handles.append(object0_handles)
    object_handles.append(object1_handles)
    object_handles.append(object2_handles)

    return envs, dvrk_handles, object_handles, num_envs, num_per_row, soft_poses_handles


def create_viewer_cameras(gym, envs, num_envs, num_per_row):
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        print("*** Failed to create viewer")
        quit()

    cam_pos = gymapi.Vec3(0.0, -0.440001, 1.5) #z=1.5 instead of 1.0
    cam_target = gymapi.Vec3(0.0, -0.44, 0.1)

    middle_env = envs[num_envs // 2]
    gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)
    #gym.viewer_camera_look_at(viewer, envs[0], cam_pos, cam_target)

    # Camera for point cloud setup
    cam_handles = []
    cam_width = 2000
    cam_height = 2000 
    cam_positions = gymapi.Vec3(0.0, -0.4400001, 0.7)
    cam_targets = gymapi.Vec3(0.0, -0.44, 0.1)

    for i, env in enumerate(envs):
        cam_handle, cam_prop = setup_cam(gym, envs[i], cam_width, cam_height, cam_positions, cam_targets)
        cam_handles.append(cam_handle)

    return viewer, cam_handles, cam_height, cam_width


def generate_trajs(sim, gym, envs, dvrk_handles, object_handles, viewer, cam_handles, cam_height, cam_width, soft_poses_handles, args):

    # loop over different environments that have different positions of soft objects
    for i, env in enumerate(envs):

        rospy.init_node('shape_servo_control')
        #rospy.logerr("======Loading object ... " + str(args.obj_name))  

        # Some important paramters
        init_dvrk_joints(gym, envs[i], dvrk_handles[i])  # Initilize robot's joints 

        all_done = False
        state = "grasp object" #"home"

        data_recording_path = f"/home/dvrk/motion_plan_data/test_demos/env{i}"
        video_main_path = f"/home/dvrk/motion_plan_data/test_videos/env{i}"
        os.makedirs(video_main_path, exist_ok=True)
        os.makedirs(data_recording_path, exist_ok=True)

        frame_count = 0
        vis_frame_count = 0
        data_point_count = 11+len(os.listdir(data_recording_path))
        max_data_point_count = data_point_count+10
        eef_states = []
        num_image = 0
        record_video = True

        dc_client = GraspClient()
 
        close_viewer = False
        #robot for planning
        robot = Robot(gym, sim, envs[i], dvrk_handles[i])

        num_soft_objects = len(soft_poses_handles[i])
        soft_goal_index = 0

        while (not close_viewer) and (not all_done):

            soft_goal_index = soft_goal_index%num_soft_objects

            if args.headless:
                close_viewer = gym.query_viewer_has_closed(viewer)

            # step the physics
            gym.simulate(sim)
            gym.fetch_results(sim, True)

            # # Record videos
            # if record_video and vis_frame_count % 10 == 0:
            #     video_path = os.path.join(video_main_path, f"demo_{data_point_count}")
            #     if not os.path.isdir(video_path):
            #         os.mkdir(video_path)

            #     gym.render_all_camera_sensors(sim)
            #     im = gym.get_camera_image(sim, envs[i], cam_handles[i], gymapi.IMAGE_COLOR).reshape((cam_height,cam_width,4))
            #     im = Image.fromarray(im)            
            #     img_path =  os.path.join(video_path, "image" + f'{num_image:03}' + ".png")            
            #     im.save(img_path)
            #     num_image += 1
            # vis_frame_count += 1
            
            if state == "grasp object":
                rospy.loginfo("**Current state: " + state)       
                gym.set_joint_target_position(envs[i], gym.get_joint_handle(envs[i], "dvrk", "psm_main_insertion_joint"), 0.24) 
                gym.set_joint_target_position(envs[i], gym.get_joint_handle(envs[i], "dvrk", "psm_tool_gripper1_joint"), -2.5)
                gym.set_joint_target_position(envs[i], gym.get_joint_handle(envs[i], "dvrk", "psm_tool_gripper2_joint"), -3.0)         

                g_1_pos = 0.35
                g_2_pos = -0.35
                dof_states = gym.get_actor_dof_states(envs[i], dvrk_handles[i], gymapi.STATE_POS)
                if dof_states['pos'][8] < 0.35:                                     
                                        
                    gym.set_joint_target_position(envs[i], gym.get_joint_handle(envs[i], "dvrk", "psm_tool_gripper1_joint"), g_1_pos)
                    gym.set_joint_target_position(envs[i], gym.get_joint_handle(envs[i], "dvrk", "psm_tool_gripper2_joint"), g_2_pos)         
                    
                    # Save robot and object states for reset
                    gym.refresh_particle_state_tensor(sim)
                    init_robot_state = deepcopy(gym.get_actor_rigid_body_states(envs[i], dvrk_handles[i], gymapi.STATE_ALL))
                    state = "get shape servo plan"

            #Todo
            if state == "get shape servo plan":
                rospy.loginfo("**Current state: " + state) 

                object_pose = soft_poses_handles[i][soft_goal_index]
                cartesian_pose = Pose()
                cartesian_pose.orientation.x = 0
                cartesian_pose.orientation.y = 0.707107
                cartesian_pose.orientation.z = 0.707107
                cartesian_pose.orientation.w = 0
                cartesian_pose.position.x = -object_pose.p.x
                cartesian_pose.position.y = -object_pose.p.y
                cartesian_pose.position.z = object_pose.p.z - ROBOT_Z_OFFSET

                # Set up moveit with the above pose as the goal
                plan_traj = dc_client.arm_moveit_planner_client(go_home=False, cartesian_goal=cartesian_pose, current_position=robot.get_full_joint_positions())
                if (not plan_traj):
                    print("++++++++++++++++++++++ FAIL OBJECT POS: ", object_pose.p.x, object_pose.p.y, object_pose.p.z, '\n')
                    rospy.logerr('Can not find moveit plan to shape servo. Ignore this grasp.\n')  
                    state = "reset"
                    soft_goal_index = 0
                    #continue
                else:
                    print("++++++++++++++++++++++ SUCCESS OBJECT POS: ", object_pose.p.x, object_pose.p.y, object_pose.p.z, '\n')
                    #print("++++++++++++++++++++++ SUCCESS PLAN: ", plan_traj, '\n')
                    state = "move to goal"
                    traj_index = 0

                frame_count = 0

            if state == "move to goal":      
                if frame_count % 5 == 0:
                    eef_states.append(deepcopy(gym.get_actor_rigid_body_states(envs[i], dvrk_handles[i], gymapi.STATE_ALL)[-3]))
                frame_count += 1  

                # Set target joint positions
                dof_states = robot.get_full_joint_positions()
                plan_traj_with_gripper = [plan+[g_1_pos,g_2_pos] for plan in plan_traj]
                pos_targets = np.array(plan_traj_with_gripper[traj_index], dtype=np.float32)
                gym.set_actor_dof_position_targets(envs[i], dvrk_handles[i], pos_targets)
                                
                #print("++++++++++++++++++++++ MOVE TO TARGET: ", pos_targets, '\n')

                if traj_index <= len(plan_traj) - 2:
                    if np.allclose(dof_states[:8], pos_targets[:8], rtol=0, atol=0.1):
                        traj_index += 1 
                else:
                    if np.allclose(dof_states[:8], pos_targets[:8], rtol=0, atol=0.02):
                        traj_index += 1   

                if traj_index == len(plan_traj) and soft_goal_index!=2:
                    traj_index = 0  
                    rospy.loginfo(f"Succesfully executed moveit arm plan to object{soft_goal_index}")                       
                    soft_goal_index+=1
                    state = "get shape servo plan"

                if traj_index == len(plan_traj) and soft_goal_index==2:
                    traj_index = 0
                    rospy.loginfo(f"Succesfully executed moveit arm plan to object{soft_goal_index}")                       
                    soft_goal_index +=1
                    # Save each data point to a pickle file
                    eef_states.append(deepcopy(gym.get_actor_rigid_body_states(envs[i], dvrk_handles[i], gymapi.STATE_ALL)[-3]))

                    if args.save_data:
                        data = eef_states
                        with open(os.path.join(data_recording_path, "sample " + str(data_point_count) + ".pickle"), 'wb') as handle:
                            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)                    
                    

                    print("+++++++++++data_point_count:", data_point_count)
                    data_point_count += 1            

                    # frame_count = 0
                    # eef_states = []
                    state = "reset"


            if state == "reset":   
                rospy.logwarn("==== RESETTING ====")
                frame_count = 0
                num_image = 0
                frame_count = 0
                eef_states = []

                gym.set_actor_rigid_body_states(envs[i], dvrk_handles[i], init_robot_state, gymapi.STATE_ALL) 
                print("Sucessfully reset robot")                

                state = "get shape servo plan"


            if data_point_count >= max_data_point_count:
                all_done = True
                        

            # update the viewer
            gym.step_graphics(sim)
            if not args.headless:
                gym.draw_viewer(viewer, sim, True)
            # Wait for dt to elapse in real time.
            # This synchronizes the physics simulation with the rendering rate.
            #gym.sync_frame_time(sim)   





if __name__ == "__main__":
    gym = gymapi.acquire_gym()

    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--obj_name', default='box_0', type=str, help="select variations of a primitive shape")
    parser.add_argument('--headless', default="False", type=str, help="Index of grasp candidate to test")
    parser.add_argument('--data_recording_path', default="/home/dvrk/shape_servo_data/generalization/multi_boxes_1000Pa/data", type=str, help="path to save the recorded data")
    parser.add_argument('--object_meshes_path', default="/home/dvrk/sim_data/Custom/Custom_mesh/multi_boxes_1000Pa", type=str, help="path to the objects' tet meshe files")
    parser.add_argument('--max_data_point_count', default=30000, type=int, help="path to the objects' tet meshe files")
    parser.add_argument('--save_data', default=True, type=bool, help="True: save recorded data to pickles files")
    args = parser.parse_args()
    args.headless = args.headless == "True"

    sim, sim_params = create_sim(gym)
    create_ground_plane(gym, sim)
    envs, dvrk_handles, object_handles, num_envs, num_per_row, soft_poses_handles = create_envs(gym, sim)

    if not args.headless:
        viewer, cam_handles, cam_height, cam_width = create_viewer_cameras(gym, envs, num_envs, num_per_row)

    start_time = timeit.default_timer()   

    #generate trajectories and save data
    generate_trajs(sim, gym, envs, dvrk_handles, object_handles, viewer, cam_handles, cam_height, cam_width, soft_poses_handles, args)

    print("All done !")
    print("Elapsed time", timeit.default_timer() - start_time)
    if not args.headless:
        gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
        
