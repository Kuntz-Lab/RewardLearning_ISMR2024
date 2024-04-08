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
# import pickle5 as pickle
import timeit
import open3d

from geometry_msgs.msg import PoseStamped, Pose, TransformStamped

from util.isaac_utils import *
from util.grasp_utils import GraspClient
from util.isaac_utils import fix_object_frame, get_pykdl_client

from core import Robot
from behaviors import MoveToPose, TaskVelocityControl, TaskVelocityControl2
import argparse
import transformations

ROBOT_Z_OFFSET = 0.1 #0.2 #0.25


def default_dvrk_asset(gym, sim):
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


    # Load deformable object
    asset_root = "/home/dvrk/Documents/IsaacGym_Preview_3_Package/isaacgym/assets/urdf"
    soft_asset_file = "new_ball.urdf"    
    # soft_pose = gymapi.Transform()
    # soft_pose.p = gymapi.Vec3(0.0, -0.40, 0.02)
    soft_poses = []
    num_balls = 9 #15
    init_pose = np.array([0.0, -0.40, 0.02])
    origin_x = -0.05
    origin_y = -0.42 #-0.45
    for j in range(num_balls):
        # pose = list(np.array(init_pose)+np.random.uniform(low=[-0.15,-0.15,0], high=[0.15,0.1,0.001], size=3))
        # soft_pose = gymapi.Transform()
        # soft_pose.p = gymapi.Vec3(*pose)
        # soft_poses.append(soft_pose)

        x = origin_x + j // 3 * 0.05
        y = origin_y + j % 3 * 0.05
        soft_pose = gymapi.Transform()
        soft_pose.p = gymapi.Vec3(x, y, 0.02)
        soft_poses.append(soft_pose)

    soft_thickness = 0.0005    # important to add some thickness to the soft body to avoid interpenetrations

    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = False #True
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
    env_objs = []


    for i in range(num_envs):
        # create env
        env = gym.create_env(sim, env_lower, env_upper, num_per_row)
        envs.append(env)

        # add dvrk
        dvrk_handle = gym.create_actor(env, dvrk_asset, dvrk_pose, "dvrk", i, 1, segmentationId=11)    
        dvrk_handles.append(dvrk_handle)    
        

        # add obj            
        # soft_actor = gym.create_actor(env, soft_asset, soft_pose, "soft", i, 0)
        # # soft_actor = gym.create_actor(env, soft_asset, soft_pose, "soft", i+1, 1)
        # color = gymapi.Vec3(1,0,0)
        # gym.set_rigid_body_color(env, soft_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
        # object_handles.append(soft_actor)

        for j in range(num_balls):
            env_obj = gym.create_env(sim, env_lower, env_upper, num_per_row)
            soft_actor = gym.create_actor(env_obj, soft_asset, soft_poses[j], "soft", i, 0)
            # color = gymapi.Vec3(0.75,0,0)
            color = gymapi.Vec3(*list(np.random.uniform(low=0, high=1, size=3)))
            gym.set_rigid_body_color(env_obj, soft_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
            object_handles.append(soft_actor)   
            env_objs.append(env_obj)         

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
    print("======vel_limits:", vel_limits)

    for env in envs:
        gym.set_actor_dof_properties(env, dvrk_handles[i], dof_props)    # set dof properties 


    # Viewer camera setup
    if not args.headless:
        # # cam_pos = gymapi.Vec3(0.5, -0.5, 0.25)
        # cam_pos = gymapi.Vec3(0.0, -1, 0.25)
        # cam_target = gymapi.Vec3(0.0, -0.36, 0.1)

        # cam_pos = gymapi.Vec3(0.0, -0.440001, 1)
        # cam_target = gymapi.Vec3(0.0, -0.44, 0.1)

        # cam_pos = gymapi.Vec3(0.0, -0.40, 1)
        # cam_target = gymapi.Vec3(0.0, -0.44, 0.1)

        cam_pos = gymapi.Vec3(0.2, -0.34, 0.4)
        cam_target = gymapi.Vec3(0.0, -0.44, 0.1)

        middle_env = envs[num_envs // 2 + num_per_row // 2]
        gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)

    
    # # Camera for point cloud setup
    # cam_handles = []
    # cam_width = 256
    # cam_height = 256 
    # cam_positions = gymapi.Vec3(0.17, -0.62, 0.2)
    # cam_targets = gymapi.Vec3(0.0, 0.40-0.86, 0.01)
    # cam_handle, cam_prop = setup_cam(gym, envs[0], cam_width, cam_height, cam_positions, cam_targets)
    # # for i, env in enumerate(envs):
    # #     cam_handles.append(cam_handle)


       

    '''
    Main simulation stuff starts from here
    '''
    rospy.init_node('shape_servo_control')
    rospy.logerr("======Loading object ... " + str(args.obj_name))  

    # Some important paramters
    init_dvrk_joints(gym, envs[i], dvrk_handles[i], joint_angles=[0,0,0,0,0.2,0,0,0,0.35,-0.35])  # Initilize robot's joints    

    all_done = False
    state = "home"
    
    
    # data_recording_path = args.data_recording_path #"/home/dvrk/shape_servo_data/generalization/multi_boxes_1000Pa/data"
    terminate_count = 0 
    sample_count = 0
    frame_count = 0
    group_count = 0
    reset_count = 0
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
    init_ball_states = []
    first_time = True

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
            gym.set_joint_target_position(envs[0], gym.get_joint_handle(envs[0], "dvrk", "psm_main_insertion_joint"), 0.103)            
            
            if frame_count == 10:
                rospy.loginfo("**Current state: " + state)
                frame_count = 0

                # # Save robot and object states for reset 
                # gym.refresh_particle_state_tensor(sim)
                # saved_object_state = deepcopy(gymtorch.wrap_tensor(gym.acquire_particle_state_tensor(sim))) 
                if first_time:
                    init_robot_state = deepcopy(gym.get_actor_rigid_body_states(envs[i], dvrk_handles[i], gymapi.STATE_ALL))  
                    for j in range(num_balls):
                        init_ball_states.append(deepcopy(gym.get_actor_rigid_body_states(env_objs[j], object_handles[j], gymapi.STATE_ALL)) )         
                

                # Go to next state
                # state = "get shape servo plan"
                state = "get shape servo plan"

                _,init_pose = get_pykdl_client(robot.get_arm_joint_positions())
                init_eulers = transformations.euler_from_matrix(init_pose)

                dof_props['driveMode'][:8].fill(gymapi.DOF_MODE_VEL)
                dof_props["stiffness"][:8].fill(0.0)
                dof_props["damping"][:8].fill(200.0)
                gym.set_actor_dof_properties(env, dvrk_handles[i], dof_props)    
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
            

            frame_count = 0
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

            frame_count += 1
            if frame_count % 2 == 0:
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
        
        reset_count += 1
        if reset_count % 500 == 0:
            for j in range(num_balls):
                gym.set_actor_rigid_body_states(env_objs[j], object_handles[j], init_ball_states[j], gymapi.STATE_ALL) 

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

