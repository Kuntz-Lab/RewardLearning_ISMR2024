import sys
import roslib.packages as rp
pkg_path = rp.get_pkg_dir('shape_servo_control')
sys.path.append(pkg_path + '/src')
sys.path.append("./pointcloud_representation_learning")

from util.isaac_utils import *
import torch
import os
import numpy as np
import pickle
import open3d

import timeit
import torch.nn.functional as F

print(3.e-4)
####################################################
# # Step 1: Convert the string to a float
# input_string = "0.1999"
# float_number = float(input_string)
# print(float_number)

# # Step 2: Format the float to have four decimal places
# formatted_float = "{:.4f}".format(float_number)

# # Alternatively, you can use f-string (Python 3.6+)
# # formatted_float = f"{float_number:.4f}"

# print(formatted_float)
###########################################
# a = torch.tensor([[1,2], [3, 4]])
# print(a>=2)
# print("where", torch.where(a>=2))
###########################################################
# remain_mask = np.array([[0,0], [0, 1], [1, 0], [1, 1]]) #(4, 2)
# reshaped_remain_mask = np.expand_dims(remain_mask, axis=2) #(4, 2, 1)
# reshaped_remain_mask = np.repeat(reshaped_remain_mask, 3, axis=2) # (num_envs, max_num_balls, 3)
# print("reshaped remain mask: ", reshaped_remain_mask)
# drop_xyz = np.array([[[100, 100, -100]]])
# drop_xyz = np.repeat(drop_xyz, 2, axis=1)
# drop_xyz = np.repeat(drop_xyz, 4, axis=0) # (num_envs, max_num_balls, 3)
# print("drop xyz: ", drop_xyz)
##########################################
# print("start test pad")
# t4d = torch.empty(3, 3, 4, 2)
# p1d = (1, 1) # pad last dim by 1 on each side
# out = F.pad(t4d, p1d, "constant", 0)  # effectively zero padding
# print(out.size())
# #torch.Size([3, 3, 4, 4])
# p2d = (1, 1, 2, 2) # pad last dim by (1, 1) and 2nd to last by (2, 2)
# out = F.pad(t4d, p2d, "constant", 0)
# print(out.size())
# #torch.Size([3, 3, 8, 4])
# t4d = torch.empty(3, 3, 4, 2)
# p3d = (0, 1, 2, 1, 3, 3) # pad by (0, 1), (2, 1), and (3, 3)
# out = F.pad(t4d, p3d, "constant", 0)
# print(out.size())
# #torch.Size([3, 9, 7, 3])
# print("end pad test")


# #print(1e8)
# xyz = np.array([[1,2,3], [4,5,6], [7,8,9], [10,11,12]])
# print(xyz[0:-1])
##########################################################################

def farthest_point_sample_batched(point, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    B, N, D = point.shape
    xyz = point[:, :,:3]
    centroids = np.zeros((B, npoint))
    distance = np.ones((B, N)) * 1e10
    farthest = np.random.uniform(low=0, high=N, size=(B,)).astype(np.int32)#np.random.randint(0, N)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[np.arange(0, B), farthest, :] # (B, D)
        centroid = np.expand_dims(centroid, axis=1) # (B, 1, D)
        dist = np.sum((xyz - centroid) ** 2, -1) # (B, N)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1) # (B,)
    point = point[np.arange(0, B).reshape(-1, 1), centroids.astype(np.int32), :]
    return point

# def farthest_point_sample(point, npoint):
#     """
#     Input:
#         xyz: pointcloud data, [N, D]
#         npoint: number of samples
#     Return:
#         centroids: sampled pointcloud index, [npoint, D]
#     """
#     N, D = point.shape
#     xyz = point[:,:3]
#     centroids = np.zeros((npoint,))
#     distance = np.ones((N,)) * 1e10
#     farthest = np.random.randint(0, N)
#     for i in range(npoint):
#         centroids[i] = farthest
#         centroid = xyz[farthest, :]
#         dist = np.sum((xyz - centroid) ** 2, -1)
#         mask = dist < distance
#         distance[mask] = dist[mask]
#         farthest = np.argmax(distance, -1)
#     point = point[centroids.astype(np.int32)]
#     return point

######################################################
# if vis_count%2==0:
#     ################################################
#     # video_main_path = "/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_test_push/image300"
#     # os.makedirs(video_main_path, exist_ok=True)
#     # video_path = os.path.join(video_main_path, f"img_{vis_count}.png")

#     # gym.render_all_camera_sensors(sim)
#     # im = gym.get_camera_image(sim, envs[0], cam_handles[0], gymapi.IMAGE_COLOR).reshape((300,300,4))
#     # im = Image.fromarray(im)                      
#     # im.save(video_path)
#     # print("++++++++++++++++++before")
#     # pc = get_partial_point_cloud(gym, sim, envs[0], cam_handles[0], cam_prop, color=[1,0,0], min_z = 0.05, visualization=False)
#     # plan_config["pcs"].append(pc)
    
#     #print("++++++++++++++++++after")
#     vis_count+=1
    ################################################
######################################################

# a = np.array([[[1,2,3], [4,5,6]], [[7,8,9], [10,11,12]]]) #(2, 2, 3)
# print(a[[[0],[1]], [[0, 1], [0,1]], :])

#######################################################

pc = np.random.uniform(low=-1, high=1, size=(500, 150, 3))
start_time = timeit.default_timer()
for i in range(500):
    pcd = np.array(down_sampling(pc[i, :, :], num_pts=256))
print("Elapsed time (s): ", timeit.default_timer() - start_time)

pc = np.random.uniform(low=-1, high=1, size=(500, 150, 3))
start_time = timeit.default_timer()
pc = np.array(farthest_point_sample_batched(pc, npoint=256))
print(pc.shape)
print("batch Elapsed time (s): ", timeit.default_timer() - start_time)

# ################################################
# emb = torch.tensor([[1,2,3], [4,5,6]])
# row = torch.tensor([[100,200,300]])
# emb[0] = row
# print(emb)

# ##################################################
# l = [torch.tensor([[1,2,3], [4,5,6]]), torch.tensor([[7,8,9], [10,11,12]])]
# print(torch.cat(l, dim=1))

# ###########################################################################
# empty = torch.tensor([[]])
# row = torch.tensor([[100,200,300]])
# print(torch.cat((empty, row), dim=1))

############################################################################

# with open(os.path.join("/home/dvrk/rl_pc/pc.pickle"), 'rb') as handle:
#     data = pickle.load(handle)  
# print("length: ", len(data)) 
# for pc in data:
#     pcd = pc
#     pcd2 = open3d.geometry.PointCloud()
#     pcd2.points = open3d.utility.Vector3dVector(np.array(pcd)) 
#     pcd2.paint_uniform_color([1, 0, 0])
#     open3d.visualization.draw_geometries([pcd2])  

##########################################################################################
##############################dropping balls by moving issues##########################################
###########################################################################################
#!/usr/bin/env python3
# from __future__ import print_function, division, absolute_import

# import numpy as np
# import os


# from isaacgym import gymutil, gymtorch, gymapi
# #from .base.vec_task import VecTask

# #import from Bao's motion planner
# import sys
# import roslib.packages as rp
# pkg_path = rp.get_pkg_dir('shape_servo_control')
# sys.path.append(pkg_path + '/src')
# sys.path.append("/home/dvrk/dvrk_ws/src/dvrk_env/shape_servo_control" + '/src')
# import os
# import numpy as np
# from isaacgym import gymapi
# from isaacgym import gymtorch
# from isaacgym import gymutil
# from copy import deepcopy
# import rospy
# import pickle
# import timeit
# import open3d
# import torch

# from geometry_msgs.msg import PoseStamped, Pose

# from util.isaac_utils import *
# #from util.grasp_utils import GraspClient

# from core import Robot
# from behaviors import MoveToPose, TaskVelocityControl
# import argparse
# from PIL import Image

# sys.path.append("/home/dvrk/catkin_ws/src/dvrk_env/shape_servo_control/src/teleoperation/traceSurface")
# sys.path.append("/home/dvrk/catkin_ws/src/dvrk_env/shape_servo_control/src/teleoperation")
# sys.path.append("/home/dvrk/catkin_ws/src/dvrk_env/shape_servo_control/src/teleoperation/traceSurface/pointcloud_representation_learning")
# from curve import *
# from compute_partial_pc import get_partial_pointcloud_vectorized
# from reward_lstm import RewardLSTM
# from architecture import AutoEncoder
# import random
# import time

# from rlgpu.utils.torch_jit_utils import *
# from rlgpu.tasks.base.base_task import BaseTask

# import copy



# #############Robot z offset###############
# ROBOT_Z_OFFSET = 0.25
# DEGREE = 1
# DIAMETER = 0.02
# MIN_NUM_BALLS = 2
# MAX_NUM_BALLS = 2

# def setup_cam_gpu(gym, env, cam_width, cam_height, cam_pos, cam_target):
#     cam_props = gymapi.CameraProperties()
#     cam_props.width = cam_width
#     cam_props.height = cam_height    
#     #cam_props.enable_tensors = True
#     cam_handle = gym.create_camera_sensor(env, cam_props)
#     gym.set_camera_location(cam_handle, env, cam_pos, cam_target)
#     return cam_handle, cam_props

# def sample_balls_poses(num_balls, degree):
#     balls_poses = []

#     weights_list, xy_curve_weights, balls_xyz = get_balls_xyz_curve(num_balls, degree=degree, offset=2*DIAMETER)

#     for xyz in balls_xyz:
#         ball_pose = gymapi.Transform()
#         pose = [xyz[0], xyz[1], xyz[2]]
#         ball_pose.p = gymapi.Vec3(*pose)
#         balls_poses.append(ball_pose)
        
#     return weights_list, xy_curve_weights, balls_xyz, balls_poses

# def default_poses():
#     poses = []
#     for i in range(MAX_NUM_BALLS):
#         ball_pose = gymapi.Transform()
#         pose = [100,100,-100]
#         ball_pose.p = gymapi.Vec3(*pose)
#         poses.append(ball_pose)
#     return poses

# def default_xyz():
#     poses = []
#     for i in range(MAX_NUM_BALLS):
#         pose = [100,100, -100]
#         poses.append(pose)
#     return poses

# def set_balls_poses():
#     rand_num_balls = 2 # random.randint(MIN_NUM_BALLS, MAX_NUM_BALLS) #1
#     weights_list, xy_curve_weights, balls_xyz, balls_poses = sample_balls_poses(num_balls=rand_num_balls, degree=DEGREE)
#     # Load ball objects with maximum amount
#     all_balls_poses = default_poses()
#     all_balls_xyz = default_xyz()
#     all_balls_poses[0:rand_num_balls] = balls_poses
#     all_balls_xyz[0:rand_num_balls] = balls_xyz
    
#     return weights_list, xy_curve_weights, balls_xyz, all_balls_poses, all_balls_xyz, rand_num_balls

# def to_obj_emb(model, device, pcd):
#     pcd_tensor = torch.from_numpy(pcd.transpose(1,0)).unsqueeze(0).float().to(device)
#     with torch.no_grad():
#         emb = model(pcd_tensor, get_global_embedding=True)
#     return emb

# def gt_reward_function(eef_pose, balls_xyz):
#     reward = 0
#     for i, ball_pose in enumerate(balls_xyz):
#         if ball_pose[0] == 100 and ball_pose[1] ==100 and ball_pose[2] ==-100:
#             continue
#         max_reward = 200
#         radius = 0.00025
#         for r in range(20):
#             if is_overlap(eef_pose, balls_xyz[i], max_dist=radius*(2**r)):
#                 reward = max(reward, max_reward*(0.5**r))
#     return reward


# ############################################################################################################################################

# class TraceCurve(BaseTask):

#     def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
#         np.random.seed(2021)
#         self.cfg = cfg
#         self.sim_params = sim_params
#         self.physics_engine = physics_engine
#         self.cfg["device_type"] = device_type
#         self.cfg["device_id"] = device_id
#         self.cfg["headless"] = headless

#         self.max_episode_length = self.cfg["env"]["episodeLength"]
#         self.max_push_effort = self.cfg["env"]["maxEffort"]

#         self.cfg["env"]["numObservations"] = 9
#         self.cfg["env"]["numActions"] = 10 #num_dofs per dvrk

#         #Values to be filled in at runtime
#         self.states = {}                        # will be dict filled with relevant states to use for reward calculation
#         self.handles = {}                       # will be dict mapping names to relevant sim handles
#         self.num_dofs = None                    # Total number of DOFs per env
#         self.actions = None                     # Current actions to be deployed
#         self._balls_states = None                
    
#         # Tensor placeholders
#         self._root_state = None             # State of root body        (n_envs, 13)
#         self._dof_state = None  # State of all joints       (n_envs, n_dof)
#         self._rigid_body_state = None  # State of all rigid bodies             (n_envs, n_bodies, 13)
#         self._eef_state = None  # end effector state (at grasping point)
#         self._global_indices = None         # Unique indices corresponding to all envs in flattened array

#         super().__init__(cfg=self.cfg)

#         self._refresh()

#         if not self.headless:
#             cam_pos = gymapi.Vec3(0.0, -0.440001, 2)
#             cam_target = gymapi.Vec3(0.0, -0.44, 0.1)
#             #middle_env = self.envs[self.num_envs // 2 + int(np.sqrt(self.num_envs)) // 2]
#             self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

#         # Camera for point cloud setup
#         self.cam_handles = []
#         cam_width = 300
#         cam_height = 300
       
#         cam_targets = gymapi.Vec3(0.0, -0.4, 0.05)
#         cam_positions = gymapi.Vec3(0.2, -0.7, 0.2)
        
#         for i, env in enumerate(self.envs):
#             cam_handle, self.cam_prop = setup_cam_gpu(self.gym, self.envs[i], cam_width, cam_height, cam_positions, cam_targets)
#             self.cam_handles.append(cam_handle)

#         self.encoder =  AutoEncoder(num_points=256, embedding_size=256).to(self.device)
#         self.encoder.load_state_dict(torch.load("/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_AE_balls_corrected/weights_straight3D_partial_flat_2ball/weights_1/epoch 150"))
        
#         self.which_ball = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)


#     def default_dvrk_asset(self):
#         '''
#         load the dvrk asset
#         '''
#         # dvrk asset
#         asset_options = gymapi.AssetOptions()
#         asset_options.armature = 0.001
#         asset_options.fix_base_link = True
#         asset_options.thickness = 0.00025#0.0001

#         asset_options.flip_visual_attachments = False
#         asset_options.collapse_fixed_joints = True
#         asset_options.disable_gravity = True
#         asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
#         asset_options.max_angular_velocity = 40000.

#         asset_root = self.cfg["env"]["asset"]["dvrkAssetRoot"]
#         dvrk_asset_file =self.cfg["env"]["asset"]["dvrkAssetFile"]
#         print("Loading asset '%s' from '%s'" % (dvrk_asset_file, asset_root))
#         return self.gym.load_asset(self.sim, asset_root, dvrk_asset_file, asset_options)


#     def create_sim(self):
#         # set the up axis to be z-up given that assets are y-up by default
#         #self.up_axis = self.cfg["sim"]["up_axis"]
#         self.sim_params.up_axis = gymapi.UP_AXIS_Z
#         self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
#         self._create_ground_plane()
#         self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        
#     def _create_ground_plane(self):
#         # add ground plane
#         plane_params = gymapi.PlaneParams()
#         plane_params.normal = gymapi.Vec3(0, 0, 1)# z-up ground
#         self.gym.add_ground(self.sim, plane_params)

    

#     def _create_envs(self, num_envs, spacing, num_per_row):
#         # define plane on which environments are initialized
#         lower = gymapi.Vec3(-spacing, 0.0, -spacing)
#         upper = gymapi.Vec3(spacing, spacing, spacing)

#         # load robot asset
#         dvrk_asset = self.default_dvrk_asset()
#         dvrk_pose = gymapi.Transform()
#         dvrk_pose.p = gymapi.Vec3(0.0, -0.1, ROBOT_Z_OFFSET)
#         dvrk_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)    
       
#         self.num_dvrk_bodies = self.gym.get_asset_rigid_body_count(dvrk_asset)
#         self.num_dvrk_dofs = self.gym.get_asset_dof_count(dvrk_asset)
#         print("num dvrk bodies: ", self.num_dvrk_bodies)
#         print("num dvrk dofs: ", self.num_dvrk_dofs)

#         #################### assume all on the same x-y plane ######################
#         weights_list, xy_curve_weights, self.balls_xyz, all_balls_poses, self.all_balls_xyz, self.rand_num_balls = set_balls_poses()

#         asset_root = self.cfg["env"]["asset"]["ballAssetRoot"]
#         ball_asset_file = self.cfg["env"]["asset"]["ballAssetFile"]
#         asset_options = gymapi.AssetOptions()
#         asset_options.fix_base_link = False
#         asset_options.disable_gravity = True
#         asset_options.thickness = 0.001

#         ball_asset = self.gym.load_asset(self.sim, asset_root, ball_asset_file, asset_options)   

        
#         # cache some common handles for later use
#         self.envs = []
#         self.dvrk_handles = []
#         self.ball_handles = []
#         self.ball_seg_Ids = [[] for i in range(self.num_envs)] # shape: (num_envs, MAX_NUM_BALLS)
#         self.dropped_ball_seg_Ids = [[] for i in range(self.num_envs)]
#         ball_name = 5
    
#         for i in range(self.num_envs):
            
#             # create env
#             env = self.gym.create_env(self.sim, lower, upper, num_per_row)
#             self.envs.append(env)

#             # add dvrk
#             dvrk_handle = self.gym.create_actor(env, dvrk_asset, dvrk_pose, "dvrk", i+1, 1, segmentationId=11)    
#             self.dvrk_handles.append(dvrk_handle)  

            
#             # add ball obj       
#             for j in range(MAX_NUM_BALLS):     
#                 seg_Id = ball_name*10
#                 ball_actor = self.gym.create_actor(env, ball_asset, all_balls_poses[j], f"{ball_name}", ball_name, segmentation_Id=seg_Id)
#                 self.ball_seg_Ids[i].append(seg_Id)
#                 color = gymapi.Vec3(*list(np.random.uniform(low=[0,0,0], high=[1,1,1], size=3)))
#                 self.gym.set_rigid_body_color(env, ball_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
#                 self.ball_handles.append(ball_actor)
#                 ball_name += 1  

#             #init_dvrk_joints(self.gym, self.envs[i], self.dvrk_handles[i])

        

#         # DOF Properties and Drive Modes 
#         dof_props = self.gym.get_actor_dof_properties(self.envs[0], self.dvrk_handles[0])
#         dof_props["driveMode"].fill(gymapi.DOF_MODE_VEL) #original: gymapi.DOF_MODE_EFFORT
#         dof_props["stiffness"].fill(200.0)
#         dof_props["damping"].fill(40.0)
#         dof_props["stiffness"][8:].fill(1)
#         dof_props["damping"][8:].fill(2)  
        
#         for env in self.envs:
#             self.gym.set_actor_dof_properties(env, self.dvrk_handles[i], dof_props)    # set dof properties 

#         # Setup data
#         self.init_data()

#     def init_data(self):
#         # Setup sim handles
#         self.handles = {
#             "eef": self.gym.find_actor_rigid_body_handle(self.envs[0], self.dvrk_handles[0], "psm_tool_yaw_link")
#         }
       
#         # Get total DOFs
#         self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

#         # Setup tensor buffers
#         _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
#         _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
#         _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
#         self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(self.num_envs, -1, 13)
#         self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(self.num_envs, -1, 2)
#         self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(self.num_envs, -1, 13)



#         self._eef_state = self._rigid_body_state[:, self.handles["eef"], :]
        
#         self._balls_states = self._root_state[:, 1:1+MAX_NUM_BALLS, :]
      

#         # Initialize indices
#         self._global_indices = torch.arange(self.num_envs * (1+MAX_NUM_BALLS), dtype=torch.int32,
#                                            device=self.device).view(self.num_envs, -1)

#         #self._init_root_state = torch.clone(self._root_state)

#         #multi_env_ids_int32 = self._global_indices[:, 0].flatten()
        
        


#     def _update_states(self):
#         self.states.update({
#             # dvrk
#             "eef_pos": self._eef_state[:, :3],
#             "eef_quat": self._eef_state[:, 3:7],
#             "eef_vel": self._eef_state[:, 7:],
#             "balls_pos": self._balls_states[:, :, :3],
#             "ball0_pos": self._balls_states[:, 0, :3],
#             "ball1_pos": self._balls_states[:, 1, :3],
#             "eef_ball0_diff": self._balls_states[:, 0, :3] - self._eef_state[:, :3],
#             "eef_ball1_diff": self._balls_states[:, 1, :3] - self._eef_state[:, :3]
#         })

#         # print(self.states["balls_pos"])

#     def _refresh(self):
#         self.gym.refresh_actor_root_state_tensor(self.sim)
#         self.gym.refresh_dof_state_tensor(self.sim)
#         self.gym.refresh_rigid_body_state_tensor(self.sim)
#         self.gym.refresh_jacobian_tensors(self.sim)
#         self.gym.refresh_mass_matrix_tensors(self.sim)

#         # Refresh states
#         self._update_states()


#     def compute_reward(self, actions):
#         self.rew_buf[:], self.reset_buf[:] = compute_dvrk_reward(
#             self.reset_buf, self.progress_buf, self.actions, self.states, self.max_episode_length, self.obs_buf, self.which_ball)
        

#     def compute_observations(self, env_ids=None):
#         self._refresh()

#         if env_ids is None:
#             env_ids = np.arange(self.num_envs)

#         obj_embs = torch.ones([self.num_envs, self.encoder.embedding_size], dtype=torch.float64, device=self.device)
#         self.gym.render_all_camera_sensors(self.sim)
#         for i in env_ids:
#             pc = get_partial_pointcloud_vectorized(self.gym, self.sim, self.envs[i], self.cam_handles[i], self.cam_prop, color=None, min_z = 0.05, visualization=True, device=self.device)
#             # emb = to_obj_emb(self.encoder, self.device, pc)
#             # obj_embs[i] = emb
        
#         #print("obj_emb", obj_embs.shape)
#         #self.obs_buf = torch.cat((torch.clone(self.states["eef_pos"]), obj_embs), dim=-1).float()
#         self.obs_buf = torch.cat((self.states["eef_pos"], self.states["ball0_pos"], self.states["ball1_pos"]), dim=-1).float()
#         return self.obs_buf

#     def reset_idx(self, env_ids):
#         print("reseting ...")
#         env_ids = env_ids.cpu().numpy()
#         #env_ids_int32 = env_ids.to(dtype=torch.int32)
#         num_reset = len(env_ids)

#         self.which_ball = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
#         self.dropped_ball_seg_Ids = copy.deepcopy(self.ball_seg_Ids)

#         # print("##############", self._balls_states[env_ids, :, :3].shape, torch.tensor(self.all_balls_xyz, device=self.device, dtype=float).float().shape)
#         #weights_list, xy_curve_weights, self.balls_xyz, all_balls_poses, self.all_balls_xyz, self.rand_num_balls = set_balls_poses()
    
#         root_state = torch.clone(self._root_state).cpu().numpy()
#         balls_states = root_state[:, 1:1+MAX_NUM_BALLS, :]

#         # Write these new init balls pos to the sim states
#         new_pos = torch.tensor(self.all_balls_xyz, device=self.device, dtype=torch.float).unsqueeze(0).expand(num_reset, -1, -1).cpu().numpy()
#         balls_states[env_ids, :, :3]=new_pos
#         # no rotation
#         balls_states[env_ids, :, 3:7]=torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).repeat(num_reset, MAX_NUM_BALLS, 1).cpu().numpy()
#         # 0 velocity
#         balls_states[env_ids, :, 7:10]=torch.tensor([0.0, 0.0, 0.0], device=self.device).repeat(num_reset, MAX_NUM_BALLS, 1).cpu().numpy()

#         # reset balls states
#         multi_env_ids_cubes_int32 = self._global_indices[env_ids, 1:].flatten()
#         self.gym.set_actor_root_state_tensor_indexed(
#             self.sim, gymtorch.unwrap_tensor(torch.tensor(root_state).to(self.device)),
#             gymtorch.unwrap_tensor(multi_env_ids_cubes_int32), len(multi_env_ids_cubes_int32))

        
        
#         # reset dvrk
#         multi_env_ids_int32 = self._global_indices[env_ids, 0].flatten()
        
#         # self.gym.set_dof_state_tensor_indexed(self.sim,
#         #                                       gymtorch.unwrap_tensor(self._init_dof_state),
#         #                                       gymtorch.unwrap_tensor(multi_env_ids_int32),
#         #                                       len(multi_env_ids_int32))

#         dof_state = torch.clone(self._dof_state).cpu().numpy()
#         dof_state[env_ids, :, 0] = np.array([-0.3193448984150404, 0.08059437628908588, 0.04633803400231634, -0.11489753865971103, 0.23808314280302934, \
#         -0.0005194664992932008, 0.010978008812426507, 0.3189222925076953, 0.35, -0.35])
#         self.gym.set_dof_state_tensor_indexed(self.sim,
#                                               gymtorch.unwrap_tensor(torch.tensor(dof_state).to(self.device)),
#                                               gymtorch.unwrap_tensor(multi_env_ids_int32),
#                                               len(multi_env_ids_int32))

        
#         self.progress_buf[env_ids] = 0
#         self.reset_buf[env_ids] = 0

    

#     def drop_ball(self):
#         #drop balls when overlap with eef
#         eef_ball_diff = torch.sum((self._balls_states[:, :, :3] - self._eef_state[:, :3].unsqueeze(1))**2, dim=-1) # (num_envs, max_num_balls)
#         overlap = eef_ball_diff.unsqueeze(-1).expand(-1,-1,3).cpu().numpy() <= 0.00025

#         # set_pos = torch.tensor([1,0,0.1], device=self.device).repeat(self._balls_states.shape[0], self._balls_states.shape[1], 1).cpu().numpy()

#         # root_state = torch.clone(self._root_state).cpu().numpy()
#         # balls_states = root_state[:, 1:1+MAX_NUM_BALLS, :]
        
#         # balls_states[:, :, :3] = np.where(eef_ball_diff.unsqueeze(-1).expand(-1,-1,3).cpu().numpy() <= 0.00025, set_pos, balls_states[:, :, :3])

#         # multi_env_ids_cubes_int32 = self._global_indices[np.arange(self.num_envs), 1:].flatten()
#         # self.gym.set_actor_root_state_tensor_indexed(
#         #     self.sim, gymtorch.unwrap_tensor(torch.tensor(root_state).to(self.device)),
#         #     gymtorch.unwrap_tensor(multi_env_ids_cubes_int32), len(multi_env_ids_cubes_int32))


#     def update_which_ball(self):
#         balls_pos = self.states["balls_pos"]
#         eef_pos = self.states["eef_pos"]
#         eef_ball_diff = torch.sum((balls_pos[:, :, :3] - eef_pos[:, :3].unsqueeze(1))**2, dim=-1) # (num_envs, max_num_balls)

#         for i in range(self.rand_num_balls):
#             self.which_ball = torch.where(eef_ball_diff[:,i] <= 0.00025, self.which_ball+1, self.which_ball)

#         self.which_ball = torch.clamp(self.which_ball, 0, self.rand_num_balls-1)


#     def pre_physics_step(self, actions):
#         self.actions = actions.clone().to(self.device)
#         #print(self.num_envs * self.num_dof,"++++++++++++++++++++", actions.shape)
#         actions_tensor = torch.zeros(self.num_envs * self.num_dofs, device=self.device, dtype=torch.float)
#         #mask out the gripper
#         mask = torch.ones_like(actions)
#         mask[:,9]=0
#         mask[:,8]=0
#         actions_tensor += (actions*mask).reshape(self.num_envs*self.num_dofs)*self.max_push_effort
#         forces = gymtorch.unwrap_tensor(actions_tensor)
#         #self.gym.set_dof_actuation_force_tensor(self.sim, forces)
#         self.gym.set_dof_velocity_target_tensor(self.sim, forces)
        

#     def post_physics_step(self):
#         print(f"ball states: {self._balls_states[:, :, :3].cpu().numpy()}")
#         print(f"eef state: {self._eef_state[:, :3].cpu().numpy()}")
#         print(f"eef ball diff: {torch.sum((self._balls_states[:, :, :3] - self._eef_state[:, :3].unsqueeze(1))**2, dim=-1).cpu().numpy()}")
#         self.progress_buf += 1

#         self.drop_ball()
#         self.update_which_ball()
        
#         env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        
#         if len(env_ids) > 0:
#             self.reset_idx(env_ids)

        
#         self.compute_observations()
#         self.compute_reward(self.actions)

        
        
        


# #####################################################################
# ###=========================jit functions=========================###
# #####################################################################


# @torch.jit.script
# def compute_dvrk_reward(
#     reset_buf, progress_buf, actions, states, max_episode_length, obs_buf, which_ball
# ):
#     # type: (Tensor, Tensor, Tensor, Dict[str, Tensor], float, Tensor, Tensor) -> Tuple[Tensor, Tensor]

#     ############# ground truth ###################
#     balls_pos = states["balls_pos"]
#     eef_pos = states["eef_pos"]
#     eef_ball_diff = torch.sum((balls_pos[:, :, :3] - eef_pos[:, :3].unsqueeze(1))**2, dim=-1) # (num_envs, max_num_balls)
#     with torch.no_grad():
#         rewards_tensor = torch.where(which_ball==0, -eef_ball_diff[:,0], -eef_ball_diff[:,1])

#     print("reward:", rewards_tensor[0])
#     print("which_ball:", which_ball[0])
    
#     # Compute resets
#     reset_buf = torch.where((progress_buf >= max_episode_length - 1) , torch.ones_like(reset_buf), reset_buf)

#     return rewards_tensor.detach(), reset_buf