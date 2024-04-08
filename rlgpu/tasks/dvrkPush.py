#!/usr/bin/env python3
# from __future__ import print_function, division, absolute_import

import numpy as np
import os


from isaacgym import gymutil, gymtorch, gymapi
#from .base.vec_task import VecTask

#import from Bao's motion planner
import sys
import roslib.packages as rp
pkg_path = rp.get_pkg_dir('shape_servo_control')
sys.path.append(pkg_path + '/src')
sys.path.append("/home/dvrk/catkin_ws/src/dvrk_env/shape_servo_control" + '/src')
import os
import numpy as np
from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym import gymutil
from copy import deepcopy
import rospy
import pickle
import timeit
import open3d
import torch

from geometry_msgs.msg import PoseStamped, Pose

from util.isaac_utils import *
#from util.grasp_utils import GraspClient

import argparse
from PIL import Image

from behaviors import TaskVelocityControlMultiRobot
from core import MultiRobot

from util.isaac_utils import *
sys.path.append('/home/dvrk/catkin_ws/src/dvrk_env/shape_servo_control/src/teleoperation')
sys.path.append('/home/dvrk/catkin_ws/src/dvrk_env/shape_servo_control/src/teleoperation/pushTask/pointcloud_representation_learning')
sys.path.append("/home/dvrk/catkin_ws/src/dvrk_env/shape_servo_control/src/teleoperation/pc_utils")
from get_isaac_partial_pc import get_partial_pointcloud_vectorized_push_box#, get_partial_pointcloud_vectorized
from compute_partial_pc import farthest_point_sample_batched
from pushTask.reward_learning.reward import RewardNetPointCloud
from pushTask.pointcloud_representation_learning.architecture import AutoEncoder

from rlgpu.utils.torch_jit_utils import *
from rlgpu.tasks.base.base_task import BaseTask

#############Robot z offset###############
ROBOT_Z_OFFSET = 0.25
CONE_Z_OFFSET = 0.01 #0.005
OVERLAP_TOLERANCE = 0.001 #0.0008
THICKNESS = 0.001 #0.000025
NUM_ADDITIONAL_ACTOR = 0


def to_obj_emb_not_batch(model, device, pcd):
    pcd_tensor = torch.from_numpy(pcd.transpose(1,0)).unsqueeze(0).float().to(device)
    emb = model(pcd_tensor, get_global_embedding=True)
    return emb

def to_obj_emb(model, device, pcds):
    '''
    pcds has shape (num_batch, num_points, point_dim)
    '''
    pcd_tensor = torch.from_numpy(pcds.transpose(0,2,1)).float().to(device)
    with torch.no_grad():
        emb = model(pcd_tensor, get_global_embedding=True)
    return emb

def is_overlap_xy(p1, p2, max_dist=OVERLAP_TOLERANCE):
    return (p1.p.x-p2.p.x)**2 + (p1.p.y-p2.p.y)**2 <=max_dist

def init_dvrk_joints(gym, env, dvrk_handle, joint_angles=None):
    dvrk_dof_states = gym.get_actor_dof_states(env, dvrk_handle, gymapi.STATE_NONE)
    if joint_angles is None:
        dvrk_dof_states['pos'][8] = 1.5
        dvrk_dof_states['pos'][9] = 0.8
    else:
        dvrk_dof_states['pos'] = joint_angles

def setup_cam_gpu(gym, env, cam_width, cam_height, cam_pos, cam_target):
    cam_props = gymapi.CameraProperties()
    cam_props.width = cam_width
    cam_props.height = cam_height    
    #cam_props.enable_tensors = True
    cam_handle = gym.create_camera_sensor(env, cam_props)
    gym.set_camera_location(cam_handle, env, cam_pos, cam_target)
    return cam_handle, cam_props

############################################################################################################################################

class DvrkPush(BaseTask):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        np.random.seed(2021)
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless

        self.train_with_learned_reward = self.cfg["reward"]["train_with_learned_reward"]
        self.use_dof_vel_action = self.cfg["control"]["dof_vel"]
        self.use_eef_vel_action = self.cfg["control"]["eef_vel"]
        self.use_eef_pos_action = self.cfg["control"]["eef_pos"]
        assert(int(self.use_dof_vel_action) + int(self.use_eef_vel_action) + int(self.use_eef_pos_action) == 1)
        
        ###################### determine state and action space (only use x-y eef pos/vel, the z-value is fixed #######################
        if self.use_eef_vel_action:
            if self.train_with_learned_reward:
                self.cfg["env"]["numObservations"] = 256 + 2 
            else:
                self.cfg["env"]["numObservations"] = (2+2) + 2
            self.cfg["env"]["numActions"] = 2
            self.cfg["env"]["episodeLength"] = 30
        elif self.use_eef_pos_action:
            if self.train_with_learned_reward:
                self.cfg["env"]["numObservations"] = 256 + 2
            else:
                self.cfg["env"]["numObservations"] = (2+2) + 2
            self.cfg["env"]["numActions"] = 2
            self.cfg["env"]["episodeLength"] = 10 #30 #10
        elif self.use_dof_vel_action:
            if self.train_with_learned_reward:
                self.cfg["env"]["numObservations"] = 256 + 10 + 2
            else:
                self.cfg["env"]["numObservations"] = 6 + 10 + 2
        
            self.cfg["env"]["numActions"] = 10 #num_dofs per dvrk
            self.cfg["env"]["episodeLength"] = 30 #10

        print("!!!!!!!! obs dim: ", self.cfg["env"]["numObservations"]) 
        print("!!!!!!!! action dim: ", self.cfg["env"]["numActions"]) 
        ###############################################################################

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.max_push_effort = self.cfg["env"]["maxEffort"]
        self.use_abs_eef_action = self.cfg["control"]["abs_eef_pos_action"]

        #Values to be filled in at runtime
        self.states = {}                        # will be dict filled with relevant states to use for reward calculation
        self.handles = {}                       # will be dict mapping names to relevant sim handles
        self.num_dofs = None                    # Total number of DOFs per env
        self.actions = None                     # Current actions to be deployed
        self._box_state = None                # Current state of cubeA for the current env
        self._cone_state = None                # Current state of cubeB for the current env
    
        # Tensor placeholders
        self._root_state = None             # State of root body        (n_envs, 13)
        self._dof_state = None  # State of all joints       (n_envs, n_dof)
        self._rigid_body_state = None  # State of all rigid bodies             (n_envs, n_bodies, 13)
        self._eef_state = None  # end effector state (at grasping point)
        self._global_indices = None         # Unique indices corresponding to all envs in flattened array

        super().__init__(cfg=self.cfg)

        self._refresh()

        ################ camera for viewer ############
        if not self.headless:
            cam_pos = gymapi.Vec3(0.0, -0.440001, 2)
            cam_target = gymapi.Vec3(0.0, -0.44, 0.1)
            #middle_env = self.envs[self.num_envs // 2 + int(np.sqrt(self.num_envs)) // 2]
            self.gym.viewer_camera_look_at(self.viewer, self.envs[0], cam_pos, cam_target)

        ############ Camera for point cloud setup #########
        self.cam_handles = []
        cam_width = 300#600#300
        cam_height = 300#600#300
       
        # cam_targets = gymapi.Vec3(0.0, -0.44, 0.1)
        # cam_positions = gymapi.Vec3(0.7, -0.54, 0.76)
        cam_targets = gymapi.Vec3(0.0, -0.4, 0.02)
        cam_positions = gymapi.Vec3(0.2, -0.7, 0.5)
            
        
        for i, env in enumerate(self.envs):
            cam_handle, self.cam_prop = setup_cam_gpu(self.gym, self.envs[i], cam_width, cam_height, cam_positions, cam_targets)
            self.cam_handles.append(cam_handle)

        ################# Camera for recording rgb image
        self.img_cam_handles = []
        cam_width = 300
        cam_height = 300
       
        cam_targets = gymapi.Vec3(0.0, -0.4, 0.02)
        cam_positions = gymapi.Vec3(0.2, -0.7, 0.3)
        
        for i, env in enumerate(self.envs):
            cam_handle, self.img_cam_prop = setup_cam_gpu(self.gym, self.envs[i], cam_width, cam_height, cam_positions, cam_targets)
            self.img_cam_handles.append(cam_handle)

        if self.train_with_learned_reward:
            reward_model_path = os.path.join(self.cfg["reward"]["rewardModelPath"], self.cfg["reward"]["rewardModelVersion"])
            self.reward_net = RewardNetPointCloud()
            self.reward_net.to(self.device)
            self.reward_net.load_state_dict(torch.load(reward_model_path))

            self.encoder = AutoEncoder(num_points=256, embedding_size=256).to(self.device)
            encoder_model_path = self.cfg["AE"]["AEModelPath"]
            self.encoder.load_state_dict(torch.load(encoder_model_path))
            self.encoder.eval()

            self.obj_embs = torch.ones([self.num_envs, self.encoder.embedding_size], dtype=torch.float64, device=self.device)

        self.reset_pc=True

        # for saving data of each episode for visualization
        self.target_is_reached = torch.zeros((self.num_envs,)).to(self.device)
        self.rl_data_dict = {"gt_cum_reward": np.zeros((self.num_envs,)), "learned_cum_reward": np.zeros((self.num_envs,)), "images": [], "num_target_reached":None, "target_reached_per_step": [], "eef_box_dist": []}
        self.rl_data_idx = 0
        self.save_idx = 10
        self.rl_data_path = "/home/dvrk/RL_data/dvrkPush/gt_reward_eef_pos/data"
        os.makedirs(self.rl_data_path, exist_ok=True)

        ####################################
        dummy_desired_positions = np.random.uniform(low=[-0.1, 0, 0.03], high=[0.1, 0.07, 0.05], size=(self.num_envs, 3))
        dt = 1./60.
        self.robots = MultiRobot(self.gym, self.sim, self.envs, self.dvrk_handles, 8, self.num_envs)
        self.tvc_behavior = TaskVelocityControlMultiRobot(list(dummy_desired_positions), self.robots, self.num_envs, dt, 3, vel_limits=self.vel_limits, error_threshold = 5e-3) #5e-3            



    def default_dvrk_asset(self):
        '''
        load the dvrk asset
        '''
        # dvrk asset
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.001
        asset_options.fix_base_link = True
        asset_options.thickness = THICKNESS#0.0001

        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.max_angular_velocity = 40000000000.

        asset_root = self.cfg["env"]["asset"]["dvrkAssetRoot"]
        dvrk_asset_file =self.cfg["env"]["asset"]["dvrkAssetFile"]
        print("Loading asset '%s' from '%s'" % (dvrk_asset_file, asset_root))
        return self.gym.load_asset(self.sim, asset_root, dvrk_asset_file, asset_options)

    def create_sim(self):
        # set the up axis to be z-up given that assets are y-up by default
        #self.up_axis = self.cfg["sim"]["up_axis"]
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        
    def _create_ground_plane(self):
        # add ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)# z-up ground
        # plane_params.static_friction = 10
        # plane_params.dynamic_friction = 10

        self.gym.add_ground(self.sim, plane_params)

    

    def _create_envs(self, num_envs, spacing, num_per_row):
        # define plane on which environments are initialized
        lower = gymapi.Vec3(-spacing, 0.0, -spacing)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        # load robot asset
        dvrk_asset = self.default_dvrk_asset()
        dvrk_pose = gymapi.Transform()
        dvrk_pose.p = gymapi.Vec3(0.0, -0.1, ROBOT_Z_OFFSET)
        dvrk_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)    
       
        self.num_dvrk_bodies = self.gym.get_asset_rigid_body_count(dvrk_asset)
        self.num_dvrk_dofs = self.gym.get_asset_dof_count(dvrk_asset)
        print("num dvrk bodies: ", self.num_dvrk_bodies)
        print("num dvrk dofs: ", self.num_dvrk_dofs)

        # Load box object
        #init_pose = [0.0, -0.5, 0.05] 
        #init_pose = [0.0, -0.5, 0.015 + THICKNESS]
        init_pose = [0.0, -0.5, 0.03 + THICKNESS]
        self.init_pose = init_pose

        self.box_xyzs = np.array([[-1.0 for j in range(3)] for i in range(self.num_envs)])
        self.cone_xyzs = np.array([[-1.0 for j in range(3)] for i in range(self.num_envs)])
        self.box_poses = [None for i in range(self.num_envs)]
        self.cone_poses = [None for i in range(self.num_envs)]
        for i in range(self.num_envs):
            self.box_pose = gymapi.Transform()
            # pose = list(np.array(init_pose) + np.array([0.02, 0, 0.015]))
            # pose = list(np.array(init_pose) + np.random.uniform(low=[-0.1,-0.05,0], high=[0.1,0.05,0], size=3) + np.array([0, 0, 0.015]))
            pose = np.array(init_pose) + np.random.uniform(low=[-0.1,-0.05,0], high=[0.1,0.05,0], size=3)
            self.box_pose.p = gymapi.Vec3(*list(pose))
            self.box_xyzs[i] = pose
            self.box_poses[i] = self.box_pose

            self.cone_pose = gymapi.Transform()
            pose = np.array(init_pose) + np.random.uniform(low=[-0.1,-0.05,0], high=[0.1,0.05,0], size=3)
            # pose = list(np.array(init_pose) + np.array([-0.1, 0, 0]))
            pose[2] += CONE_Z_OFFSET
            self.cone_pose.p = gymapi.Vec3(*list(pose))
            self.cone_xyzs[i] = pose
            self.cone_poses[i] = self.cone_pose

            while is_overlap_xy(self.cone_pose, self.box_pose):
                pose = np.array(init_pose) + np.random.uniform(low=[-0.1,-0.05,0], high=[0.1,0.05,0], size=3)
                pose[2] += CONE_Z_OFFSET
                self.cone_pose.p = gymapi.Vec3(*list(pose))
                self.cone_xyzs[i] = pose
                self.cone_poses[i] = self.cone_pose

        print("box_xyzs", self.box_xyzs.shape)


        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.disable_gravity = False
        asset_options.thickness = THICKNESS
        asset_options.density = 1000 #1000
        # asset_options.max_angular_velocity = 0.0001
        # asset_options.max_linear_velocity = 0.0001
        asset_options.linear_damping = 100000
        asset_options.angular_damping = 100000
        # asset_options.override_inertia = True
        
        asset_options.armature = 20 #10
        asset_options.use_physx_armature = True

        #box_asset = self.gym.create_box(self.sim, *[0.06, 0.06, 0.06], asset_options)#self.gym.load_asset(self.sim, asset_root, box_asset_file, asset_options) 
        box_asset = self.gym.create_sphere(self.sim, 0.03, asset_options)
        cone_asset = self.gym.create_box(self.sim, *[0.04, 0.04, 0.08], asset_options)#self.gym.create_sphere(self.sim, 0.02, asset_options)#self.gym.create_box(self.sim, *[0.04, 0.04, 0.08], asset_options) #self.gym.load_asset(self.sim, asset_root, cone_asset_file, asset_options) 


        
        
        # table for supporting the box
        # table_pos = [0.0, 0.0, THICKNESS+0.025/2]
        # table_thickness = 0.025
        # table_opts = gymapi.AssetOptions()
        # table_opts.fix_base_link = True
        # table_opts.thickness = THICKNESS
        # table_asset = self.gym.create_box(self.sim, *[2.0, 2.0, table_thickness], table_opts)
        # table_pose = gymapi.Transform()
        # table_pose.p = gymapi.Vec3(*table_pos)

        # cache some common handles for later use
        self.envs = []
        self.dvrk_handles = []
        self.box_handles = []
        self.cone_handles = []
        self.table_handles  = []
    
        for i in range(self.num_envs):
            
            # create env
            env = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.envs.append(env)

            # add dvrk
            dvrk_handle = self.gym.create_actor(env, dvrk_asset, dvrk_pose, "dvrk", i, 1, segmentationId=11)    
            self.dvrk_handles.append(dvrk_handle)    

            # add obj (start and goal)   
            box_id = self.gym.create_actor(env, box_asset, self.box_poses[i], "box", i, 0)
            color = gymapi.Vec3(0,1,0)
            self.gym.set_rigid_body_color(env, box_id, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
            self.box_handles.append(box_id)

            cone_id = self.gym.create_actor(env, cone_asset, self.cone_poses[i], "cone", i+2, 0)
            color = gymapi.Vec3(1,0,0)
            self.gym.set_rigid_body_color(env,  cone_id, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
            self.cone_handles.append(cone_id)

            # table_actor = self.gym.create_actor(env, table_asset, table_pose, "table", i, 1, segmentationId=11)
            # color = gymapi.Vec3(0,0,1)
            # self.gym.set_rigid_body_color(env,  table_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
            # self.table_handles.append(table_actor)


            init_dvrk_joints(self.gym, self.envs[i], self.dvrk_handles[i])

        # DOF Properties and Drive Modes 
        dof_props = self.gym.get_actor_dof_properties(self.envs[0], self.dvrk_handles[0])
        dof_props["driveMode"].fill(gymapi.DOF_MODE_VEL) #original: gymapi.DOF_MODE_EFFORT
        # dof_props["stiffness"].fill(200.0)
        # dof_props["damping"].fill(40.0)
        # dof_props["stiffness"][8:].fill(1)
        # dof_props["damping"][8:].fill(2)  
        dof_props["stiffness"].fill(400000)
        dof_props["damping"].fill(100000)
        dof_props["stiffness"][8:].fill(10000)
        dof_props["damping"][8:].fill(20000)  
        
        self.vel_limits = dof_props['velocity']
        
        for i, env in enumerate(self.envs):
            self.gym.set_actor_dof_properties(env, self.dvrk_handles[i], dof_props) 

        # Setup data
        self.init_data()

    def init_data(self):
        # Setup sim handles
        self.handles = {
            "eef": self.gym.find_actor_rigid_body_handle(self.envs[0], self.dvrk_handles[0], "psm_tool_yaw_link"),  #??? why 0
            "box_env_idx": self.gym.find_actor_rigid_body_handle(self.envs[0], self.box_handles[0], "box"),
            "cone_env_idx": self.gym.find_actor_rigid_body_handle(self.envs[0], self.cone_handles[0], "cone")
        }

        # Get total DOFs
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        # Setup tensor buffers
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(self.num_envs, -1, 13)
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(self.num_envs, -1, 2)
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(self.num_envs, -1, 13)


        self._eef_state = self._rigid_body_state[:, self.handles["eef"], :]
        self._init_dof_state = torch.clone(gymtorch.wrap_tensor(_dof_state_tensor))
        self._init_root_state = torch.clone(gymtorch.wrap_tensor(_actor_root_state_tensor))
        
        self._box_state = self._root_state[:, 1, :]
        self._cone_state = self._root_state[:, 2, :]
      

        # Initialize indices
        self._global_indices = torch.arange(self.num_envs * (3+NUM_ADDITIONAL_ACTOR), dtype=torch.int32,
                                           device=self.device).view(self.num_envs, -1)

    def _update_states(self):
        self.states.update({
            # dvrk
            "eef_pos": self._eef_state[:, :3],
            "eef_quat": self._eef_state[:, 3:7],
            "eef_vel": self._eef_state[:, 7:],
            "dof_pos": self._dof_state[:, :, 0],
            # Cubes and cones
            "box_quat": self._box_state[:, 3:7],
            "box_pos": self._box_state[:, :3],
            "box_lin_vel": self._box_state[:, 7:10],
            "box_ang_vel": self._box_state[:, 10:13],
            "box_pos_relative": self._box_state[:, :3] - self._eef_state[:, :3],
            "cone_quat": self._cone_state[:, 3:7],
            "cone_pos": self._cone_state[:, :3],
            "box_to_cone_pos": self._cone_state[:, :3] - self._box_state[:, :3],
            "eef_to_box_pos": self._eef_state[:, :3] - self._box_state[:, :3]
        })

    def _refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

        # Refresh states
        self._update_states()
        #print(self.states["box_quat"])

    def gt_reward_function(self, use_explore_reward=False):
        eef_box_d = torch.norm(self.states["eef_to_box_pos"], dim=-1).to("cuda")
        explore_reward = -eef_box_d
        
        box_cone_d = torch.norm(self.states["box_to_cone_pos"], dim=-1).to("cuda")
        exploit_reward = -box_cone_d

        close_to_box = eef_box_d.clone().detach() < 0.03
        print("***eef close to box?: ", close_to_box[0])
        self.rl_data_dict["eef_box_dist"].append(eef_box_d.cpu().detach().numpy())

        if use_explore_reward:
            total_reward = torch.where(close_to_box, exploit_reward, explore_reward).to(self.device)
        else:
            total_reward = exploit_reward
        return total_reward


    def compute_reward(self, actions):
        
        gt_rewards = self.gt_reward_function(use_explore_reward=False)
        print("gt reward: ", np.around(gt_rewards[0].cpu().detach().numpy(), decimals=4))
        if not self.train_with_learned_reward:
            #reset_buf = torch.where(torch.logical_or((progress_buf >= max_episode_length - 1),(box_cone_d <= 0.05)) , torch.ones_like(reset_buf), reset_buf)
            reset_buf = torch.where((self.progress_buf >= self.max_episode_length - 1) , torch.ones_like(self.reset_buf), self.reset_buf)
            self.rew_buf[:] = torch.clone(gt_rewards).to(self.device)
            self.reset_buf[:] = torch.clone(reset_buf).to(self.device)
        else:
            if self.use_dof_vel_action:
                emb_state = self.obs_buf[:, 12:].unsqueeze(1) # (num_envs, 1, 256)
            else:
                emb_state = self.obs_buf[:, 2:].unsqueeze(1) # (num_envs, 1, 256)
            with torch.no_grad():
                rewards, _ = self.reward_net.cum_return(emb_state) # (num_envs, 1)
            rewards = rewards.reshape(-1)
            print(f"learned reward: {rewards[0].item()}")

            reset_buf = torch.where((self.progress_buf >= self.max_episode_length - 1) , torch.ones_like(self.reset_buf), self.reset_buf)
            self.rew_buf[:] = rewards
            self.reset_buf[:] = torch.clone(reset_buf).to(self.device)

            self.rl_data_dict["learned_cum_reward"] += rewards.cpu().detach().numpy()

        self.rl_data_dict["gt_cum_reward"] += gt_rewards.cpu().detach().numpy()

        

    def compute_observations(self, env_ids=None):
        self._refresh()

        if env_ids is None:
            env_ids = np.arange(self.num_envs)

        print(f"box pose: ", self.states["box_pos"][0])
        print(f"cone pose: ", self.states["cone_pos"][0])

        self.gym.render_all_camera_sensors(self.sim)

        # save image to data_dict
        if self.rl_data_idx % self.save_idx == 0:
            im = self.gym.get_camera_image(self.sim, self.envs[0], self.img_cam_handles[0], gymapi.IMAGE_COLOR).reshape((300,300,4))
            im = np.expand_dims(im, axis=0)
            self.rl_data_dict["images"].append(im)

            num_target_reached = torch.clone(self.target_is_reached).cpu().numpy()
            self.rl_data_dict["target_reached_per_step"].append(num_target_reached)

        if self.use_dof_vel_action:
            if self.train_with_learned_reward:
                self.obs_buf = torch.cat((torch.clone(self.states["dof_pos"]), torch.clone(self.states["eef_pos"][:, :2]), torch.clone(self.obj_embs)), dim=-1)
            else:
                self.obs_buf = torch.cat((torch.clone(self.states["dof_pos"]), torch.clone(self.states["eef_pos"][:, :2]), torch.clone(self.states["box_pos"][:, :2]), torch.clone(self.states["cone_pos"][:, :2])), dim=-1)
        else:
            if self.train_with_learned_reward:
                self.obs_buf = torch.cat((torch.clone(self.states["eef_pos"][:, :2]), torch.clone(self.obj_embs)), dim=-1)
            else:
                self.obs_buf = torch.cat((torch.clone(self.states["eef_pos"][:, :2]), torch.clone(self.states["box_pos"][:, :2]), torch.clone(self.states["cone_pos"][:, :2])), dim=-1)
        
        return self.obs_buf

    def reset_idx(self, env_ids):
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        num_reset = len(env_ids)

        ##################### save data dict for visualization and resey data dict
        
        num_target_reached = torch.clone(self.target_is_reached).cpu().numpy()
        self.rl_data_dict["num_target_reached"] = num_target_reached
        if self.rl_data_idx % self.save_idx == 0:
            print(f"!!!!!!!!!!!!!!!!!!!!!! save rl data at idx {self.rl_data_idx} !!!!!!!!!!!!!!!!!!!!")
            with open(os.path.join(self.rl_data_path, f"episode {self.rl_data_idx}.pickle"), 'wb') as handle:
                pickle.dump(self.rl_data_dict, handle, protocol=3)
        self.rl_data_idx += 1

        self.target_is_reached = torch.zeros((self.num_envs,)).to(self.device)
        self.rl_data_dict = {"gt_cum_reward": np.zeros((self.num_envs,)), "learned_cum_reward": np.zeros((self.num_envs,)), "images": [], "num_target_reached":None, "target_reached_per_step": [], "eef_box_dist": []}

        ######################## reset box and cone (target) ###########################
        # self.box_pose = gymapi.Transform()
        # pose = list(np.array(self.init_pose) + np.random.uniform(low=[-0.1,-0.05,0], high=[0.1,0.05,0], size=3) + + np.array([0, 0, 0.015]))
        # self.box_pose.p = gymapi.Vec3(*pose)
        # self.cone_pose = gymapi.Transform()
        # pose = list(np.array(self.init_pose) + np.random.uniform(low=[-0.1,-0.05,0], high=[0.1,0.05,0], size=3))
        # pose[2] += CONE_Z_OFFSET
        # self.cone_pose.p = gymapi.Vec3(*pose)
        # while is_overlap(self.cone_pose, self.box_pose):
        #     pose = list(np.array(self.init_pose) + np.random.uniform(low=[-0.1,-0.05,0], high=[0.1,0.05,0], size=3))
        #     pose[2] += CONE_Z_OFFSET
        #     self.cone_pose.p = gymapi.Vec3(*pose)

        self.box_xyzs = np.array([[-1.0 for j in range(3)] for i in range(self.num_envs)])
        self.cone_xyzs = np.array([[-1.0 for j in range(3)] for i in range(self.num_envs)])
        self.box_poses = [None for i in range(self.num_envs)]
        self.cone_poses = [None for i in range(self.num_envs)]
        for i in range(self.num_envs):
            self.box_pose = gymapi.Transform()
            pose = np.array(self.init_pose) + np.random.uniform(low=[-0.1,-0.05,0], high=[0.1,0.05,0], size=3)
            self.box_pose.p = gymapi.Vec3(*list(pose))
            self.box_xyzs[i] = pose
            self.box_poses[i] = self.box_pose

            self.cone_pose = gymapi.Transform()
            pose = np.array(self.init_pose) + np.random.uniform(low=[-0.1,-0.05,0], high=[0.1,0.05,0], size=3)
            pose[2] += CONE_Z_OFFSET
            self.cone_pose.p = gymapi.Vec3(*list(pose))
            self.cone_xyzs[i] = pose
            self.cone_poses[i] = self.cone_pose

            while is_overlap_xy(self.cone_pose, self.box_pose):
                pose = np.array(self.init_pose) + np.random.uniform(low=[-0.1,-0.05,0], high=[0.1,0.05,0], size=3)
                pose[2] += CONE_Z_OFFSET
                self.cone_pose.p = gymapi.Vec3(*list(pose))
                self.cone_xyzs[i] = pose
                self.cone_poses[i] = self.cone_pose

        
        # self._box_state[env_ids, :3]=torch.tensor([self.box_pose.p.x, self.box_pose.p.y, self.box_pose.p.z], device=self.device).repeat(num_reset, 1)
        # self._cone_state[env_ids, :3]=torch.tensor([self.cone_pose.p.x, self.cone_pose.p.y, self.cone_pose.p.z], device=self.device).repeat(num_reset, 1)
        # # no rotation
        # self._box_state[env_ids, 3:7]=torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).repeat(num_reset, 1)
        # # 0 velocity
        # self._box_state[env_ids, 7:10]=torch.tensor([0.0, 0.0, 0.0], device=self.device).repeat(num_reset, 1)
        # self._box_state[env_ids, 10:13]=torch.tensor([0.0, 0.0, 0.0], device=self.device).repeat(num_reset, 1)

        self._box_state[env_ids, :3]=torch.tensor(self.box_xyzs, device=self.device).float()
        self._cone_state[env_ids, :3]=torch.tensor(self.cone_xyzs, device=self.device).float()
        # no rotation
        self._box_state[env_ids, 3:7]=torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).repeat(num_reset, 1)
        # 0 velocity
        self._box_state[env_ids, 7:10]=torch.tensor([0.0, 0.0, 0.0], device=self.device).repeat(num_reset, 1)
        self._box_state[env_ids, 10:13]=torch.tensor([0.0, 0.0, 0.0], device=self.device).repeat(num_reset, 1)
        
        
        multi_env_ids_cubes_int32 = self._global_indices[env_ids, 1:3].flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(multi_env_ids_cubes_int32), len(multi_env_ids_cubes_int32))

        ###################### reset dvrk ############################
        multi_env_ids_int32 = self._global_indices[env_ids, 0].flatten()
        dof_state = torch.clone(self._dof_state).cpu().numpy()
        dof_state[env_ids.cpu().numpy(), :, 0] = np.array([0.26925390614924577, 0.21282163028362158, 0.056475521072966536, -0.2626643147870063, 0.2399000201368886, -8.883981296955778e-06, 0.0065760339301758775, -0.2693216131913684, 0.35, -0.35])
        
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(torch.tensor(dof_state).to(self.device)), #self._init_dof_state
                                              gymtorch.unwrap_tensor(multi_env_ids_int32),
                                              len(multi_env_ids_int32))


        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def update_target_is_reached(self):
        sqr = self.states["box_to_cone_pos"]*self.states["box_to_cone_pos"]
        xy_sqr_dist = torch.sum(sqr[:, :2], dim=-1) # shape (num_envs,)
        self.target_is_reached = torch.where(xy_sqr_dist<=OVERLAP_TOLERANCE, torch.ones_like(xy_sqr_dist), self.target_is_reached)
        print(f"target_is_reached: {self.target_is_reached[0]}")

    def transform_coordinate(self, actions):
        actions = torch.clone(actions)
        actions[:, 0:2]*=-1
        # actions[:, 2] -= ROBOT_Z_OFFSET
        return actions

    def sigmoid_clip(self, input_tensor, min_value, max_value):
        scaled_tensor = torch.sigmoid(input_tensor)
        clipped_tensor = min_value + (max_value - min_value) * scaled_tensor
        return clipped_tensor



    def pre_physics_step(self, actions):
        if self.reset_pc and self.train_with_learned_reward:
            "skip pre physics step for resetting pc"
            return 
        self._refresh()

        if self.use_dof_vel_action:
            self.actions = actions.clone().to(self.device)
            #print(self.num_envs * self.num_dof,"++++++++++++++++++++", actions.shape)
            actions_tensor = torch.zeros(self.num_envs * self.num_dofs, device=self.device, dtype=torch.float)
            #mask out the gripper
            mask = torch.ones_like(actions)
            mask[:,9]=0
            mask[:,8]=0
            actions_tensor += (actions*mask).reshape(self.num_envs*self.num_dofs)*self.max_push_effort
            #actions_tensor*=0
            forces = gymtorch.unwrap_tensor(actions_tensor)
            #self.gym.set_dof_actuation_force_tensor(self.sim, forces)
            self.gym.set_dof_velocity_target_tensor(self.sim, forces)
        elif self.use_eef_vel_action:
            ########################## actions are eef vel ###############################
            print(f"vel raw action:{actions}")
            
            # eef_limits = self.tvc_behavior.get_eef_vel_limits()
            # print(f"eef limits: {eef_limits}")
            # eef_limit_max_norm = torch.max(torch.norm(eef_limits[:, :3], dim=1))
            # print("max action norm: ", eef_limit_max_norm)
            
            # norm = torch.norm(actions, dim=1).unsqueeze(1) #(num_envs, 1)
            # actions = torch.where(norm>=1, torch.clone(actions)/norm*1, torch.clone(actions)) #(num_envs, 3)
            actions_xyz = torch.zeros((self.num_envs, 3), device=self.device).float()
            actions_xyz[:, :2] = torch.clone(actions)
            actions = actions_xyz
            print(f"vel scaled action:{actions}")
           
            actions = torch.nn.functional.pad(actions, (0, 3), "constant", 0) # (num_envs, 6)
            assert(actions.shape[1]==6)
            dof_vel = self.tvc_behavior.get_action_from_eef_vel(actions, device="cpu").to(self.device) # (num_envs, 8)
            all_dof_vel = torch.nn.functional.pad(dof_vel, (0, 2), 'constant', 0) # add 0 velocities for the last 2 joints of the eef
            actions_tensor = all_dof_vel # convert to type float32.  
            self.actions = actions_tensor.clone().to(self.device)      
            actions_tensor = actions_tensor.reshape(self.num_envs*self.num_dofs) # do the same action for all environments, then flatten to a 1D vector for compatibility.
            actions_tensor = gymtorch.unwrap_tensor(actions_tensor)  
            self.gym.set_dof_velocity_target_tensor(self.sim, actions_tensor)
        else:
            ########################## actions are eef pos ###############################
            #### pos control with action=absolute_eef_pose####
            if self.use_abs_eef_action:
                print(f"pos action:{actions[0]}")
                actions_clipped = torch.clone(actions)
                actions_clipped[:, 0] = self.sigmoid_clip(actions_clipped[:, 0], min_value=-0.1, max_value=0.1)
                actions_clipped[:, 1] = self.sigmoid_clip(actions_clipped[:, 1], min_value=-0.6, max_value=-0.4)
                # actions_clipped[:, 2] = self.sigmoid_clip(actions_clipped[:, 2], min_value=0.1, max_value=0.11)
                # actions_clipped[:, 2] = 0.04

                eef_z = 0.04
                actions_xyz = torch.ones((self.num_envs, 3), device=self.device).float() * eef_z
                actions_xyz[:, :2] = torch.clone(actions_clipped)
                actions_clipped = actions_xyz
                
                print(f"pos clipped action:{actions_clipped[0]}")

                #####################
                actions_clipped_transformed = self.transform_coordinate(actions_clipped) 
                target_xyz = actions_clipped_transformed
                current_xyz = self.robots.get_ee_cartesian_position()[:, :3].to(self.device)
                delta_xyz = target_xyz - current_xyz
                self.tvc_behavior.set_target_pose(delta_xyzs=delta_xyz.detach().cpu().numpy())

                i = 0
                print(f"start position control")
                while(not all(self.tvc_behavior.successes)):
                    all_actions = self.tvc_behavior.get_action()
                    all_actions = np.pad(all_actions, ((0, 0), (0, 2)), mode='constant') # add 0 velocities for the last 2 joints of the eef, 
                    actions_tensor = torch.from_numpy(all_actions.astype('float32')).to(self.device)    # convert to type float32.        
                    actions_tensor = actions_tensor.reshape(self.num_envs*self.num_dofs) # do the same action for all environments, then flatten to a 1D vector for compatibility.
                    actions_tensor = gymtorch.unwrap_tensor(actions_tensor)  
                    self.gym.set_dof_velocity_target_tensor(self.sim, actions_tensor)
                    i+=1     
                    self.gym.simulate(self.sim)
                    self.render()
                    self._refresh()
                    print("robot successes: ", torch.sum(self.tvc_behavior.successes.int())) #
                self.tvc_behavior.successes = torch.tensor(self.num_envs*[False])
                print("finished pos control")

        


    def get_obj_pc_emb(self):
        self.gym.render_all_camera_sensors(self.sim)
        self._refresh()
        env_ids = np.arange(self.num_envs)

        #################### collect pcs from all envs ###########################
        processed_pcs = []
        max_num_points = 800
        vinv = np.linalg.inv(np.matrix(self.gym.get_camera_view_matrix(self.sim, self.envs[0], self.cam_handles[0])))
        proj = self.gym.get_camera_proj_matrix(self.sim, self.envs[0], self.cam_handles[0])
        for i in env_ids:
            print(f"getting pc at env {i}")
            pc = get_partial_pointcloud_vectorized_push_box(self.gym, self.sim, self.envs[i], self.cam_handles[i], vinv, proj, self.cam_prop, color=None, min_z=0.009, visualization=False, device="cuda")
            processed_pc = np.zeros((max_num_points, 3))
            if len(pc)==0:
                self.reset_pc = True
                print(f"!!!!!!!!!!!! pc recording failed for env {i}, try next time")
                return
            else:
                pad_point = pc[-1, :]
                processed_pc[:len(pc), :] = pc
                processed_pc[len(pc):, :] = pad_point
            processed_pcs.append(np.expand_dims(processed_pc, axis=0))

        self.reset_pc = False

        ################## farthest point sampling in batch and convert to obj emb
        pcs = np.concatenate(processed_pcs, axis=0)
        pcs = np.array(farthest_point_sample_batched(pcs, npoint=256)) # (num_envs, 256, 3)
        emb = to_obj_emb(self.encoder, self.device, pcs) # (num_envs, 256)

        self.obj_embs = emb

        vis = False
        if vis:
            with torch.no_grad():
                points = pcs[1]
                print(points.shape)

                points = points[np.random.permutation(points.shape[0])]
            
                pcd = open3d.geometry.PointCloud()
                pcd.points = open3d.utility.Vector3dVector(np.array(points))
                pcd.paint_uniform_color([0, 1, 0])

                points_tensor = torch.from_numpy(points.transpose(1,0)).unsqueeze(0).float().to(self.device)
                reconstructed_points = self.encoder(points_tensor)
                
                reconstructed_points = np.swapaxes(reconstructed_points.squeeze().cpu().detach().numpy(),0,1)
                reconstructed_points = reconstructed_points[:,:3]
                print(reconstructed_points.shape)

                pcd2 = open3d.geometry.PointCloud()
                pcd2.points = open3d.utility.Vector3dVector(np.array(reconstructed_points))
                pcd2.paint_uniform_color([1, 0, 0])
                open3d.visualization.draw_geometries([pcd, pcd2.translate((0,0,0.1))])


    def post_physics_step(self):
        if self.train_with_learned_reward:
            self.get_obj_pc_emb()

            if self.reset_pc:
                "skip post physics step for resetting pc"
                return 
        
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)
        self.update_target_is_reached()

