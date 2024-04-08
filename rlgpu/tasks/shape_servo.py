# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import os
import torch

from rlgpu.utils.torch_jit_utils import *
from rlgpu.tasks.base.base_task import BaseTask
from isaacgym import gymtorch
from isaacgym import gymapi


class ShapeServo(BaseTask):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.action_scale = self.cfg["env"]["actionScale"]
        self.start_position_noise = self.cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self.cfg["env"]["startRotationNoise"]
        self.num_props = self.cfg["env"]["numProps"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        self.dof_vel_scale = self.cfg["env"]["dofVelocityScale"]
        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self.cfg["env"]["rotRewardScale"]
        self.around_handle_reward_scale = self.cfg["env"]["aroundHandleRewardScale"]
        self.open_reward_scale = self.cfg["env"]["openRewardScale"]
        self.finger_dist_reward_scale = self.cfg["env"]["fingerDistRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.up_axis = "z"
        self.up_axis_idx = 2

        self.distX_offset = 0.04
        self.dt = 1/60.

        # prop dimensions
        self.prop_width = 0.08
        self.prop_height = 0.08
        self.prop_length = 0.08
        self.prop_spacing = 0.09

        num_obs = 25
        num_acts = 10

        self.cfg["env"]["numObservations"] = 25
        self.cfg["env"]["numActions"] = 10

        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless

        super().__init__(cfg=self.cfg)

        # # get gym GPU state tensors
        # actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        # dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        # rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        # self.gym.refresh_actor_root_state_tensor(self.sim)
        # self.gym.refresh_dof_state_tensor(self.sim)
        # self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.franka_default_dof_pos = to_torch([1.157, -1.066, -0.155, -2.239, -1.841, 1.003, 0.469, 0.469, 0.035, 0.035], device=self.device)
        # self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.franka_dof_state = torch.randn(1, 10, 2).to(self.device)
        self.franka_dof_pos = self.franka_dof_state[..., 0]
        self.franka_dof_vel = self.franka_dof_state[..., 1]


        # self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        # self.num_bodies = self.rigid_body_states.shape[1]

        # self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(self.num_envs, -1, 13)

        # if self.num_props > 0:
        #     self.prop_states = self.root_state_tensor[:, 2:]

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.franka_dof_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        # self.global_indices = torch.arange(self.num_envs * (2 + self.num_props), dtype=torch.int32, device=self.device).view(self.num_envs, -1)

        # self.reset(torch.arange(self.num_envs, device=self.device))

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        # asset_root = "../../assets"
        # franka_asset_file = "urdf/franka_description/robots/franka_panda.urdf"
        asset_root = "/home/baothach/dvrk_shape_servo/src/dvrk_env"
        franka_asset_file = "dvrk_description/psm/psm_for_issacgym.urdf"
        cabinet_asset_file = "urdf/sektion_cabinet_model/urdf/sektion_cabinet_2.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = self.cfg["env"]["asset"].get("assetRoot", asset_root)
            franka_asset_file = self.cfg["env"]["asset"].get("assetFileNameFranka", franka_asset_file)
            cabinet_asset_file = self.cfg["env"]["asset"].get("assetFileNameCabinet", cabinet_asset_file)

        # load franka asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.use_mesh_materials = True
        franka_asset = self.gym.load_asset(self.sim, asset_root, franka_asset_file, asset_options)

        # load cabinet asset
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = False
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.armature = 0.005
        asset_root = "../../assets"
        cabinet_asset = self.gym.load_asset(self.sim, asset_root, cabinet_asset_file, asset_options)

        # franka_dof_stiffness = to_torch([400, 400, 400, 400, 400, 400, 400, 400, 1.0e6, 1.0e6], dtype=torch.float, device=self.device)
        # franka_dof_damping = to_torch([80, 80, 80, 80, 80, 80, 80, 80, 1.0e2, 1.0e2], dtype=torch.float, device=self.device)

        # self.num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        # self.num_franka_dofs = self.gym.get_asset_dof_count(franka_asset)
        # self.num_cabinet_bodies = self.gym.get_asset_rigid_body_count(cabinet_asset)
        # self.num_cabinet_dofs = self.gym.get_asset_dof_count(cabinet_asset)

        # print("num franka bodies: ", self.num_franka_bodies)
        # print("num franka dofs: ", self.num_franka_dofs)
        # print("num cabinet bodies: ", self.num_cabinet_bodies)
        # print("num cabinet dofs: ", self.num_cabinet_dofs)

        # # set franka dof properties
        franka_dof_props = self.gym.get_asset_dof_properties(franka_asset)
        # self.franka_dof_lower_limits = []
        # self.franka_dof_upper_limits = []
        # for i in range(self.num_franka_dofs):
        #     franka_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
        #     if self.physics_engine == gymapi.SIM_PHYSX:
        #         franka_dof_props['stiffness'][i] = franka_dof_stiffness[i]
        #         franka_dof_props['damping'][i] = franka_dof_damping[i]
        #     else:
        #         franka_dof_props['stiffness'][i] = 7000.0
        #         franka_dof_props['damping'][i] = 50.0

        #     self.franka_dof_lower_limits.append(franka_dof_props['lower'][i])
        #     self.franka_dof_upper_limits.append(franka_dof_props['upper'][i])

        # self.franka_dof_lower_limits = to_torch(self.franka_dof_lower_limits, device=self.device)
        # self.franka_dof_upper_limits = to_torch(self.franka_dof_upper_limits, device=self.device)
        # self.franka_dof_speed_scales = torch.ones_like(self.franka_dof_lower_limits)
        # self.franka_dof_speed_scales[[7, 8]] = 0.1
        # franka_dof_props['effort'][7] = 200
        # franka_dof_props['effort'][8] = 200

        # # set cabinet dof properties
        # cabinet_dof_props = self.gym.get_asset_dof_properties(cabinet_asset)
        # for i in range(self.num_cabinet_dofs):
        #     cabinet_dof_props['damping'][i] = 10.0

        # create prop assets
        box_opts = gymapi.AssetOptions()
        box_opts.density = 400
        prop_asset = self.gym.create_box(self.sim, self.prop_width, self.prop_height, self.prop_width, box_opts)

        franka_start_pose = gymapi.Transform()
        franka_start_pose.p = gymapi.Vec3(1.0, 0.0, 0.0)
        franka_start_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)

        cabinet_start_pose = gymapi.Transform()
        cabinet_start_pose.p = gymapi.Vec3(*get_axis_params(0.4, self.up_axis_idx))

        # compute aggregate size
        num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)
        num_cabinet_bodies = self.gym.get_asset_rigid_body_count(cabinet_asset)
        num_cabinet_shapes = self.gym.get_asset_rigid_shape_count(cabinet_asset)
        num_prop_bodies = self.gym.get_asset_rigid_body_count(prop_asset)
        num_prop_shapes = self.gym.get_asset_rigid_shape_count(prop_asset)
        max_agg_bodies = num_franka_bodies + num_cabinet_bodies + self.num_props * num_prop_bodies
        max_agg_shapes = num_franka_shapes + num_cabinet_shapes + self.num_props * num_prop_shapes

        self.frankas = []
        self.cabinets = []
        self.default_prop_states = []
        self.prop_start = []
        self.envs = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )



            franka_actor = self.gym.create_actor(env_ptr, franka_asset, franka_start_pose, "franka", i, 1, 0)
            # self.gym.set_actor_dof_properties(env_ptr, franka_actor, franka_dof_props)

        

            self.envs.append(env_ptr)
            self.frankas.append(franka_actor)
           

        self.hand_handle = self.gym.find_actor_rigid_body_handle(env_ptr, franka_actor, "panda_link7")
        self.lfinger_handle = self.gym.find_actor_rigid_body_handle(env_ptr, franka_actor, "panda_leftfinger")
        self.rfinger_handle = self.gym.find_actor_rigid_body_handle(env_ptr, franka_actor, "panda_rightfinger")
        # self.default_prop_states = to_torch(self.default_prop_states, device=self.device, dtype=torch.float).view(self.num_envs, self.num_props, 13)



    def compute_reward(self):
        self.rew_buf[:], self.reset_buf[:] = compute_franka_reward(
            self.reset_buf, self.progress_buf
        )

    def compute_observations(self):

        self.obs_buf = torch.randn(1, 25)
        # print("=====obs:", self.obs_buf.shape)
        return self.obs_buf.to(self.device)

    def reset(self, env_ids):
        print("=========resetting....")
        
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        # # reset franka
        # pos = tensor_clamp(
        #     self.franka_default_dof_pos.unsqueeze(0) + 0.25 * (torch.rand((len(env_ids), self.num_franka_dofs), device=self.device) - 0.5),
        #     self.franka_dof_lower_limits, self.franka_dof_upper_limits)
        # self.franka_dof_pos[env_ids, :] = pos
        # self.franka_dof_vel[env_ids, :] = torch.zeros_like(self.franka_dof_vel[env_ids])
        # self.franka_dof_targets[env_ids, :self.num_franka_dofs] = pos



    def pre_physics_step(self, actions):
        # print("===teo:", actions)

        # self.actions = actions.clone().to(self.device)
        # # print("===teo:", self.franka_dof_speed_scales.shape, self.actions.shape)
        # targets = self.franka_dof_targets[:, :self.num_franka_dofs] + self.franka_dof_speed_scales * self.dt * self.actions * self.action_scale
        # self.franka_dof_targets[:, :self.num_franka_dofs] = tensor_clamp(
        #     targets, self.franka_dof_lower_limits, self.franka_dof_upper_limits)
        # env_ids_int32 = torch.arange(self.num_envs, dtype=torch.int32, device=self.device)
        # self.gym.set_dof_position_target_tensor(self.sim,
                                                # gymtorch.unwrap_tensor(self.franka_dof_targets))
        self.gym.set_actor_dof_position_targets(self.envs[0], self.frankas[0], np.random.rand(10).astype('float32'))      
                              

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset(env_ids)

        self.compute_observations()
        self.compute_reward()

   

#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_franka_reward(reset_buf, progress_buf):

    rewards = torch.tensor([10])
    reset_buf = torch.tensor([0])

    # print("=======rewards shape:", rewards.shape)
    # print("=======reset_buf shape:", reset_buf.shape)


    return rewards, reset_buf


@torch.jit.script
def compute_grasp_transforms(hand_rot, hand_pos, franka_local_grasp_rot, franka_local_grasp_pos,
                             drawer_rot, drawer_pos, drawer_local_grasp_rot, drawer_local_grasp_pos
                             ):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]

    global_franka_rot, global_franka_pos = tf_combine(
        hand_rot, hand_pos, franka_local_grasp_rot, franka_local_grasp_pos)
    global_drawer_rot, global_drawer_pos = tf_combine(
        drawer_rot, drawer_pos, drawer_local_grasp_rot, drawer_local_grasp_pos)

    return global_franka_rot, global_franka_pos, global_drawer_rot, global_drawer_pos
