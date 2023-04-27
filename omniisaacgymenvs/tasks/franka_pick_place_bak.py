# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.franka import Franka
from omniisaacgymenvs.robots.articulations.cabinet import Cabinet
from omniisaacgymenvs.robots.articulations.views.franka_view import FrankaView
from omniisaacgymenvs.robots.articulations.views.cabinet_view import CabinetView

from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.prims import RigidPrim, RigidPrimView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.torch.transformations import *
from omni.isaac.core.utils.torch.rotations import *

from omni.isaac.cloner import Cloner

import numpy as np
import torch
import math

from pxr import Usd, UsdGeom

from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.tasks.base_task import BaseTask
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrim
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.utils.stage import get_stage_units
from omni.isaac.core.utils.string import find_unique_string_name
from omniisaacgymenvs.robots.articulations.views.franka_view import FrankaView
from omniisaacgymenvs.robots.articulations.franka import Franka
import random
import math

import omni.kit

from gym import spaces
import numpy as np
import torch
import math


class Franka_(RLTask):
    def __init__(
        self,
        name,
        sim_config,
        env,
        offset=None
    ) -> None:
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]

        self._max_episode_length = self._task_cfg["env"]["episodeLength"]

        self.action_scale = self._task_cfg["env"]["actionScale"]
        self.start_position_noise = self._task_cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self._task_cfg["env"]["startRotationNoise"]
        self.num_props = self._task_cfg["env"]["numProps"]

        self.dof_vel_scale = self._task_cfg["env"]["dofVelocityScale"]
        self.dist_reward_scale = self._task_cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self._task_cfg["env"]["rotRewardScale"]
        self.around_handle_reward_scale = self._task_cfg["env"]["aroundHandleRewardScale"]
        self.open_reward_scale = self._task_cfg["env"]["openRewardScale"]
        self.finger_dist_reward_scale = self._task_cfg["env"]["fingerDistRewardScale"]
        self.action_penalty_scale = self._task_cfg["env"]["actionPenaltyScale"]
        self.finger_close_reward_scale = self._task_cfg["env"]["fingerCloseRewardScale"]

        self.distX_offset = 0.04
        self.dt = 1/60.

        self._franka_position = [0.0, 0.0, 0.0]
        self._block_position = [0.5, 0.0, 0.0]
        self._reset_dist = 3.0
        self._max_push_effort = 15.0

        self.distX_offset = 0.04
        self.dt = 1/60.

        self._num_observations = 21
        self._num_actions = 9

        self._target_cube = None
        self._cube = None
        self._cube_initial_position = None
        self._cube_initial_orientation = None
        self._target_position = None
        # self.target = torch.zeros([256,3]).cuda()
        self._cube_size = None
        if self._cube_size is None:
            self._cube_size = np.array([0.0515, 0.0515, 0.0515]) / get_stage_units()
        if self._cube_initial_position is None:
            self._cube_initial_position = np.array([1.0, 0.5, 0.0]) / get_stage_units()
        if self._cube_initial_orientation is None:
            self._cube_initial_orientation = np.array([1, 0, 0, 0])
        if self._target_position is None:
            self._target_position = np.array([-0.3, -0.3, 0]) / get_stage_units()
            self._target_position[2] = self._cube_size[2] / 2.0
        self._target_position = self._target_position


        RLTask.__init__(self, name, env)
        return

    def set_up_scene(self, scene) -> None:

        self.get_franka()
        super().set_up_scene(scene)
        self._franka = FrankaView(prim_paths_expr="/World/envs/.*/franka", name="franka_view")
        scene.add_default_ground_plane()

        cube_name = find_unique_string_name(initial_name="cube", is_unique_fn=lambda x: not self.scene.object_exists(x))
        cube = DynamicCuboid(
                name=cube_name,
                position=torch.tensor([1.0, 0.5, 0.0]).cuda() + self._env_pos[0],
                orientation=self._cube_initial_orientation,
                prim_path=self.default_zero_env_path + "/cube",
                scale=self._cube_size,
                size=1.0,
                color=np.array([0, 0, 1]),
            )
        cube2 = DynamicCuboid(
                name="cube2",
                position=torch.tensor([0.5, 0.5, 0.0]).cuda() + self._env_pos[0],
                orientation=self._cube_initial_orientation,
                prim_path=self.default_zero_env_path + "/cube2",
                scale=self._cube_size,
                size=1.0,
                color=np.array([1, 0, 0]),
            )

        self._cube = RigidPrimView(prim_paths_expr="/World/envs/.*/cube", name="cube_view")
        self._cube2 = RigidPrimView(prim_paths_expr="/World/envs/.*/cube2", name="cube_view2")
        scene.add(self._franka)
        scene.add(self._franka._hands)
        scene.add(self._franka._lfingers)
        scene.add(self._franka._rfingers)
        scene.add(self._cube)
        scene.add(self._cube2)

        self.init_data()
        return

    def get_franka(self):
        franka = Franka(prim_path=self.default_zero_env_path + "/franka", name="franka")
        self._sim_config.apply_articulation_settings("franka", get_prim_at_path(franka.prim_path), self._sim_config.parse_actor_config("franka"))

    def init_data(self) -> None:
        def get_env_local_pose(env_pos, xformable, device):
            """Compute pose in env-local coordinates"""
            world_transform = xformable.ComputeLocalToWorldTransform(0)
            world_pos = world_transform.ExtractTranslation()
            world_quat = world_transform.ExtractRotationQuat()

            px = world_pos[0] - env_pos[0]
            py = world_pos[1] - env_pos[1]
            pz = world_pos[2] - env_pos[2]
            qx = world_quat.imaginary[0]
            qy = world_quat.imaginary[1]
            qz = world_quat.imaginary[2]
            qw = world_quat.real

            return torch.tensor([px, py, pz, qw, qx, qy, qz], device=device, dtype=torch.float)

        stage = get_current_stage()
        hand_pose = get_env_local_pose(self._env_pos[0],
                                       UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/franka/panda_link7")),
                                       self._device)
        lfinger_pose = get_env_local_pose(
            self._env_pos[0], UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/franka/panda_leftfinger")),
            self._device
        )
        rfinger_pose = get_env_local_pose(
            self._env_pos[0], UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/franka/panda_rightfinger")),
            self._device
        )

        finger_pose = torch.zeros(7, device=self._device)
        finger_pose[0:3] = (lfinger_pose[0:3] + rfinger_pose[0:3]) / 2.0
        finger_pose[3:7] = lfinger_pose[3:7]
        hand_pose_inv_rot, hand_pose_inv_pos = (tf_inverse(hand_pose[3:7], hand_pose[0:3]))

        grasp_pose_axis = 1
        franka_local_grasp_pose_rot, franka_local_pose_pos = tf_combine(hand_pose_inv_rot, hand_pose_inv_pos,
                                                                        finger_pose[3:7], finger_pose[0:3])
        franka_local_pose_pos += torch.tensor([0, 0.04, 0], device=self._device)
        self.franka_local_grasp_pos = franka_local_pose_pos.repeat((self._num_envs, 1))
        self.franka_local_grasp_rot = franka_local_grasp_pose_rot.repeat((self._num_envs, 1))
        
        drawer_local_grasp_pose = torch.tensor([0.0, 0.01, 0.2, 1.0, 0.0, 0.0, 0.0], device=self._device)
        self.drawer_local_grasp_pos = drawer_local_grasp_pose[0:3].repeat((self._num_envs, 1))
        self.drawer_local_grasp_rot = drawer_local_grasp_pose[3:7].repeat((self._num_envs, 1))

        self.gripper_up_axis = torch.tensor([0, 1, 0], device=self._device, dtype=torch.float).repeat((self._num_envs, 1))
        self.drawer_up_axis = torch.tensor([0, 0, 1], device=self._device, dtype=torch.float).repeat((self._num_envs, 1))


        self.franka_default_dof_pos = torch.tensor(
            [1.157, -1.066, -0.155, -2.239, -1.841, 1.003, 0.469, 0.035, 0.035], device=self._device
        )


        self.actions = torch.zeros((self._num_envs, self.num_actions), device=self._device)

    def post_reset(self):
        self.num_franka_dofs = self._franka.num_dof
        self.target = torch.zeros([self._franka.count, 3]).cuda()
        self.target_rot = torch.zeros([self._franka.count, 4]).cuda()

        self.franka_dof_pos = torch.zeros((self._franka.count, self.num_franka_dofs), device=self._device)
        dof_limits = self._franka.get_dof_limits()
        self.franka_dof_lower_limits = dof_limits[0, :, 0].to(device=self._device)
        self.franka_dof_upper_limits = dof_limits[0, :, 1].to(device=self._device)
        self.franka_dof_speed_scales = torch.ones_like(self.franka_dof_lower_limits)
        self.franka_dof_speed_scales[self._franka.gripper_indices] = 0.1
        self.franka_dof_targets = torch.zeros(
            (self._franka.count, self.num_franka_dofs), dtype=torch.float, device=self._device
        )

        self.default_cube_pos, self.default_cube_rot = self._cube.get_world_poses()
        self.cube_indices = torch.arange(self._num_envs * 1, device=self._device).view(self._num_envs, 1)

        self.target[:, 0] = self.default_cube_pos[:, 0]
        self.target[:, 1] = self.default_cube_pos[:, 1]
        self.target[:, 2] = self.default_cube_pos[:, 2] + 0.5
        
        self.target_rot[:, 0] = -1.0
        self.target_rot[:, 1] = 0.0
        self.target_rot[:, 2] = 0.0
        self.target_rot[:, 3] = 0.0

        # randomize all envs
        indices = torch.arange(self._franka.count, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def get_observations(self) -> dict:
        self.hand_pos, self.hand_rot = self._franka._hands.get_world_poses(clone=False)
        franka_dof_pos = self._franka.get_joint_positions(clone=False)
        franka_dof_vel = self._franka.get_joint_velocities(clone=False)
        self.cube_position, self.cube_orientation = self._cube.get_world_poses()
        self.cube2_position, _ = self._cube2.get_world_poses()
        self.franka_dof_pos = franka_dof_pos

        self.franka_lfinger_pos, self.franka_lfinger_rot = self._franka._lfingers.get_world_poses(clone=False)
        self.franka_rfinger_pos, self.franka_rfinger_rot = self._franka._lfingers.get_world_poses(clone=False)
        
        self.franka_grasp_rot, self.franka_grasp_pos, self.drawer_grasp_rot, self.drawer_grasp_pos = self.compute_grasp_transforms(
            self.hand_rot,
            self.hand_pos,
            self.franka_local_grasp_rot,
            self.franka_local_grasp_pos,
            self.cube_orientation,
            self.cube_position,
            self.drawer_local_grasp_rot,
            self.drawer_local_grasp_pos,
        )

        dof_pos_scaled = (
                2.0
                * (franka_dof_pos - self.franka_dof_lower_limits)
                / (self.franka_dof_upper_limits - self.franka_dof_lower_limits)
                - 1.0
        )
        to_target = self.drawer_grasp_pos - self.franka_grasp_pos
        self.obs_buf = torch.cat(
            (
                dof_pos_scaled,
                franka_dof_vel * self.dof_vel_scale,
                to_target,             
            ),
            dim=-1,
        )

        observations = {
            self._franka.name: {
                "obs_buf": self.obs_buf
            }
        }
        return observations
        
    def compute_grasp_transforms(
        self,
        hand_rot,
        hand_pos,
        franka_local_grasp_rot,
        franka_local_grasp_pos,
        drawer_rot,
        drawer_pos,
        drawer_local_grasp_rot,
        drawer_local_grasp_pos,
    ):

        global_franka_rot, global_franka_pos = tf_combine(
            hand_rot, hand_pos, franka_local_grasp_rot, franka_local_grasp_pos
        )
        global_drawer_rot, global_drawer_pos = tf_combine(
            drawer_rot, drawer_pos, drawer_local_grasp_rot, drawer_local_grasp_pos
        )

        return global_franka_rot, global_franka_pos, global_drawer_rot, global_drawer_pos

    def pre_physics_step(self, actions) -> None:
        if not self._env._world.is_playing():
            return

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        self.actions = actions.clone().to(self._device)
        targets = self.franka_dof_targets + self.franka_dof_speed_scales * self.dt * self.actions * self.action_scale
        self.franka_dof_targets[:] = tensor_clamp(targets, self.franka_dof_lower_limits, self.franka_dof_upper_limits)
        env_ids_int32 = torch.arange(self._franka.count, dtype=torch.int32, device=self._device)

        self._franka.set_joint_position_targets(self.franka_dof_targets, indices=env_ids_int32)

    def reset_idx(self, env_ids):
        indices = env_ids.to(dtype=torch.int32)
        num_indices = len(indices)
        self.num_franka_dofs = self._franka.num_dof

        # reset franka
        pos = tensor_clamp(
            self.franka_default_dof_pos.unsqueeze(0)
            + 0.25 * (torch.rand((len(env_ids), self.num_franka_dofs), device=self._device) - 0.5),
            self.franka_dof_lower_limits,
            self.franka_dof_upper_limits,
        )
        dof_pos = torch.zeros((num_indices, self._franka.num_dof), device=self._device)
        dof_vel = torch.zeros((num_indices, self._franka.num_dof), device=self._device)
        dof_pos[:, :] = pos
        self.franka_dof_targets[env_ids, :] = pos
        self.franka_dof_pos[env_ids, :] = pos

        self._franka.set_joint_position_targets(self.franka_dof_targets[env_ids], indices=indices)
        self._franka.set_joint_positions(dof_pos, indices=indices)
        self._franka.set_joint_velocities(dof_vel, indices=indices)

        # self._cube.set_world_pose(torch.from_numpy(0.6*np.array([1.0,0,0])).float())
        self._cube.set_world_poses(
                self.default_cube_pos[self.cube_indices[env_ids].flatten()],
                self.default_cube_rot[self.cube_indices[env_ids].flatten()],
                self.cube_indices[env_ids].flatten().to(torch.int32)
        )

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0


    def calculate_metrics(self) -> None:
        # gripper_position, _ = self._franka._hands.get_world_poses(clone=False)
        # self.cube_position, self.cube_orientation = self._cube.get_world_poses()
        # self.cube_position -= self._env_pos[0]
        # gripper_position -= self._env_pos[0]

        d1 = torch.norm(self.franka_grasp_pos - self.drawer_grasp_pos, p=2, dim=-1)
        dist_reward1 = 1.0 / (1.0 + d1 ** 2)
        dist_reward1 *= dist_reward1
        dist_reward1 = torch.where(d1 <= 0.02, dist_reward1 * 2, dist_reward1)
        
        # axis1 = tf_vector(franka_grasp_rot, gripper_forward_axis)
        # axis2 = tf_vector(drawer_grasp_rot, drawer_inward_axis)
        axis3 = tf_vector(self.franka_grasp_rot, self.gripper_up_axis)
        axis4 = tf_vector(self.drawer_grasp_rot, self.drawer_up_axis)
        
        # dot1 = torch.bmm(axis1.view(num_envs, 1, 3), axis2.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)  # alignment of forward axis for gripper
        dot1 = torch.bmm(axis3.view(self._num_envs, 1, 3), axis4.view(self._num_envs, 3, 1)).squeeze(-1).squeeze(-1)  # alignment of up axis for gripper

        #rot = (torch.bmm(self.franka_lfinger_rot.view(self._num_envs, 1, 4), self.target_rot.view(self._num_envs, 4, 1)).squeeze(-1)).squeeze(-1)
        # print(rot.shape)
        # quit()
        # rot_reward = 1.0 / (1.0 + rot ** 2)
        # rot_reward *= rot_reward
        # rot_reward = torch.where(d1 <= 0.02, rot_reward * 2, rot_reward)
        rot_reward = 0.5 * (torch.sign(dot1) * dot1 ** 2)

        finger_close_reward = torch.zeros_like(rot_reward)
        finger_close_reward = torch.where(d1 <=0.03, (0.04 - self.franka_dof_pos[:, 7]) + (0.04 - self.franka_dof_pos[:, 8]), finger_close_reward)

        d2 = torch.norm((self.franka_lfinger_pos + self.franka_lfinger_pos)/2.0 - self.target, p=2, dim=-1)
        dist_reward2 = 1.0 / (1.0 + d2 ** 2)
        dist_reward2 *= dist_reward2
        dist_reward2 = torch.where(d2 <= 0.02, dist_reward2 * 2, dist_reward2)

        # dist_target = torch.norm(self.cube2_position - self.cube_position, p=2, dim=-1) #((self.cube_position[:, 0] - 0.5) ** 2 + (self.cube_position[:, 1] - 0.5) ** 2 + (self.cube_position[:, 2] - 0) ** 2) ** 0.5
        # dist_reward2 = 1.0 / (1.0 + dist_target ** 2)
        # dist_reward2 *= dist_reward2
        # dist_reward2 = torch.where(dist_target <= 0.02, dist_reward2 * 2, dist_reward2)

        action_penalty = torch.sum(self.actions ** 2, dim=-1)

        rewards = self.rot_reward_scale * rot_reward + self.dist_reward_scale * (dist_reward1) #- self.action_penalty_scale * action_penalty + 
                  # + self.finger_close_reward_scale * finger_close_reward - self.action_penalty_scale * action_penalty # + dist_reward2) #- self.action_penalty_scale * action_penalty

        self.rew_buf[:] = rewards

        # return rewards

    def is_done(self) -> None:
        # self.cube_position, self.cube_orientation = self._cube.get_world_poses()
        # self.cube_position -= self._env_pos[0]
        dist_cube_to_target = torch.norm(self.cube2_position - self.cube_position, p=2, dim=-1)
        # #
        self.reset_buf = torch.where(dist_cube_to_target >= 100.0, torch.ones_like(self.reset_buf), self.reset_buf)
        self.reset_buf = torch.where(self.progress_buf >= self._max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf)
