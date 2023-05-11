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
from omniisaacgymenvs.tasks.factory.factory_control import *

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
        self.franka_finger_length = 0.053671

        self.distX_offset = 0.04
        self.dt = 1 / 60.

        self._num_observations = 24 + 18
        self._num_actions = 9

        RLTask.__init__(self, name, env)
        return

    def set_up_scene(self, scene) -> None:

        self.get_franka()

        super().set_up_scene(scene)

        self._frankas = FrankaView(prim_paths_expr="/World/envs/.*/franka", name="franka_view")

        scene.add(self._frankas)
        scene.add(self._frankas._hands)
        scene.add(self._frankas._lfingers)
        scene.add(self._frankas._rfingers)

        cube1 = DynamicCuboid(
                name="cube1",
                position=np.array([-0.55, 0.0, 0.0]) ,
                orientation=np.array([1, 0, 0, 0]),
                prim_path=self.default_zero_env_path + "/cube1",
                scale=np.array([0.0515, 0.0515, 0.0515]),
                size=1.0,
                color=np.array([0, 0, 1]),
            )
        cube2 = DynamicCuboid(
                name="cube2",
                position=np.array([0.5, 0.0, 0.0]),
                orientation=np.array([1, 0, 0, 0]),
                prim_path=self.default_zero_env_path + "/cube2",
                scale=np.array([0.0515, 0.0515, 0.0515]),
                size=1.0,
                color=np.array([1, 0, 0]),
            )

        self._cubeA = RigidPrimView(prim_paths_expr="/World/envs/.*/cube1", name="cubeA_view",
                                        reset_xform_properties=False)
        self._cubeB = RigidPrimView(prim_paths_expr="/World/envs/.*/cube2", name="cubeB_view",
                                        reset_xform_properties=False)
        scene.add(self._cubeA)
        scene.add(self._cubeB)

        self.init_data()
        return

    def get_franka(self):
        franka = Franka(prim_path=self.default_zero_env_path + "/franka", name="franka")
        self._sim_config.apply_articulation_settings("franka", get_prim_at_path(franka.prim_path),
                                                     self._sim_config.parse_actor_config("franka"))

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

        drawer_local_grasp_pose = torch.tensor([0.3, 0.01, 0.0, 1.0, 0.0, 0.0, 0.0], device=self._device)
        self.drawer_local_grasp_pos = drawer_local_grasp_pose[0:3].repeat((self._num_envs, 1))
        self.drawer_local_grasp_rot = drawer_local_grasp_pose[3:7].repeat((self._num_envs, 1))

        self.gripper_forward_axis = torch.tensor([0, 0, 1], device=self._device, dtype=torch.float).repeat(
            (self._num_envs, 1))
        self.drawer_inward_axis = torch.tensor([-1, 0, 0], device=self._device, dtype=torch.float).repeat(
            (self._num_envs, 1))
        self.gripper_up_axis = torch.tensor([0, 1, 0], device=self._device, dtype=torch.float).repeat(
            (self._num_envs, 1))
        self.drawer_up_axis = torch.tensor([0, 0, 1], device=self._device, dtype=torch.float).repeat(
            (self._num_envs, 1))

        self.franka_default_dof_pos = torch.tensor(
            [1.157, -1.066, -0.155, -2.239, -1.841, 1.003, 0.469, 0.035, 0.035], device=self._device
        )

        self.actions = torch.zeros((self._num_envs, self.num_actions), device=self._device)

    def get_observations(self) -> dict:
        self.hand_pos, self.hand_rot = self._frankas._hands.get_world_poses(clone=False)
        franka_dof_pos = self._frankas.get_joint_positions(clone=False)
        franka_dof_vel = self._frankas.get_joint_velocities(clone=False)
        self.franka_dof_pos = franka_dof_pos

        self.cubeA_pos, self.cubeA_rot = self._cubeA.get_world_poses(clone=False)
        self.cubeB_pos, self.cubeB_rot = self._cubeB.get_world_poses(clone=False)

        self.cubeA_to_cubeB_pos = self.cubeB_pos - self.cubeA_pos

        self.franka_lfinger_pos, self.franka_lfinger_rot = self._frankas._lfingers.get_world_poses(clone=False)
        self.franka_rfinger_pos, self.franka_rfinger_rot = self._frankas._rfingers.get_world_poses(clone=False)

        dof_pos_scaled = (
                2.0
                * (franka_dof_pos - self.franka_dof_lower_limits)
                / (self.franka_dof_upper_limits - self.franka_dof_lower_limits)
                - 1.0
        )
        to_target = (self.franka_lfinger_pos + self.franka_rfinger_pos)/2.0 - self.cubeA_pos

        self.obs_buf = torch.cat(
            (
                dof_pos_scaled,
                franka_dof_vel * self.dof_vel_scale,
                (self.franka_lfinger_pos + self.franka_rfinger_pos)/2.0, self.hand_rot,
                self.cubeA_pos, self.cubeA_rot,
                self.cubeB_pos, self.cubeB_rot,
                self.cubeA_to_cubeB_pos,
            ),
            dim=-1,
        )

        observations = {
            self._frankas.name: {
                "obs_buf": self.obs_buf
            }
        }
        return observations

    def pre_physics_step(self, actions) -> None:
        if not self._env._world.is_playing():
            return

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        self.actions = actions.clone().to(self._device)

        self.franka_dof_targets = self.compute_dof_pos_target(self.franka_dof_pos[:,0:7], self.finger_midpoint_pos,
                                   self.fingertip_midpoint_quat,
                                   self.fingertip_midpoint_jacobian,
                                   self.actions[:,0:3],
                                   self.actions[:,3:7],
                                   self.actions[:,7:9],
                                   self._device)

        # targets = self.franka_dof_targets + self.franka_dof_speed_scales * self.dt * self.actions * self.action_scale
        # self.franka_dof_targets[:] = tensor_clamp(targets, self.franka_dof_lower_limits, self.franka_dof_upper_limits)
        env_ids_int32 = torch.arange(self._frankas.count, dtype=torch.int32, device=self._device)

        self._frankas.set_joint_position_targets(self.franka_dof_targets, indices=env_ids_int32)

    def compute_dof_pos_target(self, arm_dof_pos,
                               fingertip_midpoint_pos,
                               fingertip_midpoint_quat,
                               jacobian,
                               ctrl_target_fingertip_midpoint_pos,
                               ctrl_target_fingertip_midpoint_quat,
                               ctrl_target_gripper_dof_pos,
                               device):
        """Compute Franka DOF position target to move fingertips towards target pose."""

        ctrl_target_dof_pos = torch.zeros((self._num_envs, 9), device=device)

        pos_error, axis_angle_error = self.get_pose_error(
            fingertip_midpoint_pos=fingertip_midpoint_pos,
            fingertip_midpoint_quat=fingertip_midpoint_quat,
            ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat,
            jacobian_type='geometric',
            rot_error_type='axis_angle')

        delta_fingertip_pose = torch.cat((pos_error, axis_angle_error), dim=1)
        delta_arm_dof_pos = self._get_delta_dof_pos(delta_pose=delta_fingertip_pose,
                                               ik_method='pinv',
                                               jacobian=jacobian,
                                               device=device)

        ctrl_target_dof_pos[:, 0:7] = arm_dof_pos + delta_arm_dof_pos
        ctrl_target_dof_pos[:, 7:9] = ctrl_target_gripper_dof_pos  # gripper finger joints

        return ctrl_target_dof_pos

    def _get_delta_dof_pos(self,delta_pose, ik_method, jacobian, device):
        """Get delta Franka DOF position from delta pose using specified IK method."""
        # References:
        # 1) https://www.cs.cmu.edu/~15464-s13/lectures/lecture6/iksurvey.pdf
        # 2) https://ethz.ch/content/dam/ethz/special-interest/mavt/robotics-n-intelligent-systems/rsl-dam/documents/RobotDynamics2018/RD_HS2018script.pdf (p. 47)

        if ik_method == 'pinv':  # Jacobian pseudoinverse
            k_val = 1.0
            jacobian_pinv = torch.linalg.pinv(jacobian)
            delta_dof_pos = k_val * jacobian_pinv @ delta_pose.unsqueeze(-1)
            delta_dof_pos = delta_dof_pos.squeeze(-1)

        elif ik_method == 'trans':  # Jacobian transpose
            k_val = 1.0
            jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
            delta_dof_pos = k_val * jacobian_T @ delta_pose.unsqueeze(-1)
            delta_dof_pos = delta_dof_pos.squeeze(-1)

        elif ik_method == 'dls':  # damped least squares (Levenberg-Marquardt)
            lambda_val = 0.1
            jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
            lambda_matrix = (lambda_val ** 2) * torch.eye(n=jacobian.shape[1], device=device)
            delta_dof_pos = jacobian_T @ torch.inverse(jacobian @ jacobian_T + lambda_matrix) @ delta_pose.unsqueeze(-1)
            delta_dof_pos = delta_dof_pos.squeeze(-1)

        elif ik_method == 'svd':  # adaptive SVD
            k_val = 1.0
            U, S, Vh = torch.linalg.svd(jacobian)
            S_inv = 1. / S
            min_singular_value = 1.0e-5
            S_inv = torch.where(S > min_singular_value, S_inv, torch.zeros_like(S_inv))
            jacobian_pinv = torch.transpose(Vh, dim0=1, dim1=2)[:, :, :6] @ torch.diag_embed(S_inv) @ torch.transpose(U,
                                                                                                                      dim0=1,
                                                                                                                      dim1=2)
            delta_dof_pos = k_val * jacobian_pinv @ delta_pose.unsqueeze(-1)
            delta_dof_pos = delta_dof_pos.squeeze(-1)

        return delta_dof_pos

    def get_pose_error(self, fingertip_midpoint_pos,
                       fingertip_midpoint_quat,
                       ctrl_target_fingertip_midpoint_pos,
                       ctrl_target_fingertip_midpoint_quat,
                       jacobian_type,
                       rot_error_type):
        """Compute task-space error between target Franka fingertip pose and current pose."""
        # Reference: https://ethz.ch/content/dam/ethz/special-interest/mavt/robotics-n-intelligent-systems/rsl-dam/documents/RobotDynamics2018/RD_HS2018script.pdf

        # Compute pos error
        # print(ctrl_target_fingertip_midpoint_pos.shape)
        # print(fingertip_midpoint_pos.shape)
        # quit()
        pos_error = ctrl_target_fingertip_midpoint_pos - fingertip_midpoint_pos

        # Compute rot error
        if jacobian_type == 'geometric':  # See example 2.9.8; note use of J_g and transformation between rotation vectors
            # Compute quat error (i.e., difference quat)
            # Reference: https://personal.utdallas.edu/~sxb027100/dock/quat.html
            fingertip_midpoint_quat_norm = torch_utils.quat_mul(fingertip_midpoint_quat,
                                                                torch_utils.quat_conjugate(fingertip_midpoint_quat))[:,
                                           0]  # scalar component
            fingertip_midpoint_quat_inv = torch_utils.quat_conjugate(
                fingertip_midpoint_quat) / fingertip_midpoint_quat_norm.unsqueeze(-1)
            quat_error = torch_utils.quat_mul(ctrl_target_fingertip_midpoint_quat, fingertip_midpoint_quat_inv)

            # Convert to axis-angle error
            axis_angle_error = axis_angle_from_quat(quat_error)

        elif jacobian_type == 'analytic':  # See example 2.9.7; note use of J_a and difference of rotation vectors
            # Compute axis-angle error
            axis_angle_error = axis_angle_from_quat(ctrl_target_fingertip_midpoint_quat) \
                               - axis_angle_from_quat(fingertip_midpoint_quat)

        if rot_error_type == 'quat':
            return pos_error, quat_error
        elif rot_error_type == 'axis_angle':
            return pos_error, axis_angle_error

    def reset_idx(self, env_ids):
        indices = env_ids.to(dtype=torch.int32)
        num_indices = len(indices)

        # reset franka
        pos = tensor_clamp(
            self.franka_default_dof_pos.unsqueeze(0)
            + 0.25 * (torch.rand((len(env_ids), self.num_franka_dofs), device=self._device) - 0.5),
            self.franka_dof_lower_limits,
            self.franka_dof_upper_limits,
        )
        dof_pos = torch.zeros((num_indices, self._frankas.num_dof), device=self._device)
        dof_vel = torch.zeros((num_indices, self._frankas.num_dof), device=self._device)
        dof_pos[:, :] = pos
        self.franka_dof_targets[env_ids, :] = pos
        self.franka_dof_pos[env_ids, :] = pos

        self._cubeA.set_world_poses(self.default_cubeA_pos[env_ids],self.default_cubeA_rot[env_ids], indices=indices)

        self._cubeB.set_world_poses(self.default_cubeB_pos[env_ids],self.default_cubeB_rot[env_ids], indices=indices)

        self._frankas.set_joint_position_targets(self.franka_dof_targets[env_ids], indices=indices)
        self._frankas.set_joint_positions(dof_pos, indices=indices)
        self._frankas.set_joint_velocities(dof_vel, indices=indices)

        self.dof_pos = self._frankas.get_joint_positions(clone=False)
        self.dof_vel = self._frankas.get_joint_velocities(clone=False)

        # jacobian shape: [4, 11, 6, 9] (The root does not have a jacobian)
        self.franka_jacobian = self._frankas._physics_view.get_jacobians()
        self.franka_mass_matrix = self._frankas.get_mass_matrices(clone=False)

        self.arm_dof_pos = self.dof_pos[:, 0:7]
        self.arm_mass_matrix = self.franka_mass_matrix[:, 0:7, 0:7]  # for Franka arm (not gripper)

        self.hand_pos, self.hand_quat = self._frankas._hands.get_world_poses(clone=False)
        # subtract from the env positions to obtain the positions relative to the envs
        self.hand_pos -= self._env_pos
        hand_velocities = self._frankas._hands.get_velocities(clone=False)
        self.hand_linvel = hand_velocities[:, 0:3]
        self.hand_angvel = hand_velocities[:, 3:6]

        self.left_finger_pos, self.left_finger_quat = self._frankas._lfingers.get_world_poses(clone=False)
        self.left_finger_pos -= self.env_pos
        left_finger_velocities = self._frankas._lfingers.get_velocities(clone=False)
        self.left_finger_linvel = left_finger_velocities[:, 0:3]
        self.left_finger_angvel = left_finger_velocities[:, 3:6]
        self.left_finger_jacobian = self.franka_jacobian[:, 8, 0:6, 0:7]

        self.right_finger_pos, self.right_finger_quat = self._frankas._rfingers.get_world_poses(clone=False)
        self.right_finger_pos -= self.env_pos
        right_finger_velocities = self._frankas._rfingers.get_velocities(clone=False)
        self.right_finger_linvel = right_finger_velocities[:, 0:3]
        self.right_finger_angvel = right_finger_velocities[:, 3:6]
        self.right_finger_jacobian = self.franka_jacobian[:, 9, 0:6, 0:7]

        # Cannot acquire proper net contact force in Isaac Sim at this moment
        self.left_finger_force = torch.zeros((self.num_envs, 3), device=self.device)
        self.right_finger_force = torch.zeros((self.num_envs, 3), device=self.device)

        self.gripper_dof_pos = self.dof_pos[:, 7:9]

        # self.fingertip_centered_pos, self.fingertip_centered_quat = self._frankas._fingertip_centered.get_world_poses(
        #     clone=False)
        # self.fingertip_centered_pos -= self.env_pos
        # fingertip_centered_velocities = self._frankas._fingertip_centered.get_velocities(clone=False)
        # self.fingertip_centered_linvel = fingertip_centered_velocities[:, 0:3]
        # self.fingertip_centered_angvel = fingertip_centered_velocities[:, 3:6]
        # self.fingertip_centered_jacobian = self.franka_jacobian[:, 10, 0:6, 0:7]

        self.finger_midpoint_pos = (self.left_finger_pos + self.right_finger_pos) / 2
        self.fingertip_midpoint_pos = self.translate_along_local_z(pos=self.finger_midpoint_pos,
                                                                 quat=self.hand_quat,
                                                                 offset=self.franka_finger_length,
                                                                 device=self.device)
        self.fingertip_midpoint_quat = (self.left_finger_quat + self.right_finger_quat) / 2

        # TODO: Add relative velocity term (see https://dynamicsmotioncontrol487379916.files.wordpress.com/2020/11/21-me258pointmovingrigidbody.pdf)
        # self.fingertip_midpoint_linvel = self.fingertip_centered_linvel + torch.cross(self.fingertip_centered_angvel,
        #                                                                               (
        #                                                                                           self.fingertip_midpoint_pos - self.fingertip_centered_pos),
        #                                                                               dim=1)

        # From sum of angular velocities (https://physics.stackexchange.com/questions/547698/understanding-addition-of-angular-velocity),
        # angular velocity of midpoint w.r.t. world is equal to sum of
        # angular velocity of midpoint w.r.t. hand and angular velocity of hand w.r.t. world.
        # Midpoint is in sliding contact (i.e., linear relative motion) with hand; angular velocity of midpoint w.r.t. hand is zero.
        # Thus, angular velocity of midpoint w.r.t. world is equal to angular velocity of hand w.r.t. world.
        # self.fingertip_midpoint_angvel = self.fingertip_centered_angvel  # always equal

        self.fingertip_midpoint_jacobian = (self.left_finger_jacobian + self.right_finger_jacobian) * 0.5

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def translate_along_local_z(self, pos, quat, offset, device):
        """Translate global body position along local Z-axis and express in global coordinates."""

        num_vecs = pos.shape[0]
        offset_vec = offset * torch.tensor([0.0, 0.0, 1.0], device=device).repeat((num_vecs, 1))
        _, translated_pos = torch_utils.tf_combine(q1=quat,
                                                   t1=pos,
                                                   q2=torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(
                                                       (num_vecs, 1)),
                                                   t2=offset_vec)

        return translated_pos

    def post_reset(self):

        self.num_franka_dofs = self._frankas.num_dof
        self.franka_dof_pos = torch.zeros((self.num_envs, self.num_franka_dofs), device=self._device)
        dof_limits = self._frankas.get_dof_limits()
        self.franka_dof_lower_limits = dof_limits[0, :, 0].to(device=self._device)
        self.franka_dof_upper_limits = dof_limits[0, :, 1].to(device=self._device)
        self.franka_dof_speed_scales = torch.ones_like(self.franka_dof_lower_limits)
        self.franka_dof_speed_scales[self._frankas.gripper_indices] = 0.1
        self.franka_dof_targets = torch.zeros(
            (self._num_envs, self.num_franka_dofs), dtype=torch.float, device=self._device
        )

        self.default_cubeA_pos, self.default_cubeA_rot = self._frankas.get_world_poses(clone=False)
        self.default_cubeB_pos, self.default_cubeB_rot = self._frankas.get_world_poses(clone=False)

        self.default_cubeA_pos += torch.tensor([-0.55, 0.0, 0.0]).cuda()
        self.default_cubeB_pos += torch.tensor([0.5, 0.0, 0.0]).cuda()
        self.env_pos = self._env_pos
        # randomize all envs
        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def calculate_metrics(self) -> None:
        self.rew_buf[:] = self.compute_franka_reward(
            self.reset_buf, self.progress_buf, self.actions, self.franka_lfinger_pos, self.franka_rfinger_pos, self.cubeA_pos, self.cubeA_to_cubeB_pos,
            self._num_envs, self.dist_reward_scale, self.rot_reward_scale, self.around_handle_reward_scale,
            self.open_reward_scale, self.finger_dist_reward_scale, self.action_penalty_scale, self.distX_offset, self._max_episode_length,
            self.franka_dof_pos,
            self.finger_close_reward_scale)

    def is_done(self) -> None:
        # reset if drawer is open or max length reached
        # self.reset_buf = torch.where(torch.norm(self.cubeA_pos) > 5.0, torch.ones_like(self.reset_buf), self.reset_buf)
        # self.reset_buf = torch.where(self.progress_buf >= self._max_episode_length - 1, torch.ones_like(self.reset_buf),
        #                              self.reset_buf)
        self.reset_buf = torch.where((self.progress_buf >= self._max_episode_length - 1) | (self.stack_reward > 0),
                                     torch.ones_like(self.reset_buf), self.reset_buf)

    def compute_franka_reward(
            self, reset_buf, progress_buf, actions, franka_lfinger_pos, franka_rfinger_pos, cubeA_pos, cubeA_to_cubeB_pos,
            num_envs, dist_reward_scale, rot_reward_scale, around_handle_reward_scale, open_reward_scale,
            finger_dist_reward_scale, action_penalty_scale, distX_offset, max_episode_length, joint_positions,
            finger_close_reward_scale
    ):
        # distance from hand to the drawer
        # d = torch.norm((franka_lfinger_pos + franka_rfinger_pos)/2.0 - cubeA_pos, p=2, dim=-1)
        # dist_reward = 1.0 / (1.0 + d ** 2)
        # dist_reward *= dist_reward
        # dist_reward = torch.where(d <= 0.02, dist_reward * 2, dist_reward)

        target_height = 0.515 + 0.515 / 2.0

        d = torch.norm((franka_lfinger_pos + franka_rfinger_pos)/2.0 - cubeA_pos, dim=-1)
        d_lf = torch.norm(cubeA_pos - franka_lfinger_pos, dim=-1)
        d_rf = torch.norm(cubeA_pos - franka_rfinger_pos, dim=-1)
        dist_reward = 1 - torch.tanh(10.0 * (d + d_lf + d_rf) / 3)

        cubeA_height = cubeA_pos[:, 2] - 0.0
        cubeA_lifted = (cubeA_height - 0.515) > 0.04
        lift_reward = cubeA_lifted

        offset = torch.zeros_like(cubeA_to_cubeB_pos)
        offset[:, 2] = (0.515 + 0.515) / 2.0
        d_ab = torch.norm(cubeA_to_cubeB_pos + offset, dim=-1)
        align_reward = (1 - torch.tanh(10.0 * d_ab)) * cubeA_lifted

        # Dist reward is maximum of dist and align reward
        dist_reward = torch.max(dist_reward, align_reward)

        # final reward for stacking successfully (only if cubeA is close to target height and corresponding location, and gripper is not grasping)
        cubeA_align_cubeB = (torch.norm(cubeA_to_cubeB_pos[:, :2], dim=-1) < 0.02)
        cubeA_on_cubeB = torch.abs(cubeA_height - target_height) < 0.02
        gripper_away_from_cubeA = (d > 0.04)
        self.stack_reward = cubeA_align_cubeB & cubeA_on_cubeB & gripper_away_from_cubeA

        rewards = torch.where(
            self.stack_reward,
            16.0 * self.stack_reward,
            0.1 * dist_reward + 1.5 * lift_reward +
            2.0 * align_reward,
        )

        return rewards


