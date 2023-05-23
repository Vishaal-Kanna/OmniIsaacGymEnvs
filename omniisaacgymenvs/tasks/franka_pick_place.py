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
from omniisaacgymenvs.robots.articulations.factory_franka import FactoryFranka
from omniisaacgymenvs.robots.articulations.views.factory_franka_view import FactoryFrankaView
from omniisaacgymenvs.robots.articulations.views.cabinet_view import CabinetView
from omniisaacgymenvs.tasks.factory.factory_control import *

from omni.isaac.core.objects import DynamicCuboid, FixedCuboid
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

def axisangle2quat(vec, eps=1e-6):
    """
    Converts scaled axis-angle to quat.
    Args:
        vec (tensor): (..., 3) tensor where final dim is (ax,ay,az) axis-angle exponential coordinates
        eps (float): Stability value below which small values will be mapped to 0
    Returns:
        tensor: (..., 4) tensor where final dim is (x,y,z,w) vec4 float quaternion
    """
    # type: (Tensor, float) -> Tensor
    # store input shape and reshape
    input_shape = vec.shape[:-1]
    vec = vec.reshape(-1, 3)

    # Grab angle
    angle = torch.norm(vec, dim=-1, keepdim=True)

    # Create return array
    quat = torch.zeros(torch.prod(torch.tensor(input_shape)), 4, device=vec.device)
    quat[:, 3] = 1.0

    # Grab indexes where angle is not zero an convert the input to its quaternion form
    idx = angle.reshape(-1) > eps
    quat[idx, :] = torch.cat([
        vec[idx, :] * torch.sin(angle[idx, :] / 2.0) / angle[idx, :],
        torch.cos(angle[idx, :] / 2.0)
    ], dim=-1)

    # Reshape and return output
    quat = quat.reshape(list(input_shape) + [4, ])

    quat_wxyz = torch.zeros(torch.prod(torch.tensor(input_shape)), 4, device=vec.device)
    quat_wxyz[:, 0] = quat[:, 3]
    quat_wxyz[:, 1:] = quat[:, :3]

    return quat_wxyz


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

        self._max_episode_length = 500 #self._task_cfg["env"]["episodeLength"]

        self.action_scale = 1.0
        self.start_position_noise = 0.25
        self.start_rotation_noise = 0.785
        self.franka_position_noise = 0.0
        self.franka_rotation_noise = 0.0
        self.franka_dof_noise = 0.25

        self.aggregate_mode = 3

        self.reward_settings = {
            "r_dist_scale": 0.1,
            "r_lift_scale": 1.5,
            "r_align_scale": 2.0,
            "r_stack_scale": 16.0,
        }

        self.control_type = 'osc'
        self._num_observations= 19 if self.control_type == "osc" else 26
        self._num_actions = 7 if self.control_type == "osc" else 8

        # Values to be filled in at runtime
        self.states = {}                        # will be dict filled with relevant states to use for reward calculation
        self.handles = {}                       # will be dict mapping names to relevant sim handles
        self.num_dofs = None                    # Total number of DOFs per env
        self.actions = None                     # Current actions to be deployed
        self._init_cubeA_state = None           # Initial state of cubeA for the current env
        self._init_cubeB_state = None           # Initial state of cubeB for the current env
        self._cubeA_state = None                # Current state of cubeA for the current env
        self._cubeB_state = None                # Current state of cubeB for the current env
        self._cubeA_id = None                   # Actor ID corresponding to cubeA for a given env
        self._cubeB_id = None                   # Actor ID corresponding to cubeB for a given env

        # Tensor placeholders
        self._root_state = None             # State of root body        (n_envs, 13)
        self._dof_state = None  # State of all joints       (n_envs, n_dof)
        self._q = None  # Joint positions           (n_envs, n_dof)
        self._qd = None                     # Joint velocities          (n_envs, n_dof)
        self._rigid_body_state = None  # State of all rigid bodies             (n_envs, n_bodies, 13)
        self._contact_forces = None     # Contact forces in sim
        self._eef_state = None  # end effector state (at grasping point)
        self._eef_lf_state = None  # end effector state (at left fingertip)
        self._eef_rf_state = None  # end effector state (at left fingertip)
        self._j_eef = None  # Jacobian for end effector
        self._mm = None  # Mass matrix
        self._arm_control = None  # Tensor buffer for controlling arm
        self._gripper_control = None  # Tensor buffer for controlling gripper
        self._pos_control = None            # Position actions
        self._effort_control = None         # Torque actions
        self._franka_effort_limits = None        # Actuator effort limits for franka
        self._global_indices = None         # Unique indices corresponding to all envs in flattened array

        self.up_axis = "z"
        self.up_axis_idx = 2

        self.env_spacing = 1.5  # 0.5
        self.franka_depth = 0.5
        self.table_height = 0.4
        self.franka_friction = 1.0
        self.table_friction = 0.3

        self.table_depth = 0.6  # depth of table
        self.table_width = 1.0  # width of table

        RLTask.__init__(self, name, env)

        self.franka_default_dof_pos = torch.tensor(
            [0, 0.1963, 0, -2.6180, 0, 2.9416, 0.7854, 0.035, 0.035], device=self.device
        )

        # OSC Gains
        self.kp = torch.tensor([150.] * 6, device=self.device)
        self.kd = 2 * torch.sqrt(self.kp)
        self.kp_null = torch.tensor([10.] * 7, device=self.device)
        self.kd_null = 2 * torch.sqrt(self.kp_null)
        # self.cmd_limit = None                   # filled in later

        # Set control limits
        self.cmd_limit = torch.tensor([0.1, 0.1, 0.1, 0.5, 0.5, 0.5], device=self.device).unsqueeze(0) if \
            self.control_type == "osc" else self._franka_effort_limits[:7].unsqueeze(0)

        return

    def set_up_scene(self, scene) -> None:

        self.import_franka_assets()
        super().set_up_scene(scene)

        self.frankas = FrankaView(prim_paths_expr="/World/envs/.*/franka", name="franka_view")

        scene.add(self.frankas)
        scene.add(self.frankas._hands)
        scene.add(self.frankas._lfingers)
        scene.add(self.frankas._rfingers)

        self._import_env_assets()
        self.cubeA = RigidPrimView(prim_paths_expr="/World/envs/.*/cubeA", name="cubeA_view")
        self.cubeB = RigidPrimView(prim_paths_expr="/World/envs/.*/cubeB", name="cubeB_view")

        # scene.add(self.frankas._fingertip_centered)

        scene.add(self.cubeA)
        scene.add(self.cubeB)

        # cube1 = DynamicCuboid(
        #         name="cube1",
        #         position=np.array([-1.0, 0.0, 0.0]) ,
        #         orientation=np.array([1, 0, 0, 0]),
        #         prim_path=self.default_zero_env_path + "/cube1",
        #         scale=*([0.5] * 3),
        #         size=1.0,
        #         color=np.array([0.6, 0.1, 0.0]),
        #     )
        # cube2 = DynamicCuboid(
        #         name="cube2",
        #         position=np.array([1.0, 0.0, 0.0]),
        #         orientation=np.array([1, 0, 0, 0]),
        #         prim_path=self.default_zero_env_path + "/cube2",
        #         scale=*([0.7] * 3),
        #         size=1.0,
        #         color=np.array([0.0, 0.4, 0.1]),
        #     )

        # franka_dof_props_pos = self.frankas.get_dof_limits()
        # self._franka_effort_limits = []
        # self.franka_dof_lower_limits = []
        # self.franka_dof_upper_limits = []
        #
        # franka_dof_stiffness = torch.tensor([0, 0, 0, 0, 0, 0, 0, 5000., 5000.], dtype=torch.float, device=self.device)
        # franka_dof_damping = torch.tensor([0, 0, 0, 0, 0, 0, 0, 1.0e2, 1.0e2], dtype=torch.float, device=self.device)
        #
        # self.num_franka_dofs = self.frankas.num_dof
        #
        # for i in range(self.num_franka_dofs):
        #     franka_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS if i > 6 else gymapi.DOF_MODE_EFFORT
        #     franka_dof_props['stiffness'][i] = franka_dof_stiffness[i]
        #     franka_dof_props['damping'][i] = franka_dof_damping[i]
        #
        #     self.franka_dof_lower_limits.append(franka_dof_props_pos[:, i, 0])
        #     self.franka_dof_upper_limits.append(franka_dof_props_pos[:, i, 1])
        #     self._franka_effort_limits.append(franka_dof_props_effort[:, i])
        #
        # self.franka_dof_lower_limits = torch.tensor(self.franka_dof_lower_limits, device=self.device)
        # self.franka_dof_upper_limits = torch.tensor(self.franka_dof_upper_limits, device=self.device)
        # self._franka_effort_limits = torch.tensor(self._franka_effort_limits, device=self.device)
        # self.franka_dof_speed_scales = torch.ones_like(self.franka_dof_lower_limits)
        # self.franka_dof_speed_scales[[7, 8]] = 0.1
        # franka_dof_props['effort'][7] = 200
        # franka_dof_props['effort'][8] = 200


        self.init_data()

        return

    def init_data(self):

        self._init_cubeA_state = torch.zeros(self._num_envs, 13, device=self.device)
        self._init_cubeB_state = torch.zeros(self._num_envs, 13, device=self.device)

        self._eef_state = torch.zeros(self._num_envs, 13, device=self.device)
        self._eef_lf_state = torch.zeros(self._num_envs, 13, device=self.device)
        self._eef_rf_state = torch.zeros(self._num_envs, 13, device=self.device)
        self._cubeA_state = torch.zeros(self._num_envs, 13, device=self.device)
        self._cubeB_state = torch.zeros(self._num_envs, 13, device=self.device)

        self.num_dofs = 9
        # self._q = torch.zeros(self._num_envs, self.num_dofs, device=self.device)
        # self._qd = torch.zeros(self._num_envs, self.num_dofs, device=self.device)

        # self.num_dofs = self.frankas.num_dof
        # self._refresh()
        #
        # jacobian = self.frankas.get_jacobians()
        # self._j_eef = jacobian[:,-3, :, :7]
        # mm = self.frankas.get_mass_matrices()
        # self._mm = mm[:, :7, :7]

        # Initialize states
        self.states.update({
            "cubeA_size": torch.ones_like(self._eef_state[:, 0]) * 0.05,
            "cubeB_size": torch.ones_like(self._eef_state[:, 0]) * 0.07,
        })

        self._pos_control = torch.zeros((self._num_envs, self.num_dofs), dtype=torch.float, device=self._device)
        self._effort_control = torch.zeros_like(self._pos_control)

        # Initialize control
        self._arm_control = self._effort_control[:, :7]
        self._gripper_control = self._pos_control[:, 7:9]

        self.default_cubeA_pos = np.array([-1.0, 0.0, 0.0])
        self.default_cubeA_rot = np.array([1, 0, 0, 0])
        self.default_cubeB_pos = np.array([1.0, 0.0, 0.0])
        self.default_cubeB_rot = np.array([1, 0, 0, 0])

        self._global_indices = torch.arange(self._num_envs * 5, dtype=torch.int32, device=self._device).view(self._num_envs, -1)



    def get_franka(self):
        franka = Franka(prim_path=self.default_zero_env_path + "/franka", name="franka")
        self._sim_config.apply_articulation_settings("franka", get_prim_at_path(franka.prim_path),
                                                     self._sim_config.parse_actor_config("franka"))

    def import_franka_assets(self):
        """Set Franka and table asset options. Import assets."""
        self._stage = get_current_stage()

        table_pos = np.array([0.0, 0.0, 0.0])
        table_orientation = np.array([1.0, 0.0, 0.0, 0.0])
        table_thickness = 0.05
        table_stand_height = 0.1
        table_stand_pos = np.array([-0.5, 0.0, 0.0 + table_thickness / 2 + table_stand_height / 2])

        franka_translation = np.array([-0.45, 0.0, 0.0 + table_thickness / 2 + table_stand_height])
        franka_orientation = np.array([0.0, 0.0, 0.0, 1.0])

        self._table_surface_pos = table_pos + np.array([0, 0, table_thickness / 2])
        self.reward_settings["table_height"] = self._table_surface_pos[2]

        self.get_franka()

        franka = FactoryFranka(
            prim_path=self.default_zero_env_path + "/franka",
            name="franka",
            translation=franka_translation,
            orientation=franka_orientation,
        )
        self._sim_config.apply_articulation_settings(
            "franka",
            get_prim_at_path(franka.prim_path),
            self._sim_config.parse_actor_config("franka")
        )

        # for link_prim in franka.prim.GetChildren():
        #     if link_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
        #         rb = PhysxSchema.PhysxRigidBodyAPI.Get(self._stage, link_prim.GetPrimPath())
        #         rb.GetDisableGravityAttr().Set(True)
        #         rb.GetRetainAccelerationsAttr().Set(False)
        #         if self.cfg_base.sim.add_damping:
        #             rb.GetLinearDampingAttr().Set(1.0)  # default = 0.0; increased to improve stability
        #             rb.GetMaxLinearVelocityAttr().Set(1.0)  # default = 1000.0; reduced to prevent CUDA errors
        #             rb.GetAngularDampingAttr().Set(5.0)  # default = 0.5; increased to improve stability
        #             rb.GetMaxAngularVelocityAttr().Set(
        #                 2 / math.pi * 180)  # default = 64.0; reduced to prevent CUDA errors
        #         else:
        #             rb.GetLinearDampingAttr().Set(0.0)
        #             rb.GetMaxLinearVelocityAttr().Set(1000.0)
        #             rb.GetAngularDampingAttr().Set(0.5)
        #             rb.GetMaxAngularVelocityAttr().Set(64 / math.pi * 180)

        table = FixedCuboid(
            prim_path=self.default_zero_env_path + "/table",
            name="table",
            translation=table_pos,
            orientation=table_orientation,
            scale=np.array([1.2, 1.2, table_thickness]),
            size=1.0,
            color=np.array([0, 0, 0]),
        )

        table_stand = FixedCuboid(
            prim_path=self.default_zero_env_path + "/table_stand",
            name="table_stand",
            translation=table_stand_pos,
            orientation=table_orientation,
            scale=np.array([0.2, 0.2, table_stand_height]),
            size=1.0,
            color=np.array([0, 0, 0]),
        )

    def _import_env_assets(self):
        """Set nut and bolt asset options. Import assets."""

        for i in range(0, self._num_envs):

            cube1 = DynamicCuboid(
                name="cube1",
                position=np.array([-1.0, 0.0, 0.0]),
                orientation=np.array([0, 0, 0, 1]),
                prim_path=f"/World/envs/env_{i}" + "/cubeA",
                scale=np.array([0.05, 0.05, 0.05]),
                size=1.0,
                color=np.array([0.6, 0.1, 0.0]),
            )
            cube2 = DynamicCuboid(
                name="cube2",
                position=np.array([1.0, 0.0, 0.0]),
                orientation=np.array([0, 0, 0, 1]),
                prim_path=f"/World/envs/env_{i}" + "/cubeB",
                scale=np.array([0.07, 0.07, 0.07]),
                size=1.0,
                color=np.array([0.0, 0.4, 0.1]),
            )

    def _update_states(self):
        self.states.update({
            # Franka
            "q": self._q[:, :],
            "q_gripper": self._q[:, -2:],
            "eef_pos": self._eef_state[:, :3],
            "eef_quat": self._eef_state[:, 3:7],
            "eef_vel": self._eef_state[:, 7:],
            "eef_lf_pos": self._eef_lf_state[:, :3],
            "eef_rf_pos": self._eef_rf_state[:, :3],
            # Cubes
            "cubeA_quat": self._cubeA_state[:, 3:7],
            "cubeA_pos": self._cubeA_state[:, :3],
            "cubeA_pos_relative": self._cubeA_state[:, :3] - self._eef_state[:, :3],
            "cubeB_quat": self._cubeB_state[:, 3:7],
            "cubeB_pos": self._cubeB_state[:, :3],
            "cubeA_to_cubeB_pos": self._cubeB_state[:, :3] - self._cubeA_state[:, :3],
        })

    def _refresh(self):
        # Refresh states
        self._q = self.frankas.get_joint_positions(clone=False)
        self._qd = self.frankas.get_joint_velocities(clone=False)

        if self._num_envs == 1:
            hand_pos, hand_rot = self.frankas._hands.get_world_poses(clone=False)
            self._eef_state[:, 7:13] = self.frankas._hands.get_velocities(clone=False)

            self._eef_state[:, :3] = hand_pos
            self._eef_state[:, 3:7] = hand_rot

            franka_lfinger_pos, franka_lfinger_rot = self.frankas._lfingers.get_world_poses(clone=False)
            franka_rfinger_pos, franka_rfinger_rot = self.frankas._rfingers.get_world_poses(clone=False)
            self._eef_lf_state[:, :3] = franka_lfinger_pos
            self._eef_lf_state[:, 3:7] = franka_lfinger_rot
            self._eef_rf_state[:, :3] = franka_rfinger_pos
            self._eef_rf_state[:, 3:7] = franka_rfinger_rot

            cubeA_pos, cubeA_rot = self.cubeA.get_world_poses(clone=False)
            self._cubeA_state[:, :3] = cubeA_pos
            self._cubeA_state[:, 3:7] = cubeA_rot
            self._cubeA_state[:, 7:13] = self.cubeA.get_velocities(clone=False)

            cubeB_pos, cubeB_rot = self.cubeB.get_world_poses(clone=False)
            self._cubeB_state[:, :3] = cubeB_pos
            self._cubeB_state[:, 3:7] = cubeB_rot
            self._cubeB_state[:, 7:13] = self.cubeB.get_velocities(clone=False)


        else:
            base_pos, _ = self.frankas.get_world_poses(clone=False)

            hand_pos, hand_rot = self.frankas._hands.get_world_poses(clone=False)
            self._eef_state[:, 7:13] = self.frankas._hands.get_velocities(clone=False)

            self._eef_state[:, :3] = hand_pos - base_pos
            self._eef_state[:, 3:7] = hand_rot

            franka_lfinger_pos, franka_lfinger_rot = self.frankas._lfingers.get_world_poses(clone=False)
            franka_rfinger_pos, franka_rfinger_rot = self.frankas._rfingers.get_world_poses(clone=False)
            self._eef_lf_state[:, :3] = franka_lfinger_pos - base_pos
            self._eef_lf_state[:, 3:7] = franka_lfinger_rot
            self._eef_rf_state[:, :3] = franka_rfinger_pos - base_pos
            self._eef_rf_state[:, 3:7] = franka_rfinger_rot


            cubeA_pos, cubeA_rot = self.cubeA.get_world_poses(clone=False)
            self._cubeA_state[:, :3] = cubeA_pos - base_pos
            self._cubeA_state[:, 3:7] = cubeA_rot
            self._cubeA_state[:, 7:13] = self.cubeA.get_velocities(clone=False)

            cubeB_pos, cubeB_rot = self.cubeB.get_world_poses(clone=False)
            self._cubeB_state[:, :3] = cubeB_pos - base_pos
            self._cubeB_state[:, 3:7] = cubeB_rot
            self._cubeB_state[:, 7:13] = self.cubeB.get_velocities(clone=False)

        self._update_states()

    def compute_observations(self) -> dict:

        self._refresh()

        obs = ["cubeA_quat", "cubeA_pos", "cubeA_to_cubeB_pos", "eef_pos", "eef_quat"]
        obs += ["q_gripper"] if self.control_type == "osc" else ["q"]
        self.obs_buf = torch.cat([self.states[ob] for ob in obs], dim=-1)

        maxs = {ob: torch.max(self.states[ob]).item() for ob in obs}

        return self.obs_buf

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)

        # Split arm and gripper command
        u_arm, u_gripper = self.actions[:, :-1], self.actions[:, -1]

        # print(u_arm, u_gripper)
        # print(self.cmd_limit, self.action_scale)

        # Control arm (scale value first)
        u_arm = u_arm * self.cmd_limit / self.action_scale
        if self.control_type == "osc":
            u_arm = self._compute_osc_torques(dpose=u_arm)
        self._arm_control[:, :] = u_arm

        if self._num_envs == -1:
            self._effort_control[:,:-2] = self._arm_control

            u_fingers = torch.zeros_like(self._gripper_control)
            u_fingers[:, 0] = torch.where(u_gripper >= 0.0, self._franka_effort_limits[-2].item(),
                                          -self._franka_effort_limits[-2].item())
            u_fingers[:, 1] = torch.where(u_gripper >= 0.0, self._franka_effort_limits[-1].item(),
                                          -self._franka_effort_limits[-1].item())
            self._effort_control[:, -2:] = u_fingers
            self.frankas.set_joint_efforts(self._effort_control)

        # Control gripper
        else:
            u_fingers = torch.zeros_like(self._gripper_control)
            u_fingers[:, 0] = torch.where(u_gripper >= 0.0, self.franka_dof_upper_limits[-2].item(),
                                          self.franka_dof_lower_limits[-2].item())
            u_fingers[:, 1] = torch.where(u_gripper >= 0.0, self.franka_dof_upper_limits[-1].item(),
                                          self.franka_dof_lower_limits[-1].item())
            # Write gripper command to appropriate tensor buffer
            self._gripper_control[:, :] = u_fingers
            # self._pos_control[:,-2:] = self._gripper_control

            # Deploy actions
            self.frankas.set_joint_position_targets(positions=self._pos_control)
            self.frankas.set_joint_efforts(self._effort_control)

    def reset_idx(self, env_ids):
        env_ids_int32 = env_ids.to(dtype=torch.long)

        self.num_dofs = self.frankas.num_dof

        self._q = torch.zeros(self._num_envs, self.num_dofs, device=self.device)
        self._qd = torch.zeros(self._num_envs, self.num_dofs, device=self.device)

        self._refresh()

        jacobian = self.frankas.get_jacobians()
        # print(self.frankas.get_dof_index('panda_joint7'))
        # quit()
        self._j_eef = jacobian[:,self.frankas.get_dof_index('panda_joint7')+1, :, :7]
        mm = self.frankas.get_mass_matrices()
        self._mm = mm[:, :7, :7]

        # franka_dof_props_pos = self.frankas.get_dof_limits()
        # self._franka_effort_limits = []
        # self.franka_dof_lower_limits = []
        # self.franka_dof_upper_limits = []
        #
        # franka_dof_stiffness = torch.tensor([0, 0, 0, 0, 0, 0, 0, 5000., 5000.], dtype=torch.float, device=self.device)
        # franka_dof_damping = torch.tensor([0, 0, 0, 0, 0, 0, 0, 1.0e2, 1.0e2], dtype=torch.float, device=self.device)
        #
        # self.num_franka_dofs = self.frankas.num_dof
        #
        # for i in range(self.num_franka_dofs):
        #     franka_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS if i > 6 else gymapi.DOF_MODE_EFFORT
        #     franka_dof_props['stiffness'][i] = franka_dof_stiffness[i]
        #     franka_dof_props['damping'][i] = franka_dof_damping[i]
        #
        #     self.franka_dof_lower_limits.append(franka_dof_props_pos[:, i, 0])
        #     self.franka_dof_upper_limits.append(franka_dof_props_pos[:, i, 1])
        #     self._franka_effort_limits.append(franka_dof_props_effort[:, i])
        #
        # self.franka_dof_lower_limits = torch.tensor(self.franka_dof_lower_limits, device=self.device)
        # self.franka_dof_upper_limits = torch.tensor(self.franka_dof_upper_limits, device=self.device)
        # self._franka_effort_limits = torch.tensor(self._franka_effort_limits, device=self.device)
        # self.franka_dof_speed_scales = torch.ones_like(self.franka_dof_lower_limits)
        # self.franka_dof_speed_scales[[7, 8]] = 0.1
        # franka_dof_props['effort'][7] = 200
        # franka_dof_props['effort'][8] = 200

        # Reset cubes, sampling cube B first, then A
        # if not self._i:
        self._reset_init_cube_state(cube='B', env_ids=env_ids_int32, check_valid=False)
        self._reset_init_cube_state(cube='A', env_ids=env_ids_int32, check_valid=True)
        # self._i = True

        # Write these new init states to the sim states

        self._cubeA_state[env_ids_int32] = self._init_cubeA_state[env_ids_int32]
        self._cubeB_state[env_ids_int32] = self._init_cubeB_state[env_ids_int32]

        if self._num_envs == 1:
            self._cubeA_state[env_ids_int32, 0] = -1.0
            self._cubeA_state[env_ids_int32, 1] = 0.0
            self._cubeA_state[env_ids_int32, 2] = 0.0
            self._cubeA_state[env_ids_int32, 3] = 0.0
            self._cubeA_state[env_ids_int32, 4] = 0.0
            self._cubeA_state[env_ids_int32, 5] = 0.0
            self._cubeA_state[env_ids_int32, 6] = 1.0

            self._cubeB_state[env_ids_int32, 0] = 0.0
            self._cubeB_state[env_ids_int32, 1] = 0.2
            self._cubeB_state[env_ids_int32, 2] = 0.0 + 0.05 / 2
            self._cubeB_state[env_ids_int32, 3] = 0.0
            self._cubeB_state[env_ids_int32, 4] = 0.0
            self._cubeB_state[env_ids_int32, 5] = 0.0
            self._cubeB_state[env_ids_int32, 6] = 1.0


        # Reset agent
        reset_noise = torch.rand((len(env_ids_int32), 9), device=self.device)
        pos = tensor_clamp(
            self.franka_default_dof_pos.unsqueeze(0) +
            self.franka_dof_noise * 2.0 * (reset_noise - 0.5),
            self.franka_dof_lower_limits.unsqueeze(0), self.franka_dof_upper_limits)

        # Overwrite gripper init pos (no noise since these are always position controlled)
        pos[:, -2:] = self.franka_default_dof_pos[-2:]

        # Reset the internal obs accordingly
        self._q[env_ids_int32, :] = pos
        self._qd[env_ids_int32, :] = torch.zeros_like(self._qd[env_ids_int32])

        # Set any position control to the current position, and any vel / effort control to be 0
        # NOTE: Task takes care of actually propagating these controls in sim using the SimActions API
        self._pos_control[env_ids_int32, :] = pos
        self._effort_control[env_ids_int32, :] = torch.zeros_like(pos)

        # Deploy updates

        # multi_env_ids_int32 = self._global_indices[env_ids_int32, 0].flatten()
        self.frankas.set_joint_position_targets(positions=self._pos_control, indices=env_ids_int32)
        self.frankas.set_joint_efforts(self._effort_control, indices=env_ids_int32)

        if self._num_envs == 1:
            self.cubeA.set_world_poses(self._cubeA_state[:, 0:3], self._cubeA_state[:, 3:7], indices=env_ids_int32)
            self.cubeB.set_world_poses(self._cubeB_state[:, 0:3], self._cubeB_state[:, 3:7], indices=env_ids_int32)

        else:
            self.base_position, _ = self.frankas.get_world_poses(clone=False)
            self.base_position[:, 2] = 0.0
            self.cubeA.set_world_poses(self._cubeA_state[:,0:3] + self.base_position, self._cubeA_state[:,3:7], indices=env_ids_int32)
            self.cubeB.set_world_poses(self._cubeB_state[:,0:3] + self.base_position, self._cubeB_state[:,3:7], indices=env_ids_int32)

        self.progress_buf[env_ids_int32] = 0
        self.reset_buf[env_ids_int32] = 0

    def _reset_init_cube_state(self, cube, env_ids, check_valid=True):
        """
        Simple method to sample @cube's position based on self.startPositionNoise and self.startRotationNoise, and
        automaticlly reset the pose internally. Populates the appropriate self._init_cubeX_state
        If @check_valid is True, then this will also make sure that the sampled position is not in contact with the
        other cube.
        Args:
            cube(str): Which cube to sample location for. Either 'A' or 'B'
            env_ids (tensor or None): Specific environments to reset cube for
            check_valid (bool): Whether to make sure sampled position is collision-free with the other cube.
        """
        # If env_ids is None, we reset all the envs
        if env_ids is None:
            env_ids = torch.arange(start=0, end=self._num_envs, device=self.device, dtype=torch.long)

        # Initialize buffer to hold sampled values
        num_resets = len(env_ids)
        sampled_cube_state = torch.zeros(num_resets, 13, device=self.device)

        # Get correct references depending on which one was selected
        if cube.lower() == 'a':
            this_cube_state_all = self._init_cubeA_state
            other_cube_state = self._init_cubeB_state[env_ids, :]
            cube_heights = self.states["cubeA_size"]
        elif cube.lower() == 'b':
            this_cube_state_all = self._init_cubeB_state
            other_cube_state = self._init_cubeA_state[env_ids, :]
            cube_heights = self.states["cubeA_size"]
        else:
            raise ValueError(f"Invalid cube specified, options are 'A' and 'B'; got: {cube}")

        # Minimum cube distance for guarenteed collision-free sampling is the sum of each cube's effective radius
        min_dists = (self.states["cubeA_size"] + self.states["cubeB_size"])[env_ids] * np.sqrt(2) / 2.0

        # We scale the min dist by 2 so that the cubes aren't too close together
        min_dists = min_dists * 2.0

        # Sampling is "centered" around middle of table
        centered_cube_xy_state = torch.tensor(self._table_surface_pos[:2], device=self.device, dtype=torch.float32)

        # Set z value, which is fixed height
        if self._num_envs==1:
            sampled_cube_state[:, 2] = self._table_surface_pos[2] + cube_heights[env_ids] / 2
        else:
            sampled_cube_state[:, 2] = self._table_surface_pos[2] + cube_heights.squeeze(-1)[env_ids] / 2


        # Initialize rotation, which is no rotation (quat w = 1)
        sampled_cube_state[:, 6] = 1.0

        # If we're verifying valid sampling, we need to check and re-sample if any are not collision-free
        # We use a simple heuristic of checking based on cubes' radius to determine if a collision would occur
        if check_valid:
            success = False
            # Indexes corresponding to envs we're still actively sampling for
            active_idx = torch.arange(num_resets, device=self.device)
            num_active_idx = len(active_idx)
            for i in range(100):
                # Sample x y values
                sampled_cube_state[active_idx, :2] = centered_cube_xy_state + \
                                                     2.0 * self.start_position_noise * (
                                                             torch.rand_like(sampled_cube_state[active_idx, :2]) - 0.5)
                # Check if sampled values are valid
                cube_dist = torch.linalg.norm(sampled_cube_state[:, :2] - other_cube_state[:, :2], dim=-1)
                active_idx = torch.nonzero(cube_dist < min_dists, as_tuple=True)[0]
                num_active_idx = len(active_idx)
                # If active idx is empty, then all sampling is valid :D
                if num_active_idx == 0:
                    success = True
                    break
            # Make sure we succeeded at sampling
            assert success, "Sampling cube locations was unsuccessful! ):"
        else:
            # We just directly sample
            sampled_cube_state[:, :2] = centered_cube_xy_state.unsqueeze(0) + \
                                              2.0 * self.start_position_noise * (
                                                      torch.rand(num_resets, 2, device=self.device) - 0.5)

        # Sample rotation value
        if self.start_rotation_noise > 0:
            aa_rot = torch.zeros(num_resets, 3, device=self.device)
            aa_rot[:, 2] = 2.0 * self.start_rotation_noise * (torch.rand(num_resets, device=self.device) - 0.5)
            sampled_cube_state[:, 3:7] = quat_mul(axisangle2quat(aa_rot), sampled_cube_state[:, 3:7])

        # Lastly, set these sampled values as the new init state
        this_cube_state_all[env_ids, :] = sampled_cube_state

    def _compute_osc_torques(self, dpose):
        # Solve for Operational Space Control # Paper: khatib.stanford.edu/publications/pdfs/Khatib_1987_RA.pdf
        # Helpful resource: studywolf.wordpress.com/2013/09/17/robot-control-4-operation-space-control/
        q, qd = self._q[:, :7], self._qd[:, :7]
        mm_inv = torch.inverse(self._mm)
        m_eef_inv = self._j_eef @ mm_inv @ torch.transpose(self._j_eef, 1, 2)
        m_eef = torch.inverse(m_eef_inv)

        # Transform our cartesian action `dpose` into joint torques `u`
        u = torch.transpose(self._j_eef, 1, 2) @ m_eef @ (
                self.kp * dpose - self.kd * self.states["eef_vel"]).unsqueeze(-1)

        # Nullspace control torques `u_null` prevents large changes in joint configuration
        # They are added into the nullspace of OSC so that the end effector orientation remains constant
        # roboticsproceedings.org/rss07/p31.pdf
        j_eef_inv = m_eef @ self._j_eef @ mm_inv
        u_null = self.kd_null * -qd + self.kp_null * (
                (self.franka_default_dof_pos[:7] - q + np.pi) % (2 * np.pi) - np.pi)
        u_null[:, 7:] *= 0
        u_null = self._mm @ u_null.unsqueeze(-1)
        u += (torch.eye(7, device=self.device).unsqueeze(0) - torch.transpose(self._j_eef, 1, 2) @ j_eef_inv) @ u_null

        # Clip the values to be within valid effort range
        u = tensor_clamp(u.squeeze(-1),
                         -self._franka_effort_limits[:7].unsqueeze(0), self._franka_effort_limits[:7].unsqueeze(0))

        return u

    def post_reset(self):
        self.num_franka_dofs = self.frankas.num_dof
        # randomize all envs
        franka_dof_props_pos = self.frankas.get_dof_limits()
        franka_dof_props_effort = self.frankas.get_max_efforts()
        self._franka_effort_limits = []
        self.franka_dof_lower_limits = []
        self.franka_dof_upper_limits = []

        franka_dof_stiffness = torch.tensor([0, 0, 0, 0, 0, 0, 0, 5000., 5000.], dtype=torch.float, device=self.device)
        franka_dof_damping = torch.tensor([0, 0, 0, 0, 0, 0, 0, 1.0e2, 1.0e2], dtype=torch.float, device=self.device)

        self.num_franka_dofs = self.frankas.num_dof

        for i in range(self.num_franka_dofs):
            self.franka_dof_lower_limits.append(franka_dof_props_pos[0, i, 0])
            self.franka_dof_upper_limits.append(franka_dof_props_pos[0, i, 1])
            self._franka_effort_limits.append(franka_dof_props_effort[0, i])

            # franka_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS if i > 6 else gymapi.DOF_MODE_EFFORT
            # franka_dof_props['stiffness'][i] = franka_dof_stiffness[i]
            # franka_dof_props['damping'][i] = franka_dof_damping[i]



        self.franka_dof_lower_limits = torch.tensor(self.franka_dof_lower_limits, device=self.device)
        self.franka_dof_upper_limits = torch.tensor(self.franka_dof_upper_limits, device=self.device)
        self._franka_effort_limits = torch.tensor(self._franka_effort_limits, device=self.device)
        self.franka_dof_speed_scales = torch.ones_like(self.franka_dof_lower_limits)
        self.franka_dof_speed_scales[[7, 8]] = 0.1
        # franka_dof_props['effort'][7] = 200
        # franka_dof_props['effort'][8] = 200

        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward()

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def is_done(self) -> None:
        # reset if drawer is open or max length reached
        # self.reset_buf = torch.where(torch.norm(self.cubeA_pos) > 5.0, torch.ones_like(self.reset_buf), self.reset_buf)
        # self.reset_buf = torch.where(self.progress_buf >= self._max_episode_length - 1, torch.ones_like(self.reset_buf),
        #                              self.reset_buf)
        self.reset_buf = torch.where(self.progress_buf >= self._max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf)
        self.reset_buf = torch.where(self.stack_reward > 0, torch.ones_like(self.reset_buf), self.reset_buf)

    def compute_reward(self):
        self.rew_buf[:], self.reset_buf[:] = self.compute_franka_reward(self.reset_buf, self.progress_buf, self.states, self.reward_settings, self._max_episode_length)

    def compute_franka_reward(self, reset_buf, progress_buf, states, reward_settings, _max_episode_length):

        # Compute per-env physical parameters
        target_height = states["cubeB_size"] + states["cubeA_size"] / 2.0
        cubeA_size = states["cubeA_size"]
        cubeB_size = states["cubeB_size"]

        # distance from hand to the cubeA
        d = torch.norm(states["cubeA_pos_relative"], dim=-1)
        d_lf = torch.norm(states["cubeA_pos"] - states["eef_lf_pos"], dim=-1)
        d_rf = torch.norm(states["cubeA_pos"] - states["eef_rf_pos"], dim=-1)
        dist_reward = 1 - torch.tanh(10.0 * (d + d_lf + d_rf) / 3)

        # reward for lifting cubeA
        cubeA_height = states["cubeA_pos"][:, 2] - reward_settings["table_height"]
        cubeA_lifted = (cubeA_height - cubeA_size) > 0.04
        lift_reward = cubeA_lifted

        # how closely aligned cubeA is to cubeB (only provided if cubeA is lifted)
        offset = torch.zeros_like(states["cubeA_to_cubeB_pos"])
        offset[:, 2] = (cubeA_size + cubeB_size) / 2
        d_ab = torch.norm(states["cubeA_to_cubeB_pos"] + offset, dim=-1)
        align_reward = (1 - torch.tanh(10.0 * d_ab)) * cubeA_lifted

        # Dist reward is maximum of dist and align reward
        dist_reward = torch.max(dist_reward, align_reward)

        # final reward for stacking successfully (only if cubeA is close to target height and corresponding location, and gripper is not grasping)
        cubeA_align_cubeB = (torch.norm(states["cubeA_to_cubeB_pos"][:, :2], dim=-1) < 0.02)
        cubeA_on_cubeB = torch.abs(cubeA_height - target_height) < 0.02
        gripper_away_from_cubeA = (d > 0.04)
        stack_reward = cubeA_align_cubeB & cubeA_on_cubeB & gripper_away_from_cubeA

        # Compose rewards

        # We either provide the stack reward or the align + dist reward
        rewards = torch.where(
            stack_reward,
            reward_settings["r_stack_scale"] * stack_reward,
            reward_settings["r_dist_scale"] * dist_reward + reward_settings["r_lift_scale"] * lift_reward +
            reward_settings[
                "r_align_scale"] * align_reward,
        )

        reset_buf = torch.where(progress_buf >= _max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)
        reset_buf = torch.where(stack_reward > 0, torch.ones_like(reset_buf), reset_buf)

        return rewards, reset_buf

