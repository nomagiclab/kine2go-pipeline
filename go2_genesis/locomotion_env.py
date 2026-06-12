# Taken from https://github.com/ziyanx02/Genesis-backflip/blob/main/locomotion_env.py
import genesis as gs
import numpy as np
import torch
from tensordict import TensorDict

from go2_genesis.utils import (
    QUAT_DIM,
    XYZ_DIM,
    gs_euler2quat,
    gs_inv_quat,
    gs_quat2euler,
    gs_quat_from_angle_axis,
    gs_quat_mul,
    gs_rand_float,
    gs_transform_by_quat,
    wrap_to_pi,
)

COMMAND_HEADING = "heading"
COMMAND_ANG_VEL_YAW = "ang_vel_yaw"
VALID_COMMAND_TYPES = (COMMAND_HEADING, COMMAND_ANG_VEL_YAW, None)
VALID_ACTION_LATENCIES = (0, 0.02)

GO2_NUM_DOFS = 12
BASE_VELOCITY_DOF_INDICES = [0, 1, 2, 3, 4, 5]
BASE_LINK_INDEX = 1
EXPLICIT_RECORD_SUBSTEPS = (0, 2)
NUM_FEET = 4
FOOT_LINK_NAMES = ["FL_foot", "RL_foot", "FR_foot", "RR_foot"]

OBS_CLIP = 100.0
COMMAND_DEADBAND = 0.2
TERMINATION_CONTACT_FORCE = 1.0
YAW_RANDOMIZATION_MAX = 3.14
MACOS_PLATFORM = "macOS"


class LocoEnv:
    def __init__(
        self,
        num_envs,
        env_cfg,
        obs_cfg,
        reward_cfg,
        command_cfg,
        show_viewer,
        eval,
        debug,
        device=gs.gpu,
        camera_res: tuple[int, int] | None = None,
    ) -> None:
        self.num_envs = 1 if num_envs == 0 else num_envs
        self.num_build_envs = num_envs
        self.num_single_obs = obs_cfg["num_obs"]
        self.num_obs = self.num_single_obs * obs_cfg["num_history_obs"]
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]

        self.headless = not show_viewer
        self.eval = eval
        self.debug = debug

        self.dt = 1 / env_cfg["control_freq"]
        if env_cfg["use_implicit_controller"]:
            self.sim_dt = self.dt
            self.sim_substeps = env_cfg["decimation"]
        else:
            self.sim_dt = self.dt / env_cfg["decimation"]
            self.sim_substeps = 1
        self.max_episode_length_s = env_cfg["episode_length_s"]
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.obs_cfg = obs_cfg
        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_cfg = reward_cfg
        self.reward_scales = reward_cfg["reward_scales"]
        self.env_cfg = env_cfg
        self.command_cfg = command_cfg

        self.command_type = env_cfg["command_type"]
        assert self.command_type in VALID_COMMAND_TYPES

        self.action_latency = env_cfg["action_latency"]
        assert self.action_latency in VALID_ACTION_LATENCIES

        self.num_dof = env_cfg["num_dofs"]
        self.device = torch.device(device)
        if camera_res is not None:
            if camera_res[0] <= 0 or camera_res[1] <= 0:
                raise ValueError(f"camera_res values must be positive, got {camera_res}")
            self.camera_res = (int(camera_res[0]), int(camera_res[1]))
        else:
            self.camera_res = None

        self._prepare_scene()

        # build
        self.scene.build(n_envs=num_envs)

        self._init_buffers()
        self._prepare_reward_function()

        # domain randomization
        self._randomize_controls()
        self._randomize_rigids()

        self.extras["observations"] = {}

    def _prepare_scene(self):
        self.scene = self._create_scene()
        self.rigid_solver = self.scene.sim.rigid_solver

        self._add_ground_or_terrain()
        self._init_base_pose_tensors()
        self.robot = self._add_robot()

        if gs.platform != MACOS_PLATFORM:
            self._set_camera()

    def _create_scene(self):
        return gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.sim_dt,
                substeps=self.sim_substeps,
            ),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=200,
                refresh_rate=200,
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(
                rendered_envs_idx=[0],
            ),
            rigid_options=gs.options.RigidOptions(
                dt=self.sim_dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_self_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=not self.headless,
        )

    def _add_ground_or_terrain(self):
        if self.env_cfg["use_terrain"]:
            self.terrain_cfg = self.env_cfg["terrain_cfg"]
            self.terrain = self.scene.add_entity(
                gs.morphs.Terrain(
                    n_subterrains=self.terrain_cfg["n_subterrains"],
                    horizontal_scale=self.terrain_cfg["horizontal_scale"],
                    vertical_scale=self.terrain_cfg["vertical_scale"],
                    subterrain_size=self.terrain_cfg["subterrain_size"],
                    subterrain_types=self.terrain_cfg["subterrain_types"],
                ),
            )
            terrain_margin_x = self.terrain_cfg["n_subterrains"][0] * self.terrain_cfg["subterrain_size"][0]
            terrain_margin_y = self.terrain_cfg["n_subterrains"][1] * self.terrain_cfg["subterrain_size"][1]
            self.terrain_margin = torch.tensor(
                [terrain_margin_x, terrain_margin_y],
                device=self.device,
                dtype=gs.tc_float,
            )
            height_field = self.terrain.geoms[0].metadata["height_field"]
            self.height_field = (
                torch.tensor(
                    height_field,
                    device=self.device,
                    dtype=gs.tc_float,
                )
                * self.terrain_cfg["vertical_scale"]
            )
        else:
            self.scene.add_entity(
                gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True),
            )

    def _init_base_pose_tensors(self):
        self.base_init_pos = torch.tensor(
            self.env_cfg["base_init_pos"],
            device=self.device,
        )
        self.base_init_quat = torch.tensor(
            self.env_cfg["base_init_quat"],
            device=self.device,
        )

    def _add_robot(self):
        return self.scene.add_entity(
            gs.morphs.URDF(
                file=self.env_cfg["urdf_path"],
                merge_fixed_links=True,
                links_to_keep=self.env_cfg["links_to_keep"],
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
            ),
            visualize_contact=False,  # Turning this off as it slows down and doesn't give much
        )

    def find_link_indices(self, names):
        link_indices = []
        for link in self.robot.links:
            flag = False
            for name in names:
                if name in link.name:
                    flag = True
            if flag:
                link_indices.append(link.idx - self.robot.link_start)
        return link_indices

    def _prepare_reward_function(self):
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt

        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, _ in self.reward_scales.items():
            if name == "termination":
                continue
            self.reward_names.append(name)
            name = "_reward_" + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {
            name: torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float) for name in self.reward_scales
        }

    def _init_buffers(self):
        self._init_base_motion_buffers()
        self._init_observation_buffers()
        self._init_reward_and_reset_buffers()
        self._init_command_buffers()
        self._init_robot_indices()
        self._init_action_state_buffers()
        self._init_episode_tracking_buffers()
        self._init_control_buffers()
        self._init_gait_buffers()

    def _init_base_motion_buffers(self):
        self.base_euler = torch.zeros(
            (self.num_envs, XYZ_DIM),
            device=self.device,
            dtype=gs.tc_float,
        )
        self.base_lin_vel = torch.zeros(
            (self.num_envs, XYZ_DIM),
            device=self.device,
            dtype=gs.tc_float,
        )
        self.base_ang_vel = torch.zeros(
            (self.num_envs, XYZ_DIM),
            device=self.device,
            dtype=gs.tc_float,
        )
        self.projected_gravity = torch.zeros(
            (self.num_envs, XYZ_DIM),
            device=self.device,
            dtype=gs.tc_float,
        )
        self.global_gravity = torch.tensor(
            np.array([0.0, 0.0, -1.0]),
            device=self.device,
            dtype=gs.tc_float,
        )
        self.forward_vec = torch.zeros(
            (self.num_envs, XYZ_DIM),
            device=self.device,
            dtype=gs.tc_float,
        )
        self.forward_vec[:, 0] = 1.0

    def _init_observation_buffers(self):
        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_single_obs),
            device=self.device,
            dtype=gs.tc_float,
        )
        self.obs_history_buf = torch.zeros(
            (self.num_envs, self.num_obs),
            device=self.device,
            dtype=gs.tc_float,
        )
        self.obs_noise = torch.zeros(
            (self.num_envs, self.num_single_obs),
            device=self.device,
            dtype=gs.tc_float,
        )
        self._prepare_obs_noise()

    def _init_reward_and_reset_buffers(self):
        self.rew_buf = torch.zeros(
            (self.num_envs,),
            device=self.device,
            dtype=gs.tc_float,
        )
        self.rew_buf_pos = torch.zeros(
            (self.num_envs,),
            device=self.device,
            dtype=gs.tc_float,
        )
        self.rew_buf_neg = torch.zeros(
            (self.num_envs,),
            device=self.device,
            dtype=gs.tc_float,
        )
        self.reset_buf = torch.ones(
            (self.num_envs,),
            device=self.device,
            dtype=torch.bool,
        )
        self.episode_length_buf = torch.zeros(
            (self.num_envs,),
            device=self.device,
            dtype=gs.tc_int,
        )
        self.time_out_buf = torch.zeros(
            (self.num_envs,),
            device=self.device,
            dtype=torch.bool,
        )

    def _init_command_buffers(self):
        if self.command_type is not None:
            self.commands = torch.zeros(
                (self.num_envs, self.num_commands),
                device=self.device,
                dtype=gs.tc_float,
            )
            self.commands_scale = torch.tensor(
                [
                    self.obs_scales["lin_vel"],
                    self.obs_scales["lin_vel"],
                    self.obs_scales["ang_vel"],
                ],
                device=self.device,
                dtype=gs.tc_float,
            )

    def _init_robot_indices(self):
        self.motor_dofs = [self.robot.get_joint(name).dofs_idx_local[0] for name in self.env_cfg["dof_names"]]

        self.termination_contact_link_indices = []
        if self.env_cfg["termination_contact_link_names"] is not None:
            self.termination_contact_link_indices = self.find_link_indices(
                self.env_cfg["termination_contact_link_names"],
            )

        self.penalized_contact_link_indices = []
        if self.env_cfg.get("penalized_contact_link_names") is not None:
            self.penalized_contact_link_indices = self.find_link_indices(
                self.env_cfg.get("penalized_contact_link_names"),
            )

        self.feet_link_indices = self.find_link_indices(
            self.env_cfg["feet_link_names"],
        )

        assert len(self.feet_link_indices) > 0
        self.feet_link_indices_world_frame = list(self.feet_link_indices)

    def _init_action_state_buffers(self):
        self.actions = torch.zeros(
            (self.num_envs, self.num_dof),
            device=self.device,
            dtype=gs.tc_float,
        )
        self.last_actions = torch.zeros(
            (self.num_envs, self.num_dof),
            device=self.device,
            dtype=gs.tc_float,
        )
        self.last_last_actions = torch.zeros(
            (self.num_envs, self.num_dof),
            device=self.device,
            dtype=gs.tc_float,
        )
        self.dof_pos = torch.zeros(
            (self.num_envs, self.num_dof),
            device=self.device,
            dtype=gs.tc_float,
        )
        self.dof_vel = torch.zeros(
            (self.num_envs, self.num_dof),
            device=self.device,
            dtype=gs.tc_float,
        )
        self.last_dof_vel = torch.zeros(
            (self.num_envs, self.num_dof),
            device=self.device,
            dtype=gs.tc_float,
        )
        self.root_vel = torch.zeros(
            (self.num_envs, XYZ_DIM),
            device=self.device,
            dtype=gs.tc_float,
        )
        self.last_root_vel = torch.zeros(
            (self.num_envs, XYZ_DIM),
            device=self.device,
            dtype=gs.tc_float,
        )
        self.base_pos = torch.zeros(
            (self.num_envs, XYZ_DIM),
            device=self.device,
            dtype=gs.tc_float,
        )
        self.base_quat = torch.zeros(
            (self.num_envs, QUAT_DIM),
            device=self.device,
            dtype=gs.tc_float,
        )
        self.link_contact_forces = torch.zeros(
            (self.num_envs, self.robot.n_links, XYZ_DIM),
            device=self.device,
            dtype=gs.tc_float,
        )

        self.feet_air_time = torch.zeros(
            (self.num_envs, len(self.feet_link_indices)),
            device=self.device,
            dtype=gs.tc_float,
        )
        self.feet_max_height = torch.zeros(
            (self.num_envs, len(self.feet_link_indices)),
            device=self.device,
            dtype=gs.tc_float,
        )
        self.last_contacts = torch.zeros(
            (self.num_envs, len(self.feet_link_indices)),
            device=self.device,
            dtype=torch.bool,
        )

    def _init_episode_tracking_buffers(self):
        self.env_identities = torch.arange(
            self.num_envs,
            device=self.device,
            dtype=gs.tc_int,
        )
        self.common_step_counter = 0
        self.extras = {}

        self.terrain_heights = torch.zeros(
            (self.num_envs,),
            device=self.device,
            dtype=gs.tc_float,
        )

    def _init_control_buffers(self):
        stiffness = self.env_cfg["PD_stiffness"]
        damping = self.env_cfg["PD_damping"]

        self.p_gains, self.d_gains = [], []
        for dof_name in self.env_cfg["dof_names"]:
            for key in stiffness:
                if key in dof_name:
                    self.p_gains.append(stiffness[key])
                    self.d_gains.append(damping[key])

        self.p_gains = torch.tensor(self.p_gains, device=self.device)
        self.d_gains = torch.tensor(self.d_gains, device=self.device)
        self.batched_p_gains = self.p_gains[None, :].repeat(self.num_envs, 1)
        self.batched_d_gains = self.d_gains[None, :].repeat(self.num_envs, 1)

        self.robot.set_dofs_kp(self.p_gains, self.motor_dofs)
        self.robot.set_dofs_kv(self.d_gains, self.motor_dofs)

        default_joint_angles = self.env_cfg["default_joint_angles"]
        self.default_dof_pos = torch.tensor(
            [default_joint_angles[name] for name in self.env_cfg["dof_names"]],
            device=self.device,
        )

        self.dof_pos_limits = torch.stack(self.robot.get_dofs_limit(self.motor_dofs), dim=1)
        self.torque_limits = self.robot.get_dofs_force_range(self.motor_dofs)[1]
        for i in range(self.dof_pos_limits.shape[0]):
            # soft limits
            m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
            r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
            self.dof_pos_limits[i, 0] = m - 0.5 * r * self.reward_cfg["soft_dof_pos_limit"]
            self.dof_pos_limits[i, 1] = m + 0.5 * r * self.reward_cfg["soft_dof_pos_limit"]

        self.motor_strengths = gs.ones((self.num_envs, self.num_dof), dtype=float)
        self.motor_offsets = gs.zeros((self.num_envs, self.num_dof), dtype=float)

    def _init_gait_buffers(self):
        self.foot_positions = torch.ones(
            self.num_envs,
            len(self.feet_link_indices),
            XYZ_DIM,
            device=self.device,
            dtype=gs.tc_float,
        )
        self.foot_quaternions = torch.ones(
            self.num_envs,
            len(self.feet_link_indices),
            QUAT_DIM,
            device=self.device,
            dtype=gs.tc_float,
        )
        self.foot_velocities = torch.ones(
            self.num_envs,
            len(self.feet_link_indices),
            XYZ_DIM,
            device=self.device,
            dtype=gs.tc_float,
        )

    def _update_buffers(self):
        # update buffers
        # [:] is for non-parallelized scene
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        base_quat_rel = gs_quat_mul(
            self.base_quat,
            gs_inv_quat(self._base_init_quat_batch()),
        )
        self.base_euler = gs_quat2euler(base_quat_rel)

        inv_quat_yaw = gs_quat_from_angle_axis(
            -self.base_euler[:, 2],
            self._z_axis(),
        )

        inv_base_quat = gs_inv_quat(self.base_quat)
        self.base_lin_vel[:] = gs_transform_by_quat(self.robot.get_vel(), inv_quat_yaw)
        self.base_ang_vel[:] = gs_transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = gs_transform_by_quat(
            self.global_gravity,
            inv_base_quat,
        )

        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)
        self.link_contact_forces[:] = torch.tensor(
            self.robot.get_links_net_contact_force(),
            device=self.device,
            dtype=gs.tc_float,
        )

        self.foot_positions[:] = self.robot.get_links_pos(self.feet_link_indices_world_frame)
        self.foot_quaternions[:] = self.robot.get_links_quat(self.feet_link_indices_world_frame)
        self.foot_velocities[:] = self.robot.get_links_vel(self.feet_link_indices_world_frame)

        if self.env_cfg["use_terrain"]:
            self._update_terrain_heights()

    def _base_init_quat_batch(self):
        return self.base_init_quat.reshape(1, -1).repeat(self.num_envs, 1)

    def _z_axis(self):
        return torch.tensor([0, 0, 1], device=self.device, dtype=torch.float)

    def _update_terrain_heights(self):
        clipped_base_pos = self.base_pos[:, :2].clamp(
            min=torch.zeros(2, device=self.device),
            max=self.terrain_margin,
        )
        height_field_ids = (clipped_base_pos / self.terrain_cfg["horizontal_scale"] - 0.5).floor().int()
        height_field_ids.clamp_(min=0)
        self.terrain_heights = self.height_field[height_field_ids[:, 0], height_field_ids[:, 1]]

    def _compute_torques(self, actions):
        actions_scaled = actions * self.env_cfg["action_scale"]
        torques = (
            self.batched_p_gains * (actions_scaled + self.default_dof_pos - self.dof_pos + self.motor_offsets)
            - self.batched_d_gains * self.dof_vel
        )
        return torques * self.motor_strengths

    def _compute_target_dof_pos(self, actions):
        actions_scaled = actions * self.env_cfg["action_scale"]
        target_dof_pos = actions_scaled + self.default_dof_pos

        return target_dof_pos

    def check_termination(self):
        self.reset_buf = torch.zeros(
            (self.num_envs,),
            device=self.device,
            dtype=torch.bool,
        )

        if self.termination_contact_link_indices:
            self.reset_buf |= torch.any(
                torch.norm(
                    self.link_contact_forces[:, self.termination_contact_link_indices, :],
                    dim=-1,
                )
                > TERMINATION_CONTACT_FORCE,
                dim=1,
            )

        self.time_out_buf = self.episode_length_buf > self.max_episode_length  # no terminal reward for time-outs
        self.reset_buf |= torch.logical_or(
            torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"],
            torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"],
        )
        if self.env_cfg["use_terrain"]:
            self.reset_buf |= torch.logical_or(
                self.base_pos[:, 0] > self.terrain_margin[0],
                self.base_pos[:, 1] > self.terrain_margin[1],
            )
            self.reset_buf |= torch.logical_or(
                self.base_pos[:, 0] < 1,
                self.base_pos[:, 1] < 1,
            )
        self.reset_buf |= self.base_pos[:, 2] < self.env_cfg["termination_if_height_lower_than"]
        self.reset_buf |= self.time_out_buf

    def compute_reward(self):
        self.rew_buf[:] = 0.0
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def get_observations(self):
        return TensorDict({"policy": self.obs_history_buf})

    def post_physics_step(self):
        self.episode_length_buf += 1
        self.common_step_counter += 1

        self._update_buffers()
        self._resample_commands_and_randomization()
        self._update_heading_command()
        self._maybe_apply_random_push()

        self.check_termination()
        self.compute_reward()

        self._reset_done_envs()
        # self.rigid_solver.forward_kinematics() # no need currently
        self.compute_observations()

        self._render_step()
        self._update_last_step_buffers()

    def _resample_commands_and_randomization(self):
        resampling_time_s = self.env_cfg["resampling_time_s"]
        envs_idx = (self.episode_length_buf % int(resampling_time_s / self.dt) == 0).nonzero(as_tuple=False).flatten()
        self._resample_commands(envs_idx)
        self._randomize_rigids(envs_idx)
        self._randomize_controls(envs_idx)

    def _update_heading_command(self):
        if self.command_type != COMMAND_HEADING:
            return

        forward = gs_transform_by_quat(self.forward_vec, self.base_quat)
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        self.commands[:, 2] = torch.clip(
            0.5 * wrap_to_pi(self.commands[:, 3] - heading),
            -1.0,
            1.0,
        )

    def _maybe_apply_random_push(self):
        push_interval_s = self.env_cfg["push_interval_s"]
        if push_interval_s <= 0 or self.debug or self.eval:
            return

        max_push_vel_xy = self.env_cfg["max_push_vel_xy"]
        dofs_vel = self.robot.get_dofs_velocity()  # (num_envs, num_dof) [0:3] ~ base_link_vel
        push_vel = gs_rand_float(-max_push_vel_xy, max_push_vel_xy, (self.num_envs, 2), self.device)
        push_vel[((self.common_step_counter + self.env_identities) % int(push_interval_s / self.dt) != 0)] = 0
        dofs_vel[:, :2] += push_vel
        self.robot.set_dofs_velocity(dofs_vel)

    def _reset_done_envs(self):
        envs_idx = self.reset_buf.nonzero(as_tuple=False).flatten()
        if self.num_build_envs > 0:
            self.reset_idx(envs_idx)

    def _render_step(self):
        if gs.platform != MACOS_PLATFORM:
            self._render_headless()
        if not self.headless and self.debug:
            self._draw_debug_vis()

    def _update_last_step_buffers(self):
        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.robot.get_vel()

    def compute_observations(self):
        obs = self._policy_obs_parts()

        if self.command_type is not None:
            obs.append(self.commands[:, :3] * self.commands_scale)

        if self.obs_cfg["include_phase"]:
            obs.extend(self._phase_obs_parts())

        self.obs_buf = torch.cat(obs, axis=-1)
        self._add_observation_noise()
        self.obs_buf = torch.clip(self.obs_buf, -OBS_CLIP, OBS_CLIP)
        self._append_observation_history()

    def _policy_obs_parts(self):
        return [
            self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
            self.projected_gravity,  # 3
            (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],
            self.dof_vel * self.obs_scales["dof_vel"],
            self.actions,
        ]

    def _phase_obs_parts(self):
        phase = torch.pi * self.episode_length_buf[:, None] * self.dt / 2
        return [
            torch.sin(phase),
            torch.cos(phase),
            torch.sin(phase / 2),
            torch.cos(phase / 2),
            torch.sin(phase / 4),
            torch.cos(phase / 4),
        ]

    def _add_observation_noise(self):
        if self.eval:
            return

        self.obs_buf += (
            gs_rand_float(
                -1.0,
                1.0,
                (self.num_envs, self.num_single_obs),
                self.device,
            )
            * self.obs_noise
        )

    def _append_observation_history(self):
        self.obs_history_buf = torch.cat(
            [self.obs_history_buf[:, self.num_single_obs :], self.obs_buf.detach()],
            dim=1,
        )

    def _prepare_obs_noise(self):
        # Per-feature noise scale, matching `_policy_obs_parts` order.
        # Only the sensors get noise; actions/commands/phase keep zero.
        obs_noise_cfg = self.obs_cfg["obs_noise"]
        sections = (
            (XYZ_DIM, obs_noise_cfg["ang_vel"]),
            (XYZ_DIM, obs_noise_cfg["gravity"]),
            (self.num_dof, obs_noise_cfg["dof_pos"]),
            (self.num_dof, obs_noise_cfg["dof_vel"]),
        )

        start = 0
        for size, value in sections:
            self.obs_noise[:, start : start + size] = value
            start += size

    def _resample_commands(self, envs_idx):
        # resample commands

        if self.command_type is None:
            return

        # lin_vel
        self.commands[envs_idx, 0] = gs_rand_float(
            *self.command_cfg["lin_vel_x_range"],
            (len(envs_idx),),
            self.device,
        )
        self.commands[envs_idx, 1] = gs_rand_float(
            *self.command_cfg["lin_vel_y_range"],
            (len(envs_idx),),
            self.device,
        )
        self.commands[envs_idx, :2] *= (torch.norm(self.commands[envs_idx, :2], dim=1) > COMMAND_DEADBAND).unsqueeze(1)

        # ang_vel
        if self.command_type == COMMAND_HEADING:
            self.commands[envs_idx, 3] = gs_rand_float(
                -YAW_RANDOMIZATION_MAX,
                YAW_RANDOMIZATION_MAX,
                (len(envs_idx),),
                self.device,
            )
        elif self.command_type == COMMAND_ANG_VEL_YAW:
            self.commands[envs_idx, 2] = gs_rand_float(
                *self.command_cfg["ang_vel_range"],
                (len(envs_idx),),
                self.device,
            )
            self.commands[envs_idx, 2] *= torch.abs(self.commands[envs_idx, 2]) > COMMAND_DEADBAND

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        self._reset_dofs(envs_idx)
        self._reset_base_pose(envs_idx)
        self._reset_base_velocity(envs_idx)
        self._resample_commands(envs_idx)
        self._reset_episode_buffers(envs_idx)
        self._fill_reset_extras(envs_idx)

    def _reset_dofs(self, envs_idx):
        dof_rand_range = self.env_cfg["dofs_position_rand_range"]
        self.dof_pos[envs_idx] = (self.default_dof_pos) + gs_rand_float(
            -dof_rand_range,
            dof_rand_range,
            (len(envs_idx), self.num_dof),
            self.device,
        )
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

    def _reset_base_pose(self, envs_idx):
        self.base_pos[envs_idx] = self.base_init_pos
        base_pos_rand_range = self.env_cfg["base_position_rand_range"]
        self.base_pos[envs_idx, :2] += gs_rand_float(
            -base_pos_rand_range,
            base_pos_rand_range,
            (len(envs_idx), 2),
            self.device,
        )
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        base_euler = gs_rand_float(
            -0.1,
            0.1,
            (len(envs_idx), 3),
            self.device,
        )

        if self.env_cfg["randomize_base_yaw"]:
            base_euler[:, 2] = gs_rand_float(0.0, YAW_RANDOMIZATION_MAX, (len(envs_idx),), self.device)

        self.base_quat[envs_idx] = gs_quat_mul(
            gs_euler2quat(base_euler),
            self.base_quat[envs_idx],
        )
        self.robot.set_pos(
            self.base_pos[envs_idx],
            zero_velocity=False,
            envs_idx=envs_idx,
        )
        self.robot.set_quat(
            self.base_quat[envs_idx],
            zero_velocity=False,
            envs_idx=envs_idx,
        )
        self.robot.zero_all_dofs_velocity(envs_idx)

        inv_base_quat = gs_inv_quat(self.base_quat)
        self.projected_gravity = gs_transform_by_quat(
            self.global_gravity,
            inv_base_quat,
        )

    def _reset_base_velocity(self, envs_idx):
        self.base_lin_vel[envs_idx] = 0  # gs_rand_float(-0.5, 0.5, (len(envs_idx), 3), self.device)
        self.base_ang_vel[envs_idx] = 0.0  # gs_rand_float(-0.5, 0.5, (len(envs_idx), 3), self.device)
        base_vel = torch.concat(
            [self.base_lin_vel[envs_idx], self.base_ang_vel[envs_idx]],
            dim=1,
        )
        self.robot.set_dofs_velocity(
            velocity=base_vel,
            dofs_idx_local=BASE_VELOCITY_DOF_INDICES,
            envs_idx=envs_idx,
        )

    def _reset_episode_buffers(self, envs_idx):
        self.obs_history_buf[envs_idx] = 0.0
        self.actions[envs_idx] = 0.0
        self.last_actions[envs_idx] = 0.0
        self.last_last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.feet_air_time[envs_idx] = 0.0
        self.feet_max_height[envs_idx] = 0.0
        self.last_contacts[envs_idx] = False
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

    def _fill_reset_extras(self, envs_idx):
        self.extras["episode"] = {}
        for key in self.episode_sums:
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.max_episode_length_s
            )
            self.episode_sums[key][envs_idx] = 0.0
        # send timeout info to the algorithm
        if self.env_cfg["send_timeouts"]:
            self.extras["time_outs"] = self.time_out_buf

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return None, None

    def step(self, actions):
        self.actions = self._clip_actions(actions)
        exec_actions = self.last_actions if self.action_latency > 0 else self.actions

        if self.env_cfg["use_implicit_controller"]:
            dof_pos_list, dof_vel_list = self._step_implicit_controller(exec_actions)
        else:
            dof_pos_list, dof_vel_list = self._step_explicit_controller(exec_actions)

        self.dof_pos_list = dof_pos_list
        self.dof_vel_list = dof_vel_list

        self.post_physics_step()

        return (
            self.get_observations(),
            self.rew_buf,
            self.reset_buf,
            self.extras,
        )

    def _clip_actions(self, actions):
        clip_actions = self.env_cfg["clip_actions"]
        return torch.clip(actions, -clip_actions, clip_actions)

    def _step_implicit_controller(self, exec_actions):
        target_dof_pos = self._compute_target_dof_pos(exec_actions)
        self.robot.control_dofs_position(target_dof_pos, self.motor_dofs)
        self.scene.step()
        return [], []

    def _step_explicit_controller(self, exec_actions):
        dof_pos_list = []
        dof_vel_list = []

        for i in range(self.env_cfg["decimation"]):
            self.torques = self._compute_torques(exec_actions)
            if self.num_build_envs == 0:
                torques = self.torques.squeeze()
                self.robot.control_dofs_force(torques, self.motor_dofs)
            else:
                self.robot.control_dofs_force(self.torques, self.motor_dofs)
            self.scene.step()
            self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
            self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)

            if i in EXPLICIT_RECORD_SUBSTEPS:
                dof_pos_list.append(self.robot.get_dofs_position().detach().cpu())
                dof_vel_list.append(self.robot.get_dofs_velocity().detach().cpu())

        return dof_pos_list, dof_vel_list

    # ------------ domain randomization----------------

    def _randomize_rigids(self, env_ids=None):
        env_ids = self._randomization_env_ids(env_ids)
        if env_ids is None:
            return

        self._apply_randomizers(
            env_ids,
            (
                ("randomize_friction", self._randomize_link_friction),
                ("randomize_base_mass", self._randomize_base_mass),
                ("randomize_com_displacement", self._randomize_com_displacement),
            ),
        )

    def _randomize_controls(self, env_ids=None):
        env_ids = self._randomization_env_ids(env_ids)
        if env_ids is None:
            return

        self._apply_randomizers(
            env_ids,
            (
                ("randomize_motor_strength", self._randomize_motor_strength),
                ("randomize_motor_offset", self._randomize_motor_offset),
                ("randomize_kp_scale", self._randomize_kp),
                ("randomize_kd_scale", self._randomize_kd),
            ),
        )

    def _randomization_env_ids(self, env_ids):
        if self.eval:
            return None
        if env_ids is None:
            return torch.arange(0, self.num_envs)
        if len(env_ids) == 0:
            return None
        return env_ids

    def _apply_randomizers(self, env_ids, randomizers):
        for cfg_key, randomizer in randomizers:
            if self.env_cfg[cfg_key]:
                randomizer(env_ids)

    def _randomize_link_friction(self, env_ids):
        min_friction, max_friction = self.env_cfg["friction_range"]

        solver = self.rigid_solver

        ratios = (
            gs.rand((len(env_ids), 1), dtype=float).repeat(1, solver.n_geoms) * (max_friction - min_friction)
            + min_friction
        )
        solver.set_geoms_friction_ratio(ratios, torch.arange(0, solver.n_geoms), env_ids)

    def _randomize_base_mass(self, env_ids):
        min_mass, max_mass = self.env_cfg["added_mass_range"]
        base_link_id = self.robot.get_link("base").idx

        added_mass = gs.rand((len(env_ids), 1), dtype=float) * (max_mass - min_mass) + min_mass

        self.rigid_solver.set_links_mass_shift(added_mass, [base_link_id], env_ids)

    def _randomize_com_displacement(self, env_ids):
        min_displacement, max_displacement = self.env_cfg["com_displacement_range"]
        base_link_id = BASE_LINK_INDEX

        com_displacement = (
            gs.rand((len(env_ids), 1, XYZ_DIM), dtype=float) * (max_displacement - min_displacement) + min_displacement
        )

        self.rigid_solver.set_links_COM_shift(com_displacement, [base_link_id], env_ids)

    def _randomize_motor_strength(self, env_ids):
        min_strength, max_strength = self.env_cfg["motor_strength_range"]
        self.motor_strengths[env_ids, :] = (
            gs.rand((len(env_ids), 1), dtype=float) * (max_strength - min_strength) + min_strength
        )

    def _randomize_motor_offset(self, env_ids):
        min_offset, max_offset = self.env_cfg["motor_offset_range"]
        self.motor_offsets[env_ids, :] = (
            gs.rand((len(env_ids), self.num_dof), dtype=float) * (max_offset - min_offset) + min_offset
        )

    def _randomize_kp(self, env_ids):
        min_scale, max_scale = self.env_cfg["kp_scale_range"]
        kp_scales = gs.rand((len(env_ids), self.num_dof), dtype=float) * (max_scale - min_scale) + min_scale
        self.batched_p_gains[env_ids, :] = kp_scales * self.p_gains[None, :]

    def _randomize_kd(self, env_ids):
        min_scale, max_scale = self.env_cfg["kd_scale_range"]
        kd_scales = gs.rand((len(env_ids), self.num_dof), dtype=float) * (max_scale - min_scale) + min_scale
        self.batched_d_gains[env_ids, :] = kd_scales * self.d_gains[None, :]

    def _draw_debug_vis(self):
        """Draws visualizations for debugging (slows down simulation a lot).
        Default behaviour: draws height measurement points
        """
        self.scene.clear_debug_objects()

        foot_poss = self.foot_positions[0].reshape(-1, 3)

        foot_poss = foot_poss.cpu()
        self.scene.draw_debug_line(foot_poss[0], foot_poss[3], radius=0.002, color=(1, 0, 0, 0.7))
        self.scene.draw_debug_line(foot_poss[1], foot_poss[2], radius=0.002, color=(1, 0, 0, 0.7))

    def _set_camera(self):
        """Set camera position and direction"""
        camera_kwargs = {
            "pos": np.array([0, -1, 1]),
            "lookat": np.array([0, 0, 0]),
            "fov": 40,
            "GUI": False,
        }
        if self.camera_res is not None:
            camera_kwargs["res"] = self.camera_res
        self._floating_camera = self.scene.add_camera(**camera_kwargs)

        self._recording = False
        self._recorded_frames = []

    def _render_headless(self):
        if self._recording:
            robot_pos = np.array(self.base_pos[0].cpu())
            self._floating_camera.set_pose(
                pos=robot_pos + np.array([-1, -1, 0.5]),
                lookat=robot_pos + np.array([0, 0, -0.1]),
            )
            frame, _, _, _ = self._floating_camera.render()
            self._recorded_frames.append(frame)

    def start_recording(self, *, record_internal=True):
        self._recorded_frames = []
        self._recording = True
        if record_internal:
            self._record_frames = True
        else:
            self._floating_camera.start_recording()

    def stop_recording(self, save_path=None):
        self._recorded_frames = []
        self._recording = False
        if save_path is not None:
            print("fps", int(1 / self.dt))
            self._floating_camera.stop_recording(save_path, fps=int(1 / self.dt))

    def save_state(self, path):
        return self.scene.save_checkpoint(path)

    def set_state(self, path):
        self.scene.load_checkpoint(path)
        self.scene.step()

    def set_robot(self, dofs_position, base_position, base_quat, envs_idx=None):
        if envs_idx is None:
            envs_idx = torch.arange(0, self.num_envs, device=self.device)

        assert dofs_position.shape == (len(envs_idx), GO2_NUM_DOFS)
        assert base_position.shape == (len(envs_idx), XYZ_DIM)
        assert base_quat.shape == (len(envs_idx), QUAT_DIM)

        self.robot.set_dofs_position(dofs_position, envs_idx=envs_idx, dofs_idx_local=self.motor_dofs)
        self.robot.set_pos(base_position, envs_idx=envs_idx)
        self.robot.set_quat(base_quat, envs_idx=envs_idx)
