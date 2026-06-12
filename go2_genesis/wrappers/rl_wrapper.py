import torch

from go2_genesis.locomotion_env import NUM_FEET, LocoEnv
from go2_genesis.utils import (
    gs_inv_quat,
    gs_quat_apply,
    gs_quat_conjugate,
    gs_quat_from_angle_axis,
    gs_quat_mul,
    gs_transform_by_quat,
)

XY_AXES = slice(0, 2)
YAW_AXIS = 2
X_AXIS_INDEX = 0
Y_AXIS_INDEX = 1
Z_AXIS_INDEX = 2

FRONT_RIGHT_LEG = slice(0, 3)
FRONT_LEFT_LEG = slice(3, 6)
REAR_RIGHT_LEG = slice(6, 9)
REAR_LEFT_LEG = slice(9, 12)
HIP_DOF = 0
THIGH_CALF_DOFS = slice(1, 3)

BACK_FEET = slice(2, 4)
COLLISION_CONTACT_FORCE = 0.1
AIR_TIME_CONTACT_FORCE = 1.0
MIN_COMMAND_NORM_FOR_AIR_TIME = 0.1
AIR_TIME_REWARD_OFFSET = 0.5

BACKFLIP_PREP_END_S = 0.5
BACKFLIP_ROTATION_START_S = 0.5
BACKFLIP_ROTATION_END_S = 1.0
JUMP_VELOCITY_START_S = 0.5
JUMP_VELOCITY_END_S = 0.75
HEIGHT_CONTROL_BEFORE_S = 0.4
HEIGHT_CONTROL_AFTER_S = 1.4
BACKFLIP_TARGET_HEIGHT = 0.3
MAX_BACKFLIP_PITCH_RATE = 7.2
MAX_JUMP_Z_VELOCITY = 3
PITCH_AXIS = (0, 1, 0)


class Go2(LocoEnv):
    def check_termination(self):
        self.reset_buf = self.episode_length_buf > self.max_episode_length

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(
            torch.square(
                self.commands[:, XY_AXES] - self.base_lin_vel[:, XY_AXES],
            ),
            dim=1,
        )
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(
            self.commands[:, YAW_AXIS] - self.base_ang_vel[:, YAW_AXIS],
        )
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_lin_vel_x(self):
        return torch.square(self.base_lin_vel[:, X_AXIS_INDEX])

    def _reward_lin_vel_y(self):
        return torch.square(self.base_lin_vel[:, Y_AXIS_INDEX])

    def _reward_lin_vel_z(self):
        return torch.square(self.base_lin_vel[:, Z_AXIS_INDEX])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, XY_AXES]), dim=1)

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, XY_AXES]), dim=1)

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(
            torch.square((self.last_dof_vel - self.dof_vel) / self.dt),
            dim=1,
        )

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = self.base_pos[:, 2]
        base_height_target = self.reward_cfg["base_height_target"]
        return torch.square(base_height - base_height_target)

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(
            1.0
            * (
                torch.norm(
                    self.link_contact_forces[:, self.penalized_contact_link_indices, :],
                    dim=-1,
                )
                > COLLISION_CONTACT_FORCE
            ),
            dim=1,
        )

    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.0)  # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.0)  # upper limit
        return torch.sum(out_of_limits, dim=1)

    def _reward_feet_air_time(self):
        # Reward long steps
        contact = self.link_contact_forces[:, self.feet_link_indices, Z_AXIS_INDEX] > AIR_TIME_CONTACT_FORCE
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.0) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum(
            (self.feet_air_time - AIR_TIME_REWARD_OFFSET) * first_contact,
            dim=1,
        )  # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, XY_AXES], dim=1) > MIN_COMMAND_NORM_FOR_AIR_TIME
        self.feet_air_time *= ~contact_filt
        return rew_airTime

    def _reward_base_xy_drift(self):
        # Penalize base position drift
        xy_pos = self.base_pos[:, XY_AXES]
        # Return per-env drift magnitude (not a single scalar)
        return torch.norm(xy_pos, dim=1)

    def _reward_actions_symmetry(self):
        front_right = self.actions[:, FRONT_RIGHT_LEG]
        front_left = self.actions[:, FRONT_LEFT_LEG]
        rear_right = self.actions[:, REAR_RIGHT_LEG]
        rear_left = self.actions[:, REAR_LEFT_LEG]

        actions_diff = torch.square(front_right[:, HIP_DOF] + front_left[:, HIP_DOF])
        actions_diff += torch.square(front_right[:, THIGH_CALF_DOFS] - front_left[:, THIGH_CALF_DOFS]).sum(dim=-1)
        actions_diff += torch.square(rear_right[:, HIP_DOF] + rear_left[:, HIP_DOF])
        actions_diff += torch.square(rear_right[:, THIGH_CALF_DOFS] - rear_left[:, THIGH_CALF_DOFS]).sum(dim=-1)
        return actions_diff

    def _reward_feet_height_before_backflip(self):
        current_time = self.episode_length_buf * self.dt
        foot_height = (self.foot_positions[:, :, Z_AXIS_INDEX]).view(self.num_envs, -1) - 0.02
        return foot_height.clamp(min=0).sum(dim=1) * (current_time < BACKFLIP_PREP_END_S)

    def _reward_feet_distance(self):
        cur_footsteps_translated = self.foot_positions - self.base_pos.unsqueeze(1)
        footsteps_in_body_frame = torch.zeros(self.num_envs, NUM_FEET, 3, device=self.device)
        for i in range(NUM_FEET):
            footsteps_in_body_frame[:, i, :] = gs_quat_apply(
                gs_quat_conjugate(self.base_quat),
                cur_footsteps_translated[:, i, :],
            )

        stance_width = 0.3 * torch.ones([self.num_envs, 1], device=self.device)
        desired_ys = torch.cat([stance_width / 2, -stance_width / 2, stance_width / 2, -stance_width / 2], dim=1)
        stance_diff = torch.square(desired_ys - footsteps_in_body_frame[:, :, Y_AXIS_INDEX]).sum(dim=1)

        return stance_diff

    def _reward_gravity_y(self):
        return torch.square(self.projected_gravity[:, Y_AXIS_INDEX])

    def _reward_height_control(self):
        # Encourage height during the takeoff and landing windows of the backflip.
        current_time = self.episode_length_buf * self.dt
        height_diff = torch.square(BACKFLIP_TARGET_HEIGHT - self.base_pos[:, Z_AXIS_INDEX]) * torch.logical_or(
            current_time < HEIGHT_CONTROL_BEFORE_S,
            current_time > HEIGHT_CONTROL_AFTER_S,
        )
        return height_diff

    def _reward_ang_vel_z(self):
        return torch.abs(self.base_ang_vel[:, Z_AXIS_INDEX])

    def _reward_ang_vel_y(self):
        current_time = self.episode_length_buf * self.dt
        ang_vel = -self.base_ang_vel[:, Y_AXIS_INDEX].clamp(
            max=MAX_BACKFLIP_PITCH_RATE,
            min=-MAX_BACKFLIP_PITCH_RATE,
        )
        return ang_vel * torch.logical_and(
            current_time > BACKFLIP_ROTATION_START_S,
            current_time < BACKFLIP_ROTATION_END_S,
        )

    def _reward_orientation_control(self):
        # Track the desired pitch progression during the backflip.
        current_time = self.episode_length_buf * self.dt
        phase = (current_time - BACKFLIP_ROTATION_START_S).clamp(min=0, max=0.5)
        quat_pitch = gs_quat_from_angle_axis(
            4 * phase * torch.pi,
            torch.tensor(PITCH_AXIS, device=self.device, dtype=torch.float),
        )

        desired_base_quat = gs_quat_mul(quat_pitch, self.base_init_quat.reshape(1, -1).repeat(self.num_envs, 1))
        inv_desired_base_quat = gs_inv_quat(desired_base_quat)
        desired_projected_gravity = gs_transform_by_quat(self.global_gravity, inv_desired_base_quat)

        orientation_diff = torch.sum(torch.square(self.projected_gravity - desired_projected_gravity), dim=1)

        return orientation_diff

    def _reward_lin_vel_z_during_jump(self):
        current_time = self.episode_length_buf * self.dt
        lin_vel = self.robot.get_vel()[:, Z_AXIS_INDEX].clamp(max=MAX_JUMP_Z_VELOCITY)
        return lin_vel * torch.logical_and(current_time > JUMP_VELOCITY_START_S, current_time < JUMP_VELOCITY_END_S)

    def _reward_similar_to_default(self):
        # Penalize joint poses far away from default pose
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_upright(self):
        # Encourage upright orientation (z-axis up)
        upright = torch.square(self.projected_gravity[:, X_AXIS_INDEX])
        return upright

    def _reward_back_feet_high(self):
        # Encourage back feet to be high above the base
        feet_height = (
            self.foot_positions[:, BACK_FEET, Z_AXIS_INDEX] - self.base_pos[:, Z_AXIS_INDEX].unsqueeze(1)
        ).mean(dim=1)
        return feet_height
