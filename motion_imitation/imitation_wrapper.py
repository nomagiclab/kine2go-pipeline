import genesis as gs
import numpy as np
import torch
from genesis.utils.geom import (
    axis_angle_to_quat,
    inv_quat,
    transform_by_quat,
    transform_quat_by_quat,
)
from tensordict import TensorDict

from go2_genesis.locomotion_env import (
    FOOT_LINK_NAMES,
    NUM_FEET,
    OBS_CLIP,
    TERMINATION_CONTACT_FORCE,
    LocoEnv,
)
from go2_genesis.utils import QUAT_DIM, XYZ_DIM
from motion_imitation.utils import calc_heading_from_quat, calc_quat_from_heading, quaternion_to_axis_angle

REF_JOINT_ROT_START = 0
REF_JOINT_ROT_NO_POS_START = 6
REF_JOINT_ROT_END = 18
REF_JOINT_VEL_START = 18
REF_JOINT_VEL_NO_ROOT_START = 24
REF_JOINT_VEL_END = 36
REF_END_EFFECTOR_POS_START = 36
REF_END_EFFECTOR_POS_END = 48
REF_ROOT_POS_START = 48
REF_ROOT_POS_END = 51
REF_ROOT_ROT_START = 51
REF_ROOT_ROT_END = 55
REF_ROOT_VEL_START = 55
REF_ROOT_VEL_END = 58
REF_ROOT_ANG_VEL_START = 58
REF_ROOT_ANG_VEL_END = 61
REF_JOINT_ROT_SLICE = slice(REF_JOINT_ROT_NO_POS_START, REF_JOINT_ROT_END)
REF_JOINT_VEL_SLICE = slice(REF_JOINT_VEL_START, REF_JOINT_VEL_END)
REF_END_EFFECTOR_POS_SLICE = slice(REF_END_EFFECTOR_POS_START, REF_END_EFFECTOR_POS_END)
REF_ROOT_POS_SLICE = slice(REF_ROOT_POS_START, REF_ROOT_POS_END)
REF_ROOT_ROT_SLICE = slice(REF_ROOT_ROT_START, REF_ROOT_ROT_END)
REF_ROOT_VEL_SLICE = slice(REF_ROOT_VEL_START, REF_ROOT_VEL_END)
REF_ROOT_ANG_VEL_SLICE = slice(REF_ROOT_ANG_VEL_START, REF_ROOT_ANG_VEL_END)

TARGET_SIZE = 19
TARGET_INDICES = [1, 2, 10, 30]

EPS = 1e-5
ROOT_QPOS_DIM = 7
DEFAULT_REF_STATE_INIT_PROB = 1.0
DEFAULT_WARMUP_TIME_S = 0.0
DEFAULT_POLICY_STEPS_PER_REF_STEP = 1
TARGET_DOF_START = REF_JOINT_ROT_NO_POS_START
TARGET_DOF_END = REF_JOINT_ROT_END
GHOST_COLOR = (1, 0, 0, 0.5)
TARGET_GHOST_OPACITIES = [0.25, 0.5, 0.75, 0.90]
TARGET_GHOST_RGB = (0, 1, 0)
ROOT_POS_PERTURB_STD = 0.025
ROOT_ROT_PERTURB_STD = 0.025 * np.pi
JOINT_POSE_PERTURB_STD = 0.05 * np.pi
ROOT_VEL_PERTURB_STD = 0.1
JOINT_VEL_PERTURB_STD = 0.05 * np.pi
RANDOM_HEADING_SCALE = 2.0 * np.pi
JOINT_ROT_REWARD_SCALE = -5.0
JOINT_VEL_REWARD_SCALE = -0.1
END_EFFECTOR_REWARD_SCALE = -40.0
ROOT_POS_REWARD_SCALE = -20.0
ROOT_ROT_REWARD_SCALE = -10.0
ROOT_VEL_REWARD_SCALE = -2.0
ROOT_ANG_VEL_REWARD_SCALE = -0.2
SMOOTH_ACTION_REWARD_SCALE = -0.1
# Match `_sync_with_ref_motion` when `sync_full_pose` is False (standing height vs. motion root z).
WARMUP_STANDING_BASE_Z = 0.33


class ImitationWrapper(LocoEnv):
    def __init__(
        self,
        motions,
        num_envs,
        env_cfg,
        obs_cfg,
        reward_cfg,
        command_cfg,
        show_viewer,
        eval,
        debug,
        device="cuda",
        camera_res: tuple[int, int] | None = None,
    ):
        super().__init__(
            num_envs,
            env_cfg,
            obs_cfg,
            reward_cfg,
            command_cfg,
            show_viewer,
            eval,
            debug,
            device,
            camera_res=camera_res,
        )
        self.preprocess_motions(motions)

        self.perturb_init_state = env_cfg.get("perturb_init_state", True)

        # Num obs is what is used to determine the size of network inputs
        self.num_obs = obs_cfg["num_obs"] * obs_cfg["num_history_obs"] + TARGET_SIZE * len(TARGET_INDICES)

        # In history buffer we do not store targets
        self.obs_history_buf = torch.zeros(
            (self.num_envs, obs_cfg["num_obs"] * obs_cfg["num_history_obs"]),
            device=self.device,
            dtype=gs.tc_float,
        )

        self.ref_state_init_prob = env_cfg.get("ref_state_init_prob", DEFAULT_REF_STATE_INIT_PROB)
        self.warmup_time_s = env_cfg.get("warmup_time_s", DEFAULT_WARMUP_TIME_S)
        self.warmup_steps = int(self.warmup_time_s / self.dt)
        self.num_policy_steps_per_ref_motion_step = env_cfg.get(
            "policy_steps_per_ref_motion_step",
            DEFAULT_POLICY_STEPS_PER_REF_STEP,
        )
        self.randomize_trajectory_heading = env_cfg.get("randomize_trajectory_heading", False)

    def _prepare_scene(self):
        super()._prepare_scene()
        self.ghost = None
        self.target_ghosts = []
        if self.debug:
            self.ghost = self._add_ghost_entity(GHOST_COLOR)
            for opacity in TARGET_GHOST_OPACITIES:
                self.target_ghosts.append(self._add_ghost_entity((*TARGET_GHOST_RGB, opacity)))

    def _add_ghost_entity(self, color):
        return self.scene.add_entity(
            gs.morphs.URDF(
                file=self.env_cfg["urdf_path"],
                merge_fixed_links=True,
                links_to_keep=self.env_cfg["links_to_keep"],
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
                collision=False,
            ),
            surface=gs.surfaces.Default(color=color),
            visualize_contact=False,
        )

    def preprocess_motions(self, motions_paths):
        """Load and preprocess a single reference motion.

        The original implementation supported multiple motions; the current
        pipeline always uses exactly one motion, so we store it as a simple
        (T, D) tensor instead of (num_motions, T, D).
        """
        assert len(motions_paths) == 1, (
            f"ImitationWrapper currently supports exactly one motion, got {len(motions_paths)}"
        )

        motion_path, motion_start, motion_end = motions_paths[0]
        motion_np = np.load(motion_path)
        motion_np = motion_np[motion_start:motion_end]

        self.motion = torch.tensor(motion_np, device=self.device)
        self.motion_length = self.motion.shape[0]
        self.motion_target_indices = TARGET_INDICES

        self.cycle_delta_pos = self.motion[-1, REF_ROOT_POS_SLICE] - self.motion[0, REF_ROOT_POS_SLICE]
        # ignore vertical displacement in cycle delta
        self.cycle_delta_pos[2] = 0.0

        rot_start = self.motion[0, REF_ROOT_ROT_SLICE].unsqueeze(0)
        rot_end = self.motion[-1, REF_ROOT_ROT_SLICE].unsqueeze(0)
        inv_rot_start = inv_quat(rot_start)
        delta_rot = transform_quat_by_quat(rot_end, inv_rot_start)
        # scalar heading change over one motion cycle
        self.cycle_delta_heading = calc_heading_from_quat(delta_rot)[0]

        max_num_cycles = (
            int(
                (self.max_episode_length.item() + max(self.motion_target_indices) + 1) // self.motion_length + 1,
            )
            + 1
        )

        # TODO: this supports only one reference motion for now
        # We calculate pos and heading offsets in advance for efficiency
        self.cycle_rot = torch.zeros((max_num_cycles, QUAT_DIM), device=self.device)
        self.cycle_translation = torch.zeros((max_num_cycles, XYZ_DIM), device=self.device)

        self.cycle_rot[0] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)

        for i in range(1, max_num_cycles):
            self._precompute_cycle_step(i)

    def _precompute_cycle_step(self, cycle_index: int):
        """Fill `cycle_rot` and `cycle_translation` entries for a given cycle index."""
        # In each cycle we rotate the robot and start the position offset with new heading
        rot_offset_rad = cycle_index * self.cycle_delta_heading
        self.cycle_rot[cycle_index] = calc_quat_from_heading(rot_offset_rad)

        rot = self.cycle_rot[cycle_index - 1]
        self.cycle_translation[cycle_index] = self.cycle_translation[cycle_index - 1] + transform_by_quat(
            self.cycle_delta_pos.unsqueeze(0),
            rot,
        )

    def _init_buffers(self):
        super()._init_buffers()
        # This will be a quaternion offset of heading of the motion.
        # Most of the motions we have just follow roughly x axis.
        # We will rotate entire trajectories to add rotation invariance to policy.
        self.motion_rot_offset_buf = torch.zeros((self.num_envs, QUAT_DIM), dtype=torch.float, device=self.device)
        self.motion_rot_offset_buf[:, 0] = 1.0
        self.motion_start_offset_buf = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self._default_joint_rot_ref = self._build_default_joint_rot_ref()

        # Foot positions relative to base (body frame) at default joint pose; used for warmup standing EE targets.
        self._cache_default_foot_offsets_body()

    def _build_default_joint_rot_ref(self):
        """Default joint angles ordered as reference expects: DOF local indices [6..17]."""
        joint_rot = torch.zeros((REF_JOINT_ROT_END - REF_JOINT_ROT_NO_POS_START,), device=self.device)
        for i, dof_idx in enumerate(self.motor_dofs):
            if REF_JOINT_ROT_NO_POS_START <= dof_idx < REF_JOINT_ROT_END:
                joint_rot[dof_idx - REF_JOINT_ROT_NO_POS_START] = self.default_dof_pos[i]
        return joint_rot

    def _cache_default_foot_offsets_body(self):
        """Compute foot offsets in base frame for default standing pose (env 0), once at init."""
        envs_idx = torch.tensor([0], device=self.device, dtype=torch.long)
        self.robot.set_dofs_position(
            position=self.default_dof_pos.unsqueeze(0),
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )
        self.robot.set_pos(self.base_init_pos.unsqueeze(0), envs_idx=envs_idx)
        self.robot.set_quat(self.base_init_quat.unsqueeze(0), envs_idx=envs_idx)
        feet_links = self._feet_link_indices(self.robot)
        feet_pos = self.robot.get_links_pos(links_idx_local=feet_links, envs_idx=envs_idx)
        root = self.robot.get_pos(envs_idx=envs_idx)
        rel_w = feet_pos - root.unsqueeze(1)
        inv_q = inv_quat(self.base_init_quat.reshape(1, -1))
        flat = rel_w.reshape(-1, XYZ_DIM)
        inv_rep = inv_q.repeat_interleave(NUM_FEET, dim=0)
        body = transform_by_quat(flat, inv_rep).reshape(1, NUM_FEET, XYZ_DIM)
        self._default_foot_offset_body = body[0].clone()

    def _feet_link_indices(self, entity):
        return [entity.get_link(name).idx - entity.link_start for name in FOOT_LINK_NAMES]

    def _placement_ref_at_motion_start(self, env_indices):
        """World-frame reference at the motion time index where tracking begins (after warmup)."""
        episode_steps = self.motion_start_offset_buf[env_indices]
        frame_indices, cycle_numbers = self._ref_indices_from_episode_steps(episode_steps)
        return self._get_ref_state_at_frame(env_indices, frame_indices, cycle_numbers)

    def _build_standing_ref_from_placement(self, placement_ref: TensorDict) -> TensorDict:
        """Standing pose at motion-start placement: default joints, yaw from motion, fixed base height."""
        n = placement_ref.batch_size[0]
        heading = calc_heading_from_quat(placement_ref["root_quat"])
        heading_quat = calc_quat_from_heading(heading)

        root_pos = placement_ref["root_pos"].clone()
        root_pos[:, 2] = WARMUP_STANDING_BASE_Z

        joint_rot = self._default_joint_rot_ref.unsqueeze(0).expand(n, -1)
        joint_vel = torch.zeros_like(placement_ref["joint_vel"])
        root_vel = torch.zeros_like(placement_ref["root_vel"])
        root_ang_vel = torch.zeros_like(placement_ref["root_ang_vel"])

        feet_b = (
            self._default_foot_offset_body
            .unsqueeze(0)
            .expand(n, NUM_FEET, XYZ_DIM)
            .reshape(
                n * NUM_FEET,
                XYZ_DIM,
            )
        )
        hq = heading_quat.repeat_interleave(NUM_FEET, dim=0)
        feet_w = transform_by_quat(feet_b, hq).reshape(n, NUM_FEET, XYZ_DIM) + root_pos.unsqueeze(1)
        end_effector_pos = feet_w.reshape(n, -1)

        return TensorDict(
            {
                "root_pos": root_pos,
                "root_quat": heading_quat,
                "joint_rot": joint_rot,
                "joint_vel": joint_vel,
                "root_vel": root_vel,
                "root_ang_vel": root_ang_vel,
                "end_effector_pos": end_effector_pos,
            },
            batch_size=[n],
            device=self.device,
        )

    def _apply_warmup_standing(self, ref: TensorDict, env_indices, *, episode_steps=None) -> TensorDict:
        """Override reference with default standing for episode-steps that are still in warmup."""
        if self.warmup_steps <= 0:
            return ref
        if episode_steps is None:
            episode_steps = self.episode_length_buf[env_indices]
        warmup_mask = episode_steps < self.warmup_steps
        if not warmup_mask.any():
            return ref

        sub = env_indices[warmup_mask]
        placement = self._placement_ref_at_motion_start(sub)
        stand = self._build_standing_ref_from_placement(placement)
        for k in ref.keys():  # noqa: SIM118
            ref[k][warmup_mask] = stand[k]
        return ref

    def _get_ref_state_at_frame(self, env_indices, frame_indices, cycle_numbers):
        """Return reference motion state in world frame for the given frame and cycle.

        All tensors have shape (len(env_indices), ...). Root/vel/end-effector are
        transformed by cycle and motion_rot_offset; joint_rot/joint_vel are in joint space.
        """
        n = len(env_indices)
        raw = self._raw_ref_state(frame_indices)
        origin_pos = self.motion[0, REF_ROOT_POS_SLICE]
        final_rotation, cycled_translation = self._cycle_transform(env_indices, cycle_numbers, origin_pos)

        root_pos = transform_by_quat(raw["root_pos"] - origin_pos, final_rotation) + cycled_translation
        root_quat = transform_quat_by_quat(raw["root_quat"], final_rotation)
        root_vel = transform_by_quat(raw["root_vel"], final_rotation)
        root_ang_vel = transform_by_quat(raw["root_ang_vel"], final_rotation)
        end_effector_pos = self._transform_end_effector_pos(
            raw["end_effector_pos"],
            origin_pos,
            final_rotation,
            cycled_translation,
            n,
        )

        return TensorDict(
            {
                "root_pos": root_pos,
                "root_quat": root_quat,
                "joint_rot": raw["joint_rot"],
                "joint_vel": raw["joint_vel"],
                "root_vel": root_vel,
                "root_ang_vel": root_ang_vel,
                "end_effector_pos": end_effector_pos,
            },
            batch_size=[n],
            device=self.device,
        )

    def _raw_ref_state(self, frame_indices):
        """Return untransformed reference tensors for selected frame indices."""
        return {
            "root_pos": self.motion[frame_indices, REF_ROOT_POS_SLICE],
            "root_quat": self.motion[frame_indices, REF_ROOT_ROT_SLICE],
            "joint_rot": self.motion[frame_indices, REF_JOINT_ROT_SLICE],
            "joint_vel": self.motion[frame_indices, REF_JOINT_VEL_SLICE],
            "root_vel": self.motion[frame_indices, REF_ROOT_VEL_SLICE],
            "root_ang_vel": self.motion[frame_indices, REF_ROOT_ANG_VEL_SLICE],
            "end_effector_pos": self.motion[frame_indices, REF_END_EFFECTOR_POS_SLICE],
        }

    def _cycle_transform(self, env_indices, cycle_numbers, origin_pos):
        final_rotation = transform_quat_by_quat(
            self.cycle_rot[cycle_numbers],
            self.motion_rot_offset_buf[env_indices],
        )
        motion_rot = self.motion_rot_offset_buf[env_indices]
        cycled_translation = transform_by_quat(self.cycle_translation[cycle_numbers] + origin_pos, motion_rot)
        return final_rotation, cycled_translation

    def _transform_end_effector_pos(self, raw_end_effector_pos, origin_pos, final_rotation, cycled_translation, n):
        # transform_by_quat expects (N, 3) and (N, 4), not (n, 4, 3); repeat quat per foot
        origin_repeat = origin_pos.unsqueeze(0).expand(n, NUM_FEET, XYZ_DIM).reshape(n * NUM_FEET, XYZ_DIM)
        ee_flat = raw_end_effector_pos.reshape(n * NUM_FEET, XYZ_DIM) - origin_repeat
        quat_repeat = final_rotation.repeat_interleave(NUM_FEET, dim=0)
        return (
            transform_by_quat(ee_flat, quat_repeat).reshape(n, NUM_FEET, XYZ_DIM) + cycled_translation.unsqueeze(1)
        ).reshape(n, -1)

    def _sync_with_ref_motion(self, envs_idx, cycle_num=None, *, ghost=False, frame_indices=None, sync_full_pose=True):
        if cycle_num is None:
            cycle_num = torch.zeros(len(envs_idx), dtype=torch.long, device=self.device)
        if frame_indices is None:
            frame_indices = self._effective_motion_steps(envs_idx) % self.motion_length
        ref = self._get_ref_state_at_frame(
            envs_idx,
            frame_indices,
            cycle_num,
        )
        ref = self._apply_warmup_standing(ref, envs_idx)
        base_pos = ref["root_pos"].clone()
        base_quat = ref["root_quat"].clone()
        joint_pos = ref["joint_rot"].clone()
        joint_vel = ref["joint_vel"].clone()

        if self.perturb_init_state and not ghost:
            base_pos, base_quat, joint_pos, joint_vel = self._apply_init_perturbations(
                base_pos,
                base_quat,
                joint_pos,
                joint_vel,
                envs_idx,
            )

        entity = self.ghost if ghost else self.robot

        if sync_full_pose:
            self._sync_full_pose(entity, envs_idx, base_pos, base_quat, joint_pos, joint_vel)
        else:
            self._sync_standing_pose(entity, envs_idx, base_pos, base_quat)

    def _sync_full_pose(self, entity, envs_idx, base_pos, base_quat, joint_pos, joint_vel):
        entity.set_pos(base_pos, envs_idx=envs_idx)
        entity.set_quat(base_quat, envs_idx=envs_idx)
        entity.set_dofs_position(
            joint_pos,
            envs_idx=envs_idx,
            dofs_idx_local=self._ref_joint_rot_dofs(),
        )
        entity.set_dofs_velocity(
            joint_vel,
            envs_idx=envs_idx,
            dofs_idx_local=self._ref_joint_vel_dofs(),
        )

    def _sync_standing_pose(self, entity, envs_idx, base_pos, base_quat):
        entity.set_dofs_position(
            position=self.default_dof_pos.unsqueeze(0).expand(len(envs_idx), -1),
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )
        entity.zero_all_dofs_velocity(envs_idx)

        base_pos[:, 2] = WARMUP_STANDING_BASE_Z
        heading = calc_heading_from_quat(base_quat)
        heading_quat = calc_quat_from_heading(heading)

        entity.set_pos(base_pos, envs_idx=envs_idx)
        entity.set_quat(heading_quat, envs_idx=envs_idx)

    def _ref_joint_rot_dofs(self):
        return torch.arange(TARGET_DOF_START, TARGET_DOF_END, device=self.device)

    def _ref_joint_vel_dofs(self):
        return torch.arange(REF_JOINT_ROT_START, REF_JOINT_ROT_END, device=self.device)

    def _apply_init_perturbations(self, base_pos, base_quat, joint_pos, joint_vel, envs_idx):
        """Apply small random perturbations to initial state."""
        # Perturbation algorithm copied from
        # https://github.com/erwincoumans/motion_imitation/blob/d0e7b963c5a301984352d25a3ee0820266fa4218/motion_imitation/envs/env_wrappers/imitation_task.py#L1192
        base_pos[:, 0] += torch.normal(0, ROOT_POS_PERTURB_STD, size=(len(envs_idx),), device=self.device)
        base_pos[:, 1] += torch.normal(0, ROOT_POS_PERTURB_STD, size=(len(envs_idx),), device=self.device)

        rand_axis = (torch.randn(size=(len(envs_idx), XYZ_DIM), device=self.device) - 0.5) * 2.0
        rand_axis = rand_axis / (torch.norm(rand_axis, dim=-1, keepdim=True) + EPS)
        rand_theta = torch.normal(0, ROOT_ROT_PERTURB_STD, size=(len(envs_idx),), device=self.device)
        rand_rot = axis_angle_to_quat(rand_theta, rand_axis)
        base_quat = transform_quat_by_quat(base_quat, rand_rot)

        joint_pos += torch.normal(0, JOINT_POSE_PERTURB_STD, size=joint_pos.shape, device=self.device)

        # We will set base velocity as part of dofs velocity. We only change velocity in xy plane
        joint_vel[:, :2] += torch.normal(0, ROOT_VEL_PERTURB_STD, size=joint_vel[:, :2].shape, device=self.device)
        joint_vel[:, REF_JOINT_ROT_NO_POS_START:REF_JOINT_ROT_END] += torch.normal(
            0,
            JOINT_VEL_PERTURB_STD,
            size=joint_vel[:, REF_JOINT_ROT_NO_POS_START:REF_JOINT_ROT_END].shape,
            device=self.device,
        )

        return base_pos, base_quat, joint_pos, joint_vel

    def reset_idx(self, envs_idx):
        super().reset_idx(envs_idx)
        self.motion_rot_offset_buf[envs_idx] = calc_quat_from_heading(self._sample_motion_headings(envs_idx))
        self.motion_start_offset_buf[envs_idx] = self._sample_motion_start_offsets(envs_idx)
        ref_init_envs = self._select_ref_init_envs(envs_idx)

        if len(ref_init_envs) > 0 and self.env_cfg.get("start_synced_with_ref_motion", True):
            self._sync_with_ref_motion(ref_init_envs, sync_full_pose=self.env_cfg.get("sync_full_pose", True))

    def _sample_motion_headings(self, envs_idx):
        headings = torch.zeros(len(envs_idx), device=self.device)
        if self.randomize_trajectory_heading:
            headings = torch.rand(len(envs_idx), device=self.device) * RANDOM_HEADING_SCALE
        return headings

    def _sample_motion_start_offsets(self, envs_idx):
        if self.env_cfg.get("randomize_init_frame", False):
            return torch.randint(
                0,
                self.motion_length,
                (len(envs_idx),),
                device=self.device,
            )
        return torch.zeros(len(envs_idx), dtype=torch.long, device=self.device)

    def _select_ref_init_envs(self, envs_idx):
        ref_init_mask = torch.rand(len(envs_idx), device=self.device) < self.ref_state_init_prob
        return envs_idx[ref_init_mask]

    def _all_env_indices(self):
        return torch.arange(self.num_envs, device=self.device)

    def _effective_motion_steps_from_episode_steps(self, env_idx, episode_steps):
        """Map episode-time steps to motion-time steps with warmup handling."""
        return (episode_steps - self.warmup_steps).clamp(min=0) + self.motion_start_offset_buf[env_idx]

    def _effective_motion_steps(self, env_idx):
        """Motion-time step per env at the current episode clock."""
        return self._effective_motion_steps_from_episode_steps(env_idx, self.episode_length_buf[env_idx])

    def _ref_indices_from_episode_steps(self, episode_steps):
        frame_indices = episode_steps % self.motion_length
        cycle_numbers = episode_steps // self.motion_length
        return frame_indices, cycle_numbers

    def _current_ref_state(self, env_idx=None):
        """Reference state aligned with current episode length for the given envs."""
        if env_idx is None:
            env_idx = self._all_env_indices()
        episode_steps = self._effective_motion_steps(env_idx)
        frame_indices, cycle_numbers = self._ref_indices_from_episode_steps(episode_steps)
        ref = self._get_ref_state_at_frame(env_idx, frame_indices, cycle_numbers)
        return self._apply_warmup_standing(ref, env_idx)

    def _compute_targets(self):
        self.scene.clear_debug_objects()

        env_idx = self._all_env_indices()
        targets = []
        for i, motion_target_index in enumerate(self.motion_target_indices):  # noqa: B905
            ref = self._target_ref_state(env_idx, motion_target_index)
            target_obs = torch.cat([ref["root_pos"], ref["root_quat"], ref["joint_rot"]], dim=-1)
            targets.append(target_obs)
            if self.debug:
                self._update_target_ghost(i, ref)

        targets = torch.cat(targets, dim=-1)

        return targets

    def _target_ref_state(self, env_idx, motion_target_index):
        target_episode_steps = self.episode_length_buf[env_idx] + motion_target_index
        target_motion_steps = self._effective_motion_steps_from_episode_steps(env_idx, target_episode_steps)
        frame_indices, cycle_numbers = self._ref_indices_from_episode_steps(target_motion_steps)
        ref = self._get_ref_state_at_frame(env_idx, frame_indices, cycle_numbers)
        return self._apply_warmup_standing(ref, env_idx, episode_steps=target_episode_steps)

    def _update_target_ghost(self, target_index: int, ref: TensorDict):
        self.target_ghosts[target_index].set_pos(ref["root_pos"])
        self.target_ghosts[target_index].set_quat(ref["root_quat"])
        self.target_ghosts[target_index].set_dofs_position(
            ref["joint_rot"],
            dofs_idx_local=self._ref_joint_rot_dofs(),
        )

    def step(self, actions):
        ret = super().step(actions)

        if self.debug:
            self._sync_debug_ghost()

        return ret

    def _sync_debug_ghost(self):
        # Use the previous reference step so visual ghosts do not snap forward at cycle boundaries.
        env_idx = self._all_env_indices()
        ref_step = (self._effective_motion_steps(env_idx) - 1).clamp(min=0)
        ghost_frame, ghost_cycle = self._ref_indices_from_episode_steps(ref_step)
        self._sync_with_ref_motion(
            env_idx,
            cycle_num=ghost_cycle,
            ghost=True,
            frame_indices=ghost_frame,
        )

    def check_termination(self):
        self.reset_buf = torch.zeros(
            (self.num_envs,),
            device=self.device,
            dtype=torch.bool,
        )

        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf = self.reset_buf | self.time_out_buf

        if self.eval:
            return

        # Falling over
        self.reset_buf = self.reset_buf | torch.any(
            torch.norm(
                self.link_contact_forces[:, self.termination_contact_link_indices, :],
                dim=-1,
            )
            > TERMINATION_CONTACT_FORCE,
            dim=1,
        )

        # Weird positions
        self.reset_buf |= torch.logical_or(
            torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"],
            torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"],
        )
        self.reset_buf |= self.base_pos[:, 2] < self.env_cfg["termination_if_height_lower_than"]

        env_idx = self._all_env_indices()
        ref = self._current_ref_state(env_idx)
        root_pos = self.robot.get_pos()
        dist_to_target = root_pos - ref["root_pos"]
        root_pos_fail_mask = (dist_to_target**2).sum(dim=-1) > self.env_cfg.get("terminate_dist_threshold", 1.0) ** 2
        self.reset_buf |= root_pos_fail_mask

    def compute_observations(self):
        # TODO: in paper IMU readings are rpy not quat, but this should be ok
        root_pos = self.robot.get_pos()
        root_rot = self.robot.get_quat()
        joint_rot = self.robot.get_qpos()
        joint_rot_not_pos = joint_rot[:, ROOT_QPOS_DIM:]  # exclude global base pos/rot DOFs
        obs = [
            root_pos,
            root_rot,
            joint_rot_not_pos,
            self.actions,
        ]

        self.obs_buf = torch.cat(obs, axis=-1)
        self.obs_buf = torch.clip(self.obs_buf, -OBS_CLIP, OBS_CLIP)

        self.obs_history_buf = torch.cat(
            [self.obs_history_buf[:, self.num_single_obs :], self.obs_buf.detach()],
            dim=1,
        )

    def get_observations(self):
        obs = torch.cat([self.obs_history_buf, self._compute_targets()], dim=-1)
        return TensorDict({"policy": obs})

    # Rewards taken directly from https://github.com/erwincoumans/motion_imitation

    # Reward for matching joint rotations
    # Rotations are expressed in joint coordinates
    def _reward_joint_rot(self, *, ghost=False):
        entity = self._reward_entity(ghost=ghost)
        joint_rot = entity.get_dofs_position()[:, REF_JOINT_ROT_NO_POS_START:]
        ref = self._current_ref_state_for_rewards()
        target_joint_rot = ref["joint_rot"]

        diff = joint_rot - target_joint_rot

        reward = torch.exp(JOINT_ROT_REWARD_SCALE * torch.sum(diff**2, dim=-1))

        return reward

    # Reward for matching joint velocicities
    # Velocities are expressed in joint coordinates
    def _reward_joint_vel(self, *, ghost=False):
        entity = self._reward_entity(ghost=ghost)
        joint_vel = entity.get_dofs_velocity()[:, REF_JOINT_ROT_NO_POS_START:]
        ref = self._current_ref_state_for_rewards()
        target_joint_vel = ref["joint_vel"][:, REF_JOINT_ROT_NO_POS_START:]

        diff = joint_vel - target_joint_vel
        reward = torch.exp(JOINT_VEL_REWARD_SCALE * torch.sum(diff**2, dim=-1))

        return reward

    # Reward for matching end-effectors positions (feet positions)
    # Positions are relative to the root of the robot
    def _reward_end_effector_pos(self, *, ghost=False):
        entity = self._reward_entity(ghost=ghost)
        feet_links = self._feet_link_indices(entity)
        feet_pos = entity.get_links_pos(links_idx_local=feet_links)

        relative_feet_pos = feet_pos - entity.get_pos().unsqueeze(1)

        ref = self._current_ref_state_for_rewards()

        relative_target_feet_pos = ref["end_effector_pos"].reshape(-1, NUM_FEET, XYZ_DIM) - ref["root_pos"].unsqueeze(1)

        root_rot = entity.get_quat()
        root_heading = calc_heading_from_quat(root_rot)
        target_root_heading = calc_heading_from_quat(ref["root_quat"])

        heading_diff = target_root_heading - root_heading
        quat_diff = calc_quat_from_heading(heading_diff)
        quat_diff = torch.repeat_interleave(inv_quat(quat_diff), NUM_FEET, dim=0)

        relative_target_feet_pos = transform_by_quat(relative_target_feet_pos.reshape(-1, XYZ_DIM), quat_diff).reshape(
            -1,
            NUM_FEET,
            XYZ_DIM,
        )

        diff = relative_feet_pos - relative_target_feet_pos
        diff = diff.reshape(diff.shape[0], -1)
        reward = torch.exp(END_EFFECTOR_REWARD_SCALE * torch.sum(diff**2, dim=-1))

        return reward

    # Reward for matching root position and orientation in world reference
    def _reward_root_pose(self, *, ghost=False):
        entity = self._reward_entity(ghost=ghost)
        root_world_pos = entity.get_pos()
        root_world_quat = entity.get_quat()

        ref = self._current_ref_state_for_rewards()

        diff_pos = root_world_pos - ref["root_pos"]
        reward_pos = torch.exp(ROOT_POS_REWARD_SCALE * torch.sum(diff_pos**2, dim=-1))

        rot_diff_quat = transform_quat_by_quat(root_world_quat, inv_quat(ref["root_quat"]))
        rot_diff_axis_angle = quaternion_to_axis_angle(rot_diff_quat)
        rot_error = torch.norm(rot_diff_axis_angle, dim=-1)
        reward_rot = torch.exp(ROOT_ROT_REWARD_SCALE * rot_error**2)
        return reward_pos + reward_rot

    # Reward for matching root linear and angular velocities in world reference
    def _reward_root_vel(self, *, ghost=False):
        entity = self._reward_entity(ghost=ghost)
        root_world_vel = entity.get_vel()
        root_world_ang_vel = entity.get_ang()

        ref = self._current_ref_state_for_rewards()

        diff_vel = root_world_vel - ref["root_vel"]
        reward_vel = torch.exp(ROOT_VEL_REWARD_SCALE * torch.sum(diff_vel**2, dim=-1))

        diff_ang_vel = root_world_ang_vel - ref["root_ang_vel"]
        reward_ang_vel = torch.exp(ROOT_ANG_VEL_REWARD_SCALE * torch.sum(diff_ang_vel**2, dim=-1))

        return reward_vel + reward_ang_vel

    def _reward_smooth_actions(self):
        diff = self.actions - self.last_actions
        reward = torch.exp(SMOOTH_ACTION_REWARD_SCALE * torch.sum(diff**2, dim=-1))
        return reward

    def _reward_entity(self, *, ghost: bool):
        return self.ghost if ghost else self.robot

    def _current_ref_state_for_rewards(self):
        return self._current_ref_state(self._all_env_indices())
