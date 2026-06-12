import tempfile

import torch

from go2_genesis.cfgs import get_default_cfgs
from go2_genesis.ghost_env import GhostEnv
from go2_genesis.locomotion_env import GO2_NUM_DOFS, OBS_CLIP, LocoEnv
from go2_genesis.utils import QUAT_DIM, XYZ_DIM

X_AXIS = 0
Z_AXIS = -1


def _get_obs_cfgs():
    """Default cfgs with 43-dim observation space."""
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_default_cfgs()
    # dof_pos + base_quat + dof_vel + base_ang_vel + actions
    obs_cfg["num_obs"] = GO2_NUM_DOFS + QUAT_DIM + GO2_NUM_DOFS + XYZ_DIM + GO2_NUM_DOFS

    return env_cfg, obs_cfg, reward_cfg, command_cfg


class Go2ObservationWrapper(GhostEnv):
    """Lightweight env that mirrors state from another LocoEnv to extract observations.

    Uses a fixed, 43-dim observation format:
        [relative_dof_pos (12), base_quat (4), dof_vel (12), base_ang_vel (3), actions (12)]

    This is an opinionated observation format chosen for trajectory gathering.
    It does not correspond to the observation space of any specific policy.
    """

    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer, eval, debug, device="cuda"):
        super().__init__(num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer, eval, debug, device)

    @classmethod
    def from_go2_env(cls, src_env: LocoEnv, *, copy_state: bool = True):
        env_cfg, obs_cfg, reward_cfg, command_cfg = _get_obs_cfgs()

        wrapper = cls(
            num_envs=src_env.num_envs,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            show_viewer=not src_env.headless,
            eval=src_env.eval,
            debug=src_env.debug,
            device=src_env.device,
        )

        if copy_state:
            wrapper.copy_state_from_env(src_env)

        return wrapper

    def copy_state_from_env(self, src_env: LocoEnv):
        with tempfile.NamedTemporaryFile(suffix=".ckpt") as tmp:
            src_env.save_state(tmp.name)
            self.set_state(tmp.name)

        self.episode_length_buf[:] = src_env.episode_length_buf
        self.actions[:] = src_env.actions
        self._update_buffers()
        self.compute_observations()

    def compute_observations(self):
        self.obs_buf = torch.cat(
            [
                self.dof_pos - self.default_dof_pos,
                self.base_quat,
                self.dof_vel,
                self.base_ang_vel,
                self.actions,
            ],
            dim=-1,
        )

        self.obs_buf = torch.clip(self.obs_buf, -OBS_CLIP, OBS_CLIP)
        self._append_observation_history()

    @staticmethod
    def _quat_to_tan_norm(quat: torch.Tensor) -> torch.Tensor:
        """Convert quaternions to tangent/normal vectors with shape ``(N, 6)``."""
        ref_tan = torch.zeros((quat.shape[0], XYZ_DIM), device=quat.device, dtype=quat.dtype)
        ref_tan[..., X_AXIS] = 1
        tan = Go2ObservationWrapper._quat_rotate(quat, ref_tan)

        ref_norm = torch.zeros((quat.shape[0], XYZ_DIM), device=quat.device, dtype=quat.dtype)
        ref_norm[..., Z_AXIS] = 1
        norm = Go2ObservationWrapper._quat_rotate(quat, ref_norm)

        norm_tan = torch.cat([tan, norm], dim=len(tan.shape) - 1)
        return norm_tan

    @staticmethod
    def _quat_rotate(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        """Rotate batched 3D vectors by batched ``wxyz`` quaternions."""
        shape = quat.shape
        q_w = quat[:, 0]
        q_vec = quat[:, 1:]
        a = vec * (2.0 * q_w**2 - 1.0)[..., None]
        b = torch.cross(q_vec, vec, dim=-1) * q_w[..., None] * 2.0
        c = q_vec * (q_vec.reshape(shape[0], 1, XYZ_DIM) @ vec.reshape(shape[0], XYZ_DIM, 1)).squeeze(-1) * 2.0
        return a + b + c

    def get_observations(self):
        return {"obs": self.obs_history_buf, "time": self.episode_length_buf}

    def get_links_pos(self) -> torch.Tensor:
        # squeeze out the batch dimension as we always evaluate on a single env
        return self.robot.get_links_pos().squeeze(0)

    def get_links_rot(self) -> torch.Tensor:
        # squeeze out the batch dimension as we always evaluate on a single env
        quat = self.robot.get_links_quat().squeeze(0)
        return self._quat_to_tan_norm(quat)  # (num_links, 6)

    def get_dataset_payload(self):
        return {
            "dof_pos": self.dof_pos.detach().clone(),
            "dof_vel": self.dof_vel.detach().clone(),
            "base_pos": self.base_pos.detach().clone(),
            "base_quat": self.base_quat.detach().clone(),
            "base_lin_vel": self.base_lin_vel.detach().clone(),
            "base_ang_vel": self.base_ang_vel.detach().clone(),
            "actions": self.actions.detach().clone(),
            "frame": self.episode_length_buf.detach().clone(),
            "links_pos": self.get_links_pos().detach().clone(),
            "links_rot": self.get_links_rot().detach().clone(),
        }
