import genesis as gs
import torch

from go2_genesis.locomotion_env import (
    GO2_NUM_DOFS,
    QUAT_DIM,
    XYZ_DIM,
    LocoEnv,
)


class GhostEnv(LocoEnv):
    def _prepare_scene(self):
        super()._prepare_scene()

        self.ghost = None
        self.target_ghosts = []
        if self.debug:
            self.ghost = self.scene.add_entity(
                gs.morphs.URDF(
                    file=self.env_cfg["urdf_path"],
                    merge_fixed_links=True,
                    links_to_keep=self.env_cfg["links_to_keep"],
                    pos=self.base_init_pos.cpu().numpy(),
                    quat=self.base_init_quat.cpu().numpy(),
                    collision=False,
                ),
                surface=gs.surfaces.Default(color=(1, 0, 0, 0.5)),
                visualize_contact=False,
            )

    def set_ghost(self, dofs_position, base_position, base_quat):
        assert self.num_envs == 1, "Ghost is only supported for single environment"
        assert dofs_position.shape == (1, GO2_NUM_DOFS)
        assert base_position.shape == (1, XYZ_DIM)
        assert base_quat.shape == (1, QUAT_DIM)

        if self.ghost is not None:
            self.ghost.set_dofs_position(
                dofs_position,
                envs_idx=None,
                dofs_idx_local=torch.arange(6, 18, device=self.device),
            )
            self.ghost.set_pos(base_position, envs_idx=None)
            self.ghost.set_quat(base_quat, envs_idx=None)
