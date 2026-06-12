import os
import pickle
import shutil
from dataclasses import dataclass
from typing import Literal

import genesis as gs
import torch
import tyro
from tyro.conf import Positional

from go2_genesis.logging_utils import LoggingOnPolicyRunner
from motion_imitation.config import get_imitation_cfgs, get_ppo_cfg
from motion_imitation.imitation_wrapper import ImitationWrapper


@dataclass
class TrainCfg:
    exp_name: Positional[str]
    motion_path: Positional[str]
    motion_start: int = 0
    motion_end: int | None = None
    num_envs: int = 4096
    max_iterations: int = 1000
    wandb_mode: Literal["online", "offline", "disabled"] = "offline"
    cpu: bool = False
    entity: str = "quadruped-rl"
    group: str | None = None
    project: str = "imitation_learning"


def main(args: TrainCfg):
    gs.init(
        backend=gs.cpu if args.cpu else gs.gpu,
        logging_level="warning",
    )

    log_dir = f"logs/{args.exp_name}"

    env_cfg, obs_cfg, reward_cfg, command_cfg = get_imitation_cfgs()

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    retargetted_motions = [(args.motion_path, args.motion_start, args.motion_end)]

    training_device = "cuda"
    if args.cpu:
        training_device = "cpu"
    if torch.backends.mps.is_available():
        training_device = "mps"

    env = ImitationWrapper(
        retargetted_motions,
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,
        eval=False,
        debug=False,
        device=training_device,
    )

    policy_cfg = get_ppo_cfg(args)
    policy_cfg["motion_range"] = (args.motion_start, args.motion_end)

    with open(f"{log_dir}/cfgs.pkl", "wb") as cfg_file:
        pickle.dump(
            [env_cfg, obs_cfg, reward_cfg, command_cfg, policy_cfg],
            cfg_file,
        )

    shutil.copyfile(
        args.motion_path,
        f"{log_dir}/motion.npy",
    )

    runner = LoggingOnPolicyRunner(env, policy_cfg, log_dir, device=training_device)

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    args = tyro.cli(TrainCfg)
    main(args)
