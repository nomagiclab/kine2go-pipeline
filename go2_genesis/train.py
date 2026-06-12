import os
import pickle
import shutil
from dataclasses import dataclass
from typing import Literal

import genesis as gs
import torch
import tyro
from tyro.conf import Positional

from go2_genesis.cfgs import get_backflip_cfgs, get_handstand_cfgs, get_ppo_cfg, get_walking_cfgs
from go2_genesis.logging_utils import LoggingOnPolicyRunner
from go2_genesis.wrappers.rl_wrapper import Go2


@dataclass
class TrainCfg:
    exp_name: Positional[str]
    num_envs: int = 4096
    max_iterations: int = 1000
    wandb_mode: Literal["online", "offline", "disabled"] = "offline"
    task: Literal["walking", "backflip", "handstand"] = "walking"
    cpu: bool = False
    entity: str = "quadruped-rl"
    group: str | None = None
    project: str = "genesis"


def main(args: TrainCfg):
    gs.init(
        backend=gs.cpu if args.cpu else gs.gpu,
        logging_level="warning",
    )

    training_device = "cuda"
    if args.cpu:
        training_device = "cpu"
    if torch.backends.mps.is_available():
        training_device = "mps"

    log_dir = f"logs/{args.exp_name}"

    if args.task == "walking":
        cfgs = get_walking_cfgs()
    elif args.task == "handstand":
        cfgs = get_handstand_cfgs()
    else:
        cfgs = get_backflip_cfgs()

    env_cfg, obs_cfg, reward_cfg, command_cfg = cfgs

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    env = Go2(
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

    with open(f"{log_dir}/cfgs.pkl", "wb") as cfg_file:
        pickle.dump(
            [env_cfg, obs_cfg, reward_cfg, command_cfg, policy_cfg],
            cfg_file,
        )

    runner = LoggingOnPolicyRunner(env, policy_cfg, log_dir, device=training_device)

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    args = tyro.cli(TrainCfg)
    main(args)
