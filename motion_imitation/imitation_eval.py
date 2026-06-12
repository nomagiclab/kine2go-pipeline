import os
import pickle
from dataclasses import dataclass

import genesis as gs
import torch
import tyro
from rsl_rl.runners import OnPolicyRunner

from motion_imitation.imitation_wrapper import ImitationWrapper


@dataclass
class EvalConfig:
    ckpt_file: tyro.conf.Positional[str]
    headless: bool = False
    cpu: bool = False
    record: bool = False
    num_episodes: int = 1
    debug: bool = False


def main(args):
    gs.init(backend=gs.cpu if args.cpu else gs.gpu, logging_level="warning", performance_mode=False)

    log_dir = os.path.dirname(args.ckpt_file)

    with open(os.path.join(log_dir, "cfgs.pkl"), "rb") as cfg_file:
        env_cfg, obs_cfg, reward_cfg, command_cfg, policy_cfg = pickle.load(cfg_file)

    motion_path = os.path.join(log_dir, "motion.npy")
    motion_start, motion_end = policy_cfg["motion_range"]
    eval_motion = [(motion_path, motion_start, motion_end)]

    device = "cuda"
    if args.cpu:
        device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"

    env_cfg["perturb_init_state"] = True
    env = ImitationWrapper(
        eval_motion,
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=not args.headless,
        eval=True,
        debug=args.debug,
        device=device,
    )

    args.max_iterations = 1
    runner = OnPolicyRunner(env, policy_cfg, log_dir, device=device)

    runner.load(args.ckpt_file, map_location=device)

    policy = runner.get_inference_policy(device=device)

    env.reset()
    obs = env.get_observations()

    with torch.no_grad():
        if args.record:
            env.start_recording(record_internal=False)

        for _ in range(args.num_episodes):
            stop = False

            while not stop:
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)

                if dones[0]:
                    stop = True

    if args.record:
        env.stop_recording(os.path.join(log_dir, "recording.mp4"))


if __name__ == "__main__":
    args = tyro.cli(EvalConfig)
    main(args)
