import copy
import os
import pickle
from dataclasses import dataclass

import genesis as gs
import torch
import tyro
from rsl_rl.runners import OnPolicyRunner

from go2_genesis.wrappers.rl_wrapper import Go2


@dataclass
class EvalConfig:
    ckpt_file: tyro.conf.Positional[str]
    headless: bool = False
    cpu: bool = False
    record: bool = False
    num_episodes: int = 1


def export_policy_as_jit(actor_critic, path, name):
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, f"{name}.pt")
    model = copy.deepcopy(actor_critic.actor).to("cpu")
    traced_script_module = torch.jit.script(model)
    traced_script_module.save(path)


def main(args):
    gs.init(backend=gs.cpu if args.cpu else gs.gpu, logging_level="warning")

    log_dir = os.path.dirname(args.ckpt_file)

    with open(os.path.join(log_dir, "cfgs.pkl"), "rb") as cfg_file:
        env_cfg, obs_cfg, reward_cfg, command_cfg, policy_cfg = pickle.load(cfg_file)

    device = "cuda" if not args.cpu else "cpu"
    env = Go2(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=not args.headless,
        eval=True,
        debug=False,
        device=device,
    )

    args.max_iterations = 1
    runner = OnPolicyRunner(env, policy_cfg, log_dir, device=device)

    runner.load(args.ckpt_file)

    policy = runner.get_inference_policy(device=device)

    env.reset()
    obs = env.get_observations()

    with torch.no_grad():
        stop = False
        if args.record:
            env.start_recording(record_internal=False)

        for _ in range(args.num_episodes):
            while not stop:
                actions = policy(obs)
                obs, rews, dones, infos = env.step(actions)

                if dones[0]:
                    stop = True

    if args.record:
        env.stop_recording(os.path.join(log_dir, "recording.mp4"))


if __name__ == "__main__":
    args = tyro.cli(EvalConfig)
    main(args)
