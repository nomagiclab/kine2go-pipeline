import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import genesis as gs
import torch
import tyro
from rsl_rl.runners import OnPolicyRunner

from go2_genesis.locomotion_env import LocoEnv
from go2_genesis.trajectory_io import TRAJ_FILENAME
from go2_genesis.wrappers.observation_wrapper import Go2ObservationWrapper
from go2_genesis.wrappers.rl_wrapper import Go2
from motion_imitation.imitation_wrapper import ImitationWrapper


@dataclass
class GatherCfg:
    # path to training checkpoint (e.g., dataset/data/<motion>/logs/model.pt)
    ckpt_path: str

    num_trajectories: int = 10
    max_steps_per_trajectory: int = 400
    save_every_n_steps: int = 1

    output_dir: str = "trajectories"

    headless: bool = True
    cpu: bool = False
    log_level: Literal["debug", "info", "warning", "error"] = "warning"


@dataclass(frozen=True)
class CheckpointPaths:
    ckpt_path: Path
    clip_dir: Path
    log_dir: Path
    cfg_path: Path
    motion_path: Path


def _resolve_checkpoint_paths(checkpoint_path: str | Path) -> CheckpointPaths:
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found at: {ckpt_path}")

    log_dir = ckpt_path.parent
    clip_dir = log_dir.parent if log_dir.name == "logs" else log_dir
    cfg_path = clip_dir / "cfgs.pkl"
    motion_path = clip_dir / "motion.npy"

    if not cfg_path.exists():
        raise FileNotFoundError(f"cfgs.pkl not found at: {cfg_path}")

    return CheckpointPaths(
        ckpt_path=ckpt_path,
        clip_dir=clip_dir,
        log_dir=log_dir,
        cfg_path=cfg_path,
        motion_path=motion_path,
    )


def _load_env_and_policy_from_checkpoint(paths: CheckpointPaths, device: str, *, headless: bool):

    with paths.cfg_path.open("rb") as cfg_file:
        env_cfg, obs_cfg, reward_cfg, command_cfg, policy_cfg = pickle.load(cfg_file)

    # This is just in case, if the reward structure has changed. We don't use rewards here anyway
    reward_cfg = {
        "tracking_sigma": 0.25,
        "soft_dof_pos_limit": 0.9,
        "base_height_target": 0.3,
        "reward_scales": {},
    }

    if "motion_range" in policy_cfg:
        if not paths.motion_path.exists():
            raise FileNotFoundError(f"motion.npy not found at: {paths.motion_path}")

        motion_path = str(paths.motion_path)
        motion_start, motion_end = policy_cfg["motion_range"]
        motions = [(motion_path, motion_start, motion_end)]

        # disable init state randomization to avoid weird movements in gathered trajectories
        env_cfg["perturb_init_state"] = False
        env_cfg["ref_state_init_prob"] = 1.0
        # but we still want to randomize the trajectory heading to make the trajectories more diverse
        env_cfg["randomize_trajectory_heading"] = True

        env = ImitationWrapper(
            motions=motions,
            num_envs=1,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            show_viewer=not headless,
            eval=True,
            debug=False,
            device=device,
        )
    else:
        env = Go2(
            num_envs=1,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            show_viewer=not headless,
            eval=True,
            debug=False,
            device=device,
        )

    log_dir = str(paths.log_dir)
    runner = OnPolicyRunner(env, policy_cfg, log_dir, device=device)
    runner.load(str(paths.ckpt_path), map_location=torch.device(device))
    policy = runner.get_inference_policy(device=device)

    return env, policy


def _should_save_step(args: GatherCfg, step_idx: int):
    return args.save_every_n_steps > 0 and (step_idx % args.save_every_n_steps == 0)


def _append_frame(env: LocoEnv, obs_env: Go2ObservationWrapper, frames: list) -> None:
    obs_env.copy_state_from_env(env)
    frames.append(obs_env.get_dataset_payload())


def _write_trajectory_file(traj_dir: Path, frames: list) -> None:
    torch.save(frames, traj_dir / TRAJ_FILENAME)


def main(args: GatherCfg):
    backend = gs.cpu if args.cpu else gs.gpu
    gs.init(backend=backend, logging_level=args.log_level)

    device = "cpu" if args.cpu else "cuda"

    paths = _resolve_checkpoint_paths(args.ckpt_path)
    env, policy = _load_env_and_policy_from_checkpoint(
        paths,
        device=device,
        headless=args.headless,
    )

    obs_env = Go2ObservationWrapper.from_go2_env(env, copy_state=True)

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    try:
        shutil.copy(str(paths.cfg_path), str(out_root / "cfgs.pkl"))
        if paths.motion_path.exists():
            shutil.copy(str(paths.motion_path), str(out_root / "motion.npy"))
    except Exception as e:
        raise RuntimeError(f"Failed to copy config files to output_dir: {e}") from e

    with torch.no_grad():
        for traj_idx in range(args.num_trajectories):
            traj_dir = out_root / f"traj_{traj_idx:04d}"
            traj_dir.mkdir(parents=True, exist_ok=True)

            env.reset()
            init_state_path = traj_dir / "init_state.pkl"
            env.save_state(init_state_path)

            obs = env.get_observations()
            step_idx = 0
            frames: list = []

            while step_idx < args.max_steps_per_trajectory:
                actions = policy(obs)

                if _should_save_step(args, step_idx):
                    _append_frame(env, obs_env, frames)

                obs, _, _, _ = env.step(actions)
                step_idx += 1

            # save the last step as well (end of episode)
            _append_frame(env, obs_env, frames)

            _write_trajectory_file(traj_dir, frames)


if __name__ == "__main__":
    args = tyro.cli(GatherCfg)
    main(args)
