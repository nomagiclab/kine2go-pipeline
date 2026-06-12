from __future__ import annotations

import pickle
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import genesis as gs
import torch
import tyro

from go2_genesis.locomotion_env import LocoEnv
from go2_genesis.trajectory_io import TRAJ_FILENAME, load_traj_frames
from go2_genesis.wrappers.rl_wrapper import Go2
from motion_imitation.imitation_wrapper import ImitationWrapper

INIT_STATE_FILENAME = "init_state.pkl"


@dataclass
class CutTrajectoryCfg:
    trajectory_path: Path
    """Path to an input trajectory folder (e.g. traj_0000)."""

    first_frame: int
    """First frame to keep (inclusive)."""

    last_frame: int
    """Last frame to keep (inclusive). Use -1 to keep through the final frame."""

    output_dir: Path | None = None
    """Optional parent output directory. Output is written to output_dir/<traj_name>/."""

    cpu: bool = False
    log_level: Literal["debug", "info", "warning", "error"] = "warning"


def _load_env_from_cfg(cfg_path: Path, *, device: str) -> LocoEnv:
    with cfg_path.open("rb") as cfg_file:
        env_cfg, obs_cfg, reward_cfg, command_cfg, policy_cfg = pickle.load(cfg_file)

    # Rewards are irrelevant for replay; keep a minimal stable config.
    reward_cfg = {
        "tracking_sigma": 0.25,
        "soft_dof_pos_limit": 0.9,
        "base_height_target": 0.3,
        "reward_scales": {},
    }

    if "motion_range" in policy_cfg:
        motion_path = cfg_path.parent / "motion.npy"
        if not motion_path.is_file():
            raise FileNotFoundError(f"motion.npy required by cfgs.pkl was not found: {motion_path}")
        motion_start, motion_end = policy_cfg["motion_range"]
        motions = [(str(motion_path), motion_start, motion_end)]
        return ImitationWrapper(
            motions=motions,
            num_envs=1,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            show_viewer=False,
            eval=True,
            debug=False,
            device=device,
        )

    return Go2(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,
        eval=True,
        debug=False,
        device=device,
    )


def _validate_inputs(cfg: CutTrajectoryCfg) -> tuple[Path, Path, Path]:
    traj_dir = cfg.trajectory_path.resolve()
    if not traj_dir.is_dir():
        raise FileNotFoundError(f"Trajectory directory not found: {traj_dir}")

    init_state_path = traj_dir / INIT_STATE_FILENAME
    traj_path = traj_dir / TRAJ_FILENAME
    cfg_path = traj_dir.parent / "cfgs.pkl"

    if not init_state_path.is_file():
        raise FileNotFoundError(f"Initial state not found: {init_state_path}")
    if not traj_path.is_file():
        raise FileNotFoundError(f"Trajectory file not found: {traj_path}")
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Dataset config not found: {cfg_path}")

    return traj_dir, init_state_path, cfg_path


def _resolve_and_validate_frame_range(first_frame: int, last_frame: int, num_frames: int) -> tuple[int, int]:
    if last_frame == -1:
        last_frame = num_frames - 1
    elif last_frame < -1:
        raise ValueError(f"last_frame must be >= 0 or -1, got {last_frame}")

    if first_frame < 0:
        raise ValueError(f"first_frame must be >= 0, got {first_frame}")
    if last_frame < first_frame:
        raise ValueError(f"last_frame must be >= first_frame, got {last_frame} < {first_frame}")
    if last_frame >= num_frames:
        raise ValueError(
            f"last_frame out of range: got {last_frame}, but trajectory has {num_frames} frames "
            f"(valid last index is {num_frames - 1})",
        )
    return first_frame, last_frame


def _extract_action(frame: dict, *, frame_idx: int, env: LocoEnv) -> torch.Tensor:
    if "actions" not in frame:
        raise ValueError(f"Frame {frame_idx} is missing 'actions'; cannot replay trajectory.")

    actions = frame["actions"]
    if not isinstance(actions, torch.Tensor):
        actions = torch.as_tensor(actions, dtype=torch.float32)
    else:
        actions = actions.to(dtype=torch.float32)

    if actions.ndim == 1:
        actions = actions.unsqueeze(0)
    if actions.ndim != 2:
        raise ValueError(f"Frame {frame_idx} has invalid action shape {tuple(actions.shape)}; expected 2D tensor.")
    if actions.shape[0] != env.num_envs:
        raise ValueError(
            f"Frame {frame_idx} action batch mismatch: got {actions.shape[0]} envs, expected {env.num_envs}.",
        )
    if actions.shape[1] != env.num_actions:
        raise ValueError(
            f"Frame {frame_idx} action dim mismatch: got {actions.shape[1]}, expected {env.num_actions}.",
        )

    return actions.to(env.device)


def _replay_to_first_frame(env: LocoEnv, *, init_state_path: Path, frames: list[dict], first_frame: int) -> None:
    env.set_state(str(init_state_path))
    env._update_buffers()

    if first_frame == 0:
        return

    # Gathered trajectories store observations before each policy step; thus the action that
    # advances frame i -> i+1 is stored in frame i+1.
    with torch.no_grad():
        for frame_idx in range(1, first_frame + 1):
            actions = _extract_action(frames[frame_idx], frame_idx=frame_idx, env=env)
            env.step(actions)


def _save_cut_result(
    *,
    env: LocoEnv,
    cut_frames: list[dict],
    source_traj_dir: Path,
    destination_traj_dir: Path,
) -> None:
    source_resolved = source_traj_dir.resolve()
    destination_resolved = destination_traj_dir.resolve()

    if destination_resolved == source_resolved:
        with tempfile.TemporaryDirectory(prefix=f"{source_traj_dir.name}_cut_", dir=str(source_traj_dir.parent)) as tmp:
            tmp_dir = Path(tmp)
            tmp_init = tmp_dir / INIT_STATE_FILENAME
            tmp_traj = tmp_dir / TRAJ_FILENAME
            env.save_state(str(tmp_init))
            torch.save(cut_frames, tmp_traj)
            tmp_init.replace(source_traj_dir / INIT_STATE_FILENAME)
            tmp_traj.replace(source_traj_dir / TRAJ_FILENAME)
        return

    destination_traj_dir.mkdir(parents=True, exist_ok=True)
    env.save_state(str(destination_traj_dir / INIT_STATE_FILENAME))
    torch.save(cut_frames, destination_traj_dir / TRAJ_FILENAME)


def _resolve_destination_traj_dir(trajectory_dir: Path, output_dir: Path | None) -> Path:
    if output_dir is None:
        return trajectory_dir
    return output_dir.resolve() / trajectory_dir.name


def main(cfg: CutTrajectoryCfg) -> None:
    trajectory_dir, init_state_path, cfg_path = _validate_inputs(cfg)
    traj_path = trajectory_dir / TRAJ_FILENAME

    frames = load_traj_frames(traj_path)
    first_frame, last_frame = _resolve_and_validate_frame_range(cfg.first_frame, cfg.last_frame, len(frames))

    backend = gs.cpu if cfg.cpu else gs.gpu
    gs.init(backend=backend, logging_level=cfg.log_level)
    device = "cpu" if cfg.cpu else "cuda"

    env = _load_env_from_cfg(cfg_path, device=device)
    _replay_to_first_frame(env, init_state_path=init_state_path, frames=frames, first_frame=first_frame)

    cut_frames = frames[first_frame : last_frame + 1]
    destination_traj_dir = _resolve_destination_traj_dir(trajectory_dir, cfg.output_dir)
    _save_cut_result(
        env=env,
        cut_frames=cut_frames,
        source_traj_dir=trajectory_dir,
        destination_traj_dir=destination_traj_dir,
    )

    print(
        f"Saved cut trajectory {trajectory_dir.name}: frames [{first_frame}, {last_frame}] "
        f"({len(cut_frames)} total) -> {destination_traj_dir}",
    )


if __name__ == "__main__":
    args = tyro.cli(CutTrajectoryCfg)
    main(args)
