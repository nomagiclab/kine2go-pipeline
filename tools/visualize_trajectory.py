import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import genesis as gs
import numpy as np
import torch
import tyro

from go2_genesis.trajectory_io import TRAJ_FILENAME, load_traj_frames
from go2_genesis.wrappers.rl_wrapper import Go2
from motion_imitation.imitation_wrapper import ImitationWrapper


@dataclass
class VisualizeCfg:
    trajectories_path: str
    trajectory_id: int = 0
    output_path: str = "trajectory.mp4"
    video_resolution: tuple[int, int] | None = None
    cpu: bool = False
    log_level: Literal["debug", "info", "warning", "error"] = "warning"


def _load_env_from_cfg(cfg_path: Path, *, device: str, camera_res: tuple[int, int] | None = None):
    with cfg_path.open("rb") as cfg_file:
        env_cfg, obs_cfg, reward_cfg, command_cfg, policy_cfg = pickle.load(cfg_file)

    reward_cfg = {
        "tracking_sigma": 0.25,
        "soft_dof_pos_limit": 0.9,
        "base_height_target": 0.3,
        "reward_scales": {},
    }  # we don't use rewards here anyway

    if "motion_range" in policy_cfg:
        motion_path = str(cfg_path.parent / "motion.npy")
        motion_start, motion_end = policy_cfg["motion_range"]
        motions = [(motion_path, motion_start, motion_end)]
        env = ImitationWrapper(
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
            camera_res=camera_res,
        )
    else:
        env = Go2(
            num_envs=1,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            show_viewer=False,
            eval=True,
            debug=False,
            device=device,
            camera_res=camera_res,
        )

    return env


def _load_trajectory_frames(traj_dir: Path) -> list[dict]:
    traj_path = traj_dir / TRAJ_FILENAME
    if traj_path.is_file():
        frames = load_traj_frames(traj_path)
        for i, fr in enumerate(frames):
            if not isinstance(fr, dict):
                raise ValueError(f"{traj_path}: frame {i} must be a dict, got {type(fr)}")
        return frames

    step_paths = [
        path for path in traj_dir.iterdir() if path.is_file() and path.suffix == ".pkl" and path.stem.isdigit()
    ]
    step_paths.sort(key=lambda path: int(path.stem))
    out: list[dict] = []
    for p in step_paths:
        data = torch.load(str(p), map_location="cpu", weights_only=False)
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict in step file {p}, got {type(data)}")
        out.append(data)
    return out


def _obs_tensor_from_frame(data: dict) -> torch.Tensor:
    if "observation" in data and isinstance(data["observation"], dict) and "obs" in data["observation"]:
        return data["observation"]["obs"]
    if "obs" in data:
        return data["obs"]
    raise ValueError(f"Frame does not contain obs under 'obs' or 'observation.obs': keys={sorted(data.keys())}")


def _extract_robot_state_from_frame(data: dict, env) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if "dof_pos" in data and "base_quat" in data:
        if "links_pos" not in data:
            raise ValueError("Dataset frame missing links_pos (needed for base link position).")
        links_pos = data["links_pos"]
        root_pos = links_pos[0]
        base_position = root_pos.unsqueeze(0).to(torch.float32)
        return (
            data["dof_pos"].to(torch.float32),
            base_position,
            data["base_quat"].to(torch.float32),
        )

    obs = _obs_tensor_from_frame(data)
    if "links_pos" not in data:
        raise ValueError("Frame does not contain links_pos (required for base position).")

    links_pos = data["links_pos"]
    root_pos = links_pos[0]
    base_position = root_pos.unsqueeze(0).to(torch.float32)

    relative_dof_pos = obs[:, :12].to(torch.float32)
    base_quat = obs[:, 12:16].to(torch.float32)

    default_dof_pos = env.default_dof_pos.detach().cpu().to(torch.float32)
    dofs_position = relative_dof_pos + default_dof_pos

    return dofs_position, base_position, base_quat


def _capture_frame(env) -> None:
    robot_pos = env.base_pos[0].detach().cpu().numpy()
    env._floating_camera.set_pose(
        pos=robot_pos + np.array([-1.0, -1.0, 0.5]),
        lookat=robot_pos + np.array([0.0, 0.0, -0.1]),
    )
    env._floating_camera.render()


def main(args: VisualizeCfg):
    root = Path(args.trajectories_path)
    cfg_path = root / "cfgs.pkl"
    traj_dir = root / f"traj_{args.trajectory_id:04d}"
    init_state_path = traj_dir / "init_state.pkl"

    if not root.is_dir():
        raise FileNotFoundError(f"Trajectory root not found: {root}")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Expected cfgs.pkl at: {cfg_path}")
    if not traj_dir.is_dir():
        raise FileNotFoundError(f"Trajectory directory not found: {traj_dir}")
    if not init_state_path.exists():
        raise FileNotFoundError(f"Initial state not found: {init_state_path}")

    frames = _load_trajectory_frames(traj_dir)
    if len(frames) < 1:
        raise ValueError("No trajectory frames found (expected traj.pkl or legacy ######.pkl files).")

    backend = gs.cpu if args.cpu else gs.gpu
    gs.init(backend=backend, logging_level=args.log_level)
    device = "cpu" if args.cpu else "cuda"

    env = _load_env_from_cfg(cfg_path, device=device, camera_res=args.video_resolution)
    if gs.platform == "macOS":
        raise RuntimeError("Headless camera recording is unavailable on macOS in this environment.")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env.set_state(str(init_state_path))
    env._update_buffers()

    env._floating_camera.start_recording()

    for frame_data in frames:
        dofs_pos, base_pos, base_quat = _extract_robot_state_from_frame(frame_data, env)
        env.set_robot(
            dofs_pos.to(env.device),
            base_pos.to(env.device),
            base_quat.to(env.device),
        )
        env.scene.step()
        env._update_buffers()
        _capture_frame(env)

    fps = int(1 / env.dt)
    env._floating_camera.stop_recording(str(output_path), fps=fps)


if __name__ == "__main__":
    cfg = tyro.cli(VisualizeCfg)
    main(cfg)
