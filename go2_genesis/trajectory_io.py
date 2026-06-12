"""Read ``traj.pkl`` contents: a plain list of per-frame dicts, or legacy ``{"frames": ...}`` wrappers."""

from __future__ import annotations

from pathlib import Path

import torch

TRAJ_FILENAME = "traj.pkl"


def load_traj_frames(path: Path | str) -> list[dict]:
    """Return the frame list from ``traj.pkl``.

    Accepts:
        - Current format: ``torch.save([frame0, frame1, ...], ...)``
        - Legacy: ``{"frames": [...]}`` with optional ``kind`` / ``version`` keys.
    """
    path = Path(path)
    data = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(data, list):
        frames = data
    elif isinstance(data, dict):
        if data.get("kind") == "states":
            raise ValueError(
                f"{path}: legacy physics-checkpoint trajectories (kind=states) are no longer supported.",
            )
        if "frames" not in data:
            raise ValueError(f"{path}: expected a list of frames or a dict with a 'frames' key")
        frames = data["frames"]
    else:
        raise TypeError(f"{path}: expected list or dict at root, got {type(data)}")

    if not isinstance(frames, list):
        raise TypeError(f"{path}: frames must be a list, got {type(frames)}")
    if not frames:
        raise ValueError(f"{path}: empty trajectory")
    if isinstance(frames[0], (bytes, bytearray)):
        raise ValueError(f"{path}: physics checkpoint frames are not supported")

    return frames
