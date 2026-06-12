from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import numpy as np

XYZ_DIM = 3
Z_AXIS_INDEX = 2
NUM_FEET = 4
DEFAULT_START_FRAME = 0
TRAJECTORY_PATH_PATTERN = r"(.+):(\d+)$"


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for dataset-specific retargeting parameters."""

    anatomy_type: Literal["dog", "horse"]
    pelvis_id: int
    neck_id: int
    hip_ids: list[int]
    toe_ids: list[int]
    ref_pos_scale: float
    ref_forward_dir_offset: tuple[float, float, float]
    pos_offset: tuple[float, float, float]
    coord_rot_euler: tuple[float, float, float]
    root_rot_euler: tuple[float, float, float]


@dataclass(frozen=True)
class MotionData:
    """Processed motion frames and optional per-frame timestamps."""

    joint_positions: list[np.ndarray]
    timestamps: np.ndarray | None = None


class RetargetingDataset(ABC):
    """Abstract base class for retargeting datasets."""

    @property
    @abstractmethod
    def config(self) -> DatasetConfig:
        """Dataset-specific retargeting configuration."""
        pass

    @abstractmethod
    def load_motion_data(
        self,
        motion_path: str,
        frame_start: int | None = None,
        frame_end: int | None = None,
    ) -> MotionData:
        """Load motion data from file and process it.

        Args:
            motion_path: Path to the motion data file.
            frame_start: Starting frame index (None for beginning).
            frame_end: Ending frame index (None for end).

        Returns:
            Processed joint positions with optional per-frame timestamps.
        """
        pass
