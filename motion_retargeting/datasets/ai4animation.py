import numpy as np
from scipy.spatial.transform import Rotation as R

from motion_retargeting.datasets.base import (
    DEFAULT_START_FRAME,
    XYZ_DIM,
    DatasetConfig,
    MotionData,
    RetargetingDataset,
)

AI4_ANIMATION_CONFIG = DatasetConfig(
    anatomy_type="dog",
    pelvis_id=0,
    neck_id=3,
    hip_ids=[6, 16, 11, 20],
    toe_ids=[10, 19, 15, 23],
    ref_pos_scale=0.825,
    ref_forward_dir_offset=(0.0, 0.0, 0.04),
    pos_offset=(0.0, 0.0, 0.0),
    coord_rot_euler=(0.5 * np.pi, 0, 0),
    root_rot_euler=(0, 0, 0.47 * np.pi),
)

FREQUENCY = 60.0  # reverse engineered from dog_clips_info.txt


class AI4AnimationDataset(RetargetingDataset):
    """Dataset for AI4Animation."""

    def __init__(self):
        self._config = AI4_ANIMATION_CONFIG
        self._coord_rot = R.from_euler("xyz", self._config.coord_rot_euler, degrees=False)
        self._root_rot = R.from_euler("xyz", self._config.root_rot_euler, degrees=False)
        self._pos_offset = np.array(self._config.pos_offset)

    @property
    def config(self) -> DatasetConfig:
        return self._config

    def load_motion_data(
        self,
        motion_path: str,
        frame_start: int | None = None,
        frame_end: int | None = None,
    ) -> MotionData:
        joint_pos_data = np.loadtxt(motion_path, delimiter=",")

        start_frame = DEFAULT_START_FRAME if (frame_start is None) else frame_start
        end_frame = joint_pos_data.shape[0] if (frame_end is None) else frame_end

        joint_pos_data = joint_pos_data[start_frame:end_frame]
        processed_positions = [self._process_joint_pos(joint_pos) for joint_pos in joint_pos_data]
        timestamps = np.arange(len(processed_positions), dtype=float) / FREQUENCY
        return MotionData(joint_positions=processed_positions, timestamps=timestamps)

    def _process_joint_pos(self, ref_joint_pos: np.ndarray) -> np.ndarray:
        pose_to_process = np.reshape(ref_joint_pos, (-1, XYZ_DIM)).copy()
        n_joints = pose_to_process.shape[0]

        for joint_idx in range(n_joints):
            curr_joint = pose_to_process[joint_idx]
            curr_joint = self._coord_rot.apply(curr_joint)
            curr_joint = self._root_rot.apply(curr_joint)
            curr_joint = curr_joint * self._config.ref_pos_scale + self._pos_offset
            pose_to_process[joint_idx] = curr_joint

        return pose_to_process
