import re

import genesis as gs
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from motion_retargeting.datasets.base import (
    DEFAULT_START_FRAME,
    TRAJECTORY_PATH_PATTERN,
    DatasetConfig,
    MotionData,
    RetargetingDataset,
)

# Indices in the generated keypoint array
# 0: Base (Pelvis)
# 1: Neck (Synthetic)
# 2: FL Hip
# 3: RL Hip
# 4: FR Hip
# 5: RR Hip
# 6: FL Foot
# 7: RL Foot
# 8: FR Foot
# 9: RR Foot

SOLO8_URDF_PATH = "motion_retargeting/data/Solo8/solo8.urdf"
SOLO8_FOOT_LINKS_TO_KEEP = ["FL_FOOT", "FR_FOOT", "HL_FOOT", "HR_FOOT"]
SOLO8_JOINT_NAMES = [
    "FL_HFE",
    "FL_KFE",
    "FR_HFE",
    "FR_KFE",
    "HL_HFE",
    "HL_KFE",
    "HR_HFE",
    "HR_KFE",
]
SOLO8_LINK_NAMES = {
    "base": "base_link",
    "FL_hip": "FL_UPPER_LEG",
    "RL_hip": "HL_UPPER_LEG",
    "FR_hip": "FR_UPPER_LEG",
    "RR_hip": "HR_UPPER_LEG",
    "FL_foot": "FL_FOOT",
    "RL_foot": "HL_FOOT",
    "FR_foot": "FR_FOOT",
    "RR_foot": "HR_FOOT",
}
BASE_POS_SLICE = slice(0, 3)
BASE_QUAT_SLICE = slice(3, 7)
DOF_POS_SLICE = slice(17, 25)
KEYPOINT_SHAPE = (10, 3)
NECK_OFFSET = [0.2, 0.0, 0.0]

SOLO8_CONFIG = DatasetConfig(
    anatomy_type="dog",
    pelvis_id=0,
    neck_id=1,
    hip_ids=[2, 3, 4, 5],  # FL, RL, FR, RR
    toe_ids=[6, 7, 8, 9],  # FL, RL, FR, RR
    ref_pos_scale=1.33,
    ref_forward_dir_offset=(0.0, 0.0, 0.0),
    pos_offset=(0.0, 0.0, 0.0),
    coord_rot_euler=(0, 0, 0),
    root_rot_euler=(0, 0, 0),
)


class Solo8Dataset(RetargetingDataset):
    """Dataset for Solo8."""

    def __init__(self):
        self._config = SOLO8_CONFIG
        self._coord_rot = R.from_euler("xyz", self._config.coord_rot_euler, degrees=False)
        self._root_rot = R.from_euler("xyz", self._config.root_rot_euler, degrees=False)
        self._pos_offset = np.array(self._config.pos_offset)
        self._init_solo8_env()

    @property
    def config(self) -> DatasetConfig:
        return self._config

    def _init_solo8_env(self):
        self._scene = gs.Scene(
            show_viewer=False,
            rigid_options=gs.options.RigidOptions(
                enable_collision=False,
                enable_joint_limit=False,
                dt=0.01,  # small dt, not simulating physics
            ),
        )

        self._robot = self._scene.add_entity(
            gs.morphs.URDF(
                file=SOLO8_URDF_PATH,
                fixed=False,
                links_to_keep=SOLO8_FOOT_LINKS_TO_KEEP,
            ),
        )

        self._scene.build(n_envs=1)

        self._joint_indices = [self._robot.get_joint(name).dof_start for name in SOLO8_JOINT_NAMES]
        self._link_names = SOLO8_LINK_NAMES

        for name in self._link_names.values():
            assert self._robot.get_link(name) is not None, f"Link {name} not found in Solo8 URDF"

    def load_motion_data(
        self,
        motion_path: str,
        frame_start: int | None = None,
        frame_end: int | None = None,
    ) -> MotionData:
        motion_path, trajectory_idx = self._get_trajectory_idx(motion_path)
        data = torch.load(motion_path, map_location="cpu")  # [n_trajectories, n_frames, frame_size]

        if trajectory_idx is not None:
            if trajectory_idx < 0 or trajectory_idx >= data.shape[0]:
                raise ValueError(f"Trajectory index {trajectory_idx} out of bounds (0-{data.shape[0] - 1})")
            data = data[trajectory_idx]
        else:
            data = data.reshape(-1, data.shape[-1])

        data = data.numpy()

        start_frame = DEFAULT_START_FRAME if (frame_start is None) else frame_start
        end_frame = data.shape[0] if (frame_end is None) else frame_end
        data = data[start_frame:end_frame]

        # Data indices from dictionary
        # "base_pos": [0, 3]
        # "base_quat": [3, 7]
        # "dof_pos": [17, 25]

        processed_frames = []

        for frame in data:
            base_pos = frame[BASE_POS_SLICE]
            base_quat = frame[BASE_QUAT_SLICE]  # (x, y, z, w)
            dof_pos = frame[DOF_POS_SLICE]  # 4 x 2 joints

            self._robot.set_pos(base_pos)
            # Genesis expects (w, x, y, z)
            self._robot.set_quat(np.array([base_quat[3], base_quat[0], base_quat[1], base_quat[2]]))
            self._robot.set_dofs_position(dof_pos, dofs_idx_local=self._joint_indices)

            # Construct keypoints
            # 0: Base
            # 1: Neck
            # 2-5: Hips
            # 6-9: Feet

            keypoints = np.zeros(KEYPOINT_SHAPE)

            keypoints[0] = self._robot.get_link(self._link_names["base"]).get_pos().cpu().numpy()

            base_rot = R.from_quat(base_quat)
            neck_offset = base_rot.apply(
                NECK_OFFSET,
            )  # neck is artificially created by offsetting the base by 0.2m (based on the URDF) forward
            keypoints[1] = keypoints[0] + neck_offset

            keypoints[2] = self._robot.get_link(self._link_names["FL_hip"]).get_pos().cpu().numpy()
            keypoints[3] = self._robot.get_link(self._link_names["RL_hip"]).get_pos().cpu().numpy()
            keypoints[4] = self._robot.get_link(self._link_names["FR_hip"]).get_pos().cpu().numpy()
            keypoints[5] = self._robot.get_link(self._link_names["RR_hip"]).get_pos().cpu().numpy()

            keypoints[6] = self._robot.get_link(self._link_names["FL_foot"]).get_pos().cpu().numpy()
            keypoints[7] = self._robot.get_link(self._link_names["RL_foot"]).get_pos().cpu().numpy()
            keypoints[8] = self._robot.get_link(self._link_names["FR_foot"]).get_pos().cpu().numpy()
            keypoints[9] = self._robot.get_link(self._link_names["RR_foot"]).get_pos().cpu().numpy()

            keypoints = self._coord_rot.apply(keypoints)
            keypoints = self._root_rot.apply(keypoints)
            keypoints = keypoints * self.config.ref_pos_scale + self._pos_offset

            processed_frames.append(keypoints)

        return MotionData(joint_positions=processed_frames, timestamps=None)

    def _get_trajectory_idx(self, motion_path: str) -> tuple[str, int | None]:
        match = re.match(TRAJECTORY_PATH_PATTERN, motion_path)
        if match:
            motion_path, traj_idx = match.groups()
            return motion_path, int(traj_idx)

        return motion_path, None
