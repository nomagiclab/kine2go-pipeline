import cvxpy as cp
import numpy as np
from scipy.spatial.transform import Rotation as R

from motion_retargeting.datasets.base import (
    DEFAULT_START_FRAME,
    NUM_FEET,
    XYZ_DIM,
    Z_AXIS_INDEX,
    DatasetConfig,
    MotionData,
    RetargetingDataset,
)

CONTACT_THRESHOLD = 15.0
SMOOTHNESS_WEIGHT = 10.0
TIMESTAMP_COLUMN = [1]
JOINT_POSITION_COLUMNS = range(2, 23)
NUM_HORSE_JOINTS = 7
FOOT_INDICES = [3, 4, 5, 6]

HORSE_CONFIG = DatasetConfig(
    anatomy_type="horse",
    pelvis_id=2,
    neck_id=1,
    hip_ids=[1, 2, 1, 2],
    toe_ids=[3, 5, 4, 6],
    ref_pos_scale=0.00025,
    ref_forward_dir_offset=(0.0, 0.0, 0.0),
    pos_offset=(0.0, 0.0, 0.0),
    coord_rot_euler=(0, 0, 0),
    root_rot_euler=(0, 0, 0),
)


class HorseDataset(RetargetingDataset):
    """Dataset for Horse kinematics data."""

    def __init__(self):
        self._config = HORSE_CONFIG
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
        timestamps = np.loadtxt(motion_path, delimiter=",", skiprows=1, usecols=TIMESTAMP_COLUMN)
        # extract columns 2-22 (joint positions: Head, Withers, Sacrum, HFL, HFR, HHL, HHR)
        joint_pos_data = np.loadtxt(motion_path, delimiter=",", skiprows=1, usecols=JOINT_POSITION_COLUMNS)

        start_frame = DEFAULT_START_FRAME if (frame_start is None) else frame_start
        end_frame = joint_pos_data.shape[0] if (frame_end is None) else frame_end

        timestamps = timestamps[start_frame:end_frame]
        joint_pos_data = joint_pos_data[start_frame:end_frame]

        # reshape to (N, 7, 3) where 7 is the number of joints (Head, Withers, Sacrum, HFL, HFR, HHL, HHR)
        joint_pos_data = joint_pos_data.reshape(-1, NUM_HORSE_JOINTS, XYZ_DIM)

        # convert to absolute coordinates
        joint_pos_data = self._estimate_absolute_motion(joint_pos_data)

        processed_positions = [self._process_joint_pos(joint_pos) for joint_pos in joint_pos_data]
        return MotionData(joint_positions=processed_positions, timestamps=timestamps)

    def _estimate_absolute_motion(self, relative_motion: np.ndarray) -> np.ndarray:
        """Estimate absolute motion from relative motion using contact constraints.

        The optimization problem minimizes the velocity of grounded feet while keeping
        changes in base velocity bounded.
        """
        n_frames = relative_motion.shape[0]

        # 1. Detect contacts based on height threshold
        feet_pos = relative_motion[:, FOOT_INDICES, :]  # (N, 4, 3)
        feet_z = feet_pos[:, :, Z_AXIS_INDEX]
        contacts = feet_z < CONTACT_THRESHOLD  # (N, 4)

        # 2. Formulate optimization problem
        # Variables: v_base (N, 3) - linear velocity of the base
        v_base = cp.Variable((n_frames, XYZ_DIM))

        # Approximate relative foot velocities
        v_feet_rel = np.zeros((n_frames, NUM_FEET, XYZ_DIM))
        v_feet_rel[1:] = feet_pos[1:] - feet_pos[:-1]

        cost = 0
        for t in range(1, n_frames):
            for i in range(NUM_FEET):
                if contacts[t, i]:
                    # Minimize absolute velocity squared
                    # v_abs = v_rel + v_base
                    cost += cp.sum_squares(v_feet_rel[t, i] + v_base[t])

        # Constraints: ||v_base,t - v_base,t-1||^2 < epsilon
        # Smoothness constraint on base velocity
        diff_v_base = v_base[1:] - v_base[:-1]
        cost += SMOOTHNESS_WEIGHT * cp.sum_squares(diff_v_base)

        problem = cp.Problem(cp.Minimize(cost))
        problem.solve(verbose=False)

        # 3. Reconstruct absolute positions
        # Integrate v_base to get p_base
        base_velocity = v_base.value  # (N, 3) displacement per frame
        if base_velocity is None:
            print("Optimization failed, returning relative motion.")
            return relative_motion

        base_pos = np.cumsum(base_velocity, axis=0)  # (N, 3) starting from 0

        # Add base position to all relative joint positions
        absolute_motion = relative_motion.copy()
        for i in range(relative_motion.shape[1]):
            absolute_motion[:, i, :] += base_pos

        return absolute_motion

    def _process_joint_pos(self, ref_joint_pos: np.ndarray) -> np.ndarray:
        pose_to_process = ref_joint_pos.copy()
        n_joints = pose_to_process.shape[0]

        for joint_idx in range(n_joints):
            curr_joint = pose_to_process[joint_idx]
            curr_joint = self._coord_rot.apply(curr_joint)
            curr_joint = self._root_rot.apply(curr_joint)
            curr_joint = curr_joint * self._config.ref_pos_scale + self._pos_offset
            pose_to_process[joint_idx] = curr_joint

        return pose_to_process
