import os
import re

import genesis as gs
import numpy as np
import tyro
from scipy.spatial.transform import Rotation as R

from motion_retargeting.config import Config, RobotConfig
from motion_retargeting.datasets import DatasetConfig, MotionData, create_dataset
from motion_retargeting.datasets.base import TRAJECTORY_PATH_PATTERN, XYZ_DIM, Z_AXIS_INDEX
from motion_retargeting.env import Go2Env

RETARGETING_RESULTS_DIR = "motion_retargeting/results"
RETARGETING_VIDEOS_DIR = "motion_retargeting/videos"
RETARGETING_VIDEO_FILENAME = "video.mp4"
RETARGETING_VIDEO_FPS = 60

LEG_ORDER = ["FL", "RL", "FR", "RR"]
FORWARD_AXIS = np.array([1.0, 0.0, 0.0])
UP_AXIS = np.array([0.0, 0.0, 1.0])


def _motion_name_from_path(motion_path: str) -> str:
    match = re.match(TRAJECTORY_PATH_PATTERN, motion_path)
    if match:
        file_path, traj_idx = match.groups()
        return f"{file_path.split('/')[-1].split('.')[0]}_traj{traj_idx}"
    return motion_path.split("/")[-1].split(".")[0]


def _compute_frame_velocities(
    *,
    frame_idx: int,
    timestamps: np.ndarray,
    root_pos: np.ndarray,
    root_rot: R,
    joints_qpos: np.ndarray,
    prev_root_pos: np.ndarray | None,
    prev_root_rot: R | None,
    prev_joints_qpos: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base_link_vel = np.zeros(XYZ_DIM)
    base_link_ang_vel = np.zeros(XYZ_DIM)
    joints_vel = np.zeros_like(joints_qpos)

    if frame_idx > 0:
        dt = float(timestamps[frame_idx] - timestamps[frame_idx - 1])
        if dt > 0.0:
            base_link_vel = (root_pos - prev_root_pos) / dt
            joints_vel = (joints_qpos - prev_joints_qpos) / dt
            rot_diff = root_rot * prev_root_rot.inv()
            base_link_ang_vel = rot_diff.as_rotvec() / dt

    return base_link_vel, base_link_ang_vel, joints_vel


def retarget_motion(
    env: Go2Env,
    motion_data: MotionData,
    motion_name: str,
    dataset_config: DatasetConfig,
    robot_config: RobotConfig,
):
    if env.scene_config.record_video:
        env.cam.start_recording()

    output_states: list[np.ndarray] = []
    init_qpos = env.default_qpos

    prev_root_pos: np.ndarray | None = None
    prev_root_rot: R | None = None
    prev_joints_qpos: np.ndarray | None = None
    timestamps = motion_data.timestamps

    for frame_idx, ref_joints_pos in enumerate(motion_data.joint_positions):
        root_pos, root_rot, joints_qpos = retarget_pose(
            env,
            ref_joints_pos,
            init_qpos,
            dataset_config,
            robot_config,
        )

        if timestamps is None:
            env.set_pose(root_pos, root_rot.as_quat(scalar_first=True), joints_qpos)
        else:
            base_link_vel, base_link_ang_vel, joints_vel = _compute_frame_velocities(
                frame_idx=frame_idx,
                timestamps=timestamps,
                root_pos=root_pos,
                root_rot=root_rot,
                joints_qpos=joints_qpos,
                prev_root_pos=prev_root_pos,
                prev_root_rot=prev_root_rot,
                prev_joints_qpos=prev_joints_qpos,
            )

            env.set_pose_and_velocity(
                root_pos,
                root_rot.as_quat(scalar_first=True),
                joints_qpos,
                base_link_vel,
                base_link_ang_vel,
                joints_vel,
            )

            prev_root_pos = root_pos
            prev_root_rot = root_rot
            prev_joints_qpos = joints_qpos

        init_qpos = env.robot.get_qpos()
        output_states.append(env.dump_state())

    if env.scene_config.record_video:
        path = os.path.join(RETARGETING_VIDEOS_DIR, motion_name)
        os.makedirs(path, exist_ok=True)
        env.cam.stop_recording(
            save_to_filename=os.path.join(path, RETARGETING_VIDEO_FILENAME),
            fps=RETARGETING_VIDEO_FPS,
        )

    return output_states


def retarget_pose(
    env: Go2Env,
    ref_joints_pos: np.ndarray,
    init_qpos: np.ndarray,
    dataset_config: DatasetConfig,
    robot_config: RobotConfig,
) -> tuple[np.ndarray, R, np.ndarray]:
    root_pos, root_rot = retarget_root_pose(ref_joints_pos, dataset_config, robot_config)
    root_pos += robot_config.sim_base_link_offset
    env.set_base_link_pose(root_pos, root_rot.as_quat(scalar_first=True))

    inv_init_rot = R.from_quat(robot_config.init_base_link_quat, scalar_first=True).inv()
    heading_rot = calculate_heading_rot(root_rot * inv_init_rot)

    target_toes_pos = _target_toes_pos(env, ref_joints_pos, dataset_config, robot_config, heading_rot)

    joints_qpos = (
        env.robot
        .inverse_kinematics_multilink(
            links=[env.robot.get_link(leg_id + "_foot") for leg_id in LEG_ORDER],
            poss=target_toes_pos,
            init_qpos=init_qpos,
            dofs_idx_local=env.motors_dof_idx,
            respect_joint_limit=True,
        )
        .squeeze()
        .cpu()
        .numpy()
    )

    return root_pos, root_rot, joints_qpos


def _target_toes_pos(
    env: Go2Env,
    ref_joints_pos: np.ndarray,
    dataset_config: DatasetConfig,
    robot_config: RobotConfig,
    heading_rot: R,
) -> np.ndarray:
    target_toes_pos = []

    for i in range(len(dataset_config.toe_ids)):
        leg_id = LEG_ORDER[i]
        ref_toe_id = dataset_config.toe_ids[i]
        ref_hip_id = dataset_config.hip_ids[i]

        ref_toe_pos = ref_joints_pos[ref_toe_id]
        ref_hip_pos = ref_joints_pos[ref_hip_id]

        sim_hip_pos = env.robot.get_link(leg_id + "_hip").get_pos().squeeze().cpu().numpy()

        ref_hip_toe_delta = ref_toe_pos - ref_hip_pos
        sim_target_toe_pos = sim_hip_pos + ref_hip_toe_delta
        sim_target_toe_pos[Z_AXIS_INDEX] = ref_toe_pos[Z_AXIS_INDEX]

        toe_offset_local = robot_config.sim_toe_offsets[leg_id]
        toe_offset_world = heading_rot.apply(toe_offset_local)
        sim_target_toe_pos += toe_offset_world

        target_toes_pos.append(sim_target_toe_pos)

    target_toes_pos = np.stack(target_toes_pos, axis=0)  # 4 legs x 3 coords
    return np.expand_dims(target_toes_pos, axis=1)  # 4 legs x 1 env x 3 coords


def calculate_heading_rot(rot: R) -> R:
    rot_dir = rot.apply(FORWARD_AXIS)
    heading = np.arctan2(rot_dir[1], rot_dir[0])
    rotvec = heading * UP_AXIS
    heading_rot = R.from_rotvec(rotvec)

    return heading_rot


def retarget_root_pose(
    ref_joints_pos: np.ndarray,
    dataset_config: DatasetConfig,
    robot_config: RobotConfig,
) -> tuple[np.ndarray, R]:
    pelvis_pos = ref_joints_pos[dataset_config.pelvis_id]
    neck_pos = ref_joints_pos[dataset_config.neck_id]

    left_shoulder_pos = ref_joints_pos[dataset_config.hip_ids[0]]
    right_shoulder_pos = ref_joints_pos[dataset_config.hip_ids[2]]
    left_hip_pos = ref_joints_pos[dataset_config.hip_ids[1]]
    right_hip_pos = ref_joints_pos[dataset_config.hip_ids[3]]

    forward_dir = neck_pos - pelvis_pos
    forward_dir += dataset_config.ref_forward_dir_offset
    forward_dir = forward_dir / np.linalg.norm(forward_dir)

    delta_shoulder = left_shoulder_pos - right_shoulder_pos
    delta_hip = left_hip_pos - right_hip_pos

    dir_shoulder = (
        delta_shoulder / np.linalg.norm(delta_shoulder)
        if dataset_config.anatomy_type == "dog"
        else np.cross(UP_AXIS, forward_dir)
    )
    dir_hip = (
        delta_hip / np.linalg.norm(delta_hip)
        if dataset_config.anatomy_type == "dog"
        else np.cross(UP_AXIS, forward_dir)
    )

    left_dir = 0.5 * (dir_shoulder + dir_hip)

    up_dir = np.cross(forward_dir, left_dir)
    up_dir = up_dir / np.linalg.norm(up_dir)

    left_dir = np.cross(up_dir, forward_dir)
    left_dir[Z_AXIS_INDEX] = 0.0  # make the base more stable
    left_dir = left_dir / np.linalg.norm(left_dir)

    rot_mat = np.array(
        [
            [forward_dir[0], left_dir[0], up_dir[0]],
            [forward_dir[1], left_dir[1], up_dir[1]],
            [forward_dir[2], left_dir[2], up_dir[2]],
        ],
    )

    root_pos = 0.5 * (pelvis_pos + neck_pos)

    root_rot = R.from_matrix(rot_mat)
    init_rot = R.from_quat(robot_config.init_base_link_quat, scalar_first=True)
    root_rot = root_rot * init_rot

    return root_pos, root_rot


def main(config: Config):
    gs.init(logging_level="warning", backend=gs.cpu)

    dataset = create_dataset(config.dataset_name)
    motion_data = dataset.load_motion_data(config.motion_path, config.frame_start, config.frame_end)

    motion_name = _motion_name_from_path(config.motion_path)

    env = Go2Env(config.robot, config.scene)

    output_states = retarget_motion(env, motion_data, motion_name, dataset.config, config.robot)

    os.makedirs(RETARGETING_RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RETARGETING_RESULTS_DIR, f"{motion_name}.npy")
    np.save(output_path, output_states)


if __name__ == "__main__":
    config = tyro.cli(Config)
    main(config)
