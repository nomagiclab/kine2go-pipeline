import genesis as gs
import numpy as np
import torch

from motion_retargeting.config import RobotConfig, SceneConfig

GO2_URDF_PATH = "urdf/go2/urdf/go2.urdf"
PLANE_URDF_PATH = "urdf/plane/plane.urdf"
FOOT_LINK_NAMES = ["FL_foot", "RL_foot", "FR_foot", "RR_foot"]
BASE_VELOCITY_DOF_INDICES = [0, 1, 2, 3, 4, 5]
MOTOR_QPOS_OFFSET = 1
NUM_ENVS = 1
VIDEO_RESOLUTION = (1920, 1080)
VIEWER_FPS_WINDOW_S = 0.5
DEBUG_POINT_RADIUS = 0.01
CAMERA_FOLLOW_OFFSET = np.array([0.0, 1.5, 0.0])


class Go2Env:
    def __init__(
        self,
        robot_config: RobotConfig,
        scene_config: SceneConfig,
    ):
        self.robot_config = robot_config
        self.scene_config = scene_config
        self.scene, self.robot = self._create_scene()

        self.motors_dof_idx = [self.robot.get_joint(name).dof_start for name in self.robot_config.joint_names]
        self.reset()

    def _create_robot(self):
        return gs.morphs.URDF(
            file=GO2_URDF_PATH,
            pos=self.robot_config.init_base_link_pos,
            quat=self.robot_config.init_base_link_quat,
            links_to_keep=FOOT_LINK_NAMES,
        )

    def _create_scene(self):
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.scene_config.dt, substeps=self.scene_config.substeps),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(VIEWER_FPS_WINDOW_S / self.scene_config.dt),
                camera_pos=self.scene_config.camera_pos,
                camera_lookat=self.scene_config.camera_lookat,
                camera_fov=self.scene_config.camera_fov,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(NUM_ENVS))),
            rigid_options=gs.options.RigidOptions(
                dt=self.scene_config.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=self.scene_config.show_viewer,
        )

        scene.add_entity(gs.morphs.URDF(file=PLANE_URDF_PATH, fixed=True))
        robot = scene.add_entity(self._create_robot())

        if self.scene_config.record_video:
            self.cam = scene.add_camera(
                res=VIDEO_RESOLUTION,
                pos=self.scene_config.camera_pos,
                lookat=self.scene_config.camera_lookat,
                fov=self.scene_config.camera_fov,
                GUI=False,
                debug=True,
            )

        scene.build(n_envs=NUM_ENVS)

        return scene, robot

    def step(self):
        self.scene.step()

    def reset(self):
        default_dof_pos = torch.tensor(
            [self.robot_config.default_joint_angles[joint_name] for joint_name in self.robot_config.joint_names],
        )

        self.robot.set_dofs_position(
            position=default_dof_pos,
            dofs_idx_local=self.motors_dof_idx,
            zero_velocity=True,
        )

        self.default_qpos = self.robot.get_qpos()

        self._update_cameras()

    def set_base_link_pose(
        self,
        pos: np.ndarray,
        quat: np.ndarray,
    ):
        self.robot.set_pos(pos, zero_velocity=True)
        self.robot.set_quat(quat, zero_velocity=True)

    def set_base_link_velocity(
        self,
        vel: np.ndarray,
        ang_vel: np.ndarray,
    ):
        base_dof_vel = np.concatenate((vel, ang_vel), axis=0)
        self.robot.set_dofs_velocity(base_dof_vel, dofs_idx_local=BASE_VELOCITY_DOF_INDICES)

    def set_leg_dofs_pose(self, joints_qpos: np.ndarray, *, zero_velocity: bool):
        motor_qpos_idx = self._motor_qpos_indices()
        leg_dof_pos = joints_qpos[motor_qpos_idx]  # skip the base joint

        self.robot.set_dofs_position(
            position=leg_dof_pos,
            dofs_idx_local=self.motors_dof_idx,
            zero_velocity=zero_velocity,
        )

    def set_leg_dofs_velocity(self, joints_vel: np.ndarray):
        motor_qpos_idx = self._motor_qpos_indices()
        leg_dof_vel = joints_vel[motor_qpos_idx]  # skip the base joint

        self.robot.set_dofs_velocity(leg_dof_vel, dofs_idx_local=self.motors_dof_idx)

    def _motor_qpos_indices(self):
        return [dof_idx + MOTOR_QPOS_OFFSET for dof_idx in self.motors_dof_idx]

    def set_pose(
        self,
        base_link_pos: np.ndarray,
        base_link_quat: np.ndarray,
        joints_qpos: np.ndarray,
    ):
        self.set_base_link_pose(base_link_pos, base_link_quat)
        self.set_leg_dofs_pose(joints_qpos, zero_velocity=True)

        self._finalize_pose_update()

    def set_pose_and_velocity(
        self,
        base_link_pos: np.ndarray,
        base_link_quat: np.ndarray,
        joints_qpos: np.ndarray,
        base_link_vel: np.ndarray,
        base_link_ang_vel: np.ndarray,
        joints_vel: np.ndarray,
    ):
        self.set_base_link_pose(base_link_pos, base_link_quat)
        self.set_base_link_velocity(base_link_vel, base_link_ang_vel)

        self.set_leg_dofs_pose(joints_qpos, zero_velocity=False)
        self.set_leg_dofs_velocity(joints_vel)

        self._finalize_pose_update()

    def _finalize_pose_update(self):
        self.scene.visualizer.update()
        self._update_cameras()
        if self.scene_config.record_video:
            self.cam.render()

    def draw_debug_points(self, points: np.ndarray, colors: list[tuple[float, float, float, float]]):
        for point, color in zip(points, colors, strict=True):
            self.scene.draw_debug_sphere(pos=point, radius=DEBUG_POINT_RADIUS, color=color)

    def dump_state(self):
        feet_pos = self._get_feet_positions()

        state = [
            self.robot.get_dofs_position().cpu().numpy(),  # [0:18]   dofs_position (18 values, indices 0-17)
            self.robot.get_dofs_velocity().cpu().numpy(),  # [18:36]  dofs_velocity (18 values, indices 18-35)
            feet_pos.cpu().numpy().reshape(1, -1),  # [36:48]  feet_pos (12 values, indices 36-47)
            self.robot.get_pos().cpu().numpy(),  # [48:51]  base position (3 values, indices 48-50)
            self.robot.get_quat().cpu().numpy(),  # [51:55]  base orientation quat (4 values, indices 51-54)
            self.robot.get_vel().cpu().numpy(),  # [55:58]  base linear velocity (3 values, indices 55-57)
            self.robot.get_ang().cpu().numpy(),  # [58:61]  base angular velocity (3 values, indices 58-60)
        ]

        state = np.concatenate([s.squeeze() for s in state], axis=0)

        return state

    def _get_feet_positions(self):
        feet_links = [self.robot.get_link(name).idx - self.robot.link_start for name in FOOT_LINK_NAMES]
        return self.robot.get_links_pos(links_idx_local=feet_links)

    def _update_cameras(self):
        cam_pos, cam_lookat = self._get_cam_pose()

        if self.scene_config.show_viewer:
            self.scene.viewer.set_camera_pose(pos=cam_pos, lookat=cam_lookat)
        if self.scene_config.record_video:
            self.cam.set_pose(pos=cam_pos, lookat=cam_lookat)

    def _get_cam_pose(self) -> tuple[np.ndarray, np.ndarray]:
        robot_pos = self.robot.get_pos().squeeze().cpu().numpy()
        return robot_pos + CAMERA_FOLLOW_OFFSET, robot_pos
