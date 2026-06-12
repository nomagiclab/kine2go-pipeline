from dataclasses import dataclass, field


@dataclass
class RobotConfig:
    # init base link pose
    init_base_link_pos: tuple[float, float, float] = (0.0, 0.0, 0.42)
    init_base_link_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    joint_names: list[str] = field(
        default_factory=lambda: [
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
        ],
    )
    # in radians
    default_joint_angles: dict[str, float] = field(
        default_factory=lambda: {
            "FL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        },
    )

    # sim offsets
    sim_base_link_offset: tuple[float, float, float] = (0.0, 0.0, -0.04)
    sim_toe_offsets: dict[str, tuple[float, float, float]] = field(
        default_factory=lambda: {
            "FL": (0.0, 0.1, 0.0),
            "RL": (0.0, 0.1, 0.01),
            "FR": (0.0, -0.1, 0.0),
            "RR": (0.0, -0.1, 0.01),
        },
    )


@dataclass
class SceneConfig:
    dt: float = 0.02
    substeps: int = 2
    max_FPS: int = int(0.5 / 0.02)
    camera_pos: tuple[float, float, float] = (5.0, 0.0, 2.0)
    camera_lookat: tuple[float, float, float] = (2.5, 0.0, 0.0)
    camera_fov: int = 40
    show_viewer: bool = False
    record_video: bool = False


@dataclass
class Config:
    motion_path: str
    dataset_name: str
    frame_start: int | None = None
    frame_end: int | None = None
    robot: RobotConfig = field(default_factory=RobotConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)
