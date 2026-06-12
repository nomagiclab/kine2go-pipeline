import torch


def get_ppo_cfg(args):
    train_cfg_dict = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
            "class_name": "ActorCritic",
        },
        "runner": {
            "checkpoint": -1,
            "experiment_name": args.exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": args.max_iterations,
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
        },
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": 24,
        "save_interval": 1000,
        "empirical_normalization": None,
        "seed": 1,
        "logger": "wandb",
        "wandb_mode": args.wandb_mode,
        "project": args.project,
        "entity": args.entity,
        "group": args.group,
        "obs_groups": {"policy": ["policy"], "critic": ["policy"]},
    }

    return train_cfg_dict


def get_imitation_cfgs():
    env_cfg = {
        "urdf_path": "urdf/go2/urdf/go2.urdf",
        "links_to_keep": ["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
        "num_actions": 12,
        "num_dofs": 12,
        # joint/link names
        "default_joint_angles": {  # [rad]
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
        "dof_names": [
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
        "termination_contact_link_names": ["base"],
        "penalized_contact_link_names": ["base", "thigh", "calf"],
        "feet_link_names": ["foot"],
        "base_link_name": ["base"],
        # PD
        "PD_stiffness": {"joint": 30.0},
        "PD_damping": {"joint": 1.5},
        "use_implicit_controller": False,
        # termination
        "termination_if_roll_greater_than": 100,
        "termination_if_pitch_greater_than": 100,
        "termination_if_height_lower_than": 0.0,
        # base pose
        "base_init_pos": [0.0, 0.0, 0.42],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        # random push
        "push_interval_s": -1,
        "max_push_vel_xy": 1.0,
        # time (second)
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "command_type": "ang_vel_yaw",  # 'ang_vel_yaw' or 'heading'
        "action_scale": 0.25,
        "action_latency": 0.02,
        "clip_actions": 100.0,
        "send_timeouts": True,
        "control_freq": 60,
        "decimation": 4,
        "feet_geom_offset": 1,
        "use_terrain": False,
        # domain randomization
        "dofs_position_rand_range": 0.3,
        "base_position_rand_range": 1.0,
        "randomize_base_yaw": True,
        "randomize_friction": False,
        "randomize_trajectory_heading": True,
        "friction_range": [0.2, 1.5],
        "randomize_base_mass": True,
        "added_mass_range": [-1.0, 3.0],
        "randomize_com_displacement": True,
        "com_displacement_range": [-0.01, 0.01],
        "randomize_motor_strength": False,
        "motor_strength_range": [0.9, 1.1],
        "randomize_motor_offset": True,
        "motor_offset_range": [-0.02, 0.02],
        "randomize_kp_scale": True,
        "kp_scale_range": [0.8, 1.2],
        "randomize_kd_scale": True,
        "kd_scale_range": [0.8, 1.2],
        # coupling
        "coupling": False,
        "ref_state_init_prob": 0.9,
        "warmup_time_s": 1,
        "terminate_dist_threshold": 5.0,
        "terminate_rot_threshold": 0.5 * torch.pi,
        # Random state perturbation
        "perturb_init_state": True,
        "randomize_init_frame": False,
        "start_synced_with_ref_motion": True,
        # If False only sync position and quat with reference motion, joints and velocities are set to default
        "sync_full_pose": False,
        # Timing
        "policy_steps_per_ref_motion_step": 1,
    }
    obs_cfg = {
        "num_obs": 31,
        "num_history_obs": 3,
        "obs_noise": {
            "ang_vel": 0.0,
            "gravity": 0.0,
            "dof_pos": 0.0,
            "dof_vel": 0.0,
        },
        "obs_scales": {
            "lin_vel": 1.0,
            "ang_vel": 1.0,
            "dof_pos": 1.0,
            "dof_vel": 1.0,
        },
    }
    reward_cfg = {
        "tracking_sigma": 0.25,
        "soft_dof_pos_limit": 0.9,
        "base_height_target": 0.3,
        "reward_scales": {
            "joint_rot": 0.5,
            "joint_vel": 0.05,
            "end_effector_pos": 0.2,
            "root_pose": 0.15,
            "root_vel": 0.1,
            "smooth_actions": 0.0,
        },
    }
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [-1.0, 1.0],
        "lin_vel_y_range": [-1.0, 1.0],
        "ang_vel_range": [-1.0, 1.0],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg
