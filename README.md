# kine2go-pipeline

*Quadruped motion retargeting and imitation learning for the Unitree Go2.*

[![Python](https://img.shields.io/badge/python-%E2%89%A53.12-blue)](pyproject.toml)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-green)](LICENSE)
[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-yellow)](https://huggingface.co/datasets/MIMUW-Robotics/kine2go)
[![arXiv](https://img.shields.io/badge/arXiv-2606.14433-b31b1b)](https://arxiv.org/abs/2606.14433)
[![Website](https://img.shields.io/badge/Website-kine2go-blue)](https://nomagiclab.github.io/kine2go/)

[Installation](#installation) · [Quick Start](#quick-start) · [Pipeline](#pipeline) · [Stage details](#stage-details) · [Tools](#tools)

---

kine2go-pipeline retargets animal and quadruped motion capture (AI4Animation dog, Vienna Horse Data Collection, Solo8) into Unitree Go2-executable reference trajectories, and trains PPO policies in [Genesis](https://genesis-world.readthedocs.io/) to imitate them. The artifacts produced with this pipeline are published as the [kine2go](https://huggingface.co/datasets/MIMUW-Robotics/kine2go) dataset on HuggingFace.

## At a Glance

|   |   |
|---|---|
| Robot | Unitree Go2 (12 DoF, 4 feet) |
| Simulator | Genesis 0.3.10 |
| RL | PPO via `rsl-rl-lib` 3.0.0 |
| Source datasets | AI4Animation, VHDC, Solo8 |
| Python | ≥ 3.12 |
| Dataset | [kine2go on HuggingFace](https://huggingface.co/datasets/MIMUW-Robotics/kine2go) |

## Pipeline

```
  raw mocap (.txt / .csv / .pt)
            │
            ▼  motion_retargeting/main.py
  retargeted trajectory (.npy)
            │
            ▼  motion_imitation/imitation.py
  trained PPO policy (model_*.pt)
            │
            ▼  tools/gather_trajectories.py
  rollout dataset
```

- `motion_retargeting/` — converts source mocap into a Go2 reference trajectory by solving per-frame inverse kinematics in Genesis.
- `motion_imitation/` + `go2_genesis/` — RL environment, training, and evaluation. `motion_imitation` adds the imitation reward and observation wrapper on top of the Go2 locomotion env in `go2_genesis`.
- `tools/` — scripts for collecting and post-processing rollout data with a trained policy.

## Dataset

Pipeline artifacts — retargeted clips, trained policies, rollouts, and videos — are published at [https://huggingface.co/datasets/MIMUW-Robotics/kine2go](https://huggingface.co/datasets/MIMUW-Robotics/kine2go). Clip names in `motion_imitation/motion_ranges.txt` mirror the dataset's per-clip folder names.

## Installation

```bash
git clone <repo-url> && cd kine2go-pipeline
uv sync
```

`uv` picks up the pinned Python version from `.python-version` and resolves all dependencies declared in `pyproject.toml`.

> **Note:** Genesis runs RL training on a CUDA-capable GPU. Retargeting uses the CPU backend and works without one. Pass `--cpu` to `motion_imitation/imitation.py` for a CPU training smoke-test.

## Quick Start

End-to-end example using the canonical `ai4_dog_walk_00` clip from `motion_imitation/motion_ranges.txt` (frames 101–580 of `dog_walk00_joint_pos.txt`):

```bash
# 1. Retarget AI4Animation dog walk to Go2
uv run -m motion_retargeting.main \
  --motion-path motion_retargeting/data/AI4Animation/dog_walk00_joint_pos.txt \
  --dataset-name ai4animation \
  --frame-start 101 --frame-end 580
# → motion_retargeting/results/dog_walk00_joint_pos.npy

# 2. Train an imitation policy on the retargeted clip
uv run -m motion_imitation.imitation ai4_dog_walk_00 \
  motion_retargeting/results/dog_walk00_joint_pos.npy \
  --num-envs 4096 --max-iterations 1000
# → logs/ai4_dog_walk_00/model_*.pt

# 3. Evaluate and record a video
uv run -m motion_imitation.imitation_eval logs/ai4_dog_walk_00/model_1000.pt --record
# → logs/ai4_dog_walk_00/recording.mp4
```

Add `--wandb-mode offline` (or `disabled`) for runs without a Weights & Biases account.

## Stage details

### Retargeting (`motion_retargeting/`)

```bash
uv run -m motion_retargeting.main \
  --motion-path <path>  --dataset-name {ai4animation,horse,solo8}  \
  [--frame-start N] [--frame-end N]  \
  [--scene.record-video] [--robot.<field>=...] [--scene.<field>=...]
```

Source data lives in `motion_retargeting/data/<source>/`. Output is written to `motion_retargeting/results/<motion_name>.npy`, where `<motion_name>` is the source filename stem. Pass `--scene.record-video` to also dump an MP4 to `motion_retargeting/videos/<motion_name>/`.

### Imitation training (`motion_imitation/` + `go2_genesis/`)

```bash
uv run -m motion_imitation.imitation EXP_NAME MOTION_PATH \
  [--motion-start N] [--motion-end N] \
  [--num-envs 4096] [--max-iterations 1000] \
  [--wandb-mode {online,offline,disabled}] [--cpu]
```

`motion_imitation/motion_ranges.txt` is the catalogue of named clips with their frame ranges; names follow the [kine2go HuggingFace dataset](https://huggingface.co/datasets/MIMUW-Robotics/kine2go) convention (`ai4_dog_*`, `solo8_*`, `vhdc_horse1_*`). Checkpoints land in `logs/<exp_name>/`, and the reference clip is copied to `logs/<exp_name>/motion.npy` so evaluation can reconstruct the frame range automatically.

### Evaluation

```bash
uv run -m motion_imitation.imitation_eval CKPT_PATH \
  [--record] [--headless] [--num-episodes 1]
```

`--record` writes `logs/<exp>/recording.mp4`.

## Tools

Scripts in `tools/` for working with rollouts:

- `gather_trajectories.py` — roll out a trained policy and save trajectories.
- `cut_trajectory.py` — trim a recorded trajectory to a frame range.
- `visualize_trajectory.py` — render an MP4 of a recorded trajectory.

## Project layout

```
kine2go-pipeline/
├── motion_retargeting/                 # stage 1: source mocap → Go2 trajectory
├── motion_imitation/                   # stage 2: imitation env, training, eval
│   └── motion_ranges.txt               # catalogue of clip names + frame ranges
├── go2_genesis/                        # Go2 locomotion env, RL training infra
├── tools/                              # rollout collection and post-processing
└── pyproject.toml
```

## Citation
If you use our work, please cite:

```
@misc{pałucki2026kine2gokinematicdatasetunitree,
      title={Kine2Go: Kinematic dataset for the Unitree Go2 robot with diverse gaits and motions}, 
      author={Władysław Pałucki and Paweł Siwak and Krzysztof Ciebiera and Marek Cygan},
      year={2026},
      eprint={2606.14433},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2606.14433}, 
}
```
