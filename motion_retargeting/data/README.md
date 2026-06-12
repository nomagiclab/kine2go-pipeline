# Motion retargeting source data

Reference motion clips used by the retargeting pipeline. Each subfolder is a
distinct upstream dataset with its own license; full license texts live in
[`LICENSES/`](LICENSES). Mixing data across subfolders is allowed only to the
extent the strictest applicable license permits.

## Sources

### `AI4Animation/` — natural dog mocap
Markered dog motion capture released with the Mode-Adaptive Neural Networks
paper. Files are paired `<clip>_pose.txt` / `<clip>_joint_pos.txt` text dumps;
`dog_clips_info.txt` maps the original recording IDs to the friendly names and
notes per-clip trim ranges.

- Upstream: <https://github.com/sebastianstarke/AI4Animation>
- License: **CC BY-NC 4.0** — see [`LICENSES/AI4Animation_CC-BY-NC-4.0.txt`](LICENSES/AI4Animation_CC-BY-NC-4.0.txt)
- Cite: Zhang et al., *Mode-Adaptive Neural Networks for Quadruped Motion Control*, ACM TOG 2018.

### `AI4Animation/synthetic/` — procedurally generated dog gaits
Synthetic trajectories (circles, ellipses, figure-eights, squares, strafing)
produced from the AI4Animation locomotion model. Treated as derivative work of
the AI4Animation dataset and therefore inherits its license.

- License: **CC BY-NC 4.0** (inherited)

### `Horse/` — Vienna Horse Data Collection (VHDC)
Per-frame kinematics CSVs for a single horse (`Horse1`) across three markersets
(`M1`/`M2`/`M3`) and two gaits (walk, trot), three repetitions each.

- Upstream: <https://horse.cs.uni-bonn.de/vhdc-home.html>
- License: **CC BY-SA 4.0** — see [`LICENSES/VHDC_CC-BY-SA-4.0.txt`](LICENSES/VHDC_CC-BY-SA-4.0.txt)
- Note: ShareAlike — adaptations distributed publicly must be released under a
  compatible CC BY-SA license.
- Cite: Vienna Horse Data Collection (VHDC).

### `Solo8/` — cassi reference motions for the Solo8 robot
Reference state tensor (`solo8_motion_data.pt`), URDF, and meshes from the
cassi project (Versatile Skill Control via Self-supervised Adversarial
Imitation of Unlabeled Mixed Motions).

- Upstream: cassi (ETH Zurich / NVIDIA)
- License: **BSD 3-Clause** — see [`LICENSES/cassi_BSD-3-Clause.txt`](LICENSES/cassi_BSD-3-Clause.txt)
- Cite: Li et al., *Versatile Skill Control via Self-supervised Adversarial Imitation of Unlabeled Mixed Motions*, IEEE ICRA 2023.

## Licensing summary

| Subfolder                 | License      |
| ------------------------- | ------------ |
| `AI4Animation/`           | CC BY-NC 4.0 |
| `AI4Animation/synthetic/` | CC BY-NC 4.0 |
| `Horse/`                  | CC BY-SA 4.0 |
| `Solo8/`                  | BSD-3-Clause |
