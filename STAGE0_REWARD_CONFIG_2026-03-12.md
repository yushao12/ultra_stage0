# Stage0 Reward Config Snapshot (2026-03-12)

Task: `Mjlab-Retargeting-Flat-Unitree-G1`
Config source: `mjlab/tasks/retargeting/retargeting_env_cfg.py`

## Active reward terms and weights

- `stage0_track`: `+5.0`
- `joint_limit`: `-1.0`
- `self_collisions`: `-1.0`

## `stage0_track` parameters

- `anchor_pos_std = 0.3`
- `ee_pos_std = 0.3`
- `body_ori_std = 0.4`
- `body_lin_vel_std = 1.0`
- `body_ang_vel_std = 3.14`
- `joint_vel_scale = 1e-4`
- `action_rate_scale = 1e-2`
- `enable_interaction_reward = False`
- `enable_contact_reward = False`

## Effective structure

`r_total = r_anchor * r_p * r_r * r_v * r_w * r_obj * r_int * r_ct * r_eng`

Current gating:

- `r_int = 1` (disabled)
- `r_ct = 1` (disabled)

## Runtime note

Recent training (`/tmp/stage0_retargeting_10000_rew5_20260312_170642.log`) shows
`stage0_track` now contributes meaningfully, but main termination shifted toward
`anchor_ori`.
