# 2026-03-09 Stage0 r_obj/r_int/r_ct Integration

## Goal
- Strictly align Stage0 reward structure with ULTRA paper formula:
  - `r_track = r_p * r_r * r_obj * r_int * r_ct * r_eng`
- Replace placeholder Stage0 terms (`r_obj/r_int/r_ct`) with actual object/contact versions.
- Connect OmniRetarget raw data (`qpos,fps`) into an mjlab-readable Stage0 motion format.

## Implemented

### 1) Motion data plumbing
- Extended motion loader/command layer to support optional fields:
  - `object_pos_w`, `object_quat_w`, `object_lin_vel_w`, `object_ang_vel_w`
  - `contact_labels`, `contact_body_names`
  - `interaction_points_local`, `interaction_weights`, `interaction_body_names`
- Kept backward compatibility:
  - old motion files without these keys still run.

### 2) Real Stage0 terms
- Implemented:
  - `stage0_object_tracking_reward` (`r_obj`)
  - `stage0_interaction_reward` (`r_int`)
  - `stage0_contact_matching_reward` (`r_ct`)
- Added these terms into `stage0_track_reward` product.
- Fallback behavior:
  - if object entity or optional references are missing, term returns `1.0` (non-breaking).

### 3) Converter script
- Added `scripts/omniretarget_qpos_to_mjlab_npz.py`:
  - input: OmniRetarget `.npz` (`qpos`, `fps`)
  - output: mjlab motion `.npz` with joint/body signals
  - when object exists, also writes object/contact/interaction fields used by Stage0 rewards

### 4) Stage0 config wiring
- Added palm body defaults for interaction/contact in retargeting env config.

## Validation run

### Syntax/compile
```bash
python -m compileall \
  mjlab/tasks/tracking/mdp/commands.py \
  mjlab/tasks/retargeting/mdp/rewards.py \
  scripts/omniretarget_qpos_to_mjlab_npz.py
```

### Converter test (object trajectory)
```bash
python scripts/omniretarget_qpos_to_mjlab_npz.py \
  --input-npz data/ultra_stage0/omniretarget/raw/robot-object-terrain/robot-object-terrain/scene_01_original.npz \
  --output-npz /tmp/scene_01_mjlab_stage0.npz \
  --output-fps 50 --device cpu
```
- Output contains:
  - base fields (`joint_pos`, `body_pos_w`, ...)
  - object/contact/interaction fields.

### Converter test (no object)
```bash
python scripts/omniretarget_qpos_to_mjlab_npz.py \
  --input-npz data/ultra_stage0/omniretarget/raw/robot-terrain/robot-terrain/climb_00_z_scale_1.0.npz \
  --output-npz /tmp/climb_00_mjlab_stage0.npz \
  --output-fps 50 --device cpu
```
- Output contains only base fields (as expected).

### Training smoke tests
```bash
python scripts/train.py Mjlab-Retargeting-Flat-Unitree-G1 \
  --motion-file=mjlab/motions/g1/dance1_subject2.npz \
  --env.scene.num-envs=64 --agent.max-iterations=1 --agent.num-steps-per-env=8
```
```bash
python scripts/train.py Mjlab-Retargeting-Flat-Unitree-G1 \
  --motion-file=/tmp/scene_01_mjlab_stage0.npz \
  --env.scene.num-envs=32 --agent.max-iterations=1 --agent.num-steps-per-env=8
```
- Both finished 1 iteration without code crash.

## Known gaps
- Current Stage0 env does not yet include explicit object entity in scene; when missing, `r_obj/r_int/r_ct` safely fallback to `1.0`.
- To fully activate object-aware rewards, next step is adding object entity + contact sensor mapping in env scene.
