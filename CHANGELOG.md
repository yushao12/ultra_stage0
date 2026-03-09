# Changelog

## 2026-03-09

### Added
- Added a minimal free-joint box object entity and XML asset for Stage0 object tracking.
- Added G1 Stage0 object wiring (`scene.entities["object"]`) and motion command binding (`object_entity_name="object"`).
- Added Omniretarget-to-MJLab conversion script for object trajectory export from `qpos` tail states.

### Changed
- Stage0 reward now defaults to object trajectory tracking first; interaction/contact rewards are gated by explicit switches.
- Omniretarget conversion defaults to object-only export (contact/interaction labels are optional via `export_contact_labels`).
- Increased Stage0 sim limits for object scenes (`nconmax>=128`, `njmax>=512`) to avoid contact/constraint overflows.

### Validated
- Generated object-only motion file from Omniretarget sample:
  - `/tmp/sub3_largebox_003_mjlab_stage0_objonly.npz`
- 1-iteration smoke training passed with object trajectory input on `Mjlab-Retargeting-Flat-Unitree-G1`.
- Confirmed no `nconmax overflow` and no `nefc overflow` after raising `njmax`.
