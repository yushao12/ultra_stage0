# ULTRA Stage0 Plan

Last updated: 2026-03-12

## Current baseline status (Track A)

- Task: `Mjlab-Retargeting-Flat-Unitree-G1`
- Object trajectory is loaded from motion npz and now synchronized into sim object root state on reset/resample.
- Reward weight rebalance already applied (`stage0_track=+5.0`, limits/collision penalties reduced to `-1.0`).

## Immediate issues observed

- Grasp robustness is still limited: policy often follows trajectory but fails to stably hold the box.
- Contact semantics are not supervised yet (`contact reward` currently disabled).

## Agreed next directions (from discussion)

1. Contact branch (harder, delayed)
- Consider enabling/adding contact reward after label reliability is confirmed.
- Keep as a planned branch, not first action.

2. Object branch (first action)
- Investigate whether original data already contains `smallbox` motion+object sequences.
- If yes, prioritize using native smallbox data instead of only shrinking sim geometry.
- If not, fallback: reduce box size/mass in sim and amplify object-tracking emphasis.

## Execution order

1. Save current reward config snapshot and commit baseline state.
2. Inspect original Omniretarget data for `smallbox` sequences and summarize findings.
3. Implement minimal object-branch changes (small/light box + stronger object tracking), then run smoke training.

## Notes

- Track A remains the control baseline.
- Track B (`Mjlab-Retargeting-Direct-Omomo-Unitree-G1`) continues as paper-aligned branch in parallel.


## Data inspection note (2026-03-12)

- Checked data/ultra_stage0/omniretarget/raw for smallbox sequences.
- Result: no smallbox motion files found in current local dump.
- Available object-related domains are mainly largebox (robot-object), plus chair/climb in object-terrain subsets.
- Fallback path: reduce sim box size/mass and strengthen object tracking reward.
