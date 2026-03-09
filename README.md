# Stage0 Repro Journal

This repository is a persistent engineering journal for Stage0 reproduction work.

## Source workspace
- Primary codebase: `/data/ouyangyu/wordkspace/pjlab/unitree_rl_mjlab`
- OmniRetarget clone: `/data/ouyangyu/wordkspace/omniretarget`

## Purpose
- Keep a compact, append-only history of each change set.
- Preserve code snapshots + rationale + validation commands.
- Make it easy to continue work after context resets.

## Update protocol (every change)
1. Create a new folder under `snapshots/<date>-<topic>/` with changed files.
2. Add one markdown note under `entries/`:
   - Problem / goal
   - Implementation details
   - Validation commands and key outputs
   - Known gaps / next steps
3. Append one line to `CHANGELOG.md`.
4. Commit with a clear message.

## Git identity used here
- user.name: `yushao12`
- user.email: `1712264659@qq.com`
