"""Stage-0 retargeting task configuration.

This file creates a stage-0 environment that reuses the existing motion command
pipeline while switching to a retargeting-oriented observation/reward design.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.retargeting import mdp
from mjlab.tasks.retargeting.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg


_STAGE0_EE_BODY_NAMES = (
  "left_ankle_roll_link",
  "right_ankle_roll_link",
  "left_wrist_yaw_link",
  "right_wrist_yaw_link",
)

_STAGE0_PALM_BODY_NAMES = (
  "left_wrist_yaw_link",
  "right_wrist_yaw_link",
)


def make_retargeting_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create the base stage-0 retargeting configuration."""
  cfg = make_tracking_env_cfg()

  policy_terms = {
    "command": ObservationTermCfg(
      func=mdp.generated_commands, params={"command_name": "motion"}
    ),
    "sim_body_pos_b": ObservationTermCfg(
      func=mdp.sim_body_pos_b, params={"command_name": "motion"}
    ),
    "sim_body_ori_b": ObservationTermCfg(
      func=mdp.sim_body_ori_b, params={"command_name": "motion"}
    ),
    "ref_body_pos_b": ObservationTermCfg(
      func=mdp.ref_body_pos_b, params={"command_name": "motion"}
    ),
    "ref_body_ori_b": ObservationTermCfg(
      func=mdp.ref_body_ori_b, params={"command_name": "motion"}
    ),
    "delta_body_pos_b": ObservationTermCfg(
      func=mdp.delta_body_pos_b, params={"command_name": "motion"}
    ),
    "delta_body_ori_b": ObservationTermCfg(
      func=mdp.delta_body_ori_b, params={"command_name": "motion"}
    ),
    "body_lin_vel_residual_w": ObservationTermCfg(
      func=mdp.body_lin_vel_residual_w, params={"command_name": "motion"}
    ),
    "body_ang_vel_residual_w": ObservationTermCfg(
      func=mdp.body_ang_vel_residual_w, params={"command_name": "motion"}
    ),
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
    "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }

  critic_terms = {
    "command": ObservationTermCfg(
      func=mdp.generated_commands, params={"command_name": "motion"}
    ),
    "sim_body_pos_b": ObservationTermCfg(
      func=mdp.sim_body_pos_b, params={"command_name": "motion"}
    ),
    "sim_body_ori_b": ObservationTermCfg(
      func=mdp.sim_body_ori_b, params={"command_name": "motion"}
    ),
    "ref_body_pos_b": ObservationTermCfg(
      func=mdp.ref_body_pos_b, params={"command_name": "motion"}
    ),
    "ref_body_ori_b": ObservationTermCfg(
      func=mdp.ref_body_ori_b, params={"command_name": "motion"}
    ),
    "delta_body_pos_b": ObservationTermCfg(
      func=mdp.delta_body_pos_b, params={"command_name": "motion"}
    ),
    "delta_body_ori_b": ObservationTermCfg(
      func=mdp.delta_body_ori_b, params={"command_name": "motion"}
    ),
    "body_lin_vel_residual_w": ObservationTermCfg(
      func=mdp.body_lin_vel_residual_w, params={"command_name": "motion"}
    ),
    "body_ang_vel_residual_w": ObservationTermCfg(
      func=mdp.body_ang_vel_residual_w, params={"command_name": "motion"}
    ),
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
    "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }

  cfg.observations["policy"] = ObservationGroupCfg(
    terms=policy_terms,
    concatenate_terms=True,
    enable_corruption=False,
  )
  cfg.observations["critic"] = ObservationGroupCfg(
    terms=critic_terms,
    concatenate_terms=True,
    enable_corruption=False,
  )

  cfg.events = {}

  # Object interaction introduces more contacts/constraints than robot-only tracking.
  cfg.sim.nconmax = max(int(cfg.sim.nconmax), 128)
  cfg.sim.njmax = max(int(cfg.sim.njmax), 512)

  cfg.rewards = {
    "stage0_track": RewardTermCfg(
      func=mdp.stage0_track_reward,
      weight=1.0,
      params={
        "command_name": "motion",
        "anchor_pos_std": 0.3,
        "ee_pos_std": 0.3,
        "body_ori_std": 0.4,
        "body_lin_vel_std": 1.0,
        "body_ang_vel_std": 3.14,
        "ee_body_names": _STAGE0_EE_BODY_NAMES,
        "body_names": None,
        "joint_vel_scale": 1.0e-4,
        "action_rate_scale": 1.0e-2,
        "enable_interaction_reward": False,
        "interaction_body_names": _STAGE0_PALM_BODY_NAMES,
        "enable_contact_reward": False,
        "contact_body_names": _STAGE0_PALM_BODY_NAMES,
      },
    ),
    "joint_limit": RewardTermCfg(
      func=mdp.joint_pos_limits,
      weight=-10.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "self_collisions": RewardTermCfg(
      func=mdp.self_collision_cost,
      weight=-10.0,
      params={"sensor_name": "self_collision"},
    ),
  }

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.pose_range = {}
  motion_cmd.velocity_range = {}
  motion_cmd.sampling_mode = "start"

  return cfg
