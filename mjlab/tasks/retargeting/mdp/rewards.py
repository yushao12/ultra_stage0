from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as F

from mjlab.utils.lab_api.math import quat_apply, quat_error_magnitude

from mjlab.tasks.tracking.mdp.commands import MotionCommand

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv


def _get_body_indexes(
  command: MotionCommand, body_names: tuple[str, ...] | None
) -> list[int]:
  return [
    i
    for i, name in enumerate(command.cfg.body_names)
    if (body_names is None) or (name in body_names)
  ]


def _exp_tracking(error: torch.Tensor, std: float) -> torch.Tensor:
  std_sq = max(std * std, 1e-8)
  return torch.exp(-error / std_sq)


def _default_interaction_body_names(command: MotionCommand) -> tuple[str, ...]:
  if command.interaction_body_names_ref:
    return command.interaction_body_names_ref
  return (
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
  )


def _resolve_body_names(
  command: MotionCommand,
  body_names: tuple[str, ...] | None,
) -> tuple[str, ...]:
  if body_names is None:
    candidates = _default_interaction_body_names(command)
  else:
    candidates = body_names

  body_name_set = set(command.cfg.body_names)
  resolved = tuple(name for name in candidates if name in body_name_set)
  return resolved


def _get_optional_object_entity(
  env: ManagerBasedRlEnv,
  command: MotionCommand,
  object_entity_name: str | None,
) -> Entity | None:
  entity_name = object_entity_name
  if entity_name is None:
    entity_name = command.cfg.object_entity_name
  if entity_name is None:
    return None
  if entity_name not in env.scene.entities:
    return None
  return cast("Entity", env.scene[entity_name])


def _object_points_world(
  object_pos_w: torch.Tensor,
  object_quat_w: torch.Tensor,
  points_local: torch.Tensor,
) -> torch.Tensor:
  num_envs = object_pos_w.shape[0]
  num_points = points_local.shape[0]
  points_local_b = points_local.unsqueeze(0).expand(num_envs, num_points, 3)
  object_quat_b = object_quat_w.unsqueeze(1).expand(num_envs, num_points, 4)
  return object_pos_w.unsqueeze(1) + quat_apply(object_quat_b, points_local_b)


def _contact_from_distance(
  body_pos_w: torch.Tensor,
  object_points_w: torch.Tensor,
  threshold: float,
) -> torch.Tensor:
  # body_pos_w: [B, I, 3], object_points_w: [B, J, 3]
  pairwise_dist_sq = torch.sum(
    torch.square(body_pos_w[:, :, None, :] - object_points_w[:, None, :, :]),
    dim=-1,
  )
  min_dist = torch.sqrt(torch.clamp(pairwise_dist_sq.min(dim=-1).values, min=0.0))
  return (min_dist < threshold).float()


def _prepare_interaction_weights(
  command: MotionCommand,
  selected_names: tuple[str, ...],
  num_points: int,
) -> torch.Tensor:
  num_links = len(selected_names)
  default = torch.full(
    (num_links, num_points),
    fill_value=1.0 / max(num_links * num_points, 1),
    device=command.device,
  )

  raw = command.interaction_weights_ref
  if raw is None:
    return default

  weights = raw
  if weights.ndim == 1:
    if weights.shape[0] != num_points:
      return default
    weights = weights.unsqueeze(0).expand(num_links, num_points)
  elif weights.ndim == 2:
    pass
  else:
    return default

  if weights.shape[0] != num_links:
    interaction_names = command.interaction_body_names_ref
    if interaction_names and weights.shape[0] == len(interaction_names):
      row_indexes = []
      for name in selected_names:
        if name not in interaction_names:
          return default
        row_indexes.append(interaction_names.index(name))
      weights = weights[row_indexes]
    elif weights.shape[0] == len(command.cfg.body_names):
      row_indexes = [command.cfg.body_names.index(name) for name in selected_names]
      weights = weights[row_indexes]
    else:
      return default

  if weights.shape[1] != num_points:
    if weights.shape[1] < num_points:
      return default
    weights = weights[:, :num_points]

  weights = torch.clamp(weights, min=0.0)
  weight_sum = torch.sum(weights)
  if weight_sum <= 1e-8:
    return default
  return weights / weight_sum


def _reference_contact_from_labels(
  command: MotionCommand,
  selected_names: tuple[str, ...],
) -> torch.Tensor | None:
  labels = command.contact_labels_ref
  if labels is None:
    return None
  if labels.ndim == 1:
    labels = labels.unsqueeze(-1)

  num_labels = labels.shape[1]
  if num_labels == len(selected_names):
    return torch.clamp(labels, min=0.0, max=1.0)

  contact_names = command.contact_body_names_ref
  if contact_names and num_labels == len(contact_names):
    indexes = []
    for name in selected_names:
      if name not in contact_names:
        return None
      indexes.append(contact_names.index(name))
    return torch.clamp(labels[:, indexes], min=0.0, max=1.0)

  if num_labels == len(command.cfg.body_names):
    indexes = [command.cfg.body_names.index(name) for name in selected_names]
    return torch.clamp(labels[:, indexes], min=0.0, max=1.0)

  if num_labels > len(selected_names):
    return torch.clamp(labels[:, : len(selected_names)], min=0.0, max=1.0)

  return None


def stage0_sparse_body_position_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]
    ),
    dim=-1,
  ).mean(-1)
  return _exp_tracking(error, std)


def stage0_body_orientation_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = quat_error_magnitude(
    command.body_quat_relative_w[:, body_indexes],
    command.robot_body_quat_w[:, body_indexes],
  )
  error_sq = torch.square(error).mean(-1)
  return _exp_tracking(error_sq, std)


def stage0_body_linear_velocity_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]
    ),
    dim=-1,
  ).mean(-1)
  return _exp_tracking(error, std)


def stage0_body_angular_velocity_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]
    ),
    dim=-1,
  ).mean(-1)
  return _exp_tracking(error, std)


def stage0_energy_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  joint_vel_scale: float = 1.0e-4,
  action_rate_scale: float = 1.0e-2,
) -> torch.Tensor:
  del command_name
  joint_vel = env.scene["robot"].data.joint_vel
  joint_vel_penalty = torch.sum(torch.square(joint_vel), dim=1)
  action_delta = env.action_manager.action - env.action_manager.prev_action
  action_rate_penalty = torch.sum(torch.square(action_delta), dim=1)
  penalty = joint_vel_scale * joint_vel_penalty + action_rate_scale * action_rate_penalty
  return torch.exp(-penalty)


def stage0_object_tracking_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_entity_name: str | None = None,
  object_pos_coef: float = 5.0,
  object_rot_coef: float = 0.5,
  object_lin_vel_coef: float = 0.1,
  object_rot_huber_delta: float = 1.0,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  if not command.has_object_reference:
    return torch.ones(env.num_envs, device=command.device)

  object_entity = _get_optional_object_entity(env, command, object_entity_name)
  if object_entity is None:
    return torch.ones(env.num_envs, device=command.device)

  object_pose_w = object_entity.data.root_link_pose_w
  object_vel_w = object_entity.data.root_link_vel_w
  sim_object_pos_w = object_pose_w[:, 0:3]
  sim_object_quat_w = object_pose_w[:, 3:7]
  sim_object_lin_vel_w = object_vel_w[:, 0:3]

  ref_object_pos_w = command.object_pos_w
  ref_object_quat_w = command.object_quat_w
  ref_object_lin_vel_w = command.object_lin_vel_w

  pos_err = torch.sum(torch.square(sim_object_pos_w - ref_object_pos_w), dim=-1)
  rot_err = quat_error_magnitude(sim_object_quat_w, ref_object_quat_w)
  rot_huber = F.huber_loss(
    rot_err,
    torch.zeros_like(rot_err),
    reduction="none",
    delta=object_rot_huber_delta,
  )
  vel_err = torch.sum(torch.square(sim_object_lin_vel_w - ref_object_lin_vel_w), dim=-1)

  return (
    torch.exp(-object_pos_coef * pos_err)
    * torch.exp(-object_rot_coef * rot_huber)
    * torch.exp(-object_lin_vel_coef * vel_err)
  )


def stage0_interaction_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_entity_name: str | None = None,
  interaction_body_names: tuple[str, ...] | None = None,
  interaction_coef: float = 20.0,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  if not command.has_object_reference:
    return torch.ones(env.num_envs, device=command.device)

  object_entity = _get_optional_object_entity(env, command, object_entity_name)
  if object_entity is None:
    return torch.ones(env.num_envs, device=command.device)

  points_local = command.interaction_points_local_ref
  if points_local is None:
    return torch.ones(env.num_envs, device=command.device)

  selected_names = _resolve_body_names(command, interaction_body_names)
  if not selected_names:
    return torch.ones(env.num_envs, device=command.device)

  body_indexes = [command.cfg.body_names.index(name) for name in selected_names]

  ref_body_pos_w = command.body_pos_w[:, body_indexes]
  sim_body_pos_w = command.robot_body_pos_w[:, body_indexes]

  ref_points_w = _object_points_world(command.object_pos_w, command.object_quat_w, points_local)

  object_pose_w = object_entity.data.root_link_pose_w
  sim_points_w = _object_points_world(
    object_pose_w[:, 0:3],
    object_pose_w[:, 3:7],
    points_local,
  )

  ref_delta = ref_body_pos_w[:, :, None, :] - ref_points_w[:, None, :, :]
  sim_delta = sim_body_pos_w[:, :, None, :] - sim_points_w[:, None, :, :]
  delta_err = torch.sum(torch.square(sim_delta - ref_delta), dim=-1)

  weights = _prepare_interaction_weights(command, selected_names, points_local.shape[0])
  weighted_err = torch.sum(delta_err * weights.unsqueeze(0), dim=(1, 2))
  return torch.exp(-interaction_coef * weighted_err)


def stage0_contact_matching_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_entity_name: str | None = None,
  contact_body_names: tuple[str, ...] | None = None,
  contact_sensor_name: str | None = None,
  contact_distance_threshold: float = 0.02,
  contact_coef: float = 5.0,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  if not command.has_object_reference:
    return torch.ones(env.num_envs, device=command.device)

  object_entity = _get_optional_object_entity(env, command, object_entity_name)
  if object_entity is None:
    return torch.ones(env.num_envs, device=command.device)

  selected_names = _resolve_body_names(command, contact_body_names)
  if not selected_names:
    return torch.ones(env.num_envs, device=command.device)

  body_indexes = [command.cfg.body_names.index(name) for name in selected_names]
  ref_body_pos_w = command.body_pos_w[:, body_indexes]
  sim_body_pos_w = command.robot_body_pos_w[:, body_indexes]

  points_local = command.interaction_points_local_ref
  if points_local is None:
    return torch.ones(env.num_envs, device=command.device)

  ref_points_w = _object_points_world(command.object_pos_w, command.object_quat_w, points_local)

  object_pose_w = object_entity.data.root_link_pose_w
  sim_points_w = _object_points_world(
    object_pose_w[:, 0:3],
    object_pose_w[:, 3:7],
    points_local,
  )

  ref_contact = _reference_contact_from_labels(command, selected_names)
  if ref_contact is None:
    ref_contact = _contact_from_distance(
      ref_body_pos_w,
      ref_points_w,
      threshold=contact_distance_threshold,
    )

  sim_contact = None
  if contact_sensor_name is not None and contact_sensor_name in env.scene.sensors:
    sensor = env.scene[contact_sensor_name]
    if sensor.data.found is not None:
      sim_contact_raw = (sensor.data.found > 0).float()
      if sim_contact_raw.ndim == 1:
        sim_contact_raw = sim_contact_raw.unsqueeze(-1)
      if sim_contact_raw.shape[1] == len(selected_names):
        sim_contact = sim_contact_raw
      elif sim_contact_raw.shape[1] == len(command.cfg.body_names):
        sim_contact = sim_contact_raw[:, body_indexes]
      elif sim_contact_raw.shape[1] > len(selected_names):
        sim_contact = sim_contact_raw[:, : len(selected_names)]

  if sim_contact is None:
    sim_contact = _contact_from_distance(
      sim_body_pos_w,
      sim_points_w,
      threshold=contact_distance_threshold,
    )

  contact_err = torch.abs(sim_contact - ref_contact).mean(dim=-1)
  return torch.exp(-contact_coef * contact_err)


def stage0_track_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  anchor_pos_std: float,
  ee_pos_std: float,
  body_ori_std: float,
  body_lin_vel_std: float,
  body_ang_vel_std: float,
  ee_body_names: tuple[str, ...] | None = None,
  body_names: tuple[str, ...] | None = None,
  joint_vel_scale: float = 1.0e-4,
  action_rate_scale: float = 1.0e-2,
  object_entity_name: str | None = None,
  object_pos_coef: float = 5.0,
  object_rot_coef: float = 0.5,
  object_lin_vel_coef: float = 0.1,
  object_rot_huber_delta: float = 1.0,
  enable_interaction_reward: bool = False,
  interaction_body_names: tuple[str, ...] | None = None,
  interaction_coef: float = 20.0,
  enable_contact_reward: bool = False,
  contact_body_names: tuple[str, ...] | None = None,
  contact_sensor_name: str | None = None,
  contact_distance_threshold: float = 0.02,
  contact_coef: float = 5.0,
) -> torch.Tensor:
  """Stage-0 product reward in ULTRA.

  Default setup only uses object trajectory tracking, while interaction/contact
  terms are gated for future label-driven experiments.
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  anchor_error = torch.sum(
    torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1
  )
  r_anchor = _exp_tracking(anchor_error, anchor_pos_std)

  r_p = stage0_sparse_body_position_reward(
    env=env,
    command_name=command_name,
    std=ee_pos_std,
    body_names=ee_body_names,
  )
  r_r = stage0_body_orientation_reward(
    env=env,
    command_name=command_name,
    std=body_ori_std,
    body_names=body_names,
  )
  r_v = stage0_body_linear_velocity_reward(
    env=env,
    command_name=command_name,
    std=body_lin_vel_std,
    body_names=body_names,
  )
  r_w = stage0_body_angular_velocity_reward(
    env=env,
    command_name=command_name,
    std=body_ang_vel_std,
    body_names=body_names,
  )

  r_obj = stage0_object_tracking_reward(
    env=env,
    command_name=command_name,
    object_entity_name=object_entity_name,
    object_pos_coef=object_pos_coef,
    object_rot_coef=object_rot_coef,
    object_lin_vel_coef=object_lin_vel_coef,
    object_rot_huber_delta=object_rot_huber_delta,
  )

  if enable_interaction_reward:
    r_int = stage0_interaction_reward(
      env=env,
      command_name=command_name,
      object_entity_name=object_entity_name,
      interaction_body_names=interaction_body_names,
      interaction_coef=interaction_coef,
    )
  else:
    r_int = torch.ones(env.num_envs, device=command.device)

  if enable_contact_reward:
    r_ct = stage0_contact_matching_reward(
      env=env,
      command_name=command_name,
      object_entity_name=object_entity_name,
      contact_body_names=contact_body_names,
      contact_sensor_name=contact_sensor_name,
      contact_distance_threshold=contact_distance_threshold,
      contact_coef=contact_coef,
    )
  else:
    r_ct = torch.ones(env.num_envs, device=command.device)

  r_eng = stage0_energy_reward(
    env=env,
    command_name=command_name,
    joint_vel_scale=joint_vel_scale,
    action_rate_scale=action_rate_scale,
  )

  return r_anchor * r_p * r_r * r_v * r_w * r_obj * r_int * r_ct * r_eng
