from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro

from mjlab.entity import Entity
from mjlab.scene import Scene
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.tasks.retargeting.config.g1.env_cfgs import unitree_g1_flat_retargeting_env_cfg
from mjlab.utils.lab_api.math import axis_angle_from_quat, quat_apply, quat_conjugate, quat_mul


G1_JOINT_NAMES = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
)


def _lerp(a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
  return a * (1.0 - blend) + b * blend


def _slerp(q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
  q0 = torch.nn.functional.normalize(q0, dim=-1)
  q1 = torch.nn.functional.normalize(q1, dim=-1)

  if t.ndim == q0.ndim - 1:
    t = t.unsqueeze(-1)

  dot = (q0 * q1).sum(dim=-1, keepdim=True)
  q1 = torch.where(dot < 0.0, -q1, q1)
  dot = (q0 * q1).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)

  theta = torch.acos(dot)
  sin_theta = torch.sin(theta)
  close = sin_theta.abs() < eps

  s0 = torch.sin((1.0 - t) * theta) / (sin_theta + eps)
  s1 = torch.sin(t * theta) / (sin_theta + eps)
  out = s0 * q0 + s1 * q1

  out = torch.where(close, (1.0 - t) * q0 + t * q1, out)
  return torch.nn.functional.normalize(out, dim=-1)


def _so3_derivative(rotations: torch.Tensor, dt: float) -> torch.Tensor:
  if rotations.shape[0] < 3:
    return torch.zeros((rotations.shape[0], 3), dtype=rotations.dtype, device=rotations.device)
  q_prev, q_next = rotations[:-2], rotations[2:]
  q_rel = quat_mul(q_next, quat_conjugate(q_prev))
  omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
  return torch.cat([omega[:1], omega, omega[-1:]], dim=0)


def _make_box_surface_points_local(half_extents: tuple[float, float, float]) -> torch.Tensor:
  hx, hy, hz = half_extents
  pts = torch.tensor(
    [
      [hx, 0.0, 0.0],
      [-hx, 0.0, 0.0],
      [0.0, hy, 0.0],
      [0.0, -hy, 0.0],
      [0.0, 0.0, hz],
      [0.0, 0.0, -hz],
      [hx, hy, hz],
      [hx, hy, -hz],
      [hx, -hy, hz],
      [hx, -hy, -hz],
      [-hx, hy, hz],
      [-hx, hy, -hz],
      [-hx, -hy, hz],
      [-hx, -hy, -hz],
    ],
    dtype=torch.float32,
  )
  return pts


def _world_object_points(object_pos_w: torch.Tensor, object_quat_w: torch.Tensor, points_local: torch.Tensor) -> torch.Tensor:
  num_frames = object_pos_w.shape[0]
  num_points = points_local.shape[0]
  local = points_local.unsqueeze(0).expand(num_frames, num_points, 3)
  quat = object_quat_w.unsqueeze(1).expand(num_frames, num_points, 4)
  return object_pos_w.unsqueeze(1) + quat_apply(quat, local)


def _compute_contact_labels(
  body_pos_w: torch.Tensor,
  object_points_w: torch.Tensor,
  contact_threshold: float,
) -> torch.Tensor:
  pairwise_dist_sq = torch.sum(
    torch.square(body_pos_w[:, :, None, :] - object_points_w[:, None, :, :]),
    dim=-1,
  )
  min_dist = torch.sqrt(torch.clamp(pairwise_dist_sq.min(dim=-1).values, min=0.0))
  return (min_dist < contact_threshold).to(dtype=torch.float32)


@dataclass
class Args:
  input_npz: str
  output_npz: str
  output_fps: float = 50.0
  device: str = "cpu"
  infer_has_object: bool = True
  has_object: bool = False
  export_contact_labels: bool = False
  contact_body_names: tuple[str, ...] = (
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
  )
  contact_threshold: float = 0.02
  object_half_extents: tuple[float, float, float] = (0.2, 0.2, 0.2)


def main(args: Args) -> None:
  data = np.load(args.input_npz, allow_pickle=True)
  if "qpos" not in data.files:
    raise ValueError("input npz must contain key 'qpos'")

  qpos = torch.tensor(data["qpos"], dtype=torch.float32)
  input_fps = float(np.array(data.get("fps", 30.0)).reshape(-1)[0])

  joint_dof = len(G1_JOINT_NAMES)
  expected_robot_dim = 4 + 3 + joint_dof
  has_object = args.has_object
  if args.infer_has_object:
    has_object = qpos.shape[1] >= expected_robot_dim + 7

  if qpos.shape[1] < expected_robot_dim:
    raise ValueError(
      f"qpos dim {qpos.shape[1]} is smaller than expected robot dim {expected_robot_dim}"
    )

  input_dt = 1.0 / input_fps
  output_dt = 1.0 / args.output_fps
  duration = max((qpos.shape[0] - 1) * input_dt, output_dt)

  times = torch.arange(0.0, duration, output_dt, dtype=torch.float32)
  phase = times / duration
  index_0 = (phase * (qpos.shape[0] - 1)).floor().long()
  index_1 = torch.minimum(index_0 + 1, torch.tensor(qpos.shape[0] - 1))
  blend = phase * (qpos.shape[0] - 1) - index_0

  base_quat_input = qpos[:, 0:4]
  base_pos_input = qpos[:, 4:7]
  joint_pos_input = qpos[:, 7 : 7 + joint_dof]

  base_pos = _lerp(base_pos_input[index_0], base_pos_input[index_1], blend.unsqueeze(-1))
  base_quat = _slerp(base_quat_input[index_0], base_quat_input[index_1], blend)
  joint_pos = _lerp(joint_pos_input[index_0], joint_pos_input[index_1], blend.unsqueeze(-1))

  base_lin_vel = torch.gradient(base_pos, spacing=output_dt, dim=0)[0]
  base_ang_vel = _so3_derivative(base_quat, output_dt)
  joint_vel = torch.gradient(joint_pos, spacing=output_dt, dim=0)[0]

  object_pos = object_quat = object_lin_vel = object_ang_vel = None
  if has_object:
    object_quat_input = qpos[:, -7:-3]
    object_pos_input = qpos[:, -3:]
    object_pos = _lerp(object_pos_input[index_0], object_pos_input[index_1], blend.unsqueeze(-1))
    object_quat = _slerp(object_quat_input[index_0], object_quat_input[index_1], blend)
    object_lin_vel = torch.gradient(object_pos, spacing=output_dt, dim=0)[0]
    object_ang_vel = _so3_derivative(object_quat, output_dt)

  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = output_dt

  env_cfg = unitree_g1_flat_retargeting_env_cfg(play=False)
  scene = Scene(env_cfg.scene, device=args.device)
  model = scene.compile()
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=args.device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  robot: Entity = scene["robot"]
  robot_joint_indexes = robot.find_joints(G1_JOINT_NAMES, preserve_order=True)[0]

  logs: dict[str, list[np.ndarray]] = {
    "joint_pos": [],
    "joint_vel": [],
    "body_pos_w": [],
    "body_quat_w": [],
    "body_lin_vel_w": [],
    "body_ang_vel_w": [],
  }

  num_frames = base_pos.shape[0]
  for frame_id in range(num_frames):
    root_state = robot.data.default_root_state.clone()
    root_state[:, 0:3] = base_pos[frame_id : frame_id + 1].to(args.device)
    root_state[:, :2] += scene.env_origins[:, :2]
    root_state[:, 3:7] = base_quat[frame_id : frame_id + 1].to(args.device)
    root_state[:, 7:10] = base_lin_vel[frame_id : frame_id + 1].to(args.device)
    root_state[:, 10:13] = base_ang_vel[frame_id : frame_id + 1].to(args.device)
    robot.write_root_state_to_sim(root_state)

    joint_pos_t = robot.data.default_joint_pos.clone()
    joint_vel_t = robot.data.default_joint_vel.clone()
    joint_pos_t[:, robot_joint_indexes] = joint_pos[frame_id : frame_id + 1].to(args.device)
    joint_vel_t[:, robot_joint_indexes] = joint_vel[frame_id : frame_id + 1].to(args.device)
    robot.write_joint_state_to_sim(joint_pos_t, joint_vel_t)

    sim.forward()
    scene.update(sim.mj_model.opt.timestep)

    logs["joint_pos"].append(robot.data.joint_pos[0].cpu().numpy().copy())
    logs["joint_vel"].append(robot.data.joint_vel[0].cpu().numpy().copy())
    logs["body_pos_w"].append(robot.data.body_link_pos_w[0].cpu().numpy().copy())
    logs["body_quat_w"].append(robot.data.body_link_quat_w[0].cpu().numpy().copy())
    logs["body_lin_vel_w"].append(robot.data.body_link_lin_vel_w[0].cpu().numpy().copy())
    logs["body_ang_vel_w"].append(robot.data.body_link_ang_vel_w[0].cpu().numpy().copy())

  out: dict[str, np.ndarray] = {
    "fps": np.array([args.output_fps], dtype=np.float32),
    "joint_pos": np.stack(logs["joint_pos"], axis=0).astype(np.float32),
    "joint_vel": np.stack(logs["joint_vel"], axis=0).astype(np.float32),
    "body_pos_w": np.stack(logs["body_pos_w"], axis=0).astype(np.float32),
    "body_quat_w": np.stack(logs["body_quat_w"], axis=0).astype(np.float32),
    "body_lin_vel_w": np.stack(logs["body_lin_vel_w"], axis=0).astype(np.float32),
    "body_ang_vel_w": np.stack(logs["body_ang_vel_w"], axis=0).astype(np.float32),
    "joint_names": np.array(G1_JOINT_NAMES, dtype=object),
    "body_names": np.array(robot.body_names, dtype=object),
  }

  if has_object and object_pos is not None and object_quat is not None:
    object_pos_w = object_pos + scene.env_origins[0:1, :]
    out["object_pos_w"] = object_pos_w.cpu().numpy().astype(np.float32)
    out["object_quat_w"] = object_quat.cpu().numpy().astype(np.float32)
    assert object_lin_vel is not None
    assert object_ang_vel is not None
    out["object_lin_vel_w"] = object_lin_vel.cpu().numpy().astype(np.float32)
    out["object_ang_vel_w"] = object_ang_vel.cpu().numpy().astype(np.float32)

    if args.export_contact_labels:
      points_local = _make_box_surface_points_local(args.object_half_extents)
      object_points_w = _world_object_points(
        object_pos_w.to(dtype=torch.float32),
        object_quat.to(dtype=torch.float32),
        points_local,
      )

      body_name_to_idx = {name: i for i, name in enumerate(robot.body_names)}
      missing = [name for name in args.contact_body_names if name not in body_name_to_idx]
      if missing:
        raise ValueError(f"contact body names not found in robot model: {missing}")

      contact_idx = [body_name_to_idx[name] for name in args.contact_body_names]
      body_pos_for_contact = torch.tensor(
        out["body_pos_w"][:, contact_idx],
        dtype=torch.float32,
      )
      contact_labels = _compute_contact_labels(
        body_pos_for_contact,
        object_points_w,
        contact_threshold=args.contact_threshold,
      )

      num_links = len(args.contact_body_names)
      num_points = points_local.shape[0]
      weights = torch.full(
        (num_links, num_points),
        fill_value=1.0 / max(num_links * num_points, 1),
        dtype=torch.float32,
      )

      out["contact_labels"] = contact_labels.cpu().numpy().astype(np.float32)
      out["contact_body_names"] = np.array(args.contact_body_names, dtype=object)
      out["interaction_points_local"] = points_local.cpu().numpy().astype(np.float32)
      out["interaction_weights"] = weights.cpu().numpy().astype(np.float32)
      out["interaction_body_names"] = np.array(args.contact_body_names, dtype=object)

  output_path = Path(args.output_npz)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  np.savez(output_path, **out)
  print(f"[OK] Saved: {output_path}")
  print(
    f"[OK] Frames: {num_frames}, has_object={has_object}, "
    f"export_contact_labels={args.export_contact_labels}"
  )


if __name__ == "__main__":
  main(tyro.cli(Args))
