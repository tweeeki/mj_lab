"""Reward terms specific to the crouching (height-tracking) task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def track_pelvis_height(
  env: "ManagerBasedRlEnv",
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward matching the commanded base (pelvis) height.

  Uses the root link world-z as the achieved height. On flat terrain this is
  equivalent to the pelvis height above the ground.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual_z = asset.data.root_link_pos_w[:, 2]
  target_z = command[:, 0]
  error = torch.square(actual_z - target_z)
  return torch.exp(-error / std**2)


def base_linear_velocity_penalty(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize xy base velocity so the robot stays in place while crouching."""
  asset: Entity = env.scene[asset_cfg.name]
  lin_vel_xy = asset.data.root_link_lin_vel_b[:, :2]
  return torch.sum(torch.square(lin_vel_xy), dim=1)


def pelvis_height_obs(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Current pelvis world-z as a scalar observation."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_pos_w[:, 2:3]


def body_orientation_l2(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize non-upright orientation of a named body (e.g. torso_link).

  Falls back to root projected gravity if no body_ids are selected.
  """
  asset: Entity = env.scene[asset_cfg.name]
  if asset_cfg.body_ids:
    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :].squeeze(1)
    gravity_w = asset.data.gravity_vec_w
    projected = quat_apply_inverse(body_quat_w, gravity_w)
    return torch.sum(torch.square(projected[:, :2]), dim=1)
  return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def forward_lean(
  env: "ManagerBasedRlEnv",
  target_pitch: float,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward a target forward pitch of the torso.

  Uses the x-component of projected gravity as a pitch proxy:
  it is ~-sin(pitch), so a small negative value corresponds to a forward lean.
  Roll is penalized (y-component pulled to zero). Matches a Gaussian around the
  target so the reward is bounded in [0, 1].
  """
  asset: Entity = env.scene[asset_cfg.name]
  pg = asset.data.projected_gravity_b
  pitch_err = torch.square(pg[:, 0] - target_pitch)
  roll_err = torch.square(pg[:, 1])
  return torch.exp(-(pitch_err + roll_err) / std**2)


def feet_slip(
  env: "ManagerBasedRlEnv",
  sensor_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize foot xy sliding while in contact. No command gating."""
  asset: Entity = env.scene[asset_cfg.name]
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  in_contact = (sensor.data.found > 0).float()  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_sq = torch.sum(torch.square(foot_vel_xy), dim=-1)  # [B, N]
  return torch.sum(vel_sq * in_contact, dim=1)


def feet_air_time(
  env: "ManagerBasedRlEnv",
  sensor_name: str,
) -> torch.Tensor:
  """Penalize any foot being off the ground — feet should stay planted while crouching."""
  sensor: ContactSensor = env.scene[sensor_name]
  air_time = sensor.data.current_air_time  # [B, N]
  return torch.sum(air_time, dim=1)


def on_target_bonus(
  env: "ManagerBasedRlEnv",
  command_name: str,
  threshold: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Binary bonus awarded each step the pelvis is within *threshold* of the target."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  err = torch.abs(asset.data.root_link_pos_w[:, 2] - command[:, 0])
  return (err < threshold).float()


class track_knee_angle:
  """Reward knee flexion matching a linear function of the commanded height.

  Map: q_target(z_cmd) interpolates linearly between (anchor_high, q_high) and
  (anchor_low, q_low). Exponential reward on mean squared error across the
  selected knee joints; paying out only when knee flexion *actually matches*
  what the target height requires forces the policy to squat with its knees
  instead of dropping the pelvis by tipping the torso.
  """

  def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRlEnv"):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    joint_ids, _ = asset.find_joints(cfg.params["asset_cfg"].joint_names)
    self._joint_ids = torch.tensor(joint_ids, device=env.device, dtype=torch.long)

  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    command_name: str,
    std: float,
    anchor_high: float,
    anchor_low: float,
    q_high: float,
    q_low: float,
    asset_cfg: SceneEntityCfg,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    z_cmd = command[:, 0:1]  # [B, 1]
    slope = (q_low - q_high) / (anchor_low - anchor_high)
    q_target = q_high + (z_cmd - anchor_high) * slope  # [B, 1]
    q_actual = asset.data.joint_pos[:, self._joint_ids]  # [B, K]
    err = torch.mean(torch.square(q_actual - q_target), dim=1)
    return torch.exp(-err / std**2)
