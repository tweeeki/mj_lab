"""Reward terms specific to the crouching (height-tracking) task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

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


class knee_height_coupling:
  """HOMIE-style coupling between knee flexion and pelvis-height error.

  Rewards the product (h - h*) * (q_knee_norm - 0.5). The product is positive
  exactly when the knees are on the "correct side" of mid-range for the current
  height error: too tall -> knees flexed; too short -> knees extended. Use a
  positive weight so this is a shaping reward on directional knee motion.
  """

  def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRlEnv"):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    joint_ids, _ = asset.find_joints(cfg.params["asset_cfg"].joint_names)
    self._joint_ids = torch.tensor(joint_ids, device=env.device, dtype=torch.long)
    lim = asset.data.soft_joint_pos_limits  # [B, J, 2]
    assert lim is not None
    self._lo = lim[:, self._joint_ids, 0]  # [B, K]
    self._hi = lim[:, self._joint_ids, 1]  # [B, K]

  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    command_name: str,
    asset_cfg: SceneEntityCfg,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    h = asset.data.root_link_pos_w[:, 2:3]
    h_star = command[:, :1]
    dh = h - h_star  # [B, 1]. >0: too tall -> want flexion.
    q = asset.data.joint_pos[:, self._joint_ids]  # [B, K]
    q_norm = (q - self._lo) / (self._hi - self._lo).clamp(min=1e-3)
    coupling = dh * (q_norm - 0.5)  # [B, K]; positive when aligned.
    return torch.sum(coupling, dim=1)
