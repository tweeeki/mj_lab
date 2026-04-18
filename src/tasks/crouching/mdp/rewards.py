"""Reward terms specific to the crouching (height-tracking) task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

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
