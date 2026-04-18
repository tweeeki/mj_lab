"""Observability metrics for the crouching task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def pelvis_z(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_pos_w[:, 2]


class mean_knee_angle:
  """Per-env mean of commanded-knee joint angles (rad)."""

  def __init__(self, cfg: MetricsTermCfg, env: "ManagerBasedRlEnv"):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    joint_ids, _ = asset.find_joints(cfg.params["asset_cfg"].joint_names)
    self._joint_ids = torch.tensor(joint_ids, device=env.device, dtype=torch.long)

  def __call__(
    self, env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.joint_pos[:, self._joint_ids].mean(dim=1)


class torso_pitch:
  """Signed torso pitch (rad). Positive = leaning forward."""

  def __init__(self, cfg: MetricsTermCfg, env: "ManagerBasedRlEnv"):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    body_ids, _ = asset.find_bodies(cfg.params["asset_cfg"].body_names)
    self._body_id = int(body_ids[0])

  def __call__(
    self, env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    body_quat_w = asset.data.body_link_quat_w[:, self._body_id, :]
    gravity_w = asset.data.gravity_vec_w
    projected = quat_apply_inverse(body_quat_w, gravity_w)
    # projected[:, 0] ≈ -sin(pitch); positive pitch = leaning forward.
    return torch.asin(-projected[:, 0].clamp(-1.0, 1.0))
