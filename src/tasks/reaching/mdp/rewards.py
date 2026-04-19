"""Reward terms for the bimanual sphere-reaching task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


class _HandSites:
  """Shared lazy lookup of the two hand site ids + command term handle."""

  def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRlEnv"):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    self._asset_name = cfg.params["asset_cfg"].name
    left_ids, _ = asset.find_sites(cfg.params["left_hand_site"])
    right_ids, _ = asset.find_sites(cfg.params["right_hand_site"])
    assert len(left_ids) == 1 and len(right_ids) == 1
    self._left_site_id = int(left_ids[0])
    self._right_site_id = int(right_ids[0])
    self._command_name = cfg.params["command_name"]

  def _targets_world(self, env: "ManagerBasedRlEnv") -> tuple[torch.Tensor, torch.Tensor]:
    asset: Entity = env.scene[self._asset_name]
    cmd = env.command_manager.get_command(self._command_name)
    assert cmd is not None
    root_pos = asset.data.root_link_pos_w
    root_quat = asset.data.root_link_quat_w
    left_w = root_pos + quat_apply(root_quat, cmd[:, 0:3])
    right_w = root_pos + quat_apply(root_quat, cmd[:, 3:6])
    return left_w, right_w

  def _hands_world(self, env: "ManagerBasedRlEnv") -> tuple[torch.Tensor, torch.Tensor]:
    asset: Entity = env.scene[self._asset_name]
    return (
      asset.data.site_pos_w[:, self._left_site_id],
      asset.data.site_pos_w[:, self._right_site_id],
    )


class reach_distance(_HandSites):
  """Exponential reward on the sum of L2 distances hand→target, per env.

  ``reward = exp(-(|d_L| + |d_R|) / std)`` so 0 error → 1, and the reward
  decays smoothly as either hand drifts.
  """

  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    command_name: str,
    left_hand_site: str,
    right_hand_site: str,
    std: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    del command_name, left_hand_site, right_hand_site, asset_cfg  # baked at init.
    left_t, right_t = self._targets_world(env)
    left_h, right_h = self._hands_world(env)
    d_left = torch.norm(left_h - left_t, dim=-1)
    d_right = torch.norm(right_h - right_t, dim=-1)
    return torch.exp(-(d_left + d_right) / std)


class reach_distance_l2(_HandSites):
  """Sum of squared hand→target distances (penalty term, weight should be negative)."""

  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    command_name: str,
    left_hand_site: str,
    right_hand_site: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    del command_name, left_hand_site, right_hand_site, asset_cfg
    left_t, right_t = self._targets_world(env)
    left_h, right_h = self._hands_world(env)
    return (
      torch.sum(torch.square(left_h - left_t), dim=-1)
      + torch.sum(torch.square(right_h - right_t), dim=-1)
    )


class reach_success_bonus(_HandSites):
  """Binary bonus: both hands within ``threshold`` meters of their targets."""

  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    command_name: str,
    left_hand_site: str,
    right_hand_site: str,
    threshold: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    del command_name, left_hand_site, right_hand_site, asset_cfg
    left_t, right_t = self._targets_world(env)
    left_h, right_h = self._hands_world(env)
    d_left = torch.norm(left_h - left_t, dim=-1)
    d_right = torch.norm(right_h - right_t, dim=-1)
    return ((d_left < threshold) & (d_right < threshold)).float()


class palm_facing_inward(_HandSites):
  """Encourage each palm's local +x (outward-from-hand normal) to point at the target.

  We read the palm site quaternion, rotate the site's local +x into world,
  and reward alignment with the unit vector from palm to target.
  This is a soft substitute for the source task's fixed-angle orientation
  check and works for any target location.
  """

  _LOCAL_X = None  # set per-device in __call__

  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    command_name: str,
    left_hand_site: str,
    right_hand_site: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    del command_name, left_hand_site, right_hand_site, asset_cfg
    asset: Entity = env.scene[self._asset_name]
    site_quat = asset.data.site_quat_w
    left_q = site_quat[:, self._left_site_id]
    right_q = site_quat[:, self._right_site_id]

    local_x = torch.zeros(env.num_envs, 3, device=env.device)
    local_x[:, 0] = 1.0
    left_normal = quat_apply(left_q, local_x)
    right_normal = quat_apply(right_q, local_x)

    left_t, right_t = self._targets_world(env)
    left_h, right_h = self._hands_world(env)
    left_dir = _safe_normalize(left_t - left_h)
    right_dir = _safe_normalize(right_t - right_h)

    left_align = torch.sum(left_normal * left_dir, dim=-1)
    right_align = torch.sum(right_normal * right_dir, dim=-1)
    # Map [-1, 1] → [0, 1] so this is a clean bounded bonus.
    return 0.5 * (left_align + right_align) * 0.5 + 0.5


def _safe_normalize(v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
  return v / (torch.norm(v, dim=-1, keepdim=True) + eps)
