"""Observation terms for the bimanual sphere-reaching task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


class hand_pos_b:
  """Left + right hand positions in the robot root frame. Shape (num_envs, 6)."""

  def __init__(self, cfg: ObservationTermCfg, env: "ManagerBasedRlEnv"):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    left_ids, _ = asset.find_sites(cfg.params["left_hand_site"])
    right_ids, _ = asset.find_sites(cfg.params["right_hand_site"])
    assert len(left_ids) == 1 and len(right_ids) == 1
    self._left = int(left_ids[0])
    self._right = int(right_ids[0])

  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    left_hand_site: str,
    right_hand_site: str,
    asset_cfg: SceneEntityCfg,
  ) -> torch.Tensor:
    del left_hand_site, right_hand_site
    asset: Entity = env.scene[asset_cfg.name]
    root_pos = asset.data.root_link_pos_w
    root_quat = asset.data.root_link_quat_w
    left_w = asset.data.site_pos_w[:, self._left]
    right_w = asset.data.site_pos_w[:, self._right]
    left_b = quat_apply_inverse(root_quat, left_w - root_pos)
    right_b = quat_apply_inverse(root_quat, right_w - root_pos)
    return torch.cat([left_b, right_b], dim=-1)


class hand_target_error_b:
  """Per-hand target-minus-hand vectors in root frame. Shape (num_envs, 6)."""

  def __init__(self, cfg: ObservationTermCfg, env: "ManagerBasedRlEnv"):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    left_ids, _ = asset.find_sites(cfg.params["left_hand_site"])
    right_ids, _ = asset.find_sites(cfg.params["right_hand_site"])
    assert len(left_ids) == 1 and len(right_ids) == 1
    self._left = int(left_ids[0])
    self._right = int(right_ids[0])

  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    command_name: str,
    left_hand_site: str,
    right_hand_site: str,
    asset_cfg: SceneEntityCfg,
  ) -> torch.Tensor:
    del left_hand_site, right_hand_site
    asset: Entity = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    assert cmd is not None
    root_pos = asset.data.root_link_pos_w
    root_quat = asset.data.root_link_quat_w
    left_w = asset.data.site_pos_w[:, self._left]
    right_w = asset.data.site_pos_w[:, self._right]
    # Targets are already root-frame in the command; hand positions need transforming.
    hand_left_b = quat_apply_inverse(root_quat, left_w - root_pos)
    hand_right_b = quat_apply_inverse(root_quat, right_w - root_pos)
    err_left = cmd[:, 0:3] - hand_left_b
    err_right = cmd[:, 3:6] - hand_right_b
    return torch.cat([err_left, err_right], dim=-1)
