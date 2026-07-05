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


class waypoint_track(_HandSites):
  """Tight tracking of the moving waypoint → enforces the commanded arm speed.

  The command term slides an internal waypoint from the current hand toward the
  final goal at the commanded speed (m/s). Rewarding the palms for staying
  *glued* to that moving waypoint (small ``std``) forces the arm to move at the
  waypoint's speed: lag → error grows; rushing ahead toward the final goal →
  ahead of the waypoint, error grows. When the waypoint reaches the goal and
  stops, the same term becomes a hold reward — so "move at X m/s" and "stop at
  the target" never conflict (cf. the twist-tracking reward, arXiv 2507.08656).

  ``self._targets_world`` reads cmd[:, 0:6], which is the waypoint (see
  ``UniformBimanualSphereCommand.command``), so this measures error to the
  waypoint, not the final goal.

  ``reward = exp(-(|d_L|^2 + |d_R|^2) / std^2)``.
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
    del command_name, left_hand_site, right_hand_site, asset_cfg
    left_t, right_t = self._targets_world(env)
    left_h, right_h = self._hands_world(env)
    d2 = (
      torch.sum(torch.square(left_h - left_t), dim=-1)
      + torch.sum(torch.square(right_h - right_t), dim=-1)
    )
    return torch.exp(-d2 / (std * std))


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


##
# Balance / posture anchors (v2). Non-saturating penalties that hold the lower
# body at the default standing pose while the arms reach. The framework's
# ``posture`` reward is exp-shaped and loses its gradient once the pose drifts
# past ~2 std — these L1/L2 terms keep pulling back no matter how far it drifts.
##


def base_height_l2(
  env: "ManagerBasedRlEnv",
  target_height: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Squared deviation of the root (pelvis) height from a fixed world target."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.square(asset.data.root_link_pos_w[:, 2] - target_height)


def root_lin_vel_z_l2(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Squared vertical velocity of the root (anti-hop / anti-bob)."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.square(asset.data.root_link_lin_vel_w[:, 2])


def joint_deviation_l1(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """L1 deviation from the default joint positions for the scoped joints."""
  asset: Entity = env.scene[asset_cfg.name]
  diff = (
    asset.data.joint_pos[:, asset_cfg.joint_ids]
    - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
  )
  return torch.sum(torch.abs(diff), dim=1)


def feet_contact(env: "ManagerBasedRlEnv", sensor_name: str) -> torch.Tensor:
  """1.0 while BOTH feet are in ground contact, else 0."""
  sensor = env.scene[sensor_name]
  found = sensor.data.found
  assert found is not None
  in_contact = (found > 0).reshape(env.num_envs, -1)
  return in_contact.all(dim=1).float()


def feet_motion_l2(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Squared linear velocity of the foot sites — feet should stay planted."""
  asset: Entity = env.scene[asset_cfg.name]
  vel = asset.data.site_lin_vel_w[:, asset_cfg.site_ids]
  return torch.sum(torch.square(vel), dim=(1, 2))


def self_collision_cost(
  env: "ManagerBasedRlEnv",
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Penalize self-collisions (copied from the velocity task).

  When the sensor provides force history (from ``history_length > 0``),
  counts substeps where any contact force exceeds *force_threshold*.
  Falls back to the instantaneous ``found`` count otherwise.
  """
  sensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    hit = (force_mag > force_threshold).any(dim=1)  # [B, H]
    return hit.sum(dim=-1).float()  # [B]
  assert data.found is not None
  return data.found.squeeze(-1)
