"""Bimanual sphere-target command for reaching tasks.

Samples two 3D target points per environment (one per hand), expressed in the
robot's root frame, and resamples on a configurable time interval. Structure
mirrors ``UniformHeightCommand``.

The command tensor shape is ``(num_envs, 6)`` where
``command[:, 0:3] = left_target_root_frame`` and
``command[:, 3:6] = right_target_root_frame``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class UniformBimanualSphereCommand(CommandTerm):
  """Two per-env 3D reach targets sampled uniformly from per-axis ranges.

  Targets live in the robot root frame (shoulder-relative feel). We sample
  each coordinate independently per hand; the left target is sampled from a
  half-space with ``y > 0`` (robot-left) and the right from ``y < 0``, enforced
  via the configured ranges, so the two spheres never cross over.
  """

  cfg: "UniformBimanualSphereCommandCfg"

  def __init__(self, cfg: "UniformBimanualSphereCommandCfg", env: "ManagerBasedRlEnv"):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.entity_name]
    # target_b   = the FINAL sampled goal per hand (N, 6), root frame.
    # waypoint_b = the MOVING target the policy actually chases (N, 6): it starts
    #   at the current hand on each resample and slews toward target_b at `speed`
    #   metres/second, so tracking it tightly == moving the arm at `speed`.
    # speed      = ONE commanded arm speed shared by both hands (N,) — the random
    #   variable, sampled per resample from cfg.speed_range. This is the number
    #   we expose to the policy and command at deploy (sim2sim + sim2real).
    self.target_b = torch.zeros(self.num_envs, 6, device=self.device)
    self.waypoint_b = torch.zeros(self.num_envs, 6, device=self.device)
    self.speed = torch.zeros(self.num_envs, device=self.device)

    self.metrics["error_left"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_right"] = torch.zeros(self.num_envs, device=self.device)

    left_ids, _ = self.robot.find_sites(cfg.left_hand_site)
    right_ids, _ = self.robot.find_sites(cfg.right_hand_site)
    assert len(left_ids) == 1 and len(right_ids) == 1, (
      f"Expected exactly one site each for "
      f"{cfg.left_hand_site!r} and {cfg.right_hand_site!r}"
    )
    self._left_site_id = int(left_ids[0])
    self._right_site_id = int(right_ids[0])

  @property
  def command(self) -> torch.Tensor:
    # (N, 7) = [left_waypoint(3), right_waypoint(3), speed(1)]. Downstream obs
    # (`generated_commands`) sees the whole vector; `hand_target_error_b` and the
    # reward terms read cmd[:, 0:6] and so measure error to the MOVING waypoint,
    # which is exactly what makes the policy track it at the commanded speed.
    return torch.cat([self.waypoint_b, self.speed.unsqueeze(-1)], dim=-1)

  def _current_hands_b(self, env_ids: torch.Tensor) -> torch.Tensor:
    """Current hand positions (both hands) in the root frame, shape (n, 6)."""
    root_pos = self.robot.data.root_link_pos_w[env_ids]
    root_quat = self.robot.data.root_link_quat_w[env_ids]
    left_w = self.robot.data.site_pos_w[env_ids, self._left_site_id]
    right_w = self.robot.data.site_pos_w[env_ids, self._right_site_id]
    left_b = quat_apply_inverse(root_quat, left_w - root_pos)
    right_b = quat_apply_inverse(root_quat, right_w - root_pos)
    return torch.cat([left_b, right_b], dim=-1)

  def _update_metrics(self) -> None:
    max_command_time = self.cfg.resampling_time_range[1]
    max_command_step = max_command_time / self._env.step_dt
    left_w, right_w = self._targets_world()
    hand_left_w = self.robot.data.site_pos_w[:, self._left_site_id]
    hand_right_w = self.robot.data.site_pos_w[:, self._right_site_id]
    self.metrics["error_left"] += (
      torch.norm(hand_left_w - left_w, dim=-1) / max_command_step
    )
    self.metrics["error_right"] += (
      torch.norm(hand_right_w - right_w, dim=-1) / max_command_step
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    r = self.cfg.ranges
    sample = lambda lo, hi: torch.empty(n, device=self.device).uniform_(lo, hi)
    self.target_b[env_ids, 0] = sample(*r.left_x)
    self.target_b[env_ids, 1] = sample(*r.left_y)
    self.target_b[env_ids, 2] = sample(*r.left_z)
    self.target_b[env_ids, 3] = sample(*r.right_x)
    self.target_b[env_ids, 4] = sample(*r.right_y)
    self.target_b[env_ids, 5] = sample(*r.right_z)
    # One shared commanded arm speed for both hands (the random variable).
    self.speed[env_ids] = sample(*self.cfg.speed_range)
    # Seed the moving waypoint at the current hand so there is no slam on
    # resample (error ~= 0), then it slews to target_b at `speed` in _update.
    self.waypoint_b[env_ids] = self._current_hands_b(env_ids)

  def _update_command(self) -> None:
    # Advance each hand's waypoint toward its final target by speed * step_dt,
    # clamped so it never overshoots. Both hands use the SAME speed; they may
    # arrive at different times if their travel distances differ.
    dt = self._env.step_dt
    max_step = (self.speed * dt).unsqueeze(-1)  # (N, 1)
    for s in (0, 3):
      d = self.target_b[:, s : s + 3] - self.waypoint_b[:, s : s + 3]
      dist = torch.norm(d, dim=-1, keepdim=True)  # (N, 1)
      step = torch.minimum(max_step, dist)
      direction = d / (dist + 1e-6)
      self.waypoint_b[:, s : s + 3] += direction * step

  def _targets_world(self) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate body-frame targets by the root quaternion and add the root xyz.

    Note: we rotate but do NOT translate the z-axis intentionally — the
    targets are anchored to the *pelvis* frame, so as the robot walks or
    leans, the targets follow. This matches the source task where goals
    move with the shoulders.
    """
    root_pos = self.robot.data.root_link_pos_w
    root_quat = self.robot.data.root_link_quat_w
    left_b = self.target_b[:, 0:3]
    right_b = self.target_b[:, 3:6]
    left_w = root_pos + quat_apply(root_quat, left_b)
    right_w = root_pos + quat_apply(root_quat, right_b)
    return left_w, right_w

  def _waypoints_world(self) -> tuple[torch.Tensor, torch.Tensor]:
    """Same transform as _targets_world but for the moving waypoint."""
    root_pos = self.robot.data.root_link_pos_w
    root_quat = self.robot.data.root_link_quat_w
    left_w = root_pos + quat_apply(root_quat, self.waypoint_b[:, 0:3])
    right_w = root_pos + quat_apply(root_quat, self.waypoint_b[:, 3:6])
    return left_w, right_w

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return
    # Solid spheres = the moving waypoints the policy chases; faint spheres =
    # the final goals they slew toward.
    wl_w, wr_w = self._waypoints_world()
    tl_w, tr_w = self._targets_world()
    wl, wr = wl_w.cpu().numpy(), wr_w.cpu().numpy()
    tl, tr = tl_w.cpu().numpy(), tr_w.cpu().numpy()
    radius = 0.04
    for batch in env_indices:
      visualizer.add_sphere(wl[batch], radius=radius, color=(0.2, 0.6, 1.0, 0.7))
      visualizer.add_sphere(wr[batch], radius=radius, color=(1.0, 0.4, 0.4, 0.7))
      visualizer.add_sphere(tl[batch], radius=radius * 0.6, color=(0.2, 0.6, 1.0, 0.25))
      visualizer.add_sphere(tr[batch], radius=radius * 0.6, color=(1.0, 0.4, 0.4, 0.25))


@dataclass(kw_only=True)
class UniformBimanualSphereCommandCfg(CommandTermCfg):
  entity_name: str
  left_hand_site: str
  right_hand_site: str

  @dataclass
  class Ranges:
    left_x: tuple[float, float]
    left_y: tuple[float, float]
    left_z: tuple[float, float]
    right_x: tuple[float, float]
    right_y: tuple[float, float]
    right_z: tuple[float, float]

  ranges: Ranges

  # One shared commanded arm speed (m/s) for both hands, sampled uniformly from
  # this range each resample. This is the "random variable"; the same scalar is
  # commanded at deploy. Tune bounds to the robot's comfortable arm-tip speed.
  speed_range: tuple[float, float] = (0.05, 0.50)

  @dataclass
  class VizCfg:
    pass

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: "ManagerBasedRlEnv") -> UniformBimanualSphereCommand:
    return UniformBimanualSphereCommand(self, env)
