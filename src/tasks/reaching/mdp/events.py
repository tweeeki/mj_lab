"""Event terms specific to the bimanual reaching task.

``push_torso_force`` applies a sustained *horizontal* external force to the
robot's torso (forward/back = x, left/right = y), to train balance robustness
against a carried/tilting load. It is adapted from
``mjlab.envs.mdp.events.apply_body_impulse`` — same sustain/cooldown/reset
lifecycle — but takes **per-axis** force ranges so the vertical (z) component can
be held at 0, which the isotropic built-in cannot do.

Training-only: this is an event, not an observation, so the policy learns to
react to the push purely from its proprioception/IMU (exactly how push-robustness
training works). It does NOT change the observation vector or the deploy contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply, sample_uniform

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer

__all__ = ["push_torso_force"]

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


class push_torso_force:
  """Sustained horizontal push on the torso, with random duration + cooldown.

  Lifecycle per env (mirrors ``apply_body_impulse``):

  1. **Cooldown** — idle for a random ``cooldown_s`` (no force applied).
  2. **Trigger** — sample a force ``(fx, fy, fz)`` with fx from ``force_range_x``
     (fwd/back), fy from ``force_range_y`` (left/right), fz from ``force_range_z``
     (default ``(0, 0)`` = no vertical), plus an optional isotropic torque from
     ``torque_range``. If ``body_point_offset`` is set, the horizontal force is
     applied off the CoM, adding a tilt torque ``offset x force`` — the leverage
     a held/tilting box exerts on the torso.
  3. **Sustain** — hold the wrench for a random ``duration_s``.
  4. **Expire** — zero the wrench and restart the cooldown.

  Forces are cleared on episode reset (see ``reset``) so nothing leaks across
  episodes. Use with ``mode="step"``.
  """

  def __init__(self, cfg, env: "ManagerBasedRlEnv"):
    self._asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    self._body_ids = cfg.params["asset_cfg"].body_ids
    self._num_envs = env.num_envs
    self._device = env.device
    self._step_dt = env.step_dt
    self._num_bodies = (
      len(self._body_ids)
      if isinstance(self._body_ids, list)
      else self._asset.num_bodies
    )
    # Per-env timers: time left on the active push, and time left until the next.
    self._time_remaining = torch.zeros(self._num_envs, device=self._device)
    self._interval_time_left = torch.zeros(self._num_envs, device=self._device)
    self._active = torch.zeros(self._num_envs, device=self._device, dtype=torch.bool)

  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    force_range_x: tuple[float, float],
    force_range_y: tuple[float, float],
    duration_s: tuple[float, float],
    cooldown_s: tuple[float, float],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    force_range_z: tuple[float, float] = (0.0, 0.0),
    torque_range: tuple[float, float] = (0.0, 0.0),
    body_point_offset: tuple[float, float, float] | None = None,
  ) -> None:
    del env, env_ids, asset_cfg  # baked at init; step events act on all envs.
    dt = self._step_dt

    # Decrement active-push timers.
    self._time_remaining[self._active] -= dt

    # Expire finished pushes: zero their wrench, start their cooldown.
    expired = self._active & (self._time_remaining <= 0)
    if expired.any():
      expired_ids = expired.nonzero(as_tuple=False).squeeze(-1)
      zeros = torch.zeros((len(expired_ids), self._num_bodies, 3), device=self._device)
      self._asset.write_external_wrench_to_sim(
        zeros, zeros, env_ids=expired_ids, body_ids=self._body_ids
      )
      self._active[expired_ids] = False
      self._time_remaining[expired_ids] = 0.0
      cd_low, cd_high = cooldown_s
      self._interval_time_left[expired_ids] = (
        torch.rand(len(expired_ids), device=self._device) * (cd_high - cd_low) + cd_low
      )

    # Decrement cooldown timers; trigger a new push where cooldown has elapsed.
    self._interval_time_left -= dt
    eligible = (~self._active) & (self._interval_time_left <= 0)
    if not eligible.any():
      return
    trigger_ids = eligible.nonzero(as_tuple=False).squeeze(-1)
    n = len(trigger_ids)

    # Per-axis horizontal force (z from force_range_z, default 0).
    forces = torch.empty((n, self._num_bodies, 3), device=self._device)
    forces[..., 0] = sample_uniform(*force_range_x, (n, self._num_bodies), self._device)
    forces[..., 1] = sample_uniform(*force_range_y, (n, self._num_bodies), self._device)
    forces[..., 2] = sample_uniform(*force_range_z, (n, self._num_bodies), self._device)
    torques = sample_uniform(*torque_range, (n, self._num_bodies, 3), self._device)

    # Off-CoM application point → tilt torque (leverage of a carried box).
    if body_point_offset is not None:
      offset_local = torch.tensor(
        body_point_offset, device=self._device, dtype=torch.float32
      )
      body_quat = self._asset.data.body_com_quat_w[trigger_ids][:, self._body_ids]
      offset_w = quat_apply(
        body_quat.reshape(-1, 4), offset_local.expand(n * self._num_bodies, 3)
      ).reshape(n, self._num_bodies, 3)
      torques = torques + torch.cross(offset_w, forces, dim=-1)

    self._asset.write_external_wrench_to_sim(
      forces, torques, env_ids=trigger_ids, body_ids=self._body_ids
    )

    # Set sustain + next-cooldown timers.
    dur_low, dur_high = duration_s
    self._time_remaining[trigger_ids] = (
      torch.rand(n, device=self._device) * (dur_high - dur_low) + dur_low
    )
    self._active[trigger_ids] = True
    cd_low, cd_high = cooldown_s
    self._interval_time_left[trigger_ids] = (
      torch.rand(n, device=self._device) * (cd_high - cd_low) + cd_low
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    """Clear any active push on reset so forces never leak across episodes."""
    if env_ids is None:
      env_ids = slice(None)
    if not bool(self._active[env_ids].any()):
      # Still clear the timers/flags for the reset envs.
      self._active[env_ids] = False
      self._time_remaining[env_ids] = 0.0
      self._interval_time_left[env_ids] = 0.0
      return
    if isinstance(env_ids, slice):
      active_ids = self._active.nonzero(as_tuple=False).squeeze(-1)
    else:
      active_ids = env_ids[self._active[env_ids]]
    if len(active_ids) > 0:
      zeros = torch.zeros((len(active_ids), self._num_bodies, 3), device=self._device)
      self._asset.write_external_wrench_to_sim(
        zeros, zeros, env_ids=active_ids, body_ids=self._body_ids
      )
    self._active[env_ids] = False
    self._time_remaining[env_ids] = 0.0
    self._interval_time_left[env_ids] = 0.0
