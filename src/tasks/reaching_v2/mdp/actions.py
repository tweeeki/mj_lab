"""Action terms for the balance-anchored reaching task (v2).

``EmaJointPositionAction`` reproduces, inside training, the whole-body action
EMA the deploy controller (``State_RLBase.cpp``) applies at 1 kHz between the
50 Hz policy updates and the motors:

    filt += alpha * (target - filt)        (per 1 kHz tick)

Why: the v2 policy trained WITHOUT this filter turned out to have a thin
actuation-lag margin — under the deploy EMA it orbits/vibrates around the
target. Training with the filter in the loop makes the policy stable under it
(standard actuator-lag modeling), so deploy can keep the EMA for its real job
(smoothing the 50 Hz action staircase into the motors).

Details:
- The filter runs in ``apply_actions``, which the env calls once per physics
  substep (200 Hz) — the time constant is matched to the 1 kHz deploy filter
  via the per-substep alpha conversion below.
- The 1 kHz-equivalent alpha is sampled log-uniformly per env at every episode
  reset, so the policy is robust across the whole deploy band instead of tuned
  to one value.
- Deploy-safe: action/obs dims are unchanged and ``last_action`` still
  observes the RAW policy output, exactly like the deploy obs pipeline.
- The filter re-seeds at the current target on episode reset, mirroring the
  deploy filter's re-seed on state entry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.actions.actions import (
  JointPositionAction,
  JointPositionActionCfg,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class EmaJointPositionActionCfg(JointPositionActionCfg):
  """Joint-position action with deploy-matched EMA + action delay in the loop."""

  # Range of the 1 kHz-equivalent EMA alpha, sampled log-uniformly per env at
  # every episode reset. Deploy hardware default is 0.10 (colleague-proven);
  # sim2sim currently runs 0.5. Training across [0.06, 0.5] makes the deployed
  # alpha a free knob anywhere in that band.
  ema_alpha_1khz_range: tuple[float, float] = (0.06, 0.5)

  # Action delay in physics substeps, sampled uniformly per env per episode
  # from {min, ..., max}. Models the deploy async staleness: the controller
  # keeps applying the PREVIOUS policy action until the policy thread delivers
  # the new one (up to one full 20 ms step = `decimation` substeps late).
  delay_substeps_range: tuple[int, int] = (0, 4)

  def build(self, env: "ManagerBasedRlEnv") -> "EmaJointPositionAction":
    return EmaJointPositionAction(self, env)


class EmaJointPositionAction(JointPositionAction):
  """JointPositionAction with per-substep EMA filtering and randomized delay."""

  def __init__(self, cfg: EmaJointPositionActionCfg, env: "ManagerBasedRlEnv"):
    super().__init__(cfg=cfg, env=env)
    self._ema_cfg = cfg
    self._physics_dt_ms = env.physics_dt * 1000.0
    self._alpha_sub = torch.zeros(env.num_envs, 1, device=env.device)
    self._delay_sub = torch.zeros(env.num_envs, 1, dtype=torch.long, device=env.device)
    self._filt: torch.Tensor | None = None
    self._prev_target: torch.Tensor | None = None
    self._substep_i = 0
    self._needs_reseed = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    self._resample_lag(slice(None))

  def _resample_lag(self, env_ids: torch.Tensor | slice) -> None:
    lo, hi = self._ema_cfg.ema_alpha_1khz_range
    dlo, dhi = self._ema_cfg.delay_substeps_range
    n = self._alpha_sub.shape[0] if isinstance(env_ids, slice) else len(env_ids)
    dev = self._alpha_sub.device
    u = torch.rand(n, 1, device=dev)
    a_1k = torch.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))
    # Same continuous time constant as the 1 kHz deploy filter:
    # (1 - a_sub) = (1 - a_1k)^(substep_ms / 1 ms)
    self._alpha_sub[env_ids] = 1.0 - (1.0 - a_1k) ** self._physics_dt_ms
    self._delay_sub[env_ids] = torch.randint(dlo, dhi + 1, (n, 1), device=dev)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    ids = slice(None) if env_ids is None else env_ids
    self._resample_lag(ids)
    self._needs_reseed[ids] = True

  def process_actions(self, actions: torch.Tensor) -> None:
    # Keep the previous step's motor target so delayed envs can hold it while
    # the "new" action is still in flight (async policy thread emulation).
    if self._prev_target is not None:
      encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
      self._prev_target = self._processed_actions - encoder_bias
    super().process_actions(actions)
    self._substep_i = 0

  def apply_actions(self) -> None:
    encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
    new_target = self._processed_actions - encoder_bias
    if self._prev_target is None:
      self._prev_target = new_target.clone()
    # Envs whose delay has not elapsed this step still track the OLD target.
    stale = self._substep_i < self._delay_sub  # (N, 1) bool
    target = torch.where(stale, self._prev_target, new_target)
    if self._filt is None:
      self._filt = target.clone()
    if self._needs_reseed.any():
      self._filt[self._needs_reseed] = new_target[self._needs_reseed]
      self._prev_target[self._needs_reseed] = new_target[self._needs_reseed]
      self._needs_reseed[:] = False
    self._filt += self._alpha_sub * (target - self._filt)
    self._entity.set_joint_position_target(self._filt, joint_ids=self._target_ids)
    self._substep_i += 1


@dataclass(kw_only=True)
class DeltaArmEmaJointPositionActionCfg(EmaJointPositionActionCfg):
  """EMA action where the ARM joints are driven as a rate-capped INCREMENTAL
  (delta) target instead of an absolute one — the SlimZorgLab mechanism, in the
  training loop so the policy LEARNS within the cap (a deploy-side rate cap on a
  policy trained without it winds up and oscillates; a cap in the action space
  is learned-around).

  Arm columns ``[arm_joint_start, arm_joint_end)`` of the processed action are
  integrated each policy step:

      arm_target[t] = clamp(arm_target[t-1] + arm_delta_max_per_step * a,
                            default ± arm_max_excursion)

  with ``a = clamp(raw_action, -1, 1)``. So one 50 Hz policy step can move an arm
  joint by at most ``arm_delta_max_per_step`` rad — a HARD rate cap of
  ``arm_delta_max_per_step * policy_hz`` rad/s. A high-frequency oscillation is
  then mechanically bounded to amplitude ≤ rate_cap / (2*pi*f). Legs + waist
  (columns < arm_joint_start) stay ABSOLUTE so balance keeps full reaction speed.
  The EMA + delay (inherited) still run on top, matching the deploy 1 kHz filter.

  DEPLOY CONTRACT CHANGES: the deploy C++ must accumulate arm deltas the same
  way (q_arm += arm_delta_max_per_step * clamp(raw,-1,1), clamped) before its
  EMA. Obs/action DIMS are unchanged (last_action still sees the raw output), but
  grabbing only the onnx is NOT enough this time — State_RLBase must be updated.
  """

  # 29-DoF trainer order: legs 0-11, waist 12-14, arms 15-28 (exclusive end).
  arm_joint_start: int = 15
  arm_joint_end: int = 29
  # Max arm move per 50 Hz policy step. a=±1 -> ±this rad. 0.02 rad/step ~= a
  # 1.0 rad/s hard cap (SlimZorg used ~0.31 rad/s). Lower = calmer/slower.
  arm_delta_max_per_step: float = 0.02
  # Anti-windup: clamp the integrated arm target to default ± this (rad).
  arm_max_excursion: float = 2.0

  def build(self, env: "ManagerBasedRlEnv") -> "DeltaArmEmaJointPositionAction":
    return DeltaArmEmaJointPositionAction(self, env)


class DeltaArmEmaJointPositionAction(EmaJointPositionAction):
  """EmaJointPositionAction whose arm columns are a rate-capped integrator."""

  def __init__(self, cfg: DeltaArmEmaJointPositionActionCfg, env: "ManagerBasedRlEnv"):
    super().__init__(cfg=cfg, env=env)
    self._d_cfg = cfg
    self._arm = slice(cfg.arm_joint_start, cfg.arm_joint_end)
    # `_offset` is the default joint pose (use_default_offset=True). Seed the
    # arm integrator there and precompute the anti-windup clamp band.
    self._arm_target = self._offset[:, self._arm].clone()
    self._arm_lo = self._offset[:, self._arm] - cfg.arm_max_excursion
    self._arm_hi = self._offset[:, self._arm] + cfg.arm_max_excursion

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    ids = slice(None) if env_ids is None else env_ids
    # Re-seed the integrator at the default pose so there is no jump at reset.
    self._arm_target[ids] = self._offset[ids, self._arm]

  def process_actions(self, actions: torch.Tensor) -> None:
    # Parent computes the ABSOLUTE processed target (raw*scale+offset) and does
    # the delay bookkeeping; it also fills `_raw_actions` with the policy output.
    super().process_actions(actions)
    # Overwrite the ARM columns with the rate-capped integrated delta target.
    a = torch.clamp(self._raw_actions[:, self._arm], -1.0, 1.0)
    self._arm_target = torch.clamp(
      self._arm_target + self._d_cfg.arm_delta_max_per_step * a,
      self._arm_lo,
      self._arm_hi,
    )
    self._processed_actions[:, self._arm] = self._arm_target
