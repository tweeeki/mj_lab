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
  # from {min, ..., max}. Models the deploy async staleness PLUS real command
  # transport lag: the controller keeps applying an OLD policy action until
  # the new one lands. v3 keeps a two-step target history, so delays up to
  # 2 * decimation substeps (two full policy steps, 40 ms) are honored exactly
  # instead of aliasing to one step as in v2.
  delay_substeps_range: tuple[int, int] = (0, 8)

  def build(self, env: "ManagerBasedRlEnv") -> "EmaJointPositionAction":
    return EmaJointPositionAction(self, env)


class EmaJointPositionAction(JointPositionAction):
  """JointPositionAction with per-substep EMA filtering and randomized delay."""

  def __init__(self, cfg: EmaJointPositionActionCfg, env: "ManagerBasedRlEnv"):
    super().__init__(cfg=cfg, env=env)
    self._ema_cfg = cfg
    self._physics_dt_ms = env.physics_dt * 1000.0
    # Substeps per policy step; delays in (decimation, 2*decimation] reach back
    # to the target from TWO steps ago (see apply_actions).
    self._decimation = max(1, round(env.step_dt / env.physics_dt))
    if cfg.delay_substeps_range[1] > 2 * self._decimation:
      raise ValueError(
        f"delay_substeps_range max {cfg.delay_substeps_range[1]} exceeds the "
        f"two-step history depth (2 * decimation = {2 * self._decimation})."
      )
    self._alpha_sub = torch.zeros(env.num_envs, 1, device=env.device)
    self._delay_sub = torch.zeros(env.num_envs, 1, dtype=torch.long, device=env.device)
    self._filt: torch.Tensor | None = None
    self._prev_target: torch.Tensor | None = None
    self._prev2_target: torch.Tensor | None = None
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
    # Shift the two-step target history so delayed envs can keep applying an
    # OLD target while the "new" action is still in flight (async policy
    # thread + transport emulation).
    if self._prev_target is not None:
      encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
      self._prev2_target = self._prev_target
      self._prev_target = self._processed_actions - encoder_bias
    super().process_actions(actions)
    self._substep_i = 0

  def apply_actions(self) -> None:
    encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
    new_target = self._processed_actions - encoder_bias
    if self._prev_target is None:
      self._prev_target = new_target.clone()
    if self._prev2_target is None:
      self._prev2_target = new_target.clone()
    # Pick the target by age: with delay d, the target produced at step k
    # becomes visible d substeps after it was produced. Substep i of the
    # current step is (i) substeps after `new_target`, (i + decimation) after
    # `prev`, and (i + 2*decimation) after `prev2`.
    use_new = self._substep_i >= self._delay_sub                       # (N, 1)
    use_prev = self._substep_i + self._decimation >= self._delay_sub  # (N, 1)
    target = torch.where(
      use_new, new_target, torch.where(use_prev, self._prev_target, self._prev2_target)
    )
    if self._filt is None:
      self._filt = target.clone()
    if self._needs_reseed.any():
      self._filt[self._needs_reseed] = new_target[self._needs_reseed]
      self._prev_target[self._needs_reseed] = new_target[self._needs_reseed]
      self._prev2_target[self._needs_reseed] = new_target[self._needs_reseed]
      self._needs_reseed[:] = False
    self._filt += self._alpha_sub * (target - self._filt)
    self._entity.set_joint_position_target(self._filt, joint_ids=self._target_ids)
    self._substep_i += 1
