"""Spectral-normalized actor model (Phase 1 sim2real smoothness fix).

Why: RL policies are not smooth by default — the Markov objective places no
constraint on how fast the action may change with the state, so the network
learns high-frequency, high-gain state->action mappings. In simulation (infinite
actuator bandwidth) that is free; on hardware (finite bandwidth) it shows up as
vibration / limit-cycle oscillation — exactly the reach arm buzz. The established
fix in the humanoid sim2real literature is to bound the policy network's Lipschitz
constant so nearby states map to nearby actions. Spectral normalization does this
directly and cheaply: it rescales each Linear weight by its largest singular value
so every layer is 1-Lipschitz, hence the whole actor is Lipschitz-bounded.

  - Learning Smooth Humanoid Locomotion through Lipschitz-Constrained Policies
    (arXiv:2410.11825)
  - Spectral Normalization for Lipschitz-Constrained Policies (arXiv:2504.08246)
  - CAPS: Regularizing Action Policies for Smooth Control (arXiv:2012.06644)

This is applied ONLY to the actor (the critic keeps the plain MLPModel), and only
to the policy MLP's Linear layers (not the obs normalizer). It runs at model
CONSTRUCTION — before PPO builds its optimizer over the actor parameters — so the
optimizer naturally tracks the reparametrized weights (`weight_orig`); no optimizer
rebuild is needed.

Wired via rl_cfg.py:
    actor=RslRlModelCfg(
        class_name="src.tasks.reaching_v2.rl.spectral_mlp:SpectralNormMLPModel",
        ...)

Tuning: standard spectral norm makes each layer exactly 1-Lipschitz. If reach
tracking ends up under-fit (arms too weak / sluggish), relax by either skipping
the final layer (see SKIP_LAST) or scaling the normalized weight up by a small
constant. If it still oscillates, this is already the strongest smoothness lever;
combine with the action-rate / arm_joint_vel reward penalties already in the task.
"""

from __future__ import annotations

import copy

import torch.nn as nn
from torch.nn.utils.parametrize import is_parametrized, remove_parametrizations
from torch.nn.utils.parametrizations import spectral_norm

from rsl_rl.models.mlp_model import MLPModel

# If True, leave the final Linear (action-mean head) un-normalized so the policy
# keeps full output range; only the hidden layers are Lipschitz-bounded. Start
# False (bound the whole net, matches the papers); flip to True if under-fit.
SKIP_LAST = False


class SpectralNormMLPModel(MLPModel):
  """MLPModel whose actor MLP Linear layers are spectral-normalized."""

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    linears = [m for m in self.mlp.modules() if isinstance(m, nn.Linear)]
    to_norm = linears[:-1] if (SKIP_LAST and len(linears) > 1) else linears
    for lin in to_norm:
      spectral_norm(lin)  # reparametrizes .weight -> weight/sigma(weight)

  def as_onnx(self, verbose: bool) -> nn.Module:
    # Export a parametrization-FREE copy so the deployed ONNX is plain Linear
    # layers with the (already normalized) weights baked in — no spectral-norm
    # machinery, no deploy-side surprises. leave_parametrized=True writes the
    # current normalized weight back as an ordinary Parameter.
    baked = copy.deepcopy(self)
    for m in baked.mlp.modules():
      if isinstance(m, nn.Linear) and is_parametrized(m, "weight"):
        remove_parametrizations(m, "weight", leave_parametrized=True)
    return MLPModel.as_onnx(baked, verbose)
