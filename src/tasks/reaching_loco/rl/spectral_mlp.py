"""Spectral-normalized actor for the loco-manipulation reaching task.

Same idea as reaching_v2/rl/spectral_mlp.py (bounds the actor's Lipschitz
constant so nearby states map to nearby actions -> smooth, sim2real-robust
control). See that file for the full rationale and refs (LCP arXiv:2410.11825,
spectral-norm 2504.08246, CAPS 2012.06644).

Difference: SKIP_LAST defaults to True here. The output layer is left
un-normalized so the LEG joints keep enough reactivity to walk/step; only the
hidden (shared representation) layers are Lipschitz-bounded. A whole-network
bound (SKIP_LAST=False) can over-smooth the fast leg responses balance/walking
need — flip it back only if the arms are still buzzy and legs have margin.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm

from rsl_rl.models.mlp_model import MLPModel

SKIP_LAST = True


class SpectralNormMLPModel(MLPModel):
  """MLPModel whose actor-MLP hidden Linear layers are spectral-normalized."""

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    linears = [m for m in self.mlp.modules() if isinstance(m, nn.Linear)]
    to_norm = linears[:-1] if (SKIP_LAST and len(linears) > 1) else linears
    for lin in to_norm:
      spectral_norm(lin)

  def as_onnx(self, verbose: bool) -> nn.Module:
    # Export with the BAKED normalized weights as plain Linear layers: build a
    # throwaway plain replica of self.mlp (reading child.weight = normalized),
    # swap it in, export, restore. Avoids deepcopy (crashes on parametrized
    # modules) and tracing the sigma computation (uses aten::vdot, unsupported
    # by ONNX). Never touches optimizer params, so training is unaffected.
    plain_layers = []
    for child in self.mlp:
      if isinstance(child, nn.Linear):
        lin = nn.Linear(
          child.in_features, child.out_features, bias=child.bias is not None
        )
        with torch.no_grad():
          lin.weight.copy_(child.weight.detach())
          if child.bias is not None:
            lin.bias.copy_(child.bias.detach())
        plain_layers.append(lin.to(child.weight.device))
      else:
        plain_layers.append(child)
    plain = nn.Sequential(*plain_layers)
    orig = self.mlp
    self.mlp = plain
    try:
      return super().as_onnx(verbose)
    finally:
      self.mlp = orig
