"""Uniform height command for crouching tasks.

Samples a scalar target pelvis (base) height per environment and resamples on a
configurable time interval, mirroring the structure of ``UniformVelocityCommand``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
  import viser

  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class UniformHeightCommand(CommandTerm):
  """Commanded base height sampled uniformly from a range."""

  cfg: "UniformHeightCommandCfg"

  def __init__(self, cfg: "UniformHeightCommandCfg", env: "ManagerBasedRlEnv"):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.entity_name]
    self.height_command = torch.zeros(self.num_envs, 1, device=self.device)

    self.metrics["error_height"] = torch.zeros(self.num_envs, device=self.device)

    self._joystick_enabled: "viser.GuiCheckboxHandle | None" = None
    self._joystick_slider: "viser.GuiSliderHandle | None" = None
    self._joystick_get_env_idx: Callable[[], int] | None = None

  @property
  def command(self) -> torch.Tensor:
    return self.height_command

  def _update_metrics(self) -> None:
    max_command_time = self.cfg.resampling_time_range[1]
    max_command_step = max_command_time / self._env.step_dt
    actual_height = self.robot.data.root_link_pos_w[:, 2]
    self.metrics["error_height"] += (
      torch.abs(self.height_command[:, 0] - actual_height) / max_command_step
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    r = torch.empty(len(env_ids), device=self.device)
    self.height_command[env_ids, 0] = r.uniform_(*self.cfg.ranges.height)

  def _update_command(self) -> None:
    pass

  def create_gui(
    self,
    name: str,
    server: "viser.ViserServer",
    get_env_idx: Callable[[], int],
  ) -> None:
    """Add a single slider to drive the commanded height in the viewer."""
    lo, hi = self.cfg.ranges.height
    with server.gui.add_folder(name.capitalize()):
      enabled = server.gui.add_checkbox("Enable", initial_value=False)
      slider = server.gui.add_slider(
        "height",
        min=lo,
        max=hi,
        step=0.01,
        initial_value=0.5 * (lo + hi),
      )
    self._joystick_enabled = enabled
    self._joystick_slider = slider
    self._joystick_get_env_idx = get_env_idx

  def compute(self, dt: float) -> None:
    super().compute(dt)
    if (
      self._joystick_enabled is not None
      and self._joystick_enabled.value
      and self._joystick_slider is not None
      and self._joystick_get_env_idx is not None
    ):
      idx = self._joystick_get_env_idx()
      self.height_command[idx, 0] = self._joystick_slider.value

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    """Draw a small horizontal disc at the commanded height above each env origin."""
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return
    base_pos_ws = self.robot.data.root_link_pos_w.cpu().numpy()
    cmds = self.height_command.cpu().numpy()
    for batch in env_indices:
      base_pos_w = base_pos_ws[batch]
      if np.linalg.norm(base_pos_w) < 1e-6:
        continue
      target = np.array([base_pos_w[0], base_pos_w[1], float(cmds[batch, 0])])
      # Short vertical arrow from ground up to the commanded height.
      ground = np.array([base_pos_w[0], base_pos_w[1], 0.0])
      visualizer.add_arrow(ground, target, color=(1.0, 0.6, 0.1, 0.8), width=0.015)


@dataclass(kw_only=True)
class UniformHeightCommandCfg(CommandTermCfg):
  entity_name: str

  @dataclass
  class Ranges:
    height: tuple[float, float]

  ranges: Ranges

  @dataclass
  class VizCfg:
    pass

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: "ManagerBasedRlEnv") -> UniformHeightCommand:
    return UniformHeightCommand(self, env)
