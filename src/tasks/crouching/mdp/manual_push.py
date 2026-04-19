"""Manual push command for interactive perturbation in the viser viewer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
  import viser
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


class ManualPushCommand(CommandTerm):
  cfg: "ManualPushCommandCfg"

  def __init__(self, cfg: "ManualPushCommandCfg", env: "ManagerBasedRlEnv"):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    self._command = torch.zeros(self.num_envs, 0, device=self.device)
    self._pending_push: dict | None = None
    self._get_env_idx: Callable[[], int] | None = None
    self._magnitude_slider = None

  @property
  def command(self) -> torch.Tensor:
    return self._command

  def _update_metrics(self) -> None:
    pass

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    pass

  def _update_command(self) -> None:
    if self._pending_push is None or self._get_env_idx is None:
      return
    idx = self._get_env_idx()
    vel = self.robot.data.root_link_vel_w[idx].clone()
    vel[0] += self._pending_push["x"]
    vel[1] += self._pending_push["y"]
    vel[2] += self._pending_push["z"]
    vel[3] += self._pending_push["roll"]
    vel[4] += self._pending_push["pitch"]
    vel[5] += self._pending_push["yaw"]
    self.robot.write_root_link_velocity_to_sim(
      vel.unsqueeze(0),
      env_ids=torch.tensor([idx], device=self.device, dtype=torch.long),
    )
    self._pending_push = None

  def create_gui(self, name, server, get_env_idx):
    self._get_env_idx = get_env_idx
    with server.gui.add_folder("Manual push"):
      magnitude = server.gui.add_slider(
        "magnitude", min=0.1, max=2.0, step=0.1,
        initial_value=self.cfg.default_magnitude,
      )
      self._magnitude_slider = magnitude

      def _mk(dx=0.0, dy=0.0, dz=0.0, droll=0.0, dpitch=0.0, dyaw=0.0):
        def _cb(_event):
          s = self._magnitude_slider.value if self._magnitude_slider else 1.0
          self._pending_push = {
            "x": dx * s, "y": dy * s, "z": dz * s,
            "roll": droll * s, "pitch": dpitch * s, "yaw": dyaw * s,
          }
        return _cb

      server.gui.add_button("Push +X (forward)").on_click(_mk(dx=+1.0))
      server.gui.add_button("Push -X (back)").on_click(_mk(dx=-1.0))
      server.gui.add_button("Push +Y (left)").on_click(_mk(dy=+1.0))
      server.gui.add_button("Push -Y (right)").on_click(_mk(dy=-1.0))
      server.gui.add_button("Push +Z (up)").on_click(_mk(dz=+1.0))
      server.gui.add_button("Push -Z (down)").on_click(_mk(dz=-1.0))
      server.gui.add_button("Yaw +").on_click(_mk(dyaw=+1.0))
      server.gui.add_button("Yaw -").on_click(_mk(dyaw=-1.0))


@dataclass(kw_only=True)
class ManualPushCommandCfg(CommandTermCfg):
  entity_name: str
  default_magnitude: float = 0.5

  @dataclass
  class VizCfg:
    pass

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: "ManagerBasedRlEnv") -> ManualPushCommand:
    return ManualPushCommand(self, env)