import os

import wandb
import yaml

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner


# Maps mjlab-side observation-term keys to the names the deploy-side C++ registers
# via REGISTER_OBSERVATION(...). The mjlab "command" term becomes
# "keyboard_height_command" so it binds to the slider-style reader in
# deploy/robots/g1_29dof/src/State_RLBase.cpp.
_OBS_NAME_MAP = {
  "base_ang_vel": "base_ang_vel",
  "projected_gravity": "projected_gravity",
  "command": "keyboard_height_command",
  "joint_pos": "joint_pos_rel",
  "joint_vel": "joint_vel_rel",
  "actions": "last_action",
}


def _obs_term_dim(name: str, num_joints: int) -> int:
  scalar_terms = {"base_ang_vel": 3, "projected_gravity": 3, "command": 1}
  if name in scalar_terms:
    return scalar_terms[name]
  return num_joints  # joint_pos, joint_vel, actions


def _resolve_per_joint(value, num_joints: int, joint_names, fallback: float) -> list[float]:
  """Resolve a (float | list | dict-of-regex) to a per-joint list of length num_joints."""
  if value is None:
    return [float(fallback)] * num_joints
  if isinstance(value, (int, float)):
    return [float(value)] * num_joints
  if isinstance(value, (list, tuple)):
    out = list(value)
    if len(out) != num_joints:
      raise ValueError(f"Expected {num_joints} values, got {len(out)}")
    return [float(x) for x in out]
  if isinstance(value, dict):
    from mjlab.utils.lab_api.string import resolve_matching_names_values

    out = [float(fallback)] * num_joints
    ids, _, vals = resolve_matching_names_values(value, list(joint_names))
    for jid, v in zip(ids, vals):
      out[jid] = float(v)
    return out
  raise TypeError(f"Unsupported per-joint spec type: {type(value)}")


def _build_deploy_cfg(env) -> dict:
  """Build a deploy.yaml-ready dict from the live training env.

  Matches the schema used by
  ``deploy/robots/g1_29dof/config/policy/velocity/v2.6/params/deploy.yaml``.

  Assumptions (pick these up as constraints if you ever port to a new robot):
  - joint_ids_map is identity: mjlab's MJCF joint order matches the deploy-stack
    robot joint order. If sim2sim shows joints firing on wrong motors, this is
    where to fix it.
  - Observation scales are 1.0 everywhere, consistent with mjlab's training
    default (the env.yaml logs `scale: null` per term).
  - default_joint_pos doubles as the action offset (matches
    ``JointPositionActionCfg(use_default_offset=True)``).
  """
  scene_robot = env.scene["robot"]
  joint_names = list(scene_robot.joint_names)
  num_joints = len(joint_names)

  # Default joint positions (one env row → list of floats).
  default_joint_pos = scene_robot.data.default_joint_pos[0].detach().cpu().numpy().tolist()

  # Per-joint stiffness / damping, collected from the actuator configs.
  stiffness = [0.0] * num_joints
  damping = [0.0] * num_joints
  articulation = scene_robot.cfg.articulation
  for act in getattr(articulation, "actuators", ()):
    target_ids, _ = scene_robot.find_joints(list(act.target_names_expr))
    for jid in target_ids:
      stiffness[jid] = float(act.stiffness)
      damping[jid] = float(act.damping)

  # Command ranges. The crouching task has a single "height" command term.
  cmd_term_cfg = env.cfg.commands["height"]
  lo, hi = cmd_term_cfg.ranges.height
  commands = {"height": {"ranges": {"height": [float(lo), float(hi)]}}}

  # Actions. JointPositionAction; scale may be a dict of regex→float, a scalar,
  # or a list.
  action_cfg = env.cfg.actions["joint_pos"]
  action_scale = _resolve_per_joint(
    getattr(action_cfg, "scale", None), num_joints, joint_names, fallback=0.25
  )
  actions = {
    "JointPositionAction": {
      "joint_names": [".*"],
      "scale": action_scale,
      "offset": default_joint_pos,
      "clip": [[-1.0, 1.0]] * num_joints,
      "joint_ids": None,
    }
  }

  # Observations. Preserve the actor-group term order; translate key names so
  # they match what the deploy stack registers at compile time.
  actor_group = env.cfg.observations["actor"]
  history_length = int(actor_group.history_length)
  observations: dict = {}
  for name, term_cfg in actor_group.terms.items():
    deploy_name = _OBS_NAME_MAP.get(name, name)
    dim = _obs_term_dim(name, num_joints)
    observations[deploy_name] = {
      "params": dict(term_cfg.params or {}),
      "clip": None,
      "scale": [1.0] * dim,
      "history_length": history_length,
    }

  return {
    "joint_ids_map": list(range(num_joints)),  # identity; fix here if sim2sim misfires
    "step_dt": float(env.step_dt),
    "stiffness": stiffness,
    "damping": damping,
    "default_joint_pos": default_joint_pos,
    "commands": commands,
    "actions": actions,
    "observations": observations,
  }


class CrouchingOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def save(self, path: str, infos=None):
    super().save(path, infos)
    policy_path = path.split("model")[0]
    filename = "policy.onnx"
    self.export_policy_to_onnx(policy_path, filename)
    run_name: str = (
      wandb.run.name if self.logger.logger_type == "wandb" and wandb.run else "local"
    )  # type: ignore[assignment]
    onnx_path = os.path.join(policy_path, filename)
    metadata = get_base_metadata(self.env.unwrapped, run_name)
    attach_metadata_to_onnx(onnx_path, metadata)

    # Emit deploy.yaml next to the onnx, and upload it to wandb so
    # unitree_rl_lab/scripts/deploy_policy.py can pull both together.
    deploy_cfg = _build_deploy_cfg(self.env.unwrapped)
    deploy_yaml_path = os.path.join(policy_path, "deploy.yaml")
    with open(deploy_yaml_path, "w") as f:
      yaml.safe_dump(deploy_cfg, f, default_flow_style=None, sort_keys=False)

    if self.logger.logger_type in ["wandb"]:
      wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))
      wandb.save(deploy_yaml_path, base_path=os.path.dirname(policy_path))
