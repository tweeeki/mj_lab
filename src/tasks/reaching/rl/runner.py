import os

import wandb
import yaml

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner


# Maps mjlab-side observation-term keys (reaching_env_cfg.py) to the names the
# deploy-side C++ registers via REGISTER_OBSERVATION(...). The training "command"
# term becomes "keyboard_reach_command" so it binds to the keyboard-driven
# sphere-y slider reader that State_Reach.cpp will register, mirroring how
# crouching renames "command" to "keyboard_height_command".
_OBS_NAME_MAP = {
  "base_ang_vel": "base_ang_vel",
  "projected_gravity": "projected_gravity",
  "command": "keyboard_reach_command",
  "hand_pos_b": "hand_pos_b",
  "hand_target_error_b": "hand_target_error_b",
  "joint_pos": "joint_pos_rel",
  "joint_vel": "joint_vel_rel",
  "actions": "last_action",
}


def _obs_term_dim(name: str, num_joints: int) -> int:
  # The three reach-specific terms are all 6-D: two 3-D hand/target vectors
  # stacked (left | right), in the robot root frame.
  scalar_terms = {
    "base_ang_vel": 3,
    "projected_gravity": 3,
    "command": 6,
    "hand_pos_b": 6,
    "hand_target_error_b": 6,
  }
  if name in scalar_terms:
    return scalar_terms[name]
  return num_joints  # joint_pos, joint_vel, actions


def _yaml_safe_params(params: dict) -> dict:
  # Drop entries whose values aren't YAML-primitive (e.g. SceneEntityCfg). The
  # deploy side only reads string/number/list params like `command_name` and
  # hand-site names; asset_cfg is training-side plumbing that safe_dump rejects.
  primitive = (str, int, float, bool, type(None))

  def ok(v):
    if isinstance(v, primitive):
      return True
    if isinstance(v, (list, tuple)):
      return all(ok(x) for x in v)
    if isinstance(v, dict):
      return all(isinstance(k, str) and ok(val) for k, val in v.items())
    return False

  return {k: v for k, v in params.items() if ok(v)}


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

  Mirrors crouching/rl/runner.py._build_deploy_cfg, adapted for the bimanual
  sphere-reach command (six per-axis ranges instead of a single height).
  """
  scene_robot = env.scene["robot"]
  joint_names = list(scene_robot.joint_names)
  num_joints = len(joint_names)

  default_joint_pos = scene_robot.data.default_joint_pos[0].detach().cpu().numpy().tolist()

  stiffness = [0.0] * num_joints
  damping = [0.0] * num_joints
  articulation = scene_robot.cfg.articulation
  for act in getattr(articulation, "actuators", ()):
    target_ids, _ = scene_robot.find_joints(list(act.target_names_expr))
    for jid in target_ids:
      stiffness[jid] = float(act.stiffness)
      damping[jid] = float(act.damping)

  # Command ranges. The reach task has a UniformBimanualSphereCommand with
  # six per-axis ranges; dump them all so the deploy side can pick fixed
  # x/z values and clamp the interactive y slider.
  reach_cmd_cfg = env.cfg.commands["reach"]
  r = reach_cmd_cfg.ranges
  commands = {
    "reach": {
      "ranges": {
        "left_x": [float(r.left_x[0]), float(r.left_x[1])],
        "left_y": [float(r.left_y[0]), float(r.left_y[1])],
        "left_z": [float(r.left_z[0]), float(r.left_z[1])],
        "right_x": [float(r.right_x[0]), float(r.right_x[1])],
        "right_y": [float(r.right_y[0]), float(r.right_y[1])],
        "right_z": [float(r.right_z[0]), float(r.right_z[1])],
      }
    }
  }

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

  actor_group = env.cfg.observations["actor"]
  history_length = int(actor_group.history_length)
  observations: dict = {}
  for name, term_cfg in actor_group.terms.items():
    deploy_name = _OBS_NAME_MAP.get(name, name)
    dim = _obs_term_dim(name, num_joints)
    observations[deploy_name] = {
      "params": _yaml_safe_params(dict(term_cfg.params or {})),
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


class ReachingOnPolicyRunner(MjlabOnPolicyRunner):
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

    deploy_cfg = _build_deploy_cfg(self.env.unwrapped)
    deploy_yaml_path = os.path.join(policy_path, "deploy.yaml")
    with open(deploy_yaml_path, "w") as f:
      yaml.safe_dump(deploy_cfg, f, default_flow_style=None, sort_keys=False)

    if self.logger.logger_type in ["wandb"]:
      wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))
      wandb.save(deploy_yaml_path, base_path=os.path.dirname(policy_path))
