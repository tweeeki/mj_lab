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
  # wrist0 variants (State_Reach.cpp): identical to joint_pos_rel/joint_vel_rel
  # except the two GUI-overridable wrist-roll channels (idx 19/26) are reported
  # at default, keeping the wrist-locked policy blind to the wrist override.
  "joint_pos": "joint_pos_rel_wrist0",
  "joint_vel": "joint_vel_rel_wrist0",
  "actions": "last_action",
}


def _obs_term_dim(name: str, num_joints: int) -> int:
  # hand_pos_b / hand_target_error_b are 6-D: two 3-D vectors (left | right) in
  # the robot root frame. The command is 7-D: the two 3-D moving waypoints plus
  # ONE shared commanded arm speed scalar (see UniformBimanualSphereCommand).
  scalar_terms = {
    "base_ang_vel": 3,
    "projected_gravity": 3,
    "command": 7,
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
  # One shared commanded arm speed (m/s); dump the range so the deploy side can
  # clamp the interactive speed command (keyboard/DDS) to what training saw.
  speed_range = getattr(reach_cmd_cfg, "speed_range", (0.05, 0.50))
  commands = {
    "reach": {
      # reach-dev deploy features (State_Reach.cpp / State_RLBase.cpp). These
      # used to be hand-added after every export and were silently lost on
      # re-export; v3 emits them with the known-good v2 values.
      "dev": {
        "seed_at_current_hand": True,
        "home_on_entry": True,
        "home_left": [0.35, 0.12, 0.20],
        "home_right": [0.35, -0.12, 0.20],
        "home_slew_mps": 0.05,
        "auto_hold": False,
        "hold_enter_m": 0.01,
        "hold_debounce_s": 0.3,
        "hold_exit_m": 0.02,
        "hold_blend_s": 0.2,
        "wrist_override": True,
        "wrist_max_joint_delta": 0.004,
      },
      "ranges": {
        "left_x": [float(r.left_x[0]), float(r.left_x[1])],
        "left_y": [float(r.left_y[0]), float(r.left_y[1])],
        "left_z": [float(r.left_z[0]), float(r.left_z[1])],
        "right_x": [float(r.right_x[0]), float(r.right_x[1])],
        "right_y": [float(r.right_y[0]), float(r.right_y[1])],
        "right_z": [float(r.right_z[0]), float(r.right_z[1])],
      },
      "speed_range": [float(speed_range[0]), float(speed_range[1])],
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
      # v3: null (was [[-1,1]]*n). Training runs unclipped, and the deploy side
      # applies this clip to the FINAL target (offset + scale*action), so ±1 rad
      # silently saturated joints whose default is near 1 (elbow default 0.87).
      # The C++ joint_actions.h skips clamping when clip is null.
      "clip": None,
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
    # Deploy-side whole-body action EMA (State_RLBase.cpp). 0.1 is inside the
    # trained ema_alpha_1khz_range band, so it's a free hardware knob up to 0.5
    # (higher = less filter lag). Emitted here so re-exports stop dropping it.
    "action_smooth_alpha": 0.1,
    # Smooth kp/kd handover on state entry (State_RLBase.cpp gain ramp).
    "gain_ramp_s": 0.5,
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
