"""Unitree G1 crouching environment configurations."""

from src.assets.robots import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from src.tasks.crouching.crouching_env_cfg import make_crouching_env_cfg


def unitree_g1_crouch_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 crouching (height tracking) configuration on flat ground."""
  cfg = make_crouching_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64

  cfg.scene.entities = {"robot": get_g1_robot_cfg()}
  cfg.viewer.body_name = "torso_link"

  # Action scaling per joint (from G1 constants).
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_ACTION_SCALE

  # Commanded height range.
  # G1 standing height ~0.80 m (pelvis), knees-bent keyframe sits at ~0.78 m
  # with visible bend. We constrain the range to keep knees safe:
  #   - upper bound 0.75 m: slightly below full standing (still clearly a stance)
  #   - lower bound 0.45 m: deep crouch without over-flexing knees (< 165 deg)
  height_cmd = cfg.commands["height"]
  height_cmd.ranges.height = (0.45, 0.75)

  # Foot geoms for friction randomization.
  geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
  )
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  # Posture regularization: keep upper body near default, allow leg joints to move
  # (they must change to follow commanded height), keep ankles/waist tight for balance.
  cfg.rewards["posture"].params["std"] = {
    # Lower body — loose on pitch joints that drive the squat.
    r".*hip_pitch.*": 0.6,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.8,
    r".*ankle_pitch.*": 0.4,
    r".*ankle_roll.*": 0.1,
    # Waist — tight, keep torso upright and stable.
    r".*waist_yaw.*": 0.15,
    r".*waist_roll.*": 0.1,
    r".*waist_pitch.*": 0.1,
    # Arms — moderate, let them counterbalance a bit.
    r".*shoulder_pitch.*": 0.2,
    r".*shoulder_roll.*": 0.15,
    r".*shoulder_yaw.*": 0.15,
    r".*elbow.*": 0.15,
    r".*wrist.*": 0.3,
  }

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}

  return cfg
