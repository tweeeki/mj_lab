"""Unitree G1 bimanual sphere-reaching environment configurations."""

from src.assets.robots import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)
from src.assets.robots.unitree_g1.g1_constants import HOME_KEYFRAME
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from src.tasks.reaching_v2.reaching_env_cfg import make_reaching_env_cfg

import src.tasks.reaching_v2.mdp as mdp


_LEFT_HAND_SITE = "left_palm"
_RIGHT_HAND_SITE = "right_palm"


def unitree_g1_reach_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the Unitree G1 bimanual sphere-reaching task (standing on ground)."""
  cfg = make_reaching_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64

  robot_cfg = get_g1_robot_cfg()
  robot_cfg.init_state = HOME_KEYFRAME
  cfg.scene.entities = {"robot": robot_cfg}
  cfg.viewer.body_name = "torso_link"

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  # v2: LOCK the forearms — zero action authority on wrist roll/pitch/yaw
  # (both arms) pins those motor targets to the default pose. The policy still
  # outputs 29 actions and obs stays 112, so the deploy contract is unchanged;
  # the C++ wrist-override can still drive wrist_roll in deploy because it
  # overwrites the motor target AFTER the policy write.
  wrist_locked_scale = dict(G1_ACTION_SCALE)
  for joint_expr in list(wrist_locked_scale):
    if "wrist" in joint_expr:
      wrist_locked_scale[joint_expr] = 0.0
  joint_pos_action.scale = wrist_locked_scale

  # Wire up hand sites for command + observations + rewards.
  reach_cmd = cfg.commands["reach"]
  reach_cmd.left_hand_site = _LEFT_HAND_SITE
  reach_cmd.right_hand_site = _RIGHT_HAND_SITE

  for obs_name in ("hand_pos_b", "hand_target_error_b"):
    cfg.observations["actor"].terms[obs_name].params["left_hand_site"] = _LEFT_HAND_SITE
    cfg.observations["actor"].terms[obs_name].params["right_hand_site"] = _RIGHT_HAND_SITE
    cfg.observations["critic"].terms[obs_name].params["left_hand_site"] = _LEFT_HAND_SITE
    cfg.observations["critic"].terms[obs_name].params["right_hand_site"] = _RIGHT_HAND_SITE

  for rew_name in (
    "reach_distance",
    "reach_distance_l2",
    "reach_success_bonus",
    "waypoint_track",
  ):
    cfg.rewards[rew_name].params["left_hand_site"] = _LEFT_HAND_SITE
    cfg.rewards[rew_name].params["right_hand_site"] = _RIGHT_HAND_SITE

  # Mild torso push (±16 N, 1/5 of the original) — bound to the torso link.
  cfg.events["push_torso"].params["asset_cfg"].body_names = ("torso_link",)

  # Foot friction (feet still carry the standing robot).
  geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
  )
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names

  # Contact sensors (same wiring as the velocity task). Scene-level only: they
  # do NOT touch the observation vector, so the deploy contract is unchanged.
  # - feet_ground_contact feeds the feet_contact reward.
  # - self_collision feeds the self_collisions penalty below: physical
  #   self-collision is already simulated (FULL_COLLISION), but without this
  #   penalty the policy is free to lean the elbows/arms against the torso.
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    ContactSensorCfg(
      name="feet_ground_contact",
      primary=ContactMatch(
        mode="subtree",
        pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
        entity="robot",
      ),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found", "force"),
      reduce="netforce",
      num_slots=1,
      track_air_time=True,
    ),
    self_collision_cfg,
  )
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )

  # Balance-anchor wiring: foot sites for the feet_motion penalty, and the
  # measured settled pelvis height for the base_height anchor (HOME_KEYFRAME
  # spawns at 0.8 and the robot settles slightly lower under gravity).
  cfg.rewards["feet_motion"].params["asset_cfg"].site_names = (
    "left_foot",
    "right_foot",
  )
  cfg.rewards["base_height"].params["target_height"] = 0.78

  # Posture stds — arms are excluded by the asset_cfg regex so they are
  # fully free to move. Everything else is held to the default keyframe.
  cfg.rewards["posture"].params["std"] = {
    r".*hip_pitch.*": 0.10,
    r".*hip_roll.*": 0.10,
    r".*hip_yaw.*": 0.10,
    r".*knee.*": 0.10,
    r".*ankle_pitch.*": 0.10,
    r".*ankle_roll.*": 0.10,
    r".*waist_yaw.*": 0.10,
    r".*waist_roll.*": 0.10,
    r".*waist_pitch.*": 0.10,
  }

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}

  return cfg
