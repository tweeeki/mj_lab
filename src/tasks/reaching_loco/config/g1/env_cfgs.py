"""Unitree G1 walk + reach + upright (loco-manipulation) env config."""

from src.assets.robots import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)
from src.assets.robots.unitree_g1.g1_constants import HOME_KEYFRAME
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from src.tasks.reaching_loco.reaching_loco_env_cfg import make_reaching_loco_env_cfg

import src.tasks.reaching_v2.mdp as rmdp

_LEFT_HAND_SITE = "left_palm"
_RIGHT_HAND_SITE = "right_palm"
_FOOT_SITES = ("left_foot", "right_foot")
_TORSO = ("torso_link",)


def unitree_g1_reach_loco_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Unitree G1: walk (marching) + reach spheres + upright, on flat ground."""
  cfg = make_reaching_loco_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64

  robot_cfg = get_g1_robot_cfg()
  robot_cfg.init_state = HOME_KEYFRAME
  cfg.scene.entities = {"robot": robot_cfg}
  cfg.viewer.body_name = "torso_link"

  # Action scale: reach's per-joint scale with the wrists LOCKED (0) so the
  # deploy C++ wrist-override owns them (same as reaching_v2).
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  wrist_locked_scale = dict(G1_ACTION_SCALE)
  for joint_expr in list(wrist_locked_scale):
    if "wrist" in joint_expr:
      wrist_locked_scale[joint_expr] = 0.0
  joint_pos_action.scale = wrist_locked_scale

  # Hand sites for the reach command + obs + rewards.
  reach_cmd = cfg.commands["reach"]
  reach_cmd.left_hand_site = _LEFT_HAND_SITE
  reach_cmd.right_hand_site = _RIGHT_HAND_SITE
  for obs_name in ("hand_pos_b", "hand_target_error_b"):
    for group in ("actor", "critic"):
      cfg.observations[group].terms[obs_name].params["left_hand_site"] = _LEFT_HAND_SITE
      cfg.observations[group].terms[obs_name].params["right_hand_site"] = _RIGHT_HAND_SITE
  for rew_name in ("reach_distance_l1", "reach_distance_l2", "reach_success_bonus"):
    cfg.rewards[rew_name].params["left_hand_site"] = _LEFT_HAND_SITE
    cfg.rewards[rew_name].params["right_hand_site"] = _RIGHT_HAND_SITE

  # Torso link wiring: push, CoM DR, orientation/ang-vel penalties.
  cfg.events["push_torso"].params["asset_cfg"].body_names = _TORSO
  cfg.events["base_com"].params["asset_cfg"].body_names = _TORSO
  cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = _TORSO
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = _TORSO

  # Foot geoms for friction DR + foot sites for clearance/slip/height.
  geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
  )
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = _FOOT_SITES
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = _FOOT_SITES
  cfg.observations["critic"].terms["foot_height"].params["asset_cfg"].site_names = (
    _FOOT_SITES
  )

  # Speed-adaptive posture stds (NON-ARM joints only — arms free to reach).
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.5,
    r".*ankle_pitch.*": 0.15,
    r".*ankle_roll.*": 0.1,
    r".*waist_yaw.*": 0.15,
    r".*waist_roll.*": 0.1,
    r".*waist_pitch.*": 0.1,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.25,
    r".*hip_yaw.*": 0.25,
    r".*knee.*": 0.5,
    r".*ankle_pitch.*": 0.25,
    r".*ankle_roll.*": 0.1,
    r".*waist_yaw.*": 0.25,
    r".*waist_roll.*": 0.1,
    r".*waist_pitch.*": 0.1,
  }

  # Contact sensors: feet-on-ground (gait/clearance/slip) + self-collision.
  feet_ground_cfg = ContactSensorCfg(
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
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (feet_ground_cfg, self_collision_cfg)
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=rmdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_torso", None)
    cfg.curriculum = {}

  return cfg
