"""Unitree G1 bimanual sphere-reaching environment configurations."""

from src.assets.robots import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)
from src.assets.robots.unitree_g1.g1_constants import HOME_KEYFRAME
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from src.tasks.reaching.reaching_env_cfg import make_reaching_env_cfg


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
  joint_pos_action.scale = G1_ACTION_SCALE

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

  # Foot friction (feet still carry the standing robot).
  geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
  )
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names

  # Push the torso link (fwd/back/left/right sustained force for balance robustness).
  cfg.events["push_torso"].params["asset_cfg"].body_names = ("torso_link",)

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
