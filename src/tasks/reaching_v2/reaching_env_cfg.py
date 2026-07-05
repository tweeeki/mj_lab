"""Bimanual sphere-reaching task configuration (v2 — balance-anchored).

Builds a base ``ManagerBasedRlEnvCfg`` where the robot must keep its default
pose while both arms track a commanded pair of 3D target spheres, one per hand.

v2 differences from ``tasks/reaching``: the lower body is anchored by
non-saturating penalties (base height, vertical root velocity, stance joint
deviation, feet contact + feet motion) so the policy cannot drift into the
crouch / wide-stance / foot-shuffle stability hacks the exp-shaped ``posture``
reward allowed. The torso push is kept but at 1/5 strength (±16 N — absorbable
without stepping). The arm/reach rewards (incl. the 7-D speed command +
``waypoint_track``) are identical to the proven v2-dev recipe. Training-only
changes: the observation vector is unchanged.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

import src.tasks.reaching_v2.mdp as mdp


def make_reaching_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create the base bimanual reaching task configuration."""

  ##
  # Observations
  ##

  actor_terms = {
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "command": ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": "reach"},
    ),
    "hand_pos_b": ObservationTermCfg(
      func=mdp.hand_pos_b,
      params={
        "left_hand_site": "",  # Set per-robot.
        "right_hand_site": "",  # Set per-robot.
        "asset_cfg": SceneEntityCfg("robot"),
      },
    ),
    "hand_target_error_b": ObservationTermCfg(
      func=mdp.hand_target_error_b,
      params={
        "command_name": "reach",
        "left_hand_site": "",  # Set per-robot.
        "right_hand_site": "",  # Set per-robot.
        "asset_cfg": SceneEntityCfg("robot"),
      },
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }

  critic_terms = {
    **actor_terms,
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
      noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
      history_length=1,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
      history_length=1,
    ),
  }

  ##
  # Actions
  ##

  actions: dict[str, ActionTermCfg] = {
    # v2: EMA-filtered joint-position action — trains the policy WITH the
    # deploy controller's whole-body action low-pass in the loop, so it stays
    # stable under that lag on sim2sim/sim2real (see mdp/actions.py). Alpha is
    # randomized per episode across the deployable band.
    "joint_pos": mdp.EmaJointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.25,  # Override per-robot.
      use_default_offset=True,
      ema_alpha_1khz_range=(0.06, 0.5),
    )
  }

  ##
  # Commands
  ##

  commands: dict[str, CommandTermCfg] = {
    "reach": mdp.UniformBimanualSphereCommandCfg(
      entity_name="robot",
      left_hand_site="",  # Set per-robot.
      right_hand_site="",  # Set per-robot.
      resampling_time_range=(3.0, 6.0),
      debug_vis=True,
      # One shared commanded arm speed (m/s) for both hands, randomized per
      # resample. Override per-robot in config/g1/env_cfgs.py if needed.
      speed_range=(0.05, 0.50),
      # Default ranges in pelvis frame: roughly in front of the robot,
      # left hand on +y side, right hand on -y side. Override per-robot.
      ranges=mdp.UniformBimanualSphereCommandCfg.Ranges(
        left_x=(0.20, 0.45),
        left_y=(0.05, 0.35),
        left_z=(-0.10, 0.25),
        right_x=(0.20, 0.45),
        right_y=(-0.35, -0.05),
        right_z=(-0.10, 0.25),
      ),
    ),
  }

  ##
  # Events
  ##

  events = {
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (-0.05, 0.05),
          "y": (-0.05, 0.05),
          "z": (0.0, 0.0),
          "yaw": (-0.3, 0.3),
        },
        "velocity_range": {},
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.05, 0.05),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per-robot.
        "operation": "abs",
        "ranges": (0.6, 1.4),
        "shared_random": True,
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "bias_range": (-0.015, 0.015),
      },
    ),
    # Mild sustained horizontal torso push for sim2real robustness. v2 runs
    # this at 1/5 of the original ±80 N: at ±16 N the robot can absorb the
    # push with ankle/hip torque alone — no step needed — so it does NOT fight
    # the planted-feet rewards below, it teaches the policy to lean against
    # small disturbances (model mismatch, cable tugs) instead of shuffling.
    # Delete this entry for a fully disturbance-free run.
    "push_torso": EventTermCfg(
      mode="step",
      func=mdp.push_torso_force,
      params={
        "force_range_x": (-16.0, 16.0),   # forward / backward (N)
        "force_range_y": (-16.0, 16.0),   # left / right (N)
        "force_range_z": (0.0, 0.0),      # horizontal only
        "torque_range": (0.0, 0.0),
        "duration_s": (0.3, 0.8),
        "cooldown_s": (2.0, 4.0),
        "body_point_offset": (0.0, 0.0, 0.2),
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot (torso).
      },
    ),
  }

  ##
  # Rewards
  ##

  rewards = {
    "alive": RewardTermCfg(func=envs_mdp.is_alive, weight=1.0),
    "reach_distance": RewardTermCfg(
      func=mdp.reach_distance,
      weight=4.0,
      params={
        "command_name": "reach",
        "left_hand_site": "",  # Set per-robot.
        "right_hand_site": "",  # Set per-robot.
        "std": 0.25,
        "asset_cfg": SceneEntityCfg("robot"),
      },
    ),
    "reach_distance_l2": RewardTermCfg(
      func=mdp.reach_distance_l2,
      weight=-1.0,
      params={
        "command_name": "reach",
        "left_hand_site": "",  # Set per-robot.
        "right_hand_site": "",  # Set per-robot.
        "asset_cfg": SceneEntityCfg("robot"),
      },
    ),
    "reach_success_bonus": RewardTermCfg(
      func=mdp.reach_success_bonus,
      weight=3.0,
      params={
        "command_name": "reach",
        "left_hand_site": "",  # Set per-robot.
        "right_hand_site": "",  # Set per-robot.
        "threshold": 0.05,
        "asset_cfg": SceneEntityCfg("robot"),
      },
    ),
    # Tight tracking of the MOVING waypoint → enforces the commanded arm speed.
    # See mdp/rewards.py::waypoint_track. std is small (glue the palm to the
    # waypoint); reach_distance above stays loose for broad guidance.
    "waypoint_track": RewardTermCfg(
      func=mdp.waypoint_track,
      weight=4.0,
      params={
        "command_name": "reach",
        "left_hand_site": "",  # Set per-robot.
        "right_hand_site": "",  # Set per-robot.
        "std": 0.05,
        "asset_cfg": SceneEntityCfg("robot"),
      },
    ),
    "posture": RewardTermCfg(
      func=envs_mdp.posture,
      weight=2.0,
      params={
        # Hold everything EXCEPT the arms near the default pose. The arms are
        # free to move to reach the targets; this keeps the legs and torso
        # from drifting — effectively locking the hips "through the reward"
        # rather than welding the pelvis.
        "asset_cfg": SceneEntityCfg(
          "robot",
          joint_names=(r"^(?!.*(shoulder|elbow|wrist)).*",),
        ),
        "std": {},  # Set per-robot.
      },
    ),
    "torso_upright": RewardTermCfg(
      func=envs_mdp.flat_orientation_l2,
      weight=-2.0,
    ),
    "is_terminated": RewardTermCfg(func=envs_mdp.is_terminated, weight=-5.0),
    "joint_acc_l2": RewardTermCfg(func=envs_mdp.joint_acc_l2, weight=-2.5e-7),
    "joint_pos_limits": RewardTermCfg(func=envs_mdp.joint_pos_limits, weight=-10.0),
    "action_rate_l2": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.02),
    # Extra penalty on ABRUPT leg motion (hip/knee/ankle), on top of the global
    # joint_acc_l2 above — discourages the feet snapping/shuffling, especially
    # kicking a foot into the air. joint_acc (jerk-like) is used on purpose
    # rather than joint_vel: it punishes *abruptness* while still allowing the
    # smooth leg motion needed to RECOVER from the push_torso disturbance above.
    # Tension knob: too heavy and the robot can't rebalance; too light and the
    # feet stay twitchy. Start ~4x the global term, tune against push strength.
    "leg_joint_acc_l2": RewardTermCfg(
      func=envs_mdp.joint_acc_l2,
      weight=-1.0e-6,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot", joint_names=(r".*(hip|knee|ankle).*",)
        ),
      },
    ),
    # --- v2 balance anchors -------------------------------------------------
    # Non-saturating penalties that hold the lower body at the default standing
    # pose. The exp-shaped `posture` reward above loses its gradient once a
    # joint drifts past ~2 std; these keep pulling back at any deviation, which
    # is what prevents the slow sink into a crouch / wide stance over training.
    # Hard anchor on absolute pelvis height. The sphere targets are pelvis-
    # frame, so crouching is free for the reach rewards — this is the only
    # term that makes sinking expensive. target_height is overridden per-robot
    # with the measured settled standing height.
    "base_height": RewardTermCfg(
      func=mdp.base_height_l2,
      weight=-30.0,
      params={
        "target_height": 0.78,  # Set per-robot (measured settled pelvis z).
        "asset_cfg": SceneEntityCfg("robot"),
      },
    ),
    # Anti-hop / anti-bob: punishes vertical pelvis velocity directly.
    "root_lin_vel_z": RewardTermCfg(func=mdp.root_lin_vel_z_l2, weight=-1.0),
    # Stance-width anchor: L1 pull on the joints that splay the legs. Scoped to
    # roll/yaw only — hip_pitch/knee/ankle_pitch height sag is base_height's job,
    # and over-constraining the sagittal joints would fight balance corrections.
    "stance_joint_deviation": RewardTermCfg(
      func=mdp.joint_deviation_l1,
      weight=-0.2,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot", joint_names=(r".*(hip_roll|hip_yaw|ankle_roll).*",)
        ),
      },
    ),
    # Feet stay planted: bonus while BOTH feet touch the ground (needs the
    # feet_ground_contact sensor, added per-robot), plus a velocity penalty on
    # the foot sites that kills the fwd/back "feeling for the ground" shuffle.
    "feet_contact": RewardTermCfg(
      func=mdp.feet_contact,
      weight=0.5,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "feet_motion": RewardTermCfg(
      func=mdp.feet_motion_l2,
      weight=-1.0,
      params={
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
    ),
  }

  ##
  # Terminations
  ##

  terminations = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    # v2: raised from 0.35 — standing height is ~0.79 and this task never
    # legitimately goes low, so ending the episode well above a crouch deletes
    # the sink-into-crouch attractor structurally instead of just penalizing it.
    "collapsed": TerminationTermCfg(
      func=envs_mdp.root_height_below_minimum,
      params={"minimum_height": 0.55},
    ),
  }

  curriculum: dict = {}
  metrics: dict = {}

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      sensors=(),
      num_envs=1,
      extent=2.0,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum=curriculum,
    metrics=metrics,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",  # Set per-robot.
      distance=2.5,
      elevation=-10.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=4,
    episode_length_s=8.0,
  )
