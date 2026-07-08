"""Base env cfg for the loco-manipulation reaching task (walk + reach + upright).

Merges the velocity (locomotion) task's leg/base rewards with reaching_v2's arm
rewards. mdp funcs are imported from BOTH source tasks (no duplication):
  vmdp = src.tasks.velocity.mdp     (gait/velocity/foot rewards, twist command)
  rmdp = src.tasks.reaching_v2.mdp  (reach rewards, sphere command, EMA action)
Both re-export mjlab.envs.mdp, so framework funcs resolve via either alias.

Design notes:
- Legs follow the velocity paradigm (twist command, foot_gait, clearance, slip,
  stand_still) — the reach standing/planted-feet rewards are intentionally NOT
  used so the feet are free to step/march.
- `pose` and `stand_still` are scoped to NON-ARM joints only; otherwise they
  would hold the arms at default and fight reaching.
- Flat ground (no terrain scan / height scan), gentle ±16 N torso push.
"""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
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

import src.tasks.velocity.mdp as vmdp
import src.tasks.reaching_v2.mdp as rmdp

# Non-arm joints (legs + waist + torso). Used to scope `pose`/`stand_still` so
# they don't constrain the arms (which must be free to reach).
_NON_ARM = (r"^(?!.*(shoulder|elbow|wrist)).*",)


def make_reaching_loco_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create the base walk + reach + upright task configuration (flat ground)."""

  ##
  # Observations
  ##

  actor_terms = {
    "base_ang_vel": ObservationTermCfg(
      func=vmdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "projected_gravity": ObservationTermCfg(
      func=vmdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    # Locomotion command (walk) + gait phase clock.
    "twist_command": ObservationTermCfg(
      func=vmdp.generated_commands, params={"command_name": "twist"}
    ),
    "phase": ObservationTermCfg(
      func=vmdp.phase, params={"period": 0.6, "command_name": "twist"}
    ),
    # Reach command (spheres) + hand FK.
    "reach_command": ObservationTermCfg(
      func=rmdp.generated_commands, params={"command_name": "reach"}
    ),
    "hand_pos_b": ObservationTermCfg(
      func=rmdp.hand_pos_b,
      params={
        "left_hand_site": "",  # Set per-robot.
        "right_hand_site": "",  # Set per-robot.
        "asset_cfg": SceneEntityCfg("robot"),
      },
    ),
    "hand_target_error_b": ObservationTermCfg(
      func=rmdp.hand_target_error_b,
      params={
        "command_name": "reach",
        "left_hand_site": "",  # Set per-robot.
        "right_hand_site": "",  # Set per-robot.
        "asset_cfg": SceneEntityCfg("robot"),
      },
    ),
    "joint_pos": ObservationTermCfg(
      func=vmdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)
    ),
    "joint_vel": ObservationTermCfg(
      func=vmdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5)
    ),
    "actions": ObservationTermCfg(func=vmdp.last_action),
  }

  critic_terms = {
    **actor_terms,
    "base_lin_vel": ObservationTermCfg(
      func=vmdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
      noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
    # Privileged foot state (critic only — same as velocity task).
    "foot_height": ObservationTermCfg(
      func=vmdp.foot_height,
      params={"asset_cfg": SceneEntityCfg("robot", site_names=())},  # Set per-robot.
    ),
    "foot_air_time": ObservationTermCfg(
      func=vmdp.foot_air_time, params={"sensor_name": "feet_ground_contact"}
    ),
    "foot_contact": ObservationTermCfg(
      func=vmdp.foot_contact, params={"sensor_name": "feet_ground_contact"}
    ),
    "foot_contact_forces": ObservationTermCfg(
      func=vmdp.foot_contact_forces, params={"sensor_name": "feet_ground_contact"}
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
  # Actions — EMA on the ARMS (deploy anti-oscillation), REACTIVE legs.
  ##
  # 2026-07-08: the arms keep the deploy-matched EMA that killed the sim2real
  # oscillation, but the LEGS (+ waist) are made reactive (alpha=1.0, no EMA).
  # The ~15-20 ms filter lag was stopping the legs from catching balance and
  # placing a crisp step (reactivity-vs-smoothness trade-off). NOTE: this means
  # the reaching_loco deploy controller must EMA only the arm joints, not all 29
  # — a deliberate change from reaching_v2's whole-body EMA (deploy is unbuilt).
  actions: dict[str, ActionTermCfg] = {
    "joint_pos": rmdp.EmaJointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.25,  # Override per-robot (wrists -> 0).
      use_default_offset=True,
      ema_alpha_1khz_range=(0.06, 0.5),
      reactive_joint_names=_NON_ARM,  # legs + waist: no EMA lag (see above).
    )
  }

  ##
  # Commands — twist (walk) + reach (spheres).
  ##

  commands: dict[str, CommandTermCfg] = {
    "twist": vmdp.UniformVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(3.0, 8.0),
      rel_standing_envs=0.0,  # 2026-07-08: was 0.2. Standing was the lazy escape
      # (no-command envs let the policy sit still and still collect reach reward).
      # Force every env to carry a walk command so it MUST move to earn reward.
      heading_command=True,
      heading_control_stiffness=0.5,
      debug_vis=True,
      ranges=vmdp.UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-0.5, 1.0),
        lin_vel_y=(-0.5, 0.5),
        ang_vel_z=(-1.0, 1.0),
        heading=(-math.pi, math.pi),
      ),
    ),
    "reach": rmdp.UniformBimanualSphereCommandCfg(
      entity_name="robot",
      left_hand_site="",  # Set per-robot.
      right_hand_site="",  # Set per-robot.
      resampling_time_range=(3.0, 6.0),
      debug_vis=True,
      speed_range=(0.05, 0.50),
      ranges=rmdp.UniformBimanualSphereCommandCfg.Ranges(
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
      func=vmdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (0.0, 0.0),
          "yaw": (-3.14, 3.14),
        },
        "velocity_range": {},
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=vmdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.0, 0.0),
        "velocity_range": (-0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    # Gentle ±16 N horizontal torso push (kept from reaching_v2, per user).
    # push_torso_force is a STEP-mode stateful event (it runs its own
    # cooldown -> trigger -> sustain cycle), so mode="step" + duration_s/
    # cooldown_s — NOT an interval event.
    "push_torso": EventTermCfg(
      mode="step",
      func=rmdp.push_torso_force,
      params={
        "force_range_x": (-16.0, 16.0),
        "force_range_y": (-16.0, 16.0),
        "force_range_z": (0.0, 0.0),
        "torque_range": (0.0, 0.0),
        "duration_s": (0.3, 0.8),
        "cooldown_s": (2.0, 4.0),
        "body_point_offset": (0.0, 0.0, 0.2),
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot (torso).
      },
    ),
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per-robot.
        "operation": "abs",
        "ranges": (0.3, 1.6),
        "shared_random": True,
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={"asset_cfg": SceneEntityCfg("robot"), "bias_range": (-0.015, 0.015)},
    ),
    # Base CoM randomization (sim2real balance robustness).
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot (torso).
        "operation": "add",
        "ranges": {0: (-0.05, 0.05), 1: (-0.05, 0.05), 2: (-0.05, 0.05)},
      },
    ),
  }

  ##
  # Rewards — LOCO (legs/base) + REACH (arms) + shared.
  ##

  rewards = {
    # -- Locomotion (from the velocity task) --------------------------------
    # 2026-07-08: track weights 1.0 -> 3.0. The policy was parking in a "stand
    # still + reach" optimum (walking barely paid vs the -200 fall penalty);
    # tripling the velocity-tracking reward makes following the twist command
    # clearly worth it, forcing it to actually move in the commanded direction.
    "track_linear_velocity": RewardTermCfg(
      func=vmdp.track_linear_velocity,
      weight=3.0,
      params={"command_name": "twist", "std": math.sqrt(0.25)},
    ),
    "track_angular_velocity": RewardTermCfg(
      func=vmdp.track_angular_velocity,
      weight=3.0,
      params={"command_name": "twist", "std": math.sqrt(0.5)},
    ),
    "body_orientation_l2": RewardTermCfg(
      func=vmdp.body_orientation_l2,
      weight=-1.0,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=())},  # Set per-robot.
    ),
    # Speed-adaptive posture — SCOPED OFF THE ARMS (arms must be free to reach).
    "pose": RewardTermCfg(
      func=vmdp.variable_posture,
      weight=1.0,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=_NON_ARM),
        "command_name": "twist",
        "std_standing": {},  # Set per-robot.
        "std_walking": {},  # Set per-robot.
        "std_running": {},  # Set per-robot.
        "walking_threshold": 0.1,
        "running_threshold": 1.5,
      },
    ),
    "body_ang_vel": RewardTermCfg(
      func=vmdp.body_angular_velocity_penalty,
      weight=-0.05,  # Override per-robot.
      params={"asset_cfg": SceneEntityCfg("robot", body_names=())},  # Set per-robot.
    ),
    "angular_momentum": RewardTermCfg(
      func=vmdp.angular_momentum_penalty,
      weight=-0.025,  # Override per-robot.
      params={"sensor_name": "robot/root_angmom"},
    ),
    # 2026-07-08: 0.5 -> 1.5. Directly rewards matching the alternating gait
    # clock (one foot in swing) when commanded to walk; standing (both feet
    # planted) only half-matches the clock, so a heavier weight actively pushes
    # the robot to lift a foot instead of shuffling in place.
    "foot_gait": RewardTermCfg(
      func=vmdp.feet_gait,
      weight=1.5,
      params={
        "period": 0.6,
        "offset": [0.0, 0.5],
        "threshold": 0.56,
        "command_threshold": 0.1,
        "command_name": "twist",
        "sensor_name": "feet_ground_contact",
      },
    ),
    "foot_clearance": RewardTermCfg(
      func=vmdp.feet_clearance,
      weight=-1.0,
      params={
        "target_height": 0.10,
        "command_name": "twist",
        "command_threshold": 0.1,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
    ),
    "foot_slip": RewardTermCfg(
      func=vmdp.feet_slip,
      weight=-0.25,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.1,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
    ),
    "soft_landing": RewardTermCfg(
      func=vmdp.soft_landing,
      weight=-1e-3,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.1,
      },
    ),
    # Stand still when NOT commanded to walk — SCOPED OFF THE ARMS.
    "stand_still": RewardTermCfg(
      func=vmdp.stand_still,
      weight=-1.0,
      params={
        "command_name": "twist",
        "command_threshold": 0.1,
        "asset_cfg": SceneEntityCfg("robot", joint_names=_NON_ARM),
      },
    ),
    # -- Reach (arms, from reaching_v2) -------------------------------------
    "reach_distance_l1": RewardTermCfg(
      func=rmdp.reach_distance_l1,
      weight=-5.0,
      params={
        "command_name": "reach",
        "left_hand_site": "",  # Set per-robot.
        "right_hand_site": "",  # Set per-robot.
        "tolerance": 0.03,
        "asset_cfg": SceneEntityCfg("robot"),
      },
    ),
    "reach_distance_l2": RewardTermCfg(
      func=rmdp.reach_distance_l2,
      weight=-1.0,
      params={
        "command_name": "reach",
        "left_hand_site": "",  # Set per-robot.
        "right_hand_site": "",  # Set per-robot.
        "asset_cfg": SceneEntityCfg("robot"),
      },
    ),
    "reach_success_bonus": RewardTermCfg(
      func=rmdp.reach_success_bonus,
      weight=3.0,
      params={
        "command_name": "reach",
        "left_hand_site": "",  # Set per-robot.
        "right_hand_site": "",  # Set per-robot.
        "threshold": 0.05,
        "asset_cfg": SceneEntityCfg("robot"),
      },
    ),
    "arm_joint_vel": RewardTermCfg(
      func=envs_mdp.joint_vel_l2,
      weight=-2.5e-4,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot", joint_names=(r".*(shoulder|elbow|wrist).*",)
        ),
      },
    ),
    # -- Shared regularization ----------------------------------------------
    "is_terminated": RewardTermCfg(func=vmdp.is_terminated, weight=-200.0),
    "joint_acc_l2": RewardTermCfg(func=vmdp.joint_acc_l2, weight=-2.5e-7),
    "joint_pos_limits": RewardTermCfg(func=vmdp.joint_pos_limits, weight=-10.0),
    "action_rate_l2": RewardTermCfg(func=vmdp.action_rate_l2, weight=-0.05),
    # self_collisions added per-robot (needs the self_collision sensor).
  }

  ##
  # Terminations
  ##

  terminations = {
    "time_out": TerminationTermCfg(func=vmdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=vmdp.bad_orientation, params={"limit_angle": math.radians(70.0)}
    ),
  }

  ##
  # Curriculum — velocity command (learnability, per 2026-07-08 research).
  ##
  # The robot was toppling instead of stepping because it was asked for the full
  # velocity range from step 0. Every humanoid-loco framework ramps this: start
  # with a GENTLE range (learn a slow walk / step-in-place first), then open up
  # to normal walking speed once flat-ground gait is solid. The final stage caps
  # forward speed at 1.0 m/s (~normal human walk) — no sprinting.
  curriculum: dict[str, CurriculumTermCfg] = {
    "command_vel": CurriculumTermCfg(
      func=vmdp.commands_vel,
      params={
        "command_name": "twist",
        "velocity_stages": [
          # Stage 0 (from step 0): gentle — slow walk, mild turning.
          {
            "step": 0,
            "lin_vel_x": (-0.3, 0.5),
            "lin_vel_y": (-0.2, 0.2),
            "ang_vel_z": (-0.5, 0.5),
          },
          # Stage 1 (~iter 1500): open up to capped normal walking speed.
          {
            "step": 1500 * 24,
            "lin_vel_x": (-0.5, 1.0),
            "lin_vel_y": (-0.5, 0.5),
            "ang_vel_z": (-1.0, 1.0),
          },
        ],
      },
    ),
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      sensors=(),  # feet_ground_contact + self_collision added per-robot.
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
    metrics={},
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",  # Set per-robot.
      distance=3.0,
      elevation=-5.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      njmax=300,
      mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
    ),
    decimation=4,
    episode_length_s=20.0,
  )
