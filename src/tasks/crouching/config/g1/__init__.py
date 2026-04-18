from mjlab.tasks.registry import register_mjlab_task
from src.tasks.crouching.rl import CrouchingOnPolicyRunner

from .env_cfgs import unitree_g1_crouch_env_cfg
from .rl_cfg import unitree_g1_crouch_ppo_runner_cfg

register_mjlab_task(
  task_id="Unitree-G1-Crouch",
  env_cfg=unitree_g1_crouch_env_cfg(),
  play_env_cfg=unitree_g1_crouch_env_cfg(play=True),
  rl_cfg=unitree_g1_crouch_ppo_runner_cfg(),
  runner_cls=CrouchingOnPolicyRunner,
)
