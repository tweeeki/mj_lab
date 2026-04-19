from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import unitree_g1_reach_env_cfg
from .rl_cfg import unitree_g1_reach_ppo_runner_cfg

register_mjlab_task(
  task_id="Unitree-G1-Reach",
  env_cfg=unitree_g1_reach_env_cfg(),
  play_env_cfg=unitree_g1_reach_env_cfg(play=True),
  rl_cfg=unitree_g1_reach_ppo_runner_cfg(),
)
