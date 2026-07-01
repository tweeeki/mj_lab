# Tasks

Each subfolder is one RL task. The pattern is identical across all of them:

```
<task>/
├── __init__.py                 ← module docstring
├── <task>_env_cfg.py           ← env builder (obs/act/rewards/...)
├── mdp/                        ← task-specific Python: commands, observations, rewards
├── rl/runner.py                ← MjlabOnPolicyRunner subclass that exports for deploy
└── config/<robot>/__init__.py  ← register_mjlab_task(...) — entry point
```

The `register_mjlab_task` call in `config/<robot>/__init__.py` is what makes
the task discoverable by `scripts/train.py` and `scripts/list_envs.py`.

## What's here

| Task | Notes |
|---|---|
| `velocity/`  | Locomotion, `wasdqe` keyboard velocity command. |
| `crouching/` | Single height slider command. Cleanest copy-paste template for new RL tasks with custom obs. |
| `reaching/`  | Bimanual sphere reach. Adds three custom obs and a 6-D pelvis-frame command. **See [docs/06_custom_reach_task.md](../../../docs/06_custom_reach_task.md) for the worked example.** |
| `tracking/`  | Motion imitation from a reference trajectory. |

## Adding a new task

1. Copy `crouching/` (or `reaching/` if you also need new obs).
2. Rename to `<my_task>/`, including the inner files.
3. Edit `<my_task>_env_cfg.py` for your obs/rewards/commands.
4. Edit `rl/runner.py`'s `_OBS_NAME_MAP` to match the names your C++
   deploy will expect.
5. Register in `config/<robot>/__init__.py`.

See [docs/03_training.md](../../../docs/03_training.md) for the full
explanation.
