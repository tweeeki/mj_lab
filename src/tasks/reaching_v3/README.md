> **v3 (real-robot lag in the loop) copy of `tasks/reaching_v2`.** Sim2real oscillation fix: randomized observation delay (0–2 policy steps) on all sensor-derived actor terms, action delay widened to 0–8 substeps (two-step target history), startup DR over joint frictionloss / armature / PD gains, `waypoint_track` std 0.05→0.10, `action_rate_l2` −0.02→−0.05, mild arm `joint_vel_l2`. Exporter emits `action_smooth_alpha` / `gain_ramp_s` / `commands.reach.dev` and `clip: null`. Obs/action dims and deploy contract unchanged (112-dim, locked wrists, 7-D speed command). Registers as `Unitree-G1-Reach-V3`.

# reaching/

Bimanual sphere-reach task. Both arms track 3D target spheres in the
pelvis frame; legs and torso are held by a `posture` reward (no welding).

## Files

| File | Purpose |
|---|---|
| `__init__.py`              | Module docstring; explains the pelvis-frame anchoring. |
| `reaching_env_cfg.py`      | Generic env: obs, actions, commands, rewards, terms, events. |
| `mdp/sphere_command.py`    | `UniformBimanualSphereCommand` — 6-D goal sampler with per-axis ranges and 3–6 s resampling. |
| `mdp/observations.py`      | `hand_pos_b`, `hand_target_error_b` — palm positions and target error in robot root frame. |
| `mdp/rewards.py`           | `reach_distance` (exp), `reach_distance_l2`, `reach_success_bonus`. |
| `rl/runner.py`             | `ReachingOnPolicyRunner` — exports `policy.onnx` + `deploy.yaml` at every checkpoint. **Read `_build_deploy_cfg`** to understand the deploy-side contract. |
| `config/g1/env_cfgs.py`    | G1-specific bindings: site names, posture-reward joint regex, sphere ranges. |
| `config/g1/rl_cfg.py`      | PPO hyperparameters. |
| `config/g1/__init__.py`    | `register_mjlab_task("Unitree-G1-Reach", ...)`. |

## Actor obs (in order, total 111 floats for 29-DoF G1)

| # | Name | Dim |
|---|------|----:|
| 1 | `base_ang_vel`        |  3 |
| 2 | `projected_gravity`   |  3 |
| 3 | `command`             |  6 (left xyz + right xyz, pelvis frame) |
| 4 | `hand_pos_b`          |  6 |
| 5 | `hand_target_error_b` |  6 |
| 6 | `joint_pos` (rel)     | 29 |
| 7 | `joint_vel` (rel)     | 29 |
| 8 | `actions` (last)      | 29 |

The order **must match** `deploy.yaml`'s `observations` order on the C++
side, because both sides concatenate in this order to build the input
vector for the policy.

## Default sphere ranges (pelvis frame, meters)

```
left_x  ∈ [0.20, 0.45]    right_x ∈ [0.20, 0.45]
left_y  ∈ [0.05, 0.35]    right_y ∈ [-0.35, -0.05]
left_z  ∈ [-0.10, 0.25]   right_z ∈ [-0.10, 0.25]
```

In front of the pelvis, left hand on +y, right hand on -y.

## Train it

```bash
cd ../../..
python scripts/train.py --task Unitree-G1-Reach --num-envs 4096
```

Output checkpoints in `logs/Unitree-G1-Reach/<run>/` include
`policy.onnx` + `deploy.yaml` next to each `.pt` — those two are the
handoff to the deploy side.

## Deploy it

See [../../../../docs/04_sim2sim.md](../../../../docs/04_sim2sim.md) and
[../../../../docs/06_custom_reach_task.md](../../../../docs/06_custom_reach_task.md).
