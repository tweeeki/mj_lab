"""Loco-manipulation: walk + stay upright + bimanual sphere reaching.

reaching_loco = the LEGS from the velocity (locomotion) task + the ARMS from
reaching_v2, co-trained. The robot follows a twist (walk) command with its legs
(marching-style gait) while both arms track a commanded pair of 3D sphere
targets, and the torso is kept upright. The actor is spectral-normalized (hidden
layers only, so the legs stay reactive enough to walk) as in reaching_v2.

This is a NEW observation/action contract (adds twist command + gait phase; the
legs now walk instead of standing planted), so it is NOT deploy-compatible with
reach/v2 — it needs its own policy + deploy wiring.
"""
