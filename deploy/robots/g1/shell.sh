#!/bin/bash
# Run this while BOTH mujoco_sim AND g1_ctrl are running.
# It answers: are they on the same DDS domain? is lowstate being published?

echo "=== 1. Processes ==="
pgrep -a mujoco_sim || echo "  mujoco_sim NOT running"
pgrep -a g1_ctrl    || echo "  g1_ctrl NOT running"

echo
echo "=== 2. DDS discovery multicast on lo ==="
# CycloneDDS default discovery is on UDP 7400 + domain*250 (so domain 0 = 7400).
# If both processes are on lo domain 0, we should see them.
echo "Looking for listeners on UDP 7400/7401 (domain 0) on 'lo'..."
ss -lun | grep -E "7400|7401|7410|7411" || echo "  No listeners found on standard DDS ports"

echo
echo "=== 3. Is anyone actually using the lo interface for DDS? ==="
sudo ss -lunp 2>/dev/null | grep -E "g1_ctrl|mujoco" || \
  ss -lunp 2>/dev/null | grep -E "g1_ctrl|mujoco" || \
  echo "  (need sudo to see per-process ports)"

echo
echo "=== 4. CycloneDDS env / config ==="
echo "CYCLONEDDS_URI = ${CYCLONEDDS_URI:-(unset)}"
env | grep -i "dds\|unitree" || echo "  No DDS env vars set"

echo
echo "=== 5. simulate/config.yaml contents (adjust path if different) ==="
SIMCFG=$(find ~/Desktop/unitree_rl_mjlab -name "config.yaml" -path "*/simulate/*" 2>/dev/null | head -1)
if [ -z "$SIMCFG" ]; then
  SIMCFG=$(find . -name "config.yaml" -path "*/simulate/*" 2>/dev/null | head -1)
fi
if [ -n "$SIMCFG" ]; then
  echo "Found at: $SIMCFG"
  grep -E "domain_id|interface|use_joystick|joystick_device" "$SIMCFG"
else
  echo "  Could not auto-locate simulate/config.yaml -- check manually"
fi

echo
echo "=== 6. Which interface is mujoco_sim actually using? ==="
# If it's listening on eth0 instead of lo, g1_ctrl won't find it.
sudo lsof -p $(pgrep mujoco_sim) 2>/dev/null | grep -i "udp\|sock" | head -5 \
  || echo "  (need sudo + mujoco_sim running)"