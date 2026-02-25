#!/bin/bash
# =========================================================
# ChronosCar RL Parking Deployment Script
# =========================================================
#
# Runs the RL parking node on the HOST (no ROS needed).
# Communicates with the CRS Docker via UDP bridge.
#
# Prerequisites:
#   1. CRS Docker running with roscore + WiFiCom + Estimator
#   2. NatNet MoCap bridge running (inside Docker)
#   3. UDP bridge running (inside Docker): python3 /project/deploy/ros_bridge.py
#
# Usage:
#   # Test without hardware (loads checkpoint, tests observation pipeline)
#   ./deploy/run_parking.sh --dry-run --checkpoint checkpoints/chronos/.../checkpoint_000XXX
#
#   # Ghost mode (reads state via UDP, computes actions, but does NOT send to car)
#   ./deploy/run_parking.sh --ghost --checkpoint checkpoints/chronos/.../checkpoint_000XXX
#
#   # Live mode (actually controls the car!)
#   ./deploy/run_parking.sh --checkpoint checkpoints/chronos/.../checkpoint_000XXX
# =========================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Activate virtual environment
if [ -f "$PROJECT_DIR/venv/bin/activate" ]; then
    source "$PROJECT_DIR/venv/bin/activate"
fi

# Set PYTHONPATH to include project root
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

echo "========================================="
echo "ChronosCar RL Parking Deployment"
echo "========================================="
echo ""
echo "No ROS needed on host -- using UDP bridge"
echo ""

# Pass all arguments through to the Python node
cd "$PROJECT_DIR"
python deploy/rl_parking_node.py \
    --scene deploy/parking_scene.yaml \
    "$@"
