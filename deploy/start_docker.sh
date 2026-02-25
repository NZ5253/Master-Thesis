#!/bin/bash
# =========================================================
# Start all CRS Docker services in a single tmux session
# =========================================================
# Launches 4 services inside the CRS Docker container:
#   1. roscore
#   2. WiFiCom + Estimator
#   3. NatNet MoCap bridge
#   4. UDP bridge (ros_bridge.py)
#
# Usage:
#   ./deploy/start_docker.sh              # start all
#   ./deploy/start_docker.sh --stop       # kill the session
#   ./deploy/start_docker.sh --attach     # attach to tmux session
#
# Once running, attach with:  tmux attach -t crs
# Switch panes: Ctrl-B then arrow keys
# =========================================================

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CRS_DIR="$PROJECT_DIR/chronoscar-main/crs-main/software"
NATNET_DIR="$PROJECT_DIR/chronoscar-main/natnet_ros_cpp"
SESSION="crs"
CONTAINER="crs"
NAMESPACE="${NAMESPACE:-BEN_CAR_WIFI}"

ROS_SETUP="source /opt/ros/noetic/setup.bash && source /code/devel/setup.bash"

# ---- Handle flags ----
if [ "$1" = "--stop" ]; then
    echo "Stopping..."
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    docker stop "$CONTAINER" 2>/dev/null || true
    echo "Done."
    exit 0
fi

if [ "$1" = "--attach" ]; then
    tmux attach -t "$SESSION"
    exit 0
fi

# ---- Check prerequisites ----
if ! command -v tmux &>/dev/null; then
    echo "ERROR: tmux not installed. Run: sudo apt install tmux"
    exit 1
fi

if ! docker ps &>/dev/null; then
    echo "ERROR: Docker not accessible. Check permissions (newgrp docker)."
    exit 1
fi

if ! docker images | grep -q "crs.*local"; then
    echo "ERROR: CRS Docker image not built. Run: ./deploy/setup_hardware.sh"
    exit 1
fi

# ---- Stop any existing session/container ----
tmux kill-session -t "$SESSION" 2>/dev/null || true
docker stop "$CONTAINER" 2>/dev/null || true
docker rm "$CONTAINER" 2>/dev/null || true
sleep 1

# ---- Start the Docker container in detached mode ----
echo "Starting CRS Docker container..."
docker run -d --name "$CONTAINER" \
    -v "$CRS_DIR:/code" \
    -v "$PROJECT_DIR:/project:ro" \
    -v "$NATNET_DIR:/code/src/natnet_ros_cpp" \
    -v /dev:/dev:rw \
    --network host --privileged \
    crs:local \
    bash -c "$ROS_SETUP && roscore"

echo "Waiting for roscore to start..."
sleep 3

# ---- Create tmux session with 4 panes ----
# Layout: 2x2 grid
#  ┌──────────────┬──────────────┐
#  │  1. roscore  │  2. CRS      │
#  ├──────────────┼──────────────┤
#  │  3. NatNet   │  4. UDP      │
#  └──────────────┴──────────────┘

# Pane 1: roscore logs
tmux new-session -d -s "$SESSION" -n "crs" \
    "docker logs -f $CONTAINER"

# Pane 2: WiFiCom + Estimator
tmux split-window -h -t "$SESSION" \
    "echo 'Starting WiFiCom + Estimator (waiting 3s)...' && sleep 3 && \
     docker exec -it $CONTAINER bash -c '$ROS_SETUP && \
     roslaunch crs_launch run_single_car.launch \
       namespace:=$NAMESPACE experiment_name:=real_world_mpc run_type:=real'"

# Pane 3: NatNet bridge
tmux split-window -v -t "$SESSION:0.0" \
    "echo 'Starting NatNet bridge (waiting 8s)...' && sleep 8 && \
     docker exec -it $CONTAINER bash -c '$ROS_SETUP && \
     roslaunch natnet_ros_cpp natnet_ros.launch'"

# Pane 4: UDP bridge
tmux split-window -v -t "$SESSION:0.1" \
    "echo 'Starting UDP bridge (waiting 10s)...' && sleep 10 && \
     docker exec -it $CONTAINER bash -c '$ROS_SETUP && \
     python3 /project/deploy/ros_bridge.py --namespace $NAMESPACE'"

# Even out the panes
tmux select-layout -t "$SESSION" tiled

echo ""
echo "========================================="
echo "CRS Docker services starting in tmux!"
echo "========================================="
echo ""
echo "  Attach:   tmux attach -t crs"
echo "  Detach:   Ctrl-B then D"
echo "  Panes:    Ctrl-B then arrow keys"
echo "  Stop all: ./deploy/start_docker.sh --stop"
echo ""
echo "  Then run the RL node on host:"
echo "  ./deploy/run_parking.sh --ghost --checkpoint checkpoints/chronos/.../best_checkpoint"
echo ""

# Auto-attach
tmux attach -t "$SESSION"
