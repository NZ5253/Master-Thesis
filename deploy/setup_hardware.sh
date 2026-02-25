#!/bin/bash
# =========================================================
# ChronosCar Hardware Setup Script
# =========================================================
# Sets up everything needed for hardware deployment:
#   1. Docker permissions
#   2. CRS Docker image (build)
#   3. CRS workspace (build inside Docker)
#   4. NatNet MoCap bridge (build inside Docker)
#
# NO ROS installation on the host is needed!
# ROS runs only inside the Docker container.
# The RL node communicates via a lightweight UDP bridge.
#
# Run: ./deploy/setup_hardware.sh
# =========================================================

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CRS_DIR="$PROJECT_DIR/chronoscar-main/crs-main/software"
NATNET_DIR="$PROJECT_DIR/chronoscar-main/natnet_ros_cpp"

# Detect docker compose command (v2 plugin vs standalone vs fallback)
if docker compose version &>/dev/null; then
    COMPOSE="docker compose"
elif command -v docker-compose &>/dev/null; then
    COMPOSE="docker-compose"
else
    COMPOSE=""
fi

echo "========================================="
echo "ChronosCar Hardware Setup"
echo "========================================="
echo "Project: $PROJECT_DIR"
echo ""

# ---- Helper ----
step_done() {
    echo ""
    echo "  [DONE] $1"
    echo ""
}

# =========================================================
# STEP 1: Docker permissions
# =========================================================
echo "--- Step 1: Docker permissions ---"

if docker ps &>/dev/null; then
    echo "  Docker already working for current user."
    step_done "Docker permissions"
else
    echo "  Adding user to docker group..."
    sudo usermod -aG docker "$USER"
    echo ""
    echo "  !! You need to log out and log back in (or run 'newgrp docker')"
    echo "  !! Then re-run this script."
    echo ""
    echo "  Quick fix (current terminal only):"
    echo "    newgrp docker"
    echo "    ./deploy/setup_hardware.sh"
    exit 1
fi

# =========================================================
# STEP 2: Build CRS Docker image
# =========================================================
echo "--- Step 2: CRS Docker image ---"

if docker images | grep -q "crs.*local"; then
    echo "  CRS Docker image already built."
    step_done "CRS Docker image"
else
    echo "  Building CRS Docker image (this takes 10-20 min first time)..."
    cd "$CRS_DIR"
    if [ -n "$COMPOSE" ]; then
        $COMPOSE build
    else
        echo "  (docker compose not found, using docker build directly)"
        docker build -t crs:local .
    fi
    step_done "CRS Docker image built"
fi

# =========================================================
# STEP 3: Build CRS workspace inside Docker
# =========================================================
echo "--- Step 3: Build CRS workspace in Docker ---"

if [ -d "$CRS_DIR/devel" ] && [ -f "$CRS_DIR/devel/setup.bash" ]; then
    echo "  CRS workspace already built."
    step_done "CRS workspace"
else
    echo "  Building CRS workspace inside Docker..."
    echo "  (This may take 5-10 min first time)"
    cd "$CRS_DIR"
    if [ -n "$COMPOSE" ]; then
        $COMPOSE run --rm crs bash -c \
            "source /opt/ros/noetic/setup.bash && cd /code && crs init && crs build"
    else
        docker run --rm -v "$CRS_DIR:/code" -v "$PROJECT_DIR:/project:ro" \
            --network host crs:local bash -c \
            "source /opt/ros/noetic/setup.bash && cd /code && crs init && crs build"
    fi
    step_done "CRS workspace built in Docker"
fi

# =========================================================
# STEP 4: Build NatNet MoCap bridge inside Docker
# =========================================================
echo "--- Step 4: NatNet MoCap bridge (in Docker) ---"

# Check if NatNet SDK is installed
if [ ! -d "$NATNET_DIR/deps/NatNetSDK" ]; then
    echo "  Installing NatNet SDK..."
    cd "$NATNET_DIR"
    bash install_sdk.sh
    step_done "NatNet SDK installed"
else
    echo "  NatNet SDK already installed."
fi

# Build NatNet bridge inside Docker using the CRS catkin workspace
# We symlink natnet_ros_cpp into the CRS workspace src/ and build it there
NATNET_BUILD_MARKER="$CRS_DIR/devel/lib/natnet_ros_cpp/natnet_ros_cpp"
if [ -f "$NATNET_BUILD_MARKER" ]; then
    echo "  NatNet bridge already built in Docker."
    step_done "NatNet bridge"
else
    echo "  Building NatNet bridge inside Docker..."
    cd "$CRS_DIR"

    # Mount natnet_ros_cpp directly into the CRS catkin workspace
    if [ -n "$COMPOSE" ]; then
        $COMPOSE run --rm crs bash -c \
            "source /opt/ros/noetic/setup.bash && \
             cd /code && \
             source devel/setup.bash && \
             catkin build natnet_ros_cpp"
    else
        docker run --rm \
            -v "$CRS_DIR:/code" \
            -v "$NATNET_DIR:/code/src/natnet_ros_cpp" \
            -v "$PROJECT_DIR:/project:ro" \
            --network host crs:local bash -c \
            "source /opt/ros/noetic/setup.bash && \
             cd /code && \
             source devel/setup.bash && \
             catkin build natnet_ros_cpp"
    fi
    step_done "NatNet bridge built in Docker"
fi

# =========================================================
# SUMMARY
# =========================================================
echo ""
echo "========================================="
echo "Hardware Setup Complete!"
echo "========================================="
echo ""
echo "Architecture:"
echo "  Host:   RL node (Python + Ray/PyTorch) <-- no ROS needed"
echo "  Docker: roscore + WiFiCom + Estimator + NatNet + UDP bridge"
echo "  Communication: UDP on localhost (network_mode: host)"
echo ""
echo "Next steps:"
echo ""
echo "1. Start CRS Docker (Terminal 1):"
echo "   cd $CRS_DIR"
if [ -n "$COMPOSE" ]; then
echo "   $COMPOSE run --rm crs"
else
echo "   docker run --rm -it --name crs -v $CRS_DIR:/code -v $PROJECT_DIR:/project:ro \\"
echo "     --network host --privileged -v /dev:/dev:rw crs:local"
fi
echo "   # Inside Docker:"
echo "   source /opt/ros/noetic/setup.bash && source /code/devel/setup.bash"
echo "   roscore"
echo ""
echo "2. Start WiFiCom + Estimator (Terminal 2, attach to Docker):"
echo "   docker exec -it crs bash"
echo "   source /opt/ros/noetic/setup.bash && source /code/devel/setup.bash"
echo "   roslaunch crs_launch run_single_car.launch \\"
echo "     namespace:=BEN_CAR_WIFI experiment_name:=real_world_mpc run_type:=real"
echo ""
echo "3. Start NatNet bridge (Terminal 3, attach to Docker):"
echo "   docker exec -it crs bash"
echo "   source /opt/ros/noetic/setup.bash && source /code/devel/setup.bash"
echo "   roslaunch natnet_ros_cpp natnet_ros.launch"
echo ""
echo "4. Start UDP bridge (Terminal 4, attach to Docker):"
echo "   docker exec -it crs bash"
echo "   source /opt/ros/noetic/setup.bash && source /code/devel/setup.bash"
echo "   python3 /project/deploy/ros_bridge.py"
echo ""
echo "5. Run RL parking node (Terminal 5, on HOST):"
echo "   ./deploy/run_parking.sh --ghost \\"
echo "     --checkpoint checkpoints/chronos/.../best_checkpoint"
echo ""
echo "========================================="
