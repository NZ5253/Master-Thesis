# ChronosCar Hardware Deployment

Deploy a trained PPO parallel parking policy on the ChronosCar (1/28 scale RC car).

## Architecture

```
Host (Ubuntu 22.04)               CRS Docker (Ubuntu 20.04 + ROS Noetic)
+------------------------+        +-----------------------------------+
| rl_parking_node.py     |  UDP   | roscore                           |
| - PPO checkpoint       |<------>| WiFiCom + Estimator               |
| - Virtual obstacles    | :5800  | NatNet MoCap bridge               |
| - 10Hz control loop    | :5801  | ros_bridge.py (ROS <-> UDP)       |
| - matplotlib viz       |        +-----------------------------------+
+------------------------+                 |              |
                                     WiFi  |        NatNet |
                                +----------v--+    +-------v------+
                                | ChronosCar  |    | OptiTrack PC |
                                | BEN_CAR_WIFI|    | 129.217.130.57|
                                +-------------+    +--------------+
```

No ROS on host. All ROS runs in Docker. Communication via UDP sockets.

## Files

| File | Where | Purpose |
|------|-------|---------|
| `rl_parking_node.py` | Host | RL inference + observation + control + calibrate + viz + collision recovery + OOD detection |
| `ros_bridge.py` | Docker | ROS <-> UDP bridge (state forwarding + 3-mode torque: stop/forward/reverse) |
| `parking_scene.yaml` | Host | Bay position, vehicle params, obstacles, safety limits |
| `start_docker.sh` | Host | Launches all 4 Docker services in tmux |
| `setup_hardware.sh` | Host | One-time build: Docker image + CRS workspace + NatNet |
| `run_parking.sh` | Host | Wrapper script for rl_parking_node.py |

## First-Time Setup

```bash
./deploy/setup_hardware.sh
```

This builds: Docker image, CRS catkin workspace, NatNet MoCap bridge.

Also update NatNet IPs in `chronoscar-main/natnet_ros_cpp/launch/natnet_ros.launch`:
- `clientIP`: 129.217.130.20 (host machine running Docker)
- `serverIP`: 129.217.130.57 (Motive PC)

## Car WiFi Configuration

The ChronosCar communicates with the host over WiFi (UDP port 20211). You must configure the car with the lab WiFi credentials and the host machine's IP address.

### Steps

1. **Press the PROG button** on the car's PCB (top-left corner, see `chronoscar-main/Chronos Hardware/Picture/PCB.png`). This puts the car into AP (Access Point) mode.

2. **Connect to `crs-car` WiFi** from your phone or laptop. The car creates its own WiFi network named `crs-car`.

3. **Open the config page** in a browser (use incognito/private window to avoid "header fields too long" errors from cookies):
   ```
   http://192.168.4.1/config
   ```

4. **Fill in all three fields** under "UDP Communication":
   - **WiFi SSID**: Your lab WiFi network name (e.g., `Robotik_5GHz`)
   - **WiFi Password**: The WiFi password
   - **IP of host**: The host machine's IP on the lab WiFi (`129.217.130.20`)
     ```bash
     # Find your host IP:
     hostname -I
     ```

5. **Save and power-cycle the car**: Turn OFF, then ON again. The car will connect to the lab WiFi.

### Verifying Car WiFi Connection

Inside Docker, check for steering data:
```bash
rostopic echo /BEN_CAR_WIFI/car_steer_state
```
If data appears, the car is connected. If no data, the car is not communicating (recheck WiFi config).

The car's IP is typically `129.217.130.30` (visible via `tcpdump -i any udp port 20211`).

## Running

### 1. Start Motive (MUST be first)

On the Windows PC (129.217.130.57), open Motive and start streaming.
NatNet bridge crashes silently if Motive isn't running.

### 2. Start Docker Services (car must be OFF)

**IMPORTANT**: Turn the car OFF before starting Docker. The CRS MPC controller_node starts automatically and will race the car at maximum speed if it's on.

```bash
./deploy/start_docker.sh
```

Creates a tmux session `crs` with 4 panes:
1. roscore
2. WiFiCom + Estimator (starts after 3s)
3. NatNet MoCap bridge (starts after 8s)
4. UDP bridge / ros_bridge.py (starts after 10s)

### 3. Kill the CRS Controller

The CRS `controller_node` publishes to `/BEN_CAR_WIFI/control_input` (the same topic our bridge uses), overriding our RL commands. You MUST kill it:

```bash
# Attach to tmux and run in any pane:
tmux attach -t crs

# Inside Docker:
rosnode kill /BEN_CAR_WIFI/controller_node
```

Verify it's gone:
```bash
rosnode list | grep controller
# Should return nothing
```

### 4. Turn the Car ON

Now it's safe to power on the car. It will connect to WiFi but won't move because the controller is killed.

### 5. Calibrate Bay Position

Place the car at the desired parking spot center, aligned with the bay direction:
```bash
python deploy/rl_parking_node.py --calibrate --checkpoint <any_checkpoint>
```

This averages 20 MoCap readings and writes the bay center + yaw to `parking_scene.yaml`.

### 6. Ghost Mode (Verify First)

Move car away from bay (~50cm), facing roughly the same direction as the bay:
```bash
./deploy/run_parking.sh --ghost \
    --checkpoint checkpoints/chronos/.../best_checkpoint
```

Verify:
- along/lateral approach 0 when car is at bay center
- yaw_err is near 0 when car is aligned with bay
- dF/dL/dR show reasonable distances to virtual obstacles (should be ~0.5-1.5, NOT 5.0)
- Live matplotlib window shows car position correctly

### 7. Live Mode

```bash
./deploy/run_parking.sh \
    --checkpoint checkpoints/chronos/.../best_checkpoint
```

**Car positioning**: Place the car ~50cm from the bay, facing roughly the same direction as the bay. The policy was trained with yaw_err within ±45 degrees. If the car faces the wrong way (yaw_err ~180°), the policy will output invalid actions.

### 8. Stop

```bash
# Stop Docker services
./deploy/start_docker.sh --stop

# Turn off the car physically
```

**CRITICAL: Always use Ctrl+C to stop the RL node. Never Ctrl+Z** -- suspended processes hold UDP ports and become zombies.

## Safe Startup Sequence (Summary)

```
1. Start Motive on Windows PC (129.217.130.57)
2. Turn car OFF
3. ./deploy/start_docker.sh          # Start Docker services
4. Wait 15 seconds for all services to initialize
5. tmux attach -t crs                # Attach to tmux
6. rosnode kill /BEN_CAR_WIFI/controller_node   # Kill MPC controller
7. Turn car ON                       # Safe now, no controller to race it
8. python deploy/rl_parking_node.py --calibrate --checkpoint <ckpt>
9. python deploy/rl_parking_node.py --ghost --checkpoint <ckpt>
10. python deploy/rl_parking_node.py --checkpoint <ckpt>   # Live mode
```

## Emergency Stop

If the car is moving and you need to stop it immediately:

1. **Physical power switch** on the car (most reliable)
2. From a Docker terminal, send a zero-velocity command:
   ```bash
   rostopic pub -1 /BEN_CAR_WIFI/control_input crs_msgs/car_input \
       "{torque: 0.0, velocity: 0.0, steer: 0.0, steer_override: false}"
   ```
3. **Do NOT rely on Ctrl+C of `rostopic pub`** -- latched/periodic messages may persist after Ctrl+C

## Modes

| Mode | Flag | Hardware Needed | Description |
|------|------|----------------|-------------|
| Dry-run | `--dry-run` | None | Tests checkpoint loading + obs pipeline |
| Calibrate | `--calibrate` | MoCap + Docker | Reads car position, saves as bay center |
| Ghost | `--ghost` | MoCap + Docker | Reads state, computes actions, no commands |
| Live | (default) | All | Full control loop |

Additional flags:
- `--no-viz`: Disable matplotlib visualization window
- `--scene <path>`: Use alternative scene config (default: `deploy/parking_scene.yaml`)

## tmux Commands

```bash
tmux attach -t crs                 # Attach to session
# Inside tmux:
Ctrl-B then arrow keys             # Switch panes
Ctrl-B then D                      # Detach
# From outside:
./deploy/start_docker.sh --stop    # Kill everything
```

## UDP Protocol

| Direction | Port | Size | Fields |
|-----------|------|------|--------|
| State (Docker -> Host) | 5800 | 56 bytes (7 doubles) | x, y, yaw, v_tot, vx_b, vy_b, steer |
| Command (Host -> Docker) | 5801 | **24 bytes (3 doubles)** | velocity, steer, **v_actual** |
| Command (old, backward compat) | 5801 | 16 bytes (2 doubles) | velocity, steer |

All values big-endian (`struct.pack("!7d", ...)` / `struct.pack("!3d", ...)`).

The 3-double command protocol includes `v_actual` (actual car velocity from MoCap) so `ros_bridge.py` can distinguish "accelerate reverse" from "brake from reverse". Without it, all negative velocity gets reverse torque even when the car should be slowing down.

## CRS car_input Message

The `ros_bridge.py` publishes `car_input` messages with:
- `velocity`: velocity setpoint (m/s) -- **primary control input** (CRS PID for forward only)
- `torque`: feedforward torque with 3-mode computation:
  - **STOP** (`|vel| < 0.01`): Active braking using v_actual direction
  - **FORWARD** (`vel > 0`): `ff_torque = (vel - 0.137) / 6.44`, floor at MIN_TORQUE=0.10
  - **REVERSE** (`vel < 0`): 3 sub-modes: BRAKE (v_actual too fast), START (standstill, -0.10), MAINTAIN (proportional with floor -0.06)
- `steer`: steering angle (rad)
- `steer_override`: False

The CRS ff_fb_controller tracks the velocity setpoint using feedforward + PID (forward only). Reverse uses direct torque because CRS PID is forward-only.

## Configuration: parking_scene.yaml

```yaml
bay:
  center_x: -0.8821     # From MoCap calibration (auto-shifted to body center)
  center_y: -0.266
  yaw: 0.1782           # Bay orientation (rad)

vehicle:
  length: 0.13           # Car body length (m)
  width: 0.065           # Car body width (m)
  wheelbase: 0.09        # Axle-to-axle distance (m)
  rear_overhang: 0.02    # Rear body beyond rear axle (m)
  max_steer: 0.35        # MUST match training (not 0.30!)
  max_vel: 0.5           # MUST match training (not 0.25!)
  max_acc: 0.5           # MUST match training

obstacles:
  neighbor:
    w: 0.13              # Neighbor car length
    h: 0.065             # Neighbor car width
    offset: 0.194        # Bay center to neighbor center distance
    curb_gap: 0.018      # Gap: car side to curb edge
    curb_thickness: 0.014

world:                   # MUST match training config (config_env_chronos.yaml)
  x_min: -1.25           # Auto re-centered on bay by rl_parking_node.py
  x_max: 1.25
  y_min: -1.25
  y_max: 1.25

safety:
  max_torque: 0.15       # Torque safety clamp
  max_steer: 0.35        # MUST match training (not 0.30!)
  max_steer_rate: 0.5    # MUST match training (not 0.3!)
  stale_timeout: 0.3     # Send stop if state older than this (s)

success:
  along_tol: 0.043       # Position tolerance along bay axis (m)
  lateral_tol: 0.043     # Position tolerance lateral (m)
  yaw_tol: 0.15          # Heading tolerance (rad, ~8.6 deg)
  v_tol: 0.05            # Velocity tolerance (m/s)
  settled_steps: 5       # Must be at goal for N consecutive steps
```

**CRITICAL**: The following MUST match training (`config_env_chronos.yaml`) or the policy outputs garbage:
- `world` boundaries: +-1.25m (auto re-centered around bay)
- `vehicle.max_steer`: 0.35 (not 0.30)
- `vehicle.max_vel`: 0.5 (not 0.25)
- `safety.max_steer`: 0.35
- `safety.max_steer_rate`: 0.5 (not 0.3)

## Troubleshooting

### Car doesn't move
- **Is the controller_node killed?** It overrides our commands on the same topic. Run `rosnode kill /BEN_CAR_WIFI/controller_node`.
- **Is velocity being sent?** Check ros_bridge logs for `vel=...` values.
- **Is torque sufficient?** `ros_bridge.py` has MIN_TORQUE=0.10 (overcomes static friction). `rl_parking_node.py` has FRICTION_VEL=0.12 (standstill boost). If car still stuck, check that v_current < 0.01 trigger is working.
- **Is the car WiFi connected?** Check `rostopic echo /BEN_CAR_WIFI/car_steer_state` for data.
- **Are safety clamps correct?** max_steer=0.35 and max_vel=0.5 must match training. Wrong values clip actions.

### Car races at max speed when Docker starts
- The CRS MPC controller_node starts automatically and sends commands immediately.
- **Fix**: Always turn car OFF before starting Docker, kill controller_node, THEN turn car ON.

### Distance features all 5.0 (dF=dL=dR=5.000)
- World boundaries in `parking_scene.yaml` don't match training config.
- Training used `world: ±1.25m`. If deployment uses ±5.0m, rays hit far walls and return 5.0.
- **Fix**: Set `world: {x_min: -1.25, x_max: 1.25, y_min: -1.25, y_max: 1.25}`.

### Policy outputs constant vel=-0.250, steer=-17deg
- The car may be facing the wrong direction (yaw_err ~180°). The policy was trained with yaw_err within ±45°. Reposition the car to face roughly the same direction as the bay.
- Distance features may be out of distribution (check dF/dL/dR values).

### All state values become NaN
- **Never send `torque = NaN`** in car_input -- it poisons the CRS estimator.
- Fix: Restart Docker services: `./deploy/start_docker.sh --stop && ./deploy/start_docker.sh`

### Port 5800 already in use
- Zombie process from Ctrl+Z. Find and kill it:
  ```bash
  ss -ulnp | grep 5800
  kill -9 <PID>
  ```

### No state data received
- Is Motive running? (must start before Docker)
- Is Docker running? (`docker ps`)
- Check NatNet IPs in `chronoscar-main/natnet_ros_cpp/launch/natnet_ros.launch`

### MoCap NaN warnings
- MoCap lost line of sight to car markers.
- RL node auto-skips NaN frames and sends stop commands.
- Reposition car so all markers are visible.

### Velocity reads ~28 m/s while stationary
- CRS estimator v_tot/vx_b are broken.
- RL node computes velocity from MoCap position deltas instead (LP filter, alpha=0.4).

### "Cannot load config from checkpoint"
- Use the actual `checkpoint_XXXXXX` directory, not the parent.
- Or use `best_checkpoint` symlink if it exists.

### "Header fields too long" on car config page
- The ESP32 web server can't handle large cookies.
- **Fix**: Use an incognito/private browser window to access `http://192.168.4.1/config`.

### Can't find crs-car WiFi
- Press the **PROG button** on the car's PCB (top-left corner) to enter AP mode.
- The car must be powered ON for AP mode to work.

### Car overshoots into neighbor obstacles
- **Hybrid velocity integrator?** If using MAX_AHEAD, braking is delayed by extra steps. Use pure closed-loop: `v_target = v_current + accel * dt` (matches training stopping behavior).
- **Velocity filter too heavy?** alpha=0.1 causes 2.3s delay. Use alpha=0.4.
- **Reverse has no braking?** Ensure 3-double protocol is active (24-byte packets). Without v_actual, ros_bridge applies constant reverse torque even when car should slow down.

### Car reaches position but yaw_err won't converge
- **Most likely cause**: Steering/velocity gain mismatch. Run `steer_calibration.py` and update `parking_scene.yaml` calibration section.
- The policy commands small forward accel (0.04 m/s^2) to correct yaw. The resulting v_target=0.004 can't overcome static friction.
- **MIN_CMD boost** should kick in at standstill (v_current < 0.02). If car still doesn't move, increase MIN_CMD from 0.05 to 0.08.
- If boost fires but car overshoots, try lowering MIN_CMD from 0.05 to 0.03.
- **Quick fix**: Relax yaw_tol in parking_scene.yaml from 0.15 (8.6 deg) to 0.30 (17 deg).

### Car shoots forward unexpectedly during fine adjustments
- **Old issue**: Was caused by FRICTION_VEL=0.12 boost firing during reverse→forward transition.
- **Current fix**: MIN_CMD=0.05 uses `accel_cmd` sign for direction, not `v_target` sign. No STANDSTILL_GRACE needed.
- If still happens, reduce MIN_CMD from 0.05 to 0.03.

### Car gets stuck after hitting virtual obstacle
- Training terminates on collision. Policy never learned recovery. dF=dL=dR=0 is undefined behavior.
- rl_parking_node.py has auto-reverse recovery (15 steps at vel=-0.12), max 5 recoveries before giving up.
- If car repeatedly collides, reposition it further from the bay.

### rostopic pub command won't stop
- `rostopic pub -1` sends a single latched message that persists.
- Send a zero-velocity command from a new terminal to override it.
- Ultimate fallback: physical power switch on the car.

## Key Deployment Concepts

### Velocity Integrator (ActionConverter)

The most critical sim-to-real component. Training uses `v_new = v + accel * dt` (instant, frictionless). Hardware has static friction requiring ~0.10 torque to start moving.

**Current design**: Closed-loop `v_target = v_current + accel * dt` with smooth MIN_CMD friction compensation:
```python
MIN_CMD = 0.05           # Minimum velocity to overcome static friction
STANDSTILL_THRESH = 0.02 # Car is "at standstill"

vel_cmd = v_target
if abs(v_current) < STANDSTILL_THRESH and abs(accel_cmd) > 0.01:
    if abs(vel_cmd) < MIN_CMD:
        vel_cmd = sign(accel_cmd) * MIN_CMD  # Use accel_cmd sign, not v_target
```

Key parameters:
- `MIN_CMD = 0.05`: Minimum velocity to generate enough torque to start
- `STANDSTILL_THRESH = 0.02`: Below this, car is considered stopped
- Uses `accel_cmd` sign (not `v_target` sign) to determine direction -- avoids zero-crossing bug during reverse→forward transitions

Also tracks `v_model` (open-loop integrator: `v_model += accel*dt`) to diagnose sim-to-real velocity gap.

### Calibration Correction

Measured with `steer_calibration.py`: car steers 1.376x and drives 1.503x commanded.
`ActionConverter.convert()` divides commands by these gains AFTER rate limiting.

### Collision Recovery

Training terminates on collision (`done=True, reward=-1000`). The policy never learned recovery.
When car enters obstacle bounding box (dF=dL=dR=0):
1. Stop immediately
2. Reverse at vel=-0.12, steer=0 for 15 steps (1.5s)
3. Reset velocity integrator + filter
4. Resume policy control
5. Give up after 5 collisions

### Calibration

The `--calibrate` mode reads the car's MoCap position and saves it as bay center. Critical: CRS reports **rear-axle** position, but training uses **body center**. Calibration auto-shifts forward by `dist_to_center = L/2 - rear_overhang = 0.045m`. Without this, `along=0.045` even when perfectly parked.

### World Boundary Re-centering

Training has the bay near the origin, so world walls at +-1.25m are symmetric. The real bay can be anywhere in the MoCap frame. `rl_parking_node.py` auto-shifts world boundaries to be centered on `bay_center`, so ray-cast distances match training distribution.

## Network Addresses

| Device | IP | Purpose |
|--------|----|---------|
| Host machine | 129.217.130.20 | Runs Docker + RL node |
| OptiTrack/Motive PC | 129.217.130.57 | MoCap server (NatNet) |
| ChronosCar | 129.217.130.30 | RC car (WiFi UDP) |
| Car AP mode | 192.168.4.1 | Config page when in AP mode |

## Testing Sequence

1. **Dry-run** -- No hardware, tests checkpoint + observation pipeline
2. **Ghost mode** -- MoCap reads, no motor commands. Verify observations.
3. **Wheels-off** -- Car propped up, verify steering/velocity direction
4. **Open floor** -- No virtual neighbors (set `neighbor.offset: 0`), verify car approaches bay
5. **Full test** -- With virtual neighbors, car parks between them
