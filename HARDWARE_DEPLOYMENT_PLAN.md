# Hardware Deployment Plan: RL Parallel Parking on ChronosCar

**Created**: 2026-02-08
**Status**: PLANNING PHASE
**Target**: Deploy trained PPO policy to real 1/28 scale ChronosCar

---

## Executive Summary

### Critical Finding: Dimension Mismatch

| Parameter | RL Training | ChronosCar Real | Ratio |
|-----------|-------------|-----------------|-------|
| Wheelbase | 0.25 m | 0.090 m | 2.78x |
| Length | 0.36 m | ~0.13 m | 2.77x |
| Width | 0.26 m | ~0.09 m | 2.89x |
| Max Steer | 0.523 rad | 0.35 rad | 1.49x |

**Impact**: The RL policy was trained on a car ~2.8x larger than ChronosCar. This requires either:
1. **Scale-compensated deployment** (faster, may work)
2. **Fine-tuning on correct dimensions** (recommended, more reliable)

---

## Phase 0: Preparation & Inventory

### 0.1 Required Files Checklist

```
deployment/
├── checkpoint/                    # Best PPO checkpoint
│   └── [copy from checkpoints/curriculum/<run_id>/phase<N>/best_checkpoint/]
├── configs/
│   ├── deploy_env.yaml           # Merged env config (to create)
│   ├── parking_scene_real.yaml   # Bay + virtual obstacles (to create)
│   └── chronos_model.yaml        # Real car params (from ChronosCar)
├── nodes/
│   └── rl_parking_node.py        # ROS node for inference (to create)
├── launch/
│   └── rl_parking.launch         # Launch file (to create)
└── README_DEPLOY.md              # Deployment instructions (to create)
```

### 0.2 Action Items

- [ ] Identify best checkpoint path
- [ ] Extract final phase config values
- [ ] Verify RLlib checkpoint can be loaded
- [ ] Confirm ChronosCar hardware is functional

---

## Phase 1: ChronosCar Hardware Baseline

### 1.1 Network Configuration

**On ChronosCar PCB:**
1. Press config button → join WiFi `crs-car`
2. Navigate to `http://192.168.4.1/config`
3. Set "IP of host" = your Ubuntu ROS PC IP
4. Power cycle car

### 1.2 Docker Environment

```bash
cd chronoscar-main/crs-main/software
crs-docker up
crs-docker run

# Inside container
roscore
```

### 1.3 Motion Capture Setup

**On host terminal (not container):**
```bash
cd chronoscar-main/natnet_ros_cpp
python3 set_ros_env.py
roslaunch natnet_ros_cpp natnet_ros.launch
```

**Edit launch file parameters:**
- `serverIP` = Motive PC IP
- `clientIP` = ROS machine IP

**Verify:**
```bash
rostopic list | grep natnet
rostopic echo /natnet/<YourRigidBodyName>/car_1
```

### 1.4 Estimator Topic Fix

**Issue**: Estimator hardcoded to `/natnet/BEN_CAR_WIFI/car_1`

**Option A (Quick)**: Rename rigid body in Motive to `BEN_CAR_WIFI`

**Option B (Proper)**: Patch estimator to use configurable topic

Edit: `chronoscar-main/crs-main/software/src/ros4crs/ros_estimators/include/ros_estimators/car_estimator/car_estimator.h`

Change subscription to use remapped topic, then rebuild:
```bash
crs build
```

---

## Phase 2: Idle Controller Setup

### 2.1 Create RL Experiment Folder

```bash
cp -r chronoscar-main/crs-main/software/experiments/real_world_mpc \
      chronoscar-main/crs-main/software/experiments/real_world_rl
```

### 2.2 Configure Idle Controller

Replace controller config with joystick (no automatic control):
```bash
cp chronoscar-main/crs-main/software/src/ros4crs/ros_controllers/config/joystick_controller.yaml \
   chronoscar-main/crs-main/software/experiments/real_world_rl/controller.yaml
```

### 2.3 Launch Idle Stack

```bash
roslaunch crs_launch run_single_car.launch \
    namespace:=BEN_CAR_WIFI \
    experiment_name:=real_world_rl
```

**Verify:**
```bash
# Should publish state but NOT control the car
rostopic echo /BEN_CAR_WIFI/estimation_node/best_state
```

---

## Phase 3: Parking Scene Configuration

### 3.1 Measure Bay Pose in MoCap Frame

1. Physically place car at desired parking spot (bay center)
2. Align car straight (yaw = 0 relative to bay)
3. Record pose:
```bash
rostopic echo /BEN_CAR_WIFI/estimation_node/best_state
# Note: x, y, yaw values
```

### 3.2 Create parking_scene_real.yaml

```yaml
# Bay pose in MoCap world frame
bay:
  center_x: <measured_x>       # From step 3.1
  center_y: <measured_y>       # From step 3.1
  yaw: <measured_yaw>          # Usually 0.0 if aligned

# Virtual neighbors (simulated, not physical)
obstacles:
  neighbor:
    enabled: true
    width: 0.36                # Training value (scaled for policy)
    height: 0.26
    offset: 0.54
    pos_jitter: 0.0            # Start without jitter

  # Disable random obstacles initially
  random:
    num_min: 0
    num_max: 0

  # Disable world walls (would cause phantom collisions)
  world:
    enabled: false
```

---

## Phase 4: Deploy Configuration Export

### 4.1 Select Deployment Phase

**Recommended**: Phase 5 or 6 (balanced precision vs. robustness)

| Phase | Tolerance | Recommended For |
|-------|-----------|-----------------|
| Phase 5 | 6 cm | Initial deployment testing |
| Phase 6 | 5 cm | Production deployment |
| Phase 7 | 4 cm | Final polish (harder) |

### 4.2 Create Merged deploy_env.yaml

Extract from curriculum_config.yaml and config_env.yaml:

```yaml
# Vehicle (TRAINING VALUES - may need scaling)
vehicle:
  L: 0.36
  W: 0.26
  wheelbase: 0.25
  max_steer: 0.523
  max_vel: 1.0
  max_acc: 1.0
  dt: 0.1

# Car center offset from rear axle
car_center_offset: 0.13      # L/2 - 0.05

# Action scaling
action_scale:
  steer: 1.0
  accel: 1.0

# Success tolerances (from selected phase)
tolerances:
  along: 0.06                # Phase 5 value
  lateral: 0.06
  yaw: 0.087                 # 5 degrees
  velocity: 0.05

# Settling detection
settling:
  v_threshold: 0.01
  steer_threshold: 0.05
  steps_required: 3

# DEPLOYMENT OVERRIDES
deployment:
  randomize_bay: false       # Fixed bay in real world
  randomize_spawn: false     # Car starts where it is
```

---

## Phase 5: Scale Compensation Strategy

### 5.1 The Mismatch Problem

The policy "thinks" the car is 2.8x larger. When it outputs steering/acceleration:
- Real car responds faster (shorter wheelbase)
- Real distances are smaller

### 5.2 Compensation Approaches

#### Strategy A: Observation Scaling (Recommended First)

Scale observations TO MATCH training expectations:

```python
# In rl_parking_node.py
SCALE_FACTOR = 2.78  # Training wheelbase / Real wheelbase

def scale_observation(real_obs):
    """Scale real-world obs to match training scale."""
    scaled = real_obs.copy()
    # Scale distances
    scaled[0] *= SCALE_FACTOR  # along
    scaled[1] *= SCALE_FACTOR  # lateral
    # yaw_err unchanged (angle)
    # v unchanged (for now)
    scaled[4] *= SCALE_FACTOR  # dist_front
    scaled[5] *= SCALE_FACTOR  # dist_left
    scaled[6] *= SCALE_FACTOR  # dist_right
    return scaled
```

#### Strategy B: Action Scaling

Scale actions FROM policy to match real car:

```python
def scale_action(policy_action):
    """Scale policy output for smaller car."""
    # Smaller car needs less aggressive steering
    steer_scale = 0.35 / 0.523  # Real max / Training max
    return [
        policy_action[0] * steer_scale,  # steer
        policy_action[1] * 0.5           # accel (reduce initially)
    ]
```

#### Strategy C: Fine-Tune (Best Long-Term)

Retrain with correct dimensions:
```yaml
# Updated config_env.yaml for ChronosCar
vehicle:
  L: 0.13         # Real length
  W: 0.09         # Real width
  wheelbase: 0.09 # Real wheelbase (lf + lr)
  max_steer: 0.35 # Real steering limit
```

### 5.3 Recommended Path

1. **Start with Strategy A+B** (no retraining)
2. **Test and tune scaling factors**
3. **If unstable, proceed to Strategy C** (fine-tune ~5M steps)

---

## Phase 6: RL Hardware Node Implementation

### 6.1 Node Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    rl_parking_node.py                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ State Sub   │───>│ Observation  │───>│ PPO Policy   │  │
│  │ best_state  │    │ Builder      │    │ Inference    │  │
│  └─────────────┘    └──────────────┘    └──────────────┘  │
│                                                │            │
│  ┌─────────────┐    ┌──────────────┐    ┌──────┴───────┐  │
│  │ Control Pub │<───│ Action       │<───│ Action       │  │
│  │ car_input   │    │ Limiter      │    │ Converter    │  │
│  └─────────────┘    └──────────────┘    └──────────────┘  │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐  │
│  │                   Safety Layer                       │  │
│  │  - Enable switch    - Stale state watchdog          │  │
│  │  - Output clamps    - Rate limiters                 │  │
│  │  - Success stop     - Emergency stop                │  │
│  └─────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 Key Implementation

```python
#!/usr/bin/env python3
"""
RL Parking Node - Deploys PPO policy on ChronosCar
"""
import rospy
import numpy as np
from crs_msgs.msg import car_state_cart, car_input
from ray.rllib.algorithms.ppo import PPO

class RLParkingNode:
    def __init__(self):
        rospy.init_node('rl_parking_node')

        # Parameters
        self.namespace = rospy.get_param('~namespace', 'BEN_CAR_WIFI')
        self.checkpoint_path = rospy.get_param('~checkpoint')
        self.enable = rospy.get_param('~enable', False)

        # Bay configuration (from parking_scene_real.yaml)
        self.bay_x = rospy.get_param('~bay/center_x')
        self.bay_y = rospy.get_param('~bay/center_y')
        self.bay_yaw = rospy.get_param('~bay/yaw')

        # Safety limits
        self.max_torque = rospy.get_param('~safety/max_torque', 0.15)
        self.max_steer = rospy.get_param('~safety/max_steer', 0.30)
        self.stale_timeout = rospy.get_param('~safety/stale_timeout', 0.2)

        # Tolerances (for success detection)
        self.tol_along = rospy.get_param('~tolerances/along', 0.06)
        self.tol_lat = rospy.get_param('~tolerances/lateral', 0.06)
        self.tol_yaw = rospy.get_param('~tolerances/yaw', 0.087)

        # Load policy
        self.policy = self._load_policy()

        # State tracking
        self.last_state_time = None
        self.last_state = None
        self.last_action = np.zeros(2)

        # Virtual obstacle manager (neighbor cars)
        self.obstacle_manager = self._init_obstacles()

        # Publishers/Subscribers
        self.cmd_pub = rospy.Publisher(
            f'/{self.namespace}/control_input',
            car_input, queue_size=1
        )
        self.state_sub = rospy.Subscriber(
            f'/{self.namespace}/estimation_node/best_state',
            car_state_cart, self.state_callback
        )

        # Control timer (10 Hz = 0.1s, matching training)
        self.timer = rospy.Timer(rospy.Duration(0.1), self.control_loop)

        rospy.loginfo(f"RL Parking Node initialized. Enable={self.enable}")

    def _load_policy(self):
        """Load PPO checkpoint."""
        # Configure minimal Ray for inference only
        from ray.rllib.algorithms.algorithm import Algorithm
        algo = Algorithm.from_checkpoint(self.checkpoint_path)
        return algo

    def _init_obstacles(self):
        """Initialize virtual obstacle manager."""
        # Simplified version - compute distances to virtual neighbors
        # based on training obstacle_manager.py logic
        pass  # Implementation details

    def state_callback(self, msg):
        """Store latest state."""
        self.last_state = msg
        self.last_state_time = rospy.Time.now()

    def build_observation(self, state):
        """
        Build 7D observation vector matching training.

        Observation: [along, lateral, yaw_err, v, dist_front, dist_left, dist_right]
        """
        # Extract world pose
        x, y, yaw = state.x, state.y, state.yaw
        v = state.v_tot

        # Compute car center (from rear axle)
        CAR_CENTER_OFFSET = 0.13  # L/2 - 0.05
        car_cx = x + CAR_CENTER_OFFSET * np.cos(yaw)
        car_cy = y + CAR_CENTER_OFFSET * np.sin(yaw)

        # Transform to bay frame
        dx = car_cx - self.bay_x
        dy = car_cy - self.bay_y
        cos_b, sin_b = np.cos(self.bay_yaw), np.sin(self.bay_yaw)

        along = dx * cos_b + dy * sin_b
        lateral = -dx * sin_b + dy * cos_b

        # Yaw error (wrapped to [-pi, pi])
        yaw_err = self._wrap_angle(self.bay_yaw - yaw)

        # Ray-cast to virtual obstacles
        dist_front, dist_left, dist_right = self.compute_distances(
            car_cx, car_cy, yaw
        )

        obs = np.array([
            along, lateral, yaw_err, v,
            dist_front, dist_left, dist_right
        ], dtype=np.float32)

        return obs

    def compute_distances(self, cx, cy, yaw):
        """Compute ray-cast distances to virtual obstacles."""
        # Simplified - use obstacle_manager logic from training
        # Returns: (front, left, right) distances
        return 5.0, 5.0, 5.0  # Max range if no obstacles

    def control_loop(self, event):
        """Main control loop at 10 Hz."""
        # Safety: check for stale state
        if self.last_state_time is None:
            return

        age = (rospy.Time.now() - self.last_state_time).to_sec()
        if age > self.stale_timeout:
            self._publish_stop("Stale state")
            return

        if not self.enable:
            return

        # Build observation
        obs = self.build_observation(self.last_state)

        # Check if already at goal (success stop)
        if self._is_at_goal(obs):
            self._publish_stop("At goal")
            return

        # Get action from policy
        action = self.policy.compute_single_action(obs, explore=False)

        # Convert and limit action
        steer, torque = self._convert_action(action)
        steer = np.clip(steer, -self.max_steer, self.max_steer)
        torque = np.clip(torque, -self.max_torque, self.max_torque)

        # Rate limiting
        steer, torque = self._apply_rate_limits(steer, torque)

        # Publish
        cmd = car_input()
        cmd.header.stamp = rospy.Time.now()
        cmd.steer = steer
        cmd.torque = torque
        cmd.velocity = float('nan')  # Use torque mode
        self.cmd_pub.publish(cmd)

        self.last_action = np.array([steer, torque])

    def _convert_action(self, action):
        """Convert normalized action to physical values."""
        # Action from policy: [steer, accel] in [-1, 1]
        # Training: steer -> [-0.523, 0.523], accel -> [-1.0, 1.0]

        # Scale for real car
        STEER_SCALE = 0.35 / 0.523  # Real max / Training max
        ACCEL_SCALE = 0.5           # Conservative initially

        steer = action[0] * 0.523 * STEER_SCALE

        # Convert acceleration to torque (simplified)
        # Real car uses torque, training uses acceleration
        accel = action[1] * 1.0 * ACCEL_SCALE
        torque = self._accel_to_torque(accel)

        return steer, torque

    def _accel_to_torque(self, accel):
        """Convert acceleration to motor torque."""
        # Simplified linear mapping - tune based on car response
        # ChronosCar: Cm1, Cm2, Cd from model_params.yaml
        return accel * 0.2  # Initial conservative gain

    def _is_at_goal(self, obs):
        """Check if car is within tolerances."""
        along, lateral, yaw_err, v = obs[:4]
        return (abs(along) < self.tol_along and
                abs(lateral) < self.tol_lat and
                abs(yaw_err) < self.tol_yaw and
                abs(v) < 0.05)

    def _apply_rate_limits(self, steer, torque):
        """Limit rate of change."""
        MAX_STEER_RATE = 0.3  # rad/step
        MAX_TORQUE_RATE = 0.1  # torque/step

        d_steer = steer - self.last_action[0]
        d_torque = torque - self.last_action[1]

        steer = self.last_action[0] + np.clip(d_steer, -MAX_STEER_RATE, MAX_STEER_RATE)
        torque = self.last_action[1] + np.clip(d_torque, -MAX_TORQUE_RATE, MAX_TORQUE_RATE)

        return steer, torque

    def _publish_stop(self, reason):
        """Publish zero command."""
        cmd = car_input()
        cmd.header.stamp = rospy.Time.now()
        cmd.steer = 0.0
        cmd.torque = 0.0
        cmd.velocity = float('nan')
        self.cmd_pub.publish(cmd)

    @staticmethod
    def _wrap_angle(angle):
        """Wrap angle to [-pi, pi]."""
        return (angle + np.pi) % (2 * np.pi) - np.pi

if __name__ == '__main__':
    try:
        node = RLParkingNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
```

---

## Phase 7: Testing Sequence

### 7.1 Ghost Mode (No Publishing)

```bash
rosrun your_pkg rl_parking_node.py \
    _enable:=false \
    _checkpoint:=/path/to/checkpoint
```

**Verify:**
- [ ] Observation values update when moving car by hand
- [ ] `along`, `lateral` near 0 when at bay center
- [ ] `yaw_err` near 0 when aligned
- [ ] Distance sensors respond to virtual obstacles

### 7.2 Wheels Off Ground

1. Prop car up so wheels spin freely
2. Enable with low limits:
```bash
_enable:=true
_safety/max_torque:=0.05
_safety/max_steer:=0.20
```

**Verify:**
- [ ] Steering direction correct (left command = wheels turn left)
- [ ] Torque direction correct (forward = forward)
- [ ] If reversed, add sign flip in `_convert_action`

### 7.3 Floor Test (No Neighbors)

1. Set virtual neighbors far away
2. Place car ~1m from bay
3. Enable with conservative limits

**Verify:**
- [ ] Car moves toward bay
- [ ] Steering reasonable (no wild oscillations)
- [ ] Car slows near goal

### 7.4 Full Test with Virtual Neighbors

1. Enable virtual neighbor obstacles
2. Test parking maneuver

**Verify:**
- [ ] Car avoids "hitting" virtual neighbors
- [ ] Parks within tolerance

---

## Phase 8: Tuning & Calibration

### 8.1 Observation Scaling

If car overshoots or undershoots:

```python
# Increase to make policy "see" larger distances
OBS_SCALE = 2.78

# Decrease if car is too aggressive
OBS_SCALE = 2.0
```

### 8.2 Action Scaling

If car turns too sharply or too slowly:

```python
STEER_SCALE = 0.5   # Reduce for less aggressive steering
ACCEL_SCALE = 0.3   # Reduce for gentler acceleration
```

### 8.3 Torque Mapping

Tune `_accel_to_torque` based on:
- ChronosCar `Cm1`, `Cm2`, `Cd` parameters
- Observed car response

---

## Phase 9: Logging & Reproducibility

### 9.1 Launch with Recording

```bash
roslaunch crs_launch run_single_car.launch \
    namespace:=BEN_CAR_WIFI \
    experiment_name:=real_world_rl \
    copy_config:=true \
    record_bag:=true
```

### 9.2 Deployment Package

Keep in version control:
```
deployment_v1/
├── checkpoint/
├── configs/
│   ├── deploy_env.yaml
│   ├── parking_scene_real.yaml
│   └── scaling_params.yaml
├── nodes/
│   └── rl_parking_node.py
├── bags/
│   └── test_run_001.bag
└── README_DEPLOY.md
```

---

## Phase 10: Optional Fine-Tuning

If scale compensation doesn't work well:

### 10.1 Update config_env.yaml

```yaml
vehicle:
  L: 0.13           # ChronosCar real length
  W: 0.09           # Real width
  wheelbase: 0.09   # Real wheelbase
  max_steer: 0.35   # Real steering limit
  max_vel: 1.0
  max_acc: 1.0

# Scale bay/spawn accordingly
```

### 10.2 Fine-Tune from Checkpoint

```bash
./quick_train.sh finetune \
    --checkpoint path/to/best_checkpoint \
    --phase phase5_neighbor_jitter \
    --timesteps 5000000
```

---

## Quick Reference

### Essential Topics

| Topic | Type | Purpose |
|-------|------|---------|
| `/<ns>/estimation_node/best_state` | `car_state_cart` | Car pose/velocity |
| `/<ns>/control_input` | `car_input` | Torque + steering command |

### Essential Commands

```bash
# Start ROS
roscore

# Start motion capture
roslaunch natnet_ros_cpp natnet_ros.launch

# Start car stack (idle controller)
roslaunch crs_launch run_single_car.launch \
    namespace:=BEN_CAR_WIFI \
    experiment_name:=real_world_rl

# Start RL node
rosrun your_pkg rl_parking_node.py \
    _enable:=true \
    _checkpoint:=/path/to/checkpoint \
    _bay/center_x:=0.0 \
    _bay/center_y:=0.0 \
    _bay/yaw:=0.0

# Kill all
pkill -f roslaunch
ray stop
```

---

## Checklist Summary

### Pre-Deployment
- [ ] Best checkpoint identified and copied
- [ ] deploy_env.yaml created
- [ ] parking_scene_real.yaml with measured bay pose
- [ ] rl_parking_node.py implemented
- [ ] Motion capture calibrated

### Hardware Validation
- [ ] Car WiFi connected
- [ ] State topic publishing
- [ ] Estimator subscription fixed
- [ ] Idle controller running

### Testing
- [ ] Ghost mode passed
- [ ] Wheels-off test passed
- [ ] Floor test (no neighbors) passed
- [ ] Full test with virtual neighbors passed

### Production
- [ ] Scaling factors tuned
- [ ] Safety limits finalized
- [ ] Logging enabled
- [ ] Deployment package versioned

---

**Next Action**: Start with Phase 0 - identify your best checkpoint and create the deployment folder structure.
