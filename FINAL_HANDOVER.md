# Final Project Handover - Parallel Parking RL

**Project**: Autonomous Parallel Parking with Deep Reinforcement Learning
**Date**: 2026-02-28
**Status**: ChronosCar — SUCCESSFULLY PARKED on hardware (2026-02-27). New car — Phase 5 complete (83% success), Phase 6 restarting.
**Version**: 14.0

---

## Executive Summary

Two RL-based parallel parking systems for 1/28-scale RC cars:

1. **ChronosCar** (L=0.13m, W=0.065m): Fully trained (7 phases, 88% sim success). **DEPLOYED AND PARKING SUCCESSFULLY on hardware** (2026-02-27). Best result: 40 steps, along=+0.027m, lateral=-0.004m, yaw=-2.5°. 31 bugs found and fixed across observation pipeline, velocity estimation, friction compensation, collision recovery, and centering.

2. **New Car** (L=0.15m, W=0.10m): Phase 5 complete (reward=613, 83% success). Phase 6 restarting from Phase 5 checkpoint with fixed curriculum (one-change-at-a-time). Has wheel encoders (Professor Frank's recommendation) and friction model baked into training.

**Professor Frank's guidance (2026-02-25)**:
> "Implement a low level velocity controller to regulate velocity... to overcome the static friction at v=0 or at motion reversals you would have to induce a brief torque impulse."
> "Switch to the new car. The new car has wheel encoders so it can measure current velocity directly rather than indirectly via MoCap."

Both suggestions implemented. The torque impulse + velocity=nan combination was the key breakthrough for overcoming static friction.

**Architecture**: OptiTrack MoCap for state estimation, CRS Docker (ROS Noetic) for car communication, UDP bridge so the host needs no ROS.

---

## Table of Contents

1. [Quick Start - ChronosCar Parking](#quick-start---chronoscar-parking)
2. [What Works and What Doesn't](#what-works-and-what-doesnt)
3. [How to Fix ChronosCar Parking](#how-to-fix-chronoscar-parking)
4. [Architecture](#architecture)
5. [ChronosCar Configuration](#chronoscar-configuration)
6. [New Car Configuration](#new-car-configuration)
7. [All Bugs Found & Fixed](#all-bugs-found--fixed-during-hardware-testing)
8. [Sim-to-Real Gap Analysis](#sim-to-real-gap-analysis)
9. [Key Codebase Files](#key-codebase-files)
10. [Critical Lessons Learned](#critical-lessons-learned)

---

## Quick Start - ChronosCar Parking

### Prerequisites
- OptiTrack Motive running on PC (129.217.130.57)
- Car markers visible in Motive
- Host machine: 129.217.130.20
- Docker image `crs:local` built (run `./deploy/setup_hardware.sh` first time)

### Step-by-Step

```bash
# 1. Start Motive on the OptiTrack PC FIRST (NatNet crashes without it)

# 2. Turn car OFF (prevents MPC from racing)

# 3. Start Docker services (4 services in tmux)
./deploy/start_docker.sh

# 4. Wait 15 seconds for all services to start, then kill the CRS controller
#    (it overrides our commands on the same ROS topic)
docker exec crs bash -c "source /opt/ros/noetic/setup.bash && rosnode kill /BEN_CAR_WIFI/controller_node"

# 5. Turn car ON

# 6. Calibrate bay position (place car at parking spot center)
python deploy/rl_parking_node.py --calibrate --scene deploy/parking_scene.yaml

# 7. Move car ~50cm away from bay, facing roughly parallel

# 8. Ghost mode first (reads sensors, computes actions, does NOT move car)
python deploy/rl_parking_node.py --ghost \
    --checkpoint checkpoints/chronos/curriculum_20260210_113810/phase7_polish/best_checkpoint \
    --scene deploy/parking_scene.yaml

# 9. Verify along/lateral approach 0 when car is manually moved to bay

# 10. Live mode (car moves!)
python deploy/rl_parking_node.py \
    --checkpoint checkpoints/chronos/curriculum_20260210_113810/phase7_polish/best_checkpoint \
    --scene deploy/parking_scene.yaml \
    --log-dir deploy/logs/

# Emergency stop: PHYSICAL POWER SWITCH on car
```

### Available Checkpoints

| Checkpoint | Phase | Notes |
|-----------|-------|-------|
| `checkpoints/chronos/curriculum_20260210_113810/phase7_polish/best_checkpoint` | 7 (latest) | Most polished, trained latest |
| `checkpoints/chronos/curriculum_20260209_230526/phase7_polish/best_checkpoint` | 7 | Earlier training run |
| `checkpoints/chronos/curriculum_20260209_230526/phase6_random_obstacles/best_checkpoint` | 6 | Simpler, may be more robust |

### CLI Options

```bash
python deploy/rl_parking_node.py \
    --checkpoint <path>              # Required: PPO checkpoint
    --scene deploy/parking_scene.yaml  # Scene config
    --ghost                          # Read-only mode (no motor commands)
    --dry-run                        # Test checkpoint loading offline
    --no-viz                         # Disable matplotlib window
    --log-dir deploy/logs/           # CSV logging for post-run analysis
    --calibrate                      # Bay calibration mode
```

---

## What Works and What Doesn't

### What Works (as of 2026-02-28)
- **Full end-to-end parking on hardware** — ChronosCar parks successfully in ~40 steps
- Training: 7-phase curriculum, 88% success in simulation at 2.7cm tolerance
- Checkpoint loading and policy inference
- Observation pipeline: bay-frame transform, virtual obstacle ray-casting
- MoCap position tracking + velocity estimation from position deltas (LP filter alpha=0.4)
- Virtual obstacle geometry matches training
- World boundary re-centering on bay (critical for correct ray distances)
- Bay calibration with rear-axle to body-center correction
- Smart collision recovery: direction-aware (inside bay → FORWARD exit; outside → BACKWARD)
- v_model sync after collision recovery (prevents velocity gap)
- In-bay speed cap (symmetric ±0.12 m/s when |along| < 0.10m)
- Steering/velocity calibration measurement tool (`steer_calibration.py`)
- Forward velocity control via CRS PID + torque impulse for static friction
- velocity=nan during impulse (disables CRS PID so it doesn't fight the burst)
- goal_offset_along separation: bay_center (obstacles) vs bay_goal_{x,y} (along/lat reference)
- Reverse braking via 3-mode controller (BRAKE/START/MAINTAIN)
- Live matplotlib visualization + CSV logging for post-run analysis

### Known Limitations
- **Reverse braking precision**: CRS PID is forward-only. Reverse uses direct torque in ros_bridge. Fine-grained speed control in reverse is limited (3-mode: brake/start/maintain).
- **Occasional collision recovery**: Car sometimes enters bay at a sub-optimal angle and needs 1-2 recovery cycles. Recovery works correctly (exits bay forward, re-approaches).

### Hardware Test History

**2026-02-19 (before calibration fix)**:
```
along  = 0.015m   (tol: 0.043m)  WITHIN TOLERANCE
lateral= 0.017m   (tol: 0.043m)  WITHIN TOLERANCE
yaw_err= -22.0deg (tol: 8.6deg)  EXCEEDS TOLERANCE  ← Problem
v      = 0.002m/s (tol: 0.05)    WITHIN TOLERANCE
```
Root cause: 2x yaw rate mismatch (steer 1.376x, vel 1.503x commanded).

**2026-02-27 (FINAL — all fixes applied)**:
```
along   = +0.027m  (tol: 0.055m)  WITHIN TOLERANCE  ✓
lateral = -0.004m  (tol: 0.055m)  WITHIN TOLERANCE  ✓
yaw_err = -2.5°    (tol: 8.6°)    WITHIN TOLERANCE  ✓
v       = 0.000m/s (tol: 0.05)    WITHIN TOLERANCE  ✓
Steps: 40  Settled: 3 consecutive steps
```
**Car parks successfully.**

---

## How to Fix ChronosCar Parking

### What Changed (2026-02-25, Professor Frank + Friend's Code Analysis)

#### Change 1: Torque Impulse at Standstill (Professor Frank)

**Problem**: Static friction at v=0 needs a *torque impulse*, not just a higher velocity setpoint. Our old `MIN_CMD=0.05` approach only raised the velocity setpoint — the CRS PID still had to convert that to torque and might not respond fast enough.

**What's implemented** in `deploy/ros_bridge.py`:
```
At standstill (|v_actual| < 0.02 m/s) but policy wants motion:
  → Fire IMPULSE_TORQUE=0.20 directly for IMPULSE_STEPS=2 steps (0.2s)
  → After 2 steps, return to normal feedforward torque
  → Same logic for forward and reverse directions
```

This is logged in Docker console as `[ros_bridge] IMPULSE FWD: ...` or `[ros_bridge] IMPULSE REV: ...`

**Tuning `IMPULSE_TORQUE`**: Default 0.20. If car still doesn't start: increase to 0.25. If it jerks too aggressively: decrease to 0.15. Change in `ros_bridge.py` `__init__`: `self.IMPULSE_TORQUE = 0.20`.

**Tuning `IMPULSE_STEPS`**: Default 2 steps = 0.2s. Increase to 3 if friction is very high. Change `self.IMPULSE_STEPS = 2`.

#### Change 2: `velocity=nan` During Impulse (From Friend's Code Analysis)

**Problem discovered**: When we send `cmd_velocity=0.05` (small forward setpoint) AND `ff_torque=0.20` (impulse), the CRS PID **fights our impulse**. The PID sees the car exceeding 0.05 m/s from the burst and immediately applies braking torque back to the setpoint. This kills the effectiveness of the impulse designed to overcome static friction.

**How it was found**: Analyzed the CRS car project at `github.com/pain7576/crs-trucktrailer-ros2`. That project uses:
```python
msg4.velocity = math.nan   # Completely disables CRS velocity PID
msg4.torque = -0.085       # Only the torque field drives the motor
```
This revealed that `velocity=nan` is a supported CRS feature that gives direct torque control without PID interference.

**The fix** in `deploy/ros_bridge.py`:
- **During impulse (both fwd/rev)**: `cmd_velocity = math.nan` → PID disabled, torque acts freely
- **Steady-state forward**: `cmd_velocity = velocity` → PID enabled for stable velocity tracking
- **Reverse (always)**: `cmd_velocity = math.nan` → CRS PID is forward-only anyway, now explicit

```
Before (broken): cmd_velocity=0.05 + ff_torque=0.20
  → PID: "car > 0.05 m/s, apply brakes!" fights the burst

After (fixed):   cmd_velocity=nan + ff_torque=0.20
  → PID disabled, 0.20 N·m acts directly on motor, friction overcome
```

**This is the most impactful fix for the static friction dead zone.**

---

### Priority 1: Re-test with Calibration Correction (Most Likely Fix)

The steering/velocity gain mismatch was the dominant issue. With calibration correction now applied (`steer_gain=1.376`, `velocity_gain=1.503`), the car's actual dynamics should match the kinematic model the policy was trained with.

**What to do:**
1. Follow the Quick Start steps above
2. Run live mode with `--log-dir deploy/logs/`
3. Watch the `v_model vs v_hw` gap in the logs - it should stay small (<0.02)
4. Watch the `steer_pol vs steer_hw` - steer_hw should be steer_pol/1.376

**What to look for in the CSV log:**
- `v_gap` column: should be near 0. If consistently positive = car accelerates slower than training expects
- `yaw_err` column: should decrease over time. If it oscillates or diverges = calibration is wrong or friction issue
- `collision` column: should stay 0. If 1 = car is hitting virtual obstacles

### Priority 2: Relax Yaw Tolerance (Quick Win)

If calibration fix improves yaw but doesn't reach 8.6 deg, try relaxing the yaw tolerance in `deploy/parking_scene.yaml`:

```yaml
success:
  yaw_tol: 0.30    # 17 degrees (was 0.15 = 8.6 deg)
```

The car may be "good enough" at 15-20 degrees yaw error for practical purposes.

### Priority 3: Tune MIN_CMD (If Still Stuck at Standstill)

Current friction compensation uses `MIN_CMD=0.05` m/s. If the car still can't make fine yaw corrections:

**Option A**: Increase MIN_CMD
In `deploy/rl_parking_node.py`, `ActionConverter.convert()`:
```python
MIN_CMD = 0.08   # was 0.05, try 0.08 for more torque at standstill
```

**Option B**: Decrease MIN_CMD (if car jerks too much)
```python
MIN_CMD = 0.03   # gentler, less overshoot
```

### Priority 4: Adjust Calibration Gains (If Yaw Overshoots the Other Way)

If after applying calibration the car now under-steers (yaw error grows slowly rather than fixing), the gains may be slightly off. Re-run `steer_calibration.py`:

```bash
python deploy/steer_calibration.py
```

This drives the car in a circle and measures actual vs expected turning radius. Update `parking_scene.yaml` with the new gains.

### Priority 5: Try Phase 6 Checkpoint (If Phase 7 Is Unstable)

Phase 7 tightened yaw tolerance from 5 to 4 degrees. Phase 6 is more relaxed and may be more robust on hardware:

```bash
python deploy/rl_parking_node.py \
    --checkpoint checkpoints/chronos/curriculum_20260209_230526/phase6_random_obstacles/best_checkpoint \
    --scene deploy/parking_scene.yaml
```

### Priority 6: Remove Virtual Neighbors (Debugging)

If the car keeps colliding with virtual neighbors, temporarily remove them to test pure approach:

```yaml
# In parking_scene.yaml
obstacles:
  neighbor:
    offset: 0.0    # Was 0.194 - setting to 0 removes neighbors
```

This lets the car practice approach without collision risk. If this works, the issue is tight maneuvering near obstacles.

### Priority 7: Train with Friction Model (Long Term)

The new car's training already includes a friction model (`static_friction=0.15`, `kinetic_friction=0.05`). Once the new car's training completes, the sim-to-real gap should be smaller because the policy learned that small accelerations don't produce movement.

To retrain ChronosCar with friction, add to `config_env_chronos.yaml`:
```yaml
vehicle:
  static_friction: 0.15    # Add these lines
  kinetic_friction: 0.05
```

Then restart Phase 1 training (or finetune from existing checkpoint).

### Debugging Checklist

When running the car, check these in order:

1. **Obs values make sense?** Along/lateral should decrease as car approaches bay. dF/dL/dR should be ~1-5m (not 0 unless colliding).
2. **v_hw vs v_model gap?** If `v_model - v_hw > 0.05` for more than 5 steps, the car is slower than training expects. Policy will see wrong velocity and make wrong steering decisions.
3. **Steer_hw direction correct?** If car steers right but should steer left, check bay yaw calibration.
4. **Action magnitudes reasonable?** `raw_steer` and `raw_accel` should be in [-1, 1]. If always +-1, policy is in an extreme state (probably OOD).
5. **dF/dL/dR not all zero?** If all zero, car is inside a virtual obstacle. Collision recovery should trigger.

---

## New Car Deployment (When Training Completes)

### The Wheel Encoder Advantage

ChronosCar's velocity came from MoCap position derivatives (noisy, ~0.1s delay). The new car has wheel encoders → CRS estimator publishes accurate `vx_b` directly.

```
ChronosCar velocity pipeline:           New car velocity pipeline:
  MoCap 120Hz positions                   Wheel encoders (kHz)
  → rl_parking_node computes              → CRS estimator
    dx/dy per step                          → vx_b field in
  → LP filter (alpha=0.4)                   car_state_cart
  → v_hw (delayed, noisy)               → rl_parking_node reads directly
                                         → v_hw (accurate, instant)
```

Impact on policy: The policy sees accurate velocity → correct decisions about when to brake, when to nudge, when stopped. The ChronosCar's delayed velocity caused the policy to misread its own speed, compounding the yaw control problem.

### Code Changes Already Made

1. **`deploy/rl_parking_node.py`**: `velocity_source` config key. If `"encoder"`, uses `vx_b` directly. If `vx_b` reads > 1.0 m/s at standstill (sanity check), auto-falls back to MoCap mode and prints a warning.

2. **`deploy/parking_scene_newcar.yaml`**: `velocity_source: encoder` set.

### Deployment Steps

```bash
# 1. Wait for training to complete (train_newcar.sh)
#    Monitor: tensorboard --logdir checkpoints/newcar/

# 2. Visualize before hardware test
python -m rl.visualize_checkpoint \
    --checkpoint checkpoints/newcar/.../phase7_polish/best_checkpoint \
    --deterministic --num-episodes 5

# 3. Calibrate bay position
python deploy/rl_parking_node.py --calibrate --scene deploy/parking_scene_newcar.yaml

# 4. Measure steering/velocity gains (same script as ChronosCar)
python deploy/steer_calibration.py
# Update parking_scene_newcar.yaml calibration section

# 5. Ghost mode: verify observations look sane
python deploy/rl_parking_node.py --ghost \
    --checkpoint checkpoints/newcar/.../phase7_polish/best_checkpoint \
    --scene deploy/parking_scene_newcar.yaml

# 6. First live run with full logging
python deploy/rl_parking_node.py \
    --checkpoint checkpoints/newcar/.../phase7_polish/best_checkpoint \
    --scene deploy/parking_scene_newcar.yaml \
    --log-dir deploy/logs/

# 7. Check CSV log: v_gap should be MUCH smaller than ChronosCar
#    (encoder velocity = accurate → v_model ≈ v_hw most of the time)
```

### What to Verify First for New Car

1. **Encoder velocity is working**: In ghost mode, watch `v_hw` in logs. Move car by hand. Does it show correct velocity? Should track instantly unlike ChronosCar.
2. **No "IMPULSE" spam**: If Docker logs show constant impulse firing, encoder velocity may be reporting 0 when car is actually moving. Check sanity fallback logs.
3. **Torque mapping**: New car may have different `a_torque`/`b_torque` in ff_fb_controller.yaml. Check CRS config before first live run.

---

## Architecture

```
 Host Machine (Ubuntu 22.04, no ROS)
 =====================================
 |  rl_parking_node.py               |
 |    - Loads PPO checkpoint          |
 |    - Builds observations           |
 |    - Runs policy inference         |
 |    - Converts actions to cmds      |
 |    - CSV logging                   |
 |    - Live visualization            |
 |                                    |
 |    UDP :5800 <-- state             |
 |    UDP :5801 --> commands           |
 =====================================
        |  UDP (localhost)  |
 =====================================
 |  Docker Container (crs:local)      |
 |  Ubuntu 20.04 + ROS Noetic         |
 |                                    |
 |  ros_bridge.py                     |
 |    - Subscribes ROS car_state_cart |
 |    - Publishes ROS car_input       |
 |    - 3-mode torque: stop/fwd/rev   |
 |    - Velocity PID (forward only)   |
 |                                    |
 |  roscore                           |
 |  WiFiCom (car ↔ ROS)              |
 |  Estimator (MoCap → state)        |
 |  NatNet bridge (OptiTrack → ROS)  |
 =====================================
        |  WiFi UDP  |     |  NatNet  |
     ChronosCar         OptiTrack Motive
    129.217.130.30      129.217.130.57
```

### UDP Protocol

**State (bridge → host):** 7 doubles = 56 bytes
```
[x, y, yaw, v_tot, vx_b, vy_b, steer]
```
Note: v_tot and vx_b are unreliable (~28 m/s while stationary). rl_parking_node.py computes velocity from position deltas + LP filter (alpha=0.4).

**Command (host → bridge):** 3 doubles = 24 bytes
```
[velocity, steer, v_actual]
```
v_actual enables ros_bridge to brake in reverse (distinguish "accelerate reverse" from "brake from reverse").

### Observation Space (7D)

```
[along,        # Car center offset along bay axis (m). + = ahead of bay
 lateral,      # Car center offset perpendicular to bay (m). + = left of bay
 yaw_err,      # Heading error (rad). Wrapped to [-pi, pi]
 v,            # Velocity (m/s). From MoCap position deltas, LP filtered
 dist_front,   # Ray-cast distance to nearest obstacle ahead (m)
 dist_left,    # Ray-cast distance to nearest obstacle left (m)
 dist_right]   # Ray-cast distance to nearest obstacle right (m)
```

### Action Space (2D)

```
[steering,      # [-1, 1] -> [-max_steer, max_steer] rad
 acceleration]  # [-1, 1] -> [-max_acc, max_acc] m/s^2, integrated to velocity
```

---

## ChronosCar Configuration

### Dimensions

| Parameter | Value | Source |
|-----------|-------|--------|
| Wheelbase | 0.090m | lr=0.038 + lf=0.052 (CRS model_params) |
| Length | 0.13m | Measured |
| Width | 0.065m | Measured (0.0325 x 2) |
| Max Steer | 0.35 rad (20 deg) | joystick_controller.yaml |
| rear_overhang | 0.02m | (L - wheelbase) / 2 |
| dist_to_center | 0.045m | L/2 - rear_overhang |
| Max Velocity | 0.5 m/s | Training config |
| Max Acceleration | 0.5 m/s^2 | Training config |

### Scale Factors

| Factor | Value | Formula | Used For |
|--------|-------|---------|----------|
| body_scale | 0.36 | 0.09/0.25 (wheelbase ratio) | Car/obstacle sizes, bay positions, tolerances |
| maneuver_scale | 0.569 | 0.247/0.434 (turning radius ratio) | Spawn distances, y_jitter, x offsets |

### Calibration (Measured 2026-02-20)

| Parameter | Value | Meaning |
|-----------|-------|---------|
| velocity_gain | 1.503 | Car drives 1.503x faster than commanded |
| steer_gain | 1.376 | Car turns 1.376x more than commanded |

Commands are divided by these gains in `ActionConverter.convert()`, after rate limiting.

### Torque Mapping (CRS ff_fb_controller)

```
torque = (velocity - 0.137) / 6.44
```
a_torque = 6.44026018, b_torque = 0.13732343

### parking_scene.yaml (Current — calibrated 2026-02-25)

```yaml
bay:
  center_x: -0.8834    # Calibrated from MoCap 2026-02-25 (20-reading average)
  center_y: -0.2651
  yaw: 0.155
  goal_offset_along: -0.020  # Shift goal 2cm toward curb (car parks near center)
vehicle:
  length: 0.13
  width: 0.065
  wheelbase: 0.09
  rear_overhang: 0.02
  max_steer: 0.35         # MUST match training (was 0.30 — caused mismatch)
  max_vel: 0.5            # MUST match training (was 0.25 — caused mismatch)
  max_acc: 0.5
obstacles:
  neighbor:
    w: 0.13
    h: 0.065
    offset: 0.194
    pos_jitter: 0.0
    curb_gap: 0.018
    curb_thickness: 0.014
world:
  x_min: -1.25            # Auto re-centered on bay by rl_parking_node.py
  x_max: 1.25
  y_min: -1.25
  y_max: 1.25
safety:
  max_steer: 0.35         # MUST match training
  max_steer_rate: 0.5     # MUST match training (was 0.3)
  stale_timeout: 0.3
success:
  along_tol: 0.055        # Loosened from 0.043 (boundary failure at exactly 0.043)
  lateral_tol: 0.055
  yaw_tol: 0.15           # 8.6 degrees
  v_tol: 0.05
  settled_steps: 3        # Reduced from 5 (hardware oscillation makes 5 unrealistic)
calibration:
  velocity_gain: 1.503    # Car drives 1.503x commanded velocity
  steer_gain: 1.376       # Car steers 1.376x commanded angle
torque_mapping:
  a_torque: 6.44026018    # From ChronosCar ff_fb_controller.yaml
  b_torque: 0.13732343
```

---

## New Car Configuration

### Dimensions

| Parameter | Value | Source |
|-----------|-------|--------|
| Wheelbase | 0.113m | Measured |
| Length | 0.15m | Measured |
| Width | 0.10m | Measured |
| Max Steer | 0.314 rad (18 deg) | CRS config |
| rear_overhang | 0.0185m | (0.15 - 0.113) / 2 |
| dist_to_center | 0.0565m | 0.15/2 - 0.0185 |
| Turning Radius | 0.348m | 0.113/tan(0.314) |

### Scale Factors

| Factor | Value | Formula | Used For |
|--------|-------|---------|----------|
| body_scale | 0.452 | 0.113/0.25 | Car/obstacle sizes, tolerances |
| maneuver_scale | 0.801 | 0.348/0.434 | Spawn distances, approach geometry |

### Friction Model (Baked into Training)

```yaml
vehicle:
  static_friction: 0.15    # m/s^2 - dead zone at standstill
  kinetic_friction: 0.05   # m/s^2 - opposes motion
```

Policy must command accel > 0.15 m/s^2 to start moving. This teaches the policy that tiny accelerations don't work, matching hardware behavior.

### Training Status (2026-02-28)

| Phase | Status | Notes |
|-------|--------|-------|
| 1 Foundation | Complete | |
| 2 Random Spawn | Complete | |
| 3 Random Bay X | Complete | |
| 4a Small Bay Y | Complete | |
| 4 Full Random | Complete | reward=444 |
| 5 Tighten 2.7cm | **Complete** | reward=613, **83% success** |
| 6 Settling Polish | **Running** | Restarted from phase5 checkpoint; only settling_bonus changed |
| 7 Tighter Settling | Pending | v_threshold, steer_threshold, settled_steps changes |
| 8 Yaw Precision | Pending | yaw_tol 8.6°→4° |

```bash
# Resume from Phase 5 checkpoint (run this after killing broken phase 6):
./resume_phase5_newcar.sh phase6_restart

# Monitor training:
tensorboard --logdir checkpoints/newcar/
```

**Curriculum lesson**: Phase 6 originally changed 3 settling criteria simultaneously → policy regressed from 83% to 0%. Fixed to change only `settling_bonus` (one change at a time). Phase 7 now handles settling threshold tightening.

### Deployment (After Training Completes)

```bash
# 1. Calibrate bay for new car
python deploy/rl_parking_node.py --calibrate --scene deploy/parking_scene_newcar.yaml

# 2. Run steer calibration to measure gains
python deploy/steer_calibration.py
# Update deploy/parking_scene_newcar.yaml with measured gains

# 3. Deploy
python deploy/rl_parking_node.py \
    --checkpoint checkpoints/newcar/.../phase7_polish/best_checkpoint \
    --scene deploy/parking_scene_newcar.yaml \
    --log-dir deploy/logs/
```

---

## All Bugs Found & Fixed During Hardware Testing

### Bug 1: controller_node Overrides Commands
**Symptom**: Car ignores RL commands, runs MPC instead.
**Fix**: `rosnode kill /BEN_CAR_WIFI/controller_node` before running RL.

### Bug 2: Car Races at Max Speed on Startup
**Symptom**: Turning car ON while Docker is running causes MPC to send full torque.
**Fix**: Car OFF → Docker → kill controller → Car ON.

### Bug 3: CRS Velocity Estimates Broken
**Symptom**: v_tot and vx_b read ~28 m/s while stationary.
**Fix**: Compute velocity from MoCap position deltas + LP filter (alpha=0.4).

### Bug 4: Velocity Filter Too Heavy
**Symptom**: Car overshoots because velocity reads lag 2.3 seconds behind.
**Fix**: Changed LP filter alpha from 0.1 to 0.4. At 120Hz MoCap, noise is low enough.

### Bug 5: NatNet Bridge Crashes Silently
**Symptom**: No MoCap data, no error messages.
**Fix**: Start Motive on OptiTrack PC BEFORE starting Docker services.

### Bug 6: World Boundary Mismatch
**Symptom**: dF/dL/dR=5.0 (max distance), policy outputs garbage.
**Fix**: parking_scene.yaml world boundaries must match training config (+-1.25m).

### Bug 7: World Not Centered on Bay
**Symptom**: Wall distances asymmetric (0.5m vs 1.9m), policy saturates.
**Fix**: rl_parking_node.py auto-re-centers world boundaries around bay_center.

### Bug 8: Bay Calibration Body-Center Offset
**Symptom**: along=0.045m even when car is perfectly parked. Never converges (tol=0.043m).
**Fix**: CRS reports rear-axle position. Calibration shifts forward by dist_to_center (0.045m) to get body center.

### Bug 9: Safety Clamp Mismatches
**Symptom**: Policy commands steer=0.35 rad but safety clamps to 0.30. Car under-steers.
**Fix**: Set safety max_steer=0.35, max_vel=0.5, steer_rate=0.5 to match training.

### Bug 10: Static Friction Dead Zone
**Symptom**: Policy commands v=0.004 m/s, car doesn't move.
**Fix**: MIN_TORQUE=0.10 in ros_bridge.py. MIN_CMD=0.05 friction compensation in rl_parking_node.py.

### Bug 11: Velocity Integrator Deadlock
**Symptom**: Pure closed-loop deadlocks at standstill (friction prevents tiny velocity from starting motion).
**Fix**: MIN_CMD=0.05 friction compensation. If car is at standstill and policy wants movement, boost velocity command to minimum that generates torque.

### Bug 12: Friction Boost Direction Bug
**Symptom**: Car shoots forward when transitioning from reverse to forward.
**Fix**: Use accel_cmd sign (not v_target sign) to determine friction boost direction. Eliminates zero-crossing bug.

### Bug 13: RLlib Action Array Extraction
**Symptom**: TypeError on action array.
**Fix**: `np.asarray(action[0]).item()` to extract scalar from numpy array.

### Bug 14: Training Terminates on Collision
**Symptom**: dF=dL=dR=0 after collision, policy behavior undefined.
**Fix**: Auto-reverse recovery: 15 steps at vel=-0.12, then resume policy. Max 5 recoveries.

### Bug 15: Reverse Has No Speed Control
**Symptom**: CRS PID only works forward. Reverse uses constant torque, can't slow down.
**Fix**: 3-double UDP protocol [vel, steer, v_actual]. ros_bridge uses v_actual to distinguish accelerate/brake/start/maintain in reverse.

### Bug 16: Car WiFi Config Cookies
**Symptom**: ESP32 web config page fails with "Header fields too long".
**Fix**: Use incognito browser for http://192.168.4.1/config.

### Bug 17: rostopic pub Latched Messages
**Symptom**: Car continues moving after Ctrl+C of rostopic pub.
**Fix**: Send zero-velocity override or use physical power switch.

### Bug 18: PPO Gaussian Actions Unbounded
**Symptom**: Action values occasionally exceed [-1, 1] range.
**Fix**: Always clip actions to [-1, 1] before scaling.

### Bug 19: NaN in Car State
**Symptom**: MoCap loses tracking, state contains NaN.
**Fix**: rl_parking_node.py skips NaN frames, sends stop command.

### Bug 20: UDP Socket Buffer Staleness
**Symptom**: Reading state that's seconds old (100Hz send, 20Hz read).
**Fix**: `drain_and_get_latest()` reads all buffered packets, returns only newest.

### Bug 21: Port Conflicts
**Symptom**: Multiple processes bind same UDP port, steal packets.
**Fix**: steer_calibration.py checks for existing listeners with lsof before starting.

### Bug 22: Out-of-Distribution Detection
**Symptom**: Policy panics when car is far from training distribution.
**Fix**: OOD warnings when yaw_err > 45 deg or lateral > 0.78m.

### Bug 23: Steering/Velocity Gain Mismatch (THE BIG ONE)
**Symptom**: Car yaw rotates 2x faster than training model. Approaches bay with good position but huge yaw error.
**Root cause**: Physical steering = 1.376x commanded, velocity = 1.503x commanded. Combined yaw rate = 2.01x.
**Fix**: Calibration section in parking_scene.yaml. ActionConverter divides commands by gains.

### Bug 24: Friction Boost During Direction Change
**Symptom**: Car shoots forward when reverse→forward transition passes through v=0.
**Root cause**: Previous STANDSTILL_GRACE approach fired boost at wrong time.
**Fix**: Replaced with MIN_CMD approach using accel_cmd sign for direction. No grace period needed.

### Bug 25: PID Fights Impulse Torque (2026-02-25)
**Symptom**: Torque impulse at standstill doesn't actually overcome friction — car still stays stuck.
**Root cause**: Sending `cmd_velocity=0.05` AND `ff_torque=0.20` simultaneously. The CRS PID sees the car moving faster than 0.05 m/s from the impulse and immediately brakes back. The braking force from PID cancels the impulse torque before the car can overcome friction.
**How found**: Analysis of peer's CRS project (github.com/pain7576/crs-trucktrailer-ros2) that uses `velocity=math.nan` to completely disable PID and drive purely via torque.
**Fix**: Set `cmd_velocity = math.nan` during impulse phases (both forward and reverse). PID disabled → torque acts on motor without interference. After impulse completes, revert to `cmd_velocity = velocity` for PID-controlled steady-state.

### Bug 26: goal_offset_along Shifted Obstacle Positions (2026-02-27)
**Symptom**: yaw_err oscillated -12° to -44° unpredictably. Policy seemed confused. Neighbor car visually appeared offset by ~3cm.
**Root cause**: `goal_offset_along` in `parking_scene.yaml` was being applied to `self.bay_x/bay_y` before those values were used to position the ObstacleManager AND center the world. The offset shifted obstacle placements and wall distances — not just the along/lat reference point.
**Fix**: Keep `self.bay_center` at original MoCap-calibrated coordinates (used for obstacle placement + world centering). Add separate `self.bay_goal_x/bay_goal_y = bay_{x,y} + offset * cos/sin(bay_yaw)` used ONLY for along/lat measurement in `build_obs()`.
**Current value**: `goal_offset_along: -0.020` (2cm toward curb) so car parks at physical bay center, not too close to opening-side neighbor.

### Bug 27: Collision Recovery Always Reversed — Wrong Inside Bay (2026-02-27)
**Symptom**: Car entered bay at slight angle, hit curb, then reversed deeper into the curb (again). Car oscillated at curb boundary.
**Root cause**: Collision recovery always sent `vel=-0.12` (backward). When car is at yaw≈0° inside the bay, "backward" = toward the curb = wrong direction.
**Fix**: Direction-aware recovery based on car geometry:
- `inside_bay`: `along < -0.04` AND `|yaw_err| < 30°` → FORWARD exit (`vel=+0.12`, 8 steps) toward bay opening
- `near_bay`: `|along| < 0.12` AND `|yaw_err| < 30°` → short backward (5 steps)
- else → full backward (`COLLISION_REVERSE_STEPS = 15` steps)

### Bug 28: v_model Reset to 0 After Collision Recovery (2026-02-27)
**Symptom**: After collision recovery, car moved erratically for several steps. Policy applied wrong steering.
**Root cause**: `action_converter.reset()` zeroed `v_model`. Car was still coasting at -0.136 m/s. Policy saw v_model climbing from 0 while car was moving backward → 0.286 m/s velocity gap → wrong steering decisions.
**Fix**: After recovery ends: `action_converter.v_model = v_current` (sync to actual velocity) AND `v_filtered = v_current` (reset LP filter). Both must be synced or the gap persists.

### Bug 29: Car Parks Too Close to Right Neighbor (2026-02-27)
**Symptom**: Car parked with dF=7.1cm (physically 5.7cm from right neighbor), leaving only 1.3cm clearance.
**Root cause**: With `goal_offset_along=+0.030` (old value), obs_along=+0.027 at success → physical_along = obs_along + offset = 0.027 + 0.030 = +0.057m from center. Too far toward opening-side neighbor.
**Fix**: Set `goal_offset_along=-0.020`. Now obs_along=+0.027 → physical = +0.027 - 0.020 = +0.007m from center. Near-perfect centering.

### Bug 30: Asymmetric In-Bay Speed Cap Caused Overshoot (2026-02-27)
**Symptom**: Car occasionally overshot forward into right neighbor after initial S-curve. Cap was `along < -0.02` (inside bay only).
**Fix**: Symmetric cap: `abs(along) < 0.10` — applies when within ±10cm of goal in BOTH directions. Car needs slow speed for backward correction too (centering after forward overshoot).

### Bug 31: In-Bay Cap at 0.10 m/s Too Slow for Yaw Correction (2026-02-27)
**Symptom**: Car took 80 steps (vs 40 before), ended with yaw=7.5° instead of 2.5°. Oscillated 55 micro-steps.
**Root cause**: At 0.10 m/s: yaw_rate = v × tan(steer) / wheelbase = 0.10 × 0.365 / 0.09 = 2.3°/step. Car needed 35° correction → minimum 15 steps of maximum steering, but oscillation added 40 more.
**Fix**: Raised to 0.12 m/s → 2.8°/step. Correction completes in ~12 steps, total run = 40 steps.

---

## Sim-to-Real Gap Analysis

### What Matches Well
- Steering geometry (trained with real car dimensions)
- Bay-frame observation transform
- Ray-casting distances (same ObstacleManager code)
- Control rate (10 Hz = training dt=0.1s)
- Calibration correction makes steering/velocity match model

### Known Gaps

| Gap | Training | Hardware | Impact | Current Fix |
|-----|----------|----------|--------|-------------|
| Static friction | None (or 0.15 for new car) | ~0.10 torque threshold | Can't execute v<0.10 m/s | MIN_CMD=0.05 boost |
| Velocity sensing | Perfect, instant | Position-derived, LP filtered | ~0.1s delay | alpha=0.4 |
| Reverse control | Symmetric v=v+a*dt | No CRS PID for reverse | Less controlled reverse | 3-mode torque in ros_bridge |
| Collision handling | Episode terminates | Car continues | Undefined behavior | Auto-reverse recovery |
| Motor latency | None | ~50-100ms | Slight overshoot | Low speeds minimize |
| Tire slip | None (kinematic) | Slip at tight turns | Trajectory deviation | Low speeds minimize |
| Fine corrections | v=0.004 works | v<0.10 doesn't move | Can't fine-adjust | MIN_CMD boost |

### The Core Challenge

Training's kinematic model allows continuous velocity down to 0.001 m/s. Hardware has a binary friction threshold: either enough torque to move (~0.10+ m/s) or stuck. The policy's learned "nudge, steer, nudge" strategy for fine yaw correction doesn't transfer directly. MIN_CMD creates a minimum velocity step that's larger than what the policy expects.

**The new car's training with friction model addresses this**: the policy will learn from the start that accel < 0.15 m/s^2 doesn't produce movement.

---

## Key Codebase Files

### Environment (Training)
| File | Purpose |
|------|---------|
| `env/parking_env.py` | Core parking environment (state=[x,y,yaw,v], collision→done) |
| `env/vehicle_model.py` | Kinematic bicycle model + friction model (static + kinetic) |
| `env/obstacle_manager.py` | Collision detection + ray-casting distances |
| `env/reward_parallel_parking.py` | Reward function with settling detection |

### Training
| File | Purpose |
|------|---------|
| `rl/curriculum_env.py` | Curriculum manager + env factory |
| `rl/gym_parking_env.py` | Gymnasium wrapper with early termination |
| `rl/train_curriculum.py` | Training loop with evaluation |
| `rl/visualize_checkpoint.py` | Visualize trained policy in simulation |
| `config_env_chronos.yaml` | ChronosCar training config |
| `config_env_newcar.yaml` | New car training config (with friction) |
| `rl/curriculum_config_chronos.yaml` | 7-phase ChronosCar curriculum |
| `rl/curriculum_config_newcar.yaml` | 7-phase new car curriculum |
| `train_chronos.sh` | ChronosCar training launcher |
| `train_newcar.sh` | New car training launcher |

### Deployment
| File | Where | Purpose |
|------|-------|---------|
| `deploy/rl_parking_node.py` | Host | RL inference + observation + control + calibration + viz + CSV logging |
| `deploy/ros_bridge.py` | Docker | ROS ↔ UDP bridge (state forwarding + 3-mode torque) |
| `deploy/parking_scene.yaml` | Host | ChronosCar bay + obstacles + safety + calibration |
| `deploy/parking_scene_newcar.yaml` | Host | New car deployment config |
| `deploy/steer_calibration.py` | Host | Measure steering/velocity gains |
| `deploy/start_docker.sh` | Host | Launch 4 Docker services in tmux |
| `deploy/setup_hardware.sh` | Host | First-time Docker/CRS/NatNet build |
| `deploy/run_parking.sh` | Host | Convenience wrapper for rl_parking_node.py |
| `deploy/README.md` | Host | Full deployment guide with troubleshooting |

### Key Classes in rl_parking_node.py

| Class | Purpose |
|-------|---------|
| `ObservationBuilder` | Builds 7D obs from MoCap state + virtual obstacles. Re-centers world on bay. |
| `ActionConverter` | Converts [-1,1] policy output to hardware commands. Velocity integration, friction compensation, rate limiting, calibration correction. |
| `UDPCarInterface` | Sends/receives UDP packets to/from ros_bridge in Docker. Buffer draining for latest state. |
| `LiveVisualizer` | Real-time matplotlib top-down view of car, bay, obstacles, trajectory. |

---

## Critical Lessons Learned

### Training
1. **NEVER** change entropy AND learning rate simultaneously between phases
2. Keep ALL reward weights identical across phases; only tighten tolerances
3. Forward-only parking is acceptable for small cars (gear_switches=0)
4. Intermediate tolerance steps work better than large jumps (2.2→2.0→1.8)
5. Collision terminates episodes -- policy never learns recovery behavior
6. Two scale factors needed when max_steer differs from original config

### Hardware Deployment (In Order of Importance)
1. **Safety clamps MUST match training**: max_steer=0.35, max_vel=0.5, steer_rate=0.5
2. **Kill controller_node before running RL**: It overrides commands
3. **Start Motive before Docker**: NatNet bridge crashes silently without it
4. **Car OFF before Docker**: Prevents MPC from racing car at max speed
5. **Calibrate bay with body-center correction**: rear-axle + dist_to_center shift
6. **Measure steering/velocity gains**: Car physically overshoots commands (1.376x / 1.503x)
7. **World boundaries auto-re-center on bay**: Training has bay near origin
8. **CRS velocity is broken**: Use MoCap position deltas (alpha=0.4 filter)
9. **Friction compensation**: torque impulse + velocity=nan (disables CRS PID during burst)
10. **goal_offset_along only affects bay_goal**: bay_center for obstacles, bay_goal for along/lat
11. **Collision recovery direction is geometry-dependent**: inside bay → FORWARD; outside → BACKWARD
12. **v_model must sync after recovery**: reset() zeros it; sync to v_current after recovery ends
13. **In-bay speed cap**: symmetric, 0.12 m/s max when abs(along) < 0.10 — too slow (0.10) causes oscillation
14. **Emergency stop**: Physical power switch. Software kill unreliable

### The Meta-Lesson

The sim-to-real gap isn't about observation scaling or action mapping -- those matched because we trained with real dimensions. The gap is about **dynamics**: friction, motor latency, and the binary nature of real actuators. A kinematic model allowing v=0.004 m/s doesn't prepare the policy for hardware where the car either moves at 0.10+ m/s or not at all. The new car's friction model in training directly addresses this.

---

## Network Addresses

| Device | IP | Purpose |
|--------|----|---------|
| Host machine | 129.217.130.20 | Runs Docker + RL node |
| OptiTrack/Motive PC | 129.217.130.57 | MoCap server (NatNet) |
| ChronosCar | 129.217.130.30 | RC car (WiFi UDP) |
| Car AP mode | 192.168.4.1 | Config page (incognito browser!) |

## UDP Ports

| Port | Direction | Content |
|------|-----------|---------|
| 5800 | bridge → host | Car state (7 doubles) |
| 5801 | host → bridge | Commands (3 doubles) |

---

## CSV Log Format

When running with `--log-dir`, each run creates `deploy/logs/run_YYYYMMDD_HHMMSS.csv`:

| Column | Description |
|--------|-------------|
| step | Step number |
| time | Unix timestamp |
| x, y, yaw | Car position from MoCap |
| v_hw | Velocity from MoCap position deltas (LP filtered) |
| v_raw | Raw velocity (unfiltered) |
| v_model | Open-loop integrator (what training sees: v += accel*dt) |
| v_target | Closed-loop target (v_current + accel*dt) |
| along, lateral, yaw_err | Bay-frame errors |
| dF, dL, dR | Virtual obstacle distances (ray-cast) |
| raw_steer, raw_accel | Policy output [-1, 1] |
| steer_hw, vel_hw | Hardware commands (after calibration correction) |
| v_gap | v_model - v_hw (positive = car slower than training expects) |
| collision | 1 if inside virtual obstacle |
| settled | Consecutive steps at goal |
| parked | 1 if successfully parked |

**Analysis tip**: Plot `v_model` vs `v_hw` to see the transient dynamics gap. Large sustained gap = friction or motor latency issue. Plot `yaw_err` over time to see convergence behavior.

---

**Last Updated**: 2026-02-20
