# Autonomous Parallel Parking with Deep Reinforcement Learning

**Status**: ChronosCar — SUCCESSFULLY PARKED on hardware. New car — training in progress (Phase 5 complete, Phase 6 running).

**Last Updated**: 2026-02-28

---

## Project Overview

This project implements autonomous parallel parking on 1/28-scale RC cars using deep reinforcement learning (PPO). The journey went from a **classical MPC baseline** → **RL simulation training** → **full hardware deployment** on the CRS ChronosCar platform, culminating in a car that parks itself reliably in under 50 steps (~5 seconds).

Two cars were trained and deployed:
- **ChronosCar** (L=0.13m, W=0.065m): Fully trained (7 phases, 88% sim success). **Deployed on hardware — parks successfully**, 40 steps average, along=2.7cm, lateral=0.4cm, yaw=2.5°.
- **New Car** (L=0.15m, W=0.10m): Training in progress. Larger car with wheel encoders (better velocity measurement). Friction model baked into training.

---

## Table of Contents

1. [Phase 1 — MPC Baseline](#1-phase-1--mpc-baseline)
2. [Phase 2 — RL Environment Design](#2-phase-2--rl-environment-design)
3. [Phase 3 — ChronosCar Training](#3-phase-3--chronoscar-training)
4. [Phase 4 — Hardware Deployment](#4-phase-4--hardware-deployment)
5. [Phase 5 — New Car Training](#5-phase-5--new-car-training)
6. [Hardware Results](#6-hardware-results)
7. [Quick Start: Run the ChronosCar](#7-quick-start-run-the-chronoscar)
8. [Quick Start: Training](#8-quick-start-training)
9. [Key Lessons Learned](#9-key-lessons-learned)
10. [Project Structure](#10-project-structure)
11. [Network & Hardware Reference](#11-network--hardware-reference)

---

## 1. Phase 1 — MPC Baseline

**Files**: `mpc/`

Before building the RL system, a classical Model Predictive Control (MPC) baseline was implemented to understand the problem difficulty.

### What Was Implemented

`mpc/teb_mpc.py` — A Timed Elastic Band (TEB) MPC controller with 3-phase staging:
1. **Phase 1: Alignment** — Drive forward/reverse to align parallel to the bay
2. **Phase 2: Entry** — Back into the bay with a steering arc
3. **Phase 3: Fine adjustment** — Small corrections to center position

`mpc/reference_trajectory.py` — Geometric reference trajectory generation (3-arc parallel parking maneuver).

`mpc/staged_controller.py` — Orchestrates the phases, handles phase transitions.

`mpc/hybrid_controller.py` — Hybrid MPC + rule-based fallback for tight gaps.

### Why We Moved to RL

The MPC baseline had fundamental limitations for this scale:
- **Fixed trajectory**: MPC follows a pre-computed geometric arc. Any variation in initial position requires replanning.
- **No obstacle adaptation**: The staging phases don't gracefully handle neighbor car position variations.
- **Sensitivity to parameters**: Small errors in velocity tracking or steering response caused large trajectory deviations.
- **Scale effects**: At 1/28 scale, the wheelbase (9cm) and turning radius (25cm) mean tiny actuation errors compound over the 3-phase maneuver.

RL learns a reactive policy that adapts to any initial position and corrects errors on every step — no trajectory pre-planning needed.

---

## 2. Phase 2 — RL Environment Design

**Files**: `env/`, `config_env.yaml`, `config_env_chronos.yaml`, `config_env_newcar.yaml`

### Environment: Kinematic Bicycle Model

`env/vehicle_model.py` — Kinematic bicycle model with configurable parameters:
- State: `[x_rear, y_rear, yaw, v]` (rear-axle position)
- Actions: `[steer_cmd, accel_cmd]` (continuous, normalized to [-1, 1])
- Wheelbase-parameterized: `v_next = v + accel*dt`, `yaw_next = yaw + v*tan(steer)/wheelbase*dt`
- New car config adds: `static_friction` and `kinetic_friction` parameters to model real-world dead zone

`env/parking_env.py` — Core Gym environment:
- Observation: 12-dimensional vector (see below)
- Bay-frame coordinates: along/lateral relative to bay axis
- Virtual obstacle ray-casting for distance features

`env/obstacle_manager.py` — Collision detection + 3-axis ray casting (forward dF, left dL, right dR)

`env/reward_parallel_parking.py` — Dense reward shaping:
- Proximity reward (distance to bay center)
- Lateral progress reward (bonus for entering bay)
- Bay entry bonus
- Action smoothness penalties
- Gear switch + steer reversal penalties
- Settling bonus (stationary at goal)
- Stall penalty (stuck oscillating)
- Collision termination (-200)
- Timeout termination (-400)
- Success reward (+200)

### Observation Space (12 features)

```
obs[0]  along       Distance along bay axis (positive = car is in front of bay opening)
obs[1]  lateral     Distance lateral to bay (positive = car is to the right)
obs[2]  yaw_err     Heading error relative to bay (0 = perfectly aligned)
obs[3]  v           Current velocity
obs[4]  dF          Ray distance forward (normalized, max 5.0)
obs[5]  dL          Ray distance left
obs[6]  dR          Ray distance right
obs[7]  dWL         Distance to left world wall
obs[8]  dWR         Distance to right world wall
obs[9]  dWF         Distance to front world wall
obs[10] dWB         Distance to rear world wall
obs[11] steer       Current steering angle
```

### Curriculum Learning (key design principle)

`rl/curriculum_env.py` — Multi-phase curriculum with auto-progression:
- Each phase has: `timesteps`, `success_threshold`, `env_config`
- Phase advances only when `success_rate >= threshold` over 100 eval episodes
- **One change at a time**: each phase modifies only ONE aspect of difficulty
- Tolerance progression: loose → tight (5.4cm → 2.7cm over 7 phases)

**Critical lesson**: Changing multiple parameters simultaneously causes policy collapse. Examples:
- Phase 5 originally changed both neighbor jitter AND tolerance → 234M steps, 0% success → added Phase 4b (jitter only at 3.2cm) as intermediate
- Phase 6 originally tightened 3 settling thresholds simultaneously → policy regressed from 83% to 0% → fixed to only change `settling_bonus`, moved threshold changes to Phase 7

---

## 3. Phase 3 — ChronosCar Training

**Files**: `config_env_chronos.yaml`, `rl/curriculum_config_chronos.yaml`, `train_chronos.sh`

### ChronosCar Dimensions

| Parameter | Value | Notes |
|-----------|-------|-------|
| Length | 0.13m | Measured |
| Width | 0.065m | Measured |
| Wheelbase | 0.090m | From CRS model_params.yaml |
| rear_overhang | 0.020m | (L - wheelbase) / 2 |
| Max steer | 0.35 rad (20°) | From joystick_controller.yaml |
| Turning radius R | 0.247m | wheelbase / tan(max_steer) |
| body_scale | 0.36 | vs original (wheelbase ratio) |
| maneuver_scale | 0.569 | vs original (R ratio) |

**DUAL SCALE FACTORS**: Because max_steer differs from the original reference design, there are two distinct scale factors:
- `body_scale = 0.36`: Used for car/obstacle sizes, bay position, position tolerances
- `maneuver_scale = 0.569`: Used for spawn distances, y_jitter, x offsets (spawn must be geometrically reachable)

Using body_scale for spawn distances gave Lateral/R=1.22 (vs correct 1.91), making parking geometrically impossible. Always use the correct scale per parameter type.

### ChronosCar 7-Phase Curriculum

| Phase | Name | along_tol | yaw_tol | Timesteps | Threshold |
|-------|------|-----------|---------|-----------|-----------|
| 1 | Foundation | 5.4cm | 8.6° | 150M | 95% |
| 2 | Random Spawn | 5.4cm | 8.6° | 60M | 85% |
| 3 | Random Bay X | 4.5cm | 8.6° | 70M | 80% |
| 4a | Small Bay Y | 3.6cm | 8.6° | 100M | 80% |
| 4 | Full Random Bay | 3.2cm | 8.6° | 150M | 80% |
| 5 | Tighten (2.7cm) | 2.7cm | 8.6° | 80M | 80% |
| 6 | Polish | 2.7cm | 8.6° | 30M | 80% |
| 7 | Yaw Precision | 2.7cm | 4.0° | 50M | 75% |

**Total**: ~690M timesteps, ~2 days on GPU.

### Training Commands

```bash
# Full training from scratch
./train_chronos.sh

# Resume from a checkpoint
./resume_phase_chronos.sh phase5_tight_tol

# Evaluate a checkpoint
python -m rl.eval_policy \
    --checkpoint checkpoints/chronos/.../best_checkpoint \
    --num-episodes 100 --deterministic
```

### Key Hyperparameters

```yaml
lr: 1e-4            # (drops to 3e-5 in later phases)
entropy_coeff: 0.02  # Phase 1-3 (0.003 from Phase 4+)
train_batch_size: 4000
num_sgd_iter: 10
num_workers: 4
```

**Critical**: Never increase entropy AND learning rate simultaneously in later phases. Increasing entropy 4x (0.003→0.008) AND LR 1.5x simultaneously destroyed the learned policy after 20M steps (13% success rate).

---

## 4. Phase 4 — Hardware Deployment

**Files**: `deploy/`, `deploy/rl_parking_node.py`, `deploy/ros_bridge.py`, `deploy/parking_scene.yaml`

### Architecture

```
Host (Ubuntu 22.04)               CRS Docker (Ubuntu 20.04 + ROS Noetic)
+----------------------------+   +----------------------------------------+
| rl_parking_node.py         |   | roscore                                 |
|   PPO checkpoint           |UDP| WiFiCom + CRS Estimator                 |
|   Obs pipeline (12 feats)  |<->| NatNet MoCap bridge                    |
|   ActionConverter          |:58| ros_bridge.py                          |
|   CollisionRecovery        |00 |   3-mode torque (impulse/fwd/rev)      |
|   CSV logging              |/5 |   velocity=nan for impulse (PID off)   |
|   matplotlib viz           |801|   3-double UDP protocol                |
+----------------------------+   +----------------------------------------+
                                           |                |
                                    WiFi   |          NatNet|
                               +-----------v--+   +---------v-------+
                               | ChronosCar   |   | OptiTrack PC    |
                               | 129.217.130.30|   | 129.217.130.57  |
                               +--------------+   +-----------------+
```

No ROS on the host machine (Ubuntu 22.04 is incompatible with ROS Noetic). All ROS runs in Docker. The RL node communicates via UDP.

### Safe Startup Sequence

```bash
# 1. Start Motive on Windows PC (MUST be first — NatNet crashes otherwise)
# 2. Turn car OFF (prevents CRS MPC from racing car at startup)
# 3. Start Docker services
./deploy/start_docker.sh

# 4. Wait 15 seconds for all services to initialize

# 5. Kill the CRS MPC controller (it overrides our commands on the same topic)
docker exec crs bash -c "source /opt/ros/noetic/setup.bash && rosnode kill /BEN_CAR_WIFI/controller_node"

# 6. Turn car ON (safe now, no controller running)

# 7. Calibrate bay position (place car at parking spot center)
python deploy/rl_parking_node.py --calibrate --scene deploy/parking_scene.yaml

# 8. Ghost mode (no motor commands, verify observations)
python deploy/rl_parking_node.py --ghost \
    --checkpoint checkpoints/chronos/.../best_checkpoint \
    --scene deploy/parking_scene.yaml

# 9. Live parking
python deploy/rl_parking_node.py \
    --checkpoint checkpoints/chronos/.../best_checkpoint \
    --scene deploy/parking_scene.yaml

# Emergency stop: PHYSICAL POWER SWITCH on car
```

### 28 Bugs Fixed During Hardware Deployment

The biggest challenge was bridging the sim-to-real gap. Every bug below caused parking failure until fixed.

#### Observation Pipeline Bugs

| # | Bug | Symptom | Fix |
|---|-----|---------|-----|
| 1 | World boundaries not centered on bay | dF/dL/dR all = 5.0 | Re-center world on bay_center in rl_parking_node.py |
| 2 | Bay calibration used rear axle, training used body center | along=0.045 even when parked | Shift calibration forward by dist_to_center=0.045m |
| 3 | goal_offset_along shifted obstacle positions | yaw_err oscillated -12° to -44° | Separate bay_center (obstacles) from bay_goal_{x,y} (along/lat only) |
| 4 | World boundaries in scene vs training mismatch | Policy garbage (OOD rays) | Ensure parking_scene.yaml world = config_env_chronos.yaml world |
| 5 | dist_to_center hardcoded as L/2 - 0.05 | Wrong for new car | Make configurable via vehicle.rear_overhang |

#### Velocity / Actuation Bugs

| # | Bug | Symptom | Fix |
|---|-----|---------|-----|
| 6 | CRS v_tot/vx_b unreliable (~28 m/s at standstill) | Policy sees garbage velocity | Compute from MoCap position deltas + LP filter |
| 7 | LP filter alpha=0.1 (too slow) | 2.3s velocity lag | Use alpha=0.4 |
| 8 | Velocity integrator deadlock | Car stuck at standstill forever | Closed-loop + MIN_CMD=0.05 friction boost |
| 9 | MIN_CMD direction wrong (used v_target sign) | Random forward burst during reverse | Use accel_cmd sign, not v_target sign |
| 10 | Reverse has no speed control | Car can't slow down in reverse | 3-double UDP [vel, steer, v_actual]; ros_bridge 3-mode reverse |
| 11 | RLlib action as nested array | action[0] = array not float | np.asarray(action[0]).item() |

#### Safety / Calibration Bugs

| # | Bug | Symptom | Fix |
|---|-----|---------|-----|
| 12 | max_steer clamp = 0.30 (not 0.35) | Actions clipped vs training | Set safety.max_steer = 0.35 |
| 13 | max_vel = 0.25 (not 0.5) | Velocity range wrong | Set vehicle.max_vel = 0.5 |
| 14 | max_steer_rate = 0.3 (not 0.5) | Steer rate clipped | Set safety.max_steer_rate = 0.5 |
| 15 | Steer 1.376x commanded | Yaw rate 2x wrong | Measure with steer_calibration.py; divide by steer_gain |
| 16 | Velocity 1.503x commanded | Odometry wrong | Divide by velocity_gain |
| 17 | Calibration gains applied before rate limiting | Rate limiting in wrong units | Apply gains AFTER rate limiting |

#### Static Friction Bugs

| # | Bug | Symptom | Fix |
|---|-----|---------|-----|
| 18 | Static friction prevents movement at low cmd | Car stuck, tiny corrections ignored | MIN_TORQUE=0.10 in ros_bridge.py |
| 19 | MIN_CMD=0.05 velocity boost: CRS PID fights it | Burst cancelled (PID brakes) | velocity=nan during impulse (disables CRS PID) |
| 20 | CRS PID velocity=0.0 fights reverse torque | Reverse torque cancelled | Also use velocity=nan for all reverse |
| 21 | Pure MIN_CMD (no torque impulse) | Too weak for static friction | IMPULSE_TORQUE=0.20 for 2 steps (0.2s) |

#### Collision Recovery Bugs

| # | Bug | Symptom | Fix |
|---|-----|---------|-----|
| 22 | Always reversed after collision | Drove deeper into bay when at yaw≈0° | Smart direction: inside bay → FORWARD; outside → BACKWARD |
| 23 | reset() zeroed v_model after recovery | +0.286 m/s gap → wrong steering | v_model = v_current after recovery |
| 24 | Training terminates on collision — no recovery | Policy undefined after collision | Hand-coded reverse recovery (15 steps) |

#### Deployment Architecture Bugs

| # | Bug | Symptom | Fix |
|---|-----|---------|-----|
| 25 | controller_node overrides RL commands | Car races at startup | rosnode kill /BEN_CAR_WIFI/controller_node |
| 26 | Start Motive after Docker | NatNet bridge crashes silently | Always start Motive first |
| 27 | UDP recvfrom() returns stale data | Policy acts on 5s-old state | drain_and_get_latest(): read all buffered, keep newest |
| 28 | controller_node races car if turned on before kill | Car drives at max speed | Car OFF → Docker → kill controller → Car ON |

#### Fine Centering Bugs

| # | Bug | Symptom | Fix |
|---|-----|---------|-----|
| 29 | goal_offset_along=+0.030 → parks too close to right neighbor | dF=7.1cm | goal_offset_along=-0.020 (shift 2cm toward curb) |
| 30 | Asymmetric in-bay speed cap | Forward overshoot into right neighbor | Symmetric cap: abs(along)<0.10 → v_model ≤ ±0.12 m/s |
| 31 | In-bay cap at 0.10 m/s too slow | 55-step oscillation, yaw=7.5° final | Raise to 0.12 m/s (2.8°/step yaw authority) |

### Key Deployment Components

#### ActionConverter (rl_parking_node.py)

Converts policy actions (normalized [-1,1]) to hardware commands:

```
accel, steer → rate limit → v_model (open-loop) → friction boost → calibration divide → vel_cmd, steer_cmd
```

- `v_model`: Open-loop integrator (`v_model += accel*dt`). What the policy expects. Feed to ros_bridge as velocity setpoint.
- `v_hw`: Measured from MoCap position deltas + LP filter (alpha=0.4). Fed back to policy as obs[3].
- Rate limiting in training units (BEFORE dividing by calibration gains).
- In-bay speed cap: `if abs(along) < 0.10: v_model = clip(v_model, -0.12, 0.12)`

#### ros_bridge.py (Docker)

3-mode torque controller:
- **STOP** (`|vel| < 0.01`): Active braking
- **FORWARD** with impulse at standstill: `velocity=nan` + `torque=0.20` for 2 steps, then `velocity=v_target` + feedforward torque
- **REVERSE**: `velocity=nan` + 3 sub-modes (BRAKE/START/MAINTAIN)

The `velocity=nan` trick disables the CRS PID controller completely during impulse phases, allowing direct torque control. Without this, the PID fights the impulse torque (brakes the car back to 0.05 m/s setpoint).

#### Calibration

```bash
# Measure steer_gain and velocity_gain:
python deploy/steer_calibration.py
# → Updates parking_scene.yaml: calibration.steer_gain, calibration.velocity_gain
```

Current ChronosCar values: `steer_gain=1.376`, `velocity_gain=1.503`.

---

## 5. Phase 5 — New Car Training

**Files**: `config_env_newcar.yaml`, `rl/curriculum_config_newcar.yaml`, `rl/curriculum_config_newcar_resume.yaml`, `train_newcar.sh`, `resume_phase5_newcar.sh`, `deploy/parking_scene_newcar.yaml`

### Why the New Car

Professor Frank's recommendation (2026-02-25):
- New car has **wheel encoders** → direct velocity measurement (eliminates MoCap position-derivative lag)
- Larger car → easier to handle mechanically
- Lesson: encoder-based velocity removes the #1 source of sim-to-real gap (velocity estimation)

### New Car Dimensions

| Parameter | Value | Notes |
|-----------|-------|-------|
| Length | 0.15m | Measured |
| Width | 0.10m | Measured |
| Wheelbase | 0.113m | Measured |
| rear_overhang | 0.0185m | (L - wheelbase) / 2 |
| Max steer | 0.314 rad (18°) | From CRS config |
| Turning radius R | 0.348m | wheelbase / tan(max_steer) |
| body_scale | 0.452 | vs original (wheelbase ratio) |
| maneuver_scale | 0.801 | vs original (R ratio) |

### Friction Model in Simulation

The new car training bakes friction into the simulation environment (`env/vehicle_model.py`):

```yaml
vehicle:
  static_friction: 0.15   # m/s² dead zone at standstill
  kinetic_friction: 0.05  # m/s² opposes motion while moving
```

Effect: policy must command accel > 0.15 m/s² to start moving. This teaches the policy to use decisive commands rather than tiny oscillations.

### Curriculum Status (as of 2026-02-28)

| Phase | Status | Notes |
|-------|--------|-------|
| 1 Foundation | Complete | |
| 2 Random Spawn | Complete | |
| 3 Random Bay X | Complete | |
| 4a Small Bay Y | Complete | |
| 4 Full Random | Complete | reward=444 |
| 5 Tighten 2.7cm | **Complete** | reward=613, **83% success** |
| 6 Polish | Running | phase6_restart from phase5 checkpoint |
| 7 Tighter Settling | Pending | |
| 8 Yaw Precision | Pending | |

```bash
# Resume Phase 6 from Phase 5 checkpoint (fresh, not degraded)
./resume_phase5_newcar.sh phase6_restart

# Continue Phase 6 from its best checkpoint
./resume_phase5_newcar.sh phase6_random_obstacles
```

### New Car Deployment Config

`deploy/parking_scene_newcar.yaml` — Uses `velocity_source: encoder` to read `vx_b` directly from CRS estimator instead of computing from MoCap. Sanity check: if `|vx_b| > 1.0` at startup, fall back to MoCap derivative.

---

## 6. Hardware Results

### ChronosCar — Best Run (2026-02-27)

```
Step 40: PARKED SUCCESSFULLY
  along   = +0.027m  (tol: 0.055m)  WITHIN TOLERANCE
  lateral = -0.004m  (tol: 0.055m)  WITHIN TOLERANCE
  yaw_err = -2.5°    (tol: 8.6°)    WITHIN TOLERANCE
  v       = 0.000m/s (tol: 0.05)    WITHIN TOLERANCE
  settled = 3 consecutive steps
```

The car executes an S-curve approach from ~50cm away, reverses into the bay, and settles at near-exact center position.

### Key Bugs That Enabled This Result (in order of impact)

1. **Steering/velocity calibration** (steer_gain=1.376, vel_gain=1.503) — without this, yaw error was ~22° permanently
2. **World boundary re-centering on bay** — without this, all ray distances were 5.0 (out of distribution)
3. **goal_offset_along separation** — without this, offset shifted neighbor positions, causing yaw oscillation
4. **Smart collision recovery direction** — without this, recovery drove car deeper into bay curb
5. **velocity=nan during impulse** — without this, CRS PID cancelled the friction override burst

---

## 7. Quick Start: Run the ChronosCar

### Prerequisites

- OptiTrack Motive running on Windows PC (129.217.130.57)
- Docker image `crs:local` built (run `./deploy/setup_hardware.sh` once)
- ChronosCar charged and configured for lab WiFi
- Trained checkpoint available

### Full Deployment Procedure

```bash
# 1. Start Motive on Windows PC (FIRST — NatNet needs it)

# 2. Turn car OFF

# 3. Launch Docker services
./deploy/start_docker.sh
# Creates tmux session 'crs' with: roscore | WiFiCom+Estimator | NatNet | ros_bridge

# 4. Wait 15s then kill CRS controller (overrides our commands)
docker exec crs bash -c "source /opt/ros/noetic/setup.bash && rosnode kill /BEN_CAR_WIFI/controller_node"

# 5. Turn car ON

# 6. Calibrate bay (place car at parking spot center, pointing along bay)
python deploy/rl_parking_node.py --calibrate --scene deploy/parking_scene.yaml

# 7. Ghost mode — verify observations (no motor commands)
python deploy/rl_parking_node.py --ghost \
    --checkpoint checkpoints/chronos/curriculum_20260210_113810/phase7_polish/best_checkpoint \
    --scene deploy/parking_scene.yaml
# Watch: along/lat → 0 as car moves to bay, dF/dL/dR not all 5.0

# 8. Move car to start position (~50cm from bay, facing same direction)

# 9. Live parking!
python deploy/rl_parking_node.py \
    --checkpoint checkpoints/chronos/curriculum_20260210_113810/phase7_polish/best_checkpoint \
    --scene deploy/parking_scene.yaml \
    --log-dir deploy/logs/
```

### Observation Quality Checks (Ghost Mode)

| What to check | Good sign | Bad sign |
|---|---|---|
| dF/dL/dR | 0.2–2.0m range | All exactly 5.0 (world boundary bug) |
| along | Decreases as car approaches bay | Stays constant (calibration bug) |
| yaw_err | Near 0 when car aligned with bay | Large when aligned (bay yaw wrong) |
| v | ≤ 0.3 m/s | 28 m/s (CRS estimator broken — use MoCap) |

### Car Positioning

- Place car **50–80cm** from bay opening
- Yaw must be within **±45°** of bay heading (policy trained with this range)
- Lateral offset: at least **25cm** from bay center line (need room for S-curve)

### CLI Reference

```bash
python deploy/rl_parking_node.py \
    --checkpoint <path>            # Required: PPO checkpoint directory
    --scene deploy/parking_scene.yaml  # Scene config (bay, vehicle, obstacles)
    --ghost                        # Read-only mode (no motor commands)
    --dry-run                      # Test checkpoint + obs pipeline offline
    --no-viz                       # Disable matplotlib window
    --log-dir deploy/logs/         # CSV logging for post-run analysis
    --calibrate                    # Bay calibration mode (place car at bay first)
```

---

## 8. Quick Start: Training

### ChronosCar (complete from scratch)

```bash
source venv/bin/activate
./train_chronos.sh
# Checkpoints → checkpoints/chronos/
# Monitor: tensorboard --logdir checkpoints/chronos/
```

### New Car (from Phase 5 checkpoint — fastest path to deployment)

```bash
source venv/bin/activate
./resume_phase5_newcar.sh phase6_restart    # Restart Phase 6 (phase5 checkpoint)
./resume_phase5_newcar.sh phase7_polish     # Start Phase 7
```

### Evaluate Any Checkpoint

```bash
python -m rl.eval_policy \
    --checkpoint checkpoints/chronos/.../best_checkpoint \
    --num-episodes 100 --deterministic

python -m rl.visualize_checkpoint \
    --checkpoint checkpoints/chronos/.../best_checkpoint \
    --num-episodes 5
```

### Monitoring

```bash
tensorboard --logdir checkpoints/chronos/
# Key metrics: custom_metrics/success_rate (phase advancement)
# Phase advances when success_rate >= threshold for 3 consecutive evals
```

---

## 9. Key Lessons Learned

### Curriculum Design

1. **One change at a time**: Every phase change must modify exactly one aspect of difficulty. Two simultaneous changes (e.g., tighten tolerance + add jitter) consistently caused catastrophic regression.

2. **Intermediate phases are cheaper than retraining**: Adding Phase 4b (jitter-only at 3.2cm) cost 100M steps but saved 234M steps of Phase 5 failure at 0% success.

3. **Never increase entropy in fine-tuning phases**: Entropy 0.003 → 0.008 in Phase 6 destroyed the policy after 20M steps. Later phases need conservative LR/entropy.

4. **Phase regression is detectable early**: If success rate is 0% after 5M steps of a new phase while pos_error_min is small (car IS reaching vicinity), the phase criteria are too hard. Stop and fix.

### Sim-to-Real

5. **Velocity estimation is critical**: v_tot/vx_b from CRS estimator were completely unreliable. MoCap position derivative with LP filter (alpha=0.4) is the only reliable source for ChronosCar.

6. **Calibrate gains on every deployment**: Steering and velocity gains can drift. `steer_calibration.py` takes 2 minutes and prevents hours of debugging.

7. **Static friction is the #1 enemy**: The dead zone (car doesn't move for torque < 0.10) must be compensated at the hardware level (impulse torque + velocity=nan). The policy cannot learn to command "extra" in sim.

8. **World boundaries must be bay-centered**: Training has the bay at (0,0). Deployment bay is at (-0.88, -0.27) in MoCap frame. Without re-centering, wall rays are completely wrong.

9. **bay_center vs bay_goal are different things**: `bay_center` defines where obstacles are placed and where the world is centered. `bay_goal_{x,y}` defines the reference point for along/lat measurement. Applying `goal_offset_along` to `bay_center` shifts obstacles — catastrophic.

### Hardware Architecture

10. **Never use `velocity=0.0` for reverse**: CRS PID treats this as a setpoint. It actively fights reverse torque. Use `velocity=nan` (disables PID) for all reverse and impulse.

11. **Kill controller_node before powering on the car**: The MPC controller publishes to the same topic as our bridge. If car is on when controller starts, it races at max speed.

12. **MoCap must start before Docker**: NatNet bridge connects at startup. If Motive isn't running, the bridge hangs silently and never delivers state data.

---

## 10. Project Structure

```
/home/naeem/Documents/final/
│
├── Documentation
│   ├── README.md                          This file — full project overview
│   ├── FINAL_HANDOVER.md                  Detailed hardware deployment reference
│   ├── HARDWARE_DEPLOYMENT_PLAN.md        Original deployment planning notes
│   └── deploy/README.md                   Deployment quick reference
│
├── Environment (simulation)
│   ├── env/parking_env.py                 Core Gym environment
│   ├── env/vehicle_model.py               Kinematic bicycle model + friction
│   ├── env/obstacle_manager.py            Ray-casting + collision detection
│   └── env/reward_parallel_parking.py     Reward function
│
├── MPC Baseline
│   ├── mpc/teb_mpc.py                     TEB-MPC controller (3-phase)
│   ├── mpc/reference_trajectory.py        Geometric trajectory generation
│   ├── mpc/staged_controller.py           Phase orchestration
│   └── mpc/hybrid_controller.py           Hybrid MPC+rule-based
│
├── RL Training
│   ├── rl/train_curriculum.py             Main training loop + curriculum manager
│   ├── rl/curriculum_env.py               Multi-phase env wrapper
│   ├── rl/gym_parking_env.py              Gym-compatible env wrapper
│   ├── rl/policy_network.py               PPO network architecture
│   ├── rl/eval_policy.py                  Evaluation script
│   └── rl/visualize_checkpoint.py         Visualization script
│
├── Configuration
│   ├── config_env.yaml                    Original reference config
│   ├── config_env_chronos.yaml            ChronosCar training config
│   ├── config_env_newcar.yaml             New car training config (+ friction)
│   ├── rl/curriculum_config.yaml          Original curriculum
│   ├── rl/curriculum_config_chronos.yaml  ChronosCar 7-phase curriculum
│   ├── rl/curriculum_config_newcar.yaml   New car 7-phase curriculum
│   └── rl/curriculum_config_newcar_resume.yaml  Phase 5→8 resume config
│
├── Hardware Deployment
│   ├── deploy/rl_parking_node.py          RL inference + control + recovery + logging
│   ├── deploy/ros_bridge.py               Docker: ROS ↔ UDP bridge + torque control
│   ├── deploy/parking_scene.yaml          ChronosCar bay + calibration + tolerances
│   ├── deploy/parking_scene_newcar.yaml   New car deployment config
│   ├── deploy/steer_calibration.py        Measure steer/velocity gains
│   ├── deploy/start_docker.sh             Launch all 4 Docker services (tmux)
│   ├── deploy/setup_hardware.sh           One-time Docker + CRS build
│   └── deploy/run_parking.sh              Wrapper for rl_parking_node.py
│
├── Training Scripts
│   ├── train_chronos.sh                   Full ChronosCar curriculum
│   ├── train_newcar.sh                    Full new car curriculum
│   ├── resume_phase_chronos.sh            Resume any ChronosCar phase
│   └── resume_phase5_newcar.sh            Resume new car from phase 5/6/7
│
└── Utilities
    ├── diagnose_training.py               Training health check
    ├── verify_all_phases.py               Config validation
    └── visualize_parking.py               Standalone visualizer
```

---

## 11. Network & Hardware Reference

| Device | IP | Purpose |
|--------|----|---------|
| Host machine | 129.217.130.20 | Runs Docker + RL node |
| OptiTrack/Motive PC | 129.217.130.57 | MoCap server (NatNet) |
| ChronosCar | 129.217.130.30 | RC car (WiFi UDP port 20211) |
| Car AP mode | 192.168.4.1 | Config page (use incognito browser) |

### UDP Protocol

| Direction | Port | Format | Fields |
|-----------|------|--------|--------|
| State (Docker → Host) | 5800 | 7 × float64 = 56 bytes | x, y, yaw, v_tot, vx_b, vy_b, steer |
| Command (Host → Docker) | 5801 | 3 × float64 = 24 bytes | velocity, steer, v_actual |

### Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Verify
python verify_all_phases.py
```

### Troubleshooting Quick Reference

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| dF/dL/dR all = 5.0 | World boundaries wrong | Check parking_scene.yaml world = ±1.25m |
| Policy outputs constant vel=-0.25 | yaw_err ~180° | Reposition car to face bay direction |
| Car doesn't move | controller_node not killed | rosnode kill /BEN_CAR_WIFI/controller_node |
| Car races at startup | Car was on before controller killed | Car OFF → Docker → kill controller → Car ON |
| along stays constant as car moves | Calibration bug | Recalibrate with --calibrate flag |
| Port 5800 already in use | Zombie from Ctrl+Z | ss -ulnp \| grep 5800; kill -9 <PID> |
| No state data | Motive not running or wrong IPs | Start Motive first; check NatNet launch file |
| v reads 28 m/s | CRS estimator broken | Normal — rl_parking_node uses MoCap derivative |

---

*For complete hardware deployment details: [FINAL_HANDOVER.md](FINAL_HANDOVER.md)*

*For deploy-specific troubleshooting: [deploy/README.md](deploy/README.md)*
