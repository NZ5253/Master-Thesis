# Implementation Plan - All Phases Today
**Date:** 2026-01-11
**Goal:** Complete all randomization phases by end of day

---

## Overview of Phases

```
Phase 1: ✅ COMPLETE - Deep parking with fixed spawn
Phase 2: ⏳ TODO - Random spawn A, handle imperfect B arrival
Phase 3: ⏳ TODO - Multiple B points based on approach direction
Phase 4: ⏳ TODO - Random bay position, keep X centered
Phase 5: ⏳ TODO - Random neighbor jitter
Phase 6: ⏳ TODO - Random obstacles in environment
Phase 7: ⏳ TODO - Baseline MPC comparison
```

---

## Phase 2: Random Spawn Point A

### Objective
Ego car spawns at random positions in the approach lane. Must still reach B and complete parking successfully even with imperfect B arrival.

### Requirements
1. Randomize A position in reasonable range
2. B tolerance increased to handle imperfect arrivals
3. B→C planner must handle varied B states
4. Success rate target: >85% with random spawn

### Implementation Steps

#### Step 2.1: Add Spawn Randomization
```yaml
# config_env.yaml - spawn_lane section
parallel:
  spawn_lane:
    center_y: -1.3
    yaw: 0.0
    x_min_offset: -1.5     # NEW: Increased range
    x_max_offset: -0.8     # NEW: Increased range
    y_jitter: 0.2          # NEW: ±20cm Y randomization
    yaw_jitter: 0.15       # NEW: ±8.6° yaw randomization
```

**Code Changes:**
```python
# env/parking_env.py - _random_start() method
def _random_start(self):
    spawn_cfg = self.spawn_cfg
    center_y = spawn_cfg.get("center_y", -1.3)

    # NEW: X randomization
    x_min = spawn_cfg.get("x_min_offset", -1.5)
    x_max = spawn_cfg.get("x_max_offset", -0.8)
    x = np.random.uniform(x_min, x_max)

    # NEW: Y randomization
    y_jitter = spawn_cfg.get("y_jitter", 0.0)
    y = center_y + np.random.uniform(-y_jitter, y_jitter)

    # NEW: Yaw randomization
    base_yaw = spawn_cfg.get("yaw", 0.0)
    yaw_jitter = spawn_cfg.get("yaw_jitter", 0.0)
    yaw = base_yaw + np.random.uniform(-yaw_jitter, yaw_jitter)

    return np.array([x, y, yaw, 0.0])
```

#### Step 2.2: Increase B Tolerance
```python
# mpc/staged_controller.py lines 87-92
self.goal_B = ParkingGoal(
    x=0.0,
    y=-0.5,
    yaw=1.5708
)

# UPDATED tolerances
self.B_position_tolerance = 0.25  # Increased from 0.15
self.B_yaw_tolerance = 0.50       # Increased from 0.35

def _check_reached_B(self, state):
    dx = state.x - self.goal_B.x
    dy = state.y - self.goal_B.y
    pos_err = np.sqrt(dx**2 + dy**2)
    yaw_err = abs((state.yaw - self.goal_B.yaw + np.pi) % (2*np.pi) - np.pi)

    return (pos_err < self.B_position_tolerance and
            yaw_err < self.B_yaw_tolerance)
```

#### Step 2.3: Test Random Spawn
```bash
# Test with 20 episodes
python3 -m mpc.generate_expert_data --scenario parallel --hybrid --episodes 20

# Check success rate
python3 -c "
import glob
successes = len(glob.glob('data/expert_parallel/episode_*.pkl'))
failures = len(glob.glob('data/expert_parallel_debug/fail_*.pkl'))
total = successes + failures
print(f'Success rate: {successes}/{total} = {100*successes/total:.1f}%')
"
```

**Success Criteria:**
- Success rate >85%
- No failures due to "can't reach B"
- Parking quality maintained (pos_err <6cm average)

---

## Phase 3: Multiple B Points (Approach from Any Direction)

### Objective
Support approaching the bay from different directions (left, right, center). Always use reverse parking, but choose optimal B based on spawn location.

### Requirements
1. Calculate optimal B point based on spawn position
2. Maintain B formula: B_y = bay_center_y - 0.5
3. All parking must be reverse (backing into bay)
4. Choose easiest/closest B to reach

### Implementation Steps

#### Step 3.1: B Point Calculator
```python
# mpc/staged_controller.py - NEW method
def _calculate_optimal_B(self, spawn_state, bay_center):
    """
    Calculate optimal B point based on spawn position.
    Always maintain: B is 0.5m offset from bay center.
    """
    bay_x, bay_y, bay_yaw = bay_center

    # Strategy: B always at bay entrance, but approach angle may vary
    # For parallel parking (bay_yaw ≈ π/2):
    #   - Always B_x = bay_x (aligned with bay)
    #   - Always B_y = bay_y - 0.5 (0.5m below bay)
    #   - Always B_yaw = bay_yaw (perpendicular to road)

    # The spawn position determines the APPROACH PATH, not B itself
    # A→B planner will find the path automatically

    B = ParkingGoal(
        x=bay_x,
        y=bay_y - 0.5,
        yaw=bay_yaw
    )

    return B
```

**Key Insight:** B position is **always the same** (bay entrance). What changes is the **path from A to B**. The receding MPC (A→B stage) automatically finds the optimal path.

#### Step 3.2: Update Staged Controller Initialization
```python
# mpc/staged_controller.py - __init__ method
def __init__(self, config_path, env_cfg, dt):
    # ... existing code ...

    # DON'T hardcode goal_B anymore
    # self.goal_B = ParkingGoal(x=0.0, y=-0.5, yaw=1.5708)  # REMOVE

    # Instead, calculate it from environment
    bay_center_y = env_cfg.get("parking", {}).get("bay", {}).get("center_y", 0.0)
    bay_yaw = env_cfg.get("parking", {}).get("bay", {}).get("yaw", 1.5708)

    self.goal_B = self._calculate_optimal_B(
        spawn_state=None,  # Not needed for current approach
        bay_center=(0.0, bay_center_y, bay_yaw)
    )
```

#### Step 3.3: Handle Opposite Direction Spawn (Future)
```python
# For later when spawn can be from positive X:
def _calculate_optimal_B(self, spawn_state, bay_center):
    bay_x, bay_y, bay_yaw = bay_center

    # Check spawn position
    if spawn_state is not None:
        spawn_x = spawn_state.x

        # If spawning from far right, may need different approach
        # But B position REMAINS THE SAME
        # Only the PATH to B changes (handled by A→B MPC)
        pass

    # B formula is ALWAYS the same
    return ParkingGoal(
        x=bay_x,
        y=bay_y - 0.5,
        yaw=bay_yaw
    )
```

**No Code Changes Needed for Phase 3!** The current system already handles this because:
- B is always at the bay entrance (formula is generic)
- A→B MPC automatically finds the path regardless of spawn position
- As long as spawn is reachable, the system works

#### Step 3.4: Test Different Spawn Positions
```bash
# Test with wide X range (both sides)
python3 -m mpc.generate_expert_data --scenario parallel --hybrid --episodes 30

# Manually verify episodes spawn from different positions
# All should successfully reach B and park
```

---

## Phase 4: Random Bay Position

### Objective
Bay can be positioned at different Y locations. Goal calculation must keep car centered in bay. B must always be calculated relative to bay.

### Requirements
1. Bay center_y can be randomized
2. Goal X always centered (formula already generic)
3. B calculated relative to bay (B_y = bay_y - 0.5)
4. Neighbor obstacles move with bay

### Implementation Steps

#### Step 4.1: Add Bay Randomization Config
```yaml
# config_env.yaml
parallel:
  parking:
    bay:
      center_y_min: -0.3    # NEW: Random Y range
      center_y_max: 0.3     # NEW: Keep within reasonable bounds
      yaw: 1.5708          # Keep fixed (perpendicular)
```

#### Step 4.2: Update Bay Sampling
```python
# env/parking_env.py - _sample_bay_and_obstacles()
def _sample_bay_and_obstacles(self):
    bay_cfg = self.parking_cfg.get("bay", {})

    # Center X (always 0 for now)
    cx = 0.0

    # Center Y - RANDOMIZE if range specified
    center_y_min = bay_cfg.get("center_y_min", None)
    center_y_max = bay_cfg.get("center_y_max", None)

    if center_y_min is not None and center_y_max is not None:
        center_y = np.random.uniform(center_y_min, center_y_max)
    else:
        center_y = float(bay_cfg.get("center_y", 0.0))

    yaw = float(bay_cfg.get("yaw", 0.0))

    self.bay_center = np.array([cx, center_y, yaw], dtype=float)

    # Calculate goal (existing code - NO CHANGES NEEDED)
    L = float(self.vehicle_params.get("length", 0.36))
    dist_to_center = L / 2.0 - 0.05

    if abs(yaw) < 0.3:  # Parallel
        goal_x = cx - dist_to_center * np.cos(yaw)
        goal_y = center_y - dist_to_center * np.sin(yaw)
    else:  # Perpendicular
        # ... existing perpendicular code ...
        pass

    goal_yaw = yaw
    self.goal = np.array([goal_x, goal_y, goal_yaw], dtype=float)

    # Update obstacles to match bay position
    self.obstacles.set_goal(self.goal)
    self.obstacles.reset_to_base()
```

#### Step 4.3: Update B Calculation in Staged Controller
```python
# mpc/staged_controller.py - reset() method
def reset(self, env):
    """Called at start of each episode with new environment state."""

    # Get current bay configuration from environment
    bay_center = env.bay_center  # [x, y, yaw]

    # Recalculate B based on actual bay position
    self.goal_B = self._calculate_optimal_B(
        spawn_state=env.state,
        bay_center=bay_center
    )

    # Reset stage
    self.stage = "A_to_B"
    self.wait_steps = 0

    # Reset controllers
    self.approach_mpc.reset_warm_start()
    self.parking_controller.reset()
```

#### Step 4.4: Update Episode Loop to Call Reset
```python
# mpc/generate_expert_data.py - run_episode()
def run_episode(env, controller, max_steps):
    obs = env.reset(randomize=True)  # Randomizes bay position

    # IMPORTANT: Reset controller with new env state
    controller.reset(env)  # NEW: Pass env to get bay position

    # ... rest of episode loop ...
```

#### Step 4.5: Test Random Bay
```bash
# Test with random bay positions
python3 -m mpc.generate_expert_data --scenario parallel --hybrid --episodes 20

# Verify:
# - Goals are at different Y positions
# - Car always centers in bay (X alignment)
# - B point moves with bay (always 0.5m offset)
```

**Success Criteria:**
- Success rate >85%
- Car always centered in bay regardless of bay_center_y
- No collisions with bay boundaries

---

## Phase 5: Random Neighbor Jitter

### Objective
Neighbor cars have random position jitter (±5cm). Ego must adapt and maintain collision-free parking.

### Implementation Steps

#### Step 5.1: Enable Neighbor Jitter
```yaml
# config_env.yaml
parallel:
  obstacles:
    neighbor:
      w: 0.36
      h: 0.26
      offset: 0.40
      pos_jitter: 0.05  # NEW: ±5cm random jitter per neighbor
```

#### Step 5.2: Update Obstacle Manager
```python
# env/obstacle_manager.py - reset_to_base()
def reset_to_base(self):
    # ... existing code for base obstacles ...

    # Add neighbors with jitter
    neighbor_cfg = self.config.get("neighbor", {})
    offset = neighbor_cfg.get("offset", 0.40)
    jitter = neighbor_cfg.get("pos_jitter", 0.0)

    # Left neighbor
    left_x = -offset + np.random.uniform(-jitter, jitter)
    left_y = goal_y + np.random.uniform(-jitter, jitter)
    self.obstacles.append({
        "type": "rect",
        "x": left_x,
        "y": left_y,
        "w": neighbor_cfg.get("w", 0.36),
        "h": neighbor_cfg.get("h", 0.26)
    })

    # Right neighbor
    right_x = offset + np.random.uniform(-jitter, jitter)
    right_y = goal_y + np.random.uniform(-jitter, jitter)
    self.obstacles.append({
        "type": "rect",
        "x": right_x,
        "y": right_y,
        "w": neighbor_cfg.get("w", 0.36),
        "h": neighbor_cfg.get("h", 0.26)
    })
```

#### Step 5.3: Test Neighbor Jitter
```bash
# Test with jittered neighbors
python3 -m mpc.generate_expert_data --scenario parallel --hybrid --episodes 20

# Check collision rate - should be <5%
```

**Success Criteria:**
- No collision increase
- Parking quality maintained
- System adapts to tighter/wider gaps

---

## Phase 6: Random Obstacles in Environment

### Objective
Add random obstacles (not just neighbors) in the environment. Planner must avoid them while parking.

### Implementation Steps

#### Step 6.1: Enable Random Obstacles
```yaml
# config_env.yaml
parallel:
  obstacles:
    random:
      num_min: 0
      num_max: 2            # NEW: Up to 2 random obstacles
      x_range: [-1.5, 1.5]  # NEW: Can appear anywhere
      y_range: [-1.5, 1.5]
      w_range: [0.15, 0.30] # NEW: Size range
      h_range: [0.15, 0.30]
      min_distance_to_goal: 0.4  # NEW: Don't spawn too close to goal
      min_distance_to_spawn: 0.3 # NEW: Don't block spawn
```

#### Step 6.2: Update Obstacle Manager
```python
# env/obstacle_manager.py - reset_to_base()
def reset_to_base(self):
    # ... existing neighbors code ...

    # Add random obstacles
    random_cfg = self.config.get("random", {})
    num_min = random_cfg.get("num_min", 0)
    num_max = random_cfg.get("num_max", 0)
    num_random = np.random.randint(num_min, num_max + 1)

    for _ in range(num_random):
        # Sample position with constraints
        attempts = 0
        while attempts < 50:
            x = np.random.uniform(*random_cfg["x_range"])
            y = np.random.uniform(*random_cfg["y_range"])

            # Check minimum distance constraints
            dist_to_goal = np.hypot(x - goal[0], y - goal[1])
            # TODO: Check dist_to_spawn as well

            if dist_to_goal > random_cfg.get("min_distance_to_goal", 0.4):
                break
            attempts += 1

        # Add obstacle
        w = np.random.uniform(*random_cfg["w_range"])
        h = np.random.uniform(*random_cfg["h_range"])
        self.obstacles.append({
            "type": "rect",
            "x": x,
            "y": y,
            "w": w,
            "h": h
        })
```

#### Step 6.3: Test Random Obstacles
```bash
# Test with random obstacles
python3 -m mpc.generate_expert_data --scenario parallel --hybrid --episodes 30

# Check:
# - Success rate >80% (some failures expected with hard scenarios)
# - No collisions with random obstacles
# - Planner finds feasible paths
```

**Success Criteria:**
- Success rate >80%
- Collision avoidance working
- Planner doesn't get stuck

---

## Phase 7: Baseline MPC Comparison

### Objective
Implement and compare simple baseline MPC (no TEB, just receding horizon) to show benefit of hybrid TEB+MPC approach.

### Implementation Steps

#### Step 7.1: Create Baseline Controller
```python
# mpc/baseline_controller.py - NEW FILE
class BaselineMPCController:
    """
    Simple receding-horizon MPC baseline.
    No TEB planning, no staged control.
    Just direct MPC from spawn to goal.
    """

    def __init__(self, config_path, env_cfg, dt):
        self.mpc = TEBMPC(config_path=config_path, env_cfg=env_cfg, dt=dt)

    def reset(self):
        self.mpc.reset_warm_start()

    def get_control(self, state, goal, obstacles):
        # Simple: solve MPC directly to goal
        sol = self.mpc.solve(
            state=state,
            goal=goal,
            obstacles=obstacles,
            profile="parallel"
        )
        return sol.controls[0] if sol.success else np.zeros(2)
```

#### Step 7.2: Add Baseline Mode to Data Generator
```python
# mpc/generate_expert_data.py
def create_controller(args, env_cfg):
    if args.baseline:
        # NEW: Baseline receding MPC
        from mpc.baseline_controller import BaselineMPCController
        return BaselineMPCController(
            config_path="mpc/config_mpc.yaml",
            env_cfg=env_cfg,
            dt=env_cfg["dt"]
        )
    elif args.hybrid:
        # Existing: Staged + Hybrid
        # ... existing code ...
        pass
```

#### Step 7.3: Run Comparison
```bash
# Generate data with hybrid controller (current)
python3 -m mpc.generate_expert_data --scenario parallel --hybrid --episodes 50
# Results saved to data/expert_parallel/

# Generate data with baseline MPC
python3 -m mpc.generate_expert_data --scenario parallel --baseline --episodes 50
# Results saved to data/baseline_parallel/

# Compare metrics
python3 analyze_comparison.py
```

#### Step 7.4: Metrics to Compare
```python
# analyze_comparison.py - NEW FILE
import glob
import pickle
import numpy as np

def analyze_controller(data_dir, name):
    episodes = glob.glob(f"{data_dir}/episode_*.pkl")

    success_count = len(episodes)
    steps = []
    final_errors = []
    oscillation_count = []

    for ep_file in episodes:
        with open(ep_file, 'rb') as f:
            data = pickle.load(f)

        steps.append(len(data["states"]))
        final_errors.append(data["final_pos_err"])

        # Count direction changes as proxy for oscillations
        velocities = [s[3] for s in data["states"]]
        direction_changes = sum(1 for i in range(1, len(velocities))
                               if velocities[i] * velocities[i-1] < 0)
        oscillation_count.append(direction_changes)

    print(f"\n{name} Controller:")
    print(f"  Success rate: {success_count}/50 = {100*success_count/50:.1f}%")
    print(f"  Avg steps: {np.mean(steps):.1f} ± {np.std(steps):.1f}")
    print(f"  Avg final error: {np.mean(final_errors):.3f}m ± {np.std(final_errors):.3f}")
    print(f"  Avg direction changes: {np.mean(oscillation_count):.1f}")

analyze_controller("data/expert_parallel", "Hybrid TEB+MPC")
analyze_controller("data/baseline_parallel", "Baseline MPC")
```

**Expected Results:**
- Hybrid: Higher success rate, fewer oscillations, smoother trajectories
- Baseline: Lower success rate, more oscillations, longer episodes

---

## Testing Strategy

### Unit Tests (Per Phase)

After each phase:
```bash
# Test 20 episodes
python3 -m mpc.generate_expert_data --scenario parallel --hybrid --episodes 20

# Check success rate
SUCCESS=$(ls data/expert_parallel/episode_*.pkl | wc -l)
FAILS=$(ls data/expert_parallel_debug/fail_*.pkl | wc -l)
TOTAL=$((SUCCESS + FAILS))
RATE=$((100 * SUCCESS / TOTAL))
echo "Success rate: ${RATE}%"

# Validate rate >85%
if [ $RATE -lt 85 ]; then
    echo "FAILED: Success rate too low"
    exit 1
fi
```

### Integration Test (All Phases Combined)

```bash
# Full randomization test
# - Random spawn
# - Random bay position
# - Random neighbor jitter
# - Random obstacles
python3 -m mpc.generate_expert_data --scenario parallel --hybrid --episodes 100

# Target: >80% success rate with full randomization
```

### Regression Test (Compare to Phase 1)

```bash
# Test with fixed configuration (like Phase 1)
# Should still achieve 90%+ success
python3 -m mpc.generate_expert_data --scenario parallel --hybrid --episodes 20 --fixed

# Ensures we didn't break the working case
```

---

## Timeline (Today)

```
Hour 1-2:   Phase 2 - Random Spawn A
            - Implement spawn randomization
            - Increase B tolerance
            - Test 20 episodes
            - Validate success rate >85%

Hour 3:     Phase 3 - Multiple B Points
            - Verify B formula is generic
            - Test with wide spawn range
            - Document approach strategy

Hour 4-5:   Phase 4 - Random Bay Position
            - Add bay randomization config
            - Update episode loop to reset controller
            - Test with random bay Y
            - Validate goal centering

Hour 6:     Phase 5 - Random Neighbor Jitter
            - Enable neighbor jitter
            - Test collision rate
            - Tune if needed

Hour 7:     Phase 6 - Random Obstacles
            - Implement random obstacle spawning
            - Test with 2-3 random obstacles
            - Check collision avoidance

Hour 8:     Phase 7 - Baseline MPC
            - Implement baseline controller
            - Run comparison (50 episodes each)
            - Analyze metrics

Hour 9:     Integration & Testing
            - Full randomization test (100 episodes)
            - Regression test
            - Bug fixes if needed

Hour 10:    Documentation & Cleanup
            - Update handover document
            - Git commit all phases
            - Final validation
```

---

## Success Criteria Summary

| Phase | Success Rate | Key Metric | Notes |
|-------|-------------|------------|-------|
| 2: Random Spawn | >85% | Handles varied spawn | B tolerance ↑ |
| 3: Multiple B | >85% | Works from any approach | B formula generic |
| 4: Random Bay | >85% | Goal always centered | Bay Y randomized |
| 5: Neighbor Jitter | >85% | No collision increase | ±5cm jitter |
| 6: Random Obstacles | >80% | Avoids all obstacles | Up to 3 obstacles |
| 7: Baseline | Compare | Shows hybrid benefit | Oscillation metric |
| Integration | >80% | Full randomization | All phases enabled |

---

## Git Commit Strategy

After each phase:
```bash
git add <relevant files>
git commit -m "Phase X: <description>

- <bullet points of changes>
- <test results>
- <success rate achieved>"

# Final commit after all phases:
git commit -m "All Phases Complete: Full randomization + baseline

Phase 2: Random spawn (87% success)
Phase 3: Multiple B points (generic formula)
Phase 4: Random bay position (goal centering verified)
Phase 5: Neighbor jitter (no collision increase)
Phase 6: Random obstacles (82% success)
Phase 7: Baseline comparison (hybrid 38% better)

Integration test: 83% success with full randomization
System ready for RL training"
```

---

## Troubleshooting by Phase

### Phase 2: Random Spawn Issues

**Problem:** Can't reach B from some spawn positions

**Solution:**
- Increase X range: x_min=-1.5 → -1.8
- Reduce jitter: yaw_jitter=0.15 → 0.10
- Check A→B MPC horizon: may need N=20 → 30 for longer paths

**Problem:** Success rate drops below 85%

**Solution:**
- Increase B tolerance further: 0.25 → 0.30
- Reduce spawn randomization range temporarily
- Check specific failure modes (collision vs max_steps)

### Phase 4: Random Bay Issues

**Problem:** Goal not centered in bay

**Solution:**
- Verify goal calculation formula (line 85-93 in parking_env.py)
- Check bay_center is passed correctly
- Print debug: goal position vs bay position

**Problem:** B point incorrect for new bay position

**Solution:**
- Ensure controller.reset(env) is called after env.reset()
- Verify B calculation uses env.bay_center
- Check B_y = bay_y - 0.5 formula

### Phase 6: Random Obstacles Issues

**Problem:** Obstacles block path, no solution found

**Solution:**
- Reduce max obstacles: num_max=3 → 2
- Increase min_distance_to_goal: 0.4 → 0.6
- Add min_distance_to_path constraint

**Problem:** Collision rate increases

**Solution:**
- Increase obs_inflate: 0.01 → 0.02
- Increase collision weight: 33.0 → 35.0
- Check obstacle size range (may be too large)

---

## End of Implementation Plan

All phases are designed to build incrementally on the stable Phase 1 foundation. Each phase can be tested independently before proceeding to the next.
