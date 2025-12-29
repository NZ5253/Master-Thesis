# Hybrid TEB-MPC Solution - The Perfect Trajectory Approach

**Date**: 2025-12-29
**Concept**: Use TEB as local planner to create committed trajectories, then MPC tracks them

## Your Key Insight

> "is there a way where TEB works as local planner and creates that perfect trajectory where can complete control steerings for particular required time and so and MPC either follows it or a hybrid TEB-MPC approach"

**This is EXACTLY the right solution!** This combines the best of both worlds:
- TEB's ability to create time-optimal trajectories with committed maneuvers
- MPC's ability to track trajectories with obstacle avoidance and disturbance rejection

## Three Hybrid Approaches

### Approach 1: TEB Planner + MPC Tracker (Recommended)

**Architecture**:
```
State → TEB Local Planner → Reference Trajectory → MPC Tracker → Controls
        (every 1-2s)         (committed segments)    (every 0.1s)
```

**How it works**:
1. **TEB runs once** at the start of each parking phase (approach, entry, final)
2. **TEB optimizes entire trajectory** to goal with time-elastic bands
3. **TEB output**: Reference trajectory with (x, y, θ, v, steering, dt) at each waypoint
4. **MPC tracks reference** every 0.1s, handling disturbances and obstacles

**Benefits**:
- ✅ TEB creates globally optimal, committed maneuvers
- ✅ MPC provides real-time tracking and robustness
- ✅ Best of both: planning + control separation
- ✅ Can re-plan if trajectory becomes invalid

**Implementation**:
```python
class HybridTEBMPC:
    def __init__(self):
        self.teb_planner = TEBPlanner()  # Global trajectory optimization
        self.mpc_tracker = MPCTracker()   # Local trajectory tracking
        self.reference_trajectory = None
        self.trajectory_valid_time = 2.0  # Re-plan every 2 seconds

    def solve(self, state, goal, obstacles):
        # Check if we need new reference trajectory
        if self._need_replan(state):
            # TEB creates committed trajectory
            self.reference_trajectory = self.teb_planner.plan(
                state, goal, obstacles,
                horizon=80,           # Full horizon to goal
                enable_teb=True,      # Variable dt
                w_time=10.0,          # Minimize time
                w_dt_precision=5.0    # Small dt when precise
            )

        # MPC tracks current segment of reference
        ref_segment = self._get_reference_segment(state)
        control = self.mpc_tracker.track(
            state, ref_segment, obstacles,
            horizon=50,           # Shorter horizon
            enable_teb=False      # Fixed dt for stability
        )

        return control
```

**TEB Planner Configuration** (runs every 1-2s):
```yaml
teb_planner:
  enable: true
  horizon: 80              # Plan to goal
  dt_min: 0.08
  dt_max: 0.30             # Allow committed maneuvers

  # Aggressive temporal optimization
  w_time: 10.0             # Minimize total time
  w_dt_smooth: 5.0         # Smooth transitions
  w_dt_precision: 5.0      # Small dt when close
  w_velocity_dt_coupling: 3.0  # Couple speed and dt

  # Standard cost weights
  w_goal_xy: 400.0
  w_goal_theta: 120.0
  # ... etc
```

**MPC Tracker Configuration** (runs every 0.1s):
```yaml
mpc_tracker:
  enable: false            # Fixed-dt for stability
  horizon: 50              # Shorter horizon
  dt: 0.10                 # Fixed 100ms

  # Trajectory tracking weights
  w_tracking_xy: 500.0     # Follow reference position
  w_tracking_theta: 150.0  # Follow reference heading
  w_tracking_v: 10.0       # Follow reference velocity

  # Standard weights (lower priority)
  w_goal_xy: 100.0         # Weak goal attraction
  w_goal_theta: 30.0       # Weak yaw alignment
  # ... etc
```

**Key Changes to Cost Function**:
```python
# MPC Tracker objective (in teb_mpc.py)
for k in range(N):
    st = X[:, k]

    # Get reference state at this time step
    ref = reference_trajectory[current_step + k]

    # PRIMARY: Track reference trajectory
    obj += self.w_tracking_xy * ((st[0] - ref.x)**2 + (st[1] - ref.y)**2)
    obj += self.w_tracking_theta * (st[2] - ref.theta)**2
    obj += self.w_tracking_v * (st[3] - ref.v)**2

    # SECONDARY: Goal attraction (weak, for safety)
    obj += self.w_goal_xy * ((st[0] - goal_x)**2 + (st[1] - goal_y)**2)
    obj += self.w_goal_theta * (st[2] - goal_theta)**2

    # Standard collision, smoothness, etc.
    # ... (unchanged)
```

---

### Approach 2: Phase-Aware TEB-MPC

**Architecture**:
```
State → Phase Detector → TEB (far) OR Fixed-dt MPC (close)
                         ↓                    ↓
                   Committed steering   Precision tracking
```

**How it works**:
1. **Far from goal** (>30cm): Use TEB with large dt_max for committed maneuvers
2. **Close to goal** (<30cm): Switch to fixed-dt MPC for precision

**Benefits**:
- ✅ Leverages TEB for gross motion
- ✅ Uses stable fixed-dt for final precision
- ✅ Simple to implement (just switch solvers)

**Implementation**:
```python
def solve(self, state, goal, obstacles):
    goal_dist = np.sqrt((state.x - goal.x)**2 + (state.y - goal.y)**2)

    if goal_dist > 0.30:  # Far from goal - use TEB for committed maneuvers
        return self.solver_teb.solve(
            state, goal, obstacles,
            enable_teb=True,
            dt_max=0.25,           # Large dt allowed
            w_time=5.0,            # Time pressure
            w_dt_precision=0.0     # Don't need precision yet
        )
    else:  # Close to goal - use fixed-dt for precision
        return self.solver_fixed.solve(
            state, goal, obstacles,
            enable_teb=False,
            dt=0.10                # Fixed dt
        )
```

**Configuration**:
```yaml
mpc:
  phase_threshold: 0.30  # Switch at 30cm from goal

  # TEB phase (far from goal)
  teb_phase:
    enable: true
    dt_min: 0.08
    dt_max: 0.25           # Allow committed maneuvers
    w_time: 5.0
    w_dt_smooth: 3.0
    w_dt_precision: 0.0    # Disabled in far phase

  # Fixed-dt phase (close to goal)
  precision_phase:
    enable: false
    dt: 0.10
```

---

### Approach 3: Constrained TEB (Time-Segmented Optimization)

**Architecture**:
```
State → TEB with temporal segments → Controls
        (constrained dt regions)
```

**How it works**:
1. **Divide horizon into segments** based on distance to goal
2. **Constrain dt per segment**:
   - Far segment (>50cm): dt ∈ [0.15, 0.30] (committed maneuvers)
   - Mid segment (20-50cm): dt ∈ [0.10, 0.15] (transition)
   - Near segment (<20cm): dt ∈ [0.08, 0.10] (precision)
3. **Single TEB optimization** with segment-wise dt bounds

**Benefits**:
- ✅ Single unified solver
- ✅ Automatic transition between phases
- ✅ Committed maneuvers far, precision close

**Implementation**:
```python
# In _build_solver() method
for k in range(N):
    st = X[:, k]
    goal_dist = ca.sqrt((st[0] - goal_x)**2 + (st[1] - goal_y)**2)

    # Segment-wise dt bounds (symbolic)
    # Far: dt ∈ [0.15, 0.30]
    # Mid: dt ∈ [0.10, 0.15]
    # Near: dt ∈ [0.08, 0.10]

    # Use conditional constraints (soft or hard)
    far_phase = ca.if_else(goal_dist > 0.50, 1.0, 0.0)
    mid_phase = ca.if_else(ca.logic_and(goal_dist >= 0.20, goal_dist <= 0.50), 1.0, 0.0)
    near_phase = ca.if_else(goal_dist < 0.20, 1.0, 0.0)

    # Soft constraints on dt bounds per phase
    obj += far_phase * 10.0 * ca.fmax(0, 0.15 - DT[k])**2  # Penalty if dt < 0.15 when far
    obj += near_phase * 10.0 * ca.fmax(0, DT[k] - 0.10)**2  # Penalty if dt > 0.10 when near
```

**Configuration**:
```yaml
teb:
  enable: true
  dt_min: 0.08
  dt_max: 0.30

  # Segment-wise dt penalties
  w_dt_segment_violation: 10.0

  # Segment thresholds
  far_threshold: 0.50   # >50cm: encourage large dt
  near_threshold: 0.20  # <20cm: enforce small dt
```

---

## Comparison of Approaches

| Aspect | TEB Planner + MPC Tracker | Phase-Aware Switch | Constrained TEB |
|--------|---------------------------|-------------------|-----------------|
| **Complexity** | High (2 solvers) | Medium (2 solvers, simple switch) | Low (1 solver, complex constraints) |
| **Committed Maneuvers** | ✅✅ Excellent | ✅ Good | ✅ Good |
| **Precision** | ✅✅ Excellent | ✅✅ Excellent | ✅ Good |
| **Robustness** | ✅✅ Best (MPC handles disturbances) | ✅ Good | ✅ Good |
| **Re-planning** | ✅ Adaptive | ⚠️ Hard switch | ✅ Smooth |
| **Implementation Effort** | High (~3-4 days) | Low (~1 day) | Medium (~2 days) |
| **Oscillation Risk** | ❌ Very low | ❌ Low | ⚠️ Medium |

## Recommended Approach: TEB Planner + MPC Tracker

**Why this is best**:

1. **Separation of concerns**:
   - TEB handles global planning (what trajectory to follow)
   - MPC handles local control (how to follow it robustly)

2. **Committed maneuvers**:
   - TEB optimizes entire trajectory at once
   - Can create segments like "steer full lock for 0.8s"
   - No myopic re-optimization every step

3. **Robust execution**:
   - MPC tracks reference but adapts to disturbances
   - Handles unexpected obstacles
   - Smooth tracking with fixed-dt (no TEB oscillations)

4. **Adaptive re-planning**:
   - If trajectory becomes invalid (obstacles move), re-plan with TEB
   - Otherwise just track

5. **Best results**:
   - Expect 2-3 gear changes (from TEB planning)
   - 1 or 0 oscillations (from MPC tracking)
   - Human-like committed steering

## Implementation Steps

### Step 1: Create TEB Planner Module
```python
# mpc/teb_planner.py
class TEBPlanner:
    def __init__(self, config):
        # Build TEB solver with aggressive temporal optimization
        self.solver = self._build_teb_solver(
            dt_min=0.08,
            dt_max=0.30,      # Large dt for committed maneuvers
            w_time=10.0,      # Strong time minimization
            w_dt_precision=5.0 # Precision when close
        )

    def plan(self, state, goal, obstacles, horizon=80):
        """Generate reference trajectory from current state to goal"""
        # Run TEB optimization over full horizon
        solution = self.solver.solve(state, goal, obstacles)

        # Extract reference trajectory
        reference = []
        for k in range(horizon):
            reference.append({
                'x': solution.X[k, 0],
                'y': solution.X[k, 1],
                'theta': solution.X[k, 2],
                'v': solution.X[k, 3],
                'steering': solution.U[k, 0],
                'accel': solution.U[k, 1],
                'dt': solution.DT[k],  # Time-elastic
                'time': np.sum(solution.DT[:k])  # Cumulative time
            })

        return ReferenceTrajectory(reference)
```

### Step 2: Modify MPC for Trajectory Tracking
```python
# mpc/mpc_tracker.py
class MPCTracker:
    def __init__(self, config):
        # Build fixed-dt MPC with tracking objective
        self.solver = self._build_tracking_solver(
            dt=0.10,          # Fixed dt for stability
            horizon=50,       # Shorter horizon
            enable_teb=False
        )

    def track(self, state, reference_trajectory, obstacles):
        """Track reference trajectory segment"""
        # Extract reference for horizon
        ref_segment = reference_trajectory.get_segment(
            start_time=current_time,
            horizon=50,
            dt=0.10
        )

        # Solve MPC with tracking objective
        solution = self.solver.solve(state, ref_segment, obstacles)

        return solution.U[0]  # First control action
```

### Step 3: Modify Cost Function
```python
# In _build_solver() for MPC Tracker
def _build_tracking_solver(self, ...):
    # ... (standard setup)

    obj = 0
    for k in range(N):
        st = X[:, k]

        # PRIMARY: Track reference trajectory
        ref = P[ref_offset + k*ref_state_size : ref_offset + (k+1)*ref_state_size]
        ref_x, ref_y, ref_theta, ref_v = ref[0], ref[1], ref[2], ref[3]

        tracking_err_xy = (st[0] - ref_x)**2 + (st[1] - ref_y)**2
        tracking_err_theta = (st[2] - ref_theta)**2
        tracking_err_v = (st[3] - ref_v)**2

        obj += self.w_tracking_xy * tracking_err_xy
        obj += self.w_tracking_theta * tracking_err_theta
        obj += self.w_tracking_v * tracking_err_v

        # SECONDARY: Goal attraction (weak, for safety margin)
        goal_err_xy = (st[0] - goal_x)**2 + (st[1] - goal_y)**2
        goal_err_theta = (st[2] - goal_theta)**2

        obj += self.w_goal_xy * goal_err_xy  # Much lower weight
        obj += self.w_goal_theta * goal_err_theta

        # Standard collision, smoothness, etc.
        # ... (unchanged)
```

### Step 4: Hybrid Controller
```python
# mpc/hybrid_controller.py
class HybridTEBMPCController:
    def __init__(self):
        self.teb_planner = TEBPlanner(config)
        self.mpc_tracker = MPCTracker(config)
        self.reference_trajectory = None
        self.last_plan_time = 0
        self.replan_interval = 1.0  # Re-plan every 1 second

    def solve(self, state, goal, obstacles):
        current_time = time.time()

        # Check if we need to re-plan
        need_replan = (
            self.reference_trajectory is None or
            current_time - self.last_plan_time > self.replan_interval or
            self._trajectory_invalid(state, obstacles)
        )

        if need_replan:
            print("[HYBRID] Re-planning with TEB...")
            self.reference_trajectory = self.teb_planner.plan(
                state, goal, obstacles
            )
            self.last_plan_time = current_time

        # Track reference with MPC
        control = self.mpc_tracker.track(
            state, self.reference_trajectory, obstacles
        )

        return control
```

## Expected Performance

### Current Fixed-dt MPC Baseline
```
Steps: 55
Precision: 2.6cm
Oscillations: 1
Pattern: Approach → drift → correct → goal
```

### Hybrid TEB-MPC (Expected)
```
Steps: 40-45 (20% fewer)
Precision: 2.0cm (better)
Oscillations: 0-1 (same or better)
Pattern: Committed maneuver → smooth tracking → goal
```

**Key improvements**:
- ✅ Fewer steps (TEB optimizes global trajectory)
- ✅ Committed maneuvers (TEB creates full-lock segments)
- ✅ Smooth execution (MPC tracks with fixed-dt)
- ✅ Better precision (both planners working together)
- ✅ No extra oscillations (MPC is stable tracker)

## Alternative: Quick Win with Phase-Aware Switch

If hybrid planner-tracker is too complex, **Phase-Aware Switch** is fastest to implement:

```python
def solve(self, state, goal, obstacles):
    goal_dist = np.sqrt((state.x - goal.x)**2 + (state.y - goal.y)**2)

    if goal_dist > 0.30:
        # Far: Use TEB for committed maneuvers
        solver = self.solver_teb
        profile = "parallel_teb"
    else:
        # Close: Use fixed-dt for precision
        solver = self.solver_fixed
        profile = "parallel_fixed"

    return solver.solve(state, goal, obstacles, profile)
```

**Configuration**:
```yaml
profiles:
  parallel_teb:  # Used when far from goal
    enable_teb: true
    dt_max: 0.25
    w_time: 5.0
    # ... aggressive settings

  parallel_fixed:  # Used when close to goal
    enable_teb: false
    dt: 0.10
    # ... precision settings
```

**Effort**: ~1 day
**Expected improvement**: 10-15% fewer steps, similar oscillations

## Recommendation

### Short-term (1-2 days)
Implement **Phase-Aware Switch** to validate concept:
- Quick to implement
- Low risk
- Immediate feedback on TEB for committed maneuvers

### Long-term (3-4 days)
Implement **TEB Planner + MPC Tracker** for optimal performance:
- Best architectural separation
- Most flexible and robust
- Industry-standard approach (Tesla, Waymo use similar)

## Files to Create/Modify

### New Files
- `mpc/teb_planner.py` - TEB global planner
- `mpc/mpc_tracker.py` - MPC trajectory tracker
- `mpc/hybrid_controller.py` - Orchestrates planner + tracker
- `mpc/reference_trajectory.py` - Reference trajectory data structure

### Modified Files
- `mpc/teb_mpc.py` - Add tracking mode support
- `mpc/config_mpc.yaml` - Add tracker configuration
- `mpc/generate_expert_data.py` - Use hybrid controller

### Configuration Example
```yaml
mpc:
  mode: "hybrid"  # Options: "fixed", "teb", "hybrid"

  hybrid:
    planner: "teb"        # Use TEB for global planning
    tracker: "mpc_fixed"  # Use fixed-dt MPC for tracking
    replan_interval: 1.0  # Re-plan every 1 second

    teb_planner:
      horizon: 80
      dt_min: 0.08
      dt_max: 0.30
      w_time: 10.0
      w_dt_precision: 5.0
      # ... other TEB settings

    mpc_tracker:
      horizon: 50
      dt: 0.10
      w_tracking_xy: 500.0
      w_tracking_theta: 150.0
      w_goal_xy: 100.0  # Weak goal attraction
      # ... other tracker settings
```

## Conclusion

Your insight about using TEB as a local planner is **exactly right**. The hybrid approach:
- ✅ Uses TEB for what it's good at: global trajectory optimization with committed maneuvers
- ✅ Uses MPC for what it's good at: robust tracking with disturbance rejection
- ✅ Avoids TEB's oscillation problem by using fixed-dt MPC for tracking
- ✅ Achieves committed steering without re-optimization oscillations

This is the proper way to eliminate zig-zag while maintaining robustness.
