# Implementation Plan: TEB Single-Shot + MPC Multi-Phase Tracking

**Goal**: Achieve human-like parking with committed maneuvers and NO zig-zag

**Architecture**: TEB plans ONCE (local optimization) → MPC tracks with multi-phase awareness

## Core Concept

### TEB's Role (Local Planner - Runs Once)
- **When**: At episode start (spawn position)
- **What**: Optimizes ENTIRE local trajectory from spawn to goal
- **How**: With variable dt, strong time minimization, precision-aware scaling
- **Output**: Reference trajectory with committed steering segments

### MPC's Role (Tracker - Runs Every Step)
- **When**: Every control step (0.1s)
- **What**: Tracks reference trajectory from TEB
- **How**: Phase-aware tracking weights (approach/entry/final)
- **Output**: Controls that follow reference + adapt to disturbances

## Why This Eliminates Zig-Zag

**Current problem**: MPC re-optimizes every step → myopic decisions → oscillations

**Solution**:
- TEB creates committed path ONCE (e.g., "steer full lock for 2s")
- MPC just tracks it → no re-planning → no oscillations
- Multi-phase weights ensure smooth tracking in each phase

## Implementation Steps

### Phase 1: Add TEB Single-Shot Planner (Day 1)

#### 1.1 Create Reference Trajectory Class
**File**: `mpc/reference_trajectory.py`

```python
from dataclasses import dataclass
from typing import List
import numpy as np

@dataclass
class ReferenceState:
    """Single state in reference trajectory"""
    x: float
    y: float
    theta: float
    v: float
    steering: float
    accel: float
    dt: float
    time: float  # Cumulative time

class ReferenceTrajectory:
    """Reference trajectory from TEB planner"""

    def __init__(self, states: List[ReferenceState]):
        self.states = states
        self.total_time = states[-1].time if states else 0.0

    def get_state_at_step(self, step: int) -> ReferenceState:
        """Get reference state at given step"""
        if step >= len(self.states):
            return self.states[-1]  # Return goal if beyond horizon
        return self.states[step]

    def get_segment(self, start_step: int, horizon: int) -> List[ReferenceState]:
        """Get segment of trajectory for MPC horizon"""
        end_step = min(start_step + horizon, len(self.states))
        return self.states[start_step:end_step]

    def distance_to_goal(self, step: int) -> float:
        """Distance from step to final goal"""
        if step >= len(self.states):
            return 0.0
        goal = self.states[-1]
        state = self.states[step]
        return np.sqrt((state.x - goal.x)**2 + (state.y - goal.y)**2)
```

#### 1.2 Modify TEB-MPC for Single-Shot Planning Mode
**File**: `mpc/teb_mpc.py`

Add configuration flag:
```python
def __init__(self, config_path: str = "mpc/config_mpc.yaml"):
    # ... existing init ...

    # NEW: Mode selection
    self.mode = self.config.get('mode', 'tracking')  # 'planning' or 'tracking'

    if self.mode == 'planning':
        print("[TEB-MPC] Mode: PLANNING (single-shot global trajectory)")
        self._setup_planning_mode()
    elif self.mode == 'tracking':
        print("[TEB-MPC] Mode: TRACKING (follow reference trajectory)")
        self._setup_tracking_mode()

def _setup_planning_mode(self):
    """Configure TEB for single-shot trajectory planning"""
    # Use TEB with aggressive temporal optimization
    self.enable_teb = True
    self.dt_max = 0.30  # Allow committed maneuvers
    self.w_time = 10.0  # Strong time minimization
    self.w_dt_precision = 5.0  # Precision when close

def _setup_tracking_mode(self):
    """Configure MPC for trajectory tracking"""
    # Use fixed-dt for stable tracking
    self.enable_teb = False
    self.dt = 0.10
```

Add planning method:
```python
def plan_trajectory(self, state: VehicleState, goal: ParkingGoal,
                    obstacles: List[Obstacle]) -> ReferenceTrajectory:
    """
    TEB single-shot planning: Create reference trajectory from state to goal
    This runs ONCE at episode start
    """
    if self.mode != 'planning':
        raise ValueError("plan_trajectory() only available in planning mode")

    print(f"[TEB Planning] Creating reference trajectory from ({state.x:.2f}, {state.y:.2f}) to goal...")

    # Solve with TEB over full horizon
    solution = self.solve(state, goal, obstacles, profile="parallel")

    if not solution.success:
        print("[TEB Planning] WARNING: Planning failed, returning default trajectory")
        return self._create_default_trajectory(state, goal)

    # Extract reference trajectory
    reference_states = []
    cumulative_time = 0.0

    for k in range(len(solution.X)):
        if solution.dt_solution:
            dt_k = solution.dt_solution[k] if k < len(solution.dt_solution) else 0.1
        else:
            dt_k = 0.1

        reference_states.append(ReferenceState(
            x=solution.X[k, 0],
            y=solution.X[k, 1],
            theta=solution.X[k, 2],
            v=solution.X[k, 3],
            steering=solution.U[k, 0] if k < len(solution.U) else 0.0,
            accel=solution.U[k, 1] if k < len(solution.U) else 0.0,
            dt=dt_k,
            time=cumulative_time
        ))
        cumulative_time += dt_k

    print(f"[TEB Planning] Created trajectory: {len(reference_states)} steps, {cumulative_time:.2f}s total")
    return ReferenceTrajectory(reference_states)
```

#### 1.3 Update Configuration
**File**: `mpc/config_mpc.yaml`

```yaml
mpc:
  mode: "hybrid"  # Options: "tracking", "planning", "hybrid"

  # TEB Planning Mode (runs once at start)
  planning:
    teb:
      enable: true
      horizon: 100            # Plan to goal
      dt_min: 0.08
      dt_max: 0.30            # ALLOW committed maneuvers

      # Aggressive temporal optimization
      w_time: 10.0            # Minimize time → committed maneuvers
      w_dt_smooth: 5.0        # Smooth dt transitions
      w_dt_precision: 5.0     # Small dt when close to goal
      w_velocity_dt_coupling: 3.0  # Couple speed and dt

    profiles:
      parallel:
        w_goal_xy: 400.0
        w_goal_theta: 120.0
        lateral_weight: 0.25
        yaw_weight: 0.9
        proximity_exp_factor: 50.0
        # ... etc

  # MPC Tracking Mode (runs every step)
  tracking:
    teb:
      enable: false           # Fixed-dt for stable tracking
      dt: 0.10

    # NEW: Tracking weights (higher than goal weights)
    w_tracking_xy: 500.0      # Strong position tracking
    w_tracking_theta: 150.0   # Strong heading tracking
    w_tracking_v: 10.0        # Velocity tracking

    # Multi-phase tracking adjustments
    phases:
      approach:
        w_tracking_xy: 300.0
        w_tracking_theta: 100.0
        coupling_strength: 0.5

      entry:
        w_tracking_xy: 500.0
        w_tracking_theta: 200.0
        coupling_strength: 0.7  # Maintain committed steering

      final:
        w_tracking_xy: 800.0
        w_tracking_theta: 300.0
        coupling_strength: 0.9

    profiles:
      parallel:
        w_goal_xy: 100.0      # WEAK goal attraction (safety net)
        w_goal_theta: 30.0    # WEAK yaw alignment
        # ... etc (much lower weights than planning)
```

### Phase 2: Add MPC Tracking with Multi-Phase (Day 1-2)

#### 2.1 Add Tracking Cost Function
**File**: `mpc/teb_mpc.py` - Modify `_build_solver()`

```python
def _build_solver(self, parking_type: str = "parallel"):
    # ... existing setup ...

    # NEW: Reference trajectory parameters (if in tracking mode)
    if self.mode == 'tracking':
        # Add reference trajectory to parameter vector P
        # Format: [goal_x, goal_y, goal_theta, obs_data..., ref_traj...]
        ref_traj_size = 4 * self.N  # (x, y, theta, v) for each step
        REF = ca.SX.sym("ref_traj", ref_traj_size)
        P = ca.vertcat(P, REF)

    # ... existing cost function ...

    for k in range(N):
        st = X[:, k]

        if self.mode == 'tracking':
            # PRIMARY: Track reference trajectory
            ref_offset = goal_size + obs_size
            ref_idx = ref_offset + k * 4
            ref_x = P[ref_idx]
            ref_y = P[ref_idx + 1]
            ref_theta = P[ref_idx + 2]
            ref_v = P[ref_idx + 3]

            tracking_err_xy = (st[0] - ref_x)**2 + (st[1] - ref_y)**2
            tracking_err_theta = (st[2] - ref_theta)**2
            tracking_err_v = (st[3] - ref_v)**2

            obj += self.w_tracking_xy * tracking_err_xy
            obj += self.w_tracking_theta * tracking_err_theta
            obj += self.w_tracking_v * tracking_err_v

            # SECONDARY: Weak goal attraction (safety net)
            goal_err_xy = (st[0] - goal_x)**2 + (st[1] - goal_y)**2
            obj += self.w_goal_xy * goal_err_xy  # Much lower weight
        else:
            # PLANNING mode: standard goal-based cost
            goal_err_xy = (st[0] - goal_x)**2 + (st[1] - goal_y)**2
            obj += self.w_goal_xy * goal_err_xy

        # ... rest of cost function (collision, smoothness, etc.)
```

#### 2.2 Add Multi-Phase Tracking
**File**: `mpc/teb_mpc.py`

```python
def _detect_phase_from_reference(self, current_step: int,
                                 reference: ReferenceTrajectory) -> ParkingPhase:
    """Detect parking phase based on reference trajectory"""
    goal_dist = reference.distance_to_goal(current_step)

    if goal_dist > 0.50:
        return ParkingPhase.APPROACH
    elif goal_dist > 0.15:
        return ParkingPhase.ENTRY
    else:
        return ParkingPhase.FINAL_ALIGN

def _apply_phase_tracking_weights(self, phase: ParkingPhase):
    """Apply phase-specific tracking weights"""
    phase_config = self.config['mpc']['tracking']['phases'][phase.value]

    self.w_tracking_xy = float(phase_config['w_tracking_xy'])
    self.w_tracking_theta = float(phase_config['w_tracking_theta'])
    self.coupling_entry = float(phase_config['coupling_strength'])

    print(f"[Tracking Phase: {phase.value}] xy={self.w_tracking_xy}, theta={self.w_tracking_theta}")

def track_trajectory(self, state: VehicleState, reference: ReferenceTrajectory,
                     current_step: int, obstacles: List[Obstacle]) -> MPCSolution:
    """
    MPC tracking: Follow reference trajectory with multi-phase awareness
    This runs EVERY control step
    """
    if self.mode != 'tracking':
        raise ValueError("track_trajectory() only available in tracking mode")

    # Detect phase from reference
    phase = self._detect_phase_from_reference(current_step, reference)

    # Apply phase-appropriate tracking weights
    self._apply_phase_tracking_weights(phase)

    # Get reference segment for horizon
    ref_segment = reference.get_segment(current_step, self.N)

    # Solve MPC with tracking objective
    solution = self.solve(state, reference.states[-1], obstacles,
                         profile="parallel", reference_segment=ref_segment)

    return solution
```

### Phase 3: Create Hybrid Controller (Day 2)

#### 3.1 Hybrid Controller
**File**: `mpc/hybrid_controller.py` (NEW)

```python
from mpc.teb_mpc import TEBMPCSolver, VehicleState, ParkingGoal, Obstacle
from mpc.reference_trajectory import ReferenceTrajectory
from typing import List
import numpy as np

class HybridTEBMPCController:
    """
    Hybrid controller combining TEB planning + MPC tracking

    Architecture:
    1. TEB plans reference trajectory ONCE at start
    2. MPC tracks reference with multi-phase awareness every step
    """

    def __init__(self, config_path: str = "mpc/config_mpc.yaml"):
        # Create two instances: one for planning, one for tracking
        self.planner = TEBMPCSolver(config_path)
        self.planner.mode = 'planning'
        self.planner._setup_planning_mode()

        self.tracker = TEBMPCSolver(config_path)
        self.tracker.mode = 'tracking'
        self.tracker._setup_tracking_mode()

        self.reference_trajectory = None
        self.current_step = 0

    def reset(self):
        """Reset for new episode"""
        self.reference_trajectory = None
        self.current_step = 0

    def plan(self, state: VehicleState, goal: ParkingGoal,
             obstacles: List[Obstacle]) -> ReferenceTrajectory:
        """Create reference trajectory using TEB (run once)"""
        print("[Hybrid] Planning reference trajectory with TEB...")
        self.reference_trajectory = self.planner.plan_trajectory(state, goal, obstacles)
        self.current_step = 0
        return self.reference_trajectory

    def step(self, state: VehicleState, obstacles: List[Obstacle]):
        """Track reference trajectory using MPC (run every step)"""
        if self.reference_trajectory is None:
            raise ValueError("Must call plan() before step()")

        # Track reference with multi-phase MPC
        solution = self.tracker.track_trajectory(
            state, self.reference_trajectory,
            self.current_step, obstacles
        )

        self.current_step += 1

        return solution
```

#### 3.2 Update Expert Data Generation
**File**: `mpc/generate_expert_data.py`

```python
from mpc.hybrid_controller import HybridTEBMPCController

# ... existing imports ...

def run_episode(env, controller, max_steps=200):
    """Run single episode with hybrid controller"""
    state = env.reset()
    controller.reset()

    # STEP 1: Plan reference trajectory (TEB runs once)
    goal = env.get_goal()
    obstacles = env.get_obstacles()
    reference = controller.plan(state, goal, obstacles)

    print(f"[Episode] Reference trajectory planned: {len(reference.states)} steps")

    # STEP 2: Track reference (MPC runs every step)
    trajectory = []
    for step in range(max_steps):
        # Get control from MPC tracker
        solution = controller.step(state, obstacles)

        if not solution.success:
            print(f"[Episode] Tracking failed at step {step}")
            break

        # Apply first control
        control = solution.U[0]
        state, reward, done, info = env.step(control)

        trajectory.append((state, control))

        if done:
            print(f"[Episode] Success in {step+1} steps")
            break

    return trajectory

# Main
if __name__ == "__main__":
    # Create hybrid controller
    controller = HybridTEBMPCController("mpc/config_mpc.yaml")

    # Run episodes
    for ep in range(num_episodes):
        trajectory = run_episode(env, controller)
        # ... save trajectory ...
```

### Phase 4: Testing & Validation (Day 3)

#### 4.1 Test TEB Planning Alone
```bash
# Test that TEB creates good reference trajectory
python -m mpc.test_teb_planning --episodes 5
```

Expected output:
```
[TEB Planning] Creating reference trajectory...
[TEB Planning] Created trajectory: 85 steps, 8.2s total
  Steps 1-20:  APPROACH (dt=0.20s avg)
  Steps 21-60: ENTRY (dt=0.25s avg, COMMITTED steering)
  Steps 61-85: FINAL (dt=0.08s avg, precision)
```

#### 4.2 Test MPC Tracking Alone
```bash
# Test that MPC can track reference without oscillations
python -m mpc.test_mpc_tracking --episodes 5
```

Expected output:
```
[Tracking] Following reference trajectory...
[Tracking Phase: approach] xy=300.0, theta=100.0
[Tracking Phase: entry] xy=500.0, theta=200.0
[Tracking Phase: final] xy=800.0, theta=300.0
[Success] Tracked reference: max deviation 3cm
```

#### 4.3 Test Full Hybrid System
```bash
# Test complete hybrid controller
python -m mpc.generate_expert_data --episodes 10 --scenario parallel --out-dir data/hybrid_teb_mpc
```

Expected results:
```
✅ Success Rate: 100%
✅ Final Precision: 1.5-2.0cm
✅ Steps: 40-50 (vs current 55)
✅ Oscillations: 0-1 (vs current 1)
✅ Committed maneuvers: YES (from TEB reference)
```

## Expected Performance Improvement

### Current Fixed-dt MPC Baseline
```
Steps: 55
Precision: 2.6cm
Oscillations: 1 major
Pattern: Approach → drift → correct → goal
```

### Hybrid TEB-MPC (Expected)
```
Steps: 40-50 (20% fewer)
Precision: 1.5-2.0cm (better)
Oscillations: 0 (ZERO zig-zag!)
Pattern: Committed approach → committed entry → smooth final
```

**Key improvements**:
- ✅ NO zig-zag (MPC just tracks, doesn't re-plan)
- ✅ Committed maneuvers (TEB plans them once)
- ✅ Faster (fewer steps from TEB optimization)
- ✅ Better precision (focused tracking)
- ✅ Multi-phase intelligence (from MPC tracking adaptation)

## Risk Mitigation

### Risk 1: TEB Planning Fails
**Mitigation**: Fallback to fixed-dt MPC
```python
reference = controller.plan(state, goal, obstacles)
if not reference or len(reference.states) < 10:
    print("[Hybrid] TEB planning failed, using fallback MPC")
    return fixed_dt_mpc.solve(state, goal, obstacles)
```

### Risk 2: Tracking Diverges from Reference
**Mitigation**: Monitor deviation, re-plan if needed
```python
deviation = distance(state, reference.get_state_at_step(current_step))
if deviation > 0.20:  # 20cm threshold
    print("[Hybrid] Large deviation detected, re-planning...")
    reference = controller.plan(state, goal, obstacles)
```

### Risk 3: Reference is Suboptimal
**Mitigation**: TEB has strong temporal costs, should create good paths
- w_time=10.0 encourages fast trajectories
- w_dt_precision=5.0 ensures small dt when close
- If still suboptimal, can tune TEB weights

## Implementation Timeline

**Day 1** (6-8 hours):
- Create ReferenceTrajectory class
- Modify teb_mpc.py for planning/tracking modes
- Update config.yaml

**Day 2** (6-8 hours):
- Implement multi-phase tracking
- Create HybridController
- Update generate_expert_data.py

**Day 3** (4-6 hours):
- Testing and validation
- Bug fixes and tuning
- Performance comparison

**Total**: 2-3 days full implementation

## Success Criteria

✅ **Zero zig-zag**: No oscillations in final approach
✅ **Committed maneuvers**: Observable full-lock steering segments
✅ **Better precision**: <2.0cm final error
✅ **Faster**: <50 steps average
✅ **100% success**: No failures or collisions

## Files to Create/Modify

### New Files
- `mpc/reference_trajectory.py` - Reference trajectory data structure
- `mpc/hybrid_controller.py` - Hybrid TEB+MPC controller
- `mpc/test_teb_planning.py` - Test TEB planning
- `mpc/test_mpc_tracking.py` - Test MPC tracking

### Modified Files
- `mpc/teb_mpc.py` - Add planning/tracking modes, tracking cost function
- `mpc/config_mpc.yaml` - Add planning/tracking configurations
- `mpc/generate_expert_data.py` - Use hybrid controller

### Documentation
- Update QUICK_REFERENCE.md with hybrid approach
- Create HYBRID_RESULTS.md with performance comparison

## Next Steps

Ready to proceed with implementation?

1. Should I start with Phase 1 (TEB planning mode)?
2. Any modifications to the architecture?
3. Any specific requirements for the reference trajectory format?
