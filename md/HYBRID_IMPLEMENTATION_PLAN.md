# Hybrid TEB+MPC Implementation Plan

**Date**: 2025-12-29
**Goal**: Achieve zero-oscillation parking with committed maneuvers
**Starting Point**: Fixed-dt MPC baseline (TEB disabled)
**Target**: Hybrid TEB planner + MPC tracker architecture

---

## Current Baseline Status

**Starting Configuration**:
- TEB: Disabled (`enable: false`)
- Fixed-dt MPC: dt = 0.1s
- Performance: 55 steps, 2.6cm precision, 1 oscillation
- Success rate: 100%
- Location: `data/test_teb_disabled/` or `data/final_baseline_teb_disabled/`

**Key Files**:
- [mpc/teb_mpc.py](mpc/teb_mpc.py) - Solver implementation
- [mpc/config_mpc.yaml](mpc/config_mpc.yaml) - Configuration
- [mpc/generate_expert_data.py](mpc/generate_expert_data.py) - Data generation

---

## Architecture Overview

### Current Architecture (Baseline)
```
Episode Loop:
  └─> MPC.solve(state, goal, obstacles) [every 0.1s]
       ├─> Optimize 50 steps ahead
       ├─> Execute first control
       └─> Repeat (RE-PLANNING every step)
```

### Target Architecture (Hybrid)
```
Episode Start:
  └─> TEB.plan(spawn, goal, obstacles) [ONCE]
       └─> Returns: ReferenceTrajectory
            ├─> States: [x, y, θ, v] for each step
            ├─> Controls: [steering, accel] for each step
            └─> Timing: dt for each step

Episode Loop:
  └─> MPC.track(state, reference, obstacles) [every 0.1s]
       ├─> Optimize tracking to reference
       ├─> Execute first control
       └─> Repeat (NO RE-PLANNING)
```

---

## Implementation Phases

---

## PHASE 1: Foundation & Testing Infrastructure (Day 1 Morning)

**Goal**: Create baseline test framework and reference trajectory data structure

### Milestone 1.1: Baseline Verification
**Duration**: 30 minutes
**Objective**: Confirm current baseline and establish metrics

**Tasks**:
1. Run 10 baseline episodes with TEB disabled
2. Extract metrics:
   - Steps to completion
   - Final precision
   - Oscillation count (moved-away phases)
   - Steering change frequency
3. Save baseline data for comparison
4. Document baseline performance

**Deliverables**:
- `data/baseline_fixed_dt/` - 10 episodes
- `BASELINE_METRICS.md` - Performance summary

**Success Criteria**:
- ✅ 10/10 episodes succeed
- ✅ Average 55 ± 3 steps
- ✅ Average 2.6 ± 0.3 cm precision
- ✅ 1 major oscillation per episode

**Test Command**:
```bash
python -m mpc.generate_expert_data --episodes 10 --scenario parallel --out-dir data/baseline_fixed_dt
```

---

### Milestone 1.2: Reference Trajectory Data Structure
**Duration**: 1 hour
**Objective**: Create data structure to hold TEB-planned trajectories

**Tasks**:
1. Create `mpc/reference_trajectory.py`
2. Implement `ReferenceState` dataclass
3. Implement `ReferenceTrajectory` class
4. Add utility methods:
   - `get_state_at_step(step)` - Get reference at step
   - `get_segment(start, horizon)` - Get reference window
   - `distance_to_goal(step)` - Remaining distance
5. Write unit tests

**Deliverables**:
- `mpc/reference_trajectory.py` (~100 lines)
- Unit tests

**Success Criteria**:
- ✅ Can store trajectory with states, controls, dt
- ✅ Can retrieve states by step index
- ✅ Can get trajectory segments
- ✅ All unit tests pass

**Code Structure**:
```python
@dataclass
class ReferenceState:
    x: float
    y: float
    theta: float
    v: float
    steering: float
    accel: float
    dt: float
    time: float  # Cumulative

class ReferenceTrajectory:
    def __init__(self, states: List[ReferenceState])
    def get_state_at_step(self, step: int) -> ReferenceState
    def get_segment(self, start: int, horizon: int) -> List[ReferenceState]
    def distance_to_goal(self, step: int) -> float
```

---

### Milestone 1.3: Test Harness for Hybrid System
**Duration**: 30 minutes
**Objective**: Create testing infrastructure for hybrid components

**Tasks**:
1. Create `mpc/test_hybrid.py`
2. Add helper functions:
   - `load_reference_from_file()` - Load pre-saved trajectories
   - `compare_trajectories()` - Compare two episode results
   - `analyze_smoothness()` - Measure steering changes
3. Add visualization helpers (optional)

**Deliverables**:
- `mpc/test_hybrid.py` (~150 lines)

**Success Criteria**:
- ✅ Can load and analyze episodes
- ✅ Can measure oscillations automatically
- ✅ Can compare baseline vs hybrid

---

## PHASE 2: TEB Planning Mode (Day 1 Afternoon)

**Goal**: Make TEB work as a one-shot trajectory planner

### Milestone 2.1: TEB Planning Configuration
**Duration**: 1 hour
**Objective**: Configure TEB for global trajectory planning

**Tasks**:
1. Update `mpc/config_mpc.yaml`:
   - Add `teb_planning` section
   - Set aggressive temporal optimization
   - Configure for committed maneuvers
2. Create planning profile separate from tracking profile

**Deliverables**:
- Updated `mpc/config_mpc.yaml`

**Configuration**:
```yaml
mpc:
  mode: "hybrid"  # Options: "fixed_dt", "teb_realtime", "hybrid"

  # TEB Planning Mode (runs ONCE at episode start)
  teb_planning:
    enable: true
    horizon: 100            # Plan to goal (increased from 50)
    dt_min: 0.08
    dt_max: 0.30            # ALLOW committed maneuvers

    # STRONG temporal optimization
    w_time: 10.0            # Minimize time → committed maneuvers
    w_dt_smooth: 5.0        # Smooth dt transitions
    w_dt_precision: 5.0     # Small dt when close to goal

    # Spatial costs (same as baseline)
    w_goal_xy: 400.0
    w_goal_theta: 120.0
    lateral_weight: 0.25
    yaw_weight: 0.9
    # ... (copy from parallel profile)
```

**Success Criteria**:
- ✅ Configuration loads without errors
- ✅ TEB planning section separated from tracking

---

### Milestone 2.2: TEB Planner Implementation
**Duration**: 2 hours
**Objective**: Implement one-shot TEB trajectory generation

**Tasks**:
1. Modify `mpc/teb_mpc.py`:
   - Add `mode` parameter ("planning" vs "tracking")
   - Implement `plan_trajectory()` method
   - Configure TEB with planning weights
2. Handle planning failures (fallback)
3. Extract trajectory from solution

**Deliverables**:
- Modified `mpc/teb_mpc.py` (+100 lines)

**Key Methods**:
```python
class TEBMPC:
    def __init__(self, config_path, mode="tracking"):
        self.mode = mode  # "planning" or "tracking"

    def plan_trajectory(self, state, goal, obstacles) -> ReferenceTrajectory:
        """
        TEB one-shot planning: Create reference from state to goal
        Returns committed trajectory with variable dt
        """
        # 1. Configure for planning mode
        self._apply_planning_config()

        # 2. Solve with TEB to goal
        solution = self.solve(state, goal, obstacles)

        # 3. Extract reference trajectory
        reference = self._solution_to_reference(solution)

        # 4. Validate and return
        return reference
```

**Success Criteria**:
- ✅ TEB can plan from spawn to goal (once)
- ✅ Returns valid ReferenceTrajectory
- ✅ Trajectory reaches goal
- ✅ dt values vary (small when close, large when far)

---

### Milestone 2.3: TEB Planner Testing
**Duration**: 1 hour
**Objective**: Verify TEB creates good trajectories

**Tasks**:
1. Create `test_teb_planner.py`
2. Test TEB planning from multiple spawn points
3. Analyze planned trajectories:
   - Check for committed maneuvers (constant steering segments)
   - Verify dt variation (large for maneuvers, small for precision)
   - Ensure goal is reached
4. Save example trajectories

**Deliverables**:
- `test_teb_planner.py`
- `data/teb_planned_trajectories/` - Example references

**Test Script**:
```python
# Test TEB planning
planner = TEBMPC(config_path, mode="planning")

for episode in range(5):
    state = env.reset()
    goal = env.get_goal()
    obstacles = env.get_obstacles()

    # Plan trajectory
    reference = planner.plan_trajectory(state, goal, obstacles)

    # Analyze
    print(f"Episode {episode}:")
    print(f"  Steps: {len(reference.states)}")
    print(f"  Total time: {reference.total_time:.2f}s")
    print(f"  dt range: [{min(ref.dt):.3f}, {max(ref.dt):.3f}]")

    # Check for committed segments
    analyze_commitment(reference)
```

**Success Criteria**:
- ✅ 5/5 trajectories reach goal
- ✅ dt varies from 0.08 (precision) to 0.25+ (committed)
- ✅ Observable committed steering segments (3+ steps same steering)
- ✅ Total time < 8 seconds

---

## PHASE 3: MPC Tracking Mode (Day 2 Morning)

**Goal**: Make MPC track reference trajectories instead of planning

### Milestone 3.1: Tracking Cost Function
**Duration**: 2 hours
**Objective**: Add reference tracking to MPC cost function

**Tasks**:
1. Modify `_build_solver()` in `mpc/teb_mpc.py`:
   - Add reference trajectory to parameter vector P
   - Implement tracking cost terms
   - Maintain collision avoidance
2. Configure tracking weights

**Deliverables**:
- Modified `_build_solver()` (+50 lines)

**Cost Function Changes**:
```python
# In _build_solver():

# Expand parameter vector: [state, goal, obstacles, prev_control, reference]
if self.mode == "tracking":
    ref_size = 4 * N  # (x, y, theta, v) for N steps
    REF = ca.SX.sym("ref_traj", ref_size)
    P = ca.vertcat(P, REF)

# In cost loop:
for k in range(N):
    st = X[:, k]
    u = U[:, k]

    if self.mode == "tracking":
        # PRIMARY: Track reference trajectory
        ref_idx = ref_offset + k * 4
        ref_x = P[ref_idx]
        ref_y = P[ref_idx + 1]
        ref_theta = P[ref_idx + 2]
        ref_v = P[ref_idx + 3]

        # Strong tracking cost
        obj += self.w_tracking_xy * ((st[0] - ref_x)**2 + (st[1] - ref_y)**2)
        obj += self.w_tracking_theta * (angle_wrap(st[2] - ref_theta))**2
        obj += self.w_tracking_v * (st[3] - ref_v)**2

        # SECONDARY: Weak goal attraction (safety net)
        obj += 10.0 * ((st[0] - goal_x)**2 + (st[1] - goal_y)**2)
    else:
        # PLANNING: Standard goal-based cost
        obj += self.w_goal_xy * ((st[0] - goal_x)**2 + (st[1] - goal_y)**2)

    # Collision avoidance (both modes)
    obj += self.w_collision * collision_cost
```

**Tracking Weights**:
```yaml
mpc_tracking:
  w_tracking_xy: 500.0      # STRONG position tracking
  w_tracking_theta: 150.0   # STRONG heading tracking
  w_tracking_v: 10.0        # Moderate velocity tracking

  w_goal_xy: 10.0           # WEAK goal (safety net)
  w_goal_theta: 5.0         # WEAK yaw (safety net)
```

**Success Criteria**:
- ✅ Solver builds with tracking mode
- ✅ Parameter vector includes reference
- ✅ Tracking cost dominates goal cost (500 >> 10)

---

### Milestone 3.2: Multi-Phase Tracking Weights
**Duration**: 1 hour
**Objective**: Adapt tracking strength based on parking phase

**Tasks**:
1. Implement phase detection from reference distance
2. Apply phase-specific tracking weights:
   - APPROACH: Moderate tracking (allow larger deviations)
   - ENTRY: Strong tracking (follow committed maneuvers)
   - FINAL: Very strong tracking (precision required)
3. Test phase transitions

**Deliverables**:
- `_apply_phase_tracking_weights()` method

**Implementation**:
```python
def _detect_phase_from_reference(self, current_step, reference):
    """Detect phase from reference trajectory"""
    goal_dist = reference.distance_to_goal(current_step)

    if goal_dist > 0.50:
        return ParkingPhase.APPROACH
    elif goal_dist > 0.15:
        return ParkingPhase.ENTRY
    else:
        return ParkingPhase.FINAL_ALIGN

def _apply_phase_tracking_weights(self, phase):
    """Apply phase-specific tracking weights"""
    if phase == ParkingPhase.APPROACH:
        self.w_tracking_xy = 300.0
        self.w_tracking_theta = 100.0
    elif phase == ParkingPhase.ENTRY:
        self.w_tracking_xy = 500.0      # STRONG
        self.w_tracking_theta = 200.0   # Follow committed path!
    elif phase == ParkingPhase.FINAL_ALIGN:
        self.w_tracking_xy = 800.0      # VERY STRONG
        self.w_tracking_theta = 300.0   # Precision
```

**Success Criteria**:
- ✅ Phase detection works from reference
- ✅ Weights adapt per phase
- ✅ Smooth transitions between phases

---

### Milestone 3.3: MPC Tracker Implementation
**Duration**: 1 hour
**Objective**: Implement `track_trajectory()` method

**Tasks**:
1. Implement `track_trajectory()` in `mpc/teb_mpc.py`
2. Handle reference segment extraction
3. Fill parameter vector with reference states
4. Call solver with tracking cost

**Deliverables**:
- `track_trajectory()` method (+80 lines)

**Implementation**:
```python
def track_trajectory(self, state, reference, current_step, obstacles):
    """
    MPC tracking: Follow reference trajectory
    Runs every control step (0.1s)
    """
    # 1. Detect phase from reference
    phase = self._detect_phase_from_reference(current_step, reference)

    # 2. Apply phase-specific tracking weights
    self._apply_phase_tracking_weights(phase)

    # 3. Get reference segment for horizon
    ref_segment = reference.get_segment(current_step, self.N)

    # 4. Fill parameter vector
    P = np.zeros(param_size)
    # ... fill state, goal, obstacles, prev_control ...

    # Fill reference trajectory
    for k, ref_state in enumerate(ref_segment):
        ref_idx = ref_offset + k * 4
        P[ref_idx] = ref_state.x
        P[ref_idx + 1] = ref_state.y
        P[ref_idx + 2] = ref_state.theta
        P[ref_idx + 3] = ref_state.v

    # 5. Solve
    solution = solver(x0=x0_init, lbx=lbx, ubx=ubx, lbg=0, ubg=0, p=P)

    return MPCSolution(...)
```

**Success Criteria**:
- ✅ Tracker method compiles and runs
- ✅ Reference correctly passed to solver
- ✅ Returns valid control

---

### Milestone 3.4: MPC Tracker Testing
**Duration**: 1 hour
**Objective**: Verify MPC can track pre-generated references

**Tasks**:
1. Create `test_mpc_tracker.py`
2. Load TEB-planned reference from Phase 2
3. Test MPC tracking:
   - Execute tracking for full episode
   - Measure deviation from reference
   - Check collision avoidance still works
4. Analyze tracking performance

**Deliverables**:
- `test_mpc_tracker.py`
- Tracking performance metrics

**Test Script**:
```python
# Test MPC tracking
tracker = TEBMPC(config_path, mode="tracking")

# Load pre-planned reference
reference = load_reference("data/teb_planned_trajectories/episode_0000.pkl")

# Track reference
env = ParkingEnv()
state = env.reset()

for step in range(len(reference.states)):
    # Get control from tracker
    solution = tracker.track_trajectory(state, reference, step, env.get_obstacles())

    # Apply control
    state, _, done, _ = env.step(solution.U[0])

    # Measure deviation
    ref_state = reference.get_state_at_step(step)
    deviation = distance(state, ref_state)

    print(f"Step {step}: deviation={deviation:.3f}m")

    if done:
        break
```

**Success Criteria**:
- ✅ Tracks reference to completion
- ✅ Max deviation < 5cm from reference
- ✅ No collisions
- ✅ Reaches goal successfully

---

## PHASE 4: Hybrid Integration (Day 2 Afternoon)

**Goal**: Combine TEB planner + MPC tracker into unified system

### Milestone 4.1: Hybrid Controller Class
**Duration**: 1 hour
**Objective**: Create orchestrator for planner + tracker

**Tasks**:
1. Create `mpc/hybrid_controller.py`
2. Implement `HybridTEBMPCController` class
3. Handle planning + tracking lifecycle
4. Add re-planning logic (if needed)

**Deliverables**:
- `mpc/hybrid_controller.py` (~150 lines)

**Implementation**:
```python
class HybridTEBMPCController:
    """
    Hybrid TEB+MPC Controller

    Architecture:
    1. TEB plans reference trajectory (ONCE at start)
    2. MPC tracks reference (every step)
    """

    def __init__(self, config_path):
        self.planner = TEBMPC(config_path, mode="planning")
        self.tracker = TEBMPC(config_path, mode="tracking")
        self.reference_trajectory = None
        self.current_step = 0

    def reset(self):
        """Reset for new episode"""
        self.reference_trajectory = None
        self.current_step = 0

    def plan(self, state, goal, obstacles):
        """Create reference trajectory using TEB (run once)"""
        print("[Hybrid] Planning reference trajectory with TEB...")
        self.reference_trajectory = self.planner.plan_trajectory(
            state, goal, obstacles
        )
        self.current_step = 0
        print(f"[Hybrid] Planned {len(self.reference_trajectory.states)} steps, "
              f"{self.reference_trajectory.total_time:.2f}s")
        return self.reference_trajectory

    def step(self, state, obstacles):
        """Track reference using MPC (run every step)"""
        if self.reference_trajectory is None:
            raise ValueError("Must call plan() before step()")

        # Check if we need to re-plan (optional)
        if self._should_replan(state):
            print("[Hybrid] Re-planning due to large deviation...")
            self.plan(state, goal, obstacles)

        # Track reference
        solution = self.tracker.track_trajectory(
            state, self.reference_trajectory,
            self.current_step, obstacles
        )

        self.current_step += 1
        return solution

    def _should_replan(self, state):
        """Check if re-planning needed (large deviation or obstacle moved)"""
        if self.current_step >= len(self.reference_trajectory.states):
            return False

        ref_state = self.reference_trajectory.get_state_at_step(self.current_step)
        deviation = np.sqrt((state.x - ref_state.x)**2 + (state.y - ref_state.y)**2)

        # Re-plan if deviation > 20cm
        return deviation > 0.20
```

**Success Criteria**:
- ✅ Controller creates planner + tracker
- ✅ Can plan and track in sequence
- ✅ Re-planning logic works (optional)

---

### Milestone 4.2: Integration with Expert Data Generation
**Duration**: 1 hour
**Objective**: Use hybrid controller in main data generation loop

**Tasks**:
1. Modify `mpc/generate_expert_data.py`:
   - Add `--mode` argument (fixed_dt, teb, hybrid)
   - Use HybridController when mode="hybrid"
   - Log planning + tracking phases
2. Test integration

**Deliverables**:
- Modified `mpc/generate_expert_data.py` (+50 lines)

**Implementation**:
```python
def run_episode_hybrid(env, controller, max_steps=200):
    """Run episode with hybrid TEB+MPC controller"""
    state = env.reset()
    controller.reset()

    # STEP 1: Plan reference trajectory (TEB runs once)
    goal = env.get_goal()
    obstacles = env.get_obstacles()
    reference = controller.plan(state, goal, obstacles)

    print(f"[Episode] Reference planned: {len(reference.states)} steps")

    # STEP 2: Track reference (MPC runs every step)
    trajectory = []
    for step in range(max_steps):
        # Get control from tracker
        solution = controller.step(state, obstacles)

        if not solution.success:
            print(f"[Episode] Tracking failed at step {step}")
            break

        # Apply control
        control = solution.U[0]
        next_state, reward, done, info = env.step(control)

        trajectory.append((state, control))
        state = next_state

        if done:
            print(f"[Episode] Success in {step+1} steps")
            break

    return trajectory

# Main
if args.mode == "hybrid":
    controller = HybridTEBMPCController(config_path)
    for ep in range(num_episodes):
        traj = run_episode_hybrid(env, controller)
        save_trajectory(traj, f"episode_{ep:04d}.pkl")
```

**Success Criteria**:
- ✅ Can run with `--mode hybrid`
- ✅ Plans once at start
- ✅ Tracks for full episode
- ✅ Saves trajectory successfully

---

### Milestone 4.3: Hybrid System Testing
**Duration**: 1 hour
**Objective**: Test full hybrid system end-to-end

**Tasks**:
1. Run 5 hybrid episodes
2. Verify planning + tracking works
3. Check for errors or crashes
4. Compare with baseline (qualitative)

**Deliverables**:
- `data/hybrid_test/` - 5 episodes

**Test Command**:
```bash
python -m mpc.generate_expert_data --episodes 5 --scenario parallel --mode hybrid --out-dir data/hybrid_test
```

**Success Criteria**:
- ✅ 5/5 episodes complete successfully
- ✅ No crashes or solver failures
- ✅ Planning takes < 5 seconds
- ✅ Tracking is stable

---

## PHASE 5: Validation & Tuning (Day 3)

**Goal**: Validate performance and tune for optimal results

### Milestone 5.1: Performance Comparison
**Duration**: 2 hours
**Objective**: Compare hybrid vs baseline quantitatively

**Tasks**:
1. Run 20 baseline episodes
2. Run 20 hybrid episodes
3. Extract metrics:
   - Steps to completion
   - Final precision
   - Oscillation count
   - Steering changes
   - Committed maneuver count
4. Statistical comparison

**Deliverables**:
- `data/comparison_baseline/` - 20 episodes
- `data/comparison_hybrid/` - 20 episodes
- `PERFORMANCE_COMPARISON.md` - Results

**Metrics to Compare**:
```python
def analyze_episodes(episode_dir):
    results = {
        'steps': [],
        'precision': [],
        'oscillations': [],
        'steering_changes': [],
        'committed_segments': []
    }

    for episode_file in glob(f"{episode_dir}/*.pkl"):
        episode = load_episode(episode_file)

        results['steps'].append(len(episode['traj']))
        results['precision'].append(final_position_error(episode))
        results['oscillations'].append(count_oscillations(episode))
        results['steering_changes'].append(count_large_steering_changes(episode))
        results['committed_segments'].append(count_committed_segments(episode))

    return results
```

**Success Criteria**:
- ✅ Hybrid has fewer steps (40-45 vs 55)
- ✅ Hybrid has better precision (<2.0cm vs 2.6cm)
- ✅ Hybrid has **zero oscillations** (vs 1)
- ✅ Hybrid has fewer steering changes (5-10 vs 30+)
- ✅ Hybrid shows committed maneuvers (3-5 segments)

---

### Milestone 5.2: Weight Tuning (If Needed)
**Duration**: 2 hours
**Objective**: Fine-tune tracking weights for optimal performance

**Tasks**:
1. Test tracking weight variations:
   - Try w_tracking_xy: [300, 500, 800]
   - Try w_tracking_theta: [100, 150, 200]
2. Analyze impact on deviation and performance
3. Select optimal weights

**Deliverables**:
- Tuning results
- Optimal weight configuration

**Success Criteria**:
- ✅ Found weights that maintain <3cm deviation from reference
- ✅ No increase in oscillations
- ✅ Stable across 10+ episodes

---

### Milestone 5.3: Edge Case Testing
**Duration**: 2 hours
**Objective**: Test robustness in various scenarios

**Tasks**:
1. Test different spawn positions (if available)
2. Test with planning failures (fallback to baseline)
3. Test re-planning logic (manual disturbance)
4. Long-run stability (100 episodes)

**Deliverables**:
- Edge case test results
- Fallback behavior verification

**Success Criteria**:
- ✅ Works across different spawns
- ✅ Graceful fallback when planning fails
- ✅ Re-planning works when deviation large
- ✅ 100% success rate over 100 episodes

---

### Milestone 5.4: Documentation & Code Cleanup
**Duration**: 2 hours
**Objective**: Document hybrid system and clean up code

**Tasks**:
1. Add docstrings to all new classes/methods
2. Create user guide: `HYBRID_USER_GUIDE.md`
3. Update [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
4. Add configuration examples
5. Code cleanup and formatting

**Deliverables**:
- `HYBRID_USER_GUIDE.md`
- Updated `QUICK_REFERENCE.md`
- Clean, documented code

**Success Criteria**:
- ✅ All public methods have docstrings
- ✅ User guide explains how to use hybrid mode
- ✅ Configuration well-documented
- ✅ Code passes linting

---

## Milestones Summary

| Phase | Milestone | Duration | Deliverable | Success Metric |
|-------|-----------|----------|-------------|----------------|
| **1. Foundation** | 1.1 Baseline Verification | 0.5h | Baseline metrics | 10/10 success |
| | 1.2 Reference Trajectory | 1h | reference_trajectory.py | Unit tests pass |
| | 1.3 Test Harness | 0.5h | test_hybrid.py | Can analyze episodes |
| **2. TEB Planning** | 2.1 Configuration | 1h | Updated config | Config loads |
| | 2.2 Planner Implementation | 2h | plan_trajectory() | Returns reference |
| | 2.3 Planner Testing | 1h | TEB test results | 5/5 plans succeed |
| **3. MPC Tracking** | 3.1 Tracking Cost | 2h | Modified solver | Tracking cost works |
| | 3.2 Multi-Phase Weights | 1h | Phase-aware tracking | Weights adapt |
| | 3.3 Tracker Implementation | 1h | track_trajectory() | Returns control |
| | 3.4 Tracker Testing | 1h | Tracking test results | Max 5cm deviation |
| **4. Integration** | 4.1 Hybrid Controller | 1h | hybrid_controller.py | Controller works |
| | 4.2 Data Generation | 1h | Modified generate | Can run hybrid mode |
| | 4.3 System Testing | 1h | 5 hybrid episodes | 5/5 success |
| **5. Validation** | 5.1 Comparison | 2h | Performance metrics | 0 oscillations |
| | 5.2 Weight Tuning | 2h | Optimal weights | <3cm deviation |
| | 5.3 Edge Cases | 2h | Robustness results | 100% success |
| | 5.4 Documentation | 2h | User guide | Complete docs |

**Total Estimated Time**: 20 hours (~2.5 days)

---

## Risk Mitigation

### Risk 1: TEB Planning Fails
**Likelihood**: Low
**Impact**: High
**Mitigation**:
- Fallback to baseline fixed-dt MPC
- Log planning failures
- Retry with relaxed constraints

### Risk 2: Tracking Diverges from Reference
**Likelihood**: Medium
**Impact**: Medium
**Mitigation**:
- Strong tracking weights (500+)
- Monitor deviation at each step
- Re-plan if deviation > 20cm

### Risk 3: Performance Not Better Than Baseline
**Likelihood**: Low (architecture is sound)
**Impact**: High
**Mitigation**:
- Start with proven configurations
- Incremental testing at each milestone
- Keep baseline code intact for fallback

### Risk 4: Implementation Takes Longer
**Likelihood**: Medium
**Impact**: Medium
**Mitigation**:
- Detailed milestones with time estimates
- Test incrementally (don't wait until end)
- Skip optional features if needed (re-planning, visualization)

---

## Success Criteria (Overall)

**Minimum Viable Product**:
- ✅ Hybrid system runs without crashes
- ✅ 100% success rate (10/10 episodes)
- ✅ Better than baseline in ≥3 metrics

**Target Performance**:
- ✅ **Zero oscillations** (vs baseline's 1)
- ✅ Fewer steps (40-45 vs 55)
- ✅ Better precision (<2.0cm vs 2.6cm)
- ✅ Observable committed maneuvers (3-5 segments)
- ✅ Smooth steering (5-10 changes vs 30+)

**Stretch Goals**:
- ✅ Works across multiple spawn positions
- ✅ Adaptive re-planning works
- ✅ 100% success over 100 episodes
- ✅ Human-like maneuvers (qualitative assessment)

---

## Next Steps

**Ready to begin?**

Start with **Phase 1, Milestone 1.1: Baseline Verification**

**First Command**:
```bash
python -m mpc.generate_expert_data --episodes 10 --scenario parallel --out-dir data/baseline_fixed_dt
```

Once baseline is verified, proceed to **Milestone 1.2: Reference Trajectory**.

---

## Files to Create/Modify

### New Files (7 total)
1. `mpc/reference_trajectory.py` - Reference trajectory data structure
2. `mpc/test_hybrid.py` - Testing infrastructure
3. `mpc/hybrid_controller.py` - Hybrid orchestrator
4. `test_teb_planner.py` - TEB planner tests
5. `test_mpc_tracker.py` - MPC tracker tests
6. `BASELINE_METRICS.md` - Baseline performance
7. `PERFORMANCE_COMPARISON.md` - Final comparison
8. `HYBRID_USER_GUIDE.md` - User documentation

### Modified Files (2 total)
1. `mpc/teb_mpc.py` - Add planning/tracking modes
2. `mpc/generate_expert_data.py` - Add hybrid mode
3. `mpc/config_mpc.yaml` - Add hybrid configuration

**Total**: ~1000 lines of new code, ~200 lines of modifications

---

Ready to start with Phase 1?
