# MPC Commitment Techniques - Making MPC Hold Steering

**Date**: 2025-12-29
**Goal**: Force MPC to commit to steering commands instead of constantly re-optimizing

## Three Proven Techniques

You've identified three excellent techniques that are **all used in production MPC systems**. Let's evaluate each for parking.

---

## Technique A: Slew Rate Penalty (Steering Smoothness)

### What It Is

Add cost to **change in steering** between consecutive steps:

```python
# Current cost (we already have this!)
obj += self.w_smooth_steer * (U[k, 0] - U[k-1, 0])**2

# Problem: Only penalizes within horizon, not actual execution
```

### The Issue With Current Implementation

**Current code** ([teb_mpc.py:519](mpc/teb_mpc.py#L519)):
```python
# Control smoothness within the horizon
for k in range(1, N):
    obj += self.w_smooth_steer * (U[k, 0] - U[k-1, 0])**2  # Future plan
    obj += self.w_smooth_accel * (U[k, 1] - U[k-1, 1])**2
```

**Problem**: This only smooths **within the planned trajectory**, NOT between actual executed commands!

```
Step 1: Plan [0.52, 0.50, 0.48, ...] → Execute 0.52
Step 2: Plan [0.40, 0.38, 0.36, ...] → Execute 0.40 (JUMP from 0.52!)
                                         ^^^^ NOT penalized!
```

### The Fix: True Slew Rate Penalty

Penalize change from **last executed control**:

```python
# In _build_solver()
# Add parameter for last executed control
U_prev = ca.SX.sym("u_prev", 2)  # [last_steering, last_accel]
P = ca.vertcat(P, U_prev)

# Penalize change from actual previous control
obj += self.w_slew_rate_steer * (U[0, 0] - U_prev[0])**2
obj += self.w_slew_rate_accel * (U[0, 1] - U_prev[1])**2
```

### Configuration

```yaml
mpc:
  # STRONG slew rate penalty (commitment to previous command)
  w_slew_rate_steer: 50.0   # NEW: Penalize steering change from last step
  w_slew_rate_accel: 20.0   # NEW: Penalize accel change from last step

  # Existing smoothness (within horizon)
  w_smooth_steer: 0.0015    # Existing: smoothness within plan
  w_smooth_accel: 0.003
```

### Expected Effect

**Current behavior**:
```
Step 1: steer=0.52
Step 2: steer=0.40  (change: 0.12 rad, ~7°) ← BAD
Step 3: steer=0.48  (change: 0.08 rad, ~5°) ← BAD
```

**With slew rate penalty (w=50.0)**:
```
Step 1: steer=0.52
Step 2: steer=0.51  (change: 0.01 rad, ~0.6°) ← GOOD
Step 3: steer=0.50  (change: 0.01 rad, ~0.6°) ← GOOD
Step 4: steer=0.49  (change: 0.01 rad, ~0.6°) ← GOOD
```

**Result**: Smooth gradual changes instead of jumps → committed steering segments

### Pros & Cons

✅ **Pros**:
- Easy to implement (just add cost term)
- Computationally cheap
- Maintains MPC reactivity (can still change if needed)
- Industry-standard technique

❌ **Cons**:
- Doesn't guarantee long holds (just makes changes expensive)
- May still oscillate if other costs dominate
- Need to tune weight carefully

### Recommendation: **HIGH PRIORITY - Implement This First**

This is the easiest and most effective fix. Should reduce oscillations significantly.

---

## Technique B: Move Blocking (Control Blocking)

### What It Is

Force multiple time steps to share the same control:

```python
# Instead of: u0, u1, u2, u3, u4, u5, ..., u49 (50 variables)
# Use:        u0=u1=u2, u3=u4=u5, u6=u7=u8, ... (17 variables)
#             └─────┘   └─────┘   └─────┘
#              Block 0   Block 1   Block 2
```

### Implementation

**Define blocking structure**:
```python
# In _build_solver()
# Define control blocks (e.g., 3 steps per block)
block_size = 3
num_blocks = N // block_size

# Decision variables: only num_blocks control values
U_blocks = ca.SX.sym("u_blocks", num_blocks, 2)  # [steer, accel]

# Expand to full horizon
U = ca.SX.zeros(N, 2)
for i in range(num_blocks):
    for j in range(block_size):
        k = i * block_size + j
        if k < N:
            U[k, :] = U_blocks[i, :]  # Same control for entire block
```

### Configuration

```yaml
mpc:
  move_blocking:
    enable: true
    block_sizes: [3, 3, 3, 5, 5, 5, 5, 5, 5, 5]  # Variable blocking
    # First 9 steps: 3-step blocks (0.3s commitment)
    # Remaining steps: 5-step blocks (0.5s commitment)
```

### Expected Effect

**Current (no blocking)**:
```
Step 1: Execute steer=0.52 for 0.1s
Step 2: Execute steer=0.40 for 0.1s (changed!)
Step 3: Execute steer=0.48 for 0.1s (changed!)
```

**With blocking (block_size=3)**:
```
Iteration 1: Plan u_block0=0.52 for steps [0,1,2]
             → Execute 0.52 for 0.1s
Iteration 2: Plan u_block0=0.52 for steps [0,1,2]
             → Execute 0.52 for 0.1s (SAME!)
Iteration 3: Plan u_block0=0.52 for steps [0,1,2]
             → Execute 0.52 for 0.1s (STILL SAME!)
Iteration 4: Plan u_block1=0.50 for steps [0,1,2]
             → Execute 0.50 for 0.1s (smooth transition)
```

**Result**: Steering holds for 0.3s chunks → human-like committed maneuvers

### Pros & Cons

✅ **Pros**:
- **Forces commitment** (cannot change within block)
- Drastically reduces oscillations
- Reduces computation (fewer variables to optimize)
- Very effective for parking scenarios

❌ **Cons**:
- More complex to implement (need to restructure optimizer)
- Less reactive (locked in for block duration)
- Need to choose block sizes carefully

### Recommendation: **MEDIUM PRIORITY - Implement After Slew Rate**

Very effective for commitment, but more complex. Good second step.

---

## Technique C: Multi-Step Execution (Execute > 1 Control)

### What It Is

Instead of executing only u[0], execute u[0:3] before re-planning:

```python
# Standard MPC
solution = mpc.solve(state, goal)
state = env.step(solution.U[0])  # Execute only first
# Discard solution.U[1:N], re-plan next step

# Multi-step execution
solution = mpc.solve(state, goal)
for i in range(3):  # Execute first 3 controls
    state = env.step(solution.U[i])
# Re-plan after 0.3s
```

### Implementation

```python
class MPCWithMultiStepExecution:
    def __init__(self, execution_steps=3):
        self.execution_steps = execution_steps
        self.planned_controls = None
        self.execution_counter = 0

    def get_control(self, state, goal, obstacles):
        # Re-plan only when execution buffer is empty
        if self.planned_controls is None or self.execution_counter >= self.execution_steps:
            solution = self.mpc.solve(state, goal, obstacles)
            self.planned_controls = solution.U
            self.execution_counter = 0

        # Execute from buffer
        control = self.planned_controls[self.execution_counter]
        self.execution_counter += 1
        return control
```

### Configuration

```yaml
mpc:
  multi_step_execution:
    enable: true
    steps: 3              # Re-plan every 3 steps (0.3s)
    adaptive: true        # Reduce steps when close to goal
    near_goal_steps: 1    # Execute only 1 step when <20cm from goal
```

### Expected Effect

**Current (re-plan every step)**:
```
t=0.0s: Plan → Execute u[0]=0.52
t=0.1s: Plan → Execute u[0]=0.40 (changed!)
t=0.2s: Plan → Execute u[0]=0.48 (changed!)
```

**With multi-step execution (steps=3)**:
```
t=0.0s: Plan → Execute u[0]=0.52
t=0.1s: (no re-plan) → Execute u[1]=0.51 (from original plan)
t=0.2s: (no re-plan) → Execute u[2]=0.50 (from original plan)
t=0.3s: Plan → Execute u[0]=0.48
t=0.4s: (no re-plan) → Execute u[1]=0.47
t=0.5s: (no re-plan) → Execute u[2]=0.46
```

**Result**: Commits to 0.3s segments, much smoother

### Pros & Cons

✅ **Pros**:
- **Simplest to implement** (just change execution logic)
- Forces commitment automatically
- Reduces computation (re-plan less often)
- Very effective for smoothness

❌ **Cons**:
- **Reduces safety** (blind for execution_steps * dt)
- Less adaptive to disturbances
- May overshoot goal if not careful
- Need adaptive logic for final approach

### Recommendation: **LOW PRIORITY - Use With Caution**

Simple but reduces reactivity. Only use for parking (low speed, controlled environment).

---

## Comparison: Which to Use?

| Technique | Implementation | Effectiveness | Safety | Computation |
|-----------|---------------|---------------|---------|-------------|
| **A. Slew Rate** | Easy (1 hour) | ✅✅ High | ✅✅ Safe | ✅ Same |
| **B. Move Blocking** | Medium (4 hours) | ✅✅✅ Very High | ✅ Good | ✅✅ Faster |
| **C. Multi-Step Exec** | Easy (1 hour) | ✅✅ High | ⚠️ Reduced | ✅✅ Faster |

## Recommended Implementation Strategy

### Phase 1: Quick Win (1-2 hours)
**Implement Slew Rate Penalty (Technique A)**
- Add `w_slew_rate_steer` and `w_slew_rate_accel` to cost function
- Pass previous control as parameter
- Test with strong weights (50.0, 20.0)

**Expected improvement**: 50-70% reduction in oscillations

### Phase 2: If Still Oscillating (4-6 hours)
**Add Move Blocking (Technique B)**
- Implement control blocking with 3-step blocks
- Reduces variables from 50 to ~17
- Guarantees 0.3s commitment

**Expected improvement**: 80-90% reduction in oscillations

### Phase 3: Final Tuning (1 hour)
**Optional Multi-Step Execution (Technique C)** for far-from-goal phase
- Execute 3 steps when >50cm from goal
- Execute 1 step when <20cm (precision mode)
- Reduces computation, maintains safety

**Expected improvement**: Computational speedup + smoothness

---

## Implementation Plan: Technique A (Slew Rate - Highest Priority)

### Step 1: Modify Solver to Accept Previous Control

**File**: `mpc/teb_mpc.py` - Modify `_build_solver()`

```python
def _build_solver(self, parking_type: str = "parallel"):
    # ... existing setup ...

    # NEW: Add previous control as parameter
    # P now includes: [goal, obstacles, prev_control]
    U_prev = ca.SX.sym("u_prev", 2)  # [steering, accel] from last execution
    P = ca.vertcat(P, U_prev)

    # ... existing cost function ...

    # NEW: Slew rate penalty (commitment to previous control)
    # This is THE KEY to preventing oscillations!
    obj += self.w_slew_rate_steer * (U[0, 0] - U_prev[0])**2
    obj += self.w_slew_rate_accel * (U[0, 1] - U_prev[1])**2

    # Existing smoothness (within horizon) - keep this too
    for k in range(1, N):
        obj += self.w_smooth_steer * (U[k, 0] - U[k-1, 0])**2
        obj += self.w_smooth_accel * (U[k, 1] - U[k-1, 1])**2

    # ... rest of solver build ...
```

### Step 2: Track Last Executed Control

**File**: `mpc/teb_mpc.py` - Modify solve() method

```python
class TEBMPCSolver:
    def __init__(self, config_path):
        # ... existing init ...
        self._last_executed_control = np.array([0.0, 0.0])  # [steering, accel]

    def solve(self, state, goal, obstacles, profile="perpendicular"):
        # ... existing setup ...

        # Build parameter vector with previous control
        P = np.zeros(param_size)
        # ... fill goal, obstacles ...

        # NEW: Add last executed control to parameters
        P[-2] = self._last_executed_control[0]  # steering
        P[-1] = self._last_executed_control[1]  # accel

        # Solve
        res = solver(x0=x0_init, lbx=lbx, ubx=ubx, lbg=0, ubg=0, p=P)

        # ... extract solution ...

        # NEW: Store first control as last executed
        self._last_executed_control = U_sol[0].copy()

        return MPCSolution(X_sol, U_sol, True, {...})
```

### Step 3: Add Configuration

**File**: `mpc/config_mpc.yaml`

```yaml
profiles:
  parallel:
    # ... existing weights ...

    # NEW: Slew rate penalties (commitment to previous control)
    w_slew_rate_steer: 50.0   # Strong penalty for steering changes
    w_slew_rate_accel: 20.0   # Moderate penalty for accel changes

    # Existing smoothness (keep these)
    w_smooth_steer: 0.0015
    w_smooth_accel: 0.003
```

### Step 4: Test

```bash
# Test with slew rate penalty
python -m mpc.generate_expert_data --episodes 5 --scenario parallel --out-dir data/test_slew_rate
```

**Expected results**:
```
Before:
  Steps 30-35: steer = [0.35, 0.40, 0.38, 0.42, 0.36, 0.41]
  Oscillations: High frequency

After:
  Steps 30-35: steer = [0.35, 0.36, 0.37, 0.38, 0.39, 0.40]
  Oscillations: Smooth gradual changes

Success Rate: 100%
Precision: ~2.5cm (similar)
Oscillations: 50-70% reduction
```

---

## Combined Approach: Slew Rate + Move Blocking (Best Results)

For **maximum commitment**, combine Techniques A and B:

1. **Slew rate penalty**: Penalizes changes from last executed control
2. **Move blocking**: Forces 3-step blocks (0.3s commitment)

**Effect**:
- Slew rate prevents jumps between blocks
- Move blocking guarantees holds within blocks
- Result: Very smooth, human-like committed steering

**Expected performance**:
```
Steps: 45-50 (vs current 55)
Precision: ~2.0cm
Oscillations: 0-1 (vs current 1)
Pattern: [hold 0.52 for 0.3s] → [hold 0.0 for 0.3s] → [hold -0.45 for 0.3s]
```

---

## Summary

### Immediate Action (Today)
✅ **Implement Slew Rate Penalty (Technique A)**
- 1-2 hours implementation
- 50-70% oscillation reduction expected
- Minimal risk, high reward

### If Still Oscillating (Tomorrow)
✅ **Add Move Blocking (Technique B)**
- 4-6 hours implementation
- 80-90% oscillation reduction
- Guaranteed commitment

### Optional Enhancement
⚠️ **Multi-Step Execution (Technique C)**
- Only if needed for computational speedup
- Use adaptive logic (3 steps far, 1 step near goal)
- Careful with safety

### Expected Final Performance

**Current Baseline**: 55 steps, 2.6cm, 1 oscillation

**With Slew Rate Only**: 50 steps, 2.4cm, 0-1 oscillations

**With Slew Rate + Blocking**: 45 steps, 2.0cm, 0 oscillations ✅

---

## Next Steps

Ready to implement Slew Rate Penalty first?

It's the quickest win and should dramatically reduce oscillations with minimal risk.
