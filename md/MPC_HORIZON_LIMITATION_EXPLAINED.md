# MPC Horizon Limitation - Why It Can't Commit to Steering

**Date**: 2025-12-29
**Question**: Why can't MPC/TEB-MPC control "steer for 1 second" like humans do?

## The Core Problem: Receding Horizon

### What MPC Does

**Current Implementation**:
```python
# In teb_mpc.py
self.N = 50              # Horizon length (number of waypoints)
self.dt = 0.10           # Time step (100ms)
# Total horizon: 50 × 0.1s = 5 seconds look-ahead
```

**At Each Control Step**:
1. MPC looks ahead 5 seconds (50 steps)
2. Optimizes control sequence to minimize cost
3. Executes ONLY the first control (0.1s)
4. **Discards the rest** of the plan
5. Next step: Repeats from scratch with new state

**This is called "Receding Horizon"**:
```
Step 1: Plan [t=0 → t=5s], execute t=0
Step 2: Plan [t=0.1 → t=5.1s], execute t=0.1  (RE-PLAN!)
Step 3: Plan [t=0.2 → t=5.2s], execute t=0.2  (RE-PLAN!)
...
```

### Why This Prevents Committed Steering

**Human Parking**:
```
"I need to steer full lock right for 1.5 seconds to get into position"
→ Commits to: steering = +0.52 rad for 15 steps
→ Executes: No changes for 1.5s, just holds steering
```

**MPC Parking**:
```
Step 1:  "Steer right 0.52 rad to minimize cost over next 5s"
Step 2:  "Hmm, I moved, let me RE-OPTIMIZE... steer 0.48 rad"
Step 3:  "Moved again, RE-OPTIMIZE... steer 0.51 rad"
Step 4:  "Getting close to wall, RE-OPTIMIZE... steer 0.30 rad"
Step 5:  "Yaw error increased, RE-OPTIMIZE... steer 0.45 rad"
...
→ Result: Constantly changing steering, never commits!
```

## Visualization of the Problem

### Committed Steering (Human)
```
Time:     0s   0.5s  1.0s  1.5s  2.0s  2.5s  3.0s
Steering: 0.52  0.52  0.52  0.52  0.0   0.0  -0.40
          └────────────────┘      └────┘  └─────┘
           COMMITTED TURN        HOLD    REVERSE TURN

Pattern: Long segments of constant steering
```

### Receding Horizon MPC
```
Time:     0s   0.1s  0.2s  0.3s  0.4s  0.5s  0.6s
Steering: 0.52  0.48  0.51  0.30  0.45  0.38  0.42
          └─┘  └─┘  └─┘  └─┘  └─┘  └─┘  └─┘
         RE-OPT RE-OPT RE-OPT (every step!)

Pattern: Constantly adjusting, never commits
```

## Why Re-Planning Happens

### Cost Function Myopia

At each step, MPC sees a **different optimization problem**:

**Step 1** (at position A):
```python
# State: Far from goal, need to turn
# Optimal: Steer hard right (0.52)
minimize: goal_distance + yaw_error + ...
→ Best action: steer = 0.52
```

**Step 2** (at position B, after steering 0.52):
```python
# State: Moved right, lateral error changed!
# NEW optimization problem (different state)
minimize: goal_distance + yaw_error + ...
→ Best action: steer = 0.48 (DIFFERENT!)
```

**Why?** The cost function is evaluated from the NEW state:
- Lateral error changed (moved sideways)
- Yaw changed (rotated)
- Distance to obstacles changed
- **Optimal solution is DIFFERENT now!**

### Example: Real Data from Tests

From our test episodes:
```
Step 30: pos_err=0.077m, steer= 0.35
Step 31: pos_err=0.081m, steer= 0.40  (increased)
Step 32: pos_err=0.085m, steer= 0.38  (decreased!)
Step 33: pos_err=0.090m, steer= 0.42  (increased)
Step 34: pos_err=0.088m, steer= 0.36  (decreased!)
Step 35: pos_err=0.092m, steer= 0.41  (increased)
```

**Pattern**: Steering oscillates every step because MPC re-optimizes!

## The Horizon Length Problem

### Why Not Just Increase Horizon?

**Attempt**: Use longer horizon to "see" the whole maneuver
```python
self.N = 200  # 200 × 0.1s = 20 seconds horizon
```

**Problems**:
1. **Computational explosion**: 200 states × 4 variables = 800 optimization variables
   - Solve time: 0.1s → 5+ seconds (too slow for real-time)
2. **Still re-optimizes every step**: Even with 20s horizon, MPC re-plans at step 2
3. **Numerical issues**: Long horizons are unstable for optimization

**Result**: Doesn't solve the commitment problem, just makes it slower.

### Why Not Reduce Re-Planning Frequency?

**Attempt**: Only re-plan every 1 second instead of every 0.1s
```python
if step % 10 == 0:  # Re-plan every 1s
    solution = mpc.solve(state, goal, obstacles)
else:
    use_previous_plan[step]
```

**Problems**:
1. **Open-loop execution**: Following old plan without feedback → drifts away
2. **No disturbance rejection**: Can't react to obstacles or errors
3. **Reduced robustness**: One bad plan = 1 second of wrong controls

**Result**: Worse performance, loses main MPC advantage (feedback).

## Why TEB Doesn't Fix This

### What We Hoped TEB Would Do

```
"TEB allows variable dt, so it can use dt=0.3s for committed maneuvers"
→ Expected: Large dt = longer steering hold
```

### What TEB Actually Does

```python
# TEB solver at each step
Step 1: Optimize DT[0:50] and U[0:50]
        → DT = [0.20, 0.18, 0.25, 0.15, ...]  (variable)
        → U  = [0.52, 0.48, 0.51, 0.30, ...]

Step 2: RE-OPTIMIZE DT[0:50] and U[0:50] from NEW state
        → DT = [0.22, 0.16, 0.28, 0.12, ...]  (DIFFERENT!)
        → U  = [0.48, 0.51, 0.45, 0.35, ...]  (DIFFERENT!)
```

**Problem**: TEB still re-optimizes every step!
- dt values change every step
- Controls change every step
- **No commitment** to previous plan

**Why?** Receding horizon applies to TEB too:
- Each step is a new optimization problem
- dt and controls are re-computed from scratch
- Previous solution is discarded (except warm-start)

### Our Test Results Confirmed This

**TEB-enabled**:
```
Step 30: dt=0.103, steer=0.35
Step 31: dt=0.097, steer=0.40  (both changed!)
Step 32: dt=0.112, steer=0.38  (both changed again!)
```

**Result**: TEB has 50 MORE variables to optimize (DT array) but still re-plans every step!

## Mathematical Explanation

### MPC Optimization at Time t

```
At time t, state x(t):

minimize  Σ cost(x[k], u[k])  for k=0..N
          k=0

subject to:
  x[k+1] = f(x[k], u[k])      (dynamics)
  x[0] = x(t)                  (current state)
  control bounds
  collision constraints

→ Solution: u*[0], u*[1], ..., u*[N-1]
→ Execute: u*[0]
```

### At Time t+dt (Next Step)

```
At time t+dt, state x(t+dt):  (NEW STATE!)

minimize  Σ cost(x[k], u[k])  for k=0..N  (SAME PROBLEM FORM)
          k=0

subject to:
  x[k+1] = f(x[k], u[k])
  x[0] = x(t+dt)               (DIFFERENT INITIAL CONDITION!)
  control bounds
  collision constraints

→ Solution: u*[0], u*[1], ..., u*[N-1]  (DIFFERENT SOLUTION!)
→ Execute: u*[0]
```

**Key insight**: Even though problem structure is same, **initial condition changed** → solution changes!

### Why Solution Changes

Consider lateral error in cost function:
```python
lateral_err = (x - goal_x)**2
```

**At Step 1**: x = 0.87, goal_x = -0.13
```
lateral_err = (0.87 - (-0.13))^2 = 1.0^2 = 1.0
→ Strong pull to move left (negative steering)
```

**At Step 2**: x = 0.85 (moved slightly left), goal_x = -0.13
```
lateral_err = (0.85 - (-0.13))^2 = 0.98^2 = 0.96
→ Pull to left is slightly weaker (DIFFERENT GRADIENT!)
→ Optimal steering is DIFFERENT!
```

**This happens EVERY step** → constant re-optimization → no commitment.

## The Solution: Separate Planning from Control

### Why Hybrid Approach Works

**TEB Planner (Runs Once)**:
```
At t=0, state x(0):

minimize  Σ cost(x[k], u[k]) + time_cost  for k=0..100
          k=0

→ Solution: u*[0..100], dt*[0..100]
→ Store ENTIRE trajectory as reference
→ NO RE-PLANNING!
```

**MPC Tracker (Runs Every Step)**:
```
At each time t, state x(t):

minimize  Σ tracking_cost(x[k], ref[k]) + ...
          k=0

where ref[k] is from TEB plan (FIXED reference)

→ Solution: u*[0..N-1]
→ Execute: u*[0]
→ Reference doesn't change, so tracking is smooth!
```

**Key Difference**:
- **Planner**: Optimizes ONCE to create committed plan
- **Tracker**: Follows plan with small corrections, doesn't re-plan

### Committed Steering Example

**TEB creates reference** (once):
```
ref[0:15]:   steering = 0.52 (hold for 1.5s)
ref[15:30]:  steering = 0.0  (straighten for 1.5s)
ref[30:45]:  steering = -0.40 (reverse turn for 1.5s)
```

**MPC tracks reference** (every step):
```
Step 1:  ref_steer=0.52, actual=0.50 (small error)
Step 2:  ref_steer=0.52, actual=0.51 (correcting)
Step 3:  ref_steer=0.52, actual=0.52 (matched!)
Step 4:  ref_steer=0.52, actual=0.52 (tracking)
...
Step 15: ref_steer=0.52, actual=0.52 (still tracking)
Step 16: ref_steer=0.0,  actual=0.48 (transition)
Step 17: ref_steer=0.0,  actual=0.30 (correcting)
Step 18: ref_steer=0.0,  actual=0.10 (almost there)
```

**Result**: Smooth committed steering segments, just like human!

## Analogy: GPS Navigation

### Current MPC (Receding Horizon)
```
Like a GPS that re-calculates route every 100ms:

"Turn right in 500m"
[100ms later] "Turn right in 450m"  (re-calculated!)
[100ms later] "Turn right in 400m"  (re-calculated!)
[100ms later] "Actually, turn left in 350m"  (changed mind!)
[100ms later] "No wait, go straight 300m"  (changed again!)

Result: Driver constantly adjusting, never commits to a turn
```

### Hybrid TEB+MPC (Plan Once, Track)
```
Like a GPS that plans once, then guides:

[At start] "Turn right in 500m, hold for 200m, then left"
[During] "Stay on course... 400m to turn"
[During] "Stay on course... 300m to turn"
[At turn] "Turn right now... good... hold steering"
[After turn] "Perfect, now prepare to turn left"

Result: Driver commits to turns, smooth driving
```

## Why This Matters for Parking

### Human Parallel Parking
```
Phase 1: "Steer full lock right, back up for 1.5s"
         → Commits to steering=+0.52 for 1.5s
         → Vehicle rotates and moves into space

Phase 2: "Now straighten steering, back up for 0.5s"
         → Commits to steering=0.0 for 0.5s
         → Vehicle moves straight back

Phase 3: "Steer full lock left, back up for 1.0s"
         → Commits to steering=-0.52 for 1.0s
         → Vehicle aligns with bay
```

**Total**: 3 committed maneuvers, ~3 seconds, parked!

### Current MPC Parallel Parking
```
Step 1-55: Constantly re-optimizing every 0.1s
           → Steering changes 50+ times
           → Oscillates approach→drift→correct
           → Takes 5.5 seconds
           → 1 zig-zag present
```

### Expected Hybrid TEB+MPC Parking
```
TEB plans (once):
  Steps 1-15:  steering=+0.52 (committed right turn)
  Steps 16-20: steering=0.0   (committed straight)
  Steps 21-35: steering=-0.45 (committed left turn)
  Steps 36-45: steering=small (precision alignment)

MPC tracks (every step):
  Just follows reference with small corrections
  No re-planning → no oscillations

Result: 3-4 committed maneuvers, ~4 seconds, ZERO zig-zag!
```

## Summary

### The Problem
- **MPC re-optimizes every 0.1s** from new state
- Each new state → different cost gradients → different optimal controls
- **Can't commit** to "steer for 1s" because plan changes every step
- Horizon length doesn't fix it (still re-plans)
- TEB doesn't fix it (re-optimizes dt AND controls every step)

### The Solution
- **TEB plans once** at start: Creates committed trajectory
- **MPC tracks** reference: Follows plan without re-planning
- Reference is fixed → tracking is smooth → no oscillations
- Multi-phase tracking adapts to approach/entry/final phases

### Why It Works
Separation of concerns:
- **Planning** (TEB): "What trajectory should I follow?" (global optimization)
- **Control** (MPC): "How do I follow it?" (local tracking)

This is how humans drive:
- **Plan**: "I'll steer right for 1.5s to get into the space"
- **Execute**: Follow plan while making small corrections

And it's how autonomous vehicles work:
- **Planner**: Hybrid A* / RRT* / TEB creates reference path
- **Controller**: MPC / Pure Pursuit tracks reference path

## Next Steps

Ready to implement the hybrid approach to achieve committed steering?

The implementation plan is in [IMPLEMENTATION_PLAN_TEB_MPC_HYBRID.md](IMPLEMENTATION_PLAN_TEB_MPC_HYBRID.md).
