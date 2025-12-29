# ROOT CAUSE ANALYSIS: Why MPC Stops at ~4.8cm Depth

## Problem Statement
The car consistently parks at **~4.8-5.0 cm** from the bay center instead of the desired **2-3 cm** "perfect" centering.

## Key Finding: **Steering is NOT Restricted!**

### The Misconception
We thought the "steering magnitude penalty" was preventing aggressive steering in the final phase.

### The Reality
```python
# Lines 373-385 in teb_mpc.py
well_aligned_depth = ca.fmin(1.0, ca.fmax(0.0, (0.025 - depth_err_abs) / 0.015))
proximity_moderate = ca.exp(-15.0 * pos_err_xy) * well_aligned_depth
proximity_final = ca.exp(-40.0 * pos_err_xy) * well_aligned_depth
steer_magnitude_penalty = (proximity_moderate * (steer_magnitude ** 4) +
                          proximity_final * 3.0 * (steer_magnitude ** 4))
```

**Analysis of Final Phase (Steps 38-52):**
- `depth_err_abs` = |rear_axle_y - goal_y| ≈ 0.048 m (4.8 cm)
- `well_aligned_depth` = (0.025 - 0.048) / 0.015 = **-1.53 → clipped to 0.0**
- Therefore: `proximity_moderate` = **0.0**, `proximity_final` = **0.0**
- **Result: Steering magnitude penalty = 0.0 (NEVER ACTIVATES!)**

The car **CAN** steer aggressively in the final phase, but the MPC solver **CHOOSES NOT TO**.

---

## The Real Problem: Insufficient Depth Incentive

### Current Cost Function Balance (Line 416-428)

```python
obj += gain * (
    self.w_goal_xy * 0.25 * lateral_err ** 2 +     # Lateral centering
    self.w_goal_xy * 4.0 * depth_penalty ** 2 +    # Penalty for backing out
    self.w_goal_xy * 0.03 * depth_reward +         # ← DEPTH REWARD (tiny!)
    # ... other costs
)
```

### Why MPC Stops at 4.8cm

1. **Weak Depth Reward**: 0.03 is extremely small
   - At 4.8cm depth: reward = -0.03 × 0.048 = -0.00144
   - Going 1cm deeper: additional reward = -0.03 × 0.01 = -0.0003
   - This is **negligible** compared to control effort costs

2. **Control Effort Costs**:
   - Steering: `w_steer * steer²` = 0.1 × (0.5)² = 0.025
   - To go deeper requires steering, which costs 100× more than the reward gained!

3. **Equilibrium Point**:
   - MPC finds equilibrium where: `depth_reward = control_effort_cost`
   - With current weights, this equilibrium is at **~4.8cm**

4. **Success Threshold**:
   - The 5cm success threshold tells MPC: "4.8cm is good enough"
   - No incentive to optimize beyond this

---

## Why Previous "Fixes" Failed

### Attempt 1: Increase Depth Reward to 0.15 (5× increase)
**Result**: Stuck at 6.4cm with max_steps failures

**Why**: Too aggressive, conflicted with other objectives:
- Solver became unstable
- Optimization couldn't converge
- Caused oscillations and increased depth error

### Attempt 2: Tighten Success Threshold to 2cm/3cm
**Result**: All episodes hit max_steps (300 steps) without success

**Why**: MPC cannot achieve that precision with current cost balance
- The equilibrium point is still ~4.8cm
- Tightening threshold doesn't change the optimization objective
- Just causes failures when actual depth doesn't meet threshold

### Attempt 3: Reduce Counter-Steering Penalty to 5.0
**Result**: Collision failures

**Why**: Allowed too much erratic steering during entry phase

---

## The Solution: Balanced Depth Reward Increase

### Strategy
Instead of 5× jump (0.03 → 0.15), use **gradual progressive increase**:

1. **Moderate increase**: 0.03 → **0.08** (2.67× increase)
   - Provides stronger incentive without overwhelming other objectives
   - Should push equilibrium from 4.8cm → ~3-3.5cm

2. **Adjust success threshold**: Keep at **3.5 cm**
   - More realistic than 2cm (which was too tight)
   - Provides clear target for MPC

3. **Fine-tune if needed**: If 0.08 works, try 0.10 for final push to 2-3cm

### Key Insight
The depth reward needs to be strong enough to **overcome control effort costs** but not so strong it **destabilizes the optimization**.

### Expected Behavior with 0.08 Depth Reward
- At 4.8cm depth: reward = -0.08 × 0.048 = -0.00384
- Going 1cm deeper: additional reward = -0.08 × 0.01 = -0.0008
- This is **2.67× stronger** incentive
- Should motivate MPC to steer more aggressively in final phase
- Combined with slow speed (already enforced), allows precise deep parking

---

## Implementation Plan

### Phase 1: Test 0.08 Depth Reward
```python
# Line 419 in teb_mpc.py
self.w_goal_xy * 0.08 * depth_reward +  # Moderate increase: 0.03 → 0.08
```

### Phase 2: Adjust Success Threshold
```python
# Lines 231-232 in parking_env.py
return (along_err < 0.035) and (lateral_err < 0.035)  # 3.5cm threshold
```

### Phase 3: Generate Dataset & Analyze
- Generate 30 episodes
- Check success rate (target: >60%)
- Check average depth (target: 3-3.5cm)
- If successful and depth still >3cm, try 0.10 reward for final optimization

---

## Why This Will Work

1. **Steering is already allowed**: The penalty never activates, so more depth reward directly translates to more aggressive steering

2. **Speed is already controlled**: Speed-steering coupling (0.7 weight) ensures slow movement during steering

3. **Progressive approach**: 2.67× increase is aggressive enough to see improvement but conservative enough to avoid instability

4. **Realistic target**: 3.5cm threshold is achievable (vs 2cm which was impossible)

5. **No structural changes**: Just tuning a single parameter that directly addresses the root cause
