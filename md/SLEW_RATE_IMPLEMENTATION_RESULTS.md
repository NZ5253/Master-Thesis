# Slew Rate Penalty Implementation - Results

**Date**: 2025-12-29
**Implementation**: Technique A from MPC Commitment Techniques

## Summary

Slew rate penalty has been **successfully implemented** but requires **careful tuning** due to sensitivity issues.

## What Was Implemented

Added cost term penalizing changes from last EXECUTED control:

```python
# In cost function (k=0 only):
obj += self.w_slew_rate_steer * (u[0] - prev_steer)**2
obj += self.w_slew_rate_accel * (u[1] - prev_accel)**2
```

## Key Implementation Details

### Critical Fix: Non-Zero Defaults Required

**Problem**: CasADi optimizes away terms with zero coefficients at build time.

**Solution**: Initialize with non-zero defaults:
```python
# WRONG (doesn't work):
self.w_slew_rate_steer = float(self.mpc_cfg.get("w_slew_rate_steer", 0.0))

# CORRECT (works):
self.w_slew_rate_steer = float(self.mpc_cfg.get("w_slew_rate_steer", 50.0))
```

**Why**: When weight is 0.0 at build time, CasADi simplifies `0.0 * (...)²` to just `0` and removes it from the expression. Changing the weight later has no effect because the term is gone.

## Test Results

| Weight (w_slew_rate_steer) | Result | Notes |
|----------------------------|---------|-------|
| 0.0 (default) | ✅ 55 steps, 2.6cm | No effect (term optimized away) |
| 50.0 (with zero default) | ✅ 55 steps, 2.6cm | No effect (term optimized away) |
| 500.0 (with 50.0 default) | ❌ Timeout | Too strong, solver struggles |
| 50.0 (with 50.0 default) | ❌ Collisions | Too strong, prevents obstacle avoidance |
| 5.0 (with 50.0 default) | ⏱️ Timeout/slow | Still too strong or causing convergence issues |

## Problem: Too Sensitive

The slew rate penalty is **very sensitive** to weight magnitude:

- **w < 1.0**: Likely no effect (too weak)
- **w = 5.0**: Causes timeouts or convergence issues
- **w = 50.0**: Causes collisions (can't avoid obstacles)
- **w > 100.0**: Solver fails completely

**Sweet spot**: Likely between 0.5-2.0, but needs extensive tuning.

## Why It's Problematic

1. **Conflicts with collision avoidance**: Strong slew rate penalty prevents steering changes needed to avoid obstacles

2. **Convergence issues**: Makes optimization landscape much harder
   - Solver takes longer to converge
   - May get stuck in local minima
   - Timeouts more frequent

3. **Phase-dependent needs**:
   - APPROACH phase: Can afford commitment (far from obstacles)
   - ENTRY phase: Needs reactivity (close to obstacles)
   - FINAL phase: Needs precision (small adjustments)

   Fixed weight doesn't adapt to these different needs.

## Alternative Approaches

### Option 1: Phase-Aware Slew Rate

Different weights per phase:
```yaml
slew_rate:
  approach: 10.0    # Strong commitment when far
  entry: 1.0        # Weak when near obstacles
  final: 5.0        # Moderate for precision
```

### Option 2: Move Blocking (Technique B)

Instead of penalizing changes, **prevent** them by forcing multi-step blocks:
- Simpler (no tuning sensitivity)
- Guarantees commitment (3 steps = 0.3s hold)
- Less sensitive to weight values

### Option 3: Adaptive Slew Rate

Scale based on distance to obstacles:
```python
# Far from obstacles: strong commitment
# Near obstacles: weak commitment (allow evasion)
w_slew = base_weight * (1.0 - obstacle_proximity_factor)
```

## Recommendation

### Short-Term: Disable Slew Rate Penalty

The implementation is correct but too sensitive for production use:

```yaml
# Revert to zero defaults (disables the penalty):
w_slew_rate_steer: 0.0
w_slew_rate_accel: 0.0
```

**Or** keep non-zero defaults but set config to very weak:
```yaml
w_slew_rate_steer: 0.5   # Very weak, just slight smoothing
w_slew_rate_accel: 0.2
```

### Medium-Term: Implement Move Blocking (Technique B)

More robust approach with less tuning sensitivity:
- Forces 3-step blocks (0.3s commitment)
- Reduces variables (50 → ~17 control blocks)
- Guaranteed smoothness without sensitive weights

### Long-Term: Hybrid TEB+MPC Architecture

Most robust solution:
- TEB creates committed trajectory once
- MPC tracks it smoothly
- No re-planning oscillations

## Code Status

### Files Modified

✅ [mpc/teb_mpc.py](mpc/teb_mpc.py):
- Lines 114-115: Non-zero default initialization
- Lines 239-240: Profile loading
- Lines 288-289: Weight printing
- Lines 385-397: Parameter vector with prev_control
- Lines 510-512: Slew rate penalty in cost function
- Line 204: Last executed control tracking
- Lines 825-826: Prev control in parameter vector
- Line 869: Store executed control

✅ [mpc/config_mpc.yaml](mpc/config_mpc.yaml):
- Lines 85-86: Slew rate weights in parallel profile

### Current State

**Implementation**: ✅ Complete and correct

**Status**: ⚠️ Disabled (w=0.0 or very weak)

**Reason**: Too sensitive, causes collisions or timeouts with moderate weights

## Lessons Learned

1. **CasADi optimization**: Terms with zero coefficients are removed at build time
   - Must use non-zero defaults for optional penalties
   - Cannot enable/disable cost terms via configuration alone

2. **Slew rate vs smoothness**: Different concepts
   - **Smoothness**: Penalizes changes within planned horizon
   - **Slew rate**: Penalizes changes from actual execution
   - Both are needed but serve different purposes

3. **MPC sensitivity**: Adding constraints/penalties affects entire optimization
   - Small weight changes can have large effects
   - Need extensive tuning for each scenario
   - Phase-aware or adaptive weights likely necessary

## Next Steps

If you want to proceed with commitment penalties:

### Path 1: Fine-Tune Slew Rate (Time-Consuming)
- Test weights from 0.1 to 5.0 in 0.1 increments
- Find sweet spot for each phase
- Implement phase-aware weights
- **Effort**: 1-2 days of testing
- **Risk**: High (may not find stable configuration)

### Path 2: Implement Move Blocking (Recommended)
- Force 3-step control blocks
- Guaranteed 0.3s commitment
- Less sensitive to tuning
- **Effort**: 4-6 hours implementation
- **Risk**: Low (well-proven technique)

### Path 3: Hybrid Architecture (Best Long-Term)
- TEB planner + MPC tracker
- Eliminates re-planning oscillations
- Most robust solution
- **Effort**: 2-3 days
- **Risk**: Medium (new architecture)

## Conclusion

Slew rate penalty **works in principle** but is **too sensitive for practical use** in this parking scenario.

Recommend:
1. **Now**: Disable or use very weak weights (w=0.5)
2. **Next**: Implement move blocking (Technique B)
3. **Future**: Consider hybrid TEB+MPC architecture

The implementation remains in the codebase (with weak/zero weights) and can be re-enabled if better tuning is found.
