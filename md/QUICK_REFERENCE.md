# TEB-MPC Baseline - Quick Reference

## Status: ✅ VERIFIED WORKING

**Date**: 2025-12-29
**Version**: Baseline with 3 fixes (Fixed-dt MPC)

### System Status
❌ TEB: **DISABLED** - Causes more oscillations (see [TEB_OSCILLATION_ANALYSIS.md](TEB_OSCILLATION_ANALYSIS.md))
✅ Profile Loading: Working correctly (verified)
✅ Config Changes: Applied at runtime (verified)
✅ All 3 fixes: Active

## Performance Summary

**Fixed-dt MPC (TEB Disabled)**:
```
✅ Success Rate: 100%
✅ Precision: 2.6 cm
✅ Steps: 55
✅ Major Oscillations: 1 (acceptable)
```

**TEB-Enabled (NOT RECOMMENDED)**:
```
✅ Success Rate: 100%
✅ Precision: 2.1 cm (0.5cm better)
⚠️  Steps: 70 (27% more steps)
❌ Major Oscillations: 2 (worse than fixed-dt)
```

## Configuration Files

### Main Config: `mpc/config_mpc.yaml`

**TEB Settings** (line 39):
```yaml
enable: false  # DISABLED - Fixed-dt MPC has fewer oscillations
```
See [TEB_OSCILLATION_ANALYSIS.md](TEB_OSCILLATION_ANALYSIS.md) for details.

**Parallel Weights** (lines 77-103):
```yaml
w_goal_xy: 400.0
w_goal_theta: 120.0
lateral_weight: 0.25
yaw_weight: 0.9
proximity_exp_factor: 50.0  # FIX #2
```

### Main Solver: `mpc/teb_mpc.py`

**Key Sections**:
- Lines 421-438: Progress penalty (FIX #1)
- Lines 443-449: Depth reward (uses FIX #2)
- Line 466: Coupling minimum (FIX #3)
- Lines 597-621: TEB dt_precision (commented for baseline)

## The 3 Implemented Fixes

| Fix | Location | What It Does |
|-----|----------|--------------|
| #1 Progress Penalty | teb_mpc.py:430-436 | Gaussian activation keeps penalty active when close |
| #2 Proximity Factor | config_mpc.yaml:103 | Tighter reward zone (20.0 → 50.0) |
| #3 Coupling Minimum | teb_mpc.py:466 | Prevents erratic steering (min 0.7) |

## Running Tests

```bash
# Activate environment
source venv/bin/activate

# Run 5 episodes
python -m mpc.generate_expert_data --episodes 5 --scenario parallel --out-dir data/test_output

# Expected: All 5 succeed, ~70 steps each, ~2cm precision
```

## Zig-Zag Explanation

**Why it happens**: Yaw cost (108) > Lateral cost (100)
**Pattern**: Car approaches to 3cm → drifts to 12cm → corrects → repeat
**Can it be eliminated**: No, not with MPC weight tuning
**Is it a problem**: No, 100% success rate achieved

## What We Tried (All Failed)

- ❌ Reduce yaw weight → Solver timeout
- ❌ Boost lateral weight → 5 moved-away (worse)
- ❌ Reduce coupling → 200+ steps, unstable
- ❌ Increase dt_max → Precision loss
- ❌ Enable dt_precision → Still oscillates

**Conclusion**: Zig-zag is fundamental to MPC architecture

## To Eliminate Zig-Zag (Future Work)

Need architectural change:
1. Path planning (Hybrid A*, Reeds-Shepp)
2. Generate committed trajectory
3. MPC tracks trajectory instead of planning

See: [committed_steering_solution.md](committed_steering_solution.md)

## Documentation

- **This file**: Quick reference
- [BASELINE_SUMMARY.md](BASELINE_SUMMARY.md): Complete baseline analysis
- [zigzag_root_cause_final.md](zigzag_root_cause_final.md): Detailed investigation
- [committed_steering_solution.md](committed_steering_solution.md): Path planning approach

## Test Data Locations

- `data/expert_baseline_final/`: Latest baseline test
- `data/expert_parallel/`: Earlier validation tests (eps 50-54)

## Code Health

✅ All fixes implemented
✅ All experiments reverted
✅ Configuration verified
✅ Tests passing
✅ 100% success rate

**Status**: Production ready
