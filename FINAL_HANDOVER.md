# Final Project Handover - Parallel Parking RL

**Project**: Autonomous Parallel Parking with Deep Reinforcement Learning
**Date**: 2026-02-01 (Updated)
**Status**: **TRAINING COMPLETE** - All 7 Phases Passed
**Version**: 3.0 (Post-Training Analysis)

---

## Executive Summary

### Training Completed Successfully

All 7 curriculum phases have been completed with the following results:

| Phase | Status | Success Rate | Threshold | Timesteps |
|-------|--------|--------------|-----------|-----------|
| Phase 1 | **PASSED** | 100% | 85% | 596K |
| Phase 2 | **PASSED** | 100% | 80% | 516K |
| Phase 3 | **PASSED** | 93% | 75% | 276K |
| Phase 4a | **PASSED** | 83% | 70% | 1.02M |
| Phase 4 | **PASSED** | 87% | 70% | 762K |
| Phase 5 | **PASSED** | 90% | 65% | 442K |
| Phase 6 | **PASSED** | 90% | 60% | 632K |

**Total Training**: ~4.2M timesteps
**Best Checkpoint**: `checkpoints/curriculum/curriculum_20260129_205818/phase6_random_obstacles/best_checkpoint`

### Critical Bug Found and Fixed

**Random obstacles were NOT being added during Phase 6 training!**

The `add_random_obstacles()` method existed but was never called. This has been **FIXED** in `env/parking_env.py`.

**Impact**: Phase 6 training was effectively the same as Phase 5 (neighbor jitter only, no random obstacles).

---

## Current State

### Checkpoint Location

```
checkpoints/curriculum/curriculum_20260129_205818/
├── phase1_foundation/final_checkpoint      (100% success)
├── phase2_random_spawn/final_checkpoint    (100% success)
├── phase3_random_bay_x/final_checkpoint    (93% success)
├── phase4a_random_bay_y_small/final_checkpoint (83% success)
├── phase4_random_bay_full/final_checkpoint (87% success)
├── phase5_neighbor_jitter/final_checkpoint (90% success)
├── phase6_random_obstacles/final_checkpoint (90% success) ← Best checkpoint
└── training_log.yaml
```

### Current Tolerance Settings (TOO LOOSE)

The current success tolerances are quite generous:

| Parameter | Current Value | Recommended | Notes |
|-----------|---------------|-------------|-------|
| `along_tol` | 0.20m (20cm) | 0.08m (8cm) | Car not centered depth-wise |
| `lateral_tol` | 0.20m (20cm) | 0.08m (8cm) | Car not centered laterally |
| `yaw_tol` | 0.20 rad (11°) | 0.10 rad (6°) | Car not aligned properly |
| `v_tol` | 0.20 m/s | 0.08 m/s | Car still moving at "success" |

These loose tolerances mean the car can be significantly off-center and still count as "success".

---

## Bugs Fixed in This Session

### Bug 1: Random Obstacles Not Appearing (FIXED)

**Problem**: Phase 6 should add 0-2 random obstacles, but none appeared during evaluation.

**Root Cause**: The `add_random_obstacles()` method in `ObstacleManager` was never called from `parking_env.py`.

**Fix Applied** in `env/parking_env.py`:
```python
# After spawn position is determined, add random obstacles
self.obstacles.add_random_obstacles(self.state)
```

**Status**: **FIXED**

---

## Known Issues (For Future Work)

### Issue 1: Loose Tolerances

The car often stops before reaching proper depth/centering because tolerances are too loose.

**Solution**: Tighten tolerances in curriculum config for future training:
```yaml
success:
  along_tol: 0.08      # 8cm (was 20cm)
  lateral_tol: 0.08    # 8cm (was 20cm)
  yaw_tol: 0.10        # ~6 degrees (was ~11 degrees)
  v_tol: 0.08          # Nearly stopped (was 20cm/s)
```

### Issue 2: Timeout on Difficult Spawns

When the car spawns at extreme positions (far yaw angle, edge of spawn range), it sometimes doesn't attempt to park and just times out.

**Possible Causes**:
1. Policy wasn't trained on enough edge cases
2. Reward shaping may not incentivize movement from extreme positions
3. Stall penalty may not be strong enough

**Potential Solutions**:
1. Fine-tune with more aggressive exploration (`entropy_coeff: 0.03+`)
2. Increase `stall_penalty` to punish standing still
3. Add curriculum phase with extreme spawn angles
4. Increase training steps on Phase 6

### Issue 3: Steering Oscillations

Some parking maneuvers show excessive steering corrections near the goal.

**Potential Solutions**:
1. Increase `action_smooth_w` penalty in reward
2. Add `steer_rate_penalty` for rapid steering changes
3. Lower action scaling near goal (already partially implemented)

---

## Future Tasks (Priority Order)

### High Priority

1. **Retrain Phase 6 with Random Obstacles**
   - The random obstacles bug is now fixed
   - Fine-tune from current Phase 5 checkpoint with actual random obstacles
   ```bash
   # Suggested command (create a fine-tuning script)
   python -m rl.train_curriculum --start-phase phase6_random_obstacles \
       --checkpoint checkpoints/curriculum/curriculum_20260129_205818/phase5_neighbor_jitter/final_checkpoint
   ```

2. **Tighten Success Tolerances**
   - Update `rl/curriculum_config.yaml` with stricter tolerances
   - Retrain or fine-tune with new tolerances
   - Target: 8cm position, 6° yaw, 8cm/s velocity

3. **Fix Timeout Behavior**
   - Investigate why car doesn't attempt parking from extreme spawns
   - Adjust reward shaping or add explicit "movement" bonus
   - Consider adding a "recovery" curriculum phase

### Medium Priority

4. **Reduce Steering Oscillations**
   - Add/increase smoothness penalties
   - Evaluate with motion quality metrics (already implemented in `train_curriculum.py`)
   - Target: <5 steering direction changes per episode

5. **Improve Evaluation Metrics**
   - Track min clearance to obstacles during parking
   - Track number of gear switches (forward/reverse changes)
   - Track time to completion

### Low Priority

6. **Add Perpendicular Parking Support**
   - Environment already supports perpendicular scenario
   - Need curriculum config for perpendicular parking
   - Could share weights from parallel parking

7. **Real-World Transfer**
   - Domain randomization for sim-to-real
   - Add noise to observations
   - Add actuator delays

---

## Quick Commands

### Evaluate Current Checkpoint
```bash
./eval_with_viz.sh \
    --checkpoint "checkpoints/curriculum/curriculum_20260129_205818/phase6_random_obstacles/best_checkpoint" \
    --num-episodes 10 \
    --speed 1.0 \
    --phase-name "phase6_random_obstacles"
```

### Verify Configuration
```bash
python verify_all_phases.py
```

### Test Random Obstacles Fix
```bash
# After the fix, random obstacles should now appear
python -c "
from rl.curriculum_env import make_curriculum_env
env = make_curriculum_env('parallel', 'phase6_random_obstacles')
obs, info = env.reset()
pe = env.env.env  # Unwrap to ParkingEnv
random_obs = [o for o in pe.obstacles.obstacles if o.get('kind') == 'random']
print(f'Random obstacles: {len(random_obs)}')
for o in random_obs:
    print(f'  - Position: ({o[\"x\"]:.2f}, {o[\"y\"]:.2f}), Size: {o[\"w\"]:.2f}x{o[\"h\"]:.2f}')
"
```

### Start Fresh Training with Tighter Tolerances
```bash
# 1. First update curriculum_config.yaml with tighter tolerances
# 2. Then run:
./quick_train.sh curriculum
```

---

## Technical Reference

### Observation Space (7D)
```
[along,        # Car-center to bay-center along bay axis (m)
 lateral,      # Car-center to bay-center perpendicular (m)
 yaw_err,      # Heading error: bay_yaw - car_yaw (rad)
 v,            # Velocity (m/s)
 dist_front,   # Distance to obstacle ahead (m)
 dist_left,    # Distance to obstacle left (m)
 dist_right]   # Distance to obstacle right (m)
```

### Action Space (2D)
```
[steering,      # [-1, 1] -> [-0.523, 0.523] rad
 acceleration]  # [-1, 1] -> [-1.0, 1.0] m/s^2
```

### Reward Components
- **Success**: +200 (terminal)
- **Collision**: -200 (terminal)
- **Timeout**: -400 (terminal)
- **Progress**: Position/yaw improvement toward goal
- **Smoothness**: Penalty for rapid action changes
- **Time**: Small per-step penalty (-0.008)

### Key Files
| File | Purpose |
|------|---------|
| `rl/curriculum_config.yaml` | 7-phase curriculum definition |
| `rl/train_curriculum.py` | Main training script |
| `rl/gym_parking_env.py` | Gymnasium wrapper |
| `env/parking_env.py` | Core environment (reset, step, success) |
| `env/obstacle_manager.py` | Obstacles & collision detection |
| `env/reward_parallel_parking.py` | RL reward function |
| `config_env.yaml` | Base environment config |

---

## Summary

### Completed
- 7-phase curriculum training completed
- All phases passed their success thresholds
- Random obstacles bug identified and fixed
- Training achieves 90% success on Phase 6

### Issues Identified
1. Random obstacles weren't being added (FIXED)
2. Success tolerances too loose (car stops early)
3. Timeout behavior on difficult spawns
4. Some steering oscillations

### Next Steps
1. Retrain Phase 6 with random obstacles (bug now fixed)
2. Tighten tolerances for proper depth/centering
3. Address timeout behavior from extreme spawns
4. Reduce steering oscillations

---

**Last Updated**: 2026-02-01
**Version**: 3.0
**Status**: Training complete, improvements needed

**Best Checkpoint**: `checkpoints/curriculum/curriculum_20260129_205818/phase6_random_obstacles/best_checkpoint`
