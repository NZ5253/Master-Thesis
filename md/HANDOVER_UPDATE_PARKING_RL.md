# Update Handover — Parking-RL (Current State + Shortcomings)

**Scope:** This document updates the earlier handover with the **current code state**, what has been changed recently (patches + behavior changes), what algorithms are used **per module**, what has been **tested**, and the **known shortcomings** that still prevent “always perfect” parking behavior.

> Goal of this codebase: generate **high-success expert demonstrations** for RL using a **staged** controller (A→B→WAIT→C) and a **TEB-like plan + fixed-dt NMPC track** pipeline.

---

## 0) Current status snapshot (what works / what regressed)

### ✅ Works now (confirmed in recent runs)
- **Hybrid staged pipeline runs end-to-end**:
  - A→B approach (receding NMPC)
  - WAIT (brake to stop)
  - B→C hybrid (TEB plan once → fixed-dt NMPC tracking)
- **Success termination is functioning** (no longer commonly ending as `max_steps` when “basically parked”).
- **Collision events reduced** after strengthening collision penalty scaling (post-smoothing).
- **CasADi NaN-gradient warnings reduced/removed in some runs** after smoothing non-smooth ops and stabilizing denominators.

### ⚠️ Current shortcomings (highest impact)
1) **Parking depth quality degraded**: vehicle stops **shallow** in the bay compared to earlier behavior (previously it executed a deeper “multi-turn / multi-correction” settling inside the bay).
2) **Collision sensitivity depends on collision smoothing + scaling**:
   - If smoothing is too “soft” → solver may accept penetration → env flags collision.
   - If collision scaling/margins are too strong → solver becomes overly conservative → shallow parking.
3) **The planner/tracker objective mismatch remains**:
   - TEB plan (variable dt) can propose a path that the tracker follows conservatively, especially near obstacles.
4) **No hard safety supervisor** (yet):
   - Collision is still handled primarily through **soft costs** (penalty terms) + env termination.

---

## 1) What changed recently (patch-level summary)

### 1.1 Termination & success consistency fixes
- **Termination priority** fixed to avoid overwriting `success` with `max_steps` in the same step:
  - Now: `collision > success > max_steps`.
- **Success tolerance control** added (via `parking.success` block in `config_env.yaml`) so the env success criteria can be tuned without code edits.
- **Expert-data logging** updated so `pos_err` and final errors match the env’s success metric (parallel uses **car-center** vs bay center).

**Net effect:** removes “looks parked but max_steps” failure mode and makes debug metrics trustworthy.

### 1.2 NaN-gradient mitigation in CasADi/Ipopt
- Replaced symbolic `ca.inf` initializers in min-distance logic with large finite values (`1e3`).
- Added small epsilons to denominators (e.g., `+1e-6`) to avoid division singularities.
- Smoothed non-differentiable primitives across the objective:
  - `fabs`, `fmin`, `fmax` replaced with smooth approximations (`sqrt(x^2+eps)` and smooth max/min).

**Net effect:** reduces/avoids `nlp_grad_f failed: NaN detected ...` warnings.

### 1.3 Collision re-tightening after smoothing
- After smoothing, collision avoidance became too “soft” in practice → collisions appeared.
- Collision penalty re-tightened with:
  - **collision margin / inflate**
  - **collision scaling** near contact (stronger penalty as distance approaches 0)
  - lower SDF smoothing epsilon (sharper but still differentiable)

**Net effect:** collisions reduced again, but **parking depth quality** became conservative/shallow.

---

## 2) Algorithms used (by module) + what was tested

This section is organized by file/module and describes:
- the algorithm used,
- what it outputs,
- what we tested and observed.

### 2.1 `env/parking_env.py` — Environment + termination
**Algorithm:** kinematic bicycle simulation (discrete time), rectangular world bounds, termination checks.

- **State**: `[x, y, yaw, v]` (rear-axle reference in dynamics)
- **Step loop**: applies `action=[steer, accel]`, clips to limits, integrates.
- **Termination**:
  - collision (from obstacle manager / bounds)
  - success (scenario-specific, parallel uses car-center alignment)
  - max_steps

**Tested:**
- Previously: “success overwritten by max_steps” caused false failures → fixed by using `elif` chain.
- Current: success works reliably with relaxed tolerances.

**Known shortcomings:**
- Success logic for parallel is geometric and can conflict with “deep bay” aesthetic unless tolerances and bay target are defined appropriately.
- No “stuck detection” or “oscillation stop” policy beyond success tolerances.

---

### 2.2 `env/obstacle_manager.py` — Obstacles + collision checking
**Algorithm:** rectangle obstacle representation, collision via oriented rectangle distance / overlap checks (and/or circle-to-rectangle approximations depending on internal usage).

**Change that mattered:** moving from **pin obstacles** (point-like / distance-based) to **rectangles** makes the collision geometry more realistic but also more restrictive.

**Tested:**
- Parallel: rectangular neighbors on both sides of bay.
- Observed: after smoothing + stronger collision margins, the vehicle keeps more clearance and parks shallow.

**Known shortcomings:**
- The optimizer sees a conservative “buffer” region; without a competing “depth reward” (or terminal constraint), it may accept shallow park as locally optimal.

---

### 2.3 `mpc/teb_mpc.py` — Core NMPC + TEB-like planner
This file contains **two related solvers** built with CasADi + Ipopt:

#### A) Fixed-dt NMPC (Tracker)
**Algorithm:** Nonlinear Model Predictive Control (NMPC)
- Horizon `N`, fixed `dt = config_env.dt` (e.g., 0.1s)
- Kinematic bicycle dynamics constraints
- Cost: goal tracking + control effort + smoothing + slew-rate + collision penalty + boundary penalty

**Tested:**
- As a pure controller (plan+track combined) and as a tracker in hybrid.
- Observed: stable tracking, but can be conservative near obstacles after collision scaling.

#### B) TEB-like planner (Plan-once reference)
**Algorithm:** “TEB-like” trajectory optimization
- Variable `DT[k]` inside `[dt_min, dt_max]`
- Objective includes timing penalty (and optionally obstacle-time scaling)
- Produces a reference trajectory (x,y,yaw,v,control) for the tracker

**Tested:**
- Hybrid mode uses planner once at B→C transition.
- Observed: planner succeeds frequently; warnings reduced after smoothing; reference can be feasible but does not guarantee deep bay behavior if the cost is conservative.

**Why NaN warnings happened (root cause):**
- Non-smooth ops (`fabs`, `fmax`, `fmin`) + symbolic `inf` caused derivative/multiplier evaluation issues in Ipopt.
- Smoothed versions reduce kinks and improve gradient stability.

**Known shortcomings:**
- Collision is still mostly a **soft cost** (penalty), not a strict inequality constraint. This can cause:
  - collision if penalty is too weak,
  - shallow solutions if penalty is too strong.
- Depth/“commitment into bay” behavior relies on cost shaping and is currently not strong enough.

---

### 2.4 `mpc/hybrid_controller.py` — Plan once, track many
**Algorithm:** hybrid controller
1) call TEB planner → get reference trajectory
2) each step: call fixed-dt MPC tracking solver to follow reference

**Tested:**
- Works with staged controller in parallel scenario.
- Past issue: returning `[0,0]` on solver fail lets vehicle drift (no friction) → mitigated by braking fallback in WAIT and (optionally) on tracking failure.

**Known shortcomings:**
- No automatic replan loop on tracking degradation; if tracking diverges, it should:
  - brake
  - replan from current state
- Planner/tracker mismatch can still yield conservative behavior.

---

### 2.5 `mpc/staged_controller.py` — A→B→WAIT→C orchestration
**Algorithm:** finite-state staged controller:
- **A→B:** approach NMPC profile
- **WAIT:** brake to stop
- **B→C:** hybrid controller for parking maneuver

**Tested:**
- This structure improved success rate vs “single MPC from far away”.

**Known shortcomings:**
- B pose selection is currently heuristic and can constrain the maneuver style.
- WAIT uses a one-step braking heuristic; could be improved with a short “stop MPC” or friction model.

---

### 2.6 `mpc/generate_expert_data.py` — Expert dataset generator
**Algorithm:** rollout runner with retry logic
- Runs episodes, records transitions, saves `.pkl` on success.
- Saves debug artifacts on failure (collision/max_steps/solver fail).
- Tracks best error during episode and prints detailed attempt summary.

**Tested:**
- After metric consistency fix, the printed `pos_err` matches env success logic for parallel.

**Known shortcomings:**
- “Retry until success” can hide systematic errors if success tolerances are too loose.
- It does not yet compute richer diagnostics (clearance minimum, solver status breakdown, constraint violation metrics).

---

## 3) Configuration: what matters most

### 3.1 `config_env.yaml`
Key knobs:
- `dt` (fixed sim dt)
- `max_steps` (currently kept at 300)
- `scenarios.parallel.approach_pose` defines B
- `scenarios.*.parking.success` defines success tolerances (if enabled)

**Current choice:** keep `max_steps=300` and rely on success tolerances to prevent timeouts.

### 3.2 `config_mpc.yaml`
Key knobs:
- profile weights (`APPROACH`, `PARALLEL`)
- collision parameters:
  - collision margin / obstacle inflate
  - collision scaling near contact
  - boundary scaling

**Trade-off in current state:**
- Stronger collision scaling → fewer collisions, but shallow bay entry.
- Weaker collision scaling → deeper entry, but higher collision risk.

---

## 4) Behavior regression: “shallow parking” (root cause + fix directions)

### Observed regression (current)
- Vehicle ends in bay but not deep enough; does not perform the earlier “two-stage deeper steer” behavior.

### Why it happens now
- Rectangular obstacle model + added safety buffer expands “keep-out” region.
- Smoothed collision cost activates earlier and pushes away continuously.
- Goal terms don’t explicitly reward “depth into bay” strongly enough relative to collision cost.

### Fix directions (recommended)
1) **Add an explicit depth objective** (parallel-specific):
   - reward decreasing longitudinal error *into* bay, not only lateral/yaw.
   - add terminal depth constraint (soft) stronger than general goal_xy.
2) **Use asymmetric collision margin**:
   - allow closer clearance when already aligned and moving slowly (final phase),
   - keep strong margins during entry.
3) **Add replan-on-stuck**:
   - if no improvement in best_pos_err for N steps, replan reference.
4) **Restore multi-turn style**:
   - introduce a staged “deepening” target: mid-goal inside bay then final goal (two terminal goals).

---

## 5) Debug & verification workflow (current best practice)

### Where failures go
- `data/expert_parallel_debug/` contains failures with:
  - termination reason
  - episode dump
  - visualization artifacts (if enabled)

### What to check first
- Is termination reason correct? (collision vs max_steps vs solver fail)
- Minimum clearance to obstacles over time (not yet automatically computed; can be added)
- Whether the failure occurs right after `WAIT -> B_to_C_hybrid` (common for tracking feasibility)

### Recommended test sweep
Run:
```bash
python3 -m mpc.generate_expert_data --scenario parallel --hybrid --episodes 20
```
Collect:
- success rate
- mean steps to success
- failure breakdown

---

## 6) “Definition of done” (updated)

We consider the system ready for large-scale expert generation when:
1) **Success rate** is consistently high (e.g., >95%) with the intended scenario distribution.
2) **Collisions** are rare and explainable (not caused by cost imbalance).
3) **No frequent solver warnings** (NaN gradients, multiplier failures).
4) **Parking quality** meets the project’s qualitative requirement:
   - “deep bay entry + settle” for parallel (not shallow stop near bay mouth).
5) Debug artifacts are sufficient to diagnose remaining failures quickly.

---

## 7) Immediate next tasks (priority ordered)

1) Implement parallel “depth commitment” shaping:
   - explicit depth target / terminal depth reward
2) Add a safety supervisor:
   - if predicted clearance < threshold, brake and/or replan
3) Add replan-on-stuck:
   - detect no improvement → replan reference trajectory
4) Add quantitative clearance logging:
   - minimum SDF distance per step; store in episode metadata

---

## 8) Commands cheat-sheet

Generate:
```bash
python3 -m mpc.generate_expert_data --scenario parallel --hybrid --episodes 1
```

Visualize:
```bash
python3 visualize_parking.py --scenario parallel --file data/expert_parallel/episode_0008.pkl
```
