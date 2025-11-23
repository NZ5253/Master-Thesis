# parking-rl (Thesis repository skeleton)
This repository contains templates and starter code for the Master's thesis:
"Reinforcement Learning for Autonomous Valet Parking with the Chronos Car (CRC)".

Structure:
- env/: Simulation environment (Gym-like)
- mpc/: MPC baseline (CasADi templates) + expert data generation
- rl/: RL agents, training scripts, BC pretraining
- real_crc/: Motion-capture interface & CRC control stubs
- simulation/: visualization & scenario runners
- data/: expert trajectories & logs
- utils/: helper functions

**Notes**
- The MPC module uses CasADi/ACADOS in full implementations. The template includes a guarded import and fallback.
- These templates are minimal but runnable for simulation-only experiments. Replace placeholders with real CRC parameters and MoCap endpoints.
