#!/bin/bash
# =========================================================
# New Car Training Script
# =========================================================
# Full curriculum training with NEW car dimensions + friction model.
#
# Dimensions:
#   - Wheelbase: 0.113m (vs 0.25m original)
#   - Length: 0.15m (vs 0.36m original)
#   - Width: 0.10m (vs 0.26m original)
#   - Max steer: 0.314 rad / 18 deg (vs 0.523 rad original)
#
# Friction model baked into training:
#   - static_friction:  0.15 m/s^2 (dead zone at standstill)
#   - kinetic_friction: 0.05 m/s^2 (opposes motion)
#
# Expected training time: ~690M steps across 7 phases
# =========================================================

set -e

echo "========================================="
echo "New Car Parallel Parking Training"
echo "========================================="
echo ""
echo "Configuration:"
echo "  Environment: config_env_newcar.yaml"
echo "  Curriculum:  rl/curriculum_config_newcar.yaml"
echo ""
echo "Vehicle dimensions:"
echo "  Wheelbase:  0.113m"
echo "  Length:     0.15m"
echo "  Width:      0.10m"
echo "  Max steer:  0.314 rad (18 deg)"
echo ""
echo "Friction model:"
echo "  static_friction:  0.15 m/s^2"
echo "  kinetic_friction: 0.05 m/s^2"
echo ""
echo "Tolerance progression:"
echo "  Phase 1: 5.4cm -> Phase 7: 2.7cm"
echo ""
echo "========================================="
echo ""

# Activate virtual environment
source venv/bin/activate

# Verify config files exist
if [ ! -f "config_env_newcar.yaml" ]; then
    echo "ERROR: config_env_newcar.yaml not found!"
    exit 1
fi

if [ ! -f "rl/curriculum_config_newcar.yaml" ]; then
    echo "ERROR: rl/curriculum_config_newcar.yaml not found!"
    exit 1
fi

# Run curriculum training
python -m rl.train_curriculum \
    --curriculum-config rl/curriculum_config_newcar.yaml \
    --train-until-success \
    --num-workers 4 \
    --num-cpus 8 \
    --num-gpus 1 \
    --checkpoint-dir checkpoints/newcar

echo ""
echo "========================================="
echo "Training complete!"
echo ""
echo "Checkpoints saved in: checkpoints/newcar/"
echo ""
echo "To monitor training:"
echo "  tensorboard --logdir checkpoints/newcar/"
echo ""
echo "To evaluate:"
echo "  python -m rl.eval_policy \\"
echo "    --checkpoint checkpoints/newcar/*/phase7_polish/best_checkpoint \\"
echo "    --num-episodes 100 --deterministic"
echo ""
echo "To visualize:"
echo "  python -m rl.visualize_checkpoint \\"
echo "    --checkpoint checkpoints/newcar/*/phase7_polish/best_checkpoint \\"
echo "    --deterministic --num-episodes 5"
echo ""
echo "Next step: Deploy with deploy/parking_scene_newcar.yaml"
echo "========================================="
