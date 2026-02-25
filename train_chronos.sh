#!/bin/bash
# =========================================================
# ChronosCar Training Script
# =========================================================
# Full curriculum training with REAL ChronosCar dimensions
#
# Dimensions:
#   - Wheelbase: 0.090m (vs 0.25m original)
#   - Length: 0.13m (vs 0.36m original)
#   - Max steer: 0.35 rad (vs 0.523 rad original)
#
# Expected training time: Similar to original (~750M steps)
# =========================================================

set -e

echo "========================================="
echo "ChronosCar Parallel Parking Training"
echo "========================================="
echo ""
echo "Configuration:"
echo "  Environment: config_env_chronos.yaml"
echo "  Curriculum:  rl/curriculum_config_chronos.yaml"
echo ""
echo "Vehicle dimensions (REAL ChronosCar):"
echo "  Wheelbase:  0.090m"
echo "  Length:     0.13m"
echo "  Width:      0.065m"
echo "  Max steer:  0.35 rad (20°)"
echo ""
echo "Tolerance progression:"
echo "  Phase 1: 4.3cm -> Phase 7: 1.4cm"
echo ""
echo "========================================="
echo ""

# Activate virtual environment
source venv/bin/activate

# Verify config files exist
if [ ! -f "config_env_chronos.yaml" ]; then
    echo "ERROR: config_env_chronos.yaml not found!"
    exit 1
fi

if [ ! -f "rl/curriculum_config_chronos.yaml" ]; then
    echo "ERROR: rl/curriculum_config_chronos.yaml not found!"
    exit 1
fi

# Run curriculum training
python -m rl.train_curriculum \
    --curriculum-config rl/curriculum_config_chronos.yaml \
    --train-until-success \
    --num-workers 4 \
    --num-cpus 8 \
    --num-gpus 1 \
    --checkpoint-dir checkpoints/chronos

echo ""
echo "========================================="
echo "Training complete!"
echo ""
echo "Checkpoints saved in: checkpoints/chronos/"
echo ""
echo "To monitor training:"
echo "  tensorboard --logdir checkpoints/chronos/"
echo ""
echo "To evaluate:"
echo "  python -m rl.eval_policy \\"
echo "    --checkpoint checkpoints/chronos/*/phase7_polish/best_checkpoint \\"
echo "    --num-episodes 100 --deterministic"
echo ""
echo "Next step: Deploy using HARDWARE_DEPLOYMENT_PLAN.md"
echo "========================================="
