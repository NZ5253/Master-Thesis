#!/bin/bash
# Fine-tune trained policy for ChronosCar real hardware
#
# This script adapts the Phase 5 policy to real ChronosCar dimensions:
#   - Wheelbase: 0.25m (training) -> 0.090m (real)
#   - Length: 0.36m (training) -> 0.13m (real)
#   - Max steer: 0.523 rad (training) -> 0.35 rad (real)
#
# Expected training time: 1-2 hours on GPU
#
# Usage:
#   ./finetune_chronos.sh                    # Use default Phase 5 checkpoint
#   ./finetune_chronos.sh <checkpoint_path>  # Use specific checkpoint

set -e

# Default to the best Phase 5 checkpoint
DEFAULT_CHECKPOINT="checkpoints/curriculum/curriculum_20260207_153829/phase5_neighbor_jitter/best_checkpoint"
CHECKPOINT=${1:-$DEFAULT_CHECKPOINT}

# Verify checkpoint exists
if [ ! -d "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    echo ""
    echo "Available checkpoints:"
    find checkpoints -name "best_checkpoint" -type d 2>/dev/null | head -10
    exit 1
fi

echo "========================================="
echo "ChronosCar Fine-Tuning"
echo "========================================="
echo ""
echo "Source checkpoint: $CHECKPOINT"
echo "Config: config_env_chronos.yaml"
echo "Curriculum: rl/curriculum_config_finetune_chronos.yaml"
echo ""
echo "Dimension changes:"
echo "  Wheelbase: 0.25m -> 0.090m (2.78x smaller)"
echo "  Length:    0.36m -> 0.13m"
echo "  Max steer: 0.523 rad -> 0.35 rad"
echo ""
echo "Expected training: ~10M timesteps (1-2 hours on GPU)"
echo "========================================="
echo ""

# Activate virtual environment
source venv/bin/activate

# Run fine-tuning
python -m rl.train_curriculum \
    --curriculum-config rl/curriculum_config_finetune_chronos.yaml \
    --resume-checkpoint "$CHECKPOINT" \
    --train-until-success \
    --num-workers 4 \
    --num-cpus 8 \
    --num-gpus 1 \
    --checkpoint-dir checkpoints/finetune_chronos

echo ""
echo "========================================="
echo "Fine-tuning complete!"
echo ""
echo "Checkpoints saved in: checkpoints/finetune_chronos/"
echo ""
echo "To evaluate:"
echo "  python -m rl.eval_policy \\"
echo "    --checkpoint checkpoints/finetune_chronos/*/finetune_full_random/best_checkpoint \\"
echo "    --num-episodes 100 --deterministic"
echo ""
echo "Next step: Deploy to hardware using HARDWARE_DEPLOYMENT_PLAN.md"
echo "========================================="
