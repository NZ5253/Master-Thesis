#!/bin/bash
# Quick training script for parallel parking RL agent
# Usage: ./quick_train.sh [mode]
# Modes: curriculum (recommended), test, normal, production

set -e  # Exit on error

MODE=${1:-curriculum}

# Activate virtual environment
source venv/bin/activate

echo "========================================="
echo "Parallel Parking RL Training"
echo "Mode: $MODE"
echo "========================================="
echo ""

case $MODE in
    curriculum)
        echo "Running CURRICULUM TRAINING (6 phases)"
        echo "Training until each phase reaches success threshold"
        echo "This is the RECOMMENDED approach for best results!"
        echo ""
        echo "Phases:"
        echo "  1. Foundation (70% success)"
        echo "  2. Random Spawn (65% success)"
        echo "  3. Random Bay X (60% success)"
        echo "  4. Full Bay Random (60% success)"
        echo "  5. Neighbor Jitter (55% success)"
        echo "  6. Random Obstacles (50% success)"
        echo ""
        python -m rl.train_curriculum \
            --train-until-success \
            --num-workers 4 \
            --num-cpus 8 \
            --num-gpus 1
        ;;

    test)
        echo "Running QUICK TEST (50k steps, 2 workers)"
        echo "Use this to verify everything works"
        echo ""
        python -m rl.train_ppo \
            --scenario parallel \
            --total-timesteps 50000 \
            --num-workers 2 \
            --num-cpus 4 \
            --eval-interval 5 \
            --save-interval 10
        ;;

    normal)
        echo "Running NORMAL TRAINING (500k steps, 4 workers)"
        echo "Expected time: 2-4 hours on 8 CPU cores"
        echo ""
        python -m rl.train_ppo \
            --scenario parallel \
            --total-timesteps 500000 \
            --num-workers 4 \
            --num-cpus 8 \
            --num-gpus 0 \
            --lr 3e-4 \
            --train-batch-size 4000 \
            --eval-interval 10 \
            --save-interval 50
        ;;

    production)
        echo "Running PRODUCTION TRAINING (1M steps, 8 workers)"
        echo "Expected time: 1-2 hours with GPU, 3-4 hours CPU-only"
        echo ""
        python -m rl.train_ppo \
            --scenario parallel \
            --total-timesteps 1000000 \
            --num-workers 8 \
            --num-cpus 16 \
            --num-gpus 1 \
            --lr 5e-4 \
            --train-batch-size 8000 \
            --sgd-minibatch-size 256 \
            --entropy-coeff 0.01 \
            --eval-interval 20 \
            --save-interval 100
        ;;

    *)
        echo "Unknown mode: $MODE"
        echo "Usage: ./quick_train.sh [curriculum|test|normal|production]"
        echo ""
        echo "Recommended: ./quick_train.sh curriculum"
        exit 1
        ;;
esac

echo ""
echo "========================================="
echo "Training complete!"
echo ""

if [ "$MODE" = "curriculum" ]; then
    echo "Check results in: checkpoints/curriculum/"
    echo ""
    echo "To monitor training:"
    echo "  tensorboard --logdir checkpoints/curriculum/"
    echo ""
    echo "To evaluate final model:"
    echo "  python -m rl.eval_policy \\"
    echo "    --checkpoint checkpoints/curriculum/curriculum_*/phase6_random_obstacles/best_checkpoint \\"
    echo "    --num-episodes 100 --deterministic --save-trajectories"
else
    echo "Check results in: checkpoints/ppo/"
    echo ""
    echo "To monitor training:"
    echo "  tensorboard --logdir checkpoints/ppo/"
    echo ""
    echo "To evaluate:"
    echo "  python -m rl.eval_policy --checkpoint checkpoints/ppo/parallel_*/best_checkpoint --num-episodes 100 --deterministic"
fi

echo "========================================="
