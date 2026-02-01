#!/bin/bash
# Quick evaluation without visualization (faster)

set -e

# Default values
CHECKPOINT=""
NUM_EPISODES=50
DETERMINISTIC="--deterministic"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --num-episodes)
            NUM_EPISODES="$2"
            shift 2
            ;;
        --explore)
            DETERMINISTIC=""
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --checkpoint <path> [--num-episodes N] [--explore]"
            exit 1
            ;;
    esac
done

if [ -z "$CHECKPOINT" ]; then
    echo "Error: --checkpoint is required"
    echo "Usage: $0 --checkpoint <path> [--num-episodes N] [--explore]"
    exit 1
fi

echo "========================================="
echo "Quick Evaluation (No Visualization)"
echo "========================================="
echo "Checkpoint: $CHECKPOINT"
echo "Episodes: $NUM_EPISODES"
echo "Deterministic: $([ -n "$DETERMINISTIC" ] && echo "Yes" || echo "No (exploring)")"
echo "========================================="
echo ""

# Run evaluation
python -m rl.test_checkpoint \
    --checkpoint "$CHECKPOINT" \
    --num-episodes "$NUM_EPISODES" \
    $DETERMINISTIC

echo ""
echo "========================================="
echo "Evaluation complete!"
echo "========================================="
