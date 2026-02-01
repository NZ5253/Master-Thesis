#!/bin/bash
# Evaluate and visualize trained checkpoint

set -e

# Activate virtual environment if it exists
if [ -d "venv/bin" ]; then
    source venv/bin/activate
fi

# Default values
CHECKPOINT=""
NUM_EPISODES=10
SPEED=1.0
DETERMINISTIC="--deterministic"
# PHASE_NAME is no longer needed by the python script but kept here to avoid breaking CLI usage
PHASE_NAME="phase1_foundation"

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
        --speed)
            SPEED="$2"
            shift 2
            ;;
        --phase-name)
            PHASE_NAME="$2"
            shift 2
            ;;
        --deterministic)
            DETERMINISTIC="--deterministic"
            shift
            ;;
        --explore)
            DETERMINISTIC=""
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --checkpoint <path> [--num-episodes N] [--speed X] [--phase-name NAME] [--deterministic|--explore]"
            exit 1
            ;;
    esac
done

if [ -z "$CHECKPOINT" ]; then
    echo "Error: --checkpoint is required"
    echo "Usage: $0 --checkpoint <path> [--num-episodes N] [--speed X] [--phase-name NAME] [--deterministic|--explore]"
    exit 1
fi

echo "========================================="
echo "Evaluation with Visualization"
echo "========================================="
echo "Checkpoint: $CHECKPOINT"
echo "Episodes: $NUM_EPISODES"
echo "Speed: ${SPEED}x"
echo "Deterministic: $([ -n "$DETERMINISTIC" ] && echo "Yes" || echo "No (exploring)")"
echo "========================================="
echo ""

# Run visualization
# Removed --phase-name (config loaded from checkpoint)
# Added --force-cpu to ensure it works on non-GPU machines
python3 -m rl.visualize_checkpoint --force-cpu \
    --checkpoint "$CHECKPOINT" \
    --num-episodes "$NUM_EPISODES" \
    --speed "$SPEED" \
    $DETERMINISTIC

echo ""
echo "========================================="
echo "Visualization complete!"
echo "========================================="