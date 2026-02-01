#!/bin/bash

# List Available Checkpoints for Resume
# This script helps find checkpoints that can be used to resume training

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo "======================================================================"
echo "Available Training Checkpoints"
echo "======================================================================"
echo ""

CHECKPOINT_BASE="checkpoints/curriculum"

if [ ! -d "$CHECKPOINT_BASE" ]; then
    echo "No checkpoints found - directory does not exist: $CHECKPOINT_BASE"
    exit 0
fi

# Find all curriculum training runs
TRAINING_RUNS=$(find "$CHECKPOINT_BASE" -maxdepth 1 -type d -name "curriculum_*" | sort -r)

if [ -z "$TRAINING_RUNS" ]; then
    echo "No training runs found in $CHECKPOINT_BASE"
    exit 0
fi

for RUN_DIR in $TRAINING_RUNS; do
    RUN_NAME=$(basename "$RUN_DIR")

    echo -e "${GREEN}Training Run: $RUN_NAME${NC}"
    echo "Path: $RUN_DIR"
    echo ""

    # List phases in this run
    PHASES=$(find "$RUN_DIR" -maxdepth 1 -type d -name "phase*" | sort)

    for PHASE_DIR in $PHASES; do
        PHASE_NAME=$(basename "$PHASE_DIR")

        # Find checkpoints in this phase
        BEST_CHECKPOINT="$PHASE_DIR/best_checkpoint"
        LATEST_CHECKPOINT=$(find "$PHASE_DIR" -maxdepth 1 -type d -name "checkpoint_*" | sort -V | tail -n 1)
        CHECKPOINT_COUNT=$(find "$PHASE_DIR" -maxdepth 1 -type d -name "checkpoint_*" | wc -l)

        echo -e "  ${BLUE}Phase: $PHASE_NAME${NC}"

        # Check if best checkpoint exists
        if [ -d "$BEST_CHECKPOINT" ]; then
            echo -e "    ${CYAN}✓ Best Checkpoint:${NC} $BEST_CHECKPOINT"
            echo "      Resume command: ./resume_training.sh $PHASE_NAME $BEST_CHECKPOINT"
        fi

        # Check if regular checkpoints exist
        if [ -n "$LATEST_CHECKPOINT" ]; then
            echo -e "    ${CYAN}✓ Latest Checkpoint:${NC} $LATEST_CHECKPOINT"
            echo "      Resume command: ./resume_training.sh $PHASE_NAME $LATEST_CHECKPOINT"
            echo "      Total checkpoints: $CHECKPOINT_COUNT"
        fi

        if [ ! -d "$BEST_CHECKPOINT" ] && [ -z "$LATEST_CHECKPOINT" ]; then
            echo "    ✗ No checkpoints found"
        fi

        echo ""
    done

    echo "----------------------------------------------------------------------"
    echo ""
done

echo "======================================================================"
echo "Resume Training Examples:"
echo "======================================================================"
echo ""
echo "To resume from a specific checkpoint, use:"
echo "  ./resume_training.sh <phase_name> <checkpoint_path>"
echo ""
echo "To continue training the latest phase:"
echo "  1. Find the latest checkpoint from the output above"
echo "  2. Run: ./resume_training.sh phase_name checkpoint_path"
echo ""
echo "To restart a phase from its best checkpoint:"
echo "  ./resume_training.sh phase_name path/to/best_checkpoint"
echo ""
