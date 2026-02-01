#!/bin/bash
# Universal Phase Resume/Start Script
# Usage: ./resume_phase.sh <phase_number> [checkpoint_dir] [--fresh]
#
# Examples:
#   ./resume_phase.sh 2                          # Resume phase 2 (best checkpoint auto-detected)
#   ./resume_phase.sh 2 curriculum_20260122_103711  # Resume phase 2 from specific checkpoint
#   ./resume_phase.sh 3 --fresh                  # Start phase 3 fresh (load phase 2 weights)
#   ./resume_phase.sh 1 --fresh                  # Start phase 1 completely fresh
#   ./resume_phase.sh 4 curriculum_20260122_103711 --fresh  # Start phase 4 fresh from specific phase 3

set -e

cd /home/naeem/Documents/final

if [[ -z "${VIRTUAL_ENV}" ]]; then
    source venv/bin/activate
fi

# Parse arguments
PHASE_NUM=${1:-2}
CHECKPOINT_ARG=${2:-""}
FRESH_MODE=${3:-""}

# Validate phase number
if ! [[ "$PHASE_NUM" =~ ^[1-7]$ ]]; then
    echo "ERROR: Phase must be 1-7, got: $PHASE_NUM"
    echo "  1=foundation, 2=random_spawn, 3=random_bay_x, 4=bay_y_small, 5=bay_full, 6=neighbor_jitter, 7=random_obstacles"
    exit 1
fi

# Convert phase number to phase name
PHASE_NAMES=("" "phase1_foundation" "phase2_random_spawn" "phase3_random_bay_x" "phase4a_random_bay_y_small" "phase4_random_bay_full" "phase5_neighbor_jitter" "phase6_random_obstacles")
PHASE_NAME="${PHASE_NAMES[$PHASE_NUM]}"

# Check if --fresh flag is present (can be in position 2 or 3)
if [ "$CHECKPOINT_ARG" = "--fresh" ] || [ "$FRESH_MODE" = "--fresh" ]; then
    FRESH_MODE="--fresh"
    # If --fresh was in position 2, shift CHECKPOINT_ARG
    if [ "$CHECKPOINT_ARG" = "--fresh" ]; then
        CHECKPOINT_ARG=""
    fi
fi

echo "========================================="
if [ "$FRESH_MODE" = "--fresh" ]; then
    echo "Starting Phase $PHASE_NUM: $PHASE_NAME (FRESH)"
else
    echo "Resuming Phase $PHASE_NUM: $PHASE_NAME"
fi
echo "========================================="
echo ""

# Determine checkpoint to load/resume from
CHECKPOINT_TO_USE=""

if [ "$FRESH_MODE" = "--fresh" ]; then
    # Fresh mode: optionally load weights from previous phase
    if [ $PHASE_NUM -gt 1 ]; then
        PREV_PHASE_NUM=$((PHASE_NUM - 1))
        PREV_PHASE_NAME="${PHASE_NAMES[$PREV_PHASE_NUM]}"
        
        echo "Fresh start - will load weights from Phase $PREV_PHASE_NUM ($PREV_PHASE_NAME)"
        
        # Find checkpoint dir for previous phase
        if [ -n "$CHECKPOINT_ARG" ]; then
            # User specified checkpoint dir
            if [ -d "$CHECKPOINT_ARG/phase${PREV_PHASE_NUM}_*" ]; then
                # Try with wildcard
                CHECKPOINT_TO_USE=$(ls -d "$CHECKPOINT_ARG/phase${PREV_PHASE_NUM}_"* 2>/dev/null | head -1)
            elif [ -d "/home/naeem/Documents/final/checkpoints/curriculum/$CHECKPOINT_ARG" ]; then
                CHECKPOINT_DIR="/home/naeem/Documents/final/checkpoints/curriculum/$CHECKPOINT_ARG"
                CHECKPOINT_TO_USE=$(ls -d "$CHECKPOINT_DIR/phase${PREV_PHASE_NUM}_"* 2>/dev/null | head -1)
            fi
        else
            # Auto-detect latest checkpoint with previous phase
            for dir in $(ls -dt /home/naeem/Documents/final/checkpoints/curriculum/curriculum_* 2>/dev/null); do
                if [ -d "$dir/phase${PREV_PHASE_NUM}"* ]; then
                    CHECKPOINT_TO_USE=$(ls -d "$dir/phase${PREV_PHASE_NUM}_"* 2>/dev/null | head -1)
                    break
                fi
            done
        fi
        
        if [ -z "$CHECKPOINT_TO_USE" ]; then
            echo "WARNING: Could not find phase $PREV_PHASE_NUM checkpoint to load weights from"
            echo "         Will start with completely fresh weights"
            CHECKPOINT_TO_USE=""
        else
            echo "✓ Found phase $PREV_PHASE_NUM checkpoint: $CHECKPOINT_TO_USE"
        fi
    else
        echo "Fresh start for Phase 1 - no previous phase to load from"
        CHECKPOINT_TO_USE=""
    fi
    echo ""
else
    # Resume mode: find best checkpoint for current phase
    echo "Looking for phase $PHASE_NUM checkpoint..."
    
    if [ -n "$CHECKPOINT_ARG" ]; then
        # User specified checkpoint dir
        if [ -d "$CHECKPOINT_ARG" ]; then
            # Full path
            CHECKPOINT_DIR="$CHECKPOINT_ARG"
        elif [ -d "/home/naeem/Documents/final/checkpoints/curriculum/$CHECKPOINT_ARG" ]; then
            # Just the name
            CHECKPOINT_DIR="/home/naeem/Documents/final/checkpoints/curriculum/$CHECKPOINT_ARG"
        else
            echo "ERROR: Checkpoint directory not found: $CHECKPOINT_ARG"
            exit 1
        fi
    else
        # Auto-detect latest checkpoint with current phase
        CHECKPOINT_DIR=""
        for dir in $(ls -dt /home/naeem/Documents/final/checkpoints/curriculum/curriculum_* 2>/dev/null); do
            if [ -d "$dir/$PHASE_NAME/best_checkpoint" ]; then
                CHECKPOINT_DIR="$dir"
                break
            fi
        done
    fi
    
    if [ -z "$CHECKPOINT_DIR" ] || [ ! -d "$CHECKPOINT_DIR/$PHASE_NAME/best_checkpoint" ]; then
        echo "ERROR: No checkpoint found for $PHASE_NAME"
        echo ""
        echo "Available checkpoint directories:"
        ls -d /home/naeem/Documents/final/checkpoints/curriculum/curriculum_* 2>/dev/null | head -10
        exit 1
    fi
    
    CHECKPOINT_TO_USE="$CHECKPOINT_DIR/$PHASE_NAME/best_checkpoint"
    echo "✓ Found checkpoint: $CHECKPOINT_TO_USE"
    echo ""
fi

# Build the command
if [ "$FRESH_MODE" = "--fresh" ]; then
    # Fresh mode - start the phase
    if [ -n "$CHECKPOINT_TO_USE" ]; then
        # Start fresh but load weights from previous phase
        echo "Starting training for $PHASE_NAME (fresh config, weights from previous phase)..."
        echo ""
    else
        # Completely fresh start
        echo "Starting training for $PHASE_NAME (completely fresh)..."
        echo ""
    fi
    python -m rl.train_curriculum \
        --start-phase "$PHASE_NAME" \
        --train-until-success \
        --num-workers 3 \
        --num-cpus 6 \
        --num-gpus 1
else
    # Resume mode - load checkpoint
    echo "Resuming training from checkpoint..."
    echo ""
    python -m rl.train_curriculum \
        --resume-from-phase "$PHASE_NAME" \
        --resume-checkpoint "$CHECKPOINT_TO_USE" \
        --num-workers 3 \
        --num-cpus 6 \
        --num-gpus 1 \
        --train-until-success
fi

echo ""
echo "========================================="
echo "Phase $PHASE_NUM training complete!"
echo "========================================="
