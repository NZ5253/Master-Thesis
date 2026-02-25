#!/bin/bash
# =========================================================
# ChronosCar Phase Resume/Start Script
# =========================================================
# Usage: ./resume_phase_chronos.sh <phase_number> [checkpoint_dir] [--fresh]
#
# Examples:
#   ./resume_phase_chronos.sh 6                          # Resume phase 6 from best checkpoint
#   ./resume_phase_chronos.sh 6 --fresh                  # Start phase 6 fresh (load phase 5 weights)
#   ./resume_phase_chronos.sh 6 curriculum_20260208_223402  # Resume from specific run
#   ./resume_phase_chronos.sh 6 curriculum_20260208_223402 --fresh  # Fresh phase 6 from specific phase 5
#
# Phase mapping (ChronosCar curriculum):
#   1 = phase1_foundation
#   2 = phase2_random_spawn
#   3 = phase3_random_bay_x
#   4 = phase4a_random_bay_y_small
#   5 = phase4_random_bay_full
#   6 = phase5_neighbor_jitter
#   7 = phase6_random_obstacles (precision depth)
#   8 = phase7_polish (ultra precision)
# =========================================================

set -e

cd /home/naeem/Documents/final

if [[ -z "${VIRTUAL_ENV}" ]]; then
    source venv/bin/activate
fi

# ---- Configuration ----
CURRICULUM_CONFIG="rl/curriculum_config_chronos.yaml"
CHECKPOINT_BASE="checkpoints/chronos"

# Phase name mapping (ChronosCar has 8 phases with phase4a)
PHASE_NAMES=(
    ""                              # 0 (unused)
    "phase1_foundation"             # 1
    "phase2_random_spawn"           # 2
    "phase3_random_bay_x"           # 3
    "phase4a_random_bay_y_small"    # 4
    "phase4_random_bay_full"        # 5
    "phase5_neighbor_jitter"        # 6
    "phase6_random_obstacles"       # 7
    "phase7_polish"                 # 8
)

# ---- Parse arguments ----
PHASE_NUM=${1:-""}
CHECKPOINT_ARG=${2:-""}
FRESH_MODE=${3:-""}

# Check if --fresh flag is present (can be in position 2 or 3)
if [ "$CHECKPOINT_ARG" = "--fresh" ] || [ "$FRESH_MODE" = "--fresh" ]; then
    FRESH_MODE="--fresh"
    if [ "$CHECKPOINT_ARG" = "--fresh" ]; then
        CHECKPOINT_ARG=""
    fi
fi

# Validate phase number
if [[ -z "$PHASE_NUM" ]] || ! [[ "$PHASE_NUM" =~ ^[1-8]$ ]]; then
    echo "ERROR: Phase must be 1-8, got: $PHASE_NUM"
    echo ""
    echo "Phase mapping:"
    for i in $(seq 1 8); do
        echo "  $i = ${PHASE_NAMES[$i]}"
    done
    echo ""
    echo "Usage: ./resume_phase_chronos.sh <phase_number> [checkpoint_dir] [--fresh]"
    exit 1
fi

PHASE_NAME="${PHASE_NAMES[$PHASE_NUM]}"

echo "========================================="
if [ "$FRESH_MODE" = "--fresh" ]; then
    echo "ChronosCar: Starting Phase $PHASE_NUM FRESH"
else
    echo "ChronosCar: Resuming Phase $PHASE_NUM"
fi
echo "  Phase: $PHASE_NAME"
echo "  Config: $CURRICULUM_CONFIG"
echo "========================================="
echo ""

# ---- Find checkpoint ----
CHECKPOINT_TO_USE=""

if [ "$FRESH_MODE" = "--fresh" ]; then
    # Fresh mode: load weights from PREVIOUS phase's best checkpoint
    if [ $PHASE_NUM -gt 1 ]; then
        PREV_PHASE_NUM=$((PHASE_NUM - 1))
        PREV_PHASE_NAME="${PHASE_NAMES[$PREV_PHASE_NUM]}"

        echo "Fresh start - loading weights from Phase $PREV_PHASE_NUM ($PREV_PHASE_NAME)"

        if [ -n "$CHECKPOINT_ARG" ]; then
            # User specified run directory
            if [ -d "$CHECKPOINT_BASE/$CHECKPOINT_ARG/$PREV_PHASE_NAME/best_checkpoint" ]; then
                CHECKPOINT_TO_USE="$CHECKPOINT_BASE/$CHECKPOINT_ARG/$PREV_PHASE_NAME/best_checkpoint"
            elif [ -d "$CHECKPOINT_ARG/$PREV_PHASE_NAME/best_checkpoint" ]; then
                CHECKPOINT_TO_USE="$CHECKPOINT_ARG/$PREV_PHASE_NAME/best_checkpoint"
            fi
        else
            # Auto-detect latest run with previous phase
            for dir in $(ls -dt "$CHECKPOINT_BASE"/curriculum_* 2>/dev/null); do
                if [ -d "$dir/$PREV_PHASE_NAME/best_checkpoint" ]; then
                    CHECKPOINT_TO_USE="$dir/$PREV_PHASE_NAME/best_checkpoint"
                    break
                fi
            done
        fi

        if [ -z "$CHECKPOINT_TO_USE" ]; then
            echo "WARNING: Could not find Phase $PREV_PHASE_NUM ($PREV_PHASE_NAME) checkpoint"
            echo "         Will start with completely fresh weights"
        else
            echo "  Found: $CHECKPOINT_TO_USE"
        fi
    else
        echo "Fresh start for Phase 1 - no previous phase to load from"
    fi
    echo ""
else
    # Resume mode: find current phase's best checkpoint
    echo "Looking for Phase $PHASE_NUM ($PHASE_NAME) checkpoint..."

    if [ -n "$CHECKPOINT_ARG" ]; then
        # User specified run directory
        if [ -d "$CHECKPOINT_BASE/$CHECKPOINT_ARG/$PHASE_NAME/best_checkpoint" ]; then
            CHECKPOINT_TO_USE="$CHECKPOINT_BASE/$CHECKPOINT_ARG/$PHASE_NAME/best_checkpoint"
        elif [ -d "$CHECKPOINT_ARG/$PHASE_NAME/best_checkpoint" ]; then
            CHECKPOINT_TO_USE="$CHECKPOINT_ARG/$PHASE_NAME/best_checkpoint"
        fi
    else
        # Auto-detect latest run with current phase
        for dir in $(ls -dt "$CHECKPOINT_BASE"/curriculum_* 2>/dev/null); do
            if [ -d "$dir/$PHASE_NAME/best_checkpoint" ]; then
                CHECKPOINT_TO_USE="$dir/$PHASE_NAME/best_checkpoint"
                break
            fi
        done
    fi

    if [ -z "$CHECKPOINT_TO_USE" ]; then
        echo "ERROR: No checkpoint found for $PHASE_NAME"
        echo ""
        echo "Available runs in $CHECKPOINT_BASE/:"
        for dir in $(ls -dt "$CHECKPOINT_BASE"/curriculum_* 2>/dev/null | head -5); do
            echo "  $(basename $dir):"
            ls -d "$dir"/phase*/ 2>/dev/null | while read p; do echo "    $(basename $p)"; done
        done
        exit 1
    fi

    echo "  Found: $CHECKPOINT_TO_USE"
    echo ""
fi

# ---- Run training ----
if [ "$FRESH_MODE" = "--fresh" ]; then
    echo "Starting training for $PHASE_NAME (fresh config, weights from previous phase)..."
    echo ""

    if [ -n "$CHECKPOINT_TO_USE" ]; then
        python -m rl.train_curriculum \
            --curriculum-config "$CURRICULUM_CONFIG" \
            --start-phase "$PHASE_NAME" \
            --resume-checkpoint "$CHECKPOINT_TO_USE" \
            --train-until-success \
            --num-workers 3 \
            --num-cpus 6 \
            --num-gpus 1 \
            --checkpoint-dir "$CHECKPOINT_BASE"
    else
        python -m rl.train_curriculum \
            --curriculum-config "$CURRICULUM_CONFIG" \
            --start-phase "$PHASE_NAME" \
            --train-until-success \
            --num-workers 3 \
            --num-cpus 6 \
            --num-gpus 1 \
            --checkpoint-dir "$CHECKPOINT_BASE"
    fi
else
    echo "Resuming training from checkpoint..."
    echo ""

    python -m rl.train_curriculum \
        --curriculum-config "$CURRICULUM_CONFIG" \
        --resume-from-phase "$PHASE_NAME" \
        --resume-checkpoint "$CHECKPOINT_TO_USE" \
        --train-until-success \
        --num-workers 3 \
        --num-cpus 6 \
        --num-gpus 1 \
        --checkpoint-dir "$CHECKPOINT_BASE"
fi

echo ""
echo "========================================="
echo "Phase $PHASE_NUM ($PHASE_NAME) training complete!"
echo "========================================="
