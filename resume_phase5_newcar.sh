#!/bin/bash
# =========================================================
# New Car Phase 5 Resume Script
# =========================================================
# Phase 5 training failed: introduced jitter + tighter tolerances
# simultaneously. Policy collapsed after 234M steps (best reward=14.5).
#
# FIX: Insert phase4b (jitter only, same 3.2cm tolerance as phase 4).
# Resume from Phase 4 final_checkpoint (reward ~444, excellent policy).
# Once policy handles jitter -> Phase 5 safely tightens to 2.7cm.
#
# Usage:
#   ./resume_phase5_newcar.sh
#
# To resume phase5 after phase4b is done:
#   ./resume_phase5_newcar.sh phase5
# =========================================================

set -e

cd /home/naeem/Documents/final

if [[ -z "${VIRTUAL_ENV}" ]]; then
    source venv/bin/activate
fi

CURRICULUM_CONFIG="rl/curriculum_config_newcar_resume.yaml"
CHECKPOINT_BASE="checkpoints/newcar"
PHASE4_RUN="curriculum_20260220_211152"
PHASE4_CHECKPOINT="$CHECKPOINT_BASE/$PHASE4_RUN/phase4_random_bay_full/final_checkpoint"

START_PHASE="${1:-phase4b_jitter_only}"

echo "========================================="
echo "New Car: Resume from Phase 4 -> Phase 4b"
echo "========================================="
echo ""
echo "Config:      $CURRICULUM_CONFIG"
echo "Start phase: $START_PHASE"
echo ""

if [ "$START_PHASE" = "phase4b_jitter_only" ]; then
    echo "Loading weights from Phase 4 final checkpoint (reward ~444):"
    echo "  $PHASE4_CHECKPOINT"

    if [ ! -d "$PHASE4_CHECKPOINT" ]; then
        echo ""
        echo "ERROR: Phase 4 final checkpoint not found!"
        echo "Expected: $PHASE4_CHECKPOINT"
        echo ""
        echo "Available phase4 checkpoints:"
        ls -d $CHECKPOINT_BASE/*/phase4_random_bay_full/ 2>/dev/null || echo "  (none found)"
        exit 1
    fi
    echo ""

    echo "Starting phase4b training..."
    python -m rl.train_curriculum \
        --curriculum-config "$CURRICULUM_CONFIG" \
        --start-phase "$START_PHASE" \
        --resume-checkpoint "$PHASE4_CHECKPOINT" \
        --train-until-success \
        --num-workers 4 \
        --num-cpus 8 \
        --num-gpus 1 \
        --checkpoint-dir "$CHECKPOINT_BASE"

elif [ "$START_PHASE" = "phase5" ] || [ "$START_PHASE" = "phase5_neighbor_jitter" ]; then
    # Find phase4b best checkpoint to start phase5
    PHASE4B_CHECKPOINT=""
    for dir in $(ls -dt "$CHECKPOINT_BASE"/curriculum_* 2>/dev/null); do
        if [ -d "$dir/phase4b_jitter_only/best_checkpoint" ]; then
            PHASE4B_CHECKPOINT="$dir/phase4b_jitter_only/best_checkpoint"
            break
        fi
    done

    if [ -z "$PHASE4B_CHECKPOINT" ]; then
        echo "ERROR: phase4b best checkpoint not found. Run phase4b first."
        exit 1
    fi

    echo "Loading weights from Phase 4b best checkpoint:"
    echo "  $PHASE4B_CHECKPOINT"
    echo ""

    echo "Starting phase5 training..."
    python -m rl.train_curriculum \
        --curriculum-config "$CURRICULUM_CONFIG" \
        --start-phase "phase5_neighbor_jitter" \
        --resume-checkpoint "$PHASE4B_CHECKPOINT" \
        --train-until-success \
        --num-workers 4 \
        --num-cpus 8 \
        --num-gpus 1 \
        --checkpoint-dir "$CHECKPOINT_BASE"

else
    # Generic: start from any phase, auto-detect checkpoint
    echo "Starting $START_PHASE..."
    python -m rl.train_curriculum \
        --curriculum-config "$CURRICULUM_CONFIG" \
        --start-phase "$START_PHASE" \
        --train-until-success \
        --num-workers 4 \
        --num-cpus 8 \
        --num-gpus 1 \
        --checkpoint-dir "$CHECKPOINT_BASE"
fi

echo ""
echo "========================================="
echo "Training complete!"
echo ""
echo "Checkpoints saved in: $CHECKPOINT_BASE/"
echo ""
echo "Next: run tensorboard --logdir $CHECKPOINT_BASE/"
echo "========================================="
