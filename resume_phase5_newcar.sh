#!/bin/bash
# =========================================================
# New Car: Resume Phase 5 from Phase 4 checkpoint
# =========================================================
# Resumes training from Phase 4 final_checkpoint (reward ~444)
# into Phase 5 (tighter tolerances 3.2cm->2.7cm, no jitter).
#
# Usage:
#   ./resume_phase5_newcar.sh               # Phase 5 -> 6 -> 7
#   ./resume_phase5_newcar.sh phase6        # Start from Phase 6
#   ./resume_phase5_newcar.sh phase7_polish # Start from Phase 7
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

START_PHASE="${1:-phase5_tight_tol}"

echo "========================================="
echo "New Car: Resume from Phase 4 -> $START_PHASE"
echo "========================================="
echo ""
echo "Config:      $CURRICULUM_CONFIG"
echo "Start phase: $START_PHASE"
echo ""

if [ "$START_PHASE" = "phase5_tight_tol" ]; then
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

    python -m rl.train_curriculum \
        --curriculum-config "$CURRICULUM_CONFIG" \
        --start-phase "$START_PHASE" \
        --resume-checkpoint "$PHASE4_CHECKPOINT" \
        --train-until-success \
        --num-workers 4 \
        --num-cpus 8 \
        --num-gpus 1 \
        --checkpoint-dir "$CHECKPOINT_BASE"

else
    # Phase 6, 7, or any later phase: auto-detect best checkpoint from previous phase
    echo "Looking for best checkpoint for phase: $START_PHASE"
    RESUME_CHECKPOINT=""
    for dir in $(ls -dt "$CHECKPOINT_BASE"/curriculum_* 2>/dev/null); do
        if [ -d "$dir/$START_PHASE/best_checkpoint" ]; then
            RESUME_CHECKPOINT="$dir/$START_PHASE/best_checkpoint"
            break
        fi
    done

    if [ -z "$RESUME_CHECKPOINT" ]; then
        # Try finding the previous phase checkpoint to warm-start
        echo "No $START_PHASE checkpoint found. Starting $START_PHASE without resume."
        python -m rl.train_curriculum \
            --curriculum-config "$CURRICULUM_CONFIG" \
            --start-phase "$START_PHASE" \
            --train-until-success \
            --num-workers 4 \
            --num-cpus 8 \
            --num-gpus 1 \
            --checkpoint-dir "$CHECKPOINT_BASE"
    else
        echo "Found: $RESUME_CHECKPOINT"
        echo ""
        python -m rl.train_curriculum \
            --curriculum-config "$CURRICULUM_CONFIG" \
            --resume-from-phase "$START_PHASE" \
            --resume-checkpoint "$RESUME_CHECKPOINT" \
            --train-until-success \
            --num-workers 4 \
            --num-cpus 8 \
            --num-gpus 1 \
            --checkpoint-dir "$CHECKPOINT_BASE"
    fi
fi

echo ""
echo "========================================="
echo "Training complete!"
echo ""
echo "Checkpoints saved in: $CHECKPOINT_BASE/"
echo ""
echo "Next: run tensorboard --logdir $CHECKPOINT_BASE/"
echo "========================================="
