#!/bin/bash
# Visualize successful parking across all curriculum phases

set -e

# Activate virtual environment if it exists
if [ -d "venv/bin" ]; then
    source venv/bin/activate
fi

CHECKPOINT_DIR="checkpoints/curriculum/curriculum_20260121_152111"
NUM_EPISODES=${1:-3}  # Default 3 episodes per phase
SPEED=${2:-1.5}  # Default 1.5x speed

echo "========================================"
echo "Visualizing All Curriculum Phases"
echo "========================================"
echo "Episodes per phase: $NUM_EPISODES"
echo "Playback speed: ${SPEED}x"
echo "========================================"
echo ""

# Array of phases to visualize
phases=(
    "phase1_foundation:Phase 1 - Foundation (Fixed Position)"
    "phase2_random_spawn:Phase 2 - Random Spawn Position"
    "phase3_random_bay_x:Phase 3 - Random Bay X Position"
    "phase4_random_bay_full:Phase 4 - Full Bay Randomization"
    "phase5_neighbor_jitter:Phase 5 - Neighbor Position Jitter"
    "phase6_random_obstacles:Phase 6 - Random Obstacles (Maximum Difficulty)"
)

for phase_info in "${phases[@]}"; do
    IFS=':' read -r phase_name phase_desc <<< "$phase_info"

    echo ""
    echo "========================================"
    echo "$phase_desc"
    echo "========================================"
    echo ""

    checkpoint_path="$CHECKPOINT_DIR/$phase_name/best_checkpoint"

    if [ -d "$checkpoint_path" ]; then
        python3 -m rl.visualize_checkpoint \
            --checkpoint "$checkpoint_path" \
            --num-episodes "$NUM_EPISODES" \
            --speed "$SPEED" \
            --phase-name "$phase_name" \
            --deterministic

        echo ""
        echo "Press Enter to continue to next phase (or Ctrl+C to stop)..."
        read -r
    else
        echo "⚠️  Checkpoint not found: $checkpoint_path"
    fi
done

echo ""
echo "========================================"
echo "All phases visualized!"
echo "========================================"
