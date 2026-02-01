#!/bin/bash
# Quick script to view the best performing checkpoint (Phase 6)

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║         Viewing Phase 6: Best Performance (2.6cm accuracy)     ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "This phase includes:"
echo "  • Full randomization (spawn + bay position)"
echo "  • Neighbor jitter (±5cm)"
echo "  • Random obstacles in environment"
echo "  • 97% success rate"
echo "  • 2.6cm total positioning error"
echo ""
echo "Starting visualization..."
echo ""

./eval_with_viz.sh \
    --checkpoint checkpoints/curriculum/curriculum_20260121_152111/phase6_random_obstacles/best_checkpoint \
    --num-episodes 5 \
    --speed 1.5 \
    --phase-name phase6_random_obstacles \
    --deterministic

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  Watch how the agent (blue rectangle) navigates obstacles     ║"
echo "║  and parks precisely in the bay with sub-3cm accuracy!        ║"
echo "╚════════════════════════════════════════════════════════════════╝"
