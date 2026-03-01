#!/usr/bin/env bash
# cleanup_checkpoints.sh
# Shows exactly what will be deleted (dry-run by default), then optionally deletes.
#
# Usage:
#   ./cleanup_checkpoints.sh          # dry-run: show what would be deleted
#   ./cleanup_checkpoints.sh --delete  # actually delete

set -e

# ============================================================
# KEEP (canonical / deployed runs)
# ============================================================
# checkpoints/curriculum/curriculum_20260203_090023  — Original RL, best full run (83% final)
# checkpoints/chronos/curriculum_20260208_223402     — ChronosCar phases 1-5
# checkpoints/chronos/curriculum_20260209_230526     — ChronosCar phase 6 (87%)
# checkpoints/chronos/curriculum_20260210_113810     — ChronosCar phase 7 (82%), DEPLOYED
# checkpoints/newcar/curriculum_20260220_211152      — New Car phases 1-4
# checkpoints/newcar/curriculum_20260225_134317      — New Car phase 5 (83%)
# checkpoints/newcar/curriculum_20260227_144843      — New Car phases 6-7 (84%), fixed curriculum

# ============================================================
# DELETE (superseded / incomplete / broken runs)
# ============================================================
DELETE=(
    # --- Original RL --- superseded by curriculum_20260203_090023 ---
    "checkpoints/curriculum/curriculum_20260122_103711"   # Phase 1 only, early test
    "checkpoints/curriculum/curriculum_20260122_144234"   # No training_log.yaml, aborted
    "checkpoints/curriculum/curriculum_20260122_180153"   # Old reward scale (285K!), wrong config
    "checkpoints/curriculum/curriculum_20260129_205818"   # Incomplete: phases 1-6, not final
    "checkpoints/curriculum/curriculum_20260204_221632"   # Incomplete: stalled at phase 4a
    "checkpoints/curriculum/curriculum_20260207_144947"   # Phase 1 only, aborted
    "checkpoints/curriculum/curriculum_20260207_153829"   # Incomplete: stalled at phase 5
    "checkpoints/curriculum/curriculum_20260208_002342"   # No training_log.yaml, aborted

    # --- New Car --- broken phase 6 run (0% success, 3-simultaneous-changes bug) ---
    "checkpoints/newcar/curriculum_20260227_120642"       # Broken phase6, no training_log.yaml
)

echo "============================================================"
echo "  CHECKPOINT CLEANUP PLAN"
echo "============================================================"
echo ""
echo "KEEP (canonical runs):"
echo "  checkpoints/curriculum/curriculum_20260203_090023  [Original RL, 83% final, 7.6M steps]"
echo "  checkpoints/chronos/curriculum_20260208_223402     [ChronosCar phases 1-5]"
echo "  checkpoints/chronos/curriculum_20260209_230526     [ChronosCar phase 6, 87%]"
echo "  checkpoints/chronos/curriculum_20260210_113810     [ChronosCar phase 7, 82%, DEPLOYED]"
echo "  checkpoints/newcar/curriculum_20260220_211152      [New Car phases 1-4]"
echo "  checkpoints/newcar/curriculum_20260225_134317      [New Car phase 5, 83%]"
echo "  checkpoints/newcar/curriculum_20260227_144843      [New Car phases 6-7, 84%, fixed]"
echo ""
echo "DELETE (superseded / incomplete / broken):"
echo ""

TOTAL=0
for dir in "${DELETE[@]}"; do
    if [ -d "$dir" ]; then
        SIZE=$(du -sh "$dir" 2>/dev/null | cut -f1)
        BYTES=$(du -sb "$dir" 2>/dev/null | cut -f1)
        TOTAL=$((TOTAL + BYTES))
        REASON=""
        case "$dir" in
            *curriculum_20260122_103711) REASON="Phase 1 only, early test" ;;
            *curriculum_20260122_144234) REASON="No training_log, aborted" ;;
            *curriculum_20260122_180153) REASON="Wrong reward scale (285K), old config" ;;
            *curriculum_20260129_205818) REASON="Incomplete run (phases 1-6, not final)" ;;
            *curriculum_20260204_221632) REASON="Stalled at phase 4a, superseded" ;;
            *curriculum_20260207_144947) REASON="Phase 1 only, aborted" ;;
            *curriculum_20260207_153829) REASON="Stalled at phase 5, superseded" ;;
            *curriculum_20260208_002342) REASON="No training_log, aborted" ;;
            *curriculum_20260227_120642) REASON="Broken phase 6 (0% success, 3-changes bug)" ;;
        esac
        echo "  [$SIZE]  $dir"
        echo "           Reason: $REASON"
    else
        echo "  [MISSING] $dir  (already deleted or wrong path)"
    fi
done

# Format total in GB
TOTAL_GB=$(echo "scale=2; $TOTAL / 1073741824" | bc)
echo ""
echo "Total to free: ~${TOTAL_GB} GB"
echo ""

if [ "$1" == "--delete" ]; then
    echo "============================================================"
    echo "  DELETING..."
    echo "============================================================"
    for dir in "${DELETE[@]}"; do
        if [ -d "$dir" ]; then
            echo "  rm -rf $dir"
            rm -rf "$dir"
            echo "  DONE"
        fi
    done
    echo ""
    echo "Cleanup complete."
    echo ""
    echo "Remaining checkpoints:"
    echo "  checkpoints/curriculum/:"
    ls checkpoints/curriculum/ 2>/dev/null
    echo "  checkpoints/chronos/:"
    ls checkpoints/chronos/ 2>/dev/null
    echo "  checkpoints/newcar/:"
    ls checkpoints/newcar/ 2>/dev/null
else
    echo "============================================================"
    echo "  DRY RUN — nothing deleted."
    echo "  Run with --delete to actually remove these directories:"
    echo "  ./cleanup_checkpoints.sh --delete"
    echo "============================================================"
fi
