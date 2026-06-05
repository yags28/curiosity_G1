#!/usr/bin/env bash
# Waits for the running DRND job to finish, then launches RDD (headless).
# Run this in a separate terminal:
#   bash scripts/run_rdd_after_drnd.sh

set -euo pipefail

REPO="/home/kevin/projects/CDL"
DRND_CKPT_DIR="$REPO/checkpoints/distant_target__drnd__seed42"
ISAAC_PY="/home/kevin/isaacsim/python.sh"
LOG="$HOME/.local/share/cdl_chain.log"

mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

echo "=== chain script started $(date) ==="
echo "Watching $DRND_CKPT_DIR for final checkpoint (step >= 9_800_000)..."

# Poll every 2 minutes until a checkpoint >= 9.8M appears
while true; do
    latest=$(ls "$DRND_CKPT_DIR"/step_*.pt 2>/dev/null \
             | sed 's/.*step_//;s/\.pt//' \
             | sort -n | tail -1)

    if [[ -n "$latest" && "$latest" -ge 9800000 ]]; then
        echo "Final DRND checkpoint found: step_${latest}.pt at $(date)"
        # Give the process 30 s to write and exit cleanly
        sleep 30
        break
    fi

    echo "  DRND at step ${latest:-0} — still running ($(date '+%H:%M'))"
    sleep 120
done

echo ""
echo "=== Launching RDD (headless) at $(date) ==="
OMNI_KIT_ACCEPT_EULA=Y "$ISAAC_PY" "$REPO/launch.py" \
    --config "$REPO/configs/local.yaml" \
    --task  distant_target \
    --curiosity rdd

echo ""
echo "=== RDD complete at $(date) ==="
echo "Run this to compare DRND vs RDD:"
echo "  python3 $REPO/plot_runs.py"
