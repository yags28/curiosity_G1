#!/usr/bin/env bash
# Waits for composite (Task 5) to finish, then runs elevated_button (Task 2).

set -euo pipefail

REPO="/home/kevin/projects/CDL"
ISAAC_PY="/home/kevin/isaacsim/python.sh"
COMPOSITE_LOG="$REPO/logs/composite__drnd__seed42/metrics.csv"

export OMNI_KIT_ACCEPT_EULA=Y

echo "=== Waiting for composite to finish ($(date)) ==="

while true; do
    if [[ -f "$COMPOSITE_LOG" ]]; then
        last_step=$(tail -1 "$COMPOSITE_LOG" | cut -d',' -f1)
        last_step=${last_step%.*}   # strip decimal
        if [[ -n "$last_step" && "$last_step" -ge 9900000 ]]; then
            echo "Composite done at step $last_step ($(date))"
            break
        fi
        echo "  Composite at step ${last_step:-0} — still running ($(date '+%H:%M'))"
    else
        echo "  Composite log not found yet ($(date '+%H:%M'))"
    fi
    sleep 120
done

sleep 30  # let the process exit cleanly

echo ""
echo "=== Starting DRND (shaped, 10 kg box): elevated_button ($(date)) ==="
"$ISAAC_PY" "$REPO/launch.py" \
    --config "$REPO/configs/local.yaml" \
    --task elevated_button \
    --curiosity drnd

echo ""
echo "=== elevated_button complete ($(date)) ==="
