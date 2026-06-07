#!/usr/bin/env bash
# Run DRND on Tasks 2-5 sequentially.
# Task 1 (distant_target) already complete.

set -euo pipefail

REPO="/home/kevin/projects/CDL"
ISAAC_PY="/home/kevin/isaacsim/python.sh"
TASKS=(elevated_button occluded_retrieval weight_lever composite)

export OMNI_KIT_ACCEPT_EULA=Y

for task in "${TASKS[@]}"; do
    echo ""
    echo "=========================================="
    echo " Starting DRND: $task  ($(date))"
    echo "=========================================="
    "$ISAAC_PY" "$REPO/launch.py" \
        --config "$REPO/configs/local.yaml" \
        --task "$task" \
        --curiosity drnd
    echo "  DRND $task complete ($(date))"
done

echo ""
echo "All 4 tasks done. Run: python3 $REPO/plot_runs.py --out all_tasks.png"
