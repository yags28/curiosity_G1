#!/usr/bin/env bash
# Daily auto-push of CDL project to GitHub.
# Scheduled via crontab at 5 pm America/New_York.

set -euo pipefail

REPO="/home/kevin/projects/CDL"
LOG="$HOME/.local/share/cdl_push.log"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M %Z')

mkdir -p "$(dirname "$LOG")"
exec >> "$LOG" 2>&1

echo "=== auto-push $TIMESTAMP ==="

cd "$REPO"

# Stage everything (respects .gitignore)
git add -A

# Skip if nothing changed
if git diff --quiet && git diff --staged --quiet; then
    echo "Nothing to commit — skipping push."
    exit 0
fi

git commit -m "auto-push: $TIMESTAMP"
git push origin main
echo "Push complete."
