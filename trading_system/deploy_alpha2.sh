#!/bin/bash
# Alpha2 Deploy: lock to commit hash. No auto restart.
# Usage: ./deploy_alpha2.sh <commit_hash>
# Requires explicit commit_hash. Records hash to logs/deploy_alpha2_hash.txt.
set -e
cd "$(dirname "$0")"
mkdir -p logs
if [ -z "$1" ]; then
  echo "Usage: ./deploy_alpha2.sh <commit_hash>"
  exit 1
fi
HASH="$1"
git checkout "$HASH" 2>/dev/null || true
echo "$HASH" > logs/deploy_alpha2_hash.txt
echo "Alpha2 deployed at commit: $HASH"
echo "Copy service: sudo cp trading_bot_alpha2.service /etc/systemd/system/"
echo "Reload: sudo systemctl daemon-reload"
echo "Start: sudo systemctl start trading_bot_alpha2"
echo "  (no auto restart - Restart=no)"
