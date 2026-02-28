#!/bin/bash
# V9.1 Deploy: lock to commit hash, then start service.
# Usage: ./deploy_v9.sh [commit_hash]
# If commit_hash omitted, uses current HEAD. Records hash to logs/deploy_hash.txt.
# MODE: V9_LIVE_MODE=PAPER|MICRO-LIVE (default MICRO-LIVE for production)
set -e
cd "$(dirname "$0")"
mkdir -p logs
HASH="${1:-$(git rev-parse HEAD)}"
git checkout "$HASH" 2>/dev/null || true
echo "$HASH" > logs/deploy_hash.txt
echo "Deployed at commit: $HASH"
echo "To start V9 MICRO-LIVE: sudo systemctl start trading_bot_v9"
echo "  (or: V9_LIVE_MODE=MICRO-LIVE python3 -m v9_live_runner)"
