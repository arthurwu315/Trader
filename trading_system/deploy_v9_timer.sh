#!/bin/bash
# Deploy V9 as systemd timer (oneshot) - Freeze policy
set -e
cd "$(dirname "$0")/.."
sudo cp systemd/trading_bot_v9_oneshot.service systemd/trading_bot_v9_oneshot.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now trading_bot_v9_oneshot.timer
echo ""
echo "=== systemctl list-timers | grep v9 ==="
systemctl list-timers | grep v9 || true
echo ""
echo "=== journalctl -u trading_bot_v9_oneshot.service -n 50 --no-pager ==="
journalctl -u trading_bot_v9_oneshot.service -n 50 --no-pager || true
