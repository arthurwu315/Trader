#!/usr/bin/env bash
set -euo pipefail

sudo cp /home/trader/trading_system/systemd/v9_telegram_ops.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now v9_telegram_ops.service

echo "v9_telegram_ops deployed and started."
systemctl status v9_telegram_ops --no-pager || true
systemctl is-enabled v9_telegram_ops || true
ps aux | grep -E "telegram_ops_bot|bots\.telegram_ops_bot" | grep -v grep || true
