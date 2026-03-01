#!/usr/bin/env bash
set -euo pipefail

sudo cp /home/trader/systemd/v9_telegram_ops.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now v9_telegram_ops.service

echo "v9_telegram_ops deployed and started."
