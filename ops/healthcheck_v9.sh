#!/usr/bin/env bash
set -euo pipefail

SNAPSHOT_CSV="/home/trader/trading_system/logs/v9_ops_snapshot.csv"

echo "=== V9 QUICK HEALTHCHECK ==="
echo "[timer next run]"
timer_line="$(systemctl list-timers | grep trading_bot_v9_oneshot | head -n 1 || true)"
if [[ -n "$timer_line" ]]; then
  echo "$timer_line"
else
  echo "NONE (timer not found)"
fi

echo
echo "[last snapshot]"
if [[ -f "$SNAPSHOT_CSV" ]]; then
  tail -n 1 "$SNAPSHOT_CSV"
else
  echo "NONE (snapshot file missing: $SNAPSHOT_CSV)"
fi

echo
echo "[snapshot age minutes]"
python3 - <<'PY'
from datetime import datetime, timezone
import csv, os
path="/home/trader/trading_system/logs/v9_ops_snapshot.csv"
if not os.path.exists(path):
    print("snapshot_age_min=UNKNOWN (file missing)")
else:
    with open(path) as f:
        rows=list(csv.reader(f))
    if not rows:
        print("snapshot_age_min=UNKNOWN (empty file)")
    else:
        last=rows[-1]
        ts=datetime.fromisoformat(last[0].replace("Z","+00:00"))
        age=(datetime.now(timezone.utc)-ts).total_seconds()/60
        print(f"snapshot_age_min={age:.1f}")
PY

echo
echo "[last runner exit status]"
systemctl show trading_bot_v9_oneshot.service -p ExecMainStatus --no-pager

echo
echo "[last connectivity test PASS/FAIL]"
conn_line="$(journalctl -q -u trading_bot_v9_oneshot.service --no-pager \
  | grep -E 'PASS: Order connectivity test succeeded|FAIL:' \
  | tail -n 1 || true)"
if [[ -n "$conn_line" ]]; then
  echo "$conn_line"
else
  echo "NONE (connectivity test not run yet)"
fi

echo
echo "[telegram ops service]"
telegram_state="$(systemctl is-active v9_telegram_ops 2>/dev/null || true)"
if [[ -n "$telegram_state" ]]; then
  echo "$telegram_state"
else
  echo "unknown"
fi

echo
echo "[hint]"
echo "To validate order connectivity now:"
echo "sudo systemctl set-environment V9_ORDER_CONNECTIVITY_TEST=1 && sudo systemctl start trading_bot_v9_oneshot.service && sudo systemctl unset-environment V9_ORDER_CONNECTIVITY_TEST"
