#!/bin/bash
# V9 hourly wrapper: 1) run v9_live_runner, 2) notify only on error (no heartbeat)
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/trading_system"
export V9_LIVE_MODE=LIVE
python3 -u -m v9_live_runner
RC=$?
cd "$ROOT"
if [ "$RC" -ne 0 ]; then
    python3 ops/send_v9_dashboard.py --error || true
fi
exit $RC
