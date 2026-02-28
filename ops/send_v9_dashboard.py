#!/usr/bin/env python3
"""
V9 Status Dashboard - External Ops Notifier.
Reads logs/v9_health_check.txt (written by v9_live_runner), parses fields,
sends Telegram status dashboard. Token/chat_id from environment (no load_dotenv).
Exit 0 if token missing (warning only, no crash).
"""
from __future__ import annotations

import os
import re
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HEALTH_FILE = REPO / "trading_system" / "logs" / "v9_health_check.txt"
TRADE_RECORDS = REPO / "trading_system" / "logs" / "v9_trade_records.csv"


def _parse_health_check() -> dict:
    """Parse key=value from first non-comment line of v9_health_check.txt."""
    out = {}
    if not HEALTH_FILE.exists():
        return out
    try:
        with open(HEALTH_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                for m in re.finditer(r"(\w+)=([^\s]+)", line):
                    out[m.group(1)] = m.group(2)
                break
    except Exception:
        pass
    return out


def _last_v9_event_time() -> str:
    """Last V9_REGIME_CORE record timestamp from v9_trade_records.csv (optional)."""
    if not TRADE_RECORDS.exists():
        return ""
    try:
        with open(TRADE_RECORDS, encoding="utf-8") as f:
            lines = f.readlines()
        for line in reversed(lines[-200:]):
            if "V9_REGIME_CORE" in line:
                parts = line.strip().split(",")
                if len(parts) >= 1 and parts[0] and not parts[0].startswith("timestamp"):
                    return parts[0]
    except Exception:
        pass
    return ""


def main() -> int:
    token = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
    chat_id = (os.getenv("TELEGRAM_CHAT_ID") or "").strip()
    if not token or not chat_id:
        print("  [WARN] TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set; skipping dashboard")
        return 0

    data = _parse_health_check()
    if not data:
        print("  [WARN] No v9_health_check.txt; skipping dashboard")
        return 0

    strategy_version = data.get("STRATEGY_VERSION", "?")
    git_commit = data.get("GIT_COMMIT", "?")
    mode = data.get("MODE", "?")
    vol_low = data.get("VOL_LOW", "?")
    vol_high = data.get("VOL_HIGH", "?")
    freeze_until = data.get("FREEZE_UNTIL", "?")
    account_equity = data.get("account_equity", "0")
    hard_cap = data.get("HARD_CAP_NOTIONAL", "0")
    current_notional = data.get("current_notional", "0")
    effective_leverage = data.get("effective_leverage", "0")
    position_count = data.get("position_count", "0")

    try:
        eq = float(account_equity)
        hc = float(hard_cap)
        nt = float(current_notional)
        lev = float(effective_leverage)
        account_equity = f"${eq:,.2f}"
        hard_cap = f"${hc:,.2f}"
        current_notional = f"${nt:,.2f}"
        effective_leverage = f"{lev:.4f}"
    except (ValueError, TypeError):
        pass

    utc8 = timezone(timedelta(hours=8))
    now_utc8 = datetime.now(utc8).strftime("%Y-%m-%d %H:%M:%S")
    last_event = _last_v9_event_time()

    msg = (
        "📊 <b>V9 系統狀態看板</b>\n\n"
        f"• STRATEGY_VERSION: {strategy_version}\n"
        f"• GIT_COMMIT: {git_commit}\n"
        f"• MODE: {mode}\n"
        f"• VOL_LOW/VOL_HIGH: {vol_low}/{vol_high}\n"
        f"• FREEZE_UNTIL: {freeze_until}\n"
        f"• account_equity: {account_equity}\n"
        f"• HARD_CAP_NOTIONAL: {hard_cap}\n"
        f"• current_notional: {current_notional}\n"
        f"• effective_leverage: {effective_leverage}\n"
        f"• 當前持倉數量: {position_count}\n"
        f"• 執行時間 (UTC+8): {now_utc8}\n"
    )
    if last_event:
        msg += f"• 最後事件: {last_event}\n"

    try:
        sys.path.insert(0, str(REPO / "trading_system"))
        from core.telegram_notifier import TelegramNotifier
        notifier = TelegramNotifier(bot_token=token, chat_id=chat_id, enabled=True)
        if notifier.send_message(msg, parse_mode="HTML"):
            print("  [OK] V9 status dashboard sent")
        else:
            print("  [WARN] Telegram send failed")
    except Exception as e:
        print(f"  [WARN] Failed to send V9 dashboard: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
