#!/usr/bin/env python3
"""
V9 Notifier - Error / Daily Summary / Manual Dashboard.
Modes: --error (runner failed), --daily (daily summary, opt-in), --force (manual).
No-args: no-op (no heartbeat). Token/chat_id from env only.
"""
from __future__ import annotations

import csv
import os
import re
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HEALTH_FILE = REPO / "trading_system" / "logs" / "v9_health_check.txt"
TRADE_RECORDS = REPO / "trading_system" / "logs" / "v9_trade_records.csv"
SNAPSHOT_CSV = REPO / "trading_system" / "logs" / "v9_ops_snapshot.csv"


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


def _build_dashboard_msg(data: dict) -> str:
    utc8 = timezone(timedelta(hours=8))
    now_utc8 = datetime.now(utc8).strftime("%Y-%m-%d %H:%M:%S")
    last_event = _last_v9_event_time()
    msg = (
        "📊 <b>V9 系統狀態看板</b>\n\n"
        f"• STRATEGY_VERSION: {data.get('STRATEGY_VERSION', '?')}\n"
        f"• GIT_COMMIT: {data.get('GIT_COMMIT', '?')}\n"
        f"• MODE: {data.get('MODE', '?')}\n"
        f"• VOL_LOW/VOL_HIGH: {data.get('VOL_LOW', '?')}/{data.get('VOL_HIGH', '?')}\n"
        f"• FREEZE_UNTIL: {data.get('FREEZE_UNTIL', '?')}\n"
        f"• account_equity: {data.get('account_equity_fmt', '0')}\n"
        f"• HARD_CAP_NOTIONAL: {data.get('hard_cap_fmt', '0')}\n"
        f"• current_notional: {data.get('current_notional_fmt', '0')}\n"
        f"• effective_leverage: {data.get('effective_leverage_fmt', '0')}\n"
        f"• 當前持倉數量: {data.get('position_count', '0')}\n"
        f"• 執行時間 (UTC+8): {now_utc8}\n"
    )
    if last_event:
        msg += f"• 最後事件: {last_event}\n"
    return msg


def _build_error_msg(data: dict) -> str:
    utc8 = timezone(timedelta(hours=8))
    now_utc8 = datetime.now(utc8).strftime("%Y-%m-%d %H:%M:%S")
    return (
        "🚨 <b>V9 ERROR</b>\n\n"
        "Runner exited non-zero. Check journalctl.\n\n"
        f"• STRATEGY_VERSION: {data.get('STRATEGY_VERSION', '?')}\n"
        f"• MODE: {data.get('MODE', '?')}\n"
        f"• account_equity: {data.get('account_equity_fmt', '?')}\n"
        f"• position_count: {data.get('position_count', '?')}\n"
        f"• 執行時間 (UTC+8): {now_utc8}\n"
    )


def _build_daily_summary_msg() -> str:
    utc8 = timezone(timedelta(hours=8))
    now_utc8 = datetime.now(utc8).strftime("%Y-%m-%d %H:%M:%S")
    eq, notional, lev, pos_count = "N/A", "N/A", "N/A", "0"
    if SNAPSHOT_CSV.exists():
        try:
            with open(SNAPSHOT_CSV, encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            if rows:
                r = rows[-1]
                try:
                    eq = f"${float(r.get('account_equity', 0) or 0):,.2f}"
                    notional = f"${float(r.get('current_notional', 0) or 0):,.2f}"
                    lev = f"{float(r.get('effective_leverage', 0) or 0):.4f}"
                    pos_count = str(int(float(r.get('position_count', 0) or 0)))
                except (ValueError, TypeError):
                    pass
        except Exception:
            pass
    trade_count_24h = 0
    if TRADE_RECORDS.exists():
        try:
            cutoff = datetime.now(timezone.utc).timestamp() - 86400
            with open(TRADE_RECORDS, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ts = row.get("timestamp", "")
                    if not ts or ts.startswith("timestamp"):
                        continue
                    try:
                        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        if dt.timestamp() >= cutoff:
                            trade_count_24h += 1
                    except Exception:
                        pass
        except Exception:
            pass
    return (
        "📋 <b>V9 Daily Summary</b>\n\n"
        f"• Equity: {eq}\n"
        f"• Notional: {notional}\n"
        f"• Leverage: {lev}\n"
        f"• Positions: {pos_count}\n"
        f"• Trades (24h): {trade_count_24h}\n"
        f"• 時間 (UTC+8): {now_utc8}\n"
    )


def _send_msg(msg: str) -> bool:
    token = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
    chat_id = (os.getenv("TELEGRAM_CHAT_ID") or "").strip()
    if not token or not chat_id:
        print("  [WARN] TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set")
        return False
    try:
        sys.path.insert(0, str(REPO / "trading_system"))
        from core.telegram_notifier import TelegramNotifier
        notifier = TelegramNotifier(bot_token=token, chat_id=chat_id, enabled=True)
        ok = notifier.send_message(msg, parse_mode="HTML")
        if ok:
            print("  [OK] Telegram sent")
        return bool(ok)
    except Exception as e:
        print(f"  [WARN] Telegram send failed: {e}")
        return False


def main() -> int:
    args = sys.argv[1:] or []
    do_error = "--error" in args
    do_daily = "--daily" in args
    do_force = "--force" in args
    dry_run = "--dry-run" in args

    if not do_error and not do_daily and not do_force:
        return 0  # No-op: no heartbeat

    if do_daily and os.getenv("V9_DAILY_SUMMARY_ENABLED", "0").strip() != "1":
        return 0  # Daily summary disabled by default

    data = _parse_health_check()
    if not data:
        if do_error or do_force:
            print("  [WARN] No v9_health_check.txt")
        return 0

    try:
        eq = float(data.get("account_equity", 0) or 0)
        hc = float(data.get("HARD_CAP_NOTIONAL", 0) or 0)
        nt = float(data.get("current_notional", 0) or 0)
        lev = float(data.get("effective_leverage", 0) or 0)
        data["account_equity_fmt"] = f"${eq:,.2f}"
        data["hard_cap_fmt"] = f"${hc:,.2f}"
        data["current_notional_fmt"] = f"${nt:,.2f}"
        data["effective_leverage_fmt"] = f"{lev:.4f}"
    except (ValueError, TypeError):
        data.setdefault("account_equity_fmt", "?")
        data.setdefault("hard_cap_fmt", "?")
        data.setdefault("current_notional_fmt", "?")
        data.setdefault("effective_leverage_fmt", "?")

    if do_error:
        msg = _build_error_msg(data)
    elif do_daily:
        msg = _build_daily_summary_msg()
    else:
        msg = _build_dashboard_msg(data)

    if dry_run:
        print("[DRY-RUN] Would send:\n" + msg)
        return 0
    _send_msg(msg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
