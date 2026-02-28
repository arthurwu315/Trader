"""
V9 Telegram Ops Bot - Read-only command interface.
Commands: /status /positions /equity /help
Data: logs/v9_ops_snapshot.csv (prefer) or account API fetch.
No trading, no strategy triggers. Independent of v9_live_runner.
"""
from __future__ import annotations

import csv
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import requests
except ImportError:
    print("requests not installed", file=sys.stderr)
    sys.exit(1)

SNAPSHOT_CSV = ROOT / "logs" / "v9_ops_snapshot.csv"
HEALTH_FILE = ROOT / "logs" / "v9_health_check.txt"


def _poll_updates(bot_token: str, offset: int) -> tuple[list, int]:
    try:
        resp = requests.get(
            f"https://api.telegram.org/bot{bot_token}/getUpdates",
            params={"offset": offset, "timeout": 30},
            timeout=35,
        )
        data = resp.json() if resp.status_code == 200 else {}
        rows = data.get("result", []) if isinstance(data, dict) else []
        next_off = offset
        for u in rows:
            next_off = max(next_off, int(u.get("update_id", 0)) + 1)
        return rows, next_off
    except Exception:
        return [], offset


def _send(bot_token: str, chat_id: str, text: str) -> bool:
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "disable_web_page_preview": True},
            timeout=10,
        )
        d = resp.json() if resp.status_code == 200 else {}
        return bool(isinstance(d, dict) and d.get("ok"))
    except Exception:
        return False


def _read_snapshot_latest() -> dict | None:
    """Read latest row from v9_ops_snapshot.csv. Returns dict or None."""
    if not SNAPSHOT_CSV.exists():
        return None
    try:
        with open(SNAPSHOT_CSV, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return None
        return dict(rows[-1])
    except Exception:
        return None


def _read_health_check() -> dict:
    """Parse v9_health_check.txt key=value."""
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


def _last_run_time() -> str:
    """Last run timestamp from health check or snapshot."""
    try:
        with open(HEALTH_FILE, encoding="utf-8") as f:
            for line in f:
                if line.strip().startswith("# Written at "):
                    return line.replace("# Written at ", "").strip()
    except Exception:
        pass
    snap = _read_snapshot_latest()
    if snap and snap.get("timestamp"):
        return snap["timestamp"]
    return "N/A"


def _fetch_account_api() -> dict | None:
    """Fetch account via Binance API (read-only)."""
    api_key = os.getenv("BINANCE_API_KEY") or os.getenv("BINANCE_FUTURES_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET") or os.getenv("BINANCE_FUTURES_API_SECRET")
    if not api_key or not api_secret:
        return None
    try:
        from core.binance_client import BinanceFuturesClient
        client = BinanceFuturesClient(
            api_key=api_key,
            api_secret=api_secret,
            base_url=os.getenv("BINANCE_DATA_URL", "https://fapi.binance.com"),
        )
        acc = client.futures_account()
        equity = float(acc.get("totalWalletBalance", 0) or 0)
        positions = acc.get("positions", []) or []
        notional = 0.0
        pos_details = []
        for p in positions:
            amt = float(p.get("positionAmt", 0) or 0)
            if amt != 0:
                mark = float(p.get("markPrice", 0) or 0)
                entry = float(p.get("entryPrice", 0) or 0)
                upnl = float(p.get("unRealizedProfit", 0) or 0)
                notional += abs(amt * mark)
                pos_details.append({
                    "symbol": p.get("symbol", ""),
                    "size": amt,
                    "entry": entry,
                    "upnl": upnl,
                })
        lev = notional / equity if equity > 0 else 0.0
        return {
            "account_equity": equity,
            "current_notional": notional,
            "effective_leverage": lev,
            "position_count": len(pos_details),
            "positions_detail": pos_details,
        }
    except Exception:
        return None


def _get_data() -> dict:
    """Data source: snapshot first, else API."""
    snap = _read_snapshot_latest()
    if snap:
        try:
            return {
                "account_equity": float(snap.get("account_equity", 0) or 0),
                "current_notional": float(snap.get("current_notional", 0) or 0),
                "effective_leverage": float(snap.get("effective_leverage", 0) or 0),
                "position_count": int(float(snap.get("position_count", 0) or 0)),
                "positions_detail": snap.get("positions_detail", ""),
                "source": "snapshot",
            }
        except (ValueError, TypeError):
            pass
    api_data = _fetch_account_api()
    if api_data:
        detail = " | ".join(
            f"{p['symbol']}:{p['size']:.6g}@{p['entry']:.2f} upnl={p['upnl']:.2f}"
            for p in api_data.get("positions_detail", [])
        )
        return {
            "account_equity": api_data["account_equity"],
            "current_notional": api_data["current_notional"],
            "effective_leverage": api_data["effective_leverage"],
            "position_count": api_data["position_count"],
            "positions_detail": detail,
            "source": "api",
        }
    return {
        "account_equity": 0.0,
        "current_notional": 0.0,
        "effective_leverage": 0.0,
        "position_count": 0,
        "positions_detail": "",
        "source": "none",
    }


def _cmd_status() -> str:
    data = _get_data()
    health = _read_health_check()
    freeze_until = health.get("FREEZE_UNTIL", "N/A")
    last_run = _last_run_time()
    return (
        "📊 V9 Status\n"
        f"• Equity: ${data['account_equity']:,.2f}\n"
        f"• Notional: ${data['current_notional']:,.2f}\n"
        f"• Leverage: {data['effective_leverage']:.4f}\n"
        f"• Freeze until: {freeze_until}\n"
        f"• Last run: {last_run}\n"
        f"• Source: {data['source']}"
    )


def _cmd_positions() -> str:
    data = _get_data()
    if not data["positions_detail"]:
        return "📋 Positions: None"
    return f"📋 Positions\n{data['positions_detail']}"


def _cmd_equity() -> str:
    data = _get_data()
    return f"💰 Equity: ${data['account_equity']:,.2f}"


def _cmd_help() -> str:
    return (
        "🤖 V9 Ops Bot (read-only)\n"
        "/status   - equity, notional, leverage, freeze, last run\n"
        "/positions - current positions\n"
        "/equity   - futures account equity\n"
        "/help     - this message"
    )


def _handle_command(cmd: str) -> str:
    c = (cmd or "").strip().lower()
    if c == "/status":
        return _cmd_status()
    if c == "/positions":
        return _cmd_positions()
    if c == "/equity":
        return _cmd_equity()
    if c == "/help":
        return _cmd_help()
    return _cmd_help()


def main() -> None:
    token = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
    chat_id = (os.getenv("TELEGRAM_CHAT_ID") or "").strip()
    if not token or not chat_id:
        print("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set", file=sys.stderr)
        sys.exit(1)

    offset = 0
    while True:
        updates, offset = _poll_updates(token, offset)
        for u in updates:
            msg = u.get("message") or {}
            text = (msg.get("text") or "").strip()
            from_chat = str(msg.get("chat", {}).get("id", ""))
            if from_chat != chat_id:
                continue
            if not text or not text.startswith("/"):
                continue
            reply = _handle_command(text)
            _send(token, chat_id, reply)


if __name__ == "__main__":
    main()
