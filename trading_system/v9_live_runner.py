"""
V9.1 Live Validation Runner.
PAPER: simulated fills, log to v9_trade_records.
MICRO-LIVE: real orders at 10% notional, log to v9_trade_records.

Strategy: V9_REGIME_CORE (frozen). Execution layer only.
Env: V9_LIVE_MODE=PAPER|MICRO-LIVE|LIVE
"""
from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

_LOG = logging.getLogger("v9_live_runner")

from config_v9 import (
    STRATEGY_VERSION,
    VOL_LOW,
    VOL_HIGH,
    FREEZE_MODE,
    FREEZE_UNTIL,
    HARD_CAP_LEVERAGE,
)


def _get_mode() -> str:
    m = (os.getenv("V9_LIVE_MODE") or "PAPER").strip().upper()
    if m == "MICRO-LIVE":
        return "MICRO-LIVE"
    if m == "LIVE":
        return "LIVE"
    return "PAPER"


def _fetch_account_metrics() -> tuple[float, float, float, int]:
    """Returns (account_equity, current_notional, effective_leverage, position_count)."""
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=ROOT / ".env", override=True)
        api_key = os.getenv("BINANCE_API_KEY") or os.getenv("BINANCE_FUTURES_API_KEY")
        api_secret = os.getenv("BINANCE_API_SECRET") or os.getenv("BINANCE_FUTURES_API_SECRET")
        if not api_key or not api_secret:
            return 0.0, 0.0, 0.0, 0
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
        position_count = 0
        for p in positions:
            amt = float(p.get("positionAmt", 0) or 0)
            if amt != 0:
                mark = float(p.get("markPrice", 0) or 0)
                notional += abs(amt * mark)
                position_count += 1
        lev = notional / equity if equity > 0 else 0.0
        return equity, notional, lev, position_count
    except Exception:
        return 0.0, 0.0, 0.0, 0


def _print_startup() -> tuple[float, float, float, int]:
    """Returns (equity, notional, lev, position_count) for status dashboard."""
    mode = _get_mode()
    commit = _get_commit_hash()
    equity, notional, lev, position_count = _fetch_account_metrics()
    hard_cap = equity * HARD_CAP_LEVERAGE if equity > 0 else 0.0

    lines = [
        f"STRATEGY_VERSION={STRATEGY_VERSION}",
        f"GIT_COMMIT={commit}",
        f"MODE={mode}",
        f"VOL_LOW={VOL_LOW}",
        f"VOL_HIGH={VOL_HIGH}",
        f"FREEZE_MODE={FREEZE_MODE}",
        f"FREEZE_UNTIL={FREEZE_UNTIL}",
        f"account_equity={equity:.2f}",
        f"HARD_CAP_NOTIONAL={hard_cap:.2f}",
        f"current_notional={notional:.2f}",
        f"effective_leverage={lev:.4f}",
    ]
    line = " ".join(lines)
    print(line)
    _write_health_check(line)
    return equity, notional, lev, position_count


def _write_health_check(line: str):
    """Write startup params to health check file (every startup)."""
    try:
        log_dir = ROOT / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / "v9_health_check.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write(line + "\n")
            f.write(f"# Written at {datetime.now(timezone.utc).isoformat()}\n")
    except Exception:
        pass


def _get_commit_hash() -> str:
    try:
        import subprocess
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, timeout=5
        )
        return (r.stdout or "").strip()[:12] if r.returncode == 0 else ""
    except Exception:
        return ""


def _send_status_dashboard(
    equity: float,
    notional: float,
    lev: float,
    position_count: int,
    commit: str,
) -> None:
    """Send V9 status dashboard to Telegram. If token missing, log warning only (no crash)."""
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=ROOT / ".env", override=True)
        token = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
        chat_id = (os.getenv("TELEGRAM_CHAT_ID") or "").strip()
        if not token or not chat_id:
            _LOG.warning("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set; skipping status dashboard")
            print("  [WARN] TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set; skipping status dashboard")
            return
        from core.telegram_notifier import TelegramNotifier
        hard_cap = equity * HARD_CAP_LEVERAGE if equity > 0 else 0.0
        utc8 = timezone(timedelta(hours=8))
        now_utc8 = datetime.now(utc8).strftime("%Y-%m-%d %H:%M:%S")
        msg = (
            "📊 <b>V9 系統狀態看板</b>\n\n"
            f"• STRATEGY_VERSION: {STRATEGY_VERSION}\n"
            f"• GIT_COMMIT: {commit}\n"
            f"• account_equity: ${equity:,.2f}\n"
            f"• HARD_CAP_NOTIONAL: ${hard_cap:,.2f}\n"
            f"• current_notional: ${notional:,.2f}\n"
            f"• effective_leverage: {lev:.4f}\n"
            f"• FREEZE_UNTIL: {FREEZE_UNTIL}\n"
            f"• 當前持倉數量: {position_count}\n"
            f"• 執行時間 (UTC+8): {now_utc8}"
        )
        notifier = TelegramNotifier(bot_token=token, chat_id=chat_id, enabled=True)
        if notifier.send_message(msg, parse_mode="HTML"):
            print("  [OK] V9 status dashboard sent")
    except Exception as e:
        _LOG.warning("Failed to send V9 status dashboard: %s", e)
        print(f"  [WARN] Failed to send V9 status dashboard: {e}")


def run_once_paper(signal_time_utc: datetime, symbol: str, side: str, price: float, qty: float,
                   regime_vol: float, reason: str):
    """PAPER: record theoretical fill."""
    from core.v9_trade_record import append_v9_trade_record
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    sig_ts = signal_time_utc.strftime("%Y-%m-%dT%H:%M:%SZ") if hasattr(signal_time_utc, "strftime") else str(signal_time_utc)
    append_v9_trade_record(
        timestamp=ts,
        symbol=symbol,
        side=side,
        price=price,
        qty=qty,
        regime_vol=regime_vol,
        reason=reason,
        fees=0.0,
        slippage_est=0.0,
        signal_time=sig_ts,
        order_time=ts,
        mode="PAPER",
    )


def run_once_micro_live(signal_time_utc: datetime, symbol: str, side: str, price: float, qty: float,
                        regime_vol: float, reason: str, client=None):
    """MICRO-LIVE: place real order at 10% notional, record actual fill."""
    from core.v9_trade_record import append_v9_trade_record
    sig_ts = signal_time_utc.strftime("%Y-%m-%dT%H:%M:%SZ") if hasattr(signal_time_utc, "strftime") else str(signal_time_utc)
    order_time = datetime.now(timezone.utc)
    fees = 0.0
    slippage_est = 0.0
    actual_price = price
    if client:
        try:
            order = client.place_order({
                "symbol": symbol,
                "side": "BUY" if side.upper() == "BUY" else "SELL",
                "type": "MARKET",
                "quantity": qty,
            })
            if order:
                actual_price = float(order.get("avgPrice") or order.get("price") or price)
                fees = float(order.get("commission", 0) or 0)
                if actual_price and price:
                    slippage_bps = abs(actual_price - price) / price * 10000
                    slippage_est = slippage_bps
        except Exception as e:
            print(f"  [ERR] MICRO-LIVE order failed: {e}")
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    ord_ts = order_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    append_v9_trade_record(
        timestamp=ts,
        symbol=symbol,
        side=side,
        price=actual_price,
        qty=qty,
        regime_vol=regime_vol,
        reason=reason,
        fees=fees,
        slippage_est=slippage_est,
        signal_time=sig_ts,
        order_time=ord_ts,
        mode="MICRO-LIVE",
    )


def main():
    equity, notional, lev, position_count = _print_startup()
    mode = _get_mode()
    if mode == "PAPER":
        print("V9.1 PAPER mode: signal logging only (no real orders)")
    else:
        print(f"V9.1 {mode} mode: real orders")
    _send_status_dashboard(equity, notional, lev, position_count, _get_commit_hash())
    print("Run tests/run_v9_live.py for full daily cycle (requires data fetch).")


if __name__ == "__main__":
    main()
