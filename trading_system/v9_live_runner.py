"""
V9.1 Live Validation Runner.
PAPER: simulated fills, log to v9_trade_records.
MICRO-LIVE: real orders at 10% notional, log to v9_trade_records.

Strategy: V9_REGIME_CORE (frozen). Execution layer only.
Env: V9_LIVE_MODE=PAPER|MICRO-LIVE|LIVE
     V9_ORDER_CONNECTIVITY_TEST=1  (optional) run order connectivity test
No Telegram/dotenv side-effects (freeze). Notifications via ops/send_v9_dashboard.py.
"""
from __future__ import annotations

import csv
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config_v9 import (
    STRATEGY_VERSION,
    VOL_LOW,
    VOL_HIGH,
    FREEZE_MODE,
    FREEZE_UNTIL,
    HARD_CAP_LEVERAGE,
)

# Ops snapshot CSV header (single line per run)
V9_OPS_SNAPSHOT_HEADER = (
    "timestamp,account_equity,available_balance,wallet_balance,current_notional,"
    "effective_leverage,position_count,open_orders_count,positions_detail"
)


def _get_mode() -> str:
    m = (os.getenv("V9_LIVE_MODE") or "PAPER").strip().upper()
    if m == "MICRO-LIVE":
        return "MICRO-LIVE"
    if m == "LIVE":
        return "LIVE"
    return "PAPER"


def _get_client():
    """Return BinanceFuturesClient if credentials exist, else None."""
    api_key = os.getenv("BINANCE_API_KEY") or os.getenv("BINANCE_FUTURES_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET") or os.getenv("BINANCE_FUTURES_API_SECRET")
    if not api_key or not api_secret:
        return None
    from core.binance_client import BinanceFuturesClient
    return BinanceFuturesClient(
        api_key=api_key,
        api_secret=api_secret,
        base_url=os.getenv("BINANCE_DATA_URL", "https://fapi.binance.com"),
    )


def _fetch_account_metrics() -> tuple[float, float, float, int]:
    """Returns (account_equity, current_notional, effective_leverage, position_count). Uses env only (no load_dotenv)."""
    try:
        client = _get_client()
        if not client:
            return 0.0, 0.0, 0.0, 0
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


def _fetch_full_snapshot() -> dict | None:
    """Fetch full account snapshot for ops logging. Returns dict or None on failure."""
    try:
        client = _get_client()
        if not client:
            return None
        acc = client.futures_account()
        equity = float(acc.get("totalWalletBalance", 0) or 0)
        avail = float(acc.get("availableBalance", 0) or 0)
        wallet = float(acc.get("totalWalletBalance", 0) or 0)
        positions = acc.get("positions", []) or []
        notional = 0.0
        position_count = 0
        pos_details = []
        for p in positions:
            amt = float(p.get("positionAmt", 0) or 0)
            if amt != 0:
                mark = float(p.get("markPrice", 0) or 0)
                entry = float(p.get("entryPrice", 0) or 0)
                upnl = float(p.get("unRealizedProfit", 0) or 0)
                notional += abs(amt * mark)
                position_count += 1
                pos_details.append({
                    "symbol": p.get("symbol", ""),
                    "size": amt,
                    "entry_price": entry,
                    "unrealized_pnl": upnl,
                })
        lev = notional / equity if equity > 0 else 0.0
        open_orders = client.get_open_orders(symbol=None)
        open_orders_count = len(open_orders) if isinstance(open_orders, list) else 0
        positions_summary = "|".join(
            f"{d['symbol']}:{d['size']:.6g}@{d['entry_price']:.2f} upnl={d['unrealized_pnl']:.2f}"
            for d in pos_details
        ) or ""
        return {
            "account_equity": equity,
            "available_balance": avail,
            "wallet_balance": wallet,
            "current_notional": notional,
            "effective_leverage": lev,
            "position_count": position_count,
            "open_orders_count": open_orders_count,
            "positions_detail": positions_summary,
        }
    except Exception:
        return None


def _write_ops_snapshot(snap: dict) -> None:
    """Print snapshot to stdout and append one row to logs/v9_ops_snapshot.csv."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [
        f"[V9 OPS SNAPSHOT] timestamp={ts}",
        f"  account_equity={snap['account_equity']:.2f}",
        f"  available_balance={snap['available_balance']:.2f}",
        f"  wallet_balance={snap['wallet_balance']:.2f}",
        f"  current_notional={snap['current_notional']:.2f}",
        f"  effective_leverage={snap['effective_leverage']:.4f}",
        f"  position_count={snap['position_count']}",
        f"  open_orders_count={snap['open_orders_count']}",
    ]
    if snap.get("positions_detail"):
        lines.append(f"  positions_detail={snap['positions_detail']}")
    for line in lines:
        print(line)
    log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = log_dir / "v9_ops_snapshot.csv"
    try:
        exists = csv_path.exists()
        with open(csv_path, "a", encoding="utf-8", newline="") as f:
            if not exists:
                f.write(V9_OPS_SNAPSHOT_HEADER + "\n")
            f.write(
                f"{ts},{snap['account_equity']:.2f},{snap['available_balance']:.2f},"
                f"{snap['wallet_balance']:.2f},{snap['current_notional']:.2f},"
                f"{snap['effective_leverage']:.6f},{snap['position_count']},"
                f"{snap['open_orders_count']},\"{snap['positions_detail'].replace(chr(34), '')}\"\n"
            )
    except Exception as e:
        print(f"  [WARN] v9_ops_snapshot.csv append failed: {e}")


def _print_startup() -> None:
    """Print banner and write health check file (for ops/send_v9_dashboard.py to parse)."""
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
        f"position_count={position_count}",
    ]
    line = " ".join(lines)
    print(line)
    _write_health_check(line)


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


def _run_order_connectivity_test() -> bool:
    """
    Ops-only connectivity test. V9_ORDER_CONNECTIVITY_TEST=1 to run.
    Uses Binance POST /fapi/v1/order/test (validates without executing).
    Returns True on success, False on failure. Never raises.
    """
    log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    test_log = log_dir / "v9_ops_order_test.log"
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def _log(msg: str) -> None:
        line = f"[{ts}] {msg}"
        print(line)
        try:
            with open(test_log, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass

    try:
        client = _get_client()
        if not client:
            _log("FAIL: No API credentials (BINANCE_API_KEY/SECRET)")
            return False
        _log("Fetching account + positions + open orders...")
        snap = _fetch_full_snapshot()
        if snap:
            _log(f"  account_equity={snap['account_equity']:.2f} position_count={snap['position_count']}")
        _log("Sending test order (POST /fapi/v1/order/test - no execution)...")
        # Binance test endpoint validates request without executing
        client.place_order_test({
            "symbol": "BTCUSDT",
            "side": "BUY",
            "type": "LIMIT",
            "timeInForce": "GTX",  # Post-only
            "quantity": "0.001",
            "price": "1000",  # Far from market, will never fill
        })
        _log("PASS: Order connectivity test succeeded")
        try:
            from core.v9_trade_record import append_v9_trade_record
            append_v9_trade_record(
                timestamp=ts,
                symbol="BTCUSDT",
                side="BUY",
                price=0.0,
                qty=0.0,
                regime_vol=0.0,
                reason="ORDER_CONNECTIVITY",
                event_type="OPS_TEST",
            )
        except Exception as e:
            _log(f"  [WARN] append_v9_trade_record: {e}")
        return True
    except Exception as e:
        err = str(e)
        _log(f"FAIL: {err}")
        if "timestamp" in err.lower() or "recvWindow" in err.lower():
            _log("  (Hint: check system clock / recvWindow)")
        if "insufficient" in err.lower() or "margin" in err.lower():
            _log("  (Hint: check margin / balance)")
        if "permission" in err.lower() or "unauthorized" in err.lower() or "401" in err:
            _log("  (Hint: API key may lack futures trading permission)")
        return False


def main():
    _print_startup()
    mode = _get_mode()
    if mode == "PAPER":
        print("V9.1 PAPER mode: signal logging only (no real orders)")
    else:
        print(f"V9.1 {mode} mode: real orders")

    snap = _fetch_full_snapshot()
    if snap:
        _write_ops_snapshot(snap)
    else:
        print("[V9 OPS SNAPSHOT] skipped (no API credentials or fetch failed)")

    if os.getenv("V9_ORDER_CONNECTIVITY_TEST", "").strip() == "1":
        print("[V9] Running order connectivity test...")
        _run_order_connectivity_test()
    else:
        print("(Set V9_ORDER_CONNECTIVITY_TEST=1 to run order connectivity test)")

    print("Run tests/run_v9_live.py for full daily cycle (requires data fetch).")


if __name__ == "__main__":
    main()
