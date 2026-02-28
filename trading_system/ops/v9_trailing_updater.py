"""
V9 Live Trailing Stop Updater.
Aligns live with backtest: monotonically tighten reduce-only STOP_MARKET.
Uses backtest formula: candidate = high - trail_mult*atr (BUY) or low + trail_mult*atr (SELL).
Only tightens; never loosens.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# V9 backtest params (from run_v8_backtest - do not change)
TRAIL_MULT = 2.5
SL_MULT = 2.5
HIGH_VOL_STOP_FACTOR = 0.7
VOL_HIGH = 4.2  # vol >= this => high vol, apply factor


def _get_client():
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


def _fetch_1d_ohlc(client, symbol: str, limit: int = 25) -> list[dict]:
    """Fetch 1D klines and return list of {timestamp, open, high, low, close}."""
    try:
        rows = client.get_klines(symbol=symbol, interval="1d", limit=limit)
        if not rows:
            return []
        out = []
        for r in rows:
            out.append({
                "timestamp": r[0],
                "open": float(r[1]),
                "high": float(r[2]),
                "low": float(r[3]),
                "close": float(r[4]),
            })
        return out
    except Exception:
        return []


def _compute_atr_14(bars: list[dict]) -> float:
    """Compute ATR(14) from OHLC bars. Returns 0 if insufficient data."""
    if len(bars) < 15:
        return 0.0
    trs = []
    for i in range(1, len(bars)):
        h = bars[i]["high"]
        l = bars[i]["low"]
        prev_c = bars[i - 1]["close"]
        tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
        trs.append(tr)
    atr = sum(trs[-14:]) / 14 if len(trs) >= 14 else 0.0
    return atr


def _is_high_vol_btc(client) -> bool:
    """Check if BTC is in high-vol regime (vol >= VOL_HIGH)."""
    bars = _fetch_1d_ohlc(client, "BTCUSDT", limit=25)
    if len(bars) < 21:
        return False
    atr20 = _compute_atr_14(bars[-21:])  # approximate ATR20
    if atr20 <= 0:
        return False
    close = bars[-1]["close"]
    if close <= 0:
        return False
    vol_pct = atr20 / close * 100.0
    return vol_pct >= VOL_HIGH


def _scan_sl_order(client, symbol: str) -> dict | None:
    """
    Find reduce-only STOP_MARKET (or STOP) for symbol.
    Returns order dict with stopPrice, orderId, reduceOnly, workingType, or None.
    """
    try:
        orders = client.get_open_orders(symbol=symbol)
        for o in orders:
            if not o.get("reduceOnly"):
                continue
            t = o.get("type", "")
            if t in ("STOP_MARKET", "STOP"):
                stop = o.get("stopPrice")
                if stop is not None:
                    return {
                        "orderId": o.get("orderId"),
                        "stopPrice": float(stop),
                        "reduceOnly": o.get("reduceOnly"),
                        "workingType": o.get("workingType", "MARK_PRICE"),
                        "side": o.get("side"),
                    }
    except Exception:
        pass
    return None


def _compute_candidate_stop(client, symbol: str, side: str, entry_price: float) -> float | None:
    """
    Compute candidate stop from backtest formula using last completed 1D bar.
    BUY: high - trail_mult * atr
    SELL: low + trail_mult * atr
    """
    bars = _fetch_1d_ohlc(client, symbol, limit=25)
    if len(bars) < 15:
        return None
    last = bars[-1]
    high = last["high"]
    low = last["low"]
    atr = _compute_atr_14(bars)
    if atr <= 0:
        atr = entry_price * 0.015
    effective_trail = TRAIL_MULT
    if _is_high_vol_btc(client):
        effective_trail = TRAIL_MULT * HIGH_VOL_STOP_FACTOR
    if side == "BUY":
        return high - effective_trail * atr
    return low + effective_trail * atr


def _get_tick_size(client, symbol: str) -> float:
    from core.order_utils import get_symbol_tick_size
    return get_symbol_tick_size(client, symbol)


def _round_price(price: float, tick_size: float) -> float:
    from core.order_utils import round_to_tick_size
    return round_to_tick_size(price, tick_size)


def _append_stop_event(event_type: str, symbol: str, side: str, old_stop: float, new_stop: float) -> None:
    from core.v9_trade_record import append_v9_trade_record
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    reason = f"old_stop={old_stop:.4f} new_stop={new_stop:.4f}"
    append_v9_trade_record(
        timestamp=ts,
        symbol=symbol,
        side=side,
        price=new_stop,
        qty=0.0,
        regime_vol=0.0,
        reason=reason,
        event_type=event_type,
    )


def _scan_all_stop_orders(client) -> dict[str, int]:
    """Scan all open orders, return {symbol: count} of reduce-only STOP_MARKET/STOP."""
    by_symbol: dict[str, int] = {}
    try:
        orders = client.get_open_orders(symbol=None)
        for o in orders or []:
            if not o.get("reduceOnly"):
                continue
            t = o.get("type", "")
            if t in ("STOP_MARKET", "STOP"):
                sym = o.get("symbol", "")
                if sym:
                    by_symbol[sym] = by_symbol.get(sym, 0) + 1
    except Exception as e:
        print(f"  [ERR] v9_trailing: scan open orders failed: {e}")
    return by_symbol


def scan_stop_orders_for_snapshot(client) -> list[dict]:
    """
    Scan open orders for reduce-only STOP/STOP_MARKET. Print verifiable log.
    Returns list of {symbol, stopPrice, orderId, type, workingType, reduceOnly} for snapshot.
    """
    stop_list: list[dict] = []
    try:
        orders = client.get_open_orders(symbol=None)
        for o in orders or []:
            if not o.get("reduceOnly"):
                continue
            t = o.get("type", "")
            if t in ("STOP_MARKET", "STOP"):
                stop_list.append({
                    "symbol": str(o.get("symbol", "")),
                    "stopPrice": o.get("stopPrice"),
                    "orderId": o.get("orderId"),
                    "type": t,
                    "workingType": str(o.get("workingType", "MARK_PRICE")),
                    "reduceOnly": o.get("reduceOnly"),
                })
    except Exception as e:
        print(f"  [ERR] v9_trailing: scan open orders failed: {e}")
    n = len(stop_list)
    print(f"  [V9 TRAILING] scan_stop_orders count={n}")
    if n > 0:
        for s in stop_list:
            print(f"    -> {s['symbol']} stopPrice={s['stopPrice']} type={s['type']} workingType={s['workingType']}")
    return stop_list


def _run_trailing_update(client, dry_run: bool, enabled: bool) -> int:
    """
    Always: scan open orders, print scanned_stop_orders log.
    If positions exist and (dry_run or enabled): compute/update stops.
    Returns 0 on success, 1 on any API error (for RC to trigger ops notify).
    """
    do_update = enabled and not dry_run

    by_symbol = _scan_all_stop_orders(client)

    try:
        acc = client.futures_account()
        positions = acc.get("positions", []) or []
    except Exception as e:
        print(f"  [ERR] v9_trailing: fetch account failed: {e}")
        return 1

    if not (dry_run or do_update):
        return 0

    rc = 0
    for p in positions:
        amt = float(p.get("positionAmt", 0) or 0)
        if amt == 0:
            continue
        symbol = p.get("symbol", "")
        entry = float(p.get("entryPrice", 0) or 0)
        side = "BUY" if amt > 0 else "SELL"
        close_side = "SELL" if amt > 0 else "BUY"

        sl_order = _scan_sl_order(client, symbol)
        current_stop = float(sl_order["stopPrice"]) if sl_order else None
        is_init = current_stop is None
        if is_init:
            bars = _fetch_1d_ohlc(client, symbol, limit=25)
            atr = _compute_atr_14(bars) if len(bars) >= 15 else entry * 0.015
            if atr <= 0:
                atr = entry * 0.015
            initial_sl = entry - SL_MULT * atr if side == "BUY" else entry + SL_MULT * atr
            current_stop = initial_sl

        candidate = _compute_candidate_stop(client, symbol, side, entry)
        if candidate is None:
            continue

        tick = _get_tick_size(client, symbol)
        candidate_r = _round_price(candidate, tick)
        current_r = _round_price(current_stop, tick)

        should_update = False
        if side == "BUY":
            should_update = candidate_r > current_r + tick
        else:
            should_update = candidate_r < current_r - tick

        # For INIT: always place stop at max(initial_sl, candidate) for BUY
        if is_init:
            if side == "BUY":
                stop_to_place = max(candidate_r, _round_price(initial_sl, tick))
            else:
                stop_to_place = min(candidate_r, _round_price(initial_sl, tick))
            should_update = True
        else:
            stop_to_place = candidate_r

        if not should_update:
            continue

        if dry_run:
            print(f"  [V9 TRAILING DRY-RUN] {symbol} {side}: old_stop={current_r:.4f} new_stop={stop_to_place:.4f} (would {'init' if is_init else 'update'})")
            _append_stop_event("STOP_DRY_RUN", symbol, side, current_r, stop_to_place)
            continue

        if not do_update:
            continue

        try:
            if sl_order:
                client.cancel_order(symbol=symbol, order_id=sl_order["orderId"])
            params = {
                "symbol": symbol,
                "side": close_side,
                "type": "STOP_MARKET",
                "stopPrice": stop_to_place,
                "closePosition": "true",
                "workingType": "MARK_PRICE",
            }
            client.place_order(params)
            event = "STOP_INIT" if is_init else "STOP_UPDATE"
            print(f"  [V9 TRAILING] {symbol} {side}: {event} old={current_r:.4f} new={stop_to_place:.4f}")
            _append_stop_event(event, symbol, side, current_r, stop_to_place)
        except Exception as e:
            print(f"  [ERR] v9_trailing {symbol}: {e}")
            rc = 1
    return rc


def run_trailing_updater() -> int:
    """Entry point. Always scans stop orders (observability). Returns 0 ok, 1 error."""
    client = _get_client()
    if not client:
        return 0
    dry_run = os.getenv("V9_TRAILING_DRY_RUN", "0").strip() == "1"
    enabled = os.getenv("V9_TRAILING_UPDATE_ENABLED", "0").strip() == "1"
    return _run_trailing_update(client, dry_run, enabled)
