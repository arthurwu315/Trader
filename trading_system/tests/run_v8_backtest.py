"""
V8 Backtest: A-only + Volatility Regime
- STRAT_A (1D macro) unchanged
- vol = BTC 1d ATR(20)/Close * 100
- V8.1: LOW=1.0, MID=1.0, HIGH=0.7 (only de-risk in high vol)
- V8.2: Revert sizing to 1.0; add high-vol conditional tighter stop (0.7x ATR trail)
- V8.3: LOW mult=0.6 to suppress low-vol exposure; MID/HIGH=1.0; trailing stop baseline
"""
from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("SKIP_CONFIG_VALIDATION", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))

from backtest_utils import fetch_klines_df

INITIAL_EQUITY = 10000.0
NOTIONAL_PCT = 0.40
MAX_CONCURRENT = 2
COST_RT_1D_PCT = 0.14
START = "2022-01-01"
END = "2024-12-31"
C_GROUP = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "OPNUSDT",
    "AZTECUSDT", "DOGEUSDT", "1000PEPEUSDT", "ENSOUSDT", "BNBUSDT",
    "ESPUSDT", "INJUSDT", "ZECUSDT", "BCHUSDT", "SIRENUSDT",
    "YGGUSDT", "POWERUSDT", "KITEUSDT", "ETCUSDT", "PIPPINUSDT",
]

VOL_LOW = 2.0
VOL_HIGH = 4.0
MULT_LOW = 1.0
MULT_MID = 1.0
MULT_HIGH = 1.0  # V8.2: revert sizing to 1.0
HIGH_VOL_STOP_FACTOR = 0.7  # V8.2: trail_mult * 0.7 when vol >= 4%


def to_utc_ts(x) -> pd.Timestamp:
    ts = pd.Timestamp(x)
    return ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")


def resample_ohlcv(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out = out.set_index("timestamp").sort_index()
    rs = (
        out.resample(interval, closed="right", label="right")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna(subset=["open", "high", "low", "close"])
        .reset_index()
    )
    return rs


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    c = out["close"].astype(float)
    h = out["high"].astype(float)
    l = out["low"].astype(float)
    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    out["atr_14"] = tr.rolling(14).mean()
    out["atr_20"] = tr.rolling(20).mean()
    out["ema_20"] = c.ewm(span=20, adjust=False).mean()
    out["ema_50"] = c.ewm(span=50, adjust=False).mean()
    out["ema_200"] = c.ewm(span=200, adjust=False).mean()
    out["roll_high_80"] = h.shift(1).rolling(80).max()
    out["roll_low_80"] = l.shift(1).rolling(80).min()
    out["roc_20"] = c.pct_change(20)
    return out


def build_btc_vol_mult_map(data_map: dict, mult_low=None, mult_mid=None, mult_high=None) -> dict:
    ml = mult_low if mult_low is not None else MULT_LOW
    mm = mult_mid if mult_mid is not None else MULT_MID
    mh = mult_high if mult_high is not None else MULT_HIGH
    btc = data_map["BTCUSDT"]["1d"]
    atr20 = btc["atr_20"].values
    close = btc["close"].astype(float).values
    ts_arr = btc["timestamp"].values
    m = {}
    for i in range(len(btc)):
        if close[i] <= 0 or not np.isfinite(atr20[i]):
            continue
        vol_pct = atr20[i] / close[i] * 100.0
        if vol_pct < VOL_LOW:
            mult = ml
        elif vol_pct < VOL_HIGH:
            mult = mm
        else:
            mult = mh
        t = to_utc_ts(ts_arr[i])
        m[t.strftime("%Y-%m-%d")] = mult
    return m


def build_btc_low_vol_set(data_map: dict) -> set:
    """Set of 'YYYY-MM-DD' date strings where BTC vol < 2% (LOW_VOL regime)."""
    btc = data_map["BTCUSDT"]["1d"]
    atr20 = btc["atr_20"].values
    close = btc["close"].astype(float).values
    ts_arr = btc["timestamp"].values
    s = set()
    for i in range(len(btc)):
        if close[i] <= 0 or not np.isfinite(atr20[i]):
            continue
        vol_pct = atr20[i] / close[i] * 100.0
        if vol_pct < VOL_LOW:
            t = to_utc_ts(ts_arr[i])
            s.add(t.strftime("%Y-%m-%d"))
    return s


def build_btc_high_vol_set(data_map: dict) -> set:
    """Set of 'YYYY-MM-DD' date strings where BTC vol >= 4% (HIGH_VOL regime)."""
    btc = data_map["BTCUSDT"]["1d"]
    atr20 = btc["atr_20"].values
    close = btc["close"].astype(float).values
    ts_arr = btc["timestamp"].values
    s = set()
    for i in range(len(btc)):
        if close[i] <= 0 or not np.isfinite(atr20[i]):
            continue
        vol_pct = atr20[i] / close[i] * 100.0
        if vol_pct >= VOL_HIGH:
            t = to_utc_ts(ts_arr[i])
            s.add(t.strftime("%Y-%m-%d"))
    return s


def _simulate_position_exit(df, side, entry_idx, entry_price, sl_price, trail_mult, tp_price, max_hold_bars):
    current_sl = sl_price
    end_idx = min(len(df) - 1, entry_idx + max_hold_bars)
    for i in range(entry_idx + 1, end_idx + 1):
        row = df.iloc[i]
        atr = float(row.get("atr_14", np.nan))
        if not np.isfinite(atr) or atr <= 0:
            atr = entry_price * 0.015
        high, low = float(row["high"]), float(row["low"])
        if trail_mult is not None and trail_mult > 0:
            if side == "BUY":
                current_sl = max(current_sl, high - trail_mult * atr)
            else:
                current_sl = min(current_sl, low + trail_mult * atr)
        if side == "BUY":
            if low <= current_sl:
                return i, current_sl
            if tp_price is not None and high >= tp_price:
                return i, tp_price
        else:
            if high >= current_sl:
                return i, current_sl
            if tp_price is not None and low <= tp_price:
                return i, tp_price
    return end_idx, float(df.iloc[end_idx]["close"])


def run_strat_a_with_sizing(data_map, btc_regime, vol_mult_map=None, high_vol_stop_set=None, high_vol_regime_set=None):
    def sig(row):
        c = float(row["close"])
        e50 = float(row.get("ema_50", np.nan))
        e200 = float(row.get("ema_200", np.nan))
        rh = float(row.get("roll_high_80", np.nan))
        rl = float(row.get("roll_low_80", np.nan))
        atr = float(row.get("atr_14", np.nan))
        if not np.isfinite([c, e50, e200, rh, rl, atr]).all():
            return None, 0, 0, None, 0
        if (float(row["high"]) - float(row["low"])) <= atr:
            return None, 0, 0, None, 0
        roc = float(row.get("roc_20", 0.0) or 0.0)
        if c > e50 > e200 and c > rh:
            return "BUY", 2.5, 2.5, None, roc
        if c < e50 < e200 and c < rl:
            return "SELL", 2.5, 2.5, None, roc
        return None, 0, 0, None, 0

    trades = []
    for s in C_GROUP:
        if s not in data_map:
            continue
        df = data_map[s]["1d"]
        if df.empty:
            continue
        i = 220
        while i < len(df) - 2:
            row = df.iloc[i]
            side, sl_mult, trail_mult, tp_mult, score = sig(row)
            if side is None:
                i += 1
                continue
            ts = to_utc_ts(row["timestamp"]).floor("1D")
            ts_key = ts.strftime("%Y-%m-%d")
            rg = btc_regime.get(ts, "unknown")
            mult = vol_mult_map.get(ts_key, 1.0) if vol_mult_map is not None else 1.0
            if (rg == "bull" and side == "SELL") or (rg == "bear" and side == "BUY"):
                i += 1
                continue
            mult = min(mult, 1.5)
            is_high_vol = (
                (high_vol_regime_set is not None and ts_key in high_vol_regime_set)
                if high_vol_regime_set is not None
                else (vol_mult_map is not None and mult < 1.0)  # V8.1-style
            )
            effective_trail = trail_mult * HIGH_VOL_STOP_FACTOR if (
                high_vol_stop_set is not None and ts_key in high_vol_stop_set
            ) else trail_mult
            entry = float(row["close"])
            atr = float(row.get("atr_14", np.nan))
            if not np.isfinite(atr) or atr <= 0:
                i += 1
                continue
            sl = entry - sl_mult * atr if side == "BUY" else entry + sl_mult * atr
            tp = entry + 2.5 * atr if side == "BUY" else entry - 2.5 * atr
            exit_idx, exit_px = _simulate_position_exit(df, side, i, entry, sl, effective_trail, tp, 30)
            raw_ret = (exit_px - entry) / entry * 100.0 if side == "BUY" else (entry - exit_px) / entry * 100.0
            net_ret = raw_ret - COST_RT_1D_PCT
            pnl = INITIAL_EQUITY * NOTIONAL_PCT * mult * (net_ret / 100.0)
            trades.append({
                "strategy": "STRAT_A",
                "symbol": s,
                "side": side,
                "entry_time": to_utc_ts(row["timestamp"]),
                "exit_time": to_utc_ts(df.iloc[exit_idx]["timestamp"]),
                "score": score,
                "ret_net_pct": net_ret,
                "pnl_usdt": pnl,
                "is_high_vol": is_high_vol if vol_mult_map is not None else False,
            })
            i = exit_idx + 1

    by_bucket = {}
    for t in sorted(trades, key=lambda x: x["entry_time"]):
        key = pd.Timestamp(t["entry_time"]).floor("1D")
        by_bucket.setdefault(key, []).append(t)
    rs = []
    for _, batch in sorted(by_bucket.items(), key=lambda kv: kv[0]):
        longs = [x for x in batch if x["side"] == "BUY"]
        shorts = [x for x in batch if x["side"] == "SELL"]
        if longs:
            rs.append(max(longs, key=lambda x: x["score"]))
        if shorts:
            rs.append(min(shorts, key=lambda x: x["score"]))
    accepted = []
    active = []
    for t in sorted(rs, key=lambda x: x["entry_time"]):
        active = [a for a in active if a["exit_time"] > t["entry_time"]]
        if len(active) >= MAX_CONCURRENT:
            continue
        accepted.append(t)
        active.append(t)
    return accepted


def compute_metrics(trades, days_override=None, high_vol_regime_set=None, low_vol_regime_set=None):
    empty = {"CAGR": 0, "MDD": 0, "Calmar": 0, "PF": 0, "Win%": 0, "Trades": 0, "Exposure": 0,
             "HighVolExposure%": 0, "HighVol MDD": 0, "non-HighVol MDD": 0,
             "LowVolExposure%": 0, "LowVol MDD": 0, "non-LowVol MDD": 0}
    if not trades:
        return empty
    eq = INITIAL_EQUITY
    curve = [eq]
    gp = gl = 0.0
    for t in sorted(trades, key=lambda x: x["exit_time"]):
        eq += t["pnl_usdt"]
        curve.append(eq)
        if t["pnl_usdt"] > 0:
            gp += t["pnl_usdt"]
        else:
            gl += abs(t["pnl_usdt"])
    peak = curve[0]
    max_dd_pct = 0.0
    for v in curve:
        if v > peak:
            peak = v
        dd = (peak - v) / peak * 100.0 if peak > 0 else 0.0
        max_dd_pct = max(max_dd_pct, dd)
    total_ret = (curve[-1] - INITIAL_EQUITY) / INITIAL_EQUITY * 100.0
    if days_override is not None:
        days = days_override
    else:
        days = (pd.Timestamp(trades[-1]["exit_time"]) - pd.Timestamp(trades[0]["entry_time"])).days or 1
    cagr = ((1 + total_ret / 100.0) ** (365.0 / max(days, 1)) - 1.0) * 100.0 if total_ret > -100 else -100.0
    calmar = cagr / max_dd_pct if max_dd_pct > 0 else 0.0
    pf = gp / gl if gl > 0 else (999.0 if gp > 0 else 0.0)
    win_pct = sum(1 for t in trades if t["pnl_usdt"] > 0) / len(trades) * 100.0
    expo_h = sum((t["exit_time"] - t["entry_time"]).total_seconds() / 3600.0 for t in trades)
    total_h = max(days, 1) * 24.0
    expo = expo_h / total_h * 100.0 if total_h > 0 else 0.0
    def _is_high_vol(t):
        if high_vol_regime_set is not None:
            d = pd.Timestamp(t["entry_time"]).strftime("%Y-%m-%d")
            return d in high_vol_regime_set
        return t.get("is_high_vol", False)

    high_vol_expo_h = sum(
        (t["exit_time"] - t["entry_time"]).total_seconds() / 3600.0
        for t in trades if _is_high_vol(t)
    )
    high_vol_expo_pct = (high_vol_expo_h / expo_h * 100.0) if expo_h > 0 else 0.0

    # HighVol / non-HighVol MDD: max drawdown of sub-curves (pnl from high-vol only / non-high-vol only)
    def _mdd_from_curve(curve):
        peak = curve[0]
        mdd = 0.0
        for v in curve:
            if v > peak:
                peak = v
            dd = (peak - v) / peak * 100.0 if peak > 0 else 0.0
            mdd = max(mdd, dd)
        return mdd

    eq_hv = INITIAL_EQUITY
    curve_hv = [eq_hv]
    eq_nhv = INITIAL_EQUITY
    curve_nhv = [eq_nhv]
    for t in sorted(trades, key=lambda x: x["exit_time"]):
        if _is_high_vol(t):
            eq_hv += t["pnl_usdt"]
        else:
            eq_nhv += t["pnl_usdt"]
        curve_hv.append(eq_hv)
        curve_nhv.append(eq_nhv)
    high_vol_mdd = _mdd_from_curve(curve_hv)
    non_high_vol_mdd = _mdd_from_curve(curve_nhv)

    def _is_low_vol(t):
        if low_vol_regime_set is not None:
            d = pd.Timestamp(t["entry_time"]).strftime("%Y-%m-%d")
            return d in low_vol_regime_set
        return False

    low_vol_expo_h = sum(
        (t["exit_time"] - t["entry_time"]).total_seconds() / 3600.0
        for t in trades if _is_low_vol(t)
    )
    low_vol_expo_pct = (low_vol_expo_h / expo_h * 100.0) if expo_h > 0 else 0.0

    eq_lv = INITIAL_EQUITY
    curve_lv = [eq_lv]
    eq_nlv = INITIAL_EQUITY
    curve_nlv = [eq_nlv]
    for t in sorted(trades, key=lambda x: x["exit_time"]):
        if _is_low_vol(t):
            eq_lv += t["pnl_usdt"]
        else:
            eq_nlv += t["pnl_usdt"]
        curve_lv.append(eq_lv)
        curve_nlv.append(eq_nlv)
    low_vol_mdd = _mdd_from_curve(curve_lv)
    non_low_vol_mdd = _mdd_from_curve(curve_nlv)

    return {
        "CAGR": cagr, "MDD": max_dd_pct, "Calmar": calmar, "PF": pf,
        "Win%": win_pct, "Trades": len(trades), "Exposure": expo,
        "HighVolExposure%": high_vol_expo_pct,
        "HighVol MDD": high_vol_mdd,
        "non-HighVol MDD": non_high_vol_mdd,
        "LowVolExposure%": low_vol_expo_pct,
        "LowVol MDD": low_vol_mdd,
        "non-LowVol MDD": non_low_vol_mdd,
    }


def filter_trades_by_year(trades, year):
    return [t for t in trades if pd.Timestamp(t["entry_time"]).year == year]


async def main():
    from bots.bot_c.config_c import get_strategy_c_config
    from core.binance_client import BinanceFuturesClient

    cfg = get_strategy_c_config()
    client = BinanceFuturesClient(
        api_key=cfg.binance_api_key or "dummy",
        api_secret=cfg.binance_api_secret or "dummy",
        base_url=os.getenv("BINANCE_DATA_URL", "https://fapi.binance.com"),
    )
    start_dt = datetime.strptime(START, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(END, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    def _fetch(sym):
        return sym, fetch_klines_df(client, sym, "1h", start_dt, end_dt)

    data_map = {}
    for sym in C_GROUP:
        _, h1 = await asyncio.to_thread(_fetch, sym)
        if h1.empty:
            continue
        h1 = add_indicators(h1)
        d1 = add_indicators(resample_ohlcv(h1, "1D"))
        data_map[sym] = {"1h": h1, "1d": d1}

    if "BTCUSDT" not in data_map:
        raise SystemExit("BTC data missing")

    btc_regime = {}
    btc = data_map["BTCUSDT"]["1d"]
    for _, r in btc.iterrows():
        ts = to_utc_ts(r["timestamp"]).floor("1D")
        close = float(r.get("close", 0) or 0)
        ema200 = float(r.get("ema_200", 0) or 0)
        if close > 0 and ema200 > 0:
            btc_regime[ts] = "bull" if close > ema200 else "bear"

    low_vol_set = build_btc_low_vol_set(data_map)
    vol_mult_v83 = build_btc_vol_mult_map(data_map, mult_low=0.6, mult_mid=1.0, mult_high=1.0)

    base_trades = run_strat_a_with_sizing(
        data_map, btc_regime, vol_mult_map=None,
        high_vol_stop_set=None, high_vol_regime_set=None,
    )
    v83_trades = run_strat_a_with_sizing(
        data_map, btc_regime, vol_mult_map=vol_mult_v83,
        high_vol_stop_set=None, high_vol_regime_set=None,
    )

    print("| Period | Version | CAGR | MDD | Calmar | PF | Win% | Trades | Exposure |")
    print("|--------|---------|------|-----|--------|-----|------|--------|----------|")
    def _m(trades, do=None):
        return compute_metrics(trades, days_override=do, low_vol_regime_set=low_vol_set)

    for name, trades in [("Baseline", base_trades), ("V8.3", v83_trades)]:
        m = _m(trades)
        print(f"| Full | {name} | {m['CAGR']:.4f} | {m['MDD']:.4f} | {m['Calmar']:.4f} | {m['PF']:.4f} | {m['Win%']:.4f} | {m['Trades']} | {m['Exposure']:.4f} |")
    days_per_year = {2022: 365, 2023: 365, 2024: 366}
    for yr in (2022, 2023, 2024):
        for name, trades in [("Baseline", base_trades), ("V8.3", v83_trades)]:
            ty = filter_trades_by_year(trades, yr)
            m = _m(ty, do=days_per_year.get(yr, 365))
            print(f"| {yr} | {name} | {m['CAGR']:.4f} | {m['MDD']:.4f} | {m['Calmar']:.4f} | {m['PF']:.4f} | {m['Win%']:.4f} | {m['Trades']} | {m['Exposure']:.4f} |")

    n_lv = sum(1 for t in base_trades if pd.Timestamp(t["entry_time"]).strftime("%Y-%m-%d") in low_vol_set)
    print(f"\nLow-vol regime days: {len(low_vol_set)} | Baseline low-vol trades: {n_lv}")
    for name, trades in [("Baseline", base_trades), ("V8.3", v83_trades)]:
        m = _m(trades)
        print(f"{name} Full: LowVol Exposure% = {m.get('LowVolExposure%', 0):.2f}% | LowVol MDD = {m.get('LowVol MDD', 0):.4f}% | non-LowVol MDD = {m.get('non-LowVol MDD', 0):.4f}%")


if __name__ == "__main__":
    asyncio.run(main())
