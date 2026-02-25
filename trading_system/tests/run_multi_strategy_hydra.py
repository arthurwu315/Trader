"""
Multi-Strategy Hydra backtest (OOS 2024-01 -> now).

Strategies:
- STRAT_A: 1D macro trend, C-group, BTC EMA200 dual-gate
- STRAT_B: 4H breakout momentum (N=20)
- STRAT_C: 1H mean reversion (BB + RSI)
- STRAT_D: 4H carry-proxy contrarian (return z-score reversal)
- STRAT_E: 1D cross-sectional momentum rotation
- STRAT_F: 4H volatility regime switch (breakout vs mean-reversion)

Outputs:
- Standalone performance table
- Blended performance table with 20% allocation grid
- Daily PnL correlation matrix
"""
from __future__ import annotations

import asyncio
import gc
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

os.environ.setdefault("SKIP_CONFIG_VALIDATION", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))

from backtest_utils import fetch_klines_df

INITIAL_EQUITY = 9568.0
NOTIONAL_PCT = 0.40
MAX_CONCURRENT = 2

START = "2024-01-01"
END = datetime.now(timezone.utc).strftime("%Y-%m-%d")
INTERVAL_1H = "1h"

# Cost model (round-trip): short-term is deliberately stricter than 1D.
COST_RT_1D_PCT = 0.14   # 0.09% fee + 0.05% slippage
COST_RT_4H_PCT = 0.24   # short-term taker: deliberately strict
COST_RT_1H_MAKER_PCT = 0.05  # mean-reversion with passive fill assumption

C_GROUP = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "OPNUSDT",
    "AZTECUSDT", "DOGEUSDT", "1000PEPEUSDT", "ENSOUSDT", "BNBUSDT",
    "ESPUSDT", "INJUSDT", "ZECUSDT", "BCHUSDT", "SIRENUSDT",
    "YGGUSDT", "POWERUSDT", "KITEUSDT", "ETCUSDT", "PIPPINUSDT",
]


@dataclass
class Trade:
    strategy: str
    symbol: str
    side: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    score: float
    ret_net_pct: float
    pnl_usdt: float
    size_mult: float = 1.0


def build_client():
    from bots.bot_c.config_c import get_strategy_c_config
    from core.binance_client import BinanceFuturesClient

    cfg = get_strategy_c_config()
    return BinanceFuturesClient(
        api_key=cfg.binance_api_key or "dummy",
        api_secret=cfg.binance_api_secret or "dummy",
        base_url=os.getenv("BINANCE_DATA_URL", "https://fapi.binance.com"),
    )


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
    c = out["close"]
    h = out["high"]
    l = out["low"]
    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    out["atr_14"] = tr.rolling(14).mean()
    out["atr_14_ma20"] = out["atr_14"].rolling(20).mean()
    out["vol_sma20"] = out["volume"].rolling(20).mean()
    out["vol_sma50"] = out["volume"].rolling(50).mean()
    out["ema_20"] = c.ewm(span=20, adjust=False).mean()
    out["ema_50"] = c.ewm(span=50, adjust=False).mean()
    out["ema_100"] = c.ewm(span=100, adjust=False).mean()
    out["ema_200"] = c.ewm(span=200, adjust=False).mean()
    out["roll_high_20"] = h.shift(1).rolling(20).max()
    out["roll_low_20"] = l.shift(1).rolling(20).min()
    out["roll_high_80"] = h.shift(1).rolling(80).max()
    out["roll_low_80"] = l.shift(1).rolling(80).min()
    out["roc_20"] = c.pct_change(20)
    out["ret_8"] = c.pct_change(8)
    out["ret_8_z"] = (out["ret_8"] - out["ret_8"].rolling(120).mean()) / out["ret_8"].rolling(120).std()
    out["atr_pct"] = out["atr_14"] / c
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    out["bb_mid"] = bb_mid
    out["bb_up"] = bb_mid + 2.0 * bb_std
    out["bb_low"] = bb_mid - 2.0 * bb_std
    out["bb_up_2_5"] = bb_mid + 2.5 * bb_std
    out["bb_low_2_5"] = bb_mid - 2.5 * bb_std
    rng = (h - l).replace(0, np.nan)
    out["lower_reclaim_ratio"] = (c - l) / rng
    out["upper_reject_ratio"] = (h - c) / rng
    delta = c.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = (-delta.clip(upper=0)).rolling(14).mean()
    rs = up / down.replace(0, np.nan)
    out["rsi_14"] = 100 - (100 / (1 + rs))
    return out


def _simulate_position_exit(
    df: pd.DataFrame,
    side: str,
    entry_idx: int,
    entry_price: float,
    sl_price: float,
    trail_atr_mult: float | None,
    tp_price: float | None,
    max_hold_bars: int,
    micro_stop_bars: int | None = None,
) -> tuple[int, float]:
    current_sl = sl_price
    end_idx = min(len(df) - 1, entry_idx + max_hold_bars)
    for i in range(entry_idx + 1, end_idx + 1):
        row = df.iloc[i]
        atr = float(row.get("atr_14", np.nan))
        if not np.isfinite(atr) or atr <= 0:
            atr = entry_price * 0.015
        high = float(row["high"])
        low = float(row["low"])
        if trail_atr_mult is not None and trail_atr_mult > 0:
            if side == "BUY":
                current_sl = max(current_sl, high - trail_atr_mult * atr)
            else:
                current_sl = min(current_sl, low + trail_atr_mult * atr)

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

        # Micro defense: if trade does not move into profit quickly, cut it.
        if micro_stop_bars is not None and (i - entry_idx) >= micro_stop_bars:
            close = float(row["close"])
            if side == "BUY" and close <= entry_price:
                return i, close
            if side == "SELL" and close >= entry_price:
                return i, close

    last = df.iloc[end_idx]
    return end_idx, float(last["close"])


def _run_symbol_strategy(
    strategy_name: str,
    symbol: str,
    df: pd.DataFrame,
    signal_fn: Callable[[pd.Series], tuple[str | None, float, float, float | None, float]],
    cost_rt_pct: float,
    max_hold_bars: int,
    btc_regime_map: dict[pd.Timestamp, str] | None = None,
    dual_gate: bool = False,
    funding_filter: Callable[[pd.Timestamp, str, str], bool] | None = None,
    micro_stop_bars: int | None = None,
) -> list[Trade]:
    if df.empty:
        return []
    trades: list[Trade] = []
    i = 220
    while i < len(df) - 2:
        row = df.iloc[i]
        side, sl_mult, trail_mult, tp_mult, score = signal_fn(row)
        if side is None:
            i += 1
            continue
        ts = to_utc_ts(row["timestamp"]).floor("1D")
        if dual_gate and btc_regime_map is not None:
            rg = btc_regime_map.get(ts, "unknown")
            if (rg == "bull" and side == "SELL") or (rg == "bear" and side == "BUY"):
                i += 1
                continue
        if funding_filter is not None and not funding_filter(to_utc_ts(row["timestamp"]), side, symbol):
            i += 1
            continue

        entry = float(row["close"])
        atr = float(row.get("atr_14", np.nan))
        if not np.isfinite(atr) or atr <= 0:
            i += 1
            continue
        sl = entry - sl_mult * atr if side == "BUY" else entry + sl_mult * atr
        tp = None
        if tp_mult is not None:
            tp = entry + tp_mult * atr if side == "BUY" else entry - tp_mult * atr
        exit_idx, exit_px = _simulate_position_exit(
            df, side, i, entry, sl, trail_mult, tp, max_hold_bars=max_hold_bars, micro_stop_bars=micro_stop_bars
        )
        raw_ret = (exit_px - entry) / entry * 100.0 if side == "BUY" else (entry - exit_px) / entry * 100.0
        net_ret = raw_ret - cost_rt_pct
        pnl = INITIAL_EQUITY * NOTIONAL_PCT * (net_ret / 100.0)
        trades.append(
            Trade(
                strategy=strategy_name,
                symbol=symbol,
                side=side,
                entry_time=to_utc_ts(row["timestamp"]),
                exit_time=to_utc_ts(df.iloc[exit_idx]["timestamp"]),
                score=float(score),
                ret_net_pct=float(net_ret),
                pnl_usdt=float(pnl),
            )
        )
        i = exit_idx + 1
    return trades


def apply_portfolio_limits(trades: list[Trade], bucket: str = "1D") -> list[Trade]:
    if not trades:
        return []
    by_bucket: dict[pd.Timestamp, list[Trade]] = {}
    for t in sorted(trades, key=lambda x: x.entry_time):
        key = pd.Timestamp(t.entry_time).floor(bucket)
        by_bucket.setdefault(key, []).append(t)

    rs_selected: list[Trade] = []
    for _, batch in sorted(by_bucket.items(), key=lambda kv: kv[0]):
        longs = [x for x in batch if x.side == "BUY"]
        shorts = [x for x in batch if x.side == "SELL"]
        if longs:
            rs_selected.append(max(longs, key=lambda x: x.score))
        if shorts:
            rs_selected.append(min(shorts, key=lambda x: x.score))

    accepted: list[Trade] = []
    active: list[Trade] = []
    for t in sorted(rs_selected, key=lambda x: x.entry_time):
        active = [a for a in active if a.exit_time > t.entry_time]
        if len(active) >= MAX_CONCURRENT:
            continue
        accepted.append(t)
        active.append(t)
    return accepted


def strategy_metrics(trades: list[Trade]) -> dict[str, float]:
    if not trades:
        return {"expectancy": 0.0, "pf": 0.0, "maxdd": 0.0, "trades": 0.0, "net": 0.0, "recovery": 0.0}
    eq = INITIAL_EQUITY
    curve = [eq]
    gp = 0.0
    gl = 0.0
    rets = []
    for t in sorted(trades, key=lambda x: x.exit_time):
        eq += t.pnl_usdt
        curve.append(eq)
        rets.append(t.ret_net_pct)
        if t.pnl_usdt > 0:
            gp += t.pnl_usdt
        elif t.pnl_usdt < 0:
            gl += abs(t.pnl_usdt)
    peak = curve[0]
    max_dd_usdt = 0.0
    max_dd_pct = 0.0
    for v in curve:
        if v > peak:
            peak = v
        dd_u = peak - v
        dd_p = (dd_u / peak * 100.0) if peak > 0 else 0.0
        max_dd_usdt = max(max_dd_usdt, dd_u)
        max_dd_pct = max(max_dd_pct, dd_p)
    pf = (gp / gl) if gl > 0 else (999.0 if gp > 0 else 0.0)
    rec = ((eq - INITIAL_EQUITY) / max_dd_usdt) if max_dd_usdt > 0 else 0.0
    return {
        "expectancy": float(sum(rets) / len(rets)),
        "pf": float(pf),
        "maxdd": float(max_dd_pct),
        "trades": float(len(trades)),
        "net": float(eq - INITIAL_EQUITY),
        "recovery": float(rec),
    }


def daily_pnl_series(trades: list[Trade]) -> pd.Series:
    if not trades:
        return pd.Series(dtype=float)
    rows = {}
    for t in trades:
        d = pd.Timestamp(t.exit_time).floor("1D")
        rows[d] = rows.get(d, 0.0) + t.pnl_usdt
    s = pd.Series(rows).sort_index()
    s.index = pd.to_datetime(s.index, utc=True)
    return s


def annualized_return(eq_curve: pd.Series) -> float:
    if eq_curve.empty:
        return 0.0
    start = float(eq_curve.iloc[0])
    end = float(eq_curve.iloc[-1])
    years = max((eq_curve.index[-1] - eq_curve.index[0]).days / 365.25, 1e-9)
    if start <= 0 or end <= 0:
        return -100.0
    return ((end / start) ** (1.0 / years) - 1.0) * 100.0


def max_drawdown_pct(eq_curve: pd.Series) -> float:
    if eq_curve.empty:
        return 0.0
    peak = eq_curve.cummax()
    dd = (peak - eq_curve) / peak.replace(0, np.nan) * 100.0
    return float(dd.max(skipna=True) or 0.0)


def alloc_grid(step: float = 0.2, n_strats: int = 6) -> list[tuple[float, ...]]:
    levels = [round(i * step, 10) for i in range(int(1 / step) + 1)]
    out = []
    for w in product(levels, repeat=n_strats):
        if abs(sum(w) - 1.0) < 1e-9:
            out.append(w)
    return out


async def load_data() -> dict[str, dict[str, pd.DataFrame]]:
    client = build_client()
    start_dt = datetime.strptime(START, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(END, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    def _fetch(sym: str) -> tuple[str, pd.DataFrame]:
        return sym, fetch_klines_df(client, sym, INTERVAL_1H, start_dt, end_dt)

    tasks = [asyncio.to_thread(_fetch, s) for s in C_GROUP]
    out: dict[str, dict[str, pd.DataFrame]] = {}
    for sym, h1 in await asyncio.gather(*tasks):
        h1 = add_indicators(h1)
        h4 = add_indicators(resample_ohlcv(h1, "4h"))
        d1 = add_indicators(resample_ohlcv(h1, "1D"))
        out[sym] = {"1h": h1, "4h": h4, "1d": d1}
    return out


async def load_funding_history_maps() -> dict[str, pd.DataFrame]:
    """
    Load recent funding history per symbol for entry-time filter.
    Uses bounded pagination for runtime control.
    """
    client = build_client()
    start_ms = int(datetime.strptime(START, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)

    def _fetch_symbol(sym: str) -> tuple[str, pd.DataFrame]:
        rows = []
        end_time = None
        pages = 0
        while pages < 6:
            params = {"symbol": sym, "limit": 1000}
            if end_time is not None:
                params["endTime"] = end_time
            try:
                data = client._call_with_retry("GET", "/fapi/v1/fundingRate", params)
            except Exception:
                break
            if not isinstance(data, list) or not data:
                break
            rows.extend(data)
            pages += 1
            first_t = int(data[0].get("fundingTime", 0) or 0)
            if first_t <= start_ms:
                break
            end_time = first_t - 1
        if not rows:
            return sym, pd.DataFrame(columns=["funding_time", "funding_rate"])
        df = pd.DataFrame(rows)
        df["funding_time"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
        df["funding_rate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
        df = df.dropna(subset=["funding_rate"]).sort_values("funding_time").drop_duplicates(subset=["funding_time"])
        return sym, df[["funding_time", "funding_rate"]].reset_index(drop=True)

    tasks = [asyncio.to_thread(_fetch_symbol, s) for s in C_GROUP]
    out = {}
    for sym, df in await asyncio.gather(*tasks):
        out[sym] = df
    return out


def build_funding_filter(funding_map: dict[str, pd.DataFrame], max_unfavorable_rate: float = 0.0004):
    def _allow(ts: pd.Timestamp, side: str, symbol: str) -> bool:
        df = funding_map.get(symbol)
        if df is None or df.empty:
            return True
        hist = df[df["funding_time"] <= ts]
        if hist.empty:
            return True
        fr = float(hist.iloc[-1]["funding_rate"])
        if side == "BUY" and fr > max_unfavorable_rate:
            return False
        if side == "SELL" and fr < -max_unfavorable_rate:
            return False
        return True
    return _allow


def latest_funding_rate(funding_map: dict[str, pd.DataFrame], symbol: str, ts: pd.Timestamp) -> float:
    df = funding_map.get(symbol)
    if df is None or df.empty:
        return 0.0
    hist = df[df["funding_time"] <= ts]
    if hist.empty:
        return 0.0
    return float(hist.iloc[-1]["funding_rate"])


def build_btc_regime_map(data_map: dict[str, dict[str, pd.DataFrame]]) -> dict[pd.Timestamp, str]:
    btc = data_map["BTCUSDT"]["1d"]
    m = {}
    for _, r in btc.iterrows():
        ts = to_utc_ts(r["timestamp"]).floor("1D")
        close = float(r.get("close", 0) or 0)
        ema200 = float(r.get("ema_200", 0) or 0)
        if close > 0 and ema200 > 0:
            m[ts] = "bull" if close > ema200 else "bear"
    return m


def run_strat_a(data_map, btc_regime):
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
        trades.extend(
            _run_symbol_strategy("STRAT_A", s, data_map[s]["1d"], sig, COST_RT_1D_PCT, max_hold_bars=30, btc_regime_map=btc_regime, dual_gate=True)
        )
    return apply_portfolio_limits(trades, bucket="1D")


def run_strat_b(data_map, btc_regime, funding_filter):
    def sig(row):
        c = float(row["close"])
        rh = float(row.get("roll_high_20", np.nan))
        rl = float(row.get("roll_low_20", np.nan))
        mid = (rh + rl) / 2.0 if np.isfinite([rh, rl]).all() else np.nan
        e20 = float(row.get("ema_20", np.nan))
        e100 = float(row.get("ema_100", np.nan))
        atrp = float(row.get("atr_pct", np.nan))
        vol = float(row.get("volume", np.nan))
        vol_sma20 = float(row.get("vol_sma20", np.nan))
        atr = float(row.get("atr_14", np.nan))
        atr_ma20 = float(row.get("atr_14_ma20", np.nan))
        if not np.isfinite([c, rh, rl, e20, e100, atrp, atr, atr_ma20, mid, vol, vol_sma20]).all():
            return None, 0, 0, None, 0
        # Phase-2: ATR expansion filter to avoid chop breakouts.
        if atr <= atr_ma20:
            return None, 0, 0, None, 0
        # Phase-3: volume expansion confirmation.
        if vol <= (vol_sma20 * 1.5):
            return None, 0, 0, None, 0
        roc = float(row.get("roc_20", 0.0) or 0.0)
        # Phase-2: tighter initial stop + much wider trail (asymmetric R:R).
        # Approximate Donchian-mid stop by translating distance into ATR units.
        stop_mult = max(0.6, min(2.2, abs(c - mid) / max(atr, 1e-9)))
        if c > rh and e20 > e100 and atrp > 0.010:
            return "BUY", stop_mult, 4.0, None, roc
        if c < rl and e20 < e100 and atrp > 0.010:
            return "SELL", stop_mult, 4.0, None, roc
        return None, 0, 0, None, 0

    trades = []
    for s in C_GROUP:
        trades.extend(
            _run_symbol_strategy(
                "STRAT_B",
                s,
                data_map[s]["4h"],
                sig,
                COST_RT_4H_PCT,
                max_hold_bars=28,
                btc_regime_map=btc_regime,
                dual_gate=True,
                funding_filter=funding_filter,
            )
        )
    return apply_portfolio_limits(trades, bucket="1D")


def run_strat_c(data_map, funding_filter, funding_map):
    # 1) Asset clustering by ATR% median (high-vol vs low-vol).
    atr_med = {}
    for s in C_GROUP:
        atrp = data_map[s]["1h"]["atr_pct"].dropna()
        atr_med[s] = float(atrp.median()) if not atrp.empty else 0.0
    split = float(np.median(list(atr_med.values()))) if atr_med else 0.0
    high_vol = {s for s, v in atr_med.items() if v >= split}

    def make_sig(symbol: str, allowed_hours: set[int] | None):
        is_high = symbol in high_vol

        def _sig(row):
            ts = to_utc_ts(row["timestamp"])
            if allowed_hours is not None and ts.hour not in allowed_hours:
                return None, 0, 0, None, 0

            c = float(row["close"])
            rsi = float(row.get("rsi_14", np.nan))
            bbu_25 = float(row.get("bb_up_2_5", np.nan))
            bbl_25 = float(row.get("bb_low_2_5", np.nan))
            if not np.isfinite([c, rsi, bbu_25, bbl_25]).all():
                return None, 0, 0, None, 0

            if is_high:
                # High-vol symbols: stricter trigger.
                bbu = float(row.get("bb_up_2_5", np.nan))
                bbl = float(row.get("bb_low_2_5", np.nan))
                # Recompute 3.0 sigma proxy from existing bands:
                # bb_up_2_5 = mid + 2.5*std, bb_up = mid + 2*std -> std = 2*(bb_up_2_5-bb_up)
                bb_up = float(row.get("bb_up", np.nan))
                bb_low = float(row.get("bb_low", np.nan))
                mid = float(row.get("bb_mid", np.nan))
                if np.isfinite([bb_up, bb_low, mid]).all():
                    std = abs(bb_up - mid) / 2.0 if abs(bb_up - mid) > 0 else abs(mid) * 0.005
                    bbu = mid + 3.0 * std
                    bbl = mid - 3.0 * std
                if c < bbl and rsi < 10:
                    roc = float(row.get("roc_20", 0.0) or 0.0)
                    atr = float(row.get("atr_14", 0.0) or 0.0)
                    atr = max(atr, 1e-6)
                    risk_adj_score = roc * (1.0 / math.sqrt(atr))
                    return "BUY", 1.0, 0.0, 1.8, risk_adj_score
                if c > bbu and rsi > 90:
                    roc = float(row.get("roc_20", 0.0) or 0.0)
                    atr = float(row.get("atr_14", 0.0) or 0.0)
                    atr = max(atr, 1e-6)
                    risk_adj_score = roc * (1.0 / math.sqrt(atr))
                    return "SELL", 1.0, 0.0, 1.8, risk_adj_score
                return None, 0, 0, None, 0

            # Low-vol symbols: baseline extreme.
            if c < bbl_25 and rsi < 15:
                roc = float(row.get("roc_20", 0.0) or 0.0)
                atr = float(row.get("atr_14", 0.0) or 0.0)
                atr = max(atr, 1e-6)
                risk_adj_score = roc * (1.0 / math.sqrt(atr))
                return "BUY", 1.0, 0.0, 1.6, risk_adj_score
            if c > bbu_25 and rsi > 85:
                roc = float(row.get("roc_20", 0.0) or 0.0)
                atr = float(row.get("atr_14", 0.0) or 0.0)
                atr = max(atr, 1e-6)
                risk_adj_score = roc * (1.0 / math.sqrt(atr))
                return "SELL", 1.0, 0.0, 1.6, risk_adj_score
            return None, 0, 0, None, 0

        return _sig

    # 2) Pass-1: no hour filter, get entry-hour expectancy profile.
    pass1 = []
    for s in C_GROUP:
        pass1.extend(
            _run_symbol_strategy(
                "STRAT_C",
                s,
                data_map[s]["1h"],
                make_sig(s, allowed_hours=None),
                COST_RT_1H_MAKER_PCT,
                max_hold_bars=18,
                funding_filter=funding_filter,
                micro_stop_bars=3,
            )
        )

    hour_stats: dict[int, list[float]] = {}
    for t in pass1:
        h = int(to_utc_ts(t.entry_time).hour)
        hour_stats.setdefault(h, []).append(float(t.ret_net_pct))
    ranked_hours = sorted(
        [(h, len(v), float(sum(v) / len(v))) for h, v in hour_stats.items()],
        key=lambda x: (x[2], x[1]),
        reverse=True,
    )
    # Keep only positive-expectancy hours with enough samples; fallback top 8.
    allowed = {h for h, n, exp in ranked_hours if n >= 8 and exp > 0}
    if not allowed:
        allowed = {h for h, _, _ in ranked_hours[:8]}

    # 3) Pass-2: apply hour filter.
    trades = []
    for s in C_GROUP:
        trades.extend(
            _run_symbol_strategy(
                "STRAT_C",
                s,
                data_map[s]["1h"],
                make_sig(s, allowed_hours=allowed),
                COST_RT_1H_MAKER_PCT,
                max_hold_bars=18,
                funding_filter=funding_filter,
                micro_stop_bars=3,
            )
        )
    # 4) Dynamic sizing score for Phase-5 (signal-weighted notional).
    for t in trades:
        h1 = data_map[t.symbol]["1h"]
        row_df = h1[pd.to_datetime(h1["timestamp"], utc=True) == pd.to_datetime(t.entry_time, utc=True)]
        if row_df.empty:
            t.size_mult = 0.5
            continue
        row = row_df.iloc[0]
        c = float(row.get("close", np.nan))
        rsi = float(row.get("rsi_14", np.nan))
        mid = float(row.get("bb_mid", np.nan))
        bb_up = float(row.get("bb_up", np.nan))
        if not np.isfinite([c, rsi, mid, bb_up]).all():
            t.size_mult = 0.5
            continue
        std = abs(bb_up - mid) / 2.0 if abs(bb_up - mid) > 0 else abs(mid) * 0.005
        bbu3 = mid + 3.0 * std
        bbl3 = mid - 3.0 * std
        extreme = (t.side == "BUY" and rsi < 10 and c < bbl3) or (t.side == "SELL" and rsi > 90 and c > bbu3)
        fr = latest_funding_rate(funding_map, t.symbol, pd.to_datetime(t.entry_time, utc=True))
        funding_subsidy = (t.side == "BUY" and fr < -0.0002) or (t.side == "SELL" and fr > 0.0002)
        if extreme and funding_subsidy:
            t.size_mult = 1.8
        elif extreme:
            t.size_mult = 1.4
        elif funding_subsidy:
            t.size_mult = 1.2
        else:
            t.size_mult = 0.5
    accepted = apply_portfolio_limits(trades, bucket="1D")
    diag = {
        "split_atr_pct": split,
        "high_vol_symbols": sorted(high_vol),
        "allowed_hours": sorted(allowed),
        "top_hour_stats": ranked_hours[:10],
    }
    return accepted, diag


def run_strat_c_liq_proxy(data_map, funding_map):
    """
    Liquidation-proxy C strategy:
    - 1H volume spike > 3x SMA50
    - Long lower-wick reclaim / Short upper-wick rejection
    - Funding extreme confirmation
    """
    trades = []
    for symbol in C_GROUP:
        df = data_map[symbol]["1h"]

        def _sig(row):
            close = float(row.get("close", np.nan))
            open_ = float(row.get("open", np.nan))
            high = float(row.get("high", np.nan))
            low = float(row.get("low", np.nan))
            vol = float(row.get("volume", np.nan))
            vol50 = float(row.get("vol_sma50", np.nan))
            lower_reclaim = float(row.get("lower_reclaim_ratio", np.nan))
            upper_reject = float(row.get("upper_reject_ratio", np.nan))
            if not np.isfinite([close, open_, high, low, vol, vol50, lower_reclaim, upper_reject]).all():
                return None, 0, 0, None, 0
            if vol50 <= 0 or vol <= (3.0 * vol50):
                return None, 0, 0, None, 0
            ts = to_utc_ts(row["timestamp"])
            fr = latest_funding_rate(funding_map, symbol, ts)

            # Long: panic flush + reclaim + negative funding extreme.
            if lower_reclaim > 0.60 and close > open_ and fr <= -0.0004:
                score = (vol / vol50) + (abs(fr) * 10000.0) + lower_reclaim
                return "BUY", 0.9, 0.0, 2.0, score
            # Short: squeeze spike + rejection + positive funding extreme.
            if upper_reject > 0.60 and close < open_ and fr >= 0.0004:
                score = (vol / vol50) + (abs(fr) * 10000.0) + upper_reject
                return "SELL", 0.9, 0.0, 2.0, score
            return None, 0, 0, None, 0

        trades.extend(
            _run_symbol_strategy(
                "STRAT_C_LIQ",
                symbol,
                df,
                _sig,
                COST_RT_1H_MAKER_PCT,
                max_hold_bars=18,
                micro_stop_bars=3,
            )
        )

    # Dynamic sizing for liquidation intensity.
    for t in trades:
        h1 = data_map[t.symbol]["1h"]
        row_df = h1[pd.to_datetime(h1["timestamp"], utc=True) == pd.to_datetime(t.entry_time, utc=True)]
        if row_df.empty:
            t.size_mult = 1.0
            continue
        row = row_df.iloc[0]
        vol = float(row.get("volume", np.nan))
        vol50 = float(row.get("vol_sma50", np.nan))
        fr = latest_funding_rate(funding_map, t.symbol, pd.to_datetime(t.entry_time, utc=True))
        if not np.isfinite([vol, vol50]).all() or vol50 <= 0:
            t.size_mult = 1.0
            continue
        spike = vol / vol50
        if spike >= 5.0 and abs(fr) >= 0.0008:
            t.size_mult = 2.0
        elif spike >= 4.0:
            t.size_mult = 1.6
        elif spike >= 3.0:
            t.size_mult = 1.2
        else:
            t.size_mult = 1.0
    return apply_portfolio_limits(trades, bucket="1D")


def run_strat_d(data_map):
    def sig(row):
        z = float(row.get("ret_8_z", np.nan))
        if not np.isfinite(z):
            return None, 0, 0, None, 0
        score = z
        if z >= 2.0:
            return "SELL", 1.6, 0.0, 1.0, score
        if z <= -2.0:
            return "BUY", 1.6, 0.0, 1.0, score
        return None, 0, 0, None, 0

    trades = []
    for s in C_GROUP:
        trades.extend(_run_symbol_strategy("STRAT_D", s, data_map[s]["4h"], sig, COST_RT_4H_PCT, max_hold_bars=12))
    return apply_portfolio_limits(trades, bucket="1D")


def run_strat_e(data_map):
    # Daily cross-sectional momentum: pick strongest long + weakest short by ROC20, hold 5 bars.
    rows = []
    for s in C_GROUP:
        d1 = data_map[s]["1d"][["timestamp", "close", "atr_14", "roc_20"]].copy()
        d1["symbol"] = s
        rows.append(d1)
    uni = pd.concat(rows, ignore_index=True).dropna(subset=["roc_20", "atr_14"])
    trades = []
    for ts, g in uni.groupby("timestamp"):
        if len(g) < 5:
            continue
        long_row = g.loc[g["roc_20"].idxmax()]
        short_row = g.loc[g["roc_20"].idxmin()]
        for side, r in (("BUY", long_row), ("SELL", short_row)):
            sym = str(r["symbol"])
            d1 = data_map[sym]["1d"].reset_index(drop=True)
            idx_arr = np.where(pd.to_datetime(d1["timestamp"], utc=True) == pd.to_datetime(ts, utc=True))[0]
            if len(idx_arr) == 0:
                continue
            i = int(idx_arr[0])
            if i + 1 >= len(d1):
                continue
            entry = float(d1.iloc[i]["close"])
            atr = float(d1.iloc[i]["atr_14"])
            if not np.isfinite(atr) or atr <= 0:
                continue
            sl = entry - 2.0 * atr if side == "BUY" else entry + 2.0 * atr
            ex_idx, ex_px = _simulate_position_exit(d1, side, i, entry, sl, 1.5, None, max_hold_bars=5)
            raw_ret = (ex_px - entry) / entry * 100.0 if side == "BUY" else (entry - ex_px) / entry * 100.0
            net_ret = raw_ret - COST_RT_1D_PCT
            pnl = INITIAL_EQUITY * NOTIONAL_PCT * (net_ret / 100.0)
            trades.append(
                Trade(
                    strategy="STRAT_E",
                    symbol=sym,
                    side=side,
                    entry_time=to_utc_ts(d1.iloc[i]["timestamp"]),
                    exit_time=to_utc_ts(d1.iloc[ex_idx]["timestamp"]),
                    score=float(r["roc_20"]),
                    ret_net_pct=float(net_ret),
                    pnl_usdt=float(pnl),
                )
            )
    return apply_portfolio_limits(trades, bucket="1D")


def run_strat_f(data_map):
    # Vol regime switch on 4H per symbol.
    def sig(row):
        atrp = float(row.get("atr_pct", np.nan))
        c = float(row["close"])
        rh = float(row.get("roll_high_20", np.nan))
        rl = float(row.get("roll_low_20", np.nan))
        bbu = float(row.get("bb_up", np.nan))
        bbl = float(row.get("bb_low", np.nan))
        rsi = float(row.get("rsi_14", np.nan))
        if not np.isfinite([atrp, c, rh, rl, bbu, bbl, rsi]).all():
            return None, 0, 0, None, 0
        # High-vol: breakout
        if atrp >= 0.012:
            if c > rh:
                return "BUY", 1.8, 2.0, None, float(row.get("roc_20", 0.0) or 0.0)
            if c < rl:
                return "SELL", 1.8, 2.0, None, float(row.get("roc_20", 0.0) or 0.0)
            return None, 0, 0, None, 0
        # Low-vol: mean reversion
        if c < bbl and rsi < 35:
            return "BUY", 1.3, 0.0, 1.0, float(row.get("roc_20", 0.0) or 0.0)
        if c > bbu and rsi > 65:
            return "SELL", 1.3, 0.0, 1.0, float(row.get("roc_20", 0.0) or 0.0)
        return None, 0, 0, None, 0

    trades = []
    for s in C_GROUP:
        trades.extend(_run_symbol_strategy("STRAT_F", s, data_map[s]["4h"], sig, COST_RT_4H_PCT, max_hold_bars=24))
    return apply_portfolio_limits(trades, bucket="1D")


def blend_and_rank(strategy_daily_pnl: dict[str, pd.Series], metrics_map: dict[str, dict[str, float]]):
    keys = list(strategy_daily_pnl.keys())
    pnl_df = pd.concat([strategy_daily_pnl[k].rename(k) for k in keys], axis=1).fillna(0.0).sort_index()
    corr = pnl_df.corr().fillna(0.0)

    combos = alloc_grid(step=0.2, n_strats=len(keys))
    ranked = []
    for w in combos:
        w_map = {k: float(wi) for k, wi in zip(keys, w)}
        # Soft correlation penalty: skip concentration in highly-correlated sleeves.
        too_corr = False
        for i, ki in enumerate(keys):
            for j in range(i + 1, len(keys)):
                kj = keys[j]
                if corr.loc[ki, kj] > 0.60 and w_map[ki] >= 0.2 and w_map[kj] >= 0.2:
                    too_corr = True
                    break
            if too_corr:
                break
        if too_corr:
            continue

        blend_pnl = sum(pnl_df[k] * w_map[k] for k in keys)
        eq = INITIAL_EQUITY + blend_pnl.cumsum()
        if eq.empty:
            continue
        dd = max_drawdown_pct(eq)
        ann = annualized_return(eq)
        net = float(eq.iloc[-1] - INITIAL_EQUITY)
        # Approximate PF from daily PnL stream.
        gp = float(blend_pnl[blend_pnl > 0].sum())
        gl = float((-blend_pnl[blend_pnl < 0]).sum())
        pf = (gp / gl) if gl > 0 else (999.0 if gp > 0 else 0.0)
        ranked.append({"weights": w_map, "ann_return_pct": ann, "maxdd_pct": dd, "net_profit": net, "pf": pf})

    ranked.sort(key=lambda x: (x["ann_return_pct"], -x["maxdd_pct"], x["net_profit"]), reverse=True)
    return corr, ranked


def simulate_compound_equity(
    trades_a: list[Trade],
    trades_c: list[Trade],
    w_a: float,
    w_c: float,
    base_notional_a: float,
    base_notional_c: float,
) -> dict[str, float]:
    events = []
    for t in trades_a:
        events.append(("A", pd.to_datetime(t.exit_time, utc=True), float(t.ret_net_pct), float(t.size_mult)))
    for t in trades_c:
        events.append(("C", pd.to_datetime(t.exit_time, utc=True), float(t.ret_net_pct), float(t.size_mult)))
    events.sort(key=lambda x: x[1])

    eq = INITIAL_EQUITY
    curve = [eq]
    times = [pd.to_datetime(START, utc=True)]
    gp = 0.0
    gl = 0.0
    for tag, ts, ret_pct, mult in events:
        base = (w_a * base_notional_a) if tag == "A" else (w_c * base_notional_c * mult)
        notional_pct_eff = max(0.0, min(base, 3.0))
        pnl = eq * notional_pct_eff * (ret_pct / 100.0)
        eq += pnl
        curve.append(eq)
        times.append(ts)
        if pnl > 0:
            gp += pnl
        elif pnl < 0:
            gl += abs(pnl)

    s = pd.Series(curve, index=pd.to_datetime(times, utc=True)).sort_index()
    dd = max_drawdown_pct(s)
    ann = annualized_return(s)
    net = float(s.iloc[-1] - INITIAL_EQUITY)
    pf = (gp / gl) if gl > 0 else (999.0 if gp > 0 else 0.0)
    return {"ann": ann, "maxdd": dd, "net": net, "pf": pf}


async def main():
    print("Hydra backtest started")
    print(f"OOS range: {START} -> {END}")
    print(f"Universe size (C-group): {len(C_GROUP)}")
    print(
        f"Cost assumptions: 1D={COST_RT_1D_PCT:.2f}% RT, "
        f"4H={COST_RT_4H_PCT:.2f}% RT, 1H-Maker={COST_RT_1H_MAKER_PCT:.2f}% RT"
    )

    data_map = await load_data()
    funding_map = await load_funding_history_maps()
    funding_filter = build_funding_filter(funding_map, max_unfavorable_rate=0.0004)
    btc_regime = build_btc_regime_map(data_map)

    strat_c_trades, strat_c_diag = run_strat_c(data_map, funding_filter, funding_map)
    strat_trades = {
        "STRAT_A": run_strat_a(data_map, btc_regime),
        "STRAT_C": strat_c_trades,
    }
    metrics_map = {k: strategy_metrics(v) for k, v in strat_trades.items()}
    daily_map = {k: daily_pnl_series(v) for k, v in strat_trades.items()}

    print("\n=== Independent Strategy Performance (A/C only) ===")
    print("| Strategy | Expectancy% | PF | MaxDD% | Recovery | Trades | NetProfit$ |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for k in ["STRAT_A", "STRAT_C"]:
        m = metrics_map[k]
        pf_str = f"{m['pf']:.2f}" if m["pf"] < 999 else ">999"
        print(
            f"| {k} | {m['expectancy']:.2f} | {pf_str} | {m['maxdd']:.2f} | "
            f"{m['recovery']:.2f} | {int(m['trades'])} | {m['net']:+.2f} |"
        )
    print("\nSTRAT_C Phase-4 diagnostics:")
    print(f"- ATR split threshold: {strat_c_diag['split_atr_pct']:.5f}")
    print(f"- High-vol symbols: {strat_c_diag['high_vol_symbols']}")
    print(f"- Allowed UTC hours: {strat_c_diag['allowed_hours']}")
    print(f"- Top hour stats (hour, trades, expectancy%): {[(h, n, round(exp, 3)) for h, n, exp in strat_c_diag['top_hour_stats']]}")

    corr, ranked = blend_and_rank(daily_map, metrics_map)
    if not ranked:
        print("\nNo valid blended portfolio found under correlation constraints.")
        return

    best_under_dd20 = [r for r in ranked if r["maxdd_pct"] < 20.0]
    best = best_under_dd20[0] if best_under_dd20 else ranked[0]

    print("\n=== Correlation Matrix (Daily PnL) ===")
    print(corr.round(2).to_string())

    print("\n=== Blended Portfolio (A/C, Top 5 by annualized return) ===")
    print("| Rank | Weights(A/C) | PF | MaxDD% | NetProfit$ | Annualized% |")
    print("|---:|---|---:|---:|---:|---:|")
    for i, r in enumerate(ranked[:5], start=1):
        w = r["weights"]
        wtxt = f"{int(w['STRAT_A']*100)}/{int(w['STRAT_C']*100)}"
        pf_str = f"{r['pf']:.2f}" if r["pf"] < 999 else ">999"
        print(f"| {i} | {wtxt} | {pf_str} | {r['maxdd_pct']:.2f} | {r['net_profit']:+.2f} | {r['ann_return_pct']:.2f} |")

    # A+C sweep with finer step.
    ac_df = pd.concat([daily_map["STRAT_A"].rename("STRAT_A"), daily_map["STRAT_C"].rename("STRAT_C")], axis=1).fillna(0.0).sort_index()
    ac_ranked = []
    for wa, wc in alloc_grid(step=0.1, n_strats=2):
        blend = ac_df["STRAT_A"] * wa + ac_df["STRAT_C"] * wc
        eq = INITIAL_EQUITY + blend.cumsum()
        if eq.empty:
            continue
        dd = max_drawdown_pct(eq)
        ann = annualized_return(eq)
        net = float(eq.iloc[-1] - INITIAL_EQUITY)
        gp = float(blend[blend > 0].sum())
        gl = float((-blend[blend < 0]).sum())
        pf = (gp / gl) if gl > 0 else (999.0 if gp > 0 else 0.0)
        ac_ranked.append({"weights": (wa, wc), "ann": ann, "dd": dd, "net": net, "pf": pf})
    ac_ranked.sort(key=lambda x: (x["ann"], -x["dd"], x["net"]), reverse=True)
    ac_best = next((x for x in ac_ranked if x["dd"] < 20.0), ac_ranked[0] if ac_ranked else None)
    if ac_best:
        wa, wc = ac_best["weights"]
        print("\n=== A+C Blend (Phase-4 focus) ===")
        print(
            f"Best A/C under DD<20 (if exists): "
            f"{int(wa*100)}/{int(wc*100)} | "
            f"PF={ac_best['pf']:.2f} | MaxDD={ac_best['dd']:.2f}% | "
            f"Net={ac_best['net']:+.2f} | Ann={ac_best['ann']:.2f}%"
        )

    wb = best["weights"]
    print("\n=== Holy Grail Candidate ===")
    print(
        "Best under MaxDD<20%:"
        f" weights A/C = {int(wb['STRAT_A']*100)}/{int(wb['STRAT_C']*100)}"
        f" | PF={best['pf']:.2f} | MaxDD={best['maxdd_pct']:.2f}% | Net={best['net_profit']:+.2f} | Ann={best['ann_return_pct']:.2f}%"
    )
    if best["ann_return_pct"] > 0:
        years = math.log(2.0) / math.log(1.0 + best["ann_return_pct"] / 100.0)
        print(f"Estimated doubling time: {years:.2f} years")
    else:
        print("Estimated doubling time: N/A (non-positive annualized return)")

    # ===== Phase-5: Risk-budget scaling + dynamic compounding on A/C =====
    print("\n=== Phase-5: Risk Budget Scaling (Compounded) ===")
    print("| A/C Weights | C Base Notional | PF | MaxDD% | NetProfit$ | Annualized% | Doubling(Y) |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    phase5_rows = []
    for w_a, w_c in ((0.6, 0.4), (0.7, 0.3), (0.0, 1.0)):
        for c_base in (0.4, 0.6, 0.8, 1.0, 1.2, 1.4):
            r = simulate_compound_equity(
                strat_trades["STRAT_A"],
                strat_trades["STRAT_C"],
                w_a=w_a,
                w_c=w_c,
                base_notional_a=0.4,
                base_notional_c=c_base,
            )
            if r["ann"] > 0:
                dy = math.log(2.0) / math.log(1.0 + r["ann"] / 100.0)
            else:
                dy = 999.0
            phase5_rows.append(
                {
                    "weights": f"{int(w_a*100)}/{int(w_c*100)}",
                    "c_base": c_base,
                    "pf": r["pf"],
                    "dd": r["maxdd"],
                    "net": r["net"],
                    "ann": r["ann"],
                    "dy": dy,
                }
            )
            dy_str = f"{dy:.2f}" if dy < 999 else "N/A"
            pf_str = f"{r['pf']:.2f}" if r["pf"] < 999 else ">999"
            print(f"| {int(w_a*100)}/{int(w_c*100)} | {c_base:.1f} | {pf_str} | {r['maxdd']:.2f} | {r['net']:+.2f} | {r['ann']:.2f} | {dy_str} |")

    feasible = [x for x in phase5_rows if x["dd"] < 18.0]
    best_phase5 = max(feasible, key=lambda x: x["ann"]) if feasible else max(phase5_rows, key=lambda x: x["ann"])
    print(
        f"\nPhase-5 best under DD<18 (if exists): "
        f"weights={best_phase5['weights']} | C_base={best_phase5['c_base']:.1f} | "
        f"PF={best_phase5['pf']:.2f} | MaxDD={best_phase5['dd']:.2f}% | "
        f"Net={best_phase5['net']:+.2f} | Ann={best_phase5['ann']:.2f}% | "
        f"Doubling={best_phase5['dy']:.2f}Y"
    )

    # ===== Phase-6: Efficient Frontier Matrix + Liquidation Proxy =====
    print("\n=== Phase-6: Efficient Frontier Matrix (Pure C) ===")
    dd_limits = [18.0, 22.0, 25.0, 30.0]
    c_base_grid = [round(x, 1) for x in np.arange(0.4, 3.1, 0.2)]
    print("| DD Limit% | Actual MaxDD% | C Base Notional | Annualized% | Doubling(Y) |")
    print("|---:|---:|---:|---:|---:|")
    baseline_rows = []
    for dd_lim in dd_limits:
        best_row = None
        for c_base in c_base_grid:
            r = simulate_compound_equity([], strat_trades["STRAT_C"], w_a=0.0, w_c=1.0, base_notional_a=0.0, base_notional_c=c_base)
            if r["ann"] <= 0:
                continue
            dy = math.log(2.0) / math.log(1.0 + r["ann"] / 100.0)
            row = {"dd_lim": dd_lim, "dd": r["maxdd"], "c_base": c_base, "ann": r["ann"], "dy": dy}
            if r["maxdd"] <= dd_lim:
                if best_row is None or row["ann"] > best_row["ann"]:
                    best_row = row
        if best_row is None:
            print(f"| {dd_lim:.0f} | N/A | N/A | N/A | N/A |")
        else:
            baseline_rows.append(best_row)
            print(f"| {dd_lim:.0f} | {best_row['dd']:.2f} | {best_row['c_base']:.1f} | {best_row['ann']:.2f} | {best_row['dy']:.2f} |")

    strat_c_liq = run_strat_c_liq_proxy(data_map, funding_map)
    m_liq = strategy_metrics(strat_c_liq)
    print(
        f"\nLiquidation-proxy STRAT_C standalone: "
        f"Expectancy={m_liq['expectancy']:.2f}% | PF={m_liq['pf']:.2f} | "
        f"MaxDD={m_liq['maxdd']:.2f}% | Trades={int(m_liq['trades'])} | Net={m_liq['net']:+.2f}"
    )
    print("\n=== Phase-6: Efficient Frontier Matrix (Liquidation C) ===")
    print("| DD Limit% | Actual MaxDD% | C Base Notional | Annualized% | Doubling(Y) |")
    print("|---:|---:|---:|---:|---:|")
    liq_rows = []
    for dd_lim in dd_limits:
        best_row = None
        for c_base in c_base_grid:
            r = simulate_compound_equity([], strat_c_liq, w_a=0.0, w_c=1.0, base_notional_a=0.0, base_notional_c=c_base)
            if r["ann"] <= 0:
                continue
            dy = math.log(2.0) / math.log(1.0 + r["ann"] / 100.0)
            row = {"dd_lim": dd_lim, "dd": r["maxdd"], "c_base": c_base, "ann": r["ann"], "dy": dy}
            if r["maxdd"] <= dd_lim:
                if best_row is None or row["ann"] > best_row["ann"]:
                    best_row = row
        if best_row is None:
            print(f"| {dd_lim:.0f} | N/A | N/A | N/A | N/A |")
        else:
            liq_rows.append(best_row)
            print(f"| {dd_lim:.0f} | {best_row['dd']:.2f} | {best_row['c_base']:.1f} | {best_row['ann']:.2f} | {best_row['dy']:.2f} |")

    report_dir = ROOT / "tests" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    corr.to_csv(report_dir / "hydra_correlation_matrix.csv")
    pd.DataFrame(ranked).head(100).to_json(report_dir / "hydra_top_allocations.json", orient="records", force_ascii=False)
    print(f"Saved reports: {report_dir / 'hydra_correlation_matrix.csv'}, {report_dir / 'hydra_top_allocations.json'}")
    gc.collect()


if __name__ == "__main__":
    asyncio.run(main())

