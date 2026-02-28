"""
Alpha Burst B1 parameter grid (B2).
Small grid: vol_expansion x3, stop_ATR_k x3, breakout_lookback x3.
Outputs to tests/reports/alpha_burst_b1_artifacts/b2_grid_results.csv
SEED=42.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("SKIP_CONFIG_VALIDATION", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backtest_utils import fetch_klines_df, simulate_trade
from bots.bot_alpha_burst.config import (
    UNIVERSE,
    EMA_TREND_PERIOD,
    ATR_PERIOD,
    ATR_MA_PERIOD,
    BREAKOUT_LOOKBACK,
    BURST_RISK_PCT,
    SEED,
)

SEED = 42
VOL_OPTS = [1.0, 1.2, 1.5]
STOP_OPTS = [1.5, 2.0, 2.5]
LOOKBACK_OPTS = [15, 20, 25]


def _add_indicators(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    df = df.copy()
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["atr"] = tr.rolling(ATR_PERIOD).mean()
    df["atr_ma"] = df["atr"].rolling(ATR_MA_PERIOD).mean()
    df["roll_high"] = df["high"].rolling(lookback).max().shift(1)
    df["roll_low"] = df["low"].rolling(lookback).min().shift(1)
    df["ema200"] = df["close"].ewm(span=EMA_TREND_PERIOD, adjust=False).mean()
    return df


def _trend_at(ts: pd.Timestamp, df_4h: pd.DataFrame) -> str:
    df = df_4h[df_4h["timestamp"] <= ts].tail(1)
    if df.empty or pd.isna(df.iloc[-1].get("ema200")):
        return "short"
    row = df.iloc[-1]
    return "long" if float(row["close"]) > float(row["ema200"]) else "short"


def _run_single(
    client,
    start_dt,
    end_dt,
    vol_thresh: float,
    stop_k: float,
    lookback: int,
    burst_equity: float = 10000.0,
) -> dict:
    np.random.seed(SEED)
    all_rs = []
    for symbol in UNIVERSE:
        df_4h = fetch_klines_df(client, symbol, "4h", start_dt, end_dt)
        df_1h = fetch_klines_df(client, symbol, "1h", start_dt, end_dt)
        if df_4h.empty or df_1h.empty or len(df_4h) < EMA_TREND_PERIOD or len(df_1h) < 250:
            continue
        df_4h = _add_indicators(df_4h, lookback)
        df_1h = _add_indicators(df_1h, lookback)
        df_1h = df_1h.dropna(subset=["atr", "atr_ma", "roll_high", "roll_low"]).reset_index(drop=True)
        if df_1h.empty:
            continue

        i = 0
        equity = burst_equity
        while i < len(df_1h) - 1:
            row = df_1h.iloc[i]
            close = float(row["close"])
            atr = float(row["atr"])
            atr_ma = float(row["atr_ma"])
            roll_high = float(row["roll_high"])
            roll_low = float(row["roll_low"])
            ts = row["timestamp"]
            if atr <= 0 or atr_ma <= 0:
                i += 1
                continue
            if atr <= atr_ma * vol_thresh:
                i += 1
                continue
            trend = _trend_at(ts, df_4h)
            if trend == "long" and close > roll_high:
                side = "BUY"
            elif trend == "short" and close < roll_low:
                side = "SELL"
            else:
                i += 1
                continue

            entry_price = close
            if side == "BUY":
                stop_price = entry_price - stop_k * atr
                tp_price = entry_price + 10 * atr
            else:
                stop_price = entry_price + stop_k * atr
                tp_price = entry_price - 10 * atr
            risk_per_unit = abs(entry_price - stop_price)
            if risk_per_unit <= 0:
                i += 1
                continue
            risk_usdt = equity * BURST_RISK_PCT
            qty = risk_usdt / risk_per_unit
            if qty <= 0:
                i += 1
                continue

            exit_idx, exit_price, _ = simulate_trade(
                df_1h, i + 1, side, entry_price, stop_price, tp_price,
                trailing_stop_atr_mult=stop_k,
            )
            if side == "BUY":
                pnl = (exit_price - entry_price) * qty
            else:
                pnl = (entry_price - exit_price) * qty
            r = pnl / risk_usdt if risk_usdt > 0 else 0
            all_rs.append(r)
            equity += pnl
            i = exit_idx + 1

    rs = np.array(all_rs) if all_rs else np.array([0.0])
    return {
        "vol_expansion": vol_thresh,
        "stop_ATR_k": stop_k,
        "breakout_lookback": lookback,
        "trade_count": len(all_rs),
        "E_R": float(np.mean(rs)),
        "WinRate": float(np.mean(rs > 0) * 100) if len(rs) else 0,
        "plateau_note": "",
    }


def main():
    from dotenv import load_dotenv
    from bots.bot_c.config_c import get_strategy_c_config
    from core.binance_client import BinanceFuturesClient

    load_dotenv(dotenv_path=ROOT / ".env", override=True)
    config = get_strategy_c_config()
    client = BinanceFuturesClient(
        api_key=config.binance_api_key or "dummy",
        api_secret=config.binance_api_secret or "dummy",
        base_url=os.getenv("BINANCE_DATA_URL", "https://fapi.binance.com"),
    )
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=365 * 3)

    out_dir = ROOT / "tests" / "reports" / "alpha_burst_b1_artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for v in VOL_OPTS:
        for s in STOP_OPTS:
            for lb in LOOKBACK_OPTS:
                res = _run_single(client, start_dt, end_dt, v, s, lb)
                rows.append(res)
                print(f"vol={v} stop={s} lb={lb} -> E[R]={res['E_R']:.4f} n={res['trade_count']}")

    df = pd.DataFrame(rows)
    out_path = out_dir / "b2_grid_results.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
