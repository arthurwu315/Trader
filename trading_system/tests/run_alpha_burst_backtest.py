"""
Alpha Burst B1 backtest.
Uses backtest_utils.fetch_klines_df. 4H trend filter, 1H entry/exit.
Vol expansion + Donchian breakout. ATR stop + ATR trailing.
SEED=42 for reproducibility.
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
    STRATEGY_ID,
    UNIVERSE,
    EMA_TREND_PERIOD,
    ATR_PERIOD,
    ATR_MA_PERIOD,
    VOL_EXPANSION_THRESHOLD,
    BREAKOUT_LOOKBACK,
    STOP_ATR_K,
    TRAILING_STOP_ATR_K,
    BURST_RISK_PCT,
    SEED,
)
from core.v9_trade_record import append_burst_trade_record, get_burst_trades_path


def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add ATR, ATR_ma, roll_high, roll_low, EMA200."""
    df = df.copy()
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["atr"] = tr.rolling(ATR_PERIOD).mean()
    df["atr_ma"] = df["atr"].rolling(ATR_MA_PERIOD).mean()
    df["roll_high"] = df["high"].rolling(BREAKOUT_LOOKBACK).max().shift(1)
    df["roll_low"] = df["low"].rolling(BREAKOUT_LOOKBACK).min().shift(1)
    df["ema200"] = df["close"].ewm(span=EMA_TREND_PERIOD, adjust=False).mean()
    return df


def _trend_at(ts: pd.Timestamp, df_4h: pd.DataFrame) -> str:
    """Return 'long' | 'short' based on 4H close vs EMA200 at ts."""
    df = df_4h[df_4h["timestamp"] <= ts].tail(1)
    if df.empty or pd.isna(df.iloc[-1].get("ema200")):
        return "short"
    row = df.iloc[-1]
    if float(row["close"]) > float(row["ema200"]):
        return "long"
    return "short"


def run_backtest(
    client,
    start_dt: datetime,
    end_dt: datetime,
    burst_equity: float = 10000.0,
    clear_trades_first: bool = True,
) -> list[dict]:
    """
    Run ALPHA_BURST_B1 backtest. Returns list of trade dicts.
    Uses SEED=42 for any randomness.
    """
    np.random.seed(SEED)
    if clear_trades_first and get_burst_trades_path().exists():
        get_burst_trades_path().unlink()

    all_trades: list[dict] = []
    equity = burst_equity

    for symbol in UNIVERSE:
        df_4h = fetch_klines_df(client, symbol, "4h", start_dt, end_dt)
        df_1h = fetch_klines_df(client, symbol, "1h", start_dt, end_dt)
        if df_4h.empty or df_1h.empty or len(df_4h) < EMA_TREND_PERIOD or len(df_1h) < 250:
            continue

        df_4h = _add_indicators(df_4h)
        df_1h = _add_indicators(df_1h)
        df_1h = df_1h.dropna(subset=["atr", "atr_ma", "roll_high", "roll_low"]).reset_index(drop=True)
        if df_1h.empty or "atr" not in df_1h.columns:
            continue

        i = 0
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

            vol_expansion = atr > (atr_ma * VOL_EXPANSION_THRESHOLD)
            if not vol_expansion:
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
                stop_price = entry_price - STOP_ATR_K * atr
                tp_price = entry_price + 10 * atr  # far TP, use trailing
            else:
                stop_price = entry_price + STOP_ATR_K * atr
                tp_price = entry_price - 10 * atr

            initial_risk_per_unit = abs(entry_price - stop_price)
            if initial_risk_per_unit <= 0:
                i += 1
                continue

            risk_usdt = equity * BURST_RISK_PCT
            qty = risk_usdt / initial_risk_per_unit
            if qty <= 0:
                i += 1
                continue

            exit_idx, exit_price, exit_time = simulate_trade(
                df_1h, i + 1, side, entry_price, stop_price, tp_price,
                trailing_stop_atr_mult=TRAILING_STOP_ATR_K,
            )
            holding_bars = exit_idx - i

            if side == "BUY":
                pnl_usdt = (exit_price - entry_price) * qty
            else:
                pnl_usdt = (entry_price - exit_price) * qty

            R_multiple = pnl_usdt / risk_usdt if risk_usdt > 0 else 0.0

            trade = {
                "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ") if hasattr(ts, "strftime") else str(ts),
                "symbol": symbol,
                "side": side,
                "entry_price": entry_price,
                "stop_price": stop_price,
                "exit_price": float(exit_price),
                "qty": qty,
                "initial_risk_usdt": risk_usdt,
                "pnl_usdt": pnl_usdt,
                "R_multiple": R_multiple,
                "holding_bars": holding_bars,
                "strategy_id": STRATEGY_ID,
            }
            all_trades.append(trade)
            equity += pnl_usdt

            append_burst_trade_record(
                timestamp=trade["timestamp"],
                symbol=symbol,
                side=side,
                entry_price=entry_price,
                stop_price=stop_price,
                exit_price=trade["exit_price"],
                qty=qty,
                initial_risk_usdt=risk_usdt,
                pnl_usdt=pnl_usdt,
                R_multiple=R_multiple,
                holding_bars=holding_bars,
                strategy_id=STRATEGY_ID,
            )
            i = exit_idx + 1

    return all_trades


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

    trades = run_backtest(client, start_dt, end_dt, burst_equity=10000.0, clear_trades_first=True)
    print(f"ALPHA_BURST_B1 backtest: {len(trades)} trades")
    print(f"Trades written to {get_burst_trades_path()}")
    if trades:
        total_pnl = sum(t["pnl_usdt"] for t in trades)
        wins = [t for t in trades if t["pnl_usdt"] > 0]
        print(f"Total PnL USDT: {total_pnl:.2f}")
        print(f"Win rate: {len(wins)/len(trades)*100:.1f}%")
        rs = [t["R_multiple"] for t in trades]
        print(f"E[R]: {sum(rs)/len(rs):.4f}")


if __name__ == "__main__":
    main()
