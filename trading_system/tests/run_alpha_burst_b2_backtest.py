"""
Alpha Burst B2 COMPRESS backtest.
Compression -> expansion burst. No Donchian breakout.
Uses backtest_utils.fetch_klines_df. 4H trend filter, 1H entry/exit.
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
from bots.bot_alpha_burst.config_b2 import (
    STRATEGY_ID,
    UNIVERSE,
    EMA_TREND_PERIOD,
    ATR_PERIOD,
    ATR_MA_PERIOD,
    COMPRESSION_ATR_LOOKBACK,
    COMPRESSION_ATR_PERCENTILE,
    COMPRESSION_MIN_BARS,
    COMPRESSION_RANGE_LOOKBACK,
    ARMED_LOOKBACK,
    EXPANSION_THRESHOLD,
    STOP_ATR_K,
    TRAILING_STOP_ATR_K,
    BURST_RISK_PCT,
    SEED,
)
from core.v9_trade_record import append_burst_b2_trade_record, get_burst_b2_trades_path


def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["atr"] = tr.rolling(ATR_PERIOD).mean()
    df["atr_ma"] = df["atr"].rolling(ATR_MA_PERIOD).mean()
    df["ema200"] = df["close"].ewm(span=EMA_TREND_PERIOD, adjust=False).mean()
    return df


def _trend_at(ts: pd.Timestamp, df_4h: pd.DataFrame) -> str:
    df = df_4h[df_4h["timestamp"] <= ts].tail(1)
    if df.empty or pd.isna(df.iloc[-1].get("ema200")):
        return "short"
    row = df.iloc[-1]
    if float(row["close"]) > float(row["ema200"]):
        return "long"
    return "short"


def _compression_at(i: int, df: pd.DataFrame) -> bool:
    if i < COMPRESSION_ATR_LOOKBACK or pd.isna(df.iloc[i]["atr"]):
        return False
    window = df["atr"].iloc[i - COMPRESSION_ATR_LOOKBACK : i]
    window = window.dropna()
    if len(window) < COMPRESSION_ATR_LOOKBACK // 2:
        return False
    pct = np.percentile(window.values, COMPRESSION_ATR_PERCENTILE)
    return float(df.iloc[i]["atr"]) <= pct


def _armed_at(i: int, df: pd.DataFrame, compression_bars: list) -> bool:
    """Armed if we had at least COMPRESSION_MIN_BARS compression bars in last ARMED_LOOKBACK bars."""
    start = max(0, i - ARMED_LOOKBACK)
    count = sum(1 for j in compression_bars if start <= j < i)
    return count >= COMPRESSION_MIN_BARS


def run_backtest(
    client,
    start_dt: datetime,
    end_dt: datetime,
    burst_equity: float = 10000.0,
    clear_trades_first: bool = True,
    write_trades: bool = True,
    param_overrides: dict = None,
) -> list[dict]:
    """Run ALPHA_BURST_B2_COMPRESS backtest. Returns list of trade dicts.
    param_overrides: e.g. {'compression_atr_pct': 25, 'expansion_threshold': 1.15}
    """
    overrides = param_overrides or {}
    comp_pct = overrides.get("compression_atr_pct", COMPRESSION_ATR_PERCENTILE)
    exp_thr = overrides.get("expansion_threshold", EXPANSION_THRESHOLD)
    range_lb = overrides.get("compression_range_lookback", COMPRESSION_RANGE_LOOKBACK)

    np.random.seed(SEED)
    if clear_trades_first and write_trades and get_burst_b2_trades_path().exists():
        get_burst_b2_trades_path().unlink()

    all_trades: list[dict] = []
    equity = burst_equity

    for symbol in UNIVERSE:
        df_4h = fetch_klines_df(client, symbol, "4h", start_dt, end_dt)
        df_1h = fetch_klines_df(client, symbol, "1h", start_dt, end_dt)
        if df_4h.empty or df_1h.empty or len(df_4h) < EMA_TREND_PERIOD or len(df_1h) < 300:
            continue

        df_4h = _add_indicators(df_4h)
        df_1h = _add_indicators(df_1h)
        df_1h["roll_high"] = df_1h["high"].rolling(range_lb).max().shift(1)
        df_1h["roll_low"] = df_1h["low"].rolling(range_lb).min().shift(1)
        df_1h = df_1h.dropna(subset=["atr", "atr_ma", "roll_high", "roll_low"]).reset_index(drop=True)
        if df_1h.empty:
            continue

        # Precompute compression bars (use overridden comp_pct)
        def _comp_at(j, d):
            if j < COMPRESSION_ATR_LOOKBACK or pd.isna(d.iloc[j]["atr"]):
                return False
            w = d["atr"].iloc[j - COMPRESSION_ATR_LOOKBACK : j].dropna()
            if len(w) < COMPRESSION_ATR_LOOKBACK // 2:
                return False
            pct = np.percentile(w.values, comp_pct)
            return float(d.iloc[j]["atr"]) <= pct

        compression_bars = [j for j in range(len(df_1h)) if _comp_at(j, df_1h)]

        i = 0
        expansion_prev = False
        while i < len(df_1h) - 1:
            row = df_1h.iloc[i]
            close = float(row["close"])
            atr = float(row["atr"])
            atr_ma = float(row["atr_ma"])
            roll_high = float(row["roll_high"])
            roll_low = float(row["roll_low"])
            ts = row["timestamp"]

            if atr <= 0 or atr_ma <= 0:
                expansion_prev = False
                i += 1
                continue

            compression_flag = i in compression_bars
            armed_flag = _armed_at(i, df_1h, compression_bars)
            expansion_flag = atr > (atr_ma * exp_thr)
            first_expansion = expansion_flag and not expansion_prev

            expansion_prev = expansion_flag

            if not (armed_flag and first_expansion):
                i += 1
                continue

            trend = _trend_at(ts, df_4h)
            if trend == "long" and close > roll_high:
                side = "BUY"
                breakout_side = "BUY"
            elif trend == "short" and close < roll_low:
                side = "SELL"
                breakout_side = "SELL"
            else:
                i += 1
                continue

            entry_price = close
            if side == "BUY":
                stop_price = entry_price - STOP_ATR_K * atr
                tp_price = entry_price + 10 * atr
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

            exit_idx, exit_price, _ = simulate_trade(
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
                "compression_flag": compression_flag,
                "armed_flag": armed_flag,
                "expansion_flag": expansion_flag,
                "compression_high": roll_high,
                "compression_low": roll_low,
                "entry_breakout_side": breakout_side,
            }
            all_trades.append(trade)
            equity += pnl_usdt

            if write_trades:
                append_burst_b2_trade_record(
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
                compression_flag=compression_flag,
                armed_flag=armed_flag,
                expansion_flag=expansion_flag,
                compression_high=roll_high,
                compression_low=roll_low,
                entry_breakout_side=breakout_side,
                write_to_v9=write_trades,
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
    print(f"ALPHA_BURST_B2_COMPRESS backtest: {len(trades)} trades")
    print(f"Trades written to {get_burst_b2_trades_path()}")
    if trades:
        total_pnl = sum(t["pnl_usdt"] for t in trades)
        wins = [t for t in trades if t["pnl_usdt"] > 0]
        rs = [t["R_multiple"] for t in trades]
        print(f"Total PnL USDT: {total_pnl:.2f}")
        print(f"Win rate: {len(wins)/len(trades)*100:.1f}%")
        print(f"E[R]: {sum(rs)/len(rs):.4f}")


if __name__ == "__main__":
    main()
