"""
Backtest Utils
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def _to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def fetch_klines_df(client, symbol: str, interval: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    cache_dir = Path(os.getenv("BACKTEST_CACHE_DIR", "/home/trader/trading_system/tests/.cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    start_key = start_dt.strftime("%Y%m%d")
    end_key = end_dt.strftime("%Y%m%d")
    cache_path = cache_dir / f"{symbol}_{interval}_{start_key}_{end_key}.csv"

    if cache_path.exists():
        try:
            df_cached = pd.read_csv(cache_path)
            df_cached["timestamp"] = pd.to_datetime(df_cached["timestamp"], utc=True)
            return df_cached
        except Exception:
            pass

    start_ms = _to_ms(start_dt)
    end_ms = _to_ms(end_dt)
    all_rows: List[list] = []

    while True:
        rows = client.get_klines(
            symbol=symbol,
            interval=interval,
            limit=1500,
            start_time=start_ms,
            end_time=end_ms,
        )
        if not rows:
            break

        all_rows.extend(rows)
        last_open_time = rows[-1][0]
        next_start = last_open_time + 1
        if next_start >= end_ms or len(rows) < 1500:
            break
        start_ms = next_start

    df = pd.DataFrame(
        all_rows,
        columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades",
            "taker_buy_base", "taker_buy_quote", "ignore"
        ],
    )
    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df = df.sort_values("timestamp").reset_index(drop=True)
    try:
        df.to_csv(cache_path, index=False)
    except Exception:
        pass
    return df


class BacktestMarketDataManager:
    def __init__(self, data_by_interval: Dict[str, pd.DataFrame]):
        self.data_by_interval = data_by_interval
        self.current_time: Optional[pd.Timestamp] = None

    def set_time(self, ts: pd.Timestamp) -> None:
        self.current_time = ts

    def get_klines_df(self, symbol: str, interval: str, limit: int = 500, use_cache: bool = True) -> pd.DataFrame:
        df = self.data_by_interval.get(interval)
        if df is None or df.empty:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        if self.current_time is None:
            return df.tail(limit).copy()
        sliced = df[df["timestamp"] <= self.current_time].tail(limit)
        return sliced.copy()


@dataclass
class TradeResult:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    side: str
    return_pct: float
    win: bool


def simulate_trade(
    df: pd.DataFrame,
    start_index: int,
    side: str,
    entry_price: float,
    stop_price: float,
    tp_price: float,
) -> tuple[int, float, pd.Timestamp]:
    for i in range(start_index, len(df)):
        row = df.iloc[i]
        high = row["high"]
        low = row["low"]

        if side == "BUY":
            hit_stop = low <= stop_price
            hit_tp = high >= tp_price
            if hit_stop and hit_tp:
                return i, stop_price, row["timestamp"]
            if hit_stop:
                return i, stop_price, row["timestamp"]
            if hit_tp:
                return i, tp_price, row["timestamp"]
        else:
            hit_stop = high >= stop_price
            hit_tp = low <= tp_price
            if hit_stop and hit_tp:
                return i, stop_price, row["timestamp"]
            if hit_stop:
                return i, stop_price, row["timestamp"]
            if hit_tp:
                return i, tp_price, row["timestamp"]

    last_row = df.iloc[-1]
    return len(df) - 1, float(last_row["close"]), last_row["timestamp"]


def simulate_trade_with_ema_exit(
    df: pd.DataFrame,
    start_index: int,
    side: str,
    entry_price: float,
    stop_price: float,
    tp_price: float,
    ema_period: int = 20,
) -> tuple[int, float, float, pd.Timestamp]:
    """
    Exit on SL/TP first, otherwise exit on EMA cross of close.
    Returns: (exit_index, exit_price, return_pct, exit_time)
    """
    ema = df["close"].ewm(span=ema_period, adjust=False).mean()

    for i in range(start_index, len(df)):
        row = df.iloc[i]
        high = row["high"]
        low = row["low"]
        close = row["close"]
        ema_val = ema.iloc[i]

        if side == "BUY":
            hit_stop = low <= stop_price
            hit_tp = high >= tp_price
            if hit_stop and hit_tp:
                exit_price = stop_price
                ret_pct = (exit_price - entry_price) / entry_price * 100
                return i, exit_price, ret_pct, row["timestamp"]
            if hit_stop:
                exit_price = stop_price
                ret_pct = (exit_price - entry_price) / entry_price * 100
                return i, exit_price, ret_pct, row["timestamp"]
            if hit_tp:
                exit_price = tp_price
                ret_pct = (exit_price - entry_price) / entry_price * 100
                return i, exit_price, ret_pct, row["timestamp"]
            if close < ema_val:
                exit_price = close
                ret_pct = (exit_price - entry_price) / entry_price * 100
                return i, exit_price, ret_pct, row["timestamp"]
        else:
            hit_stop = high >= stop_price
            hit_tp = low <= tp_price
            if hit_stop and hit_tp:
                exit_price = stop_price
                ret_pct = (entry_price - exit_price) / entry_price * 100
                return i, exit_price, ret_pct, row["timestamp"]
            if hit_stop:
                exit_price = stop_price
                ret_pct = (entry_price - exit_price) / entry_price * 100
                return i, exit_price, ret_pct, row["timestamp"]
            if hit_tp:
                exit_price = tp_price
                ret_pct = (entry_price - exit_price) / entry_price * 100
                return i, exit_price, ret_pct, row["timestamp"]
            if close > ema_val:
                exit_price = close
                ret_pct = (entry_price - exit_price) / entry_price * 100
                return i, exit_price, ret_pct, row["timestamp"]

    last_row = df.iloc[-1]
    exit_price = float(last_row["close"])
    ret_pct = (exit_price - entry_price) / entry_price * (1 if side == "BUY" else -1) * 100
    return len(df) - 1, exit_price, ret_pct, last_row["timestamp"]


def simulate_trade_two_stage(
    df: pd.DataFrame,
    start_index: int,
    side: str,
    entry_price: float,
    stop_price: float,
    tp1_price: float,
    tp2_price: float,
    partial_ratio: float = 0.5,
) -> tuple[int, float, float, pd.Timestamp]:
    """
    Two-stage exit. Returns: (exit_index, exit_price, return_pct, exit_time)
    """
    partial_ratio = max(0.0, min(1.0, partial_ratio))
    tp1_hit = False
    realized_pct = 0.0

    for i in range(start_index, len(df)):
        row = df.iloc[i]
        high = row["high"]
        low = row["low"]

        if side == "BUY":
            hit_stop = low <= stop_price
            hit_tp1 = high >= tp1_price
            hit_tp2 = high >= tp2_price

            if hit_stop and not tp1_hit:
                ret_pct = (stop_price - entry_price) / entry_price * 100
                return i, stop_price, ret_pct, row["timestamp"]

            if hit_tp1 and not tp1_hit:
                tp1_hit = True
                realized_pct += (tp1_price - entry_price) / entry_price * 100 * partial_ratio

            if hit_tp2:
                remaining_ratio = 1 - partial_ratio if tp1_hit else 1.0
                realized_pct += (tp2_price - entry_price) / entry_price * 100 * remaining_ratio
                return i, tp2_price, realized_pct, row["timestamp"]

            if hit_stop and tp1_hit:
                remaining_ratio = 1 - partial_ratio
                realized_pct += (stop_price - entry_price) / entry_price * 100 * remaining_ratio
                return i, stop_price, realized_pct, row["timestamp"]
        else:
            hit_stop = high >= stop_price
            hit_tp1 = low <= tp1_price
            hit_tp2 = low <= tp2_price

            if hit_stop and not tp1_hit:
                ret_pct = (entry_price - stop_price) / entry_price * 100
                return i, stop_price, ret_pct, row["timestamp"]

            if hit_tp1 and not tp1_hit:
                tp1_hit = True
                realized_pct += (entry_price - tp1_price) / entry_price * 100 * partial_ratio

            if hit_tp2:
                remaining_ratio = 1 - partial_ratio if tp1_hit else 1.0
                realized_pct += (entry_price - tp2_price) / entry_price * 100 * remaining_ratio
                return i, tp2_price, realized_pct, row["timestamp"]

            if hit_stop and tp1_hit:
                remaining_ratio = 1 - partial_ratio
                realized_pct += (entry_price - stop_price) / entry_price * 100 * remaining_ratio
                return i, stop_price, realized_pct, row["timestamp"]

    last_row = df.iloc[-1]
    exit_price = float(last_row["close"])
    direction = 1 if side == "BUY" else -1
    ret_pct = (exit_price - entry_price) / entry_price * direction * 100
    return len(df) - 1, exit_price, ret_pct, last_row["timestamp"]


def summarize_results(trades: List[TradeResult]) -> Dict[str, float]:
    if not trades:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "avg_return_pct": 0.0,
            "expectancy_pct": 0.0,
            "long_trades": 0,
            "long_win_rate": 0.0,
            "long_expectancy_pct": 0.0,
            "short_trades": 0,
            "short_win_rate": 0.0,
            "short_expectancy_pct": 0.0,
        }

    returns = [t.return_pct for t in trades]
    wins = [t for t in trades if t.win]
    win_rate = len(wins) / len(trades) * 100
    avg_return = sum(returns) / len(returns)

    def _side_summary(side: str) -> tuple[int, float, float]:
        side_trades = [t for t in trades if t.side == side]
        if not side_trades:
            return 0, 0.0, 0.0
        side_returns = [t.return_pct for t in side_trades]
        side_wins = [t for t in side_trades if t.win]
        side_win_rate = len(side_wins) / len(side_trades) * 100
        side_expectancy = sum(side_returns) / len(side_returns)
        return len(side_trades), side_win_rate, side_expectancy

    long_trades, long_win_rate, long_expectancy = _side_summary("BUY")
    short_trades, short_win_rate, short_expectancy = _side_summary("SELL")

    return {
        "trades": len(trades),
        "win_rate": win_rate,
        "avg_return_pct": avg_return,
        "expectancy_pct": avg_return,
        "long_trades": long_trades,
        "long_win_rate": long_win_rate,
        "long_expectancy_pct": long_expectancy,
        "short_trades": short_trades,
        "short_win_rate": short_win_rate,
        "short_expectancy_pct": short_expectancy,
    }
