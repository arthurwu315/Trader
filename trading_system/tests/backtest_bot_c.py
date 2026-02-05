"""
Backtest Bot C - Quick Backtest
"""
from __future__ import annotations

import argparse
import logging
import sys
import os
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv
import pandas as pd

from backtest_utils import (
    BacktestMarketDataManager,
    TradeResult,
    fetch_klines_df,
    simulate_trade,
    simulate_trade_with_ema_exit,
    simulate_trade_two_stage,
    summarize_results,
)

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

load_dotenv(dotenv_path=ROOT / "bots" / "bot_c" / ".env.c_testnet", override=True)

from bots.bot_c.config_c import get_strategy_c_config  # noqa: E402
from bots.bot_c.strategy_c_core import StrategyCCore  # noqa: E402
from core.binance_client import BinanceFuturesClient  # noqa: E402


def run_adx_backtest(config, data_entry, data_15m) -> list[TradeResult]:
    # Precompute indicators
    df_3m = data_entry.copy()
    df_15m = data_15m.copy()

    def ema(series, period):
        return series.ewm(span=period, adjust=False).mean()

    def atr(df, period):
        high = df["high"]
        low = df["low"]
        close = df["close"]
        prev_close = close.shift(1)
        tr = (high - low).to_frame(name="hl")
        tr["hc"] = (high - prev_close).abs()
        tr["lc"] = (low - prev_close).abs()
        return tr.max(axis=1).rolling(window=period, min_periods=period).mean()

    def adx(df, period=14):
        high = df["high"]
        low = df["low"]
        close = df["close"]
        prev_close = close.shift(1)
        tr = (high - low).to_frame(name="hl")
        tr["hc"] = (high - prev_close).abs()
        tr["lc"] = (low - prev_close).abs()
        true_range = tr.max(axis=1)

        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

        atr_val = true_range.rolling(window=period, min_periods=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period, min_periods=period).mean() / atr_val)
        minus_di = 100 * (minus_dm.rolling(window=period, min_periods=period).mean() / atr_val)
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)).replace([float("inf"), -float("inf")], 0.0) * 100
        return dx.rolling(window=period, min_periods=period).mean()

    df_15m["ema20"] = ema(df_15m["close"], 20)
    df_15m["ema50"] = ema(df_15m["close"], 50)
    df_15m["adx"] = adx(df_15m, period=int(getattr(config, "c_adx_period", 14)))

    df_3m["ema9"] = ema(df_3m["close"], 9)
    df_3m["ema20"] = ema(df_3m["close"], 20)
    df_3m["atr"] = atr(df_3m, int(getattr(config, "c_atr_period_3m", 14)))
    vol_lookback = int(getattr(config, "c_breakout_volume_lookback", 20))
    df_3m["vol_sma"] = df_3m["volume"].rolling(window=vol_lookback, min_periods=vol_lookback).mean()

    lookback = int(getattr(config, "c_breakout_lookback", 20))
    df_3m["break_high"] = df_3m["high"].rolling(window=lookback, min_periods=lookback).max().shift(1)
    df_3m["break_low"] = df_3m["low"].rolling(window=lookback, min_periods=lookback).min().shift(1)

    # Align 15m indicators to 3m by timestamp
    df_15m = df_15m[["timestamp", "close", "ema20", "ema50", "adx"]].sort_values("timestamp")
    df_3m = df_3m.sort_values("timestamp")
    df_merged = pd.merge_asof(df_3m, df_15m, on="timestamp", direction="backward", suffixes=("", "_15m"))

    trades: list[TradeResult] = []
    i = max(lookback + 2, 60)
    buffer_pct = float(getattr(config, "c_breakout_buffer_pct", 0.0002))
    vol_mult = float(getattr(config, "c_breakout_volume_mult", 1.2))
    adx_min = float(getattr(config, "c_adx_min", 20.0))
    atr_mult = float(getattr(config, "c_atr_mult", 1.5))

    while i < len(df_merged) - 2:
        row = df_merged.iloc[i]
        adx_val = float(row["adx"]) if row["adx"] == row["adx"] else 0.0
        if adx_val < adx_min:
            i += 1
            continue

        ema20_15m = float(row["ema20"])
        ema50_15m = float(row["ema50"])
        price_15m = float(row["close_15m"])

        trend_long = ema20_15m > ema50_15m and price_15m >= ema20_15m
        trend_short = ema20_15m < ema50_15m and price_15m <= ema20_15m

        close = float(row["close"])
        open_p = float(row["open"])
        high = float(row["high"])
        low = float(row["low"])
        ema9 = float(row["ema9"])
        ema20 = float(row["ema20"])
        atr_val = float(row["atr"]) if row["atr"] == row["atr"] else 0.0
        vol_ok = float(row["volume"]) >= float(row["vol_sma"]) * vol_mult if row["vol_sma"] == row["vol_sma"] else False
        break_high = float(row["break_high"]) if row["break_high"] == row["break_high"] else None
        break_low = float(row["break_low"]) if row["break_low"] == row["break_low"] else None

        signal_side = None
        if trend_long and break_high is not None:
            if close > break_high * (1 + buffer_pct) and ema9 > ema20 and vol_ok:
                signal_side = "BUY"
                sl_price = close - atr_val * atr_mult
        if signal_side is None and trend_short and break_low is not None:
            if close < break_low * (1 - buffer_pct) and ema9 < ema20 and vol_ok:
                signal_side = "SELL"
                sl_price = close + atr_val * atr_mult

        if signal_side is None:
            i += 1
            continue

        sl_pct = abs(close - sl_price) / close
        if sl_pct < config.min_stop_distance_pct:
            sl_price = close * (1 - config.min_stop_distance_pct) if signal_side == "BUY" else close * (1 + config.min_stop_distance_pct)
        if sl_pct > config.max_stop_distance_pct:
            i += 1
            continue

        round_trip_fee = config.fee_taker * 2
        slippage = config.slippage_buffer
        total_cost = round_trip_fee + slippage
        rr = float(getattr(config, "tp_rr_multiple", 2.0))
        tp_min_fixed_pct = float(getattr(config, "tp_min_fixed_pct", 0.008))

        if signal_side == "BUY":
            r_dist = close - sl_price
            tp_by_r = close + r_dist * rr
            fixed_pct = max(tp_min_fixed_pct, total_cost + config.min_tp_after_costs_pct)
            tp_by_fixed = close * (1 + fixed_pct)
            tp_price = max(tp_by_r, tp_by_fixed)
        else:
            r_dist = sl_price - close
            tp_by_r = close - r_dist * rr
            fixed_pct = max(tp_min_fixed_pct, total_cost + config.min_tp_after_costs_pct)
            tp_by_fixed = close * (1 - fixed_pct)
            tp_price = min(tp_by_r, tp_by_fixed)

        side = "BUY" if signal_side == "BUY" else "SELL"
        exit_idx, exit_price, exit_time = simulate_trade(
            df_3m, i + 1, side, close, sl_price, tp_price
        )
        direction = 1 if side == "BUY" else -1
        gross_return = (exit_price - close) / close * direction * 100
        costs = (config.fee_taker * 2 + config.slippage_buffer) * 100
        net_return = gross_return - costs

        trades.append(
            TradeResult(
                entry_time=row["timestamp"],
                exit_time=exit_time,
                entry_price=close,
                exit_price=exit_price,
                side=side,
                return_pct=net_return,
                win=net_return > 0,
            )
        )
        i = exit_idx + 1

    return trades


def setup_logging():
    logging.basicConfig(
        level=logging.ERROR,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    logging.getLogger("bots.bot_b.strategy_b_core").setLevel(logging.ERROR)


def parse_args():
    parser = argparse.ArgumentParser(description="Quick backtest for Bot C")
    parser.add_argument("--days", type=int, default=60, help="Lookback days")
    parser.add_argument("--start-date", type=str, default=None, help="Start date YYYYMMDD")
    parser.add_argument("--end-date", type=str, default=None, help="End date YYYYMMDD")
    return parser.parse_args()


def main():
    setup_logging()
    args = parse_args()

    config = get_strategy_c_config()
    data_base_url = os.getenv("BINANCE_DATA_URL", "https://fapi.binance.com")
    client = BinanceFuturesClient(
        api_key=config.binance_api_key,
        api_secret=config.binance_api_secret,
        base_url=data_base_url,
    )

    if args.start_date and args.end_date:
        start_dt = datetime.strptime(args.start_date, "%Y%m%d").replace(tzinfo=timezone.utc)
        end_dt = datetime.strptime(args.end_date, "%Y%m%d").replace(tzinfo=timezone.utc)
        days = max((end_dt - start_dt).days, 1)
    else:
        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=args.days)
        days = args.days

    data_15m = fetch_klines_df(client, config.symbol, "15m", start_dt, end_dt)
    entry_interval = getattr(config, "entry_interval", "3m")
    if entry_interval == "15m":
        data_entry = data_15m
    else:
        data_entry = fetch_klines_df(client, config.symbol, entry_interval, start_dt, end_dt)
    htf_slow = int(getattr(config, "l1_htf_slow_ema", 200))
    htf_extra_days = int((htf_slow + 5) * 4 / 24) + 5
    htf_start_dt = end_dt - timedelta(days=days + htf_extra_days)
    data_4h = fetch_klines_df(client, config.symbol, "4h", htf_start_dt, end_dt)
    if data_4h.empty:
        data_4h = fetch_klines_df(client, config.symbol, "4h", start_dt, end_dt)
    if data_entry.empty or data_15m.empty or data_4h.empty:
        logging.error("無法取得足夠K線數據")
        return
    md = BacktestMarketDataManager({entry_interval: data_entry, "15m": data_15m, "4h": data_4h})
    strategy = StrategyCCore(config, md)

    trades: list[TradeResult] = []
    i = 50
    while i < len(data_entry) - 2:
        current_time = data_entry["timestamp"].iloc[i]
        md.set_time(current_time)

        signal = None
        if str(getattr(config, "c_strategy_mode", "")).upper() == "ADX_TREND":
            break
        else:
            l1_pass, _, l1_debug = strategy.l1_gate.check_long_environment(md)
            if l1_pass:
                has_signal, _, signal = strategy.l2_gate.check_entry_pattern(
                    md,
                    l1_passed=l1_pass,
                    bar_index=i
                )

            if signal is None and getattr(config, "enable_short", True):
                l1_short_pass, _, _ = strategy.l1_gate.check_short_environment(md)
                if l1_short_pass:
                    has_signal, _, signal = strategy.l2_gate.check_entry_pattern_short(
                        md,
                        l1_passed=l1_short_pass,
                        bar_index=i
                    )

        if signal:
            entry_price = signal.entry_price
            stop_price = signal.stop_loss
            tp_price = signal.tp1_price
            side = "BUY" if signal.signal_type == "LONG" else "SELL"

            if getattr(config, "enable_ema_exit", False):
                exit_idx, exit_price, gross_return, exit_time = simulate_trade_with_ema_exit(
                    data_entry,
                    i + 1,
                    side,
                    entry_price,
                    stop_price,
                    tp_price,
                    ema_period=int(getattr(config, "ema_exit_period", 20)),
                )
            elif getattr(config, "enable_partial_tp", True) and getattr(signal, "tp2_price", None):
                exit_idx, exit_price, gross_return, exit_time = simulate_trade_two_stage(
                    data_entry,
                    i + 1,
                    side,
                    entry_price,
                    stop_price,
                    tp_price,
                    signal.tp2_price,
                    partial_ratio=float(getattr(config, "partial_tp_ratio", 0.5)),
                )
            else:
                exit_idx, exit_price, exit_time = simulate_trade(
                    data_entry, i + 1, side, entry_price, stop_price, tp_price
                )
                direction = 1 if side == "BUY" else -1
                gross_return = (exit_price - entry_price) / entry_price * direction * 100
            costs = (config.fee_taker * 2 + config.slippage_buffer) * 100
            net_return = gross_return - costs

            trades.append(
                TradeResult(
                    entry_time=current_time,
                    exit_time=exit_time,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    side=side,
                    return_pct=net_return,
                    win=net_return > 0,
                )
            )

            i = exit_idx + 1
            continue

        i += 1

    if str(getattr(config, "c_strategy_mode", "")).upper() == "ADX_TREND":
        trades = run_adx_backtest(config, data_entry, data_15m)

    summary = summarize_results(trades)
    trades_per_day = summary["trades"] / max(days, 1)
    weekly_est = summary["expectancy_pct"] * trades_per_day * 7
    print(f"Bot C Backtest (days={days})")
    print(f"Range: {start_dt.date()} -> {end_dt.date()}")
    print(f"Trades: {summary['trades']}")
    print(f"Win Rate: {summary['win_rate']:.2f}%")
    print(f"Avg Return: {summary['avg_return_pct']:.4f}%")
    print(f"Expectancy: {summary['expectancy_pct']:.4f}%")
    print(f"Trades/Day: {trades_per_day:.2f}")
    print(f"Est Weekly Return: {weekly_est:.2f}%")


if __name__ == "__main__":
    main()
