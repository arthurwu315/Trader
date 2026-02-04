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

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=args.days)

    data_15m = fetch_klines_df(client, config.symbol, "15m", start_dt, end_dt)
    entry_interval = getattr(config, "entry_interval", "3m")
    if entry_interval == "15m":
        data_entry = data_15m
    else:
        data_entry = fetch_klines_df(client, config.symbol, entry_interval, start_dt, end_dt)
    if data_entry.empty or data_15m.empty:
        logging.error("無法取得足夠K線數據")
        return
    md = BacktestMarketDataManager({entry_interval: data_entry, "15m": data_15m})
    strategy = StrategyCCore(config, md)

    trades: list[TradeResult] = []
    i = 50
    while i < len(data_entry) - 2:
        current_time = data_entry["timestamp"].iloc[i]
        md.set_time(current_time)

        signal = None
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

    summary = summarize_results(trades)
    print(f"Bot C Backtest (days={args.days})")
    print(f"Trades: {summary['trades']}")
    print(f"Win Rate: {summary['win_rate']:.2f}%")
    print(f"Avg Return: {summary['avg_return_pct']:.4f}%")
    print(f"Expectancy: {summary['expectancy_pct']:.4f}%")


if __name__ == "__main__":
    main()
