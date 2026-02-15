"""
Backtest Bot A - Quick Backtest
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv

from backtest_utils import (
    BacktestMarketDataManager,
    TradeResult,
    fetch_klines_df,
    simulate_trade,
    summarize_results,
)

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

load_dotenv(dotenv_path=ROOT / "bots" / "bot_a" / ".env.a_mainnet", override=True)

from bots.bot_a.config_a import get_micro_mvp_config  # noqa: E402
from core.binance_client import BinanceFuturesClient  # noqa: E402
from core.market_regime import MarketRegimeDetector  # noqa: E402


def setup_logging():
    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    logging.getLogger("core.market_regime").setLevel(logging.WARNING)
    logging.getLogger("core.structure_detector").setLevel(logging.ERROR)


def parse_args():
    parser = argparse.ArgumentParser(description="Quick backtest for Bot A")
    parser.add_argument("--days", type=int, default=120, help="Lookback days")
    return parser.parse_args()


def main():
    setup_logging()
    args = parse_args()

    config = get_micro_mvp_config()
    client = BinanceFuturesClient(
        api_key=config.binance_api_key,
        api_secret=config.binance_api_secret,
        base_url=config.binance_base_url,
    )

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=args.days)

    data_15m = fetch_klines_df(client, config.symbol, "15m", start_dt, end_dt)
    data_4h = fetch_klines_df(client, config.symbol, "4h", start_dt, end_dt)
    data_1d = fetch_klines_df(client, config.symbol, "1d", start_dt, end_dt)
    data_1w = fetch_klines_df(client, config.symbol, "1w", start_dt, end_dt)

    if data_15m.empty or data_4h.empty or data_1d.empty or data_1w.empty:
        logging.error("無法取得足夠K線數據")
        return

    md = BacktestMarketDataManager({
        "15m": data_15m,
        "4h": data_4h,
        "1d": data_1d,
        "1w": data_1w,
    })
    detector = MarketRegimeDetector(config, md, require_structure=True)

    trades: list[TradeResult] = []
    i = 50
    while i < len(data_15m) - 2:
        current_time = data_15m["timestamp"].iloc[i]
        md.set_time(current_time)

        decision = detector.evaluate(config.symbol)
        signal = decision.signal
        if decision.allow and signal and signal.entry_allowed:
            entry_price = float(signal.entry_price)
            stop_price = float(signal.stop_loss)
            risk_per_unit = abs(entry_price - stop_price)
            tp_price = entry_price + risk_per_unit * config.tp1_r_multiplier
            side = "BUY"

            exit_idx, exit_price, exit_time = simulate_trade(
                data_15m, i + 1, side, entry_price, stop_price, tp_price
            )

            gross_return = (exit_price - entry_price) / entry_price * 100
            costs = (config.mvp_gate_fee_taker * 2 + config.mvp_gate_slippage_buffer) * 100
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
    print(f"Bot A Backtest (days={args.days})")
    print(f"Trades: {summary['trades']}")
    print(f"Win Rate: {summary['win_rate']:.2f}%")
    print(f"Avg Return: {summary['avg_return_pct']:.4f}%")
    print(f"Expectancy: {summary['expectancy_pct']:.4f}%")


if __name__ == "__main__":
    main()
