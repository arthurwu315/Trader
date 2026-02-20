"""
動態非對稱緩衝參數掃描：K_UP × K_DOWN 網格回測，產出 Calmar 比較表。
使用方式：cd /home/trader/trading_system && python3 -m tests.run_sweeps
可設 BACKTEST_OFFLINE=1 使用本地快取，減少 API 流量。
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("SKIP_CONFIG_VALIDATION", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))

from backtest_engine_bnb import BacktestEngineBNB
from backtest_utils import fetch_klines_df
from bots.bot_c.strategy_bnb import StrategyBNB, ExitRules

SYMBOL = "BNBUSDT"
INTERVAL = "1h"
FEE_BPS = 9.0
SLIPPAGE_BPS = 5.0
POSITION_SIZE = 0.02
BACKTEST_START = "2022-01-01"
BACKTEST_END = "2023-12-31"

K_UP_VALS = (1.0, 1.5, 2.0)
K_DOWN_VALS = (0.2, 0.5, 0.8)


def load_data():
    from bots.bot_c.config_c import get_strategy_c_config
    from core.binance_client import BinanceFuturesClient
    config = get_strategy_c_config()
    client = BinanceFuturesClient(
        api_key=config.binance_api_key or "dummy",
        api_secret=config.binance_api_secret or "dummy",
        base_url=os.getenv("BINANCE_DATA_URL", "https://fapi.binance.com"),
    )
    start_dt = datetime.strptime(BACKTEST_START, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(BACKTEST_END, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    df = fetch_klines_df(client, SYMBOL, INTERVAL, start_dt, end_dt)
    if df.empty or len(df) < 200:
        raise ValueError(
            f"資料不足: {len(df)} 根。請檢查網路或設定 BACKTEST_OFFLINE=1 並提供 .cache 內 1h 資料"
        )
    return df


def build_strategy(k_up: float, k_down: float) -> StrategyBNB:
    entry_th = {
        "funding_z_threshold": 0.62,
        "rsi_z_threshold": 2.0,
        "min_score": 2,
        "price_breakout_short": 1.0,
        "k_up": k_up,
        "k_down": k_down,
    }
    exit_rules = ExitRules(
        tp_r_mult=2.0,
        tp_atr_mult=2.5,
        sl_atr_mult=2.0,
        trailing_stop_atr_mult=None,
        exit_after_bars=None,
        tp_fixed_pct=None,
    )
    return StrategyBNB(
        entry_thresholds=entry_th,
        exit_rules=exit_rules,
        position_size=POSITION_SIZE,
        direction="short",
        min_factors_required=2,
    )


def main():
    print("載入 1h 資料...")
    data = load_data()
    print(f"  K 線數: {len(data)}")
    engine = BacktestEngineBNB(position_size_pct=POSITION_SIZE, max_trades_per_day=5)
    rows = []
    for k_up in K_UP_VALS:
        for k_down in K_DOWN_VALS:
            strategy = build_strategy(k_up, k_down)
            res = engine.run(strategy, data, fee_bps=FEE_BPS, slippage_bps=SLIPPAGE_BPS)
            trades = res.get("trades_count", 0)
            total_ret = res.get("total_return_pct", 0.0)
            max_dd = max(res.get("max_drawdown_pct", 0.0), 0.01)
            years = max((data["timestamp"].max() - data["timestamp"].min()).days / 365.25, 0.25)
            annual_ret = total_ret / years if years else 0.0
            calmar = annual_ret / max_dd if max_dd else 0.0
            rows.append((k_up, k_down, trades, total_ret, max_dd, calmar))
    print("\n" + "=" * 70)
    print("K_UP   K_DOWN   Total Trades   Max Drawdown %   Calmar Ratio")
    print("=" * 70)
    for k_up, k_down, trades, total_ret, max_dd, calmar in rows:
        print(f" {k_up}      {k_down}         {trades:>4}           {max_dd:>6.2f}          {calmar:>6.2f}")
    print("=" * 70)
    best = max(rows, key=lambda r: r[5])
    print(f"最佳 Calmar: K_UP={best[0]}, K_DOWN={best[1]} -> Calmar={best[5]:.2f}")


if __name__ == "__main__":
    main()
