"""
多幣種 Squeeze 組合回測聚合器 (Portfolio Backtester)。

目標：
- SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "AVAXUSDT"]
- 統一策略參數：K=150, TRAILING_ATR_MULT=2.5
- 以 asyncio.gather 並行抓取各幣種 2022-2023 的 1H 資料
- 在回測層實作最大同時持倉限制 MAX_CONCURRENT=2
- 輸出組合績效：Total Trades, Portfolio Net Profit, Max Drawdown %, Profit Factor
"""
from __future__ import annotations

import asyncio
import gc
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("SKIP_CONFIG_VALIDATION", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))

from backtest_engine_bnb import BacktestEngineBNB
from backtest_utils import fetch_klines_df
from bots.bot_c.strategy_bnb import ExitRules, StrategyBNB

SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "AVAXUSDT"]
INTERVAL = "1h"
BACKTEST_START = "2022-01-01"
BACKTEST_END = "2023-12-31"
FEE_BPS = 9.0
SLIPPAGE_BPS = 5.0

# 統一策略參數
SQUEEZE_K = 150
TRAILING_ATR_MULT = 2.5
ATR_STOP_MULT_INIT = 2.5

# 組合層模擬
INITIAL_EQUITY_USDT = 10000.0
NOTIONAL_PER_TRADE = INITIAL_EQUITY_USDT * 0.2  # 固定名目 20% = 2000 USDT
MAX_CONCURRENT = 2


@dataclass
class PortfolioTrade:
    symbol: str
    entry_time: Any
    exit_time: Any
    return_net_pct: float
    pnl_usdt: float


def build_strategy() -> StrategyBNB:
    entry_th = {"squeeze_k": SQUEEZE_K}
    exit_rules = ExitRules(
        tp_r_mult=None,
        tp_atr_mult=None,
        sl_atr_mult=ATR_STOP_MULT_INIT,
        trailing_stop_atr_mult=TRAILING_ATR_MULT,
        exit_after_bars=None,
        tp_fixed_pct=None,
    )
    return StrategyBNB(
        entry_thresholds=entry_th,
        exit_rules=exit_rules,
        position_size=0.02,
        direction="long",
        min_factors_required=2,
    )


def build_client():
    from bots.bot_c.config_c import get_strategy_c_config
    from core.binance_client import BinanceFuturesClient

    config = get_strategy_c_config()
    return BinanceFuturesClient(
        api_key=config.binance_api_key or "dummy",
        api_secret=config.binance_api_secret or "dummy",
        base_url=os.getenv("BINANCE_DATA_URL", "https://fapi.binance.com"),
    )


def _run_single_symbol_backtest(symbol: str) -> list[PortfolioTrade]:
    start_dt = datetime.strptime(BACKTEST_START, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(BACKTEST_END, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    client = build_client()
    df = fetch_klines_df(client, symbol, INTERVAL, start_dt, end_dt)
    if df.empty:
        return []

    strategy = build_strategy()
    engine = BacktestEngineBNB(position_size_pct=0.02, max_trades_per_day=10)
    res = engine.run(
        strategy,
        df,
        fee_bps=FEE_BPS,
        slippage_bps=SLIPPAGE_BPS,
        reverse_side=False,
    )

    trades: list[PortfolioTrade] = []
    for t in res.get("trades", []):
        pnl = NOTIONAL_PER_TRADE * (float(t.return_net_pct) / 100.0)
        trades.append(
            PortfolioTrade(
                symbol=symbol,
                entry_time=t.entry_time,
                exit_time=t.exit_time,
                return_net_pct=float(t.return_net_pct),
                pnl_usdt=pnl,
            )
        )

    # 釋放單幣種資料記憶體，避免 1GB 伺服器 OOM
    del df
    del res
    gc.collect()
    return trades


async def fetch_and_backtest_all_symbols() -> list[PortfolioTrade]:
    tasks = [asyncio.to_thread(_run_single_symbol_backtest, symbol) for symbol in SYMBOLS]
    results = await asyncio.gather(*tasks)
    merged: list[PortfolioTrade] = []
    for trades in results:
        merged.extend(trades)
    return merged


def apply_portfolio_risk_limits(all_trades: list[PortfolioTrade]) -> tuple[list[PortfolioTrade], int]:
    all_trades_sorted = sorted(all_trades, key=lambda t: t.entry_time)
    accepted: list[PortfolioTrade] = []
    active: list[PortfolioTrade] = []
    skipped = 0

    for tr in all_trades_sorted:
        active = [a for a in active if a.exit_time > tr.entry_time]
        if len(active) >= MAX_CONCURRENT:
            skipped += 1
            continue
        accepted.append(tr)
        active.append(tr)

    return accepted, skipped


def portfolio_metrics(trades: list[PortfolioTrade]) -> dict[str, float]:
    if not trades:
        return {
            "total_trades": 0.0,
            "net_profit": 0.0,
            "max_drawdown_pct": 0.0,
            "profit_factor": 0.0,
        }

    # 權益曲線以平倉時點累加
    equity = INITIAL_EQUITY_USDT
    curve = [equity]
    gross_profit = 0.0
    gross_loss = 0.0

    for tr in sorted(trades, key=lambda t: t.exit_time):
        equity += tr.pnl_usdt
        curve.append(equity)
        if tr.pnl_usdt > 0:
            gross_profit += tr.pnl_usdt
        elif tr.pnl_usdt < 0:
            gross_loss += abs(tr.pnl_usdt)

    peak = curve[0]
    max_dd = 0.0
    for v in curve:
        if v > peak:
            peak = v
        dd = (peak - v) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd

    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)
    return {
        "total_trades": float(len(trades)),
        "net_profit": equity - INITIAL_EQUITY_USDT,
        "max_drawdown_pct": max_dd * 100.0,
        "profit_factor": profit_factor,
    }


async def main():
    print("載入多幣種 1H 資料並並行回測...")
    print(f"  Symbols: {', '.join(SYMBOLS)}")
    print(f"  Params: K={SQUEEZE_K}, Trail={TRAILING_ATR_MULT}")

    all_trades = await fetch_and_backtest_all_symbols()
    print(f"  原始交易筆數（合併前風控）: {len(all_trades)}")

    accepted, skipped = apply_portfolio_risk_limits(all_trades)
    metrics = portfolio_metrics(accepted)

    print("\n" + "=" * 80)
    print("Portfolio Backtest Report")
    print("=" * 80)
    print(f"Total Trades:            {int(metrics['total_trades'])}")
    print(f"Skipped (risk limit):    {skipped}")
    print(f"Portfolio Net Profit:    {metrics['net_profit']:+.2f} USDT")
    print(f"Max Drawdown %:          {metrics['max_drawdown_pct']:.2f}%")
    pf = metrics["profit_factor"]
    pf_str = f"{pf:.2f}" if pf < 999 else ">999"
    print(f"Profit Factor:           {pf_str}")
    print(f"Max Concurrent Allowed:  {MAX_CONCURRENT}")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())

