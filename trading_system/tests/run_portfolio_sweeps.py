"""
多幣種組合參數掃描 (Portfolio Sweep)

掃描網格：
- K: (100, 120, 150)
- VOL_MULT: (1.3, 1.5, 1.8)
- ROC_WINDOW: (12, 24, 36)

固定參數：
- Trailing ATR: 2.5
- MAX_CONCURRENT: 2
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

import pandas as pd

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

K_VALS = (100, 120, 150)
VOL_MULT_VALS = (1.3, 1.5, 1.8)
ROC_WINDOW_VALS = (12, 24, 36)

TRAILING_ATR_MULT = 2.5
ATR_STOP_MULT_INIT = 2.5
MAX_CONCURRENT = 2
INITIAL_EQUITY_USDT = 10000.0
NOTIONAL_PER_TRADE = INITIAL_EQUITY_USDT * 0.2  # 2000 USDT


@dataclass
class PortfolioTrade:
    symbol: str
    side: str
    entry_time: Any
    exit_time: Any
    roc: float
    pnl_usdt: float


def build_client():
    from bots.bot_c.config_c import get_strategy_c_config
    from core.binance_client import BinanceFuturesClient

    config = get_strategy_c_config()
    return BinanceFuturesClient(
        api_key=config.binance_api_key or "dummy",
        api_secret=config.binance_api_secret or "dummy",
        base_url=os.getenv("BINANCE_DATA_URL", "https://fapi.binance.com"),
    )


def build_strategy(k: int, vol_mult: float) -> StrategyBNB:
    entry_th = {"squeeze_k": k, "vol_mult": vol_mult}
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


def _fetch_symbol_df(symbol: str) -> tuple[str, pd.DataFrame]:
    start_dt = datetime.strptime(BACKTEST_START, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(BACKTEST_END, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    client = build_client()
    df = fetch_klines_df(client, symbol, INTERVAL, start_dt, end_dt)
    return symbol, df


async def load_all_symbol_data() -> dict[str, pd.DataFrame]:
    tasks = [asyncio.to_thread(_fetch_symbol_df, symbol) for symbol in SYMBOLS]
    out: dict[str, pd.DataFrame] = {}
    for symbol, df in await asyncio.gather(*tasks):
        out[symbol] = df
    return out


def _run_single_symbol_backtest(symbol: str, df: pd.DataFrame, k: int, vol_mult: float, roc_window: int) -> list[PortfolioTrade]:
    if df.empty:
        return []

    local_df = df.copy()
    local_df["roc"] = local_df["close"].pct_change(roc_window)
    roc_map = dict(zip(pd.to_datetime(local_df["timestamp"], utc=True), local_df["roc"]))

    strategy = build_strategy(k, vol_mult)
    engine = BacktestEngineBNB(position_size_pct=0.02, max_trades_per_day=10)
    res = engine.run(
        strategy,
        local_df,
        fee_bps=FEE_BPS,
        slippage_bps=SLIPPAGE_BPS,
        reverse_side=False,
    )

    out: list[PortfolioTrade] = []
    for t in res.get("trades", []):
        entry_ts = pd.Timestamp(t.entry_time)
        entry_ts = entry_ts.tz_convert("UTC") if entry_ts.tzinfo else entry_ts.tz_localize("UTC")
        roc_val = roc_map.get(entry_ts, 0.0)
        roc_val = 0.0 if pd.isna(roc_val) else float(roc_val)
        pnl = NOTIONAL_PER_TRADE * (float(t.return_net_pct) / 100.0)
        out.append(
            PortfolioTrade(
                symbol=symbol,
                side=str(t.side).upper(),
                entry_time=t.entry_time,
                exit_time=t.exit_time,
                roc=roc_val,
                pnl_usdt=pnl,
            )
        )

    del local_df
    del res
    gc.collect()
    return out


async def run_combo(data_map: dict[str, pd.DataFrame], k: int, vol_mult: float, roc_window: int) -> tuple[list[PortfolioTrade], dict[str, int]]:
    tasks = [
        asyncio.to_thread(_run_single_symbol_backtest, symbol, data_map.get(symbol, pd.DataFrame()), k, vol_mult, roc_window)
        for symbol in SYMBOLS
    ]
    result_lists = await asyncio.gather(*tasks)
    all_trades: list[PortfolioTrade] = []
    for lst in result_lists:
        all_trades.extend(lst)
    accepted, skipped = apply_portfolio_risk_limits(all_trades)
    return accepted, skipped


def apply_portfolio_risk_limits(all_trades: list[PortfolioTrade]) -> tuple[list[PortfolioTrade], dict[str, int]]:
    trades = sorted(all_trades, key=lambda t: t.entry_time)
    buckets: dict[pd.Timestamp, list[PortfolioTrade]] = {}
    for tr in trades:
        key = pd.Timestamp(tr.entry_time).floor("2h")
        buckets.setdefault(key, []).append(tr)

    rs_selected: list[PortfolioTrade] = []
    skipped_by_rs = 0
    for _, batch in sorted(buckets.items(), key=lambda kv: kv[0]):
        longs = [t for t in batch if t.side == "BUY"]
        shorts = [t for t in batch if t.side == "SELL"]
        if longs:
            rs_selected.append(max(longs, key=lambda t: t.roc))   # 做多選最強
            skipped_by_rs += max(0, len(longs) - 1)
        if shorts:
            rs_selected.append(min(shorts, key=lambda t: t.roc))  # 做空選最弱
            skipped_by_rs += max(0, len(shorts) - 1)

    rs_selected.sort(key=lambda t: t.entry_time)
    accepted: list[PortfolioTrade] = []
    active: list[PortfolioTrade] = []
    skipped_by_concurrent = 0
    for tr in rs_selected:
        active = [a for a in active if a.exit_time > tr.entry_time]
        if len(active) >= MAX_CONCURRENT:
            skipped_by_concurrent += 1
            continue
        accepted.append(tr)
        active.append(tr)

    return accepted, {
        "skipped_by_rs": skipped_by_rs,
        "skipped_by_concurrent": skipped_by_concurrent,
        "skipped_total": skipped_by_rs + skipped_by_concurrent,
    }


def portfolio_metrics(trades: list[PortfolioTrade]) -> dict[str, float]:
    if not trades:
        return {
            "trades": 0.0,
            "net_profit": 0.0,
            "max_drawdown_pct": 0.0,
            "profit_factor": 0.0,
        }

    equity = INITIAL_EQUITY_USDT
    curve = [equity]
    gp = 0.0
    gl = 0.0
    for tr in sorted(trades, key=lambda t: t.exit_time):
        equity += tr.pnl_usdt
        curve.append(equity)
        if tr.pnl_usdt > 0:
            gp += tr.pnl_usdt
        elif tr.pnl_usdt < 0:
            gl += abs(tr.pnl_usdt)

    peak = curve[0]
    max_dd = 0.0
    for v in curve:
        if v > peak:
            peak = v
        dd = (peak - v) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd

    pf = (gp / gl) if gl > 0 else (999.0 if gp > 0 else 0.0)
    return {
        "trades": float(len(trades)),
        "net_profit": equity - INITIAL_EQUITY_USDT,
        "max_drawdown_pct": max_dd * 100.0,
        "profit_factor": pf,
    }


async def main():
    print("開始組合層級參數掃描 (Portfolio Sweep)")
    print(f"  Symbols: {', '.join(SYMBOLS)}")
    print(f"  Grid: K={K_VALS}, VOL_MULT={VOL_MULT_VALS}, ROC_WINDOW={ROC_WINDOW_VALS}")
    print(f"  Fixed: Trailing ATR={TRAILING_ATR_MULT}, MAX_CONCURRENT={MAX_CONCURRENT}")
    print("  Loading data once for all symbols...")

    data_map = await load_all_symbol_data()
    for symbol in SYMBOLS:
        size = len(data_map.get(symbol, pd.DataFrame()))
        print(f"    {symbol}: {size} bars")

    rows: list[dict[str, float]] = []
    total = len(K_VALS) * len(VOL_MULT_VALS) * len(ROC_WINDOW_VALS)
    idx = 0
    for k in K_VALS:
        for vol_mult in VOL_MULT_VALS:
            for roc_window in ROC_WINDOW_VALS:
                idx += 1
                print(f"\n[{idx}/{total}] K={k}, Vol={vol_mult}, ROC={roc_window}")
                accepted, skipped = await run_combo(data_map, k, vol_mult, roc_window)
                m = portfolio_metrics(accepted)
                rows.append({
                    "k": float(k),
                    "vol": float(vol_mult),
                    "roc": float(roc_window),
                    "trades": m["trades"],
                    "skipped": float(skipped["skipped_total"]),
                    "net_profit": m["net_profit"],
                    "max_dd": m["max_drawdown_pct"],
                    "pf": m["profit_factor"],
                })
                gc.collect()

    rows_sorted = sorted(rows, key=lambda r: r["net_profit"], reverse=True)
    top10 = rows_sorted[:10]

    print("\n" + "=" * 120)
    print("Top 10 組合 (依 Portfolio Net Profit 降序)")
    print("=" * 120)
    print(f"{'K':<6}{'Vol':<8}{'ROC':<8}{'Trades':<10}{'Skipped':<10}{'NetProfit$':<14}{'MaxDD%':<10}{'PF':<8}")
    print("-" * 120)
    for r in top10:
        pf_str = f"{r['pf']:.2f}" if r["pf"] < 999 else ">999"
        print(
            f"{int(r['k']):<6}{r['vol']:<8.1f}{int(r['roc']):<8}"
            f"{int(r['trades']):<10}{int(r['skipped']):<10}"
            f"{r['net_profit']:+14.2f}{r['max_dd']:<10.2f}{pf_str:<8}"
        )
    print("=" * 120)

    robust = [
        r for r in rows_sorted
        if r["pf"] >= 1.0 and r["max_dd"] <= 5.0 and r["trades"] >= 50
    ]
    if robust:
        print("\n⭐ 發現穩健正期望組合：")
        for r in robust[:5]:
            print(
                f"  K={int(r['k'])}, Vol={r['vol']:.1f}, ROC={int(r['roc'])} | "
                f"Trades={int(r['trades'])}, Skipped={int(r['skipped'])}, "
                f"NetProfit={r['net_profit']:+.2f}, MaxDD={r['max_dd']:.2f}%, PF={r['pf']:.2f}"
            )
    else:
        print("\n尚未發現符合條件 (PF>=1.0 且 MaxDD<=5.0% 且 Trades>=50) 的組合。")


if __name__ == "__main__":
    asyncio.run(main())
