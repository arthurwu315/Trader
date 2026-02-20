"""
多幣種組合參數掃描 (Portfolio Sweep) - 1D Macro Trend Following

掃描網格：
- Donchian N: (20, 55, 80)
- EMA 慢線週期: (100, 200)
- ROC 視窗: 固定 30 日

固定參數：
- ATR 突破濾網: 1.0x ATR
- Trailing ATR: 3.0
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

DONCHIAN_N_VALS = (20, 55, 80)
EMA_SLOW_VALS = (100, 200)
ROC_WINDOW = 30

ATR_BREAK_FILTER = 1.0
TRAILING_ATR_MULT = 3.0
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


def build_strategy(donchian_n: int, ema_slow_period: int) -> StrategyBNB:
    entry_th = {
        "macro_n": donchian_n,
        "ema_slow_period": ema_slow_period,
        "atr_break_mult": ATR_BREAK_FILTER,
    }
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


def resample_1h_to_1d(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out = out.set_index("timestamp").sort_index()
    daily = (
        out.resample("1D", closed="right", label="right")
        .agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        })
        .dropna(subset=["open", "high", "low", "close"])
        .reset_index()
    )
    return daily


async def load_all_symbol_data() -> dict[str, pd.DataFrame]:
    tasks = [asyncio.to_thread(_fetch_symbol_df, symbol) for symbol in SYMBOLS]
    out: dict[str, pd.DataFrame] = {}
    for symbol, df in await asyncio.gather(*tasks):
        daily = resample_1h_to_1d(df)
        daily["roc"] = daily["close"].pct_change(ROC_WINDOW)
        out[symbol] = daily
    return out


def _run_single_symbol_backtest(symbol: str, df: pd.DataFrame, donchian_n: int, ema_slow_period: int) -> list[PortfolioTrade]:
    if df.empty:
        return []

    local_df = df.copy()
    roc_map = dict(zip(pd.to_datetime(local_df["timestamp"], utc=True), local_df["roc"]))

    strategy = build_strategy(donchian_n, ema_slow_period)
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


async def run_combo(data_map: dict[str, pd.DataFrame], donchian_n: int, ema_slow_period: int) -> tuple[list[PortfolioTrade], dict[str, int]]:
    tasks = [
        asyncio.to_thread(
            _run_single_symbol_backtest,
            symbol,
            data_map.get(symbol, pd.DataFrame()),
            donchian_n,
            ema_slow_period,
        )
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
        key = pd.Timestamp(tr.entry_time).floor("1D")
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
    print(f"  Grid: N={DONCHIAN_N_VALS}, EMA_SLOW={EMA_SLOW_VALS}, ROC_WINDOW={ROC_WINDOW}")
    print(
        f"  Fixed: ATR_BREAK={ATR_BREAK_FILTER}x, Trailing ATR={TRAILING_ATR_MULT}, "
        f"MAX_CONCURRENT={MAX_CONCURRENT}"
    )
    print("  Loading data once for all symbols...")

    data_map = await load_all_symbol_data()
    for symbol in SYMBOLS:
        size = len(data_map.get(symbol, pd.DataFrame()))
        print(f"    {symbol}: {size} bars")

    rows: list[dict[str, float]] = []
    total = len(DONCHIAN_N_VALS) * len(EMA_SLOW_VALS)
    idx = 0
    for donchian_n in DONCHIAN_N_VALS:
        for ema_slow in EMA_SLOW_VALS:
            idx += 1
            print(f"\n[{idx}/{total}] N={donchian_n}, EMA_SLOW={ema_slow}, ROC={ROC_WINDOW}")
            accepted, skipped = await run_combo(data_map, donchian_n, ema_slow)
            m = portfolio_metrics(accepted)
            rows.append({
                "n": float(donchian_n),
                "ema_slow": float(ema_slow),
                "roc": float(ROC_WINDOW),
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
    print(f"{'N':<6}{'EMA':<8}{'ROC':<8}{'Trades':<10}{'Skipped':<10}{'NetProfit$':<14}{'MaxDD%':<10}{'PF':<8}")
    print("-" * 120)
    for r in top10:
        pf_str = f"{r['pf']:.2f}" if r["pf"] < 999 else ">999"
        print(
            f"{int(r['n']):<6}{int(r['ema_slow']):<8}{int(r['roc']):<8}"
            f"{int(r['trades']):<10}{int(r['skipped']):<10}"
            f"{r['net_profit']:+14.2f}{r['max_dd']:<10.2f}{pf_str:<8}"
        )
    print("=" * 120)

    robust = [r for r in rows_sorted if r["pf"] >= 1.5]
    if robust:
        print("\n⭐ 發現 PF >= 1.5 的組合：")
        for r in robust[:5]:
            print(
                f"  N={int(r['n'])}, EMA={int(r['ema_slow'])}, ROC={int(r['roc'])} | "
                f"Trades={int(r['trades'])}, Skipped={int(r['skipped'])}, "
                f"NetProfit={r['net_profit']:+.2f}, MaxDD={r['max_dd']:.2f}%, PF={r['pf']:.2f}"
            )
    else:
        print("\n尚未發現 PF >= 1.5 的組合。")


if __name__ == "__main__":
    asyncio.run(main())
