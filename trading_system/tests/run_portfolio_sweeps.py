"""
多幣種 1D 宏觀策略槓桿壓力掃描 (Leverage Stress Sweep)

掃描網格：
- 單筆名目倉位占比 Notional%: (20%, 30%, 40%, 50%)
- 最大併發 Max Concurrent: (2, 3, 4, 5)

固定參數：
- Donchian N: 55
- EMA 慢線: 100
- ATR 突破濾網: 1.0x ATR
- Trailing ATR: 3.0
- ROC 視窗: 30 日
"""
from __future__ import annotations

import asyncio
import gc
import math
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

DONCHIAN_N = 55
EMA_SLOW = 100
ROC_WINDOW = 30

ATR_BREAK_FILTER = 1.0
TRAILING_ATR_MULT = 3.0
ATR_STOP_MULT_INIT = 2.5
INITIAL_EQUITY_USDT = 10000.0
NOTIONAL_PCT_VALS = (0.20, 0.30, 0.40, 0.50)
MAX_CONCURRENT_VALS = (2, 3, 4, 5)


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


def build_strategy() -> StrategyBNB:
    entry_th = {
        "macro_n": DONCHIAN_N,
        "ema_slow_period": EMA_SLOW,
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


def _run_single_symbol_backtest(symbol: str, df: pd.DataFrame, notional_pct: float) -> list[PortfolioTrade]:
    if df.empty:
        return []

    local_df = df.copy()
    roc_map = dict(zip(pd.to_datetime(local_df["timestamp"], utc=True), local_df["roc"]))

    strategy = build_strategy()
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
        pnl = INITIAL_EQUITY_USDT * notional_pct * (float(t.return_net_pct) / 100.0)
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


async def run_combo(
    data_map: dict[str, pd.DataFrame],
    notional_pct: float,
    max_concurrent: int,
) -> tuple[list[PortfolioTrade], dict[str, int]]:
    tasks = [
        asyncio.to_thread(
            _run_single_symbol_backtest,
            symbol,
            data_map.get(symbol, pd.DataFrame()),
            notional_pct,
        )
        for symbol in SYMBOLS
    ]
    result_lists = await asyncio.gather(*tasks)
    all_trades: list[PortfolioTrade] = []
    for lst in result_lists:
        all_trades.extend(lst)
    accepted, skipped = apply_portfolio_risk_limits(all_trades, max_concurrent=max_concurrent)
    return accepted, skipped


def apply_portfolio_risk_limits(
    all_trades: list[PortfolioTrade],
    max_concurrent: int,
) -> tuple[list[PortfolioTrade], dict[str, int]]:
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
        if len(active) >= max_concurrent:
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
            "max_drawdown_usdt": 0.0,
            "max_drawdown_pct": 0.0,
            "profit_factor": 0.0,
            "recovery_factor": 0.0,
            "annualized_return_pct": 0.0,
            "doubling_years": 999.0,
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
    max_dd_usdt = 0.0
    for v in curve:
        if v > peak:
            peak = v
        dd = (peak - v) / peak if peak > 0 else 0.0
        dd_usdt = peak - v
        if dd > max_dd:
            max_dd = dd
        if dd_usdt > max_dd_usdt:
            max_dd_usdt = dd_usdt

    pf = (gp / gl) if gl > 0 else (999.0 if gp > 0 else 0.0)
    years = max((pd.Timestamp(BACKTEST_END) - pd.Timestamp(BACKTEST_START)).days / 365.25, 1e-9)
    ending_equity = INITIAL_EQUITY_USDT + (equity - INITIAL_EQUITY_USDT)
    if ending_equity > 0:
        annualized = (ending_equity / INITIAL_EQUITY_USDT) ** (1.0 / years) - 1.0
    else:
        annualized = -1.0
    if annualized > 0:
        doubling_years = math.log(2.0) / math.log(1.0 + annualized)
    else:
        doubling_years = 999.0
    recovery = (equity - INITIAL_EQUITY_USDT) / max_dd_usdt if max_dd_usdt > 0 else 0.0
    return {
        "trades": float(len(trades)),
        "net_profit": equity - INITIAL_EQUITY_USDT,
        "max_drawdown_usdt": max_dd_usdt,
        "max_drawdown_pct": max_dd * 100.0,
        "profit_factor": pf,
        "recovery_factor": recovery,
        "annualized_return_pct": annualized * 100.0,
        "doubling_years": doubling_years,
    }


async def main():
    print("開始 1D Macro 槓桿壓力掃描 (Leverage Stress Sweep)")
    print(f"  Symbols: {', '.join(SYMBOLS)}")
    print(
        f"  Fixed Strategy: N={DONCHIAN_N}, EMA_SLOW={EMA_SLOW}, "
        f"ATR_BREAK={ATR_BREAK_FILTER}x, TRAIL={TRAILING_ATR_MULT}"
    )
    print(
        f"  Grid: Notional%={tuple(int(x * 100) for x in NOTIONAL_PCT_VALS)}, "
        f"MaxConcurrent={MAX_CONCURRENT_VALS}, ROC_WINDOW={ROC_WINDOW}"
    )
    print("  Loading data once for all symbols...")

    data_map = await load_all_symbol_data()
    for symbol in SYMBOLS:
        size = len(data_map.get(symbol, pd.DataFrame()))
        print(f"    {symbol}: {size} bars")

    rows: list[dict[str, float]] = []
    total = len(NOTIONAL_PCT_VALS) * len(MAX_CONCURRENT_VALS)
    idx = 0
    for notional_pct in NOTIONAL_PCT_VALS:
        for max_concurrent in MAX_CONCURRENT_VALS:
            idx += 1
            print(f"\n[{idx}/{total}] Notional={notional_pct*100:.0f}%, MaxConcurrent={max_concurrent}")
            accepted, skipped = await run_combo(
                data_map,
                notional_pct=notional_pct,
                max_concurrent=max_concurrent,
            )
            m = portfolio_metrics(accepted)
            rows.append({
                "notional_pct": float(notional_pct * 100.0),
                "max_concurrent": float(max_concurrent),
                "trades": m["trades"],
                "skipped": float(skipped["skipped_total"]),
                "net_profit": m["net_profit"],
                "max_dd": m["max_drawdown_pct"],
                "pf": m["profit_factor"],
                "recovery": m["recovery_factor"],
                "doubling_years": m["doubling_years"],
            })
            gc.collect()

    rows_sorted = sorted(rows, key=lambda r: r["net_profit"], reverse=True)
    print("\n" + "=" * 150)
    print("風險報酬矩陣 (Risk-Return Matrix)")
    print("=" * 150)
    print(
        f"{'Notional%':<12}{'MaxConc':<10}{'TotalTrades':<12}{'NetProfit$':<14}"
        f"{'MaxDD%':<10}{'Recovery':<10}{'PF':<8}{'Double(Y)':<10}"
    )
    print("-" * 150)
    for r in rows_sorted:
        pf_str = f"{r['pf']:.2f}" if r["pf"] < 999 else ">999"
        double_str = f"{r['doubling_years']:.2f}" if r["doubling_years"] < 999 else "N/A"
        print(
            f"{r['notional_pct']:<12.0f}{int(r['max_concurrent']):<10}{int(r['trades']):<12}"
            f"{r['net_profit']:+14.2f}{r['max_dd']:<10.2f}{r['recovery']:<10.2f}"
            f"{pf_str:<8}{double_str:<10}"
        )
    print("=" * 150)

    candidates = [r for r in rows_sorted if r["max_dd"] < 25.0]
    if candidates:
        best = max(candidates, key=lambda r: r["net_profit"])
        double_str = f"{best['doubling_years']:.2f}" if best["doubling_years"] < 999 else "N/A"
        print("\n⭐ 最佳平衡點 (NetProfit 最高且 MaxDD < 25%)：")
        print(
            f"  Notional={best['notional_pct']:.0f}% | MaxConcurrent={int(best['max_concurrent'])} | "
            f"Trades={int(best['trades'])} | NetProfit={best['net_profit']:+.2f} | "
            f"MaxDD={best['max_dd']:.2f}% | Recovery={best['recovery']:.2f} | "
            f"PF={best['pf']:.2f} | 預估翻倍時間={double_str} 年"
        )
    else:
        print("\n尚未發現 MaxDD < 25% 的組合。")


if __name__ == "__main__":
    asyncio.run(main())
