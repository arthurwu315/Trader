"""
1H 波動率壓縮突破 (Volatility Squeeze) 參數掃描。
Squeeze = BBW < bbw_mean_K * 0.8；LONG = Squeeze 且 close > BB_UP 且 close > ema_100_1h；SHORT = Squeeze 且 close < BB_LOW 且 close < ema_100_1h。
出場：1h ATR Trailing Stop。
掃描 K=(50, 80)、Trail=(3.0, 4.0)。產出 Top 5：NetProfit, MaxDD, Trades, PF。
使用方式：cd /home/trader/trading_system && python3 -m tests.run_sweeps
"""
from __future__ import annotations

import gc
import os
import re
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
INITIAL_EQUITY_USDT = 10000.0

MIN_PROFIT_FACTOR = 1.0
BACKTEST_YEARS = 2.0

# 波動率壓縮突破：BBW 觀察期 K、Trailing ATR Mult
SQUEEZE_K_VALS = (50, 80)
TRAILING_ATR_MULT_VALS = (3.0, 4.0)
ATR_STOP_MULT_INIT = 3.0


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
    if df.empty or len(df) < 100:
        raise ValueError(f"1h 資料不足: {len(df)} 根。需至少 100 根（含 BBW mean 80）")
    return df


def build_strategy(squeeze_k: int, trailing_atr_mult: float) -> StrategyBNB:
    entry_th = {"squeeze_k": squeeze_k}
    exit_rules = ExitRules(
        tp_r_mult=None,
        tp_atr_mult=None,
        sl_atr_mult=ATR_STOP_MULT_INIT,
        trailing_stop_atr_mult=trailing_atr_mult,
        exit_after_bars=None,
        tp_fixed_pct=None,
    )
    return StrategyBNB(
        entry_thresholds=entry_th,
        exit_rules=exit_rules,
        position_size=POSITION_SIZE,
        direction="long",
        min_factors_required=2,
    )


def annualized_return_pct(total_return_pct: float, years: float = BACKTEST_YEARS) -> float:
    if total_return_pct <= -100:
        return -100.0
    return ((1 + total_return_pct / 100.0) ** (1.0 / years) - 1.0) * 100.0


def compute_metrics(res: dict) -> dict:
    curve = res.get("equity_curve")
    trades = res.get("trades", [])
    if curve is not None and hasattr(curve, "__len__") and len(curve) > 0:
        final_equity = float(curve[-1])
    else:
        final_equity = 1.0
    net_profit_usdt = (final_equity - 1.0) * INITIAL_EQUITY_USDT

    wins = [t for t in trades if t.return_net_pct > 0]
    losses = [t for t in trades if t.return_net_pct < 0]
    gross_profit = sum(t.return_net_pct for t in wins)
    gross_loss = abs(sum(t.return_net_pct for t in losses))
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = 999.0 if gross_profit > 0 else 0.0

    win_rate = (len(wins) / len(trades) * 100.0) if trades else 0.0
    max_dd = res.get("max_drawdown_pct", 0.0)
    avg_win_pct = (gross_profit / len(wins)) if wins else 0.0
    avg_loss_pct = (gross_loss / len(losses)) if losses else 0.0
    payoff_ratio = (avg_win_pct / avg_loss_pct) if avg_loss_pct > 0 else (999.0 if avg_win_pct > 0 else 0.0)

    return {
        "net_profit_usdt": net_profit_usdt,
        "profit_factor": profit_factor,
        "win_rate_pct": win_rate,
        "max_drawdown_pct": max_dd,
        "payoff_ratio": payoff_ratio,
        "trades_count": len(trades),
    }


def run_grid(data, engine):
    rows = []
    for squeeze_k in SQUEEZE_K_VALS:
        for trailing in TRAILING_ATR_MULT_VALS:
            gc.collect()
            strategy = build_strategy(squeeze_k, trailing)
            res = engine.run(
                strategy, data,
                fee_bps=FEE_BPS,
                slippage_bps=SLIPPAGE_BPS,
                reverse_side=False,
            )
            m = compute_metrics(res)
            total_ret = res.get("total_return_pct", 0.0)
            ann_ret = annualized_return_pct(total_ret)
            rows.append({
                "squeeze_k": squeeze_k,
                "trailing_atr_mult": trailing,
                "trades": m["trades_count"],
                "net_profit_usdt": m["net_profit_usdt"],
                "profit_factor": m["profit_factor"],
                "win_rate_pct": m["win_rate_pct"],
                "total_return_pct": total_ret,
                "annualized_return_pct": ann_ret,
                "max_drawdown_pct": m["max_drawdown_pct"],
                "payoff_ratio": m["payoff_ratio"],
            })
            gc.collect()
    return rows


def main():
    print("載入 1h 資料（波動率壓縮突破 Volatility Squeeze）...")
    data = load_data()
    print(f"  K 線數: {len(data)}")
    gc.collect()
    engine = BacktestEngineBNB(position_size_pct=POSITION_SIZE, max_trades_per_day=10)

    print("\n[1] 掃描：Squeeze K × Trailing ATR Mult（K=50,80；Trail=3.0,4.0）...")
    rows = run_grid(data, engine)
    gc.collect()

    rows_sorted = sorted(rows, key=lambda r: r["net_profit_usdt"], reverse=True)
    top5 = rows_sorted[:5]

    print("\n" + "=" * 110)
    print("Top 5 組合 (1H Squeeze + EMA100 趨勢濾網) — NetProfit, MaxDD, Trades, PF")
    print("=" * 110)
    print(f"{'K':<5} {'Trail':<6} {'Trades':<7} {'NetProfit$':<11} {'PF':<6} {'MaxDD%':<7} {'AnnRet%':<8} {'Payoff':<6}")
    print("-" * 110)
    for r in top5:
        pf_str = f"{r['profit_factor']:.2f}" if r["profit_factor"] < 999 else ">999"
        payoff_str = f"{r['payoff_ratio']:.2f}" if r["payoff_ratio"] < 999 else ">999"
        print(
            f"{r['squeeze_k']:<5} {r['trailing_atr_mult']:<6.1f} {r['trades']:<7} {r['net_profit_usdt']:>+10.2f}   {pf_str:<6} "
            f"{r['max_drawdown_pct']:>5.1f}   {r['annualized_return_pct']:>6.1f}   {payoff_str:<6}"
        )
    print("=" * 110)

    best = rows_sorted[0]
    print(f"\n最佳組合: K={best['squeeze_k']}, Trailing={best['trailing_atr_mult']} | "
          f"Trades={best['trades']} NetProfit={best['net_profit_usdt']:+.2f} PF={best['profit_factor']:.2f} MaxDD={best['max_drawdown_pct']:.1f}%")
    if best["net_profit_usdt"] > 0:
        print("Net Profit 已轉正。")
    else:
        print("Net Profit 尚未轉正，可調整 K / Trailing 或區間再測。")

    print("\n回測掃描完成。")


if __name__ == "__main__":
    main()
