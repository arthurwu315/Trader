"""
BNB/USDT 策略批量生成與篩選
- 定義 20–50 組入場門檻、3–5 種出場規則
- 固定回測 engine、固定倉位 2%、手續費後盈利、日交易 ≤5、周報酬 ≥1%、最大回撤 ≤10%
- Walk-forward + Monte Carlo 驗證
- 輸出符合條件的策略清單與回測結果
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

# 回測僅取 K 線，不需通過 Strategy C 完整配置驗證
os.environ["SKIP_CONFIG_VALIDATION"] = "1"

import pandas as pd

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backtest_engine_bnb import BacktestEngineBNB  # noqa: E402
from backtest_utils import fetch_klines_df  # noqa: E402
from bots.bot_c.strategy_bnb import StrategyBNB, ExitRules  # noqa: E402
from core.binance_client import BinanceFuturesClient  # noqa: E402


# ---------- 固定參數（不隨策略變體改動）----------
POSITION_SIZE = 0.02
MAX_TRADES_PER_DAY = 5
FEE_BPS = 9.0   # 往返約 0.09%
SLIPPAGE_BPS = 5.0
BACKTEST_START = "2022-01-01"
BACKTEST_END = "2023-12-31"
SYMBOL = "BNBUSDT"
INTERVAL = "1h"

# 篩選條件
MIN_WEEKLY_RETURN_PCT = 1.0
MAX_DRAWDOWN_PCT = 10.0
MAX_TRADES_PER_DAY_LIMIT = 5.0


def build_entry_threshold_sets(quick: bool = False) -> List[Dict[str, float]]:
    """20–50 組入場門檻（僅改門檻，因子名稱固定）. 做多策略。"""
    sets = []
    funding_thresholds = [-0.006, -0.005, -0.004, -0.003, -0.002, -0.001, 0.0]
    oi_thresholds = [1.0, 1.1, 1.2, 1.3]
    vol_thresholds = [0.005, 0.008, 0.01, 0.015, 0.02]
    if quick:
        funding_thresholds = [-0.004, -0.002, 0.0]
        oi_thresholds = [1.0, 1.2]
        vol_thresholds = [0.008, 0.01]

    for f in funding_thresholds:
        for o in oi_thresholds:
            for v in vol_thresholds:
                sets.append({
                    "funding_rate_proxy": f,
                    "oi_proxy": o,
                    "volatility": v,
                    "price_breakout_long": 1.0,
                })
                if len(sets) >= (20 if quick else 50):
                    return sets
    # 補足：僅 3 因子（不含 funding）以增加訊號機會
    for o in (oi_thresholds if quick else [1.0, 1.1, 1.2]):
        for v in (vol_thresholds if quick else [0.005, 0.01, 0.015]):
            sets.append({
                "oi_proxy": o,
                "volatility": v,
                "price_breakout_long": 1.0,
            })
            if len(sets) >= (20 if quick else 50):
                return sets
    return sets[:50]


def build_exit_rules(quick: bool = False) -> List[ExitRules]:
    """3–5 種出場規則組合."""
    rules = [
        ExitRules(tp_r_mult=1.5, sl_atr_mult=1.5),
        ExitRules(tp_r_mult=2.0, sl_atr_mult=1.5),
        ExitRules(tp_r_mult=2.5, sl_atr_mult=1.5),
        ExitRules(tp_r_mult=2.0, sl_atr_mult=2.0, exit_after_bars=24),
        ExitRules(tp_r_mult=2.0, sl_atr_mult=1.5, tp_fixed_pct=0.01),
    ]
    return rules[:2] if quick else rules


def load_data(client, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    df = fetch_klines_df(client, SYMBOL, INTERVAL, start_dt, end_dt)
    if df.empty or len(df) < 100:
        raise ValueError("BNB 1h 資料不足，請檢查網路或 BACKTEST_OFFLINE 快取")
    return df


def run_screen(
    data: pd.DataFrame,
    entry_sets: List[Dict[str, float]],
    exit_rules_list: List[ExitRules],
    engine: BacktestEngineBNB,
    relax: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    回傳 (符合全部條件的策略, 全部有交易且盈利的原始結果用於排序/除錯)。
    relax=True 時不篩選，僅按 weekly_return_pct 排序回傳前 50 筆原始結果。
    """
    qualified = []
    all_profitable = []
    for entry_th in entry_sets:
        for exit_rules in exit_rules_list:
            min_factors = len(entry_th) if all(
                k in entry_th for k in ("funding_rate_proxy", "oi_proxy", "volatility", "price_breakout_long")
            ) else len(entry_th)
            strategy = StrategyBNB(
                entry_thresholds=entry_th,
                exit_rules=exit_rules,
                position_size=POSITION_SIZE,
                direction="long",
                min_factors_required=min_factors,
            )
            res = engine.run(strategy, data, fee_bps=FEE_BPS, slippage_bps=SLIPPAGE_BPS)
            if res.get("skip_reason") or res.get("trades_count", 0) == 0:
                continue
            weekly = res.get("weekly_return_pct") or 0
            dd = res.get("max_drawdown_pct") or 0
            tpd = res.get("trades_per_day_avg") or 0
            profitable = res.get("profitable_after_fees", False)
            rec = {
                "entry_thresholds": str(entry_th),
                "exit_rules": f"tp_r={exit_rules.tp_r_mult} sl_atr={exit_rules.sl_atr_mult} exit_bars={exit_rules.exit_after_bars} tp_fixed={exit_rules.tp_fixed_pct}",
                "total_return_pct": res["total_return_pct"],
                "sharpe": res["sharpe"],
                "max_drawdown_pct": res["max_drawdown_pct"],
                "trades_count": res["trades_count"],
                "trades_per_day_avg": res["trades_per_day_avg"],
                "weekly_return_pct": res["weekly_return_pct"],
                "win_rate_pct": res["win_rate_pct"],
                "strategy": strategy,
                "res": res,
            }
            if profitable:
                all_profitable.append(rec)
            if not profitable:
                continue
            all_profitable.append(rec)
            if relax:
                qualified.append(rec)
                continue
            if weekly < MIN_WEEKLY_RETURN_PCT:
                continue
            if dd > MAX_DRAWDOWN_PCT:
                continue
            if tpd > MAX_TRADES_PER_DAY_LIMIT:
                continue
            qualified.append(rec)
    if relax and all_profitable:
        all_profitable.sort(key=lambda x: x["weekly_return_pct"], reverse=True)
        qualified = all_profitable[:50]
    return qualified, all_profitable


def main():
    parser = argparse.ArgumentParser(description="BNB/USDT 策略批量篩選")
    parser.add_argument("--start", default=BACKTEST_START, help="回測開始 YYYY-MM-DD")
    parser.add_argument("--end", default=BACKTEST_END, help="回測結束 YYYY-MM-DD")
    parser.add_argument("--wf", action="store_true", help="對通過策略跑 Walk-forward")
    parser.add_argument("--mc", type=int, default=0, help="Monte Carlo 模擬次數，0 不跑")
    parser.add_argument("--out", type=str, default="", help="輸出 CSV 路徑")
    parser.add_argument("--quick", action="store_true", help="快速模式：較少入場/出場組合")
    parser.add_argument("--relax", action="store_true", help="放寬：輸出盈利策略中周報酬最高者（不強制周≥1%%、回撤≤10%%）")
    args = parser.parse_args()

    start_dt = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    # 使用現有 config 的 API（僅取數用）
    from bots.bot_c.config_c import get_strategy_c_config
    config = get_strategy_c_config()
    data_base_url = os.getenv("BINANCE_DATA_URL", "https://fapi.binance.com")
    client = BinanceFuturesClient(
        api_key=config.binance_api_key or "dummy",
        api_secret=config.binance_api_secret or "dummy",
        base_url=data_base_url,
    )

    print("載入 BNB 1h 資料...")
    data = load_data(client, start_dt, end_dt)
    print(f"  K 線數: {len(data)}, 區間: {data['timestamp'].min()} ~ {data['timestamp'].max()}")

    engine = BacktestEngineBNB(
        position_size_pct=POSITION_SIZE,
        max_trades_per_day=MAX_TRADES_PER_DAY,
    )
    entry_sets = build_entry_threshold_sets(quick=args.quick)
    exit_rules_list = build_exit_rules(quick=args.quick)
    print(f"入場組合數: {len(entry_sets)}, 出場組合數: {len(exit_rules_list)}")

    print("執行回測篩選...")
    qualified, all_profitable = run_screen(
        data, entry_sets, exit_rules_list, engine, relax=args.relax
    )
    if args.relax:
        print(f"放寬模式：顯示盈利策略數（按周報酬排序）: {len(qualified)}")
    else:
        print(f"符合條件的策略數: {len(qualified)}")
        if not qualified and all_profitable:
            all_profitable.sort(key=lambda x: x["weekly_return_pct"], reverse=True)
            print(f"（盈利但未達門檻的策略數: {len(all_profitable)}，可用 --relax 檢視）")

    if not qualified:
        print("無符合條件的策略，請放寬門檻或調整出場規則。")
        if args.out:
            pd.DataFrame([]).to_csv(args.out, index=False)
        return

    # 報表
    rows = []
    for i, q in enumerate(qualified):
        row = {
            "id": i + 1,
            "entry_thresholds": q["entry_thresholds"],
            "exit_rules": q["exit_rules"],
            "total_return_pct": round(q["total_return_pct"], 4),
            "sharpe": round(q["sharpe"], 4),
            "max_drawdown_pct": round(q["max_drawdown_pct"], 4),
            "trades_count": q["trades_count"],
            "trades_per_day_avg": round(q["trades_per_day_avg"], 4),
            "weekly_return_pct": round(q["weekly_return_pct"], 4),
            "win_rate_pct": round(q["win_rate_pct"], 2),
        }
        if args.wf:
            wf = engine.walk_forward(q["strategy"], data, fee_bps=FEE_BPS, slippage_bps=SLIPPAGE_BPS)
            row["wf_oos_weekly_return_avg"] = round(wf.get("oos_weekly_return_avg", 0), 4)
            row["wf_oos_max_dd_avg"] = round(wf.get("oos_max_dd_avg", 0), 4)
        if args.mc > 0:
            mc = engine.monte_carlo(q["strategy"], data, fee_bps=FEE_BPS, slippage_bps=SLIPPAGE_BPS, n_simulations=args.mc)
            row["mc_mean_weekly_pct"] = round(mc.get("mean_weekly_return_pct", 0), 4)
            row["mc_p5_weekly_pct"] = round(mc.get("p5_weekly_return_pct", 0), 4)
            row["mc_p95_weekly_pct"] = round(mc.get("p95_weekly_return_pct", 0), 4)
            row["mc_mean_max_dd_pct"] = round(mc.get("mean_max_dd_pct", 0), 4)
        rows.append(row)

    report = pd.DataFrame(rows)
    print("\n" + "=" * 80)
    print("符合條件的策略清單（扣費後盈利、日交易≤5、周報酬≥1%、最大回撤≤10%）")
    print("=" * 80)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    print(report.to_string(index=False))

    if args.out:
        report.to_csv(args.out, index=False)
        print(f"\n已寫入: {args.out}")

    print("\n建議：回測通過後請小資金 live 驗證。")


if __name__ == "__main__":
    main()
