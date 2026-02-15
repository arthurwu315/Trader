"""
樣本外驗證與參數穩定性測試
- 從 best_strategy.py 讀取參數，對 2024-01-01 至今做 OOS 回測
- 回報 2024 年周報酬率與最大回撤
- 對最佳參數做 +/- 5% 擾動，確認績效穩定性
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ["SKIP_CONFIG_VALIDATION"] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from backtest_engine_bnb import BacktestEngineBNB
from backtest_utils import fetch_klines_df
from bots.bot_c.strategy_bnb import StrategyBNB, ExitRules
from core.binance_client import BinanceFuturesClient

SYMBOL = "BNBUSDT"
INTERVAL = "1h"
FEE_BPS = 9.0
SLIPPAGE_BPS = 5.0
POSITION_SIZE = 0.02
MAX_TRADES_PER_DAY = 5
OOS_START = "2024-01-01"


def load_oos_data(start: str, end: str) -> pd.DataFrame:
    from bots.bot_c.config_c import get_strategy_c_config
    config = get_strategy_c_config()
    client = BinanceFuturesClient(
        api_key=config.binance_api_key or "dummy",
        api_secret=config.binance_api_secret or "dummy",
        base_url=os.getenv("BINANCE_DATA_URL", "https://fapi.binance.com"),
    )
    start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    df = fetch_klines_df(client, SYMBOL, INTERVAL, start_dt, end_dt)
    if df.empty or len(df) < 100:
        raise ValueError(f"OOS 資料不足: {start} ~ {end}，請檢查網路或 BACKTEST_OFFLINE 快取")
    return df


def extract_best_params():
    """從 best_strategy.py 讀取參數（解析檔案，避免相對 import）。"""
    best_py = ROOT / "bots" / "bot_c" / "best_strategy.py"
    params = {
        "funding_z_threshold": 1.75,
        "rsi_z_threshold": 1.88,
        "min_score": 2,
        "price_breakout_short": 1.0,
        "sl_atr_mult": 1.5,
        "tp_atr_mult": 2.5,
        "direction": "short",
    }
    if best_py.exists():
        text = best_py.read_text(encoding="utf-8")
        import re
        for key in ["funding_z_threshold", "rsi_z_threshold", "min_score", "sl_atr_mult", "tp_atr_mult"]:
            m = re.search(rf'["\']?{key}["\']?\s*:\s*([\d.]+)', text)
            if m:
                v = float(m.group(1)) if key != "min_score" else int(float(m.group(1)))
                params[key] = v
    return params


def build_strategy_from_params(p: dict) -> StrategyBNB:
    entry_th = {
        "funding_z_threshold": p["funding_z_threshold"],
        "rsi_z_threshold": p["rsi_z_threshold"],
        "min_score": p["min_score"],
        "price_breakout_short": p["price_breakout_short"],
    }
    exit_rules = ExitRules(
        tp_r_mult=2.0,
        tp_atr_mult=p["tp_atr_mult"],
        sl_atr_mult=p["sl_atr_mult"],
        trailing_stop_atr_mult=None,
        exit_after_bars=None,
        tp_fixed_pct=None,
    )
    return StrategyBNB(
        entry_thresholds=entry_th,
        exit_rules=exit_rules,
        position_size=POSITION_SIZE,
        direction=p["direction"],
        min_factors_required=p["min_score"],
    )


def run_oos_and_report(data_2024: pd.DataFrame, engine: BacktestEngineBNB, label: str, strategy: StrategyBNB):
    res = engine.run(strategy, data_2024, fee_bps=FEE_BPS, slippage_bps=SLIPPAGE_BPS)
    weekly = res.get("weekly_return_pct") or 0
    max_dd = res.get("max_drawdown_pct") or 0
    trades = res.get("trades_count", 0)
    total = res.get("total_return_pct") or 0
    print(f"  [{label}] 2024 周報酬率(%): {weekly:.2f}, 最大回撤(%): {max_dd:.2f}, 交易數: {trades}, 總報酬(%): {total:.2f}")
    return {"weekly_return_pct": weekly, "max_drawdown_pct": max_dd, "trades_count": trades, "total_return_pct": total}


def main():
    end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    print(f"樣本外驗證: {OOS_START} ~ {end_date}")
    print("載入 OOS 資料...")
    try:
        data = load_oos_data(OOS_START, end_date)
    except Exception as e:
        print(f"載入失敗: {e}")
        if os.getenv("BACKTEST_OFFLINE") == "1":
            print("BACKTEST_OFFLINE=1 時請確保 .cache 內有 BNBUSDT 1h 覆蓋 2024 至今")
        sys.exit(1)
    # 僅 2024 年用於報告
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    data_2024 = data[data["timestamp"].dt.year == 2024].copy()
    if data_2024.empty:
        data_2024 = data.copy()
    print(f"  K 線數: 全區間 {len(data)}, 2024 {len(data_2024)}")

    engine = BacktestEngineBNB(position_size_pct=POSITION_SIZE, max_trades_per_day=MAX_TRADES_PER_DAY)
    params = extract_best_params()
    strategy_baseline = build_strategy_from_params(params)

    print("\n--- 1. 樣本外驗證（2024）---")
    run_oos_and_report(data_2024, engine, "Baseline", strategy_baseline)

    print("\n--- 2. 參數穩定性（±5% 擾動，全 OOS 區間）---")
    baseline_res = run_oos_and_report(data, engine, "Baseline(全區間)", strategy_baseline)
    results = [("Baseline", params.copy(), baseline_res)]

    for name, delta in [
        ("funding_z -5%", ("funding_z_threshold", 0.95)),
        ("funding_z +5%", ("funding_z_threshold", 1.05)),
        ("rsi_z -5%", ("rsi_z_threshold", 0.95)),
        ("rsi_z +5%", ("rsi_z_threshold", 1.05)),
        ("sl_atr -5%", ("sl_atr_mult", 0.95)),
        ("sl_atr +5%", ("sl_atr_mult", 1.05)),
        ("tp_atr -5%", ("tp_atr_mult", 0.95)),
        ("tp_atr +5%", ("tp_atr_mult", 1.05)),
    ]:
        key, mult = delta
        p2 = params.copy()
        p2[key] = p2[key] * mult
        strat = build_strategy_from_params(p2)
        res = engine.run(strat, data, fee_bps=FEE_BPS, slippage_bps=SLIPPAGE_BPS)
        results.append((name, p2, res))
        wr = res.get("weekly_return_pct") or 0
        dd = res.get("max_drawdown_pct") or 0
        print(f"  [{name}] 周報酬: {wr:.2f}%, 最大回撤: {dd:.2f}%, 交易數: {res.get('trades_count', 0)}")

    print("\n--- 彙總 ---")
    base_weekly = baseline_res.get("weekly_return_pct") or 0
    base_dd = baseline_res.get("max_drawdown_pct") or 0
    print(f"2024 年周報酬率: {base_weekly:.2f}%, 最大回撤: {base_dd:.2f}%")
    stable = all(
        abs((r[2].get("weekly_return_pct") or 0) - base_weekly) < 2.0 and
        abs((r[2].get("max_drawdown_pct") or 0) - base_dd) < 3.0
        for r in results[1:]
    )
    print(f"參數穩定性: {'通過 (±5% 擾動後績效仍穩定)' if stable else '部分擾動下波動較大'}")


if __name__ == "__main__":
    main()
