"""
遞迴優化引擎 (orchestrator) - 自適應趨勢與風控升級
- 目標函數：卡瑪比率 Calmar = Annual Return / Max Drawdown
- 三因子投票 + EMA200 趨勢過濾；min_score [1, 2]；base_sl_atr_mult [1.0, 2.0] 追蹤止損
- 兩年內交易 < 20 筆：評分 -10 懲罰；報告檔名含時間戳記
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ["SKIP_CONFIG_VALIDATION"] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TESTS = Path(__file__).resolve().parent
LOG_DIR = TESTS / "logs"
BOT_C = ROOT / "bots" / "bot_c"
VARIANTS_JSON = BOT_C / "variants.json"
BEST_STRATEGY_PY = BOT_C / "best_strategy.py"

BATCH_SIZE = 5
MIN_DD_FOR_RATIO = 0.1
POSITION_SIZE = 0.02
MAX_TRADES_PER_DAY = 5
FEE_BPS = 9.0
SLIPPAGE_BPS = 5.0
BACKTEST_START = "2022-01-01"
BACKTEST_END = "2023-12-31"
USE_2024_OOS = os.environ.get("USE_2024_OOS", "0") == "1"
if USE_2024_OOS:
    BACKTEST_START = "2024-01-01"
    BACKTEST_END = datetime.now(timezone.utc).strftime("%Y-%m-%d")
MAX_RECURSION = 3
if USE_2024_OOS:
    DIRECTION_VALS = ["long"]
    MIN_SCORE_VALS = [1, 2]
MIN_STEP_Z = 0.1
SLEEP_BETWEEN_BATCHES = 3
SLEEP_WHEN_LOAD_HIGH = 8
LOAD_AVG_THRESHOLD = 2.0

# 放寬 Z 門檻搜尋範圍（原本 1.5 太高易 0 交易）
FUNDING_Z_MIN = 0.5
FUNDING_Z_MAX = 2.0
RSI_Z_MIN = 0.5
RSI_Z_MAX = 2.0
STEP_FUNDING_Z_INIT = 0.5   # 第一輪 → 4 點 (0.5,1,1.5,2)
STEP_RSI_Z_INIT = 0.5       # 第一輪 → 4 點
MIN_SCORE_VALS = [1, 2]     # 進場門檻：總分 >= min_score

# 最低交易頻率：少於此筆數直接負分（防呆：零交易陷阱）
MIN_TRADES_FOR_VALID = 20
PENALTY_RATIO_IF_LOW_TRADES = -10.0

# 動態止損：基礎 ATR 倍數搜尋範圍 [1.0, 2.0]
BASE_SL_ATR_MIN = 1.0
BASE_SL_ATR_MAX = 2.0
STEP_SL_ATR = 0.5  # 1.0, 1.5, 2.0
BASE_SL_VALS = [1.0, 1.5, 2.0]

# 牛熊分治：short / both / regime（regime 時價格 > EMA200 用較低 min_score 做多）
DIRECTION_VALS = ["short", "both", "regime"]
# 快速止盈：止盈 ATR 倍數 [1.5, 2.5, 3.5]
TP_ATR_VALS = [1.5, 2.5, 3.5]

# 固定
FIXED_OI = 1.2
FIXED_VOLATILITY = 0.01
FIXED_TP_R_MULT = 2.0  # 當未使用 tp_atr_mult 時之預設


def get_report_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M")


def get_load_average() -> float:
    try:
        with open("/proc/loadavg", "r") as f:
            return float(f.read().split()[0])
    except Exception:
        return 0.0


def build_candidates_round1() -> List[Tuple[float, float, int, float, float, str]]:
    """第一輪：funding_z、rsi_z、min_score、base_sl、tp_atr_mult、direction in [short, both]"""
    fz_vals = []
    x = FUNDING_Z_MIN
    while x <= FUNDING_Z_MAX:
        fz_vals.append(round(x, 2))
        x += STEP_FUNDING_Z_INIT
    rz_vals = []
    x = RSI_Z_MIN
    while x <= RSI_Z_MAX:
        rz_vals.append(round(x, 2))
        x += STEP_RSI_Z_INIT
    out = []
    for fz in fz_vals:
        for rz in rz_vals:
            for ms in MIN_SCORE_VALS:
                for base_sl in BASE_SL_VALS:
                    for tp_atr in TP_ATR_VALS:
                        for d in DIRECTION_VALS:
                            out.append((fz, rz, ms, base_sl, tp_atr, d))
    return out


def build_candidates_around_center(
    center_fz: float,
    center_rz: float,
    center_min_score: int,
    center_base_sl: float,
    center_tp_atr: float,
    center_direction: str,
    step_fz: float,
    step_rz: float,
) -> List[Tuple[float, float, int, float, float, str]]:
    """以中心為準、同 min_score / base_sl / tp_atr / 方向，步長內 5 組"""
    candidates = [
        (center_fz, center_rz, center_min_score, center_base_sl, center_tp_atr, center_direction),
        (min(FUNDING_Z_MAX, center_fz + step_fz), center_rz, center_min_score, center_base_sl, center_tp_atr, center_direction),
        (max(FUNDING_Z_MIN, center_fz - step_fz), center_rz, center_min_score, center_base_sl, center_tp_atr, center_direction),
        (center_fz, min(RSI_Z_MAX, center_rz + step_rz), center_min_score, center_base_sl, center_tp_atr, center_direction),
        (center_fz, max(RSI_Z_MIN, center_rz - step_rz), center_min_score, center_base_sl, center_tp_atr, center_direction),
    ]
    return list(dict.fromkeys([(round(a, 2), round(b, 2), c, d, e, f) for a, b, c, d, e, f in candidates]))


def load_data():
    from run_bnb_strategy_screen import load_data as _load_data
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
    return _load_data(client, start_dt, end_dt)


def params_to_variant(
    funding_z: float,
    rsi_z: float,
    min_score: int,
    base_sl_atr: float,
    tp_atr_mult: float,
    direction: str,
) -> Tuple[Dict[str, float], Any]:
    from bots.bot_c.strategy_bnb import ExitRules
    if direction == "regime":
        entry_thresholds = {
            "funding_z_long": funding_z,
            "rsi_z_long": rsi_z,
            "min_score_long": 1,
            "funding_z_short": funding_z,
            "rsi_z_short": rsi_z,
            "min_score_short": 2,
            "price_breakout_long": 1.0,
            "price_breakout_short": 1.0,
        }
    else:
        entry_thresholds = {
            "funding_z_threshold": funding_z,
            "rsi_z_threshold": rsi_z,
            "min_score": min_score,
        }
        if direction == "long":
            entry_thresholds["price_breakout_long"] = 1.0
        else:
            entry_thresholds["price_breakout_short"] = 1.0
    exit_rules = ExitRules(
        tp_r_mult=FIXED_TP_R_MULT,
        tp_atr_mult=tp_atr_mult,
        sl_atr_mult=base_sl_atr,
        trailing_stop_atr_mult=None,
        exit_after_bars=None,
        tp_fixed_pct=None,
    )
    return entry_thresholds, exit_rules


def run_single(
    data,
    engine,
    funding_z: float,
    rsi_z: float,
    min_score: int,
    base_sl_atr: float,
    tp_atr_mult: float,
    direction: str,
) -> Dict[str, Any]:
    from bots.bot_c.strategy_bnb import StrategyBNB
    if direction == "regime":
        entry_th, exit_rules = params_to_variant(funding_z, rsi_z, min_score, base_sl_atr, tp_atr_mult, direction)
        strat = StrategyBNB(
            entry_thresholds=entry_th,
            exit_rules=exit_rules,
            position_size=POSITION_SIZE,
            direction="regime",
            min_factors_required=1,
        )
        res = engine.run(strat, data, fee_bps=FEE_BPS, slippage_bps=SLIPPAGE_BPS)
        total_return_pct = res.get("total_return_pct") or 0
        max_dd = res.get("max_drawdown_pct") or 0
        return {
            "funding_z_threshold": funding_z,
            "rsi_z_threshold": rsi_z,
            "min_score": min_score,
            "base_sl_atr_mult": base_sl_atr,
            "tp_atr_mult": tp_atr_mult,
            "direction": direction,
            "result": res,
            "total_return_pct": total_return_pct,
            "weekly_return_pct": res.get("weekly_return_pct") or 0,
            "max_drawdown_pct": max_dd,
            "trades_count": res.get("trades_count", 0),
            "profitable": res.get("profitable_after_fees", False),
        }
    if direction == "both":
        # 多空並行：各跑一次，合併報酬與取最大回撤
        entry_th_long, exit_rules = params_to_variant(funding_z, rsi_z, min_score, base_sl_atr, tp_atr_mult, "long")
        entry_th_short, _ = params_to_variant(funding_z, rsi_z, min_score, base_sl_atr, tp_atr_mult, "short")
        strat_long = StrategyBNB(
            entry_thresholds=entry_th_long,
            exit_rules=exit_rules,
            position_size=POSITION_SIZE,
            direction="long",
            min_factors_required=min_score,
        )
        strat_short = StrategyBNB(
            entry_thresholds=entry_th_short,
            exit_rules=exit_rules,
            position_size=POSITION_SIZE,
            direction="short",
            min_factors_required=min_score,
        )
        res_long = engine.run(strat_long, data, fee_bps=FEE_BPS, slippage_bps=SLIPPAGE_BPS)
        res_short = engine.run(strat_short, data, fee_bps=FEE_BPS, slippage_bps=SLIPPAGE_BPS)
        total_return_pct = (res_long.get("total_return_pct") or 0) + (res_short.get("total_return_pct") or 0)
        max_dd = max(res_long.get("max_drawdown_pct") or 0, res_short.get("max_drawdown_pct") or 0)
        trades_count = (res_long.get("trades_count") or 0) + (res_short.get("trades_count") or 0)
        weekly_long = res_long.get("weekly_return_pct") or 0
        weekly_short = res_short.get("weekly_return_pct") or 0
        return {
            "funding_z_threshold": funding_z,
            "rsi_z_threshold": rsi_z,
            "min_score": min_score,
            "base_sl_atr_mult": base_sl_atr,
            "tp_atr_mult": tp_atr_mult,
            "direction": direction,
            "result": res_long,
            "total_return_pct": total_return_pct,
            "weekly_return_pct": weekly_long + weekly_short,
            "max_drawdown_pct": max_dd,
            "trades_count": trades_count,
            "profitable": total_return_pct > 0,
        }
    entry_th, exit_rules = params_to_variant(funding_z, rsi_z, min_score, base_sl_atr, tp_atr_mult, direction)
    strategy = StrategyBNB(
        entry_thresholds=entry_th,
        exit_rules=exit_rules,
        position_size=POSITION_SIZE,
        direction=direction,
        min_factors_required=min_score,
    )
    res = engine.run(strategy, data, fee_bps=FEE_BPS, slippage_bps=SLIPPAGE_BPS)
    total_return_pct = res.get("total_return_pct") or 0
    max_dd = res.get("max_drawdown_pct") or 0
    return {
        "funding_z_threshold": funding_z,
        "rsi_z_threshold": rsi_z,
        "min_score": min_score,
        "base_sl_atr_mult": base_sl_atr,
        "tp_atr_mult": tp_atr_mult,
        "direction": direction,
        "result": res,
        "total_return_pct": total_return_pct,
        "weekly_return_pct": res.get("weekly_return_pct") or 0,
        "max_drawdown_pct": max_dd,
        "trades_count": res.get("trades_count", 0),
        "profitable": res.get("profitable_after_fees", False),
    }


def compute_ratio(rec: Dict[str, Any]) -> float:
    """目標函數：卡瑪比率 Calmar = Annual Return / Max Drawdown。兩年內 < 20 筆交易給 -10 懲罰。"""
    if rec.get("trades_count", 0) < MIN_TRADES_FOR_VALID:
        return PENALTY_RATIO_IF_LOW_TRADES
    annual_return_pct = (rec.get("total_return_pct") or 0) / 2.0  # 回測區間 2 年
    dd = max(rec.get("max_drawdown_pct", 0), 0.01)
    return annual_return_pct / dd


def run_batch_with_resource_guard(
    data,
    engine,
    candidates: List[Tuple[float, float, int, float, float, str]],
) -> List[Dict[str, Any]]:
    results = []
    for start in range(0, len(candidates), BATCH_SIZE):
        batch = candidates[start : start + BATCH_SIZE]
        load_before = get_load_average()
        if load_before > LOAD_AVG_THRESHOLD:
            print(f"  [資源] load average {load_before:.2f} > {LOAD_AVG_THRESHOLD}，放慢 {SLEEP_WHEN_LOAD_HIGH}s")
            time.sleep(SLEEP_WHEN_LOAD_HIGH)
        for (fz, rz, ms, base_sl, tp_atr, d) in batch:
            rec = run_single(data, engine, fz, rz, ms, base_sl, tp_atr, d)
            rec["ratio"] = compute_ratio(rec)
            results.append(rec)
        time.sleep(SLEEP_BETWEEN_BATCHES)
        gc.collect()
        load_after = get_load_average()
        print(f"  批次 {start // BATCH_SIZE + 1}: 已跑 {len(batch)} 組, load_avg={load_after:.2f}")
    return results


def find_best(results: List[Dict[str, Any]]) -> Tuple[int, Dict[str, Any]]:
    best_idx = -1
    best_ratio = -1e9
    for i, rec in enumerate(results):
        r = rec.get("ratio", PENALTY_RATIO_IF_LOW_TRADES)
        if r > best_ratio:
            best_ratio = r
            best_idx = i
    if best_idx < 0:
        best_idx = 0
    return best_idx, results[best_idx]


def write_best_strategy_py(
    funding_z: float,
    rsi_z: float,
    min_score: int,
    base_sl_atr: float,
    tp_atr_mult: float,
    direction: str,
    ratio: float,
):
    BOT_C.mkdir(parents=True, exist_ok=True)
    direction_file = "short" if direction == "both" else direction
    if direction == "regime":
        entry_lines = f"""    "funding_z_long": {funding_z},
    "rsi_z_long": {rsi_z},
    "min_score_long": 1,
    "funding_z_short": {funding_z},
    "rsi_z_short": {rsi_z},
    "min_score_short": 2,
    "price_breakout_long": 1.0,
    "price_breakout_short": 1.0,"""
    elif direction_file == "long":
        entry_lines = f"""    "funding_z_threshold": {funding_z},
    "rsi_z_threshold": {rsi_z},
    "min_score": {min_score},
    "price_breakout_long": 1.0,"""
    else:
        entry_lines = f"""    "funding_z_threshold": {funding_z},
    "rsi_z_threshold": {rsi_z},
    "min_score": {min_score},
    "price_breakout_short": 1.0,"""
    content = f'''"""
BNB/USDT 牛熊分治 + EMA200 + 快速止盈 tp_atr={tp_atr_mult}（方向={direction!r}）
funding_z={funding_z}, rsi_z={rsi_z}, min_score={min_score}, base_sl={base_sl_atr}, Calmar={ratio:.4f}
"""
from __future__ import annotations

from typing import Optional
from strategy_bnb import StrategyBNB, ExitRules

VARIANT_ID = "best"
POSITION_SIZE = 0.02

ENTRY_THRESHOLDS = {{
{entry_lines}
}}

EXIT_RULES = ExitRules(
    tp_r_mult={FIXED_TP_R_MULT},
    tp_atr_mult={tp_atr_mult},
    sl_atr_mult={base_sl_atr},
    trailing_stop_atr_mult=None,
    exit_after_bars=None,
    tp_fixed_pct=None,
)


def get_strategy(min_factors_required: Optional[int] = None) -> StrategyBNB:
    return StrategyBNB(
        entry_thresholds=ENTRY_THRESHOLDS,
        exit_rules=EXIT_RULES,
        position_size=POSITION_SIZE,
        direction="{direction_file}",
        min_factors_required=min_factors_required or ENTRY_THRESHOLDS.get("min_score", len(ENTRY_THRESHOLDS)),
    )
'''
    with open(BEST_STRATEGY_PY, "w", encoding="utf-8") as f:
        f.write(content)


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    BOT_C.mkdir(parents=True, exist_ok=True)
    report_ts = get_report_timestamp()
    optimization_path_file = LOG_DIR / f"optimization_path_{report_ts}.json"
    final_report_file = LOG_DIR / f"final_report_{report_ts}.json"

    if USE_2024_OOS:
        print("遞迴優化引擎 [2024 牛市專項] 僅優化做多、2024-01-01 至今資料...")
    else:
        print("遞迴優化引擎 (Calmar 優化 + EMA200 趨勢 + 動態止損): 載入資料...")
    try:
        data = load_data()
    except Exception as e:
        print(f"載入資料失敗: {e}")
        if os.getenv("BACKTEST_OFFLINE") == "1":
            print("BACKTEST_OFFLINE=1 時請確保 .cache 內有 BNBUSDT 1h 覆蓋 2022-2023 區間")
        sys.exit(1)
    print(f"  K 線: {len(data)}")
    load0 = get_load_average()
    print(f"  目前 load average: {load0:.2f}")
    print(f"  搜尋: direction {DIRECTION_VALS}, tp_atr_mult {TP_ATR_VALS}, funding_z/rsi_z/base_sl/min_score, 最低交易數={MIN_TRADES_FOR_VALID} (否則 -10)")
    print(f"  報告檔: {optimization_path_file.name}, {final_report_file.name}")

    from backtest_engine_bnb import BacktestEngineBNB
    engine = BacktestEngineBNB(position_size_pct=POSITION_SIZE, max_trades_per_day=MAX_TRADES_PER_DAY)

    step_fz = STEP_FUNDING_Z_INIT
    step_rz = STEP_RSI_Z_INIT
    center_fz = 1.0
    center_rz = 1.0
    center_min_score = 1
    center_base_sl = 1.5
    center_tp_atr = 2.5
    center_direction = "short"
    optimization_path: List[Dict[str, Any]] = []
    best_overall: Dict[str, Any] = {}
    best_ratio_overall = -1e9

    # 第一輪（重心 Short + both，快速止盈 tp_atr_mult）
    candidates = build_candidates_round1()
    print(f"\n[第 1 輪] 全組合 {len(candidates)} 組 (direction {DIRECTION_VALS}, tp_atr {TP_ATR_VALS}, base_sl {BASE_SL_VALS})")
    results = run_batch_with_resource_guard(data, engine, candidates)
    best_idx, best_rec = find_best(results)
    center_fz = best_rec["funding_z_threshold"]
    center_rz = best_rec["rsi_z_threshold"]
    center_min_score = best_rec.get("min_score", 1)
    center_base_sl = best_rec.get("base_sl_atr_mult", 1.5)
    center_tp_atr = best_rec.get("tp_atr_mult", 2.5)
    center_direction = best_rec["direction"]
    best_ratio_overall = best_rec.get("ratio", PENALTY_RATIO_IF_LOW_TRADES)
    best_overall = best_rec
    path_entry = {
        "round": 1,
        "step_funding_z": step_fz,
        "step_rsi_z": step_rz,
        "n_candidates": len(candidates),
        "best_index": best_idx,
        "best_funding_z_threshold": center_fz,
        "best_rsi_z_threshold": center_rz,
        "best_min_score": center_min_score,
        "best_base_sl_atr_mult": center_base_sl,
        "best_tp_atr_mult": center_tp_atr,
        "best_direction": center_direction,
        "best_ratio": round(best_ratio_overall, 6),
        "best_trades_count": best_rec.get("trades_count", 0),
        "best_weekly_return_pct": round(best_rec.get("weekly_return_pct", 0), 4),
        "best_max_drawdown_pct": round(best_rec.get("max_drawdown_pct", 0), 4),
        "load_avg_after": round(get_load_average(), 2),
    }
    optimization_path.append(path_entry)
    print(f"  勝出: fz={center_fz}, rz={center_rz}, ms={center_min_score}, base_sl={center_base_sl}, tp_atr={center_tp_atr}, dir={center_direction}, Calmar={best_ratio_overall:.4f}, trades={best_rec.get('trades_count', 0)}")
    gc.collect()

    # 遞迴
    for recursion in range(1, MAX_RECURSION + 1):
        step_fz = max(MIN_STEP_Z, step_fz / 2)
        step_rz = max(MIN_STEP_Z, step_rz / 2)
        if step_fz <= MIN_STEP_Z and step_rz <= MIN_STEP_Z:
            print(f"\n[終止] 步長已達下限")
            break
        candidates = build_candidates_around_center(
            center_fz, center_rz, center_min_score, center_base_sl, center_tp_atr, center_direction, step_fz, step_rz
        )
        print(f"\n[第 {recursion + 1} 輪] 中心=({center_fz}, {center_rz}, ms={center_min_score}, sl={center_base_sl}, tp={center_tp_atr}, {center_direction}), 共 {len(candidates)} 組")
        results = run_batch_with_resource_guard(data, engine, candidates)
        best_idx, best_rec = find_best(results)
        new_fz = best_rec["funding_z_threshold"]
        new_rz = best_rec["rsi_z_threshold"]
        new_min_score = best_rec.get("min_score", 1)
        new_base_sl = best_rec.get("base_sl_atr_mult", 1.5)
        new_tp_atr = best_rec.get("tp_atr_mult", 2.5)
        new_direction = best_rec["direction"]
        new_ratio = best_rec.get("ratio", PENALTY_RATIO_IF_LOW_TRADES)
        path_entry = {
            "round": recursion + 1,
            "step_funding_z": round(step_fz, 4),
            "step_rsi_z": round(step_rz, 4),
            "n_candidates": len(candidates),
            "best_index": best_idx,
            "best_funding_z_threshold": new_fz,
            "best_rsi_z_threshold": new_rz,
            "best_min_score": new_min_score,
            "best_base_sl_atr_mult": new_base_sl,
            "best_tp_atr_mult": new_tp_atr,
            "best_direction": new_direction,
            "best_ratio": round(new_ratio, 6),
            "best_trades_count": best_rec.get("trades_count", 0),
            "best_weekly_return_pct": round(best_rec.get("weekly_return_pct", 0), 4),
            "best_max_drawdown_pct": round(best_rec.get("max_drawdown_pct", 0), 4),
            "load_avg_after": round(get_load_average(), 2),
        }
        optimization_path.append(path_entry)
        if new_ratio > best_ratio_overall:
            best_ratio_overall = new_ratio
            best_overall = best_rec
            center_fz = new_fz
            center_rz = new_rz
            center_min_score = new_min_score
            center_base_sl = new_base_sl
            center_tp_atr = new_tp_atr
            center_direction = new_direction
            print(f"  勝出: fz={center_fz}, rz={center_rz}, ms={center_min_score}, base_sl={center_base_sl}, tp_atr={center_tp_atr}, dir={center_direction}, Calmar={best_ratio_overall:.4f}, trades={best_rec.get('trades_count', 0)}")
        else:
            print(f"  本輪最佳 ratio={new_ratio:.4f} 未超越 {best_ratio_overall:.4f}，保留原中心")
        gc.collect()

    final_fz = best_overall.get("funding_z_threshold", center_fz)
    final_rz = best_overall.get("rsi_z_threshold", center_rz)
    final_min_score = best_overall.get("min_score", center_min_score)
    final_base_sl = best_overall.get("base_sl_atr_mult", center_base_sl)
    final_tp_atr = best_overall.get("tp_atr_mult", center_tp_atr)
    final_direction = best_overall.get("direction", center_direction)
    write_best_strategy_py(final_fz, final_rz, final_min_score, final_base_sl, final_tp_atr, final_direction, best_ratio_overall)
    print(f"\n已寫入 {BEST_STRATEGY_PY} (fz={final_fz}, rz={final_rz}, min_score={final_min_score}, base_sl={final_base_sl}, tp_atr={final_tp_atr}, dir={final_direction})")

    opt_report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "report_timestamp": report_ts,
        "best_funding_z_threshold": final_fz,
        "best_rsi_z_threshold": final_rz,
        "best_min_score": final_min_score,
        "best_base_sl_atr_mult": final_base_sl,
        "best_tp_atr_mult": final_tp_atr,
        "best_direction": final_direction,
        "best_ratio": round(best_ratio_overall, 6),
        "min_trades_for_valid": MIN_TRADES_FOR_VALID,
        "optimization_path": optimization_path,
    }
    with open(optimization_path_file, "w", encoding="utf-8") as f:
        json.dump(opt_report, f, indent=2, ensure_ascii=False)
    print(f"已寫入 {optimization_path_file}")

    entry_th, _ = params_to_variant(final_fz, final_rz, final_min_score, final_base_sl, final_tp_atr, final_direction)
    variants_out = [{
        "id": "best",
        "entry_thresholds": entry_th,
        "exit_rules": {
            "tp_r_mult": FIXED_TP_R_MULT,
            "tp_atr_mult": final_tp_atr,
            "sl_atr_mult": final_base_sl,
            "exit_after_bars": None,
            "tp_fixed_pct": None,
        },
        "direction": final_direction,
        "best_ratio": round(best_ratio_overall, 6),
    }]
    with open(VARIANTS_JSON, "w", encoding="utf-8") as f:
        json.dump(variants_out, f, indent=2, ensure_ascii=False)
    print(f"已更新 {VARIANTS_JSON}")

    final = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "report_timestamp": report_ts,
        "qualified_count": 1,
        "best_strategy": {
            "funding_z_threshold": final_fz,
            "rsi_z_threshold": final_rz,
            "min_score": final_min_score,
            "base_sl_atr_mult": final_base_sl,
            "tp_atr_mult": final_tp_atr,
            "direction": final_direction,
            "ratio": round(best_ratio_overall, 6),
            "min_trades_required": MIN_TRADES_FOR_VALID,
        },
        "optimization_path_file": str(optimization_path_file),
    }

    # 若最佳結果仍 0 交易：列出最接近進場的 5 組並回報缺了什麼條件
    if best_overall.get("trades_count", 0) == 0:
        from bots.bot_c.strategy_bnb import StrategyBNB
        closest_5_params = [
            (0.5, 0.5, 1, 1.5, 2.5, "short"),
            (0.5, 0.5, 1, 1.5, 2.5, "both"),
            (1.0, 0.5, 1, 1.5, 2.5, "short"),
            (0.5, 1.0, 1, 1.5, 2.5, "short"),
            (1.0, 1.0, 1, 1.5, 1.5, "short"),
        ]
        fallback_entries = []
        for fz, rz, ms, base_sl, tp_atr, d in closest_5_params:
            direction_for_strategy = "short" if d == "both" else d
            entry_th, exit_rules = params_to_variant(fz, rz, ms, base_sl, tp_atr, direction_for_strategy)
            strat = StrategyBNB(
                entry_thresholds=entry_th,
                exit_rules=exit_rules,
                position_size=POSITION_SIZE,
                direction=direction_for_strategy,
                min_factors_required=ms,
            )
            dist = strat.get_score_distribution(data)
            total_cand = sum(dist.get(k, 0) for k in (1, 2, 3))
            meets = sum(dist.get(k, 0) for k in range(ms, 4))
            if meets == 0 and total_cand > 0:
                missing = f"缺 {ms - max((s for s, c in dist.items() if c), default=0)} 因子才達 min_score={ms}"
            elif total_cand == 0:
                missing = "無任何 K 線達任一因子門檻（檢查 Z/breakout 計算）"
            else:
                missing = f"已有達 min_score 的 K 線數={meets}，0 交易可能為引擎/持倉邏輯"
            fallback_entries.append({
                "funding_z_threshold": fz,
                "rsi_z_threshold": rz,
                "min_score": ms,
                "base_sl_atr_mult": base_sl,
                "tp_atr_mult": tp_atr,
                "direction": d,
                "score_distribution": dist,
                "missing": missing,
            })
        final["zero_trade_fallback"] = {
            "message": "所有組合均 0 筆交易",
            "closest_5": fallback_entries,
        }
        print("  [0 交易] 已寫入最接近進場的 5 組參數與缺件說明至報告")

    with open(final_report_file, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)
    print(f"已寫入 {final_report_file}")

    print("遞迴優化引擎 完成.")


if __name__ == "__main__":
    main()
