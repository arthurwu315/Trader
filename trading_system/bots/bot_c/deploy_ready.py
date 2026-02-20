"""
1D 宏觀趨勢部署參數（Macro Portfolio Engine v1.0）
- LONG: close > ema_50 且 ema_50 > ema_slow 且 close > roll_high_N
- SHORT: close < ema_50 且 ema_50 < ema_slow 且 close < roll_low_N
- ATR 突破濾網: (high - low) > ATR_BREAK_MULT * ATR
- 出場: 3.0x ATR 追蹤（由策略引擎處理），此處提供進場與初始 SL
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

# 1D Macro 最佳組合（由回測掃描結果寫入）
DONCHIAN_N = 55
EMA_SLOW_PERIOD = 100
TRAILING_ATR_MULT = 3.0
ATR_STOP_MULT = 2.5
ATR_BREAK_MULT = 1.0

DEPLOY_PARAMS = {
    "macro_n": DONCHIAN_N,
    "ema_slow_period": EMA_SLOW_PERIOD,
    "trailing_atr_mult": TRAILING_ATR_MULT,
    "atr_stop_mult": ATR_STOP_MULT,
    "atr_break_mult": ATR_BREAK_MULT,
}

HARD_STOP_POSITION_PCT = 2.0
POSITION_SIZE = 0.02


@dataclass
class SignalOut:
    """單根 K 線的訊號輸出"""
    should_enter: bool
    side: str
    entry_price: float
    sl_price: float
    tp_price: float
    hard_stop_price: float
    atr: float
    regime: str  # "bull" | "bear"


def get_signal_from_row(
    row: Dict[str, Any],
    params: Optional[Dict[str, Any]] = None,
    last_regime: Optional[str] = None,
) -> Tuple[Optional[SignalOut], str]:
    """
    1D 宏觀趨勢：
    LONG = close > ema_50 且 ema_50 > ema_slow 且 close > 過去 N 根最高價
    SHORT = close < ema_50 且 ema_50 < ema_slow 且 close < 過去 N 根最低價
    並要求 (high-low) > ATR_BREAK_MULT * ATR。
    回傳 (signal_or_none, current_regime)。
    """
    p = params or DEPLOY_PARAMS
    close = float(row["close"])
    atr = float(row.get("atr", 0.0))
    if atr <= 0:
        atr = close * 0.02

    donchian_n = int(p.get("macro_n", DONCHIAN_N))
    if donchian_n not in (20, 55, 80):
        donchian_n = 55
    ema_slow_period = int(p.get("ema_slow_period", EMA_SLOW_PERIOD))
    if ema_slow_period not in (100, 200):
        ema_slow_period = 100
    atr_stop = float(p.get("atr_stop_mult", ATR_STOP_MULT))
    atr_break_mult = float(p.get("atr_break_mult", ATR_BREAK_MULT))

    roll_high_col = f"roll_high_{donchian_n}"
    roll_low_col = f"roll_low_{donchian_n}"
    ema_slow_col = f"ema_{ema_slow_period}"
    roll_high_val = row.get(roll_high_col)
    roll_low_val = row.get(roll_low_col)
    ema_fast_val = row.get("ema_50")
    ema_slow_val = row.get(ema_slow_col)
    if roll_high_val is None or roll_low_val is None or ema_fast_val is None or ema_slow_val is None:
        regime = last_regime if last_regime in ("bull", "bear") else "bear"
        return None, regime
    if isinstance(roll_high_val, float) and math.isnan(roll_high_val):
        regime = last_regime if last_regime in ("bull", "bear") else "bear"
        return None, regime
    if isinstance(roll_low_val, float) and math.isnan(roll_low_val):
        regime = last_regime if last_regime in ("bull", "bear") else "bear"
        return None, regime
    if isinstance(ema_fast_val, float) and math.isnan(ema_fast_val):
        regime = last_regime if last_regime in ("bull", "bear") else "bear"
        return None, regime
    if isinstance(ema_slow_val, float) and math.isnan(ema_slow_val):
        regime = last_regime if last_regime in ("bull", "bear") else "bear"
        return None, regime
    roll_high_val = float(roll_high_val)
    roll_low_val = float(roll_low_val)
    ema_fast_val = float(ema_fast_val)
    ema_slow_val = float(ema_slow_val)
    bar_range = abs(float(row["high"]) - float(row["low"]))
    if bar_range <= (atr_break_mult * atr):
        regime = "bull" if close > ema_slow_val else "bear"
        return None, regime

    # 無固定 TP，設遠價僅供介面；實務出場靠追蹤止損
    tp_far_mult = 50.0

    if close > ema_fast_val and ema_fast_val > ema_slow_val and close > roll_high_val:
        regime = "bull"
        sl_price = close - atr_stop * atr
        tp_price = close + tp_far_mult * atr
        hard_stop_price = close * (1 - HARD_STOP_POSITION_PCT / 100.0)
        return SignalOut(
            should_enter=True,
            side="BUY",
            entry_price=close,
            sl_price=sl_price,
            tp_price=tp_price,
            hard_stop_price=hard_stop_price,
            atr=atr,
            regime=regime,
        ), regime
    if close < ema_fast_val and ema_fast_val < ema_slow_val and close < roll_low_val:
        regime = "bear"
        sl_price = close + atr_stop * atr
        tp_price = close - tp_far_mult * atr
        hard_stop_price = close * (1 + HARD_STOP_POSITION_PCT / 100.0)
        return SignalOut(
            should_enter=True,
            side="SELL",
            entry_price=close,
            sl_price=sl_price,
            tp_price=tp_price,
            hard_stop_price=hard_stop_price,
            atr=atr,
            regime=regime,
        ), regime

    regime = last_regime if last_regime in ("bull", "bear") else "bear"
    return None, regime


def get_deploy_params() -> Dict[str, Any]:
    return {
        **DEPLOY_PARAMS,
        "hard_stop_position_pct": HARD_STOP_POSITION_PCT,
        "position_size": POSITION_SIZE,
    }


def check_hard_stop(side: str, entry_price: float, current_high: float, current_low: float) -> bool:
    thresh = entry_price * (HARD_STOP_POSITION_PCT / 100.0)
    if side == "BUY":
        return current_low <= entry_price - thresh
    return current_high >= entry_price + thresh


def apply_trailing_tp(
    side: str,
    entry_price: float,
    current_tp: float,
    atr: float,
    trailing_atr_mult: Optional[float],
    bar_high: float,
    bar_low: float,
) -> float:
    """4H 宏觀：寬幅 ATR 追蹤止損，趨勢不回頭則持續持倉。"""
    if trailing_atr_mult is None or trailing_atr_mult <= 0:
        return current_tp
    if side == "BUY" and bar_high > entry_price:
        new_tp = bar_high - trailing_atr_mult * atr
        if new_tp > current_tp:
            return new_tp
    if side == "SELL" and bar_low < entry_price:
        new_tp = bar_low + trailing_atr_mult * atr
        if new_tp < current_tp:
            return new_tp
    return current_tp


if __name__ == "__main__":
    print("deploy_ready: 1D 宏觀趨勢引擎")
    print(
        "  N:",
        DONCHIAN_N,
        "EMA_SLOW:",
        EMA_SLOW_PERIOD,
        "TRAIL:",
        TRAILING_ATR_MULT,
        "ATR_BREAK:",
        ATR_BREAK_MULT,
    )
