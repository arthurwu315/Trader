"""
BNB/USDT 牛熊分治 + EMA200 + 快速止盈 tp_atr=2.5（方向='short'）
funding_z=0.62, rsi_z=2.0, min_score=2, base_sl=2.0, Calmar=13.9041
"""
from __future__ import annotations

from typing import Optional
from strategy_bnb import StrategyBNB, ExitRules

VARIANT_ID = "best"
POSITION_SIZE = 0.02

ENTRY_THRESHOLDS = {
    "funding_z_threshold": 0.62,
    "rsi_z_threshold": 2.0,
    "min_score": 2,
    "price_breakout_short": 1.0,
}

EXIT_RULES = ExitRules(
    tp_r_mult=2.0,
    tp_atr_mult=2.5,
    sl_atr_mult=2.0,
    trailing_stop_atr_mult=None,
    exit_after_bars=None,
    tp_fixed_pct=None,
)


def get_strategy(min_factors_required: Optional[int] = None) -> StrategyBNB:
    return StrategyBNB(
        entry_thresholds=ENTRY_THRESHOLDS,
        exit_rules=EXIT_RULES,
        position_size=POSITION_SIZE,
        direction="short",
        min_factors_required=min_factors_required or ENTRY_THRESHOLDS.get("min_score", len(ENTRY_THRESHOLDS)),
    )
