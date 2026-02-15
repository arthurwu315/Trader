"""
BNB/USDT 因子投票策略（Funding Z + RSI Z + Price Breakout, min_score=2）
funding_z=1.75, rsi_z=1.88, direction='long', 優化比=-0.0373
"""
from __future__ import annotations

from typing import Optional
from strategy_bnb import StrategyBNB, ExitRules

VARIANT_ID = "best"
POSITION_SIZE = 0.02

ENTRY_THRESHOLDS = {
    "funding_z_threshold": 1.75,
    "rsi_z_threshold": 1.88,
    "min_score": 2,
    "price_breakout_long": 1.0,
}

EXIT_RULES = ExitRules(
    tp_r_mult=2.0,
    sl_atr_mult=2.0,
    exit_after_bars=None,
    tp_fixed_pct=None,
)


def get_strategy(min_factors_required: Optional[int] = None) -> StrategyBNB:
    return StrategyBNB(
        entry_thresholds=ENTRY_THRESHOLDS,
        exit_rules=EXIT_RULES,
        position_size=POSITION_SIZE,
        direction="long",
        min_factors_required=min_factors_required or ENTRY_THRESHOLDS.get("min_score", len(ENTRY_THRESHOLDS)),
    )
