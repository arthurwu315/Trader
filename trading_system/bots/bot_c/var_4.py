"""
BNB/USDT 策略變體 4：最嚴 Funding -0.006、緊止損 1.0、2R 止盈
對應 variants.json 之 var_4（回撤優化）
"""
from __future__ import annotations

from typing import Optional
from strategy_bnb import StrategyBNB, ExitRules

VARIANT_ID = "var_4"
POSITION_SIZE = 0.02

ENTRY_THRESHOLDS = {
    "funding_rate_proxy": -0.006,
    "oi_proxy": 1.2,
    "volatility": 0.008,
    "price_breakout_long": 1.0,
}

EXIT_RULES = ExitRules(
    tp_r_mult=2.0,
    sl_atr_mult=1.0,
    exit_after_bars=None,
    tp_fixed_pct=None,
)


def get_strategy(min_factors_required: Optional[int] = None) -> StrategyBNB:
    return StrategyBNB(
        entry_thresholds=ENTRY_THRESHOLDS,
        exit_rules=EXIT_RULES,
        position_size=POSITION_SIZE,
        direction="long",
        min_factors_required=min_factors_required or len(ENTRY_THRESHOLDS),
    )
