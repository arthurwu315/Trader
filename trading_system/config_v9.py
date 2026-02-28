"""
V9 Production Core - Regime-filtered trend model.
V9.1: HARD FREEZE - Strategy params & rules must not change.
Any change requires new version (V9.2+). Do not overwrite V9.1.
"""
STRATEGY_VERSION = "V9_REGIME_CORE"

# Fixed regime thresholds (from V8.4 robustness - plateau confirmed)
VOL_LOW = 2.2   # LOW: vol < 2.2%
VOL_HIGH = 4.2  # HIGH: vol >= 4.2%
# MID (2.2% <= vol < 4.2%): disabled, no new entries
