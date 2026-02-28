"""
V9 Production Core - Regime-filtered trend model.
Locked parameters; no further threshold optimization.
"""
STRATEGY_VERSION = "V9_REGIME_CORE"

# Fixed regime thresholds (from V8.4 robustness - plateau confirmed)
VOL_LOW = 2.2   # LOW: vol < 2.2%
VOL_HIGH = 4.2  # HIGH: vol >= 4.2%
# MID (2.2% <= vol < 4.2%): disabled, no new entries
