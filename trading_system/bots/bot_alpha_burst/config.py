"""
Alpha Burst B1 config.
Independent of V9. strategy_id = ALPHA_BURST_B1
"""
STRATEGY_ID = "ALPHA_BURST_B1"
UNIVERSE = ["BTCUSDT", "ETHUSDT"]

# Timeframes
TF_TREND = "4h"
TF_ENTRY = "1h"

# 4H trend filter: close > EMA200 = long only, close < EMA200 = short only
EMA_TREND_PERIOD = 200

# Vol expansion: ATR14 > ATR14_ma20 * threshold to allow trades
ATR_PERIOD = 14
ATR_MA_PERIOD = 20
VOL_EXPANSION_THRESHOLD = 1.2

# Donchian breakout
BREAKOUT_LOOKBACK = 20

# Exit: initial stop ATR*k, ATR trailing stop
STOP_ATR_K = 2.0
TRAILING_STOP_ATR_K = 2.0

# Position sizing: 1% burst_equity risk per trade
BURST_RISK_PCT = 0.01

# Ops: Burst DD > 25% => KILL
BURST_MAX_DD_PCT = 0.25

# Seed for reproducibility
SEED = 42
