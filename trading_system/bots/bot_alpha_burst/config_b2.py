"""
Alpha Burst B2 COMPRESS config.
Compression -> expansion burst. strategy_id = ALPHA_BURST_B2_COMPRESS
Independent of B1 (no Donchian breakout).
"""
STRATEGY_ID = "ALPHA_BURST_B2_COMPRESS"
UNIVERSE = ["BTCUSDT", "ETHUSDT"]

TF_TREND = "4h"
TF_ENTRY = "1h"

EMA_TREND_PERIOD = 200
ATR_PERIOD = 14
ATR_MA_PERIOD = 20

# Compression: ATR14 below this percentile of past N bars
COMPRESSION_ATR_LOOKBACK = 80
COMPRESSION_ATR_PERCENTILE = 30
COMPRESSION_MIN_BARS = 5
COMPRESSION_RANGE_LOOKBACK = 40
ARMED_LOOKBACK = 30

# Expansion: first ATR > atr_ma * threshold after compression
EXPANSION_THRESHOLD = 1.1

# Exit
STOP_ATR_K = 2.0
TRAILING_STOP_ATR_K = 2.0

BURST_RISK_PCT = 0.01
BURST_MAX_DD_PCT = 0.25
SEED = 42
