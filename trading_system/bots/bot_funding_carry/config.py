"""
Alpha2 Funding Carry - MVP config.
Independent of V9. strategy_id = FUND_CARRY_V1
"""
STRATEGY_ID = "FUND_CARRY_V1"
UNIVERSE = ["BTCUSDT", "ETHUSDT"]
ENTRY_ANNUALIZED_MIN = 0.20
ENTRY_FUNDING_RATE_MIN = ENTRY_ANNUALIZED_MIN / (3 * 365)
EXIT_ANNUALIZED_MAX = 0.10
EXIT_FUNDING_RATE_MAX = EXIT_ANNUALIZED_MAX / (3 * 365)
SINGLE_ASSET_LOSS_CAP_PCT = 0.01
ALPHA2_CAPITAL_PCT = 0.10
SINGLE_ASSET_CAP_PCT = 0.025
PAPER_DAYS_FIRST = 7

# Hedge deviation hard limits (engineering guardrails)
HEDGE_DEVIATION_PCT_THRESHOLD = 0.002  # 0.20% => must REBALANCE
REBALANCE_FAIL_HARD_STOP = 3  # consecutive fails or cycles out => stop bot
COOLDOWN_HOURS = 24  # no entry for 24h after exit (annualized < 10% or <= 0)

# PAPER simulated notional (for observability when no real positions)
PAPER_SIMULATED_EQUITY = 10000.0

# Alpha2 MICRO-LIVE equity DD kill (independent of V9)
ALPHA2_MAX_DD_PCT = 0.01  # 1% drawdown => stop bot
