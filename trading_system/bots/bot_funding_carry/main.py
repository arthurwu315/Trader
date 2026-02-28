"""
Alpha2 Funding Carry - main loop.
PAPER: log signals only. MICRO-LIVE: 2-5% notional (after PAPER 7d).
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("SKIP_CONFIG_VALIDATION", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bots.bot_funding_carry.config import (
    STRATEGY_ID,
    UNIVERSE,
    ENTRY_ANNUALIZED_MIN,
    EXIT_ANNUALIZED_MAX,
)


def _get_funding_rate(client, symbol: str) -> float:
    try:
        out = client._call_with_retry("GET", "/fapi/v1/premiumIndex", {"symbol": symbol})
        return float(out.get("lastFundingRate", 0) or 0)
    except Exception:
        return 0.0


def run_cycle_paper(client=None):
    """One PAPER cycle: check funding, log signals, append trade records."""
    from core.v9_trade_record import append_v9_trade_record
    mode = os.getenv("ALPHA2_MODE", "PAPER")
    now = datetime.now(timezone.utc)
    ts = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    if client is None:
        from bots.bot_c.config_c import get_strategy_c_config
        from core.binance_client import BinanceFuturesClient
        cfg = get_strategy_c_config()
        client = BinanceFuturesClient(
            base_url=os.getenv("BINANCE_DATA_URL", "https://fapi.binance.com"),
            api_key=cfg.binance_api_key or "dummy",
            api_secret=cfg.binance_api_secret or "dummy",
        )

    for symbol in UNIVERSE:
        fr = _get_funding_rate(client, symbol)
        annual = fr * 3 * 365
        if annual >= ENTRY_ANNUALIZED_MIN:  # entry: >20% annualized
            # Long spot + short perp = receive funding when positive
            # Record as "entry" for funding carry
            append_v9_trade_record(
                timestamp=ts,
                symbol=symbol,
                side="SHORT_PERP",  # hedge = long spot not tracked here
                price=0.0,  # placeholder for PAPER
                qty=0.0,
                regime_vol=annual * 100,
                reason="entry",
                fees=0.0,
                slippage_est=0.0,
                signal_time=ts,
                order_time=ts,
                mode=mode,
                strategy_id=STRATEGY_ID,
            )
        elif annual < EXIT_ANNUALIZED_MAX or annual < 0:  # exit: <10% or negative
            append_v9_trade_record(
                timestamp=ts,
                symbol=symbol,
                side="CLOSE",
                price=0.0,
                qty=0.0,
                regime_vol=annual * 100,
                reason="exit",
                fees=0.0,
                slippage_est=0.0,
                signal_time=ts,
                order_time=ts,
                mode=mode,
                strategy_id=STRATEGY_ID,
            )
    return 0


def main():
    print(f"Alpha2 {STRATEGY_ID} PAPER mode (7 days first)")
    run_cycle_paper()
    print("Cycle done. Check logs/v9_trade_records.csv (strategy_id=FUND_CARRY_V1)")


if __name__ == "__main__":
    main()
