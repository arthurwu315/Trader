"""
Alpha2 Funding Carry - main loop.
PAPER: log signals only. MICRO-LIVE: 2-5% notional (after PAPER 7d).
Every cycle writes CYCLE records with hedge/rebalance/cooldown fields.
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
    HEDGE_DEVIATION_PCT_THRESHOLD,
    REBALANCE_FAIL_HARD_STOP,
    COOLDOWN_HOURS,
)
from bots.bot_funding_carry.state import (
    is_in_cooldown,
    set_cooldown,
    should_hard_stop,
)


def _get_funding_rate(client, symbol: str) -> float:
    try:
        out = client._call_with_retry("GET", "/fapi/v1/premiumIndex", {"symbol": symbol})
        return float(out.get("lastFundingRate", 0) or 0)
    except Exception:
        return 0.0


def run_cycle_paper(client=None):
    """One PAPER cycle: check funding, cooldown, always write CYCLE per symbol."""
    from core.v9_trade_record import append_v9_trade_record, append_funding_carry_cycle
    mode = os.getenv("ALPHA2_MODE", "PAPER")
    now = datetime.now(timezone.utc)
    ts = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    if should_hard_stop(REBALANCE_FAIL_HARD_STOP):
        print("ALPHA2 HARD STOP: rebalance fail or consecutive out-of-threshold >= 3")
        sys.exit(1)

    if client is None:
        from bots.bot_c.config_c import get_strategy_c_config
        from core.binance_client import BinanceFuturesClient
        cfg = get_strategy_c_config()
        client = BinanceFuturesClient(
            base_url=os.getenv("BINANCE_DATA_URL", "https://fapi.binance.com"),
            api_key=cfg.binance_api_key or "dummy",
            api_secret=cfg.binance_api_secret or "dummy",
        )

    # PAPER: no real positions => spot/perp/net = 0, rebalance_action=NONE
    spot_notional = perp_notional = net_notional = net_notional_pct = 0.0
    rebalance_action = "NONE"
    rebalance_attempt = False
    rebalance_success = True

    for symbol in UNIVERSE:
        if is_in_cooldown(symbol):
            append_funding_carry_cycle(
                timestamp=ts,
                symbol=symbol,
                mode=mode,
                funding_rate_8h=0.0,
                funding_annualized_pct=0.0,
                signal=False,
                reason="COOLDOWN",
                strategy_id=STRATEGY_ID,
                spot_notional=spot_notional,
                perp_notional=perp_notional,
                net_notional=net_notional,
                net_notional_pct=net_notional_pct,
                rebalance_action=rebalance_action,
                rebalance_attempt=rebalance_attempt,
                rebalance_success=rebalance_success,
            )
            continue

        fr = _get_funding_rate(client, symbol)
        annual = fr * 3 * 365
        annual_pct = annual * 100

        if annual >= ENTRY_ANNUALIZED_MIN:
            signal, reason = True, "ENTRY_SIGNAL"
        elif annual < EXIT_ANNUALIZED_MAX or annual <= 0:
            signal, reason = True, "EXIT_SIGNAL"
            set_cooldown(symbol, COOLDOWN_HOURS)
        else:
            signal, reason = False, "NO_SIGNAL"

        append_funding_carry_cycle(
            timestamp=ts,
            symbol=symbol,
            mode=mode,
            funding_rate_8h=fr,
            funding_annualized_pct=annual_pct,
            signal=signal,
            reason=reason,
            strategy_id=STRATEGY_ID,
            spot_notional=spot_notional,
            perp_notional=perp_notional,
            net_notional=net_notional,
            net_notional_pct=net_notional_pct,
            rebalance_action=rebalance_action,
            rebalance_attempt=rebalance_attempt,
            rebalance_success=rebalance_success,
        )

        if annual >= ENTRY_ANNUALIZED_MIN:
            append_v9_trade_record(
                timestamp=ts,
                symbol=symbol,
                side="SHORT_PERP",
                price=0.0,
                qty=0.0,
                regime_vol=annual_pct,
                reason="entry",
                fees=0.0,
                slippage_est=0.0,
                signal_time=ts,
                order_time=ts,
                mode=mode,
                strategy_id=STRATEGY_ID,
                event_type="TRADE",
            )
        elif annual < EXIT_ANNUALIZED_MAX or annual <= 0:
            append_v9_trade_record(
                timestamp=ts,
                symbol=symbol,
                side="CLOSE",
                price=0.0,
                qty=0.0,
                regime_vol=annual_pct,
                reason="exit",
                fees=0.0,
                slippage_est=0.0,
                signal_time=ts,
                order_time=ts,
                mode=mode,
                strategy_id=STRATEGY_ID,
                event_type="TRADE",
            )
    return 0


def main():
    print(f"Alpha2 {STRATEGY_ID} PAPER mode (7 days first)")
    run_cycle_paper()
    print("Cycle done. Check logs/v9_trade_records.csv (strategy_id=FUND_CARRY_V1)")


if __name__ == "__main__":
    main()
