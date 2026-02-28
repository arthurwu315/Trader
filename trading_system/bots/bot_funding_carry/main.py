"""
Alpha2 Funding Carry - main loop.
PAPER: log signals only. MICRO-LIVE: 2-5% notional (after PAPER 7d).
Every cycle writes CYCLE records with hedge/rebalance/cooldown fields.
ALPHA2_SELF_TEST=1: simulate hedge deviation -> rebalance fail -> hard stop (PAPER only).
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
    ALPHA2_CAPITAL_PCT,
    SINGLE_ASSET_CAP_PCT,
    PAPER_SIMULATED_EQUITY,
)
from bots.bot_funding_carry.state import (
    is_in_cooldown,
    set_cooldown,
    should_hard_stop,
    inc_rebalance_fail,
    get_state_summary,
    get_cooldown_active,
)


def _get_funding_rate(client, symbol: str) -> float:
    try:
        out = client._call_with_retry("GET", "/fapi/v1/premiumIndex", {"symbol": symbol})
        return float(out.get("lastFundingRate", 0) or 0)
    except Exception:
        return 0.0


def _get_mark_price(client, symbol: str) -> float | None:
    try:
        out = client._call_with_retry("GET", "/fapi/v1/premiumIndex", {"symbol": symbol})
        p = out.get("markPrice")
        return float(p) if p is not None else None
    except Exception:
        return None


def _print_startup_banner():
    """Print hard stop status, cooldown symbols, self-test mode."""
    summary = get_state_summary()
    cooldowns = get_cooldown_active()
    self_test = os.getenv("ALPHA2_SELF_TEST", "").strip() in ("1", "true", "yes")
    lines = [
        f"Alpha2 {STRATEGY_ID} PAPER",
        f"  hard_stop_triggered: {summary['hard_stop_triggered']} | rebalance_fail: {summary['rebalance_fail_count']} | consecutive_out: {summary['consecutive_out_of_threshold']}",
        f"  cooldown: {[(s, u) for s, u in cooldowns] if cooldowns else 'none'}",
    ]
    if self_test:
        lines.append("  ALPHA2_SELF_TEST=1 (simulate deviation -> rebalance fail -> hard stop)")
    print("\n".join(lines))


def run_cycle_paper(client=None):
    """One PAPER cycle: check funding, cooldown, compute/NA notional, self-test optional."""
    from core.v9_trade_record import append_v9_trade_record, append_funding_carry_cycle
    mode = os.getenv("ALPHA2_MODE", "PAPER")
    self_test = os.getenv("ALPHA2_SELF_TEST", "").strip() in ("1", "true", "yes")
    now = datetime.now(timezone.utc)
    ts = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    _print_startup_banner()

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

    # Simulated allocation per symbol (PAPER)
    allocated = PAPER_SIMULATED_EQUITY * (SINGLE_ASSET_CAP_PCT or 0.025)

    for symbol in UNIVERSE:
        rebalance_action = "NONE"
        rebalance_attempt = False
        rebalance_success = True

        # Compute notional: price * simulated position, or NA if unavailable
        price = _get_mark_price(client, symbol)
        if price is None or price <= 0:
            spot_n, perp_n, net_n, net_pct = None, None, None, None
            notional_reason_suffix = ";NOTIONAL_UNAVAILABLE"
        elif self_test:
            # Simulate deviation > threshold (0.2% + buffer)
            net_pct = HEDGE_DEVIATION_PCT_THRESHOLD + 0.0005
            net_n = PAPER_SIMULATED_EQUITY * net_pct
            spot_n = allocated
            perp_n = -(allocated - net_n)
            notional_reason_suffix = ""
            rebalance_action = "REBALANCE"
            rebalance_attempt = True
            rebalance_success = False
            inc_rebalance_fail()
            append_v9_trade_record(
                timestamp=ts,
                symbol=symbol,
                side="REBALANCE_FAIL",
                price=0.0,
                qty=0.0,
                regime_vol=0.0,
                reason="REBALANCE_FAIL",
                fees=0.0,
                slippage_est=0.0,
                signal_time=ts,
                order_time=ts,
                mode=mode,
                strategy_id=STRATEGY_ID,
                event_type="TRADE",
            )
        else:
            # Normal PAPER: perfect hedge (simulated position)
            spot_n = allocated
            perp_n = -allocated
            net_n = 0.0
            net_pct = 0.0
            notional_reason_suffix = ""

        if is_in_cooldown(symbol):
            reason = "COOLDOWN" + notional_reason_suffix
            append_funding_carry_cycle(
                timestamp=ts,
                symbol=symbol,
                mode=mode,
                funding_rate_8h=0.0,
                funding_annualized_pct=0.0,
                signal=False,
                reason=reason,
                strategy_id=STRATEGY_ID,
                spot_notional=spot_n,
                perp_notional=perp_n,
                net_notional=net_n,
                net_notional_pct=net_pct,
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
        reason = reason + notional_reason_suffix

        append_funding_carry_cycle(
            timestamp=ts,
            symbol=symbol,
            mode=mode,
            funding_rate_8h=fr,
            funding_annualized_pct=annual_pct,
            signal=signal,
            reason=reason,
            strategy_id=STRATEGY_ID,
            spot_notional=spot_n,
            perp_notional=perp_n,
            net_notional=net_n,
            net_notional_pct=net_pct,
            rebalance_action=rebalance_action,
            rebalance_attempt=rebalance_attempt,
            rebalance_success=rebalance_success,
        )

        if self_test:
            pass  # self-test: no entry/exit TRADE records, only rebalance fail
        elif annual >= ENTRY_ANNUALIZED_MIN:
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
