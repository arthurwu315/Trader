"""
Alpha2 Funding Carry Report.
Reads v9_trade_records.csv (strategy_id=FUND_CARRY_V1).
Outputs: funding collected, hedge pnl, costs, max adverse excursion.
Robust: works with 0 TRADE events; outputs CYCLE snapshot + diagnostics when no records.
"""
from __future__ import annotations

import sys
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.v9_trade_record import read_v9_records, get_v9_records_path


def _parse_ts(s: str) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def main():
    path = get_v9_records_path()
    if not path.exists():
        print("=" * 60)
        print("Alpha2 Funding Carry Report (FUND_CARRY_V1)")
        print("=" * 60)
        print(f"File not found: {path.resolve()}")
        print("Run: python3 -m bots.bot_funding_carry.main")
        return

    records = read_v9_records(path)
    strategy_counts = Counter(r.get("strategy_id", "") for r in records)
    carry = [r for r in records if r.get("strategy_id") == "FUND_CARRY_V1"]

    if not carry:
        print("=" * 60)
        print("Alpha2 Funding Carry Report (FUND_CARRY_V1)")
        print("=" * 60)
        print("No FUND_CARRY_V1 records.")
        print(f"Path: {path.resolve()}")
        print(f"Total rows in file: {len(records)}")
        if strategy_counts:
            print("Strategy ID distribution:", dict(strategy_counts.most_common(10)))
        print("Run: python3 -m bots.bot_funding_carry.main")
        return

    cycles = [r for r in carry if r.get("side") == "CYCLE" or r.get("event_type") == "CYCLE"]
    trades = [r for r in carry if r.get("event_type") == "TRADE" or (r.get("side") not in ("CYCLE", "") and r.get("event_type") != "CYCLE")]
    if not trades:
        trades = [r for r in carry if r.get("reason") in ("entry", "exit")]

    print("=" * 60)
    print("Alpha2 Funding Carry Report (FUND_CARRY_V1)")
    print("=" * 60)
    print(f"Total FUND_CARRY_V1 records: {len(carry)}")
    print(f"CYCLE records: {len(cycles)} | TRADE events: {len(trades)}")

    if trades:
        total_fees = sum(float(r.get("fees", 0)) for r in trades)
        total_slip_bp = sum(float(r.get("slippage_est", 0)) for r in trades)
        n = len(trades)
        avg_fees = total_fees / n if n else 0
        avg_slip = total_slip_bp / n if n else 0
        funding_collected_proxy = sum(float(r.get("regime_vol", 0)) for r in trades if r.get("reason") == "entry") / 100
        print(f"Total fees: {total_fees:.4f} USDT")
        print(f"Avg fee per event: {avg_fees:.4f}")
        print(f"Avg slippage est (bp): {avg_slip:.2f}")
        print(f"Funding collected (proxy): {funding_collected_proxy:.4f}")
    else:
        print("Trades: 0")

    if cycles:
        # Latest funding snapshot (most recent per symbol)
        seen = {}
        for r in reversed(cycles):
            sym = r.get("symbol", "")
            if sym and sym not in seen:
                seen[sym] = r
        print("-" * 40)
        print("Latest funding snapshot (annualized %):")
        for sym, r in sorted(seen.items()):
            annual = r.get("funding_annualized_pct") or r.get("regime_vol") or 0
            try:
                annual = float(annual)
            except (TypeError, ValueError):
                annual = 0
            reason = r.get("reason", "?")
            print(f"  {sym}: {annual:.2f}% | reason={reason}")

        # Signal count in last 7 days
        cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        signal_count = 0
        for r in cycles:
            if r.get("reason") not in ("ENTRY_SIGNAL", "EXIT_SIGNAL") and str(r.get("signal", "")).lower() != "true":
                continue
            t = _parse_ts(r.get("timestamp", ""))
            if t:
                t_utc = t.replace(tzinfo=timezone.utc) if not t.tzinfo else t
                if t_utc >= cutoff:
                    signal_count += 1
        print(f"Signal count (last 7 days): {signal_count}")

    print("Max adverse excursion: N/A (requires position-level tracking)")
    print("Hedge PnL: N/A (PAPER mode)")


if __name__ == "__main__":
    main()
