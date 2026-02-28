"""
Alpha2 Funding Carry Report.
Reads v9_trade_records.csv (strategy_id=FUND_CARRY_V1).
Outputs: funding collected, hedge pnl, costs, max adverse excursion.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.v9_trade_record import read_v9_records, get_v9_records_path


def main():
    path = get_v9_records_path()
    records = read_v9_records(path)
    carry = [r for r in records if r.get("strategy_id") == "FUND_CARRY_V1"]
    if not carry:
        print("No FUND_CARRY_V1 records. Run Alpha2 PAPER first.")
        print(f"Expected: {path}")
        return

    total_fees = sum(float(r.get("fees", 0)) for r in carry)
    total_slip_bp = sum(float(r.get("slippage_est", 0)) for r in carry)
    n = len(carry)
    avg_fees = total_fees / n if n else 0
    avg_slip = total_slip_bp / n if n else 0

    # regime_vol stored as annualized % for funding carry
    funding_collected_proxy = sum(float(r.get("regime_vol", 0)) for r in carry if r.get("reason") == "entry") / 100
    holds = [r for r in carry if r.get("reason") in ("entry", "exit")]

    print("=" * 60)
    print("Alpha2 Funding Carry Report (FUND_CARRY_V1)")
    print("=" * 60)
    print(f"Records: {n} | Entry/Exit events: {len(holds)}")
    print(f"Total fees: {total_fees:.4f} USDT")
    print(f"Avg fee per event: {avg_fees:.4f}")
    print(f"Avg slippage est (bp): {avg_slip:.2f}")
    print(f"Funding collected (proxy from regime_vol): {funding_collected_proxy:.4f}")
    print("Max adverse excursion: N/A (requires position-level tracking)")
    print("Hedge PnL: N/A (PAPER mode)")


if __name__ == "__main__":
    main()
