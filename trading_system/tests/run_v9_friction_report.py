"""
V9.1 Friction Report.
Input: PAPER & MICRO-LIVE trade records (v9_trade_records.csv)
Output: fees (bp), slippage (bp), latency, PF/Win%/Expectancy diff (if enough samples).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.v9_trade_record import read_v9_records, get_v9_records_path


def parse_ts(s: str):
    from datetime import datetime
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def main():
    csv_path = get_v9_records_path()
    records = read_v9_records(csv_path)
    if not records:
        print("No V9 trade records found. Run PAPER or MICRO-LIVE to generate records.")
        print(f"Expected path: {csv_path}")
        return

    paper = [r for r in records if r.get("mode") == "PAPER"]
    live = [r for r in records if r.get("mode") == "MICRO-LIVE"]

    print("=" * 60)
    print("V9.1 Friction Report")
    print("=" * 60)
    print(f"Total records: {len(records)} | PAPER: {len(paper)} | MICRO-LIVE: {len(live)}")

    # Fee cost (bp) - fees / (price * qty) * 10000
    def fee_bps(r):
        notional = float(r.get("price", 0)) * float(r.get("qty", 0))
        if notional <= 0:
            return 0.0
        return float(r.get("fees", 0)) / notional * 10000

    for label, subset in [("PAPER", paper), ("MICRO-LIVE", live)]:
        if not subset:
            continue
        bps_list = [fee_bps(r) for r in subset]
        avg_bps = sum(bps_list) / len(bps_list) if bps_list else 0
        print(f"\n{label}:")
        print(f"  Avg fee cost (bp): {avg_bps:.2f}")

    # Slippage (bp)
    for label, subset in [("PAPER", paper), ("MICRO-LIVE", live)]:
        if not subset:
            continue
        slip = [float(r.get("slippage_est", 0)) for r in subset]
        avg_slip = sum(slip) / len(slip) if slip else 0
        print(f"  Avg slippage est (bp): {avg_slip:.2f}")

    # Latency (signal_time vs order_time)
    latency_sec = []
    for r in records:
        st = parse_ts(r.get("signal_time", ""))
        ot = parse_ts(r.get("order_time", ""))
        if st and ot:
            delta = (ot - st).total_seconds()
            latency_sec.append(delta)
    if latency_sec:
        print(f"\nLatency (signal_time -> order_time):")
        print(f"  Avg: {sum(latency_sec)/len(latency_sec):.2f}s | Min: {min(latency_sec):.2f}s | Max: {max(latency_sec):.2f}s")
    else:
        print("\nLatency: no signal_time/order_time pairs (sample PAPER uses same ts)")

    # PF / Win% / Expectancy diff (if enough samples)
    min_sample = 10
    if len(paper) >= min_sample and len(live) >= min_sample:
        def pnl_list(recs):
            # Entry records with price/qty - we'd need exit to calc pnl; use fees as proxy
            return [-float(r.get("fees", 0)) for r in recs]

        pnl_p = pnl_list(paper)
        pnl_l = pnl_list(live)
        gp_p = sum(x for x in pnl_p if x > 0)
        gl_p = sum(abs(x) for x in pnl_p if x < 0)
        gp_l = sum(x for x in pnl_l if x > 0)
        gl_l = sum(abs(x) for x in pnl_l if x < 0)
        pf_p = gp_p / gl_p if gl_p > 0 else 0
        pf_l = gp_l / gl_l if gl_l > 0 else 0
        win_p = sum(1 for x in pnl_p if x > 0) / len(pnl_p) * 100 if pnl_p else 0
        win_l = sum(1 for x in pnl_l if x > 0) / len(pnl_l) * 100 if pnl_l else 0
        print(f"\nLive vs Paper (fees-only proxy, N>={min_sample}):")
        print(f"  PF: Paper {pf_p:.2f} | Live {pf_l:.2f}")
        print(f"  Win%: Paper {win_p:.1f} | Live {win_l:.1f}")
    else:
        print(f"\nPF/Win%/Expectancy: need >= {min_sample} samples each (PAPER={len(paper)}, LIVE={len(live)})")


if __name__ == "__main__":
    main()
