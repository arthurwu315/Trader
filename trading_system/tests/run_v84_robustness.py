"""
V8.4 Robustness Test: Mid-vol threshold plateau check.

Tests three MID vol block intervals:
  1) vol < 1.8% / 1.8–3.8% / >=3.8%
  2) vol < 2.0% / 2.0–4.0% / >=4.0% (baseline)
  3) vol < 2.2% / 2.2–4.2% / >=4.2%

Only the MID block interval changes; entry/exit/sizing unchanged.
"""
from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("SKIP_CONFIG_VALIDATION", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))

from run_v8_backtest import (
    run_strat_a_with_sizing,
    compute_metrics,
    filter_trades_by_year,
    add_indicators,
    resample_ohlcv,
    build_btc_mid_vol_set,
    to_utc_ts,
)
from backtest_utils import fetch_klines_df

START = "2022-01-01"
END = "2024-12-31"
C_GROUP = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "OPNUSDT",
    "AZTECUSDT", "DOGEUSDT", "1000PEPEUSDT", "ENSOUSDT", "BNBUSDT",
    "ESPUSDT", "INJUSDT", "ZECUSDT", "BCHUSDT", "SIRENUSDT",
    "YGGUSDT", "POWERUSDT", "KITEUSDT", "ETCUSDT", "PIPPINUSDT",
]

CONFIGS = [
    ("1.8–3.8%", 1.8, 3.8),   # 1) vol < 1.8 / 1.8–3.8 / >=3.8
    ("2.0–4.0%", 2.0, 4.0),   # 2) baseline
    ("2.2–4.2%", 2.2, 4.2),   # 3) vol < 2.2 / 2.2–4.2 / >=4.2
]


async def main():
    from bots.bot_c.config_c import get_strategy_c_config
    from core.binance_client import BinanceFuturesClient

    cfg = get_strategy_c_config()
    client = BinanceFuturesClient(
        api_key=cfg.binance_api_key or "dummy",
        api_secret=cfg.binance_api_secret or "dummy",
        base_url=os.getenv("BINANCE_DATA_URL", "https://fapi.binance.com"),
    )
    start_dt = datetime.strptime(START, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(END, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    def _fetch(sym):
        return sym, fetch_klines_df(client, sym, "1h", start_dt, end_dt)

    data_map = {}
    for sym in C_GROUP:
        _, h1 = await asyncio.to_thread(_fetch, sym)
        if h1.empty:
            continue
        h1 = add_indicators(h1)
        d1 = add_indicators(resample_ohlcv(h1, "1D"))
        data_map[sym] = {"1h": h1, "1d": d1}

    if "BTCUSDT" not in data_map:
        raise SystemExit("BTC data missing")

    btc_regime = {}
    btc = data_map["BTCUSDT"]["1d"]
    for _, r in btc.iterrows():
        ts = to_utc_ts(r["timestamp"]).floor("1D")
        close = float(r.get("close", 0) or 0)
        ema200 = float(r.get("ema_200", 0) or 0)
        if close > 0 and ema200 > 0:
            btc_regime[ts] = "bull" if close > ema200 else "bear"

    results = {}
    for label, vl, vh in CONFIGS:
        mid_set = build_btc_mid_vol_set(data_map, vol_low=vl, vol_high=vh)
        trades = run_strat_a_with_sizing(
            data_map, btc_regime, vol_mult_map=None,
            high_vol_stop_set=None, high_vol_regime_set=None, mid_vol_block_set=mid_set,
        )
        results[label] = trades

    days_per_year = {2022: 365, 2023: 365, 2024: 366}

    print("=" * 100)
    print("V8.4 Robustness: Mid-vol block threshold")
    print("=" * 100)

    for label in [c[0] for c in CONFIGS]:
        trades = results[label]
        m_full = compute_metrics(trades)
        print(f"\n--- {label} ---")
        print("| Period | CAGR | MDD | Calmar | PF | Trades | Exposure |")
        print("|--------|------|-----|--------|-----|--------|----------|")
        print(f"| Full | {m_full['CAGR']:.4f} | {m_full['MDD']:.4f} | {m_full['Calmar']:.4f} | {m_full['PF']:.4f} | {m_full['Trades']} | {m_full['Exposure']:.4f} |")
        for yr in (2022, 2023, 2024):
            ty = filter_trades_by_year(trades, yr)
            m = compute_metrics(ty, days_override=days_per_year.get(yr, 365))
            print(f"| {yr} | {m['CAGR']:.4f} | {m['MDD']:.4f} | {m['Calmar']:.4f} | {m['PF']:.4f} | {m['Trades']} | {m['Exposure']:.4f} |")

    calmar_vals = [compute_metrics(results[c[0]])["Calmar"] for c in CONFIGS]
    labels = [c[0] for c in CONFIGS]

    print("\n" + "=" * 100)
    print("QUESTIONS")
    print("=" * 100)
    print(f"1) Calmar > 0.6 for all? {all(c > 0.6 for c in calmar_vals)}")
    for lab, c in zip(labels, calmar_vals):
        print(f"   {lab}: Calmar = {c:.4f} {'✓' if c > 0.6 else '✗'}")
    calmar_range = max(calmar_vals) - min(calmar_vals) if calmar_vals else 0
    plateau = calmar_range < 0.15
    print(f"\n2) Plateau exists? {plateau} (Calmar range = {calmar_range:.4f}; < 0.15 suggests plateau)")
    best_idx = max(range(len(calmar_vals)), key=lambda i: calmar_vals[i])
    print(f"\n3) Most stable? {labels[best_idx]} (Calmar = {calmar_vals[best_idx]:.4f})")


if __name__ == "__main__":
    asyncio.run(main())
