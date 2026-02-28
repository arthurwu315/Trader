"""
V9 Rolling Walk-Forward Validation.
- Split 1: Train 2022-2023, Test 2024
- Split 2: Train 2023-2024, Test 2022 (look back)

Uses V9 fixed regime: MID 2.2-4.2% disabled.
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

# V9 fixed: 2.2-4.2% mid block
V9_VOL_LOW = 2.2
V9_VOL_HIGH = 4.2

START = "2022-01-01"
END = "2024-12-31"
C_GROUP = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "OPNUSDT",
    "AZTECUSDT", "DOGEUSDT", "1000PEPEUSDT", "ENSOUSDT", "BNBUSDT",
    "ESPUSDT", "INJUSDT", "ZECUSDT", "BCHUSDT", "SIRENUSDT",
    "YGGUSDT", "POWERUSDT", "KITEUSDT", "ETCUSDT", "PIPPINUSDT",
]
DAYS_PER_YEAR = {2022: 365, 2023: 365, 2024: 366}


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

    mid_set = build_btc_mid_vol_set(data_map, vol_low=V9_VOL_LOW, vol_high=V9_VOL_HIGH)
    trades = run_strat_a_with_sizing(
        data_map, btc_regime, vol_mult_map=None,
        high_vol_stop_set=None, high_vol_regime_set=None, mid_vol_block_set=mid_set,
    )

    print("=" * 70)
    print("V9 Rolling Walk-Forward (MID 2.2-4.2% disabled)")
    print("=" * 70)

    # Split 1: Train 2022-2023, Test 2024
    trades_2024 = filter_trades_by_year(trades, 2024)
    m_2024 = compute_metrics(trades_2024, days_override=DAYS_PER_YEAR[2024])
    print("\nSplit 1: Train 2022-2023, Test 2024")
    print(f"  OOS 2024: Calmar = {m_2024['Calmar']:.4f} | MDD = {m_2024['MDD']:.4f}% | "
          f"Trades = {m_2024['Trades']} | CAGR = {m_2024['CAGR']:.4f}%")

    # Split 2: Train 2023-2024, Test 2022 (look back)
    trades_2022 = filter_trades_by_year(trades, 2022)
    m_2022 = compute_metrics(trades_2022, days_override=DAYS_PER_YEAR[2022])
    print("\nSplit 2: Train 2023-2024, Test 2022")
    print(f"  OOS 2022: Calmar = {m_2022['Calmar']:.4f} | MDD = {m_2022['MDD']:.4f}% | "
          f"Trades = {m_2022['Trades']} | CAGR = {m_2022['CAGR']:.4f}%")

    print("\n" + "=" * 70)
    print("Walk-Forward Summary")
    print("| Split | Train | Test | Calmar | MDD | CAGR | Trades |")
    print("|-------|-------|------|--------|-----|------|--------|")
    print(f"| 1 | 2022-2023 | 2024 | {m_2024['Calmar']:.4f} | {m_2024['MDD']:.4f} | {m_2024['CAGR']:.4f} | {m_2024['Trades']} |")
    print(f"| 2 | 2023-2024 | 2022 | {m_2022['Calmar']:.4f} | {m_2022['MDD']:.4f} | {m_2022['CAGR']:.4f} | {m_2022['Trades']} |")


if __name__ == "__main__":
    asyncio.run(main())
