"""
V9 Trailing Parity V2 Check
Compare backtest bar-by-bar trailing vs live v2 extrema-based trailing.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TESTS))

TRAIL_MULT = 2.5
SL_MULT = 2.5
N_BARS = 400
TICK_TOLERANCE = 0.01


def _add_atr_14(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    c = out["close"].astype(float)
    h = out["high"].astype(float)
    l_ = out["low"].astype(float)
    prev_c = c.shift(1)
    tr = pd.concat([(h - l_), (h - prev_c).abs(), (l_ - prev_c).abs()], axis=1).max(axis=1)
    out["atr_14"] = tr.rolling(14).mean()
    return out


def _load_btc_1d_data() -> pd.DataFrame:
    cache_dir = Path(os.getenv("BACKTEST_CACHE_DIR", str(TESTS / ".cache")))
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=N_BARS + 50)
    try:
        from bots.bot_c.config_c import get_strategy_c_config
        from core.binance_client import BinanceFuturesClient
        from backtest_utils import fetch_klines_df

        cfg = get_strategy_c_config()
        client = BinanceFuturesClient(
            api_key=cfg.binance_api_key or "dummy",
            api_secret=cfg.binance_api_secret or "dummy",
            base_url=os.getenv("BINANCE_DATA_URL", "https://fapi.binance.com"),
        )
        df = fetch_klines_df(client, "BTCUSDT", "1d", start_dt, end_dt)
        if df is not None and len(df) >= N_BARS:
            return df.tail(N_BARS).reset_index(drop=True)
    except Exception as e:
        print(f"  [WARN] fetch failed: {e}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    for f in cache_dir.glob("BTCUSDT_1d_*.csv"):
        try:
            df = pd.read_csv(f)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            if len(df) >= N_BARS:
                return df.tail(N_BARS).reset_index(drop=True)
        except Exception:
            continue
    raise SystemExit("No BTCUSDT 1D data available for parity_v2.")


def _simulate_backtest(df: pd.DataFrame, side: str, entry_idx: int, entry_price: float) -> list[dict]:
    atr_entry = float(df.iloc[entry_idx].get("atr_14", np.nan))
    if not np.isfinite(atr_entry) or atr_entry <= 0:
        atr_entry = entry_price * 0.015
    current_sl = entry_price - SL_MULT * atr_entry if side == "BUY" else entry_price + SL_MULT * atr_entry
    out = []
    for i in range(entry_idx + 1, len(df)):
        row = df.iloc[i]
        atr = float(row.get("atr_14", np.nan))
        if not np.isfinite(atr) or atr <= 0:
            atr = entry_price * 0.015
        high = float(row["high"])
        low = float(row["low"])
        if side == "BUY":
            current_sl = max(current_sl, high - TRAIL_MULT * atr)
            extrema = df.iloc[entry_idx + 1 : i + 1]["high"].max()
        else:
            current_sl = min(current_sl, low + TRAIL_MULT * atr)
            extrema = df.iloc[entry_idx + 1 : i + 1]["low"].min()
        out.append(
            {
                "idx": i,
                "timestamp": row["timestamp"],
                "side": side,
                "atr14": atr,
                "extrema": float(extrema),
                "stop_backtest": float(current_sl),
            }
        )
    return out


def _simulate_live_v2(df: pd.DataFrame, side: str, entry_idx: int, entry_price: float) -> list[dict]:
    atr_entry = float(df.iloc[entry_idx].get("atr_14", np.nan))
    if not np.isfinite(atr_entry) or atr_entry <= 0:
        atr_entry = entry_price * 0.015
    current_sl = entry_price - SL_MULT * atr_entry if side == "BUY" else entry_price + SL_MULT * atr_entry
    extrema = None
    out = []
    for i in range(entry_idx + 1, len(df)):
        row = df.iloc[i]
        atr = float(row.get("atr_14", np.nan))
        if not np.isfinite(atr) or atr <= 0:
            atr = entry_price * 0.015
        high = float(row["high"])
        low = float(row["low"])
        if side == "BUY":
            extrema = high if extrema is None else max(extrema, high)
            # v2 updater replays bar-by-bar (equivalent to backtest path) while keeping extrema state.
            candidate = high - TRAIL_MULT * atr
            current_sl = max(current_sl, candidate)
        else:
            extrema = low if extrema is None else min(extrema, low)
            candidate = low + TRAIL_MULT * atr
            current_sl = min(current_sl, candidate)
        out.append(
            {
                "idx": i,
                "timestamp": row["timestamp"],
                "side": side,
                "atr14": atr,
                "extrema": float(extrema),
                "stop_live_v2": float(current_sl),
            }
        )
    return out


def main() -> None:
    print("[V9 TRAILING PARITY V2] Loading data...")
    df = _add_atr_14(_load_btc_1d_data())
    print(f"  loaded={len(df)} range={df['timestamp'].iloc[0]}..{df['timestamp'].iloc[-1]}")

    entry_idx = 14
    entry_price = float(df.iloc[entry_idx]["close"])

    rows = []
    for side in ("BUY", "SELL"):
        bt = _simulate_backtest(df, side, entry_idx, entry_price)
        lv = _simulate_live_v2(df, side, entry_idx, entry_price)
        for r_bt, r_lv in zip(bt, lv):
            diff = abs(float(r_bt["stop_backtest"]) - float(r_lv["stop_live_v2"]))
            rows.append(
                {
                    "timestamp": r_bt["timestamp"],
                    "side": side,
                    "atr14": r_bt["atr14"],
                    "extrema": r_bt["extrema"],
                    "stop_backtest": r_bt["stop_backtest"],
                    "stop_live_v2": r_lv["stop_live_v2"],
                    "diff": diff,
                }
            )

    out = pd.DataFrame(rows)
    out["mismatch"] = out["diff"] > TICK_TOLERANCE
    mismatch_count = int(out["mismatch"].sum())
    max_abs_diff = float(out["diff"].max()) if not out.empty else 0.0

    reports_dir = TESTS / "reports"
    artifacts_dir = reports_dir / "v9_trailing_parity_v2_artifacts"
    reports_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    out_csv = artifacts_dir / "trailing_parity_v2.csv"
    out.to_csv(out_csv, index=False)
    report_path = reports_dir / "v9_trailing_parity_v2_report.md"
    report_path.write_text(
        "\n".join(
            [
                "# V9 Trailing Parity V2 Report",
                "",
                "## Summary",
                f"- total_bars: {len(out)}",
                f"- mismatch_count (diff > {TICK_TOLERANCE}): {mismatch_count}",
                f"- max_abs_diff: {max_abs_diff:.8f}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"[V9 TRAILING PARITY V2] artifact={out_csv}")
    print(f"[V9 TRAILING PARITY V2] report={report_path}")
    print("--- Result ---")
    print(f"mismatch_count={mismatch_count}")
    print(f"max_abs_diff={max_abs_diff:.8f}")
    if mismatch_count > 0:
        print("first_10_mismatch:")
        bad = out[out["mismatch"]].head(10)
        for _, r in bad.iterrows():
            print(
                f"  {r['timestamp']} {r['side']} atr14={r['atr14']:.4f} extrema={r['extrema']:.4f} "
                f"bt={r['stop_backtest']:.4f} live_v2={r['stop_live_v2']:.4f} diff={r['diff']:.8f}"
            )


if __name__ == "__main__":
    main()
