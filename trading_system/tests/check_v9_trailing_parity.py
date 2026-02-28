"""
V9 Trailing Parity Check: compare backtest vs live trailing stop calculation.
Uses same 1D OHLCV, same ATR(14), same trail_mult. Reports max_abs_diff and mismatches.
Hard constraint: does NOT modify strategy/updater logic or formulas.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Path setup
ROOT = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TESTS))

# V9 params (from v9_trailing_updater - do not change)
TRAIL_MULT = 2.5
TICK_TOLERANCE = 0.01  # BTC ~0.01; tolerance for "match"
N_BARS = 400


def _add_atr_14(df: pd.DataFrame) -> pd.DataFrame:
    """Same ATR(14) as run_v8_backtest add_indicators."""
    out = df.copy()
    c = out["close"].astype(float)
    h = out["high"].astype(float)
    l_ = out["low"].astype(float)
    prev_c = c.shift(1)
    tr = pd.concat([(h - l_), (h - prev_c).abs(), (l_ - prev_c).abs()], axis=1).max(axis=1)
    out["atr_14"] = tr.rolling(14).mean()
    return out


def _compute_atr_14_from_bars(bars: list[dict]) -> float:
    """Same as ops/v9_trailing_updater._compute_atr_14 (for live-side parity)."""
    if len(bars) < 15:
        return 0.0
    trs = []
    for i in range(1, len(bars)):
        h = bars[i]["high"]
        l_ = bars[i]["low"]
        prev_c = bars[i - 1]["close"]
        tr = max(h - l_, abs(h - prev_c), abs(l_ - prev_c))
        trs.append(tr)
    return sum(trs[-14:]) / 14 if len(trs) >= 14 else 0.0


def _backtest_trailing_sequence(
    df: pd.DataFrame,
    side: str,
    entry_idx: int,
    entry_price: float,
    sl_mult: float,
    trail_mult: float,
) -> list[tuple[int, float, float, float]]:
    """
    Simulate backtest trailing: bar-by-bar running max (BUY) or min (SELL).
    Returns list of (idx, atr, hh_or_ll, stop_backtest) for each bar from entry_idx+1.
    """
    atr_col = df["atr_14"]
    initial_sl = entry_price - sl_mult * float(atr_col.iloc[entry_idx]) if side == "BUY" else entry_price + sl_mult * float(atr_col.iloc[entry_idx])
    if not np.isfinite(initial_sl) or initial_sl <= 0:
        atr_fallback = entry_price * 0.015
        initial_sl = entry_price - sl_mult * atr_fallback if side == "BUY" else entry_price + sl_mult * atr_fallback

    result = []
    current_sl = initial_sl
    for i in range(entry_idx + 1, len(df)):
        row = df.iloc[i]
        atr = float(row.get("atr_14", np.nan))
        if not np.isfinite(atr) or atr <= 0:
            atr = entry_price * 0.015
        high = float(row["high"])
        low = float(row["low"])
        if side == "BUY":
            current_sl = max(current_sl, high - trail_mult * atr)
            hh_ll = high
        else:
            current_sl = min(current_sl, low + trail_mult * atr)
            hh_ll = low
        result.append((i, atr, hh_ll, current_sl))
    return result


def _live_trailing_at_bar(bars: list[dict], bar_idx: int, side: str, entry_price: float, trail_mult: float) -> float | None:
    """
    What live updater would compute if run at end of bar_idx.
    Uses last bar's high/low and ATR(14) from bars[0:bar_idx+1].
    """
    if bar_idx < 14:
        return None
    window = bars[: bar_idx + 1]
    atr = _compute_atr_14_from_bars(window)
    if atr <= 0:
        atr = entry_price * 0.015
    last = window[-1]
    high = last["high"]
    low = last["low"]
    if side == "BUY":
        return high - trail_mult * atr
    return low + trail_mult * atr


def load_btc_1d_data() -> pd.DataFrame:
    """Load BTCUSDT 1D data: fetch via API or read from cache."""
    cache_dir = Path(os.getenv("BACKTEST_CACHE_DIR", str(TESTS / ".cache")))
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=N_BARS + 50)

    # Try fetch via existing backtest_utils
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
        print(f"  [WARN] Fetch failed: {e}")

    # Fallback: read from cache
    cache_dir.mkdir(parents=True, exist_ok=True)
    start_key = start_dt.strftime("%Y%m%d")
    end_key = end_dt.strftime("%Y%m%d")
    cache_path = cache_dir / f"BTCUSDT_1d_{start_key}_{end_key}.csv"
    if cache_path.exists():
        try:
            df = pd.read_csv(cache_path)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            if len(df) >= N_BARS:
                return df.tail(N_BARS).reset_index(drop=True)
        except Exception as e:
            print(f"  [WARN] Cache read failed: {e}")

    # Find any covering cache
    if cache_dir.exists():
        for f in cache_dir.glob("BTCUSDT_1d_*.csv"):
            try:
                df = pd.read_csv(f)
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                if len(df) >= N_BARS:
                    return df.tail(N_BARS).reset_index(drop=True)
            except Exception:
                pass
    raise SystemExit("No BTCUSDT 1D data (fetch failed, no cache). Run run_v8_backtest first or ensure API/cache.")


def run_parity_check() -> dict:
    """Run parity check and return summary dict."""
    print("[V9 TRAILING PARITY] Loading data...")
    df = load_btc_1d_data()
    print(f"  Loaded {len(df)} bars, range {df['timestamp'].iloc[0]} .. {df['timestamp'].iloc[-1]}")
    df = _add_atr_14(df)
    bars = [
        {"timestamp": r["timestamp"], "open": r["open"], "high": r["high"], "low": r["low"], "close": r["close"]}
        for _, r in df.iterrows()
    ]

    entry_idx = 14
    entry_price = float(df.iloc[entry_idx]["close"])
    sl_mult = 2.5

    rows = []
    for side in ("BUY", "SELL"):
        bt_seq = _backtest_trailing_sequence(df, side, entry_idx, entry_price, sl_mult, TRAIL_MULT)
        for i, (idx, atr, hh_ll, stop_bt) in enumerate(bt_seq):
            stop_live = _live_trailing_at_bar(bars, idx, side, entry_price, TRAIL_MULT)
            if stop_live is None:
                continue
            diff = abs(stop_bt - stop_live)
            ts = df.iloc[idx]["timestamp"]
            rows.append({
                "side": side,
                "timestamp": ts,
                "atr": atr,
                "hh_ll": hh_ll,
                "stop_backtest": stop_bt,
                "stop_live": stop_live,
                "diff": diff,
            })

    if not rows:
        return {"mismatch_count": 0, "max_abs_diff": 0.0, "rows": pd.DataFrame(), "total_bars": 0}

    df_out = pd.DataFrame(rows)
    df_out["mismatch"] = df_out["diff"] > TICK_TOLERANCE
    mismatch_count = int(df_out["mismatch"].sum())
    max_abs_diff = float(df_out["diff"].max())

    return {
        "mismatch_count": mismatch_count,
        "max_abs_diff": max_abs_diff,
        "rows": df_out,
        "total_bars": len(df_out),
    }


def main() -> None:
    result = run_parity_check()
    rows: pd.DataFrame = result["rows"]
    mismatch_count = result["mismatch_count"]
    max_abs_diff = result["max_abs_diff"]
    total_bars = result["total_bars"]

    reports_dir = TESTS / "reports"
    artifacts_dir = reports_dir / "v9_trailing_parity_artifacts"
    reports_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # trailing_parity.csv
    if len(rows) > 0:
        out_csv = artifacts_dir / "trailing_parity.csv"
        rows.to_csv(out_csv, index=False)
        print(f"\n[V9 TRAILING PARITY] Artifact: {out_csv}")

    # Report
    report_lines = [
        "# V9 Trailing Parity Report",
        "",
        "## Summary",
        f"- **total_bars**: {total_bars}",
        f"- **mismatch_count** (diff > {TICK_TOLERANCE}): {mismatch_count}",
        f"- **max_abs_diff**: {max_abs_diff:.6f}",
        "",
        "## Method",
        "- Same 1D OHLCV, same ATR(14) SMA, trail_mult=2.5",
        "- Backtest: bar-by-bar running max (BUY) / min (SELL)",
        "- Live: single-bar formula high - trail*atr (BUY) / low + trail*atr (SELL)",
        "- Expect mismatches when price pulls back (live uses current bar only)",
        "",
    ]

    if mismatch_count > 0:
        report_lines.append("## First 10 Mismatches")
        report_lines.append("")
        mismatches = rows[rows["mismatch"]].head(10)
        for _, r in mismatches.iterrows():
            report_lines.append(
                f"- {r['timestamp']} {r['side']} atr={r['atr']:.4f} hh_ll={r['hh_ll']:.2f} "
                f"stop_bt={r['stop_backtest']:.4f} stop_live={r['stop_live']:.4f} diff={r['diff']:.6f}"
            )
        report_lines.append("")
        report_lines.append("## Diff Root Cause")
        report_lines.append("- **Single-bar vs running max/min**: Backtest keeps rolling best; live uses last bar only.")
        report_lines.append("- When price pulls back, live produces a looser (BUY) or tighter (SELL) stop than backtest.")
    else:
        report_lines.append("## Result")
        report_lines.append("All bars within tolerance. Formulas match for this dataset.")
    report_lines.append("")

    report_path = reports_dir / "v9_trailing_parity_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"[V9 TRAILING PARITY] Report: {report_path}")

    # Console output
    print(f"\n--- Result ---")
    print(f"  max_abs_diff = {max_abs_diff:.6f}")
    print(f"  mismatch_count = {mismatch_count}")
    if mismatch_count > 0 and len(rows) > 0 and "mismatch" in rows.columns:
        print(f"\n  First 10 mismatches (diff > {TICK_TOLERANCE}):")
        mismatches = rows[rows["mismatch"]].head(10)
        for _, r in mismatches.iterrows():
            print(
                f"    {r['timestamp']} {r['side']} atr={r['atr']:.4f} hh_ll={r['hh_ll']:.2f} "
                f"stop_bt={r['stop_backtest']:.4f} stop_live={r['stop_live']:.4f} diff={r['diff']:.6f}"
            )


if __name__ == "__main__":
    main()
