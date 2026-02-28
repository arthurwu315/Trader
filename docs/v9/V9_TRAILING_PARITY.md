# V9 Trailing Parity Check

## Purpose

Reproducible parity check between backtest and live trailing stop calculation. Proves alignment (or documents differences) with evidence, not claims.

## Method

- **Same inputs**: 1D OHLCV, ATR(14) SMA, trail_mult=2.5
- **Backtest side**: Bar-by-bar running max (BUY) or min (SELL) — from `run_v8_backtest._simulate_position_exit`
- **Live side**: Single-bar formula — `high - trail_mult*atr` (BUY) or `low + trail_mult*atr` (SELL) — from `ops/v9_trailing_updater._compute_candidate_stop`
- **Comparison**: For each bar (from entry+1), compute both stops; record diff and count mismatches (diff > tick tolerance)

## Data Source

- **Primary**: `fetch_klines_df` (backtest_utils) — BTCUSDT 1D, last N=400 bars from Binance API
- **Fallback**: Local cache at `tests/.cache/BTCUSDT_1d_*.csv` (from prior run_v8_backtest or other fetches)
- If neither available: script exits with instructions

## Results Summary

| Metric        | Typical Value (example run) |
|---------------|-----------------------------|
| total_bars    | 770 (385 BUY + 385 SELL)    |
| mismatch_count| 725                         |
| max_abs_diff  | ~64080 (BTC scale)          |

## Root Cause of Mismatches

1. **Single-bar vs running max/min**
   - Backtest: `current_sl = max(current_sl, high - trail*atr)` — keeps best so far
   - Live: `candidate = high - trail*atr` — uses last bar only
   - When price pulls back, live produces a **looser** stop (BUY) or **tighter** stop (SELL) than backtest

2. **Not caused by**: ATR smoothing (both SMA), timeframe (both 1D), tick rounding (tolerance 0.01)

## Artifacts

- `tests/reports/v9_trailing_parity_report.md` — summary + first 10 mismatches
- `tests/reports/v9_trailing_parity_artifacts/trailing_parity.csv` — full per-bar comparison

## Handling Mismatches

1. **Evidence first**: Run parity check, inspect report and CSV
2. **Then decide**: If mismatch root cause is acceptable (e.g., known single-bar design), document and proceed. If not, plan structural change in a future version — do not modify formulas without version bump.
