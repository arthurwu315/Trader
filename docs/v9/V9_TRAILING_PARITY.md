# V9 Trailing Parity Check

## Purpose

Reproducible parity check between backtest and live trailing stop calculation. Proves alignment (or documents differences) with evidence, not claims.

## Method

- **Same inputs**: 1D OHLCV, ATR(14) SMA, trail_mult=2.5
- **Backtest side**: Bar-by-bar running max (BUY) or min (SELL) — from `run_v8_backtest._simulate_position_exit`
- **Live v1 side**: Single-bar formula — `high - trail_mult*atr` (BUY) or `low + trail_mult*atr` (SELL)
- **Live v2 side**: Since-entry bar replay (stateful) with monotonic tightening, equivalent to backtest bar-by-bar path
- **Comparison**: For each bar (from entry+1), compute both stops; record diff and count mismatches (diff > tick tolerance)

## Data Source

- **Primary**: `fetch_klines_df` (backtest_utils) — BTCUSDT 1D, last N=400 bars from Binance API
- **Fallback**: Local cache at `tests/.cache/BTCUSDT_1d_*.csv` (from prior run_v8_backtest or other fetches)
- If neither available: script exits with instructions

## Results Summary

### v1 (single-bar candidate) – parity failed

| Metric | Example Run |
|---|---:|
| total_bars | 770 |
| mismatch_count | 725 |
| max_abs_diff | 64080.196429 |

### v2 (since-entry replay + state) – parity aligned

| Metric | Example Run |
|---|---:|
| total_bars | 770 |
| mismatch_count | 0 |
| max_abs_diff | 0.00000000 |

## Root Cause of v1 Mismatches

1. **Single-bar vs running max/min**
   - Backtest: `current_sl = max(current_sl, high - trail*atr)` — keeps best so far
   - Live: `candidate = high - trail*atr` — uses last bar only
   - When price pulls back, live produces a **looser** stop (BUY) or **tighter** stop (SELL) than backtest

2. **Not caused by**: ATR smoothing (both SMA), timeframe (both 1D), tick rounding (tolerance 0.01)

## Artifacts

- `tests/reports/v9_trailing_parity_report.md` — summary + first 10 mismatches
- `tests/reports/v9_trailing_parity_artifacts/trailing_parity.csv` — full per-bar comparison
- `tests/reports/v9_trailing_parity_v2_report.md` — v2 parity summary
- `tests/reports/v9_trailing_parity_v2_artifacts/trailing_parity_v2.csv` — v2 full comparison

## Commands

```bash
cd trading_system && python3 -m tests.check_v9_trailing_parity
cd trading_system && python3 -m tests.check_v9_trailing_parity_v2
```

## Handling Mismatches

1. **Evidence first**: Run parity check, inspect report and CSV
2. **Then decide**: If mismatch root cause is acceptable (e.g., known single-bar design), document and proceed. If not, plan structural change in a future version — do not modify formulas without version bump.
