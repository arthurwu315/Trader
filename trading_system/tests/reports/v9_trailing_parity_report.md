# V9 Trailing Parity Report

## Summary
- **total_bars**: 770
- **mismatch_count** (diff > 0.01): 725
- **max_abs_diff**: 64080.196429

## Method
- Same 1D OHLCV, same ATR(14) SMA, trail_mult=2.5
- Backtest: bar-by-bar running max (BUY) / min (SELL)
- Live: single-bar formula high - trail*atr (BUY) / low + trail*atr (SELL)
- Expect mismatches when price pulls back (live uses current bar only)

## First 10 Mismatches

- 2025-02-12 00:00:00+00:00 BUY atr=4173.3714 hh_ll=98090.90 stop_bt=88156.2179 stop_live=87657.4714 diff=498.746429
- 2025-02-13 00:00:00+00:00 BUY atr=4149.6714 hh_ll=98053.80 stop_bt=88156.2179 stop_live=87679.6214 diff=476.596429
- 2025-02-15 00:00:00+00:00 BUY atr=3887.3429 hh_ll=97941.20 stop_bt=88762.1750 stop_live=88222.8429 diff=539.332143
- 2025-02-16 00:00:00+00:00 BUY atr=3620.7286 hh_ll=97665.80 stop_bt=88762.1750 stop_live=88613.9786 diff=148.196429
- 2025-02-22 00:00:00+00:00 BUY atr=2624.6286 hh_ll=96988.00 stop_bt=92876.3750 stop_live=90426.4286 diff=2449.946429
- 2025-02-23 00:00:00+00:00 BUY atr=2536.8500 hh_ll=96602.90 stop_bt=92876.3750 stop_live=90260.7750 diff=2615.600000
- 2025-02-24 00:00:00+00:00 BUY atr=2682.7143 hh_ll=96470.00 stop_bt=92876.3750 stop_live=89763.2143 diff=3113.160714
- 2025-02-25 00:00:00+00:00 BUY atr=2883.5500 hh_ll=92500.00 stop_bt=92876.3750 stop_live=85291.1250 diff=7585.250000
- 2025-02-26 00:00:00+00:00 BUY atr=3101.9857 hh_ll=89367.10 stop_bt=92876.3750 stop_live=81612.1357 diff=11264.239286
- 2025-02-27 00:00:00+00:00 BUY atr=3209.5071 hh_ll=87066.00 stop_bt=92876.3750 stop_live=79042.2321 diff=13834.142857

## Diff Root Cause
- **Single-bar vs running max/min**: Backtest keeps rolling best; live uses last bar only.
- When price pulls back, live produces a looser (BUY) or tighter (SELL) stop than backtest.
