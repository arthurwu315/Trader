# STRATEGY_STATE_ALPHA_BURST_B2

**strategy_id**: ALPHA_BURST_B2_COMPRESS  
**Type**: Compression → expansion burst (no Donchian breakout)  
**Status**: Pre-validation (default disabled, no micro-live until validation passes)

---

## 1. Strategy Structure

| Component | Value |
|-----------|-------|
| Universe | BTCUSDT, ETHUSDT |
| Trend filter | 4H close vs EMA200 (long/short only) |
| Entry TF | 1H |
| Compression | ATR14 ≤ 30th percentile of past 80 bars |
| Armed | ≥5 compression bars in last 30 bars |
| Expansion | ATR14 > ATR14_ma20 × 1.1 |
| Entry | First expansion after armed + close breaks compression range high/low |
| Exit | ATR×k stop + ATR trailing |
| Position sizing | 1% burst equity per trade |

---

## 2. Governance

- B2 預設不得 micro-live
- 上線門檻：E[R] > 0 且 permutation p < 0.05 且成本壓力下仍為正
- B1 維持 ENABLE_ALPHA_BURST=false（預設）
- 獨立於 V9 核心

---

## 3. Commands

```bash
python3 -m tests.run_alpha_burst_b2_backtest
python3 -m tests.run_alpha_burst_b2_report
python3 -m tests.run_alpha_burst_b2_grid   # optional: plateau grid
```

---

## 4. Trade Record

- File: `logs/alpha_burst_b2_trades.csv`
- v9: `strategy_id=ALPHA_BURST_B2_COMPRESS`
- Fields: compression_flag, armed_flag, expansion_flag, compression_high, compression_low, entry_breakout_side

---

END OF STRATEGY_STATE_ALPHA_BURST_B2
