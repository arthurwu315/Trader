# Alpha Burst B2 COMPRESS Report

Generated: 2026-02-28 16:30 UTC
Trade count: 188

---

## B1: Performance by Year / Symbol

### Full sample
| Metric | Value |
|--------|-------|
| E[R] | -0.1480 |
| WinRate | 34.0426 |
| AvgWin_R | 0.7407 |
| AvgLoss_R | -0.6066 |
| p10 | -0.9462 |
| p50 | -0.2722 |
| p90 | 0.7562 |
| p95 | 1.2683 |
| trade_count | 188 |

### By year
**2023** (n=53)
- E[R]=-0.0897 WinRate=37.7%

**2024** (n=63)
- E[R]=-0.1737 WinRate=36.5%

**2025** (n=62)
- E[R]=-0.1471 WinRate=30.6%

**2026** (n=10 INSUFFICIENT POWER)
- E[R]=-0.3000 WinRate=20.0%

### By symbol
**BTCUSDT** (n=99)
- E[R]=-0.2172 WinRate=29.3%

**ETHUSDT** (n=89)
- E[R]=-0.0710 WinRate=39.3%

## B2: Parameter Grid & Plateau Regions

Grid (run `python3 -m tests.run_alpha_burst_b2_grid` to generate):
- compression_atr_pct: [20, 30, 40]
- expansion_threshold: [1.05, 1.1, 1.2]
- compression_range_lookback: [30, 40, 50]

## B3: Statistical Validation

### Permutation test (1000, SEED=42)
- Observed E[R]: -0.1480
- 95th percentile of permuted: 0.1100
- p-value (permuted >= observed): 0.9890

### Block bootstrap (1000, block=7 days, SEED=42)
- 95% CI for E[R]: [-0.2720, -0.0184]

Artifacts: `b3_permutation.csv`, `b3_bootstrap.csv`

## B4: Cost Stress

| Scenario | E[R] stressed | WinRate stressed | Avg cost (R) |
|----------|---------------|------------------|--------------|
| slippage_5bps | -0.3052 | 28.2% | 0.1572 |
| slippage_10bps | -0.3838 | 26.6% | 0.2359 |
| slippage_20bps | -0.5411 | 21.3% | 0.3931 |
| high_vol_2.0x | -0.1480 | 34.0% | N/A (2x risk) |

Artifacts: `b4_cost_stress.csv`

---

## Acceptance

```bash
python3 -m tests.run_alpha_burst_b2_backtest
python3 -m tests.run_alpha_burst_b2_report
grep ALPHA_BURST_B2_COMPRESS logs/v9_trade_records.csv | tail -n 20
cat STRATEGY_STATE_ALPHA_BURST_B2.md
```