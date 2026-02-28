# Alpha Burst B1 Report

Generated: 2026-02-28 15:17 UTC
Trade count: 581

---

## B1: Performance by Year / Symbol

### Full sample
| Metric | Value |
|--------|-------|
| E[R] | -0.0486 |
| WinRate | 38.38% |
| AvgWin_R | 0.7873 |
| AvgLoss_R | -0.5694 |
| p10 | -0.9117 |
| p50 | -0.2068 |
| p90 | 0.9212 |
| p95 | 1.5031 |
| trade_count | 581 |

### By year
**2023** (n=155)
- E[R]=-0.0078 WinRate=36.1% AvgWin_R=0.9905 AvgLoss_R=-0.5726
- p10=-0.9540 p50=-0.2051 p90=0.9339 p95=1.7496

**2024** (n=187)
- E[R]=-0.0459 WinRate=38.0% AvgWin_R=0.7597 AvgLoss_R=-0.5389
- p10=-0.8623 p50=-0.2559 p90=1.0149 p95=1.4011

**2025** (n=198)
- E[R]=-0.0796 WinRate=39.4% AvgWin_R=0.6825 AvgLoss_R=-0.5750
- p10=-0.9201 p50=-0.1992 p90=0.8512 p95=1.4575

**2026** (n=41)
- E[R]=-0.0658 WinRate=43.9% AvgWin_R=0.7185 AvgLoss_R=-0.6795
- p10=-0.9028 p50=-0.3857 p90=0.8934 p95=0.9212

### By symbol
**BTCUSDT** (n=283)
- E[R]=-0.0275 WinRate=41.7%

**ETHUSDT** (n=298)
- E[R]=-0.0687 WinRate=35.2%

## B2: Parameter Grid & Plateau Regions

Grid (run `python3 -m tests.run_alpha_burst_grid` to generate results):
- vol_expansion_threshold: [1.0, 1.2, 1.5]
- stop_ATR_k: [1.5, 2.0, 2.5]
- breakout_lookback: [15, 20, 25]

## B3: Statistical Validation

### Permutation test (1000, SEED=42)
- Observed E[R]: -0.0486
- 95th percentile of permuted: 0.0615
- p-value (permuted >= observed): 0.9160

### Block bootstrap (1000, SEED=42)
- 95% CI for E[R]: [-0.1133, 0.0185]

Artifacts: `b3_permutation.csv`, `b3_bootstrap.csv`

## B4: Cost Stress

| Scenario | E[R] stressed | WinRate stressed | Avg cost (R) |
|----------|---------------|------------------|--------------|
| slippage_5bps | -0.1818 | 33.6% | 0.1332 |
| slippage_10bps | -0.2484 | 29.6% | 0.1998 |
| slippage_20bps | -0.3815 | 25.3% | 0.3329 |
| high_vol_2.0x | -0.0486 | 38.4% | N/A (2x risk) |

Artifacts: `b4_cost_stress.csv`

---

## Acceptance

```bash
python3 -m tests.run_alpha_burst_report
grep ALPHA_BURST_B1 logs/v9_trade_records.csv | tail -n 20
cat STRATEGY_STATE_ALPHA_BURST.md
```