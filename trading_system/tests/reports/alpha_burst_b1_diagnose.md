# Alpha Burst B1 Diagnostic Report

Generated: 2026-02-28 15:39 UTC
Trade count: 342

---

## 1. R Distribution

### Quantiles
| Quantile | Value |
|----------|-------|
| p1 | -1.5271 |
| p5 | -0.9951 |
| p10 | -0.9082 |
| p50 | -0.2307 |
| p90 | 0.9910 |
| p95 | 1.4985 |
| p99 | 4.4518 |

### Top 20 positive R
See `alpha_burst_b1_diagnose_artifacts/top20_r.csv`

### Bottom 20 negative R
See `alpha_burst_b1_diagnose_artifacts/bottom20_r.csv`

---

## 2. Segment Buckets

### By regime_vol (LOW<2%, MID 2-4%, HIGH>=4%)
| bucket | trade_count | E[R] | PF | WinRate | AvgWin_R | AvgLoss_R |
|--------|-------------|------|-----|---------|----------|-----------|
| LOW | 339 | -0.0322 | 0.908 | 37.2% | 0.8536 | -0.5562 |
| MID | 3 | 0.3746 | 2.542 | 33.3% | 1.8526 | -0.3643 |

### By trend (4H close vs EMA200)
| bucket | trade_count | E[R] | PF | WinRate | AvgWin_R | AvgLoss_R |
|--------|-------------|------|-----|---------|----------|-----------|
| unknown | 342 | -0.0286 | 0.918 | 37.1% | 0.8615 | -0.5544 |

### By vol_expansion_flag
| bucket | trade_count | E[R] | PF | WinRate | AvgWin_R | AvgLoss_R |
|--------|-------------|------|-----|---------|----------|-----------|
| True | 342 | -0.0286 | 0.918 | 37.1% | 0.8615 | -0.5544 |

### By holding_bars
| bucket | trade_count | E[R] | PF | WinRate | AvgWin_R | AvgLoss_R |
|--------|-------------|------|-----|---------|----------|-----------|
| 1-3 | 104 | 0.0272 | 1.076 | 39.4% | 0.9790 | -0.5923 |
| 21+ | 17 | 0.1341 | 1.509 | 47.1% | 0.8442 | -0.4971 |
| 4-8 | 100 | -0.1712 | 0.588 | 27.0% | 0.9030 | -0.5685 |
| 9-20 | 121 | 0.0184 | 1.062 | 42.1% | 0.7477 | -0.5130 |

---

## 3. Fake Breakout (MFE/MAE)

| Metric | Value |
|--------|-------|
| avg MFE_R | 1.2356 |
| avg MAE_R | 1.1035 |
| pct MAE>1R then turned positive | 15.5% |

Artifacts: `mfe_mae.csv`

---

## 4. Permutation on Promising Buckets (200 runs)

| segment | bucket | trade_count | E[R] | perm_p |
|---------|--------|-------------|------|--------|
| holding_bars | 1-3 | 104 | 0.0272 | 0.8850 |
| holding_bars | 21+ | 17 | 0.1341 | 0.8000 |
| holding_bars | 9-20 | 121 | 0.0184 | 0.7350 |

**結論**：所有正 E[R] 桶 permutation p > 0.05，無法拒絕隨機假設。

---

## 5. Structural Candidate (Task 2)

**決策：B) 改 alpha 類型**

| 依據 | 桶統計 (2022-2024, n=342) |
|------|---------------------------|
| 流血區 | holding_bars 4-8 (n=100, E[R]=-0.17, WinRate 27%) |
| 略好區 | 1-3/9-20/21+ E[R] 略正但 perm_p 0.74–0.89 |
| MFE/MAE | avg MAE_R=1.10，15.5% 先 MAE>1R 再轉正 → 假突破特徵 |
| 支持方向 | B1 breakout 於 4-8 bar 明顯流血；無桶達顯著。建議改為 compression→expansion 或 regime flip，不沿用 B1 breakout。 |