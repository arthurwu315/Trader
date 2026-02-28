# ALPHA_BURST_B1 Postmortem

**Date**: 2026-02-28  
**Decision**: B1 禁止上線（Burst allocation=0%），不得進入 micro-live

---

## 1. Validation Results (2022–2024)

| Metric | Value |
|--------|-------|
| E[R] | -0.0486 |
| WinRate | 38.38% |
| AvgWin_R | 0.7873 |
| AvgLoss_R | -0.5694 |
| p10 / p50 / p90 / p95 | -0.91 / -0.21 / 0.92 / 1.50 |
| trade_count | 581 |
| Permutation p-value | 0.9160 |
| Bootstrap 95% CI for E[R] | [-0.1133, 0.0185] |

---

## 2. Conclusion

- E[R] < 0
- Permutation p-value 0.916 → 無法拒絕「無 edge」假設
- Bootstrap CI 含 0 且偏負
- **B1 無統計優勢，禁止上線**

---

## 3. Diagnostic Summary (2022-2024, n=342)

**holding_bars**:
| bucket | trade_count | E[R] | WinRate | perm_p |
|--------|-------------|------|---------|--------|
| 1-3 | 104 | 0.027 | 39.4% | 0.89 |
| 4-8 | 100 | -0.171 | 27.0% | - |
| 9-20 | 121 | 0.018 | 42.1% | 0.74 |
| 21+ | 17 | 0.134 | 47.1% | 0.80 |

4-8 bar 流血最重；其餘桶 permutation 未達顯著。

**regime_vol**: 339 LOW (E[R]=-0.03), 3 MID；HIGH 無。**MFE/MAE**: avg MFE_R=1.24, avg MAE_R=1.10；15.5% 先 MAE>1R 再轉正。

## 4. Next Structural Candidate

**決策：B) 改 alpha 類型**

- 解剖顯示：4-8 bar 桶 E[R]=-0.17、WinRate 27% 流血最重；其餘桶 E[R] 略正但 permutation p 均 > 0.05
- 結論：Donchian breakout 結構易產生假突破 / whipsaw，不適合微調
- 建議：compression→expansion 或 regime flip，完全放棄 B1 breakout entry

---

END OF POSTMORTEM
