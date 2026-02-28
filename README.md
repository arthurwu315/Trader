## Version & Experiment Log

### V7A1 (Experimental)

Core Change:
C 引擎 Score 計算方式由 Raw Momentum (roc_20)
改為 Risk-Adjusted Momentum:

Score = ROC_20 × (1 / sqrt(ATR_20))

Purpose:
驗證風險調整後動能是否能提升：
- 勝率
- Calmar Ratio
且最大回撤 (MDD) 不得高於 V7 基準。

Note:
本版本為單一變數控制實驗。
未加入任何額外門檻或濾網。

### V7A1 OOS Stability Validation

| Year | Version | CAGR | MDD | Calmar | PF | Win% | Trades | Exposure |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2022 | V7 | -0.1766 | 1.3236 | -0.1334 | 1.1061 | 40.0000 | 15* | 0.7897 |
| 2022 | V7A1 | -0.0734 | 1.2296 | -0.0597 | 1.1507 | 40.0000 | 15* | 0.7897 |
| 2023 | V7 | 30.3109 | 5.6887 | 5.3283 | 3.4347 | 46.6667 | 15* | 20.0526 |
| 2023 | V7A1 | 30.3109 | 5.6887 | 5.3283 | 3.4347 | 46.6667 | 15* | 20.0526 |
| 2024 | V7 | 35.3776 | 8.4320 | 4.1956 | 2.2249 | 55.5556 | 9* | 18.3541 |
| 2024 | V7A1 | 35.3776 | 8.4320 | 4.1956 | 2.2249 | 55.5556 | 9* | 18.3541 |

\* Trades < 20

Bootstrap (1000 resamples, Full period 2022-01-01 ~ 2024-12-31):

| Version | Calmar Mean | Calmar Std |
|---|---:|---:|
| V7 | 0.4953 | 0.6881 |
| V7A1 | 0.5623 | 0.7602 |

| Metric | Value |
|---|---:|
| V7A1 Calmar > V7 (%) | 52.7 |

### V7B RSI Threshold Robustness (2022–2024)

實驗：C engine RSI entry threshold 網格掃描  
- BUY: rsi < 8, 10, 12, 15, 18  
- SELL: rsi > 92, 90, 88, 85, 82  
- 其餘條件不變（scoring、risk、sizing）

完整報表：`v7b_work/trading_system/tests/reports/v7b_results_full.md`

**Calmar Top 5 (Full):**

| Rank | BUY | SELL | Calmar | CAGR | MDD | TradeCount |
|---|---|---:|---:|---:|---:|---:|
| 1 | 12 | 82 | 1.0531 | 6.20 | 5.89 | 212 |
| 2 | 15 | 90 | 0.9808 | 2.43 | 2.48 | 68 |
| 3 | 10 | 85 | 0.9075 | 3.25 | 3.59 | 95 |
| 4 | 15 | 85 | 0.7718 | 3.74 | 4.85 | 129 |
| 5 | 10 | 82 | 0.7417 | 5.11 | 6.89 | 187 |

**數據結論（僅基於回測）：**
1. **Plateau 區域**：存在。BUY 10–15 搭配 SELL 82–90 多組 Calmar ≥ 0.6。
2. **最佳區間集中**：是。最佳組合集中在 BUY 10–15、SELL 82–90；SELL 92 普遍表現差。
3. **V7B（15/85）位置**：中段偏上，非極值。Calmar 排名第 4 / 25。

### C Engine Edge Validation (2022-2024)

Edge Test Summary

**RSI < 10**

| Window | Mean(%) | Std | Win% | N |
| 6h | 0.3825 | 1.5788 | 67.27 | 55 |
| 12h | 0.5103 | 2.4114 | 69.09 | 55 |
| 24h | 0.2978 | 4.2684 | 52.73 | 55 |

t-test vs Random:
- 6h p=0.1248
- 12h p=0.1285
- 24h p=0.3616

**RSI > 90**

| Window | Mean(%) | Std | Win% | N |
| 6h | -0.0696 | 1.2699 | 44.07 | 118 |
| 12h | 0.2435 | 1.7879 | 54.24 | 118 |
| 24h | -0.0859 | 2.5143 | 56.78 | 118 |

t-test vs Random:
- 6h p=0.8478
- 12h p=0.2326
- 24h p=0.5548

**Conclusion**

- RSI 極端未達統計顯著
- p-values 全部 > 0.1
- 無法拒絕隨機假設
- C 引擎在 2022-2024 樣本中未證實存在穩定 alpha

**Strategic Decision**

- 停止對現有 C 引擎做參數優化
- 封存 RSI+BB 反轉模型
- 未來若重建 C 引擎，需基於新的 Edge 假設

### V7C (Edge Validation Phase)

Status:
- C Engine statistical edge not confirmed.
- Further parameter tuning halted.
