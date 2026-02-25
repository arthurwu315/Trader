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
