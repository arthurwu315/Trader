# V9 Trailing Stop Behavior

本文檔描述 V9_REGIME_CORE 策略中 trailing stop 的**實際執行行為**（非推測），含回測與實盤兩層。

---

## 1. 回測層（Backtest）

### 1.1 程式定位

| 項目 | 位置 |
|------|------|
| 檔案 | `trading_system/tests/run_v8_backtest.py` |
| 函式 | `_simulate_position_exit()`（約第 165–189 行） |
| 呼叫點 | `run_strat_a_with_sizing()` 內（約第 251 行） |

### 1.2 行為描述

- **K 線頻率**：1D（日線）
- **Trailing 更新頻率**：每一根日線（bar）執行一次。迴圈 `for i in range(entry_idx + 1, end_idx + 1)` 以 bar 為單位。
- **Trailing 更新規則**：
  - BUY：`current_sl = max(current_sl, high - trail_mult * atr)`
  - SELL：`current_sl = min(current_sl, low + trail_mult * atr)`
- **觸發判斷價格**：當根 bar 的 **high / low**
  - BUY：`low <= current_sl` → 出場於 `current_sl`
  - SELL：`high >= current_sl` → 出場於 `current_sl`
- **退出價格**：觸發時用 `current_sl`（止損價）或 `tp_price`（止盈價）

因此：回測層的 trailing 是「每根日線更新一次，用 bar 的 high/low 判斷觸發」，並非收盤價。

### 1.3 情境範例（回測）

| 情境 | 行為 |
|------|------|
| **a) 日內急跌** | 若某日 bar 的 `low` 觸及 `current_sl`，當日即出場；模擬為當根 bar 內觸及止損。 |
| **b) 日內觸及但收盤回來** | 若當日 `low <= current_sl`，觸發即出場；收盤拉回不影響，已假設在 bar 內先觸及。 |

---

## 2. 實盤層（Live Execution）

### 2.1 程式定位

| 項目 | 位置 |
|------|------|
| 下單邏輯 | `trading_system/core/execution_safety.py`（OrderStateMachine） |
| 保護單 | `trading_system/core/protection_guard.py` |
| 止損下單 | `ProtectionGuard.place_stop_loss()` → `_place_reduce_only_order()` |
| Trailing 更新 | `trading_system/ops/v9_trailing_updater.py` |

### 2.2 行為描述

- **止損型態**：STOP_MARKET（reduce-only，closePosition）
- **觸發價格來源**：`workingType = "MARK_PRICE"`（標記價格）
- **Trailing 更新**：每小時 runner 執行時，依 backtest 公式計算 candidate_stop，若可**單調遞緊**則取消舊單並下新 STOP_MARKET。環境變數 `V9_TRAILING_UPDATE_ENABLED=1` 才實際更新；`V9_TRAILING_DRY_RUN=1` 僅計算與記錄。
- **觸發判斷**：交易所依 **Mark Price** 是否觸及 `stopPrice` 來觸發 STOP_MARKET。

### 2.3 Behavioral Equivalence 與差異

- **公式一致**：Live 使用與 backtest 相同之 `high - trail_mult*atr`（BUY）、`low + trail_mult*atr`（SELL），以及 HIGH_VOL_STOP_FACTOR。
- **差異**：Live 無 bar high/low 觸發；僅依 Mark Price 觸及 stopPrice。日內觸及但未成交之情況與 backtest 模擬可能不同。

### 2.3 情境範例（實盤）

| 情境 | 行為 |
|------|------|
| **a) 日內急跌** | 若 Mark Price 觸及止損價，即觸發 STOP_MARKET，日內可出場。 |
| **b) 日內觸及但收盤回來** | 若 Mark Price 曾觸及止損，即已觸發；收盤拉回不影響，訂單已執行。 |

---

## 3. 回測 vs 實盤對照

| 項目 | 回測 | 實盤 |
|------|------|------|
| 更新頻率 | 每根日線 bar | 每小時 runner（V9_TRAILING_UPDATE_ENABLED=1） |
| 觸發價格 | Bar 的 high/low | Mark Price |
| Trailing | 有（ATR-based） | 有（同公式，單調遞緊） |
| 日內出場 | 可（模擬 intrabar 觸及） | 可（Mark Price 觸及即執行） |

---

## 4. 備註

- `config_a.py` 中有 `enable_trailing_stop: bool = False`，與上述實盤行為一致。
- 本文件僅描述現有程式邏輯，未改變任何策略參數或邏輯。
