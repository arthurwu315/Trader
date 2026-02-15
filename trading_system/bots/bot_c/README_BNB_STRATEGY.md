# BNB/USDT 策略 (bot_c) — 說明

## 目標

- **標的**: BNB/USDT，1h 時間框架  
- **約一天 5 筆**、扣幣安手續費後仍盈利、**周報酬 ≥ 1%**、固定倉位 2%、**最大回撤 ≤ 10%**  
- 回測區間：**2022/01 – 2023/12**  
- 入場因子：Funding rate（代理）、OI（代理）、波動率、價格突破  
- 出場：止盈、止損、固定 ATR 倍數或固定時間  

## 檔案結構

| 檔案 | 說明 |
|------|------|
| `strategy_bnb.py` | `StrategyBNB` 類：入場門檻 `entry_thresholds`、出場規則 `ExitRules`、`generate_signal()`；因子計算 `add_factor_columns()`（固定，不隨策略變體改） |
| `tests/backtest_engine_bnb.py` | **固定回測引擎**：手續費/滑點、固定倉位 2%、Sharpe/最大回撤/日交易數/周報酬；Walk-forward 與 Monte Carlo（不修改此引擎） |
| `tests/run_bnb_strategy_screen.py` | 批量生成 20–50 組入場門檻 × 3–5 種出場規則 → 回測 → 篩選符合條件策略 → 輸出報表 |

## 使用方式

### 1. 回測區間 2022–2023（需可拉取 K 線）

```bash
cd trading_system/tests
export SKIP_CONFIG_VALIDATION=1
python3 run_bnb_strategy_screen.py --start 2022-01-01 --end 2023-12-31
```

- 會自動向 Binance 拉取 BNB 1h（或使用 `BACKTEST_CACHE_DIR` / `BACKTEST_OFFLINE=1` 快取）。  
- 通過條件：扣費後盈利、日交易 ≤ 5、周報酬 ≥ 1%、最大回撤 ≤ 10%。

### 2. 輸出 CSV、Walk-forward、Monte Carlo

```bash
python3 run_bnb_strategy_screen.py --start 2022-01-01 --end 2023-12-31 --wf --mc 100 --out bnb_qualified.csv
```

- `--wf`: 對通過策略跑 Walk-forward 驗證  
- `--mc 100`: Monte Carlo 100 次  
- `--out`: 寫入符合條件策略清單（含回測結果、Sharpe、最大回撤、日交易數、周報酬）

### 3. 快速 / 放寬

- `--quick`: 減少入場/出場組合，加快單次測試  
- `--relax`: 不強制周報酬 ≥ 1% 與回撤 ≤ 10%，只列出「扣費後盈利」的策略（按周報酬排序），方便除錯或放寬門檻  

## 策略介面（僅改入場/出場門檻）

```python
from bots.bot_c.strategy_bnb import StrategyBNB, ExitRules

entry_thresholds = {
    "funding_rate_proxy": -0.002,  # 做多：小於此值（空頭付費）
    "oi_proxy": 1.2,
    "volatility": 0.01,
    "price_breakout_long": 1.0,
}
exit_rules = ExitRules(tp_r_mult=2.0, sl_atr_mult=1.5)
strategy = StrategyBNB(
    entry_thresholds=entry_thresholds,
    exit_rules=exit_rules,
    position_size=0.02,
    direction="long",
)
# 回測由固定引擎執行，不在此改
result = engine.run(strategy, market_data_1h, fee_bps=9, slippage_bps=5)
```

- **入場**：可只使用部分因子（例如不含 `funding_rate_proxy`）；可用 `min_factors_required` 改為「至少 N 個因子通過」即進場。  
- **出場**：僅改 `ExitRules` 參數（`tp_r_mult`, `sl_atr_mult`, `exit_after_bars`, `tp_fixed_pct`）。

## 因子說明（回測用代理）

- **funding_rate_proxy**: 8 根 K 線收盤報酬（約 8h），做多時取「小於門檻」表示空頭付費環境。  
- **oi_proxy**: 成交量 / 24h 均量。  
- **volatility**: ATR(14) / close。  
- **price_breakout_long / short**: 收盤突破前 N 根高/低。  

實盤若有真實 Funding / OI 歷史，可替換為真實欄位並沿用同一門檻結構。

## 最佳實務

- **固定搜尋空間**：只改 entry/exit 門檻；回測引擎與風控模組不讓 AI 改動。  
- **一次生成多組變體**：用 `run_bnb_strategy_screen.py` 批量產生、快速淘汰。  
- **小資金驗證**：回測通過後先小倉位 live 再放大。
