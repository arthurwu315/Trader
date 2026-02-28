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

---

## Version & Decision Log

### V8 Governance (2025-02)

**1. 封存 C 引擎**
- 原因：Edge Test 在 2022-2024 無統計顯著（p-values 全 > 0.1）
- 動作：`ENABLE_C_ENGINE=false`（預設），`config_c.enable_c_engine` 開關
- 建議在 `.env` 新增 `ENABLE_C_ENGINE=false`（該檔不納版控）
- futures_run 若被呼叫且 ENABLE_C_ENGINE=false 會立即退出

**2. 停用舊 bot 自動重啟**
- 檔案：`trading_system/trading_bot.service`
- 變更：`Restart=always` → `Restart=no`
- 指令：`sudo systemctl disable --now trading_bot.service`（若已安裝）

**3. 新版本 V8**
- 設計：A-only + Volatility Regime Position Sizing
- 服務：`ExecStart` 改為 `bots.bot_a.main_bot_a`
- Vol regime：BTC 1d ATR20/Close*100 → LOW(<2%): mult=1.5, MID(2-4%): 1.0, HIGH(>=4%): 0.6
- 回測：`python3 -m tests.run_v8_backtest`

**V8 回測對照 (Full 2022-2024)**

| Version | Period | CAGR | MDD | Calmar | PF | Win% | Trades | Exposure |
|---------|--------|------|-----|--------|-----|------|--------|----------|
| Baseline | Full | 8.3348 | 34.7510 | 0.2398 | 1.1294 | 47.69 | 65 | 47.93 |
| V8_VolRegime | Full | 4.3086 | 37.8898 | 0.1137 | 1.0716 | 47.69 | 65 | 47.93 |

### V8.1 (Experimental)

**Governance fix:**
- systemd ExecStart normalized to A-only bot (`bots.bot_a.main_bot_a`)
- futures_run.py no longer exits with 0 when C disabled; uses `sys.exit(1)` and prints `C engine disabled (deprecated)` to prevent false "healthy" state

**Strategy change:**
- Vol regime sizing changed to "only de-risk in high vol"
- multipliers: LOW=1.0, MID=1.0, HIGH=0.7 (no low-vol leverage)

**Backtest (Full 2022-2024 + yearly)**

| Period | Version | CAGR | MDD | Calmar | PF | Win% | Trades | Exposure |
|--------|---------|------|-----|--------|-----|------|--------|----------|
| Full | Baseline | 8.3348 | 34.7510 | 0.2398 | 1.1294 | 47.69 | 65 | 47.93 |
| Full | V8.1 | 5.3312 | 37.0857 | 0.1438 | 1.0870 | 47.69 | 65 | 47.93 |
| 2022 | Baseline | -2.4556 | 6.1105 | -0.4019 | 0.6132 | 33.33 | 3 | 6.58 |
| 2022 | V8.1 | -2.8791 | 5.4553 | -0.5278 | 0.4862 | 33.33 | 3 | 6.58 |
| 2023 | Baseline | 21.3615 | 10.2185 | 2.0905 | 1.4689 | 57.69 | 26 | 40.00 |
| 2023 | V8.1 | 18.5763 | 10.4335 | 1.7804 | 1.4201 | 57.69 | 26 | 40.00 |
| 2024 | Baseline | -0.4543 | 38.9282 | -0.0117 | 0.9950 | 41.67 | 36 | 54.64 |
| 2024 | V8.1 | -4.0745 | 39.9283 | -0.1020 | 0.9511 | 41.67 | 36 | 54.64 |

- **HighVolExposure%** (V8.1 Full): 35.41% — proportion of exposure hours in HIGH vol regime (vol≥4%)

### V8.2 (Experimental)

- Reverted position multiplier to 1.0 (no sizing change)
- Added volatility-conditional tighter stop: high vol regime (vol≥4%) uses 0.7× ATR stop distance
- Goal: reduce drawdown in high-vol periods without reducing position size

**Backtest (Full 2022-2024 + yearly)**

| Period | Version | CAGR | MDD | Calmar | PF | Win% | Trades | Exposure |
|--------|---------|------|-----|--------|-----|------|--------|----------|
| Full | Baseline | 8.3348 | 34.7510 | 0.2398 | 1.1294 | 47.69 | 65 | 47.93 |
| Full | V8.2 | -4.1221 | 50.9746 | -0.0809 | 0.9432 | 45.71 | 70 | 44.56 |
| 2022 | Baseline | -2.4556 | 6.1105 | -0.4019 | 0.6132 | 33.33 | 3 | 6.58 |
| 2022 | V8.2 | 1.5165 | 7.4620 | 0.2032 | 1.1853 | 33.33 | 3 | 1.10 |
| 2023 | Baseline | 21.3615 | 10.2185 | 2.0905 | 1.4689 | 57.69 | 26 | 40.00 |
| 2023 | V8.2 | 10.6868 | 11.0947 | 0.9632 | 1.2308 | 51.85 | 27 | 38.63 |
| 2024 | Baseline | -0.4543 | 38.9282 | -0.0117 | 0.9950 | 41.67 | 36 | 54.64 |
| 2024 | V8.2 | -20.6714 | 55.7757 | -0.3706 | 0.7833 | 42.50 | 40 | 54.37 |

- **HighVol MDD** (equity curve from high-vol trades only): Baseline 11.12% | V8.2 23.09%
- **non-HighVol MDD** (equity curve from non-high-vol trades only): Baseline 43.01% | V8.2 42.97%
- **Conclusion**: 0.7× trail in high vol increased HighVol MDD and overall MDD; single-variable experiment did not achieve goal.

### V8.3 (Experimental)

- Suppress low-vol regime exposure: LOW vol (vol<2%) multiplier = 0.6
- MID / HIGH unchanged (1.0)
- Trailing stop reverted to baseline

**Backtest (Full 2022-2024 + yearly)**

| Period | Version | CAGR | MDD | Calmar | PF | Win% | Trades | Exposure |
|--------|---------|------|-----|--------|-----|------|--------|----------|
| Full | Baseline | 8.3348 | 34.7510 | 0.2398 | 1.1294 | 47.69 | 65 | 47.93 |
| Full | V8.3 | 8.3348 | 34.7510 | 0.2398 | 1.1294 | 47.69 | 65 | 47.93 |
| 2022 | Baseline | -2.4556 | 6.1105 | -0.4019 | 0.6132 | 33.33 | 3 | 6.58 |
| 2022 | V8.3 | -2.4556 | 6.1105 | -0.4019 | 0.6132 | 33.33 | 3 | 6.58 |
| 2023 | Baseline | 21.3615 | 10.2185 | 2.0905 | 1.4689 | 57.69 | 26 | 40.00 |
| 2023 | V8.3 | 21.3615 | 10.2185 | 2.0905 | 1.4689 | 57.69 | 26 | 40.00 |
| 2024 | Baseline | -0.4543 | 38.9282 | -0.0117 | 0.9950 | 41.67 | 36 | 54.64 |
| 2024 | V8.3 | -0.4543 | 38.9282 | -0.0117 | 0.9950 | 41.67 | 36 | 54.64 |

- **LowVol Exposure%**: 0.00% (no trades occurred on low-vol days in 2022-2024)
- **LowVol MDD / non-LowVol MDD**: 0.00% / 34.75% (Baseline & V8.3 identical)
- **Note**: 18 low-vol days in sample; 0 trades coincided. Experiment structurally correct; no observable effect in this period.

### V8.4 (Experimental)

- MidVol regime (2%–4%) disabled: no new entries when vol in this range
- Only trade in LowVol (<2%) and HighVol (≥4%)
- Goal: test whether mid-vol is the main MDD source

**Backtest (Full 2022-2024 + yearly)**

| Period | Version | CAGR | MDD | Calmar | PF | Win% | Trades | Exposure |
|--------|---------|------|-----|--------|-----|------|--------|----------|
| Full | Baseline | 8.3348 | 34.7510 | 0.2398 | 1.1294 | 47.69 | 65 | 47.93 |
| Full | V8.4 | 8.5903 | 11.0001 | 0.7809 | 1.5325 | 43.75 | 16 | 19.55 |
| 2022 | Baseline | -2.4556 | 6.1105 | -0.4019 | 0.6132 | 33.33 | 3 | 6.58 |
| 2022 | V8.4 | 1.4116 | 2.3882 | 0.5911 | 1.5689 | 50.00 | 2 | 6.03 |
| 2023 | Baseline | 21.3615 | 10.2185 | 2.0905 | 1.4689 | 57.69 | 26 | 40.00 |
| 2023 | V8.4 | 10.7512 | 2.9731 | 3.6162 | 4.6162 | 66.67 | 3 | 11.51 |
| 2024 | Baseline | -0.4543 | 38.9282 | -0.0117 | 0.9950 | 41.67 | 36 | 54.64 |
| 2024 | V8.4 | 6.5921 | 12.1057 | 0.5445 | 1.2218 | 36.36 | 11 | 23.22 |

- **Baseline Full**: HighVol Exposure 35.41% | MidVol Exposure 64.59% | LowVol Exposure 0.00%
- **Baseline**: MidVol MDD = 43.01% | HighVol MDD = 11.12% → **MidVol is the main MDD source**
- **V8.4 Full**: HighVol 100% | MidVol 0% | LowVol 0% | MidVol MDD = 0% | HighVol MDD = 11.00%
- **Conclusion**: Blocking mid-vol cut MDD from 34.75% to 11.00%; Calmar improved 0.24 → 0.78. Hypothesis confirmed.

**V8.4 Robustness Test (mid-vol threshold)**

Three MID block intervals tested (only block interval changes):

| Config | MID block | Full CAGR | Full MDD | Full Calmar | Full PF | Trades | Exposure |
|--------|-----------|-----------|----------|-------------|---------|--------|----------|
| 1 | 1.8–3.8% | 8.54 | 12.27 | 0.696 | 1.33 | 23 | 23.83 |
| 2 | 2.0–4.0% (baseline) | 8.59 | 11.00 | 0.781 | 1.53 | 16 | 19.55 |
| 3 | 2.2–4.2% | 8.88 | 11.00 | 0.808 | 1.66 | 15 | 16.71 |

**By year**

| Config | 2022 Calmar | 2023 Calmar | 2024 Calmar |
|--------|-------------|-------------|-------------|
| 1.8–3.8% | 0.59 | 0.36 | 1.13 |
| 2.0–4.0% | 0.59 | 3.62 | 0.54 |
| 2.2–4.2% | 0.59 | 3.62 | 0.58 |

**Answers**
1. Calmar > 0.6 for all? **Yes** (0.70, 0.78, 0.81)
2. Plateau exists? **Yes** (Calmar range 0.11 < 0.15)
3. Most stable? **2.2–4.2%** (highest Calmar 0.81)

Run: `python3 -m tests.run_v84_robustness`

### V9 – Production Core

- **Regime-filtered trend model**: Mid-vol (2.2–4.2%) disabled
- **Robustness confirmed**: Calmar plateau 0.70–0.81 across threshold variants
- **MDD reduced**: 34% → 11% (baseline vs V9)
- **Locked params**: LOW &lt;2.2%, MID 2.2–4.2% (blocked), HIGH ≥4.2%; no further threshold optimization

**Config:** `STRATEGY_VERSION = "V9_REGIME_CORE"` (`config_v9.py`, `config_a.strategy_version`)

**Walk-Forward (Rolling)**

| Split | Train | Test | Calmar | MDD | CAGR | Trades |
|-------|-------|------|--------|-----|------|--------|
| 1 | 2022–2023 | 2024 | 0.58 | 12.11% | 7.04% | 10 |
| 2 | 2023–2024 | 2022 | 0.59 | 2.39% | 1.41% | 2 |

Run: `python3 -m tests.run_v9_walkforward`

---

## V9.1 Live Validation Plan

- **Strategy**: V9_REGIME_CORE (Frozen) — 策略邏輯禁止更動，變更一律開新版本號 (V9.2+)
- **Modes**: PAPER / MICRO-LIVE
- **Position**: MICRO-LIVE uses 10% notional (of base 40%)
- **What we validate**:
  1) fees & slippage
  2) latency (signal_time vs order_time)
  3) execution correctness

**Exit Criteria (Two-Stage)**

- **Stage 1 (Fast)**: 14 days OR 10 execution events (entry/exit/rebalance) → verify correctness, fees, slippage, latency only
- **Stage 2**: 8 weeks OR 30 trades → evaluate PF and behavior
- **Kill Switch**: If MICRO-LIVE equity drawdown &gt; 3% or any abnormal behavior → stop bot immediately

**V9.1 Fast-Track Live Validation**

- **Start MICRO-LIVE**:
  1. `cd trading_system && ./deploy_v9.sh` (locks current commit)
  2. `sudo cp trading_bot_v9.service /etc/systemd/system/ && sudo systemctl daemon-reload`
  3. `sudo systemctl start trading_bot_v9`
  4. Check: `journalctl -u trading_bot_v9 -f` for `STRATEGY_VERSION=... MODE=MICRO-LIVE GIT_COMMIT=...`
  5. Health check: `logs/v9_health_check.txt` (written every startup)
- **Commit hash lock**: `logs/deploy_hash.txt` after `./deploy_v9.sh [hash]`
- **Manual run**: `V9_LIVE_MODE=MICRO-LIVE python3 -m v9_live_runner`

**Trade record**: `logs/v9_trade_records.csv` (same format for PAPER & MICRO-LIVE)

| Field | Description |
|-------|-------------|
| timestamp | Fill/exchange time |
| symbol | Trading pair |
| side | BUY / SELL |
| price | Fill price |
| qty | Quantity |
| regime_vol | Vol % at entry |
| reason | entry / exit |
| fees | Transaction fees |
| slippage_est | Slippage (bp, optional) |
| signal_time | Signal generation time |
| order_time | Order send time |
| mode | PAPER / MICRO-LIVE |

**Friction report**: `python3 -m tests.run_v9_friction_report`

**Deploy**: `./deploy_v9.sh [commit_hash]` — Production must lock commit hash (`logs/deploy_hash.txt`)

---

## Alpha2: Funding Carry (Experimental) – FUND_CARRY_V1

- **Purpose**: Low-correlation cash flow, add live friction samples; market neutral
- **Strategy**: Long spot + short perp to collect funding when rate &gt; threshold
- **Universe**: BTC, ETH (MVP)
- **Entry**: funding_rate_8h annualized &gt; 20%
- **Exit**: annualized &lt; 10% or negative, or single-asset loss &gt; 1%
- **Risk**: Alpha2 capital cap 10% of total; single asset 2–3%; MICRO-LIVE 2–5% notional first
- **Deploy**: PAPER 7 days first → MICRO-LIVE 2–5%
- **Start PAPER**: `python3 -m bots.bot_funding_carry.main`
- **Alpha2 and V9 are fully independent**: separate service, separate kill switch, no shared logic

**Independent service** (`trading_bot_alpha2.service`)

- File: `trading_system/trading_bot_alpha2.service`
- `Restart=no` (no auto restart)
- `Environment=ALPHA2_LIVE_MODE=MICRO-LIVE`
- `ExecStart=/usr/bin/python3 -u -m bots.bot_funding_carry.main`
- Install: `sudo cp trading_bot_alpha2.service /etc/systemd/system/ && sudo systemctl daemon-reload`
- Start: `sudo systemctl start trading_bot_alpha2`
- Rollback: `sudo systemctl stop trading_bot_alpha2`; deploy previous commit via `./deploy_alpha2.sh <hash>`

**Alpha2 kill switch (independent of V9)**

- **ALPHA2_MAX_DD_PCT = 1%** (equity drawdown from alpha2_equity_peak)
- Tracks `alpha2_equity_peak` in `logs/alpha2_equity_peak.json`
- If DD &gt; 1% → stop bot (SystemExit), write `event_type=KILL`, `reason=ALPHA2_DD_LIMIT`
- Manual DD simulation: set `logs/alpha2_equity_peak.json` to high value, then `ALPHA2_SIMULATE_DD=1 ALPHA2_LIVE_MODE=MICRO-LIVE python3 -m bots.bot_funding_carry.main`

**Deploy** (`deploy_alpha2.sh`)

- Usage: `./deploy_alpha2.sh <commit_hash>`
- Locks commit, records to `logs/deploy_alpha2_hash.txt`
- No auto restart

**Observability (PAPER)**

- Every cycle writes **CYCLE** records (one per symbol) to `logs/v9_trade_records.csv` (strategy_id=FUND_CARRY_V1)
- CYCLE fields: timestamp_utc, symbol, mode(PAPER), funding_rate_8h, funding_annualized_pct, signal(boolean), reason(NO_SIGNAL / ENTRY_SIGNAL / EXIT_SIGNAL)
- When entry/exit signal: additional **TRADE** record (event_type=TRADE)

**Trade record format (extended for Alpha2)**

| Field | Description |
|-------|-------------|
| event_type | CYCLE \| TRADE |
| funding_rate_8h | 8h funding rate (CYCLE only) |
| funding_annualized_pct | Annualized % (CYCLE only) |
| signal | true/false (CYCLE only) |
| reason | NO_SIGNAL \| ENTRY_SIGNAL \| EXIT_SIGNAL \| COOLDOWN \| REBALANCE_FAIL (CYCLE); entry/exit (TRADE) |
| spot_notional, perp_notional, net_notional | Position notionals (CYCLE) |
| net_notional_pct | Net vs equity % (CYCLE) |
| rebalance_action | NONE \| REBALANCE \| EXIT \| KILL |
| rebalance_attempt, rebalance_success | Boolean (CYCLE) |

**Hedge deviation hard limits**

- `abs(net_notional_pct) > 0.20%` ⇒ must trigger REBALANCE
- Consecutive 3 cycles out of threshold or 3 rebalance fails ⇒ **hard stop** (bot exits)
- Order/API exceptions: write `reason=REBALANCE_FAIL` to trade_records and count toward hard stop

**Cooldown (anti-churn)**

- Exit due to annualized &lt; 10% or ≤ 0 ⇒ 24h cooldown for that symbol
- During cooldown: no entry, only CYCLE record with `reason=COOLDOWN`

**Notional / NA rules**

- CYCLE spot_notional, perp_notional, net_notional, net_notional_pct: computed from price × simulated position when available
- When price unavailable: write `NA` (empty/NA) and append `NOTIONAL_UNAVAILABLE` to reason — do not write fixed 0

**Self-test (PAPER only)**

- `ALPHA2_SELF_TEST=1 python3 -m bots.bot_funding_carry.main`
- Simulates net_notional_pct &gt; threshold → rebalance_action=REBALANCE → rebalance fail count → hard stop after 3 fails
- No real orders; validates hedge-deviation → rebalance → hard-stop chain

**Report**: `python3 -m tests.run_funding_carry_report`

- KPIs (last 7 days): max_abs_net_notional_pct (excludes NA; "notional unavailable" if all NA), rebalance_count, rebalance_fail_count, cooldown_count, hard_stop_triggered
- Does not treat fixed 0 as hedge-stable evidence
- With 0 TRADE events: outputs KPIs + latest funding snapshot + signal count
- With trades: funding collected proxy, fees, slippage
- Diagnostic: when no FUND_CARRY_V1 records, prints file path, row count, strategy_id distribution

---

## Engineering Policy (Hard Rule)

1. **任何策略/參數/部署變更都必須更新 README.md**
2. **必須 commit 並 git push 才算完成**
3. **README 需包含**：變更目的、影響範圍、回測對照表、以及是否影響風控
4. **任何 service/部署檔變更** 必須在 README 記錄「檔名 + 變更點 + 回滾方法」
5. **任何部署模式/服務檔變更** 也必須記錄並 push
6. **Production 必須鎖定 commit hash**（在 service 或 config 明記；可用 deploy_v9.sh）
7. **多策略** 必須「獨立版本號、獨立 strategy_id、獨立風控上限」，禁止互相調用核心邏輯造成污染
