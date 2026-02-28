# Strategy State – Alpha Burst B1

Alpha Burst 策略狀態文件。獨立於 V9 核心。

---

## 1. Strategy Identity

| 參數 | 值 |
|------|-----|
| strategy_id | ALPHA_BURST_B1 |
| Universe | BTCUSDT, ETHUSDT |
| Timeframe Trend | 4H |
| Timeframe Entry/Exit | 1H |
| Type | Vol-expansion Donchian breakout |

---

## 2. Rules

### Trend Filter (4H)

- close > EMA200 ⇒ long only
- close < EMA200 ⇒ short only

### Vol Expansion

- ATR14 > ATR14_ma20 × 1.2 ⇒ allow trades
- Otherwise: no entry

### Entry (1H)

- Long: close > roll_high_20 (Donchian breakout)
- Short: close < roll_low_20

### Exit

- Initial stop: ATR × k (k=2.0)
- ATR trailing stop (k=2.0)

### Position Sizing

- 1% burst_equity risk per trade
- qty = risk_usdt / (entry - stop) per unit

---

## 3. Trade Record

| Field | Description |
|-------|-------------|
| timestamp | Exit time |
| symbol | BTCUSDT / ETHUSDT |
| side | BUY / SELL |
| entry_price | Entry price |
| stop_price | Initial stop |
| exit_price | Exit price |
| qty | Quantity |
| initial_risk_usdt | 1% of burst equity |
| pnl_usdt | Realized PnL |
| R_multiple | pnl_usdt / initial_risk_usdt |
| holding_bars | Bars in trade |
| strategy_id | ALPHA_BURST_B1 |

File: `logs/alpha_burst_b1_trades.csv`

---

## 4. Ops

- **Burst DD > 25%** ⇒ KILL (stop bot)
- Reconcile: dry-run (log only)
- State: `logs/alpha_burst_state.json` (equity_peak)

---

## 5. Report

- Output: `tests/reports/alpha_burst_b1_report.md`
- Artifacts: `tests/reports/alpha_burst_b1_artifacts/`
- B1: E[R], WinRate, AvgWin_R, AvgLoss_R, p10/p50/p90/p95 by year/symbol
- B2: Grid (vol_expansion, stop_ATR_k, breakout_lookback); plateau
- B3: Permutation 1000, block bootstrap 1000 (SEED=42)
- B4: Cost stress (slippage 5/10/20 bps, high-vol 2x, fee)

---

## 6. Statistical Validation (SEED=42)

- **Permutation test**: 1000 shuffles; sample entry bars within allowed set; preserve holding_bars distribution
- **Block bootstrap**: 1000 paths; block_size=7 days; bootstrap trade-level R
- **Cost assumptions**: fee taker 0.04%; slippage 5/10/20 bps; high-vol 2x

## 7. Commands

```bash
# Backtest (produces trades)
python3 -m tests.run_alpha_burst_backtest

# Report
python3 -m tests.run_alpha_burst_report

# Grid (B2)
python3 -m tests.run_alpha_burst_grid

# PAPER / MICRO-LIVE (direct run)
ALPHA_BURST_MODE=PAPER python3 -m bots.bot_alpha_burst.main
ALPHA_BURST_MODE=MICRO-LIVE python3 -m bots.bot_alpha_burst.main
```

## 8. Acceptance

```bash
python3 -m tests.run_alpha_burst_report
grep ALPHA_BURST_B1 logs/v9_trade_records.csv | tail -n 20
cat STRATEGY_STATE_ALPHA_BURST.md
```

## 9. Independence

- **Do NOT modify**: config_v9.py, config_a.py, v9_live_runner.py, deploy_v9.sh, bot_a
- Alpha Burst uses `append_burst_trade_record()` in core/v9_trade_record.py (write_to_v9=True → v9_trade_records.csv)
- V9 logic unchanged

---

END OF STRATEGY STATE – ALPHA BURST
