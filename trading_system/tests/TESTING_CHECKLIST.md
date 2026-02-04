# Trading System Checklist

## Bot B (Testnet)

- 確認 `.env.b_testnet` 有正確 key/secret
- 先跑 dry-run：
  - `python3 /home/trader/trading_system/tests/bot_b_full_test.py --dry-run`
- 再跑完整測試：
  - `python3 /home/trader/trading_system/tests/bot_b_full_test.py`
- 檢查 `/tmp/bot_b_full_test.log` 中是否出現：
  - `Regime阻擋` / `L1/L2通過`
  - `止損單送出成功` / `止盈單送出成功`
  - `保護單確認成功`

## Bot A (Mainnet Preflight)

- 確認 `.env.a_mainnet` 有正確 key/secret
- 跑 preflight（不下單）：
  - `python3 /home/trader/trading_system/tests/bot_a_preflight.py`
- 檢查 `/tmp/bot_a_preflight.log`：
  - `RiskCheck: allowed=True`
  - `RegimeDecision: allow=True`（或合理阻擋原因）

## 上線前總檢查

- ProtectionGuard 定期掃描已啟用
- RiskManager 初始權益設置成功
- RegimeDetector log 清楚可追溯
- Algo Order 回傳 algoId 正常
