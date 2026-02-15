# Bot B 完整測試流程

此流程會在 **TESTNET** 上實際下單，用於驗證風控、下單與保護單守護。

## 前置條件

- `.env.b_testnet` 已填入 testnet key/secret
- `config_b.py` 仍為 `mode=TESTNET` 且 `binance_base_url` 指向 testnet

## 快速測試

Dry-run（不下單，只做連線/風控/價格檢查）：

```
python3 /home/trader/trading_system/tests/bot_b_full_test.py --dry-run
```

完整測試（會下單、掛 SL/TP、再清理）：

```
python3 /home/trader/trading_system/tests/bot_b_full_test.py
```

## 進階參數

- 指定交易對：
```
python3 /home/trader/trading_system/tests/bot_b_full_test.py --symbol BNBUSDT
```

- 指定數量（最小化成本）：
```
python3 /home/trader/trading_system/tests/bot_b_full_test.py --qty 0.01
```

- 指定方向：
```
python3 /home/trader/trading_system/tests/bot_b_full_test.py --side SELL
```

- 不做清理（保留倉位/掛單）：
```
python3 /home/trader/trading_system/tests/bot_b_full_test.py --skip-cleanup
```

## 日誌

- `/tmp/bot_b_full_test.log`

## 注意事項

- 測試僅限 TESTNET
- 若下單失敗，請先檢查：
  - API key 權限
  - Testnet 餘額
  - 最小名義價值（min notional）
