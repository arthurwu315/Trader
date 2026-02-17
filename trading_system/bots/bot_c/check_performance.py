"""
輕量級戰報腳本：讀取 logs/paper_signals.json，統計進場、浮動盈虧、心跳。
執行：cd /home/trader/trading_system && python3 -m bots.bot_c.check_performance
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# 載入 Strategy C 的 .env（與 paper_run / main_bot_c 一致）
try:
    from dotenv import load_dotenv
    _bot_dir = Path(__file__).resolve().parent
    _env_live = _bot_dir / ".env.c_live"
    _env_testnet = _bot_dir / ".env.c_testnet"
    if os.getenv("MODE") == "LIVE" and _env_live.exists():
        load_dotenv(dotenv_path=_env_live)
    elif _env_testnet.exists():
        load_dotenv(dotenv_path=_env_testnet)
except Exception:
    pass

LOG_DIR = ROOT / "logs"
PAPER_SIGNALS_FILE = LOG_DIR / "paper_signals.json"
HEARTBEAT_FILE = LOG_DIR / "paper_last_heartbeat.txt"
SYMBOL = "BNBUSDT"
# 合約成本：Net_Profit = (Price_Change * Leverage) - (Entry_Fee + Exit_Fee)
LEVERAGE = 3
FEE_MAKER_PCT = 0.02   # 0.02%
FEE_TAKER_PCT = 0.04   # 0.04%


def load_signals():
    if not PAPER_SIGNALS_FILE.exists():
        return []
    try:
        with open(PAPER_SIGNALS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def get_current_price():
    try:
        from bots.bot_c.config_c import get_strategy_c_config
        from core.binance_client import BinanceFuturesClient
        cfg = get_strategy_c_config()
        client = BinanceFuturesClient(
            base_url=os.getenv("BINANCE_DATA_URL", "https://fapi.binance.com"),
            api_key=cfg.binance_api_key or "dummy",
            api_secret=cfg.binance_api_secret or "dummy",
        )
        out = client.get_ticker_price(SYMBOL)
        return float(out.get("price", 0)) if out else None
    except Exception:
        return None


def main():
    signals = load_signals()

    # 成交統計
    total = len(signals)
    longs = sum(1 for s in signals if (s.get("side") or "").upper() == "BUY")
    shorts = sum(1 for s in signals if (s.get("side") or "").upper() == "SELL")

    print("========== 戰報 (Paper Signals) ==========")
    print(f"進場總筆數: {total}")
    print(f"  多單: {longs}  空單: {shorts}")

    # 浮動盈虧：最後一筆視為當前持倉
    current_price = get_current_price()
    if signals and current_price and current_price > 0:
        last = signals[-1]
        entry = float(last.get("entry_price", 0))
        side = (last.get("side") or "").upper()
        if entry and entry > 0:
            if side == "BUY":
                price_change_pct = (current_price - entry) / entry * 100
            else:
                price_change_pct = (entry - current_price) / entry * 100
            # Net = (Price_Change * Leverage) - (Entry_Fee + Exit_Fee); 進出場皆以 Taker 估
            gross_pct = price_change_pct * LEVERAGE
            round_trip_fee_pct = 2 * FEE_TAKER_PCT
            net_pct = gross_pct - round_trip_fee_pct
            bar_time = last.get("bar_time", "?")
            print(f"浮動盈虧 (最後持倉 {bar_time} {side} @ {entry}): 價格 {price_change_pct:+.2f}% | 淨(含槓桿{LEVERAGE}x與手續費): {net_pct:+.2f}%")
        else:
            print("浮動盈虧: 無法計算 (entry_price 無效)")
    else:
        if not signals:
            print("浮動盈虧: 無持倉")
        else:
            print("浮動盈虧: 無法取得市價，略過")

    # 已實現盈虧：paper 模式未記錄出場，僅提示
    print("已實現盈虧: N/A (paper 模式未記錄 TP/SL 出場)")

    # 心跳檢查
    if HEARTBEAT_FILE.exists():
        try:
            with open(HEARTBEAT_FILE, "r", encoding="utf-8") as f:
                line = (f.read() or "").strip()
            print(f"最後 Heartbeat: {line or '無時間戳'}")
        except Exception:
            print("最後 Heartbeat: 讀取失敗")
    else:
        print("最後 Heartbeat: 無記錄 (paper_run 尚未寫入)")

    print("==========================================")


if __name__ == "__main__":
    main()
