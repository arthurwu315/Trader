"""
模擬盤監控 - 實時訊號記錄器
- 每小時拉取最新 K 線，套用 deploy_ready 邏輯
- 若有進場訊號則寫入 logs/paper_signals.json（不實際下單）
- 斷線保護：連續 3 小時拉不到資料則寫入告警並可觸發系統通知
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# 載入 Strategy C 的 .env（與 main_bot_c 一致：TESTNET 用 .env.c_testnet，LIVE 用 .env.c_live）
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

SYMBOL = "BNBUSDT"
INTERVAL = "1h"
KLINES_LIMIT = 250
LOG_DIR = ROOT / "logs"
PAPER_SIGNALS_FILE = LOG_DIR / "paper_signals.json"
HEARTBEAT_FILE = LOG_DIR / "paper_last_heartbeat.txt"
DISCONNECT_ALERT_FILE = LOG_DIR / "paper_disconnect_alert.log"
CONSECUTIVE_FAIL_THRESHOLD = 3


def get_client():
    from bots.bot_c.config_c import get_strategy_c_config
    from core.binance_client import BinanceFuturesClient
    config = get_strategy_c_config()
    return BinanceFuturesClient(
        base_url=os.getenv("BINANCE_DATA_URL", "https://fapi.binance.com"),
        api_key=config.binance_api_key or "dummy",
        api_secret=config.binance_api_secret or "dummy",
    )


def fetch_latest_klines(client, symbol: str = SYMBOL, interval: str = INTERVAL, limit: int = KLINES_LIMIT):
    """拉取最新 K 線，返回 DataFrame (timestamp, open, high, low, close, volume)。"""
    import pandas as pd
    rows = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    if not rows:
        return None
    df = pd.DataFrame(
        rows,
        columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades",
            "taker_buy_base", "taker_buy_quote", "ignore",
        ],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


def add_factors(df):
    from bots.bot_c.strategy_bnb import add_factor_columns
    return add_factor_columns(df)


def ensure_log_dir():
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def append_paper_signal(record: dict):
    ensure_log_dir()
    records = []
    if PAPER_SIGNALS_FILE.exists():
        try:
            with open(PAPER_SIGNALS_FILE, "r", encoding="utf-8") as f:
                records = json.load(f)
        except Exception:
            records = []
    if not isinstance(records, list):
        records = []
    records.append(record)
    with open(PAPER_SIGNALS_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


def send_disconnect_alert():
    ensure_log_dir()
    msg = f"[{datetime.now(timezone.utc).isoformat()}] 斷線保護：連續 {CONSECUTIVE_FAIL_THRESHOLD} 小時無法取得 K 線資料，請檢查網路或 Binance API。\n"
    with open(DISCONNECT_ALERT_FILE, "a", encoding="utf-8") as f:
        f.write(msg)
    sys.stderr.write(msg)
    try:
        os.system("wall --version >/dev/null 2>&1 && echo '" + msg.strip().replace("'", "'\\''") + "' | wall 2>/dev/null || true")
    except Exception:
        pass


def _get_telegram_notifier():
    try:
        from bots.bot_c.config_c import get_strategy_c_config
        from core.telegram_notifier import TelegramNotifier
        cfg = get_strategy_c_config()
        return TelegramNotifier(
            cfg.telegram_bot_token, cfg.telegram_chat_id, cfg.enable_telegram
        )
    except Exception:
        return None


def _write_heartbeat_file(now_iso: str):
    ensure_log_dir()
    try:
        with open(HEARTBEAT_FILE, "w", encoding="utf-8") as f:
            f.write(now_iso)
    except Exception:
        pass


def _send_daily_summary(notifier, last_sent_date: str) -> str:
    from datetime import date, timedelta
    today = date.today()
    if last_sent_date and last_sent_date == today.isoformat():
        return last_sent_date
    now_local = datetime.now()
    if now_local.hour != 8:
        return last_sent_date
    yesterday = (today - timedelta(days=1)).isoformat()
    records = []
    if PAPER_SIGNALS_FILE.exists():
        try:
            with open(PAPER_SIGNALS_FILE, "r", encoding="utf-8") as f:
                records = json.load(f)
        except Exception:
            records = []
    if not isinstance(records, list):
        records = []
    yesterday_signals = [r for r in records if (r.get("time_utc") or "").startswith(yesterday)]
    total = len(records)
    longs = sum(1 for s in records if (s.get("side") or "").upper() == "BUY")
    shorts = sum(1 for s in records if (s.get("side") or "").upper() == "SELL")
    msg = (
        f"📊 昨日戰報摘要 ({yesterday})\n"
        f"昨日訊號筆數: {len(yesterday_signals)}\n"
        f"累計進場總筆數: {total} (多: {longs} / 空: {shorts})\n"
        f"⏰ {now_local.strftime('%Y-%m-%d %H:%M')}"
    )
    if notifier and getattr(notifier, "send_message", None):
        notifier.send_message(msg)
    return today.isoformat()


def run_once(client, consecutive_fail: int, telegram_notifier=None, last_summary_date: str = ""):
    df = fetch_latest_klines(client)
    if df is None or len(df) < 200:
        return consecutive_fail + 1, last_summary_date
    df = add_factors(df)
    row = df.iloc[-1].to_dict()
    from bots.bot_c.deploy_ready import get_signal_from_row, get_deploy_params
    signal = get_signal_from_row(row, get_deploy_params())
    if signal and signal.should_enter:
        ts = df.iloc[-1]["timestamp"]
        bar_time = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
        record = {
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "bar_time": bar_time,
            "side": signal.side,
            "entry_price": round(signal.entry_price, 4),
            "sl_price": round(signal.sl_price, 4),
            "tp_price": round(signal.tp_price, 4),
            "hard_stop_price": round(signal.hard_stop_price, 4),
            "regime": signal.regime,
        }
        append_paper_signal(record)
        print(f"  [訊號] {record['side']} @ {record['entry_price']}  sl={record['sl_price']}  tp={record['tp_price']}")
        if telegram_notifier and getattr(telegram_notifier, "send_message", None):
            tg_msg = (
                f"📊 Signal: {record['side']}\n"
                f"Entry: {record['entry_price']} | SL: {record['sl_price']} | TP: {record['tp_price']}\n"
                f"Bar: {bar_time}"
            )
            telegram_notifier.send_message(tg_msg)

    last_summary_date = _send_daily_summary(telegram_notifier, last_summary_date)

    # 每小時掃描後無論有無訊號都輸出一行 Heartbeat，並寫入檔案供 check_performance 讀取
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    _write_heartbeat_file(datetime.now(timezone.utc).isoformat())
    price = round(float(row.get("close", 0)), 2)
    ema200_raw = row.get("ema_200")
    ema200 = round(float(ema200_raw), 2) if ema200_raw is not None and str(ema200_raw) != "nan" else None
    regime = "Bull" if (ema200 is not None and price > ema200) else ("Bear" if ema200 is not None else "N/A")
    sig_str = signal.side if (signal and signal.should_enter) else None
    ema_str = ema200 if ema200 is not None else "N/A"
    print(f"[Heartbeat] {now} - Price: {price}, EMA200: {ema_str}, Regime: {regime}, Signal: {sig_str}")
    return 0, last_summary_date


def main():
    print("模擬盤監控啟動：每小時拉取 BNBUSDT 1h K 線，套用 deploy_ready 邏輯，訊號寫入 logs/paper_signals.json")
    client = get_client()
    telegram_notifier = _get_telegram_notifier()
    consecutive_fail = 0
    last_summary_date = ""
    while True:
        try:
            consecutive_fail, last_summary_date = run_once(client, consecutive_fail, telegram_notifier, last_summary_date)
            if consecutive_fail >= CONSECUTIVE_FAIL_THRESHOLD:
                send_disconnect_alert()
                consecutive_fail = 0
        except Exception as e:
            consecutive_fail += 1
            sys.stderr.write(f"[paper_run] 本小時拉取失敗: {e}\n")
            if consecutive_fail >= CONSECUTIVE_FAIL_THRESHOLD:
                send_disconnect_alert()
                consecutive_fail = 0
        time.sleep(3600)


if __name__ == "__main__":
    main()
