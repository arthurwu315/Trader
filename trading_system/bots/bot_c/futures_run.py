"""
Binance Futures Testnet 實戰腳本
- 每小時掃描 1h K 線，套用 deploy_ready 邏輯
- 有訊號時於 Testnet 下 MARKET 單並掛 STOP_MARKET（2% 硬止損）
- 啟動時自動設定 3x 槓桿、逐倉 (ISOLATED)
- 使用 Testnet: https://testnet.binancefuture.com
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
TESTNET_URL = "https://testnet.binancefuture.com"
LOG_DIR = ROOT / "logs"
SIGNALS_FILE = LOG_DIR / "paper_signals.json"
HEARTBEAT_FILE = LOG_DIR / "paper_last_heartbeat.txt"
DISCONNECT_ALERT_FILE = LOG_DIR / "paper_disconnect_alert.log"
CONSECUTIVE_FAIL_THRESHOLD = 3
HARD_STOP_PCT = 2.0  # 2% 硬止損
LEVERAGE = 3
RISK_PCT_OF_EQUITY = 0.0025  # 0.25% 風險


def get_client():
    from bots.bot_c.config_c import get_strategy_c_config
    from core.binance_client import BinanceFuturesClient
    cfg = get_strategy_c_config()
    base = os.getenv("BINANCE_BASE_URL", TESTNET_URL)
    return BinanceFuturesClient(
        base_url=base,
        api_key=cfg.binance_api_key or "dummy",
        api_secret=cfg.binance_api_secret or "dummy",
    )


def fetch_latest_klines(client, symbol: str = SYMBOL, interval: str = INTERVAL, limit: int = KLINES_LIMIT):
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


def init_futures_settings(client, symbol: str, leverage: int = LEVERAGE, margin_type: str = "ISOLATED"):
    """啟動時設定槓桿與逐倉。"""
    try:
        try:
            client.set_margin_type(symbol=symbol, margin_type=margin_type)
        except Exception as e:
            if "No need to change margin type" not in str(e):
                print(f"  [WARN] marginType: {e}")
        time.sleep(0.3)
        client.set_leverage(symbol=symbol, leverage=leverage)
        print(f"  [OK] {symbol} 槓桿={leverage}x, 保證金={margin_type}")
    except Exception as e:
        print(f"  [ERR] init_futures_settings: {e}")
        raise


def get_available_balance(client) -> float:
    try:
        balances = client.get_balance()
        for b in balances or []:
            if b.get("asset") == "USDT":
                return float(b.get("availableBalance", 0) or 0)
    except Exception:
        pass
    return 0.0


def compute_qty(available_usdt: float, entry_price: float, risk_pct: float = RISK_PCT_OF_EQUITY, sl_pct: float = HARD_STOP_PCT) -> float:
    """依 0.25% 風險與 2% 止損反推名義價值與數量。"""
    if available_usdt <= 0 or entry_price <= 0:
        return 0.0
    risk_usdt = available_usdt * risk_pct
    # 2% 價格變動 → 名義價值 = risk_usdt / (sl_pct/100)
    notional = risk_usdt / (sl_pct / 100.0)
    qty = notional / entry_price
    return round(qty, 3)


def has_open_position(client, symbol: str) -> bool:
    try:
        positions = client.get_position_risk(symbol=symbol)
        for p in positions or []:
            amt = float(p.get("positionAmt", 0) or 0)
            if amt != 0:
                return True
    except Exception:
        pass
    return False


def place_market_order(client, symbol: str, side: str, quantity: float) -> dict | None:
    """下市價單開倉。"""
    if quantity <= 0:
        return None
    try:
        params = {
            "symbol": symbol,
            "side": side.upper(),
            "type": "MARKET",
            "quantity": quantity,
        }
        out = client.place_order(params)
        return out
    except Exception as e:
        print(f"  [ERR] place_market_order: {e}")
        return None


def place_stop_market_close(client, symbol: str, side: str, stop_price: float) -> dict | None:
    """掛 STOP_MARKET 平倉（2% 硬止損）。Long 用 SELL 觸發，Short 用 BUY 觸發。"""
    try:
        close_side = "SELL" if side.upper() == "BUY" else "BUY"
        params = {
            "symbol": symbol,
            "side": close_side,
            "type": "STOP_MARKET",
            "stopPrice": round(stop_price, 2),
            "closePosition": "true",
        }
        out = client.place_order(params)
        return out
    except Exception as e:
        print(f"  [ERR] place_stop_market_close: {e}")
        return None


def append_signal_record(record: dict):
    ensure_log_dir()
    records = []
    if SIGNALS_FILE.exists():
        try:
            with open(SIGNALS_FILE, "r", encoding="utf-8") as f:
                records = json.load(f)
        except Exception:
            records = []
    if not isinstance(records, list):
        records = []
    records.append(record)
    with open(SIGNALS_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


def _write_heartbeat(now_iso: str):
    ensure_log_dir()
    try:
        with open(HEARTBEAT_FILE, "w", encoding="utf-8") as f:
            f.write(now_iso)
    except Exception:
        pass


def _get_telegram_notifier():
    try:
        from bots.bot_c.config_c import get_strategy_c_config
        from core.telegram_notifier import TelegramNotifier
        cfg = get_strategy_c_config()
        return TelegramNotifier(cfg.telegram_bot_token, cfg.telegram_chat_id, cfg.enable_telegram)
    except Exception:
        return None


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
    if SIGNALS_FILE.exists():
        try:
            with open(SIGNALS_FILE, "r", encoding="utf-8") as f:
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
        f"📊 <b>昨日戰報 (Testnet)</b> ({yesterday})\n"
        f"昨日訊號: {len(yesterday_signals)} | 累計: {total} (多: {longs} / 空: {shorts})\n"
        f"⏰ {now_local.strftime('%Y-%m-%d %H:%M')}"
    )
    if notifier and getattr(notifier, "send_message", None):
        notifier.send_message(msg)
    return today.isoformat()


def send_disconnect_alert():
    ensure_log_dir()
    msg = f"[{datetime.now(timezone.utc).isoformat()}] 斷線：連續 {CONSECUTIVE_FAIL_THRESHOLD} 小時無法取得 K 線\n"
    with open(DISCONNECT_ALERT_FILE, "a", encoding="utf-8") as f:
        f.write(msg)
    sys.stderr.write(msg)


def run_once(client, telegram_notifier=None, last_summary_date: str = ""):
    df = fetch_latest_klines(client)
    if df is None or len(df) < 200:
        return 1, last_summary_date
    df = add_factors(df)
    row = df.iloc[-1].to_dict()
    from bots.bot_c.deploy_ready import get_signal_from_row, get_deploy_params, HARD_STOP_POSITION_PCT
    signal = get_signal_from_row(row, get_deploy_params())
    hard_stop_pct = HARD_STOP_POSITION_PCT

    if signal and signal.should_enter:
        ts = df.iloc[-1]["timestamp"]
        bar_time = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
        if has_open_position(client, SYMBOL):
            print(f"  [SKIP] 已有持倉，本根不重複下單")
        else:
            balance = get_available_balance(client)
            qty = compute_qty(balance, signal.entry_price, RISK_PCT_OF_EQUITY, hard_stop_pct)
            if qty <= 0:
                print(f"  [SKIP] 餘額不足或 qty=0 (balance={balance:.2f})")
            else:
                order = place_market_order(client, SYMBOL, signal.side, qty)
                if order:
                    sl_price = signal.hard_stop_price
                    stop_ok = place_stop_market_close(client, SYMBOL, signal.side, sl_price)
                    record = {
                        "time_utc": datetime.now(timezone.utc).isoformat(),
                        "bar_time": bar_time,
                        "side": signal.side,
                        "entry_price": round(signal.entry_price, 4),
                        "sl_price": round(sl_price, 4),
                        "tp_price": round(signal.tp_price, 4),
                        "hard_stop_price": round(sl_price, 4),
                        "regime": signal.regime,
                        "qty": qty,
                        "order_id": order.get("orderId"),
                    }
                    append_signal_record(record)
                    print(f"  [FILL] {signal.side} qty={qty} @ {signal.entry_price}  SL={sl_price}  orderId={order.get('orderId')}")
                    if telegram_notifier and getattr(telegram_notifier, "send_message", None):
                        telegram_notifier.send_message(
                            f"📊 <b>Testnet: {signal.side}</b>\n"
                            f"Entry: {signal.entry_price} | SL: {sl_price} | qty: {qty}\nBar: {bar_time}"
                        )
                else:
                    print(f"  [ERR] 市價單未成交")

    last_summary_date = _send_daily_summary(telegram_notifier, last_summary_date)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    _write_heartbeat(datetime.now(timezone.utc).isoformat())
    price = round(float(row.get("close", 0)), 2)
    ema200_raw = row.get("ema_200")
    ema200 = round(float(ema200_raw), 2) if ema200_raw is not None and str(ema200_raw) != "nan" else None
    regime = "Bull" if (ema200 is not None and price > ema200) else ("Bear" if ema200 is not None else "N/A")
    sig_str = signal.side if (signal and signal.should_enter) else None
    ema_str = ema200 if ema200 is not None else "N/A"
    print(f"[Heartbeat] {now} - Price: {price}, EMA200: {ema_str}, Regime: {regime}, Signal: {sig_str}")
    return 0, last_summary_date


def main():
    print("Futures Testnet 實戰啟動：每小時掃描 BNBUSDT 1h，deploy_ready 邏輯，2% 硬止損")
    client = get_client()
    init_futures_settings(client, SYMBOL, leverage=LEVERAGE, margin_type="ISOLATED")
    telegram_notifier = _get_telegram_notifier()
    consecutive_fail = 0
    last_summary_date = ""
    while True:
        try:
            consecutive_fail, last_summary_date = run_once(client, telegram_notifier, last_summary_date)
            if consecutive_fail >= CONSECUTIVE_FAIL_THRESHOLD:
                send_disconnect_alert()
                consecutive_fail = 0
        except Exception as e:
            consecutive_fail += 1
            sys.stderr.write(f"[futures_run] 本小時失敗: {e}\n")
            if consecutive_fail >= CONSECUTIVE_FAIL_THRESHOLD:
                send_disconnect_alert()
                consecutive_fail = 0
        time.sleep(3600)


if __name__ == "__main__":
    main()
