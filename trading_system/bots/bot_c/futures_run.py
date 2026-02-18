"""
Binance Futures Testnet 實戰腳本（1h 決策對齊 Calmar 13.9 回測）
- 決策週期 1h：進場訊號僅在每根 1h K 線收盤後評估（與回測一致）
- 數據源：ATR / SL / TP / price_breakout / Z-Score / EMA200 均來自 1h，不混用 3m
- Heartbeat 每 3 分鐘輸出一次，僅監控價格與連線，不觸發開倉/平倉
- 有訊號時於 Testnet 下 MARKET 單並掛 STOP_MARKET（2% 硬止損）
- 使用 Testnet: https://testnet.binancefuture.com
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

# 台灣時區 (UTC+8)：print / Telegram 顯示用此；paper_signals.json、heartbeat 寫入維持 UTC
TZ_TAIWAN = timezone(timedelta(hours=8))


def _now_taiwan() -> datetime:
    """目前時間（台灣）。"""
    return datetime.now(TZ_TAIWAN)


def _format_taiwan(dt: datetime | None) -> str:
    """將 datetime 轉為台灣時間顯示字串；若為 naive 則視為 UTC 再轉 +8。"""
    if dt is None:
        return "N/A"
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(TZ_TAIWAN).strftime("%Y-%m-%d %H:%M:%S")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    _env_root = ROOT / ".env"
    if _env_root.exists():
        load_dotenv(dotenv_path=_env_root)
except Exception:
    pass

SYMBOL = "BNBUSDT"
# 決策週期 1h（與回測 Calmar 13.9 一致）：進場/止損/止盈/breakout 全用 1h
INTERVAL_ENTRY = "1h"
INTERVAL_FILTER = "1h"
LOOKBACK_ENTRY = 220  # 1h：足夠 EMA200(200)+Z-Score(168)+ATR(14)+breakout(20)
LOOKBACK_FILTER = 220
# 僅在每小時 0–9 分內才評估進場（對應上一根 1h 收盤）
DECISION_WINDOW_START_MIN = 0
DECISION_WINDOW_END_MIN = 9
TESTNET_URL = "https://testnet.binancefuture.com"
LOG_DIR = ROOT / "logs"
SIGNALS_FILE = LOG_DIR / "paper_signals.json"
TRADE_HISTORY_CSV = LOG_DIR / "trade_history.csv"
HEARTBEAT_FILE = LOG_DIR / "paper_last_heartbeat.txt"
DISCONNECT_ALERT_FILE = LOG_DIR / "paper_disconnect_alert.log"
TRADE_HISTORY_HEADER = "entry_time_tw,exit_time_tw,side,qty,entry_price,exit_price,pnl_usdt,pnl_pct,fees,funding"
CONSECUTIVE_FAIL_THRESHOLD = 3
LOOP_SLEEP_SEC = 180  # 每 3 分鐘一輪
HARD_STOP_PCT = 2.0   # 2% 硬止損
LEVERAGE = 3
RISK_PCT_OF_EQUITY = 0.0025  # 0.25% 風險
# 每日總結觸發小時（目前=7 為測試用，驗證通知後請改回 8）
SUMMARY_TRIGGER_HOUR = 8


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


def fetch_klines(client, symbol: str, interval: str, limit: int):
    """拉取指定週期 K 線，僅當前循環所需數量以節省記憶體。"""
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


def _minutes_to_next_1h_close() -> int:
    """距離下一根 1h K 線收盤還剩幾分鐘（UTC）。"""
    now = datetime.now(timezone.utc)
    return 60 - now.minute - (1 if now.second >= 30 else 0)


def fetch_merged_row(client, symbol: str = SYMBOL):
    """
    1h 決策模式：所有進場/止損/止盈/breakout/Z-Score/EMA200 均來自 1h K 線。
    回傳 (merged_last_closed, r1h_current, minutes_to_1h)。
    - merged_last_closed: 上一根「已收盤」的 1h row，供訊號判定（與回測一致）
    - r1h_current: 當前 1h 根（可能未收盤），供 heartbeat 顯示
    - minutes_to_1h: 距離下一根 1h 收盤還剩幾分
    """
    df_1h = fetch_klines(client, symbol, INTERVAL_ENTRY, LOOKBACK_ENTRY)
    if df_1h is None or len(df_1h) < 200:
        return None, None, None
    df_1h = add_factors(df_1h)
    # 上一根已收盤的 1h（回測同邏輯：在 bar 收盤時做決策）
    r_last_closed = df_1h.iloc[-2].to_dict() if len(df_1h) >= 2 else df_1h.iloc[-1].to_dict()
    r_current = df_1h.iloc[-1].to_dict()
    merged_last_closed = {
        "close": r_last_closed["close"],
        "ema_200": r_last_closed.get("ema_200"),
        "atr": r_last_closed["atr"],
        "funding_z_score": r_last_closed.get("funding_z_score", 0),
        "rsi_z_score": r_last_closed.get("rsi_z_score", 0),
        "price_breakout_long": r_last_closed.get("price_breakout_long", 0),
        "price_breakout_short": r_last_closed.get("price_breakout_short", 0),
        "timestamp": r_last_closed["timestamp"],
    }
    return merged_last_closed, r_current, _minutes_to_next_1h_close()


def ensure_log_dir():
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def get_position_info(client, symbol: str) -> dict | None:
    """取得當前持倉摘要（開倉價、數量、未實現盈虧、保證金模式）。無倉位回傳 None。"""
    try:
        positions = client.get_position_risk(symbol=symbol)
        for p in positions or []:
            amt = float(p.get("positionAmt", 0) or 0)
            if amt == 0:
                continue
            return {
                "positionAmt": amt,
                "entryPrice": float(p.get("entryPrice", 0) or 0),
                "unrealizedProfit": float(p.get("unrealizedProfit", 0) or 0),
                "marginType": (p.get("marginType") or "UNKNOWN").upper(),
                "side": "BUY" if amt > 0 else "SELL",
            }
    except Exception:
        pass
    return None


def init_futures_settings(client, symbol: str, leverage: int = LEVERAGE, margin_type: str = "ISOLATED", has_position: bool = False):
    """啟動時設定槓桿與逐倉；若有持倉則不強制切換模式，僅記錄警告並繼續。"""
    try:
        if not has_position:
            try:
                client.set_margin_type(symbol=symbol, margin_type=margin_type)
                time.sleep(0.3)
            except Exception as e:
                err_str = str(e)
                if "No need to change margin type" in err_str:
                    pass
                else:
                    print(f"  [WARN] set_margin_type 跳過或失敗（不中斷）: {err_str}")
        try:
            client.set_leverage(symbol=symbol, leverage=leverage)
            print(f"  [OK] {symbol} 槓桿={leverage}x, 保證金目標={margin_type}")
        except Exception as e:
            print(f"  [WARN] set_leverage: {e}")
    except Exception as e:
        print(f"  [WARN] init_futures_settings 非致命: {e}")


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
    """掛 STOP_MARKET 平倉（2% 硬止損）。回傳含 orderId 的結果供 Telegram 顯示。"""
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


def get_margin_type_from_api(client, symbol: str) -> str:
    """從倉位 API 讀取當前保證金模式（全倉/逐倉）。"""
    try:
        positions = client.get_position_risk(symbol=symbol)
        for p in positions or []:
            mt = (p.get("marginType") or "").strip().upper()
            if mt:
                return "逐倉" if mt == "ISOLATED" else "全倉"
    except Exception:
        pass
    return "N/A"


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


def _get_recent_income_for_close(client, symbol: str, limit: int = 30) -> tuple[float, float, float]:
    """取得最近一筆平倉相關的已實現盈虧、手續費、資金費（供寫入 trade_history）。"""
    try:
        items = client.get_income_history(symbol=symbol, limit=limit)
        realized, funding, commission = 0.0, 0.0, 0.0
        now_ms = int(time.time() * 1000)
        for x in (items or []):
            if x.get("asset") != "USDT":
                continue
            t = int(x.get("time", 0) or 0)
            if now_ms - t > 60000 * 10:  # 只取 10 分鐘內
                continue
            inc = float(x.get("income", 0) or 0)
            it = x.get("incomeType", "")
            if it == "REALIZED_PNL":
                realized += inc
            elif it == "FUNDING_FEE":
                funding += inc
            elif it == "COMMISSION":
                commission += inc
        return realized, funding, commission
    except Exception:
        pass
    return 0.0, 0.0, 0.0


def append_trade_history_row(
    entry_time_tw: str,
    exit_time_tw: str,
    side: str,
    qty: float,
    entry_price: float,
    exit_price: float,
    pnl_usdt: float,
    pnl_pct: float,
    fees: float,
    funding: float,
):
    """平倉時追加一筆到 trade_history.csv（永恆帳本）。"""
    ensure_log_dir()
    write_header = not TRADE_HISTORY_CSV.exists()
    try:
        with open(TRADE_HISTORY_CSV, "a", encoding="utf-8", newline="") as f:
            if write_header:
                f.write(TRADE_HISTORY_HEADER + "\n")
            f.write(f"{entry_time_tw},{exit_time_tw},{side},{qty},{entry_price},{exit_price},{pnl_usdt:.2f},{pnl_pct:.2f},{fees:.2f},{funding:.2f}\n")
    except Exception as e:
        print(f"  [WARN] append_trade_history 失敗: {e}")


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


def _get_wallet_balance_usdt(client) -> float:
    """從 API 獲取 USDT 總餘額（Total Wallet Balance）。失敗回傳 0，不拋錯。"""
    try:
        for b in (client.get_balance() or []):
            if b.get("asset") == "USDT":
                return float(b.get("balance", 0) or 0)
    except Exception:
        pass
    return 0.0


def _get_daily_realized_pnl(client, symbol: str, hours: int = 24) -> float:
    """過去 hours 小時內已實現盈虧 + 資金費。失敗回傳 0，不拋錯。"""
    try:
        cutoff_ms = int(time.time() * 1000) - hours * 3600 * 1000
        items = client.get_income_history(symbol=symbol, limit=500)
        total = 0.0
        for x in items or []:
            if x.get("asset") != "USDT":
                continue
            t = x.get("incomeType", "")
            if t not in ("REALIZED_PNL", "FUNDING_FEE"):
                continue
            if int(x.get("time", 0) or 0) < cutoff_ms:
                continue
            total += float(x.get("income", 0) or 0)
        return total
    except Exception:
        pass
    return 0.0


def _get_daily_commission(client, symbol: str, hours: int = 24) -> float:
    """過去 hours 小時內手續費合計。失敗回傳 0，不拋錯。"""
    try:
        cutoff_ms = int(time.time() * 1000) - hours * 3600 * 1000
        items = client.get_income_history(symbol=symbol, limit=500)
        total = 0.0
        for x in items or []:
            if x.get("asset") != "USDT":
                continue
            if x.get("incomeType") != "COMMISSION":
                continue
            if int(x.get("time", 0) or 0) < cutoff_ms:
                continue
            total += float(x.get("income", 0) or 0)
        return total
    except Exception:
        pass
    return 0.0


def _send_daily_summary(client, notifier, last_sent_date: str) -> str:
    """每日實戰總結：API 總餘額、昨日盈虧、持倉浮動 + 本地訊號統計。觸發為台灣時間 8 點。"""
    from datetime import date, timedelta
    now_tw = _now_taiwan()
    today_tw = now_tw.date()
    if last_sent_date and last_sent_date == today_tw.isoformat():
        return last_sent_date
    if now_tw.hour != SUMMARY_TRIGGER_HOUR:
        return last_sent_date
    yesterday_tw = today_tw - timedelta(days=1)
    yesterday = yesterday_tw.isoformat()
    current_time = now_tw.strftime("%Y-%m-%d %H:%M")

    balance = 0.0
    daily_pnl = 0.0
    daily_fees = 0.0
    position_info = get_position_info(client, SYMBOL)
    try:
        balance = _get_wallet_balance_usdt(client)
        daily_pnl = _get_daily_realized_pnl(client, SYMBOL, hours=24)
        daily_fees = _get_daily_commission(client, SYMBOL, hours=24)
    except Exception as e:
        print(f"  [WARN] 每日總結 API 取得失敗（不中斷）: {e}")

    pnl_pct = (daily_pnl / balance * 100) if balance and balance > 0 else 0.0
    if position_info:
        pos = position_info
        current_position_info = (
            f"{pos['side']} {abs(pos['positionAmt'])} BNB | "
            f"開倉價 {pos['entryPrice']} | 浮動盈虧 {pos['unrealizedProfit']:+.2f} USDT"
        )
    else:
        current_position_info = "無持倉"

    records = []
    if SIGNALS_FILE.exists():
        try:
            with open(SIGNALS_FILE, "r", encoding="utf-8") as f:
                records = json.load(f)
        except Exception:
            records = []
    if not isinstance(records, list):
        records = []
    # 昨日訊號：依台灣「昨日」篩選（time_utc 為 UTC，轉台灣日期再比對）
    def _utc_str_to_taiwan_date(utc_str: str) -> date | None:
        try:
            from datetime import datetime as dt_parse
            if not utc_str:
                return None
            # 支援 ISO 含 +00:00 或 Z
            t = dt_parse.fromisoformat(utc_str.replace("Z", "+00:00"))
            return t.astimezone(TZ_TAIWAN).date()
        except Exception:
            return None
    yesterday_signals = [r for r in records if _utc_str_to_taiwan_date(r.get("time_utc") or "") == yesterday_tw]
    total_count = len(records)
    longs = sum(1 for s in records if (s.get("side") or "").upper() == "BUY")
    shorts = sum(1 for s in records if (s.get("side") or "").upper() == "SELL")
    count = len(yesterday_signals)

    msg = (
        "📊 <b>【Strategy C】每日實戰總結</b>\n"
        f"📅 日期：{yesterday}\n"
        "-------------------------\n"
        "💰 <b>帳戶狀態</b>\n"
        f"總餘額：{balance:.2f} USDT\n"
        f"昨日盈虧：{daily_pnl:+.2f} USDT ({pnl_pct:+.2f}%)\n"
        f"昨日手續費：{daily_fees:.2f} USDT\n"
        "📈 <b>交易統計</b>\n"
        f"昨日訊號：{count} 筆\n"
        f"累計總筆數：{total_count} (多:{longs} / 空:{shorts})\n"
        "🛡️ <b>當前持倉</b>\n"
        f"{current_position_info}\n"
        f"⏰ 報時：{current_time}"
    )
    if notifier and getattr(notifier, "send_message", None):
        try:
            notifier.send_message(msg)
        except Exception as e:
            print(f"  [WARN] 每日總結 Telegram 發送失敗: {e}")
    return today_tw.isoformat()


def send_disconnect_alert():
    ensure_log_dir()
    msg = f"[{_now_taiwan().strftime('%Y-%m-%d %H:%M:%S')} UTC+8] 斷線：連續 {CONSECUTIVE_FAIL_THRESHOLD} 輪無法取得 K 線\n"
    with open(DISCONNECT_ALERT_FILE, "a", encoding="utf-8") as f:
        f.write(msg)
    sys.stderr.write(msg)


def run_once(client, telegram_notifier=None, last_summary_date: str = "", last_position_amt: float = 0.0):
    merged, r1h_current, minutes_to_1h = fetch_merged_row(client)
    if merged is None or r1h_current is None:
        return 1, last_summary_date, last_position_amt
    row = merged
    in_decision_window = (
        DECISION_WINDOW_END_MIN is not None
        and datetime.now(timezone.utc).minute >= DECISION_WINDOW_START_MIN
        and datetime.now(timezone.utc).minute <= DECISION_WINDOW_END_MIN
    )
    # 持倉狀態（用於偵測平倉並寫入 trade_history.csv）
    pos = get_position_info(client, SYMBOL)
    current_amt = pos["positionAmt"] if pos else 0.0
    if last_position_amt != 0 and current_amt == 0:
        try:
            time.sleep(2)
            exit_time_tw = _now_taiwan().strftime("%Y-%m-%d %H:%M:%S")
            entry_time_tw = exit_time_tw
            records = []
            if SIGNALS_FILE.exists():
                with open(SIGNALS_FILE, "r", encoding="utf-8") as f:
                    records = json.load(f)
            if isinstance(records, list) and records:
                last_rec = records[-1]
                entry_ts = last_rec.get("time_utc") or ""
                if entry_ts:
                    from datetime import datetime as dt_parse
                    t = dt_parse.fromisoformat(entry_ts.replace("Z", "+00:00"))
                    entry_time_tw = t.astimezone(TZ_TAIWAN).strftime("%Y-%m-%d %H:%M:%S")
                side = (last_rec.get("side") or "").upper()
                qty = float(last_rec.get("qty", 0) or 0)
                entry_price = float(last_rec.get("entry_price", 0) or 0)
                exit_price = entry_price
                try:
                    trades = client.get_user_trades(SYMBOL, limit=10)
                    if trades:
                        latest = trades[-1]
                        exit_price = float(latest.get("price", 0) or 0)
                except Exception:
                    pass
                realized, funding, commission = _get_recent_income_for_close(client, SYMBOL)
                pnl_usdt = realized + funding
                fees = commission
                if entry_price and qty:
                    pnl_pct = (pnl_usdt / (entry_price * qty) * 100)
                else:
                    pnl_pct = 0.0
                append_trade_history_row(
                    entry_time_tw, exit_time_tw, side, qty, entry_price, exit_price,
                    pnl_usdt, pnl_pct, fees, funding,
                )
                print(f"  [帳本] 平倉已寫入 trade_history.csv | 出場價 {exit_price} | PnL {pnl_usdt:+.2f} USDT")
                if telegram_notifier and getattr(telegram_notifier, "send_message", None):
                    try:
                        telegram_notifier.send_message(
                            f"🚨 <b>【平倉通知】</b> 趨勢反轉或觸發止損！\n"
                            f"出場價：{exit_price}\n"
                            f"預估損益：{pnl_usdt:+.2f} USDT"
                        )
                    except Exception as tg_err:
                        print(f"  [WARN] 平倉 Telegram 發送失敗: {tg_err}")
        except Exception as e:
            print(f"  [WARN] 平倉寫入帳本失敗: {e}")
    from bots.bot_c.deploy_ready import get_signal_from_row, get_deploy_params, HARD_STOP_POSITION_PCT
    # 僅在 1h 收盤後之決策窗口內評估進場（與回測一致）
    signal = get_signal_from_row(row, get_deploy_params()) if in_decision_window else None
    hard_stop_pct = HARD_STOP_POSITION_PCT

    if in_decision_window and signal and signal.should_enter:
        ts = row.get("timestamp")
        bar_time = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
        if current_amt != 0:
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
                    stop_order = place_stop_market_close(client, SYMBOL, signal.side, sl_price)
                    stop_order_id = stop_order.get("orderId") if stop_order else None
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
                        "stop_order_id": stop_order_id,
                    }
                    append_signal_record(record)
                    print(f"  [FILL] {signal.side} qty={qty} @ {signal.entry_price}  SL={sl_price}  orderId={order.get('orderId')} stopOrderId={stop_order_id}")
                    if telegram_notifier and getattr(telegram_notifier, "send_message", None):
                        margin_mode = get_margin_type_from_api(client, SYMBOL)
                        fz = row.get("funding_z_score")
                        rz = row.get("rsi_z_score")
                        fz_str = round(float(fz), 2) if fz is not None and str(fz) != "nan" else "N/A"
                        rz_str = round(float(rz), 2) if rz is not None and str(rz) != "nan" else "N/A"
                        telegram_notifier.send_message(
                            f"📊 <b>Testnet: {signal.side}</b>\n"
                            f"開倉模式: {margin_mode} | Entry: {signal.entry_price} | SL: {sl_price} | qty: {qty}\n"
                            f"止損單 ID: {stop_order_id or 'N/A'}\n"
                            f"1h Z-Score: FundingZ={fz_str} RSI_Z={rz_str}\n"
                            f"Bar: {bar_time}"
                        )
                else:
                    print(f"  [ERR] 市價單未成交")

    last_summary_date = _send_daily_summary(client, telegram_notifier, last_summary_date)
    now = _now_taiwan().strftime("%Y-%m-%d %H:%M:%S")
    _write_heartbeat(datetime.now(timezone.utc).isoformat())
    # Heartbeat 用當前價（優先 3m 最後收盤價以每 3 分鐘更新，僅監控用）
    try:
        df_3m = fetch_klines(client, SYMBOL, "3m", 2)
        price = round(float(df_3m.iloc[-1]["close"]), 2) if df_3m is not None and len(df_3m) else round(float(r1h_current.get("close", 0)), 2)
    except Exception:
        price = round(float(r1h_current.get("close", 0)), 2)
    mins_left = minutes_to_1h if minutes_to_1h is not None else _minutes_to_next_1h_close()
    ema200_raw = row.get("ema_200")
    ema200 = round(float(ema200_raw), 2) if ema200_raw is not None and str(ema200_raw) != "nan" else None
    regime = "Bull" if (ema200 is not None and price > ema200) else ("Bear" if ema200 is not None else "N/A")
    fz = row.get("funding_z_score")
    rz = row.get("rsi_z_score")
    fz_str = round(float(fz), 2) if fz is not None and str(fz) != "nan" else "N/A"
    rz_str = round(float(rz), 2) if rz is not None and str(rz) != "nan" else "N/A"
    sig_str = signal.side if (in_decision_window and signal and signal.should_enter) else None
    ema_str = ema200 if ema200 is not None else "N/A"
    print(f"[1h 決策模式] 當前價格: {price} | 距離下一根 1h 收盤還剩: {mins_left} 分 | FundingZ: {fz_str} RSI_Z: {rz_str} | EMA200: {ema_str} Regime: {regime} | Signal: {sig_str}")
    return 0, last_summary_date, current_amt


def trim_log_lines(log_path: Path, keep_lines: int = 10000) -> None:
    """將日誌檔保留最近 keep_lines 行，避免塞爆磁碟。"""
    if not log_path.exists():
        return
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        if len(lines) <= keep_lines:
            return
        with open(log_path, "w", encoding="utf-8") as f:
            f.writelines(lines[-keep_lines:])
        print(f"  [LOG] 已滾動 {log_path.name}，保留最近 {keep_lines} 行")
    except Exception as e:
        print(f"  [WARN] 日誌滾動跳過: {e}")


def main():
    print("Futures Testnet 實戰啟動：1h 決策模式（對齊 Calmar 13.9 回測），每小時 0–9 分評估進場，Heartbeat 每 3 分鐘，2% 硬止損")
    ensure_log_dir()
    trim_log_lines(LOG_DIR / "paper_out.log", 10000)
    trim_log_lines(LOG_DIR / "paper_err.log", 10000)

    client = get_client()
    position = get_position_info(client, SYMBOL)
    if position:
        print(f"  [現有持倉接管] {position['side']} 數量={position['positionAmt']} 開倉價={position['entryPrice']} 未實現盈虧={position['unrealizedProfit']} 保證金模式={position['marginType']}")
        print("  將繼續追蹤，不重複開倉；2% 止損邏輯維持運作。")
    init_futures_settings(client, SYMBOL, leverage=LEVERAGE, margin_type="ISOLATED", has_position=bool(position))

    telegram_notifier = _get_telegram_notifier()
    consecutive_fail = 0
    last_summary_date = ""
    last_position_amt = 0.0
    if position:
        last_position_amt = position["positionAmt"]
    while True:
        try:
            consecutive_fail, last_summary_date, last_position_amt = run_once(
                client, telegram_notifier, last_summary_date, last_position_amt
            )
            if consecutive_fail >= CONSECUTIVE_FAIL_THRESHOLD:
                send_disconnect_alert()
                consecutive_fail = 0
        except Exception as e:
            consecutive_fail += 1
            sys.stderr.write(f"[futures_run] 本輪失敗: {e}\n")
            if consecutive_fail >= CONSECUTIVE_FAIL_THRESHOLD:
                send_disconnect_alert()
                consecutive_fail = 0
        time.sleep(LOOP_SLEEP_SEC)


if __name__ == "__main__":
    main()
