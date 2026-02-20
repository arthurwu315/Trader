"""
Binance Futures 實戰腳本 - 1D 宏觀趨勢組合 (Macro Portfolio)
- 決策週期 1d：每日 UTC 00:05~00:15（UTC+8 08:05~08:15）評估一次
- 掃描多幣種，若同日多訊號則以 ROC30 做 RS 仲裁擇優下單
- 維持 MAX_CONCURRENT 風控上限並掛 ATR 初始止損
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

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "AVAXUSDT",
    "ADAUSDT", "XRPUSDT", "DOTUSDT", "LINKUSDT", "LTCUSDT",
    "BCHUSDT", "MATICUSDT", "UNIUSDT", "ICPUSDT", "NEARUSDT",
]
PRIMARY_SYMBOL = SYMBOLS[0]
# 決策週期 1d：每天 UTC 00:05~00:15（台灣 08:05~08:15）評估一次
INTERVAL_ENTRY = "1d"
INTERVAL_FILTER = "1d"
LOOKBACK_ENTRY = 320   # 1d：足夠 EMA200 / Donchian 80 / ATR14 / ROC30
LOOKBACK_FILTER = 320
DECISION_WINDOW_START_MINUTE_UTC = 5
DECISION_WINDOW_END_MINUTE_UTC = 15
# 實盤上線時設為 False；Testnet 測試時設為 True
TESTNET = False
TESTNET_URL = "https://testnet.binancefuture.com"
MAINNET_URL = "https://fapi.binance.com"
LOG_DIR = ROOT / "logs"
SIGNALS_FILE = LOG_DIR / "paper_signals.json"
TRADE_HISTORY_CSV = LOG_DIR / "trade_history.csv"
HEARTBEAT_FILE = LOG_DIR / "paper_last_heartbeat.txt"
REGIME_FILE = LOG_DIR / "paper_last_regime.txt"
DISCONNECT_ALERT_FILE = LOG_DIR / "paper_disconnect_alert.log"
TRADE_HISTORY_HEADER = "entry_time_tw,exit_time_tw,side,qty,entry_price,exit_price,pnl_usdt,pnl_pct,fees,funding"
CONSECUTIVE_FAIL_THRESHOLD = 3
LOOP_SLEEP_SEC = 300  # 每 5 分鐘一輪
HARD_STOP_PCT = 2.0   # 2% 硬止損
LEVERAGE = 3
RISK_PCT_OF_EQUITY = 0.0025  # 0.25% 風險
MAX_CONCURRENT = 2
NOTIONAL_PCT_OF_EQUITY = 0.50
LEVERAGE_WARN_THRESHOLD = 1.5
FUNDING_ALERT_RATE = 0.0005      # 0.05% / 8h
FUNDING_SHORT_SKIP_ANNUAL = 0.20 # 做空年化資費 > 20% 則跳過
SPREAD_ALERT_PCT = 0.15
CIRCUIT_DRAWDOWN_PCT = 20.0
CIRCUIT_COOLDOWN_HOURS = 48
RISK_STATE_FILE = LOG_DIR / "paper_risk_state.json"
# 每日總結觸發小時（目前=7 為測試用，驗證通知後請改回 8）
SUMMARY_TRIGGER_HOUR = 8


def get_client():
    from bots.bot_c.config_c import get_strategy_c_config
    from core.binance_client import BinanceFuturesClient
    cfg = get_strategy_c_config()
    base = os.getenv("BINANCE_BASE_URL", MAINNET_URL if not TESTNET else TESTNET_URL)
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


# 1d = 1440 分鐘
MINUTES_PER_1D = 1440


def _minutes_to_next_1d_close() -> int:
    """距離下一根 1d K 線收盤還剩幾分鐘（UTC）。"""
    now = datetime.now(timezone.utc)
    total_min = now.hour * 60 + now.minute
    offset = total_min % MINUTES_PER_1D
    return MINUTES_PER_1D - offset - (1 if now.second >= 30 else 0)


def _in_daily_decision_window(now_utc: datetime) -> bool:
    return (
        now_utc.hour == 0
        and DECISION_WINDOW_START_MINUTE_UTC <= now_utc.minute <= DECISION_WINDOW_END_MINUTE_UTC
    )


def fetch_merged_row(client, symbol: str):
    """
    1d 決策模式：進場/止損/roll_high_N/roll_low_N 均來自 1d K 線。
    回傳 (merged_last_closed, r_current, minutes_to_1d)。
    """
    df = fetch_klines(client, symbol, INTERVAL_ENTRY, LOOKBACK_ENTRY)
    if df is None or len(df) < 100:
        return None, None, None
    df["roc_30"] = df["close"].pct_change(30)
    df = add_factors(df)
    r_last_closed = df.iloc[-2].to_dict() if len(df) >= 2 else df.iloc[-1].to_dict()
    r_current = df.iloc[-1].to_dict()
    merged_last_closed = dict(r_last_closed)
    return merged_last_closed, r_current, _minutes_to_next_1d_close()


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


def _get_total_equity_usdt(client) -> float:
    """總淨值（Wallet + Unrealized PnL）。"""
    try:
        acc = client.get_account()
        wallet = float(acc.get("totalWalletBalance", 0) or 0)
        upnl = float(acc.get("totalUnrealizedProfit", 0) or 0)
        return wallet + upnl
    except Exception:
        return _get_wallet_balance_usdt(client)


def _get_total_open_notional_usdt(client) -> float:
    """當前所有持倉名目總和。"""
    total = 0.0
    try:
        positions = client.get_position_risk()
        for p in positions or []:
            amt = float(p.get("positionAmt", 0) or 0)
            if amt == 0:
                continue
            mark = float(p.get("markPrice", 0) or 0)
            total += abs(amt * mark)
    except Exception:
        pass
    return total


def _compute_qty_by_notional(equity_usdt: float, entry_price: float, notional_pct: float = NOTIONAL_PCT_OF_EQUITY) -> float:
    if equity_usdt <= 0 or entry_price <= 0:
        return 0.0
    notional = equity_usdt * notional_pct
    return round(notional / entry_price, 3)


def _get_funding_rate(client, symbol: str) -> float:
    """回傳最新 funding rate（每 8 小時）。"""
    try:
        out = client._call_with_retry("GET", "/fapi/v1/premiumIndex", {"symbol": symbol})
        return float(out.get("lastFundingRate", 0) or 0)
    except Exception:
        return 0.0


def _get_spread_pct(client, symbol: str) -> float:
    """回傳即時 spread 百分比。"""
    try:
        t = client.get_24h_ticker(symbol)
        bid = float(t.get("bidPrice", 0) or 0)
        ask = float(t.get("askPrice", 0) or 0)
        mid = (bid + ask) / 2.0
        if mid <= 0:
            return 0.0
        return (ask - bid) / mid * 100.0
    except Exception:
        return 0.0


def _load_risk_state() -> dict:
    if not RISK_STATE_FILE.exists():
        return {}
    try:
        data = json.loads(RISK_STATE_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_risk_state(state: dict) -> None:
    try:
        RISK_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        RISK_STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _parse_iso_utc(ts: str) -> datetime | None:
    try:
        if not ts:
            return None
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def _load_signal_records() -> list[dict]:
    if not SIGNALS_FILE.exists():
        return []
    try:
        items = json.loads(SIGNALS_FILE.read_text(encoding="utf-8"))
        return items if isinstance(items, list) else []
    except Exception:
        return []


def _find_latest_entry_record(symbol: str) -> dict | None:
    records = _load_signal_records()
    for rec in reversed(records):
        if str(rec.get("symbol", "")) == symbol and rec.get("time_utc"):
            return rec
    return None


def _get_funding_fee_since(client, symbol: str, since_utc: datetime) -> float:
    try:
        items = client.get_income_history(symbol=symbol, limit=1000)
        total = 0.0
        cutoff_ms = int(since_utc.timestamp() * 1000)
        for x in items or []:
            if x.get("asset") != "USDT":
                continue
            if x.get("incomeType") != "FUNDING_FEE":
                continue
            t = int(x.get("time", 0) or 0)
            if t < cutoff_ms:
                continue
            total += float(x.get("income", 0) or 0)
        return total
    except Exception:
        return 0.0


def _get_position_health_snapshot(client, symbol: str, pos: dict) -> tuple[str, str | None]:
    """
    回傳 (健康度文字, 可選警示文字)。
    指標：持有天數、累積資費、MFE/MAE(ATR 倍數估計)。
    """
    rec = _find_latest_entry_record(symbol)
    if not rec:
        return f"{symbol}: 無入場審計紀錄，暫無健康度資料", None

    entry_time_utc = _parse_iso_utc(str(rec.get("time_utc", "")))
    if not entry_time_utc:
        return f"{symbol}: 入場時間格式異常，暫無健康度資料", None

    entry_price = float(rec.get("entry_price", pos.get("entryPrice", 0)) or 0)
    sl_price = float(rec.get("sl_price", entry_price) or entry_price)
    if entry_price <= 0:
        return f"{symbol}: 入場價異常，暫無健康度資料", None

    hold_days = max((datetime.now(timezone.utc) - entry_time_utc).total_seconds() / 86400.0, 0.0)
    funding_fee = _get_funding_fee_since(client, symbol, entry_time_utc)

    # 用入場 SL 反推 ATR（ATR_STOP_MULT=2.5）
    atr_est = abs(entry_price - sl_price) / 2.5 if abs(entry_price - sl_price) > 0 else max(entry_price * 0.01, 1e-9)
    start_ms = int(entry_time_utc.timestamp() * 1000)
    kl = client.get_klines(symbol=symbol, interval="1d", limit=200, start_time=start_ms)
    highs: list[float] = []
    lows: list[float] = []
    for row in kl or []:
        try:
            highs.append(float(row[2]))
            lows.append(float(row[3]))
        except Exception:
            continue
    if not highs or not lows:
        return f"{symbol}: 持有 {hold_days:.1f} 天 | Funding {funding_fee:+.2f} USDT | MFE/MAE=N/A", None

    side = str(pos.get("side", "")).upper()
    if side == "BUY":
        mfe = (max(highs) - entry_price) / atr_est
        mae = (entry_price - min(lows)) / atr_est
    else:
        mfe = (entry_price - min(lows)) / atr_est
        mae = (max(highs) - entry_price) / atr_est

    upnl = float(pos.get("unrealizedProfit", 0) or 0)
    warning = None
    if upnl > 0 and funding_fee < 0 and abs(funding_fee) >= (0.1 * upnl):
        warning = f"{symbol} 持有成本過高: Funding {funding_fee:+.2f} 已達浮盈 {upnl:+.2f} 的 10%以上"

    line = (
        f"{symbol}: 持有 {hold_days:.1f} 天 | Funding {funding_fee:+.2f} USDT | "
        f"MFE {mfe:.2f} ATR / MAE {mae:.2f} ATR"
    )
    return line, warning


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
    position_info = get_position_info(client, PRIMARY_SYMBOL)
    try:
        balance = _get_wallet_balance_usdt(client)
        daily_pnl = _get_daily_realized_pnl(client, PRIMARY_SYMBOL, hours=24)
        daily_fees = _get_daily_commission(client, PRIMARY_SYMBOL, hours=24)
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


def _load_regime_map() -> dict[str, str]:
    if not REGIME_FILE.exists():
        return {}
    try:
        data = json.loads(REGIME_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_regime_map(regime_map: dict[str, str]) -> None:
    try:
        REGIME_FILE.parent.mkdir(parents=True, exist_ok=True)
        REGIME_FILE.write_text(json.dumps(regime_map, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _count_open_positions(client) -> int:
    total = 0
    for symbol in SYMBOLS:
        if has_open_position(client, symbol):
            total += 1
    return total


def _refresh_circuit_state(state: dict, equity: float, now_utc: datetime) -> dict:
    month_key = now_utc.strftime("%Y-%m")
    if state.get("month_key") != month_key:
        state["month_key"] = month_key
        state["month_peak_equity"] = equity
    peak = float(state.get("month_peak_equity", equity) or equity)
    if equity > peak:
        peak = equity
        state["month_peak_equity"] = equity
    drawdown_pct = ((peak - equity) / peak * 100.0) if peak > 0 else 0.0
    state["latest_drawdown_pct"] = drawdown_pct
    until = state.get("circuit_until_utc", "")
    active = False
    if until:
        try:
            active = now_utc < datetime.fromisoformat(until.replace("Z", "+00:00"))
        except Exception:
            active = False
    if drawdown_pct >= CIRCUIT_DRAWDOWN_PCT and not active:
        until_dt = now_utc + timedelta(hours=CIRCUIT_COOLDOWN_HOURS)
        state["circuit_until_utc"] = until_dt.isoformat()
        state["circuit_triggered_at_utc"] = now_utc.isoformat()
        state["circuit_last_alert_date"] = now_utc.date().isoformat()
        active = True
    state["circuit_active"] = active
    return state


def _send_macro_control_report(
    notifier,
    report_date: str,
    equity: float,
    equity_change_pct: float,
    top3: list[str],
    decision: str,
    warnings: list[str],
    real_leverage: float,
    audit_lines: list[str],
    health_lines: list[str],
) -> None:
    if not notifier or not getattr(notifier, "send_message", None):
        return
    warn_txt = "；".join(warnings) if warnings else "無"
    msg = (
        "📊 <b>[1D Macro 實盤報告]</b>\n"
        f"📅 日期: {report_date}\n"
        f"💰 當前淨值: {equity:.2f} USDT ({equity_change_pct:+.2f}%)\n"
        f"🎯 RS 候選名單: {', '.join(top3) if top3 else 'None'}\n"
        f"🛡️ 決策結果: {decision}\n"
        f"🧾 決策審計: {' | '.join(audit_lines) if audit_lines else '無'}\n"
        f"🩺 持倉健康度: {' | '.join(health_lines) if health_lines else '無持倉'}\n"
        f"📐 真實槓桿率: {real_leverage:.2f}x\n"
        f"⚠️ 異常提醒: {warn_txt}\n"
        "🆘 緊急指令預留: /close_all (目前僅保留說明，尚未啟用自動執行)"
    )
    try:
        notifier.send_message(msg)
    except Exception as e:
        print(f"  [WARN] Telegram 中控報告發送失敗: {e}")


def run_once(client, telegram_notifier=None, last_summary_date: str = "", last_scan_date: str = ""):
    from bots.bot_c.deploy_ready import get_signal_from_row, get_deploy_params, HARD_STOP_POSITION_PCT

    now_utc = datetime.now(timezone.utc)
    today_utc = now_utc.date().isoformat()
    in_decision_window = _in_daily_decision_window(now_utc)
    params = get_deploy_params()
    regime_map = _load_regime_map()
    risk_state = _load_risk_state()
    equity = _get_total_equity_usdt(client)
    risk_state = _refresh_circuit_state(risk_state, equity, now_utc)
    _save_risk_state(risk_state)

    candidates: list[dict] = []
    candidate_symbols: list[str] = []
    warning_msgs: list[str] = []
    audit_lines: list[str] = []
    health_lines: list[str] = []
    decision_text = "續抱"

    if in_decision_window and last_scan_date != today_utc:
        for symbol in SYMBOLS:
            merged, _, _ = fetch_merged_row(client, symbol)
            if merged is None:
                continue
            last_regime = regime_map.get(symbol)
            signal, current_regime = get_signal_from_row(merged, params, last_regime=last_regime)
            regime_map[symbol] = current_regime
            if signal and signal.should_enter:
                roc_30 = float(merged.get("roc_30", 0.0) or 0.0)
                funding_rate = _get_funding_rate(client, symbol)
                spread_pct = _get_spread_pct(client, symbol)
                annual_funding = max(funding_rate, 0.0) * 3.0 * 365.0
                # 做空 funding 過高直接跳過，避免長線侵蝕
                if signal.side == "SELL" and annual_funding > FUNDING_SHORT_SKIP_ANNUAL:
                    warning_msgs.append(
                        f"{symbol} 做空跳過: funding年化 {annual_funding*100:.2f}% > {FUNDING_SHORT_SKIP_ANNUAL*100:.0f}%"
                    )
                    audit_lines.append(
                        f"{symbol}: 年化資費 {annual_funding*100:.2f}% > {FUNDING_SHORT_SKIP_ANNUAL*100:.0f}%"
                    )
                    continue
                if abs(funding_rate) > FUNDING_ALERT_RATE:
                    warning_msgs.append(f"{symbol} Funding {funding_rate*100:.3f}%/8h 偏高")
                if spread_pct > SPREAD_ALERT_PCT:
                    warning_msgs.append(f"{symbol} Spread {spread_pct:.3f}% 偏大")
                    audit_lines.append(f"{symbol}: Spread {spread_pct:.3f}% > {SPREAD_ALERT_PCT:.3f}%")
                    continue
                candidates.append(
                    {
                        "symbol": symbol,
                        "signal": signal,
                        "row": merged,
                        "roc_30": roc_30,
                        "funding_rate": funding_rate,
                        "spread_pct": spread_pct,
                    }
                )
                candidate_symbols.append(symbol)
        _save_regime_map(regime_map)

        selected = None
        longs = [c for c in candidates if c["signal"].side == "BUY"]
        shorts = [c for c in candidates if c["signal"].side == "SELL"]
        best_long = max(longs, key=lambda x: x["roc_30"]) if longs else None
        best_short = min(shorts, key=lambda x: x["roc_30"]) if shorts else None
        if best_long and best_short:
            selected = best_long if abs(best_long["roc_30"]) >= abs(best_short["roc_30"]) else best_short
        elif best_long:
            selected = best_long
        elif best_short:
            selected = best_short

        selected_symbol = selected["symbol"] if selected else "None"
        print(
            f"[Macro Scan] 掃描日期: {today_utc} | 候選訊號: {candidate_symbols or ['None']} | "
            f"RS 仲裁選擇: {selected_symbol}"
        )
        top3 = sorted(candidates, key=lambda x: abs(x["roc_30"]), reverse=True)[:3]
        top3_fmt = [f"{x['symbol']}({x['roc_30']:+.2%})" for x in top3]
        rs_rank = sorted(candidates, key=lambda x: abs(x["roc_30"]), reverse=True)
        for i, c in enumerate(rs_rank, start=1):
            if selected and c["symbol"] != selected["symbol"]:
                audit_lines.append(f"{c['symbol']}: RS 排名不足 (位居第 {i})")

        if selected:
            if risk_state.get("circuit_active", False):
                decision_text = "Circuit Breaker 啟動，暫停新倉"
                print("  [RISK] Circuit Breaker 啟動，跳過新進場")
                if telegram_notifier and getattr(telegram_notifier, "send_message", None):
                    telegram_notifier.send_message(
                        "🚨 <b>緊急止損警告</b>\n"
                        f"當月峰值回撤已達 {risk_state.get('latest_drawdown_pct', 0.0):.2f}%\n"
                        f"新倉暫停至: {risk_state.get('circuit_until_utc', 'N/A')}"
                    )
            else:
                open_count = _count_open_positions(client)
                if open_count >= MAX_CONCURRENT:
                    decision_text = "倉位已滿 Skip"
                    print(f"  [SKIP] 已達 MAX_CONCURRENT={MAX_CONCURRENT}")
                elif has_open_position(client, selected["symbol"]):
                    decision_text = f"{selected['symbol']} 已有持倉 Skip"
                    print(f"  [SKIP] {selected['symbol']} 已有持倉")
                else:
                    signal = selected["signal"]
                    row = selected["row"]
                    symbol = selected["symbol"]
                    qty = _compute_qty_by_notional(equity, signal.entry_price, NOTIONAL_PCT_OF_EQUITY)
                    if qty <= 0:
                        decision_text = "淨值不足 Skip"
                        print(f"  [SKIP] 淨值不足或 qty=0 (equity={equity:.2f})")
                    else:
                        order = place_market_order(client, symbol, signal.side, qty)
                        if order:
                            decision_text = f"進場 {symbol}"
                            sl_price = signal.sl_price
                            stop_order = place_stop_market_close(client, symbol, signal.side, sl_price)
                            stop_order_id = stop_order.get("orderId") if stop_order else None
                            ts = row.get("timestamp")
                            bar_time = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
                            append_signal_record(
                                {
                                    "time_utc": datetime.now(timezone.utc).isoformat(),
                                    "symbol": symbol,
                                    "bar_time": bar_time,
                                    "side": signal.side,
                                    "entry_price": round(signal.entry_price, 4),
                                    "sl_price": round(sl_price, 4),
                                    "tp_price": round(signal.tp_price, 4),
                                    "hard_stop_price": round(sl_price, 4),
                                    "regime": signal.regime,
                                    "roc_30": round(float(selected["roc_30"]), 6),
                                    "funding_rate": round(float(selected["funding_rate"]), 8),
                                    "spread_pct": round(float(selected["spread_pct"]), 6),
                                    "qty": qty,
                                    "order_id": order.get("orderId"),
                                    "stop_order_id": stop_order_id,
                                }
                            )
                            print(
                                f"  [FILL] {symbol} {signal.side} qty={qty} @ {signal.entry_price} "
                                f"SL={sl_price} orderId={order.get('orderId')}"
                            )
                            if telegram_notifier and getattr(telegram_notifier, "send_message", None):
                                margin_mode = get_margin_type_from_api(client, symbol)
                                telegram_notifier.send_message(
                                    f"📊 <b>Macro 1D: {symbol} {signal.side}</b>\n"
                                    f"開倉模式: {margin_mode} | Entry: {signal.entry_price} | SL: {sl_price} | qty: {qty}\n"
                                    f"ROC30: {selected['roc_30']:+.2%} | Funding: {selected['funding_rate']*100:.3f}%/8h | "
                                    f"Spread: {selected['spread_pct']:.3f}%"
                                )
                        else:
                            decision_text = "下單失敗"
                            print("  [ERR] 市價單未成交")
        else:
            top3_fmt = []
            decision_text = "無有效訊號，續抱"

        total_notional = _get_total_open_notional_usdt(client)
        leverage_now = (total_notional / equity) if equity > 0 else 0.0
        if leverage_now > LEVERAGE_WARN_THRESHOLD:
            warning_msgs.append(f"真實槓桿 {leverage_now:.2f}x > {LEVERAGE_WARN_THRESHOLD:.2f}x")

        prev_equity = float(risk_state.get("last_report_equity", equity) or equity)
        equity_change_pct = ((equity - prev_equity) / prev_equity * 100.0) if prev_equity > 0 else 0.0

        # 持倉健康度（天數 / 累積資費 / MFE/MAE）
        for s in SYMBOLS:
            pos = get_position_info(client, s)
            if not pos:
                continue
            line, warn = _get_position_health_snapshot(client, s, pos)
            health_lines.append(line)
            if warn:
                warning_msgs.append(warn)

        _send_macro_control_report(
            telegram_notifier,
            report_date=_now_taiwan().strftime("%Y-%m-%d"),
            equity=equity,
            equity_change_pct=equity_change_pct,
            top3=top3_fmt,
            decision=decision_text,
            warnings=warning_msgs,
            real_leverage=leverage_now,
            audit_lines=audit_lines,
            health_lines=health_lines,
        )
        risk_state["last_report_equity"] = equity
        risk_state["last_report_date"] = today_utc
        _save_risk_state(risk_state)

        last_scan_date = today_utc

    # 保留原每日總結，避免中斷既有監控習慣
    last_summary_date = _send_daily_summary(client, telegram_notifier, last_summary_date)
    _write_heartbeat(datetime.now(timezone.utc).isoformat())
    mins_left = _minutes_to_next_1d_close()
    open_count = _count_open_positions(client)
    total_notional = _get_total_open_notional_usdt(client)
    lev = (total_notional / equity) if equity > 0 else 0.0
    print(
        f"[1D 決策模式] 距離下一根 1d 收盤: {mins_left} 分 | OpenPositions: {open_count}/{MAX_CONCURRENT} | "
        f"Equity: {equity:.2f} | Leverage: {lev:.2f}x"
    )
    return 0, last_summary_date, last_scan_date


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
    print("Futures 實戰啟動：1D 宏觀組合引擎，每日 UTC 00:05~00:15 (UTC+8 08:05~08:15) 掃描一次")
    print(f"  監控幣種數: {len(SYMBOLS)} | MAX_CONCURRENT: {MAX_CONCURRENT}")
    ensure_log_dir()
    trim_log_lines(LOG_DIR / "paper_out.log", 10000)
    trim_log_lines(LOG_DIR / "paper_err.log", 10000)

    client = get_client()
    for symbol in SYMBOLS:
        position = get_position_info(client, symbol)
        if position:
            print(
                f"  [現有持倉接管] {symbol} {position['side']} 數量={position['positionAmt']} "
                f"開倉價={position['entryPrice']} 未實現盈虧={position['unrealizedProfit']}"
            )
        init_futures_settings(client, symbol, leverage=LEVERAGE, margin_type="ISOLATED", has_position=bool(position))

    telegram_notifier = _get_telegram_notifier()
    consecutive_fail = 0
    last_summary_date = ""
    last_scan_date = ""
    while True:
        try:
            consecutive_fail, last_summary_date, last_scan_date = run_once(
                client, telegram_notifier, last_summary_date, last_scan_date
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
