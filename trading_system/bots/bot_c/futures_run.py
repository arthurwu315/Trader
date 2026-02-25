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
import threading
import time
import concurrent.futures as cf
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests

# 即時輸出日誌到 journald（避免 stdout 緩衝延遲）
os.environ.setdefault("PYTHONUNBUFFERED", "1")

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

STRATEGY_VERSION = "v2.0-A+C-DualEngine"
# C Group（Top50 + 流動性過濾）固定 20 幣清單
C_GROUP_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "OPNUSDT",
    "AZTECUSDT", "DOGEUSDT", "1000PEPEUSDT", "ENSOUSDT", "BNBUSDT",
    "ESPUSDT", "INJUSDT", "ZECUSDT", "BCHUSDT", "SIRENUSDT",
    "YGGUSDT", "POWERUSDT", "KITEUSDT", "ETCUSDT", "PIPPINUSDT",
]
MONITOR_SYMBOLS = list(C_GROUP_SYMBOLS)
PRIMARY_SYMBOL = MONITOR_SYMBOLS[0]
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
NOTIONAL_PCT_OF_EQUITY = 0.40
NOTIONAL_REDUCED_PCT = 0.30
DRAWDOWN_REDUCE_NOTIONAL_PCT = 12.0
# v2.0: 策略層基礎倉位（以總淨值百分比）
STRAT_A_BASE_NOTIONAL_PCT = 0.40
STRAT_C_BASE_NOTIONAL_PCT = 0.80
# v2.0: STRAT_C 微型驗證模式（最小名義金額）
TEST_MODE = os.getenv("TEST_MODE", "true").strip().lower() in {"1", "true", "yes", "on"}
TEST_MODE_MIN_NOTIONAL_USDT = float(os.getenv("TEST_MODE_MIN_NOTIONAL_USDT", "10"))
LEVERAGE_WARN_THRESHOLD = 1.5
FUNDING_ALERT_RATE = 0.0005      # 0.05% / 8h
FUNDING_SHORT_SKIP_ANNUAL = 0.20 # 做空年化資費 > 20% 則跳過
SPREAD_ALERT_PCT = 0.15
CIRCUIT_DRAWDOWN_PCT = 25.0
CIRCUIT_COOLDOWN_HOURS = 48
RISK_STATE_FILE = LOG_DIR / "paper_risk_state.json"
ADMIN_CHAT_ID_ENV = os.getenv("ADMIN_CHAT_ID", "").strip()
ALLOWED_CHAT_IDS = {
    v.strip()
    for v in os.getenv("ALLOWED_CHAT_IDS", "").split(",")
    if v.strip()
}
if ADMIN_CHAT_ID_ENV:
    ALLOWED_CHAT_IDS.add(ADMIN_CHAT_ID_ENV)
# 每日總結觸發小時（目前=7 為測試用，驗證通知後請改回 8）
SUMMARY_TRIGGER_HOUR = 8
C_MICRO_STOP_HOURS = 3
C_SCAN_STALE_EXIT_SECONDS = int(1.5 * 3600)
BINANCE_API_TIMEOUT_SEC = 20


def get_client():
    from bots.bot_c.config_c import get_strategy_c_config
    from core.binance_client import BinanceFuturesClient
    cfg = get_strategy_c_config()
    base = os.getenv("BINANCE_BASE_URL", MAINNET_URL if not TESTNET else TESTNET_URL)
    return BinanceFuturesClient(
        base_url=base,
        api_key=cfg.binance_api_key or "dummy",
        api_secret=cfg.binance_api_secret or "dummy",
        timeout=BINANCE_API_TIMEOUT_SEC,
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


def _get_effective_notional_pct(risk_state: dict) -> float:
    dd = float(risk_state.get("latest_drawdown_pct", 0.0) or 0.0)
    if dd >= DRAWDOWN_REDUCE_NOTIONAL_PCT:
        return NOTIONAL_REDUCED_PCT
    return NOTIONAL_PCT_OF_EQUITY


def _get_effective_strategy_notional_pct(
    risk_state: dict,
    base_pct: float,
    signal_mult: float = 1.0,
) -> float:
    dd = float(risk_state.get("latest_drawdown_pct", 0.0) or 0.0)
    if dd >= DRAWDOWN_REDUCE_NOTIONAL_PCT:
        base = min(base_pct, NOTIONAL_REDUCED_PCT)
    else:
        base = base_pct
    mult = max(0.5, min(signal_mult, 1.8))
    return max(0.05, min(base * mult, 1.8))


def _compute_qty_test_mode(entry_price: float) -> float:
    if entry_price <= 0:
        return 0.0
    return round(TEST_MODE_MIN_NOTIONAL_USDT / entry_price, 3)


def get_btc_regime(client) -> str:
    """BTC 大盤濾鏡：BTC > EMA200 => bull，否則 bear。"""
    try:
        merged, _, _ = fetch_merged_row(client, "BTCUSDT")
        if not merged:
            return "unknown"
        close = float(merged.get("close", 0) or 0)
        ema200 = float(merged.get("ema_200", 0) or 0)
        if close <= 0 or ema200 <= 0:
            return "unknown"
        return "bull" if close > ema200 else "bear"
    except Exception:
        return "unknown"


def _compute_rsi(series, period: int = 14):
    delta = series.diff()
    up = delta.clip(lower=0).rolling(period).mean()
    down = (-delta.clip(upper=0)).rolling(period).mean()
    rs = up / down.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))


def _calc_c_signal_from_1h(df, symbol: str, funding_rate: float) -> dict | None:
    """STRAT_C: 1H 極端撿屍 + 高低波動分群 + 清算代理條件。"""
    if df is None or len(df) < 120:
        return None
    row = df.iloc[-2]  # 僅使用已收盤 1h K
    close = float(row["close"])
    open_ = float(row["open"])
    high = float(row["high"])
    low = float(row["low"])
    atr = float(row.get("atr_14", 0) or 0)
    if close <= 0 or atr <= 0:
        return None
    vol = float(row.get("volume", 0) or 0)
    vol50 = float(row.get("vol_sma50", 0) or 0)
    rsi = float(row.get("rsi_14", 0) or 0)
    bb_mid = float(row.get("bb_mid", 0) or 0)
    bb_up = float(row.get("bb_up", 0) or 0)
    bb_low = float(row.get("bb_low", 0) or 0)
    if bb_mid <= 0 or bb_up <= 0 or bb_low <= 0:
        return None

    rng = max(high - low, 1e-9)
    lower_reclaim = (close - low) / rng
    upper_reject = (high - close) / rng
    std = abs(bb_up - bb_mid) / 2.0 if abs(bb_up - bb_mid) > 0 else abs(bb_mid) * 0.005
    bb_up_25 = bb_mid + 2.5 * std
    bb_low_25 = bb_mid - 2.5 * std
    bb_up_30 = bb_mid + 3.0 * std
    bb_low_30 = bb_mid - 3.0 * std

    high_vol = {
        "1000PEPEUSDT", "AZTECUSDT", "ENSOUSDT", "ESPUSDT", "INJUSDT",
        "KITEUSDT", "PIPPINUSDT", "POWERUSDT", "SIRENUSDT", "YGGUSDT",
    }
    is_high_vol = symbol in high_vol
    # Funding filter: 避免逆勢高成本。
    long_ok = funding_rate <= 0.0004
    short_ok = funding_rate >= -0.0004

    # 清算潮代理：爆量+長/上影回收，搭配 funding 極值。
    if vol50 > 0 and vol > 3.0 * vol50 and lower_reclaim > 0.60 and close > open_ and funding_rate <= -0.0004:
        score = (vol / vol50) + abs(funding_rate) * 10000.0 + lower_reclaim
        return {"side": "BUY", "entry_price": close, "sl_price": close - 1.0 * atr, "score": score, "strategy": "C"}
    if vol50 > 0 and vol > 3.0 * vol50 and upper_reject > 0.60 and close < open_ and funding_rate >= 0.0004:
        score = (vol / vol50) + abs(funding_rate) * 10000.0 + upper_reject
        return {"side": "SELL", "entry_price": close, "sl_price": close + 1.0 * atr, "score": score, "strategy": "C"}

    # 常規極端撿屍（高波動組更嚴格）
    if is_high_vol:
        if close < bb_low_30 and rsi < 10 and long_ok:
            return {"side": "BUY", "entry_price": close, "sl_price": close - 1.0 * atr, "score": 1.4, "strategy": "C"}
        if close > bb_up_30 and rsi > 90 and short_ok:
            return {"side": "SELL", "entry_price": close, "sl_price": close + 1.0 * atr, "score": 1.4, "strategy": "C"}
    else:
        if close < bb_low_25 and rsi < 15 and long_ok:
            return {"side": "BUY", "entry_price": close, "sl_price": close - 1.0 * atr, "score": 1.2, "strategy": "C"}
        if close > bb_up_25 and rsi > 85 and short_ok:
            return {"side": "SELL", "entry_price": close, "sl_price": close + 1.0 * atr, "score": 1.2, "strategy": "C"}
    return None


def _fetch_1h_with_indicators(client, symbol: str, limit: int = 320):
    import pandas as pd
    df = fetch_klines(client, symbol, "1h", limit)
    if df is None or len(df) < 120:
        return None
    out = df.copy()
    prev_close = out["close"].shift(1)
    tr = pd.concat(
        [
            out["high"] - out["low"],
            (out["high"] - prev_close).abs(),
            (out["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["atr_14"] = tr.rolling(14).mean()
    out["vol_sma50"] = out["volume"].rolling(50).mean()
    out["bb_mid"] = out["close"].rolling(20).mean()
    bb_std = out["close"].rolling(20).std()
    out["bb_up"] = out["bb_mid"] + 2.0 * bb_std
    out["bb_low"] = out["bb_mid"] - 2.0 * bb_std
    out["rsi_14"] = _compute_rsi(out["close"], 14)
    return out


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
        "📊 【Strategy C】每日實戰總結\n"
        f"📅 日期：{yesterday}\n"
        "-------------------------\n"
        "💰 帳戶狀態\n"
        f"總餘額：{balance:.2f} USDT\n"
        f"昨日盈虧：{daily_pnl:+.2f} USDT ({pnl_pct:+.2f}%)\n"
        f"昨日手續費：{daily_fees:.2f} USDT\n"
        "📈 交易統計\n"
        f"昨日訊號：{count} 筆\n"
        f"累計總筆數：{total_count} (多:{longs} / 空:{shorts})\n"
        "🛡️ 當前持倉\n"
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
    for symbol in MONITOR_SYMBOLS:
        if has_open_position(client, symbol):
            total += 1
    return total


def _refresh_circuit_state(state: dict, equity: float, now_utc: datetime) -> dict:
    if bool(state.get("circuit_permanent_lock", False)):
        state["circuit_active"] = True
        state["circuit_until_utc"] = "9999-12-31T00:00:00+00:00"
        state["latest_drawdown_pct"] = float(state.get("latest_drawdown_pct", 0.0) or 0.0)
        return state
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
        "📊 [1D Macro 實盤報告]\n"
        f"📅 日期: {report_date}\n"
        f"💰 當前淨值: {equity:.2f} USDT ({equity_change_pct:+.2f}%)\n"
        f"🎯 RS 候選名單: {', '.join(top3) if top3 else 'None'}\n"
        f"🛡️ 決策結果: {decision}\n"
        f"🧾 決策審計: {' | '.join(audit_lines) if audit_lines else '無'}\n"
        f"🩺 持倉健康度: {' | '.join(health_lines) if health_lines else '無持倉'}\n"
        f"📐 真實槓桿率: {real_leverage:.2f}x\n"
        f"⚠️ 異常提醒: {warn_txt}\n"
        "🆘 緊急指令: /close_all（30 秒內輸入 /confirm_kill 以執行）"
    )
    try:
        notifier.send_message(msg)
    except Exception as e:
        print(f"  [WARN] Telegram 中控報告發送失敗: {e}")


def _get_exchange_open_symbols(client) -> set[str]:
    out: set[str] = set()
    try:
        positions = client.get_position_risk()
        for p in positions or []:
            amt = float(p.get("positionAmt", 0) or 0)
            symbol = str(p.get("symbol", ""))
            if amt != 0 and symbol:
                out.add(symbol)
    except Exception:
        pass
    return out


def _select_top_monitor_symbols(client, limit: int = 50) -> list[str]:
    """從 USDT-M 永續依 24h 成交量選前 N，排除穩定幣基礎資產。"""
    stable_bases = {"USDT", "USDC", "FDUSD", "BUSD", "TUSD", "USDP", "DAI", "USTC"}
    try:
        info = client.get_exchange_info()
        eligible = {}
        for s in info.get("symbols", []):
            symbol = str(s.get("symbol", ""))
            if s.get("contractType") != "PERPETUAL":
                continue
            if s.get("quoteAsset") != "USDT":
                continue
            if s.get("status") != "TRADING":
                continue
            if str(s.get("baseAsset", "")).upper() in stable_bases:
                continue
            if not symbol.endswith("USDT"):
                continue
            eligible[symbol] = 0.0

        tickers = client._call_with_retry("GET", "/fapi/v1/ticker/24hr", {})
        for row in tickers if isinstance(tickers, list) else []:
            symbol = str(row.get("symbol", ""))
            if symbol in eligible:
                eligible[symbol] = float(row.get("quoteVolume", 0) or 0.0)

        ranked = sorted(eligible.items(), key=lambda kv: kv[1], reverse=True)
        out = [s for s, _ in ranked[:limit] if s]
        return out if out else MONITOR_SYMBOLS
    except Exception as e:
        print(f"  [WARN] 自動篩選前50失敗，改用預設清單: {e}")
        return MONITOR_SYMBOLS


def _get_position_details(client, max_items: int = 6) -> list[str]:
    out: list[str] = []
    try:
        positions = client.get_position_risk()
        for p in positions or []:
            amt = float(p.get("positionAmt", 0) or 0)
            if amt == 0:
                continue
            symbol = str(p.get("symbol", ""))
            side = "BUY" if amt > 0 else "SELL"
            entry = float(p.get("entryPrice", 0) or 0)
            upnl = float(p.get("unrealizedProfit", 0) or 0)
            out.append(f"{symbol}:{side} qty={abs(amt):.4f} entry={entry:.4f} uPnL={upnl:+.2f}")
        out.sort()
    except Exception:
        pass
    return out[:max_items]


def _check_server_time_drift_ms(client) -> int | None:
    try:
        srv = client._call_with_retry("GET", "/fapi/v1/time", {})
        server_ms = int(srv.get("serverTime", 0) or 0)
        local_ms = int(time.time() * 1000)
        return abs(server_ms - local_ms)
    except Exception:
        return None


def _execute_close_all(client) -> tuple[int, float]:
    """核按鈕：取消所有掛單 + 市價平所有倉位。"""
    closed = 0
    for symbol in MONITOR_SYMBOLS:
        try:
            client.cancel_all_orders(symbol)
        except Exception:
            pass
        try:
            positions = client.get_position_risk(symbol=symbol)
            for p in positions or []:
                amt = float(p.get("positionAmt", 0) or 0)
                if amt == 0:
                    continue
                side = "SELL" if amt > 0 else "BUY"
                qty = round(abs(amt), 6)
                if qty <= 0:
                    continue
                client.place_order(
                    {
                        "symbol": symbol,
                        "side": side,
                        "type": "MARKET",
                        "quantity": qty,
                        "reduceOnly": "true",
                    }
                )
                closed += 1
        except Exception as e:
            print(f"  [WARN] close_all {symbol} 失敗: {e}")
    balance = _get_wallet_balance_usdt(client)
    return closed, balance


def _poll_telegram_updates(bot_token: str, offset: int) -> tuple[list[dict], int]:
    """輪詢 Telegram getUpdates。"""
    url = f"https://api.telegram.org/bot{bot_token}/getUpdates"
    try:
        resp = requests.get(url, params={"offset": offset, "timeout": 1}, timeout=5)
        data = resp.json() if resp.status_code == 200 else {}
        rows = data.get("result", []) if isinstance(data, dict) else []
        next_offset = offset
        for u in rows:
            next_offset = max(next_offset, int(u.get("update_id", 0)) + 1)
        return rows, next_offset
    except Exception:
        return [], offset


def _tg_send_plain(bot_token: str, chat_id: str, text: str) -> int | None:
    """純文字送訊息（不帶 parse_mode，避免 Markdown/HTML 解析失敗）。"""
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        resp = requests.post(
            url,
            json={"chat_id": chat_id, "text": text, "disable_web_page_preview": True},
            timeout=8,
        )
        data = resp.json() if resp.status_code == 200 else {}
        if isinstance(data, dict) and data.get("ok") and isinstance(data.get("result"), dict):
            return data["result"].get("message_id")
    except Exception:
        pass
    return None


def _tg_edit_plain(bot_token: str, chat_id: str, message_id: int, text: str) -> bool:
    try:
        url = f"https://api.telegram.org/bot{bot_token}/editMessageText"
        resp = requests.post(
            url,
            json={"chat_id": chat_id, "message_id": message_id, "text": text, "disable_web_page_preview": True},
            timeout=8,
        )
        data = resp.json() if resp.status_code == 200 else {}
        return bool(isinstance(data, dict) and data.get("ok"))
    except Exception:
        return False


def _fetch_merged_row_with_timeout(client, symbol: str, timeout_sec: float = 3.0):
    with cf.ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(fetch_merged_row, client, symbol)
        return fut.result(timeout=timeout_sec)


def _next_reconciliation_time_tw() -> str:
    now_tw = _now_taiwan()
    target = now_tw.replace(hour=8, minute=5, second=0, microsecond=0)
    if now_tw >= target:
        target = target + timedelta(days=1)
    return target.strftime("%Y-%m-%d %H:%M:%S")


def _last_heartbeat_display() -> str:
    """讀取最後一次主循環 heartbeat 時間（顯示用，台灣時間）。"""
    try:
        if not HEARTBEAT_FILE.exists():
            return "N/A"
        raw = HEARTBEAT_FILE.read_text(encoding="utf-8").strip()
        if not raw:
            return "N/A"
        dt = _parse_iso_utc(raw)
        return _format_taiwan(dt) if dt else "N/A"
    except Exception:
        return "N/A"


def _build_status_message(client) -> str:
    equity = _get_total_equity_usdt(client)
    open_syms = sorted(_get_exchange_open_symbols(client))
    pos_details = _get_position_details(client)
    risk_state = _load_risk_state()
    btc_regime = get_btc_regime(client)
    locked = bool(risk_state.get("circuit_permanent_lock", False)) or bool(risk_state.get("circuit_active", False))
    risk_text = "Locked" if locked else "Normal"
    now_str = datetime.now(TZ_TAIWAN).strftime("%Y-%m-%d %H:%M:%S")
    effective_notional_pct = _get_effective_notional_pct(risk_state)
    notional = equity * effective_notional_pct
    c_last_scan = str(risk_state.get("c_last_scan_hour_utc", "N/A"))
    c_last_note = str(risk_state.get("c_last_check_note", "N/A"))
    heartbeat_text = _last_heartbeat_display()
    return (
        "🛰️ [系統狀態看板]\n"
        f"🧠 策略版本: {STRATEGY_VERSION}\n"
        f"🌐 BTC Regime: {btc_regime}\n"
        f"🧩 引擎狀態: A(1D)=ON | C(1H)=ON | TEST_MODE={'ON' if TEST_MODE else 'OFF'}\n"
        f"💰 當前淨值: {equity:.2f} USDT\n"
        f"📌 當前持倉: {open_syms if open_syms else ['None']}\n"
        f"📋 持倉詳情: {pos_details if pos_details else ['None']}\n"
        f"🎯 Monitor count: {len(MONITOR_SYMBOLS)}\n"
        f"💸 單筆下單金額(Notional): {notional:.2f} USDT ({effective_notional_pct*100:.0f}% Equity)\n"
        f"🛡️ 風控狀態: {risk_text}\n"
        f"⚙️ 系統心跳: {heartbeat_text}\n"
        f"⏱️ C 最近進場檢測: {c_last_scan} ({c_last_note})\n"
        f"🕒 更新時間: {now_str} (UTC+8)\n"
        "🔄 資料來源: Binance 即時查詢\n"
        f"🧮 下一次對帳時間: {_next_reconciliation_time_tw()} (UTC+8)\n"
        "💡 輸入 /scan 查看 C Group 20 檔幣種的進場預警與診斷。"
    )


def _build_help_message() -> str:
    return (
        f"🤖 1D Macro Bot 控制中心 ({STRATEGY_VERSION})\n"
        "-------------------------\n"
        "📈 狀態監控\n"
        "/status - 查看淨值、持倉、風控狀態\n"
        "/sync_now - 強制執行帳實對帳\n\n"
        "🔍 市場掃描\n"
        "/scan - 查看 C Group 20 檔進場預警與未達標原因\n\n"
        "🛡️ 安全控制\n"
        "/close_all - 緊急清倉並永久鎖定 (核按鈕)\n"
        "/unlock_trading - 解除熔斷與永久鎖定\n\n"
        "📜 目前參數\n"
        "策略: 1D Donchian (N=80, EMA=200, Trail=2.5)\n"
        "風控: 40% Notional / 2 倉位（回撤>=12%降至30%）\n"
        "權限: 已鎖定白名單管理員"
    )


def _select_rs_candidates(candidates: list[dict], slots: int) -> list[dict]:
    if slots <= 0 or not candidates:
        return []
    longs = [c for c in candidates if c["side"] == "BUY"]
    shorts = [c for c in candidates if c["side"] == "SELL"]
    ranked_longs = sorted(longs, key=lambda x: x["roc_30"], reverse=True)
    ranked_shorts = sorted(shorts, key=lambda x: x["roc_30"])
    long_top = abs(ranked_longs[0]["roc_30"]) if ranked_longs else -1.0
    short_top = abs(ranked_shorts[0]["roc_30"]) if ranked_shorts else -1.0
    if long_top >= short_top and ranked_longs:
        return ranked_longs[:slots]
    if ranked_shorts:
        return ranked_shorts[:slots]
    return []


def _build_scan_message(client) -> str:
    from bots.bot_c.deploy_ready import get_signal_from_row, get_deploy_params

    _refresh_monitor_symbols(client)
    params = get_deploy_params()
    n = int(params.get("macro_n", 55))
    ema_slow = int(params.get("ema_slow_period", 100))
    now_str = datetime.now(TZ_TAIWAN).strftime("%Y-%m-%d %H:%M:%S")
    equity = _get_total_equity_usdt(client)

    reason_store: dict[str, str] = {}
    opportunities: list[tuple[str, float, str]] = []
    breakout_candidates: list[dict] = []

    for symbol in MONITOR_SYMBOLS:
        try:
            merged, _, _ = fetch_merged_row(client, symbol)
            if merged is None:
                reason_store[symbol] = "[資料不足] K線不足"
                continue

            close = float(merged.get("close", 0) or 0)
            if close <= 0:
                reason_store[symbol] = "[資料異常] close<=0"
                continue
            roll_high = merged.get(f"roll_high_{n}")
            roll_low = merged.get(f"roll_low_{n}")
            ema_val = merged.get(f"ema_{ema_slow}")
            if roll_high is None or roll_low is None or ema_val is None:
                reason_store[symbol] = "[過濾中] 指標尚未就緒"
                continue
            roll_high = float(roll_high)
            roll_low = float(roll_low)
            ema_val = float(ema_val)
            dist_long = ((roll_high - close) / close) * 100.0
            dist_short = ((close - roll_low) / close) * 100.0
            near = min(abs(dist_long), abs(dist_short))
            if near < 3.0:
                if abs(dist_long) <= abs(dist_short):
                    opportunities.append((symbol, dist_long, "LONG"))
                else:
                    opportunities.append((symbol, dist_short, "SHORT"))

            signal, _ = get_signal_from_row(merged, params, last_regime=None)
            if signal and signal.should_enter:
                funding_rate = _get_funding_rate(client, symbol)
                spread_pct = _get_spread_pct(client, symbol)
                annual_funding = max(funding_rate, 0.0) * 3.0 * 365.0
                if signal.side == "SELL" and annual_funding > FUNDING_SHORT_SKIP_ANNUAL:
                    reason_store[symbol] = f"[資費過高] 年化 {annual_funding*100:.2f}%"
                    continue
                if spread_pct > SPREAD_ALERT_PCT:
                    reason_store[symbol] = f"[盤整中] Spread {spread_pct:.3f}%"
                    continue
                breakout_candidates.append(
                    {
                        "symbol": symbol,
                        "side": signal.side,
                        "roc_30": float(merged.get("roc_30", 0.0) or 0.0),
                    }
                )
            else:
                if close < ema_val:
                    reason_store[symbol] = "[過濾中] 價格在 EMA 下方"
                elif max(abs(dist_long), abs(dist_short)) > 5.0:
                    reason_store[symbol] = "[盤整中] 距離突破口 > 5%"
                else:
                    reason_store[symbol] = "[盤整中] 尚未觸發突破"
        except Exception as e:
            reason_store[symbol] = f"[掃描錯誤] {type(e).__name__}"

    selected = _select_rs_candidates(breakout_candidates, slots=MAX_CONCURRENT)
    selected_set = {x["symbol"] for x in selected}
    for c in breakout_candidates:
        if c["symbol"] not in selected_set:
            reason_store[c["symbol"]] = "[RS排名後段] 雖突破但未進前2"
        else:
            reason_store[c["symbol"]] = f"[已入選] {c['side']} ROC={c['roc_30']:+.2%}"

    opp_sorted = sorted(opportunities, key=lambda x: abs(x[1]))
    hot_lines = [f"{s}: 距離{d:+.2f}% ({side})" for s, d, side in opp_sorted[:12]]

    # 精簡輸出：先顯示機會與前段診斷，避免超過 4096 字元
    diag_lines = []
    for sym in MONITOR_SYMBOLS:
        if sym in reason_store:
            diag_lines.append(f"{sym}: {reason_store[sym]}")
    diag_lines = diag_lines[:28]

    return (
        "🔍 C Group 監控報告 (20 Symbols)\n"
        f"🕒 掃描時間: {now_str} (UTC+8)\n"
        f"💰 當前資產: {equity:.2f} USDT\n\n"
        "🔥 接近突破 (距離 < 3%)\n"
        f"{chr(10).join(hot_lines) if hot_lines else 'None'}\n\n"
        "💤 觀察中 / 原因診斷\n"
        f"{chr(10).join(diag_lines) if diag_lines else 'None'}"
    )


def _handle_scan_command(notifier, bot_token: str, chat_id: str) -> None:
    # DEBUG 第一時間確認指令已到達
    notifier.send_message("DEBUG: 已接收到掃描指令", parse_mode=None)
    try:
        from bots.bot_c.deploy_ready import get_signal_from_row, get_deploy_params

        cmd_client = get_client()
        _refresh_monitor_symbols(cmd_client)
        params = get_deploy_params()
        n = int(params.get("macro_n", 55))
        ema_slow = int(params.get("ema_slow_period", 100))
        total = len(MONITOR_SYMBOLS)
        equity = _get_total_equity_usdt(cmd_client)
        btc_regime = get_btc_regime(cmd_client)
        blocked_side = "SELL" if btc_regime == "bull" else ("BUY" if btc_regime == "bear" else "NONE")

        progress_id = _tg_send_plain(bot_token, chat_id, f"⏳ 掃描中: 0/{total} ...")
        reason_store: dict[str, str] = {}
        opportunities: list[tuple[str, float, str]] = []
        breakout_candidates: list[dict] = []
        batch_lines: list[str] = []

        for idx, symbol in enumerate(MONITOR_SYMBOLS, start=1):
            try:
                merged, _, _ = _fetch_merged_row_with_timeout(cmd_client, symbol, timeout_sec=3.0)
                if merged is None:
                    reason = "[資料不足] K線不足"
                    reason_store[symbol] = reason
                    batch_lines.append(f"{symbol}: {reason}")
                    continue

                close = float(merged.get("close", 0) or 0)
                if close <= 0:
                    reason = "[資料異常] close<=0"
                    reason_store[symbol] = reason
                    batch_lines.append(f"{symbol}: {reason}")
                    continue

                roll_high = merged.get(f"roll_high_{n}")
                roll_low = merged.get(f"roll_low_{n}")
                ema_val = merged.get(f"ema_{ema_slow}")
                if roll_high is None or roll_low is None or ema_val is None:
                    reason = "[過濾中] 指標尚未就緒"
                    reason_store[symbol] = reason
                    batch_lines.append(f"{symbol}: {reason}")
                    continue

                roll_high = float(roll_high)
                roll_low = float(roll_low)
                ema_val = float(ema_val)
                dist_long = ((roll_high - close) / close) * 100.0
                dist_short = ((close - roll_low) / close) * 100.0
                near = min(abs(dist_long), abs(dist_short))
                if near < 3.0:
                    if abs(dist_long) <= abs(dist_short):
                        opportunities.append((symbol, dist_long, "LONG"))
                    else:
                        opportunities.append((symbol, dist_short, "SHORT"))

                signal, _ = get_signal_from_row(merged, params, last_regime=None)
                if signal and signal.should_enter:
                    if blocked_side != "NONE" and signal.side == blocked_side:
                        if btc_regime == "bull":
                            reason = "[DualGate屏蔽] 大盤看多，空頭訊號已屏蔽"
                        else:
                            reason = "[DualGate屏蔽] 大盤看空，多頭訊號已屏蔽"
                        reason_store[symbol] = reason
                        batch_lines.append(f"{symbol}: {reason}")
                        continue
                    funding_rate = _get_funding_rate(cmd_client, symbol)
                    spread_pct = _get_spread_pct(cmd_client, symbol)
                    annual_funding = max(funding_rate, 0.0) * 3.0 * 365.0
                    if signal.side == "SELL" and annual_funding > FUNDING_SHORT_SKIP_ANNUAL:
                        reason = f"[資費過高] 年化 {annual_funding*100:.2f}%"
                        reason_store[symbol] = reason
                        batch_lines.append(f"{symbol}: {reason}")
                        continue
                    if spread_pct > SPREAD_ALERT_PCT:
                        reason = f"[盤整中] Spread {spread_pct:.3f}%"
                        reason_store[symbol] = reason
                        batch_lines.append(f"{symbol}: {reason}")
                        continue
                    breakout_candidates.append(
                        {"symbol": symbol, "side": signal.side, "roc_30": float(merged.get("roc_30", 0.0) or 0.0)}
                    )
                    batch_lines.append(f"{symbol}: [候選] {signal.side} ROC={float(merged.get('roc_30', 0.0) or 0.0):+.2%}")
                else:
                    if close < ema_val:
                        reason = "[過濾中] 價格在 EMA 下方"
                    elif max(abs(dist_long), abs(dist_short)) > 5.0:
                        reason = "[盤整中] 距離突破口 > 5%"
                    else:
                        reason = "[盤整中] 尚未觸發突破"
                    reason_store[symbol] = reason
                    batch_lines.append(f"{symbol}: {reason}")
            except cf.TimeoutError:
                reason_store[symbol] = "[超時] 單幣掃描 > 3s，已跳過"
                batch_lines.append(f"{symbol}: [超時] >3s")
            except Exception as e:
                reason_store[symbol] = f"[掃描錯誤] {type(e).__name__}"
                batch_lines.append(f"{symbol}: [錯誤] {type(e).__name__}")

            if idx % 10 == 0 or idx == total:
                if progress_id:
                    _tg_edit_plain(bot_token, chat_id, progress_id, f"⏳ 掃描中: {idx}/{total} ...")
                _tg_send_plain(
                    bot_token,
                    chat_id,
                    f"🔎 掃描分段 {max(1, idx-9)}-{idx}/{total}\n" + "\n".join(batch_lines[-10:]),
                )

        selected = _select_rs_candidates(breakout_candidates, slots=MAX_CONCURRENT)
        selected_set = {x["symbol"] for x in selected}
        for c in breakout_candidates:
            if c["symbol"] not in selected_set:
                reason_store[c["symbol"]] = "[RS排名後段] 雖突破但未進前2"
            else:
                reason_store[c["symbol"]] = f"[已入選] {c['side']} ROC={c['roc_30']:+.2%}"

        opp_sorted = sorted(opportunities, key=lambda x: abs(x[1]))
        hot_lines = [f"{s}: 距離{d:+.2f}% ({side})" for s, d, side in opp_sorted[:12]]
        diag_lines = [f"{sym}: {reason_store[sym]}" for sym in MONITOR_SYMBOLS if sym in reason_store][:28]
        now_str = datetime.now(TZ_TAIWAN).strftime("%Y-%m-%d %H:%M:%S")
        summary = (
            "🔍 C Group(20) 宏觀預警報告\n"
            f"🕒 掃描時間: {now_str} (UTC+8)\n"
            f"🌐 BTC Regime: {btc_regime}\n"
            f"🧠 引擎: A(1D breakout) + C(1H sniper)\n"
            f"🚦 方向限制: {'大盤看多，僅允許 LONG' if btc_regime == 'bull' else ('大盤看空，僅允許 SHORT' if btc_regime == 'bear' else 'Regime 未知，暫不屏蔽')}\n"
            f"💰 當前資產: {equity:.2f} USDT\n\n"
            "🔥 接近突破 (距離 < 3%)\n"
            f"{chr(10).join(hot_lines) if hot_lines else 'None'}\n\n"
            "💤 觀察中\n"
            f"{chr(10).join(diag_lines) if diag_lines else 'None'}"
        )
        _tg_send_plain(bot_token, chat_id, summary)
        if progress_id:
            _tg_edit_plain(bot_token, chat_id, progress_id, f"✅ 掃描完成: {total}/{total}")
    except Exception as e:
        notifier.send_message(f"❌ /scan 執行失敗：{e}", parse_mode=None)


def _refresh_monitor_symbols(client) -> None:
    global MONITOR_SYMBOLS, PRIMARY_SYMBOL
    MONITOR_SYMBOLS = list(C_GROUP_SYMBOLS)
    if MONITOR_SYMBOLS:
        PRIMARY_SYMBOL = MONITOR_SYMBOLS[0]


def _ensure_runtime_files() -> None:
    ensure_log_dir()
    # 初始化風險狀態檔，並設定僅擁有者可讀寫
    if not RISK_STATE_FILE.exists():
        _save_risk_state(
            {
                "month_key": _now_taiwan().strftime("%Y-%m"),
                "month_peak_equity": 0.0,
                "latest_drawdown_pct": 0.0,
                "circuit_active": False,
                "circuit_permanent_lock": False,
                "expected_open_symbols": [],
            }
        )
    try:
        os.chmod(RISK_STATE_FILE, 0o600)
    except Exception:
        pass


def _telegram_command_loop():
    """背景命令循環：/close_all 雙重確認。"""
    try:
        notifier = _get_telegram_notifier()
        if not notifier or not getattr(notifier, "enabled", False):
            return
        chat_id = str(notifier.chat_id)
        bot_token = str(notifier.bot_token)
        if not bot_token or not chat_id:
            return
        allowed_ids = set(ALLOWED_CHAT_IDS)
        allowed_ids.add(chat_id)
        cmd_client = get_client()
        # 指令註冊表（本腳本使用輪詢架構，等效於 CommandHandler 註冊）
        command_registry = {
            "/scan": lambda n: _handle_scan_command(n, bot_token, chat_id),
        }
        offset = 0
        while True:
            updates, offset = _poll_telegram_updates(bot_token, offset)
            for u in updates:
                msg = u.get("message", {}) if isinstance(u, dict) else {}
                text = str(msg.get("text", "") or "").strip()
                from_chat = str((msg.get("chat") or {}).get("id", ""))
                if from_chat not in allowed_ids or not text:
                    continue
                now_utc = datetime.now(timezone.utc)
                state = _load_risk_state()

                if text == "/close_all":
                    deadline = (now_utc + timedelta(seconds=30)).isoformat()
                    state["kill_confirm_deadline_utc"] = deadline
                    _save_risk_state(state)
                    notifier.send_message(
                        "⚠️ 收到 /close_all。\n"
                        "請在 30 秒內輸入 /confirm_kill 以執行全平倉與永久熔斷。"
                    )
                elif text == "/confirm_kill":
                    # 強制即時查詢：每次執行核按鈕都重建 client
                    cmd_client = get_client()
                    deadline = _parse_iso_utc(str(state.get("kill_confirm_deadline_utc", "")))
                    if not deadline or now_utc > deadline:
                        notifier.send_message("❌ /confirm_kill 超時，請重新輸入 /close_all。")
                        state.pop("kill_confirm_deadline_utc", None)
                        _save_risk_state(state)
                        continue

                    closed_cnt, balance = _execute_close_all(cmd_client)
                    state["circuit_permanent_lock"] = True
                    state["circuit_active"] = True
                    state["circuit_until_utc"] = "9999-12-31T00:00:00+00:00"
                    state["expected_open_symbols"] = []
                    state.pop("kill_confirm_deadline_utc", None)
                    _save_risk_state(state)
                    notifier.send_message(
                        "🧨 [核按鈕已執行]\n"
                        f"已嘗試平倉筆數: {closed_cnt}\n"
                        f"當前帳戶可用餘額: {balance:.2f} USDT\n"
                        "交易已永久鎖定（circuit_permanent_lock=true）。"
                    )
                elif text == "/unlock_trading":
                    deadline = (now_utc + timedelta(seconds=30)).isoformat()
                    state["unlock_confirm_deadline_utc"] = deadline
                    _save_risk_state(state)
                    notifier.send_message(
                        "⚠️ 收到 /unlock_trading。\n"
                        "確定要解除永久鎖定並恢復自動交易嗎？\n"
                        "請在 30 秒內輸入 /confirm_unlock。"
                    )
                elif text == "/confirm_unlock":
                    deadline = _parse_iso_utc(str(state.get("unlock_confirm_deadline_utc", "")))
                    if not deadline or now_utc > deadline:
                        notifier.send_message("❌ /confirm_unlock 超時，請重新輸入 /unlock_trading。")
                        state.pop("unlock_confirm_deadline_utc", None)
                        _save_risk_state(state)
                        continue
                    state["circuit_permanent_lock"] = False
                    state["circuit_active"] = False
                    state["circuit_until_utc"] = ""
                    state["latest_drawdown_pct"] = 0.0
                    state.pop("unlock_confirm_deadline_utc", None)
                    _save_risk_state(state)
                    notifier.send_message(
                        "✅ [系統已恢復]\n"
                        "交易鎖定已解除，監控中。\n"
                        "下一個決策窗口為 08:05 (UTC+8)。"
                    )
                elif text == "/status":
                    # 強制即時查詢：status 不使用舊 client 狀態
                    cmd_client = get_client()
                    _refresh_monitor_symbols(cmd_client)
                    notifier.send_message(_build_status_message(cmd_client))
                elif text == "/sync_now":
                    # 強制即時查詢：sync_now 重新建立 client 並覆蓋本地狀態
                    cmd_client = get_client()
                    _refresh_monitor_symbols(cmd_client)
                    ex = sorted(_get_exchange_open_symbols(cmd_client))
                    equity = _get_total_equity_usdt(cmd_client)
                    now_str = datetime.now(TZ_TAIWAN).strftime("%Y-%m-%d %H:%M:%S")
                    pos_details = _get_position_details(cmd_client)
                    effective_notional_pct = _get_effective_notional_pct(state)
                    notional = equity * effective_notional_pct
                    state["expected_open_symbols"] = ex
                    _save_risk_state(state)
                    notifier.send_message(
                        "🔄 [手動對帳完成]\n"
                        f"交易所持倉已同步: {ex if ex else ['None']}\n"
                        f"💰 Equity: {equity:.2f} USDT\n"
                        f"📋 持倉詳情: {pos_details if pos_details else ['None']}\n"
                        f"💸 Notional: {notional:.2f} USDT ({effective_notional_pct*100:.0f}% Equity)\n"
                        f"🕒 更新時間: {now_str} (UTC+8)"
                    )
                elif text == "/help":
                    notifier.send_message(_build_help_message())
                elif text == "/scan":
                    command_registry["/scan"](notifier)
            time.sleep(2)
    except Exception as e:
        print(f"  [WARN] Telegram 指令循環異常: {e}")


def run_once(
    client,
    telegram_notifier=None,
    last_summary_date: str = "",
    last_scan_date: str = "",
    force_startup_scan: bool = False,
):
    from bots.bot_c.deploy_ready import get_signal_from_row, get_deploy_params, HARD_STOP_POSITION_PCT

    now_utc = datetime.now(timezone.utc)
    today_utc = now_utc.date().isoformat()
    in_decision_window = _in_daily_decision_window(now_utc)
    params = get_deploy_params()
    regime_map = _load_regime_map()
    risk_state = _load_risk_state()
    equity = _get_total_equity_usdt(client)
    risk_state = _refresh_circuit_state(risk_state, equity, now_utc)
    effective_notional_pct = _get_effective_notional_pct(risk_state)
    _save_risk_state(risk_state)
    # 伺服器時間同步檢查（>1s 警告）
    drift_ms = _check_server_time_drift_ms(client)
    if drift_ms is not None and drift_ms > 1000:
        warning = f"伺服器時間偏移 {drift_ms}ms > 1000ms，請檢查 NTP"
        print(f"  [WARN] {warning}")
        if telegram_notifier and getattr(telegram_notifier, "send_message", None):
            telegram_notifier.send_message(f"⚠️ [時間同步警告]\n{warning}")

    candidates: list[dict] = []
    candidate_symbols: list[str] = []
    warning_msgs: list[str] = []
    audit_lines: list[str] = []
    health_lines: list[str] = []
    decision_text = "續抱"

    a_scan_due = (in_decision_window and last_scan_date != today_utc) or force_startup_scan
    if a_scan_due:
        _refresh_monitor_symbols(client)
        btc_regime = get_btc_regime(client)
        blocked_side = "SELL" if btc_regime == "bull" else ("BUY" if btc_regime == "bear" else "NONE")
        # 每日對帳：本地預期持倉 vs 交易所真實持倉
        exchange_open = _get_exchange_open_symbols(client)
        local_open = set(regime_map.get("_open_symbols", []))
        if local_open != exchange_open:
            risk_state["expected_open_symbols"] = sorted(exchange_open)
            regime_map["_open_symbols"] = sorted(exchange_open)
            _save_regime_map(regime_map)
            _save_risk_state(risk_state)
            if telegram_notifier and getattr(telegram_notifier, "send_message", None):
                telegram_notifier.send_message(
                    "🚨 [同步異常]\n"
                    f"本地持倉: {sorted(local_open)}\n"
                    f"交易所持倉: {sorted(exchange_open)}\n"
                    "已強制校準本地狀態，請檢查！"
                )

        for symbol in MONITOR_SYMBOLS:
            merged, _, _ = fetch_merged_row(client, symbol)
            if merged is None:
                continue
            last_regime = regime_map.get(symbol)
            signal, current_regime = get_signal_from_row(merged, params, last_regime=last_regime)
            regime_map[symbol] = current_regime
            if signal and signal.should_enter:
                if blocked_side != "NONE" and signal.side == blocked_side:
                    if btc_regime == "bull":
                        audit_lines.append(f"{symbol}: DualGate 屏蔽空頭（BTC Bull）")
                    else:
                        audit_lines.append(f"{symbol}: DualGate 屏蔽多頭（BTC Bear）")
                    continue
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

        longs = [c for c in candidates if c["signal"].side == "BUY"]
        shorts = [c for c in candidates if c["signal"].side == "SELL"]
        top3 = sorted(candidates, key=lambda x: abs(x["roc_30"]), reverse=True)[:3]
        top3_fmt = [f"{x['symbol']}({x['roc_30']:+.2%})" for x in top3]
        open_count = _count_open_positions(client)
        available_slots = max(0, MAX_CONCURRENT - open_count)
        selected_candidates: list[dict] = []
        if available_slots > 0:
            ranked_longs = sorted(longs, key=lambda x: x["roc_30"], reverse=True)
            ranked_shorts = sorted(shorts, key=lambda x: x["roc_30"])
            long_top = abs(ranked_longs[0]["roc_30"]) if ranked_longs else -1.0
            short_top = abs(ranked_shorts[0]["roc_30"]) if ranked_shorts else -1.0
            # 同批次以同方向優先：做多取最強前N；做空取最弱前N
            if long_top >= short_top and ranked_longs:
                selected_candidates = ranked_longs[:available_slots]
            elif ranked_shorts:
                selected_candidates = ranked_shorts[:available_slots]
        print(
            f"[Macro Scan] 掃描日期: {today_utc} | BTC Regime: {btc_regime} | "
            f"候選訊號: {candidate_symbols or ['None']} | "
            f"RS 仲裁選擇: {[c['symbol'] for c in selected_candidates] if selected_candidates else ['None']}"
        )

        selected_symbols = {c["symbol"] for c in selected_candidates}
        if selected_candidates:
            for i, c in enumerate(sorted(candidates, key=lambda x: abs(x["roc_30"]), reverse=True), start=1):
                if c["symbol"] not in selected_symbols:
                    audit_lines.append(f"{c['symbol']}: RS 排名不足 (位居第 {i})")

        if selected_candidates:
            if risk_state.get("circuit_active", False):
                decision_text = "Circuit Breaker 啟動，暫停新倉"
                print("  [RISK] Circuit Breaker 啟動，跳過新進場")
                if telegram_notifier and getattr(telegram_notifier, "send_message", None):
                    telegram_notifier.send_message(
                        "🚨 緊急止損警告\n"
                        f"當月峰值回撤已達 {risk_state.get('latest_drawdown_pct', 0.0):.2f}%\n"
                        f"新倉暫停至: {risk_state.get('circuit_until_utc', 'N/A')}"
                    )
            else:
                if open_count >= MAX_CONCURRENT:
                    decision_text = "倉位已滿 Skip"
                    print(f"  [SKIP] 已達 MAX_CONCURRENT={MAX_CONCURRENT}")
                else:
                    filled_symbols: list[str] = []
                    for selected in selected_candidates:
                        if _count_open_positions(client) >= MAX_CONCURRENT:
                            break
                        if has_open_position(client, selected["symbol"]):
                            audit_lines.append(f"{selected['symbol']}: 已有持倉跳過")
                            continue
                        signal = selected["signal"]
                        row = selected["row"]
                        symbol = selected["symbol"]
                        notional_pct_a = _get_effective_strategy_notional_pct(
                            risk_state,
                            STRAT_A_BASE_NOTIONAL_PCT,
                            signal_mult=1.0,
                        )
                        qty = _compute_qty_by_notional(equity, signal.entry_price, notional_pct_a)
                        if TEST_MODE:
                            qty = _compute_qty_test_mode(signal.entry_price)
                        if qty <= 0:
                            audit_lines.append(f"{symbol}: 淨值不足 qty=0")
                            continue
                        order = place_market_order(client, symbol, signal.side, qty)
                        if not order:
                            audit_lines.append(f"{symbol}: 下單失敗")
                            continue

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
                        filled_symbols.append(symbol)
                        print(
                            f"  [FILL] {symbol} {signal.side} qty={qty} @ {signal.entry_price} "
                            f"SL={sl_price} orderId={order.get('orderId')}"
                        )
                        if telegram_notifier and getattr(telegram_notifier, "send_message", None):
                            margin_mode = get_margin_type_from_api(client, symbol)
                            telegram_notifier.send_message(
                                f"📊 Macro 1D: {symbol} {signal.side}\n"
                                f"開倉模式: {margin_mode} | Entry: {signal.entry_price} | SL: {sl_price} | qty: {qty}\n"
                                f"ROC30: {selected['roc_30']:+.2%} | Funding: {selected['funding_rate']*100:.3f}%/8h | "
                                f"Spread: {selected['spread_pct']:.3f}% | A-Notional: {notional_pct_a*100:.0f}%"
                            )
                    if filled_symbols:
                        decision_text = f"進場 {filled_symbols}"
                    else:
                        decision_text = "候選皆被跳過"
                    synced_open = sorted(_get_exchange_open_symbols(client))
                    risk_state["expected_open_symbols"] = synced_open
                    regime_map["_open_symbols"] = synced_open
                    _save_regime_map(regime_map)
                    _save_risk_state(risk_state)
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
        for s in MONITOR_SYMBOLS:
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

    # STRAT_C 1H 引擎：每小時只掃一次（同一 event loop 內）
    now_hour_key = now_utc.strftime("%Y-%m-%d %H")
    last_c_hour = str(risk_state.get("c_last_scan_hour_utc", ""))
    c_scan_due = force_startup_scan or (now_hour_key != last_c_hour)
    if c_scan_due:
        # 掃描一開始就寫入，避免中途出錯時看不到「曾嘗試掃描」的痕跡
        risk_state["c_last_scan_hour_utc"] = f"{now_hour_key} (start)"
        risk_state["c_last_scan_ts_utc"] = now_utc.isoformat()
        risk_state["c_last_check_note"] = "Scan Started"
        _save_risk_state(risk_state)
        btc_regime = get_btc_regime(client)
        btc_regime_note = "Bull Regime" if btc_regime == "bull" else ("Bear Regime" if btc_regime == "bear" else "Unknown Regime")
        blocked_side = "SELL" if btc_regime == "bull" else ("BUY" if btc_regime == "bear" else "NONE")
        open_count = _count_open_positions(client)
        c_entry_count = 0
        if not risk_state.get("circuit_active", False) and open_count < MAX_CONCURRENT:
            for symbol in MONITOR_SYMBOLS:
                if _count_open_positions(client) >= MAX_CONCURRENT:
                    break
                if has_open_position(client, symbol):
                    continue
                funding_rate = _get_funding_rate(client, symbol)
                h1 = _fetch_1h_with_indicators(client, symbol, limit=320)
                sig = _calc_c_signal_from_1h(h1, symbol, funding_rate)
                if not sig:
                    continue
                if blocked_side != "NONE" and sig["side"] == blocked_side:
                    continue
                signal_mult = float(sig.get("score", 1.0) or 1.0)
                # score -> 0.5x~1.8x
                if signal_mult >= 8:
                    signal_mult = 1.8
                elif signal_mult >= 5:
                    signal_mult = 1.6
                elif signal_mult >= 3:
                    signal_mult = 1.2
                else:
                    signal_mult = 0.5
                notional_pct_c = _get_effective_strategy_notional_pct(
                    risk_state,
                    STRAT_C_BASE_NOTIONAL_PCT,
                    signal_mult=signal_mult,
                )
                qty = _compute_qty_by_notional(equity, float(sig["entry_price"]), notional_pct_c)
                if TEST_MODE:
                    qty = _compute_qty_test_mode(float(sig["entry_price"]))
                if qty <= 0:
                    continue
                order = place_market_order(client, symbol, str(sig["side"]), qty)
                if not order:
                    continue
                c_entry_count += 1
                stop_order = place_stop_market_close(client, symbol, str(sig["side"]), float(sig["sl_price"]))
                stop_order_id = stop_order.get("orderId") if stop_order else None
                c_meta = risk_state.get("c_open_meta", {})
                if not isinstance(c_meta, dict):
                    c_meta = {}
                c_meta[symbol] = {
                    "entry_time_utc": datetime.now(timezone.utc).isoformat(),
                    "entry_price": float(sig["entry_price"]),
                    "side": str(sig["side"]),
                    "strategy": "C",
                }
                risk_state["c_open_meta"] = c_meta
                _save_risk_state(risk_state)
                if telegram_notifier and getattr(telegram_notifier, "send_message", None):
                    telegram_notifier.send_message(
                        f"🎯 STRAT_C 進場\n"
                        f"{symbol} {sig['side']} qty={qty}\n"
                        f"Entry={float(sig['entry_price']):.4f} | SL={float(sig['sl_price']):.4f}\n"
                        f"Funding={funding_rate*100:.3f}%/8h | C-Notional={notional_pct_c*100:.0f}%\n"
                        f"TEST_MODE={'ON' if TEST_MODE else 'OFF'}"
                    )
            if c_entry_count > 0:
                risk_state["c_last_check_note"] = f"{btc_regime_note} - Entry Triggered ({c_entry_count})"
            else:
                risk_state["c_last_check_note"] = f"{btc_regime_note} - Skipped"
        elif risk_state.get("circuit_active", False):
            risk_state["c_last_check_note"] = f"{btc_regime_note} - Circuit Breaker Active"
        else:
            risk_state["c_last_check_note"] = f"{btc_regime_note} - Max Concurrent Reached"

        # C 微停損：持倉 3 小時仍未脫離成本區，強制平倉
        c_meta = risk_state.get("c_open_meta", {})
        if isinstance(c_meta, dict):
            for symbol, meta in list(c_meta.items()):
                entry_ts = _parse_iso_utc(str(meta.get("entry_time_utc", "")))
                if not entry_ts:
                    c_meta.pop(symbol, None)
                    continue
                hold_hours = (now_utc - entry_ts).total_seconds() / 3600.0
                if hold_hours < C_MICRO_STOP_HOURS:
                    continue
                pos = get_position_info(client, symbol)
                if not pos:
                    c_meta.pop(symbol, None)
                    continue
                upnl = float(pos.get("unrealizedProfit", 0) or 0)
                if upnl > 0:
                    continue
                side = "SELL" if float(pos.get("positionAmt", 0) or 0) > 0 else "BUY"
                qty = round(abs(float(pos.get("positionAmt", 0) or 0)), 6)
                if qty <= 0:
                    c_meta.pop(symbol, None)
                    continue
                try:
                    client.place_order(
                        {
                            "symbol": symbol,
                            "side": side,
                            "type": "MARKET",
                            "quantity": qty,
                            "reduceOnly": "true",
                        }
                    )
                    c_meta.pop(symbol, None)
                    if telegram_notifier and getattr(telegram_notifier, "send_message", None):
                        telegram_notifier.send_message(
                            f"🛑 STRAT_C 微停損\n{symbol} 持倉超過 {C_MICRO_STOP_HOURS}h 且未獲利，已平倉。"
                        )
                except Exception:
                    pass
        risk_state["c_open_meta"] = c_meta
        risk_state["c_last_scan_hour_utc"] = now_hour_key
        risk_state["c_last_scan_ts_utc"] = now_utc.isoformat()
        _save_risk_state(risk_state)

    # 自我電擊：C 掃描停滯超過 1.5 小時，主動退出讓 systemd 拉起
    c_last_scan_ts = _parse_iso_utc(str(risk_state.get("c_last_scan_ts_utc", "")))
    if c_last_scan_ts is not None:
        stale_sec = (now_utc - c_last_scan_ts).total_seconds()
        if stale_sec > C_SCAN_STALE_EXIT_SECONDS:
            msg = (
                f"[FATAL] C scan stale {int(stale_sec)}s > "
                f"{C_SCAN_STALE_EXIT_SECONDS}s, exiting for systemd restart"
            )
            print(msg)
            raise SystemExit(1)

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
    _ensure_runtime_files()
    trim_log_lines(LOG_DIR / "paper_out.log", 10000)
    trim_log_lines(LOG_DIR / "paper_err.log", 10000)

    client = get_client()
    _refresh_monitor_symbols(client)
    print(f"  監控幣種數: {len(MONITOR_SYMBOLS)} | MAX_CONCURRENT: {MAX_CONCURRENT}")
    for symbol in MONITOR_SYMBOLS:
        position = get_position_info(client, symbol)
        if position:
            print(
                f"  [現有持倉接管] {symbol} {position['side']} 數量={position['positionAmt']} "
                f"開倉價={position['entryPrice']} 未實現盈虧={position['unrealizedProfit']}"
            )
        init_futures_settings(client, symbol, leverage=LEVERAGE, margin_type="ISOLATED", has_position=bool(position))

    telegram_notifier = _get_telegram_notifier()
    # 背景指令循環：支援 /close_all -> /confirm_kill 雙重確認
    cmd_thread = threading.Thread(target=_telegram_command_loop, daemon=True)
    cmd_thread.start()
    consecutive_fail = 0
    last_summary_date = ""
    last_scan_date = ""
    first_loop = True
    while True:
        try:
            consecutive_fail, last_summary_date, last_scan_date = run_once(
                client,
                telegram_notifier,
                last_summary_date,
                last_scan_date,
                force_startup_scan=first_loop,
            )
            first_loop = False
            if consecutive_fail >= CONSECUTIVE_FAIL_THRESHOLD:
                send_disconnect_alert()
                consecutive_fail = 0
        except Exception as e:
            first_loop = False
            consecutive_fail += 1
            sys.stderr.write(f"[futures_run] 本輪失敗: {e}\n")
            if consecutive_fail >= CONSECUTIVE_FAIL_THRESHOLD:
                send_disconnect_alert()
                consecutive_fail = 0
        time.sleep(LOOP_SLEEP_SEC)


if __name__ == "__main__":
    main()
