"""
輕量級戰報腳本：以交易所 API 為準（持倉浮動盈虧、已實現+資金費），杜絕幻影獲利。
執行：cd /home/trader/trading_system && python3 -m bots.bot_c.check_performance
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    _env_root = ROOT / ".env"
    if _env_root.exists():
        load_dotenv(dotenv_path=_env_root)
except Exception:
    pass

LOG_DIR = ROOT / "logs"
PAPER_SIGNALS_FILE = LOG_DIR / "paper_signals.json"
TRADE_HISTORY_CSV = LOG_DIR / "trade_history.csv"
HEARTBEAT_FILE = LOG_DIR / "paper_last_heartbeat.txt"
SYMBOL = "BNBUSDT"
FEE_TAKER_PCT = 0.04   # 0.04%，與淨利說明一致
# 實盤切換日：僅統計此日之後的已實現盈虧，避免測試/手動平倉干擾
MAINNET_SWITCH_DATE = "2026-02-18"


def get_client():
    from bots.bot_c.config_c import get_strategy_c_config
    from core.binance_client import BinanceFuturesClient
    cfg = get_strategy_c_config()
    base = os.getenv("BINANCE_BASE_URL", "https://testnet.binancefuture.com")
    return BinanceFuturesClient(
        base_url=base,
        api_key=cfg.binance_api_key or "dummy",
        api_secret=cfg.binance_api_secret or "dummy",
    )


def load_signals():
    if not PAPER_SIGNALS_FILE.exists():
        return []
    try:
        with open(PAPER_SIGNALS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def count_reset_trades() -> int:
    """實盤重置後筆數：從 trade_history.csv 資料行數計算（不含 header）。"""
    if not TRADE_HISTORY_CSV.exists():
        return 0
    try:
        with open(TRADE_HISTORY_CSV, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if not lines:
            return 0
        # 第一行若為 header 則不計
        if lines[0].lower().startswith("entry_time"):
            return max(0, len(lines) - 1)
        return len(lines)
    except Exception:
        return 0


def api_wallet_balance(client) -> tuple[float, float]:
    """從 futures account 取得 USDT 錢包總餘額與可用保證金。回傳 (total_wallet, available)。"""
    try:
        acc = client.get_account()
        if not acc:
            return 0.0, 0.0
        # 頂層欄位（部分 API 回傳）
        total = acc.get("totalWalletBalance")
        available = acc.get("availableBalance")
        if total is not None and available is not None:
            return float(total or 0), float(available or 0)
        # 從 assets 找 USDT
        for a in acc.get("assets") or []:
            if (a.get("asset") or "").strip().upper() == "USDT":
                w = float(a.get("walletBalance") or 0)
                av = float(a.get("availableBalance") or 0)
                return w, av
        return 0.0, 0.0
    except Exception:
        return 0.0, 0.0


def api_position_pnl(client, symbol: str) -> tuple[float, str, float | None]:
    """從 API 讀取持倉未實現盈虧（真實數據）。回傳 (unrealized_usdt, margin_type, entry_price)。"""
    try:
        positions = client.get_position_risk(symbol=symbol)
        for p in positions or []:
            amt = float(p.get("positionAmt", 0) or 0)
            if amt == 0:
                continue
            up = float(p.get("unrealizedProfit", 0) or 0)
            mt = (p.get("marginType") or "N/A").upper()
            ep = float(p.get("entryPrice", 0) or 0)
            return up, "逐倉" if mt == "ISOLATED" else "全倉", ep
    except Exception:
        pass
    return 0.0, "N/A", None


def api_realized_and_funding(
    client, symbol: str, limit: int = 500, since_ts_ms: int | None = None
) -> tuple[float, float]:
    """從 API 讀取已實現盈虧與資金費。若 since_ts_ms 有值則只統計該時間之後的紀錄（實盤切換後）。"""
    try:
        items = client.get_income_history(symbol=symbol, limit=limit)
        realized = 0.0
        funding = 0.0
        for x in items or []:
            if x.get("asset") != "USDT":
                continue
            if since_ts_ms is not None:
                t_ms = int(x.get("time", 0) or 0)
                if t_ms < since_ts_ms:
                    continue
            inc = float(x.get("income", 0) or 0)
            t = x.get("incomeType", "")
            if t == "REALIZED_PNL":
                realized += inc
            elif t == "FUNDING_FEE":
                funding += inc
        return realized, funding
    except Exception:
        pass
    return 0.0, 0.0


def main():
    signals = load_signals()
    total = len(signals)
    longs = sum(1 for s in signals if (s.get("side") or "").upper() == "BUY")
    shorts = sum(1 for s in signals if (s.get("side") or "").upper() == "SELL")
    reset_count = count_reset_trades()

    print("========== 戰報 (API 真實數據) ==========")
    print(f"進場總筆數 (訊號檔): {total}  多單: {longs}  空單: {shorts}")
    print(f"實盤重置後筆數 (帳本): {reset_count}")

    try:
        client = get_client()
    except Exception as e:
        print(f"浮動盈虧 / 已實現: 無法連接 API ({e})")
        client = None

    if client:
        total_wallet, available_balance = api_wallet_balance(client)
        print(f"\n💰 帳戶即時餘額")
        print(f"   錢包總餘額 (totalWalletBalance): {total_wallet:.2f} USDT")
        print(f"   可用保證金 (availableBalance): {available_balance:.2f} USDT")

        unrealized_usdt, margin_type, entry_price = api_position_pnl(client, SYMBOL)
        # 僅統計實盤切換日之後的已實現，避免測試/手動平倉干擾
        try:
            from datetime import datetime, timezone
            since_dt = datetime.strptime(MAINNET_SWITCH_DATE, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            since_ts_ms = int(since_dt.timestamp() * 1000)
        except Exception:
            since_ts_ms = None
        realized_pnl, funding_fee = api_realized_and_funding(
            client, SYMBOL, limit=500, since_ts_ms=since_ts_ms
        )
        print(f"\n持倉浮動盈虧 (交易所): {unrealized_usdt:+.2f} USDT  保證金模式: {margin_type}")
        label_since = f" (切換實盤後 since {MAINNET_SWITCH_DATE})" if since_ts_ms else ""
        print(f"已實現盈虧 (交易所){label_since}: {realized_pnl:+.2f} USDT")
        print(f"已實現資金費 (交易所){label_since}: {funding_fee:+.2f} USDT")
        net_realized = realized_pnl + funding_fee
        print(f"真實淨利 (已實現+資金費){label_since}: {net_realized:+.2f} USDT")
    else:
        print("浮動盈虧 / 已實現: 略過（無 API）")

    if HEARTBEAT_FILE.exists():
        try:
            with open(HEARTBEAT_FILE, "r", encoding="utf-8") as f:
                line = (f.read() or "").strip()
            print(f"最後 Heartbeat: {line or '無時間戳'}")
        except Exception:
            print("最後 Heartbeat: 讀取失敗")
    else:
        print("最後 Heartbeat: 無記錄")

    print("==========================================")


if __name__ == "__main__":
    main()
