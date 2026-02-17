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
FEE_TAKER_PCT = 0.04   # 0.04%，與淨利說明一致


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


def api_realized_and_funding(client, symbol: str, limit: int = 500) -> tuple[float, float]:
    """從 API 讀取已實現盈虧與資金費（真實數據）。回傳 (realized_pnl_sum, funding_fee_sum)。"""
    try:
        items = client.get_income_history(symbol=symbol, limit=limit)
        realized = 0.0
        funding = 0.0
        for x in items or []:
            if x.get("asset") != "USDT":
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

    print("========== 戰報 (API 真實數據) ==========")
    print(f"進場總筆數: {total}  多單: {longs}  空單: {shorts}")

    try:
        client = get_client()
    except Exception as e:
        print(f"浮動盈虧 / 已實現: 無法連接 API ({e})")
        client = None

    if client:
        unrealized_usdt, margin_type, entry_price = api_position_pnl(client, SYMBOL)
        realized_pnl, funding_fee = api_realized_and_funding(client, SYMBOL)
        print(f"持倉浮動盈虧 (交易所): {unrealized_usdt:+.2f} USDT  保證金模式: {margin_type}")
        # 真實淨利 = 已實現盈虧 + 已實現資金費（交易所已扣手續費）
        net_realized = realized_pnl + funding_fee
        print(f"已實現盈虧 (交易所): {realized_pnl:+.2f} USDT")
        print(f"已實現資金費 (交易所): {funding_fee:+.2f} USDT")
        print(f"真實淨利 (已實現+資金費): {net_realized:+.2f} USDT")
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
