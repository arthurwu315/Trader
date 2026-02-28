"""
Alpha Burst B1 - PAPER / MICRO-LIVE entry point.
Independent of V9. Does NOT modify V9 core.
Env: ALPHA_BURST_MODE=PAPER|MICRO-LIVE
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

os.environ.setdefault("SKIP_CONFIG_VALIDATION", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bots.bot_alpha_burst.config import STRATEGY_ID
from bots.bot_alpha_burst.ops import check_burst_dd_kill


def main():
    mode = os.getenv("ALPHA_BURST_MODE", "PAPER").strip().upper()
    mode = "MICRO-LIVE" if mode == "MICRO-LIVE" else "PAPER"
    print(f"Alpha Burst {STRATEGY_ID} {mode} mode")

    from dotenv import load_dotenv
    from bots.bot_c.config_c import get_strategy_c_config
    from core.binance_client import BinanceFuturesClient
    from tests.run_alpha_burst_backtest import run_backtest

    load_dotenv(dotenv_path=ROOT / ".env", override=True)
    cfg = get_strategy_c_config()
    client = BinanceFuturesClient(
        api_key=cfg.binance_api_key or "dummy",
        api_secret=cfg.binance_api_secret or "dummy",
        base_url=os.getenv("BINANCE_DATA_URL", "https://fapi.binance.com"),
    )
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=365 * 3)
    burst_equity = 10000.0

    if mode == "MICRO-LIVE":
        # MICRO-LIVE: check DD kill before run; no real orders in MVP
        if check_burst_dd_kill(burst_equity):
            from core.v9_trade_record import append_v9_trade_record
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            append_v9_trade_record(
                timestamp=ts,
                symbol="ALPHA_BURST",
                side="KILL",
                price=0.0,
                qty=0.0,
                regime_vol=0.0,
                reason="BURST_DD_LIMIT",
                fees=0.0,
                slippage_est=0.0,
                signal_time=ts,
                order_time=ts,
                mode=mode,
                strategy_id=STRATEGY_ID,
                event_type="KILL",
            )
            print("ALPHA_BURST KILL: burst equity DD > 25%")
            sys.exit(1)

    trades = run_backtest(client, start_dt, end_dt, burst_equity=burst_equity, clear_trades_first=True)
    print(f"Cycle done: {len(trades)} trades. Check logs/alpha_burst_b1_trades.csv")
    print("Run: python3 -m tests.run_alpha_burst_report")


if __name__ == "__main__":
    main()
