"""
Alpha Burst B2 grid (<=27 combinations).
Run: python3 -m tests.run_alpha_burst_b2_grid
Output: tests/reports/alpha_burst_b2_artifacts/b2_grid_results.csv
"""
from __future__ import annotations

import csv
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

os.environ.setdefault("SKIP_CONFIG_VALIDATION", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_alpha_burst_b2_backtest import run_backtest


def main():
    from dotenv import load_dotenv
    from bots.bot_c.config_c import get_strategy_c_config
    from core.binance_client import BinanceFuturesClient

    load_dotenv(dotenv_path=ROOT / ".env", override=True)
    cfg = get_strategy_c_config()
    client = BinanceFuturesClient(
        api_key=cfg.binance_api_key or "dummy",
        api_secret=cfg.binance_api_secret or "dummy",
        base_url=os.getenv("BINANCE_DATA_URL", "https://fapi.binance.com"),
    )
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=365 * 3)

    comp_opts = [20, 30, 40]
    exp_opts = [1.05, 1.1, 1.2]
    range_opts = [30, 40, 50]
    n_max = 27

    art_dir = Path(__file__).resolve().parent / "reports" / "alpha_burst_b2_artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)
    grid_csv = art_dir / "b2_grid_results.csv"

    rows = []
    idx = 0
    for cp in comp_opts:
        for et in exp_opts:
            for rl in range_opts:
                if idx >= n_max:
                    break
                trades = run_backtest(
                    client, start_dt, end_dt,
                    burst_equity=10000.0,
                    clear_trades_first=False,
                    write_trades=False,
                    param_overrides={
                        "compression_atr_pct": cp,
                        "expansion_threshold": et,
                        "compression_range_lookback": rl,
                    },
                )
                import numpy as np
                rs = [t["R_multiple"] for t in trades]
                wins = [r for r in rs if r > 0]
                m = {
                    "trade_count": len(trades),
                    "E_R": np.mean(rs) if rs else 0,
                    "WinRate": len(wins) / len(rs) * 100 if rs else 0,
                }
                rows.append({
                    "compression_atr_pct": cp,
                    "expansion_threshold": et,
                    "compression_range_lookback": rl,
                    "trade_count": m["trade_count"],
                    "E_R": m["E_R"],
                    "WinRate": m["WinRate"],
                })
                idx += 1
                print(f"  [{idx}/27] cp={cp} et={et} rl={rl} n={m['trade_count']} E[R]={m['E_R']:.4f}")
            if idx >= n_max:
                break
        if idx >= n_max:
            break

    with open(grid_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "compression_atr_pct", "expansion_threshold", "compression_range_lookback",
            "trade_count", "E_R", "WinRate",
        ])
        w.writeheader()
        w.writerows(rows)
    print(f"Grid written to {grid_csv}")


if __name__ == "__main__":
    main()
