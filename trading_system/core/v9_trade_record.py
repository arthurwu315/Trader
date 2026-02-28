"""
V9.1 Trade record format (PAPER & MICRO-LIVE share same schema).
"""
import json
from pathlib import Path
from datetime import datetime, timezone

V9_TRADE_RECORD_HEADER = (
    "timestamp,symbol,side,price,qty,regime_vol,reason,fees,slippage_est,"
    "signal_time,order_time,mode,strategy_id"
)


def ensure_log_dir():
    log_dir = Path(__file__).resolve().parents[1] / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def get_v9_records_path() -> Path:
    return ensure_log_dir() / "v9_trade_records.csv"


def get_v9_records_json_path() -> Path:
    return ensure_log_dir() / "v9_trade_records.jsonl"


def append_v9_trade_record(
    timestamp: str,
    symbol: str,
    side: str,
    price: float,
    qty: float,
    regime_vol: float,
    reason: str,
    fees: float = 0.0,
    slippage_est: float = 0.0,
    signal_time: str = "",
    order_time: str = "",
    mode: str = "PAPER",
    strategy_id: str = "V9_REGIME_CORE",
):
    """Append one trade record to CSV and JSONL."""
    log_dir = ensure_log_dir()
    csv_path = log_dir / "v9_trade_records.csv"
    json_path = log_dir / "v9_trade_records.jsonl"

    row = {
        "timestamp": timestamp,
        "symbol": symbol,
        "side": side,
        "price": price,
        "qty": qty,
        "regime_vol": regime_vol,
        "reason": reason,
        "fees": fees,
        "slippage_est": slippage_est,
        "signal_time": signal_time,
        "order_time": order_time,
        "mode": mode,
        "strategy_id": strategy_id,
    }

    write_header = not csv_path.exists()
    try:
        with open(csv_path, "a", encoding="utf-8", newline="") as f:
            if write_header:
                f.write(V9_TRADE_RECORD_HEADER + "\n")
            f.write(
                f"{timestamp},{symbol},{side},{price:.8f},{qty:.8f},{regime_vol:.4f},{reason},"
                f"{fees:.4f},{slippage_est:.4f},{signal_time},{order_time},{mode},{strategy_id}\n"
            )
    except Exception as e:
        print(f"  [WARN] append_v9_trade_record CSV: {e}")

    try:
        with open(json_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"  [WARN] append_v9_trade_record JSONL: {e}")


def read_v9_records(csv_path: Path = None):
    """Read V9 trade records from CSV. Returns list of dicts."""
    if csv_path is None:
        csv_path = get_v9_records_path()
    if not csv_path.exists():
        return []
    import csv
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                row["price"] = float(row.get("price", 0))
                row["qty"] = float(row.get("qty", 0))
                row["regime_vol"] = float(row.get("regime_vol", 0))
                row["fees"] = float(row.get("fees", 0))
                row["slippage_est"] = float(row.get("slippage_est", 0))
            except (ValueError, KeyError):
                pass
            row.setdefault("strategy_id", "V9_REGIME_CORE")
            rows.append(row)
    return rows
