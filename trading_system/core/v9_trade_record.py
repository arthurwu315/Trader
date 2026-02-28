"""
V9.1 Trade record format (PAPER & MICRO-LIVE share same schema).
Extended for Alpha2 CYCLE records: event_type, funding_rate_8h, funding_annualized_pct, signal.
"""
import csv
import json
from pathlib import Path
from datetime import datetime, timezone

# Project root: trading_system/
_PROJECT_ROOT = Path(__file__).resolve().parents[1]

V9_TRADE_RECORD_HEADER = (
    "timestamp,symbol,side,price,qty,regime_vol,reason,fees,slippage_est,"
    "signal_time,order_time,mode,strategy_id"
)
V9_TRADE_RECORD_HEADER_EXTENDED = (
    V9_TRADE_RECORD_HEADER
    + ",event_type,funding_rate_8h,funding_annualized_pct,signal"
)
V9_FUNDING_CARRY_EXTENDED = (
    V9_TRADE_RECORD_HEADER_EXTENDED
    + ",spot_notional,perp_notional,net_notional,net_notional_pct,"
    "rebalance_action,rebalance_attempt,rebalance_success"
)


def ensure_log_dir() -> Path:
    """Ensure logs dir exists. Uses project root (absolute path)."""
    log_dir = _PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def get_v9_records_path() -> Path:
    return ensure_log_dir() / "v9_trade_records.csv"


def get_v9_records_json_path() -> Path:
    return ensure_log_dir() / "v9_trade_records.jsonl"


def _ensure_funding_carry_header(csv_path: Path) -> None:
    """If file exists without funding carry columns, migrate to full Alpha2 header."""
    if not csv_path.exists():
        return
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            first = f.readline().strip()
        if "spot_notional" in first and "rebalance_action" in first:
            return
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        fc_fieldnames = [
            "timestamp", "symbol", "side", "price", "qty", "regime_vol", "reason",
            "fees", "slippage_est", "signal_time", "order_time", "mode", "strategy_id",
            "event_type", "funding_rate_8h", "funding_annualized_pct", "signal",
            "spot_notional", "perp_notional", "net_notional", "net_notional_pct",
            "rebalance_action", "rebalance_attempt", "rebalance_success",
        ]
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            f.write(V9_FUNDING_CARRY_EXTENDED + "\n")
            w = csv.DictWriter(f, fieldnames=fc_fieldnames, extrasaction="ignore")
            for r in rows:
                for k in ("event_type", "funding_rate_8h", "funding_annualized_pct", "signal",
                          "spot_notional", "perp_notional", "net_notional", "net_notional_pct",
                          "rebalance_action", "rebalance_attempt", "rebalance_success"):
                    r.setdefault(k, "")
                w.writerow(r)
    except Exception:
        pass


def _ensure_extended_header(csv_path: Path) -> None:
    """If file exists with old header, migrate to extended header."""
    if not csv_path.exists():
        return
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            first = f.readline().strip()
        if "event_type" in first:
            _ensure_funding_carry_header(csv_path)
            return
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        ext_fieldnames = [
            "timestamp", "symbol", "side", "price", "qty", "regime_vol", "reason",
            "fees", "slippage_est", "signal_time", "order_time", "mode", "strategy_id",
            "event_type", "funding_rate_8h", "funding_annualized_pct", "signal",
            "spot_notional", "perp_notional", "net_notional", "net_notional_pct",
            "rebalance_action", "rebalance_attempt", "rebalance_success",
        ]
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            f.write(V9_FUNDING_CARRY_EXTENDED + "\n")
            w = csv.DictWriter(f, fieldnames=ext_fieldnames, extrasaction="ignore")
            for r in rows:
                for k in ("event_type", "funding_rate_8h", "funding_annualized_pct", "signal",
                          "spot_notional", "perp_notional", "net_notional", "net_notional_pct",
                          "rebalance_action", "rebalance_attempt", "rebalance_success"):
                    r.setdefault(k, "")
                w.writerow(r)
    except Exception:
        pass


def _ensure_header_and_extended(csv_path: Path, use_extended: bool = False) -> None:
    """Ensure file exists with correct header. Migrate if needed for extended."""
    if csv_path.exists():
        if use_extended:
            _ensure_extended_header(csv_path)
        return
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        hdr = V9_TRADE_RECORD_HEADER_EXTENDED if use_extended else V9_TRADE_RECORD_HEADER
        f.write(hdr + "\n")


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
    event_type: str = "",
    funding_rate_8h: str = "",
    funding_annualized_pct: str = "",
    signal: str = "",
):
    """Append one trade record to CSV and JSONL. Extended fields for Alpha2 CYCLE."""
    log_dir = ensure_log_dir()
    csv_path = log_dir / "v9_trade_records.csv"
    json_path = log_dir / "v9_trade_records.jsonl"

    _ensure_header_and_extended(csv_path, use_extended=True)

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
        "event_type": event_type or "",
        "funding_rate_8h": funding_rate_8h or "",
        "funding_annualized_pct": funding_annualized_pct or "",
        "signal": signal or "",
    }

    try:
        with open(csv_path, "a", encoding="utf-8", newline="") as f:
            f.write(
                f"{timestamp},{symbol},{side},{price:.8f},{qty:.8f},{regime_vol:.4f},{reason},"
                f"{fees:.4f},{slippage_est:.4f},{signal_time},{order_time},{mode},{strategy_id},"
                f"{event_type or ''},{funding_rate_8h or ''},{funding_annualized_pct or ''},{signal or ''},"
                f",,,,,,,\n"
            )
    except Exception as e:
        print(f"  [WARN] append_v9_trade_record CSV: {e}")

    try:
        with open(json_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"  [WARN] append_v9_trade_record JSONL: {e}")


def append_funding_carry_cycle(
    timestamp: str,
    symbol: str,
    mode: str,
    funding_rate_8h: float,
    funding_annualized_pct: float,
    signal: bool,
    reason: str,
    strategy_id: str = "FUND_CARRY_V1",
    spot_notional: float = 0.0,
    perp_notional: float = 0.0,
    net_notional: float = 0.0,
    net_notional_pct: float = 0.0,
    rebalance_action: str = "NONE",
    rebalance_attempt: bool = False,
    rebalance_success: bool = True,
) -> None:
    """Append Alpha2 CYCLE record with hedge/rebalance fields."""
    csv_path = get_v9_records_path()
    _ensure_header_and_extended(csv_path, use_extended=True)
    _ensure_funding_carry_header(csv_path)
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        f.write(
            f"{timestamp},{symbol},CYCLE,0,0,{funding_annualized_pct:.4f},{reason},"
            f"0,0,{timestamp},{timestamp},{mode},{strategy_id},"
            f"CYCLE,{funding_rate_8h:.8f},{funding_annualized_pct:.2f},{str(signal).lower()},"
            f"{spot_notional:.4f},{perp_notional:.4f},{net_notional:.4f},{net_notional_pct:.4f},"
            f"{rebalance_action},{str(rebalance_attempt).lower()},{str(rebalance_success).lower()}\n"
        )


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
