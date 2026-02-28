"""
Alpha2 Funding Carry - state persistence.
Cooldown (per symbol) and rebalance fail tracking.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOGS = ROOT / "logs"
COOLDOWN_PATH = LOGS / "alpha2_cooldown.json"
STATE_PATH = LOGS / "alpha2_state.json"


def _ensure_logs():
    LOGS.mkdir(parents=True, exist_ok=True)


def load_cooldown() -> dict[str, str]:
    """{symbol: "ISO timestamp until"}"""
    _ensure_logs()
    if not COOLDOWN_PATH.exists():
        return {}
    try:
        with open(COOLDOWN_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_cooldown(data: dict[str, str]) -> None:
    _ensure_logs()
    with open(COOLDOWN_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=0)


def set_cooldown(symbol: str, hours: int) -> None:
    until = datetime.now(timezone.utc)
    from datetime import timedelta
    until = (until + timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
    d = load_cooldown()
    d[symbol] = until
    save_cooldown(d)


def is_in_cooldown(symbol: str) -> bool:
    d = load_cooldown()
    until_s = d.get(symbol, "")
    if not until_s:
        return False
    try:
        until = datetime.fromisoformat(until_s.replace("Z", "+00:00"))
        return datetime.now(timezone.utc) < until
    except Exception:
        return False


def clear_cooldown(symbol: str) -> None:
    d = load_cooldown()
    d.pop(symbol, None)
    save_cooldown(d)


def load_state() -> dict:
    _ensure_logs()
    if not STATE_PATH.exists():
        return {"rebalance_fail_count": 0, "consecutive_out_of_threshold": 0}
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"rebalance_fail_count": 0, "consecutive_out_of_threshold": 0}


def save_state(data: dict) -> None:
    _ensure_logs()
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=0)


def inc_rebalance_fail() -> None:
    s = load_state()
    s["rebalance_fail_count"] = s.get("rebalance_fail_count", 0) + 1
    save_state(s)


def inc_consecutive_out() -> None:
    s = load_state()
    s["consecutive_out_of_threshold"] = s.get("consecutive_out_of_threshold", 0) + 1
    save_state(s)


def reset_consecutive_out() -> None:
    s = load_state()
    s["consecutive_out_of_threshold"] = 0
    save_state(s)


def should_hard_stop(rebalance_fail_threshold: int = 3) -> bool:
    s = load_state()
    return (
        s.get("rebalance_fail_count", 0) >= rebalance_fail_threshold
        or s.get("consecutive_out_of_threshold", 0) >= rebalance_fail_threshold
    )
