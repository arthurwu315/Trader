"""
Alpha Burst B1 Ops.
- Burst DD > 25% => KILL
- Reconcile dry-run (log only)
"""
from __future__ import annotations

import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
BURST_STATE_PATH = _PROJECT_ROOT / "logs" / "alpha_burst_state.json"
BURST_MAX_DD_PCT = 0.25


def _read_state() -> dict:
    import json
    if not BURST_STATE_PATH.exists():
        return {}
    try:
        with open(BURST_STATE_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def _write_state(state: dict) -> None:
    import json
    BURST_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BURST_STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def check_burst_dd_kill(current_equity: float) -> bool:
    """
    Check if burst drawdown exceeds 25%. If so, return True (caller should KILL).
    Updates peak in state file.
    """
    state = _read_state()
    peak = state.get("equity_peak")
    if peak is None or peak <= 0:
        state["equity_peak"] = current_equity
        _write_state(state)
        return False
    if current_equity > peak:
        state["equity_peak"] = current_equity
        _write_state(state)
        return False
    dd = (peak - current_equity) / peak
    return dd > BURST_MAX_DD_PCT


def reconcile_dry_run(expected: list[dict], actual: list[dict], log_path: Path | None = None) -> None:
    """
    Reconcile expected vs actual trades. Dry-run: log only, no side effects.
    expected/actual: list of {symbol, side, entry_price, qty, ...}
    """
    import logging
    logger = logging.getLogger("alpha_burst.ops")
    if log_path:
        h = logging.FileHandler(log_path)
        logger.addHandler(h)
    n_exp = len(expected)
    n_act = len(actual)
    if n_exp != n_act:
        logger.warning(f"Reconcile DRY-RUN: count mismatch expected={n_exp} actual={n_act}")
    # Simple length check; full field-by-field reconciliation can be extended
    logger.info(f"Reconcile DRY-RUN: expected={n_exp} actual={n_act}")
