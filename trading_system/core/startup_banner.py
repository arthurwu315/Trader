"""
V9.1 Startup banner: print STRATEGY_VERSION, VOL_LOW, VOL_HIGH, MODE, GIT_COMMIT.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def get_commit_hash() -> str:
    try:
        import subprocess
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, timeout=5
        )
        return (r.stdout or "").strip()[:12] if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def print_startup_banner(mode: str = None):
    try:
        from config_v9 import STRATEGY_VERSION, VOL_LOW, VOL_HIGH
    except ImportError:
        STRATEGY_VERSION = "V9_REGIME_CORE"
        VOL_LOW = 2.2
        VOL_HIGH = 4.2
    m = mode if mode is not None else os.getenv("V9_LIVE_MODE", "LIVE")
    commit = get_commit_hash()
    msg = (
        f"STRATEGY_VERSION={STRATEGY_VERSION} VOL_LOW={VOL_LOW} VOL_HIGH={VOL_HIGH} "
        f"MODE={m} GIT_COMMIT={commit}"
    )
    print(msg)
    sys.stdout.flush()
