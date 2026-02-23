#!/usr/bin/env python3
"""
Minimal Telegram health check sender.

Usage:
  python3 tests/test_telegram_health.py
"""

from __future__ import annotations

import os
import sys

import requests


def main() -> int:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        print("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID in environment.")
        return 1

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": "System Restarted and Healthy",
        "disable_web_page_preview": True,
    }
    try:
        resp = requests.post(url, json=payload, timeout=10)
        ok = resp.status_code == 200
        print(f"status_code={resp.status_code}")
        print(resp.text[:500])
        return 0 if ok else 2
    except Exception as exc:
        print(f"request_failed={exc}")
        return 3


if __name__ == "__main__":
    sys.exit(main())
