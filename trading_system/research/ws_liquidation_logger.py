"""
Manhattan Plan Phase-1: Binance Futures websocket logger.

Collects:
- Global liquidation stream: !forceOrder@arr
- C-group mark price stream (includes funding-related fields)

Stores to SQLite:
- liquidation_events
- mark_price_events

This script is independent from live trading runtime.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import websockets


DEFAULT_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "OPNUSDT",
    "AZTECUSDT", "DOGEUSDT", "1000PEPEUSDT", "ENSOUSDT", "BNBUSDT",
    "ESPUSDT", "INJUSDT", "ZECUSDT", "BCHUSDT", "SIRENUSDT",
    "YGGUSDT", "POWERUSDT", "KITEUSDT", "ETCUSDT", "PIPPINUSDT",
]


@dataclass
class LoggerConfig:
    db_path: Path
    symbols: list[str]
    ws_base_url: str
    ping_interval: int
    reconnect_backoff_sec: int


class SQLiteSink:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS liquidation_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recv_time_utc TEXT NOT NULL,
                event_time_ms INTEGER,
                symbol TEXT,
                side TEXT,
                order_type TEXT,
                time_in_force TEXT,
                qty REAL,
                price REAL,
                avg_price REAL,
                status TEXT,
                trade_time_ms INTEGER,
                raw_json TEXT
            );
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS mark_price_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recv_time_utc TEXT NOT NULL,
                event_time_ms INTEGER,
                symbol TEXT,
                mark_price REAL,
                index_price REAL,
                estimated_settle_price REAL,
                funding_rate REAL,
                next_funding_time_ms INTEGER,
                raw_json TEXT
            );
            """
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_liq_symbol_time ON liquidation_events(symbol, event_time_ms);"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_mark_symbol_time ON mark_price_events(symbol, event_time_ms);"
        )
        self.conn.commit()

    def insert_liquidation(self, payload: dict) -> None:
        o = payload.get("o", {}) if isinstance(payload, dict) else {}
        recv_time = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            """
            INSERT INTO liquidation_events (
                recv_time_utc, event_time_ms, symbol, side, order_type,
                time_in_force, qty, price, avg_price, status, trade_time_ms, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                recv_time,
                payload.get("E"),
                o.get("s"),
                o.get("S"),
                o.get("o"),
                o.get("f"),
                _to_float(o.get("q")),
                _to_float(o.get("p")),
                _to_float(o.get("ap")),
                o.get("X"),
                o.get("T"),
                json.dumps(payload, ensure_ascii=False),
            ),
        )

    def insert_mark_price(self, payload: dict) -> None:
        recv_time = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            """
            INSERT INTO mark_price_events (
                recv_time_utc, event_time_ms, symbol, mark_price, index_price,
                estimated_settle_price, funding_rate, next_funding_time_ms, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                recv_time,
                payload.get("E"),
                payload.get("s"),
                _to_float(payload.get("p")),
                _to_float(payload.get("i")),
                _to_float(payload.get("P")),
                _to_float(payload.get("r")),
                payload.get("T"),
                json.dumps(payload, ensure_ascii=False),
            ),
        )

    def commit(self) -> None:
        self.conn.commit()

    def close(self) -> None:
        try:
            self.conn.commit()
        finally:
            self.conn.close()


def _to_float(v):
    try:
        return float(v)
    except Exception:
        return None


def _build_streams(symbols: list[str]) -> list[str]:
    streams = ["!forceOrder@arr"]
    for s in symbols:
        streams.append(f"{s.lower()}@markPrice@1s")
    return streams


async def _run_stream(cfg: LoggerConfig, sink: SQLiteSink, stop_event: asyncio.Event) -> None:
    streams = _build_streams(cfg.symbols)
    stream_qs = "/".join(streams)
    url = f"{cfg.ws_base_url}/stream?streams={stream_qs}"
    liq_count = 0
    mark_count = 0
    last_commit = time.time()
    last_log = time.time()

    while not stop_event.is_set():
        try:
            async with websockets.connect(url, ping_interval=cfg.ping_interval) as ws:
                print(f"[ws] connected: {url}")
                async for message in ws:
                    if stop_event.is_set():
                        break
                    try:
                        data = json.loads(message)
                    except Exception:
                        continue
                    stream = str(data.get("stream", ""))
                    payload = data.get("data", {})
                    if stream == "!forceOrder@arr":
                        sink.insert_liquidation(payload)
                        liq_count += 1
                    elif stream.endswith("@markPrice@1s"):
                        sink.insert_mark_price(payload)
                        mark_count += 1

                    now = time.time()
                    if now - last_commit >= 2.0:
                        sink.commit()
                        last_commit = now
                    if now - last_log >= 15.0:
                        print(f"[ws] heartbeat liq={liq_count} mark={mark_count}")
                        last_log = now
        except Exception as e:
            print(f"[ws] disconnected: {type(e).__name__}: {e}")
            sink.commit()
            await asyncio.sleep(cfg.reconnect_backoff_sec)


def _parse_args() -> LoggerConfig:
    parser = argparse.ArgumentParser(description="Binance liquidation/funding websocket logger")
    parser.add_argument(
        "--db",
        default="/home/trader/trading_system/research/data/liquidation_feed.db",
        help="SQLite DB path",
    )
    parser.add_argument(
        "--symbols",
        default=",".join(DEFAULT_SYMBOLS),
        help="Comma-separated symbols for markPrice stream",
    )
    parser.add_argument(
        "--ws-base",
        default=os.getenv("BINANCE_WS_BASE", "wss://fstream.binance.com"),
        help="Binance futures websocket base URL",
    )
    parser.add_argument("--ping-interval", type=int, default=20)
    parser.add_argument("--reconnect-backoff", type=int, default=5)
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    return LoggerConfig(
        db_path=Path(args.db),
        symbols=symbols,
        ws_base_url=args.ws_base.rstrip("/"),
        ping_interval=args.ping_interval,
        reconnect_backoff_sec=args.reconnect_backoff,
    )


async def _main_async() -> None:
    cfg = _parse_args()
    sink = SQLiteSink(cfg.db_path)
    stop_event = asyncio.Event()

    loop = asyncio.get_running_loop()

    def _handle_stop(*_):
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handle_stop)
        except NotImplementedError:
            pass

    print(f"[boot] logging {len(cfg.symbols)} symbols to {cfg.db_path}")
    try:
        await _run_stream(cfg, sink, stop_event)
    finally:
        sink.close()
        print("[boot] stopped")


if __name__ == "__main__":
    asyncio.run(_main_async())

