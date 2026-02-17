"""
匯出近期交易摘要：讀取 trade_history.csv 或 paper_signals.json，打印表格方便複製到 Excel。
執行：cd /home/trader/trading_system && python3 -m bots.bot_c.export_trades [--rows 20] [--source csv|json]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "logs"
CSV_PATH = LOG_DIR / "trade_history.csv"
JSON_PATH = LOG_DIR / "paper_signals.json"


def export_from_csv(rows: int = 30) -> None:
    if not CSV_PATH.exists():
        print("trade_history.csv 不存在")
        return
    with open(CSV_PATH, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        data = list(reader)
    data = data[-rows:] if len(data) > rows else data
    if not data:
        print("無交易記錄")
        return
    headers = list(data[0].keys())
    col_widths = [max(len(str(h)), max(len(str(r.get(h, ""))) for r in data)) for h in headers]
    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    print(sep)
    print("|" + "|".join(f" {h:<{col_widths[i]}} " for i, h in enumerate(headers)) + "|")
    print(sep)
    for r in data:
        print("|" + "|".join(f" {str(r.get(h, '')):<{col_widths[i]}} " for i, h in enumerate(headers)) + "|")
    print(sep)
    print(f"共 {len(data)} 筆（最近 {rows} 筆內）")


def export_from_json(rows: int = 30) -> None:
    if not JSON_PATH.exists():
        print("paper_signals.json 不存在")
        return
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        data = []
    data = data[-rows:] if len(data) > rows else data
    if not data:
        print("無訊號記錄")
        return
    headers = ["time_utc", "side", "qty", "entry_price", "sl_price", "tp_price", "regime"]
    col_widths = [12, 6, 8, 12, 10, 10, 6]
    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    print(sep)
    print("|" + "|".join(f" {h:<{col_widths[i]}} " for i, h in enumerate(headers)) + "|")
    print(sep)
    for r in data:
        row = [str(r.get(h, ""))[:col_widths[i]] for i, h in enumerate(headers)]
        print("|" + "|".join(f" {row[i]:<{col_widths[i]}} " for i in range(len(headers))) + "|")
    print(sep)
    print(f"共 {len(data)} 筆（最近 {rows} 筆內）")


def main():
    ap = argparse.ArgumentParser(description="匯出近期交易摘要")
    ap.add_argument("--rows", type=int, default=20, help="顯示筆數")
    ap.add_argument("--source", choices=["csv", "json"], default="csv", help="資料來源")
    args = ap.parse_args()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if args.source == "csv":
        export_from_csv(rows=args.rows)
    else:
        export_from_json(rows=args.rows)


if __name__ == "__main__":
    main()
