"""
Alpha Burst B1 diagnostic breakdown.
Reads logs/alpha_burst_b1_trades.csv, enriches with regime_vol/trend/MFE/MAE from 1H/4H data.
Output: tests/reports/alpha_burst_b1_diagnose.md, tests/reports/alpha_burst_b1_diagnose_artifacts/*.csv

Run: python3 -m tests.run_alpha_burst_b1_diagnose
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("SKIP_CONFIG_VALIDATION", "1")
os.environ["BACKTEST_OFFLINE"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bots.bot_alpha_burst.config import (
    UNIVERSE,
    EMA_TREND_PERIOD,
    ATR_PERIOD,
    ATR_MA_PERIOD,
    VOL_EXPANSION_THRESHOLD,
)
from core.v9_trade_record import get_burst_trades_path, read_burst_trades
from backtest_utils import fetch_klines_df


def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["atr"] = tr.rolling(ATR_PERIOD).mean()
    df["atr_ma"] = df["atr"].rolling(ATR_MA_PERIOD).mean()
    df["ema200"] = df["close"].ewm(span=EMA_TREND_PERIOD, adjust=False).mean()
    return df


def _regime_vol_at(ts: pd.Timestamp, df_1h: pd.DataFrame) -> float:
    df = df_1h[df_1h["timestamp"] <= ts].tail(ATR_PERIOD + 5)
    if df.empty or len(df) < ATR_PERIOD or pd.isna(df.iloc[-1].get("atr")):
        return float("nan")
    row = df.iloc[-1]
    atr = float(row["atr"])
    close = float(row["close"])
    if close <= 0:
        return float("nan")
    return atr / close * 100


def _trend_at(ts: pd.Timestamp, df_4h: pd.DataFrame) -> str:
    df = df_4h[df_4h["timestamp"] <= ts].tail(1)
    if df.empty or pd.isna(df.iloc[-1].get("ema200")):
        return "unknown"
    row = df.iloc[-1]
    if float(row["close"]) > float(row["ema200"]):
        return "long"
    return "short"


def _compute_mfe_mae(
    df_1h: pd.DataFrame,
    entry_ts: pd.Timestamp,
    holding_bars: int,
    side: str,
    entry_price: float,
    stop_price: float,
    initial_risk_per_unit: float,
) -> tuple[float, float, bool]:
    """
    Returns (mfe_r, mae_r, mae_gt1_then_positive).
    """
    mask = df_1h["timestamp"] >= entry_ts
    subset = df_1h.loc[mask].head(holding_bars + 2)
    if subset.empty or initial_risk_per_unit <= 0:
        return float("nan"), float("nan"), False

    mfe_r = float("-inf")
    mae_r = float("-inf")
    mae_exceeded_1 = False

    for _, row in subset.iterrows():
        high = float(row["high"])
        low = float(row["low"])
        if side == "BUY":
            fav = (high - entry_price) / initial_risk_per_unit
            adv = (entry_price - low) / initial_risk_per_unit
        else:
            fav = (entry_price - low) / initial_risk_per_unit
            adv = (high - entry_price) / initial_risk_per_unit
        mfe_r = max(mfe_r, fav)
        mae_r = max(mae_r, adv)
        if adv > 1.0:
            mae_exceeded_1 = True

    mfe_r = mfe_r if mfe_r > float("-inf") else 0.0
    mae_r = mae_r if mae_r > float("-inf") else 0.0
    return mfe_r, mae_r, mae_exceeded_1


def _bucket_regime_vol(vol: float) -> str:
    if pd.isna(vol):
        return "UNKNOWN"
    if vol < 2.0:
        return "LOW"
    if vol < 4.0:
        return "MID"
    return "HIGH"


def _bucket_holding_bars(hb: int) -> str:
    if hb <= 3:
        return "1-3"
    if hb <= 8:
        return "4-8"
    if hb <= 20:
        return "9-20"
    return "21+"


def _permutation_e_r(rs: list[float], n_perm: int = 200, seed: int = 42) -> float:
    rs = [r for r in rs if not (r != r)]  # drop nan
    if len(rs) < 10:
        return 1.0
    obs = np.mean(rs)
    rng = np.random.default_rng(seed)
    perms = []
    for _ in range(n_perm):
        perm = rng.permutation(rs)
        perms.append(np.mean(perm))
    perms = np.array(perms)
    p = np.mean(perms >= obs) if obs >= 0 else np.mean(perms <= obs)
    return float(p)


def _segment_stats(trades: list[dict], key: str) -> dict:
    from collections import defaultdict
    buckets = defaultdict(list)
    for t in trades:
        b = t.get(key, "UNKNOWN")
        buckets[b].append(t["R_multiple"])
    out = []
    for b, rs in sorted(buckets.items()):
        rs_clean = [r for r in rs if r == r]
        if not rs_clean:
            continue
        wins = [r for r in rs_clean if r > 0]
        losses = [r for r in rs_clean if r <= 0]
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        pf = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 1.0)
        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = np.mean(losses) if losses else 0.0
        out.append({
            "bucket": b,
            "trade_count": len(rs_clean),
            "E[R]": np.mean(rs_clean),
            "PF": pf,
            "WinRate": len(wins) / len(rs_clean) * 100,
            "AvgWin_R": avg_win,
            "AvgLoss_R": avg_loss,
        })
    return out


def main():
    from dotenv import load_dotenv
    from bots.bot_c.config_c import get_strategy_c_config
    from core.binance_client import BinanceFuturesClient

    load_dotenv(dotenv_path=ROOT / ".env", override=True)
    config = get_strategy_c_config()
    client = BinanceFuturesClient(
        api_key=config.binance_api_key or "dummy",
        api_secret=config.binance_api_secret or "dummy",
        base_url=os.getenv("BINANCE_DATA_URL", "https://fapi.binance.com"),
    )

    burst_path = get_burst_trades_path()
    if not burst_path.exists():
        print(f"No trades file: {burst_path}")
        sys.exit(1)

    trades = read_burst_trades(burst_path)
    if not trades:
        print("No trades to diagnose")
        sys.exit(1)

    # Restrict to 2022-2024 for cache compatibility and validation period
    t_start = pd.Timestamp("2022-01-01", tz="UTC")
    t_end = pd.Timestamp("2024-12-31 23:59:59", tz="UTC")
    trades = [t for t in trades if t_start <= pd.Timestamp(t["timestamp"], tz="UTC") <= t_end]
    if not trades:
        print("No trades in 2022-2024")
        sys.exit(1)

    start_dt = datetime(2022, 1, 1, tzinfo=timezone.utc)
    end_dt = datetime(2024, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

    data_1h = {}
    data_4h = {}
    for sym in UNIVERSE:
        df1 = fetch_klines_df(client, sym, "1h", start_dt, end_dt)
        df4 = fetch_klines_df(client, sym, "4h", start_dt, end_dt)
        if not df1.empty and len(df1) >= ATR_PERIOD + EMA_TREND_PERIOD:
            data_1h[sym] = _add_indicators(df1)
        if not df4.empty and len(df4) >= EMA_TREND_PERIOD:
            data_4h[sym] = _add_indicators(df4)

    # Enrich trades
    for t in trades:
        ts = pd.Timestamp(t["timestamp"], tz="UTC")
        sym = t["symbol"]
        side = t["side"]
        entry = float(t["entry_price"])
        stop = float(t["stop_price"])
        risk_per_unit = abs(entry - stop)
        hb = int(t["holding_bars"])

        t["regime_vol"] = float("nan")
        t["regime_vol_bucket"] = "UNKNOWN"
        t["trend"] = "unknown"
        t["vol_expansion_flag"] = True
        t["holding_bars_bucket"] = _bucket_holding_bars(hb)

        if sym in data_1h:
            vol = _regime_vol_at(ts, data_1h[sym])
            t["regime_vol"] = vol
            t["regime_vol_bucket"] = _bucket_regime_vol(vol)
        if sym in data_4h:
            t["trend"] = _trend_at(ts, data_4h[sym])
        if sym in data_1h and risk_per_unit > 0:
            mfe, mae, mae1pos = _compute_mfe_mae(
                data_1h[sym], ts, hb, side, entry, stop, risk_per_unit
            )
            t["mfe_r"] = mfe
            t["mae_r"] = mae
            t["mae_gt1_then_positive"] = mae1pos and t["R_multiple"] > 0

    # Output dir
    report_dir = ROOT / "tests" / "reports"
    art_dir = report_dir / "alpha_burst_b1_diagnose_artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)

    rs = [t["R_multiple"] for t in trades]
    rs_clean = [r for r in rs if r == r]

    # 1) R distribution
    quantiles = {}
    for p in [1, 5, 10, 50, 90, 95, 99]:
        quantiles[f"p{p}"] = np.percentile(rs_clean, p) if rs_clean else float("nan")

    sorted_by_r = sorted(trades, key=lambda x: x["R_multiple"], reverse=True)
    top20 = sorted_by_r[:20]
    bot20 = sorted_by_r[-20:]

    # Export top/bottom
    def _row(t):
        et = pd.Timestamp(t["timestamp"], tz="UTC") + pd.Timedelta(hours=int(t["holding_bars"]))
        return {
            "symbol": t["symbol"],
            "entry_time": t["timestamp"],
            "exit_time": et.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "regime_vol": t.get("regime_vol", float("nan")),
            "vol_expansion_flag": t.get("vol_expansion_flag", True),
            "holding_bars": t["holding_bars"],
            "R": t["R_multiple"],
        }

    pd.DataFrame([_row(t) for t in top20]).to_csv(art_dir / "top20_r.csv", index=False)
    pd.DataFrame([_row(t) for t in bot20]).to_csv(art_dir / "bottom20_r.csv", index=False)

    # 2) Segment buckets
    seg_regime = _segment_stats(trades, "regime_vol_bucket")
    seg_trend = _segment_stats(trades, "trend")
    seg_vol_exp = _segment_stats(trades, "vol_expansion_flag")
    seg_holding = _segment_stats(trades, "holding_bars_bucket")

    pd.DataFrame(seg_regime).to_csv(art_dir / "segment_regime_vol.csv", index=False)
    pd.DataFrame(seg_trend).to_csv(art_dir / "segment_trend.csv", index=False)
    pd.DataFrame(seg_vol_exp).to_csv(art_dir / "segment_vol_expansion.csv", index=False)
    pd.DataFrame(seg_holding).to_csv(art_dir / "segment_holding_bars.csv", index=False)

    # 3) MFE/MAE
    has_mfe = [t for t in trades if "mfe_r" in t and t["mfe_r"] == t["mfe_r"]]
    avg_mfe = np.mean([t["mfe_r"] for t in has_mfe]) if has_mfe else float("nan")
    avg_mae = np.mean([t["mae_r"] for t in has_mfe]) if has_mfe else float("nan")
    mae1_then_pos = [t for t in has_mfe if t.get("mae_gt1_then_positive")]
    pct_mae1_then_pos = len(mae1_then_pos) / len(has_mfe) * 100 if has_mfe else float("nan")

    mfe_mae_rows = [
        {"trade_idx": i, "symbol": t["symbol"], "entry_time": t["timestamp"], "R": t["R_multiple"],
         "mfe_r": t.get("mfe_r", float("nan")), "mae_r": t.get("mae_r", float("nan")),
         "mae_gt1_then_positive": t.get("mae_gt1_then_positive", False)}
        for i, t in enumerate(trades)
    ]
    pd.DataFrame(mfe_mae_rows).to_csv(art_dir / "mfe_mae.csv", index=False)

    # 4) Permutation on promising buckets
    key_by_seg = {"regime_vol": "regime_vol_bucket", "trend": "trend", "holding_bars": "holding_bars_bucket"}
    perm_results = []
    for seg, name in [
        (seg_regime, "regime_vol"),
        (seg_trend, "trend"),
        (seg_holding, "holding_bars"),
    ]:
        tkey = key_by_seg[name]
        for row in seg:
            bucket = row["bucket"]
            er = row["E[R]"]
            n = row["trade_count"]
            if n >= 15 and er > 0:
                bucket_trades = [t for t in trades if t.get(tkey) == bucket]
                brs = [t["R_multiple"] for t in bucket_trades]
                pval = _permutation_e_r(brs, n_perm=200)
                perm_results.append({"segment": name, "bucket": str(bucket), "trade_count": n, "E[R]": er, "perm_p": pval})

    pd.DataFrame(perm_results).to_csv(art_dir / "permutation_promising.csv", index=False)

    # Build report
    lines = [
        "# Alpha Burst B1 Diagnostic Report",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"Trade count: {len(trades)}",
        "",
        "---",
        "",
        "## 1. R Distribution",
        "",
        "### Quantiles",
        "| Quantile | Value |",
        "|----------|-------|",
    ]
    for k, v in quantiles.items():
        lines.append(f"| {k} | {v:.4f} |")
    lines.extend([
        "",
        "### Top 20 positive R",
        "See `alpha_burst_b1_diagnose_artifacts/top20_r.csv`",
        "",
        "### Bottom 20 negative R",
        "See `alpha_burst_b1_diagnose_artifacts/bottom20_r.csv`",
        "",
        "---",
        "",
        "## 2. Segment Buckets",
        "",
        "### By regime_vol (LOW<2%, MID 2-4%, HIGH>=4%)",
    ])
    lines.append("| bucket | trade_count | E[R] | PF | WinRate | AvgWin_R | AvgLoss_R |")
    lines.append("|--------|-------------|------|-----|---------|----------|-----------|")
    for row in seg_regime:
        lines.append(f"| {row['bucket']} | {row['trade_count']} | {row['E[R]']:.4f} | {row['PF']:.3f} | {row['WinRate']:.1f}% | {row['AvgWin_R']:.4f} | {row['AvgLoss_R']:.4f} |")

    lines.extend([
        "",
        "### By trend (4H close vs EMA200)",
        "| bucket | trade_count | E[R] | PF | WinRate | AvgWin_R | AvgLoss_R |",
        "|--------|-------------|------|-----|---------|----------|-----------|",
    ])
    for row in seg_trend:
        lines.append(f"| {row['bucket']} | {row['trade_count']} | {row['E[R]']:.4f} | {row['PF']:.3f} | {row['WinRate']:.1f}% | {row['AvgWin_R']:.4f} | {row['AvgLoss_R']:.4f} |")

    lines.extend([
        "",
        "### By vol_expansion_flag",
        "| bucket | trade_count | E[R] | PF | WinRate | AvgWin_R | AvgLoss_R |",
        "|--------|-------------|------|-----|---------|----------|-----------|",
    ])
    for row in seg_vol_exp:
        lines.append(f"| {str(row['bucket'])} | {row['trade_count']} | {row['E[R]']:.4f} | {row['PF']:.3f} | {row['WinRate']:.1f}% | {row['AvgWin_R']:.4f} | {row['AvgLoss_R']:.4f} |")

    lines.extend([
        "",
        "### By holding_bars",
        "| bucket | trade_count | E[R] | PF | WinRate | AvgWin_R | AvgLoss_R |",
        "|--------|-------------|------|-----|---------|----------|-----------|",
    ])
    for row in seg_holding:
        lines.append(f"| {row['bucket']} | {row['trade_count']} | {row['E[R]']:.4f} | {row['PF']:.3f} | {row['WinRate']:.1f}% | {row['AvgWin_R']:.4f} | {row['AvgLoss_R']:.4f} |")

    lines.extend([
        "",
        "---",
        "",
        "## 3. Fake Breakout (MFE/MAE)",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| avg MFE_R | {avg_mfe:.4f} |",
        f"| avg MAE_R | {avg_mae:.4f} |",
        f"| pct MAE>1R then turned positive | {pct_mae1_then_pos:.1f}% |",
        "",
        "Artifacts: `mfe_mae.csv`",
        "",
        "---",
        "",
        "## 4. Permutation on Promising Buckets (200 runs)",
        "",
    ])
    if perm_results:
        lines.append("| segment | bucket | trade_count | E[R] | perm_p |")
        lines.append("|---------|--------|-------------|------|--------|")
        for r in perm_results:
            lines.append(f"| {r['segment']} | {r['bucket']} | {r['trade_count']} | {r['E[R]']:.4f} | {r['perm_p']:.4f} |")
    else:
        lines.append("No bucket with E[R]>0 and n>=15.")

    report_path = report_dir / "alpha_burst_b1_diagnose.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report: {report_path}")
    print(f"Artifacts: {art_dir}")


if __name__ == "__main__":
    main()
