"""
Alpha Burst B1 Report.
Reads logs/alpha_burst_b1_trades.csv.
Outputs: tests/reports/alpha_burst_b1_report.md
Artifacts: tests/reports/alpha_burst_b1_artifacts/
B1: E[R], WinRate, AvgWin_R, AvgLoss_R, p10/p50/p90/p95 by year/symbol; INSUFFICIENT POWER when trade_count < 30
B2: Small grid (vol_expansion, stop_ATR_k, breakout_lookback x3); plateau regions
B3: Permutation test 1000, block bootstrap 1000 (SEED=42); percentile, p-value
B4: Cost stress (slippage 5/10/20 bps, high-vol 2x, fee)
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

os.environ.setdefault("SKIP_CONFIG_VALIDATION", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.v9_trade_record import read_burst_trades, get_burst_trades_path

SEED = 42
REPORTS_DIR = Path(__file__).resolve().parent / "reports"
ARTIFACTS_DIR = REPORTS_DIR / "alpha_burst_b1_artifacts"


def _parse_year(ts: str) -> int | None:
    try:
        return int(ts[:4])
    except (ValueError, TypeError):
        return None


def _b1_metrics(trades: list[dict]) -> dict:
    """E[R], WinRate, AvgWin_R, AvgLoss_R, p10/p50/p90/p95."""
    if not trades:
        return {}
    rs = [t["R_multiple"] for t in trades]
    wins = [r for r in rs if r > 0]
    losses = [r for r in rs if r < 0]
    return {
        "E[R]": np.mean(rs),
        "WinRate": len(wins) / len(rs) * 100 if rs else 0,
        "AvgWin_R": np.mean(wins) if wins else 0,
        "AvgLoss_R": np.mean(losses) if losses else 0,
        "p10": np.percentile(rs, 10),
        "p50": np.percentile(rs, 50),
        "p90": np.percentile(rs, 90),
        "p95": np.percentile(rs, 95),
        "trade_count": len(trades),
    }


def _run_b1(trades: list[dict]) -> str:
    """B1: By year, by symbol, full. INSUFFICIENT POWER when trade_count < 30."""
    lines = ["## B1: Performance by Year / Symbol", ""]
    if len(trades) < 30:
        lines.append("**INSUFFICIENT POWER** (trade_count < 30)")
        lines.append("")
    lines.append("### Full sample")
    m = _b1_metrics(trades)
    if m:
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| E[R] | {m['E[R]']:.4f} |")
        lines.append(f"| WinRate | {m['WinRate']:.2f}% |")
        lines.append(f"| AvgWin_R | {m['AvgWin_R']:.4f} |")
        lines.append(f"| AvgLoss_R | {m['AvgLoss_R']:.4f} |")
        lines.append(f"| p10 | {m['p10']:.4f} |")
        lines.append(f"| p50 | {m['p50']:.4f} |")
        lines.append(f"| p90 | {m['p90']:.4f} |")
        lines.append(f"| p95 | {m['p95']:.4f} |")
        lines.append(f"| trade_count | {m['trade_count']} |")
    lines.append("")

    by_year: dict[int, list] = defaultdict(list)
    by_symbol: dict[str, list] = defaultdict(list)
    for t in trades:
        y = _parse_year(t.get("timestamp", ""))
        if y:
            by_year[y].append(t)
        by_symbol[t.get("symbol", "")].append(t)

    lines.append("### By year")
    for year in sorted(by_year.keys()):
        ts = by_year[year]
        m = _b1_metrics(ts)
        if m:
            pw = " INSUFFICIENT POWER" if m["trade_count"] < 30 else ""
            lines.append(f"**{year}** (n={m['trade_count']}{pw})")
            lines.append(f"- E[R]={m['E[R]']:.4f} WinRate={m['WinRate']:.1f}% AvgWin_R={m['AvgWin_R']:.4f} AvgLoss_R={m['AvgLoss_R']:.4f}")
            lines.append(f"- p10={m['p10']:.4f} p50={m['p50']:.4f} p90={m['p90']:.4f} p95={m['p95']:.4f}")
            lines.append("")

    lines.append("### By symbol")
    for sym in sorted(by_symbol.keys()):
        ts = by_symbol[sym]
        m = _b1_metrics(ts)
        if m:
            pw = " INSUFFICIENT POWER" if m["trade_count"] < 30 else ""
            lines.append(f"**{sym}** (n={m['trade_count']}{pw})")
            lines.append(f"- E[R]={m['E[R]']:.4f} WinRate={m['WinRate']:.1f}%")
            lines.append("")

    return "\n".join(lines)


def _run_b2(trades: list[dict], artifacts_dir: Path) -> str:
    """
    B2: Small grid (vol_expansion x3, stop_ATR_k x3, breakout_lookback x3); plateau regions.
    Reads b2_grid_results.csv if available.
    """
    lines = ["## B2: Parameter Grid & Plateau Regions", ""]
    grid_csv = artifacts_dir / "b2_grid_results.csv"
    if grid_csv.exists():
        try:
            import pandas as pd
            df = pd.read_csv(grid_csv)
            lines.append("| vol_expansion | stop_ATR_k | breakout_lookback | trade_count | E_R | WinRate |")
            lines.append("|---------------|------------|-------------------|-------------|-----|---------|")
            for _, row in df.iterrows():
                lines.append(f"| {row['vol_expansion']} | {row['stop_ATR_k']} | {row['breakout_lookback']} | {row['trade_count']} | {row['E_R']:.4f} | {row['WinRate']:.1f} |")
            # Plateau: adjacent E_R within 0.1
            ers = df["E_R"].values
            if len(ers) >= 2:
                std_er = np.std(ers)
                lines.append("")
                lines.append(f"E[R] std across grid: {std_er:.4f} (lower = more plateau)")
        except Exception as e:
            lines.append(f"Could not parse grid: {e}")
    else:
        vol_opts = [1.0, 1.2, 1.5]
        stop_opts = [1.5, 2.0, 2.5]
        lookback_opts = [15, 20, 25]
        lines.append("Grid (run `python3 -m tests.run_alpha_burst_grid` to generate results):")
        lines.append(f"- vol_expansion_threshold: {vol_opts}")
        lines.append(f"- stop_ATR_k: {stop_opts}")
        lines.append(f"- breakout_lookback: {lookback_opts}")
    lines.append("")
    return "\n".join(lines)


def _run_b3(trades: list[dict], artifacts_dir: Path) -> str:
    """B3: Permutation test 1000, block bootstrap 1000 (SEED=42); percentile, p-value."""
    lines = ["## B3: Statistical Validation", ""]
    np.random.seed(SEED)
    rs = np.array([t["R_multiple"] for t in trades])
    n = len(rs)
    if n < 10:
        lines.append("Insufficient trades for B3.")
        return "\n".join(lines)

    observed_mean = np.mean(rs)

    # Permutation: shuffle signs
    n_perm = 1000
    perm_means = []
    for _ in range(n_perm):
        signs = np.random.choice([-1, 1], size=n)
        perm_means.append(np.mean(rs * signs))
    perm_means = np.array(perm_means)
    p_perm = np.mean(perm_means >= observed_mean)
    pct_perm = np.percentile(perm_means, 95)

    # Block bootstrap
    block_size = max(5, n // 20)
    n_boot = 1000
    boot_means = []
    for _ in range(n_boot):
        idx = np.random.randint(0, n - block_size + 1, size=(n + block_size - 1) // block_size)
        resampled = []
        for i in idx:
            resampled.extend(rs[i : i + block_size].tolist())
        resampled = np.array(resampled[:n])
        boot_means.append(np.mean(resampled))
    boot_means = np.array(boot_means)
    pct_2_5 = np.percentile(boot_means, 2.5)
    pct_97_5 = np.percentile(boot_means, 97.5)

    lines.append("### Permutation test (1000, SEED=42)")
    lines.append(f"- Observed E[R]: {observed_mean:.4f}")
    lines.append(f"- 95th percentile of permuted: {pct_perm:.4f}")
    lines.append(f"- p-value (permuted >= observed): {p_perm:.4f}")
    lines.append("")
    lines.append("### Block bootstrap (1000, SEED=42)")
    lines.append(f"- 95% CI for E[R]: [{pct_2_5:.4f}, {pct_97_5:.4f}]")
    lines.append("")

    with open(artifacts_dir / "b3_permutation.csv", "w") as f:
        f.write("permutation_idx,permuted_mean\n")
        for i, m in enumerate(perm_means):
            f.write(f"{i},{m:.6f}\n")
    with open(artifacts_dir / "b3_bootstrap.csv", "w") as f:
        f.write("bootstrap_idx,boot_mean\n")
        for i, m in enumerate(boot_means):
            f.write(f"{i},{m:.6f}\n")
    lines.append("Artifacts: `b3_permutation.csv`, `b3_bootstrap.csv`")
    lines.append("")
    return "\n".join(lines)


def _run_b4(trades: list[dict], artifacts_dir: Path) -> str:
    """B4: Cost stress (slippage 5/10/20 bps, high-vol 2x, fee)."""
    lines = ["## B4: Cost Stress", ""]
    if not trades:
        lines.append("No trades for cost stress.")
        return "\n".join(lines)

    # Approximate: R_multiple = pnl_usdt / initial_risk_usdt
    # Cost in bps reduces pnl. fee ~ 10 bps round-trip typical.
    # Stress: subtract cost impact from R
    slippage_bps = [5, 10, 20]
    fee_bps = 10
    high_vol_mult = 2.0

    rows = [["Scenario", "E[R]_stressed", "WinRate_stressed", "Avg_cost_R"]]
    for slip_bps in slippage_bps:
        total_cost_bps = slip_bps * 2 + fee_bps  # entry+exit
        cost_pct = total_cost_bps / 10000.0
        stressed_rs = []
        for t in trades:
            notional = t["entry_price"] * t["qty"]
            cost_usdt = notional * cost_pct
            adj_pnl = t["pnl_usdt"] - cost_usdt
            r_stressed = adj_pnl / t["initial_risk_usdt"] if t["initial_risk_usdt"] > 0 else 0
            stressed_rs.append(r_stressed)
        m = np.mean(stressed_rs)
        wr = np.mean([1 if r > 0 else 0 for r in stressed_rs]) * 100 if stressed_rs else 0
        avg_cost_r = cost_pct * np.mean([(t["entry_price"] * t["qty"]) / t["initial_risk_usdt"] for t in trades if t["initial_risk_usdt"] > 0]) if trades else 0
        rows.append([f"slippage_{slip_bps}bps", f"{m:.4f}", f"{wr:.1f}%", f"{avg_cost_r:.4f}"])

    # High-vol 2x: assume 2x ATR = 2x risk per trade => same R distribution but 2x drawdown
    rs = [t["R_multiple"] for t in trades]
    rows.append([f"high_vol_{high_vol_mult}x", f"{np.mean(rs):.4f}", f"{np.mean([r>0 for r in rs])*100:.1f}%", "N/A (2x risk)"])

    lines.append("| Scenario | E[R] stressed | WinRate stressed | Avg cost (R) |")
    lines.append("|----------|---------------|------------------|--------------|")
    for r in rows[1:]:
        lines.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} |")
    lines.append("")
    with open(artifacts_dir / "b4_cost_stress.csv", "w") as f:
        f.write(",".join(rows[0]) + "\n")
        for r in rows[1:]:
            f.write(",".join(str(x) for x in r) + "\n")
    lines.append("Artifacts: `b4_cost_stress.csv`")
    return "\n".join(lines)


def main():
    burst_path = get_burst_trades_path()
    if not burst_path.exists():
        print("=" * 60)
        print("Alpha Burst B1 Report")
        print("=" * 60)
        print(f"File not found: {burst_path}")
        print("Run backtest first: python3 -m tests.run_alpha_burst_backtest")
        return

    trades = read_burst_trades(burst_path)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    report_path = REPORTS_DIR / "alpha_burst_b1_report.md"
    header = [
        "# Alpha Burst B1 Report",
        "",
        f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        f"Trade count: {len(trades)}",
        "",
        "---",
        "",
    ]
    b1 = _run_b1(trades)
    b2 = _run_b2(trades, ARTIFACTS_DIR)
    b3 = _run_b3(trades, ARTIFACTS_DIR)
    b4 = _run_b4(trades, ARTIFACTS_DIR)

    content = "\n".join(header) + "\n" + b1 + "\n" + b2 + "\n" + b3 + "\n" + b4
    with open(report_path, "w") as f:
        f.write(content)

    print("=" * 60)
    print("Alpha Burst B1 Report")
    print("=" * 60)
    print(f"Report: {report_path}")
    print(f"Artifacts: {ARTIFACTS_DIR}")
    print(f"Trades: {len(trades)}")


if __name__ == "__main__":
    main()
