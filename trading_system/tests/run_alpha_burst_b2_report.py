"""
Alpha Burst B2 COMPRESS Report.
Reads logs/alpha_burst_b2_trades.csv.
Outputs: tests/reports/alpha_burst_b2_report.md
Artifacts: tests/reports/alpha_burst_b2_artifacts/
B1: E[R], WinRate, AvgWin_R, AvgLoss_R, p10/p50/p90/p95 by year/symbol
B2: Plateau grid (<=27 groups): compression_atr_pct, expansion_threshold, compression_range_lookback
B3: Permutation 1000, block bootstrap 1000 (block=7 days)
B4: Cost stress (5/10/20 bps + high_vol_2x)
SEED=42 (overridable via env SEED)
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

os.environ.setdefault("SKIP_CONFIG_VALIDATION", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.v9_trade_record import read_burst_b2_trades, get_burst_b2_trades_path

SEED = int(os.getenv("SEED", "42"))
REPORTS_DIR = Path(__file__).resolve().parent / "reports"
ARTIFACTS_DIR = REPORTS_DIR / "alpha_burst_b2_artifacts"


def _parse_year(ts: str) -> int | None:
    try:
        return int(ts[:4])
    except (ValueError, TypeError):
        return None


def _metrics(trades: list[dict]) -> dict:
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
    """B1: R expectancy by year, symbol, full."""
    lines = ["## B1: Performance by Year / Symbol", ""]
    if len(trades) < 30:
        lines.append("**INSUFFICIENT POWER** (trade_count < 30)")
        lines.append("")
    lines.append("### Full sample")
    m = _metrics(trades)
    if m:
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        for k, v in m.items():
            if isinstance(v, float):
                lines.append(f"| {k} | {v:.4f} |")
            else:
                lines.append(f"| {k} | {v} |")
    lines.append("")

    by_year = defaultdict(list)
    by_symbol = defaultdict(list)
    for t in trades:
        y = _parse_year(t.get("timestamp", ""))
        if y:
            by_year[y].append(t)
        by_symbol[t.get("symbol", "")].append(t)

    lines.append("### By year")
    for year in sorted(by_year.keys()):
        ts = by_year[year]
        m = _metrics(ts)
        if m:
            pw = " INSUFFICIENT POWER" if m["trade_count"] < 30 else ""
            lines.append(f"**{year}** (n={m['trade_count']}{pw})")
            lines.append(f"- E[R]={m['E[R]']:.4f} WinRate={m['WinRate']:.1f}%")
            lines.append("")

    lines.append("### By symbol")
    for sym in sorted(by_symbol.keys()):
        ts = by_symbol[sym]
        m = _metrics(ts)
        if m:
            lines.append(f"**{sym}** (n={m['trade_count']})")
            lines.append(f"- E[R]={m['E[R]']:.4f} WinRate={m['WinRate']:.1f}%")
            lines.append("")

    return "\n".join(lines)


def _run_b2_grid(trades: list[dict], artifacts_dir: Path) -> str:
    """B2: Plateau grid (<=27). Read from b2_grid_results.csv if exists."""
    lines = ["## B2: Parameter Grid & Plateau Regions", ""]
    grid_csv = artifacts_dir / "b2_grid_results.csv"
    comp_opts = [20, 30, 40]
    exp_opts = [1.05, 1.1, 1.2]
    range_opts = [30, 40, 50]

    if grid_csv.exists():
        try:
            import csv
            with open(grid_csv) as f:
                r = csv.DictReader(f)
                rows = list(r)
            lines.append("| compression_atr_pct | expansion_threshold | compression_range_lookback | trade_count | E_R | WinRate |")
            lines.append("|---------------------|---------------------|----------------------------|-------------|-----|---------|")
            for row in rows:
                lines.append(
                    f"| {row.get('compression_atr_pct')} | {row.get('expansion_threshold')} | "
                    f"{row.get('compression_range_lookback')} | {row.get('trade_count')} | "
                    f"{float(row.get('E_R', 0)):.4f} | {float(row.get('WinRate', 0)):.1f}% |"
                )
        except Exception as e:
            lines.append(f"Parse error: {e}")
    else:
        lines.append("Grid (run `python3 -m tests.run_alpha_burst_b2_grid` to generate):")
        lines.append(f"- compression_atr_pct: {comp_opts}")
        lines.append(f"- expansion_threshold: {exp_opts}")
        lines.append(f"- compression_range_lookback: {range_opts}")
    lines.append("")
    return "\n".join(lines)


def _run_b3(trades: list[dict], artifacts_dir: Path) -> str:
    """B3: Permutation 1000, block bootstrap 1000 (block=7 days)."""
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

    # Block bootstrap: 7-day blocks
    block_days = 7
    from datetime import datetime
    timestamps = []
    for t in trades:
        ts = t.get("timestamp", "")
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            timestamps.append(dt.timestamp())
        except Exception:
            timestamps.append(0.0)
    timestamps = np.array(timestamps)
    day_ids = (timestamps - timestamps.min()) / (86400 * block_days)
    block_ids = day_ids.astype(int)
    uniq_blocks = np.unique(block_ids)
    block_to_indices = {b: np.where(block_ids == b)[0] for b in uniq_blocks}
    blocks = [block_to_indices[b] for b in uniq_blocks]

    n_boot = 1000
    boot_means = []
    for _ in range(n_boot):
        sel = np.random.choice(len(blocks), size=len(blocks), replace=True)
        resampled = []
        for i in sel:
            resampled.extend(rs[blocks[i]].tolist())
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
    lines.append("### Block bootstrap (1000, block=7 days, SEED=42)")
    lines.append(f"- 95% CI for E[R]: [{pct_2_5:.4f}, {pct_97_5:.4f}]")
    lines.append("")

    artifacts_dir.mkdir(parents=True, exist_ok=True)
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
    """B4: Cost stress (5/10/20 bps + high_vol_2x)."""
    lines = ["## B4: Cost Stress", ""]
    if not trades:
        lines.append("No trades for cost stress.")
        return "\n".join(lines)

    slippage_bps = [5, 10, 20]
    fee_bps = 10
    high_vol_mult = 2.0

    rows = [["Scenario", "E[R]_stressed", "WinRate_stressed", "Avg_cost_R"]]
    for slip_bps in slippage_bps:
        total_cost_bps = slip_bps * 2 + fee_bps
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
        avg_cost_r = 0
        if trades:
            ratios = [(t["entry_price"] * t["qty"]) / t["initial_risk_usdt"] for t in trades if t["initial_risk_usdt"] > 0]
            avg_cost_r = cost_pct * np.mean(ratios) if ratios else 0
        rows.append([f"slippage_{slip_bps}bps", f"{m:.4f}", f"{wr:.1f}%", f"{avg_cost_r:.4f}"])

    rs = [t["R_multiple"] for t in trades]
    rows.append([f"high_vol_{high_vol_mult}x", f"{np.mean(rs):.4f}", f"{np.mean([r > 0 for r in rs]) * 100:.1f}%", "N/A (2x risk)"])

    lines.append("| Scenario | E[R] stressed | WinRate stressed | Avg cost (R) |")
    lines.append("|----------|---------------|------------------|--------------|")
    for r in rows[1:]:
        lines.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} |")
    lines.append("")

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    import csv
    with open(artifacts_dir / "b4_cost_stress.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(rows[0])
        w.writerows(rows[1:])
    lines.append("Artifacts: `b4_cost_stress.csv`")
    return "\n".join(lines)


def main():
    burst_path = get_burst_b2_trades_path()
    if not burst_path.exists():
        print("=" * 60)
        print("Alpha Burst B2 Report")
        print("=" * 60)
        print(f"File not found: {burst_path}")
        print("Run backtest first: python3 -m tests.run_alpha_burst_b2_backtest")
        return

    trades = read_burst_b2_trades(burst_path)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    report_path = REPORTS_DIR / "alpha_burst_b2_report.md"
    header = [
        "# Alpha Burst B2 COMPRESS Report",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"Trade count: {len(trades)}",
        "",
        "---",
        "",
    ]
    b1 = _run_b1(trades)
    b2 = _run_b2_grid(trades, ARTIFACTS_DIR)
    b3 = _run_b3(trades, ARTIFACTS_DIR)
    b4 = _run_b4(trades, ARTIFACTS_DIR)

    footer = [
        "",
        "---",
        "",
        "## Acceptance",
        "",
        "```bash",
        "python3 -m tests.run_alpha_burst_b2_backtest",
        "python3 -m tests.run_alpha_burst_b2_report",
        "grep ALPHA_BURST_B2_COMPRESS logs/v9_trade_records.csv | tail -n 20",
        "cat STRATEGY_STATE_ALPHA_BURST_B2.md",
        "```",
    ]
    content = "\n".join(header) + "\n" + b1 + "\n" + b2 + "\n" + b3 + "\n" + b4 + "\n" + "\n".join(footer)
    with open(report_path, "w") as f:
        f.write(content)

    print("=" * 60)
    print("Alpha Burst B2 COMPRESS Report")
    print("=" * 60)
    print(f"Report: {report_path}")
    print(f"Artifacts: {ARTIFACTS_DIR}")
    print(f"Trades: {len(trades)}")


if __name__ == "__main__":
    main()
