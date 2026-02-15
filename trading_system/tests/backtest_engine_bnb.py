"""
固定回測引擎 - BNB/USDT 1h 策略
不修改此引擎與風控邏輯，僅供 StrategyBNB 呼叫。
- 含手續費與滑點
- 固定倉位 2%
- 計算 Sharpe、最大回撤、日交易數、周報酬
- Walk-forward 與 Monte Carlo 驗證
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# 固定依賴策略模組（僅讀取 StrategyBNB 與 add_factor_columns）
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bots.bot_c.strategy_bnb import StrategyBNB, ExitRules, add_factor_columns  # noqa: E402
from backtest_utils import simulate_trade  # noqa: E402


@dataclass
class BNBTrade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    side: str
    return_gross_pct: float
    fee_slippage_pct: float
    return_net_pct: float
    win: bool


class BacktestEngineBNB:
    """
    固定回測引擎：只接受 StrategyBNB + 1h OHLCV，輸出標準化指標。
    風控與引擎邏輯不隨策略變體改動。
    """

    def __init__(
        self,
        position_size_pct: float = 0.02,
        max_trades_per_day: Optional[int] = None,
    ):
        self.position_size_pct = position_size_pct
        self.max_trades_per_day = max_trades_per_day

    def run(
        self,
        strategy: StrategyBNB,
        market_data: pd.DataFrame,
        fee_bps: float = 9.0,
        slippage_bps: float = 5.0,
    ) -> Dict[str, Any]:
        """
        market_data: 1h OHLCV，會自動呼叫 add_factor_columns 補因子。
        回傳: total_return_pct, sharpe, max_drawdown_pct, trades_count, trades_per_day_avg,
              weekly_return_pct, profitable_after_fees, trades[], equity_curve
        """
        if market_data.empty or len(market_data) < 50:
            return self._empty_result("insufficient_data")

        df = add_factor_columns(market_data.copy())
        signals_df = strategy.generate_signal(df)

        fee_pct = fee_bps / 10000.0 * 2  # 往返
        slippage_pct = slippage_bps / 10000.0 * 2
        cost_per_trade_pct = fee_pct + slippage_pct

        trades: List[BNBTrade] = []
        i = 0
        daily_count: Dict[pd.Timestamp, int] = {}

        while i < len(df) - 1:
            row = signals_df.iloc[i]
            if row["signal"] != 1:
                i += 1
                continue

            side = row["side"]
            entry_price = float(row["entry_price"])
            sl_price = float(row["sl_price"])
            tp_price = float(row["tp_price"])
            entry_time = df.iloc[i]["timestamp"]
            exit_after_bars = strategy.exit_rules.exit_after_bars

            start_idx = i + 1
            if exit_after_bars is not None:
                exit_idx = min(i + exit_after_bars, len(df) - 1)
                exit_row = df.iloc[exit_idx]
                exit_price = float(exit_row["close"])
                exit_time = exit_row["timestamp"]
                # 仍檢查區間內是否先觸 SL/TP
                for j in range(start_idx, exit_idx + 1):
                    r = df.iloc[j]
                    h, l_ = float(r["high"]), float(r["low"])
                    if side == "BUY":
                        if l_ <= sl_price:
                            exit_price, exit_time = sl_price, r["timestamp"]
                            exit_idx = j
                            break
                        if h >= tp_price:
                            exit_price, exit_time = tp_price, r["timestamp"]
                            exit_idx = j
                            break
                    else:
                        if h >= sl_price:
                            exit_price, exit_time = sl_price, r["timestamp"]
                            exit_idx = j
                            break
                        if l_ <= tp_price:
                            exit_price, exit_time = tp_price, r["timestamp"]
                            exit_idx = j
                            break
            else:
                exit_idx, exit_price, exit_time = simulate_trade(
                    df, start_idx, side, entry_price, sl_price, tp_price
                )

            direction = 1 if side == "BUY" else -1
            gross_pct = (exit_price - entry_price) / entry_price * direction * 100
            net_pct = gross_pct - (fee_pct + slippage_pct) * 100

            day_key = pd.Timestamp(entry_time).date() if hasattr(entry_time, "date") else pd.Timestamp(entry_time).normalize().date()
            day_ts = pd.Timestamp(day_key)
            daily_count[day_ts] = daily_count.get(day_ts, 0) + 1

            if self.max_trades_per_day is not None and daily_count[day_ts] > self.max_trades_per_day:
                i += 1
                continue

            trades.append(BNBTrade(
                entry_time=entry_time,
                exit_time=exit_time,
                entry_price=entry_price,
                exit_price=exit_price,
                side=side,
                return_gross_pct=gross_pct,
                fee_slippage_pct=cost_per_trade_pct * 100,
                return_net_pct=net_pct,
                win=net_pct > 0,
            ))
            i = exit_idx + 1

        return self._compute_metrics(trades, df, cost_per_trade_pct)

    def _compute_metrics(
        self,
        trades: List[BNBTrade],
        df: pd.DataFrame,
        cost_per_trade_pct: float,
    ) -> Dict[str, Any]:
        if not trades:
            return self._empty_result("no_trades")

        returns = [t.return_net_pct for t in trades]
        total_return_pct = sum(returns)
        n = len(trades)
        days = (df["timestamp"].max() - df["timestamp"].min()).days or 1
        trades_per_day = n / max(days / 7 * 7 / 7, 1/7)
        weeks = max(days / 7, 1)
        weekly_return_pct = total_return_pct / weeks

        # 簡化 equity curve: 每筆交易用 position_size 的淨報酬累加
        equity = 1.0
        position_size = self.position_size_pct
        curve = [1.0]
        for t in trades:
            equity *= (1 + (t.return_net_pct / 100.0) * position_size)
            curve.append(equity)
        curve = np.array(curve)
        peak = np.maximum.accumulate(curve)
        dd = (peak - curve) / peak
        max_drawdown_pct = float(np.max(dd)) * 100 if len(dd) > 0 else 0.0

        # Sharpe (年化): 用日報酬近似。每筆報酬視為發生在一天
        if n > 1:
            ret_std = np.std(returns)
            sharpe = (np.mean(returns) / ret_std * np.sqrt(252)) if ret_std > 0 else 0.0
        else:
            sharpe = 0.0

        profitable = total_return_pct > 0
        win_rate = sum(1 for t in trades if t.win) / n * 100

        return {
            "total_return_pct": total_return_pct,
            "sharpe": sharpe,
            "max_drawdown_pct": max_drawdown_pct,
            "trades_count": n,
            "trades_per_day_avg": n / max(days, 1) * 1.0,
            "weekly_return_pct": weekly_return_pct,
            "profitable_after_fees": profitable,
            "win_rate_pct": win_rate,
            "days": days,
            "trades": trades,
            "equity_curve": curve,
        }

    def _empty_result(self, reason: str) -> Dict[str, Any]:
        return {
            "total_return_pct": 0.0,
            "sharpe": 0.0,
            "max_drawdown_pct": 0.0,
            "trades_count": 0,
            "trades_per_day_avg": 0.0,
            "weekly_return_pct": 0.0,
            "profitable_after_fees": False,
            "win_rate_pct": 0.0,
            "days": 0,
            "trades": [],
            "equity_curve": np.array([1.0]),
            "skip_reason": reason,
        }

    def walk_forward(
        self,
        strategy: StrategyBNB,
        market_data: pd.DataFrame,
        fee_bps: float = 9.0,
        slippage_bps: float = 5.0,
        train_pct: float = 0.6,
        n_folds: int = 3,
    ) -> Dict[str, Any]:
        """
        Walk-forward: 將資料依時間切成 n_folds 段，每段用前 train_pct 訓練（此引擎不調參，僅跑回測），
        後 (1-train_pct) 做 OOS 驗證。回傳各 fold 的 OOS 指標與彙總。
        """
        if market_data.empty or len(market_data) < 200:
            return {"folds": [], "oos_weekly_return_avg": 0.0, "oos_max_dd_avg": 0.0}

        df = market_data.sort_values("timestamp").reset_index(drop=True)
        n = len(df)
        folds = []
        for k in range(n_folds):
            start = int(n * k / n_folds)
            end = int(n * (k + 1) / n_folds)
            if end - start < 100:
                continue
            segment = df.iloc[start:end]
            train_end = start + int((end - start) * train_pct)
            train_df = df.iloc[start:train_end]
            oos_df = df.iloc[train_end:end]
            if len(oos_df) < 24:
                continue
            res_train = self.run(strategy, train_df, fee_bps, slippage_bps)
            res_oos = self.run(strategy, oos_df, fee_bps, slippage_bps)
            folds.append({
                "fold": k,
                "train_return_pct": res_train.get("total_return_pct", 0),
                "oos_return_pct": res_oos.get("total_return_pct", 0),
                "oos_weekly_return_pct": res_oos.get("weekly_return_pct", 0),
                "oos_max_dd_pct": res_oos.get("max_drawdown_pct", 0),
                "oos_trades": res_oos.get("trades_count", 0),
            })
        if not folds:
            return {"folds": [], "oos_weekly_return_avg": 0.0, "oos_max_dd_avg": 0.0}
        oos_weekly_avg = sum(f["oos_weekly_return_pct"] for f in folds) / len(folds)
        oos_dd_avg = sum(f["oos_max_dd_pct"] for f in folds) / len(folds)
        return {
            "folds": folds,
            "oos_weekly_return_avg": oos_weekly_avg,
            "oos_max_dd_avg": oos_dd_avg,
        }

    def monte_carlo(
        self,
        strategy: StrategyBNB,
        market_data: pd.DataFrame,
        fee_bps: float = 9.0,
        slippage_bps: float = 5.0,
        n_simulations: int = 100,
        seed: Optional[int] = 42,
    ) -> Dict[str, Any]:
        """
        Monte Carlo: 先跑一次完整回測得到交易列表，再對「交易順序」做 bootstrap 重抽，
        重算 equity curve 與 max_dd / 報酬。回傳分位數等。
        """
        res = self.run(strategy, market_data, fee_bps, slippage_bps)
        trades: List[BNBTrade] = res.get("trades", [])
        if len(trades) < 2:
            return {
                "mean_weekly_return_pct": res.get("weekly_return_pct", 0),
                "mean_max_dd_pct": res.get("max_drawdown_pct", 0),
                "p5_weekly_return_pct": res.get("weekly_return_pct", 0),
                "p95_weekly_return_pct": res.get("weekly_return_pct", 0),
                "simulations": 0,
            }

        if seed is not None:
            random.seed(seed)
        position_size = self.position_size_pct
        weekly_returns = []
        max_dds = []
        n = len(trades)
        weeks = max((market_data["timestamp"].max() - market_data["timestamp"].min()).days / 7, 1)

        for _ in range(n_simulations):
            idx = [random.randint(0, n - 1) for _ in range(n)]
            sim_returns = [trades[i].return_net_pct for i in idx]
            equity = 1.0
            curve = [1.0]
            for r in sim_returns:
                equity *= (1 + (r / 100.0) * position_size)
                curve.append(equity)
            curve = np.array(curve)
            peak = np.maximum.accumulate(curve)
            dd = (peak - curve) / np.maximum(peak, 1e-12)
            max_dds.append(float(np.max(dd)) * 100)
            total_ret = (equity - 1.0) * 100
            weekly_returns.append(total_ret / weeks)

        weekly_returns = np.array(weekly_returns)
        max_dds = np.array(max_dds)
        return {
            "mean_weekly_return_pct": float(np.mean(weekly_returns)),
            "mean_max_dd_pct": float(np.mean(max_dds)),
            "p5_weekly_return_pct": float(np.percentile(weekly_returns, 5)),
            "p95_weekly_return_pct": float(np.percentile(weekly_returns, 95)),
            "p5_max_dd_pct": float(np.percentile(max_dds, 95)),
            "p95_max_dd_pct": float(np.percentile(max_dds, 5)),
            "simulations": n_simulations,
        }
