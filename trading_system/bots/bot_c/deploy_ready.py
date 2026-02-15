"""
BNB/USDT 實盤接入檔 - 牛熊分治 (Regime-Specific)
- 進場邏輯：三因子投票 (Funding Z, RSI Z, Price Breakout) + EMA200 趨勢
- 牛熊分治：EMA200 之上用 LONG_Z（寬鬆做多），之下用 SHORT 門檻（嚴謹做空）
- 風控：2% 硬止損、快速止盈、可選追蹤止盈
- 當前部署：Calmar 13.9 優化 Short（Funding Z=0.62, RSI Z=2.0, SL ATR=2.0, TP ATR=2.5）
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

# 牛熊分治 Z-Score 門檻
LONG_Z_THRESH = 1.0
LONG_MIN_SCORE = 1
# Short 部署參數（Ratio 13.9 優化結果）
SHORT_FUNDING_Z = 0.62
SHORT_RSI_Z = 2.0
SHORT_MIN_SCORE = 2

# 與 best_strategy / 優化報告同步
DEPLOY_PARAMS = {
    "funding_z_long": LONG_Z_THRESH,
    "rsi_z_long": LONG_Z_THRESH,
    "min_score_long": LONG_MIN_SCORE,
    "funding_z_short": SHORT_FUNDING_Z,
    "rsi_z_short": SHORT_RSI_Z,
    "min_score_short": SHORT_MIN_SCORE,
    "price_breakout_long": 1.0,
    "price_breakout_short": 1.0,
    "sl_atr_mult": 2.0,
    "tp_atr_mult": 2.5,
    "trailing_stop_atr_mult": None,
}

# 風控常數
HARD_STOP_POSITION_PCT = 2.0
POSITION_SIZE = 0.02


@dataclass
class SignalOut:
    """單根 K 線的訊號輸出"""
    should_enter: bool
    side: str
    entry_price: float
    sl_price: float
    tp_price: float
    hard_stop_price: float
    atr: float
    regime: str  # "bull" | "bear"


def _vote_score_long(row: Dict[str, Any], z_thresh: float) -> int:
    """做多三因子：Funding Z <= -z_thresh, RSI Z <= -z_thresh, Price Breakout Long."""
    score = 0
    z_f = row.get("funding_z_score", 0)
    if z_f is not None and not (isinstance(z_f, float) and (z_f != z_f)):
        if float(z_f) <= -z_thresh:
            score += 1
    z_r = row.get("rsi_z_score", 0)
    if z_r is not None and not (isinstance(z_r, float) and (z_r != z_r)):
        if float(z_r) <= -z_thresh:
            score += 1
    if row.get("price_breakout_long", 0) >= 1:
        score += 1
    return score


def _vote_score_short(
    row: Dict[str, Any],
    funding_z_thresh: float,
    rsi_z_thresh: float,
) -> int:
    """做空三因子：Funding Z >= funding_z_thresh, RSI Z >= rsi_z_thresh, Price Breakout Short."""
    score = 0
    z_f = row.get("funding_z_score", 0)
    if z_f is not None and not (isinstance(z_f, float) and (z_f != z_f)):
        if float(z_f) >= funding_z_thresh:
            score += 1
    z_r = row.get("rsi_z_score", 0)
    if z_r is not None and not (isinstance(z_r, float) and (z_r != z_r)):
        if float(z_r) >= rsi_z_thresh:
            score += 1
    if row.get("price_breakout_short", 0) >= 1:
        score += 1
    return score


def get_signal_from_row(row: Dict[str, Any], params: Optional[Dict[str, Any]] = None) -> Optional[SignalOut]:
    """
    牛熊分治：依 close 與 ema_200 決定 regime。
    - 價格 > EMA200（牛市）：用 LONG_Z_THRESH、LONG_MIN_SCORE 評估做多。
    - 價格 < EMA200（熊市）：用 SHORT_Z_THRESH、SHORT_MIN_SCORE 評估做空。
    """
    p = params or DEPLOY_PARAMS
    close = float(row["close"])
    ema200 = row.get("ema_200")
    if ema200 is None or (isinstance(ema200, float) and ema200 != ema200):
        ema200 = close
    else:
        ema200 = float(ema200)
    atr = float(row.get("atr", 0.0))
    if atr <= 0:
        atr = close * 0.02

    if close > ema200:
        regime = "bull"
        z_thresh = p.get("funding_z_long", LONG_Z_THRESH)
        min_score = int(p.get("min_score_long", LONG_MIN_SCORE))
        score = _vote_score_long(row, z_thresh)
        if score < min_score:
            return None
        sl_atr = p.get("sl_atr_mult", 1.5)
        tp_atr = p.get("tp_atr_mult", 2.5)
        sl_price = close - sl_atr * atr
        tp_price = close + tp_atr * atr
        hard_stop_price = close * (1 - HARD_STOP_POSITION_PCT / 100.0)
        return SignalOut(
            should_enter=True,
            side="BUY",
            entry_price=close,
            sl_price=sl_price,
            tp_price=tp_price,
            hard_stop_price=hard_stop_price,
            atr=atr,
            regime=regime,
        )
    else:
        regime = "bear"
        fz = p.get("funding_z_short", SHORT_FUNDING_Z)
        rz = p.get("rsi_z_short", SHORT_RSI_Z)
        min_score = int(p.get("min_score_short", SHORT_MIN_SCORE))
        score = _vote_score_short(row, fz, rz)
        if score < min_score:
            return None
        sl_atr = p.get("sl_atr_mult", 1.5)
        tp_atr = p.get("tp_atr_mult", 2.5)
        sl_price = close + sl_atr * atr
        tp_price = close - tp_atr * atr
        hard_stop_price = close * (1 + HARD_STOP_POSITION_PCT / 100.0)
        return SignalOut(
            should_enter=True,
            side="SELL",
            entry_price=close,
            sl_price=sl_price,
            tp_price=tp_price,
            hard_stop_price=hard_stop_price,
            atr=atr,
            regime=regime,
        )


def get_deploy_params() -> Dict[str, Any]:
    """回傳目前部署參數（含牛熊門檻與風控）。"""
    return {
        **DEPLOY_PARAMS,
        "long_z_thresh": LONG_Z_THRESH,
        "short_funding_z": SHORT_FUNDING_Z,
        "short_rsi_z": SHORT_RSI_Z,
        "hard_stop_position_pct": HARD_STOP_POSITION_PCT,
        "position_size": POSITION_SIZE,
    }


def check_hard_stop(side: str, entry_price: float, current_high: float, current_low: float) -> bool:
    """檢查是否觸及 2% 硬止損。Long: low <= entry*0.98；Short: high >= entry*1.02。"""
    thresh = entry_price * (HARD_STOP_POSITION_PCT / 100.0)
    if side == "BUY":
        return current_low <= entry_price - thresh
    return current_high >= entry_price + thresh


def apply_trailing_tp(
    side: str,
    entry_price: float,
    current_tp: float,
    atr: float,
    trailing_atr_mult: float,
    bar_high: float,
    bar_low: float,
) -> float:
    """追蹤止盈：有利移動時鎖利。"""
    if trailing_atr_mult is None or trailing_atr_mult <= 0:
        return current_tp
    if side == "BUY" and bar_high > entry_price:
        new_tp = bar_high - trailing_atr_mult * atr
        if new_tp > current_tp:
            return new_tp
    if side == "SELL" and bar_low < entry_price:
        new_tp = bar_low + trailing_atr_mult * atr
        if new_tp < current_tp:
            return new_tp
    return current_tp


if __name__ == "__main__":
    print("deploy_ready: 牛熊分治門檻已載入 (Short: Funding Z=0.62, RSI Z=2.0, SL ATR=2.0, TP ATR=2.5)")
    print("  LONG_Z_THRESH (牛市做多):", LONG_Z_THRESH)
    print("  SHORT_FUNDING_Z / SHORT_RSI_Z (熊市做空):", SHORT_FUNDING_Z, SHORT_RSI_Z)
    print("  EMA200 過濾：僅在價格 < EMA200 時允許做空，防止狂暴牛市逆勢空")
    print("  HARD_STOP_POSITION_PCT:", HARD_STOP_POSITION_PCT)
    print("  DEPLOY_PARAMS:", get_deploy_params())
