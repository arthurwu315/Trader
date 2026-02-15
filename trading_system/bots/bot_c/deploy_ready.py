"""
BNB/USDT 實盤接入檔 - 最佳 Short 策略
- 進場邏輯：三因子投票 (Funding Z, RSI Z, Price Breakout) + EMA200 趨勢過濾
- 風控：2% 硬止損、快速止盈 (tp_atr_mult)、可選追蹤止盈
- 供實盤或模擬盤呼叫：給定 K 線與因子，回傳是否進場及 sl/tp
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

# 與 best_strategy.py 同步之參數（部署前請確認）
DEPLOY_PARAMS = {
    "funding_z_threshold": 1.75,
    "rsi_z_threshold": 1.88,
    "min_score": 2,
    "price_breakout_short": 1.0,
    "sl_atr_mult": 1.5,
    "tp_atr_mult": 2.5,
    "trailing_stop_atr_mult": None,
    "direction": "short",
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


def _vote_score_short(row: Dict[str, Any], params: Dict[str, Any]) -> int:
    """Short 三因子投票：Funding Z、RSI Z、Price Breakout。"""
    score = 0
    fz_th = params.get("funding_z_threshold")
    if fz_th is not None:
        z = row.get("funding_z_score", 0)
        if z is not None and not (isinstance(z, float) and (z != z)):
            if float(z) >= fz_th:
                score += 1
    rz_th = params.get("rsi_z_threshold")
    if rz_th is not None:
        z = row.get("rsi_z_score", 0)
        if z is not None and not (isinstance(z, float) and (z != z)):
            if float(z) >= rz_th:
                score += 1
    if row.get("price_breakout_short", 0) >= 1:
        score += 1
    return score


def get_signal_from_row(row: Dict[str, Any], params: Optional[Dict[str, Any]] = None) -> Optional[SignalOut]:
    """
    給定單根 K 線（含因子欄位），判斷是否進場 Short 並回傳 sl/tp。
    需已含: close, atr, ema_200, funding_z_score, rsi_z_score, price_breakout_short
    """
    p = params or DEPLOY_PARAMS
    min_score = int(p.get("min_score", 2))
    score = _vote_score_short(row, p)
    if score < min_score:
        return None
    close = float(row["close"])
    ema200 = row.get("ema_200")
    if ema200 is not None and close >= float(ema200):
        return None
    atr = float(row.get("atr", 0.0))
    if atr <= 0:
        atr = close * 0.02
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
    )


def get_deploy_params() -> Dict[str, Any]:
    """回傳目前部署參數（含風控）。"""
    return {
        **DEPLOY_PARAMS,
        "hard_stop_position_pct": HARD_STOP_POSITION_PCT,
        "position_size": POSITION_SIZE,
    }


def check_hard_stop(side: str, entry_price: float, current_high: float, current_low: float) -> bool:
    """檢查是否觸及 2% 硬止損。Short: 若 high >= entry * 1.02 則觸發。"""
    if side != "SELL":
        return False
    return current_high >= entry_price * (1 + HARD_STOP_POSITION_PCT / 100.0)


def apply_trailing_tp(
    side: str,
    entry_price: float,
    current_tp: float,
    atr: float,
    trailing_atr_mult: float,
    bar_high: float,
    bar_low: float,
) -> float:
    """
    追蹤止盈：Short 時若價格有利（low < entry），將 tp 下移鎖利。
    新 tp = min(當前 tp, low + trailing_atr_mult * atr)
    """
    if side != "SELL" or trailing_atr_mult is None or trailing_atr_mult <= 0:
        return current_tp
    if bar_low < entry_price:
        new_tp = bar_low + trailing_atr_mult * atr
        if new_tp < current_tp:
            return new_tp
    return current_tp


if __name__ == "__main__":
    print("deploy_ready: 進場邏輯與風控常數已載入")
    print("  HARD_STOP_POSITION_PCT:", HARD_STOP_POSITION_PCT)
    print("  DEPLOY_PARAMS:", get_deploy_params())
