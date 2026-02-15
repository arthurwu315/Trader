"""
BNB/USDT 策略 - 1h 時間框架
入場因子: Funding rate (proxy)、OI (proxy)、波動率、價格突破
出場: 止盈、止損、固定 ATR 倍數或固定時間
僅改策略邏輯與入場/出場門檻，不依賴風控模組內部實作。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


# ---------- 因子計算（回測用，固定不隨策略變體改動）----------

def add_factor_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    在 OHLCV DataFrame 上計算入場因子欄位（固定邏輯，不屬於策略門檻）。
    - funding_rate_proxy: 8h 收盤報酬 (多空壓力代理)
    - oi_proxy: 成交量 / 24h 均量
    - volatility: ATR(14) / close
    - price_breakout_long: close > 過去 lookback 根 high 的最大值
    - price_breakout_short: close < 過去 lookback 根 low 的最小值
    """
    out = df.copy()
    close = out["close"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)
    volume = out["volume"].astype(float)

    roll_168 = 168  # 168 小時滾動窗（1h K 線 = 168 根）

    # Funding proxy: 8 根 K 線報酬 (約 8h)
    out["funding_rate_proxy"] = close.pct_change(8)
    # 滾動 168 小時的 Funding 平均值與標準差 → Z-Score（自適應）
    out["funding_roll_mean_168"] = out["funding_rate_proxy"].rolling(roll_168, min_periods=roll_168).mean()
    out["funding_roll_std_168"] = out["funding_rate_proxy"].rolling(roll_168, min_periods=roll_168).std()
    out["funding_z_score"] = (out["funding_rate_proxy"] - out["funding_roll_mean_168"]) / out["funding_roll_std_168"].replace(0, np.nan)
    out["funding_z_score"] = out["funding_z_score"].fillna(0)

    out["oi_proxy"] = volume / volume.rolling(24, min_periods=1).mean().replace(0, np.nan)
    out["oi_proxy"] = out["oi_proxy"].fillna(1.0)

    # ATR(14) / close，並保留 atr 供 SL/TP 計算
    atr_period = 14
    tr = high - low
    prev_close = close.shift(1)
    tr = tr.combine((high - prev_close).abs(), max).combine((low - prev_close).abs(), max)
    atr = tr.rolling(atr_period, min_periods=atr_period).mean()
    out["atr"] = atr
    out["volatility"] = (atr / close).fillna(0)

    # 突破: 收盤突破前 N 根最高/最低
    lookback = 20
    out["roll_high"] = high.rolling(lookback, min_periods=lookback).max().shift(1)
    out["roll_low"] = low.rolling(lookback, min_periods=lookback).min().shift(1)
    out["price_breakout_long"] = (close > out["roll_high"]).astype(int)
    out["price_breakout_short"] = (close < out["roll_low"]).astype(int)

    # 動能因子：RSI(14)、MACD(12,26,9)
    rsi_period = 14
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(rsi_period, min_periods=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(rsi_period, min_periods=rsi_period).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi"] = (100 - (100 / (1 + rs))).fillna(50)
    # RSI 滾動 168 小時 Z-Score（自適應）
    out["rsi_roll_mean_168"] = out["rsi"].rolling(roll_168, min_periods=roll_168).mean()
    out["rsi_roll_std_168"] = out["rsi"].rolling(roll_168, min_periods=roll_168).std()
    out["rsi_z_score"] = (out["rsi"] - out["rsi_roll_mean_168"]) / out["rsi_roll_std_168"].replace(0, np.nan)
    out["rsi_z_score"] = out["rsi_z_score"].fillna(0)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["macd_line"] = ema12 - ema26
    out["macd_signal"] = out["macd_line"].ewm(span=9, adjust=False).mean()
    out["macd_above_signal"] = (out["macd_line"] > out["macd_signal"]).astype(int)
    out["macd_below_signal"] = (out["macd_line"] < out["macd_signal"]).astype(int)

    # 趨勢過濾：EMA 200，僅在價格 > EMA200 做多、< EMA200 做空
    out["ema_200"] = close.ewm(span=200, adjust=False).mean()

    return out


# ---------- 出場規則定義 ----------

@dataclass
class ExitRules:
    """出場規則（固定結構，僅參數不同）"""
    # 止盈: 固定倍數 R (R = 進場到止損距離)，或 None 表示用 atr_tp_mult
    tp_r_mult: Optional[float] = 2.0
    # 止損: 基礎 ATR 倍數（搜尋範圍建議 [1.0, 2.0]）
    sl_atr_mult: float = 1.5
    # 追蹤止損: 價格有利移動後以 ATR 倍數追蹤，None 表示不啟用
    trailing_stop_atr_mult: Optional[float] = None
    # 固定時間出場（K 線根數），None 表示不啟用
    exit_after_bars: Optional[int] = None
    # 固定止盈 %（可選，若設則覆蓋 tp_r_mult 的結果取較大者）
    tp_fixed_pct: Optional[float] = None


# ---------- 策略類 ----------

class StrategyBNB:
    """
    BNB/USDT 多因子入場 + 可配置出場。
    支援「因子投票機制」：當 entry_thresholds 含 min_score 時，僅以 Funding Z、RSI Z、Price Breakout 三因子計分，
    總分 >= min_score 即進場（例如 min_score=2 表示至少 2 個因子達標即可）。
    """

    def __init__(
        self,
        entry_thresholds: Dict[str, float],
        exit_rules: ExitRules,
        position_size: float = 0.02,
        direction: str = "long",
        min_factors_required: Optional[int] = None,
    ):
        self.entry_thresholds = entry_thresholds
        self.exit_rules = exit_rules
        self.position_size = position_size
        self.direction = direction.lower()
        self._use_vote = "min_score" in entry_thresholds
        self._min_score = int(entry_thresholds.get("min_score", 2))
        if min_factors_required is not None:
            self.min_factors_required = min_factors_required
        elif self._use_vote:
            self.min_factors_required = self._min_score
        else:
            self.min_factors_required = len([k for k in entry_thresholds if k != "min_score"])

    def _vote_score(self, row: pd.Series) -> int:
        """三因子投票：Funding Z、RSI Z、Price Breakout，各達標得 1 分。"""
        score = 0
        fz_th = self.entry_thresholds.get("funding_z_threshold")
        if fz_th is not None:
            z = row.get("funding_z_score", 0)
            if not pd.isna(z):
                z = float(z)
                if self.direction == "long" and z <= -fz_th:
                    score += 1
                elif self.direction == "short" and z >= fz_th:
                    score += 1
        rz_th = self.entry_thresholds.get("rsi_z_threshold")
        if rz_th is not None:
            z = row.get("rsi_z_score", 0)
            if not pd.isna(z):
                z = float(z)
                if self.direction == "long" and z <= -rz_th:
                    score += 1
                elif self.direction == "short" and z >= rz_th:
                    score += 1
        if self.direction == "long":
            if row.get("price_breakout_long", 0) >= 1:
                score += 1
        else:
            if row.get("price_breakout_short", 0) >= 1:
                score += 1
        return score

    def get_score_distribution(self, market_data: pd.DataFrame) -> Dict[int, int]:
        """回傳各得分出現次數（用於診斷 0 交易）。key=1,2,3 為得分，value=該得分的 K 線數。"""
        df = market_data.copy()
        counts = {1: 0, 2: 0, 3: 0}
        for i in range(1, len(df)):
            row = df.iloc[i]
            s = self._vote_score(row)
            if 1 <= s <= 3:
                counts[s] += 1
        return counts

    def generate_signal(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        market_data: 需已含 add_factor_columns 的欄位。
        返回: 與 market_data 同 index 的 DataFrame，含 signal (1=進場), side, entry_price, sl_price, tp_price 等。
        """
        df = market_data.copy()
        n = len(df)
        signal = pd.Series(0, index=df.index)
        side_ser = pd.Series("", index=df.index)
        entry_price = pd.Series(np.nan, index=df.index)
        sl_price = pd.Series(np.nan, index=df.index)
        tp_price = pd.Series(np.nan, index=df.index)

        use_vote = self._use_vote
        min_score = self._min_score

        for i in range(1, n):
            row = df.iloc[i]
            score = 0
            required = self.min_factors_required

            if use_vote:
                score = self._vote_score(row)
                if score < min_score:
                    continue
            else:
                for factor, threshold in self.entry_thresholds.items():
                    if factor == "min_score":
                        continue
                    if factor not in row.index:
                        continue
                    val = row[factor]
                    if pd.isna(val):
                        continue
                    # 依因子類型判斷：大於或小於門檻
                    if factor == "funding_z_threshold":
                        z = row.get("funding_z_score", 0)
                        if pd.isna(z):
                            continue
                        z = float(z)
                        if self.direction == "long" and z <= -threshold:
                            score += 1
                        elif self.direction == "short" and z >= threshold:
                            score += 1
                    elif factor == "rsi_z_threshold":
                        z = row.get("rsi_z_score", 0)
                        if pd.isna(z):
                            continue
                        z = float(z)
                        if self.direction == "long" and z <= -threshold:
                            score += 1
                        elif self.direction == "short" and z >= threshold:
                            score += 1
                    elif factor == "funding_rate_proxy":
                        if self.direction == "long" and val < threshold:
                            score += 1
                        elif self.direction == "short" and val > threshold:
                            score += 1
                    elif factor == "oi_proxy":
                        if val >= threshold:
                            score += 1
                    elif factor == "volatility":
                        if val >= threshold:
                            score += 1
                    elif factor == "price_breakout_long":
                        if val >= threshold:
                            score += 1
                    elif factor == "price_breakout_short":
                        if val >= threshold:
                            score += 1
                    elif factor == "rsi_long_max":
                        rsi_val = row.get("rsi", 50)
                        if self.direction == "long" and not pd.isna(rsi_val) and float(rsi_val) <= threshold:
                            score += 1
                    elif factor == "rsi_short_min":
                        rsi_val = row.get("rsi", 50)
                        if self.direction == "short" and not pd.isna(rsi_val) and float(rsi_val) >= threshold:
                            score += 1
                    elif factor == "macd_above_signal":
                        macd_val = row.get("macd_above_signal", 0)
                        if self.direction == "long" and float(macd_val) >= threshold:
                            score += 1
                    elif factor == "macd_below_signal":
                        macd_val = row.get("macd_below_signal", 0)
                        if self.direction == "short" and float(macd_val) >= threshold:
                            score += 1
                    else:
                        if float(val) >= threshold:
                            score += 1
                if score < required:
                    continue

            # 趨勢過濾：僅在價格 > EMA200 做多、< EMA200 做空
            close_val = float(row["close"])
            ema200 = row.get("ema_200")
            if pd.notna(ema200):
                ema200 = float(ema200)
                if self.direction == "long" and close_val <= ema200:
                    continue
                if self.direction == "short" and close_val >= ema200:
                    continue

            close = close_val
            atr = float(row.get("atr", 0.0))
            if atr <= 0 and "volatility" in row.index:
                atr = float(row["volatility"]) * close
            if atr <= 0:
                atr = close * 0.02

            er = self.exit_rules
            if self.direction == "long":
                sl = close - er.sl_atr_mult * atr
                r_dist = close - sl
                if er.tp_r_mult is not None:
                    tp = close + er.tp_r_mult * r_dist
                else:
                    tp = close + er.sl_atr_mult * atr * 2
                if er.tp_fixed_pct is not None:
                    tp = max(tp, close * (1 + er.tp_fixed_pct))
                side = "BUY"
            else:
                sl = close + er.sl_atr_mult * atr
                r_dist = sl - close
                if er.tp_r_mult is not None:
                    tp = close - er.tp_r_mult * r_dist
                else:
                    tp = close - er.sl_atr_mult * atr * 2
                if er.tp_fixed_pct is not None:
                    tp = min(tp, close * (1 - er.tp_fixed_pct))
                side = "SELL"

            signal.iloc[i] = 1
            side_ser.iloc[i] = side
            entry_price.iloc[i] = close
            sl_price.iloc[i] = sl
            tp_price.iloc[i] = tp

        return pd.DataFrame({
            "signal": signal,
            "side": side_ser,
            "entry_price": entry_price,
            "sl_price": sl_price,
            "tp_price": tp_price,
        }, index=df.index)

    def backtest(
        self,
        market_data: pd.DataFrame,
        engine: Any,
        fee_bps: float = 9.0,
        slippage_bps: float = 5.0,
    ) -> Dict[str, Any]:
        """
        market_data: 1h OHLCV + 因子欄位
        engine: 固定回測引擎（介面: engine.run(strategy, market_data, fee_bps, slippage_bps) -> dict）
        """
        return engine.run(self, market_data, fee_bps, slippage_bps)


def compute_atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """計算 ATR 供策略與引擎使用。"""
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev = close.shift(1)
    tr = (high - low).combine((high - prev).abs(), max).combine((low - prev).abs(), max)
    return tr.rolling(period, min_periods=period).mean()
