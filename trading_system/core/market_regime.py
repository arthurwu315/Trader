"""
Market Regime Filter (L1 + L2)
市場環境過濾系統
"""
import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict
from datetime import datetime

from core.structure_detector import StructureDetector

logger = logging.getLogger(__name__)

class MarketRegimeFilter:
    """市場環境過濾器 - L1和L2層"""
    
    def __init__(self, config):
        self.config = config
        self.last_l1_check = None
        self.last_l2_check = None
        self.l1_allow_long = False
        self.l2_allow_execution = False
        
        # L2節流機制 - 防止同一趨勢段過度交易
        self.l2_trade_count_in_regime = 0  # 當前regime的交易次數
        self.l2_last_regime_id = None  # 上個regime標識
        self.l2_max_trades_per_regime = getattr(config, "l2_max_trades_per_regime", 2)
    
    # ==================== L1 方向層 ====================
    
    def check_l1_directional_gate(self, weekly_df: pd.DataFrame, 
                                   daily_df: pd.DataFrame) -> Tuple[bool, str]:
        """
        L1層: 方向過濾
        
        檢查週線和日線趨勢
        只有趨勢向上時才允許交易
        
        Returns:
            (allow_long, reason)
        """
        try:
            # 週線檢查
            weekly_check, weekly_reason = self._check_weekly_trend(weekly_df)
            
            # 日線檢查
            daily_check, daily_reason = self._check_daily_trend(daily_df)
            
            # 兩者都必須通過
            allow_long = weekly_check and daily_check
            
            if allow_long:
                reason = "✓ L1通過: 週線↑ AND 日線↑"
            else:
                reason = f"✗ L1阻擋: {weekly_reason} | {daily_reason}"
            
            self.l1_allow_long = allow_long
            self.last_l1_check = datetime.now()
            
            logger.info(f"L1檢查: {reason}")
            
            return allow_long, reason
            
        except Exception as e:
            logger.error(f"L1檢查錯誤: {e}")
            return False, f"L1錯誤: {e}"
    
    def _check_weekly_trend(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """檢查週線趨勢 - 使用多週期確認避免頻繁切換"""
        if len(df) < self.config.weekly_ema_period + 10:
            return False, "週線數據不足"
        
        # 計算EMA
        ema = df['close'].ewm(span=self.config.weekly_ema_period, adjust=False).mean()
        
        # 當前價格
        current_price = df['close'].iloc[-1]
        current_ema = ema.iloc[-1]
        ema_3w_ago = ema.iloc[-4]  # 3週前 (包含當週共4個點)
        ema_6w_ago = ema.iloc[-7]  # 6週前
        
        relaxed_l1 = getattr(self.config, 'l1_relaxed_mode', False) or getattr(self.config, 'weekly_use_relaxed_filter', False)
        if relaxed_l1:
            # Study模式: 只需6週向上 OR 價格>EMA
            trend_6w = current_ema > ema_6w_ago
            price_above_ema = current_price > current_ema
            growth_6w = (current_ema - ema_6w_ago) / ema_6w_ago if ema_6w_ago > 0 else 0
            
            if trend_6w or price_above_ema:
                return True, f"週線✓(Study:6W成長{growth_6w:+.2%})"
            else:
                return False, "週線✗(Study:6W未向上且價格<EMA)"
        
        # Live模式: 需要3週+6週都向上
        trend_3w = current_ema > ema_3w_ago  # 3週向上
        trend_6w = current_ema > ema_6w_ago  # 6週向上
        
        # 計算3週增長率 (用於顯示)
        growth_3w = (current_ema - ema_3w_ago) / ema_3w_ago if ema_3w_ago > 0 else 0
        
        # 檢查2: 價格在EMA上方（或允許一定容忍）
        tolerance = float(getattr(self.config, "l1_price_tolerance_pct", 0.0))
        price_above_ema = current_price >= current_ema * (1 - tolerance)
        
        # 兩個條件都要滿足
        if trend_3w and trend_6w and price_above_ema:
            return True, f"週線✓(Live:3W成長{growth_3w:+.2%})"
        elif not (trend_3w and trend_6w):
            return False, f"週線✗(Live:EMA未確認向上)"
        else:
            return False, "週線✗(Live:價格<EMA)"
    
    def _check_daily_trend(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """檢查日線趨勢 - 使用多日確認避免頻繁切換"""
        if len(df) < self.config.daily_ema_period + 10:
            return False, "日線數據不足"
        
        # 計算EMA
        ema = df['close'].ewm(span=self.config.daily_ema_period, adjust=False).mean()
        
        # 當前價格
        current_price = df['close'].iloc[-1]
        current_ema = ema.iloc[-1]
        ema_3d_ago = ema.iloc[-4]  # 3天前
        ema_5d_ago = ema.iloc[-6]  # 5天前
        
        # 檢查1: 多日EMA向上確認
        trend_3d = current_ema > ema_3d_ago  # 3天向上
        trend_5d = current_ema > ema_5d_ago  # 5天向上
        
        # 計算3日增長率
        growth_3d = (current_ema - ema_3d_ago) / ema_3d_ago if ema_3d_ago > 0 else 0
        
        # 檢查2: 價格在EMA上方（或允許一定容忍）
        tolerance = float(getattr(self.config, "l1_price_tolerance_pct", 0.0))
        price_above_ema = current_price >= current_ema * (1 - tolerance)
        
        relaxed_l1 = getattr(self.config, 'l1_relaxed_mode', False) or getattr(self.config, 'daily_use_relaxed_filter', False)
        if relaxed_l1:
            if trend_3d or price_above_ema:
                return True, f"日線✓(Relaxed:3D成長{growth_3d:+.2%})"
            return False, "日線✗(Relaxed:EMA未向上且價格<EMA)"

        if trend_3d and trend_5d and price_above_ema:
            return True, f"日線✓(3D成長{growth_3d:+.2%})"
        elif not (trend_3d and trend_5d):
            return False, "日線✗(EMA未確認向上)"
        else:
            return False, "日線✗(價格<EMA)"
    
    # ==================== L2 環境層 ====================
    
    def check_l2_regime_filter(self, h4_df: pd.DataFrame, 
                                daily_df: pd.DataFrame) -> Tuple[bool, str]:
        """
        L2層: 環境過濾
        
        檢查4H環境是否適合執行
        滿足以下任一組條件即可:
        A. 波動收斂
        B. 趨勢延續
        
        加入節流機制: 同一regime最多允許N筆交易
        
        Returns:
            (allow_execution, reason)
        """
        if not self.l1_allow_long:
            return False, "L2跳過: L1未通過"
        
        try:
            # 檢查A: 波動收斂
            volatility_check, vol_reason = self._check_volatility_contraction(h4_df)
            
            # 檢查B: 趨勢延續
            trend_check, trend_reason = self._check_trend_continuation(h4_df)
            
            # 任一通過即可
            basic_allow = volatility_check or trend_check
            
            if not basic_allow:
                # 基本條件未通過,重置計數
                self.l2_trade_count_in_regime = 0
                self.l2_last_regime_id = None
                reason = f"✗ L2阻擋: {vol_reason} AND {trend_reason}"
                self.l2_allow_execution = False
                self.last_l2_check = datetime.now()
                logger.info(f"L2檢查: {reason}")
                return False, reason
            # 基本條件通過,檢查節流
            current_regime_id = self._get_regime_id(h4_df)
            
            # 檢查是否新的regime
            if current_regime_id != self.l2_last_regime_id:
                # 新regime,重置計數
                self.l2_trade_count_in_regime = 0
                self.l2_last_regime_id = current_regime_id
                logger.info(f"L2: 檢測到新regime {current_regime_id},重置交易計數")
            
            # 檢查是否超過交易次數限制
            if self.l2_trade_count_in_regime >= self.l2_max_trades_per_regime:
                reason = f"✗ L2節流: 本regime已交易{self.l2_trade_count_in_regime}次,達上限"
                self.l2_allow_execution = False
                self.last_l2_check = datetime.now()
                logger.info(f"L2檢查: {reason}")
                return False, reason
            
            # 通過所有檢查
            reasons = []
            if volatility_check:
                reasons.append(vol_reason)
            if trend_check:
                reasons.append(trend_reason)
            
            reason = f"✓ L2通過: {' + '.join(reasons)} (本regime {self.l2_trade_count_in_regime}/{self.l2_max_trades_per_regime})"
            
            self.l2_allow_execution = True
            self.last_l2_check = datetime.now()
            
            logger.info(f"L2檢查: {reason}")
            
            return True, reason
            
        except Exception as e:
            logger.error(f"L2檢查錯誤: {e}")
            return False, f"L2錯誤: {e}"
    
    def _get_regime_id(self, df: pd.DataFrame) -> str:
        """
        獲取當前regime標識
        使用4H EMA20的位置和斜率判斷
        """
        if len(df) < 30:
            return "unknown"
        
        # 計算EMA20
        ema20 = df['close'].ewm(span=20, adjust=False).mean()
        current_ema = ema20.iloc[-1]
        ema_10bars_ago = ema20.iloc[-11]
        
        # 計算斜率
        slope = (current_ema - ema_10bars_ago) / ema_10bars_ago
        
        # 根據斜率分區
        if slope > 0.02:
            return f"strong_up_{int(current_ema/1000)}"  # 強上升
        elif slope > 0.005:
            return f"up_{int(current_ema/1000)}"  # 上升
        elif slope > -0.005:
            return f"flat_{int(current_ema/1000)}"  # 盤整
        else:
            return f"down_{int(current_ema/1000)}"  # 下降
    
    def notify_trade_executed(self):
        """
        通知L2層有交易執行
        由主程式在實際進場後調用
        """
        self.l2_trade_count_in_regime += 1
        logger.info(f"L2: 記錄交易,本regime已{self.l2_trade_count_in_regime}筆")
    
    def _check_volatility_contraction(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """檢查波動收斂"""
        if len(df) < self.config.atr_lookback_periods + 14:
            return False, "波動✗(數據不足)"
        
        # 計算ATR
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        
        # 方法1: ATR連續下降
        decline_bars = self.config.atr_decline_bars
        if getattr(self.config, "l2_relaxed_mode", False):
            decline_bars = max(2, decline_bars - 2)
        recent_atr = atr.iloc[-decline_bars:]
        is_declining = all(recent_atr.iloc[i] < recent_atr.iloc[i-1] 
                          for i in range(1, len(recent_atr)))
        
        # 方法2: ATR低於歷史中位數
        historical_atr = atr.iloc[-self.config.atr_lookback_periods:]
        current_atr = atr.iloc[-1]
        atr_percentile = (historical_atr < current_atr).sum() / len(historical_atr) * 100
        threshold = self.config.atr_percentile_threshold
        if getattr(self.config, "l2_relaxed_mode", False):
            threshold = min(80.0, threshold + 20)
        is_low = atr_percentile < threshold
        
        if is_declining:
            return True, f"波動收斂✓(連續{self.config.atr_decline_bars}根下降)"
        elif is_low:
            return True, f"波動低位✓({atr_percentile:.0f}分位)"
        else:
            return False, f"波動✗({atr_percentile:.0f}分位)"
    
    def _check_trend_continuation(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """檢查趨勢延續"""
        if len(df) < self.config.regime_ema_period + 5:
            return False, "趨勢✗(數據不足)"
        
        # 計算EMA20
        ema20 = df['close'].ewm(span=self.config.regime_ema_period, adjust=False).mean()
        current_price = df['close'].iloc[-1]
        current_ema = ema20.iloc[-1]
        
        # 檢查1: 價格在EMA20上方
        price_above_ema = current_price > current_ema
        tolerance_pct = getattr(self.config, "regime_ema_tolerance_pct", 0.0)
        relaxed_l2 = getattr(self.config, "l2_relaxed_mode", False)
        price_near_ema = current_price >= current_ema * (1 - tolerance_pct)
        
        # 檢查2: 沒有結構破壞(未形成Lower Low)
        if self.config.structure_break_enabled:
            structure_intact = self._check_structure_intact(df)
        else:
            structure_intact = True
        
        if (price_above_ema or (relaxed_l2 and price_near_ema)) and structure_intact:
            distance_pct = (current_price - current_ema) / current_ema * 100
            return True, f"趨勢延續✓(價格>EMA {distance_pct:.2f}%)"
        elif not price_above_ema:
            if relaxed_l2 and price_near_ema:
                return True, f"趨勢延續✓(Relaxed:價格接近EMA {distance_pct:.2f}%)"
            return False, "趨勢✗(價格<EMA)"
        else:
            return False, "趨勢✗(結構破壞)"
    
    def _check_structure_intact(self, df: pd.DataFrame) -> bool:
        """檢查結構是否完整(未形成LL)"""
        # 找最近的兩個低點
        lows = df['low'].iloc[-20:]
        
        if len(lows) < 10:
            return True  # 數據不足,假設完整
        
        # 簡化判斷: 最近低點不低於前一個低點
        recent_low = lows.iloc[-5:].min()
        previous_low = lows.iloc[-15:-5].min()
        
        return recent_low >= previous_low * 0.98  # 允許2%誤差
    
    # ==================== 輔助方法 ====================
    
    def get_status(self) -> Dict:
        """獲取當前狀態"""
        return {
            'l1_allow_long': self.l1_allow_long,
            'l1_last_check': self.last_l1_check,
            'l2_allow_execution': self.l2_allow_execution,
            'l2_last_check': self.last_l2_check,
            'overall_status': 'READY' if (self.l1_allow_long and self.l2_allow_execution) else 'BLOCKED'
        }
    
    def get_status_string(self) -> str:
        """獲取狀態字串"""
        status = self.get_status()
        
        l1_icon = "🟢" if status['l1_allow_long'] else "🔴"
        l2_icon = "🟢" if status['l2_allow_execution'] else "🔴"
        overall_icon = "✅" if status['overall_status'] == 'READY' else "⛔"
        
        return f"{overall_icon} L1:{l1_icon} L2:{l2_icon}"


@dataclass
class RegimeDecision:
    allow: bool
    reason: str
    regime_label: str
    risk_multiplier: float = 1.0
    signal: Optional[object] = None
    details: Optional[Dict] = None


class MarketRegimeDetector:
    """
    混合式市況偵測器
    L1/L2: 趨勢/波動
    L3: 結構訊號
    """

    def __init__(self, config, market_data, require_structure: bool = True):
        self.config = config
        self.market_data = market_data
        self.regime_filter = MarketRegimeFilter(config)
        self.structure_detector = StructureDetector(config)
        self.require_structure = require_structure

    def evaluate(self, symbol: str) -> RegimeDecision:
        weekly_df = self.market_data.get_klines_df(symbol, "1w", limit=200)
        daily_df = self.market_data.get_klines_df(symbol, "1d", limit=200)
        h4_df = self.market_data.get_klines_df(symbol, "4h", limit=300)
        m15_df = self.market_data.get_klines_df(symbol, "15m", limit=300)

        l1_allow, l1_reason = self.regime_filter.check_l1_directional_gate(
            weekly_df, daily_df
        )
        l2_allow, l2_reason = self.regime_filter.check_l2_regime_filter(
            h4_df, daily_df
        )

        structure_signal = self.structure_detector.detect_entry_setup(m15_df)

        if not l1_allow:
            return RegimeDecision(
                allow=False,
                reason=l1_reason,
                regime_label="blocked_l1",
                risk_multiplier=0.0,
                signal=structure_signal,
                details={"l1_reason": l1_reason, "l2_reason": l2_reason, "structure_reason": structure_signal.reason},
            )

        if not l2_allow:
            return RegimeDecision(
                allow=False,
                reason=l2_reason,
                regime_label="blocked_l2",
                risk_multiplier=0.0,
                signal=structure_signal,
                details={"l1_reason": l1_reason, "l2_reason": l2_reason, "structure_reason": structure_signal.reason},
            )

        if self.require_structure and not structure_signal.entry_allowed:
            return RegimeDecision(
                allow=False,
                reason=structure_signal.reason,
                regime_label="wait_structure",
                risk_multiplier=0.0,
                signal=structure_signal,
                details={"l1_reason": l1_reason, "l2_reason": l2_reason, "structure_reason": structure_signal.reason},
            )

        reason = "L1/L2通過" if not self.require_structure else "L1/L2/L3通過"
        return RegimeDecision(
            allow=True,
            reason=reason,
            regime_label="allow_trade",
            risk_multiplier=1.0,
            signal=structure_signal,
            details={"l1_reason": l1_reason, "l2_reason": l2_reason, "structure_reason": structure_signal.reason},
        )

# 測試
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 創建測試數據
    dates = pd.date_range(end=datetime.now(), periods=200, freq='W')
    weekly_df = pd.DataFrame({
        'close': np.linspace(80000, 92000, 200) + np.random.randn(200) * 1000,
        'high': np.linspace(81000, 93000, 200) + np.random.randn(200) * 1000,
        'low': np.linspace(79000, 91000, 200) + np.random.randn(200) * 1000,
    }, index=dates)
    
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    daily_df = pd.DataFrame({
        'close': np.linspace(88000, 92000, 100) + np.random.randn(100) * 500,
        'high': np.linspace(88500, 92500, 100) + np.random.randn(100) * 500,
        'low': np.linspace(87500, 91500, 100) + np.random.randn(100) * 500,
    }, index=dates)
    
    dates = pd.date_range(end=datetime.now(), periods=240, freq='4H')
    h4_df = pd.DataFrame({
        'close': np.linspace(90000, 92000, 240) + np.random.randn(240) * 300,
        'high': np.linspace(90300, 92300, 240) + np.random.randn(240) * 300,
        'low': np.linspace(89700, 91700, 240) + np.random.randn(240) * 300,
    }, index=dates)
    
    # 測試
    from config_v3 import get_config
    config = get_config()
    
    regime_filter = MarketRegimeFilter(config)
    
    # L1檢查
    l1_pass, l1_reason = regime_filter.check_l1_directional_gate(weekly_df, daily_df)
    print(f"\nL1結果: {l1_pass}")
    print(f"L1原因: {l1_reason}")
    
    # L2檢查
    l2_pass, l2_reason = regime_filter.check_l2_regime_filter(h4_df, daily_df)
    print(f"\nL2結果: {l2_pass}")
    print(f"L2原因: {l2_reason}")
    
    # 狀態
    print(f"\n狀態: {regime_filter.get_status_string()}")
