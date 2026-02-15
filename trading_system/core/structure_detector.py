"""
Structure Detector (L3)
15分鐘結構檢測與執行層
"""
import logging
import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class StructureSignal:
    """結構訊號"""
    symbol: str
    signal_type: str  # 'breakout' or 'pullback'
    entry_allowed: bool
    entry_price: float
    stop_loss: float
    reason: str
    
    # 結構細節
    structure_high: Optional[float] = None
    structure_low: Optional[float] = None
    ema20: Optional[float] = None
    vwap: Optional[float] = None
    
    # 風險指標
    stop_distance_pct: float = 0
    atr: float = 0
    
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class StructureDetector:
    """L3執行層 - 結構檢測器"""
    
    def __init__(self, config):
        self.config = config
        self.last_signal = None
    
    def detect_entry_setup(self, m15_df: pd.DataFrame, 
                           m5_df: Optional[pd.DataFrame] = None) -> StructureSignal:
        """
        檢測進場設置
        
        主要在15m上檢測結構
        可選用5m做精確確認
        
        Args:
            m15_df: 15分鐘K線數據
            m5_df: 5分鐘K線數據(可選)
        
        Returns:
            StructureSignal
        """
        symbol = self.config.symbol
        
        try:
            # 檢查數據充足性
            if len(m15_df) < self.config.structure_lookback_bars + 20:
                return StructureSignal(
                    symbol=symbol,
                    signal_type='none',
                    entry_allowed=False,
                    entry_price=0,
                    stop_loss=0,
                    reason="數據不足"
                )
            
            # 計算技術指標
            indicators = self._calculate_indicators(m15_df)
            
            # 識別當前結構
            structure = self._identify_structure(m15_df, indicators)
            
            # 檢測進場型態
            signal = self._detect_entry_pattern(m15_df, indicators, structure)
            
            self.last_signal = signal
            return signal
            
        except Exception as e:
            logger.error(f"結構檢測錯誤: {e}")
            return StructureSignal(
                symbol=symbol,
                signal_type='error',
                entry_allowed=False,
                entry_price=0,
                stop_loss=0,
                reason=f"錯誤: {e}"
            )
    
    def _calculate_indicators(self, df: pd.DataFrame) -> Dict:
        """計算技術指標"""
        # EMA20
        ema20 = df['close'].ewm(span=self.config.execution_ema_period, adjust=False).mean()
        
        # VWAP (簡化版:當日均價加權)
        if 'volume' in df.columns:
            vwap = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        else:
            vwap = df['close']  # fallback
        
        # ATR
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        
        return {
            'ema20': ema20,
            'vwap': vwap,
            'atr': atr,
            'current_price': df['close'].iloc[-1],
            'current_high': df['high'].iloc[-1],
            'current_low': df['low'].iloc[-1]
        }
    
    def _identify_structure(self, df: pd.DataFrame, indicators: Dict) -> Dict:
        """
        識別結構 (HH, HL, LH, LL) - 機械化定義
        
        Swing點定義:
        - swing_high: 高點高於左右各N根
        - swing_low: 低點低於左右各N根
        
        趨勢結構:
        - HH: 最新swing_high > 前一個swing_high
        - HL: 最新swing_low > 前一個swing_low
        """
        lookback = self.config.structure_lookback_bars
        recent_df = df.iloc[-lookback:]
        
        swing_window = 2  # 左右各2根確認swing點
        
        # 找出所有swing highs
        swing_highs = []
        for i in range(swing_window, len(recent_df) - swing_window):
            high = recent_df['high'].iloc[i]
            left_highs = recent_df['high'].iloc[i-swing_window:i]
            right_highs = recent_df['high'].iloc[i+1:i+swing_window+1]
            
            if high > left_highs.max() and high > right_highs.max():
                swing_highs.append({
                    'index': i,
                    'price': high,
                    'time': recent_df.index[i]
                })
        
        # 找出所有swing lows
        swing_lows = []
        for i in range(swing_window, len(recent_df) - swing_window):
            low = recent_df['low'].iloc[i]
            left_lows = recent_df['low'].iloc[i-swing_window:i]
            right_lows = recent_df['low'].iloc[i+1:i+swing_window+1]
            
            if low < left_lows.min() and low < right_lows.min():
                swing_lows.append({
                    'index': i,
                    'price': low,
                    'time': recent_df.index[i]
                })
        
        # 分析結構
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            # 最近兩個swing高點
            recent_high = swing_highs[-1]['price']
            previous_high = swing_highs[-2]['price']
            
            # 最近兩個swing低點
            recent_low = swing_lows[-1]['price']
            previous_low = swing_lows[-2]['price']
            
            # 判斷趨勢結構
            is_higher_high = recent_high > previous_high
            is_higher_low = recent_low > previous_low
            
            if is_higher_high and is_higher_low:
                structure_type = "uptrend"  # HH + HL = 上升趨勢
            elif is_higher_high and not is_higher_low:
                structure_type = "consolidation"  # HH + LL = 盤整
            else:
                structure_type = "weak"  # 非上升結構
            
            return {
                'type': structure_type,
                'recent_high': recent_high,
                'recent_low': recent_low,
                'previous_high': previous_high,
                'previous_low': previous_low,
                'swing_highs': swing_highs,
                'swing_lows': swing_lows
            }
        else:
            # 結構點不足,使用簡化方法
            recent_high = recent_df['high'].iloc[-10:].max()
            recent_low = recent_df['low'].iloc[-10:].min()
            previous_high = recent_df['high'].iloc[-20:-10].max()
            previous_low = recent_df['low'].iloc[-20:-10].min()
            
            is_higher_high = recent_high > previous_high
            is_higher_low = recent_low > previous_low
            
            structure_type = "uptrend" if (is_higher_high and is_higher_low) else "weak"
            
            return {
                'type': structure_type,
                'recent_high': recent_high,
                'recent_low': recent_low,
                'previous_high': previous_high,
                'previous_low': previous_low,
                'swing_highs': [],
                'swing_lows': []
            }
    
    def _detect_entry_pattern(self, df: pd.DataFrame, 
                              indicators: Dict, 
                              structure: Dict) -> StructureSignal:
        """檢測進場型態"""
        
        symbol = self.config.symbol
        current_price = indicators['current_price']
        atr = indicators['atr'].iloc[-1]
        
        # 先檢查結構
        allow_weak = bool(getattr(self.config, "allow_weak_structure", False))
        if structure['type'] == 'weak' and not allow_weak:
            return StructureSignal(
                symbol=symbol,
                signal_type='none',
                entry_allowed=False,
                entry_price=current_price,
                stop_loss=0,
                reason="結構不清晰",
                atr=atr
            )
        if structure['type'] == 'weak' and allow_weak:
            logger.warning("⚠️ 弱結構放行（allow_weak_structure=True）")
            structure['type'] = 'consolidation'
        
        # 型態1: 突破進場
        if getattr(self.config, "allow_breakout_entry", True):
            breakout_signal = self._check_breakout_entry(df, indicators, structure)
            if breakout_signal['valid']:
                return self._create_signal_from_setup(
                    symbol, 'breakout', breakout_signal, indicators, structure
                )
        
        # 型態2: 回踩進場
        if getattr(self.config, "allow_pullback_entry", True):
            pullback_signal = self._check_pullback_entry(df, indicators, structure)
            if pullback_signal['valid']:
                return self._create_signal_from_setup(
                    symbol, 'pullback', pullback_signal, indicators, structure
                )
        
        # 沒有訊號
        return StructureSignal(
            symbol=symbol,
            signal_type='none',
            entry_allowed=False,
            entry_price=current_price,
            stop_loss=0,
            reason="等待結構確認",
            atr=atr
        )
    
    def _check_breakout_entry(self, df: pd.DataFrame, 
                              indicators: Dict, 
                              structure: Dict) -> Dict:
        """檢查突破進場"""
        current_price = indicators['current_price']
        resistance = structure['recent_high']
        
        # 突破位 = 前高 + 緩衝
        breakout_level = resistance * (1 + self.config.breakout_buffer_pct)
        
        # 檢查是否突破
        if current_price > breakout_level:
            # 檢查回踩是否守住
            recent_lows = df['low'].iloc[-5:]
            lowest_after_break = recent_lows.min()
            
            # 止損放在近期低點下方
            stop_loss = structure['recent_low'] * 0.998
            stop_distance_pct = abs(current_price - stop_loss) / current_price
            
            # 檢查止損距離是否合理
            if stop_distance_pct > self.config.leverage_tier_3_max_sl:
                return {'valid': False, 'reason': '止損距離過大'}
            
            return {
                'valid': True,
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'stop_distance_pct': stop_distance_pct,
                'reason': f"突破{resistance:.2f},回踩守住"
            }
        
        return {'valid': False, 'reason': '未突破前高'}
    
    def _check_pullback_entry(self, df: pd.DataFrame, 
                              indicators: Dict, 
                              structure: Dict) -> Dict:
        """檢查回踩進場"""
        current_price = indicators['current_price']
        ema20 = indicators['ema20'].iloc[-1]
        vwap = indicators['vwap'].iloc[-1]
        
        # 檢查是否在上升結構中
        if structure['type'] != 'uptrend':
            return {'valid': False, 'reason': '非上升結構'}
        
        # 檢查價格是否接近支撐
        near_ema = abs(current_price - ema20) / ema20 < 0.005  # 0.5%內
        near_vwap = abs(current_price - vwap) / vwap < 0.005
        
        if not (near_ema or near_vwap):
            return {'valid': False, 'reason': '未回踩支撐'}
        
        # 檢查是否在結構高低點之間
        in_range = structure['recent_low'] < current_price < structure['recent_high']
        
        if not in_range:
            return {'valid': False, 'reason': '不在結構範圍'}
        
        # 檢查最近是否有反彈確認
        recent_closes = df['close'].iloc[-3:]
        is_bouncing = recent_closes.iloc[-1] > recent_closes.iloc[-2]
        
        if not is_bouncing:
            return {'valid': False, 'reason': '未確認反彈'}
        
        # 止損放在結構低點下方
        stop_loss = structure['recent_low'] * 0.998
        stop_distance_pct = abs(current_price - stop_loss) / current_price
        
        # 檢查止損距離
        if stop_distance_pct > self.config.leverage_tier_3_max_sl:
            return {'valid': False, 'reason': '止損距離過大'}
        
        support_level = ema20 if near_ema else vwap
        
        return {
            'valid': True,
            'entry_price': current_price,
            'stop_loss': stop_loss,
            'stop_distance_pct': stop_distance_pct,
            'reason': f"回踩{'EMA20' if near_ema else 'VWAP'}@{support_level:.2f}反彈"
        }
    
    def _create_signal_from_setup(self, symbol: str, signal_type: str,
                                   setup: Dict, indicators: Dict,
                                   structure: Dict) -> StructureSignal:
        """從設置創建訊號"""
        return StructureSignal(
            symbol=symbol,
            signal_type=signal_type,
            entry_allowed=True,
            entry_price=setup['entry_price'],
            stop_loss=setup['stop_loss'],
            reason=setup['reason'],
            structure_high=structure['recent_high'],
            structure_low=structure['recent_low'],
            ema20=indicators['ema20'].iloc[-1],
            vwap=indicators['vwap'].iloc[-1],
            stop_distance_pct=setup['stop_distance_pct'],
            atr=indicators['atr'].iloc[-1]
        )

# 測試
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 創建測試數據 - 模擬上升趨勢
    dates = pd.date_range(end=datetime.now(), periods=100, freq='15min')
    
    # 創建HH + HL結構
    base_price = 91000
    prices = []
    for i in range(100):
        if i < 30:
            price = base_price + i * 10 + np.random.randn() * 50
        elif i < 50:  # 回踩
            price = base_price + 300 - (i-30) * 5 + np.random.randn() * 50
        else:  # 突破
            price = base_price + 200 + (i-50) * 15 + np.random.randn() * 50
        prices.append(price)
    
    m15_df = pd.DataFrame({
        'close': prices,
        'high': [p + np.random.rand() * 100 for p in prices],
        'low': [p - np.random.rand() * 100 for p in prices],
        'volume': np.random.randint(100, 1000, 100)
    }, index=dates)
    
    # 測試
    from config_v3 import get_config
    config = get_config()
    
    detector = StructureDetector(config)
    signal = detector.detect_entry_setup(m15_df)
    
    print(f"\n訊號類型: {signal.signal_type}")
    print(f"允許進場: {signal.entry_allowed}")
    print(f"進場價格: ${signal.entry_price:,.2f}")
    print(f"止損價格: ${signal.stop_loss:,.2f}")
    print(f"止損距離: {signal.stop_distance_pct:.2%}")
    print(f"原因: {signal.reason}")
    
    if signal.entry_allowed:
        print(f"\n結構資訊:")
        print(f"  結構高點: ${signal.structure_high:,.2f}")
        print(f"  結構低點: ${signal.structure_low:,.2f}")
        print(f"  EMA20: ${signal.ema20:,.2f}")
        print(f"  ATR: ${signal.atr:.2f}")
