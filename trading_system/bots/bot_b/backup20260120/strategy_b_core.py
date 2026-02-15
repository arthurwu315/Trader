"""
Strategy B: Micro Momentum (3m/15m) - V5.2 修正版
老手B3規格 - 狀態機實作

修正問題:
1. ✅ HH判斷用[-N:-1]不包含當前K
2. ✅ Breakout與Pullback分離(狀態機)
3. ✅ 移除同根K互斥條件
4. ✅ 進場不更新勝負狀態
5. ✅ 成本內建Gate
6. ✅ 止損距離使用config參數 (V5.3修正)
7. ✅ 版本號同步 (V5.2)
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


# ==================== 狀態機定義 ====================

class SetupState(Enum):
    """Setup狀態"""
    IDLE = "IDLE"                          # 無setup
    BREAKOUT_DETECTED = "BREAKOUT"         # 檢測到突破
    PULLBACK_WAITING = "PULLBACK_WAIT"     # 等待回踩
    CONFIRMED = "CONFIRMED"                # 確認完成,可進場


@dataclass
class BreakoutSetup:
    """Breakout Setup狀態記錄"""
    state: SetupState = SetupState.IDLE
    
    # Breakout資訊
    breakout_level: Optional[float] = None
    breakout_time: Optional[datetime] = None
    breakout_swing_low: Optional[float] = None
    breakout_bar_index: Optional[int] = None
    
    # Pullback資訊
    pullback_touched: bool = False
    pullback_low: Optional[float] = None
    
    # 確認資訊
    confirmed: bool = False
    confirm_time: Optional[datetime] = None
    
    def reset(self):
        """重置setup"""
        self.state = SetupState.IDLE
        self.breakout_level = None
        self.breakout_time = None
        self.breakout_swing_low = None
        self.breakout_bar_index = None
        self.pullback_touched = False
        self.pullback_low = None
        self.confirmed = False
        self.confirm_time = None


# ==================== 數據結構 ====================

@dataclass
class StrategyBSignal:
    """策略B訊號"""
    signal_type: str  # "LONG" / "SHORT"
    pattern: str      # "BREAKOUT_PULLBACK" / "EMA_MOMENTUM"
    entry_price: float
    stop_loss: float
    tp1_price: float
    stop_distance_pct: float
    expected_tp1_pct: float
    confidence: float
    reason: str
    timestamp: datetime
    
    # 15m環境
    ema20_15m: float
    
    # 3m資訊
    ema9_3m: float
    ema20_3m: float
    breakout_level: Optional[float] = None
    swing_low: Optional[float] = None


@dataclass
class StrategyBState:
    """策略B狀態"""
    # 交易計數
    trades_today: int = 0
    trades_this_hour: int = 0
    last_trade_time: Optional[datetime] = None
    last_hour_reset: Optional[datetime] = None
    
    # 連虧追蹤
    consecutive_losses: int = 0
    consecutive_wins: int = 0
    last_trade_result: Optional[str] = None  # "WIN"/"LOSS"
    
    # 冷卻
    in_cooldown: bool = False
    cooldown_until: Optional[datetime] = None


# ==================== L0 Gate ====================

class L0Gate:
    """L0 Gate - 保命層"""
    
    def __init__(self, config):
        self.config = config
        
    def check(
        self,
        state: StrategyBState,
        execution_safety,  # OrderStateMachine物件
        has_lock: bool,
        has_emergency: bool,
        has_position: bool
    ) -> Tuple[bool, str]:
        """
        L0檢查
        
        Args:
            state: 策略狀態
            execution_safety: OrderStateMachine物件
            has_lock: 是否有全局鎖
            has_emergency: 是否有緊急標記
            has_position: 是否有倉位
        """
        
        logger.info("\n" + "="*60)
        logger.info("🔒 L0 Gate 檢查")
        logger.info("="*60)
        
        # 1. 系統安全
        logger.info("\n📋 系統安全檢查:")
        
        if has_lock:
            return False, "L0_LOCKED"
        logger.info("  ✅ 無全局鎖")
        
        if has_emergency:
            return False, "L0_EMERGENCY"
        logger.info("  ✅ 無緊急標記")
        
        # ✅ 執行安全檢查 (必須有is_safe()方法)
        if not hasattr(execution_safety, "is_safe"):
            return False, "L0_EXECUTION_SAFETY_MISSING"
        
        if not execution_safety.is_safe():
            return False, "L0_EXECUTION_UNSAFE"
        
        logger.info("  ✅ 執行安全OK")
        
        # 2. 倉位互斥
        if has_position:
            logger.warning("  ❌ 已有倉位(互斥模式)")
            return False, "CROSS_STRATEGY_EXCLUSIVE"
        logger.info("  ✅ 無倉位衝突")
        
        # 3. 每日限制
        logger.info(f"\n📋 頻率限制:")
        logger.info(f"  今日交易: {state.trades_today}/{self.config.max_trades_per_day}")
        
        if state.trades_today >= self.config.max_trades_per_day:
            return False, "L0_DAILY_LIMIT"
        logger.info("  ✅ 未達每日限制")
        
        # 4. 每小時限制
        now = datetime.now()
        
        if state.last_hour_reset is None or \
           (now - state.last_hour_reset).total_seconds() >= 3600:
            state.trades_this_hour = 0
            state.last_hour_reset = now
        
        logger.info(f"  本小時交易: {state.trades_this_hour}/{self.config.max_trades_per_hour}")
        
        if state.trades_this_hour >= self.config.max_trades_per_hour:
            return False, "L0_HOURLY_LIMIT"
        logger.info("  ✅ 未達小時限制")
        
        # 5. 連虧冷卻
        logger.info(f"\n📋 連虧檢查:")
        logger.info(f"  連續虧損: {state.consecutive_losses}")
        
        if state.in_cooldown:
            if state.cooldown_until and now < state.cooldown_until:
                remaining = (state.cooldown_until - now).total_seconds() / 60
                logger.warning(f"  ❌ 冷卻中,剩餘{remaining:.1f}分鐘")
                return False, "L0_COOLDOWN"
            else:
                # 冷卻結束
                state.in_cooldown = False
                state.cooldown_until = None
                logger.info("  ✅ 冷卻期已過")
        
        logger.info("  ✅ 無冷卻限制")
        
        logger.info("\n✅ L0 Gate 通過")
        logger.info("="*60)
        
        return True, "L0_PASS"


# ==================== L1 Gate ====================

class L1Gate:
    """L1 Gate - 15m環境"""
    
    def __init__(self, config):
        self.config = config
        
    def check_long_environment(self, market_data) -> Tuple[bool, str, Dict]:
        """檢查15m多頭環境"""
        
        logger.info("\n" + "="*60)
        logger.info("🌍 L1 Gate: 15m環境檢查")
        logger.info("="*60)
        
        debug = {}
        
        try:
            df_15m = market_data.get_klines_df(
                symbol=self.config.symbol,
                interval='15m',
                limit=50
            )
            
            if df_15m is None or len(df_15m) < 30:
                return False, "L1_DATA_INSUFFICIENT", {}
            
            # 計算EMA20
            ema20 = self._calculate_ema(df_15m['close'], 20)
            ema20_current = ema20.iloc[-1]
            current_price = df_15m['close'].iloc[-1]
            
            debug['ema20_15m'] = ema20_current
            debug['price'] = current_price
            
            # 1. 價格 > EMA20
            logger.info(f"\n1️⃣ 價格位置:")
            logger.info(f"  價格: ${current_price:.2f}")
            logger.info(f"  EMA20: ${ema20_current:.2f}")
            
            if current_price <= ema20_current:
                logger.warning("  ❌ 價格未在EMA20上方")
                return False, "L1_15M_NOT_UPTREND", debug
            logger.info("  ✅ 價格在EMA20上方")
            
            # 2. EMA20斜率向上
            logger.info(f"\n2️⃣ EMA20斜率:")
            
            ema20_3 = ema20.iloc[-3:].values
            
            if not (ema20_3[2] > ema20_3[1] > ema20_3[0]):
                logger.warning("  ❌ EMA20未連續向上")
                return False, "L1_EMA_NOT_RISING", debug
            logger.info("  ✅ EMA20連續向上")
            
            # 3. 結構完整(最近swing low未破)
            logger.info(f"\n3️⃣ 結構檢查:")
            
            swing_low = self._find_last_swing_low(df_15m)
            
            if swing_low is None:
                logger.info("  ⚠️ 未找到swing low,放行")
                return True, "L1_PASS", debug
            
            if current_price <= swing_low:
                logger.warning(f"  ❌ 跌破swing low ${swing_low:.2f}")
                return False, "L1_STRUCTURE_BROKEN", debug
            
            logger.info(f"  ✅ 結構完整(swing low: ${swing_low:.2f})")
            
            logger.info("\n✅ L1 Gate 通過")
            logger.info("="*60)
            
            return True, "L1_PASS", debug
            
        except Exception as e:
            logger.error(f"L1檢查失敗: {e}", exc_info=True)
            return False, f"L1_ERROR: {str(e)}", {}
    
    def _calculate_ema(self, series, period):
        """計算EMA"""
        return series.ewm(span=period, adjust=False).mean()
    
    def _find_last_swing_low(self, df, lookback=10):
        """找最近swing low (fractal: 左右各2根)"""
        if len(df) < 5:
            return None
        
        lows = df['low'].values[-lookback:]
        
        for i in range(len(lows)-3, 1, -1):
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
               lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                return lows[i]
        
        return None


# ==================== L2 Gate - 狀態機版本 ====================

class L2Gate:
    """L2 Gate - 3m進場邏輯(狀態機)"""
    
    def __init__(self, config):
        self.config = config
        
        # 狀態機
        self.setup = BreakoutSetup()
        
        # 參數
        self.breakout_lookback = 20  # 1小時
        self.breakout_buffer = 0.0002  # 0.02%
        self.retest_buffer = 0.0001  # 0.01%
        self.pullback_max_bars = 12  # 36分鐘
        
    def check_entry_pattern(self, market_data, l1_passed: bool, bar_index: int) -> Tuple[bool, str, Optional[StrategyBSignal]]:
        """
        檢查3m進場型態(狀態機)
        
        Args:
            market_data: 市場數據
            l1_passed: L1是否通過
            bar_index: 當前K線索引(用於追蹤時間)
        """
        
        if not l1_passed:
            self.setup.reset()
            return False, "L1_NOT_PASSED", None
        
        logger.info("\n" + "="*60)
        logger.info("🎯 L2 Gate: 3m型態檢查(狀態機)")
        logger.info(f"當前狀態: {self.setup.state.value}")
        logger.info("="*60)
        
        try:
            df_3m = market_data.get_klines_df(
                symbol=self.config.symbol,
                interval='3m',
                limit=100
            )
            
            if df_3m is None or len(df_3m) < 30:
                return False, "L2_DATA_INSUFFICIENT", None
            
            # 計算指標
            ema9 = self._calculate_ema(df_3m['close'], 9).iloc[-1]
            ema20 = self._calculate_ema(df_3m['close'], 20).iloc[-1]
            
            current_high = df_3m['high'].iloc[-1]
            current_low = df_3m['low'].iloc[-1]
            current_close = df_3m['close'].iloc[-1]
            current_open = df_3m['open'].iloc[-1]
            
            # 狀態機邏輯
            if self.setup.state == SetupState.IDLE:
                # 檢測breakout
                has_breakout, breakout_info = self._detect_breakout(df_3m, bar_index)
                
                if has_breakout:
                    self.setup.state = SetupState.BREAKOUT_DETECTED
                    self.setup.breakout_level = breakout_info['level']
                    self.setup.breakout_time = datetime.now()
                    self.setup.breakout_swing_low = breakout_info['swing_low']
                    self.setup.breakout_bar_index = bar_index
                    
                    logger.info(f"\n🎯 Breakout檢測!")
                    logger.info(f"  突破位: ${self.setup.breakout_level:.2f}")
                    logger.info(f"  Swing Low: ${self.setup.breakout_swing_low:.2f}")
                    
                    # 繼續檢查pullback
                
            if self.setup.state == SetupState.BREAKOUT_DETECTED:
                # 檢查是否超時
                bars_since_breakout = bar_index - self.setup.breakout_bar_index
                
                if bars_since_breakout > self.pullback_max_bars:
                    logger.warning(f"  ⏰ Pullback超時({bars_since_breakout}>{self.pullback_max_bars})")
                    self.setup.reset()
                    return False, "L2_PULLBACK_TIMEOUT", None
                
                # 檢查結構是否破壞
                if current_low <= self.setup.breakout_swing_low:
                    logger.warning(f"  💔 結構破壞(跌破${self.setup.breakout_swing_low:.2f})")
                    self.setup.reset()
                    return False, "L2_STRUCTURE_BROKEN", None
                
                # 檢查pullback
                has_pullback = self._check_pullback(
                    current_low, 
                    ema9, 
                    ema20, 
                    self.setup.breakout_level
                )
                
                if has_pullback:
                    self.setup.pullback_touched = True
                    self.setup.pullback_low = current_low
                    
                    logger.info(f"\n📉 Pullback觸碰!")
                    logger.info(f"  回踩低點: ${current_low:.2f}")
                    
                    # 繼續檢查確認
            
            if self.setup.pullback_touched and not self.setup.confirmed:
                # 檢查確認K
                has_confirm = self._check_confirmation(
                    current_high,
                    current_close,
                    current_open,
                    ema9,
                    df_3m
                )
                
                if has_confirm:
                    self.setup.confirmed = True
                    self.setup.confirm_time = datetime.now()
                    self.setup.state = SetupState.CONFIRMED
                    
                    logger.info(f"\n✅ 確認K出現!")
                    
                    # 生成訊號
                    signal = self._generate_signal(
                        current_close,
                        self.setup,
                        ema9,
                        ema20
                    )
                    
                    # 重置setup準備下一次
                    self.setup.reset()
                    
                    if signal is None:
                        return False, "L2_SIGNAL_REJECTED_BY_GATES", None

                    return True, "BREAKOUT_PULLBACK", signal
            
            # 無訊號
            return False, f"L2_STATE_{self.setup.state.value}", None
            
        except Exception as e:
            logger.error(f"L2檢查失敗: {e}", exc_info=True)
            self.setup.reset()
            return False, f"L2_ERROR: {str(e)}", None
    
    def _detect_breakout(self, df, bar_index) -> Tuple[bool, Optional[Dict]]:
        """
        檢測breakout
        
        ✅ 修正: 用[-N:-1]不包含當前K
        """
        
        # ✅ 關鍵修正: 不包含當前K!
        highs_before_current = df['high'].iloc[-self.breakout_lookback-1:-1]
        breakout_level = highs_before_current.max()
        
        current_close = df['close'].iloc[-1]
        
        # 突破條件
        if current_close > breakout_level * (1 + self.breakout_buffer):
            # 找breakout前的swing low
            swing_low = self._find_swing_low_before_breakout(df)
            
            return True, {
                'level': breakout_level,
                'swing_low': swing_low
            }
        
        return False, None
    
    def _check_pullback(self, current_low, ema9, ema20, breakout_level) -> bool:
        """
        檢查pullback (用low觸碰,不用close!)
        
        ✅ 修正: 用low不用close
        """
        
        # 任一條件滿足
        if current_low <= ema9:
            logger.info(f"  觸碰EMA9: ${current_low:.2f} <= ${ema9:.2f}")
            return True
        
        if current_low <= ema20:
            logger.info(f"  觸碰EMA20: ${current_low:.2f} <= ${ema20:.2f}")
            return True
        
        retest_level = breakout_level * (1 + self.retest_buffer)
        if current_low <= retest_level:
            logger.info(f"  回測突破位: ${current_low:.2f} <= ${retest_level:.2f}")
            return True
        
        return False
    
    def _check_confirmation(self, high, close, open_price, ema9, df) -> bool:
        """
        檢查確認K
        
        ✅ 修正: 不要求close同時在EMA9上又在EMA9下
        """
        
        # 方式1: 收回EMA9上方
        if close > ema9:
            logger.info(f"  確認: 收回EMA9上(${close:.2f} > ${ema9:.2f})")
            return True
        
        # 方式2: 小型HH(高點突破+陽線)
        prev_high = df['high'].iloc[-2]
        
        if high > prev_high and close > open_price:
            logger.info(f"  確認: 小HH+陽線")
            return True
        
        return False
    
    def _generate_signal(self, entry_price, setup, ema9, ema20) -> Optional[StrategyBSignal]:
        """生成訊號(包含成本檢查)"""
        
        # 計算止損
        sl_price = min(setup.pullback_low, setup.breakout_swing_low) * 0.9999
        sl_pct = abs(entry_price - sl_price) / entry_price
        
        logger.info(f"\n📊 訊號計算:")
        logger.info(f"  進場: ${entry_price:.2f}")
        logger.info(f"  止損: ${sl_price:.2f} ({sl_pct:.2%})")
        
        # ✅ V5.3修正: 止損範圍檢查使用config參數
        if sl_pct < self.config.min_stop_distance_pct:
            logger.warning(f"  ❌ 止損過近 ({sl_pct:.2%} < {self.config.min_stop_distance_pct:.2%})")
            return None
        
        if sl_pct > self.config.max_stop_distance_pct:
            logger.warning(f"  ❌ 止損過寬,拒絕交易 ({sl_pct:.2%} > {self.config.max_stop_distance_pct:.2%})")
            return None
        
        logger.info("  ✅ 止損範圍OK")
        
        # ✅ V4: 從config讀取成本參數
        round_trip_fee = self.config.fee_taker * 2  # 進+出都用taker
        slippage = self.config.slippage_buffer
        total_cost = round_trip_fee + slippage
        
        # TP1 = 1R 或 0.35% 取較大
        tp1_r = entry_price + (entry_price - sl_price)  # 1R
        tp1_fixed = entry_price * 1.0050  # +0.35%
        tp1_price = max(tp1_r, tp1_fixed)
        
        tp1_pct = (tp1_price - entry_price) / entry_price
        tp1_net = tp1_pct - total_cost  # 扣除成本後淨利
        
        logger.info(f"\n💰 成本檢查:")
        logger.info(f"  往返費用: {round_trip_fee:.2%}")
        logger.info(f"  滑點緩衝: {slippage:.2%}")
        logger.info(f"  總成本: {total_cost:.2%}")
        logger.info(f"  TP1距離: {tp1_pct:.2%}")
        logger.info(f"  扣成本後: {tp1_net:.2%}")
        logger.info(f"  最小要求: {self.config.min_tp_after_costs_pct:.2%}")
        
        # ✅ V4: 硬檢查! 扣成本後必須 >= min_tp_after_costs_pct
        if tp1_net < self.config.min_tp_after_costs_pct:
            logger.warning(f"  ❌ 扣成本後淨利不足 ({tp1_net:.2%} < {self.config.min_tp_after_costs_pct:.2%})")
            return None
        
        logger.info("  ✅ 成本Gate通過")
        
        # 生成訊號
        signal = StrategyBSignal(
            signal_type="LONG",
            pattern="BREAKOUT_PULLBACK",
            entry_price=entry_price,
            stop_loss=sl_price,
            tp1_price=tp1_price,
            stop_distance_pct=sl_pct,
            expected_tp1_pct=tp1_pct,
            confidence=0.75,
            reason=f"Breakout@{setup.breakout_level:.2f} → Pullback → Confirm",
            timestamp=datetime.now(),
            ema20_15m=0.0,  # 外層填入
            ema9_3m=ema9,
            ema20_3m=ema20,
            breakout_level=setup.breakout_level,
            swing_low=setup.breakout_swing_low
        )
        
        logger.info(f"\n🎯 訊號生成成功!")
        logger.info(f"  TP1: ${tp1_price:.2f} (+{tp1_pct:.2%})")
        
        return signal
    
    def _calculate_ema(self, series, period):
        """計算EMA"""
        return series.ewm(span=period, adjust=False).mean()
    
    def _find_swing_low_before_breakout(self, df, lookback=20):
        """找breakout前的swing low"""
        if len(df) < lookback:
            return df['low'].iloc[-lookback:].min()
        
        lows = df['low'].iloc[-lookback:-1].values
        
        for i in range(len(lows)-3, 1, -1):
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
               lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                return lows[i]
        
        return lows.min()


# ==================== 主策略類 ====================

class StrategyBCore:
    """策略B核心 - V2修正版"""
    
    def __init__(self, config, market_data):
        self.config = config
        self.market_data = market_data
        
        # Gate
        self.l0_gate = L0Gate(config)
        self.l1_gate = L1Gate(config)
        self.l2_gate = L2Gate(config)
        
        # 狀態
        self.state = StrategyBState()
        
        # K線計數器
        self.bar_counter = 0
        
        logger.info("="*60)
        logger.info("🚀 Strategy B Core V5.2 初始化")
        logger.info("✅ 狀態機版本")
        logger.info("✅ HH用[-N:-1]")
        logger.info("✅ Breakout/Pullback分離")
        logger.info("✅ 成本內建")
        logger.info("✅ 接口對齊")
        logger.info("✅ 止損距離使用config參數")
        logger.info("="*60)
    
    def check_for_signal(
        self,
        execution_safety,  # OrderStateMachine物件
        has_lock: bool = False,
        has_emergency: bool = False,
        has_position: bool = False
    ) -> Optional[StrategyBSignal]:
        """
        檢查訊號
        
        Args:
            execution_safety: OrderStateMachine物件 (不是bool!)
            has_lock: 是否有全局鎖
            has_emergency: 是否有緊急標記
            has_position: 是否有倉位
        """
        
        self.bar_counter += 1
        
        logger.info("\n" + "🔍"*30)
        logger.info(f"🔍 Strategy B V5.2: 檢查訊號 (Bar #{self.bar_counter})")
        logger.info("🔍"*30)
        
        try:
            # L0
            l0_pass, l0_reason = self.l0_gate.check(
                self.state,
                execution_safety,  # 傳物件
                has_lock,
                has_emergency,
                has_position
            )
            
            if not l0_pass:
                logger.info(f"🚫 {l0_reason}")
                return None
            
            # L1
            l1_pass, l1_reason, l1_debug = self.l1_gate.check_long_environment(
                self.market_data
            )
            
            if not l1_pass:
                logger.info(f"🚫 {l1_reason}")
                return None
            
            # L2 (狀態機)
            has_signal, pattern, signal = self.l2_gate.check_entry_pattern(
                self.market_data,
                l1_passed=l1_pass,
                bar_index=self.bar_counter
            )
            
            if not has_signal:
                logger.info(f"🚫 {pattern}")
                return None
            if signal is None:
                logger.info(f"🚫 {pattern} (signal=None, likely rejected by L2 gates)")
                return None
            # 填入15m資訊
            signal.ema20_15m = l1_debug.get('ema20_15m', 0.0)
            
            # ✅ 訊號確認!
            logger.info("\n" + "🎯"*30)
            logger.info("🎯 訊號確認! (V5.2)")
            logger.info("🎯"*30)
            logger.info(f"型態: {signal.pattern}")
            logger.info(f"進場: ${signal.entry_price:.2f}")
            logger.info(f"止損: ${signal.stop_loss:.2f} ({signal.stop_distance_pct:.2%})")
            logger.info(f"TP1: ${signal.tp1_price:.2f} (+{signal.expected_tp1_pct:.2%})")
            logger.info(f"Breakout: ${signal.breakout_level:.2f}")
            logger.info("🎯"*30)
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ 訊號檢查失敗: {e}", exc_info=True)
            return None
    
    def record_trade_entry(self):
        """
        記錄交易進場
        
        ✅ 修正: 不更新勝負狀態!
        """
        now = datetime.now()
        
        # 更新計數
        if self.state.last_trade_time is None or \
           self.state.last_trade_time.date() != now.date():
            self.state.trades_today = 1
        else:
            self.state.trades_today += 1
        
        if self.state.last_hour_reset is None or \
           (now - self.state.last_hour_reset).total_seconds() >= 3600:
            self.state.trades_this_hour = 1
        else:
            self.state.trades_this_hour += 1
        
        self.state.last_trade_time = now
        
        logger.info(f"📊 進場記錄: 今日{self.state.trades_today}筆, 本小時{self.state.trades_this_hour}筆")
        
        # ✅ 不更新勝負! 要等平倉後!
    
    def record_trade_exit(self, is_win: bool):
        """
        記錄交易出場(平倉後調用)
        
        Args:
            is_win: True=盈利, False=虧損
        """
        
        if is_win:
            self.state.consecutive_losses = 0
            self.state.consecutive_wins += 1
            self.state.last_trade_result = "WIN"
            logger.info(f"✅ 盈利交易! 連勝{self.state.consecutive_wins}次")
            
            # 解除冷卻
            if self.state.in_cooldown:
                self.state.in_cooldown = False
                self.state.cooldown_until = None
                logger.info("  解除冷卻!")
        
        else:
            self.state.consecutive_wins = 0
            self.state.consecutive_losses += 1
            self.state.last_trade_result = "LOSS"
            logger.warning(f"❌ 虧損交易! 連虧{self.state.consecutive_losses}次")
            
            # ✅ 使用正確的config欄位名
            # 檢查冷卻
            if self.state.consecutive_losses >= self.config.max_consecutive_losses:
                self.state.in_cooldown = True
                self.state.cooldown_until = datetime.now() + timedelta(
                    minutes=self.config.cooldown_minutes_after_loss
                )
                logger.warning(f"🧊 觸發冷卻! 至{self.state.cooldown_until}")


# ==================== 導出 ====================

__all__ = [
    'StrategyBCore',
    'StrategyBSignal',
    'StrategyBState',
    'SetupState',
    'BreakoutSetup'
]
