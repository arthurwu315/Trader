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

V5.2.1 Hotfix (你這次要的修正):
A) ✅ L2: signal=None 時回傳 has_signal=False，避免 NoneType 爆炸
B) ✅ L2: pullback_low 只會記錄更低的 low；pullback 觸碰後進入 PULLBACK_WAITING
C) ✅ SL: 改用 pullback_low 當主要止損，不再被 breakout 前 swing low 拉太遠
D) ✅ TP1: 固定TP距離至少 >= (成本 + min_tp_after_costs + buffer) ，避免常被成本Gate擋
E) ✅ L0: 跨日自動歸零 trades_today
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


# ==================== 狀態機定義 ====================

class SetupState(Enum):
    """Setup狀態"""
    IDLE = "IDLE"                          # 無setup
    BREAKOUT_DETECTED = "BREAKOUT"         # 檢測到突破
    PULLBACK_WAITING = "PULLBACK_WAIT"     # 等待回踩後確認
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


# ==================== Short Setup ====================

@dataclass
class BreakdownSetup:
    """Breakdown Setup狀態記錄"""
    state: SetupState = SetupState.IDLE

    # Breakdown資訊
    breakdown_level: Optional[float] = None
    breakdown_time: Optional[datetime] = None
    breakdown_swing_high: Optional[float] = None
    breakdown_bar_index: Optional[int] = None

    # Pullback資訊
    pullback_touched: bool = False
    pullback_high: Optional[float] = None

    # 確認資訊
    confirmed: bool = False
    confirm_time: Optional[datetime] = None

    def reset(self):
        """重置setup"""
        self.state = SetupState.IDLE
        self.breakdown_level = None
        self.breakdown_time = None
        self.breakdown_swing_high = None
        self.breakdown_bar_index = None
        self.pullback_touched = False
        self.pullback_high = None
        self.confirmed = False
        self.confirm_time = None

# ==================== 數據結構 ====================

@dataclass
class StrategyCSignal:
    """策略C訊號"""
    signal_type: str
    pattern: str
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
    tp2_price: Optional[float] = None


@dataclass
class StrategyCState:
    """策略C狀態"""
    trades_today: int = 0
    trades_this_hour: int = 0
    last_trade_time: Optional[datetime] = None
    last_hour_reset: Optional[datetime] = None

    consecutive_losses: int = 0
    consecutive_wins: int = 0
    last_trade_result: Optional[str] = None  # "WIN"/"LOSS"

    in_cooldown: bool = False
    cooldown_until: Optional[datetime] = None


# ==================== L0 Gate ====================

class L0Gate:
    """L0 Gate - 保命層"""

    def __init__(self, config):
        self.config = config

    def check(
        self,
        state: StrategyCState,
        execution_safety,
        has_lock: bool,
        has_emergency: bool,
        has_position: bool
    ) -> Tuple[bool, str]:

        logger.info("\n" + "=" * 60)
        logger.info("🔒 L0 Gate 檢查")
        logger.info("=" * 60)

        now = datetime.now()

        # ✅ 跨日自動歸零 (避免隔天被 L0_DAILY_LIMIT 鎖死)
        if state.last_trade_time is None or state.last_trade_time.date() != now.date():
            state.trades_today = 0

        # 1. 系統安全
        logger.info("\n📋 系統安全檢查:")

        if has_lock:
            return False, "L0_LOCKED"
        logger.info("  ✅ 無全局鎖")

        if has_emergency:
            return False, "L0_EMERGENCY"
        logger.info("  ✅ 無緊急標記")

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
        if state.last_hour_reset is None or (now - state.last_hour_reset).total_seconds() >= 3600:
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
                state.in_cooldown = False
                state.cooldown_until = None
                logger.info("  ✅ 冷卻期已過")

        logger.info("  ✅ 無冷卻限制")

        logger.info("\n✅ L0 Gate 通過")
        logger.info("=" * 60)

        return True, "L0_PASS"


# ==================== L1 Gate ====================

class L1Gate:
    """L1 Gate - 15m環境"""

    def __init__(self, config):
        self.config = config

    def check_long_environment(self, market_data) -> Tuple[bool, str, Dict]:
        """檢查15m多頭環境（放寬版：允許EMA20下方小幅回檔）"""

        logger.info("\n" + "=" * 60)
        logger.info("🌍 L1 Gate: 15m環境檢查")
        logger.info("=" * 60)

        debug = {}

        try:
            if not bool(getattr(self.config, "enable_long", True)):
                return False, "LONG_DISABLED", {}

            df_15m = market_data.get_klines_df(
                symbol=self.config.symbol,
                interval='15m',
                limit=50
            )

            if df_15m is None or len(df_15m) < 30:
                return False, "L1_DATA_INSUFFICIENT", {}

            # 計算EMA20
            ema20 = self._calculate_ema(df_15m['close'], 20)
            ema20_current = float(ema20.iloc[-1])
            current_price = float(df_15m['close'].iloc[-1])

            debug['ema20_15m'] = ema20_current
            debug['price'] = current_price
            # === Debug: 永遠印 EMA20 最近幾根（即使第1關就被擋）===
            rising_bars = int(getattr(self.config, "l1_ema20_rising_bars", 3))
            debug_bars = max(5, rising_bars + 2)

            ema_tail = ema20.iloc[-debug_bars:].values
            ema_tail_str = ", ".join([f"{v:.2f}" for v in ema_tail])

            diffs = []
            for i in range(1, len(ema_tail)):
                diffs.append(float(ema_tail[i]) - float(ema_tail[i - 1]))
            diffs_str = ", ".join([f"{d:+.4f}" for d in diffs])

            logger.info(f"  EMA20 最近{debug_bars}根: [{ema_tail_str}]")
            logger.info(f"  EMA20 相鄰差值: [{diffs_str}]")

            # ===== 1) 價格位置（放寬）=====
            tolerance = float(getattr(self.config, "l1_ema20_tolerance_pct", 0.006))  # 預設0.6%
            lower_bound = ema20_current * (1 - tolerance)
            diff_pct = (current_price / ema20_current - 1) * 100.0
            gap = current_price - lower_bound
            logger.info(f"\n1️⃣ 價格位置:")
            logger.info(f"  價格: ${current_price:.2f}")
            logger.info(f"  EMA20: ${ema20_current:.2f}")
            logger.info(f"  與EMA20偏離: {diff_pct:.2f}% (容忍下方: {tolerance:.2%})")
                       
            logger.info(f"  距離放行門檻差: {gap:+.2f} USDT")

            if current_price < lower_bound:
                logger.warning(
                    f"  ❌ 價格過低於EMA20下方 "
                    f"(price {current_price:.2f} < lower_bound {lower_bound:.2f})"
                )
                return False, "L1_15M_NOT_UPTREND", debug

            if current_price <= ema20_current:
                logger.info("  ⚠️ 價格略低於EMA20但在容忍範圍內（回檔放行）")
            else:
                logger.info("  ✅ 價格在EMA20上方")

            # ===== 2) EMA20斜率向上（保留，避免放到空頭）=====
            logger.info(f"\n2️⃣ EMA20斜率:")


            ema_tail = ema20.iloc[-debug_bars:].values
            ema_tail_str = ", ".join([f"{v:.2f}" for v in ema_tail])

            # 計算每根之間變化（bp / %）
            diffs = []
            for i in range(1, len(ema_tail)):
                prev = float(ema_tail[i - 1])
                cur = float(ema_tail[i])
                diffs.append(cur - prev)

            diffs_str = ", ".join([f"{d:+.4f}" for d in diffs])

            logger.info(f"  EMA20 最近{debug_bars}根: [{ema_tail_str}]")
            logger.info(f"  EMA20 相鄰差值: [{diffs_str}]")
            #///////////////////
            rising_bars = int(getattr(self.config, "l1_ema20_rising_bars", 3))
            if len(ema20) < rising_bars:
                return False, "L1_DATA_INSUFFICIENT", debug

            ema_tail = ema20.iloc[-rising_bars:].values
            # 連續向上：後一個必須 > 前一個
            ema_rising = True
            for i in range(1, len(ema_tail)):
                if not (ema_tail[i] > ema_tail[i - 1]):
                    ema_rising = False
                    break

            if not ema_rising:
                logger.warning("  ❌ EMA20未連續向上")
                return False, "L1_EMA_NOT_RISING", debug

            logger.info("  ✅ EMA20連續向上")

            # ===== 2.5) EMA20斜率門檻（強趨勢過濾）=====
            slope_bars = int(getattr(self.config, "l1_ema20_slope_bars", 5))
            min_slope_pct = float(getattr(self.config, "l1_min_ema20_slope_pct", 0.15))
            if slope_bars > 0 and min_slope_pct > 0:
                if len(ema20) < slope_bars + 1:
                    return False, "L1_DATA_INSUFFICIENT", debug
                ema_prev = float(ema20.iloc[-slope_bars - 1])
                ema_slope_pct = (ema20_current / ema_prev - 1) * 100.0
                logger.info(f"\n2️⃣-1 EMA20斜率:")
                logger.info(f"  EMA20[{slope_bars}根前]: {ema_prev:.2f}")
                logger.info(f"  EMA20當前: {ema20_current:.2f}")
                logger.info(f"  斜率變化: {ema_slope_pct:.2f}% (門檻: {min_slope_pct:.2f}%)")
                if ema_slope_pct < min_slope_pct:
                    logger.warning("  ❌ EMA20斜率不足（趨勢偏弱）")
                    return False, "L1_EMA_SLOPE_TOO_WEAK", debug

            # ===== 2.52) 宏觀趨勢濾網（週線/日線）=====
            macro_ok, macro_reason = self._check_macro_long_trend(market_data)
            if not macro_ok:
                logger.warning(f"  ❌ 宏觀阻擋: {macro_reason}")
                return False, "L1_MACRO_BLOCKED", debug

            # ===== 2.55) EMA50 趨勢確認 =====
            use_ema50_filter = bool(getattr(self.config, "l1_use_ema50_filter", True))
            ema50 = None
            if use_ema50_filter:
                if len(df_15m) < 60:
                    return False, "L1_DATA_INSUFFICIENT", debug
                ema50_series = self._calculate_ema(df_15m['close'], 50)
                ema50_current = float(ema50_series.iloc[-1])
                ema50 = ema50_current

                logger.info(f"\n2️⃣-1 EMA50趨勢:")
                logger.info(f"  EMA20: {ema20_current:.2f}")
                logger.info(f"  EMA50: {ema50_current:.2f}")
                if ema20_current <= ema50_current:
                    logger.warning("  ❌ EMA20未站上EMA50（趨勢不足）")
                    return False, "L1_EMA50_TREND_TOO_WEAK", debug

                ema50_rising_bars = int(getattr(self.config, "l1_ema50_rising_bars", 3))
                if len(ema50_series) < ema50_rising_bars:
                    return False, "L1_DATA_INSUFFICIENT", debug
                ema50_tail = ema50_series.iloc[-ema50_rising_bars:].values
                ema50_rising = True
                for i in range(1, len(ema50_tail)):
                    if not (ema50_tail[i] > ema50_tail[i - 1]):
                        ema50_rising = False
                        break
                if not ema50_rising:
                    logger.warning("  ❌ EMA50未連續向上")
                    return False, "L1_EMA50_NOT_RISING", debug
                logger.info("  ✅ EMA50向上且EMA20在上方")

            # ===== 2.6) 成交量過濾 =====
            vol_lookback = int(getattr(self.config, "l1_volume_lookback", 20))
            vol_mult = float(getattr(self.config, "l1_volume_sma_mult", 1.2))
            if vol_lookback > 0 and vol_mult > 0:
                if len(df_15m) < vol_lookback + 1:
                    return False, "L1_DATA_INSUFFICIENT", debug
                volume_series = df_15m['volume'].iloc[-vol_lookback - 1:-1]
                vol_sma = float(volume_series.mean())
                current_vol = float(df_15m['volume'].iloc[-1])
                logger.info(f"\n2️⃣-2 成交量:")
                logger.info(f"  當前量: {current_vol:.2f}")
                logger.info(f"  {vol_lookback} SMA: {vol_sma:.2f}")
                logger.info(f"  需要 >= {vol_mult:.2f}x SMA: {vol_sma * vol_mult:.2f}")
                if current_vol < vol_sma * vol_mult:
                    logger.warning("  ❌ 成交量不足（動能偏弱）")
                    return False, "L1_VOLUME_TOO_WEAK", debug

            # ===== 2.7) 高週期趨勢濾網 =====
            htf_ok, htf_reason = self._check_htf_trend(market_data, "LONG")
            if not htf_ok:
                logger.warning(f"  ❌ HTF阻擋: {htf_reason}")
                return False, "L1_HTF_BLOCKED", debug

            # ===== 2.8) ATR 波動濾網 =====
            atr_period = int(getattr(self.config, "l1_atr_period", 14))
            atr_lookback = int(getattr(self.config, "l1_atr_lookback", 100))
            atr_min_pct = float(getattr(self.config, "l1_atr_min_percentile", 40.0))
            if len(df_15m) >= max(atr_period + 5, atr_lookback + atr_period + 5):
                atr = self._calculate_atr(df_15m, atr_period)
                atr_tail = atr.dropna().iloc[-atr_lookback:]
                if not atr_tail.empty:
                    current_atr = float(atr_tail.iloc[-1])
                    threshold = float(np.percentile(atr_tail.values, atr_min_pct))
                    logger.info(f"\n2️⃣-3 ATR:")
                    logger.info(f"  ATR({atr_period}) 目前: {current_atr:.4f}")
                    logger.info(f"  近{atr_lookback}分位(>= {atr_min_pct:.1f}): {threshold:.4f}")
                    if current_atr < threshold:
                        logger.warning("  ❌ 波動不足（ATR偏低）")
                        return False, "L1_ATR_TOO_LOW", debug

            # ===== 3) 結構完整(最近swing low未破)（保留）=====
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
            logger.info("=" * 60)

            return True, "L1_PASS", debug

        except Exception as e:
            logger.error(f"L1檢查失敗: {e}", exc_info=True)
            return False, f"L1_ERROR: {str(e)}", {}

    def check_short_environment(self, market_data) -> Tuple[bool, str, Dict]:
        """檢查15m空頭環境（允許EMA20上方小幅回彈）"""

        logger.info("\n" + "=" * 60)
        logger.info("🌍 L1 Gate: 15m空頭環境檢查")
        logger.info("=" * 60)

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
            ema20_current = float(ema20.iloc[-1])
            current_price = float(df_15m['close'].iloc[-1])

            debug['ema20_15m'] = ema20_current
            debug['price'] = current_price

            # ===== 1) 價格位置（放寬）=====
            tolerance = float(
                getattr(
                    self.config,
                    "l1_ema20_tolerance_short_pct",
                    getattr(self.config, "l1_ema20_tolerance_pct", 0.006)
                )
            )
            upper_bound = ema20_current * (1 + tolerance)
            diff_pct = (current_price / ema20_current - 1) * 100.0
            gap = upper_bound - current_price
            logger.info(f"\n1️⃣ 價格位置:")
            logger.info(f"  價格: ${current_price:.2f}")
            logger.info(f"  EMA20: ${ema20_current:.2f}")
            logger.info(f"  與EMA20偏離: {diff_pct:.2f}% (容忍上方: {tolerance:.2%})")
            logger.info(f"  距離放行門檻差: {gap:+.2f} USDT")

            if current_price > upper_bound:
                logger.warning(
                    f"  ❌ 價格過高於EMA20上方 "
                    f"(price {current_price:.2f} > upper_bound {upper_bound:.2f})"
                )
                return False, "L1_15M_NOT_DOWNTREND", debug

            if current_price >= ema20_current:
                logger.info("  ⚠️ 價格略高於EMA20但在容忍範圍內（回彈放行）")
            else:
                logger.info("  ✅ 價格在EMA20下方")

            # ===== 2) EMA20斜率向下 =====
            logger.info(f"\n2️⃣ EMA20斜率:")
            falling_bars = int(getattr(self.config, "l1_ema20_falling_bars", 3))
            if len(ema20) < falling_bars:
                return False, "L1_DATA_INSUFFICIENT", debug

            ema_tail = ema20.iloc[-falling_bars:].values
            ema_falling = True
            for i in range(1, len(ema_tail)):
                if not (ema_tail[i] < ema_tail[i - 1]):
                    ema_falling = False
                    break

            if not ema_falling:
                logger.warning("  ❌ EMA20未連續向下")
                return False, "L1_EMA_NOT_FALLING", debug

            logger.info("  ✅ EMA20連續向下")

            # ===== 2.5) EMA20斜率門檻（強趨勢過濾）=====
            slope_bars = int(getattr(self.config, "l1_ema20_slope_bars", 5))
            min_slope_pct = float(getattr(self.config, "l1_min_ema20_slope_pct", 0.10))
            if slope_bars > 0 and min_slope_pct > 0:
                if len(ema20) < slope_bars + 1:
                    return False, "L1_DATA_INSUFFICIENT", debug
                ema_prev = float(ema20.iloc[-slope_bars - 1])
                ema_slope_pct = (ema20_current / ema_prev - 1) * 100.0
                logger.info(f"\n2️⃣-1 EMA20斜率:")
                logger.info(f"  EMA20[{slope_bars}根前]: {ema_prev:.2f}")
                logger.info(f"  EMA20當前: {ema20_current:.2f}")
                logger.info(f"  斜率變化: {ema_slope_pct:.2f}% (門檻: -{min_slope_pct:.2f}%)")
                if ema_slope_pct > -min_slope_pct:
                    logger.warning("  ❌ EMA20斜率不足（空頭偏弱）")
                    return False, "L1_EMA_SLOPE_TOO_WEAK", debug

            # ===== 2.55) EMA50 空頭趨勢確認 =====
            use_ema50_short = bool(
                getattr(
                    self.config,
                    "l1_use_ema50_filter_short",
                    getattr(self.config, "l1_use_ema50_filter", True),
                )
            )
            if use_ema50_short:
                if len(df_15m) < 60:
                    return False, "L1_DATA_INSUFFICIENT", debug
                ema50_series = self._calculate_ema(df_15m['close'], 50)
                ema50_falling_bars = int(getattr(self.config, "l1_ema50_falling_bars", 3))
                if len(ema50_series) < ema50_falling_bars:
                    return False, "L1_DATA_INSUFFICIENT", debug
                ema50_tail = ema50_series.iloc[-ema50_falling_bars:].values
                ema50_falling = True
                for i in range(1, len(ema50_tail)):
                    if not (ema50_tail[i] < ema50_tail[i - 1]):
                        ema50_falling = False
                        break
                if not ema50_falling:
                    logger.warning("  ❌ EMA50未連續向下")
                    return False, "L1_EMA50_NOT_FALLING", debug
                logger.info("  ✅ EMA50連續向下")

            # ===== 2.55) 宏觀趨勢濾網（週線/日線）=====
            macro_ok, macro_reason = self._check_macro_short_trend(market_data)
            if not macro_ok:
                logger.warning(f"  ❌ 宏觀阻擋: {macro_reason}")
                return False, "L1_MACRO_BLOCKED", debug

            # ===== 2.6) 成交量過濾 =====
            vol_lookback = int(getattr(self.config, "l1_volume_lookback", 20))
            vol_mult = float(getattr(self.config, "l1_volume_sma_mult", 1.1))
            if vol_lookback > 0 and vol_mult > 0:
                if len(df_15m) < vol_lookback + 1:
                    return False, "L1_DATA_INSUFFICIENT", debug
                volume_series = df_15m['volume'].iloc[-vol_lookback - 1:-1]
                vol_sma = float(volume_series.mean())
                current_vol = float(df_15m['volume'].iloc[-1])
                logger.info(f"\n2️⃣-2 成交量:")
                logger.info(f"  當前量: {current_vol:.2f}")
                logger.info(f"  {vol_lookback} SMA: {vol_sma:.2f}")
                logger.info(f"  需要 >= {vol_mult:.2f}x SMA: {vol_sma * vol_mult:.2f}")
                if current_vol < vol_sma * vol_mult:
                    logger.warning("  ❌ 成交量不足（動能偏弱）")
                    return False, "L1_VOLUME_TOO_WEAK", debug

            # ===== 2.7) 高週期趨勢濾網 =====
            htf_ok, htf_reason = self._check_htf_trend(market_data, "SHORT")
            if not htf_ok:
                logger.warning(f"  ❌ HTF阻擋: {htf_reason}")
                return False, "L1_HTF_BLOCKED", debug

            # ===== 2.8) ATR 波動濾網 =====
            atr_period = int(getattr(self.config, "l1_atr_period", 14))
            atr_lookback = int(getattr(self.config, "l1_atr_lookback", 100))
            atr_min_pct = float(getattr(self.config, "l1_atr_min_percentile", 40.0))
            if len(df_15m) >= max(atr_period + 5, atr_lookback + atr_period + 5):
                atr = self._calculate_atr(df_15m, atr_period)
                atr_tail = atr.dropna().iloc[-atr_lookback:]
                if not atr_tail.empty:
                    current_atr = float(atr_tail.iloc[-1])
                    threshold = float(np.percentile(atr_tail.values, atr_min_pct))
                    logger.info(f"\n2️⃣-3 ATR:")
                    logger.info(f"  ATR({atr_period}) 目前: {current_atr:.4f}")
                    logger.info(f"  近{atr_lookback}分位(>= {atr_min_pct:.1f}): {threshold:.4f}")
                    if current_atr < threshold:
                        logger.warning("  ❌ 波動不足（ATR偏低）")
                        return False, "L1_ATR_TOO_LOW", debug

            # ===== 3) 結構完整(最近swing high未破) =====
            logger.info(f"\n3️⃣ 結構檢查:")
            swing_high = self._find_last_swing_high(df_15m)

            if swing_high is None:
                logger.info("  ⚠️ 未找到swing high,放行")
                return True, "L1_PASS", debug

            if current_price >= swing_high:
                logger.warning(f"  ❌ 漲破swing high ${swing_high:.2f}")
                return False, "L1_STRUCTURE_BROKEN", debug

            logger.info(f"  ✅ 結構完整(swing high: ${swing_high:.2f})")

            logger.info("\n✅ L1 Gate 通過")
            logger.info("=" * 60)

            return True, "L1_PASS", debug

        except Exception as e:
            logger.error(f"L1檢查失敗: {e}", exc_info=True)
            return False, f"L1_ERROR: {str(e)}", {}

    def _calculate_ema(self, series, period):
        return series.ewm(span=period, adjust=False).mean()

    def _calculate_atr(self, df, period: int):
        high = df['high']
        low = df['low']
        close = df['close']
        prev_close = close.shift(1)
        tr = (high - low).to_frame(name="hl")
        tr["hc"] = (high - prev_close).abs()
        tr["lc"] = (low - prev_close).abs()
        true_range = tr.max(axis=1)
        return true_range.rolling(window=period, min_periods=period).mean()

    def _find_last_swing_low(self, df, lookback=10):
        if len(df) < 5:
            return None

        lows = df['low'].values[-lookback:]

        for i in range(len(lows) - 3, 1, -1):
            if lows[i] < lows[i - 1] and lows[i] < lows[i - 2] and \
               lows[i] < lows[i + 1] and lows[i] < lows[i + 2]:
                return lows[i]

        return None

    def _find_last_swing_high(self, df, lookback=10):
        if len(df) < 5:
            return None

        highs = df['high'].values[-lookback:]

        for i in range(len(highs) - 3, 1, -1):
            if highs[i] > highs[i - 1] and highs[i] > highs[i - 2] and \
               highs[i] > highs[i + 1] and highs[i] > highs[i + 2]:
                return highs[i]

        return None

    def _check_htf_trend(self, market_data, direction: str) -> Tuple[bool, str]:
        """高週期趨勢過濾（例如4h EMA50/EMA200）"""
        if not bool(getattr(self.config, "l1_use_htf_filter", True)):
            return True, "HTF_DISABLED"

        interval = str(getattr(self.config, "l1_htf_interval", "4h"))
        fast = int(getattr(self.config, "l1_htf_fast_ema", 50))
        slow = int(getattr(self.config, "l1_htf_slow_ema", 200))
        tolerance = float(getattr(self.config, "l1_htf_tolerance_pct", 0.0))
        fast_tolerance = float(getattr(self.config, "l1_htf_fast_tolerance_pct", 0.0))
        slope_bars = int(getattr(self.config, "l1_htf_slope_bars", 10))
        min_slope_pct = float(getattr(self.config, "l1_htf_min_slope_pct", 0.0))
        min_spread_pct = float(getattr(self.config, "l1_htf_min_spread_pct", 0.0))

        df = market_data.get_klines_df(self.config.symbol, interval, limit=max(slow + 20, 220))
        if df is None or len(df) < slow + 5:
            return False, "HTF_DATA_INSUFFICIENT"

        ema_fast = self._calculate_ema(df['close'], fast)
        ema_slow = self._calculate_ema(df['close'], slow)
        fast_now = float(ema_fast.iloc[-1])
        slow_now = float(ema_slow.iloc[-1])
        price_now = float(df['close'].iloc[-1])

        if min_spread_pct > 0:
            spread_pct = abs(fast_now - slow_now) / slow_now
            if spread_pct < min_spread_pct:
                return False, "HTF_SPREAD_TOO_SMALL"

        if min_slope_pct > 0 and len(ema_slow) >= slope_bars + 1:
            slow_now = float(ema_slow.iloc[-1])
            slow_then = float(ema_slow.iloc[-1 - slope_bars])
            slope_pct = (slow_now - slow_then) / slow_then
            if direction == "LONG" and slope_pct < min_slope_pct:
                return False, "HTF_SLOPE_TOO_FLAT"
            if direction == "SHORT" and slope_pct > -min_slope_pct:
                return False, "HTF_SLOPE_TOO_FLAT"

        if direction == "LONG":
            if fast_now <= slow_now:
                return False, "HTF_TREND_WEAK"
            if price_now < slow_now * (1 - tolerance):
                return False, "HTF_PRICE_BELOW_SLOW"
            if price_now < fast_now * (1 - fast_tolerance):
                return False, "HTF_PRICE_BELOW_FAST"
        else:
            if fast_now >= slow_now:
                return False, "HTF_TREND_WEAK"
            if price_now > slow_now * (1 + tolerance):
                return False, "HTF_PRICE_ABOVE_SLOW"
            if price_now > fast_now * (1 + fast_tolerance):
                return False, "HTF_PRICE_ABOVE_FAST"

        return True, "HTF_PASS"

    def _check_macro_short_trend(self, market_data) -> Tuple[bool, str]:
        """宏觀趨勢濾網（週線/日線）"""
        if not bool(getattr(self.config, "l1_use_macro_filter_short", False)):
            return True, "MACRO_DISABLED"

        weekly_period = int(getattr(self.config, "weekly_ema_period", 21))
        daily_period = int(getattr(self.config, "daily_ema_period", 21))
        tolerance = float(getattr(self.config, "l1_macro_short_tolerance_pct", 0.0))
        mode = str(getattr(self.config, "l1_macro_short_mode", "both")).lower()

        weekly_df = market_data.get_klines_df(self.config.symbol, "1w", limit=weekly_period + 10)
        daily_df = market_data.get_klines_df(self.config.symbol, "1d", limit=daily_period + 10)
        if weekly_df is None or len(weekly_df) < weekly_period + 7:
            return False, "MACRO_WEEKLY_DATA_INSUFFICIENT"
        if daily_df is None or len(daily_df) < daily_period + 7:
            return False, "MACRO_DAILY_DATA_INSUFFICIENT"

        weekly_ema = self._calculate_ema(weekly_df["close"], weekly_period)
        weekly_now = float(weekly_ema.iloc[-1])
        weekly_3w_ago = float(weekly_ema.iloc[-4])
        weekly_6w_ago = float(weekly_ema.iloc[-7])
        weekly_price = float(weekly_df["close"].iloc[-1])
        weekly_trend_down = weekly_now < weekly_3w_ago and weekly_now < weekly_6w_ago
        weekly_price_ok = weekly_price <= weekly_now * (1 + tolerance)

        daily_ema = self._calculate_ema(daily_df["close"], daily_period)
        daily_now = float(daily_ema.iloc[-1])
        daily_3d_ago = float(daily_ema.iloc[-4])
        daily_5d_ago = float(daily_ema.iloc[-6])
        daily_price = float(daily_df["close"].iloc[-1])
        daily_trend_down = daily_now < daily_3d_ago and daily_now < daily_5d_ago
        daily_price_ok = daily_price <= daily_now * (1 + tolerance)

        weekly_ok = weekly_trend_down and weekly_price_ok
        daily_ok = daily_trend_down and daily_price_ok
        if mode == "weekly" and weekly_ok:
            return True, "MACRO_PASS_WEEKLY"
        if mode == "daily" and daily_ok:
            return True, "MACRO_PASS_DAILY"
        if mode == "any" and (weekly_ok or daily_ok):
            return True, "MACRO_PASS_ANY"
        if mode == "both" and weekly_ok and daily_ok:
            return True, "MACRO_PASS_BOTH"

        reasons = []
        if not weekly_trend_down:
            reasons.append("W_TREND_UP")
        if not weekly_price_ok:
            reasons.append("W_PRICE_ABOVE_EMA")
        if not daily_trend_down:
            reasons.append("D_TREND_UP")
        if not daily_price_ok:
            reasons.append("D_PRICE_ABOVE_EMA")
        return False, "+".join(reasons)

    def _check_macro_long_trend(self, market_data) -> Tuple[bool, str]:
        """宏觀趨勢濾網（週線/日線）"""
        if not bool(getattr(self.config, "l1_use_macro_filter_long", False)):
            return True, "MACRO_DISABLED"

        weekly_period = int(getattr(self.config, "weekly_ema_period", 21))
        daily_period = int(getattr(self.config, "daily_ema_period", 21))
        tolerance = float(getattr(self.config, "l1_macro_long_tolerance_pct", 0.0))
        mode = str(getattr(self.config, "l1_macro_long_mode", "both")).lower()

        weekly_df = market_data.get_klines_df(self.config.symbol, "1w", limit=weekly_period + 10)
        daily_df = market_data.get_klines_df(self.config.symbol, "1d", limit=daily_period + 10)
        if weekly_df is None or len(weekly_df) < weekly_period + 7:
            return False, "MACRO_WEEKLY_DATA_INSUFFICIENT"
        if daily_df is None or len(daily_df) < daily_period + 7:
            return False, "MACRO_DAILY_DATA_INSUFFICIENT"

        weekly_ema = self._calculate_ema(weekly_df["close"], weekly_period)
        weekly_now = float(weekly_ema.iloc[-1])
        weekly_3w_ago = float(weekly_ema.iloc[-4])
        weekly_6w_ago = float(weekly_ema.iloc[-7])
        weekly_price = float(weekly_df["close"].iloc[-1])
        weekly_trend_up = weekly_now > weekly_3w_ago and weekly_now > weekly_6w_ago
        weekly_price_ok = weekly_price >= weekly_now * (1 - tolerance)

        daily_ema = self._calculate_ema(daily_df["close"], daily_period)
        daily_now = float(daily_ema.iloc[-1])
        daily_3d_ago = float(daily_ema.iloc[-4])
        daily_5d_ago = float(daily_ema.iloc[-6])
        daily_price = float(daily_df["close"].iloc[-1])
        daily_trend_up = daily_now > daily_3d_ago and daily_now > daily_5d_ago
        daily_price_ok = daily_price >= daily_now * (1 - tolerance)

        weekly_ok = weekly_trend_up and weekly_price_ok
        daily_ok = daily_trend_up and daily_price_ok
        if mode == "weekly" and weekly_ok:
            return True, "MACRO_PASS_WEEKLY"
        if mode == "daily" and daily_ok:
            return True, "MACRO_PASS_DAILY"
        if mode == "any" and (weekly_ok or daily_ok):
            return True, "MACRO_PASS_ANY"
        if mode == "both" and weekly_ok and daily_ok:
            return True, "MACRO_PASS_BOTH"

        reasons = []
        if not weekly_trend_up:
            reasons.append("W_TREND_DOWN")
        if not weekly_price_ok:
            reasons.append("W_PRICE_BELOW_EMA")
        if not daily_trend_up:
            reasons.append("D_TREND_DOWN")
        if not daily_price_ok:
            reasons.append("D_PRICE_BELOW_EMA")
        return False, "+".join(reasons)


# ==================== L2 Gate - 狀態機版本 ====================

class L2Gate:
    """L2 Gate - 3m進場邏輯(狀態機)"""

    def __init__(self, config):
        self.config = config
        self.setup = BreakoutSetup()
        self.short_setup = BreakdownSetup()

        self.breakout_lookback = int(getattr(self.config, "structure_lookback_bars", 20))
        self.breakout_buffer = float(getattr(self.config, "breakout_buffer_pct", 0.0002))
        self.retest_buffer = float(getattr(self.config, "retest_buffer_pct", 0.0001))
        self.pullback_max_bars = int(getattr(self.config, "pullback_max_bars", 12))

        # ✅ 可調 buffer，沒設就用原本 0.01%
        self.stop_buffer_pct = getattr(self.config, "stop_buffer_pct", 0.0001)
        # ✅ TP1 固定距離安全緩衝（避免剛好貼邊被浮動成本吃掉）
        self.tp_safety_buffer_pct = getattr(self.config, "tp_safety_buffer_pct", 0.0001)

    def check_entry_pattern(
        self,
        market_data,
        l1_passed: bool,
        bar_index: int
    ) -> Tuple[bool, str, Optional[StrategyCSignal]]:

        if not l1_passed:
            self.setup.reset()
            return False, "L1_NOT_PASSED", None

        logger.info("\n" + "=" * 60)
        logger.info("🎯 L2 Gate: 進場型態檢查(狀態機)")
        logger.info(f"當前狀態: {self.setup.state.value}")
        logger.info("=" * 60)

        try:
            entry_interval = getattr(self.config, "entry_interval", "3m")
            df_3m = market_data.get_klines_df(
                symbol=self.config.symbol,
                interval=entry_interval,
                limit=100
            )

            if df_3m is None or len(df_3m) < 30:
                return False, "L2_DATA_INSUFFICIENT", None

            ema9_series = self._calculate_ema(df_3m['close'], 9)
            ema20_series = self._calculate_ema(df_3m['close'], 20)

            ema9 = float(ema9_series.iloc[-1])
            ema20 = float(ema20_series.iloc[-1])


            current_high = df_3m['high'].iloc[-1]
            current_low = df_3m['low'].iloc[-1]
            current_close = df_3m['close'].iloc[-1]
            current_open = df_3m['open'].iloc[-1]

            # ========== 1) IDLE：偵測 Breakout ==========
            if self.setup.state == SetupState.IDLE:
                has_breakout, breakout_info = self._detect_breakout(df_3m)

                if has_breakout:
                    self.setup.state = SetupState.BREAKOUT_DETECTED
                    self.setup.breakout_level = breakout_info['level']
                    self.setup.breakout_time = datetime.now()
                    self.setup.breakout_swing_low = breakout_info['swing_low']
                    self.setup.breakout_bar_index = bar_index

                    logger.info(f"\n🎯 Breakout檢測!")
                    logger.info(f"  突破位: ${self.setup.breakout_level:.2f}")
                    logger.info(f"  Swing Low: ${self.setup.breakout_swing_low:.2f}")

            # 共同：breakout 後的超時/結構檢查
            if self.setup.state in (SetupState.BREAKOUT_DETECTED, SetupState.PULLBACK_WAITING):
                bars_since_breakout = bar_index - (self.setup.breakout_bar_index or bar_index)

                if bars_since_breakout > self.pullback_max_bars:
                    logger.warning(f"  ⏰ Pullback超時({bars_since_breakout}>{self.pullback_max_bars})")
                    self.setup.reset()
                    return False, "L2_PULLBACK_TIMEOUT", None

                if self.setup.breakout_swing_low is not None and current_low <= self.setup.breakout_swing_low:
                    logger.warning(f"  💔 結構破壞(跌破${self.setup.breakout_swing_low:.2f})")
                    self.setup.reset()
                    return False, "L2_STRUCTURE_BROKEN", None

            # ========== 2) BREAKOUT_DETECTED：等回踩 ==========
            if self.setup.state == SetupState.BREAKOUT_DETECTED:
                has_pullback = self._check_pullback(
                    current_low,
                    ema9,
                    ema20,
                    self.setup.breakout_level
                )

                if has_pullback:
                    self.setup.pullback_touched = True
                    # ✅ pullback_low 只會記錄更低的 low（不會被更高 low 覆寫）
                    if self.setup.pullback_low is None:
                        self.setup.pullback_low = current_low
                    else:
                        self.setup.pullback_low = min(self.setup.pullback_low, current_low)

                    self.setup.state = SetupState.PULLBACK_WAITING

                    logger.info(f"\n📉 Pullback觸碰!")
                    logger.info(f"  回踩低點: ${self.setup.pullback_low:.2f}")

            # ========== 3) PULLBACK_WAITING：等確認K ==========
            if self.setup.state == SetupState.PULLBACK_WAITING and not self.setup.confirmed:
                has_confirm = self._check_confirmation(
                    current_high,
                    current_close,
                    current_open,
                    ema9,
                    df_3m,
                    breakout_level=self.setup.breakout_level
                )

                if has_confirm:
                    # 額外品質過濾（Strategy C 專用）
                    if bool(getattr(self.config, "l2_require_ema_alignment", False)):
                        if ema9 <= ema20:
                            logger.warning("  ❌ EMA9未在EMA20上方（趨勢未對齊）")
                            return False, "L2_EMA_ALIGN_FAIL", None
                    max_pullback = float(getattr(self.config, "l2_max_pullback_depth_pct", 0.0))
                    if max_pullback > 0 and self.setup.breakout_level is not None:
                        if self.setup.pullback_low is not None:
                            min_allowed = self.setup.breakout_level * (1 - max_pullback)
                            if self.setup.pullback_low < min_allowed:
                                logger.warning("  ❌ Pullback過深（品質過濾）")
                                return False, "L2_PULLBACK_TOO_DEEP", None

                    self.setup.confirmed = True
                    self.setup.confirm_time = datetime.now()
                    self.setup.state = SetupState.CONFIRMED

                    logger.info(f"\n✅ 確認K出現!")

                    signal = self._generate_signal(
                        entry_price=float(current_close),
                        setup=self.setup,
                        ema9=float(ema9),
                        ema20=float(ema20),
                        ema20_series=ema20_series
                    )


                    self.setup.reset()

                    # ✅ 關鍵：被 Gate 擋掉就不要回 True
                    if signal is None:
                        return False, "L2_SIGNAL_REJECTED_BY_GATES", None

                    return True, "BREAKOUT_PULLBACK", signal

            # ========== 4) Trend Pullback (Strategy C 新核心) ==========
            if self.setup.state == SetupState.IDLE and bool(getattr(self.config, "l2_use_trend_pullback", False)):
                signal = self._check_trend_pullback(
                    df_3m=df_3m,
                    ema9=ema9,
                    ema20=ema20,
                    ema20_series=ema20_series,
                    entry_price=float(current_close),
                    current_low=float(current_low),
                    current_high=float(current_high),
                    current_open=float(current_open),
                )
                if signal is not None:
                    return True, "TREND_PULLBACK", signal

            return False, f"L2_STATE_{self.setup.state.value}", None

        except Exception as e:
            logger.error(f"L2檢查失敗: {e}", exc_info=True)
            self.setup.reset()
            return False, f"L2_ERROR: {str(e)}", None

    def check_entry_pattern_short(
        self,
        market_data,
        l1_passed: bool,
        bar_index: int
    ) -> Tuple[bool, str, Optional[StrategyCSignal]]:

        if not l1_passed:
            self.short_setup.reset()
            return False, "L1_NOT_PASSED", None

        logger.info("\n" + "=" * 60)
        logger.info("🎯 L2 Gate: 進場型態檢查(空頭狀態機)")
        logger.info(f"當前狀態: {self.short_setup.state.value}")
        logger.info("=" * 60)

        try:
            entry_interval = getattr(self.config, "entry_interval", "3m")
            df_3m = market_data.get_klines_df(
                symbol=self.config.symbol,
                interval=entry_interval,
                limit=100
            )

            if df_3m is None or len(df_3m) < 30:
                return False, "L2_DATA_INSUFFICIENT", None

            ema9_series = self._calculate_ema(df_3m['close'], 9)
            ema20_series = self._calculate_ema(df_3m['close'], 20)

            ema9 = float(ema9_series.iloc[-1])
            ema20 = float(ema20_series.iloc[-1])

            current_high = df_3m['high'].iloc[-1]
            current_low = df_3m['low'].iloc[-1]
            current_close = df_3m['close'].iloc[-1]
            current_open = df_3m['open'].iloc[-1]

            # ========== 1) IDLE：偵測 Breakdown ==========
            if self.short_setup.state == SetupState.IDLE:
                has_breakdown, breakdown_info = self._detect_breakdown(df_3m)

                if has_breakdown:
                    self.short_setup.state = SetupState.BREAKOUT_DETECTED
                    self.short_setup.breakdown_level = breakdown_info['level']
                    self.short_setup.breakdown_time = datetime.now()
                    self.short_setup.breakdown_swing_high = breakdown_info['swing_high']
                    self.short_setup.breakdown_bar_index = bar_index

                    logger.info(f"\n🎯 Breakdown檢測!")
                    logger.info(f"  跌破位: ${self.short_setup.breakdown_level:.2f}")
                    logger.info(f"  Swing High: ${self.short_setup.breakdown_swing_high:.2f}")

            # 共同：breakdown 後的超時/結構檢查
            if self.short_setup.state in (SetupState.BREAKOUT_DETECTED, SetupState.PULLBACK_WAITING):
                bars_since_breakdown = bar_index - (self.short_setup.breakdown_bar_index or bar_index)

                if bars_since_breakdown > self.pullback_max_bars:
                    logger.warning(f"  ⏰ Pullback超時({bars_since_breakdown}>{self.pullback_max_bars})")
                    self.short_setup.reset()
                    return False, "L2_PULLBACK_TIMEOUT", None

                if self.short_setup.breakdown_swing_high is not None and current_high >= self.short_setup.breakdown_swing_high:
                    logger.warning(f"  💔 結構破壞(突破${self.short_setup.breakdown_swing_high:.2f})")
                    self.short_setup.reset()
                    return False, "L2_STRUCTURE_BROKEN", None

            # ========== 2) BREAKOUT_DETECTED：等回踩 ==========
            if self.short_setup.state == SetupState.BREAKOUT_DETECTED:
                has_pullback = self._check_pullback_short(
                    current_high,
                    ema9,
                    ema20,
                    self.short_setup.breakdown_level
                )

                if has_pullback:
                    self.short_setup.pullback_touched = True
                    if self.short_setup.pullback_high is None:
                        self.short_setup.pullback_high = current_high
                    else:
                        self.short_setup.pullback_high = max(self.short_setup.pullback_high, current_high)

                    self.short_setup.state = SetupState.PULLBACK_WAITING

                    logger.info(f"\n📈 Pullback觸碰!")
                    logger.info(f"  回踩高點: ${self.short_setup.pullback_high:.2f}")

            # ========== 3) PULLBACK_WAITING：等確認K ==========
            if self.short_setup.state == SetupState.PULLBACK_WAITING and not self.short_setup.confirmed:
                has_confirm = self._check_confirmation_short(
                    current_low,
                    current_close,
                    current_open,
                    ema9,
                    df_3m,
                    breakdown_level=self.short_setup.breakdown_level
                )

                if has_confirm:
                    # 額外品質過濾（Strategy C 專用）
                    if bool(getattr(self.config, "l2_require_ema_alignment", False)):
                        if ema9 >= ema20:
                            logger.warning("  ❌ EMA9未在EMA20下方（趨勢未對齊）")
                            return False, "L2_EMA_ALIGN_FAIL", None
                    max_pullback = float(getattr(self.config, "l2_max_pullback_depth_pct", 0.0))
                    if max_pullback > 0 and self.short_setup.breakdown_level is not None:
                        if self.short_setup.pullback_high is not None:
                            max_allowed = self.short_setup.breakdown_level * (1 + max_pullback)
                            if self.short_setup.pullback_high > max_allowed:
                                logger.warning("  ❌ Pullback過深（品質過濾）")
                                return False, "L2_PULLBACK_TOO_DEEP", None

                    self.short_setup.confirmed = True
                    self.short_setup.confirm_time = datetime.now()
                    self.short_setup.state = SetupState.CONFIRMED

                    logger.info(f"\n✅ 確認K出現!")

                    signal = self._generate_signal_short(
                        entry_price=float(current_close),
                        setup=self.short_setup,
                        ema9=float(ema9),
                        ema20=float(ema20),
                        ema20_series=ema20_series
                    )

                    self.short_setup.reset()

                    if signal is None:
                        return False, "L2_SIGNAL_REJECTED_BY_GATES", None

                    return True, "BREAKDOWN_PULLBACK", signal

            # ========== 4) Trend Pullback Short (Strategy C 新核心) ==========
            if self.short_setup.state == SetupState.IDLE and bool(getattr(self.config, "l2_use_trend_pullback", False)):
                signal = self._check_trend_pullback_short(
                    df_3m=df_3m,
                    ema9=ema9,
                    ema20=ema20,
                    ema20_series=ema20_series,
                    entry_price=float(current_close),
                    current_low=float(current_low),
                    current_high=float(current_high),
                    current_open=float(current_open),
                )
                if signal is not None:
                    return True, "TREND_PULLBACK", signal

            return False, f"L2_STATE_{self.short_setup.state.value}", None

        except Exception as e:
            logger.error(f"L2檢查失敗: {e}", exc_info=True)
            self.short_setup.reset()
            return False, f"L2_ERROR: {str(e)}", None

    def _check_trend_pullback(
        self,
        df_3m,
        ema9: float,
        ema20: float,
        ema20_series,
        entry_price: float,
        current_low: float,
        current_high: float,
        current_open: float,
    ) -> Optional[StrategyCSignal]:
        """
        Strategy C: EMA 趨勢回踩 + 動能確認
        - 15m 環境已由 L1 過濾
        - 3m 觸碰 EMA9/EMA20 後，收回 EMA9 並收陽
        """
        # Pullback 深度限制
        max_pullback = float(getattr(self.config, "l2_max_pullback_depth_pct", 0.003))
        if max_pullback > 0 and current_low < ema20 * (1 - max_pullback):
            return None

        pullback_hit = current_low <= ema9 or current_low <= ema20
        if not pullback_hit:
            return None

        # 確認K：收回 EMA9 + 收陽
        min_body_pct = float(getattr(self.config, "l2_confirm_body_pct", 0.0))
        body_pct = abs(entry_price - current_open) / entry_price
        if min_body_pct > 0 and body_pct < min_body_pct:
            return None
        if entry_price <= ema9 or entry_price <= current_open:
            return None

        # EMA 對齊（可選）
        if bool(getattr(self.config, "l2_require_ema_alignment", False)):
            if ema9 <= ema20:
                return None

        pullback_low = current_low
        return self._generate_signal_trend(
            entry_price=entry_price,
            pullback_low=pullback_low,
            ema9=ema9,
            ema20=ema20,
            ema20_series=ema20_series,
        )

    def _check_trend_pullback_short(
        self,
        df_3m,
        ema9: float,
        ema20: float,
        ema20_series,
        entry_price: float,
        current_low: float,
        current_high: float,
        current_open: float,
    ) -> Optional[StrategyCSignal]:
        """
        Strategy C: EMA 趨勢回踩 + 動能確認（空頭）
        - 15m 環境已由 L1 過濾
        - 3m 觸碰 EMA9/EMA20 後，收回 EMA9 下並收陰
        """
        max_pullback = float(getattr(self.config, "l2_max_pullback_depth_pct", 0.003))
        if max_pullback > 0 and current_high > ema20 * (1 + max_pullback):
            return None

        pullback_hit = current_high >= ema9 or current_high >= ema20
        if not pullback_hit:
            return None

        min_body_pct = float(getattr(self.config, "l2_confirm_body_pct", 0.0))
        body_pct = abs(entry_price - current_open) / entry_price
        if min_body_pct > 0 and body_pct < min_body_pct:
            return None
        if entry_price >= ema9 or entry_price >= current_open:
            return None

        if bool(getattr(self.config, "l2_require_ema_alignment", False)):
            if ema9 >= ema20:
                return None

        pullback_high = current_high
        return self._generate_signal_trend_short(
            entry_price=entry_price,
            pullback_high=pullback_high,
            ema9=ema9,
            ema20=ema20,
            ema20_series=ema20_series,
        )

    def _generate_signal_trend(
        self,
        entry_price: float,
        pullback_low: float,
        ema9: float,
        ema20: float,
        ema20_series=None,
    ) -> Optional[StrategyCSignal]:
        if pullback_low is None:
            return None

        stop_buf = float(getattr(self.config, "stop_buffer_pct", self.stop_buffer_pct))
        sl_price = float(pullback_low) * (1 - stop_buf)
        sl_pct = abs(entry_price - sl_price) / entry_price

        if sl_pct < self.config.min_stop_distance_pct:
            sl_price = entry_price * (1 - self.config.min_stop_distance_pct)
            sl_pct = self.config.min_stop_distance_pct
        if sl_pct > self.config.max_stop_distance_pct:
            return None

        round_trip_fee = self.config.fee_taker * 2
        slippage = self.config.slippage_buffer
        total_cost = round_trip_fee + slippage

        rr = float(getattr(self.config, "tp_rr_multiple", 2.0))
        tp_min_fixed_pct = float(getattr(self.config, "tp_min_fixed_pct", 0.008))
        r_dist = (entry_price - sl_price)
        tp_by_r = entry_price + r_dist * rr
        min_required_tp1_pct = total_cost + self.config.min_tp_after_costs_pct + self.tp_safety_buffer_pct
        fixed_pct = max(tp_min_fixed_pct, min_required_tp1_pct)
        tp_by_fixed = entry_price * (1 + fixed_pct)
        tp1_price = max(tp_by_r, tp_by_fixed)
        tp1_pct = (tp1_price - entry_price) / entry_price
        tp1_net = tp1_pct - total_cost
        if tp1_net < self.config.min_tp_after_costs_pct:
            return None

        rr2 = float(getattr(self.config, "tp2_rr_multiple", 2.0))
        tp2_price = entry_price + r_dist * rr2

        return StrategyCSignal(
            signal_type="LONG",
            pattern="TREND_PULLBACK",
            entry_price=entry_price,
            stop_loss=sl_price,
            tp1_price=tp1_price,
            tp2_price=tp2_price,
            stop_distance_pct=sl_pct,
            expected_tp1_pct=tp1_pct,
            confidence=0.70,
            reason="Trend pullback → EMA9 reclaim",
            timestamp=datetime.now(),
            ema20_15m=0.0,
            ema9_3m=ema9,
            ema20_3m=ema20,
            breakout_level=None,
            swing_low=None,
        )

    def _generate_signal_trend_short(
        self,
        entry_price: float,
        pullback_high: float,
        ema9: float,
        ema20: float,
        ema20_series=None,
    ) -> Optional[StrategyCSignal]:
        if pullback_high is None:
            return None

        stop_buf = float(getattr(self.config, "stop_buffer_pct", self.stop_buffer_pct))
        sl_price = float(pullback_high) * (1 + stop_buf)
        sl_pct = abs(entry_price - sl_price) / entry_price

        if sl_pct < self.config.min_stop_distance_pct:
            sl_price = entry_price * (1 + self.config.min_stop_distance_pct)
            sl_pct = self.config.min_stop_distance_pct
        if sl_pct > self.config.max_stop_distance_pct:
            return None

        round_trip_fee = self.config.fee_taker * 2
        slippage = self.config.slippage_buffer
        total_cost = round_trip_fee + slippage

        rr = float(getattr(self.config, "tp_rr_multiple", 2.0))
        tp_min_fixed_pct = float(getattr(self.config, "tp_min_fixed_pct", 0.008))
        r_dist = (sl_price - entry_price)
        tp_by_r = entry_price - r_dist * rr
        min_required_tp1_pct = total_cost + self.config.min_tp_after_costs_pct + self.tp_safety_buffer_pct
        fixed_pct = max(tp_min_fixed_pct, min_required_tp1_pct)
        tp_by_fixed = entry_price * (1 - fixed_pct)
        tp1_price = min(tp_by_r, tp_by_fixed)
        tp1_pct = (entry_price - tp1_price) / entry_price
        tp1_net = tp1_pct - total_cost
        if tp1_net < self.config.min_tp_after_costs_pct:
            return None

        rr2 = float(getattr(self.config, "tp2_rr_multiple", 2.0))
        tp2_price = entry_price - r_dist * rr2

        return StrategyCSignal(
            signal_type="SHORT",
            pattern="TREND_PULLBACK",
            entry_price=entry_price,
            stop_loss=sl_price,
            tp1_price=tp1_price,
            tp2_price=tp2_price,
            stop_distance_pct=sl_pct,
            expected_tp1_pct=tp1_pct,
            confidence=0.70,
            reason="Trend pullback → EMA9 breakdown",
            timestamp=datetime.now(),
            ema20_15m=0.0,
            ema9_3m=ema9,
            ema20_3m=ema20,
            breakout_level=None,
            swing_low=None,
        )

    def _detect_breakout(self, df) -> Tuple[bool, Optional[Dict]]:
        # ✅ 不包含當前K
        highs_before_current = df['high'].iloc[-self.breakout_lookback - 1:-1]
        breakout_level = highs_before_current.max()

        current_close = df['close'].iloc[-1]

        threshold = breakout_level * (1 + self.breakout_buffer)
        logger.info(
            f"[BREAKOUT_CHECK] close={current_close:.2f}, "
            f"level={breakout_level:.2f}, threshold={threshold:.2f}, "
            f"delta={(current_close/threshold - 1)*100:.3f}%"
        )


        current_high = df['high'].iloc[-1]
        threshold_high = breakout_level * (1 + self.breakout_buffer)

        # high 有效突破
        high_break = current_high > threshold_high

        # close 至少站在突破位附近（避免只是影線亂刺）
        close_ok = current_close >= breakout_level * (1 + self.breakout_buffer * 0.25)

        if high_break and close_ok:
            # ✅ 量能確認：突破K量必須高於均量
            vol_lookback = int(getattr(self.config, "l2_breakout_volume_lookback", 20))
            vol_mult = float(getattr(self.config, "l2_breakout_volume_mult", 1.5))
            if vol_lookback > 0 and vol_mult > 0:
                if len(df) < vol_lookback + 1:
                    logger.warning("  ❌ 成交量資料不足（breakout）")
                    return False, None
                vol_series = df['volume'].iloc[-vol_lookback - 1:-1]
                vol_sma = float(vol_series.mean())
                current_vol = float(df['volume'].iloc[-1])
                logger.info(
                    f"[BREAKOUT_VOLUME] cur={current_vol:.2f}, "
                    f"sma={vol_sma:.2f}, need>={vol_sma * vol_mult:.2f}"
                )
                if current_vol < vol_sma * vol_mult:
                    logger.warning("  ❌ 突破量能不足")
                    return False, None

            swing_low = self._find_swing_low_before_breakout(df)
            return True, {'level': breakout_level, 'swing_low': swing_low}


        return False, None

    def _detect_breakdown(self, df) -> Tuple[bool, Optional[Dict]]:
        # ✅ 不包含當前K
        lows_before_current = df['low'].iloc[-self.breakout_lookback - 1:-1]
        breakdown_level = lows_before_current.min()

        current_close = df['close'].iloc[-1]

        threshold = breakdown_level * (1 - self.breakout_buffer)
        logger.info(
            f"[BREAKDOWN_CHECK] close={current_close:.2f}, "
            f"level={breakdown_level:.2f}, threshold={threshold:.2f}, "
            f"delta={(current_close/threshold - 1)*100:.3f}%"
        )

        current_low = df['low'].iloc[-1]
        threshold_low = breakdown_level * (1 - self.breakout_buffer)

        # low 有效跌破
        low_break = current_low < threshold_low

        # close 至少站在跌破位附近
        close_ok = current_close <= breakdown_level * (1 - self.breakout_buffer * 0.25)

        if low_break and close_ok:
            # ✅ 量能確認：突破K量必須高於均量
            vol_lookback = int(getattr(self.config, "l2_breakout_volume_lookback", 20))
            vol_mult = float(getattr(self.config, "l2_breakout_volume_mult", 1.5))
            if vol_lookback > 0 and vol_mult > 0:
                if len(df) < vol_lookback + 1:
                    logger.warning("  ❌ 成交量資料不足（breakdown）")
                    return False, None
                vol_series = df['volume'].iloc[-vol_lookback - 1:-1]
                vol_sma = float(vol_series.mean())
                current_vol = float(df['volume'].iloc[-1])
                logger.info(
                    f"[BREAKDOWN_VOLUME] cur={current_vol:.2f}, "
                    f"sma={vol_sma:.2f}, need>={vol_sma * vol_mult:.2f}"
                )
                if current_vol < vol_sma * vol_mult:
                    logger.warning("  ❌ 跌破量能不足")
                    return False, None

            swing_high = self._find_swing_high_before_breakdown(df)
            return True, {'level': breakdown_level, 'swing_high': swing_high}

        return False, None

    def _check_pullback(self, current_low, ema9, ema20, breakout_level) -> bool:
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

    def _check_pullback_short(self, current_high, ema9, ema20, breakdown_level) -> bool:
        if current_high >= ema9:
            logger.info(f"  觸碰EMA9: ${current_high:.2f} >= ${ema9:.2f}")
            return True

        if current_high >= ema20:
            logger.info(f"  觸碰EMA20: ${current_high:.2f} >= ${ema20:.2f}")
            return True

        retest_level = breakdown_level * (1 - self.retest_buffer)
        if current_high >= retest_level:
            logger.info(f"  回測跌破位: ${current_high:.2f} >= ${retest_level:.2f}")
            return True

        return False

    def _check_confirmation(self, high, close, open_price, ema9, df, breakout_level=None) -> bool:
        min_body_pct = float(getattr(self.config, "l2_confirm_body_pct", 0.0))
        if min_body_pct > 0:
            body_pct = abs(close - open_price) / close
            if body_pct < min_body_pct:
                logger.info(f"  確認K實體過小 ({body_pct:.3%} < {min_body_pct:.3%})")
                return False
        confirm_buf = float(getattr(self.config, "l2_confirm_breakout_buffer_pct", 0.0))
        if breakout_level is not None and confirm_buf > 0:
            if close < breakout_level * (1 + confirm_buf):
                logger.info("  確認: 收盤未站穩突破位")
                return False

        if close > ema9:
            logger.info(f"  確認: 收回EMA9上(${close:.2f} > ${ema9:.2f})")
            return True

        prev_high = df['high'].iloc[-2]
        if high > prev_high and close > open_price:
            logger.info("  確認: 小HH+陽線")
            return True

        return False

    def _check_confirmation_short(self, low, close, open_price, ema9, df, breakdown_level=None) -> bool:
        min_body_pct = float(getattr(self.config, "l2_confirm_body_pct", 0.0))
        if min_body_pct > 0:
            body_pct = abs(close - open_price) / close
            if body_pct < min_body_pct:
                logger.info(f"  確認K實體過小 ({body_pct:.3%} < {min_body_pct:.3%})")
                return False
        confirm_buf = float(getattr(self.config, "l2_confirm_breakout_buffer_pct", 0.0))
        if breakdown_level is not None and confirm_buf > 0:
            if close > breakdown_level * (1 - confirm_buf):
                logger.info("  確認: 收盤未站穩跌破位")
                return False

        if close < ema9:
            logger.info(f"  確認: 收回EMA9下(${close:.2f} < ${ema9:.2f})")
            return True

        prev_low = df['low'].iloc[-2]
        if low < prev_low and close < open_price:
            logger.info("  確認: 小LL+陰線")
            return True

        return False

    def _generate_signal(self, entry_price, setup, ema9, ema20, ema20_series=None) -> Optional[StrategyCSignal]:

        # ====================
        # SL (Hotfix C + 安全增強): 以 pullback_low 為主 + stop buffer
        # 若止損過近 → 拉到 min_stop_distance（因為 bot 端是 risk-based sizing，所以不會放大風險）
        # ====================
        if setup.pullback_low is None:
            logger.warning("  ❌ 無法計算止損：pullback_low=None")
            return None
        # ✅ 3m 趨勢確認：EMA20 最近 N 根必須連續上升（提升勝率，抵消 TP 拉大造成的勝率下降）
        bars = int(getattr(self.config, "l2_ema20_rising_bars", 3))
        if ema20_series is not None and len(ema20_series) >= bars:
            tail = ema20_series.iloc[-bars:].values
            rising = True
            for i in range(1, len(tail)):
                if not (tail[i] > tail[i - 1]):
                    rising = False
                    break
            if not rising:
                logger.warning(f"  ❌ 3m EMA20未連續向上({bars}根)，放棄此訊號（提高勝率用）")
                return None

        raw_sl_anchor = float(setup.pullback_low)

        stop_buf = float(getattr(self.config, "stop_buffer_pct", self.stop_buffer_pct))
        sl_price = raw_sl_anchor * (1 - stop_buf)
        sl_pct = abs(entry_price - sl_price) / entry_price

        logger.info(f"\n📊 訊號計算:")
        logger.info(f"  進場: ${entry_price:.2f}")
        logger.info(f"  止損(原始): ${sl_price:.2f} ({sl_pct:.2%})")
        logger.info(f"  SL anchor(pullback_low): {setup.pullback_low}")
        logger.info(f"  Swing low(僅供結構/診斷): {setup.breakout_swing_low}")
        logger.info(f"  stop_buffer_pct: {stop_buf:.4%}")

        # ✅ 止損太近：拉到最小止損距離
        if sl_pct < self.config.min_stop_distance_pct:
            adjusted_sl = entry_price * (1 - self.config.min_stop_distance_pct)
            logger.warning(
                f"  ⚠️ 止損過近，調整到最小止損距離: "
                f"{sl_pct:.2%} -> {self.config.min_stop_distance_pct:.2%} "
                f"(SL {sl_price:.2f} -> {adjusted_sl:.2f})"
            )
            sl_price = adjusted_sl
            sl_pct = self.config.min_stop_distance_pct

        # ✅ 只保留「過寬」拒單（過近已調整）
        if sl_pct > self.config.max_stop_distance_pct:
            logger.warning(
                f"  ❌ 止損過寬,拒絕交易 ({sl_pct:.2%} > {self.config.max_stop_distance_pct:.2%})"
            )
            return None

        logger.info("  ✅ 止損範圍OK")


        # 成本
        round_trip_fee = self.config.fee_taker * 2
        slippage = self.config.slippage_buffer
        total_cost = round_trip_fee + slippage

        # ✅ 目標：把平均贏拉大，避免短打被成本磨掉
        # TP1：以 RR 倍數為主（例如 2R），同時保底固定距離（例如 0.8%）
        rr = float(getattr(self.config, "tp_rr_multiple", 2.0))
        if bool(getattr(self.config, "dynamic_rr_enabled", True)) and ema20_series is not None:
            slope_bars = int(getattr(self.config, "rr_slope_bars", 5))
            slope_th = float(getattr(self.config, "rr_slope_threshold_pct", 0.08))
            rr_boost = float(getattr(self.config, "rr_slope_boost", 0.3))
            if len(ema20_series) >= slope_bars + 1:
                ema_prev = float(ema20_series.iloc[-slope_bars - 1])
                ema_now = float(ema20_series.iloc[-1])
                ema_slope_pct = (ema_now / ema_prev - 1) * 100.0
                if ema_slope_pct >= slope_th:
                    rr += rr_boost
        tp_min_fixed_pct = float(getattr(self.config, "tp_min_fixed_pct", 0.008))

        # 以 R 計算（R = entry - SL）
        r_dist = (entry_price - sl_price)
        tp_by_r = entry_price + r_dist * rr

        # 固定距離至少要覆蓋 (成本 + 你設定的最小淨利 + safety buffer) 且 >= tp_min_fixed_pct
        min_required_tp1_pct = total_cost + self.config.min_tp_after_costs_pct + self.tp_safety_buffer_pct
        fixed_pct = max(tp_min_fixed_pct, min_required_tp1_pct)
        tp_by_fixed = entry_price * (1 + fixed_pct)

        tp1_price = max(tp_by_r, tp_by_fixed)
        tp1_pct = (tp1_price - entry_price) / entry_price
        tp1_net = tp1_pct - total_cost

        rr2 = float(getattr(self.config, "tp2_rr_multiple", 2.0))
        tp2_price = entry_price + r_dist * rr2

        logger.info(f"\n💰 成本檢查:")
        logger.info(f"  往返費用: {round_trip_fee:.2%}")
        logger.info(f"  滑點緩衝: {slippage:.2%}")
        logger.info(f"  總成本: {total_cost:.2%}")
        logger.info(f"  TP(RR={rr:.2f}): {((tp_by_r/entry_price)-1):.2%}")
        logger.info(f"  TP(固定>= {fixed_pct:.2%}): {((tp_by_fixed/entry_price)-1):.2%}")
        logger.info(f"  最終TP距離: {tp1_pct:.2%}")
        logger.info(f"  扣成本後: {tp1_net:.2%}")
        logger.info(f"  最小要求(淨利): {self.config.min_tp_after_costs_pct:.2%}")

        # ✅ 這裡只做「下限保護」：若連最低淨利都不到，拒絕
        if tp1_net < self.config.min_tp_after_costs_pct:
            logger.warning(f"  ❌ 扣成本後淨利不足 ({tp1_net:.2%} < {self.config.min_tp_after_costs_pct:.2%})")
            return None


        logger.info("  ✅ 成本Gate通過")

        signal = StrategyCSignal(
            signal_type="LONG",
            pattern="BREAKOUT_PULLBACK",
            entry_price=entry_price,
            stop_loss=sl_price,
            tp1_price=tp1_price,
            tp2_price=tp2_price,
            stop_distance_pct=sl_pct,
            expected_tp1_pct=tp1_pct,
            confidence=0.75,
            reason=f"Breakout@{setup.breakout_level:.2f} → Pullback → Confirm",
            timestamp=datetime.now(),
            ema20_15m=0.0,
            ema9_3m=ema9,
            ema20_3m=ema20,
            breakout_level=setup.breakout_level,
            swing_low=setup.breakout_swing_low
        )

        logger.info(f"\n🎯 訊號生成成功!")
        logger.info(f"  TP1: ${tp1_price:.2f} (+{tp1_pct:.2%})")

        return signal

    def _generate_signal_short(self, entry_price, setup, ema9, ema20, ema20_series=None) -> Optional[StrategyCSignal]:

        if setup.pullback_high is None:
            logger.warning("  ❌ 無法計算止損：pullback_high=None")
            return None

        # ✅ 3m 趨勢確認：EMA20 最近 N 根必須連續下降
        bars = int(getattr(self.config, "l2_ema20_falling_bars", 3))
        if ema20_series is not None and len(ema20_series) >= bars:
            tail = ema20_series.iloc[-bars:].values
            falling = True
            for i in range(1, len(tail)):
                if not (tail[i] < tail[i - 1]):
                    falling = False
                    break
            if not falling:
                logger.warning(f"  ❌ 3m EMA20未連續向下({bars}根)，放棄此訊號（提高勝率用）")
                return None

        raw_sl_anchor = float(setup.pullback_high)

        stop_buf = float(getattr(self.config, "stop_buffer_pct", self.stop_buffer_pct))
        sl_price = raw_sl_anchor * (1 + stop_buf)
        sl_pct = abs(entry_price - sl_price) / entry_price

        logger.info(f"\n📊 訊號計算:")
        logger.info(f"  進場: ${entry_price:.2f}")
        logger.info(f"  止損(原始): ${sl_price:.2f} ({sl_pct:.2%})")
        logger.info(f"  SL anchor(pullback_high): {setup.pullback_high}")
        logger.info(f"  Swing high(僅供結構/診斷): {setup.breakdown_swing_high}")
        logger.info(f"  stop_buffer_pct: {stop_buf:.4%}")

        if sl_pct < self.config.min_stop_distance_pct:
            adjusted_sl = entry_price * (1 + self.config.min_stop_distance_pct)
            logger.warning(
                f"  ⚠️ 止損過近，調整到最小止損距離: "
                f"{sl_pct:.2%} -> {self.config.min_stop_distance_pct:.2%} "
                f"(SL {sl_price:.2f} -> {adjusted_sl:.2f})"
            )
            sl_price = adjusted_sl
            sl_pct = self.config.min_stop_distance_pct

        if sl_pct > self.config.max_stop_distance_pct:
            logger.warning(
                f"  ❌ 止損過寬,拒絕交易 ({sl_pct:.2%} > {self.config.max_stop_distance_pct:.2%})"
            )
            return None

        logger.info("  ✅ 止損範圍OK")

        # 成本
        round_trip_fee = self.config.fee_taker * 2
        slippage = self.config.slippage_buffer
        total_cost = round_trip_fee + slippage

        rr = float(getattr(self.config, "tp_rr_multiple", 2.0))
        if bool(getattr(self.config, "dynamic_rr_enabled", True)) and ema20_series is not None:
            slope_bars = int(getattr(self.config, "rr_slope_bars", 5))
            slope_th = float(getattr(self.config, "rr_slope_threshold_pct", 0.08))
            rr_boost = float(getattr(self.config, "rr_slope_boost", 0.3))
            if len(ema20_series) >= slope_bars + 1:
                ema_prev = float(ema20_series.iloc[-slope_bars - 1])
                ema_now = float(ema20_series.iloc[-1])
                ema_slope_pct = (ema_now / ema_prev - 1) * 100.0
                if ema_slope_pct <= -slope_th:
                    rr += rr_boost
        tp_min_fixed_pct = float(getattr(self.config, "tp_min_fixed_pct", 0.008))

        # 以 R 計算（R = SL - entry）
        r_dist = (sl_price - entry_price)
        tp_by_r = entry_price - r_dist * rr

        min_required_tp1_pct = total_cost + self.config.min_tp_after_costs_pct + self.tp_safety_buffer_pct
        fixed_pct = max(tp_min_fixed_pct, min_required_tp1_pct)
        tp_by_fixed = entry_price * (1 - fixed_pct)

        tp1_price = min(tp_by_r, tp_by_fixed)
        tp1_pct = (entry_price - tp1_price) / entry_price
        tp1_net = tp1_pct - total_cost

        rr2 = float(getattr(self.config, "tp2_rr_multiple", 2.0))
        tp2_price = entry_price - r_dist * rr2

        logger.info(f"\n💰 成本檢查:")
        logger.info(f"  往返費用: {round_trip_fee:.2%}")
        logger.info(f"  滑點緩衝: {slippage:.2%}")
        logger.info(f"  總成本: {total_cost:.2%}")
        logger.info(f"  TP(RR={rr:.2f}): {((entry_price - tp_by_r)/entry_price):.2%}")
        logger.info(f"  TP(固定>= {fixed_pct:.2%}): {((entry_price - tp_by_fixed)/entry_price):.2%}")
        logger.info(f"  最終TP距離: {tp1_pct:.2%}")
        logger.info(f"  扣成本後: {tp1_net:.2%}")
        logger.info(f"  最小要求(淨利): {self.config.min_tp_after_costs_pct:.2%}")

        if tp1_net < self.config.min_tp_after_costs_pct:
            logger.warning(f"  ❌ 扣成本後淨利不足 ({tp1_net:.2%} < {self.config.min_tp_after_costs_pct:.2%})")
            return None

        logger.info("  ✅ 成本Gate通過")

        signal = StrategyCSignal(
            signal_type="SHORT",
            pattern="BREAKDOWN_PULLBACK",
            entry_price=entry_price,
            stop_loss=sl_price,
            tp1_price=tp1_price,
            tp2_price=tp2_price,
            stop_distance_pct=sl_pct,
            expected_tp1_pct=tp1_pct,
            confidence=0.75,
            reason=f"Breakdown@{setup.breakdown_level:.2f} → Pullback → Confirm",
            timestamp=datetime.now(),
            ema20_15m=0.0,
            ema9_3m=ema9,
            ema20_3m=ema20,
            breakout_level=setup.breakdown_level,
            swing_low=setup.breakdown_swing_high
        )

        logger.info(f"\n🎯 訊號生成成功!")
        logger.info(f"  TP1: ${tp1_price:.2f} (+{tp1_pct:.2%})")

        return signal

    def _calculate_ema(self, series, period):
        return series.ewm(span=period, adjust=False).mean()

    def _find_swing_low_before_breakout(self, df, lookback=20):
        if len(df) < lookback:
            return df['low'].iloc[-lookback:].min()

        lows = df['low'].iloc[-lookback:-1].values

        for i in range(len(lows) - 3, 1, -1):
            if lows[i] < lows[i - 1] and lows[i] < lows[i - 2] and \
               lows[i] < lows[i + 1] and lows[i] < lows[i + 2]:
                return lows[i]

        return lows.min()

    def _find_swing_high_before_breakdown(self, df, lookback=20):
        if len(df) < lookback:
            return df['high'].iloc[-lookback:].max()

        highs = df['high'].iloc[-lookback:-1].values

        for i in range(len(highs) - 3, 1, -1):
            if highs[i] > highs[i - 1] and highs[i] > highs[i - 2] and \
               highs[i] > highs[i + 1] and highs[i] > highs[i + 2]:
                return highs[i]

        return highs.max()


# ==================== 主策略類 ====================

class StrategyCCore:
    """策略B核心 - V5.2"""

    def __init__(self, config, market_data):
        self.config = config
        self.market_data = market_data

        self.l0_gate = L0Gate(config)
        self.l1_gate = L1Gate(config)
        self.l2_gate = L2Gate(config)

        self.state = StrategyCState()
        self.bar_counter = 0

        logger.info("=" * 60)
        logger.info("🚀 Strategy B Core V5.2 初始化")
        logger.info("✅ 狀態機版本")
        logger.info("✅ 成本內建")
        logger.info("✅ 止損距離使用config參數")
        logger.info("=" * 60)

    def check_for_signal(
        self,
        execution_safety,
        has_lock: bool = False,
        has_emergency: bool = False,
        has_position: bool = False
    ) -> Optional[StrategyCSignal]:

        self.bar_counter += 1

        logger.info("\n" + "🔍" * 30)
        logger.info(f"🔍 Strategy C: 檢查訊號 (Bar #{self.bar_counter})")
        logger.info("🔍" * 30)

        try:
            # === Strategy C: Macro regime filter (direction selection) ===
            allow_long = bool(getattr(self.config, "enable_long", True))
            allow_short = bool(getattr(self.config, "enable_short", True))
            if bool(getattr(self.config, "c_use_macro_regime", False)):
                long_ok, _ = self.l1_gate._check_macro_long_trend(self.market_data)
                short_ok, _ = self.l1_gate._check_macro_short_trend(self.market_data)
                if long_ok and not short_ok:
                    allow_short = False
                elif short_ok and not long_ok:
                    allow_long = False
                elif not long_ok and not short_ok:
                    logger.info("🚫 MACRO_REGIME_BLOCKED")
                    return None
            l0_pass, l0_reason = self.l0_gate.check(
                self.state,
                execution_safety,
                has_lock,
                has_emergency,
                has_position
            )

            if not l0_pass:
                logger.info(f"🚫 {l0_reason}")
                return None

            l1_pass = False
            l1_reason = "LONG_DISABLED"
            l1_debug = {}
            if allow_long:
                l1_pass, l1_reason, l1_debug = self.l1_gate.check_long_environment(self.market_data)
            if not l1_pass:
                logger.info(f"🚫 {l1_reason}")
            else:
                has_signal, pattern, signal = self.l2_gate.check_entry_pattern(
                    self.market_data,
                    l1_passed=l1_pass,
                    bar_index=self.bar_counter
                )

                if has_signal and signal is not None:
                    signal.ema20_15m = l1_debug.get('ema20_15m', 0.0)

                    logger.info("\n" + "🎯" * 30)
                    logger.info("🎯 訊號確認! (Strategy C)")
                    logger.info("🎯" * 30)
                    logger.info(f"型態: {signal.pattern}")
                    logger.info(f"方向: {signal.signal_type}")
                    logger.info(f"進場: ${signal.entry_price:.2f}")
                    logger.info(f"止損: ${signal.stop_loss:.2f} ({signal.stop_distance_pct:.2%})")
                    logger.info(f"TP1: ${signal.tp1_price:.2f} (+{signal.expected_tp1_pct:.2%})")
                    logger.info(f"Breakout: ${signal.breakout_level:.2f}")
                    logger.info("🎯" * 30)

                    return signal

                if not has_signal:
                    logger.info(f"🚫 {pattern}")

            if not allow_short:
                return None

            l1_short_pass, l1_short_reason, l1_short_debug = self.l1_gate.check_short_environment(self.market_data)
            if not l1_short_pass:
                logger.info(f"🚫 {l1_short_reason}")
                return None

            has_signal, pattern, signal = self.l2_gate.check_entry_pattern_short(
                self.market_data,
                l1_passed=l1_short_pass,
                bar_index=self.bar_counter
            )

            if not has_signal:
                logger.info(f"🚫 {pattern}")
                return None

            if signal is None:
                logger.info(f"🚫 {pattern} (signal=None, rejected by L2 gates)")
                return None

            signal.ema20_15m = l1_short_debug.get('ema20_15m', 0.0)

            logger.info("\n" + "🎯" * 30)
            logger.info("🎯 訊號確認! (Strategy C)")
            logger.info("🎯" * 30)
            logger.info(f"型態: {signal.pattern}")
            logger.info(f"方向: {signal.signal_type}")
            logger.info(f"進場: ${signal.entry_price:.2f}")
            logger.info(f"止損: ${signal.stop_loss:.2f} ({signal.stop_distance_pct:.2%})")
            logger.info(f"TP1: ${signal.tp1_price:.2f} (+{signal.expected_tp1_pct:.2%})")
            logger.info(f"Breakout: ${signal.breakout_level:.2f}")
            logger.info("🎯" * 30)

            return signal

        except Exception as e:
            logger.error(f"❌ 訊號檢查失敗: {e}", exc_info=True)
            return None

    def record_trade_entry(self):
        now = datetime.now()

        if self.state.last_trade_time is None or self.state.last_trade_time.date() != now.date():
            self.state.trades_today = 1
        else:
            self.state.trades_today += 1

        if self.state.last_hour_reset is None or (now - self.state.last_hour_reset).total_seconds() >= 3600:
            self.state.trades_this_hour = 1
            self.state.last_hour_reset = now
        else:
            self.state.trades_this_hour += 1

        self.state.last_trade_time = now
        logger.info(f"📊 進場記錄: 今日{self.state.trades_today}筆, 本小時{self.state.trades_this_hour}筆")

    def record_trade_exit(self, is_win: bool):
        if is_win:
            self.state.consecutive_losses = 0
            self.state.consecutive_wins += 1
            self.state.last_trade_result = "WIN"
            logger.info(f"✅ 盈利交易! 連勝{self.state.consecutive_wins}次")

            if self.state.in_cooldown:
                self.state.in_cooldown = False
                self.state.cooldown_until = None
                logger.info("  解除冷卻!")

        else:
            self.state.consecutive_wins = 0
            self.state.consecutive_losses += 1
            self.state.last_trade_result = "LOSS"
            logger.warning(f"❌ 虧損交易! 連虧{self.state.consecutive_losses}次")

            if self.state.consecutive_losses >= self.config.max_consecutive_losses:
                self.state.in_cooldown = True
                self.state.cooldown_until = datetime.now() + timedelta(
                    minutes=self.config.cooldown_minutes_after_loss
                )
                logger.warning(f"🧊 觸發冷卻! 至{self.state.cooldown_until}")


__all__ = [
    'StrategyCCore',
    'StrategyCSignal',
    'StrategyCState',
    'SetupState',
    'BreakoutSetup',
    'BreakdownSetup'
]
