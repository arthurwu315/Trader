"""
Strategy B Configuration - 唯一真相規格
完全按照老手要求統一命名

V5.3修正:
- 新增倉位硬上限 (max_notional_pct_of_equity, max_margin_pct_of_available)
- 調高min_tp_after_costs_pct到0.25% (避免貼邊)
- 強制testnet必須有testnet key/secret (不fallback)
- 目錄自動創建
- ✅ 新增 stop_buffer_pct / tp_safety_buffer_pct 供策略端使用
"""
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ConfigStrategyB:
    """策略B配置 - 唯一真相規格"""

    # ==================== 基本 ====================
    strategy_id: str = "B"
    strategy_version: str = "B-UNIFIED-V5.3"
    strategy_tag: str = "STRATEGY_B"
    symbol: str = "BNBUSDT"
    mode: str = "TESTNET"

    # ==================== 交易節流 ====================
    max_trades_per_day: int = 6
    max_trades_per_hour: int = 2
    cooldown_minutes_after_trade: int = 30
    cooldown_minutes_after_loss: int = 60
    max_consecutive_losses: int = 3
    enable_long: bool = False
    enable_short: bool = True

    # ==================== 風險與槓桿 ====================
    risk_per_trade_pct: float = 0.0005  # 0.05%
    max_leverage: int = 3
    min_stop_distance_pct: float = 0.0018  # 0.18%
    max_position_pct: float = 0.30  # 單筆名義倉位上限
    max_leverage_usage: float = 0.30  # 可用保證金上限
    max_daily_loss_pct: float = 0.03  # 單日最大虧損
    max_total_loss_pct: float = 0.10  # 總權益最大回撤
    max_total_loss_amount: Optional[float] = None  # 絕對金額上限（可選）
    min_equity_threshold_pct: float = 0.80  # 權益跌破閾值
    max_price_change_pct: float = 0.05  # 1h 內波動上限
    enable_trailing_stop: bool = False
    trailing_activate_atr: float = 2.0
    trailing_callback_atr: float = 1.0

    # ==================== L1 放寬參數 ====================
    # ✅ 允許價格略低於15m EMA20（避免回檔期完全沒訊號）
    # 0.006 = 0.6%
    l1_ema20_tolerance_pct: float = 0.006
    l1_ema20_tolerance_short_pct: float = 0.002

    # ✅ 要求EMA20至少連續向上幾根（原本用3根）
    l1_ema20_rising_bars: int = 2
    l1_ema20_falling_bars: int = 3

    # ✅ 強趨勢過濾：EMA20 斜率門檻（%）
    l1_ema20_slope_bars: int = 8
    l1_min_ema20_slope_pct: float = 0.05

    # ✅ 高週期趨勢濾網（4h）
    l1_use_htf_filter: bool = True
    l1_htf_interval: str = "4h"
    l1_htf_fast_ema: int = 21
    l1_htf_slow_ema: int = 55
    l1_htf_tolerance_pct: float = 0.0
    l1_htf_fast_tolerance_pct: float = 0.001
    l1_htf_slope_bars: int = 12
    l1_htf_min_slope_pct: float = 0.003
    l1_htf_min_spread_pct: float = 0.003

    # ✅ 宏觀趨勢濾網（週線/日線）- 避免逆勢放空
    l1_use_macro_filter_short: bool = True
    l1_macro_short_mode: str = "both"  # weekly | daily | both | any
    l1_macro_short_tolerance_pct: float = 0.005
    # ✅ 宏觀趨勢濾網（週線/日線）- 只做順勢多單
    l1_use_macro_filter_long: bool = False
    l1_macro_long_mode: str = "any"  # weekly | daily | both | any
    l1_macro_long_tolerance_pct: float = 0.003


    # ✅ L1 波動濾網（ATR 15m）
    l1_atr_period: int = 14
    l1_atr_lookback: int = 100
    l1_atr_min_percentile: float = 40.0

    # ✅ 成交量過濾：當前量 >= SMA * 倍數
    l1_volume_lookback: int = 20
    l1_volume_sma_mult: float = 1.0

    # ✅ 趨勢強度：EMA20 > EMA50 且 EMA50 向上
    l1_use_ema50_filter: bool = True
    l1_ema50_rising_bars: int = 3

    # ✅ EMA50 空頭過濾（15m）
    l1_use_ema50_filter_short: bool = False
    l1_ema50_falling_bars: int = 3

    # ✅ L2 突破量能確認
    l2_breakout_volume_lookback: int = 20
    l2_breakout_volume_mult: float = 1.3
    l2_confirm_body_pct: float = 0.0005
    l2_confirm_breakout_buffer_pct: float = 0.0005

    
    # ✅ BNB 實務：0.50% 太容易被「合理波動」擋掉，先給 0.60%
    max_stop_distance_pct: float = 0.0080  # 0.80%

    # ✅ 止損buffer（原本策略寫死 0.9999 = 0.01%）
    stop_buffer_pct: float = 0.0001  # 0.01%

    # ✅ TP1安全緩衝（避免剛好貼邊被浮動成本吃掉）
    tp_safety_buffer_pct: float = 0.0002  # 0.02%

    # ✅ V5.3新增: 倉位硬上限
    max_notional_pct_of_equity: float = 0.3
    max_margin_pct_of_available: float = 0.3
    # ==================== 進取型報酬參數（目標 5~10%/週 的必要條件之一） ====================
    # TP 以 R 倍數為主：例如 2.0 = 2R
    tp_rr_multiple: float = 1.2

    # ✅ 動態RR：趨勢強度高時提高RR
    dynamic_rr_enabled: bool = True
    rr_slope_bars: int = 5
    rr_slope_threshold_pct: float = 0.04
    rr_slope_boost: float = 0.5

    # TP 最低固定距離（避免變成 0.3~0.4% 的短打被成本磨掉）
    # 建議先從 0.8% 起跳；若勝率掉太多再調回 0.6~0.7%
    tp_min_fixed_pct: float = 0.004  # 0.40%

    # ✅ 分批止盈
    enable_partial_tp: bool = True
    partial_tp_ratio: float = 0.5
    tp2_rr_multiple: float = 2.5
    enable_ema_exit: bool = False
    ema_exit_period: int = 20

    # 3m 趨勢確認：EMA20 最近 N 根必須連續上升（提高勝率）
    l2_ema20_rising_bars: int = 2
    l2_ema20_falling_bars: int = 3

    # 名義上限（要衝週報酬通常得提高；先保守到 25%~35%）


    # ==================== 成本模型 ====================
    fee_maker: float = 0.00018  # 0.018%
    fee_taker: float = 0.00045  # 0.045%
    slippage_buffer: float = 0.0005  # 0.05%

    # ✅ V5.3: 0.25%
    min_tp_after_costs_pct: float = 0.0018  # 0.18%

    # ==================== 資料週期 ====================
    tf_filter: str = "15m"
    tf_entry: str = "3m"
    lookback_filter: int = 200
    lookback_entry: int = 300

    # ==================== L1/L2/L3 (Regime) ====================
    weekly_ema_period: int = 21
    daily_ema_period: int = 21
    regime_timeframe: str = "4h"
    execution_timeframe: str = "15m"
    entry_interval: str = "3m"
    l2_max_trades_per_regime: int = 2
    l1_relaxed_mode: bool = False
    l2_relaxed_mode: bool = False
    daily_use_relaxed_filter: bool = False
    weekly_use_relaxed_filter: bool = False
    l1_price_tolerance_pct: float = 0.0
    regime_ema_period: int = 20
    regime_ema_tolerance_pct: float = 0.0
    atr_lookback_periods: int = 100
    atr_decline_bars: int = 5
    atr_percentile_threshold: float = 30.0
    structure_break_enabled: bool = True
    allow_breakout_entry: bool = True
    allow_pullback_entry: bool = False
    allow_weak_structure: bool = False

    execution_ema_period: int = 20
    structure_lookback_bars: int = 20
    breakout_buffer_pct: float = 0.001
    retest_buffer_pct: float = 0.0005
    pullback_max_bars: int = 10
    leverage_tier_3_max_sl: float = 0.006

    # ==================== 環境設定 ====================
    binance_api_key: str = os.getenv("BINANCE_API_KEY", "")
    binance_api_secret: str = os.getenv("BINANCE_API_SECRET", "")
    binance_base_url: str = "https://testnet.binancefuture.com"

    # ==================== MVP Gate ====================
    mvp_gate_account_min_available_usdt: float = 300.0
    mvp_gate_account_max_margin_ratio: float = 0.50
    mvp_gate_min_notional: float = 5.0
    mvp_gate_log_decisions: bool = True

    # ==================== 執行安全 ====================
    global_lock_path: str = "/tmp/futures_account_lock_strategy_b"
    global_lock_timeout: int = 10

    execution_max_retries: int = 3
    execution_fill_timeout: int = 30
    execution_query_interval: int = 1
    protection_scan_interval_sec: int = 60
    protection_reconcile_on_startup: bool = True

    # ==================== Telegram ====================
    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")
    enable_telegram: bool = bool(telegram_bot_token and telegram_chat_id)
    telegram_prefix: str = "⚡ Strategy-B"

    # ==================== 數據庫與日誌 ====================
    database_path: str = "/home/trader/trading_system/bots/bot_b/db/strategy_b_unified.db"
    log_file: str = "/home/trader/trading_system/bots/bot_b/logs/strategy_b_unified.log"
    log_level: str = "INFO"

    # ==================== 系統設定 ====================
    main_loop_interval: int = 10
    signal_check_interval: int = 180  # 3分鐘
    margin_type: str = "ISOLATED"

    def __post_init__(self):
        self._ensure_directories()
        if not os.getenv("SKIP_CONFIG_VALIDATION"):
            self._validate_config()

    def _ensure_directories(self):
        log_dir = Path(self.log_file).parent
        if not log_dir.exists():
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
                print(f"✅ 創建日誌目錄: {log_dir}")
            except Exception as e:
                print(f"⚠️ 無法創建日誌目錄 {log_dir}: {e}")

        db_dir = Path(self.database_path).parent
        if not db_dir.exists():
            try:
                db_dir.mkdir(parents=True, exist_ok=True)
                print(f"✅ 創建數據庫目錄: {db_dir}")
            except Exception as e:
                print(f"⚠️ 無法創建數據庫目錄 {db_dir}: {e}")

    def _validate_config(self):
        print("\n" + "=" * 60)
        print("🔍 Strategy B 唯一真相規格 配置驗證 (V5.3)")
        print("=" * 60)

        errors = []
        warnings = []

        print(f"\n1. 基本:")
        print(f"   策略ID: {self.strategy_id}")
        print(f"   版本: {self.strategy_version}")
        print(f"   商品: {self.symbol}")
        print(f"   模式: {self.mode}")

        print(f"\n2. API Key:")
        if not self.binance_api_key:
            errors.append("❌ BINANCE_API_KEY 未設置 (testnet模式必須設置)")
        if not self.binance_api_secret:
            errors.append("❌ BINANCE_API_SECRET 未設置 (testnet模式必須設置)")

        if self.binance_api_key and self.binance_api_secret:
            print(f"   ✅ TESTNET Key已設置 ({self.binance_api_key[:8]}...)")
            print(f"   ✅ TESTNET Secret已設置 ({self.binance_api_secret[:4]}...****)")

        print(f"\n3. 交易節流:")
        print(f"   每日: {self.max_trades_per_day}筆")
        print(f"   每小時: {self.max_trades_per_hour}筆")
        print(f"   交易後冷卻: {self.cooldown_minutes_after_trade}分")
        print(f"   虧損後冷卻: {self.cooldown_minutes_after_loss}分")
        print(f"   最大連虧: {self.max_consecutive_losses}次")

        print(f"\n4. 風險與槓桿:")
        print(f"   單筆風險: {self.risk_per_trade_pct:.2%}")
        print(f"   槓桿: {self.max_leverage}x")
        print(f"   止損範圍: {self.min_stop_distance_pct:.2%} - {self.max_stop_distance_pct:.2%}")
        print(f"   止損buffer: {self.stop_buffer_pct:.2%}")

        print(f"\n5. 成本模型:")
        print(f"   Maker費率: {self.fee_maker:.2%}")
        print(f"   Taker費率: {self.fee_taker:.2%}")
        print(f"   滑點緩衝: {self.slippage_buffer:.2%}")
        round_trip = self.fee_taker * 2 + self.slippage_buffer
        print(f"   往返成本: ~{round_trip:.2%}")
        print(f"   最小淨利: {self.min_tp_after_costs_pct:.2%}")
        print(f"   TP安全緩衝: {self.tp_safety_buffer_pct:.2%}")

        if self.min_tp_after_costs_pct <= round_trip:
            warnings.append(f"⚠️ 最小淨利({self.min_tp_after_costs_pct:.2%}) <= 成本({round_trip:.2%})")

        print(f"\n6. 資料週期:")
        print(f"   過濾週期: {self.tf_filter}")
        print(f"   進場週期: {self.tf_entry}")
        print(f"   過濾回看: {self.lookback_filter}根")
        print(f"   進場回看: {self.lookback_entry}根")

        print(f"\n7. 目錄檢查:")
        log_dir = Path(self.log_file).parent
        db_dir = Path(self.database_path).parent

        if log_dir.exists():
            print(f"   ✅ 日誌目錄存在: {log_dir}")
        else:
            warnings.append(f"⚠️ 日誌目錄不存在: {log_dir}")

        if db_dir.exists():
            print(f"   ✅ 數據庫目錄存在: {db_dir}")
        else:
            warnings.append(f"⚠️ 數據庫目錄不存在: {db_dir}")

        print("\n" + "=" * 60)

        if warnings:
            print("⚠️ 警告:")
            for warn in warnings:
                print(f"   {warn}")

        if errors:
            print("🚨 致命錯誤:")
            for err in errors:
                print(f"   {err}")
            print("\n⛔ 配置驗證失敗!")
            sys.exit(1)
        else:
            print("✅ 所有檢查通過")

        print("=" * 60)
        print(f"\n🚀 {self.strategy_version}")
        print(f"模式: {self.mode}")
        print(f"風險: {self.risk_per_trade_pct:.2%} × {self.max_leverage}x")
        print(f"限制: {self.max_trades_per_day}筆/日, {self.max_trades_per_hour}筆/時")
        print(f"成本Gate: TP扣成本 >= {self.min_tp_after_costs_pct:.2%}")
        print("=" * 60 + "\n")


def get_strategy_b_config():
    return ConfigStrategyB()


if __name__ == "__main__":
    print("正在載入Strategy B 唯一真相規格配置 (V5.3)...")
    config = get_strategy_b_config()
    print("\n✅ 配置載入成功!")
