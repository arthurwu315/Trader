"""
Trading Bot V3-Micro-MVP Configuration - MAINNET
真錢小額測試版 (極度保守)
"""
import os
import sys
from dataclasses import dataclass
from typing import Optional

@dataclass
class ConfigV3MicroMVP:
    """V3-Micro-MVP 配置 - 真錢版"""
    
    # ==================== 環境檢查 (第一優先) ====================
    
    def __post_init__(self):
        """初始化後立即執行安全檢查"""
        self._critical_safety_checks()
    
    def _critical_safety_checks(self):
        """關鍵安全檢查 - 失敗即停機"""
        print("\n" + "="*60)
        print("🚨 V3-Micro-MVP 關鍵安全檢查")
        print("="*60)
        
        errors = []
        warnings = []
        
        # 檢查1: 環境模式
        print(f"1. 環境模式: {self.binance_env}")
        
        if self.binance_env == "MAINNET":
            print(f"   Base URL: {self.binance_base_url}")
            
            if "fapi.binance.com" not in self.binance_base_url:
                errors.append("❌ MAINNET模式但Base URL不是真錢!")
            else:
                print("   ✅ Base URL正確")
        
        # 檢查2: 風險限制
        print("\n2. 風險限制檢查:")
        print(f"   單筆風險: {self.risk_per_trade_pct:.2%}")
        
        if self.risk_per_trade_pct > 0.001:  # 0.1%
            errors.append(f"❌ 真錢首測風險過高! {self.risk_per_trade_pct:.2%}")
        else:
            print("   ✅ 風險極低")
        
        # 檢查3: 槓桿限制
        print(f"   最大槓桿: {self.max_leverage}x")
        
        if self.max_leverage > 5:
            errors.append(f"❌ 槓桿過高! {self.max_leverage}x > 5x")
        else:
            print("   ✅ 槓桿安全")
        
        # 檢查4: 交易頻率
        print(f"   每日限制: {self.max_trades_per_day}筆")
        print(f"   每週限制: {self.max_trades_per_week}筆")
        
        if self.max_trades_per_day > 1:
            warnings.append(f"⚠️ 每日限制>1: {self.max_trades_per_day}筆")
        else:
            print("   ✅ 每日限制正確")
        
        if self.max_trades_per_week > 2:
            warnings.append(f"⚠️ 每週限制>2: {self.max_trades_per_week}筆")
        else:
            print("   ✅ 每週限制正確")
        
        # 檢查5: API Key
        print("\n3. API Key檢查:")
        
        if not self.binance_api_key:
            errors.append("❌ API Key未設置!")
        else:
            print(f"   ✅ API Key已設置 ({self.binance_api_key[:8]}...)")
        
        # 檢查6: MVP Gate
        print("\n4. MVP Gate參數檢查:")
        print(f"   最小可用餘額: ${self.mvp_gate_account_min_available_usdt:.2f}")
        
        if self.mvp_gate_account_min_available_usdt < 500:
            warnings.append(f"⚠️ 最小餘額較低: ${self.mvp_gate_account_min_available_usdt}")
        else:
            print("   ✅ 最小可用餘額合理")
        
        print(f"   最大保證金率: {self.mvp_gate_account_max_margin_ratio:.0%}")
        
        if self.mvp_gate_account_max_margin_ratio > 0.65:
            warnings.append(f"⚠️ 保證金率較高: {self.mvp_gate_account_max_margin_ratio:.0%}")
        else:
            print("   ✅ 保證金率上限安全")
        
        # 總結
        print("\n" + "="*60)
        
        if errors:
            print("🚨 發現致命錯誤:")
            for err in errors:
                print(f"   {err}")
            print("\n⛔ 安全檢查失敗 - 拒絕啟動!")
            print("="*60)
            sys.exit(1)
        
        if warnings:
            print("⚠️ 發現警告:")
            for warn in warnings:
                print(f"   {warn}")
            print("\n繼續啟動但請注意...")
        else:
            print("✅ 所有安全檢查通過")
        
        print("="*60 + "\n")
    
    # ==================== 基礎設定 ====================
    version_name: str = "V3-Micro-MVP (真錢測試版)"
    strategy_tag: str = "V3_MICRO_MAINNET"
    symbol: str = "BTCUSDT"
    
    # ⚠️ 真錢環境設定
    binance_env: str = "MAINNET"
    binance_api_key: str = os.getenv("BINANCE_API_KEY", "")
    binance_api_secret: str = os.getenv("BINANCE_API_SECRET", "")
    binance_base_url: str = "https://fapi.binance.com"  # 真錢URL
    
    testnet_mode: bool = False
    paper_trading_mode: bool = False
    
    # ==================== 超嚴格風控 (真錢降低) ====================
    risk_per_trade_pct: float = 0.0005  # 0.05% (從0.1%降低)
    max_leverage: int = 3  # 3x (從5x降低)
    max_position_pct: float = 0.20  # 單筆名義倉位上限
    max_leverage_usage: float = 0.30  # 可用保證金上限
    max_daily_loss_pct: float = 0.03  # 單日最大虧損
    max_total_loss_pct: float = 0.10  # 總權益最大回撤
    max_total_loss_amount: Optional[float] = None  # 絕對金額上限（可選）
    min_equity_threshold_pct: float = 0.80  # 權益跌破閾值
    max_price_change_pct: float = 0.05  # 1h 內波動上限
    enable_trailing_stop: bool = False
    trailing_activate_atr: float = 2.0
    trailing_callback_atr: float = 1.0
    
    max_trades_per_day: int = 1
    max_trades_per_week: int = 2
    
    # ==================== MVP Gate 參數 (真錢更嚴格) ====================
    mvp_gate_account_min_available_usdt: float = 500.0  # 從300提高到500
    mvp_gate_account_max_margin_ratio: float = 0.50  # 從65%降到50%
    
    mvp_gate_min_notional: float = 5.0
    
    mvp_gate_fee_maker: float = 0.00018
    mvp_gate_fee_taker: float = 0.00045
    mvp_gate_slippage_buffer: float = 0.00050
    mvp_gate_min_tp_pct: float = 0.0029
    
    mvp_gate_log_decisions: bool = True
    
    # 全局鎖設定
    global_lock_path: str = "/tmp/futures_account_lock_mainnet"
    global_lock_timeout: int = 10
    
    # 執行安全設定
    execution_max_retries: int = 3
    execution_fill_timeout: int = 30
    execution_query_interval: int = 1
    protection_scan_interval_sec: int = 300
    protection_reconcile_on_startup: bool = True
    
    # ==================== L1/L2/L3 ====================
    weekly_ema_period: int = 21
    daily_ema_period: int = 21
    regime_timeframe: str = "4h"
    execution_timeframe: str = "15m"
    
    l2_max_trades_per_regime: int = 1
    l1_relaxed_mode: bool = True
    l2_relaxed_mode: bool = True
    daily_use_relaxed_filter: bool = True
    weekly_use_relaxed_filter: bool = True
    l1_price_tolerance_pct: float = 0.02
    regime_ema_period: int = 20
    regime_ema_tolerance_pct: float = 0.004
    atr_lookback_periods: int = 100
    atr_decline_bars: int = 5
    atr_percentile_threshold: float = 30.0
    structure_break_enabled: bool = True
    
    allow_breakout_entry: bool = True
    allow_pullback_entry: bool = True
    allow_weak_structure: bool = True

    execution_ema_period: int = 20
    structure_lookback_bars: int = 20
    breakout_buffer_pct: float = 0.001
    leverage_tier_3_max_sl: float = 0.006
    
    # ==================== 出場策略 ====================
    exit_atr_timeframe: str = "4h"
    tp1_r_multiplier: float = 1.5
    tp1_exit_percentage: float = 0.5
    tp2_r_multiplier: float = 2.5
    trailing_stop_enabled: bool = True
    
    # ==================== Telegram ====================
    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")
    enable_telegram: bool = bool(telegram_bot_token and telegram_chat_id)
    telegram_prefix: str = "💰 MAINNET-Micro"
    
    # ==================== 數據庫與日誌 ====================
    database_path: str = "/home/trader/trading_system/bots/bot_a/db/strategy_a_unified.db"
    log_file: str = "/home/trader/trading_system/bots/bot_a/logs/strategy_a_unified.log"
    log_level: str = "INFO"
    
    # ==================== 系統設定 ====================
    main_loop_interval: int = 20
    strategy_check_interval: int = 300
    margin_type: str = "ISOLATED"

def get_micro_mvp_config():
    """獲取Micro MVP配置並執行安全檢查"""
    return ConfigV3MicroMVP()

# 測試
if __name__ == "__main__":
    print("正在載入V3-Micro-MVP真錢配置...")
    config = get_micro_mvp_config()
    
    print(f"\n✅ 配置載入成功")
    print(f"版本: {config.version_name}")
    print(f"環境: {config.binance_env}")
    print(f"Base URL: {config.binance_base_url}")
    print(f"風險: {config.risk_per_trade_pct:.2%}")
    print(f"槓桿: {config.max_leverage}x")
