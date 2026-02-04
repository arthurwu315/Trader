"""
Strategy C Configuration - 高報酬版
在不改動 Strategy B 的前提下，提升風險/頻率
"""
import os
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ConfigStrategyC:
    """策略C配置 - 高報酬版"""
    
    # ==================== 基本 ====================
    strategy_id: str = "C"
    strategy_version: str = "C-UNIFIED-V1.0"
    strategy_tag: str = "STRATEGY_C"
    symbol: str = "BNBUSDT"
    mode: str = os.getenv("MODE", "TESTNET")
    
    # ==================== 交易節流 ====================
    max_trades_per_day: int = 12
    max_trades_per_hour: int = 3
    cooldown_minutes_after_trade: int = 20
    cooldown_minutes_after_loss: int = 45
    max_consecutive_losses: int = 3
    enable_long: bool = True
    enable_short: bool = False
    
    # ==================== 風險與槓桿 ====================
    risk_per_trade_pct: float = 0.0030  # 0.30%
    max_leverage: int = 3
    min_stop_distance_pct: float = 0.0018  # 0.18%
    max_stop_distance_pct: float = 0.0080  # 0.80%
    max_position_pct: float = 0.40
    max_leverage_usage: float = 0.55

    # ==================== L1/L2（頻率調整） ====================
    # L1: EMA20 趨勢放寬
    l1_ema20_tolerance_pct: float = 0.0010  # 允許價格略低於EMA20
    l1_ema20_rising_bars: int = 3
    l1_use_structure_filter: bool = True

    # L2: Breakout/回踩放寬
    l2_breakout_lookback: int = 18
    l2_breakout_buffer_pct: float = 0.00015
    l2_retest_buffer_pct: float = 0.00008
    l2_pullback_max_bars: int = 14

    # 與 strategy_c_core L2Gate 對齊（與 B 同參數名）
    structure_lookback_bars: int = 20
    breakout_buffer_pct: float = 0.0002
    retest_buffer_pct: float = 0.0001
    pullback_max_bars: int = 12
    stop_buffer_pct: float = 0.0001
    tp_safety_buffer_pct: float = 0.0001
    entry_interval: str = "3m"

    # ==================== TP 調整（維持正期望值） ====================
    tp_rr_multiple: float = 1.5
    tp_min_fixed_pct: float = 0.006  # 0.60%
    
    # ✅ V5.3新增: 倉位硬上限 (防止stop過近導致qty爆大)
    max_notional_pct_of_equity: float = 0.45
    max_margin_pct_of_available: float = 0.45
    
    # ==================== 成本模型 ====================
    fee_maker: float = 0.00018  # 0.018%
    fee_taker: float = 0.00045  # 0.045%
    slippage_buffer: float = 0.0005  # 0.05%
    # ✅ 最小淨利
    min_tp_after_costs_pct: float = 0.0020  # 0.20%
    
    # ==================== 資料週期 ====================
    tf_filter: str = "15m"
    tf_entry: str = "3m"
    lookback_filter: int = 200
    lookback_entry: int = 300
    
    # ==================== 環境設定 ====================
    # ✅ V5.3修正: 強制testnet必須有testnet key，不fallback避免混用
    binance_api_key: str = os.getenv("BINANCE_API_KEY", "")
    binance_api_secret: str = os.getenv("BINANCE_API_SECRET", "")
    binance_base_url: str = os.getenv("BINANCE_BASE_URL", "https://testnet.binancefuture.com")
    
    # ==================== MVP Gate ====================
    mvp_gate_account_min_available_usdt: float = 300.0
    mvp_gate_account_max_margin_ratio: float = 0.50
    mvp_gate_min_notional: float = 5.0
    mvp_gate_log_decisions: bool = True
    
    # ==================== 執行安全 ====================
    global_lock_path: str = "/tmp/futures_account_lock_strategy_c"
    global_lock_timeout: int = 10
    
    execution_max_retries: int = 3
    execution_fill_timeout: int = 30
    execution_query_interval: int = 1
    
    # ==================== Telegram ====================
    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")
    enable_telegram: bool = bool(telegram_bot_token and telegram_chat_id)
    telegram_prefix: str = "⚡ Strategy-C"
    
    # ==================== 數據庫與日誌 ====================
    database_path: str = "/home/trader/trading_system/bots/bot_c/db/strategy_c_unified.db"
    log_file: str = "/home/trader/trading_system/bots/bot_c/logs/strategy_c_unified.log"
    log_level: str = "INFO"
    
    # ==================== 系統設定 ====================
    main_loop_interval: int = 10
    signal_check_interval: int = 180  # 3分鐘
    margin_type: str = "ISOLATED"
    
    def __post_init__(self):
        """配置驗證"""
        self._ensure_directories()
        self._validate_config()
    
    def _ensure_directories(self):
        """
        ✅ V5.2新增: 確保必要的目錄存在
        避免因目錄不存在導致IOError
        """
        # 確保日誌目錄存在
        log_dir = Path(self.log_file).parent
        if not log_dir.exists():
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
                print(f"✅ 創建日誌目錄: {log_dir}")
            except Exception as e:
                print(f"⚠️ 無法創建日誌目錄 {log_dir}: {e}")
        
        # 確保數據庫目錄存在
        db_dir = Path(self.database_path).parent
        if not db_dir.exists():
            try:
                db_dir.mkdir(parents=True, exist_ok=True)
                print(f"✅ 創建數據庫目錄: {db_dir}")
            except Exception as e:
                print(f"⚠️ 無法創建數據庫目錄 {db_dir}: {e}")
    
    def _validate_config(self):
        """驗證配置"""
        
        print("\n" + "="*60)
        print("🔍 Strategy C 高報酬版 配置驗證 (V1.0)")
        print("="*60)
        
        errors = []
        warnings = []
        
        # 1. 基本
        print(f"\n1. 基本:")
        print(f"   策略ID: {self.strategy_id}")
        print(f"   版本: {self.strategy_version}")
        print(f"   商品: {self.symbol}")
        print(f"   模式: {self.mode}")
        
        # 2. API Key
        print(f"\n2. API Key:")
        if not self.binance_api_key:
            errors.append("❌ BINANCE_API_KEY 未設置 (testnet模式必須設置)")
        else:
            print(f"   ✅ TESTNET Key已設置 ({self.binance_api_key[:8]}...)")
        
        # 3. 交易節流
        print(f"\n3. 交易節流:")
        print(f"   每日: {self.max_trades_per_day}筆")
        print(f"   每小時: {self.max_trades_per_hour}筆")
        print(f"   交易後冷卻: {self.cooldown_minutes_after_trade}分")
        print(f"   虧損後冷卻: {self.cooldown_minutes_after_loss}分")
        print(f"   最大連虧: {self.max_consecutive_losses}次")
        
        # 4. 風險
        print(f"\n4. 風險與槓桿:")
        print(f"   單筆風險: {self.risk_per_trade_pct:.2%}")
        print(f"   槓桿: {self.max_leverage}x")
        print(f"   止損範圍: {self.min_stop_distance_pct:.2%} - {self.max_stop_distance_pct:.2%}")
        
        # 5. 成本
        print(f"\n5. 成本模型:")
        print(f"   Maker費率: {self.fee_maker:.2%}")
        print(f"   Taker費率: {self.fee_taker:.2%}")
        print(f"   滑點緩衝: {self.slippage_buffer:.2%}")
        round_trip = self.fee_taker * 2 + self.slippage_buffer
        print(f"   往返成本: ~{round_trip:.2%}")
        print(f"   最小淨利: {self.min_tp_after_costs_pct:.2%}")
        
        if self.min_tp_after_costs_pct <= round_trip:
            warnings.append(f"⚠️ 最小淨利({self.min_tp_after_costs_pct:.2%}) <= 成本({round_trip:.2%})")
        
        # 6. 資料週期
        print(f"\n6. 資料週期:")
        print(f"   過濾週期: {self.tf_filter}")
        print(f"   進場週期: {self.tf_entry}")
        print(f"   過濾回看: {self.lookback_filter}根")
        print(f"   進場回看: {self.lookback_entry}根")
        
        # 7. 目錄檢查
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
        
        # 總結
        print("\n" + "="*60)
        
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
        
        print("="*60)
        print(f"\n🚀 {self.strategy_version}")
        print(f"模式: {self.mode}")
        print(f"風險: {self.risk_per_trade_pct:.2%} × {self.max_leverage}x")
        print(f"限制: {self.max_trades_per_day}筆/日, {self.max_trades_per_hour}筆/時")
        print(f"成本Gate: TP扣成本 >= {self.min_tp_after_costs_pct:.2%}")
        print("="*60 + "\n")


def get_strategy_c_config():
    """獲取策略C配置"""
    return ConfigStrategyC()


if __name__ == "__main__":
    print("正在載入Strategy C 高報酬版 配置 (V1.0)...")
    config = get_strategy_c_config()
    print("\n✅ 配置載入成功!")
