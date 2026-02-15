"""
Strategy B Main Bot - 完全正確版 V5.3
所有接口完全對齊,不會再有AttributeError/TypeError

V5.3更新:
- 新增倉位硬上限 (max_notional, max_margin)
- 防止stop過近導致qty爆大
"""
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]  # trading_system
sys.path.insert(0, str(ROOT)) 



from dotenv import load_dotenv

# 依 MODE 載入：正式網用 .env.b_live，否則用 .env.b_testnet
_bot_dir = Path(__file__).resolve().parent
_env_live = _bot_dir / ".env.b_live"
_env_testnet = _bot_dir / ".env.b_testnet"
load_dotenv(dotenv_path=_env_live if os.getenv("MODE") == "LIVE" else _env_testnet)

import logging
import time
import sqlite3

from datetime import datetime
from typing import Optional, Tuple
from pathlib import Path

from config_b import get_strategy_b_config
from strategy_b_core import StrategyBCore

from core.binance_client import BinanceFuturesClient
from core.market_data import MarketDataManager
from core.telegram_notifier import TelegramNotifier
from core.execution_safety import OrderStateMachine
from core.global_lock import global_account_lock
from core.emergency_handler import EmergencyHandler

from core.mvp_gate import (
    mvp_gate_check, get_account_snapshot, log_gate_decision,
    MVPGateConfig, CandidateTrade, EnvState
)

config = get_strategy_b_config()

# ✅ V5.3: 確保日誌目錄存在後再設置logging
log_dir = Path(config.log_file).parent
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=getattr(logging, config.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class StrategyBBot:
    """Strategy B Trading Bot - 完全正確版 V5.3"""
    
    def __init__(self):
        logger.info("="*60)
        logger.info("⚡ Strategy B Bot 完全正確版 V5.3 初始化")
        logger.info("="*60)
        
        self.config = get_strategy_b_config()
        
        # Binance客戶端
        self.client = BinanceFuturesClient(
            api_key=self.config.binance_api_key,
            api_secret=self.config.binance_api_secret,
            base_url=self.config.binance_base_url
        )
        
        self.market_data = MarketDataManager(self.client)
        
        # Telegram
        self.telegram = TelegramNotifier(
            bot_token=self.config.telegram_bot_token,
            chat_id=self.config.telegram_chat_id,
            enabled=self.config.enable_telegram
        )
        
        # EmergencyHandler
        self.emergency_handler = EmergencyHandler(
            self.client,
            self.telegram if self.config.enable_telegram else None
        )
        
        # ✅ OrderStateMachine必須有is_safe()方法
        self.order_sm = OrderStateMachine(self.client, self.emergency_handler)
        self.order_sm.fill_timeout = self.config.execution_fill_timeout
        self.order_sm.query_interval = self.config.execution_query_interval
        
        # ✅ MVP Gate - 參數名完全對齊!
        self.mvp_config = MVPGateConfig(
            account_min_available_usdt=self.config.mvp_gate_account_min_available_usdt,
            account_max_total_initial_margin_ratio=self.config.mvp_gate_account_max_margin_ratio,
            min_notional=self.config.mvp_gate_min_notional,
            fee_maker=self.config.fee_maker,
            fee_taker=self.config.fee_taker,
            slippage_buffer=self.config.slippage_buffer,
            min_tp_pct=self.config.min_tp_after_costs_pct  # ✅ 不硬編!
        )
        
        self.env_state = EnvState()
        
        # 策略B核心
        self.strategy_core = StrategyBCore(self.config, self.market_data)
        
        # 數據庫
        self.db_conn = None
        self._init_database()
        
        # 時間追蹤
        self.last_signal_check = None
        
        # ✅ Telegram啟動通知 - 欄位名完全對齊!
        if self.config.enable_telegram:
            self.telegram.send_message(
                f"⚡ <b>Strategy B 完全正確版 V5.3 啟動</b>\n\n"
                f"版本: {self.config.strategy_version}\n"  # ✅ strategy_version
                f"商品: {self.config.symbol}\n"
                f"模式: {self.config.mode}\n"  # ✅ mode
                f"風險: {self.config.risk_per_trade_pct:.2%}\n"
                f"限制: {self.config.max_trades_per_day}筆/日, {self.config.max_trades_per_hour}筆/時\n"  # ✅ 正確欄位名
                f"成本Gate: TP扣成本 >= {self.config.min_tp_after_costs_pct:.2%}"
            )
        
        logger.info("\n✅ Strategy B Bot V5.3 初始化完成")
        logger.info("="*60 + "\n")
    
    def _init_database(self):
        """初始化數據庫"""
        try:
            # ✅ V5.3: 確保數據庫目錄存在
            db_dir = Path(self.config.database_path).parent
            db_dir.mkdir(parents=True, exist_ok=True)
            
            self.db_conn = sqlite3.connect(self.config.database_path)
            cursor = self.db_conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS strategy_b_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    signal_type TEXT,
                    pattern TEXT,
                    entry_price REAL,
                    stop_loss REAL,
                    tp1_price REAL,
                    qty REAL,
                    leverage INTEGER,
                    entry_order_id TEXT,
                    sl_order_id TEXT,
                    tp_order_id TEXT,
                    result TEXT,
                    pnl REAL,
                    notes TEXT
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS strategy_b_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    trades_today INTEGER DEFAULT 0,
                    trades_this_hour INTEGER DEFAULT 0,
                    consecutive_losses INTEGER DEFAULT 0,
                    last_trade_time TIMESTAMP,
                    cooldown_until TIMESTAMP
                )
            """)
            
            cursor.execute("INSERT OR IGNORE INTO strategy_b_state (id) VALUES (1)")
            
            self.db_conn.commit()
            logger.info("✅ 數據庫初始化完成")
            
        except Exception as e:
            logger.error(f"❌ 數據庫初始化失敗: {e}")
            self.db_conn = None
    
    def run(self):
        """主循環"""
        logger.info("="*60)
        logger.info("🚀 Strategy B Bot V5.3 啟動")
        logger.info("="*60)
        logger.info(f"心跳: {self.config.main_loop_interval}秒")
        logger.info(f"訊號: {self.config.signal_check_interval}秒")
        logger.info("="*60 + "\n")
        
        try:
            while True:
                current_time = datetime.now()
                
                logger.info(f"💓 {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
                
                should_check = (
                    self.last_signal_check is None or
                    (current_time - self.last_signal_check).total_seconds() >= self.config.signal_check_interval
                )
                
                if should_check:
                    logger.info("\n🔍 檢查策略訊號...")
                    self._check_and_execute_signal()
                    self.last_signal_check = current_time
                
                time.sleep(self.config.main_loop_interval)
                
        except KeyboardInterrupt:
            logger.info("\n⚠️ 收到停止信號")
            self._shutdown()
        except Exception as e:
            logger.error(f"\n❌ Bot運行異常: {e}", exc_info=True)
            self._shutdown()
    
    def _check_global_lock_status(self) -> bool:
        """
        ✅ V5.3新增: 檢查全局鎖狀態
        
        用於L0 Gate判斷是否有其他策略正在下單
        
        NOTE: 目前單策略運行時永遠返回False
        TODO: A+B雙策略並行時，需要實作真正的鎖狀態讀取
              可以通過檢查lock file是否被其他進程持有來實現
        
        Returns:
            bool: True=有鎖(其他策略正在下單), False=無鎖
        """
        lock_path = self.config.global_lock_path
        
        # 方案1: 簡單檢查lock file是否存在
        # 注意: 這不是完美的互斥檢測，但可以作為初步檢查
        # 真正的互斥是在下單時用 with global_account_lock() 確保的
        
        # 目前單策略運行，直接返回False
        # 未來A+B並行時，可以改為:
        # return os.path.exists(lock_path + ".locked")
        
        return False
    
    def _get_flags(self) -> Tuple[bool, bool, bool]:
        """
        ✅ 獲取所有flags
        Returns: (has_lock, has_emergency, has_position)
        
        NOTE V5.3:
        - has_lock: 目前單策略運行時永遠False，下單時用global_account_lock()確保互斥
        - 未來A+B雙策略並行時，需要實作_check_global_lock_status()真正讀取鎖狀態
        """
        has_lock = self._check_global_lock_status()  # ✅ 改用方法，方便未來擴展
        has_emergency = getattr(self.emergency_handler, 'should_stop', False)
        has_position = self._has_any_position()
        
        return has_lock, has_emergency, has_position
    
    def _check_and_execute_signal(self):
        """檢查訊號並執行"""
        try:
            # ✅ 獲取flags
            has_lock, has_emergency, has_position = self._get_flags()
            
            # ✅ 統一的check_for_signal接口
            signal = self.strategy_core.check_for_signal(
                execution_safety=self.order_sm,  # OrderStateMachine物件(有is_safe())
                has_lock=has_lock,
                has_emergency=has_emergency,
                has_position=has_position
            )
            
            if signal is None:
                logger.info("無訊號")
                return
            
            logger.info("\n" + "🎯"*30)
            logger.info("🎯 收到訊號,準備執行")
            logger.info("🎯"*30)
            
            self._execute_trade(signal)
            
        except Exception as e:
            logger.error(f"❌ 訊號檢查失敗: {e}", exc_info=True)
    
    def _execute_trade(self, signal):
        """執行交易"""
        
        try:
            # 計算倉位
            logger.info("\n📊 計算倉位...")
            
            account = self.client.get_account()
            account_equity = float(account.get('totalWalletBalance', 0))
            available_balance = float(account.get('availableBalance', 0))
            
            logger.info(f"帳戶權益: ${account_equity:.2f}")
            logger.info(f"可用餘額: ${available_balance:.2f}")
            
            risk_amount = account_equity * self.config.risk_per_trade_pct
            logger.info(f"單筆風險: ${risk_amount:.2f} ({self.config.risk_per_trade_pct:.2%})")
            
            stop_distance = abs(signal.entry_price - signal.stop_loss)
            qty_from_risk = risk_amount / stop_distance
            
            leverage = self.config.max_leverage
            
            # ✅ V5.3新增: 倉位硬上限檢查
            # 1. 名義價值上限
            max_notional = account_equity * self.config.max_notional_pct_of_equity
            qty_from_notional = max_notional / signal.entry_price
            
            # 2. 保證金上限  
            max_margin = available_balance * self.config.max_margin_pct_of_available
            max_notional_from_margin = max_margin * leverage
            qty_from_margin = max_notional_from_margin / signal.entry_price
            
            # 取最小值
            qty = min(qty_from_risk, qty_from_notional, qty_from_margin)
            
            logger.info(f"\n📊 倉位限制檢查:")
            logger.info(f"  風險計算qty: {qty_from_risk:.6f}")
            logger.info(f"  名義價值上限qty: {qty_from_notional:.6f} (max notional: ${max_notional:.2f})")
            logger.info(f"  保證金上限qty: {qty_from_margin:.6f} (max margin: ${max_margin:.2f})")
            logger.info(f"  → 最終qty: {qty:.6f}")
            
            if qty < qty_from_risk:
                logger.warning(f"⚠️ qty被硬上限縮減: {qty_from_risk:.6f} → {qty:.6f}")
            
            notional = qty * signal.entry_price
            required_margin = notional / leverage * 1.05
            
            logger.info(f"\n計算結果:")
            logger.info(f"  數量: {qty:.6f} BTC")
            logger.info(f"  槓桿: {leverage}x")
            logger.info(f"  名義價值: ${notional:.2f}")
            logger.info(f"  所需保證金: ${required_margin:.2f}")
            
            # ✅ 構建CandidateTrade - 使用strategy_tag
            candidate = CandidateTrade(
                symbol=self.config.symbol,
                side="BUY" if signal.signal_type == "LONG" else "SELL",
                entry_type="MARKET",
                entry_price=signal.entry_price,
                stop_price=signal.stop_loss,
                tp_price=signal.tp1_price,
                qty=qty,
                leverage=leverage,
                notional=notional,
                required_margin_est=required_margin,
                risk_usdt=risk_amount,
                expected_tp_pct=signal.expected_tp1_pct,
                strategy_tag=self.config.strategy_tag  # ✅ 使用strategy_tag
            )
            
            # 全局鎖 + MVP Gate
            logger.info("\n🔒 獲取全局鎖並執行MVP Gate...")
            
            with global_account_lock(self.config.global_lock_path, self.config.global_lock_timeout):
                snapshot = get_account_snapshot(self.client)
                
                logger.info(f"帳戶快照:")
                logger.info(f"  可用: ${snapshot.available_balance:.2f}")
                logger.info(f"  總權益: ${snapshot.total_wallet_balance:.2f}")
                
                allow, reason, debug = mvp_gate_check(
                    snapshot, candidate, self.env_state, self.mvp_config
                )
                
                if self.config.mvp_gate_log_decisions and self.db_conn:
                    log_gate_decision(
                        self.db_conn,
                        'ALLOW' if allow else 'REJECT',
                        reason,
                        debug
                    )
                
                if not allow:
                    logger.warning(f"❌ MVP Gate拒單: {reason}")
                    
                    if self.config.enable_telegram:
                        self.telegram.send_message(
                            f"🚫 <b>Strategy B Gate拒單</b>\n\n"
                            f"原因: {reason}\n"
                            f"型態: {signal.pattern}"
                        )
                    
                    return
                
                logger.info(f"✅ MVP Gate通過,執行交易")
                
                self._execute_real_trade(signal, candidate)
                
        except Exception as e:
            logger.error(f"❌ 交易執行失敗: {e}", exc_info=True)
            
            if self.config.enable_telegram:
                self.telegram.send_message(
                    f"❌ <b>Strategy B 交易失敗</b>\n\n"
                    f"錯誤: {str(e)}"
                )
    
    def _execute_real_trade(self, signal, candidate: CandidateTrade):
        """執行真實交易"""
        
        logger.info("\n" + "💰"*30)
        logger.info("💰 執行真實交易")
        logger.info("💰"*30)
        
        try:
            # ✅ execute_trade_with_safety(candidate, strategy_tag)
            result = self.order_sm.execute_trade_with_safety(
                candidate,
                self.config.strategy_tag,  # ✅ 使用strategy_tag
                max_retries=self.config.execution_max_retries
            )
            
            if result['success']:
                logger.info(f"\n✅ 真錢交易執行成功")
                logger.info(f"   進場單: {result['entry_order_id']}")
                logger.info(f"   止損單: {result['sl_order_id']}")
                if result.get('tp_order_id'):
                    logger.info(f"   止盈單: {result['tp_order_id']}")
                
                # 只記錄進場
                self.strategy_core.record_trade_entry()
                
                # 記錄到數據庫
                if self.db_conn:
                    cursor = self.db_conn.cursor()
                    cursor.execute("""
                        INSERT INTO strategy_b_trades 
                        (signal_type, pattern, entry_price, stop_loss, tp1_price, qty, leverage, 
                         entry_order_id, sl_order_id, tp_order_id, result, notes)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        signal.signal_type,
                        signal.pattern,
                        signal.entry_price,
                        signal.stop_loss,
                        signal.tp1_price,
                        candidate.qty,
                        candidate.leverage,
                        result['entry_order_id'],
                        result['sl_order_id'],
                        result.get('tp_order_id'),
                        'EXECUTED',
                        str(result)
                    ))
                    self.db_conn.commit()
                
                # Telegram通知
                if self.config.enable_telegram:
                    total_cost = self.config.fee_taker * 2 + self.config.slippage_buffer
                    tp1_net = signal.expected_tp1_pct - total_cost
                    
                    self.telegram.send_message(
                        f"✅ <b>Strategy B 開倉成功</b>\n\n"
                        f"型態: {signal.pattern}\n"
                        f"方向: {signal.signal_type}\n"
                        f"進場: ${signal.entry_price:.2f}\n"
                        f"數量: {candidate.qty:.6f}\n"
                        f"槓桿: {candidate.leverage}x\n\n"
                        f"止損: ${signal.stop_loss:.2f} ({signal.stop_distance_pct:.2%})\n"
                        f"TP1: ${signal.tp1_price:.2f} (+{signal.expected_tp1_pct:.2%})\n"
                        f"扣成本後: +{tp1_net:.2%}\n\n"
                        f"Breakout: ${signal.breakout_level:.2f}\n"
                        f"進場單: {result['entry_order_id']}\n"
                        f"止損單: {result['sl_order_id']}"
                    )
                
                logger.info("\n" + "💰"*30)
                logger.info("💰 交易完成")
                logger.info("💰"*30)
            
            else:
                logger.error(f"❌ 下單失敗: {result.get('error', 'Unknown')}")
                
                if self.config.enable_telegram:
                    self.telegram.send_message(
                        f"❌ <b>Strategy B 下單失敗</b>\n\n"
                        f"錯誤: {result.get('error', 'Unknown')}"
                    )
            
        except Exception as e:
            logger.error(f"❌ 真實交易執行失敗: {e}", exc_info=True)
            
            if self.config.enable_telegram:
                self.telegram.send_message(
                    f"❌ <b>Strategy B 執行異常</b>\n\n"
                    f"錯誤: {str(e)}"
                )
    
    def _has_any_position(self) -> bool:
        """
        ✅ 必備方法: 檢查是否有倉位
        簡化版: BTCUSDT positionAmt != 0
        """
        try:
            positions = self.client.get_position_risk(symbol=self.config.symbol)
            
            for pos in positions:
                amt = float(pos.get('positionAmt', 0))
                if abs(amt) > 0:
                    logger.info(f"⚠️ 發現倉位: {amt}, 互斥模式不開新倉")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"檢查倉位失敗: {e}")
            return True  # 保守起見
    
    def _shutdown(self):
        """關閉bot"""
        logger.info("\n" + "="*60)
        logger.info("關閉 Strategy B Bot V5.3...")
        logger.info("="*60)
        
        if self.db_conn:
            self.db_conn.close()
            logger.info("✅ 數據庫已關閉")
        
        if self.config.enable_telegram:
            self.telegram.send_message(
                f"⚠️ <b>Strategy B V5.3 已停止</b>\n\n"
                f"時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
        
        logger.info("✅ 已安全關閉")
        sys.exit(0)


if __name__ == "__main__":
    logger.info("\n" + "="*60)
    logger.info("⚡ Strategy B: 完全正確版 V5.3")
    logger.info("="*60)
    logger.info("所有接口完全對齊 - 按照老手規格!")
    logger.info("✅ 止損距離使用config參數")
    logger.info("✅ API Key環境變數優化")
    logger.info("✅ 目錄自動創建")
    logger.info("✅ has_lock讀取準備 (未來A+B互斥用)")
    logger.info("="*60 + "\n")
    
    try:
        bot = StrategyBBot()
        bot.run()
    except Exception as e:
        logger.error(f"❌ Bot啟動失敗: {e}", exc_info=True)
        sys.exit(1)
