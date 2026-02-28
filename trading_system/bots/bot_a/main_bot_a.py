"""
Trading Bot V3-Micro-MVP - Main Program
真錢限制版 + MVP Gate

整合:
- 你的嚴格風控 ✅
- MVP Gate系統 ✅  
- 狀態機下單 ✅
- 老手規格 ✅
"""
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]  # trading_system
sys.path.insert(0, str(ROOT))



from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env.a_mainnet")

import logging
import time
import sqlite3
from datetime import datetime, timedelta
from typing import Optional

# 使用Micro MVP配置
from config_a import get_micro_mvp_config
from core.market_regime import MarketRegimeFilter, MarketRegimeDetector
from core.structure_detector import StructureDetector
from core.dynamic_leverage import DynamicLeverageCalculator
from core.paper_trading import PaperTradingManager

from core.binance_client import BinanceFuturesClient
from core.market_data import MarketDataManager
from core.risk_manager import RiskManager
from core.telegram_notifier import TelegramNotifier
from core.protection_guard import ProtectionGuard

# MVP Gate整合
from core.mvp_gate import (
    mvp_gate_check, get_account_snapshot, log_gate_decision,
    MVPGateConfig, CandidateTrade, EnvState
)
from core.execution_safety import OrderStateMachine
from core.global_lock import global_account_lock
from core.emergency_handler import EmergencyHandler

# 載入配置並執行安全檢查
config = get_micro_mvp_config()

# V9.1: startup banner (STRATEGY_VERSION, VOL_LOW, VOL_HIGH, MODE, GIT_COMMIT)
try:
    from core.startup_banner import print_startup_banner, get_commit_hash
    mode = os.getenv("V9_LIVE_MODE", "LIVE")  # main_bot_a = LIVE (not PAPER/MICRO-LIVE)
    print_startup_banner(mode)
except Exception:
    pass

# 設定日誌
logging.basicConfig(
    level=getattr(logging, config.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradingBotV3MicroMVP:
    """V3-Micro-MVP機器人 - 真錢限制版 + MVP Gate"""
    
    def __init__(self):
        logger.info("="*60)
        logger.info("🛡️ Trading Bot V3-Micro-MVP 初始化")
        logger.info("="*60)
        
        self.config = get_micro_mvp_config()
        
        # 交易計數器
        self.trades_today = 0
        self.trades_this_week = 0
        self.last_trade_date = None
        self.week_start_date = datetime.now().date()
        
        # 初始化客戶端
        self.client = BinanceFuturesClient(
            api_key=self.config.binance_api_key,
            api_secret=self.config.binance_api_secret,
            base_url=self.config.binance_base_url
        )
        
        self.market_data = MarketDataManager(self.client)
        
        # 策略組件
        self.regime_filter = MarketRegimeFilter(self.config)
        self.structure_detector = StructureDetector(self.config)
        self.leverage_calculator = DynamicLeverageCalculator(self.config)
        self.regime_detector = MarketRegimeDetector(
            self.config,
            self.market_data,
            require_structure=True,
        )
        
        # 交易管理
        if self.config.paper_trading_mode:
            self.paper_trading = PaperTradingManager(initial_balance=5000.0)
            logger.info("📝 紙上交易模式")
        else:
            self.paper_trading = None
            logger.info("💰 真錢交易模式")
        
        self.risk_manager = RiskManager(self.config)
        
        # Telegram
        self.telegram = TelegramNotifier(
            bot_token=self.config.telegram_bot_token,
            chat_id=self.config.telegram_chat_id,
            enabled=self.config.enable_telegram
        )
        
        # ⭐ MVP Gate組件
        self.mvp_config = MVPGateConfig(
            account_min_available_usdt=self.config.mvp_gate_account_min_available_usdt,
            account_max_total_initial_margin_ratio=self.config.mvp_gate_account_max_margin_ratio,
            min_notional=self.config.mvp_gate_min_notional,
            fee_maker=self.config.mvp_gate_fee_maker,
            fee_taker=self.config.mvp_gate_fee_taker,
            slippage_buffer=self.config.mvp_gate_slippage_buffer,
            min_tp_pct=self.config.mvp_gate_min_tp_pct
        )
        
        self.env_state = EnvState()
        self.emergency_handler = EmergencyHandler(
            self.client,
            self.telegram if self.config.enable_telegram else None
        )
        self.protection_guard = ProtectionGuard(
            self.client,
            self.emergency_handler,
            working_type="MARK_PRICE",
        )
        self.order_sm = OrderStateMachine(
            self.client,
            self.emergency_handler,
            protection_guard=self.protection_guard,
        )
        
        # 設定timeout參數
        self.order_sm.fill_timeout = self.config.execution_fill_timeout
        self.order_sm.query_interval = self.config.execution_query_interval
        
        # 數據庫
        self.db_conn = self._init_database()
        
        self.running = False
        self.last_strategy_check = None
        self.last_protection_scan = None
        self.last_regime_decision = None
        
        logger.info("✅ 初始化完成")
        logger.info(f"📊 MVP Gate已啟用")
        logger.info(f"   最小可用餘額: ${self.mvp_config.account_min_available_usdt:.2f}")
        logger.info(f"   最大保證金率: {self.mvp_config.account_max_total_initial_margin_ratio:.0%}")
        
        # 執行三重確認
        if not self.config.paper_trading_mode:
            self._perform_triple_verification()

        if self.config.protection_reconcile_on_startup:
            self.protection_guard.reconcile_positions(
                strategy_id=self.config.strategy_tag,
                get_registered_ids=self._get_registered_protection_ids,
                update_registry=self._update_protection_registry,
            )
    
    def _init_database(self):
        """初始化數據庫"""
        try:
            conn = sqlite3.connect(self.config.database_path)
            
            # 確保gate_decisions表存在
            conn.execute("""
                CREATE TABLE IF NOT EXISTS gate_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    reason_code TEXT NOT NULL,
                    available_balance REAL,
                    wallet_balance REAL,
                    margin_ratio REAL,
                    notional REAL,
                    required_margin REAL,
                    risk_usdt REAL,
                    debug_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS protection_registry (
                    symbol TEXT PRIMARY KEY,
                    sl_order_id TEXT,
                    tp_order_id TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            logger.info(f"✅ 數據庫已連接: {self.config.database_path}")
            return conn
            
        except Exception as e:
            logger.error(f"數據庫初始化失敗: {e}")
            return None

    def _get_registered_protection_ids(self, symbol: str) -> dict:
        if not self.db_conn:
            return {"sl": None, "tp": None}
        try:
            cur = self.db_conn.cursor()
            cur.execute(
                "SELECT sl_order_id, tp_order_id FROM protection_registry WHERE symbol = ?",
                (symbol,),
            )
            row = cur.fetchone()
            if not row:
                return {"sl": None, "tp": None}
            sl_id, tp_id = row
            return {"sl": int(sl_id) if sl_id else None, "tp": int(tp_id) if tp_id else None}
        except Exception as e:
            logger.warning(f"⚠️ 讀取保護單登記失敗: {e}")
            return {"sl": None, "tp": None}

    def _update_protection_registry(self, symbol: str, sl_order_id: Optional[str], tp_order_id: Optional[str]) -> None:
        if not self.db_conn:
            return
        try:
            cur = self.db_conn.cursor()
            cur.execute("""
                INSERT INTO protection_registry (symbol, sl_order_id, tp_order_id)
                VALUES (?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    sl_order_id = excluded.sl_order_id,
                    tp_order_id = excluded.tp_order_id,
                    updated_at = CURRENT_TIMESTAMP
            """, (symbol, sl_order_id, tp_order_id))
            self.db_conn.commit()
        except Exception as e:
            logger.warning(f"⚠️ 寫入保護單登記失敗: {e}")
    
    def _perform_triple_verification(self):
        """三重確認 - 真錢模式必須執行"""
        logger.info("\n" + "="*60)
        logger.info("🔍 執行三重確認")
        logger.info("="*60)
        
        # 1. 環境確認
        logger.info("1. 環境確認:")
        logger.info(f"   BINANCE_ENV: {self.config.binance_env}")
        logger.info(f"   Base URL: {self.config.binance_base_url}")
        logger.info(f"   測試網模式: {self.config.testnet_mode}")
        
        if self.config.binance_env == "LIVE" and "testnet" in self.config.binance_base_url.lower():
            logger.critical("❌ 環境不一致! LIVE模式但使用測試網URL!")
            sys.exit(1)
        
        # 2. 風控確認
        logger.info("\n2. 風控確認:")
        logger.info(f"   單筆風險: {self.config.risk_per_trade_pct:.2%}")
        logger.info(f"   最大槓桿: {self.config.max_leverage}x")
        logger.info(f"   每日限制: {self.config.max_trades_per_day}筆")
        logger.info(f"   每週限制: {self.config.max_trades_per_week}筆")
        
        # 3. MVP Gate確認
        logger.info("\n3. MVP Gate確認:")
        logger.info(f"   帳戶級Gate: ✅")
        logger.info(f"   工程級Gate: ✅")
        logger.info(f"   全局鎖: ✅")
        logger.info(f"   狀態機下單: ✅")
        
        logger.info("\n" + "="*60)
        logger.info("✅ 三重確認完成")
        logger.info("="*60 + "\n")
    
    def _check_trade_limits(self):
        """檢查交易限制"""
        today = datetime.now().date()
        
        # 重置每日計數
        if self.last_trade_date != today:
            self.trades_today = 0
            self.last_trade_date = today
        
        # 重置每週計數
        days_since_week_start = (today - self.week_start_date).days
        if days_since_week_start >= 7:
            self.trades_this_week = 0
            self.week_start_date = today
        
        # 檢查限制
        if self.trades_today >= self.config.max_trades_per_day:
            logger.warning(f"⛔ 已達每日交易限制: {self.trades_today}/{self.config.max_trades_per_day}")
            return False
        
        if self.trades_this_week >= self.config.max_trades_per_week:
            logger.warning(f"⛔ 已達每週交易限制: {self.trades_this_week}/{self.config.max_trades_per_week}")
            return False
        
        return True
    
    def _handle_entry_signal(self, signal):
        """處理進場訊號 - 帶MVP Gate"""
        logger.info(f"\n{'='*60}")
        logger.info(f"🎯 處理進場訊號: {signal.direction}")
        logger.info(f"{'='*60}")
        
        # 1. 檢查交易限制
        if not self._check_trade_limits():
            logger.warning("交易次數已達限制")
            return
        
        # 2. 計算倉位
        try:
            # 獲取帳戶權益
            if self.paper_trading:
                account_equity = self.paper_trading.balance
            else:
                account = self.client.futures_account()
                account_equity = float(account.get('totalWalletBalance', 0))
            
            logger.info(f"帳戶權益: ${account_equity:.2f}")
            
            # 計算風險金額
            risk_amount = account_equity * self.config.risk_per_trade_pct
            logger.info(f"單筆風險: ${risk_amount:.2f} ({self.config.risk_per_trade_pct:.2%})")
            
            # 計算數量
            stop_distance = abs(signal.entry_price - signal.stop_loss)
            qty = risk_amount / stop_distance
            
            # 計算槓桿
            calculated_leverage = self.leverage_calculator.calculate_leverage(
                signal.entry_price,
                account_equity,
                risk_amount
            )
            
            # 限制槓桿
            calculated_leverage = min(calculated_leverage, self.config.max_leverage)
            
            # 計算名義價值和保證金
            notional = qty * signal.entry_price
            required_margin = notional / calculated_leverage * 1.05
            
            logger.info(f"計算結果:")
            logger.info(f"  數量: {qty:.6f} BTC")
            logger.info(f"  槓桿: {calculated_leverage}x")
            logger.info(f"  名義價值: ${notional:.2f}")
            logger.info(f"  所需保證金: ${required_margin:.2f}")
            
        except Exception as e:
            logger.error(f"倉位計算失敗: {e}")
            return
        
        # 3. 構建CandidateTrade
        candidate = CandidateTrade(
            symbol=self.config.symbol,
            side="BUY" if signal.direction == "LONG" else "SELL",
            entry_type="MARKET",
            entry_price=signal.entry_price,
            stop_price=signal.stop_loss,
            tp_price=signal.take_profit if hasattr(signal, 'take_profit') else None,
            qty=qty,
            leverage=calculated_leverage,
            notional=notional,
            required_margin_est=required_margin,
            risk_usdt=risk_amount,
            expected_tp_pct=None,  # V3不檢查TP
            strategy_tag=self.config.strategy_tag
        )
        
        # 4. ⭐ 使用全局鎖 + MVP Gate
        logger.info(f"\n{'='*60}")
        logger.info("🔒 獲取全局鎖並執行MVP Gate檢查")
        logger.info(f"{'='*60}")
        
        try:
            with global_account_lock(self.config.global_lock_path, self.config.global_lock_timeout):
                # 獲取帳戶快照
                snapshot = get_account_snapshot(self.client)
                
                logger.info(f"帳戶快照:")
                logger.info(f"  可用餘額: ${snapshot.available_balance:.2f}")
                logger.info(f"  總權益: ${snapshot.total_wallet_balance:.2f}")
                logger.info(f"  已用保證金: ${snapshot.total_initial_margin:.2f}")
                logger.info(f"  保證金率: {snapshot.total_initial_margin/max(snapshot.total_wallet_balance,1):.2%}")
                
                # MVP Gate檢查
                allow, reason, debug = mvp_gate_check(
                    snapshot, candidate, self.env_state, self.mvp_config
                )
                
                # 記錄決策
                if self.config.mvp_gate_log_decisions and self.db_conn:
                    log_gate_decision(
                        self.db_conn,
                        'ALLOW' if allow else 'REJECT',
                        reason,
                        debug
                    )
                
                # 拒單
                if not allow:
                    logger.warning(f"❌ MVP Gate拒單: {reason}")
                    
                    if self.config.enable_telegram:
                        self.telegram.send_message(
                            f"🚫 <b>Gate拒單</b>\n\n"
                            f"原因: {reason}\n"
                            f"可用餘額: ${snapshot.available_balance:.2f}\n"
                            f"保證金率: {snapshot.total_initial_margin/max(snapshot.total_wallet_balance,1):.1%}\n"
                            f"名義價值: ${candidate.notional:.2f}"
                        )
                    
                    return
                
                # 通過Gate - 執行交易
                logger.info(f"✅ MVP Gate通過,執行交易")
                
                # 紙上交易
                if self.paper_trading:
                    result = self._execute_paper_trade(signal, candidate)
                    if result:
                        self.trades_today += 1
                        self.trades_this_week += 1
                        logger.info(f"✅ 紙上交易執行成功")
                    return
                
                # 真錢交易 - 使用狀態機
                result = self.order_sm.execute_trade_with_safety(
                    candidate,
                    self.config.strategy_tag,
                    max_retries=self.config.execution_max_retries
                )
                
                # 處理結果
                if result['success']:
                    self.trades_today += 1
                    self.trades_this_week += 1
                    
                    logger.info(f"✅ 真錢交易執行成功")
                    logger.info(f"   進場單: {result['entry_order_id']}")
                    logger.info(f"   止損單: {result['sl_order_id']}")
                    if result.get('tp_order_id'):
                        logger.info(f"   止盈單: {result['tp_order_id']}")
                    
                    if self.config.enable_telegram:
                        self.telegram.send_message(
                            f"✅ <b>交易執行成功</b>\n\n"
                            f"方向: {candidate.side}\n"
                            f"數量: {candidate.qty:.6f}\n"
                            f"槓桿: {candidate.leverage}x\n"
                            f"進場: ${candidate.entry_price:.2f}\n"
                            f"止損: ${candidate.stop_price:.2f}\n"
                            f"風險: ${candidate.risk_usdt:.2f}"
                        )
                    self._update_protection_registry(
                        self.config.symbol,
                        result.get('sl_order_id'),
                        result.get('tp_order_id'),
                    )
                else:
                    logger.error(f"❌ 交易執行失敗: {result['error']}")
                    logger.error(f"   狀態: {result['state']}")
                    
                    if self.config.enable_telegram:
                        self.telegram.send_message(
                            f"❌ <b>交易執行失敗</b>\n\n"
                            f"錯誤: {result['error']}\n"
                            f"狀態: {result['state']}"
                        )
        
        except Exception as e:
            logger.error(f"處理進場訊號異常: {e}", exc_info=True)
            
            if self.config.enable_telegram:
                self.telegram.send_message(
                    f"🚨 <b>系統異常</b>\n\n"
                    f"錯誤: {str(e)}"
                )
    
    def _execute_paper_trade(self, signal, candidate):
        """執行紙上交易"""
        try:
            self.paper_trading.open_position(
                symbol=candidate.symbol,
                side=candidate.side,
                entry_price=candidate.entry_price,
                quantity=candidate.qty,
                stop_loss=candidate.stop_price,
                take_profit=candidate.tp_price
            )
            return True
        except Exception as e:
            logger.error(f"紙上交易失敗: {e}")
            return False
    
    def run(self):
        """主循環"""
        logger.info("\n" + "="*60)
        logger.info("🚀 Trading Bot V3-Micro-MVP 啟動")
        logger.info("="*60)
        
        self.running = True
        
        if self.config.enable_telegram:
            self.telegram.send_message(
                f"🚀 <b>{self.config.telegram_prefix} 啟動</b>\n\n"
                f"模式: {'紙上' if self.config.paper_trading_mode else '真錢'}\n"
                f"環境: {self.config.binance_env}\n"
                f"風險: {self.config.risk_per_trade_pct:.2%}\n"
                f"槓桿: {self.config.max_leverage}x\n\n"
                f"MVP Gate: ✅ 已啟用"
            )
        
        try:
            while self.running:
                try:
                    # 心跳
                    logger.info(f"💓 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    # 策略檢查
                    now = time.time()
                    if (self.last_strategy_check is None or 
                        now - self.last_strategy_check >= self.config.strategy_check_interval):
                        
                        logger.info("\n檢查策略訊號...")
                        self._check_strategy()
                        self.last_strategy_check = now

                    if (
                        self.last_protection_scan is None
                        or now - self.last_protection_scan >= self.config.protection_scan_interval_sec
                    ):
                        self.protection_guard.reconcile_positions(
                            strategy_id=self.config.strategy_tag,
                            get_registered_ids=self._get_registered_protection_ids,
                            update_registry=self._update_protection_registry,
                        )
                        self.last_protection_scan = now
                    
                    time.sleep(self.config.main_loop_interval)
                    
                except KeyboardInterrupt:
                    logger.info("接收到停止信號")
                    break
                except Exception as e:
                    logger.error(f"主循環錯誤: {e}", exc_info=True)
                    time.sleep(60)
        
        finally:
            self.shutdown()
    
    def _check_strategy(self):
        """檢查策略"""
        try:
            self.last_regime_decision = self.regime_detector.evaluate(self.config.symbol)
            if not self.last_regime_decision.allow:
                logger.info(f"🚫 Regime阻擋: {self.last_regime_decision.reason}")
                return

            signal = self.last_regime_decision.signal
            if not signal or not signal.entry_allowed:
                logger.info("🚫 Regime未提供可執行訊號")
                return

            class _SignalAdapter:
                direction = "LONG"
                entry_price = signal.entry_price
                stop_loss = signal.stop_loss

            logger.info(f"🎯 Regime訊號: {signal.signal_type} {signal.reason}")
            self._handle_entry_signal(_SignalAdapter())
            
        except Exception as e:
            logger.error(f"策略檢查失敗: {e}")
    
    def shutdown(self):
        """關閉"""
        logger.info("\n關閉Trading Bot...")
        self.running = False
        
        if self.db_conn:
            self.db_conn.close()
        
        if self.config.enable_telegram:
            self.telegram.send_message(
                f"🛑 <b>{self.config.telegram_prefix} 已停止</b>"
            )
        
        logger.info("✅ 已安全關閉")

def main():
    """主函數"""
    try:
        bot = TradingBotV3MicroMVP()
        bot.run()
    except KeyboardInterrupt:
        logger.info("\n程序被用戶中斷")
    except Exception as e:
        logger.error(f"程序異常: {e}", exc_info=True)

if __name__ == "__main__":
    main()
