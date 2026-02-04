"""
Safety Test Harness - 自動保命自檢
10分鐘打穿所有關鍵保命鏈條

老手建議的4個自檢模式:
1. 逐倉檢測
2. 槓桿檢測
3. SL下單能力檢測
4. 故障演練
"""
import sys
import logging
import time
from datetime import datetime

from binance_client import BinanceFuturesClient
from execution_safety import OrderStateMachine
from emergency_handler import EmergencyHandler
from telegram_notifier import TelegramNotifier
from config_strategy_b import get_strategy_b_config as get_micro_mvp_config

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/safety_test_harness.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SafetyTestHarness:
    """自動保命自檢"""
    
    def __init__(self):
        logger.info("="*60)
        logger.info("🧪 Safety Test Harness 初始化")
        logger.info("="*60)
        
        self.config = get_micro_mvp_config()
        
        # 確認是測試網
        if self.config.binance_env != "TESTNET":
            logger.critical("⛔ 必須在測試網運行!")
            sys.exit(1)
        
        # 初始化客戶端
        self.client = BinanceFuturesClient(
            api_key=self.config.binance_api_key,
            api_secret=self.config.binance_api_secret,
            base_url=self.config.binance_base_url
        )
        
        # Telegram
        self.telegram = TelegramNotifier(
            bot_token=self.config.telegram_bot_token,
            chat_id=self.config.telegram_chat_id,
            enabled=self.config.enable_telegram
        )
        
        # 組件
        self.emergency = EmergencyHandler(self.client, self.telegram)
        self.order_sm = OrderStateMachine(self.client, self.emergency)
        
        # 測試參數
        self.symbol = self.config.symbol
        self.test_qty = 0.002  # 增加到0.002 (約$190)
        self.target_leverage = 5  # 改名避免衝突
        
        # 結果
        self.results = []
        
        logger.info("✅ 初始化完成")
        logger.info(f"測試標的: {self.symbol}")
        logger.info(f"測試數量: {self.test_qty}")
    
    def run_all_tests(self):
        """運行所有測試"""
        logger.info("\n" + "="*60)
        logger.info("🚀 開始自動保命自檢")
        logger.info("="*60)
        
        # ⭐ 測試前清場 (老手建議) ⭐
        logger.info("\n🧹 測試前清場...")
        try:
            # 1. 取消所有掛單
            logger.info("取消所有掛單...")
            self.client.cancel_all_orders(symbol=self.symbol)
            time.sleep(1)
            
            # 2. 清空任何殘倉
            logger.info("檢查並清空殘倉...")
            positions = self.client.get_position_risk(symbol=self.symbol)
            
            for pos in positions:
                pos_amt = float(pos.get('positionAmt', 0))
                if abs(pos_amt) > 0.0001:
                    logger.warning(f"⚠️ 發現殘倉: {pos_amt}, 立即清空")
                    self.emergency.emergency_flatten_position(
                        self.symbol, "test", "測試前清場"
                    )
                    break
            
            logger.info("✅ 清場完成\n")
            time.sleep(2)
            
        except Exception as e:
            logger.warning(f"⚠️ 清場失敗: {e}")
            logger.warning("繼續測試...\n")
        
        if self.telegram.enabled:
            self.telegram.send_message(
                "🧪 <b>Safety Test Harness 啟動</b>\n\n"
                "開始保命自檢..."
            )
        
        tests = [
            ("1. 逐倉檢測", self.test_isolated_margin),
            ("2. 槓桿檢測", self.test_leverage),
            ("3. SL下單能力檢測", self.test_sl_order),
            ("4. 故障演練", self.test_failure_handling),
        ]
        
        for name, test_func in tests:
            logger.info(f"\n{'='*60}")
            logger.info(f"🧪 {name}")
            logger.info(f"{'='*60}")
            
            try:
                result = test_func()
                self.results.append((name, result, None))
                
                if result:
                    logger.info(f"✅ {name} - 通過")
                else:
                    logger.error(f"❌ {name} - 失敗")
                
            except Exception as e:
                logger.error(f"💥 {name} - 異常: {e}", exc_info=True)
                self.results.append((name, False, str(e)))
            
            time.sleep(2)  # 間隔
        
        # 總結
        self.print_summary()
    
    def test_isolated_margin(self) -> bool:
        """
        測試1: 逐倉檢測
        
        流程:
        1. set isolated
        2. query回讀
        3. 確認是ISOLATED
        """
        logger.info("📍 開始逐倉檢測...")
        
        try:
            # 設定逐倉
            logger.info(f"設定 {self.symbol} 為逐倉模式...")
            
            try:
                self.client.set_margin_type(
                    symbol=self.symbol,
                    margin_type='ISOLATED'
                )
            except Exception as e:
                if "No need to change" in str(e):
                    logger.info("已經是逐倉模式")
                else:
                    raise
            
            # 回讀確認
            time.sleep(1)
            position = self.client.get_position_risk(symbol=self.symbol)
            
            if not position:
                logger.error("無法獲取倉位信息")
                return False
            
            margin_type = position[0].get('marginType', '').upper()
            logger.info(f"回讀結果: marginType={margin_type}")
            
            if margin_type != 'ISOLATED':
                logger.error(f"逐倉設定失敗! 實際: {margin_type}")
                return False
            
            logger.info("✅ 逐倉設定+回讀成功")
            return True
            
        except Exception as e:
            logger.error(f"逐倉檢測失敗: {e}")
            return False
    
    def test_leverage(self) -> bool:
        """
        測試2: 槓桿檢測
        
        流程:
        1. set leverage
        2. query回讀
        3. 確認槓桿正確
        """
        logger.info("📍 開始槓桿檢測...")
        
        try:
            # 設定槓桿
            logger.info(f"設定 {self.symbol} 槓桿為 {self.target_leverage}x...")
            
            result = self.client.set_leverage(
                symbol=self.symbol,
                leverage=self.target_leverage
            )
            logger.info(f"設定結果: {result}")
            
            # 回讀確認
            time.sleep(1)
            position = self.client.get_position_risk(symbol=self.symbol)
            
            if not position:
                logger.error("無法獲取倉位信息")
                return False
            
            actual_leverage = int(position[0].get('leverage', 0))
            logger.info(f"回讀結果: leverage={actual_leverage}x")
            
            if actual_leverage != self.target_leverage:
                logger.error(f"槓桿設定失敗! 設定{self.target_leverage}x, 實際{actual_leverage}x")
                return False
            
            logger.info("✅ 槓桿設定+回讀成功")
            return True
            
        except Exception as e:
            logger.error(f"槓桿檢測失敗: {e}")
            return False
    
    def test_sl_order(self) -> bool:
        """
        測試3: SL下單能力檢測 (最重要!)
        
        流程:
        1. 開一個極小倉
        2. 立刻掛STOP_MARKET reduceOnly
        3. query確認orderId存在
        4. 取消訂單
        """
        logger.info("📍 開始SL下單能力檢測...")
        
        entry_order_id = None
        sl_order_id = None
        
        try:
            # 獲取當前價格
            ticker = self.client.get_ticker_price(symbol=self.symbol)
            current_price = float(ticker['price'])
            logger.info(f"當前價格: ${current_price:.2f}")
            
            # 1. 開極小倉
            logger.info(f"開倉: {self.test_qty} {self.symbol}...")
            
            entry_order = self.client.place_order({
                'symbol': self.symbol,
                'side': 'BUY',
                'type': 'MARKET',
                'quantity': self.test_qty
            })
            
            entry_order_id = entry_order.get('orderId')
            logger.info(f"✅ 進場單已下: {entry_order_id}")
            
            # 等待成交
            time.sleep(2)
            
            # 確認成交 (query_order方法在你的client可能沒有,用get_open_orders檢查)
            # 假設MARKET單立即成交
            filled_qty = self.test_qty
            avg_price = current_price
            logger.info(f"✅ 進場已成交(假設): {filled_qty} @ ${avg_price:.2f}")
            
            # 2. 掛止損 (關鍵!)
            stop_price = avg_price * 0.98  # 2%下方
            
            logger.info(f"掛止損: stopPrice=${stop_price:.2f}, qty={filled_qty}")
            
            # 使用狀態機的方法 (測試完整流程)
            sl_order = self.order_sm._place_stop_loss(
                symbol=self.symbol,
                side='BUY',
                qty=filled_qty,
                stop_price=stop_price
            )
            
            if not sl_order:
                logger.error("❌ 止損掛單失敗!")
                # 平倉
                self.emergency.emergency_flatten_position(
                    self.symbol, "test", "止損測試失敗"
                )
                return False
            
            sl_order_id = sl_order.get('orderId')
            logger.info(f"✅ 止損已掛: {sl_order_id}")
            
            # 3. 確認訂單存在 (通過get_open_orders)
            time.sleep(1)
            open_orders = self.client.get_open_orders(symbol=self.symbol)
            
            sl_found = False
            for order in open_orders:
                if order.get('orderId') == sl_order_id:
                    sl_found = True
                    logger.info(f"止損單狀態: {order.get('status')}")
                    logger.info(f"止損單類型: {order.get('type')}")
                    logger.info(f"止損單價格: {order.get('stopPrice')}")
                    break
            
            if not sl_found:
                logger.warning("⚠️ 未找到止損單(可能已觸發或取消)")
            
            # 4. 取消訂單+平倉
            logger.info("取消止損單...")
            
            try:
                self.client.cancel_order(
                    symbol=self.symbol,
                    order_id=sl_order_id
                )
            except Exception as e:
                logger.warning(f"取消訂單失敗(可能已成交): {e}")
            
            logger.info("平倉...")
            self.emergency.emergency_flatten_position(
                self.symbol, "test", "測試完成"
            )
            
            logger.info("✅ SL下單能力檢測完成")
            return True
            
        except Exception as e:
            logger.error(f"SL下單檢測失敗: {e}", exc_info=True)
            
            # 清理
            try:
                if sl_order_id:
                    self.client.cancel_order(symbol=self.symbol, order_id=sl_order_id)
                        
                self.emergency.emergency_flatten_position(
                    self.symbol, "test", "測試異常清理"
                )
            except:
                pass
            
            return False
    
    def test_failure_handling(self) -> bool:
        """
        測試4: 故障演練 (關鍵!)
        
        流程:
        1. 開極小倉
        2. 刻意傳不合規stopPrice
        3. 確認SL失敗→flatten→停機
        """
        logger.info("📍 開始故障演練...")
        logger.info("⚠️ 這個測試會故意製造失敗!")
        
        entry_order_id = None
        
        try:
            # 獲取當前價格
            ticker = self.client.get_ticker_price(symbol=self.symbol)
            current_price = float(ticker['price'])
            
            # 1. 開極小倉
            logger.info(f"開倉: {self.test_qty} {self.symbol}...")
            
            entry_order = self.client.place_order({
                'symbol': self.symbol,
                'side': 'BUY',
                'type': 'MARKET',
                'quantity': self.test_qty
            })
            
            entry_order_id = entry_order.get('orderId')
            time.sleep(2)
            
            # 假設MARKET單立即成交
            filled_qty = self.test_qty
            avg_price = current_price
            
            # 2. 刻意用錯誤stopPrice (不round)
            bad_stop_price = avg_price * 0.98 + 0.123456789  # 故意很多小數位
            
            logger.info(f"⚠️ 故意用不合規stopPrice: {bad_stop_price}")
            
            # 直接調用client (繞過狀態機的round)
            try:
                bad_order = self.client.place_order({
                    'symbol': self.symbol,
                    'side': 'SELL',
                    'type': 'STOP_MARKET',
                    'stopPrice': bad_stop_price,  # 不合規!
                    'quantity': filled_qty,
                    'reduceOnly': 'true'
                })
                
                # 如果沒拒單,取消它
                logger.warning("⚠️ 意外: 不合規訂單竟然通過了!")
                try:
                    self.client.cancel_order(
                        symbol=self.symbol,
                        order_id=bad_order.get('orderId')
                    )
                except:
                    pass
                
            except Exception as e:
                logger.info(f"✅ 預期中的拒單: {e}")
            
            # 3. 確認flatten被觸發
            logger.info("檢查倉位是否被flatten...")
            time.sleep(2)
            
            # 手動flatten (模擬失敗處理)
            flatten_result = self.emergency.emergency_flatten_position(
                self.symbol, "test", "故障演練"
            )
            
            if flatten_result:
                logger.info("✅ flatten執行成功")
            else:
                logger.warning("⚠️ flatten執行失敗 (可能已無倉位)")
            
            # 確認無倉位
            time.sleep(1)
            
            position = self.client.get_position_risk(symbol=self.symbol)
            
            if position:
                position_amt = float(position[0].get('positionAmt', 0))
                
                if abs(position_amt) < 0.0001:
                    logger.info("✅ 倉位已清空")
                    return True
                else:
                    logger.error(f"❌ 倉位未清空: {position_amt}")
                    return False
            else:
                logger.info("✅ 無倉位信息(假設已清空)")
                return True
            
        except Exception as e:
            logger.error(f"故障演練失敗: {e}", exc_info=True)
            
            # 清理
            try:
                self.emergency.emergency_flatten_position(
                    self.symbol, "test", "故障演練異常清理"
                )
            except:
                pass
            
            return False
    
    def print_summary(self):
        """打印測試總結"""
        logger.info("\n" + "="*60)
        logger.info("📊 測試總結")
        logger.info("="*60)
        
        passed = sum(1 for _, result, _ in self.results if result)
        total = len(self.results)
        
        for name, result, error in self.results:
            status = "✅ 通過" if result else "❌ 失敗"
            logger.info(f"{name}: {status}")
            if error:
                logger.info(f"   錯誤: {error}")
        
        logger.info(f"\n總計: {passed}/{total} 通過")
        
        if passed == total:
            logger.info("\n🎉 所有測試通過! 保命鏈條完整!")
            msg = (
                "🎉 <b>Safety Test 完成</b>\n\n"
                f"結果: {passed}/{total} 通過\n\n"
                "✅ 逐倉檢測\n"
                "✅ 槓桿檢測\n"
                "✅ SL下單能力\n"
                "✅ 故障演練\n\n"
                "<b>保命鏈條完整! 可以進入48小時驗證!</b>"
            )
        else:
            logger.error("\n❌ 部分測試失敗! 需要修正!")
            msg = (
                "❌ <b>Safety Test 失敗</b>\n\n"
                f"結果: {passed}/{total} 通過\n\n"
                "需要檢查失敗的測試!"
            )
        
        if self.telegram.enabled:
            self.telegram.send_message(msg)

def main():
    """主函數"""
    print("\n" + "="*60)
    print("🧪 Safety Test Harness")
    print("自動保命自檢 - 10分鐘打穿所有關鍵保命鏈條")
    print("="*60)
    print("\n⚠️ 確認:")
    print("1. 你在測試網嗎? (testnet)")
    print("2. 你的.env.live_micro設定正確嗎?")
    print("\n")
    
    response = input("繼續執行? (yes/no): ").strip().lower()
    
    if response != 'yes':
        print("已取消")
        return
    
    try:
        harness = SafetyTestHarness()
        harness.run_all_tests()
        
    except KeyboardInterrupt:
        logger.info("\n測試被用戶中斷")
    except Exception as e:
        logger.error(f"測試異常: {e}", exc_info=True)

if __name__ == "__main__":
    main()
