"""
Emergency Handler
緊急平倉和停機處理
"""
import logging
import sys
from datetime import datetime

logger = logging.getLogger(__name__)

class EmergencyHandler:
    """緊急處理器"""
    
    def __init__(self, binance_client, telegram_notifier=None):
        self.client = binance_client
        self.telegram = telegram_notifier
        self.emergency_triggered = False
    
    def emergency_flatten_position(self, symbol, strategy_id, reason):
        """
        緊急平倉
        
        Args:
            symbol: 交易對
            strategy_id: 策略ID
            reason: 觸發原因
        """
        logger.critical(f"🚨🚨🚨 緊急平倉觸發!")
        logger.critical(f"策略: {strategy_id}")
        logger.critical(f"標的: {symbol}")
        logger.critical(f"原因: {reason}")
        
        try:
            # 獲取當前倉位
            positions = self.client.get_position_risk(symbol=symbol)
            
            for pos in positions:
                pos_amt = float(pos.get('positionAmt', 0))
                
                if abs(pos_amt) > 0:
                    logger.warning(f"發現倉位: {pos_amt} {symbol}")
                    
                    # 市價平倉
                    side = 'SELL' if pos_amt > 0 else 'BUY'
                    qty = abs(pos_amt)
                    
                    order = self.client.place_order({
                        'symbol': symbol,
                        'side': side,
                        'type': 'MARKET',
                        'quantity': qty,
                        'reduceOnly': 'true'
                    })
                    
                    logger.warning(f"✅ 平倉訂單: {order.get('orderId')}")
                    
                    if self.telegram:
                        self.telegram.send_message(
                            f"🚨 <b>緊急平倉執行</b>\n\n"
                            f"策略: {strategy_id}\n"
                            f"標的: {symbol}\n"
                            f"數量: {qty}\n"
                            f"原因: {reason}\n"
                            f"時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                        )
            
            # 取消所有掛單
            self.cancel_all_orders(symbol, strategy_id)
            
            logger.warning("✅ 緊急平倉完成")
            return True
            
        except Exception as e:
            logger.error(f"❌ 緊急平倉失敗: {e}", exc_info=True)
            
            if self.telegram:
                self.telegram.send_message(
                    f"🚨🚨🚨 <b>緊急平倉失敗!</b>\n\n"
                    f"策略: {strategy_id}\n"
                    f"錯誤: {str(e)}\n\n"
                    f"<b>請立即手動處理!</b>"
                )
            
            return False
    
    def cancel_all_orders(self, symbol, strategy_id):
        """取消所有掛單"""
        try:
            result = self.client.cancel_all_orders(symbol=symbol)
            logger.warning(f"✅ 取消所有掛單: {result}")
            return True
        except Exception as e:
            logger.error(f"❌ 取消掛單失敗: {e}")
            return False
    
    def emergency_stop_strategy(self, strategy_id, reason):
        """
        緊急停止策略
        
        Args:
            strategy_id: 策略ID
            reason: 停止原因
        """
        logger.critical(f"⛔⛔⛔ 緊急停止策略!")
        logger.critical(f"策略: {strategy_id}")
        logger.critical(f"原因: {reason}")
        
        self.emergency_triggered = True
        
        if self.telegram:
            self.telegram.send_message(
                f"⛔ <b>策略緊急停止</b>\n\n"
                f"策略: {strategy_id}\n"
                f"原因: {reason}\n"
                f"時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                f"<b>Bot已停止運行</b>\n"
                f"需要手動重啟"
            )
        
        # 停止程式
        logger.critical("⛔ 程式即將停止...")
        sys.exit(1)
    
    def check_emergency_triggered(self):
        """檢查是否觸發緊急狀態"""
        return self.emergency_triggered

# 測試
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("緊急處理器模組測試")
    print("✅ 模組載入成功")
