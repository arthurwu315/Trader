"""
Telegram Notifier
發送交易通知到 Telegram
"""
import requests
import logging
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class TelegramNotifier:
    """Telegram 通知器"""
    
    def __init__(self, bot_token: Optional[str], chat_id: Optional[str], enabled: bool = True):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = enabled and bot_token and chat_id
        
        if self.enabled:
            logger.info("✅ Telegram 通知已啟用")
        else:
            logger.info("⚠️ Telegram 通知未啟用")
    
    def send_message(self, message: str, parse_mode: Optional[str] = None) -> bool:
        """
        發送訊息到 Telegram
        
        Args:
            message: 訊息內容
            parse_mode: 解析模式（None/Markdown/HTML）；預設 None 走純文字
        
        Returns:
            是否發送成功
        """
        if not self.enabled:
            return False
        
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            
            payload = {
                "chat_id": self.chat_id,
                "text": message,
                "disable_web_page_preview": True,
            }
            if parse_mode:
                payload["parse_mode"] = parse_mode
            
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                logger.debug("✅ Telegram 訊息已發送")
                return True
            else:
                logger.error(f"❌ Telegram 發送失敗: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Telegram 發送錯誤: {e}")
            return False
    
    def notify_startup(self, config) -> None:
        """機器人啟動通知"""
        message = f"""
🚀 交易機器人已啟動

📊 配置資訊:
• 標的: {config.symbol}
• 槓桿: {config.leverage}x
• 保證金: {config.margin_type}
• 環境: {'測試網' if config.testnet_mode else '實盤'}

⏰ 時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

🛡️ 風險設定:
• 單筆風險: {config.risk_per_trade_pct:.1%}
• 總風險上限: {config.max_total_loss_pct:.1%}
• 單日虧損限制: {config.max_daily_loss_pct:.1%}
"""
        self.send_message(message)
    
    def notify_entry(self, symbol: str, side: str, quantity: float, price: float, 
                     stop_loss: float, take_profit_1: float, take_profit_2: float) -> None:
        """進場通知"""
        message = f"""
📈 開倉通知

🎯 {symbol} - {side}

💰 進場資訊:
• 數量: {quantity}
• 價格: ${price:,.2f}
• 名義價值: ${quantity * price:,.2f}

🛡️ 風控設定:
• 止損: ${stop_loss:,.2f} ({((price - stop_loss) / price * 100):.2f}%)
• 止盈1: ${take_profit_1:,.2f} ({((take_profit_1 - price) / price * 100):.2f}%)
• 止盈2: ${take_profit_2:,.2f} ({((take_profit_2 - price) / price * 100):.2f}%)

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        self.send_message(message)
    
    def notify_exit(self, symbol: str, side: str, quantity: float, 
                    entry_price: float, exit_price: float, pnl: float, pnl_pct: float) -> None:
        """平倉通知"""
        emoji = "✅" if pnl > 0 else "❌"
        color = "盈利" if pnl > 0 else "虧損"
        
        message = f"""
{emoji} 平倉通知 - {color}

🎯 {symbol} - {side}

💰 交易資訊:
• 數量: {quantity}
• 進場: ${entry_price:,.2f}
• 出場: ${exit_price:,.2f}

📊 損益:
• 金額: ${pnl:,.2f}
• 百分比: {pnl_pct:,.2f}%

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        self.send_message(message)
    
    def notify_strategy_signal(self, signal_reason: str, details: dict = None) -> None:
        """策略訊號通知"""
        message = f"""
💡 策略訊號更新

{signal_reason}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        self.send_message(message)
    
    def notify_risk_warning(self, warning_type: str, details: str) -> None:
        """風險警告通知"""
        message = f"""
⚠️ 風險警告

類型: {warning_type}

詳情:
{details}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        self.send_message(message)
    
    def notify_critical_error(self, error_type: str, error_message: str) -> None:
        """嚴重錯誤通知"""
        message = f"""
🚨 嚴重錯誤

類型: {error_type}

錯誤訊息:
{error_message}

⚠️ 請立即檢查機器人狀態!

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        self.send_message(message)
    
    def notify_total_loss_limit(self, initial_equity: float, current_equity: float, 
                                 loss_pct: float, limit_pct: float) -> None:
        """總風險上限觸發通知"""
        message = f"""
🚨🚨🚨 總風險上限觸發 🚨🚨🚨

⛔ 機器人已停止交易!

📊 帳戶狀態:
• 初始權益: ${initial_equity:,.2f}
• 當前權益: ${current_equity:,.2f}
• 虧損金額: ${initial_equity - current_equity:,.2f}
• 虧損百分比: {loss_pct:.2%}
• 設定上限: {limit_pct:.2%}

⚠️ 請立即:
1. 檢查所有倉位
2. 分析虧損原因
3. 決定是否繼續運行
4. 如需重啟,請手動操作

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        self.send_message(message)
    
    def notify_daily_summary(self, trades_count: int, win_count: int, 
                            total_pnl: float, current_equity: float) -> None:
        """每日總結通知"""
        win_rate = (win_count / trades_count * 100) if trades_count > 0 else 0
        
        message = f"""
📊 每日交易總結

📈 交易統計:
• 交易次數: {trades_count}
• 獲利次數: {win_count}
• 勝率: {win_rate:.1f}%

💰 損益:
• 今日損益: ${total_pnl:,.2f}
• 當前權益: ${current_equity:,.2f}

⏰ {datetime.now().strftime('%Y-%m-%d')}
"""
        self.send_message(message)

# 測試函數
if __name__ == "__main__":
    import os
    from config_v2 import get_config
    
    logging.basicConfig(level=logging.INFO)
    
    config = get_config()
    
    notifier = TelegramNotifier(
        bot_token=config.telegram_bot_token,
        chat_id=config.telegram_chat_id,
        enabled=config.enable_telegram
    )
    
    if notifier.enabled:
        print("發送測試訊息...")
        notifier.send_message("🧪 測試訊息\n\nTelegram 通知功能正常! ✅")
    else:
        print("❌ Telegram 未啟用")
        print("請在 .env 中設定:")
        print("- TELEGRAM_BOT_TOKEN")
        print("- TELEGRAM_CHAT_ID")
