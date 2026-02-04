"""
Paper Trading Manager
紙上交易管理器 - V3專用
"""
import logging
from datetime import datetime
from typing import Optional, Dict, List
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class PaperPosition:
    """紙上交易倉位"""
    symbol: str
    side: str
    entry_time: datetime
    entry_price: float
    quantity: float
    leverage: int
    stop_loss: float
    take_profit_1: float
    take_profit_2: Optional[float]
    
    # 狀態
    status: str = "OPEN"  # OPEN, PARTIAL_CLOSED, CLOSED
    remaining_quantity: float = None
    
    # 出場資訊
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: float = 0
    pnl_pct: float = 0
    exit_reason: str = ""
    
    # 策略資訊
    entry_signal_type: str = ""
    structure_high: Optional[float] = None
    structure_low: Optional[float] = None
    
    def __post_init__(self):
        if self.remaining_quantity is None:
            self.remaining_quantity = self.quantity

class PaperTradingManager:
    """紙上交易管理器"""
    
    def __init__(self, initial_balance: float = 5000):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.equity = initial_balance
        
        self.positions: Dict[str, PaperPosition] = {}
        self.closed_trades: List[PaperPosition] = []
        
        self.total_trades = 0
        self.win_trades = 0
        self.total_pnl = 0
        
        logger.info(f"📝 紙上交易初始化: ${initial_balance:,.2f}")
    
    def open_position(self, signal, leverage: int, quantity: float) -> bool:
        """
        開倉 (紙上)
        
        Args:
            signal: 策略訊號
            leverage: 槓桿倍數
            quantity: 數量
        
        Returns:
            是否成功
        """
        if signal.symbol in self.positions:
            logger.warning(f"已有{signal.symbol}倉位,跳過")
            return False
        
        # 計算風險距離 (1R)
        risk_distance = abs(signal.entry_price - signal.stop_loss)
        
        position = PaperPosition(
            symbol=signal.symbol,
            side="BUY",  # 目前只做多
            entry_time=datetime.now(),
            entry_price=signal.entry_price,
            quantity=quantity,
            leverage=leverage,
            stop_loss=signal.stop_loss,
            take_profit_1=signal.entry_price + (risk_distance * 1.5),  # 1.5R
            take_profit_2=signal.entry_price + (risk_distance * 2.5),  # 2.5R
            entry_signal_type=signal.signal_type,
            structure_high=signal.structure_high,
            structure_low=signal.structure_low
        )
        
        self.positions[signal.symbol] = position
        
        logger.info(f"📝 紙上開倉:")
        logger.info(f"   標的: {signal.symbol}")
        logger.info(f"   價格: ${signal.entry_price:,.2f}")
        logger.info(f"   數量: {quantity:.4f}")
        logger.info(f"   槓桿: {leverage}x")
        logger.info(f"   止損: ${position.stop_loss:,.2f} (1R)")
        logger.info(f"   止盈1: ${position.take_profit_1:,.2f} (1.5R)")
        logger.info(f"   止盈2: ${position.take_profit_2:,.2f} (2.5R)")
        
        return True
    
    def update_positions(self, current_prices: Dict[str, float]):
        """
        更新所有倉位 (檢查止損/止盈)
        
        Args:
            current_prices: {symbol: price}
        """
        for symbol, position in list(self.positions.items()):
            if position.status == "CLOSED":
                continue
            
            if symbol not in current_prices:
                continue
            
            current_price = current_prices[symbol]
            
            # 檢查止損
            if current_price <= position.stop_loss:
                self.close_position(
                    symbol, 
                    exit_price=position.stop_loss,
                    reason="止損觸發",
                    partial=False
                )
                continue
            
            # 檢查TP1
            if position.status == "OPEN" and current_price >= position.take_profit_1:
                self.close_position(
                    symbol,
                    exit_price=position.take_profit_1,
                    reason="TP1觸發",
                    partial=True,
                    partial_pct=0.5
                )
                continue
            
            # 檢查TP2 (剩餘倉位)
            if position.status == "PARTIAL_CLOSED" and current_price >= position.take_profit_2:
                self.close_position(
                    symbol,
                    exit_price=position.take_profit_2,
                    reason="TP2觸發",
                    partial=False
                )
    
    def close_position(self, symbol: str, exit_price: float, reason: str,
                      partial: bool = False, partial_pct: float = 0.5):
        """
        平倉 (紙上)
        
        Args:
            symbol: 交易對
            exit_price: 出場價格
            reason: 平倉原因
            partial: 是否部分平倉
            partial_pct: 部分平倉百分比
        """
        if symbol not in self.positions:
            logger.warning(f"{symbol}無倉位")
            return
        
        position = self.positions[symbol]
        
        # 計算平倉數量
        if partial:
            close_qty = position.remaining_quantity * partial_pct
            position.remaining_quantity -= close_qty
            position.status = "PARTIAL_CLOSED"
        else:
            close_qty = position.remaining_quantity
            position.remaining_quantity = 0
            position.status = "CLOSED"
        
        # 計算盈虧
        pnl = (exit_price - position.entry_price) * close_qty
        pnl_pct = (pnl / (position.entry_price * close_qty)) * 100
        
        # 更新總盈虧
        self.total_pnl += pnl
        self.equity += pnl
        
        # 記錄
        position.exit_time = datetime.now()
        position.exit_price = exit_price
        position.pnl += pnl
        position.pnl_pct = (position.pnl / (position.entry_price * position.quantity)) * 100
        position.exit_reason = reason
        
        # 統計
        if position.status == "CLOSED":
            self.total_trades += 1
            if position.pnl > 0:
                self.win_trades += 1
            self.closed_trades.append(position)
            del self.positions[symbol]
        
        emoji = "✅" if pnl > 0 else "❌"
        logger.info(f"📝 紙上{'部分' if partial else '全部'}平倉 {emoji}")
        logger.info(f"   標的: {symbol}")
        logger.info(f"   進場: ${position.entry_price:,.2f}")
        logger.info(f"   出場: ${exit_price:,.2f}")
        logger.info(f"   數量: {close_qty:.4f}")
        logger.info(f"   損益: ${pnl:.2f} ({pnl_pct:+.2f}%)")
        logger.info(f"   原因: {reason}")
        logger.info(f"   當前權益: ${self.equity:,.2f}")
    
    def get_statistics(self) -> Dict:
        """獲取統計數據"""
        win_rate = (self.win_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        
        return {
            'initial_balance': self.initial_balance,
            'current_equity': self.equity,
            'total_pnl': self.total_pnl,
            'return_pct': (self.equity - self.initial_balance) / self.initial_balance * 100,
            'total_trades': self.total_trades,
            'win_trades': self.win_trades,
            'loss_trades': self.total_trades - self.win_trades,
            'win_rate': win_rate,
            'open_positions': len(self.positions)
        }
    
    def get_open_positions(self) -> List[Dict]:
        """獲取開倉倉位"""
        return [asdict(pos) for pos in self.positions.values()]
    
    def get_closed_trades(self, limit: Optional[int] = None) -> List[Dict]:
        """獲取平倉記錄"""
        trades = [asdict(pos) for pos in self.closed_trades]
        if limit:
            return trades[-limit:]
        return trades

# 測試
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    from structure_detector import StructureSignal
    
    # 創建測試訊號
    signal = StructureSignal(
        symbol="BTCUSDT",
        signal_type="breakout",
        entry_allowed=True,
        entry_price=91500,
        stop_loss=91200,
        reason="測試",
        atr=650
    )
    
    # 初始化紙上交易
    paper = PaperTradingManager(initial_balance=5000)
    
    # 開倉
    paper.open_position(signal, leverage=10, quantity=0.05)
    
    # 模擬價格變動
    print("\n模擬價格變動:")
    print("-" * 50)
    
    # 上漲到TP1
    paper.update_positions({"BTCUSDT": 93450})
    
    # 繼續上漲到TP2
    paper.update_positions({"BTCUSDT": 94750})
    
    # 統計
    stats = paper.get_statistics()
    print("\n📊 紙上交易統計:")
    print(f"初始資金: ${stats['initial_balance']:,.2f}")
    print(f"當前權益: ${stats['current_equity']:,.2f}")
    print(f"總盈虧: ${stats['total_pnl']:,.2f} ({stats['return_pct']:+.2f}%)")
    print(f"總交易: {stats['total_trades']}")
    print(f"勝率: {stats['win_rate']:.1f}%")
