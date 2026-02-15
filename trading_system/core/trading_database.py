"""
Trading Database Module
交易數據記錄和查詢
"""
import sqlite3
import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

class TradingDatabase:
    """交易數據庫"""
    
    def __init__(self, db_path: str = "/home/trader/bot_v2/trading.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self) -> None:
        """初始化數據庫表"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 交易記錄表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_time TEXT NOT NULL,
                exit_time TEXT,
                entry_price REAL NOT NULL,
                exit_price REAL,
                quantity REAL NOT NULL,
                stop_loss REAL,
                take_profit_1 REAL,
                take_profit_2 REAL,
                pnl REAL,
                pnl_pct REAL,
                status TEXT DEFAULT 'OPEN',
                exit_reason TEXT,
                strategy_signal TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 權益記錄表 (每小時記錄一次)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS equity_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                equity REAL NOT NULL,
                balance REAL NOT NULL,
                unrealized_pnl REAL DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 策略訊號記錄表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                entry_allowed BOOLEAN,
                entry_price REAL,
                stop_loss REAL,
                take_profit_1 REAL,
                take_profit_2 REAL,
                reason TEXT,
                details TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 風險事件記錄表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS risk_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                description TEXT,
                equity_before REAL,
                equity_after REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
        logger.info(f"✅ 數據庫初始化完成: {self.db_path}")
    
    def record_trade_entry(self, symbol: str, side: str, entry_price: float,
                          quantity: float, stop_loss: float, take_profit_1: float,
                          take_profit_2: float, strategy_signal: str = "") -> int:
        """
        記錄進場
        
        Returns:
            trade_id: 交易ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        now = datetime.now().isoformat()
        
        cursor.execute("""
            INSERT INTO trades (
                symbol, side, entry_time, entry_price, quantity,
                stop_loss, take_profit_1, take_profit_2, 
                status, strategy_signal
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?)
        """, (symbol, side, now, entry_price, quantity,
              stop_loss, take_profit_1, take_profit_2, strategy_signal))
        
        trade_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        logger.info(f"✅ 記錄進場: Trade #{trade_id}")
        return trade_id
    
    def record_trade_exit(self, trade_id: int, exit_price: float,
                         exit_reason: str = "") -> None:
        """記錄平倉"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 獲取進場信息
        cursor.execute("""
            SELECT entry_price, quantity, side 
            FROM trades WHERE id = ?
        """, (trade_id,))
        
        result = cursor.fetchone()
        if not result:
            logger.error(f"❌ 找不到交易 #{trade_id}")
            conn.close()
            return
        
        entry_price, quantity, side = result
        
        # 計算損益
        if side == "BUY":
            pnl = (exit_price - entry_price) * quantity
        else:
            pnl = (entry_price - exit_price) * quantity
        
        pnl_pct = (pnl / (entry_price * quantity)) * 100
        
        now = datetime.now().isoformat()
        
        # 更新交易記錄
        cursor.execute("""
            UPDATE trades 
            SET exit_time = ?,
                exit_price = ?,
                pnl = ?,
                pnl_pct = ?,
                status = 'CLOSED',
                exit_reason = ?
            WHERE id = ?
        """, (now, exit_price, pnl, pnl_pct, exit_reason, trade_id))
        
        conn.commit()
        conn.close()
        
        logger.info(f"✅ 記錄平倉: Trade #{trade_id}, PnL: ${pnl:.2f} ({pnl_pct:.2f}%)")
    
    def record_equity(self, equity: float, balance: float, 
                     unrealized_pnl: float = 0) -> None:
        """記錄權益"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        now = datetime.now().isoformat()
        
        cursor.execute("""
            INSERT INTO equity_history (timestamp, equity, balance, unrealized_pnl)
            VALUES (?, ?, ?, ?)
        """, (now, equity, balance, unrealized_pnl))
        
        conn.commit()
        conn.close()
    
    def record_signal(self, symbol: str, signal_type: str, entry_allowed: bool,
                     entry_price: Optional[float], stop_loss: Optional[float],
                     take_profit_1: Optional[float], take_profit_2: Optional[float],
                     reason: str, details: str = "") -> None:
        """記錄策略訊號"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        now = datetime.now().isoformat()
        
        cursor.execute("""
            INSERT INTO signals (
                timestamp, symbol, signal_type, entry_allowed,
                entry_price, stop_loss, take_profit_1, take_profit_2,
                reason, details
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (now, symbol, signal_type, entry_allowed,
              entry_price, stop_loss, take_profit_1, take_profit_2,
              reason, details))
        
        conn.commit()
        conn.close()
    
    def record_risk_event(self, event_type: str, severity: str,
                         description: str, equity_before: float = 0,
                         equity_after: float = 0) -> None:
        """記錄風險事件"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        now = datetime.now().isoformat()
        
        cursor.execute("""
            INSERT INTO risk_events (
                timestamp, event_type, severity, description,
                equity_before, equity_after
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (now, event_type, severity, description,
              equity_before, equity_after))
        
        conn.commit()
        conn.close()
        
        logger.warning(f"⚠️ 風險事件: {event_type} - {description}")
    
    def get_all_trades(self, status: Optional[str] = None) -> List[Dict]:
        """獲取所有交易"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        if status:
            cursor.execute("SELECT * FROM trades WHERE status = ? ORDER BY entry_time DESC", (status,))
        else:
            cursor.execute("SELECT * FROM trades ORDER BY entry_time DESC")
        
        trades = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return trades
    
    def get_closed_trades(self, limit: Optional[int] = None) -> List[Dict]:
        """獲取已平倉交易"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        if limit:
            cursor.execute("""
                SELECT * FROM trades 
                WHERE status = 'CLOSED' 
                ORDER BY exit_time DESC 
                LIMIT ?
            """, (limit,))
        else:
            cursor.execute("""
                SELECT * FROM trades 
                WHERE status = 'CLOSED' 
                ORDER BY exit_time DESC
            """)
        
        trades = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return trades
    
    def get_statistics(self) -> Dict:
        """獲取交易統計"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 總交易次數
        cursor.execute("SELECT COUNT(*) FROM trades WHERE status = 'CLOSED'")
        total_trades = cursor.fetchone()[0]
        
        # 獲利次數
        cursor.execute("SELECT COUNT(*) FROM trades WHERE status = 'CLOSED' AND pnl > 0")
        win_trades = cursor.fetchone()[0]
        
        # 虧損次數
        cursor.execute("SELECT COUNT(*) FROM trades WHERE status = 'CLOSED' AND pnl < 0")
        loss_trades = cursor.fetchone()[0]
        
        # 總損益
        cursor.execute("SELECT SUM(pnl) FROM trades WHERE status = 'CLOSED'")
        total_pnl = cursor.fetchone()[0] or 0
        
        # 平均盈利
        cursor.execute("SELECT AVG(pnl) FROM trades WHERE status = 'CLOSED' AND pnl > 0")
        avg_win = cursor.fetchone()[0] or 0
        
        # 平均虧損
        cursor.execute("SELECT AVG(pnl) FROM trades WHERE status = 'CLOSED' AND pnl < 0")
        avg_loss = cursor.fetchone()[0] or 0
        
        # 最大盈利
        cursor.execute("SELECT MAX(pnl) FROM trades WHERE status = 'CLOSED'")
        max_win = cursor.fetchone()[0] or 0
        
        # 最大虧損
        cursor.execute("SELECT MIN(pnl) FROM trades WHERE status = 'CLOSED'")
        max_loss = cursor.fetchone()[0] or 0
        
        # 勝率
        win_rate = (win_trades / total_trades * 100) if total_trades > 0 else 0
        
        # 盈虧比
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        # 權益曲線
        cursor.execute("""
            SELECT MIN(equity) as min_equity, MAX(equity) as max_equity
            FROM equity_history
        """)
        equity_result = cursor.fetchone()
        min_equity = equity_result[0] if equity_result[0] else 0
        max_equity = equity_result[1] if equity_result[1] else 0
        
        conn.close()
        
        return {
            'total_trades': total_trades,
            'win_trades': win_trades,
            'loss_trades': loss_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_win': max_win,
            'max_loss': max_loss,
            'profit_factor': profit_factor,
            'min_equity': min_equity,
            'max_equity': max_equity,
        }
    
    def get_equity_history(self, days: int = 30) -> List[Dict]:
        """獲取權益歷史"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM equity_history 
            WHERE timestamp >= datetime('now', '-' || ? || ' days')
            ORDER BY timestamp ASC
        """, (days,))
        
        history = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return history

# 測試
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    db = TradingDatabase()
    
    # 測試記錄進場
    trade_id = db.record_trade_entry(
        symbol="BTCUSDT",
        side="BUY",
        entry_price=90000,
        quantity=0.01,
        stop_loss=89000,
        take_profit_1=91000,
        take_profit_2=92000,
        strategy_signal="趨勢突破"
    )
    
    print(f"Trade ID: {trade_id}")
    
    # 測試記錄平倉
    # db.record_trade_exit(trade_id, 91000, "止盈1")
    
    # 測試記錄權益
    db.record_equity(equity=5000, balance=5000, unrealized_pnl=10)
    
    # 測試獲取統計
    stats = db.get_statistics()
    print("\n統計數據:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
