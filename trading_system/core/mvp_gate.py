"""
MVP Gate Engine
最小可行Gate - 只保命,不做Pool分配

按照老手規格實作:
- 帳戶級Gate (4個檢查)
- 工程級Gate (3個檢查)  
- 費用Gate (1個檢查)
- Reason codes (固定10個)
"""
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
from datetime import datetime

logger = logging.getLogger(__name__)

# ==================== Reason Codes (固定) ====================
ALLOW = "ALLOW"

# 帳戶級
REJECT_ACC_AVAILABLE_LOW = "REJECT_ACC_AVAILABLE_LOW"
REJECT_ACC_MARGIN_RATIO_HIGH = "REJECT_ACC_MARGIN_RATIO_HIGH"
REJECT_ACC_KILL_SWITCH = "REJECT_ACC_KILL_SWITCH"
REJECT_ACC_COOLDOWN = "REJECT_ACC_COOLDOWN"
REJECT_ACC_EXTREME_VOL_PAUSE = "REJECT_ACC_EXTREME_VOL_PAUSE"

# 工程級
REJECT_ORDER_SAFETY_ISOLATED_FAILED = "REJECT_ORDER_SAFETY_ISOLATED_FAILED"
REJECT_ORDER_SAFETY_LEVERAGE_FAILED = "REJECT_ORDER_SAFETY_LEVERAGE_FAILED"
REJECT_ORDER_MIN_NOTIONAL = "REJECT_ORDER_MIN_NOTIONAL"

# 費用級
REJECT_FEE_TOO_HIGH_FOR_TARGET = "REJECT_FEE_TOO_HIGH_FOR_TARGET"

# ==================== 數據類 ====================

@dataclass
class AccountSnapshot:
    """帳戶快照"""
    ts: int
    available_balance: float
    total_wallet_balance: float
    total_initial_margin: float
    total_maint_margin: float
    unrealized_pnl: float

@dataclass
class CandidateTrade:
    """候選交易"""
    symbol: str
    side: str  # BUY/SELL
    entry_type: str  # MARKET/STOP_MARKET
    entry_price: float
    stop_price: float
    tp_price: Optional[float]  # V3可空
    qty: float
    leverage: int
    notional: float
    required_margin_est: float
    risk_usdt: float
    expected_tp_pct: Optional[float]  # 短線用,V3可None
    strategy_tag: str  # "V3_MICRO"

@dataclass
class EnvState:
    """環境狀態"""
    kill_switch_active: bool = False
    cooldown_active: bool = False
    extreme_vol_pause_active: bool = False
    emergency_stop: bool = False

@dataclass
class MVPGateConfig:
    """MVP Gate配置"""
    # 帳戶級
    account_min_available_usdt: float = 300.0
    account_max_total_initial_margin_ratio: float = 0.65
    
    # 工程級
    min_notional: float = 5.0  # Binance BTCUSDT最小名義價值
    
    # 費用級 (你的實際費率)
    fee_maker: float = 0.00018  # 0.018%
    fee_taker: float = 0.00045  # 0.045%
    slippage_buffer: float = 0.00050  # 0.05%
    min_tp_pct: float = 0.0029  # 0.29% (0.09% + 0.05% + 0.15%)

# ==================== MVP Gate 主函數 ====================

def mvp_gate_check(
    snapshot: AccountSnapshot,
    candidate: CandidateTrade,
    env: EnvState,
    config: MVPGateConfig
) -> Tuple[bool, str, Dict]:
    """
    MVP Gate檢查 - 唯一入口
    
    Returns:
        (allow: bool, reason_code: str, debug: dict)
    """
    debug = {
        'timestamp': datetime.now().isoformat(),
        'symbol': candidate.symbol,
        'side': candidate.side,
        'strategy': candidate.strategy_tag,
        'available_balance': snapshot.available_balance,
        'wallet_balance': snapshot.total_wallet_balance,
        'total_initial_margin': snapshot.total_initial_margin,
        'notional': candidate.notional,
        'required_margin': candidate.required_margin_est,
        'risk_usdt': candidate.risk_usdt,
        'leverage': candidate.leverage
    }
    
    # 1. 帳戶級Gate
    allow, reason = check_account_gate(snapshot, env, config)
    if not allow:
        logger.warning(f"❌ Gate拒單: {reason}")
        logger.warning(f"   可用餘額: ${snapshot.available_balance:.2f}")
        logger.warning(f"   保證金率: {_calc_margin_ratio(snapshot):.2%}")
        return False, reason, debug
    
    # 2. 工程級Gate
    allow, reason = check_order_safety_gate(candidate, config)
    if not allow:
        logger.warning(f"❌ Gate拒單: {reason}")
        logger.warning(f"   名義價值: ${candidate.notional:.2f}")
        logger.warning(f"   槓桿: {candidate.leverage}x")
        return False, reason, debug
    
    # 3. 費用Gate (只有短線需要)
    if candidate.expected_tp_pct is not None:
        allow, reason = check_fee_gate(candidate, config)
        if not allow:
            logger.warning(f"❌ Gate拒單: {reason}")
            logger.warning(f"   預期TP: {candidate.expected_tp_pct:.2%}")
            logger.warning(f"   最小TP: {config.min_tp_pct:.2%}")
            return False, reason, debug
    
    # 全部通過
    logger.info(f"✅ Gate通過: {candidate.strategy_tag} {candidate.side}")
    logger.info(f"   名義價值: ${candidate.notional:.2f}")
    logger.info(f"   風險: ${candidate.risk_usdt:.2f}")
    logger.info(f"   槓桿: {candidate.leverage}x")
    
    return True, ALLOW, debug

# ==================== 帳戶級Gate ====================

def check_account_gate(
    snapshot: AccountSnapshot,
    env: EnvState,
    config: MVPGateConfig
) -> Tuple[bool, str]:
    """
    帳戶級Gate檢查
    
    檢查:
    1. 可用餘額
    2. 保證金使用率
    3. Kill switch
    4. Cooldown
    5. 極端波動暫停
    """
    # 1. 可用餘額不足
    if snapshot.available_balance < config.account_min_available_usdt:
        return False, REJECT_ACC_AVAILABLE_LOW
    
    # 2. 保證金使用率過高
    margin_ratio = _calc_margin_ratio(snapshot)
    if margin_ratio > config.account_max_total_initial_margin_ratio:
        return False, REJECT_ACC_MARGIN_RATIO_HIGH
    
    # 3. Kill switch
    if env.kill_switch_active:
        return False, REJECT_ACC_KILL_SWITCH
    
    # 4. Cooldown
    if env.cooldown_active:
        return False, REJECT_ACC_COOLDOWN
    
    # 5. 極端波動暫停
    if env.extreme_vol_pause_active:
        return False, REJECT_ACC_EXTREME_VOL_PAUSE
    
    return True, ALLOW

def _calc_margin_ratio(snapshot: AccountSnapshot) -> float:
    """計算保證金使用率"""
    if snapshot.total_wallet_balance < 1e-9:
        return 999.0  # 避免除零
    return snapshot.total_initial_margin / snapshot.total_wallet_balance

# ==================== 工程級Gate ====================

def check_order_safety_gate(
    candidate: CandidateTrade,
    config: MVPGateConfig
) -> Tuple[bool, str]:
    """
    工程級Gate檢查
    
    檢查:
    1. 最小名義價值
    
    注意: 逐倉和槓桿檢查在execute_trade前做!
    """
    # 1. 最小名義價值
    if candidate.notional < config.min_notional:
        return False, REJECT_ORDER_MIN_NOTIONAL
    
    return True, ALLOW

# ==================== 費用Gate ====================

def check_fee_gate(
    candidate: CandidateTrade,
    config: MVPGateConfig
) -> Tuple[bool, str]:
    """
    費用Gate檢查
    
    檢查: TP是否足夠覆蓋成本
    """
    if candidate.expected_tp_pct is None:
        return True, ALLOW  # 不檢查
    
    if candidate.expected_tp_pct < config.min_tp_pct:
        return False, REJECT_FEE_TOO_HIGH_FOR_TARGET
    
    return True, ALLOW

# ==================== 輔助函數 ====================

def get_account_snapshot(binance_client) -> AccountSnapshot:
    """
    獲取帳戶快照
    
    從Binance API獲取最新帳戶狀態
    """
    try:
        account_info = binance_client.futures_account()
        
        return AccountSnapshot(
            ts=int(datetime.now().timestamp()),
            available_balance=float(account_info.get('availableBalance', 0)),
            total_wallet_balance=float(account_info.get('totalWalletBalance', 0)),
            total_initial_margin=float(account_info.get('totalInitialMargin', 0)),
            total_maint_margin=float(account_info.get('totalMaintMargin', 0)),
            unrealized_pnl=float(account_info.get('totalUnrealizedProfit', 0))
        )
    except Exception as e:
        logger.error(f"獲取帳戶快照失敗: {e}")
        # 返回保守值 (會被Gate擋掉)
        return AccountSnapshot(
            ts=int(datetime.now().timestamp()),
            available_balance=0,
            total_wallet_balance=0,
            total_initial_margin=999999,
            total_maint_margin=999999,
            unrealized_pnl=0
        )

def log_gate_decision(db_conn, decision: str, reason_code: str, debug: Dict):
    """
    記錄Gate決策到數據庫 (gate_decisions)
    decision: "ALLOW" or "REJECT"
    """
    try:
        import json
        cursor = db_conn.cursor()
        cursor.execute("""
            INSERT INTO gate_decisions (
                timestamp, symbol, decision, reason_code,
                available_balance, wallet_balance, margin_ratio,
                notional, required_margin, risk_usdt,
                debug_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            debug.get('timestamp'),
            debug.get('symbol'),
            decision,  # ✅ 直接寫入字串，不做常數比對
            reason_code,
            debug.get('available_balance'),
            debug.get('wallet_balance'),
            (debug.get('total_initial_margin', 0) / max(debug.get('wallet_balance', 1), 1e-9)),
            debug.get('notional'),
            debug.get('required_margin'),
            debug.get('risk_usdt'),
            json.dumps(debug, ensure_ascii=False)
        ))
        db_conn.commit()
        logger.debug(f"Gate決策已記錄: {reason_code}")
    except Exception as e:
        logger.error(f"記錄Gate決策失敗: {e}")


# 測試
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 MVP Gate引擎測試\n")
    
    # 測試配置
    config = MVPGateConfig()
    
    # 測試快照
    snapshot = AccountSnapshot(
        ts=int(datetime.now().timestamp()),
        available_balance=500.0,
        total_wallet_balance=5000.0,
        total_initial_margin=2000.0,
        total_maint_margin=1000.0,
        unrealized_pnl=100.0
    )
    
    # 測試候選交易
    candidate = CandidateTrade(
        symbol="BTCUSDT",
        side="BUY",
        entry_type="MARKET",
        entry_price=50000.0,
        stop_price=49800.0,
        tp_price=50500.0,
        qty=0.01,
        leverage=5,
        notional=500.0,
        required_margin_est=100.0,
        risk_usdt=5.0,
        expected_tp_pct=0.01,
        strategy_tag="V3_MICRO"
    )
    
    # 測試環境
    env = EnvState()
    
    # 執行Gate檢查
    print("測試1: 正常情況")
    allow, reason, debug = mvp_gate_check(snapshot, candidate, env, config)
    print(f"結果: {'通過' if allow else '拒絕'} - {reason}\n")
    
    # 測試可用餘額不足
    print("測試2: 可用餘額不足")
    snapshot.available_balance = 200.0
    allow, reason, debug = mvp_gate_check(snapshot, candidate, env, config)
    print(f"結果: {'通過' if allow else '拒絕'} - {reason}\n")
    
    # 測試保證金率過高
    print("測試3: 保證金率過高")
    snapshot.available_balance = 500.0
    snapshot.total_initial_margin = 4000.0
    allow, reason, debug = mvp_gate_check(snapshot, candidate, env, config)
    print(f"結果: {'通過' if allow else '拒絕'} - {reason}\n")
    
    print("✅ 測試完成!")
