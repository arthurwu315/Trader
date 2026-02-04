"""
Dynamic Leverage Calculator
動態槓桿計算系統
根據止損距離自動調整槓桿
"""
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

class DynamicLeverageCalculator:
    """動態槓桿計算器"""
    
    def __init__(self, config):
        self.config = config
    
    def calculate_leverage_and_position(self, 
                                       entry_price: float,
                                       stop_loss: float,
                                       account_equity: float,
                                       symbol: str = "BTCUSDT") -> dict:
        """
        計算動態槓桿和倉位
        
        核心邏輯:
        1. 計算止損距離百分比
        2. 根據止損距離決定槓桿倍數
        3. 計算固定風險下的倉位大小
        
        Args:
            entry_price: 進場價格
            stop_loss: 止損價格
            account_equity: 帳戶權益
            symbol: 交易對
        
        Returns:
            {
                'leverage': int,
                'quantity': float,
                'notional_value': float,
                'risk_amount': float,
                'stop_distance_pct': float,
                'tier': str,
                'allowed': bool,
                'reason': str
            }
        """
        
        # 計算止損距離
        stop_distance = abs(entry_price - stop_loss)
        stop_distance_pct = stop_distance / entry_price
        
        logger.info(f"止損距離: ${stop_distance:.2f} ({stop_distance_pct:.4%})")
        
        # 根據止損距離決定槓桿
        leverage, tier, tier_reason = self._determine_leverage(stop_distance_pct)
        
        if leverage == 0:
            return {
                'leverage': 0,
                'quantity': 0,
                'notional_value': 0,
                'risk_amount': 0,
                'stop_distance_pct': stop_distance_pct,
                'tier': tier,
                'allowed': False,
                'reason': tier_reason
            }
        
        # 計算風險金額
        risk_amount = account_equity * self.config.risk_per_trade_pct
        
        # 計算數量 (正確公式)
        # quantity = 風險金額 / 止損距離
        # 但要注意: 止損距離是USD,所以直接除即可
        quantity = risk_amount / stop_distance
        
        logger.info(f"倉位計算:")
        logger.info(f"  風險金額: ${risk_amount:.2f}")
        logger.info(f"  止損距離: ${stop_distance:.2f}")
        logger.info(f"  數量: {quantity:.6f} BTC")
        
        # 計算名義價值
        notional_value = quantity * entry_price
        
        logger.info(f"  名義價值: ${notional_value:,.2f}")
        
        # 計算實際使用的保證金
        margin_required = notional_value / leverage
        
        # 檢查保證金是否足夠
        if margin_required > account_equity:
            return {
                'leverage': leverage,
                'quantity': 0,
                'notional_value': 0,
                'risk_amount': risk_amount,
                'stop_distance_pct': stop_distance_pct,
                'tier': tier,
                'allowed': False,
                'reason': f"保證金不足(需要${margin_required:.2f},有${account_equity:.2f})"
            }
        
        # 檢查是否超過最大槓桿
        actual_leverage_used = notional_value / account_equity
        if actual_leverage_used > self.config.max_leverage:
            # 調整數量
            max_notional = account_equity * self.config.max_leverage
            quantity = max_notional / entry_price
            notional_value = quantity * entry_price
            
            logger.warning(f"觸及最大槓桿限制{self.config.max_leverage}x,調整倉位")
        
        logger.info(f"槓桿層級: {tier}")
        logger.info(f"使用槓桿: {leverage}x")
        logger.info(f"數量: {quantity:.4f}")
        logger.info(f"名義價值: ${notional_value:,.2f}")
        logger.info(f"風險金額: ${risk_amount:.2f}")
        
        return {
            'leverage': leverage,
            'quantity': quantity,
            'notional_value': notional_value,
            'risk_amount': risk_amount,
            'stop_distance_pct': stop_distance_pct,
            'margin_required': margin_required,
            'tier': tier,
            'allowed': True,
            'reason': f"{tier}: {leverage}x槓桿"
        }
    
    def _determine_leverage(self, stop_distance_pct: float) -> Tuple[int, str, str]:
        """
        根據止損距離決定槓桿
        
        Returns:
            (leverage, tier_name, reason)
        """
        
        # Tier 1: ≤0.25% → 12-15x (結構極好)
        if stop_distance_pct <= self.config.leverage_tier_1_max_sl:
            leverage = self.config.leverage_tier_1_range[1]  # 使用最高
            return leverage, "Tier1(結構極佳)", f"止損{stop_distance_pct:.2%}≤0.25%"
        
        # Tier 2: 0.25-0.4% → 8-12x (結構良好)
        elif stop_distance_pct <= self.config.leverage_tier_2_max_sl:
            leverage = self.config.leverage_tier_2_range[1]  # 使用最高
            return leverage, "Tier2(結構良好)", f"止損{stop_distance_pct:.2%}≤0.4%"
        
        # Tier 3: 0.4-0.6% → 4-6x (結構尚可)
        elif stop_distance_pct <= self.config.leverage_tier_3_max_sl:
            leverage = self.config.leverage_tier_3_range[1]  # 使用最高
            return leverage, "Tier3(結構尚可)", f"止損{stop_distance_pct:.2%}≤0.6%"
        
        # Tier 4: >0.6% → 不交易 (結構不佳)
        else:
            return 0, "Tier4(拒絕)", f"止損{stop_distance_pct:.2%}>0.6%,結構不乾淨"
    
    def adjust_quantity_for_exchange_rules(self, 
                                          quantity: float,
                                          symbol_info: dict) -> float:
        """
        根據交易所規則調整數量
        
        Args:
            quantity: 原始數量
            symbol_info: 交易對規則
        
        Returns:
            調整後的數量
        """
        min_qty = symbol_info.get('min_qty', 0.001)
        max_qty = symbol_info.get('max_qty', 1000)
        step_size = symbol_info.get('step_size', 0.001)
        
        # 調整到步長
        adjusted_qty = round(quantity / step_size) * step_size
        
        # 檢查最小值
        if adjusted_qty < min_qty:
            logger.warning(f"數量{adjusted_qty}小於最小值{min_qty}")
            return 0
        
        # 檢查最大值
        if adjusted_qty > max_qty:
            logger.warning(f"數量{adjusted_qty}超過最大值{max_qty},調整為{max_qty}")
            adjusted_qty = max_qty
        
        return adjusted_qty

# 測試
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    from config_v3 import get_config
    config = get_config()
    
    calculator = DynamicLeverageCalculator(config)
    
    # 測試不同止損距離
    test_cases = [
        (91500, 91300, "極小止損"),  # 0.22%
        (91500, 91200, "小止損"),    # 0.33%
        (91500, 91000, "中等止損"),  # 0.55%
        (91500, 90500, "大止損"),    # 1.09%
    ]
    
    account_equity = 5000
    
    print("\n動態槓桿計算測試")
    print("=" * 80)
    
    for entry, stop, desc in test_cases:
        print(f"\n{desc}:")
        print(f"進場: ${entry:,} | 止損: ${stop:,}")
        
        result = calculator.calculate_leverage_and_position(
            entry_price=entry,
            stop_loss=stop,
            account_equity=account_equity
        )
        
        print(f"結果: {result['tier']}")
        print(f"槓桿: {result['leverage']}x")
        print(f"允許: {result['allowed']}")
        print(f"原因: {result['reason']}")
        
        if result['allowed']:
            print(f"數量: {result['quantity']:.4f}")
            print(f"名義價值: ${result['notional_value']:,.2f}")
            print(f"風險金額: ${result['risk_amount']:.2f}")
