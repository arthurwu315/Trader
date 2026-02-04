"""
Order Utils
下單/保護單的精度與交易所參數工具
"""
import logging
from decimal import Decimal, ROUND_DOWN
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


# ==================== Exchange Info 兼容 ====================

def _fetch_exchange_info(client) -> Dict:
    """
    兼容不同 Binance client 的 exchangeInfo 取得方法
    - 有些 client 叫 exchange_info()
    - 你的 BinanceFuturesClient 可能叫 get_exchange_info()
    """
    if hasattr(client, "exchange_info") and callable(getattr(client, "exchange_info")):
        return client.exchange_info()
    if hasattr(client, "get_exchange_info") and callable(getattr(client, "get_exchange_info")):
        return client.get_exchange_info()
    raise AttributeError("Client missing exchange info method: exchange_info()/get_exchange_info()")


# ==================== 價格精度處理 ====================

def round_to_tick_size(price: float, tick_size: float = 0.1) -> float:
    """
    將價格 round 到 tickSize
    例如: 50123.456 -> 50123.4 (ROUND_DOWN)
    """
    if tick_size == 0:
        return price

    decimal_price = Decimal(str(price))
    decimal_tick = Decimal(str(tick_size))

    rounded = (decimal_price / decimal_tick).quantize(Decimal('1'), rounding=ROUND_DOWN) * decimal_tick
    return float(rounded)


def get_symbol_tick_size(client, symbol: str) -> float:
    """
    獲取 symbol 的 tickSize
    從 exchangeInfo 獲取, 失敗則返回默認值 0.1
    """
    try:
        exchange_info = _fetch_exchange_info(client)

        for s in exchange_info.get('symbols', []):
            if s.get('symbol') == symbol:
                for f in s.get('filters', []):
                    if f.get('filterType') == 'PRICE_FILTER':
                        tick_size = float(f.get('tickSize', 0.1))
                        logger.debug(f"{symbol} tickSize: {tick_size}")
                        return tick_size

        logger.warning(f"無法獲取{symbol}的tickSize,使用默認值0.1")
        return 0.1

    except Exception as e:
        logger.error(f"獲取tickSize失敗: {e}")
        return 0.1


# ==================== 數量精度處理 ====================

def round_qty_to_step(qty: float, step_size: float = 0.001) -> float:
    """
    將數量 round 到 stepSize
    例如: 0.0123456 -> 0.012 (ROUND_DOWN)
    """
    if step_size == 0:
        return qty

    decimal_qty = Decimal(str(qty))
    decimal_step = Decimal(str(step_size))

    rounded = (decimal_qty / decimal_step).quantize(Decimal('1'), rounding=ROUND_DOWN) * decimal_step
    return float(rounded)


def get_symbol_step_size(client, symbol: str) -> Tuple[float, float]:
    """
    獲取 symbol 的 stepSize 和 minQty
    返回: (step_size, min_qty)
    """
    try:
        exchange_info = _fetch_exchange_info(client)

        for s in exchange_info.get('symbols', []):
            if s.get('symbol') == symbol:
                step_size = 0.001
                min_qty = 0.001

                for f in s.get('filters', []):
                    if f.get('filterType') == 'LOT_SIZE':
                        step_size = float(f.get('stepSize', 0.001))
                        min_qty = float(f.get('minQty', 0.001))
                        logger.debug(f"{symbol} stepSize: {step_size}, minQty: {min_qty}")

                return (step_size, min_qty)

        logger.warning(f"無法獲取{symbol}的stepSize,使用默認值0.001")
        return (0.001, 0.001)

    except Exception as e:
        logger.error(f"獲取stepSize失敗: {e}")
        return (0.001, 0.001)
