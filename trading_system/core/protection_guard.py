"""
Protection Guard
持續性止損/止盈守護
"""
import logging
import time
from typing import Optional, Dict, Tuple, Callable

from core.order_utils import (
    round_to_tick_size,
    round_qty_to_step,
    get_symbol_tick_size,
    get_symbol_step_size,
)

logger = logging.getLogger(__name__)


class ProtectionGuard:
    """
    保護單守衛層
    - 去重
    - 重試
    - 重啟補單
    """

    def __init__(self, client, emergency_handler, working_type: str = "MARK_PRICE"):
        self.client = client
        self.emergency = emergency_handler
        self.working_type = working_type

        self._tick_size_cache: Dict[str, float] = {}
        self._step_size_cache: Dict[str, float] = {}
        self._min_qty_cache: Dict[str, float] = {}

    # ==================== 精度與快取 ====================

    def _get_tick_size(self, symbol: str) -> float:
        if symbol not in self._tick_size_cache:
            self._tick_size_cache[symbol] = get_symbol_tick_size(self.client, symbol)
        return self._tick_size_cache[symbol]

    def _get_step_size(self, symbol: str) -> Tuple[float, float]:
        if symbol not in self._step_size_cache:
            step_size, min_qty = get_symbol_step_size(self.client, symbol)
            self._step_size_cache[symbol] = step_size
            self._min_qty_cache[symbol] = min_qty
        return (self._step_size_cache[symbol], self._min_qty_cache[symbol])

    # ==================== 掃描/驗證 ====================

    def scan_existing_protective_orders(self, symbol: str) -> Dict[str, Optional[Dict]]:
        """
        掃描目前是否已有 SL / TP 保護單（reduceOnly）
        回傳: {"sl": order or None, "tp": order or None}
        """
        result = {"sl": None, "tp": None}

        try:
            open_orders = self.client.get_open_orders(symbol=symbol)
            for order in open_orders:
                if not order.get("reduceOnly"):
                    continue

                order_type = order.get("type")
                if order_type == "STOP_MARKET":
                    result["sl"] = order
                elif order_type == "TAKE_PROFIT_MARKET":
                    result["tp"] = order

        except Exception as e:
            logger.error(f"掃描保護單失敗: {e}")

        return result

    def verify_protective_orders(
        self,
        symbol: str,
        expect_tp: bool = True,
        max_wait: int = 5,
        algo_ids: Optional[Dict[str, Optional[int]]] = None,
    ) -> bool:
        for _ in range(max_wait):
            existing = self.scan_existing_protective_orders(symbol)

            sl_ok = existing["sl"] is not None
            tp_ok = (existing["tp"] is not None) if expect_tp else True

            if algo_ids:
                sl_algo_id = algo_ids.get("sl")
                tp_algo_id = algo_ids.get("tp") if expect_tp else None
                if sl_algo_id and not sl_ok:
                    sl_ok = self._verify_algo_order(sl_algo_id)
                if tp_algo_id and not tp_ok:
                    tp_ok = self._verify_algo_order(tp_algo_id)

            if sl_ok and tp_ok:
                return True

            time.sleep(1)

        logger.error("驗證保護單超時仍未確認成功")
        return False

    # ==================== 下單 ====================

    def _should_try_close_position_fallback(self, error_msg: str) -> bool:
        keywords = (
            "Order type not supported",
            "orderType",
            "Invalid orderType",
            "Unknown order",
        )
        return any(k in error_msg for k in keywords)

    def _place_reduce_only_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        stop_price: float,
        quantity: float,
    ) -> Dict:
        params = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "stopPrice": stop_price,
            "quantity": quantity,
            "reduceOnly": "true",
            "workingType": self.working_type,
        }
        return self.client.place_order(params)

    def _place_close_position_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        stop_price: float,
    ) -> Dict:
        params = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "stopPrice": stop_price,
            "closePosition": "true",
            "workingType": self.working_type,
        }
        return self.client.place_order(params)

    def _place_algo_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        trigger_price: float,
        quantity: Optional[float],
        close_position: bool = False,
    ) -> Dict:
        params = {
            "algoType": "CONDITIONAL",
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "triggerPrice": trigger_price,
            "workingType": self.working_type,
            "priceProtect": "TRUE",
        }
        if close_position:
            params["closePosition"] = "true"
        else:
            params["quantity"] = quantity
            params["reduceOnly"] = "true"
        return self.client.place_algo_order(params)

    def place_stop_loss(self, symbol: str, side: str, qty: float, stop_price: float) -> Optional[Dict]:
        close_side = 'SELL' if side == 'BUY' else 'BUY'
        tick_size = self._get_tick_size(symbol)
        step_size, min_qty = self._get_step_size(symbol)

        rounded_stop_price = round_to_tick_size(stop_price, tick_size)
        rounded_qty = round_qty_to_step(qty, step_size)

        if rounded_qty < min_qty:
            logger.error(f"止損數量{rounded_qty}小於最小值{min_qty}")
            return None

        logger.info(f"📍 嘗試掛止損: {symbol} {close_side} @ {rounded_stop_price}")

        try:
            return self._place_reduce_only_order(
                symbol, close_side, "STOP_MARKET", rounded_stop_price, rounded_qty
            )
        except Exception as e:
            msg = str(e)
            logger.warning(f"止損 reduceOnly 失敗: {msg}")
            if self._should_try_close_position_fallback(msg):
                try:
                    logger.warning("改用 closePosition 方式掛止損")
                    return self._place_close_position_order(
                        symbol, close_side, "STOP_MARKET", rounded_stop_price
                    )
                except Exception as e2:
                    logger.error(f"止損 closePosition 失敗: {e2}")
                try:
                    logger.warning("改用 Algo Order 方式掛止損")
                    return self._place_algo_order(
                        symbol, close_side, "STOP_MARKET", rounded_stop_price, rounded_qty
                    )
                except Exception as e3:
                    logger.error(f"止損 Algo Order 失敗: {e3}")
            return None

    def place_take_profit(self, symbol: str, side: str, qty: float, tp_price: float) -> Optional[Dict]:
        close_side = 'SELL' if side == 'BUY' else 'BUY'
        tick_size = self._get_tick_size(symbol)
        step_size, min_qty = self._get_step_size(symbol)

        rounded_tp_price = round_to_tick_size(tp_price, tick_size)
        rounded_qty = round_qty_to_step(qty, step_size)

        if rounded_qty < min_qty:
            logger.error(f"止盈數量{rounded_qty}小於最小值{min_qty}")
            return None

        logger.info(f"📍 嘗試掛止盈: {symbol} {close_side} @ {rounded_tp_price}")

        try:
            return self._place_reduce_only_order(
                symbol, close_side, "TAKE_PROFIT_MARKET", rounded_tp_price, rounded_qty
            )
        except Exception as e:
            msg = str(e)
            logger.warning(f"止盈 reduceOnly 失敗: {msg}")
            if self._should_try_close_position_fallback(msg):
                try:
                    logger.warning("改用 closePosition 方式掛止盈")
                    return self._place_close_position_order(
                        symbol, close_side, "TAKE_PROFIT_MARKET", rounded_tp_price
                    )
                except Exception as e2:
                    logger.error(f"止盈 closePosition 失敗: {e2}")
                try:
                    logger.warning("改用 Algo Order 方式掛止盈")
                    return self._place_algo_order(
                        symbol, close_side, "TAKE_PROFIT_MARKET", rounded_tp_price, rounded_qty
                    )
                except Exception as e3:
                    logger.error(f"止盈 Algo Order 失敗: {e3}")
            return None

    # ==================== 核心確保流程 ====================

    def ensure_protection_orders(
        self,
        symbol: str,
        side: str,
        qty: float,
        stop_price: float,
        tp_price: Optional[float],
        max_retries: int = 3,
        strategy_id: Optional[str] = None,
        existing_algo_ids: Optional[Dict[str, Optional[int]]] = None,
    ) -> Dict[str, Optional[str]]:
        """
        強制確保止損與止盈掛單存在（含去重、自補、驗證、風控）
        Returns:
            {"ok": bool, "sl_order_id": Optional[str], "tp_order_id": Optional[str]}
        """
        last_sl_id = None
        last_tp_id = None
        last_sl_algo_id = None
        last_tp_algo_id = None

        if existing_algo_ids:
            sl_algo_id = existing_algo_ids.get("sl")
            tp_algo_id = existing_algo_ids.get("tp")
            sl_ok = sl_algo_id is not None and self._verify_algo_order(sl_algo_id)
            tp_ok = (tp_price is None) or (tp_algo_id is not None and self._verify_algo_order(tp_algo_id))
            if sl_ok and tp_ok:
                return {
                    "ok": True,
                    "sl_order_id": str(sl_algo_id),
                    "tp_order_id": str(tp_algo_id) if tp_algo_id is not None else None,
                }

        for attempt in range(max_retries):
            logger.info(f"🔁 保護單掛單嘗試 {attempt+1}/{max_retries}")

            existing = self.scan_existing_protective_orders(symbol)

            # ===== SL =====
            if existing["sl"] is None:
                sl_order = self.place_stop_loss(symbol, side, qty, stop_price)
                if sl_order:
                    last_sl_id = str(sl_order.get("orderId") or sl_order.get("algoId"))
                    last_sl_algo_id = sl_order.get("algoId")
                    logger.info(f"✅ 止損單送出成功: {last_sl_id}")
            else:
                last_sl_id = str(existing["sl"].get("orderId"))
                logger.info(f"🔒 已存在止損單 {last_sl_id}，略過重掛")

            # ===== TP =====
            if tp_price:
                if existing["tp"] is None:
                    tp_order = self.place_take_profit(symbol, side, qty, tp_price)
                    if tp_order:
                        last_tp_id = str(tp_order.get("orderId") or tp_order.get("algoId"))
                        last_tp_algo_id = tp_order.get("algoId")
                        logger.info(f"✅ 止盈單送出成功: {last_tp_id}")
                else:
                    last_tp_id = str(existing["tp"].get("orderId"))
                    logger.info(f"🔒 已存在止盈單 {last_tp_id}，略過重掛")

            # ===== 驗證 =====
            if self.verify_protective_orders(
                symbol,
                expect_tp=bool(tp_price),
                max_wait=5,
                algo_ids={"sl": last_sl_algo_id, "tp": last_tp_algo_id},
            ):
                logger.info("🛡️ 保護單確認成功")
                return {
                    "ok": True,
                    "sl_order_id": last_sl_id,
                    "tp_order_id": last_tp_id,
                }

            logger.warning("⚠️ 尚未確認保護單，準備重試")
            time.sleep(1)

        logger.critical("🚨🚨🚨 多次嘗試後仍無法確認保護單，啟動緊急平倉")

        if strategy_id:
            self.emergency.emergency_flatten_position(
                symbol, strategy_id, "止損/止盈掛單失敗"
            )

        return {
            "ok": False,
            "sl_order_id": last_sl_id,
            "tp_order_id": last_tp_id,
        }

    def _verify_algo_order(self, algo_id: int) -> bool:
        try:
            info = self.client.query_algo_order(algo_id=algo_id)
            status = str(info.get("algoStatus", "")).upper()
            return status not in {"CANCELED", "EXPIRED", "REJECTED"}
        except Exception as e:
            logger.warning(f"Algo單查詢失敗(algoId={algo_id}): {e}")
            return False

    # ==================== 重啟補單 ====================

    def _reconstruct_sl_tp(self, symbol: str, side: str, entry_price: float) -> Tuple[float, float]:
        """
        系統重啟且無法取得原策略 SL/TP 時，使用保守風控補掛
        """
        if side == "BUY":
            stop_price = entry_price * 0.997
            tp_price = entry_price * 1.008
        else:
            stop_price = entry_price * 1.003
            tp_price = entry_price * 0.992

        logger.warning(f"⚠️ 使用預設風控 SL/TP: SL={stop_price}, TP={tp_price}")
        return stop_price, tp_price

    def reconcile_positions(
        self,
        strategy_id: str = "RECONCILE",
        get_registered_ids: Optional[Callable[[str], Dict[str, Optional[int]]]] = None,
        update_registry: Optional[Callable[[str, Optional[str], Optional[str]], None]] = None,
    ) -> None:
        """
        啟動時掃描所有倉位，確保每個倉位都有 SL/TP，缺失則補掛
        """
        try:
            positions = self.client.get_position_risk()
            for pos in positions:
                qty = float(pos.get("positionAmt", 0))
                if qty == 0:
                    continue

                symbol = pos.get("symbol")
                side = "BUY" if qty > 0 else "SELL"
                entry_price = float(pos.get("entryPrice", 0))

                logger.warning(f"🔄 發現未結束倉位 {symbol} qty={qty}，檢查保護單")

                stop_price, tp_price = self._reconstruct_sl_tp(symbol, side, entry_price)

                existing_algo_ids = get_registered_ids(symbol) if get_registered_ids else None
                result = self.ensure_protection_orders(
                    symbol=symbol,
                    side=side,
                    qty=abs(qty),
                    stop_price=stop_price,
                    tp_price=tp_price,
                    max_retries=5,
                    strategy_id=strategy_id,
                    existing_algo_ids=existing_algo_ids,
                )
                if update_registry and result.get("ok"):
                    update_registry(symbol, result.get("sl_order_id"), result.get("tp_order_id"))

        except Exception as e:
            logger.error(f"重啟保護單修復失敗: {e}", exc_info=True)
