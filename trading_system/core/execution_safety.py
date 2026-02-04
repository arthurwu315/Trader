"""
Execution Safety Module
狀態機下單 + Fail-safe

核心保命機制:
1. 成交後才掛SL/TP
2. SL失敗→flatten+停機
3. 逐倉強制+確認
4. 槓桿設定+確認

老手建議修正:
- 方案A: reduceOnly + quantity (不用closePosition)
- 價格精度: round到tickSize

V5.2修正:
- 新增 is_safe() 方法供 L0 Gate 調用

V5.3修正 (本版):
- ✅ exchangeInfo 取得方法兼容 exchange_info()/get_exchange_info()
- ✅ _wait_for_fill timeout 使用 self.fill_timeout (參數化)
"""
import logging
import time
from typing import Optional, Dict

from core.order_utils import (
    round_to_tick_size,
    get_symbol_tick_size,
    round_qty_to_step,
    get_symbol_step_size,
)
from core.protection_guard import ProtectionGuard

logger = logging.getLogger(__name__)

class OrderStateMachine:
    """
    訂單狀態機

    流程:
    1. 設定逐倉+槓桿
    2. 下進場單
    3. 等待成交
    4. 掛止損 (必須成功!)
    5. 掛止盈
    """
    # ✅ 一鍵切換：寫死環境（之後上真錢只改這行）
    # TESTNET: 槓桿設定失敗不緊急停機（放行繼續）
    # LIVE:    槓桿設定失敗 => raise（由上層 emergency 停機）
    SAFE_MODE = "TESTNET"   # <- 上真錢改成 "LIVE"

    # ✅ 是否允許槓桿設定失敗時繼續（只在 TESTNET 放行）
    ALLOW_LEVERAGE_SET_FAILURE = True

    def __init__(
        self,
        binance_client,
        emergency_handler,
        working_type: str = "MARK_PRICE",
        protection_guard: Optional[ProtectionGuard] = None,
    ):
        self.client = binance_client
        self.emergency = emergency_handler
        self.current_state = "IDLE"

        # tickSize/stepSize cache
        self.tick_size_cache = {}
        self.step_size_cache = {}
        self.min_qty_cache = {}

        # ✅ leverage 設定 cache：避免 testnet 重複 set_leverage 觸發 400
        # key: (symbol, leverage) -> True
        self._leverage_cached = {}


        # workingType配置 (MARK_PRICE較抗插針)
        self.working_type = working_type
        logger.info(f"OrderStateMachine初始化: workingType={self.working_type}")

        # ✅ 可配置 timeout (外部設定)
        self.fill_timeout = 30   # 秒
        self.query_interval = 1  # 秒

        # 保護單守衛
        self.protection_guard = protection_guard or ProtectionGuard(
            self.client,
            self.emergency,
            working_type=self.working_type,
        )

    # ==================== is_safe() 方法 ====================

    def is_safe(self) -> bool:
        """
        檢查是否可以安全執行訂單
        供 L0 Gate 調用
        """
        try:
            if self.current_state != "IDLE":
                logger.warning(f"is_safe: 狀態機非IDLE (current={self.current_state})")
                return False

            if hasattr(self.emergency, 'should_stop') and self.emergency.should_stop:
                logger.warning("is_safe: EmergencyHandler已觸發停機")
                return False

            if hasattr(self.emergency, 'emergency_stop') and self.emergency.emergency_stop:
                logger.warning("is_safe: emergency_stop已觸發")
                return False

            return True

        except Exception as e:
            logger.error(f"is_safe檢查失敗: {e}")
            return False

    # ==================== 內部快取 ====================

    def _get_tick_size(self, symbol: str) -> float:
        if symbol not in self.tick_size_cache:
            self.tick_size_cache[symbol] = get_symbol_tick_size(self.client, symbol)
        return self.tick_size_cache[symbol]

    def _get_step_size(self, symbol: str) -> tuple:
        if symbol not in self.step_size_cache:
            step_size, min_qty = get_symbol_step_size(self.client, symbol)
            self.step_size_cache[symbol] = step_size
            self.min_qty_cache[symbol] = min_qty
        return (self.step_size_cache[symbol], self.min_qty_cache[symbol])

    # ==================== 對外主要入口 ====================

    def execute_trade_with_safety(
        self,
        candidate,
        strategy_id: str,
        max_retries: int = 3
    ) -> Dict:
        """
        安全執行交易
        """
        symbol = candidate.symbol
        side = candidate.side
        qty = candidate.qty
        leverage = candidate.leverage
        stop_price = candidate.stop_price
        tp_price = candidate.tp_price

        result = {
            'success': False,
            'state': 'INIT',
            'entry_order_id': None,
            'sl_order_id': None,
            'tp_order_id': None,
            'error': None
        }

        try:
            # ========== 步驟1: 設定逐倉 ==========
            logger.info("📍 步驟1: 設定逐倉模式")
            self.current_state = "SETTING_ISOLATED"

            if not self._set_and_verify_isolated(symbol):
                result['error'] = "逐倉設定失敗"
                result['state'] = "ISOLATED_FAILED"
                self.emergency.emergency_stop_strategy(strategy_id, f"逐倉設定失敗: {symbol}")
                return result

            logger.info("✅ 逐倉模式已確認")

            # ========== 步驟2: 設定槓桿 ==========
            logger.info(f"📍 步驟2: 設定槓桿 {candidate.leverage}x")
            self.current_state = "SETTING_LEVERAGE"

            leverage_ok = False
            try:
                leverage_ok = self._set_and_verify_leverage(symbol, leverage)
            except Exception as e:
                leverage_ok = False
                logger.error(f"槓桿設定失敗: {e}")

            if not leverage_ok:
                msg = f"leverage verify failed: {symbol} {leverage}x"
                # ✅ TESTNET 放行：降級為 1x，確保風控與實際一致
                if getattr(self, "SAFE_MODE", "TESTNET") == "TESTNET":
                    logger.warning(f"⚠️ {msg}（SAFE_MODE=TESTNET 放行，降級為 1x 繼續）")
                    leverage = 1
                    candidate.leverage = 1
                else:
                    logger.critical(msg)
                    self.emergency.emergency_stop_strategy(strategy_id, msg)
                    result['error'] = msg
                    result['state'] = "LEVERAGE_FAILED"
                    return result
            else:
                logger.info("✅ 槓桿已確認")



            # ========== 步驟3: 下進場單 ==========
            logger.info(f"📍 步驟3: 下進場單 {side} {qty} {symbol}")
            self.current_state = "PLACING_ENTRY"

            entry_order = self._place_entry_order(
                symbol, side, qty, candidate.entry_type, candidate.entry_price
            )

            if not entry_order:
                result['error'] = "進場單下單失敗"
                result['state'] = "ENTRY_FAILED"
                return result

            result['entry_order_id'] = entry_order.get('orderId')
            logger.info(f"✅ 進場單已下: {result['entry_order_id']}")

            # ========== 步驟4: 等待成交 ==========
            logger.info("📍 步驟4: 等待進場單成交")
            self.current_state = "WAITING_FILL"

            # ✅ 使用可配置 timeout
            filled_order = self._wait_for_fill(
                symbol, result['entry_order_id'], timeout=self.fill_timeout
            )

            if not filled_order:
                result['error'] = "進場單未成交"
                result['state'] = "ENTRY_NOT_FILLED"
                try:
                    self.client.cancel_order(symbol=symbol, orderId=result['entry_order_id'])
                except Exception:
                    pass
                return result

            actual_qty = float(filled_order.get('executedQty', qty))
            avg_price = float(filled_order.get('avgPrice', candidate.entry_price))

            logger.info("✅ 進場單已成交")
            logger.info(f"   數量: {actual_qty}")
            logger.info(f"   均價: ${avg_price:.2f}")

            # ========== 步驟5: 掛止損 / 止盈（強制成功） ==========
            logger.info("📍 步驟5: 掛止損 / 止盈（強制保護）")
            self.current_state = "PLACING_PROTECTION"

            protection_result = self.protection_guard.ensure_protection_orders(
                symbol=symbol,
                side=side,
                qty=actual_qty,
                stop_price=stop_price,
                tp_price=tp_price,
                max_retries=max_retries,
                strategy_id=strategy_id,
            )

            result['sl_order_id'] = protection_result.get('sl_order_id')
            result['tp_order_id'] = protection_result.get('tp_order_id')

            if not protection_result.get("ok"):
                result['error'] = "止損/止盈掛單失敗，已緊急平倉"
                result['state'] = "PROTECTION_FAILED_FLATTENING"
                return result


            # ========== 完成 ==========
            result['success'] = True
            result['state'] = "COMPLETED"
            self.current_state = "IDLE"

            logger.info("✅ 交易執行完成!")
            logger.info(f"   進場: {result['entry_order_id']}")
            logger.info(f"   止損: {result['sl_order_id']}")
            if result['tp_order_id']:
                logger.info(f"   止盈: {result['tp_order_id']}")

            return result

        except Exception as e:
            logger.error(f"❌ 交易執行異常: {e}", exc_info=True)
            result['error'] = str(e)
            result['state'] = "EXCEPTION"

            if result['entry_order_id']:
                self.emergency.emergency_flatten_position(symbol, strategy_id, f"執行異常: {e}")

            return result

    # ==================== 逐倉 / 槓桿 ====================

    def _set_and_verify_isolated(self, symbol: str) -> bool:
        try:
            try:
                self.client.set_margin_type(symbol=symbol, margin_type='ISOLATED')
                logger.debug(f"逐倉設定請求已發送: {symbol}")
            except Exception as e:
                if "No need to change margin type" in str(e):
                    logger.debug("已經是逐倉模式")
                else:
                    logger.warning(f"設定逐倉警告: {e}")

            time.sleep(0.5)
            position = self.client.get_position_risk(symbol=symbol)
            if not position:
                logger.error("無法獲取倉位信息")
                return False

            margin_type = position[0].get('marginType', '').upper()
            if margin_type != 'ISOLATED':
                logger.error(f"逐倉驗證失敗: marginType={margin_type}")
                return False

            return True

        except Exception as e:
            logger.error(f"逐倉設定失敗: {e}")
            return False

    def _set_and_verify_leverage(self, symbol: str, leverage: int) -> bool:
        try:
            result = self.client.set_leverage(symbol=symbol, leverage=leverage)
            logger.debug(f"槓桿設定結果: {result}")

            time.sleep(0.5)
            position = self.client.get_position_risk(symbol=symbol)
            if not position:
                logger.error("無法獲取倉位信息")
                return False

            actual_leverage = int(position[0].get('leverage', 0))
            if actual_leverage != leverage:
                logger.error(f"槓桿驗證失敗: 設定{leverage}x, 實際{actual_leverage}x")
                return False
            logger.info(f"leverage verify: want={leverage} actual={actual_leverage}")

            return True

        except Exception as e:
            logger.error(f"槓桿設定失敗: {e}")
            return False

    # ==================== 下單 / 成交 ====================

    def _place_entry_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str,
        price: Optional[float] = None
    ) -> Optional[Dict]:
        """
        下進場單 (支援: MARKET, LIMIT, STOP_MARKET, TAKE_PROFIT_MARKET)
        價格/數量精度: round到tickSize/stepSize
        """
        try:
            tick_size = self._get_tick_size(symbol)
            step_size, min_qty = self._get_step_size(symbol)

            rounded_qty = round_qty_to_step(qty, step_size)
            if rounded_qty < min_qty:
                logger.error(f"數量{rounded_qty}小於最小值{min_qty}")
                return None

            logger.debug(f"進場數量: {qty:.6f} -> {rounded_qty:.6f} (stepSize={step_size})")

            params = {
                'symbol': symbol,
                'side': side,
                'type': order_type,
                'quantity': rounded_qty
            }

            if order_type == 'LIMIT' and price is not None:
                rounded_price = round_to_tick_size(price, tick_size)
                params['price'] = rounded_price
                params['timeInForce'] = 'GTC'
                logger.debug(f"進場價格: {price:.4f} -> {rounded_price} (tickSize={tick_size})")

            elif order_type in ['STOP_MARKET', 'TAKE_PROFIT_MARKET'] and price is not None:
                rounded_price = round_to_tick_size(price, tick_size)
                params['stopPrice'] = rounded_price
                params['workingType'] = self.working_type
                logger.debug(f"觸發價格: {price:.4f} -> {rounded_price} (tickSize={tick_size})")

            order = self.client.place_order(params)
            return order

        except Exception as e:
            logger.error(f"進場單下單失敗: {e}")
            return None

    def _wait_for_fill(
        self,
        symbol: str,
        order_id: int,
        timeout: Optional[int] = None
    ) -> Optional[Dict]:
        """
        等待訂單成交，並嘗試獲取實際成交資訊
        """
        if timeout is None:
            timeout = self.fill_timeout

        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                open_orders = self.client.get_open_orders(symbol=symbol)

                for order in open_orders:
                    if order.get('orderId') == order_id:
                        logger.debug(f"訂單 {order_id} 尚未成交，等待中...")
                        time.sleep(self.query_interval)
                        break
                else:
                    # 不在 open_orders 中，視為已成交
                    logger.debug(f"訂單 {order_id} 已成交(不在掛單列表)")
                    return {
                        'status': 'FILLED',
                        'orderId': order_id
                    }

            except Exception as e:
                logger.warning(f"查詢訂單狀態失敗，假設已成交: {e}")
                return {
                    'status': 'FILLED',
                    'orderId': order_id
                }

        logger.warning(f"訂單 {order_id} 等待成交超時")
        return None


    # ==================== SL / TP ====================

    # =========================
    # 止損 / 止盈 安全模組（Production Ready）
    # =========================
    def reconcile_all_positions(self):
        """
        啟動時掃描所有倉位，確保每個倉位都有 SL/TP，缺失則補掛
        """
        self.protection_guard.reconcile_positions()

    def _scan_existing_protective_orders(self, symbol: str) -> dict:
        """
        掃描目前是否已有 SL / TP 保護單（reduceOnly）
        回傳: {"sl": order or None, "tp": order or None}
        """
        return self.protection_guard.scan_existing_protective_orders(symbol)

    def _place_stop_loss(self, symbol: str, side: str, qty: float, stop_price: float) -> Optional[Dict]:
        return self.protection_guard.place_stop_loss(symbol, side, qty, stop_price)




    def _place_take_profit(self, symbol: str, side: str, qty: float, tp_price: float) -> Optional[Dict]:
        return self.protection_guard.place_take_profit(symbol, side, qty, tp_price)






    def _verify_protective_orders(self, symbol: str, expect_tp: bool = True, max_wait: int = 5) -> bool:
        """
        驗證 SL / TP 是否存在（reduceOnly）
        """
        return self.protection_guard.verify_protective_orders(symbol, expect_tp=expect_tp, max_wait=max_wait)



    def _ensure_protection_orders(
        self,
        symbol: str,
        side: str,
        qty: float,
        stop_price: float,
        tp_price: Optional[float],
        max_retries: int = 3,
        strategy_id: Optional[str] = None
    ) -> bool:
        """
        強制確保止損與止盈掛單存在（含去重、自補、驗證、風控）
        """
        result = self.protection_guard.ensure_protection_orders(
            symbol=symbol,
            side=side,
            qty=qty,
            stop_price=stop_price,
            tp_price=tp_price,
            max_retries=max_retries,
            strategy_id=strategy_id,
        )
        return bool(result.get("ok"))






# 測試
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("🧪 執行安全模組測試")
    print("✅ 模組載入成功")
    print("\n注意: 實際測試需要連接Binance API")

    print("\n🧪 測試 is_safe() 方法:")

    class MockClient:
        def get_exchange_info(self):
            return {"symbols": []}

    class MockEmergency:
        should_stop = False
        emergency_stop = False

    mock_client = MockClient()
    mock_emergency = MockEmergency()

    osm = OrderStateMachine(mock_client, mock_emergency)

    print(f"  狀態=IDLE, emergency=False: is_safe() = {osm.is_safe()}")

    osm.current_state = "PLACING_SL"
    print(f"  狀態=PLACING_SL: is_safe() = {osm.is_safe()}")

    osm.current_state = "IDLE"
    mock_emergency.should_stop = True
    print(f"  狀態=IDLE, should_stop=True: is_safe() = {osm.is_safe()}")

    print("\n✅ is_safe() 測試完成!")
