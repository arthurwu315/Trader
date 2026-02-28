"""
Enhanced Binance Futures REST Client
專業級API客戶端: 限速、重試、完整錯誤處理
"""
import time
import hmac
import hashlib
import urllib.parse
import requests
from typing import Any, Dict, Optional, Tuple, Callable
from collections import deque
from threading import Lock
import logging

logger = logging.getLogger(__name__)

class RateLimiter:
    """API限速器"""
    def __init__(self, max_calls: int, time_window: int = 60):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = deque()
        self.lock = Lock()
    
    def wait_if_needed(self) -> None:
        """如果超過限速則等待"""
        with self.lock:
            now = time.time()
            # 移除時間窗口外的記錄
            while self.calls and self.calls[0] < now - self.time_window:
                self.calls.popleft()
            
            # 如果已達上限,等待
            if len(self.calls) >= self.max_calls:
                sleep_time = self.calls[0] + self.time_window - now + 0.1
                if sleep_time > 0:
                    logger.warning(f"達到API限速,等待 {sleep_time:.2f} 秒")
                    time.sleep(sleep_time)
                    # 重新計算
                    now = time.time()
                    while self.calls and self.calls[0] < now - self.time_window:
                        self.calls.popleft()
            
            self.calls.append(now)

class BinanceAPIError(Exception):
    """幣安API錯誤"""
    def __init__(self, code: int, message: str, response: Any = None):
        self.code = code
        self.message = message
        self.response = response
        super().__init__(f"[{code}] {message}")

class BinanceFuturesClient:
    """專業級幣安合約REST客戶端"""


    def __init__(
        self,
        base_url: str,
        api_key: str,
        api_secret: str,
        max_calls_per_minute: int = 1200,
        timeout: int = 30,
        max_retries: int = 3,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.api_secret = api_secret.encode()
        self.timeout = timeout
        self.max_retries = max_retries
        
        # 限速器
        self.rate_limiter = RateLimiter(max_calls_per_minute, 60)
        
        # Session (連接池)
        self.session = requests.Session()
        self.session.headers.update({
            "X-MBX-APIKEY": self.api_key,
            "Content-Type": "application/json",
        })
    
    def _sign(self, params: Dict[str, Any]) -> str:
        """簽名"""
        query_string = urllib.parse.urlencode(params, doseq=True)
        signature = hmac.new(
            self.api_secret,
            query_string.encode(),
            hashlib.sha256
        ).hexdigest()
        return query_string + "&signature=" + signature
    
    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        signed: bool = False,
    ) -> Tuple[int, Any]:
        """
        底層請求方法
        
        Returns:
            (status_code, response_data)
        """
        params = params or {}
        
        # 限速
        self.rate_limiter.wait_if_needed()
        
        # 簽名請求
        if signed:
            params["recvWindow"] = 5000
            params["timestamp"] = int(time.time() * 1000)
            query_string = self._sign(params)
            url = f"{self.base_url}{endpoint}?{query_string}"
        else:
            url = f"{self.base_url}{endpoint}"
            if params:
                url += "?" + urllib.parse.urlencode(params)
        
        # 發送請求
        try:
            response = self.session.request(
                method,
                url,
                timeout=self.timeout
            )
            
            # 解析響應
            try:
                data = response.json()
            except ValueError:
                data = response.text
            
            return response.status_code, data
            
        except requests.Timeout:
            raise BinanceAPIError(0, f"請求超時 (>{self.timeout}s)")
        except requests.ConnectionError as e:
            raise BinanceAPIError(0, f"連接錯誤: {e}")
        except Exception as e:
            raise BinanceAPIError(0, f"未知錯誤: {e}")
    
    def _call_with_retry(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        signed: bool = False,
    ) -> Any:
        """
        帶重試的API呼叫
        
        Returns:
            response_data (成功時)
            
        Raises:
            BinanceAPIError
        """
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                status, data = self._request(method, endpoint, params, signed)
                
                # 成功
                if 200 <= status < 300:
                    return data
                
                # 失敗 - 分析錯誤類型
                error_msg = str(data)
                if isinstance(data, dict):
                    error_msg = data.get("msg", str(data))
                
                # 特定錯誤碼處理
                if status == 429:  # Rate limit
                    wait_time = 60
                    logger.warning(f"API限速 (429),等待 {wait_time}s")
                    time.sleep(wait_time)
                    continue
                
                if status in (502, 503, 504):  # 服務器錯誤
                    if attempt < self.max_retries - 1:
                        wait_time = 2 ** attempt
                        logger.warning(f"服務器錯誤 ({status}),重試 {attempt+1}/{self.max_retries}")
                        time.sleep(wait_time)
                        continue
                
                # 其他錯誤直接拋出
                raise BinanceAPIError(status, error_msg, data)
                
            except BinanceAPIError as e:
                last_error = e
                if e.code == 0:  # 網路錯誤,可重試
                    if attempt < self.max_retries - 1:
                        wait_time = 2 ** attempt
                        logger.warning(f"網路錯誤,重試 {attempt+1}/{self.max_retries}: {e}")
                        time.sleep(wait_time)
                        continue
                raise
        
        # 重試耗盡
        if last_error:
            raise last_error
        raise BinanceAPIError(0, "重試耗盡")
    
    # ===== Public API =====
    def exchange_info(self, symbol: str = None):
    # 給 execution_safety 用的相容層
        return self.get_exchange_info(symbol)

    def get_exchange_info(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """獲取交易規則"""
        params = {"symbol": symbol} if symbol else {}
        return self._call_with_retry("GET", "/fapi/v1/exchangeInfo", params)
    

    def exchange_info(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """別名: 相容execution_safety.py"""
        return self.get_exchange_info(symbol)
    def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> list:
        """
        獲取K線數據
        
        Args:
            interval: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
            limit: 最多1500
        """
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": min(limit, 1500),
        }
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        
        return self._call_with_retry("GET", "/fapi/v1/klines", params)
    
    def get_ticker_price(self, symbol: str) -> Dict[str, Any]:
        """獲取最新價格"""
        return self._call_with_retry("GET", "/fapi/v1/ticker/price", {"symbol": symbol})
    
    def get_24h_ticker(self, symbol: str) -> Dict[str, Any]:
        """獲取24小時統計"""
        return self._call_with_retry("GET", "/fapi/v1/ticker/24hr", {"symbol": symbol})
    
    # ===== Private API =====
    
    def get_account(self) -> Dict[str, Any]:
        """獲取帳戶資訊"""
        return self._call_with_retry("GET", "/fapi/v2/account", {}, signed=True)

    def futures_account(self):
    # backward-compatible alias
        return self.get_account()

    def get_server_time(self):
    # if you don't have server time endpoint wrapped, you can return None or call exchange endpoint if exists
        return None

    def get_balance(self) -> list:
        """獲取帳戶餘額"""
        return self._call_with_retry("GET", "/fapi/v2/balance", {}, signed=True)
    
    def get_position_risk(self, symbol: Optional[str] = None) -> list:
        """獲取倉位資訊"""
        params = {"symbol": symbol} if symbol else {}
        return self._call_with_retry("GET", "/fapi/v2/positionRisk", params, signed=True)
    
    def get_open_orders(self, symbol: Optional[str] = None) -> list:
        """獲取掛單。symbol=None 時回傳所有標的掛單。"""
        params: Dict[str, Any] = {} if symbol is None else {"symbol": symbol}
        return self._call_with_retry("GET", "/fapi/v1/openOrders", params, signed=True)

    def place_order_test(self, params: Dict[str, Any]) -> None:
        """測試下單請求（不實際執行）。用於連通性驗證。"""
        self._call_with_retry("POST", "/fapi/v1/order/test", params, signed=True)

    def place_order(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """下單"""
        return self._call_with_retry("POST", "/fapi/v1/order", params, signed=True)

    def place_algo_order(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """下單 - Algo Order"""
        return self._call_with_retry("POST", "/fapi/v1/algoOrder", params, signed=True)

    def query_algo_order(
        self,
        algo_id: Optional[int] = None,
        client_algo_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """查詢 Algo Order"""
        params: Dict[str, Any] = {}
        if algo_id is not None:
            params["algoId"] = algo_id
        if client_algo_id is not None:
            params["clientAlgoId"] = client_algo_id
        return self._call_with_retry("GET", "/fapi/v1/algoOrder", params, signed=True)
    
    def cancel_order(self, symbol: str, order_id: Optional[int] = None, client_order_id: Optional[str] = None) -> Dict[str, Any]:
        """取消訂單"""
        params = {"symbol": symbol}
        if order_id:
            params["orderId"] = order_id
        if client_order_id:
            params["origClientOrderId"] = client_order_id
        return self._call_with_retry("DELETE", "/fapi/v1/order", params, signed=True)
    
    def cancel_all_orders(self, symbol: str) -> Dict[str, Any]:
        """取消所有訂單"""
        return self._call_with_retry("DELETE", "/fapi/v1/allOpenOrders", {"symbol": symbol}, signed=True)
    
    def set_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]:
        """設定槓桿"""
        return self._call_with_retry("POST", "/fapi/v1/leverage", {"symbol": symbol, "leverage": leverage}, signed=True)
    
    def set_margin_type(self, symbol: str, margin_type: str) -> Dict[str, Any]:
        """設定保證金模式 (ISOLATED/CROSS)"""
        return self._call_with_retry("POST", "/fapi/v1/marginType", {"symbol": symbol, "marginType": margin_type}, signed=True)
    
    def get_income_history(self, symbol: Optional[str] = None, limit: int = 100) -> list:
        """獲取資金流水"""
        params = {"limit": limit}
        if symbol:
            params["symbol"] = symbol
        return self._call_with_retry("GET", "/fapi/v1/income", params, signed=True)

    def get_user_trades(self, symbol: str, limit: int = 30) -> list:
        """獲取帳戶成交紀錄（含價格、realizedPnl），用於平倉時取得真實出場價。"""
        return self._call_with_retry("GET", "/fapi/v1/userTrades", {"symbol": symbol, "limit": limit}, signed=True)

    # ===== Compatibility aliases for execution_safety =====

    def futures_create_order(self, **params):
        """相容 execution_safety: futures_create_order -> place_order"""
        return self.place_order(params)

    def futures_get_open_orders(self, symbol: str):
        """相容 execution_safety: futures_get_open_orders -> get_open_orders"""
        return self.get_open_orders(symbol)

    def futures_cancel_order(self, symbol: str, orderId: Optional[int] = None, origClientOrderId: Optional[str] = None):
        """相容 execution_safety: futures_cancel_order -> cancel_order"""
        return self.cancel_order(symbol, order_id=orderId, client_order_id=origClientOrderId)

    def futures_cancel_all_open_orders(self, symbol: str):
        """相容 execution_safety: futures_cancel_all_open_orders -> cancel_all_orders"""
        return self.cancel_all_orders(symbol)

    def futures_position_information(self, symbol: Optional[str] = None):
        """相容 execution_safety: futures_position_information -> get_position_risk"""
        return self.get_position_risk(symbol)

    def futures_account(self):
        """相容 execution_safety: futures_account -> get_account"""
        return self.get_account()





if __name__ == "__main__":
    # 測試
    import os
    logging.basicConfig(level=logging.INFO)
    
    client = BinanceFuturesClient(
        base_url=os.getenv("BINANCE_BASE_URL", "https://testnet.binancefuture.com"),
        api_key=os.getenv("BINANCE_API_KEY", ""),
        api_secret=os.getenv("BINANCE_API_SECRET", ""),
    )
    
    try:
        # 測試公開API
        price = client.get_ticker_price("BTCUSDT")
        print(f"✅ BTC價格: {price}")
        
        # 測試私有API
        balance = client.get_balance()
        print(f"✅ 帳戶餘額: {balance[:2]}")
        
    except BinanceAPIError as e:
        print(f"❌ API錯誤: {e}")
