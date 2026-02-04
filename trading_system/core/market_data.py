"""
Market Data Manager
處理K線數據、技術指標計算、數據緩存
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class MarketDataManager:
    """市場數據管理器"""
    
    def __init__(self, client):
        self.client = client
        self.cache: Dict[str, pd.DataFrame] = {}
        self.cache_ttl_seconds = 300  # 5分鐘緩存
        self.cache_timestamps: Dict[str, float] = {}
    
    def get_klines_df(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        獲取K線數據並轉為DataFrame
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        import time
        
        cache_key = f"{symbol}_{interval}_{limit}"
        
        # 檢查緩存
        if use_cache and cache_key in self.cache:
            cache_age = time.time() - self.cache_timestamps.get(cache_key, 0)
            if cache_age < self.cache_ttl_seconds:
                logger.debug(f"使用緩存數據: {cache_key} (age: {cache_age:.1f}s)")
                return self.cache[cache_key].copy()
        
        # 從API獲取
        klines = self.client.get_klines(symbol, interval, limit)
        
        if not klines:
            raise ValueError(f"無法獲取K線數據: {symbol} {interval}")
        
        # 轉換為DataFrame
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # 類型轉換
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        # 只保留需要的列
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # 更新緩存
        self.cache[cache_key] = df.copy()
        self.cache_timestamps[cache_key] = time.time()
        
        logger.debug(f"獲取K線數據: {symbol} {interval} x{len(df)}")
        return df
    
    def calculate_ema(self, df: pd.DataFrame, period: int, column: str = 'close') -> pd.Series:
        """計算EMA"""
        return df[column].ewm(span=period, adjust=False).mean()
    
    def calculate_sma(self, df: pd.DataFrame, period: int, column: str = 'close') -> pd.Series:
        """計算SMA"""
        return df[column].rolling(window=period).mean()
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        計算ATR (Average True Range)
        
        Returns:
            Series of ATR values
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ATR = EMA of TR
        atr = tr.ewm(span=period, adjust=False).mean()
        
        return atr
    
    def calculate_atr_percentage(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """計算ATR百分比 (ATR / close)"""
        atr = self.calculate_atr(df, period)
        return (atr / df['close']) * 100
    
    def calculate_rsi(self, df: pd.DataFrame, period: int = 14, column: str = 'close') -> pd.Series:
        """計算RSI"""
        delta = df[column].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_bollinger_bands(
        self,
        df: pd.DataFrame,
        period: int = 20,
        std_dev: float = 2.0,
        column: str = 'close'
    ) -> Dict[str, pd.Series]:
        """計算布林通道"""
        sma = df[column].rolling(window=period).mean()
        std = df[column].rolling(window=period).std()
        
        return {
            'middle': sma,
            'upper': sma + (std * std_dev),
            'lower': sma - (std * std_dev),
        }
    
    def get_technical_indicators(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
    ) -> pd.DataFrame:
        """
        獲取帶技術指標的K線數據
        
        Returns:
            DataFrame with OHLCV + indicators
        """
        df = self.get_klines_df(symbol, interval, limit)
        
        # 計算常用指標
        df['ema_21'] = self.calculate_ema(df, 21)
        df['ema_50'] = self.calculate_ema(df, 50)
        df['sma_20'] = self.calculate_sma(df, 20)
        
        df['atr_14'] = self.calculate_atr(df, 14)
        df['atr_pct'] = self.calculate_atr_percentage(df, 14)
        
        df['rsi_14'] = self.calculate_rsi(df, 14)
        
        # 成交量均線
        df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
        
        return df
    
    def check_trend_alignment(
        self,
        weekly_df: pd.DataFrame,
        daily_df: pd.DataFrame,
        ema_period: int = 21,
    ) -> Dict[str, Any]:
        """
        檢查多時間週期趨勢一致性
        
        Returns:
            {
                'weekly_bullish': bool,
                'daily_bullish': bool,
                'aligned': bool,
                'weekly_ema': float,
                'daily_ema': float,
            }
        """
        # 週線趨勢
        w_ema = self.calculate_ema(weekly_df, ema_period)
        w_close = weekly_df['close'].iloc[-1]
        w_ema_val = w_ema.iloc[-1]
        w_ema_prev = w_ema.iloc[-2] if len(w_ema) > 1 else w_ema_val
        
        weekly_bullish = (w_close > w_ema_val) and (w_ema_val > w_ema_prev)
        
        # 日線趨勢
        d_ema = self.calculate_ema(daily_df, ema_period)
        d_close = daily_df['close'].iloc[-1]
        d_ema_val = d_ema.iloc[-1]
        d_ema_prev = d_ema.iloc[-2] if len(d_ema) > 1 else d_ema_val
        
        daily_bullish = (d_close > d_ema_val) and (d_ema_val > d_ema_prev)
        
        return {
            'weekly_bullish': weekly_bullish,
            'daily_bullish': daily_bullish,
            'aligned': weekly_bullish and daily_bullish,
            'weekly_ema': w_ema_val,
            'daily_ema': d_ema_val,
            'weekly_close': w_close,
            'daily_close': d_close,
        }
    
    def detect_volatility_contraction(
        self,
        df: pd.DataFrame,
        atr_period: int = 14,
        lookback_window: int = 90,
        low_percentile: float = 0.35,
    ) -> Dict[str, Any]:
        """
        檢測波動性收縮
        
        Returns:
            {
                'contracted': bool,
                'current_atr_pct': float,
                'threshold': float,
                'percentile_rank': float,
            }
        """
        if len(df) < lookback_window + atr_period:
            return {
                'contracted': False,
                'current_atr_pct': 0,
                'threshold': 0,
                'percentile_rank': 0,
                'error': 'insufficient data'
            }
        
        # 計算ATR%序列
        atr_pct = self.calculate_atr_percentage(df, atr_period)
        
        # 取最近lookback_window的ATR%
        recent_atr = atr_pct.iloc[-lookback_window:].dropna()
        
        if len(recent_atr) == 0:
            return {
                'contracted': False,
                'current_atr_pct': 0,
                'threshold': 0,
                'percentile_rank': 0,
                'error': 'no valid atr data'
            }
        
        # 當前ATR%
        current = atr_pct.iloc[-1]
        
        # 計算閾值
        threshold = recent_atr.quantile(low_percentile)
        
        # 計算當前ATR%在歷史中的位置
        percentile_rank = (recent_atr < current).sum() / len(recent_atr)
        
        return {
            'contracted': current < threshold,
            'current_atr_pct': float(current),
            'threshold': float(threshold),
            'percentile_rank': float(percentile_rank),
        }
    
    def detect_breakout(
        self,
        df: pd.DataFrame,
        lookback: int = 3,
        buffer_pct: float = 0.002,
    ) -> Dict[str, Any]:
        """
        檢測突破前N天最高點
        
        Returns:
            {
                'breakout_level': float,  # 突破價位
                'current_price': float,
                'distance_pct': float,  # 距離突破點的百分比
                'high_n_days': float,  # N天最高點
            }
        """
        if len(df) < lookback + 1:
            return {
                'breakout_level': 0,
                'current_price': 0,
                'distance_pct': 0,
                'high_n_days': 0,
                'error': 'insufficient data'
            }
        
        # 前N天最高點 (不包含當前K線)
        high_n = df['high'].iloc[-(lookback+1):-1].max()
        
        # 突破價格
        breakout_price = high_n * (1 + buffer_pct)
        
        # 當前價格
        current = df['close'].iloc[-1]
        
        # 距離
        distance = (current - breakout_price) / breakout_price
        
        return {
            'breakout_level': float(breakout_price),
            'current_price': float(current),
            'distance_pct': float(distance * 100),
            'high_n_days': float(high_n),
        }
    
    def check_volume_surge(
        self,
        df: pd.DataFrame,
        ma_period: int = 20,
        surge_ratio: float = 1.2,
    ) -> Dict[str, Any]:
        """
        檢查成交量放大
        
        Returns:
            {
                'volume_surge': bool,
                'current_volume': float,
                'avg_volume': float,
                'ratio': float,
            }
        """
        if len(df) < ma_period + 1:
            return {
                'volume_surge': False,
                'current_volume': 0,
                'avg_volume': 0,
                'ratio': 0,
                'error': 'insufficient data'
            }
        
        current_vol = df['volume'].iloc[-1]
        avg_vol = df['volume'].iloc[-(ma_period+1):-1].mean()
        
        if avg_vol == 0:
            return {
                'volume_surge': False,
                'current_volume': float(current_vol),
                'avg_volume': 0,
                'ratio': 0,
            }
        
        ratio = current_vol / avg_vol
        
        return {
            'volume_surge': ratio >= surge_ratio,
            'current_volume': float(current_vol),
            'avg_volume': float(avg_vol),
            'ratio': float(ratio),
        }

if __name__ == "__main__":
    # 測試
    import os
    from binance_client import BinanceFuturesClient
    
    logging.basicConfig(level=logging.INFO)
    
    client = BinanceFuturesClient(
        base_url=os.getenv("BINANCE_BASE_URL", "https://testnet.binancefuture.com"),
        api_key=os.getenv("BINANCE_API_KEY", ""),
        api_secret=os.getenv("BINANCE_API_SECRET", ""),
    )
    
    dm = MarketDataManager(client)
    
    # 測試獲取K線
    daily = dm.get_klines_df("BTCUSDT", "1d", 100)
    print(f"✅ 日線數據: {len(daily)} 條")
    print(daily.tail())
    
    # 測試技術指標
    indicators = dm.get_technical_indicators("BTCUSDT", "1d", 100)
    print(f"\n✅ 技術指標:")
    print(indicators[['close', 'ema_21', 'atr_pct', 'rsi_14']].tail())
    
    # 測試波動性檢測
    vol = dm.detect_volatility_contraction(daily)
    print(f"\n✅ 波動性檢測: {vol}")
