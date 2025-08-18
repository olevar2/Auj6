"""
Yahoo Finance Data Provider for the AUJ Platform.

This provider uses the free yfinance library to serve as a secondary/fallback
data source for historical OHLCV data when MetaTrader 5 data is not available
or for additional market coverage. No API key required.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
from decimal import Decimal
import yfinance as yf
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .base_provider import (
    BaseDataProvider, DataProviderType, DataProviderCapabilities,
    Timeframe, ConnectionStatus, DataType
)
from ..core.exceptions import DataProviderError, ConnectionError, DataNotAvailableError
from ..core.logging_setup import get_logger
from ..core.unified_config import UnifiedConfigManager

logger = get_logger(__name__)


class YahooFinanceProvider(BaseDataProvider):
    """
    Yahoo Finance provider for historical OHLCV data using the free yfinance library.

    Serves as a secondary data source and provides broader market coverage
    including stocks, indices, and additional forex pairs. No API key required.
    """

    def __init__(self, config_manager: UnifiedConfigManager, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Yahoo Finance provider.

        Args:
            config_manager: Unified configuration manager instance
            config: Configuration including timeout and retry settings (deprecated)
                   No API key required - uses free yfinance library
        """
        # Initialize forex symbols
        self.forex_symbols = {
            'EURUSD': 'EURUSD=X',
            'GBPUSD': 'GBPUSD=X',
            'USDJPY': 'USDJPY=X',
            'USDCHF': 'USDCHF=X',
            'AUDUSD': 'AUDUSD=X',
            'USDCAD': 'USDCAD=X',
            'NZDUSD': 'NZDUSD=X',
            'EURJPY': 'EURJPY=X',
            'GBPJPY': 'GBPJPY=X',
            'EURGBP': 'EURGBP=X'
        }

        # Major stock indices and ETFs
        self.stock_symbols = {
            'SPY': 'SPY',  # S&P 500 ETF
            'QQQ': 'QQQ',  # NASDAQ ETF
            'DIA': 'DIA',  # Dow Jones ETF
            'VIX': '^VIX', # Volatility Index
            'GOLD': 'GC=F', # Gold Futures
            'OIL': 'CL=F'   # Oil Futures
        }

        # Timeframe mapping for Yahoo Finance - MUST be initialized before parent constructor
        self.timeframe_mapping = {
            Timeframe.M1: "1m",
            Timeframe.M5: "5m",
            Timeframe.M15: "15m",
            Timeframe.M30: "30m",
            Timeframe.H1: "1h",
            Timeframe.H4: "4h",  # Not directly supported, will use 1h and resample
            Timeframe.D1: "1d",
            Timeframe.W1: "1wk",
            Timeframe.MN1: "1mo"
        }

        # Combined available symbols - MUST be initialized before parent constructor
        self.available_symbols = list(self.forex_symbols.keys()) + list(self.stock_symbols.keys())

        # Call parent constructor AFTER all required attributes are set
        super().__init__(
            name="Yahoo_Finance_Provider",
            provider_type=DataProviderType.YAHOO_FINANCE,
            config_manager=config_manager,
            config=config
        )

        # Yahoo Finance specific settings from unified configuration
        self.timeout = config_manager.get_int('yahoo.timeout', 30)
        self.max_retries = config_manager.get_int('yahoo.max_retries', 3)
        self.retry_delay = config_manager.get_float('yahoo.retry_delay', 1.0)

        # Set up session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=self.retry_delay,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        logger.info(f"Yahoo Finance Provider initialized with {len(self.available_symbols)} symbols (free yfinance library)")

    def _define_capabilities(self) -> DataProviderCapabilities:
        """Define Yahoo Finance provider capabilities."""
        return DataProviderCapabilities(
            supports_ohlcv=True,
            supports_tick=False,
            supports_news=False,
            supports_order_book=False,
            supports_market_depth=False,
            supports_live=True,  # Limited live data
            supports_historical=True,
            supported_symbols=self.available_symbols,
            supported_timeframes=list(self.timeframe_mapping.keys())
        )

    async def connect(self) -> bool:
        """
        Establish connection to Yahoo Finance.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.connection_status = ConnectionStatus.CONNECTING
            logger.info("Testing Yahoo Finance connection...")

            # Test connection with a simple request
            test_symbol = 'EURUSD=X'
            ticker = yf.Ticker(test_symbol)

            # Try to get some basic info
            info = ticker.info
            if info and 'symbol' in info:
                self.connection_status = ConnectionStatus.CONNECTED
                logger.info("Yahoo Finance Provider connected successfully")
                return True
            else:
                raise ConnectionError("Failed to retrieve test data from Yahoo Finance")

        except Exception as e:
            self._handle_error(e, "connection test")
            return False

    async def disconnect(self):
        """Disconnect from Yahoo Finance."""
        try:
            if self.session:
                self.session.close()
            self.connection_status = ConnectionStatus.DISCONNECTED
            logger.info("Disconnected from Yahoo Finance")
        except Exception as e:
            logger.error(f"Error during Yahoo Finance disconnection: {str(e)}")

    async def is_connected(self) -> bool:
        """
        Check if provider can reach Yahoo Finance.

        Returns:
            True if connected, False otherwise
        """
        try:
            # Simple connectivity test
            response = self.session.get('https://finance.yahoo.com', timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def _convert_symbol(self, symbol: str) -> str:
        """Convert internal symbol to Yahoo Finance format."""
        # Check forex symbols first
        if symbol in self.forex_symbols:
            return self.forex_symbols[symbol]

        # Check stock symbols
        if symbol in self.stock_symbols:
            return self.stock_symbols[symbol]

        # Default: return as-is (might work for some stocks)
        return symbol

    async def get_ohlcv_data(self,
                           symbol: str,
                           timeframe: Timeframe,
                           start_time: Optional[datetime] = None,
                           end_time: Optional[datetime] = None,
                           count: Optional[int] = None) -> Optional[pd.DataFrame]:
        """
        Get OHLCV data from Yahoo Finance.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe for the data
            start_time: Start time for historical data
            end_time: End time for historical data
            count: Number of candles to retrieve

        Returns:
            DataFrame with OHLCV data or None if not available
        """
        try:
            # Validate request
            self.validate_request(DataType.OHLCV, symbol, timeframe=timeframe)

            # Convert symbol to Yahoo format
            yahoo_symbol = self._convert_symbol(symbol)

            # Convert timeframe
            yahoo_interval = self.timeframe_mapping.get(timeframe)
            if yahoo_interval is None:
                raise DataProviderError(f"Unsupported timeframe: {timeframe}")

            # Prepare time parameters
            if count and not start_time:
                # Calculate start time based on count and timeframe
                start_time = self._calculate_start_time_from_count(count, timeframe)
                end_time = datetime.now()
            elif not start_time:
                # Default: last 1000 periods
                start_time = self._calculate_start_time_from_count(1000, timeframe)
                end_time = datetime.now()
            elif not end_time:
                end_time = datetime.now()

            # Create ticker object
            ticker = yf.Ticker(yahoo_symbol)

            # Get historical data
            hist = ticker.history(
                start=start_time,
                end=end_time,
                interval=yahoo_interval,
                auto_adjust=True,
                prepost=False
            )

            if hist.empty:
                logger.warning(f"No OHLCV data available for {symbol} from Yahoo Finance")
                self._update_request_stats(False)
                return None

            # Reset index to make datetime a column
            hist = hist.reset_index()

            # Rename columns to standard format
            hist = hist.rename(columns={
                'Date': 'timestamp',
                'Datetime': 'timestamp',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })

            # Select required columns
            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            available_cols = [col for col in required_cols if col in hist.columns]
            hist = hist[available_cols]

            # Convert price columns to Decimal
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                if col in hist.columns:
                    hist[col] = hist[col].apply(lambda x: Decimal(str(x)) if pd.notna(x) else Decimal('0'))

            # Convert volume to Decimal (handle NaN values)
            if 'volume' in hist.columns:
                hist['volume'] = hist['volume'].apply(
                    lambda x: Decimal(str(int(x))) if pd.notna(x) and x > 0 else Decimal('0')
                )
            else:
                # Add default volume column if missing
                hist['volume'] = Decimal('0')

            # Ensure timestamp is datetime
            if 'timestamp' in hist.columns:
                hist['timestamp'] = pd.to_datetime(hist['timestamp'])

            # Sort by timestamp
            hist = hist.sort_values('timestamp').reset_index(drop=True)

            # Handle 4H timeframe by resampling 1H data
            if timeframe == Timeframe.H4 and yahoo_interval == "1h":
                hist = self._resample_to_4h(hist)

            # Limit to requested count if specified
            if count and len(hist) > count:
                hist = hist.tail(count).reset_index(drop=True)

            logger.debug(f"Retrieved {len(hist)} OHLCV candles for {symbol} from Yahoo Finance")
            self._update_request_stats(True)

            return hist

        except Exception as e:
            self._handle_error(e, f"OHLCV data retrieval for {symbol}")
            return None

    async def get_tick_data(self,
                          symbol: str,
                          start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None,
                          count: Optional[int] = None) -> Optional[pd.DataFrame]:
        """
        Tick data not supported by Yahoo Finance.

        Returns:
            None (tick data not available)
        """
        logger.info(f"Tick data not available from Yahoo Finance for {symbol}")
        return None

    def _calculate_start_time_from_count(self, count: int, timeframe: Timeframe) -> datetime:
        """Calculate start time based on count and timeframe."""
        # Timeframe to timedelta mapping
        timedelta_mapping = {
            Timeframe.M1: timedelta(minutes=1),
            Timeframe.M5: timedelta(minutes=5),
            Timeframe.M15: timedelta(minutes=15),
            Timeframe.M30: timedelta(minutes=30),
            Timeframe.H1: timedelta(hours=1),
            Timeframe.H4: timedelta(hours=4),
            Timeframe.D1: timedelta(days=1),
            Timeframe.W1: timedelta(weeks=1),
            Timeframe.MN1: timedelta(days=30)
        }

        delta = timedelta_mapping.get(timeframe, timedelta(hours=1))
        return datetime.now() - (delta * count * 1.2)  # Add 20% buffer for weekends/holidays

    def _resample_to_4h(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resample 1H data to 4H timeframe."""
        try:
            df = df.set_index('timestamp')

            resampled = df.resample('4H').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()

            return resampled.reset_index()
        except Exception as e:
            logger.error(f"Failed to resample to 4H: {str(e)}")
            return df

    async def get_current_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get current price for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Current price data or None
        """
        try:
            yahoo_symbol = self._convert_symbol(symbol)
            ticker = yf.Ticker(yahoo_symbol)

            # Get current data
            data = ticker.history(period='1d', interval='1m').tail(1)
            if data.empty:
                return None

            latest = data.iloc[-1]
            return {
                'symbol': symbol,
                'timestamp': data.index[-1],
                'price': Decimal(str(latest['Close'])),
                'open': Decimal(str(latest['Open'])),
                'high': Decimal(str(latest['High'])),
                'low': Decimal(str(latest['Low'])),
                'volume': Decimal(str(int(latest['Volume']))) if latest['Volume'] > 0 else Decimal('0')
            }

        except Exception as e:
            logger.error(f"Failed to get current price for {symbol}: {str(e)}")
            return None
