"""
Base Data Provider Interface for the AUJ Platform.

This module defines the abstract base class that all data providers must implement,
providing a standardized interface for accessing various market data sources.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from enum import Enum
import pandas as pd
from decimal import Decimal
import logging

# Import data contracts
from ..core.data_contracts import OHLCVData, TickData, NewsEvent

# Define minimal exceptions to avoid circular imports
class DataProviderError(Exception):
    """Data provider related error."""
    pass

class DataNotAvailableError(Exception):
    """Data not available error."""
    pass

class ConnectionError(Exception):
    """Connection related error."""
    pass

# Use basic logging to avoid circular imports
logger = logging.getLogger(__name__)


class DataProviderType(str, Enum):
    """Types of data providers."""
    MT5 = "MT5"
    YAHOO_FINANCE = "YAHOO_FINANCE"
    NEWS = "NEWS"
    ECONOMIC_CALENDAR = "ECONOMIC_CALENDAR"
    NEWS_AND_ECONOMIC = "NEWS_AND_ECONOMIC"
    ORDER_BOOK = "ORDER_BOOK"
    PLACEHOLDER = "PLACEHOLDER"


class DataType(str, Enum):
    """Types of data that can be provided."""
    OHLCV = "OHLCV"
    TICK = "TICK"
    NEWS = "NEWS"
    ECONOMIC_CALENDAR = "ECONOMIC_CALENDAR"
    ORDER_BOOK = "ORDER_BOOK"
    MARKET_DEPTH = "MARKET_DEPTH"
    VOLUME_PROFILE = "VOLUME_PROFILE"


class Timeframe(str, Enum):
    """Supported timeframes for OHLCV data."""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"
    MN1 = "1M"


class ConnectionStatus(str, Enum):
    """Connection status of data provider."""
    CONNECTED = "CONNECTED"
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    FAILED = "FAILED"
    ERROR = "ERROR"


class DataProviderCapabilities:
    """Defines what data types a provider can supply."""

    def __init__(self,
                 supports_ohlcv: bool = False,
                 supports_tick: bool = False,
                 supports_news: bool = False,
                 supports_economic_calendar: bool = False,
                 supports_order_book: bool = False,
                 supports_market_depth: bool = False,
                 supports_live: bool = False,
                 supports_historical: bool = False,
                 supported_symbols: Optional[List[str]] = None,
                 supported_timeframes: Optional[List[Timeframe]] = None):
        """
        Initialize provider capabilities.

        Args:
            supports_ohlcv: Can provide OHLCV candlestick data
            supports_tick: Can provide tick-by-tick data
            supports_news: Can provide news data
            supports_economic_calendar: Can provide economic calendar data
            supports_order_book: Can provide order book data
            supports_market_depth: Can provide market depth data
            supports_live: Can provide real-time data
            supports_historical: Can provide historical data
            supported_symbols: List of supported trading symbols
            supported_timeframes: List of supported timeframes
        """
        self.supports_ohlcv = supports_ohlcv
        self.supports_tick = supports_tick
        self.supports_news = supports_news
        self.supports_economic_calendar = supports_economic_calendar
        self.supports_order_book = supports_order_book
        self.supports_market_depth = supports_market_depth
        self.supports_live = supports_live
        self.supports_historical = supports_historical
        self.supported_symbols = supported_symbols or []
        self.supported_timeframes = supported_timeframes or []

    def can_provide(self, data_type: DataType) -> bool:
        """Check if provider can supply a specific data type."""
        capability_map = {
            DataType.OHLCV: self.supports_ohlcv,
            DataType.TICK: self.supports_tick,
            DataType.NEWS: self.supports_news,
            DataType.ECONOMIC_CALENDAR: self.supports_economic_calendar,
            DataType.ORDER_BOOK: self.supports_order_book,
            DataType.MARKET_DEPTH: self.supports_market_depth,
            DataType.VOLUME_PROFILE: self.supports_ohlcv  # Derived from OHLCV
        }
        return capability_map.get(data_type, False)

    def supports_symbol(self, symbol: str) -> bool:
        """Check if provider supports a specific symbol."""
        if not self.supported_symbols:
            return True  # No restrictions
        return symbol in self.supported_symbols

    def supports_timeframe(self, timeframe: Timeframe) -> bool:
        """Check if provider supports a specific timeframe."""
        if not self.supported_timeframes:
            return True  # No restrictions
        return timeframe in self.supported_timeframes


class BaseDataProvider(ABC):
    """
    Abstract base class for all data providers in the AUJ Platform.

    This class defines the contract that all data providers must implement,
    ensuring consistent behavior across different data sources.
    """

    def __init__(self, name: str, provider_type: DataProviderType, config_manager=None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the data provider.

        Args:
            name: Human-readable name of the provider
            provider_type: Type of provider (MT5, Yahoo Finance, etc.)
            config_manager: Configuration manager instance (optional)
            config: Provider-specific configuration (deprecated, use config_manager)
        """
        self.name = name
        self.provider_type = provider_type
        self.config_manager = config_manager
        # Backward compatibility - will be removed in next version
        self.config = config or {}
        self.connection_status = ConnectionStatus.DISCONNECTED
        self.last_error: Optional[str] = None
        self.request_count = 0
        self.successful_requests = 0
        self.last_request_time: Optional[datetime] = None

        # Initialize capabilities
        self.capabilities = self._define_capabilities()

        logger.info(f"Initialized data provider: {name} ({provider_type})")

    @abstractmethod
    def _define_capabilities(self) -> DataProviderCapabilities:
        """
        Define what data types this provider can supply.
        Must be implemented by each provider.
        """
        pass

    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish connection to the data source.

        Returns:
            True if connection successful, False otherwise
        """
        pass

    @abstractmethod
    async def disconnect(self):
        """Disconnect from the data source."""
        pass

    @abstractmethod
    async def is_connected(self) -> bool:
        """
        Check if provider is currently connected.

        Returns:
            True if connected, False otherwise
        """
        pass

    @abstractmethod
    async def get_ohlcv_data(self,
                           symbol: str,
                           timeframe: Timeframe,
                           start_time: Optional[datetime] = None,
                           end_time: Optional[datetime] = None,
                           count: Optional[int] = None) -> Optional[pd.DataFrame]:
        """
        Get OHLCV candlestick data.

        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            timeframe: Timeframe for the data
            start_time: Start time for historical data
            end_time: End time for historical data
            count: Number of candles to retrieve (alternative to time range)

        Returns:
            DataFrame with OHLC data or None if not available
        """
        pass

    @abstractmethod
    async def get_tick_data(self,
                          symbol: str,
                          start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None,
                          count: Optional[int] = None) -> Optional[pd.DataFrame]:
        """
        Get tick-by-tick data.

        Args:
            symbol: Trading symbol
            start_time: Start time for historical data
            end_time: End time for historical data
            count: Number of ticks to retrieve

        Returns:
            DataFrame with tick data or None if not available
        """
        pass

    async def get_news_data(self,
                          symbols: Optional[List[str]] = None,
                          start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None,
                          count: Optional[int] = None) -> Optional[List[NewsEvent]]:
        """
        Get news data (placeholder implementation for providers that don't support news).

        Args:
            symbols: List of symbols to get news for
            start_time: Start time for news
            end_time: End time for news
            count: Number of news items to retrieve

        Returns:
            List of news events or None if not available
        """
        if not self.capabilities.supports_news:
            logger.warning(f"Provider {self.name} does not support news data")
            return None

        # Default implementation returns None - override in subclasses
        return None

    async def get_order_book_data(self, symbol: str, depth: int = 10) -> Optional[Dict[str, Any]]:
        """
        Get order book data (placeholder implementation).

        Args:
            symbol: Trading symbol
            depth: Depth of order book

        Returns:
            Order book data or None if not available
        """
        if not self.capabilities.supports_order_book:
            logger.warning(f"Provider {self.name} does not support order book data")
            return None

        # Default implementation returns None - override in subclasses
        return None

    async def get_market_depth_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get market depth data (placeholder implementation).

        Args:
            symbol: Trading symbol

        Returns:
            Market depth data or None if not available
        """
        if not self.capabilities.supports_market_depth:
            logger.warning(f"Provider {self.name} does not support market depth data")
            return None

        # Default implementation returns None - override in subclasses
        return None

    def validate_request(self, data_type: DataType, symbol: str, **kwargs) -> bool:
        """
        Validate if this provider can handle a data request.

        Args:
            data_type: Type of data being requested
            symbol: Trading symbol
            **kwargs: Additional parameters

        Returns:
            True if request can be handled, False otherwise

        Raises:
            DataProviderError: If validation fails with specific reason
        """
        # Check if provider supports this data type
        if not self.capabilities.can_provide(data_type):
            raise DataProviderError(
                f"Provider {self.name} does not support {data_type.value} data",
                error_code="UNSUPPORTED_DATA_TYPE",
                context={"provider": self.name, "data_type": data_type.value}
            )

        # Check if provider supports this symbol
        if not self.capabilities.supports_symbol(symbol):
            raise DataProviderError(
                f"Provider {self.name} does not support symbol {symbol}",
                error_code="UNSUPPORTED_SYMBOL",
                context={"provider": self.name, "symbol": symbol}
            )

        # Check timeframe if provided
        timeframe = kwargs.get('timeframe')
        if timeframe and not self.capabilities.supports_timeframe(timeframe):
            raise DataProviderError(
                f"Provider {self.name} does not support timeframe {timeframe}",
                error_code="UNSUPPORTED_TIMEFRAME",
                context={"provider": self.name, "timeframe": timeframe.value}
            )

        return True

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the provider.

        Returns:
            Health status information
        """
        try:
            is_connected = await self.is_connected()

            return {
                "provider_name": self.name,
                "provider_type": self.provider_type.value,
                "connection_status": self.connection_status.value,
                "is_connected": is_connected,
                "last_error": self.last_error,
                "request_count": self.request_count,
                "successful_requests": self.successful_requests,
                "success_rate": self.successful_requests / max(self.request_count, 1),
                "last_request_time": self.last_request_time.isoformat() if self.last_request_time else None,
                "capabilities": {
                    "supports_ohlcv": self.capabilities.supports_ohlcv,
                    "supports_tick": self.capabilities.supports_tick,
                    "supports_news": self.capabilities.supports_news,
                    "supports_order_book": self.capabilities.supports_order_book,
                    "supports_market_depth": self.capabilities.supports_market_depth,
                    "supports_live": self.capabilities.supports_live,
                    "supports_historical": self.capabilities.supports_historical,
                    "supported_symbols_count": len(self.capabilities.supported_symbols),
                    "supported_timeframes_count": len(self.capabilities.supported_timeframes)
                }
            }
        except Exception as e:
            logger.error(f"Health check failed for provider {self.name}: {str(e)}")
            return {
                "provider_name": self.name,
                "provider_type": self.provider_type.value,
                "connection_status": ConnectionStatus.ERROR.value,
                "is_connected": False,
                "error": str(e)
            }

    def _update_request_stats(self, success: bool):
        """Update request statistics."""
        self.request_count += 1
        if success:
            self.successful_requests += 1
        self.last_request_time = datetime.utcnow()

    def _handle_error(self, error: Exception, operation: str) -> None:
        """
        Handle and log errors consistently.

        Args:
            error: The exception that occurred
            operation: Description of the operation that failed
        """
        error_msg = f"Provider {self.name} failed during {operation}: {str(error)}"
        self.last_error = error_msg
        self.connection_status = ConnectionStatus.ERROR
        logger.error(error_msg)

        # Update request stats
        self._update_request_stats(False)

    def get_capabilities(self) -> DataProviderCapabilities:
        """Get provider capabilities."""
        return self.capabilities

    def get_supported_symbols(self) -> List[str]:
        """Get list of supported symbols."""
        return self.capabilities.supported_symbols.copy()

    def get_supported_timeframes(self) -> List[Timeframe]:
        """Get list of supported timeframes."""
        return self.capabilities.supported_timeframes.copy()

    def __str__(self) -> str:
        return f"{self.name} ({self.provider_type.value}) - {self.connection_status.value}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', type='{self.provider_type.value}')"
