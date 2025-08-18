"""
Data Providers Package for AUJ Platform
=======================================

This package contains all data providers for fetching market data, news,
and economic calendar information from various sources.
"""

from .base_provider import (
    BaseDataProvider,
    DataProviderType,
    DataType,
    Timeframe,
    ConnectionStatus,
    DataProviderCapabilities
)

from .data_provider_manager import DataProviderManager, get_data_provider_manager
from .unified_news_economic_provider import UnifiedNewsEconomicProvider
# MT5 Provider deprecated - use MetaApiProvider for Linux deployment
# from .unified_mt5_provider import UnifiedMT5Provider
from .metaapi_provider import MetaApiProvider

# Import other providers

# Legacy MT5 providers deprecated - use MetaApiProvider instead
# try:
#     from .tick_data_provider import MT5TickDataProvider
# except ImportError:
#     MT5TickDataProvider = None

# try:
#     from .ohlcv_provider import MT5OHLCVProvider
# except ImportError:
#     MT5OHLCVProvider = None

try:
    from .yahoo_finance_provider import YahooFinanceProvider
except ImportError:
    YahooFinanceProvider = None

# Import real providers instead of placeholders
from .real_order_book_provider import RealOrderBookProvider, OrderBookProvider
from .real_market_depth_provider import RealMarketDepthProvider, MarketDepthProvider

__all__ = [
    # Base classes
    'BaseDataProvider',
    'DataProviderType',
    'DataType',
    'Timeframe',
    'ConnectionStatus',
    'DataProviderCapabilities',

    # Manager
    'DataProviderManager',
    'get_data_provider_manager',

    # Unified Providers
    'UnifiedNewsEconomicProvider',
    # 'UnifiedMT5Provider',  # Deprecated - use MetaApiProvider
    'MetaApiProvider',

    # Market Data Providers
    # Legacy MT5 providers removed - use MetaApiProvider instead
    # 'MT5TickDataProvider',
    # 'MT5OHLCVProvider',
    'YahooFinanceProvider',

    # Real Providers (replacing placeholders)
    'RealOrderBookProvider',
    'OrderBookProvider',  # Alias for compatibility
    'RealMarketDepthProvider',
    'MarketDepthProvider',  # Alias for compatibility
]
