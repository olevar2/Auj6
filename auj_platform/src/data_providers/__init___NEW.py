"""
Data Providers Package for AUJ Platform
=======================================

This package contains all data providers for fetching market data, news,
and economic calendar information from various sources.

Linux-optimized with MetaApi as primary provider.
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
from .metaapi_provider import MetaApiProvider

try:
    from .yahoo_finance_provider import YahooFinanceProvider
except ImportError:
    YahooFinanceProvider = None

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
    'MetaApiProvider',

    # Market Data Providers
    'YahooFinanceProvider',

    # Real Providers
    'RealOrderBookProvider',
    'OrderBookProvider',
    'RealMarketDepthProvider',
    'MarketDepthProvider',
]
