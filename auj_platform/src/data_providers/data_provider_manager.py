"""
Data Provider Manager for AUJ Platform - Linux Optimized

Central manager optimized for Linux deployment using MetaApi as primary provider.
Removes dependencies on Windows-only MT5 library and provides comprehensive
data management for cross-platform trading operations.
"""

import asyncio
import platform
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
import pandas as pd
from enum import Enum
import logging

from .base_provider import BaseDataProvider, DataType, Timeframe, ConnectionStatus
from .metaapi_provider import MetaApiProvider
from .yahoo_finance_provider import YahooFinanceProvider
from .unified_news_economic_provider import UnifiedNewsEconomicProvider
from .real_order_book_provider import RealOrderBookProvider
from .real_market_depth_provider import RealMarketDepthProvider

# Import core components
try:
    from ..core.unified_config import UnifiedConfigManager, get_unified_config
except ImportError:
    UnifiedConfigManager = None
    get_unified_config = None

try:
    from ..core.platform_detection import PlatformDetection
except ImportError:
    PlatformDetection = None

try:
    from ..core.data_contracts import NewsEvent, EconomicCalendar
except ImportError:
    NewsEvent = dict
    EconomicCalendar = dict

# Define exceptions
class DataProviderError(Exception):
    """Data provider related error."""
    pass

class DataNotAvailableError(Exception):
    """Data not available error."""
    pass

class ConfigurationError(Exception):
    """Configuration related error."""
    pass

logger = logging.getLogger(__name__)


class ProviderPriority(Enum):
    """Provider priority levels for Linux deployment."""
    PRIMARY = "primary"
    SECONDARY = "secondary"
    FALLBACK = "fallback"
    DISABLED = "disabled"


class DataProviderManager:
    """
    Linux-optimized Data Provider Manager using MetaApi.
    
    Completely removes Windows-only MT5 dependencies and uses
    MetaApi as the primary source for all trading data on Linux.
    """

    def __init__(self, config_manager=None, config: Optional[Dict[str, Any]] = None):
        """Initialize the data provider manager for Linux deployment."""
        try:
            self.config_manager = config_manager or (get_unified_config() if get_unified_config else None)
            self.config = config or (self.config_manager.get_dict('data_providers', {}) if self.config_manager else {})
        except Exception:
            self.config_manager = config_manager
            self.config = config or {}
        
        # Provider management
        self.providers: Dict[str, BaseDataProvider] = {}
        self.provider_priorities: Dict[str, ProviderPriority] = {}
        self.provider_specializations: Dict[DataType, List[str]] = {}
        self.health_status: Dict[str, Dict[str, Any]] = {}
        
        # Linux deployment flag
        self.is_linux = platform.system().lower() == 'linux'
        self.deployment_mode = "linux_optimized" if self.is_linux else "cross_platform"
        
        # Health monitoring
        self.last_health_check = datetime.min
        self.health_check_interval = timedelta(minutes=2)  # More frequent on Linux
        
        # Live data streaming
        self.streaming_active = False
        self.streaming_symbols: List[str] = []
        self.streaming_task: Optional[asyncio.Task] = None
        self.streaming_interval = 0.5  # Faster updates for Linux
        
        # Initialization state
        self._initialized = False
        
        # Platform detection
        if PlatformDetection:
            self.platform_info = PlatformDetection.get_system_info()
            self.recommended_config = PlatformDetection.get_recommended_broker_config(self.config)
        else:
            self.platform_info = {"platform": platform.system().lower()}
            self.recommended_config = {}
        
        logger.info(f"Data Provider Manager initialized for {self.deployment_mode} deployment")
        logger.info(f"Platform: {self.platform_info.get('platform', 'unknown')}")

    async def initialize(self) -> None:
        """Initialize the component with Linux-optimized providers."""
        if self._initialized:
            return
            
        logger.info("🚀 Initializing Linux Data Provider Manager...")
        await self._initialize_linux_providers()
        await self._setup_provider_specializations()
        self._initialized = True
        logger.info("✅ Data Provider Manager initialization completed for Linux")

    async def _initialize_linux_providers(self):
        """Initialize Linux-compatible providers only."""
        try:
            # 1. Initialize MetaApi Provider (Primary for all trading data)
            metaapi_enabled = True
            if self.config_manager and hasattr(self.config_manager, 'get_bool'):
                metaapi_enabled = self.config_manager.get_bool('metaapi.enabled', True)
            elif self.config:
                metaapi_enabled = self.config.get('metaapi', {}).get('enabled', True)
                
            if metaapi_enabled:
                try:
                    metaapi_provider = MetaApiProvider(self.config_manager, self.config)
                    self.providers['metaapi'] = metaapi_provider
                    self.provider_priorities['metaapi'] = ProviderPriority.PRIMARY
                    
                    if await metaapi_provider.connect():
                        logger.info("✅ MetaApi Provider connected successfully")
                        
                        # Get symbols for capability update
                        symbols = await metaapi_provider.get_symbols()
                        logger.info(f"   📊 Available symbols: {len(symbols)}")
                    else:
                        logger.warning("⚠️ MetaApi Provider failed to connect")
                        self.provider_priorities['metaapi'] = ProviderPriority.DISABLED
                        
                except Exception as e:
                    logger.error(f"❌ Failed to initialize MetaApi Provider: {e}")
                    self.provider_priorities['metaapi'] = ProviderPriority.DISABLED

            # 2. Initialize Yahoo Finance Provider (Fallback for OHLCV)
            yahoo_enabled = True
            if self.config_manager and hasattr(self.config_manager, 'get_bool'):
                yahoo_enabled = self.config_manager.get_bool('yahoo.enabled', True)
            elif self.config:
                yahoo_enabled = self.config.get('yahoo', {}).get('enabled', True)
                
            if yahoo_enabled:
                try:
                    yahoo_provider = YahooFinanceProvider(self.config_manager, self.config)
                    self.providers['yahoo_finance'] = yahoo_provider
                    self.provider_priorities['yahoo_finance'] = ProviderPriority.FALLBACK

                    if await yahoo_provider.connect():
                        logger.info("✅ Yahoo Finance Provider connected successfully")
                    else:
                        logger.warning("⚠️ Yahoo Finance Provider failed to connect")
                        self.provider_priorities['yahoo_finance'] = ProviderPriority.DISABLED
                        
                except Exception as e:
                    logger.warning(f"⚠️ Failed to initialize Yahoo Finance Provider: {e}")
                    self.provider_priorities['yahoo_finance'] = ProviderPriority.DISABLED

            # 3. Initialize News & Economic Provider
            news_enabled = True
            if self.config_manager and hasattr(self.config_manager, 'get_bool'):
                news_enabled = self.config_manager.get_bool('news.enabled', True)
            elif self.config:
                news_enabled = self.config.get('news', {}).get('enabled', True)

            if news_enabled:
                try:
                    news_provider = UnifiedNewsEconomicProvider(self.config_manager, self.config)
                    self.providers['news_economic'] = news_provider
                    self.provider_priorities['news_economic'] = ProviderPriority.PRIMARY

                    if await news_provider.connect():
                        logger.info("✅ News & Economic Provider connected successfully")
                    else:
                        logger.warning("⚠️ News & Economic Provider failed to connect")
                        self.provider_priorities['news_economic'] = ProviderPriority.DISABLED
                        
                except Exception as e:
                    logger.warning(f"⚠️ Failed to initialize News & Economic Provider: {e}")
                    self.provider_priorities['news_economic'] = ProviderPriority.DISABLED

            # 4. Initialize Order Book Provider (MetaApi-based)
            try:
                order_book_provider = RealOrderBookProvider(self.config_manager, self.config)
                self.providers['order_book'] = order_book_provider
                self.provider_priorities['order_book'] = ProviderPriority.SECONDARY
                if await order_book_provider.connect():
                    logger.info("✅ Order Book Provider initialized")
                else:
                    self.provider_priorities['order_book'] = ProviderPriority.DISABLED
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize Order Book Provider: {e}")

            # 5. Initialize Market Depth Provider (MetaApi-based)
            try:
                market_depth_provider = RealMarketDepthProvider(self.config_manager, self.config)
                self.providers['market_depth'] = market_depth_provider
                self.provider_priorities['market_depth'] = ProviderPriority.SECONDARY
                if await market_depth_provider.connect():
                    logger.info("✅ Market Depth Provider initialized")
                else:
                    self.provider_priorities['market_depth'] = ProviderPriority.DISABLED
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize Market Depth Provider: {e}")

            # Log final provider status
            connected_count = sum(1 for name, provider in self.providers.items() 
                                if self.provider_priorities.get(name) != ProviderPriority.DISABLED)
            
            logger.info(f"📊 Initialized {len(self.providers)} providers, {connected_count} active")
            
            # Log provider status
            for name, provider in self.providers.items():
                priority = self.provider_priorities.get(name, ProviderPriority.DISABLED)
                status = "🟢 Active" if priority != ProviderPriority.DISABLED else "🔴 Disabled"
                logger.info(f"  📡 {name}: {status} ({priority.value})")

        except Exception as e:
            logger.error(f"❌ Failed to initialize Linux providers: {str(e)}")
            raise DataProviderError(f"Provider initialization failed: {str(e)}")

    async def _setup_provider_specializations(self):
        """Define provider specializations for Linux deployment."""
        self.provider_specializations = {
            DataType.OHLCV: ['metaapi', 'yahoo_finance'],
            DataType.TICK: ['metaapi'],
            DataType.NEWS: ['news_economic'],
            DataType.ECONOMIC_CALENDAR: ['news_economic'],
            DataType.ORDER_BOOK: ['order_book', 'metaapi'],
            DataType.MARKET_DEPTH: ['market_depth', 'metaapi']
        }
        
        logger.debug(f"Provider specializations configured for {len(self.provider_specializations)} data types")

    def _get_providers_for_data_type(self, data_type: DataType) -> List[str]:
        """Get ordered list of providers for a specific data type."""
        provider_names = self.provider_specializations.get(data_type, [])
        
        # Filter by availability and priority
        available_providers = []
        for name in provider_names:
            if (name in self.providers and 
                self.provider_priorities.get(name) != ProviderPriority.DISABLED):
                available_providers.append(name)
        
        # Sort by priority (PRIMARY first, then SECONDARY, then FALLBACK)
        priority_order = {
            ProviderPriority.PRIMARY: 0,
            ProviderPriority.SECONDARY: 1,
            ProviderPriority.FALLBACK: 2
        }
        
        available_providers.sort(
            key=lambda name: priority_order.get(
                self.provider_priorities.get(name, ProviderPriority.FALLBACK), 
                99
            )
        )
        
        return available_providers

    # Data Retrieval Methods

    async def get_ohlcv_data(self,
                           symbol: str,
                           timeframe: Timeframe,
                           start_time: Optional[datetime] = None,
                           end_time: Optional[datetime] = None,
                           count: Optional[int] = None) -> Optional[pd.DataFrame]:
        """Get OHLCV data with Linux-optimized provider priority."""
        providers_to_try = self._get_providers_for_data_type(DataType.OHLCV)
        
        for provider_name in providers_to_try:
            if provider_name in self.providers:
                provider = self.providers[provider_name]
                try:
                    if await provider.is_connected():
                        data = await provider.get_ohlcv_data(
                            symbol, timeframe, start_time, end_time, count
                        )
                        if data is not None and not data.empty:
                            logger.debug(f"✅ Got OHLCV data from {provider_name}: {len(data)} candles")
                            await self._update_provider_success(provider_name)
                            return data
                    else:
                        logger.debug(f"⚠️ {provider_name} not connected, trying next provider")
                except Exception as e:
                    logger.warning(f"⚠️ {provider_name} failed for OHLCV {symbol}: {e}")
                    await self._update_provider_failure(provider_name)
                    continue
        
        logger.error(f"❌ No provider could supply OHLCV data for {symbol}")
        raise DataNotAvailableError(f"OHLCV data not available for {symbol}")

    async def get_tick_data(self,
                          symbol: str,
                          start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None,
                          count: Optional[int] = None) -> Optional[pd.DataFrame]:
        """Get tick data from MetaApi (primary source on Linux)."""
        providers_to_try = self._get_providers_for_data_type(DataType.TICK)
        
        for provider_name in providers_to_try:
            if provider_name in self.providers:
                provider = self.providers[provider_name]
                try:
                    if await provider.is_connected():
                        data = await provider.get_tick_data(symbol, start_time, end_time, count)
                        if data is not None and not data.empty:
                            logger.debug(f"✅ Got tick data from {provider_name}: {len(data)} ticks")
                            await self._update_provider_success(provider_name)
                            return data
                except Exception as e:
                    logger.warning(f"⚠️ {provider_name} failed for tick data {symbol}: {e}")
                    await self._update_provider_failure(provider_name)
                    continue
        
        logger.error(f"❌ No provider could supply tick data for {symbol}")
        raise DataNotAvailableError(f"Tick data not available for {symbol}")

    async def get_news_data(self,
                          symbols: Optional[List[str]] = None,
                          start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None,
                          count: Optional[int] = None) -> Optional[List]:
        """Get news data from unified news provider."""
        providers_to_try = self._get_providers_for_data_type(DataType.NEWS)
        
        for provider_name in providers_to_try:
            if provider_name in self.providers:
                provider = self.providers[provider_name]
                try:
                    if await provider.is_connected():
                        data = await provider.get_news_data(symbols, start_time, end_time, count)
                        if data is not None:
                            logger.debug(f"✅ Got news data from {provider_name}: {len(data)} items")
                            await self._update_provider_success(provider_name)
                            return data
                except Exception as e:
                    logger.warning(f"⚠️ {provider_name} failed for news data: {e}")
                    await self._update_provider_failure(provider_name)
                    continue
        
        logger.warning("⚠️ No news provider available")
        return []

    async def get_economic_calendar_data(self,
                                       start_time: Optional[datetime] = None,
                                       end_time: Optional[datetime] = None,
                                       importance: Optional[str] = None) -> Optional[List]:
        """Get economic calendar data."""
        providers_to_try = self._get_providers_for_data_type(DataType.ECONOMIC_CALENDAR)
        
        for provider_name in providers_to_try:
            if provider_name in self.providers:
                provider = self.providers[provider_name]
                try:
                    if await provider.is_connected() and hasattr(provider, 'get_economic_calendar_data'):
                        data = await provider.get_economic_calendar_data(start_time, end_time, importance)
                        if data is not None:
                            logger.debug(f"✅ Got economic calendar data from {provider_name}: {len(data)} events")
                            await self._update_provider_success(provider_name)
                            return data
                except Exception as e:
                    logger.warning(f"⚠️ {provider_name} failed for economic calendar: {e}")
                    await self._update_provider_failure(provider_name)
                    continue
        
        logger.warning("⚠️ No economic calendar provider available")
        return []

    # Provider Management

    async def get_provider_by_name(self, name: str) -> Optional[BaseDataProvider]:
        """Get provider by name."""
        return self.providers.get(name)

    async def get_metaapi_provider(self) -> Optional[MetaApiProvider]:
        """Get MetaApi provider for trading operations."""
        return self.providers.get('metaapi')

    async def get_primary_trading_provider(self) -> Optional[BaseDataProvider]:
        """Get the primary trading provider (MetaApi on Linux)."""
        return await self.get_metaapi_provider()

    async def _update_provider_success(self, provider_name: str):
        """Update provider success statistics."""
        if provider_name not in self.health_status:
            self.health_status[provider_name] = {"success_count": 0, "failure_count": 0}
        
        self.health_status[provider_name]["success_count"] += 1
        self.health_status[provider_name]["last_success"] = datetime.now()

    async def _update_provider_failure(self, provider_name: str):
        """Update provider failure statistics."""
        if provider_name not in self.health_status:
            self.health_status[provider_name] = {"success_count": 0, "failure_count": 0}
        
        self.health_status[provider_name]["failure_count"] += 1
        self.health_status[provider_name]["last_failure"] = datetime.now()

    # Health and Monitoring

    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check on all providers."""
        if datetime.now() - self.last_health_check < self.health_check_interval:
            # Return cached results if recent check
            return self._get_cached_health_status()
        
        health_results = {}
        
        for name, provider in self.providers.items():
            try:
                health = await provider.health_check()
                health_results[name] = health
                
                # Update our tracking
                priority = self.provider_priorities.get(name, ProviderPriority.DISABLED)
                health_results[name]["priority"] = priority.value
                health_results[name]["statistics"] = self.health_status.get(name, {})
                
            except Exception as e:
                health_results[name] = {
                    "provider_name": name,
                    "status": "ERROR",
                    "error": str(e),
                    "priority": self.provider_priorities.get(name, ProviderPriority.DISABLED).value
                }
        
        # Overall health assessment
        active_providers = [result for result in health_results.values() 
                           if result.get("is_connected", False)]
        connected_count = len(active_providers)
        
        overall_health = {
            "overall_status": "HEALTHY" if connected_count > 0 else "CRITICAL",
            "deployment_mode": self.deployment_mode,
            "platform": self.platform_info.get("platform", "unknown"),
            "providers_connected": connected_count,
            "total_providers": len(self.providers),
            "primary_provider": "metaapi" if "metaapi" in self.providers else "none",
            "metaapi_status": health_results.get("metaapi", {}).get("is_connected", False),
            "timestamp": datetime.now().isoformat(),
            "providers": health_results
        }
        
        self.last_health_check = datetime.now()
        return overall_health

    def _get_cached_health_status(self) -> Dict[str, Any]:
        """Get cached health status."""
        return {
            "overall_status": "CACHED",
            "deployment_mode": self.deployment_mode,
            "platform": self.platform_info.get("platform", "unknown"),
            "providers_connected": len([p for p in self.providers.values() 
                                      if self.provider_priorities.get(p.name, ProviderPriority.DISABLED) != ProviderPriority.DISABLED]),
            "total_providers": len(self.providers),
            "last_check": self.last_health_check.isoformat(),
            "cache_note": "Use fresh health_check() for real-time status"
        }

    # Live Data Streaming

    async def start_live_data_stream(self, symbols: List[str], callback=None):
        """Start live data streaming from MetaApi."""
        if not symbols:
            logger.warning("No symbols provided for live streaming")
            return
            
        metaapi_provider = await self.get_metaapi_provider()
        if not metaapi_provider:
            logger.error("❌ MetaApi provider not available for live streaming")
            return
            
        try:
            self.streaming_symbols = symbols
            self.streaming_active = True
            
            # Subscribe to real-time data for all symbols
            for symbol in symbols:
                await metaapi_provider.subscribe_to_ticks(symbol, callback)
                await metaapi_provider.subscribe_to_prices(symbol, callback)
            
            logger.info(f"✅ Started live data stream for {len(symbols)} symbols")
            
        except Exception as e:
            logger.error(f"❌ Failed to start live data stream: {e}")
            self.streaming_active = False

    async def stop_live_data_stream(self, symbols: Optional[List[str]] = None, callback=None):
        """Stop live data streaming."""
        metaapi_provider = await self.get_metaapi_provider()
        if not metaapi_provider:
            return
            
        symbols_to_stop = symbols or self.streaming_symbols
        
        try:
            for symbol in symbols_to_stop:
                await metaapi_provider.unsubscribe_from_ticks(symbol, callback)
                await metaapi_provider.unsubscribe_from_prices(symbol, callback)
            
            if not symbols:  # Stop all streaming
                self.streaming_active = False
                self.streaming_symbols = []
            
            logger.info(f"✅ Stopped live data stream for {len(symbols_to_stop)} symbols")
            
        except Exception as e:
            logger.error(f"❌ Failed to stop live data stream: {e}")

    async def cleanup(self):
        """Cleanup all providers."""
        logger.info("🧹 Cleaning up Data Provider Manager...")
        
        # Stop streaming if active
        if self.streaming_active:
            await self.stop_live_data_stream()
        
        # Disconnect all providers
        for name, provider in self.providers.items():
            try:
                await provider.disconnect()
                logger.info(f"✅ {name} disconnected")
            except Exception as e:
                logger.error(f"❌ Error disconnecting {name}: {e}")
        
        # Clear all data
        self.providers.clear()
        self.provider_priorities.clear()
        self.provider_specializations.clear()
        self.health_status.clear()
        
        logger.info("✅ Data Provider Manager cleanup completed")

    def get_deployment_info(self) -> Dict[str, Any]:
        """Get deployment information."""
        return {
            "deployment_mode": self.deployment_mode,
            "platform_info": self.platform_info,
            "is_linux": self.is_linux,
            "provider_count": len(self.providers),
            "primary_provider": "metaapi",
            "supports_trading": "metaapi" in self.providers,
            "supports_real_time": self.streaming_active,
            "initialization_status": self._initialized,
            "recommended_config": self.recommended_config
        }


# Factory function for getting the manager
def get_data_provider_manager(config_manager=None) -> DataProviderManager:
    """Get the data provider manager instance optimized for Linux."""
    return DataProviderManager(config_manager=config_manager)