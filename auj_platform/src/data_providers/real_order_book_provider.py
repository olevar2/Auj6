"""
Real Order Book Provider for the AUJ Platform.

This provider uses MetaAPI for order book data, providing modern
cloud-based market depth functionality.
"""

from datetime import datetime
from typing import Any, Dict, Optional

from ..core.logging_setup import get_logger
from ..core.unified_config import UnifiedConfigManager
from .base_provider import (
    BaseDataProvider,
    ConnectionStatus,
    DataProviderCapabilities,
    DataProviderType,
)
from .metaapi_provider import MetaApiProvider

logger = get_logger(__name__)


class RealOrderBookProvider(BaseDataProvider):
    """
    Real order book provider that uses MetaAPI for market depth data.

    This provider uses MetaAPI cloud services for order book functionality,
    providing modern, scalable market depth analysis.
    """

    def __init__(self, config_manager: Optional[UnifiedConfigManager] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the real order book provider.

        Args:
            config_manager: Configuration manager instance
            config: Additional configuration parameters
        """
        # Initialize MetaAPI provider first, before calling super().__init__
        self.metaapi_provider = MetaApiProvider(config_manager=config_manager, config=config)
        
        super().__init__(
            name="OrderBook_Provider_Real",
            provider_type=DataProviderType.ORDER_BOOK,
            config_manager=config_manager,
            config=config
        )

        logger.info("Real Order Book Provider initialized - using MetaAPI cloud services")

    def _define_capabilities(self) -> DataProviderCapabilities:
        """Define real order book capabilities."""
        metaapi_caps = getattr(self.metaapi_provider, 'capabilities', None)
        supported_symbols = metaapi_caps.supported_symbols if metaapi_caps else []
        
        return DataProviderCapabilities(
            supports_ohlcv=False,
            supports_tick=False,
            supports_news=False,
            supports_order_book=True,
            supports_market_depth=True,
            supports_live=True,
            supports_historical=False,
            supported_symbols=supported_symbols,
            supported_timeframes=[]  # Order book doesn't use timeframes
        )

    async def connect(self) -> bool:
        """Connect to MetaAPI for order book data."""
        try:
            success = await self.metaapi_provider.connect()
            if success:
                self.connection_status = ConnectionStatus.CONNECTED
                logger.info("Order book provider connected to MetaAPI")
                return True
            else:
                self.connection_status = ConnectionStatus.FAILED
                logger.error("Failed to connect order book provider to MetaAPI")
                return False
        except Exception as e:
            self.connection_status = ConnectionStatus.FAILED
            logger.error(f"Order book provider connection error: {e}")
            return False

    async def disconnect(self) -> bool:
        """Disconnect from MetaAPI."""
        try:
            success = await self.metaapi_provider.disconnect()
            if success:
                self.connection_status = ConnectionStatus.DISCONNECTED
                logger.info("Order book provider disconnected from MetaAPI")
                return True
            else:
                logger.warning("Order book provider disconnect had issues")
                return False
        except Exception as e:
            logger.error(f"Order book provider disconnect error: {e}")
            return False

    async def get_order_book_data(self, symbol: str, depth: int = 10) -> Optional[Dict[str, Any]]:
        """
        Get real order book data for a symbol using MetaAPI.

        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            depth: Number of levels to retrieve (default: 10)

        Returns:
            Dictionary containing order book data with bids, asks, spread, etc.
            None if data is not available
        """
        try:
            if self.connection_status != ConnectionStatus.CONNECTED:
                logger.warning("Order book provider not connected, attempting to connect...")
                if not await self.connect():
                    logger.error("Failed to establish connection for order book data")
                    return None

            # Get order book data from MetaAPI
            # Note: MetaAPI might have different method names - this is a conceptual implementation
            order_book = await self.metaapi_provider.get_market_depth(symbol, depth)

            if order_book is None:
                logger.warning(f"No order book data available for {symbol} via MetaAPI")
                return None

            # Add metadata
            order_book.update({
                'provider': self.name,
                'data_source': 'MetaAPI',
                'depth_requested': depth,
                'depth_received': len(order_book.get('bids', [])) + len(order_book.get('asks', [])),
                'timestamp': datetime.now().isoformat()
            })

            logger.debug(f"Retrieved order book for {symbol} with {depth} levels via MetaAPI")
            return order_book

        except Exception as e:
            logger.error(f"Failed to get order book data for {symbol} via MetaAPI: {e}")
            return None

    async def get_market_depth_analysis(self, symbol: str, depth: int = 20) -> Optional[Dict[str, Any]]:
        """
        Get enhanced market depth analysis with liquidity metrics.

        Args:
            symbol: Trading symbol
            depth: Number of levels to analyze

        Returns:
            Dictionary with market depth analysis including liquidity metrics
        """
        try:
            order_book = await self.get_order_book_data(symbol, depth)
            if not order_book:
                return None

            bids = order_book.get('bids', [])
            asks = order_book.get('asks', [])

            if not bids or not asks:
                logger.warning(f"Incomplete order book data for {symbol}")
                return order_book

            # Calculate liquidity metrics
            bid_volume = sum(level['volume'] for level in bids)
            ask_volume = sum(level['volume'] for level in asks)
            total_volume = bid_volume + ask_volume

            # Calculate weighted average prices
            if bid_volume > 0:
                bid_avg_price = sum(level['price'] * level['volume'] for level in bids) / bid_volume
            else:
                bid_avg_price = bids[0]['price'] if bids else 0

            if ask_volume > 0:
                ask_avg_price = sum(level['price'] * level['volume'] for level in asks) / ask_volume
            else:
                ask_avg_price = asks[0]['price'] if asks else 0

            # Calculate order book imbalance
            imbalance = (bid_volume - ask_volume) / total_volume if total_volume > 0 else 0

            # Add analysis to order book data
            order_book.update({
                'analysis': {
                    'bid_volume': bid_volume,
                    'ask_volume': ask_volume,
                    'total_volume': total_volume,
                    'bid_avg_price': bid_avg_price,
                    'ask_avg_price': ask_avg_price,
                    'imbalance': imbalance,
                    'imbalance_interpretation': self._interpret_imbalance(imbalance),
                    'liquidity_score': min(total_volume / 100, 10.0),  # Simple liquidity scoring
                    'depth_levels': len(bids) + len(asks)
                }
            })

            return order_book

        except Exception as e:
            logger.error(f"Failed to get market depth analysis for {symbol}: {e}")
            return None

    def _interpret_imbalance(self, imbalance: float) -> str:
        """Interpret order book imbalance."""
        if imbalance > 0.2:
            return "Strong buying pressure"
        elif imbalance > 0.1:
            return "Moderate buying pressure"
        elif imbalance < -0.2:
            return "Strong selling pressure"
        elif imbalance < -0.1:
            return "Moderate selling pressure"
        else:
            return "Balanced"

    async def validate_connection(self) -> bool:
        """Validate connection to MetaAPI."""
        try:
            return await self.metaapi_provider.validate_connection()
        except Exception as e:
            logger.error(f"Order book provider connection validation failed: {e}")
            return False
    
    async def is_connected(self) -> bool:
        """Check if provider is connected."""
        return await self.metaapi_provider.is_connected()
    
    async def get_ohlcv_data(self, symbol: str, timeframe, start_time=None, end_time=None, count=None):
        """Order book provider doesn't support OHLCV data."""
        raise NotImplementedError("Order book provider doesn't support OHLCV data")
    
    async def get_tick_data(self, symbol: str, start_time=None, end_time=None, count=None):
        """Order book provider doesn't support tick data."""
        raise NotImplementedError("Order book provider doesn't support tick data")

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the order book provider."""
        metaapi_health = self.metaapi_provider.get_health_status()

        return {
            'provider_name': self.name,
            'provider_type': self.provider_type.value,
            'connection_status': self.connection_status.value,
            'metaapi_health': metaapi_health,
            'capabilities': {
                'supports_order_book': True,
                'supports_market_depth_analysis': True,
                'supports_real_time': True
            }
        }


# For backward compatibility, create an alias
OrderBookProvider = RealOrderBookProvider# For backward compatibility, create an alias
OrderBookProvider = RealOrderBookProvider
