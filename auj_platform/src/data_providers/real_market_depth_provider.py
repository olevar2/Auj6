"""
Real Market Depth Provider for the AUJ Platform.

This provider uses MetaAPI for market depth data, providing modern 
cloud-based market depth functionality with comprehensive analysis.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd
import asyncio

from .base_provider import (
    BaseDataProvider, DataProviderType, DataProviderCapabilities, 
    Timeframe, ConnectionStatus, DataType
)
from .metaapi_provider import MetaApiProvider
from ..core.exceptions import DataNotAvailableError
from ..core.logging_setup import get_logger
from ..core.unified_config import UnifiedConfigManager

logger = get_logger(__name__)


class RealMarketDepthProvider(BaseDataProvider):
    """
    Real market depth provider that uses MetaAPI for market depth data.
    
    This provider uses MetaAPI cloud services for market depth functionality,
    providing enhanced market depth analysis including liquidity distribution, 
    price level analysis, and market microstructure insights.
    """
    
    def __init__(self, config_manager: Optional[UnifiedConfigManager] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the real market depth provider.
        
        Args:
            config_manager: Configuration manager instance
            config: Additional configuration parameters
        """
        # Initialize MetaAPI provider first, before calling super().__init__
        self.metaapi_provider = MetaApiProvider(config_manager=config_manager, config=config)
        
        super().__init__(
            name="MarketDepth_Provider_Real",
            provider_type=DataProviderType.ORDER_BOOK,
            config_manager=config_manager,
            config=config
        )
        
        # Configuration for market depth analysis
        self.default_depth = self.config_manager.get_int('default_market_depth', 20) if config_manager else 20
        self.max_depth = self.config_manager.get_int('max_market_depth', 50) if config_manager else 50
        
        logger.info("Real Market Depth Provider initialized - using MetaAPI cloud services")
    
    def _define_capabilities(self) -> DataProviderCapabilities:
        """Define real market depth capabilities."""
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
            supported_timeframes=[]  # Market depth doesn't use timeframes
        )
    
    async def connect(self) -> bool:
        """Connect to MetaAPI for market depth data."""
        try:
            success = await self.metaapi_provider.connect()
            if success:
                self.connection_status = ConnectionStatus.CONNECTED
                logger.info("Market depth provider connected to MetaAPI")
                return True
            else:
                self.connection_status = ConnectionStatus.FAILED
                logger.error("Failed to connect market depth provider to MetaAPI")
                return False
        except Exception as e:
            self.connection_status = ConnectionStatus.FAILED
            logger.error(f"Market depth provider connection error: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from MetaAPI."""
        try:
            success = await self.metaapi_provider.disconnect()
            if success:
                self.connection_status = ConnectionStatus.DISCONNECTED
                logger.info("Market depth provider disconnected from MetaAPI")
                return True
            else:
                logger.warning("Market depth provider disconnect had issues")
                return False
        except Exception as e:
            logger.error(f"Market depth provider disconnect error: {e}")
            return False
    
    async def get_market_depth_data(self, symbol: str, depth: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive market depth data for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            depth: Number of levels to retrieve (default: configured default)
            
        Returns:
            Dictionary containing comprehensive market depth data
            None if data is not available
        """
        try:
            if self.connection_status != ConnectionStatus.CONNECTED:
                logger.warning("Market depth provider not connected, attempting to connect...")
                if not await self.connect():
                    logger.error("Failed to establish connection for market depth data")
                    return None
            
            # Use configured default if not specified
            if depth is None:
                depth = self.default_depth
            
            # Ensure depth doesn't exceed maximum
            depth = min(depth, self.max_depth)
            
            # Get order book data from MetaAPI (market depth is essentially order book data)
            order_book = await self.metaapi_provider.get_market_depth(symbol, depth)
            
            if order_book is None:
                logger.warning(f"No market depth data available for {symbol} via MetaAPI")
                return None
            
            # Enhance with market depth specific analysis
            market_depth = await self._enhance_with_depth_analysis(order_book, symbol)
            
            logger.debug(f"Retrieved market depth for {symbol} with {depth} levels")
            return market_depth
            
        except Exception as e:
            logger.error(f"Failed to get market depth data for {symbol}: {e}")
            return None
    
    async def _enhance_with_depth_analysis(self, order_book: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """
        Enhance order book data with market depth specific analysis.
        
        Args:
            order_book: Raw order book data from MT5
            symbol: Trading symbol
            
        Returns:
            Enhanced market depth data with additional analysis
        """
        try:
            bids = order_book.get('bids', [])
            asks = order_book.get('asks', [])
            
            if not bids or not asks:
                logger.warning(f"Incomplete market depth data for {symbol}")
                return order_book
            
            # Calculate liquidity distribution
            liquidity_analysis = self._calculate_liquidity_distribution(bids, asks)
            
            # Calculate price level gaps
            price_gaps = self._calculate_price_gaps(bids, asks)
            
            # Calculate market pressure indicators
            pressure_indicators = self._calculate_market_pressure(bids, asks)
            
            # Calculate depth quality metrics
            quality_metrics = self._calculate_depth_quality(bids, asks)
            
            # Add comprehensive market depth analysis
            order_book.update({
                'market_depth_analysis': {
                    'liquidity_distribution': liquidity_analysis,
                    'price_gaps': price_gaps,
                    'market_pressure': pressure_indicators,
                    'depth_quality': quality_metrics,
                    'timestamp': datetime.now().isoformat(),
                    'provider': self.name
                }
            })
            
            return order_book
            
        except Exception as e:
            logger.error(f"Failed to enhance market depth analysis for {symbol}: {e}")
            return order_book
    
    def _calculate_liquidity_distribution(self, bids: List[Dict], asks: List[Dict]) -> Dict[str, Any]:
        """Calculate liquidity distribution across price levels."""
        
        # Calculate cumulative volumes
        bid_cumulative = []
        ask_cumulative = []
        
        bid_total = 0
        for bid in bids:
            bid_total += bid['volume']
            bid_cumulative.append(bid_total)
        
        ask_total = 0
        for ask in asks:
            ask_total += ask['volume']
            ask_cumulative.append(ask_total)
        
        # Calculate liquidity concentration (how much liquidity is in top levels)
        top_3_bid_volume = sum(bid['volume'] for bid in bids[:3]) if len(bids) >= 3 else bid_total
        top_3_ask_volume = sum(ask['volume'] for ask in asks[:3]) if len(asks) >= 3 else ask_total
        
        bid_concentration = (top_3_bid_volume / bid_total) if bid_total > 0 else 0
        ask_concentration = (top_3_ask_volume / ask_total) if ask_total > 0 else 0
        
        return {
            'total_bid_volume': bid_total,
            'total_ask_volume': ask_total,
            'total_liquidity': bid_total + ask_total,
            'bid_concentration_top3': bid_concentration,
            'ask_concentration_top3': ask_concentration,
            'liquidity_balance': (bid_total - ask_total) / (bid_total + ask_total) if (bid_total + ask_total) > 0 else 0,
            'bid_cumulative_volumes': bid_cumulative,
            'ask_cumulative_volumes': ask_cumulative
        }
    
    def _calculate_price_gaps(self, bids: List[Dict], asks: List[Dict]) -> Dict[str, Any]:
        """Calculate price gaps between levels."""
        
        bid_gaps = []
        ask_gaps = []
        
        # Calculate gaps between bid levels
        for i in range(1, len(bids)):
            gap = bids[i-1]['price'] - bids[i]['price']
            bid_gaps.append(gap)
        
        # Calculate gaps between ask levels
        for i in range(1, len(asks)):
            gap = asks[i]['price'] - asks[i-1]['price']
            ask_gaps.append(gap)
        
        # Calculate spread
        spread = asks[0]['price'] - bids[0]['price'] if bids and asks else 0
        
        return {
            'spread': spread,
            'avg_bid_gap': sum(bid_gaps) / len(bid_gaps) if bid_gaps else 0,
            'avg_ask_gap': sum(ask_gaps) / len(ask_gaps) if ask_gaps else 0,
            'max_bid_gap': max(bid_gaps) if bid_gaps else 0,
            'max_ask_gap': max(ask_gaps) if ask_gaps else 0,
            'bid_gap_consistency': 1.0 - (max(bid_gaps) - min(bid_gaps)) / max(bid_gaps) if bid_gaps and max(bid_gaps) > 0 else 1.0,
            'ask_gap_consistency': 1.0 - (max(ask_gaps) - min(ask_gaps)) / max(ask_gaps) if ask_gaps and max(ask_gaps) > 0 else 1.0
        }
    
    def _calculate_market_pressure(self, bids: List[Dict], asks: List[Dict]) -> Dict[str, Any]:
        """Calculate market pressure indicators."""
        
        # Calculate volume-weighted prices for top levels
        top_levels = 5
        
        bid_vwap = 0
        ask_vwap = 0
        bid_vol = 0
        ask_vol = 0
        
        for i, bid in enumerate(bids[:top_levels]):
            bid_vwap += bid['price'] * bid['volume']
            bid_vol += bid['volume']
        
        for i, ask in enumerate(asks[:top_levels]):
            ask_vwap += ask['price'] * ask['volume']
            ask_vol += ask['volume']
        
        bid_vwap = bid_vwap / bid_vol if bid_vol > 0 else 0
        ask_vwap = ask_vwap / ask_vol if ask_vol > 0 else 0
        
        # Calculate pressure ratios
        volume_ratio = bid_vol / ask_vol if ask_vol > 0 else float('inf')
        
        return {
            'bid_volume_top5': bid_vol,
            'ask_volume_top5': ask_vol,
            'volume_ratio_top5': volume_ratio,
            'bid_vwap_top5': bid_vwap,
            'ask_vwap_top5': ask_vwap,
            'pressure_direction': 'buying' if volume_ratio > 1.2 else 'selling' if volume_ratio < 0.8 else 'neutral',
            'pressure_strength': abs(1 - volume_ratio) if volume_ratio != float('inf') else 1.0
        }
    
    def _calculate_depth_quality(self, bids: List[Dict], asks: List[Dict]) -> Dict[str, Any]:
        """Calculate market depth quality metrics."""
        
        # Calculate depth uniformity
        bid_volumes = [bid['volume'] for bid in bids]
        ask_volumes = [ask['volume'] for ask in asks]
        
        bid_volume_std = pd.Series(bid_volumes).std() if bid_volumes else 0
        ask_volume_std = pd.Series(ask_volumes).std() if ask_volumes else 0
        bid_volume_mean = pd.Series(bid_volumes).mean() if bid_volumes else 0
        ask_volume_mean = pd.Series(ask_volumes).mean() if ask_volumes else 0
        
        bid_uniformity = 1 - (bid_volume_std / bid_volume_mean) if bid_volume_mean > 0 else 0
        ask_uniformity = 1 - (ask_volume_std / ask_volume_mean) if ask_volume_mean > 0 else 0
        
        return {
            'depth_levels': len(bids) + len(asks),
            'bid_levels': len(bids),
            'ask_levels': len(asks),
            'bid_uniformity': max(0, bid_uniformity),  # Ensure non-negative
            'ask_uniformity': max(0, ask_uniformity),  # Ensure non-negative
            'overall_quality_score': (max(0, bid_uniformity) + max(0, ask_uniformity)) / 2,
            'depth_completeness': min(len(bids), len(asks)) / max(len(bids), len(asks)) if bids and asks else 0
        }
    
    async def validate_connection(self) -> bool:
        """Validate connection to MetaAPI."""
        try:
            return await self.metaapi_provider.validate_connection()
        except Exception as e:
            logger.error(f"Market depth provider connection validation failed: {e}")
            return False
    
    async def is_connected(self) -> bool:
        """Check if provider is connected."""
        return await self.metaapi_provider.is_connected()
    
    async def get_ohlcv_data(self, symbol: str, timeframe, start_time=None, end_time=None, count=None):
        """Market depth provider doesn't support OHLCV data."""
        raise NotImplementedError("Market depth provider doesn't support OHLCV data")
    
    async def get_tick_data(self, symbol: str, start_time=None, end_time=None, count=None):
        """Market depth provider doesn't support tick data."""
        raise NotImplementedError("Market depth provider doesn't support tick data")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the market depth provider."""
        metaapi_health = self.metaapi_provider.get_health_status()
        
        return {
            'provider_name': self.name,
            'provider_type': self.provider_type.value,
            'connection_status': self.connection_status.value,
            'configuration': {
                'default_depth': self.default_depth,
                'max_depth': self.max_depth
            },
            'metaapi_health': metaapi_health,
            'capabilities': {
                'supports_market_depth': True,
                'supports_liquidity_analysis': True,
                'supports_price_gap_analysis': True,
                'supports_market_pressure_analysis': True,
                'supports_real_time': True
            }
        }


# For backward compatibility, create an alias
MarketDepthProvider = RealMarketDepthProvider