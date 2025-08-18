"""
Unified MT5 Provider (DEPRECATED - USE MetaAPI)

This provider has been deprecated in favor of MetaAPI for cross-platform compatibility.
Use MetaApiProvider instead for all trading and data access operations.

Migration Path:
- Replace UnifiedMT5Provider with MetaApiProvider
- Update broker interface configuration to use MetaAPI
- See documentation for MetaAPI migration guide
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from decimal import Decimal
import pandas as pd

from .base_provider import BaseDataProvider, DataProviderType, DataProviderCapabilities, ConnectionStatus
from ..core.exceptions import DataProviderError, ConnectionError, TradingError
from ..core.logging_setup import get_logger

logger = get_logger(__name__)


class UnifiedMT5Provider(BaseDataProvider):
    """
    DEPRECATED: Unified MetaTrader 5 Provider
    
    This provider has been replaced by MetaApiProvider for Linux compatibility.
    All trading and data access functionality is now handled by MetaAPI.
    """

    # Singleton instance management (maintained for compatibility)
    _instance = None
    _lock = asyncio.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(UnifiedMT5Provider, cls).__new__(cls)
        return cls._instance

    def __init__(self, config_manager=None, config: Optional[Dict[str, Any]] = None):
        """Initialize deprecated provider with warning."""
        logger.warning("UnifiedMT5Provider is DEPRECATED. Use MetaApiProvider instead.")
        
        if hasattr(self, '_initialized'):
            return
            
        # Basic initialization to maintain compatibility
        self.config_manager = config_manager
        self.config = config or {}
        self._initialized = True
        self.available_symbols = []
        
        super().__init__(
            name="unified_mt5_deprecated",
            provider_type=DataProviderType.MT5,
            config_manager=config_manager,
            config=config or {}
        )

    async def initialize(self) -> bool:
        """Return False as provider is deprecated."""
        logger.warning("UnifiedMT5Provider.initialize() called - provider is deprecated")
        return False
        
    async def connect(self) -> bool:
        """Return False as provider is deprecated."""
        logger.warning("UnifiedMT5Provider connection attempted - provider is deprecated")
        return False
        
    async def disconnect(self) -> None:
        """No-op for deprecated provider."""
        pass

    # Trading Interface Methods (stubs for compatibility)
    async def place_order(self, symbol: str, order_type: str, volume: float, 
                         price: Optional[float] = None, **kwargs) -> Optional[Dict[str, Any]]:
        """Raise error as provider is deprecated."""
        logger.error("UnifiedMT5Provider.place_order called - provider is deprecated. Use MetaApiProvider.")
        raise TradingError("UnifiedMT5Provider is deprecated. Use MetaApiProvider for trading operations.")

    async def close_position(self, position_id: str) -> bool:
        """Raise error as provider is deprecated."""
        logger.error("UnifiedMT5Provider.close_position called - provider is deprecated. Use MetaApiProvider.")
        raise TradingError("UnifiedMT5Provider is deprecated. Use MetaApiProvider for trading operations.")
        
    async def modify_order(self, order_id: str, **kwargs) -> bool:
        """Raise error as provider is deprecated."""
        logger.error("UnifiedMT5Provider.modify_order called - provider is deprecated. Use MetaApiProvider.")
        raise TradingError("UnifiedMT5Provider is deprecated. Use MetaApiProvider for trading operations.")
        
    async def cancel_order(self, order_id: str) -> bool:
        """Raise error as provider is deprecated."""
        logger.error("UnifiedMT5Provider.cancel_order called - provider is deprecated. Use MetaApiProvider.")
        raise TradingError("UnifiedMT5Provider is deprecated. Use MetaApiProvider for trading operations.")    # Data Provider Methods (stubs for compatibility)
    async def get_ohlcv_data(self, symbol: str, timeframe, start_time: Optional[datetime] = None,
                           end_time: Optional[datetime] = None, count: Optional[int] = None) -> Optional[pd.DataFrame]:
        """Raise error as provider is deprecated."""
        logger.error("UnifiedMT5Provider.get_ohlcv_data called - provider is deprecated. Use MetaApiProvider.")
        raise DataProviderError("UnifiedMT5Provider is deprecated. Use MetaApiProvider for data access.")
        
    async def get_tick_data(self, symbol: str, start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None, count: Optional[int] = None) -> Optional[pd.DataFrame]:
        """Raise error as provider is deprecated."""
        logger.error("UnifiedMT5Provider.get_tick_data called - provider is deprecated. Use MetaApiProvider.")
        raise DataProviderError("UnifiedMT5Provider is deprecated. Use MetaApiProvider for data access.")
        
    async def get_symbols(self) -> List[str]:
        """Return empty list as provider is deprecated."""
        logger.warning("UnifiedMT5Provider.get_symbols called - provider is deprecated")
        return []
        
    async def get_account_info(self) -> Optional[Dict[str, Any]]:
        """Raise error as provider is deprecated."""
        logger.error("UnifiedMT5Provider.get_account_info called - provider is deprecated. Use MetaApiProvider.")
        raise DataProviderError("UnifiedMT5Provider is deprecated. Use MetaApiProvider for account info.")
        
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Return empty list as provider is deprecated."""
        logger.warning("UnifiedMT5Provider.get_positions called - provider is deprecated")
        return []
        
    async def get_orders(self) -> List[Dict[str, Any]]:
        """Return empty list as provider is deprecated."""
        logger.warning("UnifiedMT5Provider.get_orders called - provider is deprecated")
        return []

    def _define_capabilities(self) -> DataProviderCapabilities:
        """Return deprecated capabilities."""
        return DataProviderCapabilities(
            supports_ohlcv=False,
            supports_tick=False,
            supports_live=False,
            supports_historical=False
        )

    # Additional compatibility methods
    def is_connected(self) -> bool:
        """Return False as provider is deprecated."""
        return False
        
    async def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Return None as provider is deprecated."""
        logger.warning(f"UnifiedMT5Provider.get_symbol_info({symbol}) called - provider is deprecated")
        return None