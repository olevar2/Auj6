"""
MetaTrader 5 OHLCV Data Provider (DEPRECATED - USE MetaAPI)

This provider has been deprecated in favor of MetaAPI for cross-platform compatibility.
Use MetaApiProvider instead for OHLCV data access.

Migration Path:
- Replace MT5OHLCVProvider with MetaApiProvider
- Update configuration to use metaapi_config.yaml
- See documentation for MetaAPI migration guide
"""

import logging
from typing import Dict, List, Optional, Any
import pandas as pd
from datetime import datetime

from .base_provider import (
    BaseDataProvider, DataProviderType, DataProviderCapabilities,
    Timeframe, ConnectionStatus, DataType
)
from ..core.exceptions import DataProviderError
from ..core.logging_setup import get_logger

logger = get_logger(__name__)


class MT5OHLCVProvider(BaseDataProvider):
    """
    DEPRECATED: MetaTrader 5 OHLCV data provider.
    
    This provider has been replaced by MetaApiProvider for Linux compatibility.
    Use MetaApiProvider for all new implementations.
    """

    def __init__(self, config_manager, config: Optional[Dict[str, Any]] = None):
        """Initialize deprecated provider with warning."""
        logger.warning("MT5OHLCVProvider is DEPRECATED. Use MetaApiProvider instead.")
        
        # Basic initialization to maintain compatibility
        self.available_symbols = []
        super().__init__(
            name="mt5_ohlcv_deprecated",
            provider_type=DataProviderType.MT5,
            config_manager=config_manager,
            config=config or {}
        )
        
    def _define_capabilities(self) -> DataProviderCapabilities:
        """Return deprecated capabilities."""
        return DataProviderCapabilities(
            supports_ohlcv=False,
            supports_live=False,
            supports_historical=False
        )
        
    async def connect(self) -> bool:
        """Return False as provider is deprecated."""
        logger.warning("MT5OHLCVProvider connection attempted - provider is deprecated")
        return False
        
    async def disconnect(self) -> None:
        """No-op for deprecated provider."""
        pass
        
    async def get_ohlcv_data(self, symbol: str, timeframe: Timeframe,
                           start_time: Optional[datetime] = None,
                           end_time: Optional[datetime] = None, count: Optional[int] = None) -> Optional[pd.DataFrame]:
        """Return None as provider is deprecated."""
        logger.error("MT5OHLCVProvider.get_ohlcv_data called - provider is deprecated. Use MetaApiProvider.")
        raise DataProviderError("MT5OHLCVProvider is deprecated. Use MetaApiProvider for OHLCV data.")
        
    async def get_tick_data(self, symbol: str, start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None, count: Optional[int] = None) -> Optional[pd.DataFrame]:
        """Return None as provider is deprecated."""
        logger.error("MT5OHLCVProvider.get_tick_data called - provider is deprecated. Use MetaApiProvider.")
        raise DataProviderError("MT5OHLCVProvider is deprecated. Use MetaApiProvider for tick data.")
        
    async def get_symbols(self) -> List[str]:
        """Return empty list as provider is deprecated."""
        return []