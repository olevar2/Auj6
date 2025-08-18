"""
MT5 Broker Interface (DEPRECATED - USE MetaAPI)

This broker interface has been deprecated in favor of MetaAPI for cross-platform compatibility.
Use MetaApiBroker instead for all trading operations.

Migration Path:
- Replace MT5Broker with MetaApiBroker
- Update configuration to use metaapi_config.yaml
- See documentation for MetaAPI migration guide
"""

from typing import Dict, List, Optional, Any
from decimal import Decimal
from datetime import datetime
import logging

from .base_broker import BaseBroker
from ..core.exceptions import TradingError
from ..core.logging_setup import get_logger

logger = get_logger(__name__)


class MT5Broker(BaseBroker):
    """
    DEPRECATED: MetaTrader 5 broker interface implementation.
    
    This broker interface has been replaced by MetaApiBroker for Linux compatibility.
    Use MetaApiBroker for all new implementations.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize deprecated broker with warning."""
        logger.warning("MT5Broker is DEPRECATED. Use MetaApiBroker instead.")
        super().__init__(config)
        self._is_initialized = False

    async def initialize(self) -> bool:
        """Return False as broker is deprecated."""
        logger.warning("MT5Broker initialization attempted - broker is deprecated")
        return False
        
    async def connect(self) -> bool:
        """Return False as broker is deprecated."""
        logger.warning("MT5Broker connection attempted - broker is deprecated")
        return False
        
    async def disconnect(self) -> None:
        """No-op for deprecated broker."""
        pass

    # Trading Interface Methods (stubs for compatibility)
    async def place_market_order(self, symbol: str, order_type: str, volume: float, **kwargs) -> Optional[str]:
        """Raise error as broker is deprecated."""
        logger.error("MT5Broker.place_market_order called - broker is deprecated. Use MetaApiBroker.")
        raise TradingError("MT5Broker is deprecated. Use MetaApiBroker for trading operations.")

    async def place_limit_order(self, symbol: str, order_type: str, volume: float, price: float, **kwargs) -> Optional[str]:
        """Raise error as broker is deprecated."""
        logger.error("MT5Broker.place_limit_order called - broker is deprecated. Use MetaApiBroker.")
        raise TradingError("MT5Broker is deprecated. Use MetaApiBroker for trading operations.")
        
    async def place_stop_order(self, symbol: str, order_type: str, volume: float, price: float, **kwargs) -> Optional[str]:
        """Raise error as broker is deprecated."""
        logger.error("MT5Broker.place_stop_order called - broker is deprecated. Use MetaApiBroker.")
        raise TradingError("MT5Broker is deprecated. Use MetaApiBroker for trading operations.")        
    async def modify_order(self, order_id: str, **kwargs) -> bool:
        """Raise error as broker is deprecated."""
        logger.error("MT5Broker.modify_order called - broker is deprecated. Use MetaApiBroker.")
        raise TradingError("MT5Broker is deprecated. Use MetaApiBroker for trading operations.")
        
    async def cancel_order(self, order_id: str) -> bool:
        """Raise error as broker is deprecated."""
        logger.error("MT5Broker.cancel_order called - broker is deprecated. Use MetaApiBroker.")
        raise TradingError("MT5Broker is deprecated. Use MetaApiBroker for trading operations.")
        
    async def close_position(self, position_id: str) -> bool:
        """Raise error as broker is deprecated."""
        logger.error("MT5Broker.close_position called - broker is deprecated. Use MetaApiBroker.")
        raise TradingError("MT5Broker is deprecated. Use MetaApiBroker for trading operations.")
        
    async def get_account_info(self) -> Optional[Dict[str, Any]]:
        """Return None as broker is deprecated."""
        logger.warning("MT5Broker.get_account_info called - broker is deprecated")
        return None
        
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Return empty list as broker is deprecated."""
        logger.warning("MT5Broker.get_positions called - broker is deprecated")
        return []
        
    async def get_orders(self) -> List[Dict[str, Any]]:
        """Return empty list as broker is deprecated."""
        logger.warning("MT5Broker.get_orders called - broker is deprecated")
        return []
        
    def is_connected(self) -> bool:
        """Return False as broker is deprecated."""
        return False