"""
Base Broker Interface

Abstract base class defining the standard interface for all broker integrations.
"""

from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Dict, List, Optional, Any
from datetime import datetime


class BaseBroker(ABC):
    """
    Abstract base class for broker interfaces.
    
    Defines the standard interface that all broker implementations must follow.
    """
    
    def __init__(self, config: Dict[str, Any]):
        # Note: This class now requires config_manager parameter in __init__
        # self.config_manager = config_manager or UnifiedConfigManager()
        """
        Initialize broker interface.
        
        Args:
            config: Broker-specific configuration
        """
        self.config = config
        self.connected = False
        self.enabled = self.config_manager.get_bool('enabled', False)
    
    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize broker connection.
        
        Returns:
            True if initialization successful
        """
        self.connected = True
        return True
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup broker resources."""
        self.connected = False
    
    @abstractmethod
    async def test_connection(self) -> bool:
        """
        Test broker connection.
        
        Returns:
            True if connection is working
        """
        return self.connected and self.enabled
    
    # Trading Methods
    
    @abstractmethod
    async def place_order(self,
                         symbol: str,
                         order_type: str,
                         volume: float,
                         price: Optional[float] = None,
                         sl: Optional[float] = None,
                         tp: Optional[float] = None,
                         comment: str = "",
                         magic: int = 0) -> Dict[str, Any]:
        """
        Place a trading order.
        
        Args:
            symbol: Trading symbol
            order_type: Order type (BUY, SELL, BUY_LIMIT, SELL_LIMIT, etc.)
            volume: Order volume (lot size)
            price: Order price (for limit/stop orders)
            sl: Stop loss price
            tp: Take profit price
            comment: Order comment
            magic: Magic number for order identification
            
        Returns:
            Dict containing order result information
        """
        raise NotImplementedError("Subclasses must implement place_order()")
    
    @abstractmethod
    async def close_position(self,
                           symbol: str,
                           position_id: Optional[int] = None,
                           volume: Optional[float] = None) -> Dict[str, Any]:
        """
        Close a position or part of it.
        
        Args:
            symbol: Trading symbol
            position_id: Specific position ID (optional)
            volume: Volume to close (if None, close all)
            
        Returns:
            Dict containing close result information
        """
        raise NotImplementedError("Subclasses must implement close_position()")
    
    @abstractmethod
    async def get_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get current positions.
        
        Args:
            symbol: Optional symbol filter
            
        Returns:
            List of position dictionaries
        """
        raise NotImplementedError("Subclasses must implement get_positions()")
    
    @abstractmethod
    async def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information.
        
        Returns:
            Dict containing account information
        """
        raise NotImplementedError("Subclasses must implement get_account_info()")
    
    @abstractmethod
    async def modify_position(self,
                            position_id: int,
                            sl: Optional[float] = None,
                            tp: Optional[float] = None) -> Dict[str, Any]:
        """
        Modify position stop loss and take profit.
        
        Args:
            position_id: Position identifier
            sl: New stop loss price
            tp: New take profit price
            
        Returns:
            Dict containing modification result
        """
        raise NotImplementedError("Subclasses must implement modify_position()")
    
    # Market Data Methods
    
    @abstractmethod
    async def get_current_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get current price for symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dict containing current price data
        """
        raise NotImplementedError("Subclasses must implement get_current_price()")
    
    async def get_symbols(self) -> List[str]:
        """
        Get available trading symbols.
        
        Returns:
            List of available symbols
        """
        return []
    
    async def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get symbol information.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dict containing symbol information
        """
        return None