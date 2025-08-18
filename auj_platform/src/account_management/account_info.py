"""
Account Information Classes

Data structures for account and position information.
"""

from dataclasses import dataclass
from decimal import Decimal
from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum


class PositionType(Enum):
    """Position type enumeration."""
    LONG = "LONG"
    SHORT = "SHORT"


class PositionStatus(Enum):
    """Position status enumeration."""
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    PENDING = "PENDING"


@dataclass
class PositionInfo:
    """Position information data structure."""
    
    position_id: str
    symbol: str
    position_type: PositionType
    size: Decimal
    entry_price: Decimal
    current_price: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    timestamp: datetime
    status: PositionStatus
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    commission: Decimal = Decimal('0')
    swap: Decimal = Decimal('0')
    comment: str = ""
    
    @property
    def total_pnl(self) -> Decimal:
        """Calculate total PnL including realized and unrealized."""
        return self.realized_pnl + self.unrealized_pnl
    
    @property
    def market_value(self) -> Decimal:
        """Calculate current market value of position."""
        return self.size * self.current_price


@dataclass
class AccountInfo:
    """Account information data structure."""
    
    account_id: str
    balance: Decimal
    equity: Decimal
    margin_used: Decimal
    margin_free: Decimal
    margin_level: Decimal
    currency: str
    leverage: int
    profit: Decimal
    timestamp: datetime
    server_name: str = ""
    trade_allowed: bool = True
    trade_expert: bool = True
    margin_so_mode: int = 0
    margin_so_call: Decimal = Decimal('0')
    margin_so_so: Decimal = Decimal('0')
    margin_initial: Decimal = Decimal('0')
    margin_maintenance: Decimal = Decimal('0')
    assets: Decimal = Decimal('0')
    liabilities: Decimal = Decimal('0')
    commission_blocked: Decimal = Decimal('0')
    
    @property
    def available_margin(self) -> Decimal:
        """Calculate available margin for trading."""
        return self.margin_free
    
    @property
    def margin_utilization(self) -> Decimal:
        """Calculate margin utilization percentage."""
        if self.equity <= 0:
            return Decimal('100')
        return (self.margin_used / self.equity) * Decimal('100')
    
    def can_open_position(self, required_margin: Decimal) -> bool:
        """Check if account can open position with required margin."""
        return self.margin_free >= required_margin and self.trade_allowed