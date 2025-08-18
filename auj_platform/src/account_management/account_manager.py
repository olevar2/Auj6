"""
Account Manager Implementation

This module provides comprehensive account management functionality including
balance tracking, position management, and account operations.
"""

import asyncio
import logging
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Any, Union
from contextlib import asynccontextmanager

from .account_info import AccountInfo, PositionInfo, PositionType, PositionStatus


class AccountManager:
    """
    Account Manager for handling all account-related operations.

    Provides comprehensive account management including:
    - Real-time account balance tracking
    - Position management and monitoring
    - Margin calculations and validation
    - Trade validation and approval
    - Account safety checks
    """

    def __init__(self, config: Dict[str, Any], broker_interface=None, config_manager=None):
        from ..core.unified_config import UnifiedConfigManager
        self.config_manager = config_manager or UnifiedConfigManager()
        """
        Initialize Account Manager.

        Args:
            config: Configuration dictionary
            broker_interface: Broker interface for account operations
        """
        self.config = config
        self.broker_interface = broker_interface
        self.logger = logging.getLogger(__name__)

        # Account state
        self.current_account_info: Optional[AccountInfo] = None
        self.positions: Dict[str, PositionInfo] = {}
        self.last_update: Optional[datetime] = None

        # Configuration parameters
        self.update_interval = self.config_manager.get_int('account_update_interval', 5)  # seconds
        self.max_margin_utilization = Decimal(str(self.config_manager.get_int('max_margin_utilization', 80)))
        self.min_free_margin = Decimal(str(self.config_manager.get_int('min_free_margin', 1000)))
        self.emergency_margin_level = Decimal(str(self.config_manager.get_int('emergency_margin_level', 100)))

        # Monitoring and safety
        self.monitoring_enabled = True
        self.safety_checks_enabled = True
        self.position_monitoring_task: Optional[asyncio.Task] = None

        self.logger.info("Account Manager initialized")

    async def initialize(self) -> None:
        """Initialize the account manager and start monitoring."""
        try:
            # Get initial account information
            await self.refresh_account_info()

            # Load existing positions
            await self.refresh_positions()

            # Start position monitoring
            if self.monitoring_enabled:
                self.position_monitoring_task = asyncio.create_task(
                    self._position_monitoring_loop()
                )

            self.logger.info("Account Manager initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize Account Manager: {e}")
            raise

    async def cleanup(self) -> None:
        """Cleanup resources and stop monitoring."""
        if self.position_monitoring_task:
            self.position_monitoring_task.cancel()
            try:
                await self.position_monitoring_task
            except asyncio.CancelledError:
                pass

        self.logger.info("Account Manager cleanup completed")

    async def get_account_info(self, force_refresh: bool = False) -> AccountInfo:
        """
        Get current account information.

        Args:
            force_refresh: Force refresh from broker

        Returns:
            Current account information
        """
        if force_refresh or self._needs_account_refresh():
            await self.refresh_account_info()

        if self.current_account_info is None:
            raise RuntimeError("Account information not available")

        return self.current_account_info

    async def refresh_account_info(self) -> None:
        """Refresh account information from broker."""
        try:
            if self.broker_interface and hasattr(self.broker_interface, 'get_account_info'):
                # Get real account info from broker
                account_data = await self.broker_interface.get_account_info()
                self.current_account_info = AccountInfo(**account_data)
            else:
                # Create mock account info for development/testing
                self.current_account_info = self._create_mock_account_info()

            self.last_update = datetime.utcnow()
            self.logger.debug("Account information refreshed")

        except Exception as e:
            self.logger.error(f"Failed to refresh account info: {e}")
            # Create emergency mock account if real data fails
            if self.current_account_info is None:
                self.current_account_info = self._create_mock_account_info()

    def _create_mock_account_info(self) -> AccountInfo:
        """Create mock account info for development/testing."""
        return AccountInfo(
            account_id="MOCK_ACCOUNT_001",
            balance=Decimal('10000.00'),
            equity=Decimal('10000.00'),
            margin_used=Decimal('0.00'),
            margin_free=Decimal('10000.00'),
            margin_level=Decimal('0.00'),
            currency="USD",
            leverage=100,
            profit=Decimal('0.00'),
            timestamp=datetime.utcnow(),
            server_name="Mock Server",
            trade_allowed=True,
            trade_expert=True
        )

    def _needs_account_refresh(self) -> bool:
        """Check if account info needs refresh."""
        if self.last_update is None:
            return True

        time_since_update = (datetime.utcnow() - self.last_update).total_seconds()
        return time_since_update > self.update_interval

    async def get_positions(self, symbol: Optional[str] = None) -> List[PositionInfo]:
        """
        Get current positions.

        Args:
            symbol: Optional symbol filter

        Returns:
            List of current positions
        """
        await self.refresh_positions()

        if symbol:
            return [pos for pos in self.positions.values() if pos.symbol == symbol]

        return list(self.positions.values())

    async def refresh_positions(self) -> None:
        """Refresh positions from broker."""
        try:
            if self.broker_interface and hasattr(self.broker_interface, 'get_positions'):
                # Get real positions from broker
                positions_data = await self.broker_interface.get_positions()
                self.positions = {
                    pos['position_id']: PositionInfo(**pos)
                    for pos in positions_data
                }
            else:
                # Clear positions for mock/development mode
                self.positions = {}

            self.logger.debug(f"Refreshed {len(self.positions)} positions")

        except Exception as e:
            self.logger.error(f"Failed to refresh positions: {e}")

    async def can_open_position(self,
                               symbol: str,
                               position_size: Decimal,
                               position_type: PositionType) -> tuple[bool, str]:
        """
        Check if a position can be opened.

        Args:
            symbol: Trading symbol
            position_size: Position size
            position_type: Position type

        Returns:
            Tuple of (can_open, reason)
        """
        try:
            account_info = await self.get_account_info()

            # Basic account checks
            if not account_info.trade_allowed:
                return False, "Trading not allowed on account"

            # Calculate required margin (simplified calculation)
            estimated_margin = position_size * Decimal('100')  # Simplified estimate

            # Check margin availability
            if account_info.margin_free < estimated_margin:
                return False, f"Insufficient margin: {account_info.margin_free} < {estimated_margin}"

            # Check margin utilization
            new_margin_used = account_info.margin_used + estimated_margin
            new_utilization = (new_margin_used / account_info.equity) * Decimal('100')

            if new_utilization > self.max_margin_utilization:
                return False, f"Margin utilization would exceed limit: {new_utilization}% > {self.max_margin_utilization}%"

            # Check minimum free margin after trade
            remaining_margin = account_info.margin_free - estimated_margin
            if remaining_margin < self.min_free_margin:
                return False, f"Would leave insufficient margin: {remaining_margin} < {self.min_free_margin}"

            return True, "Position can be opened"

        except Exception as e:
            self.logger.error(f"Error checking position capability: {e}")
            return False, f"Error checking position: {e}"

    async def _position_monitoring_loop(self) -> None:
        """Background task for monitoring positions and account."""
        self.logger.info("Starting position monitoring loop")

        while self.monitoring_enabled:
            try:
                # Refresh account and positions
                await self.refresh_account_info()
                await self.refresh_positions()

                # Perform safety checks
                if self.safety_checks_enabled:
                    await self._perform_safety_checks()

                # Wait for next update
                await asyncio.sleep(self.update_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in position monitoring: {e}")
                await asyncio.sleep(self.update_interval)

        self.logger.info("Position monitoring loop stopped")

    async def _perform_safety_checks(self) -> None:
        """Perform account safety checks."""
        if not self.current_account_info:
            return

        # Check margin level
        if (self.current_account_info.margin_level > 0 and
            self.current_account_info.margin_level < self.emergency_margin_level):
            self.logger.warning(
                f"EMERGENCY: Margin level critically low: {self.current_account_info.margin_level}%"
            )

        # Check margin utilization
        utilization = self.current_account_info.margin_utilization
        if utilization > self.max_margin_utilization:
            self.logger.warning(
                f"WARNING: Margin utilization high: {utilization}%"
            )

    async def get_position_by_id(self, position_id: str) -> Optional[PositionInfo]:
        """Get position by ID."""
        await self.refresh_positions()
        return self.positions.get(position_id)

    async def get_total_exposure(self, symbol: Optional[str] = None) -> Decimal:
        """
        Calculate total exposure for symbol or all positions.

        Args:
            symbol: Optional symbol filter

        Returns:
            Total exposure amount
        """
        positions = await self.get_positions(symbol)
        return sum(pos.market_value for pos in positions)

    async def get_total_pnl(self, symbol: Optional[str] = None) -> Decimal:
        """
        Calculate total PnL for symbol or all positions.

        Args:
            symbol: Optional symbol filter

        Returns:
            Total PnL amount
        """
        positions = await self.get_positions(symbol)
        return sum(pos.total_pnl for pos in positions)
