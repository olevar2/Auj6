"""
Execution Handler for AUJ Platform.

This module implements the final gatekeeper that receives trade plans and places orders
with comprehensive validation, error handling, and execution optimization.
The handler ensures every trade executed serves the noble mission of generating
sustainable profits to support sick children and families in need.

Key Features:
- Final trade validation and authorization
- Multiple broker/platform integration
- Order execution optimization
- Slippage monitoring and control
- Transaction cost analysis
- Execution performance tracking
- Error handling and retry mechanisms
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from decimal import Decimal, ROUND_HALF_UP
import uuid
import logging
from contextlib import asynccontextmanager

from ..core.data_contracts import (
    TradeSignal, TradeDirection, TradeStatus, ConfidenceLevel,
    AccountInfo, GradedDeal, DealGrade
)
from ..core.exceptions import ExecutionError, ValidationError, BrokerError
from ..core.logging_setup import get_logger
from ..core.unified_config import UnifiedConfigManager
from .dynamic_risk_manager import DynamicRiskManager, RiskMetrics

logger = get_logger(__name__)


class ExecutionStatus(str, Enum):
    """Execution status enumeration."""
    PENDING = "PENDING"
    VALIDATED = "VALIDATED"
    SUBMITTED = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"
    EXPIRED = "EXPIRED"


class OrderType(str, Enum):
    """Order type enumeration."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    TRAILING_STOP = "TRAILING_STOP"


class ExecutionVenue(str, Enum):
    """Execution venue enumeration."""
    MT5 = "MT5"  # Deprecated - kept for backward compatibility
    METAAPI = "METAAPI"  # Primary venue for Linux deployment
    INTERACTIVE_BROKERS = "INTERACTIVE_BROKERS"
    OANDA = "OANDA"
    DEMO = "DEMO"
    PAPER_TRADING = "PAPER_TRADING"


@dataclass
class ExecutionOrder:
    """Comprehensive execution order structure."""
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    signal_id: str = ""
    symbol: str = ""
    direction: TradeDirection = TradeDirection.BUY
    order_type: OrderType = OrderType.MARKET
    quantity: Decimal = Decimal('0')
    price: Optional[Decimal] = None
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    venue: ExecutionVenue = ExecutionVenue.DEMO
    status: ExecutionStatus = ExecutionStatus.PENDING

    # Execution details
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    filled_price: Optional[Decimal] = None
    filled_quantity: Decimal = Decimal('0')
    remaining_quantity: Decimal = Decimal('0')

    # Performance metrics
    slippage: Optional[Decimal] = None
    execution_time_ms: Optional[int] = None
    transaction_costs: Decimal = Decimal('0')

    # Risk and validation
    risk_metrics: Optional[RiskMetrics] = None
    validation_errors: List[str] = field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.remaining_quantity:
            self.remaining_quantity = self.quantity


@dataclass
class ExecutionReport:
    """Comprehensive execution report."""
    execution_id: str
    order: ExecutionOrder
    success: bool
    execution_time_seconds: float
    total_slippage: Decimal
    total_costs: Decimal
    fill_rate: float  # Percentage of order filled
    warnings: List[str]
    errors: List[str]
    performance_grade: DealGrade
    timestamp: datetime = field(default_factory=datetime.utcnow)


class ExecutionHandler:
    """
    Final gatekeeper for trade execution with comprehensive validation,
    optimization, and performance tracking.
    """

    def __init__(self,
                 config_manager: UnifiedConfigManager,
                 risk_manager: DynamicRiskManager,
                 messaging_service: Optional[Any] = None):
        """
        Initialize the Execution Handler.

        Args:
            config_manager: Unified configuration manager instance
            risk_manager: Dynamic risk manager instance
            messaging_service: Optional injected messaging service
        """
        self.config_manager = config_manager
        self.risk_manager = risk_manager
        self.messaging_service = messaging_service  # Injected dependency

        # These will be initialized in the initialize() method
        self.account_manager = None
        self.broker_interfaces = {}
        self.performance_tracker = None

        # Execution state
        self.active_orders: Dict[str, ExecutionOrder] = {}
        self.execution_queue: List[ExecutionOrder] = []
        self.execution_history: List[ExecutionReport] = []

        # Configuration parameters - using unified configuration manager
        self.max_slippage_percent = config_manager.get_float('trading.max_slippage_percent', 0.5)
        self.order_timeout_seconds = config_manager.get_int('trading.order_timeout_seconds', 30)
        self.retry_attempts = config_manager.get_int('trading.retry_attempts', 3)
        self.retry_delay_seconds = config_manager.get_int('trading.retry_delay_seconds', 1)
        self.min_order_value = Decimal(str(config_manager.get_float('trading.min_order_value', 100)))

        # Performance tracking
        self.total_executions = 0
        self.successful_executions = 0
        self.total_slippage = Decimal('0')
        self.total_costs = Decimal('0')

        # Execution optimization
        self.venue_performance: Dict[ExecutionVenue, Dict[str, float]] = {}
        self.symbol_execution_stats: Dict[str, Dict[str, Any]] = {}

        # Messaging integration
        self.message_broker = None
        self.messaging_enabled = False

        logger.info("Execution Handler initialized with unified configuration management")

    async def initialize(self) -> None:
        """
        Initialize the Execution Handler with required components.

        This method must be called after construction to complete initialization.
        """
        try:
            # Import and initialize account manager
            from ..account_management import AccountManager
            account_manager_config = self.config_manager.get_dict('account_manager', {})
            self.account_manager = AccountManager(
                config=account_manager_config,
                broker_interface=None  # Will be set when broker interfaces are added
            )
            await self.account_manager.initialize()
            logger.info("Account Manager initialized")

            # Initialize broker interfaces (MT5 for now)
            await self._initialize_broker_interfaces()

            # Set performance tracker (should be passed from main.py later)
            # For now, we'll handle this gracefully
            self.performance_tracker = None  # Will be properly integrated later

            # Initialize messaging if configured
            await self._initialize_messaging()

            logger.info("Execution Handler initialization completed successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Execution Handler: {e}")
            raise

    async def _initialize_broker_interfaces(self) -> None:
        """Initialize broker interfaces."""
        try:
            # Initialize MetaApi interface (primary for Linux deployment)
            metaapi_config = self.config_manager.get_dict('brokers.metaapi', {})
            if metaapi_config.get('enabled', True):  # Default to enabled for Linux deployment
                from ..broker_interfaces.metaapi_broker import MetaApiBroker
                metaapi_broker = MetaApiBroker(metaapi_config)
                if await metaapi_broker.connect():
                    self.broker_interfaces[ExecutionVenue.METAAPI] = metaapi_broker

                    # Set broker interface for account manager
                    if self.account_manager:
                        self.account_manager.broker_interface = metaapi_broker

                    logger.info("MetaApi Broker initialized with trading capabilities")
                else:
                    logger.warning("Failed to initialize MetaApi Broker")

            # Legacy MT5 interface (deprecated - will show warning)
            mt5_config = self.config_manager.get_dict('brokers.mt5', {})
            if mt5_config.get('enabled', False):
                logger.warning("MT5 broker interface is deprecated. Use MetaApi for Linux deployment.")
                logger.warning("MT5 interface disabled - use 'brokers.metaapi.enabled: true' instead")

            # Add other broker interfaces here (OANDA, Interactive Brokers, etc.)
            oanda_config = self.config_manager.get_dict('brokers.oanda', {})
            if oanda_config.get('enabled', False):
                # Initialize OANDA broker
                logger.info("OANDA broker configuration found but not yet implemented")

            if not self.broker_interfaces:
                logger.warning("No broker interfaces initialized - trading will not be available")

        except Exception as e:
            logger.warning(f"Could not initialize broker interfaces: {e}")
            # Continue without broker interfaces for development mode

    async def _initialize_messaging(self) -> None:
        """Initialize messaging system using injected dependency."""
        try:
            if self.messaging_service:
                # Use injected messaging service
                self.messaging_enabled = True
                logger.info("✅ Using injected messaging service")

                # Send initialization status
                await self.messaging_service.publish_system_status(
                    component="execution_handler",
                    status="initialized",
                    details={
                        "broker_interfaces": len(self.broker_interfaces),
                        "account_manager": self.account_manager is not None
                    }
                )
            else:
                # Fallback to creating messaging service if not injected
                messaging_config = self.config_manager.get_dict('messaging', {})
                if messaging_config.get('enabled', False):
                    from ..messaging.messaging_service import MessagingServiceFactory

                    # Create config dict for messaging service (legacy compatibility)
                    legacy_config = self.config_manager.get_all()
                    self.messaging_service = await MessagingServiceFactory.create_messaging_service(legacy_config)

                    if self.messaging_service:
                        self.messaging_enabled = True
                        logger.info("✅ Created fallback messaging service")

                        # Send initialization status
                        await self.messaging_service.publish_system_status(
                            component="execution_handler",
                            status="initialized",
                            details={
                                "broker_interfaces": len(self.broker_interfaces),
                                "account_manager": self.account_manager is not None
                            }
                        )
                    else:
                        logger.warning("⚠️ Failed to create fallback messaging service")
                else:
                    logger.info("📝 Messaging system disabled in configuration")

        except Exception as e:
            logger.warning(f"Could not initialize messaging: {e}")
            self.messaging_service = None

    async def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            if self.account_manager:
                await self.account_manager.cleanup()

            for broker_interface in self.broker_interfaces.values():
                if hasattr(broker_interface, 'cleanup'):
                    await broker_interface.cleanup()

            # Cleanup messaging service (new dependency injection pattern)
            if self.messaging_service:
                await self.messaging_service.stop()
                logger.info("✅ Messaging service stopped")

            logger.info("Execution Handler cleanup completed")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    async def execute_trade_signal(self, signal: TradeSignal) -> ExecutionReport:
        """
        Execute a trade signal with full validation and optimization.

        Args:
            signal: Trade signal to execute

        Returns:
            ExecutionReport with execution details
        """
        execution_start = datetime.utcnow()
        execution_id = f"exec_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{signal.id[:8]}"

        logger.info(f"Executing trade signal {signal.id}: {signal.direction.value} {signal.symbol}")

        try:
            # Phase 1: Final signal validation
            is_valid, validation_errors = await self._final_signal_validation(signal)
            if not is_valid:
                return self._create_failed_report(
                    execution_id, signal, validation_errors, execution_start
                )

            # Phase 2: Account and risk checks
            account_info = await self.account_manager.get_account_info()
            risk_check_passed, risk_errors = await self._perform_final_risk_check(signal, account_info)
            if not risk_check_passed:
                return self._create_failed_report(
                    execution_id, signal, risk_errors, execution_start
                )

            # Phase 3: Position sizing and risk calculation
            position_size, risk_metrics = await self.risk_manager.calculate_position_size(
                signal=signal,
                account_info=account_info,
                current_price=await self._get_current_price(signal.symbol)
            )

            if position_size <= Decimal('0'):
                return self._create_failed_report(
                    execution_id, signal, ["Position size calculated as zero"], execution_start
                )

            # Phase 4: Create execution order
            execution_order = await self._create_execution_order(
                signal, position_size, risk_metrics, account_info
            )

            # Phase 5: Venue selection and optimization
            optimal_venue = await self._select_optimal_venue(signal.symbol, position_size)
            execution_order.venue = optimal_venue

            # Phase 6: Execute the order
            execution_report = await self._execute_order(execution_order, execution_start)

            # Phase 7: Post-execution processing
            await self._post_execution_processing(execution_report)

            return execution_report

        except Exception as e:
            logger.error(f"Trade execution failed for signal {signal.id}: {str(e)}")
            return self._create_failed_report(
                execution_id, signal, [f"Execution error: {str(e)}"], execution_start
            )

    async def _final_signal_validation(self, signal: TradeSignal) -> Tuple[bool, List[str]]:
        """Perform final comprehensive signal validation."""
        validation_errors = []

        try:
            # Basic signal integrity
            if not signal.symbol:
                validation_errors.append("Missing symbol")

            if not signal.direction:
                validation_errors.append("Missing direction")

            if signal.confidence < 0.3:
                validation_errors.append(f"Confidence too low: {signal.confidence:.3f}")

            # Price validation
            current_price = await self._get_current_price(signal.symbol)
            if not current_price:
                validation_errors.append(f"Cannot get current price for {signal.symbol}")

            # Stop loss and take profit validation
            if signal.stop_loss and current_price:
                if signal.direction == TradeDirection.BUY and signal.stop_loss >= current_price:
                    validation_errors.append("Stop loss too high for BUY order")
                elif signal.direction == TradeDirection.SELL and signal.stop_loss <= current_price:
                    validation_errors.append("Stop loss too low for SELL order")

            if signal.take_profit and current_price:
                if signal.direction == TradeDirection.BUY and signal.take_profit <= current_price:
                    validation_errors.append("Take profit too low for BUY order")
                elif signal.direction == TradeDirection.SELL and signal.take_profit >= current_price:
                    validation_errors.append("Take profit too high for SELL order")

            # Market hours validation
            if not await self._check_market_hours(signal.symbol):
                validation_errors.append(f"Market closed for {signal.symbol}")

            # Liquidity validation
            if not await self._check_symbol_liquidity(signal.symbol):
                validation_errors.append(f"Insufficient liquidity for {signal.symbol}")

            return len(validation_errors) == 0, validation_errors

        except Exception as e:
            logger.error(f"Signal validation failed: {str(e)}")
            return False, [f"Validation error: {str(e)}"]

    async def _perform_final_risk_check(self, signal: TradeSignal, account_info: AccountInfo) -> Tuple[bool, List[str]]:
        """Perform final risk checks before execution."""
        try:
            # Use risk manager's validation
            is_valid, errors = await self.risk_manager.validate_trade_signal(signal)

            # Additional execution-specific checks
            if account_info.balance <= Decimal('100'):
                errors.append("Account balance too low")

            if account_info.margin_available <= Decimal('50'):
                errors.append("Insufficient margin available")

            # Check for account restrictions
            if hasattr(account_info, 'trading_enabled') and not account_info.trading_enabled:
                errors.append("Trading disabled on account")

            return is_valid and len(errors) == 0, errors

        except Exception as e:
            logger.error(f"Final risk check failed: {str(e)}")
            return False, [f"Risk check error: {str(e)}"]
    async def _create_execution_order(self,
                                    signal: TradeSignal,
                                    position_size: Decimal,
                                    risk_metrics: RiskMetrics,
                                    account_info: AccountInfo) -> ExecutionOrder:
        """Create a comprehensive execution order from the signal."""
        try:
            # Determine order type based on signal characteristics
            order_type = await self._determine_optimal_order_type(signal, risk_metrics)

            # Get current market price
            current_price = await self._get_current_price(signal.symbol)

            # Calculate execution price based on order type
            execution_price = await self._calculate_execution_price(
                signal, current_price, order_type
            )

            execution_order = ExecutionOrder(
                signal_id=signal.id,
                symbol=signal.symbol,
                direction=signal.direction,
                order_type=order_type,
                quantity=position_size,
                price=execution_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                risk_metrics=risk_metrics,
                metadata={
                    'signal_confidence': signal.confidence,
                    'signal_confidence_level': signal.confidence_level.value,
                    'generating_agent': signal.generating_agent,
                    'supporting_agents': signal.supporting_agents,
                    'strategy': signal.strategy,
                    'account_id': account_info.account_id,
                    'original_signal': signal.dict() if hasattr(signal, 'dict') else str(signal)
                }
            )

            return execution_order

        except Exception as e:
            logger.error(f"Failed to create execution order: {str(e)}")
            raise ExecutionError(f"Order creation failed: {str(e)}")

    async def _determine_optimal_order_type(self, signal: TradeSignal, risk_metrics: RiskMetrics) -> OrderType:
        """Determine the optimal order type based on signal and market conditions."""
        try:
            # High confidence signals with low slippage risk -> Market orders
            if (signal.confidence >= 0.8 and
                risk_metrics.risk_level.value in ['VERY_LOW', 'LOW']):
                return OrderType.MARKET

            # Medium confidence signals -> Limit orders for better fills
            elif signal.confidence >= 0.6:
                return OrderType.LIMIT

            # Lower confidence signals -> More conservative limit orders
            else:
                return OrderType.LIMIT

        except Exception as e:
            logger.warning(f"Failed to determine order type: {str(e)}")
            return OrderType.LIMIT  # Conservative default

    async def _calculate_execution_price(self,
                                       signal: TradeSignal,
                                       current_price: Decimal,
                                       order_type: OrderType) -> Optional[Decimal]:
        """Calculate optimal execution price based on order type and market conditions."""
        try:
            if order_type == OrderType.MARKET:
                return None  # Market orders don't have a specific price

            # For limit orders, calculate optimal limit price
            if order_type == OrderType.LIMIT:
                # Get bid/ask spread
                spread = await self._get_bid_ask_spread(signal.symbol)

                if signal.direction == TradeDirection.BUY:
                    # For buy orders, place slightly below current price
                    limit_price = current_price - (spread * Decimal('0.3'))
                else:
                    # For sell orders, place slightly above current price
                    limit_price = current_price + (spread * Decimal('0.3'))

                return limit_price.quantize(Decimal('0.00001'), rounding=ROUND_HALF_UP)

            return current_price

        except Exception as e:
            logger.warning(f"Failed to calculate execution price: {str(e)}")
            return current_price

    async def _select_optimal_venue(self, symbol: str, position_size: Decimal) -> ExecutionVenue:
        """Select the optimal execution venue based on performance and conditions."""
        try:
            # Get venue performance data
            available_venues = list(self.broker_interfaces.keys())

            if not available_venues:
                return ExecutionVenue.DEMO  # Fallback

            # Simple venue selection logic (can be enhanced with ML)
            best_venue = available_venues[0]
            best_score = 0.0

            for venue in available_venues:
                score = await self._calculate_venue_score(venue, symbol, position_size)
                if score > best_score:
                    best_score = score
                    best_venue = venue

            return best_venue

        except Exception as e:
            logger.warning(f"Venue selection failed: {str(e)}")
            return ExecutionVenue.DEMO

    async def _calculate_venue_score(self, venue: ExecutionVenue, symbol: str, position_size: Decimal) -> float:
        """Calculate venue score based on historical performance."""
        try:
            venue_stats = self.venue_performance.get(venue, {})

            # Base score factors
            execution_speed = venue_stats.get('avg_execution_speed', 0.5)
            fill_rate = venue_stats.get('fill_rate', 0.8)
            avg_slippage = venue_stats.get('avg_slippage', 0.001)

            # Calculate composite score
            score = (
                execution_speed * 0.3 +
                fill_rate * 0.4 +
                (1.0 - avg_slippage) * 0.3
            )

            return max(0.0, min(1.0, score))

        except Exception as e:
            logger.warning(f"Venue score calculation failed: {str(e)}")
            return 0.5

    async def _execute_order(self, order: ExecutionOrder, execution_start: datetime) -> ExecutionReport:
        """Execute the order with retry logic and performance tracking."""
        execution_errors = []
        execution_warnings = []

        try:
            # Get broker interface
            broker_interface = self.broker_interfaces.get(order.venue)
            if not broker_interface:
                raise ExecutionError(f"No broker interface available for {order.venue.value}")

            # Add to active orders
            self.active_orders[order.order_id] = order
            order.status = ExecutionStatus.VALIDATED
            order.submitted_at = datetime.utcnow()

            # Execute with retry logic
            execution_success = False
            retry_count = 0

            while not execution_success and retry_count < self.retry_attempts:
                try:
                    # Submit order to broker
                    broker_order_id = await self._submit_order_to_broker(broker_interface, order)
                    order.status = ExecutionStatus.SUBMITTED

                    # Wait for fill with timeout
                    fill_result = await self._wait_for_fill(broker_interface, broker_order_id, order)

                    if fill_result['success']:
                        execution_success = True
                        await self._process_successful_fill(order, fill_result)
                    else:
                        execution_warnings.extend(fill_result.get('warnings', []))
                        retry_count += 1

                        if retry_count < self.retry_attempts:
                            await asyncio.sleep(self.retry_delay_seconds)

                except Exception as e:
                    retry_count += 1
                    error_msg = f"Execution attempt {retry_count} failed: {str(e)}"
                    execution_errors.append(error_msg)
                    logger.warning(error_msg)

                    if retry_count < self.retry_attempts:
                        await asyncio.sleep(self.retry_delay_seconds)

            # Calculate execution metrics
            execution_time = (datetime.utcnow() - execution_start).total_seconds()

            # Create execution report
            report = ExecutionReport(
                execution_id=f"exec_{order.order_id}",
                order=order,
                success=execution_success,
                execution_time_seconds=execution_time,
                total_slippage=order.slippage or Decimal('0'),
                total_costs=order.transaction_costs,
                fill_rate=float(order.filled_quantity / order.quantity) if order.quantity > 0 else 0.0,
                warnings=execution_warnings,
                errors=execution_errors,
                performance_grade=self._calculate_execution_grade(order, execution_success)
            )

            # Remove from active orders
            if order.order_id in self.active_orders:
                del self.active_orders[order.order_id]

            # Update statistics
            self._update_execution_statistics(report)

            return report

        except Exception as e:
            error_msg = f"Order execution failed: {str(e)}"
            execution_errors.append(error_msg)
            logger.error(error_msg)

            # Clean up
            if order.order_id in self.active_orders:
                del self.active_orders[order.order_id]

            execution_time = (datetime.utcnow() - execution_start).total_seconds()

            return ExecutionReport(
                execution_id=f"exec_{order.order_id}",
                order=order,
                success=False,
                execution_time_seconds=execution_time,
                total_slippage=Decimal('0'),
                total_costs=Decimal('0'),
                fill_rate=0.0,
                warnings=execution_warnings,
                errors=execution_errors,
                performance_grade=DealGrade.F
            )

    async def _submit_order_to_broker(self, broker_interface, order: ExecutionOrder) -> str:
        """Submit order to broker and return broker order ID."""
        try:
            # Map order direction to MT5 order type
            if order.direction == TradeDirection.BUY:
                if order.order_type == OrderType.MARKET:
                    mt5_order_type = "BUY"
                elif order.order_type == OrderType.LIMIT:
                    mt5_order_type = "BUY_LIMIT"
                elif order.order_type == OrderType.STOP:
                    mt5_order_type = "BUY_STOP"
                else:
                    mt5_order_type = "BUY"
            else:  # SELL
                if order.order_type == OrderType.MARKET:
                    mt5_order_type = "SELL"
                elif order.order_type == OrderType.LIMIT:
                    mt5_order_type = "SELL_LIMIT"
                elif order.order_type == OrderType.STOP:
                    mt5_order_type = "SELL_STOP"
                else:
                    mt5_order_type = "SELL"

            # Submit order to MT5
            result = await broker_interface.place_order(
                symbol=order.symbol,
                order_type=mt5_order_type,
                volume=float(order.quantity),
                price=float(order.price) if order.price else None,
                sl=float(order.stop_loss) if order.stop_loss else None,
                tp=float(order.take_profit) if order.take_profit else None,
                comment=f"AUJ:{order.order_id}"
            )

            if result.get('success'):
                broker_order_id = result.get('order_id') or result.get('deal_id')
                if broker_order_id:
                    logger.info(f"Order {order.order_id} submitted to {order.venue.value} as {broker_order_id}")
                    return str(broker_order_id)
                else:
                    raise BrokerError("Order submitted but no order ID returned")
            else:
                error_msg = result.get('error', 'Unknown broker error')
                raise BrokerError(f"Broker rejected order: {error_msg}")

        except Exception as e:
            logger.error(f"Failed to submit order to broker: {str(e)}")
            raise BrokerError(f"Failed to submit order to broker: {str(e)}")

    async def _wait_for_fill(self, broker_interface, broker_order_id: str, order: ExecutionOrder) -> Dict[str, Any]:
        """Wait for order fill with timeout and enhanced monitoring."""
        try:
            timeout_time = datetime.utcnow() + timedelta(seconds=self.order_timeout_seconds)
            check_interval = 0.1  # More frequent checks for better responsiveness
            last_status = None
            partial_fills = []

            logger.info(f"Monitoring fill for order {order.order_id}, broker ID: {broker_order_id}")

            while datetime.utcnow() < timeout_time:
                if order.order_type == OrderType.MARKET:
                    # Enhanced market order monitoring
                    fill_result = await self._check_market_order_fill(
                        broker_interface, broker_order_id, order, partial_fills
                    )
                    if fill_result['success'] or fill_result.get('terminal_error'):
                        return fill_result

                else:
                    # Enhanced pending order monitoring
                    fill_result = await self._check_pending_order_status(
                        broker_interface, broker_order_id, order, partial_fills
                    )
                    if fill_result['success'] or fill_result.get('terminal_error'):
                        return fill_result

                # Log status changes
                current_status = fill_result.get('status')
                if current_status != last_status:
                    logger.info(f"Order {order.order_id} status: {current_status}")
                    last_status = current_status

                # Adaptive wait based on order type
                await asyncio.sleep(check_interval)
                if order.order_type == OrderType.MARKET:
                    check_interval = min(check_interval * 1.1, 1.0)  # Slower checks for market orders

            # Timeout reached - enhanced timeout handling
            return await self._handle_fill_timeout(broker_interface, broker_order_id, order, partial_fills)

        except Exception as e:
            logger.error(f"Error waiting for fill: {str(e)}")
            return {
                'success': False,
                'fill_data': {'status': 'ERROR', 'error': str(e)},
                'warnings': [f'Error during fill monitoring: {str(e)}']
            }

    async def _check_market_order_fill(self, broker_interface, broker_order_id: str,
                                     order: ExecutionOrder, partial_fills: List) -> Dict[str, Any]:
        """Enhanced market order fill checking with partial fill support."""
        try:
            # Check positions for filled market orders
            positions = await broker_interface.get_positions(order.symbol)

            # Look for positions matching our order
            for position in positions:
                if (position.get('comment', '').startswith(f"AUJ:{order.order_id}") or
                    str(position.get('ticket')) == str(broker_order_id)):

                    fill_price = Decimal(str(position.get('price_open', 0)))
                    fill_quantity = Decimal(str(position.get('volume', 0)))
                    commission = Decimal(str(position.get('commission', 0)))

                    # Calculate slippage
                    expected_price = order.price or fill_price
                    slippage = abs(fill_price - expected_price)

                    return {
                        'success': True,
                        'status': 'FILLED',
                        'fill_data': {
                            'status': 'FILLED',
                            'fill_price': fill_price,
                            'fill_quantity': fill_quantity,
                            'commission': commission,
                            'position_id': position.get('ticket'),
                            'slippage': slippage,
                            'execution_time': datetime.utcnow()
                        },
                        'warnings': []
                    }

            # Check for rejection or errors
            if hasattr(broker_interface, 'get_last_error'):
                last_error = await broker_interface.get_last_error()
                if last_error:
                    return {
                        'success': False,
                        'status': 'REJECTED',
                        'fill_data': {'status': 'REJECTED', 'error': last_error},
                        'warnings': [f'Order rejected: {last_error}'],
                        'terminal_error': True
                    }

            return {'success': False, 'status': 'PENDING'}

        except Exception as e:
            logger.error(f"Error checking market order fill: {e}")
            return {
                'success': False,
                'status': 'ERROR',
                'fill_data': {'status': 'ERROR', 'error': str(e)},
                'warnings': [f'Error checking fill: {str(e)}'],
                'terminal_error': True
            }

    async def _check_pending_order_status(self, broker_interface, broker_order_id: str,
                                        order: ExecutionOrder, partial_fills: List) -> Dict[str, Any]:
        """Enhanced pending order status checking with comprehensive monitoring."""
        try:
            # Check if order is still pending
            if hasattr(broker_interface, 'get_order_status'):
                order_status = await broker_interface.get_order_status(broker_order_id)
                if order_status:
                    status = order_status.get('status', 'UNKNOWN')

                    if status == 'FILLED':
                        return {
                            'success': True,
                            'status': 'FILLED',
                            'fill_data': {
                                'status': 'FILLED',
                                'fill_price': Decimal(str(order_status.get('fill_price', 0))),
                                'fill_quantity': Decimal(str(order_status.get('fill_quantity', 0))),
                                'commission': Decimal(str(order_status.get('commission', 0))),
                                'position_id': order_status.get('position_id'),
                                'execution_time': datetime.utcnow()
                            },
                            'warnings': []
                        }
                    elif status == 'PARTIALLY_FILLED':
                        # Handle partial fills
                        partial_fill = {
                            'quantity': Decimal(str(order_status.get('filled_quantity', 0))),
                            'price': Decimal(str(order_status.get('fill_price', 0))),
                            'time': datetime.utcnow()
                        }
                        partial_fills.append(partial_fill)
                        return {'success': False, 'status': 'PARTIALLY_FILLED'}

                    elif status in ['REJECTED', 'CANCELLED', 'EXPIRED']:
                        return {
                            'success': False,
                            'status': status,
                            'fill_data': {'status': status, 'reason': order_status.get('reason', 'Unknown')},
                            'warnings': [f'Order {status.lower()}: {order_status.get("reason", "Unknown")}'],
                            'terminal_error': True
                        }

            # Fallback: check positions (in case order was filled but not properly tracked)
            positions = await broker_interface.get_positions(order.symbol)
            for position in positions:
                if str(position.get('ticket')) == str(broker_order_id):
                    return {
                        'success': True,
                        'status': 'FILLED',
                        'fill_data': {
                            'status': 'FILLED',
                            'fill_price': Decimal(str(position.get('price_open', 0))),
                            'fill_quantity': Decimal(str(position.get('volume', 0))),
                            'commission': Decimal(str(position.get('commission', 0))),
                            'position_id': position.get('ticket'),
                            'execution_time': datetime.utcnow()
                        },
                        'warnings': []
                    }

            return {'success': False, 'status': 'PENDING'}

        except Exception as e:
            logger.error(f"Error checking pending order status: {e}")
            return {
                'success': False,
                'status': 'ERROR',
                'fill_data': {'status': 'ERROR', 'error': str(e)},
                'warnings': [f'Error checking order status: {str(e)}'],
                'terminal_error': True
            }

    async def _handle_fill_timeout(self, broker_interface, broker_order_id: str,
                                 order: ExecutionOrder, partial_fills: List) -> Dict[str, Any]:
        """Enhanced timeout handling with cancellation attempts."""
        try:
            logger.warning(f"Order {order.order_id} timed out, attempting recovery")

            # Try to cancel if it's a pending order
            if order.order_type != OrderType.MARKET:
                cancel_result = await self.cancel_order_properly(broker_interface, broker_order_id, order)
                if cancel_result['success']:
                    return {
                        'success': False,
                        'status': 'CANCELLED_ON_TIMEOUT',
                        'fill_data': {
                            'status': 'CANCELLED_ON_TIMEOUT',
                            'partial_fills': partial_fills,
                            'timeout_seconds': self.order_timeout_seconds
                        },
                        'warnings': ['Order cancelled due to timeout']
                    }

            # If cancellation failed or market order, check one more time
            final_check = await self._check_market_order_fill(broker_interface, broker_order_id, order, partial_fills)
            if final_check['success']:
                return final_check

            return {
                'success': False,
                'status': 'TIMEOUT',
                'fill_data': {
                    'status': 'TIMEOUT',
                    'partial_fills': partial_fills,
                    'timeout_seconds': self.order_timeout_seconds
                },
                'warnings': ['Order timed out without fill']
            }

        except Exception as e:
            logger.error(f"Error handling timeout: {e}")
            return {
                'success': False,
                'status': 'TIMEOUT_ERROR',
                'fill_data': {'status': 'TIMEOUT_ERROR', 'error': str(e)},
                'warnings': [f'Error during timeout handling: {str(e)}']
            }

            # Attempt to cancel timed-out order
            try:
                if hasattr(broker_interface, 'cancel_order'):
                    cancel_result = await broker_interface.cancel_order(broker_order_id)
                    logger.info(f"Attempted to cancel order {broker_order_id}: {cancel_result}")
            except Exception as cancel_error:
                logger.warning(f"Failed to cancel order {broker_order_id}: {cancel_error}")

            return {
                'success': False,
                'fill_data': {'status': 'TIMEOUT'},
                'warnings': ['Order execution timeout']
            }

        except Exception as e:
            logger.error(f"Error waiting for fill: {str(e)}")
            return {
                'success': False,
                'fill_data': {'status': 'ERROR'},
                'warnings': [f"Fill monitoring error: {str(e)}"]
            }

        except Exception as e:
            logger.error(f"Error waiting for fill: {str(e)}")
            return {
                'success': False,
                'fill_data': {'status': 'ERROR'},
                'warnings': [f"Fill monitoring error: {str(e)}"]
            }

    async def _get_order_status_from_positions(self, broker_interface, order: ExecutionOrder, broker_order_id: str) -> Optional[Dict[str, Any]]:
        """Get order status by checking positions and trades."""
        try:
            # Check current positions
            positions = await broker_interface.get_positions(order.symbol)

            for position in positions:
                # Match by comment or ticket
                if (position.get('comment', '').startswith(f"AUJ:{order.order_id}") or
                    str(position.get('ticket')) == str(broker_order_id)):

                    return {
                        'status': 'FILLED',
                        'fill_price': position.get('price_open'),
                        'fill_quantity': position.get('volume'),
                        'commission': position.get('commission', 0),
                        'position_id': position.get('ticket'),
                        'current_price': position.get('price_current'),
                        'profit': position.get('profit', 0)
                    }

            # If not found in positions, the order might have been rejected or not yet executed
            return None

        except Exception as e:
            logger.error(f"Error getting order status from positions: {e}")
            return None

    async def _process_successful_fill(self, order: ExecutionOrder, fill_result: Dict[str, Any]):
        """Process successful order fill and update order details."""
        try:
            fill_data = fill_result['fill_data']

            # Update order with fill information
            order.status = ExecutionStatus.FILLED
            order.filled_at = datetime.utcnow()
            order.filled_price = Decimal(str(fill_data.get('fill_price', 0)))
            order.filled_quantity = Decimal(str(fill_data.get('fill_quantity', 0)))
            order.remaining_quantity = order.quantity - order.filled_quantity
            order.transaction_costs = Decimal(str(fill_data.get('commission', 0)))

            # Calculate slippage
            if order.price and order.filled_price:
                if order.direction == TradeDirection.BUY:
                    order.slippage = order.filled_price - order.price
                else:
                    order.slippage = order.price - order.filled_price

            # Calculate execution time
            if order.submitted_at:
                execution_ms = int((order.filled_at - order.submitted_at).total_seconds() * 1000)
                order.execution_time_ms = execution_ms

            logger.info(f"Order {order.order_id} filled: {order.filled_quantity} @ {order.filled_price}")

        except Exception as e:
            logger.error(f"Failed to process successful fill: {str(e)}")
            raise ExecutionError(f"Fill processing failed: {str(e)}")
    def _calculate_execution_grade(self, order: ExecutionOrder, success: bool) -> DealGrade:
        """Calculate execution quality grade."""
        try:
            if not success:
                return DealGrade.F

            score = 100.0  # Start with perfect score

            # Deduct points for slippage
            if order.slippage:
                slippage_percent = abs(float(order.slippage)) / float(order.filled_price or 1) * 100
                if slippage_percent > self.max_slippage_percent:
                    score -= min(30, slippage_percent * 10)  # Max 30 point deduction
                else:
                    score -= slippage_percent * 5  # Lighter penalty for acceptable slippage

            # Deduct points for execution time
            if order.execution_time_ms:
                if order.execution_time_ms > 5000:  # > 5 seconds
                    score -= min(20, (order.execution_time_ms - 5000) / 1000)  # Max 20 point deduction

            # Deduct points for partial fills
            if order.filled_quantity < order.quantity:
                fill_ratio = float(order.filled_quantity / order.quantity)
                score -= (1.0 - fill_ratio) * 25  # Max 25 point deduction

            # Deduct points for high transaction costs
            if order.transaction_costs and order.filled_price and order.filled_quantity:
                cost_percent = float(order.transaction_costs) / (float(order.filled_price) * float(order.filled_quantity)) * 100
                if cost_percent > 0.1:  # > 0.1%
                    score -= min(15, cost_percent * 50)  # Max 15 point deduction

            # Convert score to grade
            if score >= 95:
                return DealGrade.A_PLUS
            elif score >= 90:
                return DealGrade.A
            elif score >= 85:
                return DealGrade.A_MINUS
            elif score >= 80:
                return DealGrade.B_PLUS
            elif score >= 75:
                return DealGrade.B
            elif score >= 70:
                return DealGrade.B_MINUS
            elif score >= 65:
                return DealGrade.C_PLUS
            elif score >= 60:
                return DealGrade.C
            elif score >= 55:
                return DealGrade.C_MINUS
            elif score >= 50:
                return DealGrade.D
            else:
                return DealGrade.F

        except Exception as e:
            logger.warning(f"Execution grade calculation failed: {str(e)}")
            return DealGrade.C  # Default grade

    def _update_execution_statistics(self, report: ExecutionReport):
        """Update execution performance statistics."""
        try:
            self.total_executions += 1

            if report.success:
                self.successful_executions += 1
                self.total_slippage += report.total_slippage
                self.total_costs += report.total_costs

                # Update venue performance
                venue = report.order.venue
                if venue not in self.venue_performance:
                    self.venue_performance[venue] = {
                        'total_executions': 0,
                        'successful_executions': 0,
                        'total_slippage': 0.0,
                        'total_execution_time': 0.0
                    }

                venue_stats = self.venue_performance[venue]
                venue_stats['total_executions'] += 1
                venue_stats['successful_executions'] += 1
                venue_stats['total_slippage'] += float(report.total_slippage)
                venue_stats['total_execution_time'] += report.execution_time_seconds

                # Calculate derived metrics
                venue_stats['success_rate'] = venue_stats['successful_executions'] / venue_stats['total_executions']
                venue_stats['avg_slippage'] = venue_stats['total_slippage'] / venue_stats['successful_executions']
                venue_stats['avg_execution_speed'] = venue_stats['total_execution_time'] / venue_stats['successful_executions']
                venue_stats['fill_rate'] = report.fill_rate

                # Update symbol statistics
                symbol = report.order.symbol
                if symbol not in self.symbol_execution_stats:
                    self.symbol_execution_stats[symbol] = {
                        'total_executions': 0,
                        'successful_executions': 0,
                        'avg_slippage': 0.0,
                        'avg_execution_time': 0.0
                    }

                symbol_stats = self.symbol_execution_stats[symbol]
                symbol_stats['total_executions'] += 1
                symbol_stats['successful_executions'] += 1

                # Update averages
                old_avg_slippage = symbol_stats['avg_slippage']
                old_avg_time = symbol_stats['avg_execution_time']
                n = symbol_stats['successful_executions']

                symbol_stats['avg_slippage'] = (old_avg_slippage * (n-1) + float(report.total_slippage)) / n
                symbol_stats['avg_execution_time'] = (old_avg_time * (n-1) + report.execution_time_seconds) / n

            # Add to execution history
            self.execution_history.append(report)
            if len(self.execution_history) > 1000:  # Keep last 1000 executions
                self.execution_history.pop(0)

        except Exception as e:
            logger.error(f"Failed to update execution statistics: {str(e)}")

    async def _post_execution_processing(self, report: ExecutionReport):
        """Perform post-execution processing and notifications."""
        try:
            # Record execution in performance tracker
            if self.performance_tracker and report.success:
                await self.performance_tracker.record_execution(
                    signal_id=report.order.signal_id,
                    execution_report=report
                )

            # Update risk manager with new position
            if report.success and self.risk_manager:
                await self.risk_manager.update_position_risk(
                    position_id=report.order.order_id,
                    current_pnl=Decimal('0')  # Initial PnL is zero
                )

            # Log execution result
            if report.success:
                logger.info(f"Execution successful: {report.order.symbol} "
                          f"{report.order.direction.value} {report.order.filled_quantity} "
                          f"@ {report.order.filled_price} (Grade: {report.performance_grade.value})")
            else:
                logger.warning(f"Execution failed: {report.order.symbol} "
                             f"{report.order.direction.value} - Errors: {', '.join(report.errors)}")

        except Exception as e:
            logger.error(f"Post-execution processing failed: {str(e)}")

    def _create_failed_report(self,
                            execution_id: str,
                            signal: TradeSignal,
                            errors: List[str],
                            execution_start: datetime) -> ExecutionReport:
        """Create a failed execution report."""
        execution_time = (datetime.utcnow() - execution_start).total_seconds()

        failed_order = ExecutionOrder(
            signal_id=signal.id,
            symbol=signal.symbol,
            direction=signal.direction,
            status=ExecutionStatus.REJECTED,
            validation_errors=errors
        )

        return ExecutionReport(
            execution_id=execution_id,
            order=failed_order,
            success=False,
            execution_time_seconds=execution_time,
            total_slippage=Decimal('0'),
            total_costs=Decimal('0'),
            fill_rate=0.0,
            warnings=[],
            errors=errors,
            performance_grade=DealGrade.F
        )

    # Helper methods for market data and validation

    async def _get_current_price(self, symbol: str) -> Optional[Decimal]:
        """Get current market price for symbol."""
        try:
            # Get real market price from broker interfaces
            for venue, broker_interface in self.broker_interfaces.items():
                if hasattr(broker_interface, 'get_current_price'):
                    price_data = await broker_interface.get_current_price(symbol)
                    if price_data and 'bid' in price_data and 'ask' in price_data:
                        # Use mid price for execution calculations
                        bid = Decimal(str(price_data['bid']))
                        ask = Decimal(str(price_data['ask']))
                        mid_price = (bid + ask) / Decimal('2')
                        logger.debug(f"Current price for {symbol}: {mid_price}")
                        return mid_price

            # Fallback: no broker interfaces available
            logger.warning(f"No price data available for {symbol} - no broker interfaces connected")
            return None

        except Exception as e:
            logger.warning(f"Failed to get current price for {symbol}: {str(e)}")
            return None

    async def _get_bid_ask_spread(self, symbol: str) -> Decimal:
        """Get current bid-ask spread for symbol."""
        try:
            # This would typically fetch from market data provider
            # For now, return a typical forex spread
            return Decimal('0.0002')  # 2 pips for major pairs
        except Exception as e:
            logger.warning(f"Failed to get spread for {symbol}: {str(e)}")
            return Decimal('0.0005')  # Conservative fallback

    async def _check_market_hours(self, symbol: str) -> bool:
        """Check if market is open for the symbol."""
        try:
            # Simplified market hours check
            # In reality, this would check specific market hours for each instrument
            current_time = datetime.utcnow()
            current_hour = current_time.hour
            current_weekday = current_time.weekday()

            # Basic forex market hours (24/5)
            if current_weekday < 5:  # Monday to Friday
                return True
            elif current_weekday == 6 and current_hour >= 21:  # Sunday after 21:00 UTC
                return True
            else:
                return False  # Weekend

        except Exception as e:
            logger.warning(f"Market hours check failed for {symbol}: {str(e)}")
            return True  # Default to allowing trading

    async def _check_symbol_liquidity(self, symbol: str) -> bool:
        """Check if symbol has sufficient liquidity."""
        try:
            # This would typically check recent volume, spread, and market depth
            # For now, return True for major currency pairs
            major_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD']
            return symbol in major_pairs
        except Exception as e:
            logger.warning(f"Liquidity check failed for {symbol}: {str(e)}")
            return True  # Default to allowing trading

    # Public interface methods

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an active order."""
        try:
            if order_id not in self.active_orders:
                logger.warning(f"Order {order_id} not found in active orders")
                return False

            order = self.active_orders[order_id]
            broker_interface = self.broker_interfaces.get(order.venue)

            if not broker_interface:
                logger.error(f"No broker interface for venue {order.venue.value}")
                return False

            # Cancel with broker
            success = await broker_interface.cancel_order(order_id)

            if success:
                order.status = ExecutionStatus.CANCELLED
                del self.active_orders[order_id]
                logger.info(f"Order {order_id} cancelled successfully")

            return success

        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {str(e)}")
            return False

    async def cancel_order_properly(self, broker_interface, broker_order_id: str, order: ExecutionOrder) -> Dict[str, Any]:
        """
        Enhanced order cancellation with proper error handling and validation.

        Args:
            broker_interface: Broker interface to use for cancellation
            broker_order_id: Broker-specific order ID
            order: ExecutionOrder object

        Returns:
            Dict with cancellation result and details
        """
        try:
            logger.info(f"Attempting to cancel order {order.order_id} (broker ID: {broker_order_id})")

            # First verify order still exists and is cancellable
            order_status = await self._check_order_cancellable(broker_interface, broker_order_id, order)
            if not order_status['cancellable']:
                return {
                    'success': False,
                    'reason': order_status['reason'],
                    'order_id': order.order_id,
                    'broker_order_id': broker_order_id
                }

            # Attempt cancellation with broker
            cancel_attempts = 0
            max_attempts = 3

            while cancel_attempts < max_attempts:
                try:
                    if hasattr(broker_interface, 'cancel_order'):
                        cancel_result = await broker_interface.cancel_order(int(broker_order_id))
                    else:
                        # Fallback for interfaces without cancel_order method
                        cancel_result = await self._fallback_cancel_order(broker_interface, broker_order_id, order)

                    if cancel_result.get('success'):
                        # Update order status
                        order.status = ExecutionStatus.CANCELLED
                        order.cancelled_time = datetime.utcnow()

                        # Remove from active orders if present
                        if order.order_id in self.active_orders:
                            del self.active_orders[order.order_id]

                        logger.info(f"Successfully cancelled order {order.order_id}")

                        return {
                            'success': True,
                            'order_id': order.order_id,
                            'broker_order_id': broker_order_id,
                            'cancellation_time': datetime.utcnow(),
                            'attempts': cancel_attempts + 1,
                            'broker_result': cancel_result
                        }
                    else:
                        cancel_attempts += 1
                        error_msg = cancel_result.get('error', 'Unknown error')
                        logger.warning(f"Cancel attempt {cancel_attempts} failed: {error_msg}")

                        if cancel_attempts < max_attempts:
                            await asyncio.sleep(0.5)  # Brief delay before retry

                except Exception as attempt_error:
                    cancel_attempts += 1
                    logger.warning(f"Cancel attempt {cancel_attempts} exception: {attempt_error}")

                    if cancel_attempts < max_attempts:
                        await asyncio.sleep(0.5)

            # All attempts failed
            logger.error(f"Failed to cancel order {order.order_id} after {max_attempts} attempts")
            return {
                'success': False,
                'reason': f'Failed after {max_attempts} attempts',
                'order_id': order.order_id,
                'broker_order_id': broker_order_id,
                'attempts': cancel_attempts
            }

        except Exception as e:
            logger.error(f"Error in cancel_order_properly: {e}")
            return {
                'success': False,
                'reason': f'Exception during cancellation: {str(e)}',
                'order_id': order.order_id,
                'broker_order_id': broker_order_id
            }

    async def _check_order_cancellable(self, broker_interface, broker_order_id: str, order: ExecutionOrder) -> Dict[str, Any]:
        """Check if order can be cancelled."""
        try:
            # Check if order still exists as pending
            if hasattr(broker_interface, 'get_order_status'):
                order_status = await broker_interface.get_order_status(broker_order_id)
                if order_status:
                    status = order_status.get('status', 'UNKNOWN')
                    if status in ['FILLED', 'CANCELLED', 'REJECTED', 'EXPIRED']:
                        return {
                            'cancellable': False,
                            'reason': f'Order already {status.lower()}'
                        }
                    elif status == 'PENDING':
                        return {'cancellable': True, 'reason': 'Order is pending'}

            # Fallback: check if it exists in broker's pending orders
            if hasattr(broker_interface, 'get_pending_orders'):
                pending_orders = await broker_interface.get_pending_orders()
                for pending_order in pending_orders:
                    if str(pending_order.get('ticket')) == str(broker_order_id):
                        return {'cancellable': True, 'reason': 'Order found in pending orders'}

            # If we can't verify, assume it might be cancellable
            return {'cancellable': True, 'reason': 'Unable to verify status, attempting cancellation'}

        except Exception as e:
            logger.warning(f"Error checking order cancellable status: {e}")
            return {'cancellable': True, 'reason': 'Error checking status, attempting cancellation'}

    async def _fallback_cancel_order(self, broker_interface, broker_order_id: str, order: ExecutionOrder) -> Dict[str, Any]:
        """Fallback cancellation method for brokers without native cancel_order."""
        try:
            # Some brokers might use close_position for market orders that became positions
            if order.order_type == OrderType.MARKET:
                # Check if it became a position
                positions = await broker_interface.get_positions(order.symbol)
                for position in positions:
                    if str(position.get('ticket')) == str(broker_order_id):
                        # Close the position instead
                        close_result = await broker_interface.close_position(
                            order.symbol,
                            position_id=position.get('ticket')
                        )
                        return close_result

            # For other order types, return failure
            return {
                'success': False,
                'error': 'Broker interface does not support order cancellation'
            }

        except Exception as e:
            return {
                'success': False,
                'error': f'Fallback cancellation failed: {str(e)}'
            }

    def get_active_orders(self) -> Dict[str, ExecutionOrder]:
        """Get all currently active orders."""
        return self.active_orders.copy()

    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get comprehensive execution statistics."""
        try:
            success_rate = self.successful_executions / max(self.total_executions, 1)
            avg_slippage = float(self.total_slippage) / max(self.successful_executions, 1)
            avg_costs = float(self.total_costs) / max(self.successful_executions, 1)

            return {
                'total_executions': self.total_executions,
                'successful_executions': self.successful_executions,
                'success_rate': success_rate,
                'average_slippage': avg_slippage,
                'average_costs': avg_costs,
                'active_orders_count': len(self.active_orders),
                'venue_performance': self.venue_performance.copy(),
                'symbol_statistics': self.symbol_execution_stats.copy(),
                'recent_executions': len(self.execution_history),
                'configuration': {
                    'max_slippage_percent': self.max_slippage_percent,
                    'order_timeout_seconds': self.order_timeout_seconds,
                    'retry_attempts': self.retry_attempts,
                    'min_order_value': float(self.min_order_value)
                }
            }
        except Exception as e:
            logger.error(f"Failed to get execution statistics: {str(e)}")
            return {'error': str(e)}

    def get_recent_executions(self, count: int = 50) -> List[Dict[str, Any]]:
        """Get recent execution reports."""
        try:
            recent_reports = self.execution_history[-count:] if count > 0 else self.execution_history

            return [
                {
                    'execution_id': report.execution_id,
                    'symbol': report.order.symbol,
                    'direction': report.order.direction.value,
                    'quantity': float(report.order.quantity),
                    'filled_quantity': float(report.order.filled_quantity),
                    'filled_price': float(report.order.filled_price) if report.order.filled_price else None,
                    'success': report.success,
                    'execution_time_seconds': report.execution_time_seconds,
                    'slippage': float(report.total_slippage),
                    'costs': float(report.total_costs),
                    'fill_rate': report.fill_rate,
                    'grade': report.performance_grade.value,
                    'venue': report.order.venue.value,
                    'timestamp': report.timestamp.isoformat(),
                    'errors': report.errors,
                    'warnings': report.warnings
                }
                for report in recent_reports
            ]
        except Exception as e:
            logger.error(f"Failed to get recent executions: {str(e)}")
            return []

    async def update_configuration(self, new_config: Dict[str, Any]):
        """Update execution handler configuration."""
        try:
            for key, value in new_config.items():
                if hasattr(self, key):
                    old_value = getattr(self, key)
                    setattr(self, key, value)
                    logger.info(f"Execution config updated: {key} {old_value} -> {value}")
                else:
                    logger.warning(f"Unknown execution config parameter: {key}")
        except Exception as e:
            logger.error(f"Failed to update configuration: {str(e)}")
            raise ExecutionError(f"Configuration update failed: {str(e)}")

    def __str__(self) -> str:
        return f"ExecutionHandler(active_orders={len(self.active_orders)}, success_rate={self.successful_executions/max(self.total_executions,1):.2%})"

    def __repr__(self) -> str:
        return (f"ExecutionHandler(total_executions={self.total_executions}, "
                f"successful_executions={self.successful_executions}, "
                f"active_orders={len(self.active_orders)}, "
                f"venues={list(self.broker_interfaces.keys())})")

# Additional utility classes and types for execution handling

class ExecutionMetrics:
    """Metrics for execution performance analysis."""

    def __init__(self):
        self.total_volume = Decimal('0')
        self.total_trades = 0
        self.successful_trades = 0
        self.total_slippage = Decimal('0')
        self.total_costs = Decimal('0')
        self.execution_times = []
        self.grade_distribution = {grade: 0 for grade in DealGrade}
        self.venue_metrics = {}
        self.symbol_metrics = {}

    def add_execution(self, report: ExecutionReport):
        """Add execution report to metrics."""
        self.total_trades += 1

        if report.success:
            self.successful_trades += 1
            self.total_volume += report.order.filled_quantity
            self.total_slippage += report.total_slippage
            self.total_costs += report.total_costs
            self.execution_times.append(report.execution_time_seconds)
            self.grade_distribution[report.performance_grade] += 1

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        return self.successful_trades / max(self.total_trades, 1)

    @property
    def average_slippage(self) -> Decimal:
        """Calculate average slippage."""
        return self.total_slippage / max(self.successful_trades, 1)

    @property
    def average_costs(self) -> Decimal:
        """Calculate average transaction costs."""
        return self.total_costs / max(self.successful_trades, 1)

    @property
    def average_execution_time(self) -> float:
        """Calculate average execution time."""
        return sum(self.execution_times) / max(len(self.execution_times), 1)


class ExecutionQualityMonitor:
    """Monitor and analyze execution quality over time."""

    def __init__(self, lookback_periods: int = 24):
        self.lookback_periods = lookback_periods
        self.hourly_metrics = []
        self.quality_alerts = []

    def update_hourly_metrics(self, execution_handler: ExecutionHandler):
        """Update hourly execution metrics."""
        try:
            current_metrics = ExecutionMetrics()
            current_hour = datetime.utcnow().replace(minute=0, second=0, microsecond=0)

            # Analyze executions from the last hour
            for report in execution_handler.execution_history:
                if report.timestamp >= current_hour:
                    current_metrics.add_execution(report)

            self.hourly_metrics.append({
                'timestamp': current_hour,
                'metrics': current_metrics,
                'total_executions': current_metrics.total_trades,
                'success_rate': current_metrics.success_rate,
                'avg_slippage': float(current_metrics.average_slippage),
                'avg_execution_time': current_metrics.average_execution_time
            })

            # Keep only recent periods
            if len(self.hourly_metrics) > self.lookback_periods:
                self.hourly_metrics.pop(0)

            # Check for quality degradation
            self._check_quality_alerts(current_metrics)

        except Exception as e:
            logger.error(f"Failed to update hourly metrics: {str(e)}")

    def _check_quality_alerts(self, current_metrics: ExecutionMetrics):
        """Check for execution quality issues."""
        try:
            alerts = []

            # Success rate alert
            if current_metrics.success_rate < 0.95:  # Less than 95%
                alerts.append(f"Low success rate: {current_metrics.success_rate:.2%}")

            # High slippage alert
            if current_metrics.average_slippage > Decimal('0.0005'):  # > 5 pips
                alerts.append(f"High average slippage: {current_metrics.average_slippage}")

            # Slow execution alert
            if current_metrics.average_execution_time > 3.0:  # > 3 seconds
                alerts.append(f"Slow execution: {current_metrics.average_execution_time:.2f}s")

            # High cost alert
            if current_metrics.average_costs > Decimal('0.001'):  # > 0.1%
                alerts.append(f"High transaction costs: {current_metrics.average_costs}")

            # Add timestamp to alerts
            if alerts:
                timestamp = datetime.utcnow()
                for alert in alerts:
                    self.quality_alerts.append({
                        'timestamp': timestamp,
                        'alert': alert,
                        'severity': 'warning'
                    })

                # Keep last 100 alerts
                if len(self.quality_alerts) > 100:
                    self.quality_alerts = self.quality_alerts[-100:]

        except Exception as e:
            logger.error(f"Failed to check quality alerts: {str(e)}")

    def set_message_broker(self, message_broker):
        """Set the message broker for execution notifications."""
        self.message_broker = message_broker
        self.messaging_enabled = message_broker is not None
        if self.messaging_enabled:
            logger.info("Message broker enabled for execution handler")

    async def publish_execution_update(self, execution_report: ExecutionReport):
        """Publish execution update to message queue."""
        if not self.messaging_enabled or not self.message_broker:
            return

        try:
            # Prepare execution update message
            update_data = {
                "execution_id": execution_report.execution_id,
                "symbol": execution_report.order.symbol,
                "side": execution_report.order.side.value,
                "quantity": float(execution_report.order.quantity),
                "price": float(execution_report.order.price) if execution_report.order.price else None,
                "status": "filled" if execution_report.success else "failed",
                "execution_time": execution_report.execution_time_seconds,
                "slippage": float(execution_report.total_slippage),
                "costs": float(execution_report.total_costs),
                "fill_rate": execution_report.fill_rate,
                "grade": execution_report.performance_grade.value,
                "timestamp": execution_report.timestamp.isoformat(),
                "venue": execution_report.order.venue.value
            }

            # Determine routing key and priority
            routing_key = "trading.execution.update"
            priority = 7 if execution_report.success else 9  # Higher priority for failures

            await self.message_broker.publish_message(
                message_body=str(update_data),
                exchange_name="auj.platform",
                routing_key=routing_key,
                priority=priority
            )

            logger.debug(f"Published execution update for {execution_report.execution_id}")

        except Exception as e:
            logger.error(f"Failed to publish execution update: {e}")

    async def publish_risk_alert(self, alert_type: str, alert_data: Dict[str, Any]):
        """Publish risk alert to message queue."""
        if not self.messaging_enabled or not self.message_broker:
            return

        try:
            risk_alert = {
                "alert_type": alert_type,
                "source": "execution_handler",
                "timestamp": datetime.utcnow().isoformat(),
                "severity": alert_data.get("severity", "warning"),
                "data": alert_data
            }

            await self.message_broker.publish_message(
                message_body=str(risk_alert),
                exchange_name="auj.platform",
                routing_key="risk.execution_alert",
                priority=10  # High priority for risk alerts
            )

            logger.debug(f"Published risk alert: {alert_type}")

        except Exception as e:
            logger.error(f"Failed to publish risk alert: {e}")

    def get_quality_summary(self) -> Dict[str, Any]:
        """Get execution quality summary."""
        try:
            if not self.hourly_metrics:
                return {'status': 'no_data'}

            recent_metrics = self.hourly_metrics[-6:]  # Last 6 hours

            avg_success_rate = sum(m['success_rate'] for m in recent_metrics) / len(recent_metrics)
            avg_slippage = sum(m['avg_slippage'] for m in recent_metrics) / len(recent_metrics)
            avg_execution_time = sum(m['avg_execution_time'] for m in recent_metrics) / len(recent_metrics)

            # Determine overall quality grade
            quality_score = 100.0
            if avg_success_rate < 0.95:
                quality_score -= (0.95 - avg_success_rate) * 200
            if avg_slippage > 0.0003:
                quality_score -= (avg_slippage - 0.0003) * 10000
            if avg_execution_time > 2.0:
                quality_score -= (avg_execution_time - 2.0) * 10

            if quality_score >= 90:
                overall_grade = 'excellent'
            elif quality_score >= 80:
                overall_grade = 'good'
            elif quality_score >= 70:
                overall_grade = 'acceptable'
            elif quality_score >= 60:
                overall_grade = 'poor'
            else:
                overall_grade = 'critical'

            return {
                'status': 'active',
                'overall_grade': overall_grade,
                'quality_score': quality_score,
                'avg_success_rate': avg_success_rate,
                'avg_slippage': avg_slippage,
                'avg_execution_time': avg_execution_time,
                'recent_alerts': len([a for a in self.quality_alerts
                                    if a['timestamp'] > datetime.utcnow() - timedelta(hours=1)]),
                'total_periods': len(self.hourly_metrics)
            }

        except Exception as e:
            logger.error(f"Failed to get quality summary: {str(e)}")
            return {'status': 'error', 'error': str(e)}

    async def modify_position(self,
                            position_id: Union[str, int],
                            sl: Optional[float] = None,
                            tp: Optional[float] = None,
                            symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Enhanced position modification with validation and error handling.

        Args:
            position_id: Position identifier (ticket number)
            sl: New stop loss level
            tp: New take profit level
            symbol: Trading symbol (for validation)

        Returns:
            Dict with modification result and details
        """
        try:
            logger.info(f"Modifying position {position_id} - SL: {sl}, TP: {tp}")

            # Validate inputs
            if sl is None and tp is None:
                return {
                    'success': False,
                    'error': 'At least one of stop loss or take profit must be specified',
                    'position_id': position_id
                }

            # Get the appropriate broker interface
            broker_interface = self._get_primary_broker_interface()
            if not broker_interface:
                return {
                    'success': False,
                    'error': 'No broker interface available',
                    'position_id': position_id
                }

            # Validate position exists
            position_info = await self._validate_position_exists(broker_interface, position_id, symbol)
            if not position_info['exists']:
                return {
                    'success': False,
                    'error': position_info['reason'],
                    'position_id': position_id
                }

            position = position_info['position']

            # Validate stop loss and take profit levels
            validation_result = await self._validate_sl_tp_levels(
                position, sl, tp, broker_interface
            )
            if not validation_result['valid']:
                return {
                    'success': False,
                    'error': validation_result['reason'],
                    'position_id': position_id
                }

            # Perform the modification
            modification_attempts = 0
            max_attempts = 3

            while modification_attempts < max_attempts:
                try:
                    # Use broker interface to modify position
                    if hasattr(broker_interface, 'modify_position'):
                        modify_result = await broker_interface.modify_position(
                            position_id=int(position_id),
                            sl=sl,
                            tp=tp
                        )
                    else:
                        # Fallback method
                        modify_result = await self._fallback_modify_position(
                            broker_interface, position_id, sl, tp, position
                        )

                    if modify_result.get('success'):
                        logger.info(f"Successfully modified position {position_id}")

                        # Update our tracking if the position was in our active orders
                        self._update_position_tracking(position_id, sl, tp)

                        return {
                            'success': True,
                            'position_id': position_id,
                            'symbol': position.get('symbol'),
                            'old_sl': position.get('sl'),
                            'old_tp': position.get('tp'),
                            'new_sl': sl,
                            'new_tp': tp,
                            'modification_time': datetime.utcnow(),
                            'attempts': modification_attempts + 1,
                            'broker_result': modify_result
                        }
                    else:
                        modification_attempts += 1
                        error_msg = modify_result.get('error', 'Unknown error')
                        logger.warning(f"Modification attempt {modification_attempts} failed: {error_msg}")

                        if modification_attempts < max_attempts:
                            await asyncio.sleep(0.3)  # Brief delay before retry

                except Exception as attempt_error:
                    modification_attempts += 1
                    logger.warning(f"Modification attempt {modification_attempts} exception: {attempt_error}")

                    if modification_attempts < max_attempts:
                        await asyncio.sleep(0.3)

            # All attempts failed
            logger.error(f"Failed to modify position {position_id} after {max_attempts} attempts")
            return {
                'success': False,
                'error': f'Failed after {max_attempts} attempts',
                'position_id': position_id,
                'attempts': modification_attempts
            }

        except Exception as e:
            logger.error(f"Error in modify_position: {e}")
            return {
                'success': False,
                'error': f'Exception during modification: {str(e)}',
                'position_id': position_id
            }

    async def _validate_position_exists(self, broker_interface, position_id: Union[str, int],
                                      symbol: Optional[str]) -> Dict[str, Any]:
        """Validate that position exists and can be modified."""
        try:
            # Get all positions
            if hasattr(broker_interface, 'get_positions'):
                if symbol:
                    positions = await broker_interface.get_positions(symbol)
                else:
                    positions = await broker_interface.get_positions()

                # Look for our position
                for position in positions:
                    if str(position.get('ticket')) == str(position_id):
                        return {
                            'exists': True,
                            'position': position,
                            'reason': 'Position found'
                        }

                return {
                    'exists': False,
                    'position': None,
                    'reason': f'Position {position_id} not found'
                }
            else:
                return {
                    'exists': True,
                    'position': {'ticket': position_id},
                    'reason': 'Unable to verify, assuming exists'
                }

        except Exception as e:
            logger.warning(f"Error validating position exists: {e}")
            return {
                'exists': True,
                'position': {'ticket': position_id},
                'reason': 'Error during validation, assuming exists'
            }

    async def _validate_sl_tp_levels(self, position: Dict, sl: Optional[float],
                                   tp: Optional[float], broker_interface) -> Dict[str, Any]:
        """Validate stop loss and take profit levels are reasonable."""
        try:
            symbol = position.get('symbol')
            current_price = position.get('price_open') or position.get('price_current')

            if not current_price:
                # Get current market price
                if hasattr(broker_interface, 'get_current_price'):
                    price_data = await broker_interface.get_current_price(symbol)
                    if price_data:
                        current_price = price_data.get('bid') or price_data.get('ask') or price_data.get('price')

            if not current_price:
                return {'valid': True, 'reason': 'Unable to validate price levels'}

            current_price = float(current_price)
            position_type = position.get('type', 0)  # 0 = buy, 1 = sell

            # Validate stop loss
            if sl is not None:
                if position_type == 0:  # Buy position
                    if sl >= current_price:
                        return {
                            'valid': False,
                            'reason': f'Stop loss {sl} must be below current price {current_price} for buy position'
                        }
                else:  # Sell position
                    if sl <= current_price:
                        return {
                            'valid': False,
                            'reason': f'Stop loss {sl} must be above current price {current_price} for sell position'
                        }

            # Validate take profit
            if tp is not None:
                if position_type == 0:  # Buy position
                    if tp <= current_price:
                        return {
                            'valid': False,
                            'reason': f'Take profit {tp} must be above current price {current_price} for buy position'
                        }
                else:  # Sell position
                    if tp >= current_price:
                        return {
                            'valid': False,
                            'reason': f'Take profit {tp} must be below current price {current_price} for sell position'
                        }

            return {'valid': True, 'reason': 'Levels are valid'}

        except Exception as e:
            logger.warning(f"Error validating SL/TP levels: {e}")
            return {'valid': True, 'reason': 'Unable to validate levels'}

    async def _fallback_modify_position(self, broker_interface, position_id: Union[str, int],
                                      sl: Optional[float], tp: Optional[float],
                                      position: Dict) -> Dict[str, Any]:
        """Fallback modification method for brokers without native modify_position."""
        try:
            # Some brokers might require closing and reopening with new levels
            # This is a simplified fallback - in reality, you'd implement broker-specific logic
            logger.warning(f"Using fallback modification for position {position_id}")

            return {
                'success': False,
                'error': 'Broker interface does not support position modification'
            }

        except Exception as e:
            return {
                'success': False,
                'error': f'Fallback modification failed: {str(e)}'
            }

    def _update_position_tracking(self, position_id: Union[str, int],
                                sl: Optional[float], tp: Optional[float]) -> None:
        """Update our internal position tracking after successful modification."""
        try:
            # Update any matching orders in our active orders
            for order_id, order in self.active_orders.items():
                if (hasattr(order, 'position_id') and
                    str(order.position_id) == str(position_id)):
                    if sl is not None:
                        order.stop_loss = Decimal(str(sl))
                    if tp is not None:
                        order.take_profit = Decimal(str(tp))
                    order.modified_time = datetime.utcnow()
                    logger.debug(f"Updated tracking for order {order_id}")
                    break

        except Exception as e:
            logger.warning(f"Error updating position tracking: {e}")

    def _get_primary_broker_interface(self):
        """Get the primary broker interface for operations."""
        if not self.broker_interfaces:
            return None

        # Prefer MT5 if available
        if ExecutionVenue.MT5 in self.broker_interfaces:
            return self.broker_interfaces[ExecutionVenue.MT5]

        # Otherwise, return the first available
        return next(iter(self.broker_interfaces.values()))

    async def get_open_positions(self) -> List[Dict[str, Any]]:
        """
        Get all open positions from the connected broker.

        Returns:
            List of open position dictionaries
        """
        try:
            if not self.broker_interface:
                logger.warning("No broker interface available for position retrieval")
                return []

            # Get positions from the primary broker interface
            if hasattr(self.broker_interface, 'get_positions'):
                positions = await self.broker_interface.get_positions()
                if positions:
                    return positions

            # Alternative: try get_all_positions if available
            if hasattr(self.broker_interface, 'get_all_positions'):
                positions = await self.broker_interface.get_all_positions()
                if positions:
                    return positions

            logger.info("No open positions found")
            return []

        except Exception as e:
            logger.error(f"Failed to get open positions: {e}")
            return []
