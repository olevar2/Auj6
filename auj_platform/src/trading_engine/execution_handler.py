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
- Deal monitoring integration
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
from collections import defaultdict

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
    
    FIXES IMPLEMENTED:
    - Added DealMonitoringTeams integration
    - Added asyncio.Lock for race condition prevention
    - Fixed broker priority (METAAPI over MT5)
    - Removed unreachable code
    - Added exponential backoff
    - Comprehensive error handling
    """

    def __init__(self,
                 config_manager: UnifiedConfigManager,
                 risk_manager: DynamicRiskManager,
                 messaging_service: Optional[Any] = None,
                 deal_monitoring_teams: Optional[Any] = None):
        """
        Initialize the Execution Handler.

        Args:
            config_manager: Unified configuration manager instance
            risk_manager: Dynamic risk manager instance
            messaging_service: Optional injected messaging service
            deal_monitoring_teams: DealMonitoringTeams instance for position tracking
        """
        self.config_manager = config_manager
        self.risk_manager = risk_manager
        self.messaging_service = messaging_service
        self.deal_monitoring_teams = deal_monitoring_teams  # NEW: DealMonitoringTeams integration

        # These will be initialized in the initialize() method
        self.account_manager = None
        self.broker_interfaces = {}
        self.performance_tracker = None

        # Execution state with concurrency control
        self.active_orders: Dict[str, ExecutionOrder] = {}
        self.active_orders_lock = asyncio.Lock()  # NEW: Race condition fix
        self.execution_queue: List[ExecutionOrder] = []
        self.execution_history: List[ExecutionReport] = []

        # Configuration parameters
        self.max_slippage_percent = config_manager.get_float('trading.max_slippage_percent', 0.5)
        self.order_timeout_seconds = config_manager.get_int('trading.order_timeout_seconds', 30)
        self.retry_attempts = config_manager.get_int('trading.retry_attempts', 3)
        self.base_retry_delay = config_manager.get_float('trading.retry_delay_seconds', 1.0)  # For exponential backoff
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

        logger.info("Execution Handler initialized with DealMonitoringTeams integration")

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
                broker_interface=None
            )
            await self.account_manager.initialize()
            logger.info("Account Manager initialized")

            # Initialize broker interfaces
            await self._initialize_broker_interfaces()

            # Initialize messaging if configured
            await self._initialize_messaging()

            logger.info("Execution Handler initialization completed successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Execution Handler: {e}")
            # NEW: Cleanup on initialization failure
            await self.cleanup()
            raise

    async def _initialize_broker_interfaces(self) -> None:
        """Initialize broker interfaces with correct priority."""
        try:
            # FIXED: Initialize MetaApi interface FIRST (primary for Linux deployment)
            metaapi_config = self.config_manager.get_dict('brokers.metaapi', {})
            if metaapi_config.get('enabled', True):
                from ..broker_interfaces.metaapi_broker import MetaApiBroker
                metaapi_broker = MetaApiBroker(metaapi_config, self.config_manager)
                if await metaapi_broker.initialize():
                    self.broker_interfaces[ExecutionVenue.METAAPI] = metaapi_broker
                    
                    if self.account_manager:
                        self.account_manager.broker_interface = metaapi_broker

                    logger.info("✅ MetaApi Broker initialized (PRIMARY)")
                else:
                    logger.warning("⚠️ Failed to initialize MetaApi Broker")

            # Legacy MT5 interface (deprecated)
            mt5_config = self.config_manager.get_dict('brokers.mt5', {})
            if mt5_config.get('enabled', False):
                logger.warning("⚠️ MT5 broker interface is DEPRECATED. Use MetaApi for production.")

            if not self.broker_interfaces:
                logger.warning("⚠️ No broker interfaces initialized - trading will not be available")

        except Exception as e:
            logger.warning(f"Could not initialize broker interfaces: {e}")

    async def _initialize_messaging(self) -> None:
        """Initialize messaging system using injected dependency."""
        try:
            if self.messaging_service:
                self.messaging_enabled = True
                logger.info("✅ Using injected messaging service")

                await self.messaging_service.publish_system_status(
                    component="execution_handler",
                    status="initialized",
                    details={
                        "broker_interfaces": len(self.broker_interfaces),
                        "account_manager": self.account_manager is not None,
                        "deal_monitoring": self.deal_monitoring_teams is not None
                    }
                )
            else:
                logger.info("📝 Messaging system disabled")

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

        logger.info(f"🎯 Executing trade signal {signal.id}: {signal.direction.value} {signal.symbol}")

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
            logger.error(f"❌ Trade execution failed for signal {signal.id}: {str(e)}")
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
        errors = []
        
        try:
            # Additional execution-specific checks
            if account_info.balance <= Decimal('100'):
                errors.append("Account balance too low")

            if account_info.margin_available <= Decimal('50'):
                errors.append("Insufficient margin available")

            # Check for account restrictions
            if hasattr(account_info, 'trading_enabled') and not account_info.trading_enabled:
                errors.append("Trading disabled on account")

            return len(errors) == 0, errors

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
            order_type = await self._determine_optimal_order_type(signal, risk_metrics)
            current_price = await self._get_current_price(signal.symbol)
            execution_price = await self._calculate_execution_price(signal, current_price, order_type)

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
                    'account_id': account_info.account_id
                }
            )

            return execution_order

        except Exception as e:
            logger.error(f"Failed to create execution order: {str(e)}")
            raise ExecutionError(f"Order creation failed: {str(e)}")

    async def _determine_optimal_order_type(self, signal: TradeSignal, risk_metrics: RiskMetrics) -> OrderType:
        """Determine the optimal order type based on signal and market conditions."""
        try:
            if signal.confidence >= 0.8 and risk_metrics.risk_level.value in ['VERY_LOW', 'LOW']:
                return OrderType.MARKET
            elif signal.confidence >= 0.6:
                return OrderType.LIMIT
            else:
                return OrderType.LIMIT

        except Exception as e:
            logger.warning(f"Failed to determine order type: {str(e)}")
            return OrderType.LIMIT

    async def _calculate_execution_price(self,
                                       signal: TradeSignal,
                                       current_price: Decimal,
                                       order_type: OrderType) -> Optional[Decimal]:
        """Calculate optimal execution price based on order type."""
        try:
            if order_type == OrderType.MARKET:
                return None

            if order_type == OrderType.LIMIT:
                spread = await self._get_bid_ask_spread(signal.symbol)

                if signal.direction == TradeDirection.BUY:
                    limit_price = current_price - (spread * Decimal('0.3'))
                else:
                    limit_price = current_price + (spread * Decimal('0.3'))

                return limit_price.quantize(Decimal('0.00001'), rounding=ROUND_HALF_UP)

            return current_price

        except Exception as e:
            logger.warning(f"Failed to calculate execution price: {str(e)}")
            return current_price

    async def _select_optimal_venue(self, symbol: str, position_size: Decimal) -> ExecutionVenue:
        """
        Select the optimal execution venue based on performance and conditions.
        FIXED: Prefers METAAPI over deprecated MT5.
        """
        try:
            available_venues = list(self.broker_interfaces.keys())

            if not available_venues:
                return ExecutionVenue.DEMO

            # FIXED: Prefer METAAPI if available
            if ExecutionVenue.METAAPI in available_venues:
                return ExecutionVenue.METAAPI

            # Otherwise use best available venue
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
            return ExecutionVenue.METAAPI if ExecutionVenue.METAAPI in self.broker_interfaces else ExecutionVenue.DEMO

    async def _calculate_venue_score(self, venue: ExecutionVenue, symbol: str, position_size: Decimal) -> float:
        """Calculate venue score based on historical performance."""
        try:
            venue_stats = self.venue_performance.get(venue, {})

            execution_speed = venue_stats.get('avg_execution_speed', 0.5)
            fill_rate = venue_stats.get('fill_rate', 0.8)
            avg_slippage = venue_stats.get('avg_slippage', 0.001)

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
        """
        Execute the order with retry logic and performance tracking.
        FIXED: Exponential backoff, proper error recovery.
        """
        execution_errors = []
        execution_warnings = []

        try:
            broker_interface = self.broker_interfaces.get(order.venue)
            if not broker_interface:
                raise ExecutionError(f"No broker interface available for {order.venue.value}")

            # FIXED: Thread-safe active orders management
            async with self.active_orders_lock:
                self.active_orders[order.order_id] = order
            
            order.status = ExecutionStatus.VALIDATED
            order.submitted_at = datetime.utcnow()

            # Execute with retry logic and exponential backoff
            execution_success = False
            retry_count = 0

            while not execution_success and retry_count < self.retry_attempts:
                try:
                    # Submit order
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
                            # FIXED: Exponential backoff
                            delay = self.base_retry_delay * (2 ** retry_count)
                            logger.warning(f"⚠️ Retry {retry_count}/{self.retry_attempts} after {delay}s")
                            await asyncio.sleep(delay)

                except Exception as e:
                    retry_count += 1
                    error_msg = f"Execution attempt {retry_count} failed: {str(e)}"
                    execution_errors.append(error_msg)
                    logger.warning(error_msg)

                    if retry_count < self.retry_attempts:
                        # FIXED: Exponential backoff
                        delay = self.base_retry_delay * (2 ** retry_count)
                        await asyncio.sleep(delay)

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

            # FIXED: Thread-safe cleanup
            async with self.active_orders_lock:
                if order.order_id in self.active_orders:
                    del self.active_orders[order.order_id]

            # Update statistics
            self._update_execution_statistics(report)

            return report

        except Exception as e:
            error_msg = f"Order execution failed: {str(e)}"
            execution_errors.append(error_msg)
            logger.error(error_msg)

            # FIXED: Thread-safe cleanup
            async with self.active_orders_lock:
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
            # Map order direction to broker order type
            if order.direction == TradeDirection.BUY:
                if order.order_type == OrderType.MARKET:
                    broker_order_type = "BUY"
                elif order.order_type == OrderType.LIMIT:
                    broker_order_type = "BUY_LIMIT"
                else:
                    broker_order_type = "BUY"
            else:
                if order.order_type == OrderType.MARKET:
                    broker_order_type = "SELL"
                elif order.order_type == OrderType.LIMIT:
                    broker_order_type = "SELL_LIMIT"
                else:
                    broker_order_type = "SELL"

            # Submit order
            result = await broker_interface.place_order(
                symbol=order.symbol,
                order_type=broker_order_type,
                volume=float(order.quantity),
                price=float(order.price) if order.price else None,
                sl=float(order.stop_loss) if order.stop_loss else None,
                tp=float(order.take_profit) if order.take_profit else None,
                comment=f"AUJ:{order.order_id}"
            )

            if result.get('success'):
                broker_order_id = result.get('order_id') or result.get('deal_id')
                if broker_order_id:
                    logger.info(f"✅ Order {order.order_id} submitted as {broker_order_id}")
                    return str(broker_order_id)
                else:
                    raise BrokerError("Order submitted but no order ID returned")
            else:
                error_msg = result.get('error', 'Unknown broker error')
                raise BrokerError(f"Broker rejected order: {error_msg}")

        except Exception as e:
            logger.error(f"Failed to submit order to broker: {str(e)}")
            raise BrokerError(f"Failed to submit order: {str(e)}")

    async def _wait_for_fill(self, broker_interface, broker_order_id: str, order: ExecutionOrder) -> Dict[str, Any]:
        """
        Wait for order fill with timeout and enhanced monitoring.
        FIXED: Removed unreachable code, proper timeout handling.
        """
        try:
            timeout_time = datetime.utcnow() + timedelta(seconds=self.order_timeout_seconds)
            check_interval = 0.1

            logger.info(f"📊 Monitoring fill for order {order.order_id}")

            while datetime.utcnow() < timeout_time:
                if order.order_type == OrderType.MARKET:
                    fill_result = await self._check_market_order_fill(broker_interface, broker_order_id, order)
                    if fill_result['success'] or fill_result.get('terminal_error'):
                        return fill_result
                else:
                    fill_result = await self._check_pending_order_status(broker_interface, broker_order_id, order)
                    if fill_result['success'] or fill_result.get('terminal_error'):
                        return fill_result

                await asyncio.sleep(check_interval)
                if order.order_type == OrderType.MARKET:
                    check_interval = min(check_interval * 1.1, 1.0)

            # FIXED: Timeout handling before exception
            return await self._handle_fill_timeout(broker_interface, broker_order_id, order)

        except Exception as e:
            logger.error(f"Error waiting for fill: {str(e)}")
            return {
                'success': False,
                'fill_data': {'status': 'ERROR', 'error': str(e)},
                'warnings': [f'Error during fill monitoring: {str(e)}']
            }

    async def _check_market_order_fill(self, broker_interface, broker_order_id: str, order: ExecutionOrder) -> Dict[str, Any]:
        """Enhanced market order fill checking."""
        try:
            positions = await broker_interface.get_positions(order.symbol)

            for position in positions:
                if (position.get('comment', '').startswith(f"AUJ:{order.order_id}") or
                    str(position.get('ticket')) == str(broker_order_id)):

                    fill_price = Decimal(str(position.get('price_open', 0)))
                    fill_quantity = Decimal(str(position.get('volume', 0)))
                    commission = Decimal(str(position.get('commission', 0)))

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

    async def _check_pending_order_status(self, broker_interface, broker_order_id: str, order: ExecutionOrder) -> Dict[str, Any]:
        """Enhanced pending order status checking."""
        try:
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
                    elif status in ['REJECTED', 'CANCELLED', 'EXPIRED']:
                        return {
                            'success': False,
                            'status': status,
                            'fill_data': {'status': status, 'reason': order_status.get('reason', 'Unknown')},
                            'warnings': [f'Order {status.lower()}: {order_status.get("reason", "Unknown")}'],
                            'terminal_error': True
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

    async def _handle_fill_timeout(self, broker_interface, broker_order_id: str, order: ExecutionOrder) -> Dict[str, Any]:
        """
        Enhanced timeout handling with cancellation attempts.
        FIXED: Moved cancellation BEFORE exception handling.
        """
        try:
            logger.warning(f"⏱️ Order {order.order_id} timed out, attempting recovery")

            # Try to cancel if it's a pending order
            if order.order_type != OrderType.MARKET:
                try:
                    if hasattr(broker_interface, 'cancel_order'):
                        cancel_result = await broker_interface.cancel_order(broker_order_id)
                        if cancel_result.get('success'):
                            logger.info(f"✅ Successfully cancelled timed-out order {broker_order_id}")
                            return {
                                'success': False,
                                'status': 'CANCELLED_ON_TIMEOUT',
                                'fill_data': {
                                    'status': 'CANCELLED_ON_TIMEOUT',
                                    'timeout_seconds': self.order_timeout_seconds
                                },
                                'warnings': ['Order cancelled due to timeout']
                            }
                except Exception as cancel_error:
                    logger.warning(f"Failed to cancel order: {cancel_error}")

            # Final check
            final_check = await self._check_market_order_fill(broker_interface, broker_order_id, order)
            if final_check['success']:
                return final_check

            return {
                'success': False,
                'status': 'TIMEOUT',
                'fill_data': {
                    'status': 'TIMEOUT',
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

    async def _process_successful_fill(self, order: ExecutionOrder, fill_result: Dict[str, Any]):
        """Process successful order fill and update order details."""
        try:
            fill_data = fill_result['fill_data']

            order.status = ExecutionStatus.FILLED
            order.filled_at = datetime.utcnow()
            order.filled_price = Decimal(str(fill_data.get('fill_price', 0)))
            order.filled_quantity = Decimal(str(fill_data.get('fill_quantity', 0)))
            order.remaining_quantity = order.quantity - order.filled_quantity
            order.transaction_costs = Decimal(str(fill_data.get('commission', 0)))

            # Calculate slippage
            if fill_data.get('slippage'):
                order.slippage = Decimal(str(fill_data['slippage']))
            elif order.price and order.filled_price:
                if order.direction == TradeDirection.BUY:
                    order.slippage = order.filled_price - order.price
                else:
                    order.slippage = order.price - order.filled_price

            # Calculate execution time
            if order.submitted_at:
                execution_ms = int((order.filled_at - order.submitted_at).total_seconds() * 1000)
                order.execution_time_ms = execution_ms

            logger.info(f"✅ Order {order.order_id} filled: {order.filled_quantity} @ {order.filled_price}")

        except Exception as e:
            logger.error(f"Failed to process successful fill: {str(e)}")
            raise ExecutionError(f"Fill processing failed: {str(e)}")

    def _calculate_execution_grade(self, order: ExecutionOrder, success: bool) -> DealGrade:
        """Calculate execution quality grade."""
        try:
            if not success:
                return DealGrade.F

            score = 100.0

            # Deduct for slippage
            if order.slippage:
                slippage_percent = abs(float(order.slippage)) / float(order.filled_price or 1) * 100
                if slippage_percent > self.max_slippage_percent:
                    score -= min(30, slippage_percent * 10)
                else:
                    score -= slippage_percent * 5

            # Deduct for execution time
            if order.execution_time_ms:
                if order.execution_time_ms > 5000:
                    score -= min(20, (order.execution_time_ms - 5000) / 1000)

            # Deduct for partial fills
            if order.filled_quantity < order.quantity:
                fill_ratio = float(order.filled_quantity / order.quantity)
                score -= (1.0 - fill_ratio) * 25

            # Convert to grade
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
            elif score >= 60:
                return DealGrade.C
            else:
                return DealGrade.D

        except Exception as e:
            logger.warning(f"Execution grade calculation failed: {str(e)}")
            return DealGrade.C

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

                venue_stats['success_rate'] = venue_stats['successful_executions'] / venue_stats['total_executions']
                venue_stats['avg_slippage'] = venue_stats['total_slippage'] / venue_stats['successful_executions']
                venue_stats['avg_execution_speed'] = venue_stats['total_execution_time'] / venue_stats['successful_executions']
                venue_stats['fill_rate'] = report.fill_rate

            # Add to execution history
            self.execution_history.append(report)
            if len(self.execution_history) > 1000:
                self.execution_history.pop(0)

        except Exception as e:
            logger.error(f"Failed to update execution statistics: {e}")

    async def _post_execution_processing(self, report: ExecutionReport):
        """
        Perform post-execution processing and notifications.
        FIXED: Added DealMonitoringTeams integration.
        """
        try:
            # Record execution in performance tracker
            if self.performance_tracker and report.success:
                await self.performance_tracker.record_execution(
                    signal_id=report.order.signal_id,
                    execution_report=report
                )

            # FIXED: NEW - Send to DealMonitoringTeams
            if report.success and self.deal_monitoring_teams:
                try:
                    await self.deal_monitoring_teams.add_position(
                        deal_id=report.order.order_id,
                        symbol=report.order.symbol,
                        direction=report.order.direction.value,
                        entry_price=report.order.filled_price,
                        quantity=report.order.filled_quantity,
                        stop_loss=report.order.stop_loss,
                        take_profit=report.order.take_profit,
                        entry_time=report.order.filled_at or datetime.utcnow(),
                        confidence_level=report.order.metadata.get('signal_confidence', 0.0),
                        agent_source=report.order.metadata.get('generating_agent', ''),
                        strategy_type=report.order.metadata.get('strategy', '')
                    )
                    logger.info(f"✅ Position {report.order.order_id} sent to DealMonitoringTeams")
                except Exception as e:
                    logger.error(f"❌ Failed to send position to DealMonitoringTeams: {e}")

            # Publish execution update
            if self.messaging_enabled and self.messaging_service:
                await self._publish_execution_update(report)

            # Send alerts for poor execution
            if report.success and report.performance_grade.value in ['D', 'F']:
                logger.warning(f"⚠️ Poor execution quality: Grade {report.performance_grade.value}")
                if self.messaging_enabled:
                    await self._publish_risk_alert('poor_execution', {
                        'order_id': report.order.order_id,
                        'grade': report.performance_grade.value,
                        'slippage': float(report.total_slippage)
                    })

        except Exception as e:
            logger.error(f"Post-execution processing failed: {e}")

    async def _publish_execution_update(self, execution_report: ExecutionReport):
        """Publish execution update to message queue."""
        if not self.messaging_enabled or not self.messaging_service:
            return

        try:
            update_data = {
                "execution_id": execution_report.execution_id,
                "symbol": execution_report.order.symbol,
                "direction": execution_report.order.direction.value,
                "quantity": float(execution_report.order.quantity),
                "price": float(execution_report.order.price) if execution_report.order.price else None,
                "status": "filled" if execution_report.success else "failed",
                "execution_time": execution_report.execution_time_seconds,
                "slippage": float(execution_report.total_slippage),
                "grade": execution_report.performance_grade.value,
                "timestamp": execution_report.timestamp.isoformat()
            }

            await self.messaging_service.publish_trade_execution(update_data)
            logger.debug(f"📢 Published execution update for {execution_report.execution_id}")

        except Exception as e:
            logger.error(f"Failed to publish execution update: {e}")

    async def _publish_risk_alert(self, alert_type: str, alert_data: Dict[str, Any]):
        """Publish risk alert to message queue."""
        if not self.messaging_enabled or not self.messaging_service:
            return

        try:
            risk_alert = {
                "alert_type": alert_type,
                "source": "execution_handler",
                "timestamp": datetime.utcnow().isoformat(),
                "severity": alert_data.get("severity", "warning"),
                "data": alert_data
            }

            await self.messaging_service.publish_risk_alert(risk_alert)
            logger.debug(f"🚨 Published risk alert: {alert_type}")

        except Exception as e:
            logger.error(f"Failed to publish risk alert: {e}")

    async def modify_position(self,
                            position_id: Union[str, int],
                            sl: Optional[float] = None,
                            tp: Optional[float] = None,
                            symbol: Optional[str] = None) -> Dict[str, Any]:
        """Enhanced position modification with validation and error handling."""
        try:
            logger.info(f"🔧 Modifying position {position_id} - SL: {sl}, TP: {tp}")

            if sl is None and tp is None:
                return {
                    'success': False,
                    'error': 'At least one of SL or TP must be specified',
                    'position_id': position_id
                }

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
            
            # Validate levels
            validation_result = await self._validate_sl_tp_levels(position, sl, tp, broker_interface)
            if not validation_result['valid']:
                return {
                    'success': False,
                    'error': validation_result['reason'],
                    'position_id': position_id
                }

            # Perform modification with retry
            modification_attempts = 0
            max_attempts = 3

            while modification_attempts < max_attempts:
                try:
                    if hasattr(broker_interface, 'modify_position'):
                        modify_result = await broker_interface.modify_position(
                            position_id=int(position_id),
                            sl=sl,
                            tp=tp
                        )
                    else:
                        return {
                            'success': False,
                            'error': 'Broker does not support position modification',
                            'position_id': position_id
                        }

                    if modify_result.get('success'):
                        logger.info(f"✅ Successfully modified position {position_id}")
                        return {
                            'success': True,
                            'position_id': position_id,
                            'symbol': position.get('symbol'),
                            'old_sl': position.get('sl'),
                            'old_tp': position.get('tp'),
                            'new_sl': sl,
                            'new_tp': tp,
                            'modification_time': datetime.utcnow()
                        }
                    else:
                        modification_attempts += 1
                        if modification_attempts < max_attempts:
                            await asyncio.sleep(0.3)

                except Exception as attempt_error:
                    modification_attempts += 1
                    logger.warning(f"Modification attempt {modification_attempts} failed: {attempt_error}")
                    if modification_attempts < max_attempts:
                        await asyncio.sleep(0.3)

            return {
                'success': False,
                'error': f'Failed after {max_attempts} attempts',
                'position_id': position_id
            }

        except Exception as e:
            logger.error(f"Error in modify_position: {e}")
            return {
                'success': False,
                'error': f'Exception: {str(e)}',
                'position_id': position_id
            }

    async def _validate_position_exists(self, broker_interface, position_id: Union[str, int], symbol: Optional[str]) -> Dict[str, Any]:
        """Validate that position exists."""
        try:
            if hasattr(broker_interface, 'get_positions'):
                if symbol:
                    positions = await broker_interface.get_positions(symbol)
                else:
                    positions = await broker_interface.get_positions()

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
            logger.warning(f"Error validating position: {e}")
            return {
                'exists': True,
                'position': {'ticket': position_id},
                'reason': 'Error during validation'
            }

    async def _validate_sl_tp_levels(self, position: Dict, sl: Optional[float], tp: Optional[float], broker_interface) -> Dict[str, Any]:
        """Validate SL/TP levels."""
        try:
            symbol = position.get('symbol')
            current_price = position.get('price_open') or position.get('price_current')

            if not current_price:
                if hasattr(broker_interface, 'get_current_price'):
                    price_data = await broker_interface.get_current_price(symbol)
                    if price_data:
                        current_price = price_data.get('bid') or price_data.get('ask') or price_data.get('price')

            if not current_price:
                return {'valid': True, 'reason': 'Unable to validate price levels'}

            current_price = float(current_price)
            position_type = position.get('type', 0)

            # Validate stop loss
            if sl is not None:
                if position_type == 0:  # Buy
                    if sl >= current_price:
                        return {
                            'valid': False,
                            'reason': f'SL {sl} must be below current price {current_price} for buy'
                        }
                else:  # Sell
                    if sl <= current_price:
                        return {
                            'valid': False,
                            'reason': f'SL {sl} must be above current price {current_price} for sell'
                        }

            # Validate take profit
            if tp is not None:
                if position_type == 0:  # Buy
                    if tp <= current_price:
                        return {
                            'valid': False,
                            'reason': f'TP {tp} must be above current price {current_price} for buy'
                        }
                else:  # Sell
                    if tp >= current_price:
                        return {
                            'valid': False,
                            'reason': f'TP {tp} must be below current price {current_price} for sell'
                        }

            return {'valid': True, 'reason': 'Levels are valid'}

        except Exception as e:
            logger.warning(f"Error validating SL/TP: {e}")
            return {'valid': True, 'reason': 'Unable to validate'}

    def _get_primary_broker_interface(self):
        """
        Get the primary broker interface.
        FIXED: Prefers METAAPI over deprecated MT5.
        """
        if not self.broker_interfaces:
            return None

        # FIXED: Prefer METAAPI
        if ExecutionVenue.METAAPI in self.broker_interfaces:
            return self.broker_interfaces[ExecutionVenue.METAAPI]

        # Otherwise return first available
        return next(iter(self.broker_interfaces.values()))

    async def get_open_positions(self) -> List[Dict[str, Any]]:
        """
        Get all open positions from the connected broker.
        FIXED: Uses self.broker_interfaces (plural) instead of undefined self.broker_interface.
        """
        try:
            broker_interface = self._get_primary_broker_interface()
            
            if not broker_interface:
                logger.warning("No broker interface available for position retrieval")
                return []

            if hasattr(broker_interface, 'get_positions'):
                positions = await broker_interface.get_positions()
                if positions:
                    return positions

            if hasattr(broker_interface, 'get_all_positions'):
                positions = await broker_interface.get_all_positions()
                if positions:
                    return positions

            logger.info("No open positions found")
            return []

        except Exception as e:
            logger.error(f"Failed to get open positions: {e}")
            return []

    def _create_failed_report(self, execution_id: str, signal: TradeSignal, errors: List[str], start_time: datetime) -> ExecutionReport:
        """Create a failed execution report."""
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        
        dummy_order = ExecutionOrder(
            signal_id=signal.id,
            symbol=signal.symbol,
            direction=signal.direction,
            validation_errors=errors
        )

        return ExecutionReport(
            execution_id=execution_id,
            order=dummy_order,
            success=False,
            execution_time_seconds=execution_time,
            total_slippage=Decimal('0'),
            total_costs=Decimal('0'),
            fill_rate=0.0,
            warnings=[],
            errors=errors,
            performance_grade=DealGrade.F
        )

    async def _get_current_price(self, symbol: str) -> Optional[Decimal]:
        """Get current price for symbol."""
        try:
            broker_interface = self._get_primary_broker_interface()
            if not broker_interface:
                return None

            if hasattr(broker_interface, 'get_current_price'):
                price_data = await broker_interface.get_current_price(symbol)
                if price_data:
                    price = price_data.get('bid') or price_data.get('ask') or price_data.get('price')
                    return Decimal(str(price)) if price else None

            return None

        except Exception as e:
            logger.warning(f"Failed to get current price for {symbol}: {e}")
            return None

    async def _get_bid_ask_spread(self, symbol: str) -> Decimal:
        """Get bid-ask spread for symbol."""
        try:
            broker_interface = self._get_primary_broker_interface()
            if not broker_interface:
                return Decimal('0.0001')

            if hasattr(broker_interface, 'get_current_price'):
                price_data = await broker_interface.get_current_price(symbol)
                if price_data and 'bid' in price_data and 'ask' in price_data:
                    spread = Decimal(str(price_data['ask'])) - Decimal(str(price_data['bid']))
                    return spread

            return Decimal('0.0001')  # Default 1 pip

        except Exception as e:
            logger.warning(f"Failed to get spread for {symbol}: {e}")
            return Decimal('0.0001')

    async def _check_market_hours(self, symbol: str) -> bool:
        """Check if market is open for symbol."""
        # Simplified - always return True for 24/7 forex
        return True

    async def _check_symbol_liquidity(self, symbol: str) -> bool:
        """Check symbol liquidity."""
        # Simplified - always return True
        return True

    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return {
            'total_executions': self.total_executions,
            'successful_executions': self.successful_executions,
            'success_rate': self.successful_executions / max(self.total_executions, 1),
            'total_slippage': float(self.total_slippage),
            'total_costs': float(self.total_costs),
            'venue_performance': self.venue_performance,
            'active_orders_count': len(self.active_orders)
        }
