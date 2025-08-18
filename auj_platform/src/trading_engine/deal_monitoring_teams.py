"""
Deal Monitoring Teams for AUJ Platform

This module implements real-time monitoring of active trades with focus on
risk alerts and real-time position surveillance. Historical performance and 
grading is handled by PerformanceTracker to avoid duplication.

Key Features:
- Real-time monitoring of open positions only
- Risk alert system for active trades
- Team-based monitoring approach
- Automatic escalation protocols
- Real-time position tracking
- Integration with PerformanceTracker for historical data
"""

import asyncio
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Callable
from dataclasses import dataclass, field
from pydantic import BaseModel, Field

from ..core.data_contracts import TradeSignal, AgentDecision, DealGrade
from ..core.exceptions import AUJException
from ..analytics.performance_tracker import PerformanceTracker

logger = logging.getLogger(__name__)


class DealStatus(Enum):
    """Deal status enumeration."""
    OPEN = "open"
    CLOSED = "closed"
    PENDING = "pending"
    CANCELLED = "cancelled"
    PARTIAL = "partial"


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class MonitoringTeam(Enum):
    """Monitoring team types."""
    RISK_TEAM = "risk_team"
    PERFORMANCE_TEAM = "performance_team"
    TECHNICAL_TEAM = "technical_team"
    COORDINATION_TEAM = "coordination_team"


@dataclass
class DealPosition:
    """Represents a trading position being monitored in real-time."""
    deal_id: str
    symbol: str
    direction: str  # 'BUY' or 'SELL'
    entry_price: Decimal
    current_price: Decimal
    quantity: Decimal
    entry_time: datetime
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    status: DealStatus = DealStatus.OPEN
    confidence_level: float = 0.0
    risk_score: float = 0.0
    agent_source: str = ""
    strategy_type: str = ""
    duration_hours: float = 0.0
    last_update: datetime = field(default_factory=datetime.utcnow)
    alerts: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def unrealized_pnl(self) -> Decimal:
        """Calculate current unrealized P&L for real-time monitoring."""
        try:
            if self.direction.upper() == 'BUY':
                return (self.current_price - self.entry_price) * self.quantity
            else:  # SELL
                return (self.entry_price - self.current_price) * self.quantity
        except Exception:
            return Decimal('0')
    
    @property 
    def pnl_percentage(self) -> float:
        """Calculate P&L percentage for risk monitoring."""
        try:
            if self.entry_price > 0:
                return float(self.unrealized_pnl / (self.entry_price * self.quantity)) * 100
            return 0.0
        except Exception:
            return 0.0


@dataclass
class MonitoringAlert:
    """Represents a monitoring alert."""
    alert_id: str
    deal_id: str
    team: MonitoringTeam
    severity: AlertSeverity
    message: str
    timestamp: datetime
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    escalated: bool = False
    escalation_level: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class DealMonitoringError(AUJException):
    """Exception raised for deal monitoring errors."""
    pass

class DealMonitoringTeams:
    """
    Real-time monitoring system for active trades with team-based surveillance.

    This class coordinates multiple monitoring teams to ensure comprehensive
    oversight of all active trading positions and risk management. Historical
    performance tracking and grading is delegated to PerformanceTracker.
    """

    def __init__(self,
                 performance_tracker: PerformanceTracker,
                 risk_manager=None,
                 alert_callback: Optional[Callable] = None):
        """
        Initialize the Deal Monitoring Teams system.

        Args:
            performance_tracker: Required performance tracking instance (single source of truth)
            risk_manager: Dynamic risk manager instance
            alert_callback: Optional callback for alert notifications
        """
        from ..core.unified_config import get_unified_config
        self.config_manager = get_unified_config()
        
        if not performance_tracker:
            raise ValueError("PerformanceTracker is required - it's the single source of truth for trade performance")
        
        self.performance_tracker = performance_tracker
        self.risk_manager = risk_manager
        self.alert_callback = alert_callback

        # Active positions being monitored (real-time only)
        self.active_positions: Dict[str, DealPosition] = {}

        # Alert management
        self.active_alerts: Dict[str, MonitoringAlert] = {}
        self.alert_history: List[MonitoringAlert] = []

        # Team configuration (focused on real-time monitoring)
        self.team_configs = {
            MonitoringTeam.RISK_TEAM: {
                'max_risk_per_deal': 0.02,  # 2% max risk per deal
                'max_portfolio_risk': 0.10,  # 10% max portfolio risk
                'stop_loss_threshold': 0.015,  # 1.5% stop loss trigger
                'correlation_limit': 0.7,  # Maximum correlation between positions
                'monitoring_interval': 30  # seconds
            },
            MonitoringTeam.TECHNICAL_TEAM: {
                'price_alert_threshold': 0.005,  # 0.5% price movement
                'volume_alert_threshold': 2.0,  # 200% volume increase
                'spread_alert_threshold': 0.0005,  # 5 pips spread
                'connection_monitoring': True,
                'monitoring_interval': 15  # seconds
            },
            MonitoringTeam.COORDINATION_TEAM: {
                'escalation_threshold': 3,  # Number of alerts before escalation
                'team_coordination_interval': 180,  # 3 minutes
                'status_report_interval': 600,  # 10 minutes
                'monitoring_interval': 120  # seconds
            },
            MonitoringTeam.PERFORMANCE_TEAM: {
                'performance_check_interval': 300,  # 5 minutes
                'profitability_threshold': 0.005,  # 0.5% minimum profit
                'loss_alert_threshold': 0.02,  # 2% loss alert
                'monitoring_interval': 60  # seconds
            }
        }

        # Monitoring state
        self.is_monitoring = False
        self.monitoring_tasks: Dict[MonitoringTeam, asyncio.Task] = {}
        self.last_status_report = datetime.utcnow()

        # Performance metrics
        self.total_positions_monitored = 0
        self.total_alerts_generated = 0
        self.total_escalations = 0
        self.monitoring_uptime = 0.0
        self.last_monitoring_start = None

        logger.info("Deal Monitoring Teams system initialized")


    async def initialize(self) -> None:
        """Initialize the component with required dependencies."""
        logger.info("Initializing Deal Monitoring Teams...")
        
        # Validate team configurations
        self._validate_team_configurations()
        
        # Initialize monitoring state
        self.is_monitoring = False
        self.monitoring_tasks = {}
        self.total_positions_monitored = 0
        self.total_alerts_generated = 0
        self.total_escalations = 0
        self.monitoring_uptime = 0.0
        self.last_monitoring_start = None
        self.last_status_report = datetime.utcnow()
        
        # Initialize position tracking dictionaries
        self.active_positions = {}
        self.position_alerts = defaultdict(list)
        self.team_status = {team: "READY" for team in MonitoringTeam}
        
        logger.info("Deal Monitoring Teams initialization completed successfully")
        
    def _validate_team_configurations(self) -> None:
        """Validate that all monitoring teams have proper configurations."""
        required_teams = {MonitoringTeam.RISK_TEAM, MonitoringTeam.TECHNICAL_TEAM, 
                         MonitoringTeam.COORDINATION_TEAM, MonitoringTeam.PERFORMANCE_TEAM}
        
        configured_teams = set(self.team_configs.keys())
        missing_teams = required_teams - configured_teams
        
        if missing_teams:
            raise ValueError(f"Missing configurations for teams: {missing_teams}")
            
        # Validate each team configuration
        for team, config in self.team_configs.items():
            if 'monitoring_interval' not in config:
                raise ValueError(f"Missing monitoring_interval for team {team}")
            if config['monitoring_interval'] <= 0:
                raise ValueError(f"Invalid monitoring_interval for team {team}")
                
        logger.info("Team configurations validation passed")

    async def start_monitoring(self):
        """Start all monitoring teams."""
        try:
            if self.is_monitoring:
                logger.warning("Monitoring is already active")
                return

            self.is_monitoring = True
            self.last_monitoring_start = datetime.utcnow()

            # Start monitoring tasks for each team
            for team in MonitoringTeam:
                task = asyncio.create_task(self._team_monitoring_loop(team))
                self.monitoring_tasks[team] = task
                logger.info(f"Started {team.value} monitoring")

            logger.info("All monitoring teams started successfully")

        except Exception as e:
            logger.error(f"Failed to start monitoring: {str(e)}")
            await self.stop_monitoring()
            raise DealMonitoringError(f"Monitoring startup failed: {str(e)}")

    async def stop_monitoring(self):
        """Stop all monitoring teams."""
        try:
            self.is_monitoring = False

            # Cancel all monitoring tasks
            for team, task in self.monitoring_tasks.items():
                if not task.cancelled():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                logger.info(f"Stopped {team.value} monitoring")

            self.monitoring_tasks.clear()

            # Update monitoring uptime
            if self.last_monitoring_start:
                uptime = (datetime.utcnow() - self.last_monitoring_start).total_seconds()
                self.monitoring_uptime += uptime

            logger.info("All monitoring teams stopped")

        except Exception as e:
            logger.error(f"Error stopping monitoring: {str(e)}")

    async def add_position(self, deal_id: str, trade_signal: TradeSignal,
                          entry_price: Decimal, quantity: Decimal) -> bool:
        """
        Add a new position to monitoring.

        Args:
            deal_id: Unique identifier for the deal
            trade_signal: Original trade signal
            entry_price: Actual entry price
            quantity: Position size

        Returns:
            bool: True if position added successfully
        """
        try:
            if deal_id in self.active_positions:
                logger.warning(f"Position {deal_id} already being monitored")
                return False

            # Create new position
            position = DealPosition(
                deal_id=deal_id,
                symbol=trade_signal.symbol,
                direction=trade_signal.direction.value,
                entry_price=entry_price,
                current_price=entry_price,  # Initial current price
                quantity=quantity,
                entry_time=datetime.utcnow(),
                stop_loss=trade_signal.stop_loss,
                take_profit=trade_signal.take_profit,
                confidence_level=trade_signal.confidence,
                agent_source=getattr(trade_signal, 'agent_source', 'unknown'),
                strategy_type=getattr(trade_signal, 'strategy_type', 'unknown'),
                metadata={
                    'signal_id': trade_signal.id,
                    'entry_conditions': getattr(trade_signal, 'entry_conditions', {}),
                    'risk_parameters': getattr(trade_signal, 'risk_parameters', {})
                }
            )

            self.active_positions[deal_id] = position
            self.total_positions_monitored += 1

            # Generate initial monitoring alert
            await self._generate_alert(
                deal_id=deal_id,
                team=MonitoringTeam.COORDINATION_TEAM,
                severity=AlertSeverity.LOW,
                message=f"New position {deal_id} added to monitoring: {position.symbol} {position.direction}"
            )

            logger.info(f"Added position {deal_id} to monitoring: {position.symbol} {position.direction}")
            return True

        except Exception as e:
            logger.error(f"Failed to add position {deal_id}: {str(e)}")
            return False

    async def update_position(self, deal_id: str, current_price: Decimal,
                            current_time: Optional[datetime] = None) -> bool:
        """
        Update position with current market data.

        Args:
            deal_id: Position identifier
            current_price: Current market price
            current_time: Current timestamp (optional)

        Returns:
            bool: True if updated successfully
        """
        try:
            if deal_id not in self.active_positions:
                logger.warning(f"Position {deal_id} not found for update")
                return False

            position = self.active_positions[deal_id]
            old_pnl = position.unrealized_pnl

            # Update position data
            position.current_price = current_price
            position.last_update = current_time or datetime.utcnow()

            # Calculate P&L
            if position.direction == 'BUY':
                position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
            else:  # SELL
                position.unrealized_pnl = (position.entry_price - current_price) * position.quantity

            # Update max profit/drawdown
            if position.unrealized_pnl > position.max_profit:
                position.max_profit = position.unrealized_pnl

            if position.unrealized_pnl < position.max_drawdown:
                position.max_drawdown = position.unrealized_pnl

            # Update duration
            position.duration_hours = (position.last_update - position.entry_time).total_seconds() / 3600

            # Update grade based on performance using PerformanceTracker
            if self.performance_tracker:
                try:
                    metrics = await self.performance_tracker.get_deal_performance(deal_id)
                    if metrics:
                        pnl_pct = metrics.get('pnl_percentage', 0)
                        if pnl_pct >= 5:
                            position.grade = DealGrade.EXCELLENT
                        elif pnl_pct >= 2:
                            position.grade = DealGrade.GOOD
                        elif pnl_pct >= 0:
                            position.grade = DealGrade.AVERAGE
                        elif pnl_pct >= -2:
                            position.grade = DealGrade.POOR
                        else:
                            position.grade = DealGrade.CRITICAL
                except Exception as e:
                    logger.warning(f"Could not get performance metrics for {deal_id}: {e}")
                    # Fallback to simple calculation
                    pnl_pct = (position.unrealized_pnl / (position.entry_price * position.quantity)) * 100
                    if pnl_pct >= 5:
                        position.grade = DealGrade.EXCELLENT
                    elif pnl_pct >= 2:
                        position.grade = DealGrade.GOOD
                    elif pnl_pct >= 0:
                        position.grade = DealGrade.AVERAGE
                    elif pnl_pct >= -2:
                        position.grade = DealGrade.POOR
                    else:
                        position.grade = DealGrade.CRITICAL

            # Check for significant P&L changes
            pnl_change = abs(float(position.unrealized_pnl - old_pnl))
            if pnl_change > 100:  # Significant change threshold
                await self._generate_alert(
                    deal_id=deal_id,
                    team=MonitoringTeam.PERFORMANCE_TEAM,
                    severity=AlertSeverity.MEDIUM if pnl_change < 500 else AlertSeverity.HIGH,
                    message=f"Significant P&L change: {position.unrealized_pnl:.2f} (change: {position.unrealized_pnl - old_pnl:.2f})"
                )

            return True

        except Exception as e:
            logger.error(f"Failed to update position {deal_id}: {str(e)}")
            return False

    async def close_position(self, deal_id: str, close_price: Decimal,
                           close_time: Optional[datetime] = None) -> bool:
        """
        Close a monitored position.

        Args:
            deal_id: Position identifier
            close_price: Final closing price
            close_time: Closing timestamp (optional)

        Returns:
            bool: True if closed successfully
        """
        try:
            if deal_id not in self.active_positions:
                logger.warning(f"Position {deal_id} not found for closing")
                return False

            position = self.active_positions[deal_id]
            close_time = close_time or datetime.utcnow()

            # Calculate final P&L
            if position.direction == 'BUY':
                final_pnl = (close_price - position.entry_price) * position.quantity
            else:  # SELL
                final_pnl = (position.entry_price - close_price) * position.quantity

            # Update position for closing
            position.current_price = close_price
            position.realized_pnl = final_pnl
            position.unrealized_pnl = Decimal('0')
            position.status = DealStatus.CLOSED
            position.last_update = close_time
            position.duration_hours = (close_time - position.entry_time).total_seconds() / 3600
            
            # Final grade calculation using PerformanceTracker
            if self.performance_tracker:
                try:
                    metrics = await self.performance_tracker.get_deal_performance(deal_id)
                    if metrics:
                        pnl_pct = metrics.get('pnl_percentage', 0)
                        if pnl_pct >= 5:
                            position.grade = DealGrade.EXCELLENT
                        elif pnl_pct >= 2:
                            position.grade = DealGrade.GOOD
                        elif pnl_pct >= 0:
                            position.grade = DealGrade.AVERAGE
                        elif pnl_pct >= -2:
                            position.grade = DealGrade.POOR
                        else:
                            position.grade = DealGrade.CRITICAL
                except Exception as e:
                    # Fallback to simple calculation for final grade
                    if final_pnl > 0:
                        position.grade = DealGrade.GOOD
                    else:
                        position.grade = DealGrade.POOR

            # Move to history
            self.position_history.append(position)
            del self.active_positions[deal_id]

            # Record with performance tracker
            if self.performance_tracker:
                await self.performance_tracker.record_trade_close(
                    deal_id=deal_id,
                    final_pnl=float(final_pnl),
                    duration_hours=position.duration_hours,
                    grade=position.grade
                )

            # Generate closing alert
            profit_status = "PROFIT" if final_pnl > 0 else "LOSS"
            await self._generate_alert(
                deal_id=deal_id,
                team=MonitoringTeam.PERFORMANCE_TEAM,
                severity=AlertSeverity.LOW,
                message=f"Position {deal_id} closed: {profit_status} of {final_pnl:.2f}"
            )

            logger.info(f"Closed position {deal_id}: {profit_status} of {final_pnl:.2f}")
            return True

        except Exception as e:
            logger.error(f"Failed to close position {deal_id}: {str(e)}")
            return False

    async def _team_monitoring_loop(self, team: MonitoringTeam):
        """Main monitoring loop for a specific team."""
        config = self.team_configs[team]
        interval = config['monitoring_interval']

        logger.info(f"Starting {team.value} monitoring loop (interval: {interval}s)")

        try:
            while self.is_monitoring:
                try:
                    # Execute team-specific monitoring
                    if team == MonitoringTeam.RISK_TEAM:
                        await self._risk_team_monitoring()
                    elif team == MonitoringTeam.PERFORMANCE_TEAM:
                        await self._performance_team_monitoring()
                    elif team == MonitoringTeam.TECHNICAL_TEAM:
                        await self._technical_team_monitoring()
                    elif team == MonitoringTeam.COORDINATION_TEAM:
                        await self._coordination_team_monitoring()

                    await asyncio.sleep(interval)

                except Exception as e:
                    logger.error(f"{team.value} monitoring error: {str(e)}")
                    await asyncio.sleep(interval * 2)  # Back off on error

        except asyncio.CancelledError:
            logger.info(f"{team.value} monitoring loop cancelled")
        except Exception as e:
            logger.error(f"Fatal error in {team.value} monitoring loop: {str(e)}")

    async def _risk_team_monitoring(self):
        """Risk team monitoring implementation."""
        config = self.team_configs[MonitoringTeam.RISK_TEAM]

        for deal_id, position in self.active_positions.items():
            try:
                # Check individual position risk
                position_risk = abs(float(position.unrealized_pnl)) / float(position.entry_price * position.quantity)

                if position_risk > config['max_risk_per_deal']:
                    await self._generate_alert(
                        deal_id=deal_id,
                        team=MonitoringTeam.RISK_TEAM,
                        severity=AlertSeverity.HIGH,
                        message=f"Position risk exceeded: {position_risk:.2%} > {config['max_risk_per_deal']:.2%}"
                    )

                # Check stop loss distance
                if position.stop_loss:
                    current_price = float(position.current_price)
                    stop_loss = float(position.stop_loss)
                    entry_price = float(position.entry_price)

                    if position.direction == 'BUY':
                        stop_distance = (current_price - stop_loss) / entry_price
                    else:
                        stop_distance = (stop_loss - current_price) / entry_price

                    if stop_distance < config['stop_loss_threshold']:
                        await self._generate_alert(
                            deal_id=deal_id,
                            team=MonitoringTeam.RISK_TEAM,
                            severity=AlertSeverity.MEDIUM,
                            message=f"Approaching stop loss: distance {stop_distance:.2%}"
                        )

                # Update risk score
                position.risk_score = self._calculate_risk_score(position)

            except Exception as e:
                logger.error(f"Risk monitoring error for {deal_id}: {str(e)}")

        # Check portfolio-level risk
        await self._check_portfolio_risk()

    async def _performance_team_monitoring(self):
        """Performance team monitoring implementation."""
        config = self.team_configs[MonitoringTeam.PERFORMANCE_TEAM]

        for deal_id, position in self.active_positions.items():
            try:
                # Get performance grade from PerformanceTracker
                old_grade = position.grade
                if self.performance_tracker:
                    # Get current performance metrics from PerformanceTracker
                    metrics = await self.performance_tracker.get_deal_performance(deal_id)
                    if metrics:
                        # Simple grading based on P&L percentage
                        pnl_pct = metrics.get('pnl_percentage', 0)
                        if pnl_pct >= 5:
                            position.grade = DealGrade.EXCELLENT
                        elif pnl_pct >= 2:
                            position.grade = DealGrade.GOOD
                        elif pnl_pct >= 0:
                            position.grade = DealGrade.AVERAGE
                        elif pnl_pct >= -2:
                            position.grade = DealGrade.POOR
                        else:
                            position.grade = DealGrade.CRITICAL

                # Alert on grade changes
                if old_grade and old_grade != position.grade:
                    severity = AlertSeverity.LOW
                    if old_grade.value > position.grade.value:  # Grade degraded
                        severity = AlertSeverity.MEDIUM

                    await self._generate_alert(
                        deal_id=deal_id,
                        team=MonitoringTeam.PERFORMANCE_TEAM,
                        severity=severity,
                        message=f"Deal grade changed: {old_grade.value} -> {position.grade.value}"
                    )

                # Check profit targets
                if self.config_manager.get_bool('profit_target_monitoring', False):
                    await self._check_profit_targets(position)

            except Exception as e:
                logger.error(f"Performance monitoring error for {deal_id}: {str(e)}")

    async def _technical_team_monitoring(self):
        """Technical team monitoring implementation."""
        config = self.team_configs[MonitoringTeam.TECHNICAL_TEAM]

        for deal_id, position in self.active_positions.items():
            try:
                # Check for significant price movements
                price_change = abs(float(position.current_price - position.entry_price)) / float(position.entry_price)

                if price_change > config['price_alert_threshold']:
                    await self._generate_alert(
                        deal_id=deal_id,
                        team=MonitoringTeam.TECHNICAL_TEAM,
                        severity=AlertSeverity.LOW,
                        message=f"Significant price movement: {price_change:.2%}"
                    )

                # Check connection and data quality
                time_since_update = (datetime.utcnow() - position.last_update).total_seconds()
                if time_since_update > 300:  # 5 minutes without update
                    await self._generate_alert(
                        deal_id=deal_id,
                        team=MonitoringTeam.TECHNICAL_TEAM,
                        severity=AlertSeverity.MEDIUM,
                        message=f"Stale data: {time_since_update:.0f}s since last update"
                    )

            except Exception as e:
                logger.error(f"Technical monitoring error for {deal_id}: {str(e)}")

    async def _coordination_team_monitoring(self):
        """Coordination team monitoring implementation."""
        config = self.team_configs[MonitoringTeam.COORDINATION_TEAM]

        try:
            # Check for alert escalations
            await self._process_alert_escalations()

            # Generate status reports
            if (datetime.utcnow() - self.last_status_report).total_seconds() > config['status_report_interval']:
                await self._generate_status_report()
                self.last_status_report = datetime.utcnow()

            # Coordinate team activities
            await self._coordinate_team_activities()

        except Exception as e:
            logger.error(f"Coordination monitoring error: {str(e)}")

    def get_deal_grade_from_tracker(self, deal_id: str) -> Optional[DealGrade]:
        """Get deal grade from PerformanceTracker (single source of truth)."""
        try:
            # Query PerformanceTracker for historical grade
            if hasattr(self.performance_tracker, 'get_trade_grade'):
                return self.performance_tracker.get_trade_grade(deal_id)
            
            # Fallback: if PerformanceTracker doesn't have grade yet, return None
            return None
            
        except Exception as e:
            logger.error(f"Error getting deal grade from tracker: {str(e)}")
            return None
            if position.duration_hours < 1:  # Quick profit
                score += 20
            elif position.duration_hours < 6:  # Good timing
                score += 15
            elif position.duration_hours < 24:  # Acceptable
                score += 10
            else:  # Long hold
                score += 5

            # Risk management (10% weight)
            if position.max_drawdown == 0:  # No drawdown
                score += 10
            elif abs(float(position.max_drawdown)) < float(position.entry_price * position.quantity * Decimal('0.01')):  # < 1%
                score += 5

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
            logger.warning(f"Deal grade calculation failed: {str(e)}")
            return DealGrade.C  # Default grade

    def _calculate_risk_score(self, position: DealPosition) -> float:
        """Calculate position risk score."""
        try:
            risk_score = 0.0

            # P&L risk
            pnl_risk = abs(float(position.unrealized_pnl)) / float(position.entry_price * position.quantity)
            risk_score += pnl_risk * 40  # 40% weight

            # Duration risk
            if position.duration_hours > 72:  # > 3 days
                risk_score += 20
            elif position.duration_hours > 24:  # > 1 day
                risk_score += 10

            # Volatility risk (estimated from price movement)
            price_volatility = abs(float(position.current_price - position.entry_price)) / float(position.entry_price)
            risk_score += price_volatility * 30  # 30% weight

            # Confidence risk (inverse)
            confidence_risk = (1.0 - position.confidence_level) * 10  # 10% weight
            risk_score += confidence_risk

            return min(100.0, risk_score)  # Cap at 100

        except Exception as e:
            logger.warning(f"Risk score calculation failed: {str(e)}")
            return 50.0  # Default risk score

    async def _generate_alert(self, deal_id: str, team: MonitoringTeam,
                            severity: AlertSeverity, message: str) -> str:
        """Generate and process a monitoring alert."""
        try:
            alert_id = f"alert_{team.value}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

            alert = MonitoringAlert(
                alert_id=alert_id,
                deal_id=deal_id,
                team=team,
                severity=severity,
                message=message,
                timestamp=datetime.utcnow()
            )

            self.active_alerts[alert_id] = alert
            self.total_alerts_generated += 1

            # Add alert to position
            if deal_id in self.active_positions:
                self.active_positions[deal_id].alerts.append(alert_id)

            # Call alert callback if provided
            if self.alert_callback:
                try:
                    await self.alert_callback(alert)
                except Exception as e:
                    logger.error(f"Alert callback failed: {str(e)}")

            logger.info(f"Generated {severity.value} alert: {message}")
            return alert_id

        except Exception as e:
            logger.error(f"Failed to generate alert: {str(e)}")
            return ""

    async def _check_portfolio_risk(self):
        """Check portfolio-level risk metrics."""
        try:
            if not self.active_positions:
                return

            # Calculate total portfolio P&L
            total_pnl = sum(float(pos.unrealized_pnl) for pos in self.active_positions.values())

            # Calculate total exposure
            total_exposure = sum(float(pos.entry_price * pos.quantity) for pos in self.active_positions.values())

            if total_exposure > 0:
                portfolio_risk = abs(total_pnl) / total_exposure
                max_portfolio_risk = self.team_configs[MonitoringTeam.RISK_TEAM]['max_portfolio_risk']

                if portfolio_risk > max_portfolio_risk:
                    await self._generate_alert(
                        deal_id="PORTFOLIO",
                        team=MonitoringTeam.RISK_TEAM,
                        severity=AlertSeverity.CRITICAL,
                        message=f"Portfolio risk exceeded: {portfolio_risk:.2%} > {max_portfolio_risk:.2%}"
                    )

            # Check position correlations
            await self._check_position_correlations()

        except Exception as e:
            logger.error(f"Portfolio risk check failed: {str(e)}")

    async def _check_position_correlations(self):
        """Check for highly correlated positions."""
        try:
            positions = list(self.active_positions.values())
            correlation_limit = self.team_configs[MonitoringTeam.RISK_TEAM]['correlation_limit']

            for i, pos1 in enumerate(positions):
                for pos2 in positions[i+1:]:
                    # Simple correlation check based on currency pairs
                    if self._are_correlated(pos1.symbol, pos2.symbol):
                        await self._generate_alert(
                            deal_id=f"{pos1.deal_id},{pos2.deal_id}",
                            team=MonitoringTeam.RISK_TEAM,
                            severity=AlertSeverity.MEDIUM,
                            message=f"High correlation detected: {pos1.symbol} and {pos2.symbol}"
                        )

        except Exception as e:
            logger.error(f"Correlation check failed: {str(e)}")

    def _are_correlated(self, symbol1: str, symbol2: str) -> bool:
        """Simple correlation check for currency pairs."""
        try:
            # Extract currencies from symbols (assuming format like EURUSD)
            if len(symbol1) >= 6 and len(symbol2) >= 6:
                base1, quote1 = symbol1[:3], symbol1[3:6]
                base2, quote2 = symbol2[:3], symbol2[3:6]

                # Check if they share currencies
                return base1 in [base2, quote2] or quote1 in [base2, quote2]

            return False

        except Exception:
            return False

    async def _check_profit_targets(self, position: DealPosition):
        """Check if position has reached profit targets."""
        try:
            if not position.take_profit:
                return

            current_price = float(position.current_price)
            take_profit = float(position.take_profit)
            entry_price = float(position.entry_price)

            # Calculate progress towards take profit
            if position.direction == 'BUY':
                if current_price >= take_profit:
                    progress = 1.0  # Reached target
                else:
                    progress = (current_price - entry_price) / (take_profit - entry_price)
            else:  # SELL
                if current_price <= take_profit:
                    progress = 1.0  # Reached target
                else:
                    progress = (entry_price - current_price) / (entry_price - take_profit)

            # Generate alerts based on progress
            if progress >= 1.0:
                await self._generate_alert(
                    deal_id=position.deal_id,
                    team=MonitoringTeam.PERFORMANCE_TEAM,
                    severity=AlertSeverity.LOW,
                    message=f"Take profit target reached: {take_profit}"
                )
            elif progress >= 0.8:
                await self._generate_alert(
                    deal_id=position.deal_id,
                    team=MonitoringTeam.PERFORMANCE_TEAM,
                    severity=AlertSeverity.LOW,
                    message=f"Near take profit: {progress:.1%} progress"
                )

        except Exception as e:
            logger.error(f"Profit target check failed for {position.deal_id}: {str(e)}")

    async def _process_alert_escalations(self):
        """Process and escalate critical alerts."""
        try:
            escalation_threshold = self.team_configs[MonitoringTeam.COORDINATION_TEAM]['escalation_threshold']

            for alert_id, alert in self.active_alerts.items():
                if alert.resolved or alert.escalated:
                    continue

                # Check escalation criteria
                age_minutes = (datetime.utcnow() - alert.timestamp).total_seconds() / 60

                should_escalate = (
                    alert.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY] or
                    age_minutes > 30  # Escalate if unresolved for 30 minutes
                )

                if should_escalate:
                    alert.escalated = True
                    alert.escalation_level += 1
                    self.total_escalations += 1

                    escalated_message = f"ESCALATED: {alert.message} (Level {alert.escalation_level})"

                    await self._generate_alert(
                        deal_id=alert.deal_id,
                        team=MonitoringTeam.COORDINATION_TEAM,
                        severity=AlertSeverity.CRITICAL,
                        message=escalated_message
                    )

                    logger.warning(f"Alert {alert_id} escalated to level {alert.escalation_level}")

        except Exception as e:
            logger.error(f"Alert escalation processing failed: {str(e)}")

    async def _generate_status_report(self):
        """Generate comprehensive status report."""
        try:
            # Collect metrics
            active_count = len(self.active_positions)
            total_pnl = sum(float(pos.unrealized_pnl) for pos in self.active_positions.values())
            avg_grade = self._calculate_average_grade()
            high_risk_positions = sum(1 for pos in self.active_positions.values() if pos.risk_score > 70)
            unresolved_alerts = len([a for a in self.active_alerts.values() if not a.resolved])

            status_message = (
                f"MONITORING STATUS REPORT:\n"
                f"Active Positions: {active_count}\n"
                f"Total Unrealized P&L: {total_pnl:.2f}\n"
                f"Average Deal Grade: {avg_grade}\n"
                f"High Risk Positions: {high_risk_positions}\n"
                f"Unresolved Alerts: {unresolved_alerts}\n"
                f"Monitoring Uptime: {self.monitoring_uptime / 3600:.1f} hours"
            )

            await self._generate_alert(
                deal_id="SYSTEM",
                team=MonitoringTeam.COORDINATION_TEAM,
                severity=AlertSeverity.LOW,
                message=status_message
            )

        except Exception as e:
            logger.error(f"Status report generation failed: {str(e)}")

    def _calculate_average_grade(self) -> str:
        """Calculate average deal grade."""
        try:
            if not self.active_positions:
                return "N/A"

            grades = [pos.grade for pos in self.active_positions.values() if pos.grade]
            if not grades:
                return "N/A"

            # Convert grades to numeric values for averaging
            grade_values = {
                DealGrade.A_PLUS: 12, DealGrade.A: 11, DealGrade.A_MINUS: 10,
                DealGrade.B_PLUS: 9, DealGrade.B: 8, DealGrade.B_MINUS: 7,
                DealGrade.C_PLUS: 6, DealGrade.C: 5, DealGrade.C_MINUS: 4,
                DealGrade.D: 3, DealGrade.F: 1
            }

            avg_value = sum(grade_values.get(grade, 5) for grade in grades) / len(grades)

            # Convert back to grade
            for grade, value in grade_values.items():
                if avg_value >= value - 0.5:
                    return grade.value

            return DealGrade.C.value

        except Exception as e:
            logger.warning(f"Average grade calculation failed: {str(e)}")
            return "N/A"

    async def _coordinate_team_activities(self):
        """Coordinate activities between monitoring teams."""
        try:
            # Check team health and load balancing
            team_loads = {}
            for team in MonitoringTeam:
                team_alerts = [a for a in self.active_alerts.values() if a.team == team and not a.resolved]
                team_loads[team] = len(team_alerts)

            # Adjust monitoring intervals based on load
            for team, load in team_loads.items():
                if load > 10:  # High load
                    self.team_configs[team]['monitoring_interval'] = max(15, self.team_configs[team]['monitoring_interval'] * 0.8)
                elif load < 2:  # Low load
                    self.team_configs[team]['monitoring_interval'] = min(300, self.team_configs[team]['monitoring_interval'] * 1.2)

        except Exception as e:
            logger.error(f"Team coordination failed: {str(e)}")

    # Public interface methods

    async def resolve_alert(self, alert_id: str) -> bool:
        """Manually resolve an alert."""
        try:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.resolved = True
                alert.resolution_time = datetime.utcnow()

                # Move to history
                self.alert_history.append(alert)
                del self.active_alerts[alert_id]

                logger.info(f"Alert {alert_id} resolved manually")
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to resolve alert {alert_id}: {str(e)}")
            return False

    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get comprehensive monitoring summary."""
        try:
            return {
                'active_positions': len(self.active_positions),
                'position_history': len(self.position_history),
                'active_alerts': len(self.active_alerts),
                'alert_history': len(self.alert_history),
                'total_monitored': self.total_positions_monitored,
                'total_alerts': self.total_alerts_generated,
                'total_escalations': self.total_escalations,
                'monitoring_uptime_hours': self.monitoring_uptime / 3600,
                'is_monitoring': self.is_monitoring,
                'team_status': {
                    team.value: {
                        'active': team in self.monitoring_tasks,
                        'interval': self.team_configs[team]['monitoring_interval']
                    }
                    for team in MonitoringTeam
                },
                'risk_metrics': self._get_risk_metrics(),
                'performance_metrics': self._get_performance_metrics_from_tracker()
            }
        except Exception as e:
            logger.error(f"Failed to get monitoring summary: {str(e)}")
            return {'error': str(e)}

    def _get_risk_metrics(self) -> Dict[str, Any]:
        """Get current risk metrics."""
        try:
            if not self.active_positions:
                return {'status': 'no_positions'}

            total_pnl = sum(float(pos.unrealized_pnl) for pos in self.active_positions.values())
            total_exposure = sum(float(pos.entry_price * pos.quantity) for pos in self.active_positions.values())
            high_risk_count = sum(1 for pos in self.active_positions.values() if pos.risk_score > 70)
            avg_risk_score = sum(pos.risk_score for pos in self.active_positions.values()) / len(self.active_positions)

            return {
                'total_unrealized_pnl': total_pnl,
                'total_exposure': total_exposure,
                'portfolio_risk_ratio': abs(total_pnl) / total_exposure if total_exposure > 0 else 0,
                'high_risk_positions': high_risk_count,
                'average_risk_score': avg_risk_score,
                'risk_distribution': self._get_risk_distribution()
            }
        except Exception as e:
            logger.error(f"Risk metrics calculation failed: {str(e)}")
            return {'error': str(e)}

    def _get_performance_metrics_from_tracker(self) -> Dict[str, Any]:
        """Get performance metrics from PerformanceTracker (single source of truth)."""
        try:
            # Get active trades count from our real-time monitoring
            active_count = len(self.active_positions)
            
            # Get historical performance data from PerformanceTracker
            if hasattr(self.performance_tracker, 'get_platform_summary'):
                platform_summary = self.performance_tracker.get_platform_summary()
                
                return {
                    'active_positions_count': active_count,
                    'platform_summary': platform_summary,
                    'real_time_monitoring_status': 'active',
                    'data_source': 'PerformanceTracker'
                }
            
            # Fallback if PerformanceTracker method not available
            return {
                'active_positions_count': active_count,
                'real_time_monitoring_status': 'active',
                'data_source': 'real_time_only',
                'note': 'Historical metrics unavailable'
            }
            
        except Exception as e:
            logger.error(f"Error getting performance metrics from tracker: {str(e)}")
            return {
                'error': str(e),
                'active_positions_count': len(self.active_positions)
            }

    def _get_risk_distribution(self) -> Dict[str, int]:
        """Get risk score distribution."""
        distribution = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}

        for position in self.active_positions.values():
            if position.risk_score < 30:
                distribution['low'] += 1
            elif position.risk_score < 60:
                distribution['medium'] += 1
            elif position.risk_score < 80:
                distribution['high'] += 1
            else:
                distribution['critical'] += 1

        return distribution

    def _get_grade_distribution(self) -> Dict[str, int]:
        """Get grade distribution."""
        distribution = {}

        for position in self.active_positions.values():
            if position.grade:
                grade_key = position.grade.value
                distribution[grade_key] = distribution.get(grade_key, 0) + 1

        return distribution

    def _get_best_performer(self) -> Optional[Dict[str, Any]]:
        """Get best performing position."""
        try:
            if not self.active_positions:
                return None

            best_pos = max(self.active_positions.values(), key=lambda p: float(p.unrealized_pnl))

            return {
                'deal_id': best_pos.deal_id,
                'symbol': best_pos.symbol,
                'unrealized_pnl': float(best_pos.unrealized_pnl),
                'grade': best_pos.grade.value if best_pos.grade else 'N/A'
            }
        except Exception:
            return None

    def _get_worst_performer(self) -> Optional[Dict[str, Any]]:
        """Get worst performing position."""
        try:
            if not self.active_positions:
                return None

            worst_pos = min(self.active_positions.values(), key=lambda p: float(p.unrealized_pnl))

            return {
                'deal_id': worst_pos.deal_id,
                'symbol': worst_pos.symbol,
                'unrealized_pnl': float(worst_pos.unrealized_pnl),
                'grade': worst_pos.grade.value if worst_pos.grade else 'N/A'
            }
        except Exception:
            return None

    def get_recent_alerts(self, count: int = 50) -> List[Dict[str, Any]]:
        """Get recent alerts."""
        try:
            all_alerts = list(self.active_alerts.values()) + self.alert_history
            recent_alerts = sorted(all_alerts, key=lambda a: a.timestamp, reverse=True)[:count]

            return [
                {
                    'alert_id': alert.alert_id,
                    'deal_id': alert.deal_id,
                    'team': alert.team.value,
                    'severity': alert.severity.value,
                    'message': alert.message,
                    'timestamp': alert.timestamp.isoformat(),
                    'resolved': alert.resolved,
                    'escalated': alert.escalated,
                    'escalation_level': alert.escalation_level
                }
                for alert in recent_alerts
            ]
        except Exception as e:
            logger.error(f"Failed to get recent alerts: {str(e)}")
            return []

    def __str__(self) -> str:
        return f"DealMonitoringTeams(active={len(self.active_positions)}, alerts={len(self.active_alerts)}, monitoring={self.is_monitoring})"

    def __repr__(self) -> str:
        return (f"DealMonitoringTeams(active_positions={len(self.active_positions)}, "
                f"active_alerts={len(self.active_alerts)}, "
                f"total_monitored={self.total_positions_monitored}, "
                f"is_monitoring={self.is_monitoring})")

    # Monitoring Callback Methods
    async def _handle_monitoring_event(self, event: Dict[str, Any]):
        """
        Handle monitoring event from external systems.
        
        Args:
            event: Monitoring event data containing type, data, and metadata
        """
        try:
            event_type = event.get('type', 'unknown')
            event_data = event.get('data', {})
            timestamp = event.get('timestamp', datetime.utcnow())
            
            logger.info(f"🔍 Processing monitoring event: {event_type}")
            
            if event_type == 'price_update':
                await self._handle_price_update_event(event_data)
            elif event_type == 'position_update':
                await self._handle_position_update_event(event_data)
            elif event_type == 'risk_alert':
                await self._handle_risk_alert_event(event_data)
            elif event_type == 'system_alert':
                await self._handle_system_alert_event(event_data)
            elif event_type == 'performance_update':
                await self._handle_performance_update_event(event_data)
            else:
                logger.warning(f"⚠️ Unknown monitoring event type: {event_type}")
                
        except Exception as e:
            logger.error(f"❌ Error handling monitoring event: {e}")
    
    async def _handle_price_update_event(self, event_data: Dict[str, Any]):
        """Handle price update events for position monitoring."""
        try:
            symbol = event_data.get('symbol')
            new_price = Decimal(str(event_data.get('price', 0)))
            
            if not symbol or new_price <= 0:
                return
            
            # Update all positions for this symbol
            updated_positions = []
            for deal_id, position in self.active_positions.items():
                if position.symbol == symbol:
                    await self.update_position(deal_id, new_price)
                    updated_positions.append(deal_id)
            
            if updated_positions:
                logger.debug(f"📈 Updated {len(updated_positions)} positions for {symbol}")
                
        except Exception as e:
            logger.error(f"❌ Error handling price update event: {e}")
    
    async def _handle_position_update_event(self, event_data: Dict[str, Any]):
        """Handle position update events from trading engine."""
        try:
            deal_id = event_data.get('deal_id')
            update_type = event_data.get('update_type')
            
            if not deal_id:
                return
            
            if update_type == 'opened':
                await self._handle_position_opened(event_data)
            elif update_type == 'modified':
                await self._handle_position_modified(event_data)
            elif update_type == 'closed':
                await self._handle_position_closed(event_data)
            else:
                logger.warning(f"⚠️ Unknown position update type: {update_type}")
                
        except Exception as e:
            logger.error(f"❌ Error handling position update event: {e}")
    
    async def _handle_risk_alert_event(self, event_data: Dict[str, Any]):
        """Handle risk alert events from risk management system."""
        try:
            alert_type = event_data.get('alert_type', 'general')
            severity = event_data.get('severity', 'medium')
            message = event_data.get('message', 'Risk alert received')
            deal_ids = event_data.get('deal_ids', [])
            
            # Map severity to AlertSeverity enum
            severity_mapping = {
                'low': AlertSeverity.LOW,
                'medium': AlertSeverity.MEDIUM,
                'high': AlertSeverity.HIGH,
                'critical': AlertSeverity.CRITICAL,
                'emergency': AlertSeverity.EMERGENCY
            }
            
            alert_severity = severity_mapping.get(severity.lower(), AlertSeverity.MEDIUM)
            
            # Generate alerts for affected deals
            if deal_ids:
                for deal_id in deal_ids:
                    if deal_id in self.active_positions:
                        await self._generate_alert(
                            deal_id=deal_id,
                            team=MonitoringTeam.RISK_TEAM,
                            severity=alert_severity,
                            message=f"Risk Alert ({alert_type}): {message}"
                        )
            else:
                # Portfolio-level risk alert
                await self._generate_alert(
                    deal_id="PORTFOLIO",
                    team=MonitoringTeam.RISK_TEAM,
                    severity=alert_severity,
                    message=f"Portfolio Risk Alert ({alert_type}): {message}"
                )
                
        except Exception as e:
            logger.error(f"❌ Error handling risk alert event: {e}")
    
    async def _handle_system_alert_event(self, event_data: Dict[str, Any]):
        """Handle system alert events from platform monitoring."""
        try:
            component = event_data.get('component', 'system')
            alert_message = event_data.get('message', 'System alert received')
            severity = event_data.get('severity', 'medium')
            
            # Map severity
            severity_mapping = {
                'low': AlertSeverity.LOW,
                'medium': AlertSeverity.MEDIUM,
                'high': AlertSeverity.HIGH,
                'critical': AlertSeverity.CRITICAL,
                'emergency': AlertSeverity.EMERGENCY
            }
            
            alert_severity = severity_mapping.get(severity.lower(), AlertSeverity.MEDIUM)
            
            # Generate system alert
            await self._generate_alert(
                deal_id="SYSTEM",
                team=MonitoringTeam.COORDINATION_TEAM,
                severity=alert_severity,
                message=f"System Alert ({component}): {alert_message}"
            )
            
        except Exception as e:
            logger.error(f"❌ Error handling system alert event: {e}")
    
    async def _handle_performance_update_event(self, event_data: Dict[str, Any]):
        """Handle performance update events from performance tracker."""
        try:
            deal_id = event_data.get('deal_id')
            performance_data = event_data.get('performance_data', {})
            
            if not deal_id or deal_id not in self.active_positions:
                return
            
            position = self.active_positions[deal_id]
            
            # Update position with performance data
            if 'grade' in performance_data:
                old_grade = position.grade
                new_grade = DealGrade(performance_data['grade'])
                position.grade = new_grade
                
                # Alert on significant grade changes
                if old_grade and old_grade != new_grade:
                    severity = AlertSeverity.LOW
                    if old_grade.value > new_grade.value:  # Grade degraded
                        severity = AlertSeverity.MEDIUM
                    
                    await self._generate_alert(
                        deal_id=deal_id,
                        team=MonitoringTeam.PERFORMANCE_TEAM,
                        severity=severity,
                        message=f"Performance grade updated: {old_grade.value} → {new_grade.value}"
                    )
            
            # Update other performance metrics
            if 'confidence_level' in performance_data:
                position.confidence_level = performance_data['confidence_level']
            
            if 'risk_score' in performance_data:
                position.risk_score = performance_data['risk_score']
                
        except Exception as e:
            logger.error(f"❌ Error handling performance update event: {e}")
    
    async def _process_deal_update(self, deal_data: Dict[str, Any]):
        """
        Process deal monitoring updates from external systems.
        
        Args:
            deal_data: Deal update data containing position info and metrics
        """
        try:
            deal_id = deal_data.get('deal_id')
            if not deal_id:
                logger.warning("⚠️ Deal update missing deal_id")
                return
            
            update_type = deal_data.get('update_type', 'status')
            logger.debug(f"📊 Processing deal update: {deal_id} ({update_type})")
            
            if update_type == 'price_update':
                await self._process_price_update(deal_id, deal_data)
            elif update_type == 'pnl_update':
                await self._process_pnl_update(deal_id, deal_data)
            elif update_type == 'risk_update':
                await self._process_risk_update(deal_id, deal_data)
            elif update_type == 'status_update':
                await self._process_status_update(deal_id, deal_data)
            elif update_type == 'metrics_update':
                await self._process_metrics_update(deal_id, deal_data)
            else:
                logger.warning(f"⚠️ Unknown deal update type: {update_type}")
                
        except Exception as e:
            logger.error(f"❌ Error processing deal update: {e}")
    
    async def _process_price_update(self, deal_id: str, deal_data: Dict[str, Any]):
        """Process price update for a specific deal."""
        try:
            if deal_id not in self.active_positions:
                logger.warning(f"⚠️ Deal {deal_id} not found for price update")
                return
            
            new_price = deal_data.get('current_price')
            if new_price is not None:
                await self.update_position(deal_id, Decimal(str(new_price)))
                
        except Exception as e:
            logger.error(f"❌ Error processing price update for {deal_id}: {e}")
    
    async def _process_pnl_update(self, deal_id: str, deal_data: Dict[str, Any]):
        """Process P&L update for a specific deal."""
        try:
            if deal_id not in self.active_positions:
                return
            
            position = self.active_positions[deal_id]
            
            # Update P&L metrics
            if 'unrealized_pnl' in deal_data:
                old_pnl = position.unrealized_pnl
                new_pnl = Decimal(str(deal_data['unrealized_pnl']))
                position.unrealized_pnl = new_pnl
                
                # Check for significant P&L changes
                pnl_change = abs(float(new_pnl - old_pnl))
                if pnl_change > 100:  # Significant change threshold
                    await self._generate_alert(
                        deal_id=deal_id,
                        team=MonitoringTeam.PERFORMANCE_TEAM,
                        severity=AlertSeverity.MEDIUM if pnl_change < 500 else AlertSeverity.HIGH,
                        message=f"Significant P&L change: {new_pnl:.2f} (Δ{new_pnl - old_pnl:+.2f})"
                    )
            
        except Exception as e:
            logger.error(f"❌ Error processing P&L update for {deal_id}: {e}")
    
    async def _process_risk_update(self, deal_id: str, deal_data: Dict[str, Any]):
        """Process risk update for a specific deal."""
        try:
            if deal_id not in self.active_positions:
                return
            
            position = self.active_positions[deal_id]
            
            # Update risk metrics
            if 'risk_score' in deal_data:
                old_risk = position.risk_score
                new_risk = deal_data['risk_score']
                position.risk_score = new_risk
                
                # Alert on significant risk changes
                if abs(new_risk - old_risk) > 20:  # 20 point change threshold
                    severity = AlertSeverity.MEDIUM if new_risk > old_risk else AlertSeverity.LOW
                    await self._generate_alert(
                        deal_id=deal_id,
                        team=MonitoringTeam.RISK_TEAM,
                        severity=severity,
                        message=f"Risk score change: {old_risk:.1f} → {new_risk:.1f}"
                    )
            
        except Exception as e:
            logger.error(f"❌ Error processing risk update for {deal_id}: {e}")
    
    async def _process_status_update(self, deal_id: str, deal_data: Dict[str, Any]):
        """Process status update for a specific deal."""
        try:
            if deal_id not in self.active_positions:
                return
            
            position = self.active_positions[deal_id]
            
            # Update position status
            if 'status' in deal_data:
                old_status = position.status
                new_status = DealStatus(deal_data['status'])
                position.status = new_status
                
                # Alert on status changes
                if old_status != new_status:
                    await self._generate_alert(
                        deal_id=deal_id,
                        team=MonitoringTeam.COORDINATION_TEAM,
                        severity=AlertSeverity.LOW,
                        message=f"Deal status changed: {old_status.value} → {new_status.value}"
                    )
            
        except Exception as e:
            logger.error(f"❌ Error processing status update for {deal_id}: {e}")
    
    async def _process_metrics_update(self, deal_id: str, deal_data: Dict[str, Any]):
        """Process general metrics update for a specific deal."""
        try:
            if deal_id not in self.active_positions:
                return
            
            position = self.active_positions[deal_id]
            
            # Update various metrics
            metrics = deal_data.get('metrics', {})
            
            for metric_name, metric_value in metrics.items():
                if hasattr(position, metric_name):
                    setattr(position, metric_name, metric_value)
                else:
                    # Store in metadata if not a direct attribute
                    position.metadata[metric_name] = metric_value
            
            # Update last update time
            position.last_update = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"❌ Error processing metrics update for {deal_id}: {e}")
    
    async def _handle_position_opened(self, event_data: Dict[str, Any]):
        """Handle new position opened event."""
        try:
            deal_id = event_data.get('deal_id')
            symbol = event_data.get('symbol')
            direction = event_data.get('direction')
            entry_price = Decimal(str(event_data.get('entry_price', 0)))
            quantity = Decimal(str(event_data.get('quantity', 0)))
            
            if not all([deal_id, symbol, direction, entry_price > 0, quantity > 0]):
                logger.warning(f"⚠️ Incomplete position opened data: {event_data}")
                return
            
            # Create a mock trade signal for compatibility
            from ..core.data_contracts import TradeSignal, TradeDirection, PriorityLevel
            
            mock_signal = TradeSignal(
                id=f"auto_{deal_id}",
                symbol=symbol,
                direction=TradeDirection(direction.upper()),
                entry_price=entry_price,
                stop_loss=event_data.get('stop_loss'),
                take_profit=event_data.get('take_profit'),
                confidence=event_data.get('confidence', 0.5),
                timestamp=datetime.utcnow(),
                priority=PriorityLevel.MEDIUM,
                agent_source=event_data.get('agent_source', 'external'),
                reasoning="External position notification"
            )
            
            # Add to monitoring
            await self.add_position(deal_id, mock_signal, entry_price, quantity)
            
        except Exception as e:
            logger.error(f"❌ Error handling position opened event: {e}")
    
    async def _handle_position_modified(self, event_data: Dict[str, Any]):
        """Handle position modification event."""
        try:
            deal_id = event_data.get('deal_id')
            if deal_id not in self.active_positions:
                return
            
            position = self.active_positions[deal_id]
            
            # Update modified fields
            if 'stop_loss' in event_data:
                old_sl = position.stop_loss
                new_sl = Decimal(str(event_data['stop_loss'])) if event_data['stop_loss'] else None
                position.stop_loss = new_sl
                
                if old_sl != new_sl:
                    await self._generate_alert(
                        deal_id=deal_id,
                        team=MonitoringTeam.RISK_TEAM,
                        severity=AlertSeverity.LOW,
                        message=f"Stop loss modified: {old_sl} → {new_sl}"
                    )
            
            if 'take_profit' in event_data:
                old_tp = position.take_profit
                new_tp = Decimal(str(event_data['take_profit'])) if event_data['take_profit'] else None
                position.take_profit = new_tp
                
                if old_tp != new_tp:
                    await self._generate_alert(
                        deal_id=deal_id,
                        team=MonitoringTeam.PERFORMANCE_TEAM,
                        severity=AlertSeverity.LOW,
                        message=f"Take profit modified: {old_tp} → {new_tp}"
                    )
            
        except Exception as e:
            logger.error(f"❌ Error handling position modified event: {e}")
    
    async def _handle_position_closed(self, event_data: Dict[str, Any]):
        """Handle position closure event."""
        try:
            deal_id = event_data.get('deal_id')
            close_price = event_data.get('close_price')
            close_time = event_data.get('close_time')
            
            if not deal_id or close_price is None:
                logger.warning(f"⚠️ Incomplete position closed data: {event_data}")
                return
            
            # Parse close time if provided as string
            if isinstance(close_time, str):
                close_time = datetime.fromisoformat(close_time.replace('Z', '+00:00'))
            elif close_time is None:
                close_time = datetime.utcnow()
            
            # Close the position
            await self.close_position(deal_id, Decimal(str(close_price)), close_time)
            
        except Exception as e:
            logger.error(f"❌ Error handling position closed event: {e}")
    async def perform_enhanced_security_monitoring(self, deal_id: str, deal_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform enhanced security monitoring for trading deals.
        Comprehensive security analysis with anomaly detection.
        """
        try:
            security_analysis = {
                'deal_id': deal_id,
                'timestamp': datetime.utcnow().isoformat(),
                'security_checks': {},
                'risk_assessments': {},
                'anomalies_detected': [],
                'compliance_status': 'COMPLIANT',
                'recommendations': []
            }
            
            # Perform various security checks
            await self._check_position_size_limits(deal_id, deal_data, security_analysis)
            await self._check_trading_patterns(deal_id, deal_data, security_analysis)
            await self._check_market_manipulation_signs(deal_id, deal_data, security_analysis)
            await self._check_compliance_violations(deal_id, deal_data, security_analysis)
            await self._check_unauthorized_access(deal_id, deal_data, security_analysis)
            await self._check_data_integrity(deal_id, deal_data, security_analysis)
            
            # Generate risk score
            security_analysis['overall_risk_score'] = await self._calculate_security_risk_score(security_analysis)
            
            # Determine final compliance status
            await self._determine_compliance_status(security_analysis)
            
            # Store security analysis
            await self._store_security_analysis(security_analysis)
            
            return security_analysis
            
        except Exception as e:
            self.logger.error(f"Enhanced security monitoring failed for deal {deal_id}: {e}")
            return {
                'deal_id': deal_id,
                'timestamp': datetime.utcnow().isoformat(),
                'status': 'ERROR',
                'error': str(e)
            }
    
    async def _check_position_size_limits(self, deal_id: str, deal_data: Dict[str, Any], analysis: Dict[str, Any]):
        """Check position size against security limits"""
        try:
            position_size = deal_data.get('position_size', 0)
            symbol = deal_data.get('symbol', '')
            
            # Get security limits for this symbol
            max_position_size = self.config_manager.get_float(f'security.max_position_size.{symbol}', 10000)
            
            if position_size > max_position_size:
                analysis['anomalies_detected'].append({
                    'type': 'POSITION_SIZE_VIOLATION',
                    'severity': 'HIGH',
                    'current_size': position_size,
                    'max_allowed': max_position_size,
                    'violation_percentage': ((position_size - max_position_size) / max_position_size) * 100
                })
                analysis['security_checks']['position_size'] = 'VIOLATION'
            else:
                analysis['security_checks']['position_size'] = 'PASSED'
                
        except Exception as e:
            analysis['security_checks']['position_size'] = f'ERROR: {e}'
    
    async def _check_trading_patterns(self, deal_id: str, deal_data: Dict[str, Any], analysis: Dict[str, Any]):
        """Check for suspicious trading patterns"""
        try:
            # Check for rapid-fire trading
            recent_trades = await self._get_recent_trades_count(deal_data.get('symbol', ''), minutes=5)
            if recent_trades > 10:
                analysis['anomalies_detected'].append({
                    'type': 'RAPID_TRADING',
                    'severity': 'MEDIUM',
                    'recent_trades': recent_trades,
                    'threshold': 10
                })
            
            # Check for unusual timing patterns
            trading_hour = datetime.utcnow().hour
            if trading_hour < 6 or trading_hour > 22:  # Outside normal trading hours
                analysis['anomalies_detected'].append({
                    'type': 'OFF_HOURS_TRADING',
                    'severity': 'LOW',
                    'trading_hour': trading_hour
                })
            
            analysis['security_checks']['trading_patterns'] = 'PASSED' if not analysis['anomalies_detected'] else 'SUSPICIOUS'
            
        except Exception as e:
            analysis['security_checks']['trading_patterns'] = f'ERROR: {e}'
    
    async def _check_market_manipulation_signs(self, deal_id: str, deal_data: Dict[str, Any], analysis: Dict[str, Any]):
        """Check for potential market manipulation signs"""
        try:
            # Check for price manipulation patterns
            entry_price = deal_data.get('entry_price', 0)
            current_price = deal_data.get('current_price', entry_price)
            
            if entry_price > 0:
                price_deviation = abs(current_price - entry_price) / entry_price
                if price_deviation > 0.05:  # 5% deviation
                    analysis['anomalies_detected'].append({
                        'type': 'UNUSUAL_PRICE_MOVEMENT',
                        'severity': 'MEDIUM',
                        'price_deviation_percent': price_deviation * 100,
                        'threshold_percent': 5.0
                    })
            
            # Check for coordinated trading (placeholder)
            coordinated_risk = await self._check_coordinated_trading(deal_data)
            if coordinated_risk > 0.7:
                analysis['anomalies_detected'].append({
                    'type': 'COORDINATED_TRADING_RISK',
                    'severity': 'HIGH',
                    'risk_score': coordinated_risk
                })
            
            analysis['security_checks']['market_manipulation'] = 'PASSED'
            
        except Exception as e:
            analysis['security_checks']['market_manipulation'] = f'ERROR: {e}'
    
    async def _check_compliance_violations(self, deal_id: str, deal_data: Dict[str, Any], analysis: Dict[str, Any]):
        """Check for regulatory compliance violations"""
        try:
            # Check leverage limits
            leverage = deal_data.get('leverage', 1)
            max_leverage = self.config_manager.get_float('compliance.max_leverage', 30)
            
            if leverage > max_leverage:
                analysis['anomalies_detected'].append({
                    'type': 'LEVERAGE_VIOLATION',
                    'severity': 'HIGH',
                    'current_leverage': leverage,
                    'max_allowed': max_leverage
                })
            
            # Check symbol trading permissions
            symbol = deal_data.get('symbol', '')
            allowed_symbols = self.config_manager.get_list('compliance.allowed_symbols', [])
            
            if allowed_symbols and symbol not in allowed_symbols:
                analysis['anomalies_detected'].append({
                    'type': 'UNAUTHORIZED_SYMBOL',
                    'severity': 'HIGH',
                    'symbol': symbol,
                    'allowed_symbols': allowed_symbols
                })
            
            analysis['security_checks']['compliance'] = 'PASSED'
            
        except Exception as e:
            analysis['security_checks']['compliance'] = f'ERROR: {e}'
    
    async def _check_unauthorized_access(self, deal_id: str, deal_data: Dict[str, Any], analysis: Dict[str, Any]):
        """Check for unauthorized access patterns"""
        try:
            # Check if deal originates from authorized agents/systems
            originating_agent = deal_data.get('originating_agent', '')
            authorized_agents = self.config_manager.get_list('security.authorized_agents', [])
            
            if authorized_agents and originating_agent not in authorized_agents:
                analysis['anomalies_detected'].append({
                    'type': 'UNAUTHORIZED_AGENT',
                    'severity': 'CRITICAL',
                    'agent': originating_agent,
                    'authorized_agents': authorized_agents
                })
            
            # Check IP/location if available
            source_ip = deal_data.get('source_ip', '')
            if source_ip and not await self._is_authorized_ip(source_ip):
                analysis['anomalies_detected'].append({
                    'type': 'UNAUTHORIZED_IP',
                    'severity': 'HIGH',
                    'source_ip': source_ip
                })
            
            analysis['security_checks']['authorization'] = 'PASSED'
            
        except Exception as e:
            analysis['security_checks']['authorization'] = f'ERROR: {e}'
    
    async def _check_data_integrity(self, deal_id: str, deal_data: Dict[str, Any], analysis: Dict[str, Any]):
        """Check data integrity and validation"""
        try:
            # Check required fields
            required_fields = ['symbol', 'position_size', 'entry_price', 'direction']
            missing_fields = [field for field in required_fields if not deal_data.get(field)]
            
            if missing_fields:
                analysis['anomalies_detected'].append({
                    'type': 'MISSING_DATA_FIELDS',
                    'severity': 'HIGH',
                    'missing_fields': missing_fields
                })
            
            # Check data consistency
            position_size = deal_data.get('position_size', 0)
            entry_price = deal_data.get('entry_price', 0)
            
            if position_size <= 0:
                analysis['anomalies_detected'].append({
                    'type': 'INVALID_POSITION_SIZE',
                    'severity': 'CRITICAL',
                    'position_size': position_size
                })
            
            if entry_price <= 0:
                analysis['anomalies_detected'].append({
                    'type': 'INVALID_ENTRY_PRICE',
                    'severity': 'CRITICAL',
                    'entry_price': entry_price
                })
            
            analysis['security_checks']['data_integrity'] = 'PASSED'
            
        except Exception as e:
            analysis['security_checks']['data_integrity'] = f'ERROR: {e}'
    
    async def _calculate_security_risk_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate overall security risk score"""
        try:
            anomalies = analysis.get('anomalies_detected', [])
            
            if not anomalies:
                return 0.0
            
            # Weight by severity
            severity_weights = {
                'LOW': 0.2,
                'MEDIUM': 0.5,
                'HIGH': 0.8,
                'CRITICAL': 1.0
            }
            
            total_score = sum(severity_weights.get(anomaly.get('severity', 'LOW'), 0.2) for anomaly in anomalies)
            max_possible_score = len(anomalies) * 1.0
            
            return min(total_score / max_possible_score if max_possible_score > 0 else 0.0, 1.0)
            
        except Exception as e:
            self.logger.error(f"Risk score calculation failed: {e}")
            return 0.5  # Default medium risk
    
    async def _determine_compliance_status(self, analysis: Dict[str, Any]):
        """Determine final compliance status"""
        try:
            critical_anomalies = [a for a in analysis.get('anomalies_detected', []) if a.get('severity') == 'CRITICAL']
            high_anomalies = [a for a in analysis.get('anomalies_detected', []) if a.get('severity') == 'HIGH']
            
            if critical_anomalies:
                analysis['compliance_status'] = 'CRITICAL_VIOLATION'
            elif high_anomalies:
                analysis['compliance_status'] = 'VIOLATION'
            elif analysis.get('anomalies_detected', []):
                analysis['compliance_status'] = 'WARNING'
            else:
                analysis['compliance_status'] = 'COMPLIANT'
                
        except Exception as e:
            analysis['compliance_status'] = 'ERROR'
            self.logger.error(f"Compliance status determination failed: {e}")
    
    async def _store_security_analysis(self, analysis: Dict[str, Any]):
        """Store security analysis for audit and trending"""
        try:
            # Store in memory buffer
            if not hasattr(self, '_security_analysis_history'):
                from collections import deque
                self._security_analysis_history = deque(maxlen=1000)
            
            self._security_analysis_history.append(analysis)
            
            # Store in database if available
            if hasattr(self, 'database') and self.database:
                await self.database.store_security_analysis(analysis)
                
        except Exception as e:
            self.logger.error(f"Failed to store security analysis: {e}")
    
    # Helper methods
    async def _get_recent_trades_count(self, symbol: str, minutes: int = 5) -> int:
        """Get count of recent trades for symbol"""
        # Placeholder - would query actual trade history
        return 0
    
    async def _check_coordinated_trading(self, deal_data: Dict[str, Any]) -> float:
        """Check for coordinated trading patterns"""
        # Placeholder for sophisticated analysis
        return 0.0
    
    async def _is_authorized_ip(self, ip_address: str) -> bool:
        """Check if IP address is authorized"""
        authorized_ips = self.config_manager.get_list('security.authorized_ips', [])
        return not authorized_ips or ip_address in authorized_ips
    
    def get_security_monitoring_summary(self) -> Dict[str, Any]:
        """Get summary of security monitoring activities"""
        try:
            if not hasattr(self, '_security_analysis_history'):
                return {'status': 'no_data', 'message': 'No security monitoring data available'}
            
            recent_analyses = list(self._security_analysis_history)[-50:]  # Last 50 analyses
            
            total_analyses = len(recent_analyses)
            violations = len([a for a in recent_analyses if a.get('compliance_status') in ['VIOLATION', 'CRITICAL_VIOLATION']])
            warnings = len([a for a in recent_analyses if a.get('compliance_status') == 'WARNING'])
            
            # Count anomaly types
            anomaly_types = {}
            for analysis in recent_analyses:
                for anomaly in analysis.get('anomalies_detected', []):
                    anomaly_type = anomaly.get('type', 'UNKNOWN')
                    anomaly_types[anomaly_type] = anomaly_types.get(anomaly_type, 0) + 1
            
            return {
                'total_analyses': total_analyses,
                'violations': violations,
                'warnings': warnings,
                'compliance_rate': ((total_analyses - violations) / total_analyses * 100) if total_analyses > 0 else 100,
                'most_common_anomalies': sorted(anomaly_types.items(), key=lambda x: x[1], reverse=True)[:5],
                'last_analysis_time': recent_analyses[-1].get('timestamp') if recent_analyses else None
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}