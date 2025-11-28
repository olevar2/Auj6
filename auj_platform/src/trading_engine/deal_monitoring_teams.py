"""
Deal Monitoring Teams for AUJ Platform

This module implements real-time monitoring of active trades with focus on
risk alerts and real-time position surveillance.

Key Features:
- Real-time monitoring of open positions
- Risk alert system for active trades
- Team-based monitoring approach
- Automatic escalation protocols
- Integration with PerformanceTracker and HierarchyManager

FIXES IMPLEMENTED:
- Added missing DealPosition attributes (max_profit, max_drawdown, realized_pnl, grade)
- Added position_history initialization in __init__
- Fixed DealGrade values (using correct enum values)
- Added HierarchyManager integration
- Removed unreachable code blocks
"""

import asyncio
import logging
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Callable
from dataclasses import dataclass, field

from ..core.data_contracts import TradeSignal, DealGrade
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
    """
    Represents a trading position being monitored in real-time.
    
    FIXED: Added missing attributes (max_profit, max_drawdown, realized_pnl, grade)
    """
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
    
    # FIXED: NEW ATTRIBUTES - Missing in original
    max_profit: Decimal = Decimal('0')
    max_drawdown: Decimal = Decimal('0')
    realized_pnl: Optional[Decimal] = None
    grade: Optional[DealGrade] = None
    
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
    oversight of all active trading positions and risk management.
    
    FIXES IMPLEMENTED:
    - Added HierarchyManager integration
    - Added position_history initialization
    - Fixed DealGrade enum usage
    - Removed unreachable code
    """

    def __init__(self,
                 performance_tracker: PerformanceTracker,
                 hierarchy_manager=None,  # FIXED: NEW - HierarchyManager integration
                 risk_manager=None,
                 alert_callback: Optional[Callable] = None):
        """
        Initialize the Deal Monitoring Teams system.

        Args:
            performance_tracker: Required performance tracking instance
            hierarchy_manager: HierarchyManager for recording trade results
            risk_manager: Dynamic risk manager instance
            alert_callback: Optional callback for alert notifications
        """
        from ..core.unified_config import get_unified_config
        self.config_manager = get_unified_config()
        
        if not performance_tracker:
            raise ValueError("PerformanceTracker is required")
        
        self.performance_tracker = performance_tracker
        self.hierarchy_manager = hierarchy_manager  # FIXED: NEW
        self.risk_manager = risk_manager
        self.alert_callback = alert_callback

        # Active positions being monitored (real-time only)
        self.active_positions: Dict[str, DealPosition] = {}
        
        # FIXED: NEW - Position history initialization (was missing)
        self.position_history: List[DealPosition] = []

        # Alert management
        self.active_alerts: Dict[str, MonitoringAlert] = {}
        self.alert_history: List[MonitoringAlert] = []
        
        # FIXED: Moved from initialize() to __init__
        self.position_alerts: defaultdict = defaultdict(list)

        # Team configuration
        self.team_configs = {
            MonitoringTeam.RISK_TEAM: {
                'max_risk_per_deal': 0.02,
                'max_portfolio_risk': 0.10,
                'stop_loss_threshold': 0.015,
                'correlation_limit': 0.7,
                'monitoring_interval': 30
            },
            MonitoringTeam.TECHNICAL_TEAM: {
                'price_alert_threshold': 0.005,
                'volume_alert_threshold': 2.0,
                'spread_alert_threshold': 0.0005,
                'connection_monitoring': True,
                'monitoring_interval': 15
            },
            MonitoringTeam.COORDINATION_TEAM: {
                'escalation_threshold': 3,
                'team_coordination_interval': 180,
                'status_report_interval': 600,
                'monitoring_interval': 120
            },
            MonitoringTeam.PERFORMANCE_TEAM: {
                'performance_check_interval': 300,
                'profitability_threshold': 0.005,
                'loss_alert_threshold': 0.02,
                'monitoring_interval': 60
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

        logger.info("Deal Monitoring Teams system initialized with HierarchyManager integration")

    async def initialize(self) -> None:
        """Initialize the component with required dependencies."""
        logger.info("Initializing Deal Monitoring Teams...")
        
        # Validate team configurations
        self._validate_team_configurations()
        
        # Initialize monitoring state
        self.is_monitoring = False
        self.monitoring_tasks = {}
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

            for team, task in self.monitoring_tasks.items():
                if not task.cancelled():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                logger.info(f"Stopped {team.value} monitoring")

            self.monitoring_tasks.clear()

            if self.last_monitoring_start:
                uptime = (datetime.utcnow() - self.last_monitoring_start).total_seconds()
                self.monitoring_uptime += uptime

            logger.info("All monitoring teams stopped")

        except Exception as e:
            logger.error(f"Error stopping monitoring: {str(e)}")

    async def add_position(self, 
                          deal_id: str, 
                          symbol: str,
                          direction: str,
                          entry_price: Decimal, 
                          quantity: Decimal,
                          stop_loss: Optional[Decimal] = None,
                          take_profit: Optional[Decimal] = None,
                          entry_time: Optional[datetime] = None,
                          confidence_level: float = 0.0,
                          agent_source: str = "",
                          strategy_type: str = "",
                          **kwargs) -> bool:
        """
        Add a new position to monitoring.

        Args:
            deal_id: Unique identifier for the deal
            symbol: Trading symbol
            direction: BUY or SELL
            entry_price: Actual entry price
            quantity: Position size
            stop_loss: Stop loss level
            take_profit: Take profit level
            entry_time: Entry timestamp
            confidence_level: Signal confidence
            agent_source: Source agent name
            strategy_type: Strategy identifier

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
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                current_price=entry_price,
                quantity=quantity,
                entry_time=entry_time or datetime.utcnow(),
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence_level=confidence_level,
                agent_source=agent_source,
                strategy_type=strategy_type,
                metadata=kwargs
            )

            self.active_positions[deal_id] = position
            self.total_positions_monitored += 1

            # Generate initial monitoring alert
            await self._generate_alert(
                deal_id=deal_id,
                team=MonitoringTeam.COORDINATION_TEAM,
                severity=AlertSeverity.LOW,
                message=f"New position {deal_id} added: {symbol} {direction}"
            )

            logger.info(f"✅ Added position {deal_id} to monitoring: {symbol} {direction}")
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

            # Update max profit/drawdown
            current_pnl = position.unrealized_pnl
            if current_pnl > position.max_profit:
                position.max_profit = current_pnl

            if current_pnl < position.max_drawdown:
                position.max_drawdown = current_pnl

            # Update duration
            position.duration_hours = (position.last_update - position.entry_time).total_seconds() / 3600

            # FIXED: Update grade using correct DealGrade enum values
            position.grade = self._calculate_deal_grade(position)

            # Check for significant P&L changes
            pnl_change = abs(float(current_pnl - old_pnl))
            if pnl_change > 100:
                await self._generate_alert(
                    deal_id=deal_id,
                    team=MonitoringTeam.PERFORMANCE_TEAM,
                    severity=AlertSeverity.MEDIUM if pnl_change < 500 else AlertSeverity.HIGH,
                    message=f"Significant P&L change: {current_pnl:.2f} (change: {current_pnl - old_pnl:.2f})"
                )

            return True

        except Exception as e:
            logger.error(f"Failed to update position {deal_id}: {str(e)}")
            return False

    async def close_position(self, deal_id: str, close_price: Decimal,
                           close_time: Optional[datetime] = None) -> bool:
        """
        Close a monitored position.
        
        FIXED: Added HierarchyManager integration.

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
            if position.direction.upper() == 'BUY':
                final_pnl = (close_price - position.entry_price) * position.quantity
            else:  # SELL
                final_pnl = (position.entry_price - close_price) * position.quantity

            # Update position for closing
            position.current_price = close_price
            position.realized_pnl = final_pnl
            position.status = DealStatus.CLOSED
            position.last_update = close_time
            position.duration_hours = (close_time - position.entry_time).total_seconds() / 3600
            
            # FIXED: Calculate final grade using correct enum values
            position.grade = self._calculate_deal_grade(position)

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

            # FIXED: NEW - Record with HierarchyManager
            if self.hierarchy_manager and position.agent_source:
                try:
                    await self.hierarchy_manager.record_trade_result(
                        agent_name=position.agent_source,
                        trade_result={
                            'pnl': float(final_pnl),
                            'grade': position.grade.value if position.grade else 'F',
                            'duration_hours': position.duration_hours,
                            'symbol': position.symbol,
                            'direction': position.direction,
                            'confidence': position.confidence_level
                        }
                    )
                    logger.info(f"✅ Recorded trade result with HierarchyManager for agent {position.agent_source}")
                except Exception as e:
                    logger.error(f"❌ Failed to record with HierarchyManager: {e}")

            # Generate closing alert
            profit_status = "PROFIT" if final_pnl > 0 else "LOSS"
            await self._generate_alert(
                deal_id=deal_id,
                team=MonitoringTeam.PERFORMANCE_TEAM,
                severity=AlertSeverity.LOW,
                message=f"Position {deal_id} closed: {profit_status} of {final_pnl:.2f}, Grade: {position.grade.value if position.grade else 'N/A'}"
            )

            logger.info(f"✅ Closed position {deal_id}: {profit_status} of {final_pnl:.2f}, Grade: {position.grade.value if position.grade else 'N/A'}")
            return True

        except Exception as e:
            logger.error(f"Failed to close position {deal_id}: {str(e)}")
            return False

    def _calculate_deal_grade(self, position: DealPosition) -> DealGrade:
        """
        Calculate deal grade based on performance.
        
        FIXED: Uses correct DealGrade enum values (A_PLUS, A, B, C, D, F)
        NOT the incorrect values (EXCELLENT, GOOD, AVERAGE)
        """
        try:
            # Use realized_pnl if closed, otherwise unrealized
            pnl = position.realized_pnl if position.realized_pnl is not None else position.unrealized_pnl
            
            if position.entry_price <= 0 or position.quantity <= 0:
                return DealGrade.F
            
            # Calculate P&L percentage
            pnl_pct = float(pnl / (position.entry_price * position.quantity)) * 100
            
            # FIXED: Using correct DealGrade enum values
            if pnl_pct >= 5.0:
                grade = DealGrade.A_PLUS
            elif pnl_pct >= 3.0:
                grade = DealGrade.A
            elif pnl_pct >= 1.5:
                grade = DealGrade.A_MINUS
            elif pnl_pct >= 0.5:
                grade = DealGrade.B_PLUS
            elif pnl_pct >= 0.0:
                grade = DealGrade.B
            elif pnl_pct >= -1.0:
                grade = DealGrade.B_MINUS
            elif pnl_pct >= -2.0:
                grade = DealGrade.C
            elif pnl_pct >= -3.0:
                grade = DealGrade.D
            else:
                grade = DealGrade.F
            
            return grade

        except Exception as e:
            logger.warning(f"Deal grade calculation failed: {str(e)}")
            return DealGrade.C

    async def _team_monitoring_loop(self, team: MonitoringTeam):
        """Main monitoring loop for a specific team."""
        config = self.team_configs[team]
        interval = config['monitoring_interval']

        logger.info(f"Starting {team.value} monitoring loop (interval: {interval}s)")

        try:
            while self.is_monitoring:
                try:
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
                    await asyncio.sleep(interval * 2)

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

                    if position.direction.upper() == 'BUY':
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
        for deal_id, position in self.active_positions.items():
            try:
                old_grade = position.grade
                
                # Update grade
                position.grade = self._calculate_deal_grade(position)

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

        except Exception as e:
            logger.error(f"Coordination monitoring error: {str(e)}")

    def _calculate_risk_score(self, position: DealPosition) -> float:
        """Calculate position risk score."""
        try:
            risk_score = 0.0

            # P&L risk
            pnl_risk = abs(float(position.unrealized_pnl)) / float(position.entry_price * position.quantity)
            risk_score += pnl_risk * 40

            # Duration risk
            if position.duration_hours > 72:
                risk_score += 20
            elif position.duration_hours > 24:
                risk_score += 10

            # Volatility risk
            price_volatility = abs(float(position.current_price - position.entry_price)) / float(position.entry_price)
            risk_score += price_volatility * 30

            # Confidence risk
            confidence_risk = (1.0 - position.confidence_level) * 10
            risk_score += confidence_risk

            return min(100.0, risk_score)

        except Exception as e:
            logger.warning(f"Risk score calculation failed: {str(e)}")
            return 50.0

    async def _generate_alert(self, deal_id: str, team: MonitoringTeam,
                            severity: AlertSeverity, message: str) -> str:
        """Generate and process a monitoring alert."""
        try:
            alert_id = f"alert_{team.value}_{uuid.uuid4().hex[:8]}"

            alert = MonitoringAlert(
                alert_id=alert_id,
                deal_id=deal_id,
                team=team,
                severity=severity,
                message=message,
                timestamp=datetime.utcnow()
            )

            self.active_alerts[alert_id] = alert
            self.alert_history.append(alert)
            self.position_alerts[deal_id].append(alert_id)
            self.total_alerts_generated += 1

            # Call alert callback if provided
            if self.alert_callback:
                try:
                    await self.alert_callback(alert)
                except Exception as e:
                    logger.error(f"Alert callback failed: {e}")

            logger.debug(f"Generated alert {alert_id}: {message}")
            return alert_id

        except Exception as e:
            logger.error(f"Failed to generate alert: {str(e)}")
            return ""

    async def _process_alert_escalations(self):
        """Process alert escalations."""
        config = self.team_configs[MonitoringTeam.COORDINATION_TEAM]
        escalation_threshold = config['escalation_threshold']

        for deal_id, alert_ids in self.position_alerts.items():
            unresolved_count = sum(1 for aid in alert_ids if not self.active_alerts.get(aid, MonitoringAlert(
                alert_id="", deal_id="", team=MonitoringTeam.COORDINATION_TEAM,
                severity=AlertSeverity.LOW, message="", timestamp=datetime.utcnow()
            )).resolved)

            if unresolved_count >= escalation_threshold:
                await self._escalate_position_alerts(deal_id)

    async def _escalate_position_alerts(self, deal_id: str):
        """Escalate alerts for a position."""
        try:
            await self._generate_alert(
                deal_id=deal_id,
                team=MonitoringTeam.COORDINATION_TEAM,
                severity=AlertSeverity.CRITICAL,
                message=f"Multiple unresolved alerts for position {deal_id} - ESCALATED"
            )
            self.total_escalations += 1
            logger.warning(f"🚨 Escalated alerts for position {deal_id}")

        except Exception as e:
            logger.error(f"Failed to escalate alerts for {deal_id}: {e}")

    async def _generate_status_report(self):
        """Generate monitoring status report."""
        try:
            status = {
                'timestamp': datetime.utcnow().isoformat(),
                'active_positions': len(self.active_positions),
                'total_monitored': self.total_positions_monitored,
                'active_alerts': len(self.active_alerts),
                'total_alerts': self.total_alerts_generated,
                'escalations': self.total_escalations,
                'uptime_hours': self.monitoring_uptime / 3600
            }
            
            logger.info(f"📊 Monitoring Status: {status}")

        except Exception as e:
            logger.error(f"Failed to generate status report: {e}")

    async def _check_portfolio_risk(self):
        """Check portfolio-level risk."""
        try:
            if not self.active_positions:
                return

            total_exposure = sum(
                float(p.entry_price * p.quantity) for p in self.active_positions.values()
            )
            
            total_unrealized = sum(
                float(p.unrealized_pnl) for p in self.active_positions.values()
            )

            if total_exposure > 0:
                portfolio_risk = abs(total_unrealized / total_exposure)
                
                max_risk = self.team_configs[MonitoringTeam.RISK_TEAM]['max_portfolio_risk']
                if portfolio_risk > max_risk:
                    await self._generate_alert(
                        deal_id="PORTFOLIO",
                        team=MonitoringTeam.RISK_TEAM,
                        severity=AlertSeverity.CRITICAL,
                        message=f"Portfolio risk exceeded: {portfolio_risk:.2%} > {max_risk:.2%}"
                    )

        except Exception as e:
            logger.error(f"Portfolio risk check failed: {e}")

    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get comprehensive monitoring summary."""
        return {
            'active_positions_count': len(self.active_positions),
            'position_history_count': len(self.position_history),
            'active_alerts_count': len(self.active_alerts),
            'total_alerts_generated': self.total_alerts_generated,
            'total_escalations': self.total_escalations,
            'monitoring_uptime_hours': self.monitoring_uptime / 3600,
            'is_monitoring': self.is_monitoring,
            'teams_active': len(self.monitoring_tasks)
        }

    async def cleanup(self):
        """Cleanup resources."""
        try:
            await self.stop_monitoring()
            self.active_positions.clear()
            self.active_alerts.clear()
            logger.info("Deal Monitoring Teams cleanup completed")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
