"""
Enhanced Performance Tracker for the AUJ Platform.

This module tracks all trade results with full context, including whether trades
were generated during in-sample (training) or out-of-sample (validation) periods.
This is crucial for the anti-overfitting framework and hierarchy management.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from decimal import Decimal
from enum import Enum
import json
from pathlib import Path
import pandas as pd
from collections import defaultdict, deque
import statistics
import asyncio
import sqlite3

from ..core.exceptions import PerformanceTrackingError, DatabaseError
from ..core.data_contracts import (
    TradeSignal, GradedDeal, AgentDecision, TradeStatus, DealGrade,
    MarketRegime, ConfidenceLevel, AgentRank
)
from ..core.unified_database_manager import get_unified_database, get_unified_database_sync
from ..core.logging_setup import get_logger

logger = get_logger(__name__)


class ValidationPeriodType(str, Enum):
    """Type of validation period for a trade."""
    IN_SAMPLE = "IN_SAMPLE"           # Training period
    OUT_OF_SAMPLE = "OUT_OF_SAMPLE"   # Validation period
    LIVE_TRADING = "LIVE_TRADING"     # Live market trading
    BACKTEST = "BACKTEST"             # Historical backtest


class PerformanceMetricType(str, Enum):
    """Types of performance metrics tracked."""
    WIN_RATE = "WIN_RATE"
    TOTAL_PNL = "TOTAL_PNL"
    SHARPE_RATIO = "SHARPE_RATIO"
    MAX_DRAWDOWN = "MAX_DRAWDOWN"
    PROFIT_FACTOR = "PROFIT_FACTOR"
    AVERAGE_WIN = "AVERAGE_WIN"
    AVERAGE_LOSS = "AVERAGE_LOSS"
    CONSISTENCY_SCORE = "CONSISTENCY_SCORE"


class TradeContextRecord:
    """Comprehensive record of trade context and execution."""

    def __init__(self,
                 trade_id: str,
                 original_signal: TradeSignal,
                 validation_period_type: ValidationPeriodType,
                 generating_agent: str,
                 supporting_agents: List[str],
                 market_regime: Optional[MarketRegime] = None,
                 confidence_level: Optional[ConfidenceLevel] = None):
        # Note: This class now requires config_manager parameter in __init__
        # self.config_manager = config_manager or UnifiedConfigManager()
        """
        Initialize trade context record.

        Args:
            trade_id: Unique trade identifier
            original_signal: Original trade signal
            validation_period_type: Whether this was in-sample, out-of-sample, etc.
            generating_agent: Agent that generated the signal
            supporting_agents: Agents that supported the decision
            market_regime: Market regime when trade was initiated
            confidence_level: Confidence level of the signal
        """
        self.trade_id = trade_id
        self.original_signal = original_signal
        self.validation_period_type = validation_period_type
        self.generating_agent = generating_agent
        self.supporting_agents = supporting_agents
        self.market_regime = market_regime
        self.confidence_level = confidence_level
        self.timestamp = datetime.utcnow()

        # Execution tracking
        self.entry_time: Optional[datetime] = None
        self.exit_time: Optional[datetime] = None
        self.entry_price: Optional[Decimal] = None
        self.exit_price: Optional[Decimal] = None
        self.position_size: Optional[Decimal] = None
        self.actual_stop_loss: Optional[Decimal] = None
        self.actual_take_profit: Optional[Decimal] = None

        # Performance metrics
        self.pnl: Optional[Decimal] = None
        self.pnl_percentage: Optional[float] = None
        self.max_profit: Optional[Decimal] = None
        self.max_drawdown: Optional[Decimal] = None
        self.slippage: Optional[Decimal] = None
        self.commission: Optional[Decimal] = None

        # Quality assessment
        self.execution_quality: Optional[float] = None
        self.grade: Optional[DealGrade] = None
        self.grade_factors: Dict[str, float] = {}

        # Risk metrics
        self.risk_adjusted_return: Optional[float] = None
        self.volatility_during_trade: Optional[float] = None
        self.beta_to_market: Optional[float] = None

        # Additional context
        self.indicators_used = original_signal.indicators_used.copy()
        self.strategy_name = original_signal.strategy
        self.timeframe = original_signal.timeframe
        self.metadata = original_signal.metadata.copy()


class PerformanceTracker:
    """
    Enhanced performance tracker with anti-overfitting focus.

    This tracker maintains detailed records of all trades with their validation context,
    enabling robust performance evaluation that distinguishes between in-sample and
    out-of-sample results.
    """

    def __init__(self,
                 config: Optional[Dict[str, Any]] = None,
                 database = None,
                 walk_forward_validator = None,
                 database_path: Optional[str] = None):
        # Note: This class now requires config_manager parameter in __init__
        # self.config_manager = config_manager or UnifiedConfigManager()
        """
        Initialize the performance tracker.

        Args:
            config: Configuration parameters
            database: Database manager instance
            walk_forward_validator: WalkForward validator for anti-overfitting
            database_path: Path to SQLite database for persistence (fallback)
        """
        self.config = config or {}
        self.config_manager = self.config  # For compatibility with existing code patterns
        self.database = database or get_unified_database_sync()
        self.walk_forward_validator = walk_forward_validator
        # Keep database_path for legacy compatibility but use unified abstraction
        self.database_path = database_path or self.config.get('performance_database_path', "data/performance_tracking.db")

        # In-memory tracking structures
        self.active_trades: Dict[str, TradeContextRecord] = {}
        
        # ✅ FIX #7F: Use deque instead of unbounded dict
        max_completed = self.config.get('max_completed_trades_in_memory', 10000)
        self.completed_trades: deque = deque(maxlen=max_completed)
        self.completed_trades_index: Dict[str, TradeContextRecord] = {}  # For fast lookup
        
        self.agent_performance: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.strategy_performance: Dict[str, Dict[str, Any]] = defaultdict(dict)

        # Performance caches for quick access
        self.performance_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_expiry: Dict[str, datetime] = {}
        self.cache_duration = timedelta(minutes=5)
        self.max_cache_size = self.config.get('max_cache_size', 1000)

        # Validation tracking
        self.validation_periods: Dict[str, Dict[str, datetime]] = {}
        self.current_validation_mode: ValidationPeriodType = ValidationPeriodType.LIVE_TRADING

        # Performance windows for rolling metrics
        self.rolling_window_size = self.config.get('rolling_window_size', 100)
        self.agent_rolling_performance: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.rolling_window_size))
        
        # ✅ FIX #7F: Add periodic cleanup
        self.cleanup_interval_hours = self.config.get('cleanup_interval_hours', 24)
        self.last_cleanup = datetime.utcnow()

        # Initialize database
        self._initialize_database()

        # Note: Historical data loading is deferred to avoid event loop issues
        # Call load_historical_data() manually when needed
        self._historical_data_loaded = False

        logger.info("Enhanced PerformanceTracker initialized with unified database abstraction")

    def _initialize_database(self):
        """Initialize database with required tables using unified database abstraction."""
        try:
            # Use unified database abstraction for schema creation
            with self.database.get_sync_session() as session:
                from sqlalchemy import text

                # Trades table with validation context
                session.execute(text("""
                    CREATE TABLE IF NOT EXISTS trades (
                        trade_id TEXT PRIMARY KEY,
                        signal_id TEXT,
                        validation_period_type TEXT,
                        generating_agent TEXT,
                        supporting_agents TEXT,
                        market_regime TEXT,
                        confidence_level TEXT,
                        symbol TEXT,
                        direction TEXT,
                        strategy_name TEXT,
                        timeframe TEXT,

                        signal_timestamp TIMESTAMP,
                        entry_time TIMESTAMP,
                        exit_time TIMESTAMP,

                        entry_price DECIMAL,
                        exit_price DECIMAL,
                        position_size DECIMAL,
                        stop_loss DECIMAL,
                        take_profit DECIMAL,

                        pnl DECIMAL,
                        pnl_percentage REAL,
                        max_profit DECIMAL,
                        max_drawdown DECIMAL,
                        slippage DECIMAL,
                        commission DECIMAL,

                        execution_quality REAL,
                        grade TEXT,
                        grade_factors TEXT,
                        risk_adjusted_return REAL,
                        volatility_during_trade REAL,
                        beta_to_market REAL,

                        indicators_used TEXT,
                        metadata TEXT,

                        -- Anti-overfitting tracking
                        in_sample_performance REAL,
                        out_of_sample_performance REAL,
                        sample_bias_score REAL,
                        overfitting_risk_score REAL,

                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))

                # Agent performance summary table
                session.execute(text("""
                    CREATE TABLE IF NOT EXISTS agent_performance (
                        agent_name TEXT,
                        period_start TIMESTAMP,
                        period_end TIMESTAMP,
                        validation_type TEXT,

                        total_trades INTEGER,
                        winning_trades INTEGER,
                        losing_trades INTEGER,
                        win_rate REAL,

                        total_pnl DECIMAL,
                        average_pnl DECIMAL,
                        best_trade DECIMAL,
                        worst_trade DECIMAL,

                        sharpe_ratio REAL,
                        max_drawdown DECIMAL,
                        profit_factor REAL,
                        consistency_score REAL,

                        PRIMARY KEY (agent_name, period_start, validation_type)
                    )
                """))

                # Validation periods table
                session.execute(text("""
                    CREATE TABLE IF NOT EXISTS validation_periods (
                        period_id TEXT PRIMARY KEY,
                        period_type TEXT,
                        start_time TIMESTAMP,
                        end_time TIMESTAMP,
                        description TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))

                # Create indexes for performance
                session.execute(text("CREATE INDEX IF NOT EXISTS idx_trades_agent ON trades(generating_agent)"))
                session.execute(text("CREATE INDEX IF NOT EXISTS idx_trades_validation ON trades(validation_period_type)"))
                session.execute(text("CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(signal_timestamp)"))
                session.execute(text("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)"))

                session.commit()

            logger.info("Performance tracking database initialized with unified abstraction")

        except Exception as e:
            logger.error(f"Failed to initialize performance database: {str(e)}")
            raise DatabaseError(f"Database initialization failed: {str(e)}")

    def _initialize_sqlite_database(self):
        """Initialize SQLite database for performance tracking."""
        try:
            # Use unified database manager instead of direct sqlite3.connect
            # Agent performance summary table
            self.database.execute_query_sync("""
                CREATE TABLE IF NOT EXISTS agent_performance (
                    agent_name TEXT,
                    period_start TIMESTAMP,
                    period_end TIMESTAMP,
                    validation_type TEXT,

                    total_trades INTEGER,
                    winning_trades INTEGER,
                    losing_trades INTEGER,
                    win_rate REAL,

                    total_pnl DECIMAL,
                    average_pnl DECIMAL,
                    best_trade DECIMAL,
                    worst_trade DECIMAL,

                    sharpe_ratio REAL,
                    max_drawdown DECIMAL,
                    profit_factor REAL,
                    consistency_score REAL,

                    PRIMARY KEY (agent_name, period_start, validation_type)
                )
            """, use_cache=False)

            # Validation periods table
            self.database.execute_query_sync("""
                CREATE TABLE IF NOT EXISTS validation_periods (
                    period_id TEXT PRIMARY KEY,
                    period_type TEXT,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """, use_cache=False)

            # Create indexes for performance
            self.database.execute_query_sync("CREATE INDEX IF NOT EXISTS idx_trades_agent ON trades(generating_agent)", use_cache=False)
            self.database.execute_query_sync("CREATE INDEX IF NOT EXISTS idx_trades_validation ON trades(validation_period_type)", use_cache=False)
            self.database.execute_query_sync("CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(signal_timestamp)", use_cache=False)
            self.database.execute_query_sync("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)", use_cache=False)

            logger.info("Performance tracking database initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize performance database: {str(e)}")
            raise DatabaseError(f"Database initialization failed: {str(e)}")

    async def initialize(self):
        """
        Initialize the PerformanceTracker with async operations.

        This method handles all async initialization tasks like loading historical data.
        Called by main.py after PerformanceTracker construction.
        """
        try:
            # Load historical data
            if not self._historical_data_loaded:
                await self._load_historical_data()
                self._historical_data_loaded = True

            # Initialize integration with walk-forward validator if available
            if self.walk_forward_validator:
                self.integrate_with_walk_forward_validator(self.walk_forward_validator)

            logger.info("PerformanceTracker async initialization completed successfully")

        except Exception as e:
            logger.error(f"Failed to initialize PerformanceTracker: {str(e)}")
            raise

    async def _load_historical_data(self):
        """Load historical performance data using unified database abstraction."""
        try:
            # Use unified database abstraction to load historical trades
            query = """
                SELECT * FROM trades
                WHERE signal_timestamp >= date('now', '-30 days')
                ORDER BY signal_timestamp DESC
            """

            trades_data = self.database.execute_query_sync(query, use_cache=True, cache_tags=['trades'])

            if trades_data:
                for row_dict in trades_data:
                    trade_record = self._dict_to_trade_record(row_dict)

                    if row_dict['exit_time']:
                        # ✅ FIX #7F: Use deque + index for completed trades
                        self.completed_trades.append(trade_record)
                        self.completed_trades_index[row_dict['trade_id']] = trade_record
                    else:
                        self.active_trades[row_dict['trade_id']] = trade_record

                logger.info(f"Loaded {len(trades_data)} historical trades from unified database")

        except Exception as e:
            logger.error(f"Failed to load historical data: {str(e)}")

    def _dict_to_trade_record(self, row_dict: Dict[str, Any]) -> TradeContextRecord:
        """Convert database dictionary to TradeContextRecord."""
        # Reconstruct TradeSignal
        signal = TradeSignal(
            id=row_dict['signal_id'],
            timestamp=pd.to_datetime(row_dict['signal_timestamp']),
            symbol=row_dict['symbol'],
            direction=row_dict['direction'],
            confidence=0.0,  # Will be updated from other fields
            confidence_level=ConfidenceLevel(row_dict['confidence_level']),
            strategy=row_dict['strategy_name'],
            generating_agent=row_dict['generating_agent'],
            indicators_used=json.loads(row_dict['indicators_used'] or '[]'),
            timeframe=row_dict['timeframe'],
            metadata=json.loads(row_dict['metadata'] or '{}')
        )

        # Create TradeContextRecord
        trade_record = TradeContextRecord(
            trade_id=row_dict['trade_id'],
            original_signal=signal,
            validation_period_type=ValidationPeriodType(row_dict['validation_period_type']),
            generating_agent=row_dict['generating_agent'],
            supporting_agents=json.loads(row_dict['supporting_agents'] or '[]'),
            market_regime=MarketRegime(row_dict['market_regime']) if row_dict['market_regime'] else None,
            confidence_level=ConfidenceLevel(row_dict['confidence_level']) if row_dict['confidence_level'] else None
        )

        # Update execution details
        if row_dict['entry_time']:
            trade_record.entry_time = pd.to_datetime(row_dict['entry_time'])
        if row_dict['exit_time']:
            trade_record.exit_time = pd.to_datetime(row_dict['exit_time'])

        trade_record.entry_price = Decimal(str(row_dict['entry_price'])) if row_dict['entry_price'] else None
        trade_record.exit_price = Decimal(str(row_dict['exit_price'])) if row_dict['exit_price'] else None
        trade_record.position_size = Decimal(str(row_dict['position_size'])) if row_dict['position_size'] else None
        trade_record.pnl = Decimal(str(row_dict['pnl'])) if row_dict['pnl'] else None
        trade_record.pnl_percentage = row_dict['pnl_percentage']
        trade_record.grade = DealGrade(row_dict['grade']) if row_dict['grade'] else None
        trade_record.grade_factors = json.loads(row_dict['grade_factors'] or '{}')

        return trade_record

    def set_validation_period(self,
                            period_type: ValidationPeriodType,
                            start_time: Optional[datetime] = None,
                            description: Optional[str] = None):
        """
        Set the current validation period type.

        Args:
            period_type: Type of validation period
            start_time: Start time of the period
            description: Optional description
        """
        self.current_validation_mode = period_type

        # Store period information
        period_id = f"{period_type.value}_{int(datetime.utcnow().timestamp())}"
        self.validation_periods[period_id] = {
            'type': period_type,
            'start_time': start_time or datetime.utcnow(),
            'description': description
        }

        # Save to database using unified abstraction
        try:
            self.database.execute_query_sync("""
                INSERT INTO validation_periods
                (period_id, period_type, start_time, description)
                VALUES (:period_id, :period_type, :start_time, :description)
            """, {
                'period_id': period_id,
                'period_type': period_type.value,
                'start_time': start_time or datetime.utcnow(),
                'description': description
            }, use_cache=False, cache_tags=['validation_periods'])
        except Exception as e:
            logger.error(f"Failed to save validation period: {str(e)}")

        logger.info(f"Validation period set to: {period_type.value}")

    def record_trade_signal(self,
                          signal: TradeSignal,
                          supporting_agents: Optional[List[str]] = None,
                          market_regime: Optional[MarketRegime] = None) -> str:
        """
        Record a new trade signal with full context.

        Args:
            signal: Trade signal to record
            supporting_agents: Agents that supported this signal
            market_regime: Current market regime

        Returns:
            Trade ID for tracking
        """
        trade_id = f"trade_{signal.id}_{int(datetime.utcnow().timestamp())}"

        # Create comprehensive trade record
        trade_record = TradeContextRecord(
            trade_id=trade_id,
            original_signal=signal,
            validation_period_type=self.current_validation_mode,
            generating_agent=signal.generating_agent,
            supporting_agents=supporting_agents or signal.supporting_agents,
            market_regime=market_regime,
            confidence_level=signal.confidence_level
        )

        # Store in active trades
        self.active_trades[trade_id] = trade_record

        logger.info(f"Recorded trade signal: {trade_id} ({self.current_validation_mode.value})")

        return trade_id

    def record_trade_execution(self,
                             trade_id: str,
                             entry_price: Decimal,
                             position_size: Decimal,
                             entry_time: Optional[datetime] = None,
                             slippage: Optional[Decimal] = None,
                             commission: Optional[Decimal] = None):
        """
        Record trade execution details.

        Args:
            trade_id: Trade identifier
            entry_price: Actual entry price
            position_size: Position size
            entry_time: Execution time
            slippage: Slippage from expected price
            commission: Trading commission
        """
        if trade_id not in self.active_trades:
            raise PerformanceTrackingError(f"Trade {trade_id} not found in active trades")

        trade_record = self.active_trades[trade_id]
        trade_record.entry_time = entry_time or datetime.utcnow()
        trade_record.entry_price = entry_price
        trade_record.position_size = position_size
        trade_record.slippage = slippage
        trade_record.commission = commission

        logger.info(f"Recorded execution for trade {trade_id}: ${entry_price} x {position_size}")

    def record_trade_closure(self,
                           trade_id: str,
                           exit_price: Decimal,
                           exit_time: Optional[datetime] = None,
                           reason: str = "Normal Exit") -> GradedDeal:
        """
        Record trade closure and calculate comprehensive performance metrics.

        Args:
            trade_id: Trade identifier
            exit_price: Exit price
            exit_time: Exit time
            reason: Reason for closure

        Returns:
            GradedDeal with complete performance assessment
        """
        if trade_id not in self.active_trades:
            raise PerformanceTrackingError(f"Trade {trade_id} not found in active trades")

        trade_record = self.active_trades[trade_id]
        trade_record.exit_time = exit_time or datetime.utcnow()
        trade_record.exit_price = exit_price

        # Calculate performance metrics
        self._calculate_trade_performance(trade_record)

        # Grade the trade
        graded_deal = self._grade_trade(trade_record, reason)

        # ✅ FIX #7F: Move to completed trades (deque + index)
        del self.active_trades[trade_id]
        self.completed_trades.append(trade_record)
        self.completed_trades_index[trade_id] = trade_record

        # Update rolling performance metrics
        self._update_rolling_metrics(trade_record)

        # Save to database
        self._save_trade_to_database(trade_record)

        # Invalidate relevant caches
        self._invalidate_performance_cache(trade_record.generating_agent)
        
        # ✅ FIX #7F: Periodic cleanup
        self._maybe_run_cleanup()

        logger.info(f"Trade {trade_id} closed: PnL=${trade_record.pnl}, Grade={graded_deal.grade}")

        return graded_deal

    def _calculate_trade_performance(self, trade_record: TradeContextRecord):
        """Calculate comprehensive performance metrics for a trade."""
        if not all([trade_record.entry_price, trade_record.exit_price, trade_record.position_size]):
            logger.warning(f"Incomplete trade data for {trade_record.trade_id}")
            return

        # Basic PnL calculation
        direction_multiplier = 1 if trade_record.original_signal.direction.value == "BUY" else -1
        price_change = trade_record.exit_price - trade_record.entry_price
        gross_pnl = direction_multiplier * price_change * trade_record.position_size

        # Adjust for commission
        commission = trade_record.commission or Decimal('0')
        trade_record.pnl = gross_pnl - commission

        # Calculate percentage return
        if trade_record.entry_price > 0:
            trade_record.pnl_percentage = float(
                (trade_record.pnl / (trade_record.entry_price * trade_record.position_size)) * 100
            )

        # Calculate trade duration
        if trade_record.entry_time and trade_record.exit_time:
            duration = trade_record.exit_time - trade_record.entry_time
            trade_record.metadata['duration_minutes'] = duration.total_seconds() / 60

        # Execution quality assessment
        expected_price = trade_record.original_signal.entry_price
        if expected_price:
            slippage_percentage = abs(float((trade_record.entry_price - expected_price) / expected_price)) * 100
            trade_record.execution_quality = max(0.0, 1.0 - (slippage_percentage / 5.0))  # 5% slippage = 0 quality

        logger.debug(f"Calculated performance for {trade_record.trade_id}: PnL={trade_record.pnl}")

    def _grade_trade(self, trade_record: TradeContextRecord, reason: str) -> GradedDeal:
        """Grade a completed trade based on multiple factors."""
        # Initialize grading factors
        grade_factors = {}

        # 1. Profitability (40% weight)
        if trade_record.pnl and trade_record.pnl > 0:
            profitability_score = min(float(trade_record.pnl) / 100.0, 1.0)  # $100 = perfect score
        else:
            profitability_score = max(-1.0, float(trade_record.pnl or 0) / 100.0)
        grade_factors['profitability'] = profitability_score * 0.4

        # 2. Execution quality (20% weight)
        execution_score = trade_record.execution_quality or 0.5
        grade_factors['execution'] = execution_score * 0.2

        # 3. Risk management (20% weight)
        risk_score = self._assess_risk_management(trade_record)
        grade_factors['risk_management'] = risk_score * 0.2

        # 4. Timing and duration (10% weight)
        timing_score = self._assess_timing_quality(trade_record)
        grade_factors['timing'] = timing_score * 0.1

        # 5. Market conditions bonus (10% weight)
        market_bonus = self._assess_market_conditions_bonus(trade_record)
        grade_factors['market_conditions'] = market_bonus * 0.1

        # Calculate overall score
        overall_score = sum(grade_factors.values())

        # Convert to letter grade
        if overall_score >= 0.9:
            grade = DealGrade.A_PLUS
        elif overall_score >= 0.8:
            grade = DealGrade.A
        elif overall_score >= 0.7:
            grade = DealGrade.A_MINUS
        elif overall_score >= 0.6:
            grade = DealGrade.B_PLUS
        elif overall_score >= 0.5:
            grade = DealGrade.B
        elif overall_score >= 0.4:
            grade = DealGrade.B_MINUS
        elif overall_score >= 0.3:
            grade = DealGrade.C_PLUS
        elif overall_score >= 0.2:
            grade = DealGrade.C
        elif overall_score >= 0.1:
            grade = DealGrade.C_MINUS
        elif overall_score >= 0.0:
            grade = DealGrade.D
        else:
            grade = DealGrade.F

        # Update trade record
        trade_record.grade = grade
        trade_record.grade_factors = grade_factors

        # Create GradedDeal
        graded_deal = GradedDeal(
            id=trade_record.trade_id,
            original_signal=trade_record.original_signal,
            entry_time=trade_record.entry_time,
            exit_time=trade_record.exit_time,
            entry_price=trade_record.entry_price,
            exit_price=trade_record.exit_price,
            position_size=trade_record.position_size,
            pnl=trade_record.pnl,
            pnl_percentage=trade_record.pnl_percentage,
            status=TradeStatus.CLOSED,
            grade=grade,
            grade_factors=grade_factors,
            execution_quality=trade_record.execution_quality,
            slippage=trade_record.slippage
        )

        return graded_deal

    def _assess_risk_management(self, trade_record: TradeContextRecord) -> float:
        """Assess risk management quality of the trade."""
        score = 0.5  # Base score

        # Check if stop loss was used appropriately
        original_sl = trade_record.original_signal.stop_loss
        if original_sl:
            score += 0.3  # Bonus for having stop loss

        # Check position sizing (assuming reasonable sizing)
        if trade_record.position_size:
            # This would need account balance context for proper assessment
            score += 0.2

        return min(1.0, score)

    def _assess_timing_quality(self, trade_record: TradeContextRecord) -> float:
        """Assess timing quality based on trade duration and market conditions."""
        if not (trade_record.entry_time and trade_record.exit_time):
            return 0.5

        duration_hours = (trade_record.exit_time - trade_record.entry_time).total_seconds() / 3600

        # Optimal duration depends on timeframe
        timeframe = trade_record.original_signal.timeframe
        if timeframe in ['1M', '5M']:
            optimal_range = (0.1, 2)  # 6 minutes to 2 hours
        elif timeframe in ['15M', '30M']:
            optimal_range = (0.5, 8)  # 30 minutes to 8 hours
        elif timeframe in ['1H', '4H']:
            optimal_range = (2, 48)   # 2 hours to 2 days
        else:
            optimal_range = (24, 168) # 1 day to 1 week

        if optimal_range[0] <= duration_hours <= optimal_range[1]:
            return 1.0
        elif duration_hours < optimal_range[0]:
            return max(0.2, duration_hours / optimal_range[0])
        else:
            return max(0.2, optimal_range[1] / duration_hours)

    def _assess_market_conditions_bonus(self, trade_record: TradeContextRecord) -> float:
        """Assess bonus based on challenging market conditions."""
        # This would require market volatility and trend data
        # For now, return neutral score
        return 0.5

    def _update_rolling_metrics(self, trade_record: TradeContextRecord):
        """Update rolling performance metrics for the agent."""
        agent_name = trade_record.generating_agent

        # Add to rolling window
        trade_summary = {
            'timestamp': trade_record.exit_time,
            'pnl': float(trade_record.pnl or 0),
            'pnl_percentage': trade_record.pnl_percentage or 0,
            'validation_type': trade_record.validation_period_type.value,
            'grade': trade_record.grade.value if trade_record.grade else 'F',
            'is_winner': (trade_record.pnl or 0) > 0
        }

        self.agent_rolling_performance[agent_name].append(trade_summary)

    def _save_trade_to_database(self, trade_record: TradeContextRecord):
        """Save completed trade to database."""
        try:
            self.database.execute_query_sync("""
                INSERT OR REPLACE INTO trades (
                    trade_id, signal_id, validation_period_type, generating_agent,
                    supporting_agents, market_regime, confidence_level, symbol,
                    direction, strategy_name, timeframe, signal_timestamp,
                    entry_time, exit_time, entry_price, exit_price, position_size,
                    stop_loss, take_profit, pnl, pnl_percentage, max_profit,
                    max_drawdown, slippage, commission, execution_quality,
                    grade, grade_factors, indicators_used, metadata, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                    trade_record.trade_id,
                    trade_record.original_signal.id,
                    trade_record.validation_period_type.value,
                    trade_record.generating_agent,
                    json.dumps(trade_record.supporting_agents),
                    trade_record.market_regime.value if trade_record.market_regime else None,
                    trade_record.confidence_level.value if trade_record.confidence_level else None,
                    trade_record.original_signal.symbol,
                    trade_record.original_signal.direction.value,
                    trade_record.strategy_name,
                    trade_record.timeframe,
                    trade_record.original_signal.timestamp,
                    trade_record.entry_time,
                    trade_record.exit_time,
                    float(trade_record.entry_price) if trade_record.entry_price else None,
                    float(trade_record.exit_price) if trade_record.exit_price else None,
                    float(trade_record.position_size) if trade_record.position_size else None,
                    float(trade_record.actual_stop_loss) if trade_record.actual_stop_loss else None,
                    float(trade_record.actual_take_profit) if trade_record.actual_take_profit else None,
                    float(trade_record.pnl) if trade_record.pnl else None,
                    trade_record.pnl_percentage,
                    float(trade_record.max_profit) if trade_record.max_profit else None,
                    float(trade_record.max_drawdown) if trade_record.max_drawdown else None,
                    float(trade_record.slippage) if trade_record.slippage else None,
                    float(trade_record.commission) if trade_record.commission else None,
                    trade_record.execution_quality,
                    trade_record.grade.value if trade_record.grade else None,
                    json.dumps(trade_record.grade_factors),
                    json.dumps(trade_record.indicators_used),
                    json.dumps(trade_record.metadata),
                    datetime.utcnow()
                ), use_cache=False)

        except Exception as e:
            logger.error(f"Failed to save trade to database: {str(e)}")

    def get_agent_performance(self,
                            agent_name: str,
                            validation_type: Optional[ValidationPeriodType] = None,
                            days_back: int = 30) -> Dict[str, Any]:
        """
        Get comprehensive performance metrics for an agent.

        Args:
            agent_name: Name of the agent
            validation_type: Filter by validation type (None for all)
            days_back: Number of days to look back

        Returns:
            Comprehensive performance metrics
        """
        # Check cache first
        cache_key = f"{agent_name}_{validation_type}_{days_back}"
        if (cache_key in self.performance_cache and
            cache_key in self.cache_expiry and
            datetime.utcnow() < self.cache_expiry[cache_key]):
            return self.performance_cache[cache_key]

        # Calculate metrics
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)

        # Filter trades - iterate through deque
        relevant_trades = []
        for trade_record in self.completed_trades:
            if (trade_record.generating_agent == agent_name and
                trade_record.exit_time and trade_record.exit_time >= cutoff_date):

                if validation_type is None or trade_record.validation_period_type == validation_type:
                    relevant_trades.append(trade_record)

        if not relevant_trades:
            return self._empty_performance_metrics(agent_name)

        # Calculate comprehensive metrics
        metrics = self._calculate_comprehensive_metrics(relevant_trades, agent_name)

        # Cache results
        self.performance_cache[cache_key] = metrics
        self.cache_expiry[cache_key] = datetime.utcnow() + self.cache_duration

        return metrics

    def _calculate_comprehensive_metrics(self, trades: List[TradeContextRecord], agent_name: str) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics from trade list."""
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t.pnl and t.pnl > 0)
        losing_trades = total_trades - winning_trades

        # Basic metrics
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        total_pnl = sum(t.pnl or Decimal('0') for t in trades)
        average_pnl = total_pnl / total_trades if total_trades > 0 else Decimal('0')

        # Best and worst trades
        pnls = [t.pnl for t in trades if t.pnl is not None]
        best_trade = max(pnls) if pnls else Decimal('0')
        worst_trade = min(pnls) if pnls else Decimal('0')

        # Profit factor
        gross_profit = sum(pnl for pnl in pnls if pnl > 0)
        gross_loss = abs(sum(pnl for pnl in pnls if pnl < 0))
        profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else float('inf')

        # Sharpe ratio
        returns = [t.pnl_percentage for t in trades if t.pnl_percentage is not None]
        sharpe_ratio = None
        if len(returns) >= 10:
            mean_return = statistics.mean(returns)
            if len(returns) > 1:
                std_return = statistics.stdev(returns)
                sharpe_ratio = mean_return / std_return if std_return > 0 else None

        # Maximum drawdown
        cumulative_pnl = Decimal('0')
        peak = Decimal('0')
        max_drawdown = Decimal('0')

        for trade in sorted(trades, key=lambda x: x.exit_time or datetime.min):
            cumulative_pnl += trade.pnl or Decimal('0')
            peak = max(peak, cumulative_pnl)
            drawdown = peak - cumulative_pnl
            max_drawdown = max(max_drawdown, drawdown)

        # Grade distribution
        grade_distribution = {}
        for trade in trades:
            grade = trade.grade.value if trade.grade else 'F'
            grade_distribution[grade] = grade_distribution.get(grade, 0) + 1

        # Validation type breakdown
        validation_breakdown = {}
        for trade in trades:
            val_type = trade.validation_period_type.value
            if val_type not in validation_breakdown:
                validation_breakdown[val_type] = {'count': 0, 'pnl': Decimal('0'), 'win_rate': 0.0}

            validation_breakdown[val_type]['count'] += 1
            validation_breakdown[val_type]['pnl'] += trade.pnl or Decimal('0')

        # Calculate win rates for each validation type
        for val_type, data in validation_breakdown.items():
            val_trades = [t for t in trades if t.validation_period_type.value == val_type]
            val_wins = sum(1 for t in val_trades if t.pnl and t.pnl > 0)
            data['win_rate'] = val_wins / len(val_trades) if val_trades else 0.0

        return {
            'agent_name': agent_name,
            'period_days': (max(t.exit_time for t in trades) - min(t.exit_time for t in trades)).days + 1,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': float(total_pnl),
            'average_pnl': float(average_pnl),
            'best_trade': float(best_trade),
            'worst_trade': float(worst_trade),
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': float(max_drawdown),
            'grade_distribution': grade_distribution,
            'validation_breakdown': {
                k: {
                    'count': v['count'],
                    'pnl': float(v['pnl']),
                    'win_rate': v['win_rate']
                }
                for k, v in validation_breakdown.items()
            },
            'last_updated': datetime.utcnow().isoformat()
        }

    def _empty_performance_metrics(self, agent_name: str) -> Dict[str, Any]:
        """Return empty performance metrics structure."""
        return {
            'agent_name': agent_name,
            'period_days': 0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'total_pnl': 0.0,
            'average_pnl': 0.0,
            'best_trade': 0.0,
            'worst_trade': 0.0,
            'profit_factor': 0.0,
            'sharpe_ratio': None,
            'max_drawdown': 0.0,
            'grade_distribution': {},
            'validation_breakdown': {},
            'last_updated': datetime.utcnow().isoformat()
        }

    def _invalidate_performance_cache(self, agent_name: str):
        """Invalidate cached performance metrics for an agent."""
        keys_to_remove = [key for key in self.performance_cache.keys() if key.startswith(agent_name)]
        for key in keys_to_remove:
            if key in self.performance_cache:
                del self.performance_cache[key]
            if key in self.cache_expiry:
                del self.cache_expiry[key]

    def get_out_of_sample_performance_only(self, agent_name: str, days_back: int = 30) -> Dict[str, Any]:
        """
        Get performance metrics for out-of-sample trades only.

        This is crucial for anti-overfitting evaluation.
        """
        return self.get_agent_performance(
            agent_name=agent_name,
            validation_type=ValidationPeriodType.OUT_OF_SAMPLE,
            days_back=days_back
        )

    def get_platform_summary(self) -> Dict[str, Any]:
        """Get overall platform performance summary."""
        # Get all completed trades from last 30 days
        cutoff_date = datetime.utcnow() - timedelta(days=30)
        recent_trades = [
            t for t in self.completed_trades.values()
            if t.exit_time and t.exit_time >= cutoff_date
        ]

        if not recent_trades:
            return {
                'total_trades': 0,
                'active_trades': len(self.active_trades),
                'overall_pnl': 0.0,
                'win_rate': 0.0,
                'agent_count': 0,
                'validation_breakdown': {}
            }

        # Calculate overall metrics
        total_pnl = sum(t.pnl or Decimal('0') for t in recent_trades)
        winning_trades = sum(1 for t in recent_trades if t.pnl and t.pnl > 0)
        win_rate = winning_trades / len(recent_trades)

        # Agent performance summary
        agent_stats = {}
        for trade in recent_trades:
            agent = trade.generating_agent
            if agent not in agent_stats:
                agent_stats[agent] = {'trades': 0, 'pnl': Decimal('0'), 'wins': 0}

            agent_stats[agent]['trades'] += 1
            agent_stats[agent]['pnl'] += trade.pnl or Decimal('0')
            if trade.pnl and trade.pnl > 0:
                agent_stats[agent]['wins'] += 1

        # Validation type breakdown
        validation_stats = {}
        for trade in recent_trades:
            val_type = trade.validation_period_type.value
            if val_type not in validation_stats:
                validation_stats[val_type] = {'trades': 0, 'pnl': Decimal('0'), 'wins': 0}

            validation_stats[val_type]['trades'] += 1
            validation_stats[val_type]['pnl'] += trade.pnl or Decimal('0')
            if trade.pnl and trade.pnl > 0:
                validation_stats[val_type]['wins'] += 1

        return {
            'total_trades': len(recent_trades),
            'active_trades': len(self.active_trades),
            'overall_pnl': float(total_pnl),
            'win_rate': win_rate,
            'agent_count': len(agent_stats),
            'agent_performance': {
                agent: {
                    'trades': stats['trades'],
                    'pnl': float(stats['pnl']),
                    'win_rate': stats['wins'] / stats['trades']
                }
                for agent, stats in agent_stats.items()
            },
            'validation_breakdown': {
                val_type: {
                    'trades': stats['trades'],
                    'pnl': float(stats['pnl']),
                    'win_rate': stats['wins'] / stats['trades']
                }
                for val_type, stats in validation_stats.items()
            },
            'current_validation_mode': self.current_validation_mode.value,
            'last_updated': datetime.utcnow().isoformat()
        }

    def export_performance_data(self,
                              agent_name: Optional[str] = None,
                              start_date: Optional[datetime] = None,
                              end_date: Optional[datetime] = None,
                              format: str = 'csv') -> str:
        """
        Export performance data for analysis.

        Args:
            agent_name: Filter by agent name
            start_date: Start date filter
            end_date: End date filter
            format: Export format ('csv', 'json')

        Returns:
            Path to exported file
        """
        # This would implement data export functionality
        # For now, return placeholder
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        filename = f"performance_export_{timestamp}.{format}"

        logger.info(f"Performance data export requested: {filename}")
        return filename

    async def cleanup_old_data(self, days_to_keep: int = 365):
        """Clean up old performance data to manage database size."""
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)

        try:
            # Using unified database manager instead of direct sqlite3
            # Remove old trades
            result = self.database.execute_query_sync(
                "DELETE FROM trades WHERE signal_timestamp < ?",
                (cutoff_date,),
                use_cache=False
            )
            deleted_trades = result.rowcount if hasattr(result, 'rowcount') else 0

            # Remove old performance summaries
            result = self.database.execute_query_sync(
                "DELETE FROM agent_performance WHERE period_end < ?",
                (cutoff_date,),
                use_cache=False
            )
            deleted_summaries = result.rowcount if hasattr(result, 'rowcount') else 0

            logger.info(f"Cleaned up {deleted_trades} old trades and {deleted_summaries} old summaries")

        except Exception as e:
            logger.error(f"Failed to cleanup old data: {str(e)}")
    def integrate_with_walk_forward_validator(self, validator):
        """
        Integrate PerformanceTracker with WalkForwardValidator.

        This creates a tight coupling between performance tracking and validation,
        ensuring that all trades are properly classified and evaluated.

        Args:
            validator: WalkForwardValidator instance
        """
        self.walk_forward_validator = validator
        logger.info("PerformanceTracker integrated with WalkForwardValidator")

    def determine_validation_context(self,
                                   signal_timestamp: datetime,
                                   strategy_name: str) -> ValidationPeriodType:
        """
        Determine if a signal falls within a training or validation period.

        This method works with the WalkForwardValidator to classify each trade
        signal based on the current validation windows.

        Args:
            signal_timestamp: When the signal was generated
            strategy_name: Name of the strategy

        Returns:
            ValidationPeriodType indicating the context
        """
        if not hasattr(self, 'walk_forward_validator'):
            # If no validator is integrated, use current mode
            return self.current_validation_mode

        try:
            # Check if we're in a live trading environment
            if self.current_validation_mode == ValidationPeriodType.LIVE_TRADING:
                return ValidationPeriodType.LIVE_TRADING

            # Check validation periods stored in database
            cutoff_date = signal_timestamp - timedelta(days=180)  # Look back 6 months max

            # Using unified database manager instead of direct sqlite3
            cursor = self.database.execute_query_sync("""
                SELECT period_type, start_time, end_time FROM validation_periods
                WHERE start_time <= ? AND (end_time IS NULL OR end_time >= ?)
                ORDER BY start_time DESC LIMIT 1
            """, (signal_timestamp, signal_timestamp), use_cache=False)

            result = cursor.fetchone()

            if result:
                period_type = result[0]
                return ValidationPeriodType(period_type)

            # Default classification logic based on recent history
            recent_trades = self._get_recent_trades_for_strategy(strategy_name, days=90)

            if len(recent_trades) < 30:
                # If we have few trades, consider it training data
                return ValidationPeriodType.IN_SAMPLE

            # Use timestamp-based logic for classification
            # This assumes a rolling validation approach
            total_period_days = 90
            training_days = 60

            period_start = signal_timestamp - timedelta(days=total_period_days)
            training_cutoff = period_start + timedelta(days=training_days)

            if signal_timestamp <= training_cutoff:
                return ValidationPeriodType.IN_SAMPLE
            else:
                return ValidationPeriodType.OUT_OF_SAMPLE

        except Exception as e:
            logger.warning(f"Error determining validation context: {str(e)}")
            return self.current_validation_mode

    def record_trade_signal_with_auto_validation(self,
                                                signal: TradeSignal,
                                                supporting_agents: Optional[List[str]] = None,
                                                market_regime: Optional[MarketRegime] = None) -> str:
        """
        Record a trade signal with automatic validation context determination.

        This is the enhanced version that automatically determines whether the
        signal falls in training or validation period.

        Args:
            signal: Trade signal to record
            supporting_agents: Agents that supported this signal
            market_regime: Current market regime

        Returns:
            Trade ID for tracking
        """
        # Automatically determine validation context
        validation_context = self.determine_validation_context(
            signal.timestamp, signal.strategy
        )

        # Temporarily set the validation mode
        original_mode = self.current_validation_mode
        self.current_validation_mode = validation_context

        try:
            # Record the trade with the determined context
            trade_id = self.record_trade_signal(
                signal=signal,
                supporting_agents=supporting_agents,
                market_regime=market_regime
            )

            logger.info(f"Auto-classified trade {trade_id} as {validation_context.value}")
            return trade_id

        finally:
            # Restore original mode
            self.current_validation_mode = original_mode

    def _get_recent_trades_for_strategy(self, strategy_name: str, days: int = 30) -> List[TradeContextRecord]:
        """Get recent trades for a specific strategy."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        return [
            trade for trade in self.completed_trades.values()
            if (trade.strategy_name == strategy_name and
                trade.exit_time and trade.exit_time >= cutoff_date)
        ]

    def get_validation_performance_comparison(self,
                                            agent_name: str,
                                            days_back: int = 90) -> Dict[str, Any]:
        """
        Compare in-sample vs out-of-sample performance for overfitting detection.

        This is a key method for the anti-overfitting framework.

        Args:
            agent_name: Name of the agent
            days_back: Number of days to analyze

        Returns:
            Comparison metrics highlighting potential overfitting
        """
        # Get performance for both validation types
        in_sample_perf = self.get_agent_performance(
            agent_name=agent_name,
            validation_type=ValidationPeriodType.IN_SAMPLE,
            days_back=days_back
        )

        out_of_sample_perf = self.get_agent_performance(
            agent_name=agent_name,
            validation_type=ValidationPeriodType.OUT_OF_SAMPLE,
            days_back=days_back
        )

        # Calculate overfitting indicators
        overfitting_signals = {}

        # Win rate degradation
        if (in_sample_perf['win_rate'] > 0 and out_of_sample_perf['win_rate'] >= 0):
            win_rate_degradation = (in_sample_perf['win_rate'] - out_of_sample_perf['win_rate']) / in_sample_perf['win_rate']
            overfitting_signals['win_rate_degradation'] = max(0.0, win_rate_degradation)

        # Profit factor degradation
        if (in_sample_perf['profit_factor'] > 0 and out_of_sample_perf['profit_factor'] > 0):
            pf_degradation = (in_sample_perf['profit_factor'] - out_of_sample_perf['profit_factor']) / in_sample_perf['profit_factor']
            overfitting_signals['profit_factor_degradation'] = max(0.0, pf_degradation)

        # Sharpe ratio degradation
        if (in_sample_perf['sharpe_ratio'] and out_of_sample_perf['sharpe_ratio'] and
            in_sample_perf['sharpe_ratio'] > 0):
            sharpe_degradation = (in_sample_perf['sharpe_ratio'] - out_of_sample_perf['sharpe_ratio']) / in_sample_perf['sharpe_ratio']
            overfitting_signals['sharpe_degradation'] = max(0.0, sharpe_degradation)

        # Overall overfitting score (0 = no overfitting, 1 = severe overfitting)
        overfitting_scores = [score for score in overfitting_signals.values() if score is not None]
        overall_overfitting_score = statistics.mean(overfitting_scores) if overfitting_scores else 0.0

        # Performance consistency score
        consistency_factors = []

        # Trade count consistency
        if in_sample_perf['total_trades'] > 0 and out_of_sample_perf['total_trades'] > 0:
            trade_count_ratio = min(in_sample_perf['total_trades'], out_of_sample_perf['total_trades']) / max(in_sample_perf['total_trades'], out_of_sample_perf['total_trades'])
            consistency_factors.append(trade_count_ratio)

        # Performance stability
        if out_of_sample_perf['win_rate'] >= 0.4:  # Minimum acceptable performance
            consistency_factors.append(0.8)
        elif out_of_sample_perf['win_rate'] >= 0.3:
            consistency_factors.append(0.5)
        else:
            consistency_factors.append(0.2)

        consistency_score = statistics.mean(consistency_factors) if consistency_factors else 0.0

        # Recommendations
        recommendations = []

        if overall_overfitting_score > 0.3:
            recommendations.append("HIGH_OVERFITTING_RISK")
        elif overall_overfitting_score > 0.15:
            recommendations.append("MODERATE_OVERFITTING_RISK")

        if out_of_sample_perf['total_trades'] < 10:
            recommendations.append("INSUFFICIENT_OUT_OF_SAMPLE_DATA")

        if out_of_sample_perf['win_rate'] < 0.4:
            recommendations.append("POOR_OUT_OF_SAMPLE_PERFORMANCE")

        if consistency_score < 0.5:
            recommendations.append("INCONSISTENT_PERFORMANCE")

        if not recommendations and overall_overfitting_score < 0.1 and out_of_sample_perf['win_rate'] > 0.5:
            recommendations.append("APPROVED_FOR_LIVE_TRADING")

        return {
            'agent_name': agent_name,
            'analysis_period_days': days_back,
            'in_sample_performance': in_sample_perf,
            'out_of_sample_performance': out_of_sample_perf,
            'overfitting_signals': overfitting_signals,
            'overall_overfitting_score': overall_overfitting_score,
            'consistency_score': consistency_score,
            'recommendations': recommendations,
            'analysis_timestamp': datetime.utcnow().isoformat(),
            'suitable_for_live_trading': 'APPROVED_FOR_LIVE_TRADING' in recommendations
        }

    def get_anti_overfitting_metrics(self, days_back: int = 90) -> Dict[str, Any]:
        """
        Get platform-wide anti-overfitting metrics.

        This provides a system-wide view of overfitting risks across all agents.

        Args:
            days_back: Number of days to analyze

        Returns:
            System-wide overfitting analysis
        """
        # Get all active agents
        active_agents = set()
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)

        for trade in self.completed_trades.values():
            if trade.exit_time and trade.exit_time >= cutoff_date:
                active_agents.add(trade.generating_agent)

        if not active_agents:
            return {
                'total_agents_analyzed': 0,
                'agents_with_overfitting_risk': 0,
                'agents_approved_for_live': 0,
                'system_overfitting_score': 0.0,
                'recommendations': ['NO_TRADING_DATA_AVAILABLE']
            }

        # Analyze each agent
        agent_analyses = {}
        overfitting_scores = []
        approved_agents = 0
        risky_agents = 0

        for agent in active_agents:
            analysis = self.get_validation_performance_comparison(agent, days_back)
            agent_analyses[agent] = analysis

            overfitting_scores.append(analysis['overall_overfitting_score'])

            if analysis['suitable_for_live_trading']:
                approved_agents += 1
            elif analysis['overall_overfitting_score'] > 0.3:
                risky_agents += 1

        # System-wide metrics
        system_overfitting_score = statistics.mean(overfitting_scores) if overfitting_scores else 0.0

        # System recommendations
        system_recommendations = []

        if risky_agents > len(active_agents) * 0.5:
            system_recommendations.append("SYSTEM_WIDE_OVERFITTING_DETECTED")

        if approved_agents < len(active_agents) * 0.3:
            system_recommendations.append("INSUFFICIENT_VALIDATED_AGENTS")

        if approved_agents > 0:
            system_recommendations.append("VALIDATED_AGENTS_AVAILABLE")

        # Get validation data sufficiency
        total_out_of_sample_trades = sum(
            analysis['out_of_sample_performance']['total_trades']
            for analysis in agent_analyses.values()
        )

        if total_out_of_sample_trades < 100:
            system_recommendations.append("INSUFFICIENT_VALIDATION_DATA")

        return {
            'total_agents_analyzed': len(active_agents),
            'agents_with_overfitting_risk': risky_agents,
            'agents_approved_for_live': approved_agents,
            'system_overfitting_score': system_overfitting_score,
            'agent_analyses': agent_analyses,
            'total_out_of_sample_trades': total_out_of_sample_trades,
            'validation_data_quality': 'SUFFICIENT' if total_out_of_sample_trades >= 100 else 'INSUFFICIENT',
            'recommendations': system_recommendations,
            'analysis_timestamp': datetime.utcnow().isoformat()
        }

    def reset_validation_period(self, new_period_type: ValidationPeriodType, description: str = ""):
        """
        Reset to a new validation period and close the current one.

        This is used when transitioning between training and validation phases.

        Args:
            new_period_type: New validation period type
            description: Description of the transition
        """
        # Close current period if it exists
        current_time = datetime.utcnow()

        try:
            # Using unified database manager instead of direct sqlite3
            # Update the end time of the current period
            self.database.execute_query_sync("""
                UPDATE validation_periods
                SET end_time = ?
                WHERE end_time IS NULL
            """, (current_time,), use_cache=False)
        except Exception as e:
            logger.error(f"Failed to close validation period: {str(e)}")

        # Set new validation period
        self.set_validation_period(new_period_type, current_time, description)

        logger.info(f"Validation period reset to {new_period_type.value}: {description}")

    def get_trade_quality_by_validation_type(self, days_back: int = 30) -> Dict[str, Any]:
        """
        Analyze trade quality distribution by validation type.

        This helps identify if the system performs differently in training vs validation.

        Args:
            days_back: Number of days to analyze

        Returns:
            Quality analysis by validation type
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)

        # Group trades by validation type
        validation_trades = {
            ValidationPeriodType.IN_SAMPLE: [],
            ValidationPeriodType.OUT_OF_SAMPLE: [],
            ValidationPeriodType.LIVE_TRADING: []
        }

        for trade in self.completed_trades.values():
            if trade.exit_time and trade.exit_time >= cutoff_date:
                val_type = trade.validation_period_type
                if val_type in validation_trades:
                    validation_trades[val_type].append(trade)

        # Analyze quality for each type
        quality_analysis = {}

        for val_type, trades in validation_trades.items():
            if not trades:
                quality_analysis[val_type.value] = {
                    'trade_count': 0,
                    'grade_distribution': {},
                    'average_grade_score': 0.0,
                    'quality_metrics': {}
                }
                continue

            # Grade distribution
            grade_dist = {}
            grade_scores = []

            grade_score_map = {
                DealGrade.A_PLUS: 4.3, DealGrade.A: 4.0, DealGrade.A_MINUS: 3.7,
                DealGrade.B_PLUS: 3.3, DealGrade.B: 3.0, DealGrade.B_MINUS: 2.7,
                DealGrade.C_PLUS: 2.3, DealGrade.C: 2.0, DealGrade.C_MINUS: 1.7,
                DealGrade.D: 1.0, DealGrade.F: 0.0
            }

            for trade in trades:
                grade = trade.grade or DealGrade.F
                grade_dist[grade.value] = grade_dist.get(grade.value, 0) + 1
                grade_scores.append(grade_score_map.get(grade, 0.0))

            # Quality metrics
            avg_grade_score = statistics.mean(grade_scores) if grade_scores else 0.0
            high_quality_trades = sum(1 for score in grade_scores if score >= 3.0)
            high_quality_percentage = (high_quality_trades / len(trades)) * 100

            quality_analysis[val_type.value] = {
                'trade_count': len(trades),
                'grade_distribution': grade_dist,
                'average_grade_score': avg_grade_score,
                'high_quality_percentage': high_quality_percentage,
                'quality_metrics': {
                    'excellent_trades': sum(1 for score in grade_scores if score >= 4.0),
                    'good_trades': sum(1 for score in grade_scores if 3.0 <= score < 4.0),
                    'fair_trades': sum(1 for score in grade_scores if 2.0 <= score < 3.0),
                    'poor_trades': sum(1 for score in grade_scores if score < 2.0)
                }
            }

        # Cross-validation quality comparison
        in_sample_avg = quality_analysis.get(ValidationPeriodType.IN_SAMPLE.value, {}).get('average_grade_score', 0)
        out_of_sample_avg = quality_analysis.get(ValidationPeriodType.OUT_OF_SAMPLE.value, {}).get('average_grade_score', 0)

        quality_degradation = 0.0
        if in_sample_avg > 0:
            quality_degradation = max(0.0, (in_sample_avg - out_of_sample_avg) / in_sample_avg)

        return {
            'analysis_period_days': days_back,
            'validation_type_analysis': quality_analysis,
            'quality_degradation': quality_degradation,
            'quality_consistency': 1.0 - quality_degradation,
            'analysis_timestamp': datetime.utcnow().isoformat()
        }

    async def save_critical_data(self):
        """
        Save critical performance data to ensure no data loss during shutdown.

        This method is called by main.py during platform shutdown to persist
        all active trades and performance metrics.
        """
        try:
            logger.info("Saving critical performance data...")

            # Save all active trades to database
            active_trade_count = 0
            for trade_id, trade_record in self.active_trades.items():
                try:
                    # Save active trade state
                    # Using unified database manager instead of direct sqlite3
                    self.database.execute_query_sync("""
                        INSERT OR REPLACE INTO trades (
                            trade_id, signal_id, validation_period_type, generating_agent,
                            supporting_agents, market_regime, confidence_level, symbol,
                            direction, strategy_name, timeframe, signal_timestamp,
                            entry_time, entry_price, position_size, stop_loss, take_profit,
                            indicators_used, metadata, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        trade_record.trade_id,
                        trade_record.original_signal.id,
                        trade_record.validation_period_type.value,
                        trade_record.generating_agent,
                        json.dumps(trade_record.supporting_agents),
                        trade_record.market_regime.value if trade_record.market_regime else None,
                        trade_record.confidence_level.value if trade_record.confidence_level else None,
                        trade_record.original_signal.symbol,
                        trade_record.original_signal.direction.value,
                        trade_record.strategy_name,
                        trade_record.timeframe,
                        trade_record.original_signal.timestamp,
                        trade_record.entry_time,
                        float(trade_record.entry_price) if trade_record.entry_price else None,
                        float(trade_record.position_size) if trade_record.position_size else None,
                        float(trade_record.actual_stop_loss) if trade_record.actual_stop_loss else None,
                        float(trade_record.actual_take_profit) if trade_record.actual_take_profit else None,
                        json.dumps(trade_record.indicators_used),
                        json.dumps(trade_record.metadata),
                        datetime.utcnow()
                    ), use_cache=False)
                    active_trade_count += 1
                except Exception as e:
                    logger.error(f"Failed to save active trade {trade_id}: {str(e)}")

            # Save current validation period state
            try:
                # Using unified database manager instead of direct sqlite3
                self.database.execute_query_sync("""
                    INSERT OR REPLACE INTO validation_periods
                    (period_id, period_type, start_time, description)
                    VALUES (?, ?, ?, ?)
                """, (
                    f"shutdown_{int(datetime.utcnow().timestamp())}",
                    self.current_validation_mode.value,
                    datetime.utcnow(),
                    "Saved during platform shutdown"
                ), use_cache=False)
            except Exception as e:
                logger.error(f"Failed to save validation period state: {str(e)}")

            # Clear performance caches to free memory
            self.performance_cache.clear()
            self.cache_expiry.clear()

            logger.info(f"Critical performance data saved successfully:")
            logger.info(f"  - Active trades saved: {active_trade_count}")
            logger.info(f"  - Completed trades in memory: {len(self.completed_trades)}")
            logger.info(f"  - Current validation mode: {self.current_validation_mode.value}")

        except Exception as e:
            logger.error(f"Failed to save critical performance data: {str(e)}")
            # Don't raise exception during shutdown - log and continue

    def _invalidate_performance_cache(self, agent_name: str):
        '''? FIX #7G: Invalidate performance cache for an agent.'''
        keys_to_remove = [k for k in self.performance_cache.keys() if k.startswith(agent_name)]
        for key in keys_to_remove:
            del self.performance_cache[key]
            if key in self.cache_expiry:
                del self.cache_expiry[key]
    
    def _maybe_run_cleanup(self):
        '''? FIX #7F: Run cleanup if interval has passed.'''
        now = datetime.utcnow()
        hours_since_cleanup = (now - self.last_cleanup).total_seconds() / 3600
        
        if hours_since_cleanup >= self.cleanup_interval_hours:
            self._cleanup_old_data()
            self.last_cleanup = now
    
    def _cleanup_old_data(self):
        '''? FIX #7F & #7H: Clean up old in-memory data.'''
        try:
            cutoff = datetime.utcnow() - timedelta(days=30)
            
            for agent_name in list(self.agent_rolling_performance.keys()):
                recent_trades = [
                    t for t in self.agent_rolling_performance[agent_name]
                    if t.get('timestamp') and t['timestamp'] > cutoff
                ]
                
                if recent_trades:
                    self.agent_rolling_performance[agent_name] = deque(
                        recent_trades,
                        maxlen=self.rolling_window_size
                    )
                else:
                    del self.agent_rolling_performance[agent_name]
            
            if len(self.performance_cache) > self.max_cache_size:
                oldest_key = min(self.cache_expiry, key=self.cache_expiry.get)
                del self.performance_cache[oldest_key]
                del self.cache_expiry[oldest_key]
            
            valid_ids = {tr.trade_id for tr in self.completed_trades}
            invalid_ids = set(self.completed_trades_index.keys()) - valid_ids
            for trade_id in invalid_ids:
                del self.completed_trades_index[trade_id]
            
            logger.info(f'Cleaned up old data. Agents: {len(self.agent_rolling_performance)}')
            
        except Exception as e:
            logger.error(f'Cleanup failed: {e}')
    
    async def cleanup(self):
        '''? FIX #7F: Cleanup PerformanceTracker resources.'''
        try:
            self._cleanup_old_data()
            self.performance_cache.clear()
            self.cache_expiry.clear()
            self.completed_trades_index.clear()
            logger.info('PerformanceTracker cleanup completed')
        except Exception as e:
            logger.error(f'Error during cleanup: {e}')
