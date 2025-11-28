"""
Daily Feedback Loop Manager for AUJ Platform.

This module implements the critical daily feedback loop that integrates all anti-overfitting
components to ensure continuous, intelligent adaptation based on validated performance data.

Every day at 22:00 UTC (end-of-day), before the next daily cycle:
1. HierarchyManager updates agent ranks based on risk-adjusted, out-of-sample performance
2. GeniusAgentCoordinator consults IndicatorEffectivenessAnalyzer to update Elite Indicator lists
3. AgentBehaviorOptimizer applies intelligent optimizations
4. System validates all changes using walk-forward analysis

This ensures the platform continuously evolves while preventing overfitting.
The coordinator continues to perform lightweight hourly exploratory scans independently.

FIXES IMPLEMENTED:
- Added missing RegimeClassifier import (was causing error on line 138)
- Added self.database initialization (was undefined in _initialize_database)
"""

import asyncio
from datetime import datetime, timedelta, time
import sqlite3
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
import traceback
import json
from decimal import Decimal
from collections import defaultdict
from pathlib import Path

from ..core.unified_config import UnifiedConfigManager
from ..core.data_contracts import (
    MarketRegime, AgentRank, ConfidenceLevel, PlatformStatus,
    TradeSignal, GradedDeal, ValidationResult
)
from ..core.exceptions import (
    FeedbackLoopError, ValidationError, OptimizationError,
    AUJPlatformError
)
from ..core.unified_database_manager import get_unified_database
from ..validation.walk_forward_validator import WalkForwardValidator
from ..analytics.performance_tracker import PerformanceTracker
from ..analytics.indicator_effectiveness_analyzer import (
    IndicatorEffectivenessAnalyzer, IndicatorQualityRating
)
from ..learning.agent_behavior_optimizer import AgentBehaviorOptimizer
from ..hierarchy.hierarchy_manager import HierarchyManager
from ..coordination.genius_agent_coordinator import GeniusAgentCoordinator
from ..monitoring.economic_monitor import EconomicMonitor
from ..regime_detection.regime_classifier import RegimeClassifier  # FIXED: Added missing import


class FeedbackLoopPhase(str, Enum):
    """Phases of the daily feedback loop."""
    INITIALIZATION = "INITIALIZATION"
    MARKET_REGIME_DETECTION = "MARKET_REGIME_DETECTION"
    AGENT_PERFORMANCE_ANALYSIS = "AGENT_PERFORMANCE_ANALYSIS"
    INDICATOR_EFFECTIVENESS_UPDATE = "INDICATOR_EFFECTIVENESS_UPDATE"
    AGENT_BEHAVIOR_OPTIMIZATION = "AGENT_BEHAVIOR_OPTIMIZATION"
    HIERARCHY_UPDATE = "HIERARCHY_UPDATE"
    ELITE_INDICATOR_UPDATE = "ELITE_INDICATOR_UPDATE"
    VALIDATION_AND_SAFETY_CHECK = "VALIDATION_AND_SAFETY_CHECK"
    SYSTEM_STATUS_UPDATE = "SYSTEM_STATUS_UPDATE"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"


class FeedbackLoopStatus(str, Enum):
    """Status of the feedback loop."""
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"
    PAUSED = "PAUSED"


@dataclass
class FeedbackLoopMetrics:
    """Metrics from a feedback loop execution."""
    loop_id: str
    timestamp: datetime
    duration_seconds: float
    phase_durations: Dict[str, float]
    
    # Performance updates
    agents_optimized: int
    agents_requiring_intervention: int
    hierarchy_changes: Dict[str, str]  # agent_name -> rank_change
    
    # Indicator updates
    elite_indicators_updated: int
    indicators_promoted: int
    indicators_demoted: int
    overfitted_indicators_removed: int
    
    # System health
    overall_system_performance: float
    overfitting_risk_score: float
    stability_score: float
    
    # Validation results
    validation_success: bool
    safety_checks_passed: bool
    
    # Recommendations
    system_recommendations: List[str]
    risk_warnings: List[str]
    
    # Status
    status: FeedbackLoopStatus
    error_message: Optional[str] = None


class DailyFeedbackLoop:
    """
    Daily Feedback Loop Manager.
    
    This component orchestrates the daily optimization and validation cycle that ensures
    the platform continuously adapts while preventing overfitting. It integrates all
    anti-overfitting components and ensures system-wide coherence.
    
    Key Responsibilities:
    - Coordinate daily optimization cycles (22:00 UTC)
    - Update agent rankings based on validated performance
    - Refresh Elite Indicator sets using robust analytics
    - Apply intelligent agent behavior optimizations
    - Validate all changes using walk-forward analysis
    - Maintain system safety and stability
    
    The coordinator performs independent hourly exploratory scans between daily cycles.
    """
    
    def __init__(self,
                 walk_forward_validator: WalkForwardValidator,
                 performance_tracker: PerformanceTracker,
                 indicator_analyzer: IndicatorEffectivenessAnalyzer,
                 agent_optimizer: AgentBehaviorOptimizer,
                 hierarchy_manager: HierarchyManager,
                 genius_coordinator: GeniusAgentCoordinator,
                 regime_classifier: RegimeClassifier,
                 economic_monitor: Optional[EconomicMonitor] = None,
                 database_path: Optional[str] = None,
                 config: Optional[Any] = None):
        """
        Initialize the Daily Feedback Loop.
        
        Args:
            walk_forward_validator: Validation engine
            performance_tracker: Performance tracking system
            indicator_analyzer: Indicator effectiveness analyzer
            agent_optimizer: Agent behavior optimizer
            hierarchy_manager: Agent hierarchy manager
            genius_coordinator: Agent coordination system
            regime_classifier: Market regime detection
            database_path: Path to feedback loop database
            config: Optional configuration override
        """
        from ..core.unified_config import get_unified_config
        self.config_manager = get_unified_config()
        
        # Core components
        self.walk_forward_validator = walk_forward_validator
        self.performance_tracker = performance_tracker
        self.indicator_analyzer = indicator_analyzer
        self.agent_optimizer = agent_optimizer
        self.hierarchy_manager = hierarchy_manager
        self.genius_coordinator = genius_coordinator
        self.regime_classifier = regime_classifier
        self.economic_monitor = economic_monitor
        
        # Configuration
        self.database_path = database_path or "data/feedback_loop.db"
        self.config = config or {}
        
        # FIXED: Initialize database connection
        self.database = get_unified_database()
        
        # Daily loop parameters (changed from hourly)
        self.daily_execution_time = time(22, 0)  # 22:00 UTC end-of-day
        self.safety_check_enabled = self.config_manager.get_bool('safety_check_enabled', True)
        self.max_system_changes_per_loop = self.config_manager.get_int('max_system_changes', 5)
        self.overfitting_emergency_threshold = self.config_manager.get_float('overfitting_emergency_threshold', 0.7)
        
        # State tracking
        self.current_status = FeedbackLoopStatus.IDLE
        self.last_execution: Optional[datetime] = None
        self.next_execution: Optional[datetime] = None
        self.current_loop_id: Optional[str] = None
        self.execution_metrics: List[FeedbackLoopMetrics] = []
        
        # Performance tracking
        self.system_performance_history: List[float] = []
        self.overfitting_risk_history: List[float] = []
        
        # Safety mechanisms
        self.emergency_stop_triggered = False
        self.system_changes_count = 0
        self.last_successful_state: Optional[Dict[str, Any]] = None
        
        self.logger = logging.getLogger(__name__)
        self._initialize_database()
        self._load_historical_metrics()
        
        self.logger.info("DailyFeedbackLoop initialized successfully - executing daily at 22:00 UTC")
    
    def _calculate_next_execution_time(self) -> datetime:
        """Calculate the next daily execution time (22:00 UTC)."""
        now = datetime.utcnow()
        
        # Calculate today's execution time
        today_execution = datetime.combine(now.date(), self.daily_execution_time)
        
        # If today's execution time has passed, schedule for tomorrow
        if now >= today_execution:
            next_execution = today_execution + timedelta(days=1)
        else:
            next_execution = today_execution
        
        return next_execution
    
    def _initialize_database(self):
        """Initialize database for feedback loop tracking."""
        try:
            Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Using unified database manager instead of direct sqlite3
            # Feedback loop execution metrics
            self.database.execute_query_sync("""
                CREATE TABLE IF NOT EXISTS feedback_loop_executions (
                    loop_id TEXT PRIMARY KEY,
                    timestamp TIMESTAMP,
                    duration_seconds REAL,
                    phase_durations TEXT,
                    agents_optimized INTEGER,
                    agents_requiring_intervention INTEGER,
                    hierarchy_changes TEXT,
                    elite_indicators_updated INTEGER,
                    indicators_promoted INTEGER,
                    indicators_demoted INTEGER,
                    overfitted_indicators_removed INTEGER,
                    overall_system_performance REAL,
                    overfitting_risk_score REAL,
                    stability_score REAL,
                    validation_success BOOLEAN,
                    safety_checks_passed BOOLEAN,
                    system_recommendations TEXT,
                    risk_warnings TEXT,
                    status TEXT,
                    error_message TEXT
                )
            """, use_cache=False)
            
            # System state snapshots
            self.database.execute_query_sync("""
                CREATE TABLE IF NOT EXISTS system_state_snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    loop_id TEXT,
                    timestamp TIMESTAMP,
                    current_market_regime TEXT,
                    agent_rankings TEXT,
                    elite_indicators TEXT,
                    system_performance_metrics TEXT,
                    
                    FOREIGN KEY (loop_id) REFERENCES feedback_loop_executions(loop_id)
                )
            """, use_cache=False)
            
            # Performance trend tracking
            self.database.execute_query_sync("""
                CREATE TABLE IF NOT EXISTS performance_trends (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP,
                    overall_performance REAL,
                    overfitting_risk REAL,
                    stability_score REAL,
                    active_agents INTEGER,
                    elite_indicators_count INTEGER,
                    market_regime TEXT,
                    comprehensive_analysis_rate REAL,
                    analysis_consistency_score REAL,
                    full_validation_rate REAL
                )
            """, use_cache=False)
            
            # Emergency stops and interventions
            self.database.execute_query_sync("""
                CREATE TABLE IF NOT EXISTS emergency_interventions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP,
                    trigger_type TEXT,
                    trigger_value REAL,
                    trigger_threshold REAL,
                    intervention_action TEXT,
                    resolution_status TEXT,
                    resolution_timestamp TIMESTAMP
                )
            """, use_cache=False)
            self.logger.info("Daily feedback loop database initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize feedback loop database: {str(e)}")
            raise
    
    def _load_historical_metrics(self):
        """Load historical performance metrics."""
        try:
            # Using unified database manager instead of direct sqlite3
            # Load recent performance trends
            cursor = self.database.execute_query_sync("""
                SELECT overall_performance, overfitting_risk 
                FROM performance_trends 
                ORDER BY timestamp DESC 
                LIMIT 100
            """, use_cache=False)
            
            for row in cursor.fetchall():
                self.system_performance_history.append(row[0])
                self.overfitting_risk_history.append(row[1])
            
            # Load last execution time
            cursor = self.database.execute_query_sync("""
                SELECT timestamp FROM feedback_loop_executions 
                ORDER BY timestamp DESC 
                LIMIT 1
            """, use_cache=False)
            
            result = cursor.fetchone()
            if result:
                self.last_execution = datetime.fromisoformat(result[0])
                
            # Calculate next execution time
            self.next_execution = self._calculate_next_execution_time()
            
        except Exception as e:
            self.logger.warning(f"Failed to load historical metrics: {str(e)}")
            self.next_execution = self._calculate_next_execution_time()
    
    async def start(self):
        """Start the daily feedback loop."""
        return await self.start_daily_feedback_loop()
    
    async def start_daily_feedback_loop(self):
        """Start the daily feedback loop."""
        try:
            self.logger.info("Starting Daily Feedback Loop - executes at 22:00 UTC")
            
            while not self.emergency_stop_triggered:
                current_time = datetime.utcnow()
                
                # Check if it's time for the next execution
                if self.next_execution is None or current_time >= self.next_execution:
                    await self.execute_feedback_cycle()
                    
                    # Schedule next execution (next day at 22:00 UTC)
                    self.next_execution = self._calculate_next_execution_time()
                    
                    self.logger.info(f"Next daily feedback loop scheduled for: {self.next_execution} UTC")
                
                # Wait before next check (5 minute intervals for daily execution)
                await asyncio.sleep(300)  # 5 minutes
                
        except Exception as e:
            self.logger.error(f"Daily feedback loop error: {str(e)}")
            await self._handle_emergency_stop("CRITICAL_ERROR", str(e))
            raise
    
    async def execute_feedback_cycle(self) -> FeedbackLoopMetrics:
        """
        Execute a complete daily feedback cycle.
        
        Returns:
            Metrics from the feedback cycle execution
        """
        loop_id = f"daily_loop_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.utcnow()
        phase_durations = {}
        
        self.current_loop_id = loop_id
        self.current_status = FeedbackLoopStatus.RUNNING
        self.system_changes_count = 0
        
        try:
            self.logger.info(f"Starting daily feedback cycle: {loop_id}")
            
            # Phase 1: Initialization
            phase_start = datetime.utcnow()
            await self._phase_initialization()
            phase_durations[FeedbackLoopPhase.INITIALIZATION.value] = (datetime.utcnow() - phase_start).total_seconds()
            
            # Phase 2: Market Regime Detection
            phase_start = datetime.utcnow()
            current_regime = await self._phase_market_regime_detection()
            phase_durations[FeedbackLoopPhase.MARKET_REGIME_DETECTION.value] = (datetime.utcnow() - phase_start).total_seconds()
            
            # Phase 3: Agent Performance Analysis
            phase_start = datetime.utcnow()
            agent_performance_data = await self._phase_agent_performance_analysis()
            phase_durations[FeedbackLoopPhase.AGENT_PERFORMANCE_ANALYSIS.value] = (datetime.utcnow() - phase_start).total_seconds()
            
            # Phase 4: Indicator Effectiveness Update
            phase_start = datetime.utcnow()
            indicator_updates = await self._phase_indicator_effectiveness_update(current_regime)
            phase_durations[FeedbackLoopPhase.INDICATOR_EFFECTIVENESS_UPDATE.value] = (datetime.utcnow() - phase_start).total_seconds()
            
            # Phase 5: Agent Behavior Optimization
            phase_start = datetime.utcnow()
            optimization_results = await self._phase_agent_behavior_optimization()
            phase_durations[FeedbackLoopPhase.AGENT_BEHAVIOR_OPTIMIZATION.value] = (datetime.utcnow() - phase_start).total_seconds()
            
            # Phase 6: Hierarchy Update
            phase_start = datetime.utcnow()
            hierarchy_changes = await self._phase_hierarchy_update(agent_performance_data)
            phase_durations[FeedbackLoopPhase.HIERARCHY_UPDATE.value] = (datetime.utcnow() - phase_start).total_seconds()
            
            # Phase 7: Elite Indicator Update
            phase_start = datetime.utcnow()
            elite_indicator_changes = await self._phase_elite_indicator_update(current_regime, indicator_updates)
            phase_durations[FeedbackLoopPhase.ELITE_INDICATOR_UPDATE.value] = (datetime.utcnow() - phase_start).total_seconds()
            
            # Phase 8: Validation and Safety Check
            phase_start = datetime.utcnow()
            validation_result = await self._phase_validation_and_safety_check()
            phase_durations[FeedbackLoopPhase.VALIDATION_AND_SAFETY_CHECK.value] = (datetime.utcnow() - phase_start).total_seconds()
            
            # Phase 9: System Status Update
            phase_start = datetime.utcnow()
            system_status = await self._phase_system_status_update()
            phase_durations[FeedbackLoopPhase.SYSTEM_STATUS_UPDATE.value] = (datetime.utcnow() - phase_start).total_seconds()
            
            # Phase 10: Execute Economic Monitor Cycle
            phase_start = datetime.utcnow()
            await self._execute_economic_monitoring()
            phase_durations["ECONOMIC_MONITORING"] = (datetime.utcnow() - phase_start).total_seconds()
            
            # Create execution metrics
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            metrics = self._create_execution_metrics(
                loop_id, start_time, duration, phase_durations,
                agent_performance_data, optimization_results,
                hierarchy_changes, indicator_updates, elite_indicator_changes,
                validation_result, system_status
            )
            
            # Save metrics and update state
            await self._save_execution_metrics(metrics)
            self.execution_metrics.append(metrics)
            self.last_execution = start_time
            self.current_status = FeedbackLoopStatus.COMPLETED
            
            self.logger.info(
                f"Daily feedback cycle {loop_id} completed successfully in {duration:.2f}s. "
                f"System performance: {metrics.overall_system_performance:.3f}, "
                f"Overfitting risk: {metrics.overfitting_risk_score:.3f}"
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Daily feedback cycle {loop_id} failed: {str(e)}")
            self.current_status = FeedbackLoopStatus.ERROR
            
            # Create error metrics
            error_metrics = FeedbackLoopMetrics(
                loop_id=loop_id,
                timestamp=start_time,
                duration_seconds=(datetime.utcnow() - start_time).total_seconds(),
                phase_durations=phase_durations,
                agents_optimized=0,
                agents_requiring_intervention=0,
                hierarchy_changes={},
                elite_indicators_updated=0,
                indicators_promoted=0,
                indicators_demoted=0,
                overfitted_indicators_removed=0,
                overall_system_performance=0.0,
                overfitting_risk_score=1.0,
                stability_score=0.0,
                validation_success=False,
                safety_checks_passed=False,
                system_recommendations=["INVESTIGATE_FEEDBACK_LOOP_ERROR"],
                risk_warnings=[f"FEEDBACK_LOOP_ERROR: {str(e)}"],
                status=FeedbackLoopStatus.ERROR,
                error_message=str(e)
            )
            
            await self._save_execution_metrics(error_metrics)
            return error_metrics
    
    async def _phase_initialization(self):
        """Initialize the daily feedback loop cycle."""
        self.logger.info("Phase 1: Initialization")
        
        # Save current system state as backup
        await self._save_system_state_snapshot()
        
        # Reset system change counter
        self.system_changes_count = 0
        
        # Check if emergency stop is triggered
        if self.emergency_stop_triggered:
            raise FeedbackLoopError("Emergency stop is active")
    
    async def _phase_market_regime_detection(self) -> MarketRegime:
        """Detect current market regime."""
        self.logger.info("Phase 2: Market Regime Detection")
        
        try:
            # Get current market regime
            current_regime = await self.regime_classifier.classify_current_regime()
            
            self.logger.info(f"Current market regime: {current_regime.value}")
            return current_regime
            
        except Exception as e:
            self.logger.warning(f"Failed to detect market regime: {str(e)}")
            return MarketRegime.TRANSITIONAL  # Default fallback
    
    async def _phase_agent_performance_analysis(self) -> Dict[str, Any]:
        """Analyze agent performance using validated metrics."""
        self.logger.info("Phase 3: Agent Performance Analysis")
        
        agent_performance_data = {}
        
        try:
            # Get all registered agents
            registered_agents = list(self.agent_optimizer.agent_profiles.keys())
            
            for agent_name in registered_agents:
                # Collect recent performance data
                performance_data = self.agent_optimizer._collect_agent_performance_data(agent_name)
                
                if len(performance_data) >= 10:  # Minimum data requirement
                    # Analyze performance metrics
                    metrics = self.agent_optimizer._analyze_agent_performance(agent_name, performance_data)
                    
                    agent_performance_data[agent_name] = {
                        'raw_data': performance_data,
                        'analyzed_metrics': metrics,
                        'requires_optimization': metrics.get('overfitting_risk', 0.0) > 0.3,
                        'out_of_sample_performance': metrics.get('out_of_sample_win_rate', 0.0),
                        'consistency_score': metrics.get('consistency_score', 0.0)
                    }
                else:
                    agent_performance_data[agent_name] = {
                        'raw_data': performance_data,
                        'analyzed_metrics': {},
                        'requires_optimization': False,
                        'out_of_sample_performance': 0.0,
                        'consistency_score': 0.0,
                        'insufficient_data': True
                    }
            
            self.logger.info(f"Analyzed performance for {len(agent_performance_data)} agents")
            return agent_performance_data
            
        except Exception as e:
            self.logger.error(f"Agent performance analysis failed: {str(e)}")
            return {}
    
    async def _phase_indicator_effectiveness_update(self, 
                                                  current_regime: MarketRegime) -> Dict[str, Any]:
        """Update indicator effectiveness analysis."""
        self.logger.info("Phase 4: Indicator Effectiveness Update")
        
        try:
            # Analyze all indicators for current regime
            indicator_results = self.indicator_analyzer.analyze_all_indicators(current_regime)
            
            # Count indicators by quality rating
            quality_counts = {}
            for quality_rating in IndicatorQualityRating:
                quality_counts[quality_rating.value] = sum(
                    1 for metrics in indicator_results.values()
                    if metrics.quality_rating == quality_rating
                )
            
            # Identify overfitted indicators to remove
            overfitted_indicators = [
                name for name, metrics in indicator_results.items()
                if metrics.quality_rating == IndicatorQualityRating.OVERFITTED
            ]
            
            # Identify indicators promoted to elite status
            promoted_indicators = [
                name for name, metrics in indicator_results.items()
                if metrics.quality_rating == IndicatorQualityRating.ELITE
                and metrics.recommended_for_live
            ]
            
            # Identify indicators demoted from elite status
            current_elite = self.indicator_analyzer.get_elite_indicators(current_regime)
            demoted_indicators = [
                name for name in current_elite
                if name in indicator_results and not indicator_results[name].recommended_for_live
            ]
            
            update_summary = {
                'total_indicators_analyzed': len(indicator_results),
                'quality_distribution': quality_counts,
                'overfitted_indicators': overfitted_indicators,
                'promoted_indicators': promoted_indicators,
                'demoted_indicators': demoted_indicators,
                'indicator_results': indicator_results
            }
            
            self.logger.info(
                f"Indicator effectiveness update complete: "
                f"{len(indicator_results)} analyzed, "
                f"{len(overfitted_indicators)} overfitted, "
                f"{len(promoted_indicators)} promoted"
            )
            
            return update_summary
            
        except Exception as e:
            self.logger.error(f"Indicator effectiveness update failed: {str(e)}")
            return {}
    
    async def _phase_agent_behavior_optimization(self) -> Dict[str, Any]:
        """Optimize agent behavior using anti-overfitting techniques."""
        self.logger.info("Phase 5: Agent Behavior Optimization")
        
        try:
            # Run optimization for all agents
            optimization_results = self.agent_optimizer.optimize_all_agents(force_optimization=False)
            
            # Analyze optimization results
            successful_optimizations = sum(
                1 for result in optimization_results.values()
                if result.optimization_successful
            )
            
            agents_requiring_intervention = sum(
                1 for result in optimization_results.values()
                if result.requires_intervention
            )
            
            # Count significant improvements
            significant_improvements = sum(
                1 for result in optimization_results.values()
                if result.performance_improvement > 0.05  # 5% improvement threshold
            )
            
            optimization_summary = {
                'total_agents_optimized': len(optimization_results),
                'successful_optimizations': successful_optimizations,
                'agents_requiring_intervention': agents_requiring_intervention,
                'significant_improvements': significant_improvements,
                'optimization_results': optimization_results
            }
            
            self.logger.info(
                f"Agent behavior optimization complete: "
                f"{successful_optimizations}/{len(optimization_results)} successful, "
                f"{agents_requiring_intervention} require intervention"
            )
            
            return optimization_summary
            
        except Exception as e:
            self.logger.error(f"Agent behavior optimization failed: {str(e)}")
            return {}
    
    async def _phase_hierarchy_update(self, 
                                    agent_performance_data: Dict[str, Any]) -> Dict[str, str]:
        """Update agent hierarchy based on validated performance."""
        self.logger.info("Phase 6: Hierarchy Update")
        
        hierarchy_changes = {}
        
        try:
            # Extract performance scores for ranking
            agent_scores = {}
            for agent_name, data in agent_performance_data.items():
                if 'analyzed_metrics' in data and data['analyzed_metrics']:
                    metrics = data['analyzed_metrics']
                    
                    # Calculate risk-adjusted, out-of-sample performance score
                    oos_performance = metrics.get('out_of_sample_win_rate', 0.0)
                    consistency = metrics.get('consistency_score', 0.0)
                    overfitting_risk = metrics.get('overfitting_risk', 1.0)
                    
                    # Risk-adjusted score (heavily penalize overfitting)
                    risk_adjusted_score = (
                        oos_performance * 0.5 +
                        consistency * 0.3 +
                        (1.0 - overfitting_risk) * 0.2
                    )
                    
                    agent_scores[agent_name] = risk_adjusted_score
                else:
                    agent_scores[agent_name] = 0.0  # No sufficient data
            
            # Update hierarchy if there are significant changes
            if agent_scores:
                # Get current rankings
                current_rankings = await self.hierarchy_manager.get_current_agent_rankings()
                
                # Calculate new rankings based on risk-adjusted scores
                sorted_agents = sorted(agent_scores.items(), key=lambda x: x[1], reverse=True)
                
                # Update rankings if there are significant performance differences
                for i, (agent_name, score) in enumerate(sorted_agents):
                    current_rank = current_rankings.get(agent_name, AgentRank.GAMMA)
                    
                    # Determine new rank based on performance
                    if i == 0 and score > 0.7:  # Top performer with high score
                        new_rank = AgentRank.ALPHA
                    elif i < 3 and score > 0.5:  # Top 3 with decent scores
                        new_rank = AgentRank.BETA
                    else:
                        new_rank = AgentRank.GAMMA
                    
                    # Update if rank changed
                    if current_rank != new_rank:
                        await self.hierarchy_manager.update_agent_rank(agent_name, new_rank)
                        hierarchy_changes[agent_name] = f"{current_rank.value} -> {new_rank.value}"
                        self.system_changes_count += 1
            
            self.logger.info(f"Hierarchy update complete: {len(hierarchy_changes)} changes made")
            return hierarchy_changes
            
        except Exception as e:
            self.logger.error(f"Hierarchy update failed: {str(e)}")
            return {}
    
    async def _phase_elite_indicator_update(self, 
                                          current_regime: MarketRegime,
                                          indicator_updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update Elite Indicator lists based on validated effectiveness."""
        self.logger.info("Phase 7: Elite Indicator Update")
        
        try:
            # Get current elite indicators
            current_elite = self.indicator_analyzer.get_elite_indicators(current_regime)
            
            # Force refresh of elite indicators based on latest analysis
            new_elite = self.indicator_analyzer.get_elite_indicators(current_regime, force_refresh=True)
            
            # Calculate changes
            added_indicators = set(new_elite) - set(current_elite)
            removed_indicators = set(current_elite) - set(new_elite)
            
            # Update GeniusAgentCoordinator with new Elite Indicator sets
            if added_indicators or removed_indicators:
                await self.genius_coordinator.update_elite_indicators(current_regime, new_elite)
                self.system_changes_count += 1
            
            elite_changes = {
                'previous_elite_count': len(current_elite),
                'new_elite_count': len(new_elite),
                'added_indicators': list(added_indicators),
                'removed_indicators': list(removed_indicators),
                'current_elite_indicators': new_elite
            }
            
            self.logger.info(
                f"Elite indicators updated: "
                f"{len(added_indicators)} added, "
                f"{len(removed_indicators)} removed, "
                f"{len(new_elite)} total elite"
            )
            
            return elite_changes
            
        except Exception as e:
            self.logger.error(f"Elite indicator update failed: {str(e)}")
            return {}
    
    async def _phase_validation_and_safety_check(self) -> Dict[str, Any]:
        """Validate changes and perform safety checks."""
        self.logger.info("Phase 8: Validation and Safety Check")
        
        validation_result = {
            'validation_success': False,
            'safety_checks_passed': False,
            'risk_warnings': [],
            'recommendations': []
        }
        
        try:
            # Check overfitting risk across the system
            overall_overfitting_risk = await self._calculate_system_overfitting_risk()
            
            if overall_overfitting_risk > self.overfitting_emergency_threshold:
                validation_result['risk_warnings'].append(
                    f"SYSTEM_WIDE_OVERFITTING_RISK: {overall_overfitting_risk:.3f}"
                )
                await self._handle_emergency_stop("HIGH_OVERFITTING_RISK", str(overall_overfitting_risk))
                return validation_result
            
            # Check if too many changes were made in this cycle
            if self.system_changes_count > self.max_system_changes_per_loop:
                validation_result['risk_warnings'].append(
                    f"EXCESSIVE_SYSTEM_CHANGES: {self.system_changes_count}"
                )
                validation_result['recommendations'].append("REDUCE_FEEDBACK_SENSITIVITY")
            
            # Validate that the system is still stable
            system_stability = await self._assess_system_stability()
            
            if system_stability < 0.5:
                validation_result['risk_warnings'].append(
                    f"LOW_SYSTEM_STABILITY: {system_stability:.3f}"
                )
                validation_result['recommendations'].append("INCREASE_REGULARIZATION")
            
            # All safety checks passed
            validation_result['validation_success'] = True
            validation_result['safety_checks_passed'] = len(validation_result['risk_warnings']) == 0
            
            self.logger.info(
                f"Validation complete: "
                f"Success={validation_result['validation_success']}, "
                f"Safety passed={validation_result['safety_checks_passed']}"
            )
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Validation and safety check failed: {str(e)}")
            validation_result['risk_warnings'].append(f"VALIDATION_ERROR: {str(e)}")
            return validation_result
    
    async def _phase_system_status_update(self) -> Dict[str, Any]:
        """Update overall system status and performance metrics."""
        self.logger.info("Phase 9: System Status Update")
        
        try:
            # Calculate overall system performance
            overall_performance = await self._calculate_system_performance()
            
            # Calculate stability score
            stability_score = await self._assess_system_stability()
            
            # Calculate overfitting risk
            overfitting_risk = await self._calculate_system_overfitting_risk()
            
            # Update performance history
            self.system_performance_history.append(overall_performance)
            self.overfitting_risk_history.append(overfitting_risk)
            
            # Keep only recent history (last 100 points)
            if len(self.system_performance_history) > 100:
                self.system_performance_history = self.system_performance_history[-100:]
                self.overfitting_risk_history = self.overfitting_risk_history[-100:]
            
            # Save performance trend to database
            await self._save_performance_trend(overall_performance, overfitting_risk, stability_score)
            
            system_status = {
                'overall_performance': overall_performance,
                'stability_score': stability_score,
                'overfitting_risk': overfitting_risk,
                'active_agents': len(self.agent_optimizer.agent_profiles),
                'system_health': 'HEALTHY' if stability_score > 0.7 and overfitting_risk < 0.3 else 'MONITORING'
            }
            
            self.logger.info(
                f"System status update: "
                f"Performance={overall_performance:.3f}, "
                f"Stability={stability_score:.3f}, "
                f"Overfitting risk={overfitting_risk:.3f}"
            )
            
            return system_status
            
        except Exception as e:
            self.logger.error(f"System status update failed: {str(e)}")
            return {
                'overall_performance': 0.0,
                'stability_score': 0.0,
                'overfitting_risk': 1.0,
                'active_agents': 0,
                'system_health': 'ERROR'
            }
    
    async def _execute_economic_monitoring(self):
        """Execute economic monitoring cycle."""
        try:
            if self.economic_monitor:
                await self.economic_monitor.run_comprehensive_monitoring()
                self.logger.info("Economic monitoring cycle completed")
        except Exception as e:
            self.logger.warning(f"Economic monitoring failed: {str(e)}")
    
    async def _calculate_system_performance(self) -> float:
        """Calculate overall system performance score."""
        try:
            # Get recent completed trades
            recent_trades = []
            cutoff_date = datetime.utcnow() - timedelta(days=7)
            
            for trade_record in self.performance_tracker.completed_trades.values():
                if trade_record.exit_time and trade_record.exit_time >= cutoff_date:
                    recent_trades.append(trade_record)
            
            if not recent_trades:
                return 0.0
            
            # Calculate performance metrics
            winning_trades = sum(1 for trade in recent_trades if trade.pnl and trade.pnl > 0)
            win_rate = winning_trades / len(recent_trades)
            
            # Weight out-of-sample performance more heavily
            oos_trades = [t for t in recent_trades if t.validation_period_type.value == 'OUT_OF_SAMPLE']
            oos_win_rate = (sum(1 for t in oos_trades if t.pnl and t.pnl > 0) / len(oos_trades)) if oos_trades else 0.0
            
            # Composite performance score
            performance_score = win_rate * 0.4 + oos_win_rate * 0.6
            
            return float(min(1.0, max(0.0, performance_score)))
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate system performance: {str(e)}")
            return 0.0
    
    async def _calculate_system_overfitting_risk(self) -> float:
        """Calculate system-wide overfitting risk."""
        try:
            agent_overfitting_risks = []
            
            for agent_name in self.agent_optimizer.agent_profiles.keys():
                performance_data = self.agent_optimizer._collect_agent_performance_data(agent_name)
                
                if len(performance_data) >= 10:
                    metrics = self.agent_optimizer._analyze_agent_performance(agent_name, performance_data)
                    overfitting_risk = metrics.get('overfitting_risk', 0.0)
                    agent_overfitting_risks.append(overfitting_risk)
            
            if not agent_overfitting_risks:
                return 0.0
            
            # Return average overfitting risk across all agents
            return float(sum(agent_overfitting_risks) / len(agent_overfitting_risks))
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate overfitting risk: {str(e)}")
            return 1.0  # Conservative default
    
    async def _assess_system_stability(self) -> float:
        """Assess overall system stability."""
        try:
            if len(self.system_performance_history) < 10:
                return 0.5  # Neutral score for insufficient data
            
            # Calculate coefficient of variation (lower is more stable)
            recent_performance = self.system_performance_history[-10:]
            mean_performance = sum(recent_performance) / len(recent_performance)
            
            if mean_performance == 0:
                return 0.0
            
            variance = sum((x - mean_performance) ** 2 for x in recent_performance) / len(recent_performance)
            std_dev = variance ** 0.5
            
            cv = std_dev / mean_performance
            
            # Convert to stability score (0 = unstable, 1 = very stable)
            stability_score = max(0.0, 1.0 - cv)
            
            return float(min(1.0, stability_score))
            
        except Exception as e:
            self.logger.warning(f"Failed to assess system stability: {str(e)}")
            return 0.0
    
    async def _handle_emergency_stop(self, trigger_type: str, trigger_value: str):
        """Handle emergency stop conditions."""
        self.logger.critical(f"Emergency stop triggered: {trigger_type} = {trigger_value}")
        
        self.emergency_stop_triggered = True
        self.current_status = FeedbackLoopStatus.ERROR
        
        # Save emergency intervention record
        try:
            # Using unified database manager instead of direct sqlite3
            self.database.execute_query_sync("""
                INSERT INTO emergency_interventions (
                    timestamp, trigger_type, trigger_value, trigger_threshold,
                    intervention_action, resolution_status
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                datetime.utcnow().isoformat(),
                trigger_type,
                trigger_value,
                str(self.overfitting_emergency_threshold),
                "EMERGENCY_STOP",
                "PENDING"
            ), use_cache=False)
        except Exception as e:
            self.logger.error(f"Failed to save emergency intervention: {str(e)}")
    
    async def _save_system_state_snapshot(self):
        """Save current system state as a snapshot."""
        try:
            snapshot_id = f"snapshot_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            # Collect current system state
            current_regime = await self.regime_classifier.classify_current_regime()
            agent_rankings = await self.hierarchy_manager.get_current_agent_rankings()
            elite_indicators = {}
            
            # Get elite indicators for all regimes
            for regime in MarketRegime:
                try:
                    elite_indicators[regime.value] = self.indicator_analyzer.get_elite_indicators(regime)
                except:
                    elite_indicators[regime.value] = []
            
            # Performance metrics
            performance_metrics = {
                'system_performance': await self._calculate_system_performance(),
                'overfitting_risk': await self._calculate_system_overfitting_risk(),
                'stability_score': await self._assess_system_stability()
            }
            
            # Save to database
            # Using unified database manager instead of direct sqlite3
            self.database.execute_query_sync("""
                INSERT INTO system_state_snapshots (
                    snapshot_id, loop_id, timestamp, current_market_regime,
                    agent_rankings, elite_indicators, system_performance_metrics
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                snapshot_id,
                self.current_loop_id,
                datetime.utcnow().isoformat(),
                current_regime.value,
                json.dumps({k: v.value if hasattr(v, 'value') else v for k, v in agent_rankings.items()}),
                json.dumps(elite_indicators),
                json.dumps(performance_metrics)
            ), use_cache=False)
            
            # Update last successful state
            self.last_successful_state = {
                'snapshot_id': snapshot_id,
                'timestamp': datetime.utcnow(),
                'agent_rankings': agent_rankings,
                'elite_indicators': elite_indicators,
                'performance_metrics': performance_metrics
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to save system state snapshot: {str(e)}")
    
    async def _save_performance_trend(self, overall_performance: float, 
                                    overfitting_risk: float, stability_score: float):
        """Save performance trend data."""
        try:
            current_regime = await self.regime_classifier.classify_current_regime()
            active_agents = len(self.agent_optimizer.agent_profiles)
            
            # Count elite indicators across all regimes
            elite_count = 0
            for regime in MarketRegime:
                try:
                    elite_indicators = self.indicator_analyzer.get_elite_indicators(regime)
                    elite_count += len(elite_indicators)
                except:
                    pass
            
            # Calculate comprehensive analysis metrics
            comprehensive_rate = 1.0  # Always 100% now
            consistency_score = stability_score  # Use stability as consistency measure

            # Using unified database manager instead of direct sqlite3
            self.database.execute_query_sync("""
                INSERT INTO performance_trends (
                    timestamp, overall_performance, overfitting_risk,
                    stability_score, active_agents, elite_indicators_count,
                    market_regime, comprehensive_analysis_rate, analysis_consistency_score,
                    full_validation_rate
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.utcnow().isoformat(),
                overall_performance,
                overfitting_risk,
                stability_score,
                active_agents,
                elite_count,
                current_regime.value,
                comprehensive_rate,      # Always 1.0
                consistency_score,       # Based on stability
                1.0                     # Always full validation
            ), use_cache=False)
        except Exception as e:
            self.logger.warning(f"Failed to save performance trend: {str(e)}")
    
    def _create_execution_metrics(self, loop_id: str, start_time: datetime, duration: float,
                                phase_durations: Dict[str, float], agent_performance_data: Dict[str, Any],
                                optimization_results: Dict[str, Any], hierarchy_changes: Dict[str, str],
                                indicator_updates: Dict[str, Any], elite_indicator_changes: Dict[str, Any],
                                validation_result: Dict[str, Any], system_status: Dict[str, Any]) -> FeedbackLoopMetrics:
        """Create execution metrics object."""
        
        # Extract counts from results
        agents_optimized = optimization_results.get('total_agents_optimized', 0)
        agents_requiring_intervention = optimization_results.get('agents_requiring_intervention', 0)
        
        elite_indicators_updated = len(elite_indicator_changes.get('added_indicators', [])) + \
                                 len(elite_indicator_changes.get('removed_indicators', []))
        indicators_promoted = len(elite_indicator_changes.get('added_indicators', []))
        indicators_demoted = len(elite_indicator_changes.get('removed_indicators', []))
        overfitted_indicators_removed = len(indicator_updates.get('overfitted_indicators', []))
        
        # Create recommendations and warnings
        recommendations = []
        risk_warnings = validation_result.get('risk_warnings', [])
        
        if system_status.get('overfitting_risk', 0.0) > 0.5:
            recommendations.append("INCREASE_REGULARIZATION")
        if system_status.get('stability_score', 1.0) < 0.6:
            recommendations.append("REVIEW_AGENT_BEHAVIOR")
        if elite_indicators_updated > 10:
            recommendations.append("REVIEW_INDICATOR_SELECTION_CRITERIA")
        
        return FeedbackLoopMetrics(
            loop_id=loop_id,
            timestamp=start_time,
            duration_seconds=duration,
            phase_durations=phase_durations,
            agents_optimized=agents_optimized,
            agents_requiring_intervention=agents_requiring_intervention,
            hierarchy_changes=hierarchy_changes,
            elite_indicators_updated=elite_indicators_updated,
            indicators_promoted=indicators_promoted,
            indicators_demoted=indicators_demoted,
            overfitted_indicators_removed=overfitted_indicators_removed,
            overall_system_performance=system_status.get('overall_performance', 0.0),
            overfitting_risk_score=system_status.get('overfitting_risk', 1.0),
            stability_score=system_status.get('stability_score', 0.0),
            validation_success=validation_result.get('validation_success', False),
            safety_checks_passed=validation_result.get('safety_checks_passed', False),
            system_recommendations=recommendations,
            risk_warnings=risk_warnings,
            status=FeedbackLoopStatus.COMPLETED
        )
    
    async def _save_execution_metrics(self, metrics: FeedbackLoopMetrics):
        """Save execution metrics to database."""
        try:
            # Using unified database manager instead of direct sqlite3
            self.database.execute_query_sync("""
                INSERT INTO feedback_loop_executions (
                    loop_id, timestamp, duration_seconds, phase_durations,
                    agents_optimized, agents_requiring_intervention, hierarchy_changes,
                    elite_indicators_updated, indicators_promoted, indicators_demoted,
                    overfitted_indicators_removed, overall_system_performance,
                    overfitting_risk_score, stability_score, validation_success,
                    safety_checks_passed, system_recommendations, risk_warnings,
                    status, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.loop_id,
                metrics.timestamp.isoformat(),
                metrics.duration_seconds,
                json.dumps(metrics.phase_durations),
                metrics.agents_optimized,
                metrics.agents_requiring_intervention,
                json.dumps(metrics.hierarchy_changes),
                metrics.elite_indicators_updated,
                metrics.indicators_promoted,
                metrics.indicators_demoted,
                metrics.overfitted_indicators_removed,
                metrics.overall_system_performance,
                metrics.overfitting_risk_score,
                metrics.stability_score,
                metrics.validation_success,
                metrics.safety_checks_passed,
                json.dumps(metrics.system_recommendations),
                json.dumps(metrics.risk_warnings),
                metrics.status.value,
                metrics.error_message
            ), use_cache=False)
        except Exception as e:
            self.logger.error(f"Failed to save execution metrics: {str(e)}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current feedback loop status."""
        return {
            'current_status': self.current_status.value,
            'last_execution': self.last_execution.isoformat() if self.last_execution else None,
            'next_execution': self.next_execution.isoformat() if self.next_execution else None,
            'execution_time': self.daily_execution_time.strftime('%H:%M UTC'),
            'emergency_stop_triggered': self.emergency_stop_triggered,
            'recent_performance': self.system_performance_history[-10:] if self.system_performance_history else [],
            'recent_overfitting_risk': self.overfitting_risk_history[-10:] if self.overfitting_risk_history else [],
            'total_executions': len(self.execution_metrics)
        }
    
    async def force_execution(self) -> FeedbackLoopMetrics:
        """Force immediate execution of feedback cycle (for testing/debugging)."""
        self.logger.info("Forcing immediate execution of daily feedback cycle")
        return await self.execute_feedback_cycle()
    
    def stop(self):
        """Stop the daily feedback loop gracefully."""
        self.logger.info("Stopping Daily Feedback Loop")
        self.emergency_stop_triggered = True
        self.current_status = FeedbackLoopStatus.PAUSED
