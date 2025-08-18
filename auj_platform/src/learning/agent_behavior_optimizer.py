"""
Agent Behavior Optimizer for AUJ Platform.

This component implements intelligent optimization with advanced anti-overfitting techniques
to ensure AI agents develop robust, generalizable trading strategies rather than memorizing
historical data patterns.

Key Anti-Overfitting Techniques:
- Model Regularization (L1/L2 penalties for complexity)
- Performance Decay Factor (recent data weighting)
- Constrained Learning Rate (prevent drastic changes)
- Ensemble Learning (committee-based decisions)
- Adaptive Learning Rate (based on market volatility)
"""

from datetime import datetime, timedelta
import sqlite3
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
from pathlib import Path
import math
from sqlalchemy import text

from ..core.unified_config import UnifiedConfigManager
from ..core.data_contracts import (
    AgentDecision, TradeSignal, GradedDeal, MarketRegime,
    ValidationResult, IndicatorResult
)
from ..core.exceptions import (
    OptimizationError, ValidationError, InsufficientDataError,
    AUJPlatformError
)
from ..core.unified_database_manager import get_unified_database
from ..validation.walk_forward_validator import WalkForwardValidator, ValidationPeriodType
from ..analytics.performance_tracker import PerformanceTracker
from ..analytics.indicator_effectiveness_analyzer import IndicatorEffectivenessAnalyzer
from ..core.unified_database_manager import get_unified_database_sync


class RegularizationType(str, Enum):
    """Types of regularization techniques."""
    L1 = "L1"              # Lasso regularization (sparse solutions)
    L2 = "L2"              # Ridge regularization (smooth solutions)
    ELASTIC_NET = "ELASTIC_NET"  # Combination of L1 and L2
    DROPOUT = "DROPOUT"    # Random neuron dropout
    EARLY_STOPPING = "EARLY_STOPPING"  # Stop before overfitting


class LearningRateSchedule(str, Enum):
    """Learning rate schedule types."""
    CONSTANT = "CONSTANT"
    EXPONENTIAL_DECAY = "EXPONENTIAL_DECAY"
    STEP_DECAY = "STEP_DECAY"
    ADAPTIVE = "ADAPTIVE"
    PERFORMANCE_BASED = "PERFORMANCE_BASED"


@dataclass
class AgentBehaviorProfile:
    """Behavioral profile for an agent with optimization parameters."""
    agent_name: str
    
    # Learning parameters
    base_learning_rate: float = 0.01
    current_learning_rate: float = 0.01
    learning_rate_schedule: LearningRateSchedule = LearningRateSchedule.ADAPTIVE
    
    # Regularization settings
    regularization_type: RegularizationType = RegularizationType.L2
    regularization_strength: float = 0.01
    dropout_rate: float = 0.1
    
    # Performance tracking
    recent_performance_scores: deque = field(default_factory=lambda: deque(maxlen=50))
    performance_decay_factor: float = 0.95
    consistency_score: float = 0.0
    
    # Behavioral constraints
    max_position_change_per_cycle: float = 0.1  # 10% max change
    max_strategy_deviation: float = 0.2  # 20% max deviation from baseline
    confidence_threshold: float = 0.6
    
    # Ensemble settings
    ensemble_size: int = 5
    ensemble_weights: List[float] = field(default_factory=list)
    ensemble_agreement_threshold: float = 0.7
    
    # Adaptation parameters
    volatility_adjustment_factor: float = 1.0
    regime_adaptation_speed: float = 0.1
    overfitting_detection_sensitivity: float = 0.8
    
    # Metadata
    last_optimization: Optional[datetime] = None
    optimization_cycles: int = 0
    
    def __post_init__(self):
        if not self.ensemble_weights:
            # Initialize uniform ensemble weights
            self.ensemble_weights = [1.0 / self.ensemble_size] * self.ensemble_size


@dataclass
class OptimizationResult:
    """Result of an agent behavior optimization cycle."""
    agent_name: str
    timestamp: datetime
    
    # Optimization metrics
    old_learning_rate: float
    new_learning_rate: float
    regularization_adjustment: float
    
    # Performance improvements
    performance_improvement: float
    overfitting_risk_reduction: float
    consistency_improvement: float
    
    # Behavioral changes
    strategy_adjustments: Dict[str, float]
    confidence_adjustments: Dict[str, float]
    
    # Validation results
    in_sample_performance: float
    out_of_sample_performance: float
    performance_degradation: float
    
    # Recommendations
    recommendations: List[str]
    risk_warnings: List[str]
    
    # Success indicators
    optimization_successful: bool
    requires_intervention: bool


class AgentBehaviorOptimizer:
    """
    Intelligent Agent Behavior Optimizer with Advanced Anti-Overfitting Techniques.
    
    This component ensures that AI agents develop robust, generalizable trading
    strategies by applying sophisticated optimization techniques that prevent
    overfitting to historical data.
    
    Key Features:
    - Model Regularization: Applies penalties for complexity
    - Performance Decay: Weights recent performance more heavily
    - Constrained Learning: Prevents drastic changes from market noise
    - Ensemble Methods: Uses committee-based decision making
    - Adaptive Optimization: Adjusts to market conditions
    """
    
    def __init__(self,
                 walk_forward_validator: WalkForwardValidator,
                 performance_tracker: PerformanceTracker,
                 indicator_analyzer: IndicatorEffectivenessAnalyzer,
                 database_path: Optional[str] = None,
                 config: Optional[Any] = None):
        """
        Initialize the Agent Behavior Optimizer.
        
        Args:
            walk_forward_validator: Validation engine for out-of-sample testing
            performance_tracker: Performance tracking system
            indicator_analyzer: Indicator effectiveness analyzer
            database_path: Path to optimization database
            config: Optional configuration override
        """
        from ..core.unified_config import get_unified_config
        self.config_manager = get_unified_config()
        
        self.walk_forward_validator = walk_forward_validator
        self.performance_tracker = performance_tracker
        self.indicator_analyzer = indicator_analyzer
        self.database_path = database_path or "data/agent_optimization.db"
        self.config = config or {}
        
        # Use unified database abstraction instead of direct path
        self.database = get_unified_database_sync()
        
        # Optimization parameters
        self.optimization_frequency = timedelta(hours=self.config_manager.get_int('optimization_frequency_hours', 6))
        self.min_data_points = self.config_manager.get_int('min_data_points', 30)
        self.overfitting_threshold = self.config_manager.get_float('overfitting_threshold', 0.25)
        self.performance_window_days = self.config_manager.get_int('performance_window_days', 30)
        
        # Regularization parameters
        self.default_l1_strength = self.config_manager.get_float('default_l1_strength', 0.01)
        self.default_l2_strength = self.config_manager.get_float('default_l2_strength', 0.01)
        self.max_learning_rate = self.config_manager.get_float('max_learning_rate', 0.1)
        self.min_learning_rate = self.config_manager.get_float('min_learning_rate', 0.001)
        
        # Agent behavior profiles
        self.agent_profiles: Dict[str, AgentBehaviorProfile] = {}
        
        # Optimization history
        self.optimization_history: Dict[str, List[OptimizationResult]] = defaultdict(list)
        
        # Performance baselines
        self.agent_baselines: Dict[str, Dict[str, float]] = {}
        
        self.logger = logging.getLogger(__name__)
        self._initialize_database()
        self._load_agent_profiles()
        
        self.logger.info("AgentBehaviorOptimizer initialized with anti-overfitting focus")
    
    def _initialize_database(self):
        """Initialize database for optimization tracking."""
        try:
            Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Using unified database manager instead of direct sqlite3
            # Agent behavior profiles table
            with self.database.get_sync_session() as session:
                session.execute(text("""
                    CREATE TABLE IF NOT EXISTS agent_profiles (
                        agent_name TEXT PRIMARY KEY,
                        base_learning_rate REAL,
                        current_learning_rate REAL,
                        learning_rate_schedule TEXT,
                            regularization_type TEXT,
                            regularization_strength REAL,
                            dropout_rate REAL,
                            performance_decay_factor REAL,
                            consistency_score REAL,
                            max_position_change_per_cycle REAL,
                            max_strategy_deviation REAL,
                            confidence_threshold REAL,
                            ensemble_size INTEGER,
                            ensemble_weights TEXT,
                            ensemble_agreement_threshold REAL,
                            volatility_adjustment_factor REAL,
                            regime_adaptation_speed REAL,
                            overfitting_detection_sensitivity REAL,
                            last_optimization TIMESTAMP,
                            optimization_cycles INTEGER,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """))
            
                # Optimization results table
                session.execute(text("""
                    CREATE TABLE IF NOT EXISTS optimization_results (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            agent_name TEXT,
                            timestamp TIMESTAMP,
                            old_learning_rate REAL,
                            new_learning_rate REAL,
                            regularization_adjustment REAL,
                            performance_improvement REAL,
                            overfitting_risk_reduction REAL,
                            consistency_improvement REAL,
                            strategy_adjustments TEXT,
                            confidence_adjustments TEXT,
                            in_sample_performance REAL,
                            out_of_sample_performance REAL,
                            performance_degradation REAL,
                            recommendations TEXT,
                            risk_warnings TEXT,
                            optimization_successful BOOLEAN,
                            requires_intervention BOOLEAN,
                            
                            FOREIGN KEY (agent_name) REFERENCES agent_profiles(agent_name)
                        )
                    """))
            
                # Performance history table
                session.execute(text("""
                    CREATE TABLE IF NOT EXISTS agent_performance_history (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            agent_name TEXT,
                            timestamp TIMESTAMP,
                            performance_score REAL,
                            consistency_score REAL,
                            overfitting_risk REAL,
                            learning_rate REAL,
                            regularization_strength REAL,
                            market_regime TEXT,
                            validation_type TEXT,
                            
                            FOREIGN KEY (agent_name) REFERENCES agent_profiles(agent_name)
                        )
                    """))
            
                # Agent baselines table
                session.execute(text("""
                    CREATE TABLE IF NOT EXISTS agent_baselines (
                            agent_name TEXT,
                            metric_name TEXT,
                            baseline_value REAL,
                            measurement_date DATE,
                            confidence_interval_lower REAL,
                            confidence_interval_upper REAL,
                            
                            PRIMARY KEY (agent_name, metric_name),
                            FOREIGN KEY (agent_name) REFERENCES agent_profiles(agent_name)
                        )
                    """))
            
                # Create indexes
                session.execute(text("CREATE INDEX IF NOT EXISTS idx_agent_optimization_timestamp ON optimization_results(agent_name, timestamp)"))
                session.execute(text("CREATE INDEX IF NOT EXISTS idx_performance_history_agent ON agent_performance_history(agent_name, timestamp)"))
                session.commit()
            self.logger.info("Agent optimization database initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize optimization database: {str(e)}")
            raise
    
    def _load_agent_profiles(self):
        """Load existing agent profiles from database."""
        try:
            # Using unified database manager instead of direct sqlite3
            with self.database.get_sync_session() as session:
                result = session.execute(text("SELECT * FROM agent_profiles"))
                rows = result.fetchall()
                columns = result.keys()
                
                for row in rows:
                    data = dict(zip(columns, row))
                    
                    # Parse JSON fields
                    ensemble_weights = json.loads(data['ensemble_weights']) if data['ensemble_weights'] else []
                    
                    profile = AgentBehaviorProfile(
                        agent_name=data['agent_name'],
                        base_learning_rate=data['base_learning_rate'],
                        current_learning_rate=data['current_learning_rate'],
                        learning_rate_schedule=LearningRateSchedule(data['learning_rate_schedule']),
                        regularization_type=RegularizationType(data['regularization_type']),
                        regularization_strength=data['regularization_strength'],
                        dropout_rate=data['dropout_rate'],
                        performance_decay_factor=data['performance_decay_factor'],
                        consistency_score=data['consistency_score'],
                        max_position_change_per_cycle=data['max_position_change_per_cycle'],
                        max_strategy_deviation=data['max_strategy_deviation'],
                        confidence_threshold=data['confidence_threshold'],
                        ensemble_size=data['ensemble_size'],
                        ensemble_weights=ensemble_weights,
                        ensemble_agreement_threshold=data['ensemble_agreement_threshold'],
                        volatility_adjustment_factor=data['volatility_adjustment_factor'],
                        regime_adaptation_speed=data['regime_adaptation_speed'],
                        overfitting_detection_sensitivity=data['overfitting_detection_sensitivity'],
                        last_optimization=datetime.fromisoformat(data['last_optimization']) if data['last_optimization'] else None,
                        optimization_cycles=data['optimization_cycles']
                    )
                    
                    self.agent_profiles[data['agent_name']] = profile
                
                self.logger.info(f"Loaded {len(self.agent_profiles)} agent profiles from database")
                
        except Exception as e:
            self.logger.warning(f"Failed to load agent profiles: {str(e)}")
    
    def register_agent(self, 
                      agent_name: str,
                      initial_config: Optional[Dict[str, Any]] = None) -> AgentBehaviorProfile:
        """
        Register a new agent for optimization or update existing profile.
        
        Args:
            agent_name: Name of the agent to register
            initial_config: Initial configuration parameters
            
        Returns:
            Agent behavior profile
        """
        try:
            config = initial_config or {}
            
            # Create or update agent profile
            if agent_name in self.agent_profiles:
                profile = self.agent_profiles[agent_name]
                self.logger.info(f"Updating existing profile for agent: {agent_name}")
            else:
                profile = AgentBehaviorProfile(
                    agent_name=agent_name,
                    base_learning_rate=self.config_manager.get_float('base_learning_rate', 0.01),
                    learning_rate_schedule=LearningRateSchedule(self.config_manager.get_str('learning_rate_schedule', 'ADAPTIVE')),
                    regularization_type=RegularizationType(self.config_manager.get_str('regularization_type', 'L2')),
                    regularization_strength=self.config_manager.get_float('regularization_strength', 0.01),
                    dropout_rate=self.config_manager.get_float('dropout_rate', 0.1),
                    performance_decay_factor=self.config_manager.get_float('performance_decay_factor', 0.95),
                    max_position_change_per_cycle=self.config_manager.get_float('max_position_change', 0.1),
                    max_strategy_deviation=self.config_manager.get_float('max_strategy_deviation', 0.2),
                    confidence_threshold=self.config_manager.get_float('confidence_threshold', 0.6),
                    ensemble_size=self.config_manager.get_int('ensemble_size', 5),
                    ensemble_agreement_threshold=self.config_manager.get_float('ensemble_agreement_threshold', 0.7),
                    volatility_adjustment_factor=self.config_manager.get_float('volatility_adjustment_factor', 1.0),
                    regime_adaptation_speed=self.config_manager.get_float('regime_adaptation_speed', 0.1),
                    overfitting_detection_sensitivity=self.config_manager.get_float('overfitting_detection_sensitivity', 0.8)
                )
                
                self.agent_profiles[agent_name] = profile
                self.logger.info(f"Registered new agent for optimization: {agent_name}")
            
            # Save to database
            self._save_agent_profile(profile)
            
            # Initialize baseline performance metrics
            self._initialize_agent_baselines(agent_name)
            
            return profile
            
        except Exception as e:
            self.logger.error(f"Failed to register agent {agent_name}: {str(e)}")
            raise OptimizationError(f"Agent registration failed: {str(e)}") from e
    
    def optimize_agent_behavior(self, 
                               agent_name: str,
                               force_optimization: bool = False) -> OptimizationResult:
        """
        Optimize agent behavior using anti-overfitting techniques.
        
        Args:
            agent_name: Name of the agent to optimize
            force_optimization: Force optimization even if recently done
            
        Returns:
            Optimization result with improvements and recommendations
            
        Raises:
            OptimizationError: If optimization fails
            InsufficientDataError: If insufficient data for optimization
        """
        try:
            if agent_name not in self.agent_profiles:
                raise ValueError(f"Agent {agent_name} not registered for optimization")
            
            profile = self.agent_profiles[agent_name]
            
            # Check if optimization is needed
            if not force_optimization and not self._should_optimize_agent(profile):
                self.logger.info(f"Agent {agent_name} does not require optimization at this time")
                return self._create_no_optimization_result(agent_name)
            
            self.logger.info(f"Starting behavior optimization for agent: {agent_name}")
            
            # Collect agent performance data
            performance_data = self._collect_agent_performance_data(agent_name)
            
            if len(performance_data) < self.min_data_points:
                raise InsufficientDataError(
                    f"Insufficient performance data for {agent_name}. "
                    f"Need {self.min_data_points}, got {len(performance_data)}"
                )
            
            # Analyze current performance and overfitting risk
            current_metrics = self._analyze_agent_performance(agent_name, performance_data)
            
            # Apply regularization optimizations
            regularization_result = self._optimize_regularization(profile, current_metrics)
            
            # Adjust learning rate based on performance
            learning_rate_result = self._optimize_learning_rate(profile, current_metrics)
            
            # Apply performance decay factor adjustments
            decay_result = self._optimize_performance_decay(profile, current_metrics)
            
            # Optimize ensemble settings
            ensemble_result = self._optimize_ensemble_settings(profile, current_metrics)
            
            # Validate optimization results
            validation_result = self._validate_optimization_changes(agent_name, profile, current_metrics)
            
            # Create optimization result
            optimization_result = self._create_optimization_result(
                agent_name, profile, current_metrics,
                regularization_result, learning_rate_result, 
                decay_result, ensemble_result, validation_result
            )
            
            # Apply successful optimizations
            if optimization_result.optimization_successful:
                self._apply_optimization_changes(profile, optimization_result)
                profile.last_optimization = datetime.utcnow()
                profile.optimization_cycles += 1
                
                # Save updated profile
                self._save_agent_profile(profile)
            
            # Record optimization result
            self._record_optimization_result(optimization_result)
            
            self.logger.info(
                f"Optimization complete for {agent_name}: "
                f"Success={optimization_result.optimization_successful}, "
                f"Performance improvement={optimization_result.performance_improvement:.3f}"
            )
            
            return optimization_result
            
        except Exception as e:
            self.logger.error(f"Optimization failed for agent {agent_name}: {str(e)}")
            raise OptimizationError(f"Agent optimization failed: {str(e)}") from e    
    def _should_optimize_agent(self, profile: AgentBehaviorProfile) -> bool:
        """Determine if agent needs optimization."""
        if profile.last_optimization is None:
            return True
        
        time_since_optimization = datetime.utcnow() - profile.last_optimization
        return time_since_optimization >= self.optimization_frequency
    
    def _collect_agent_performance_data(self, agent_name: str) -> List[Dict[str, Any]]:
        """Collect recent performance data for an agent."""
        cutoff_date = datetime.utcnow() - timedelta(days=self.performance_window_days)
        performance_data = []
        
        # Get completed trades where this agent was involved
        for trade_record in self.performance_tracker.completed_trades.values():
            if (trade_record.exit_time and 
                trade_record.exit_time >= cutoff_date and
                trade_record.generating_agent == agent_name):
                
                data_point = {
                    'timestamp': trade_record.exit_time,
                    'pnl': float(trade_record.pnl) if trade_record.pnl else 0.0,
                    'success': trade_record.pnl and trade_record.pnl > 0,
                    'confidence': trade_record.original_signal.confidence,
                    'validation_type': trade_record.validation_period_type.value,
                    'market_regime': trade_record.market_regime.value if trade_record.market_regime else 'UNKNOWN',
                    'grade': trade_record.grade.value if trade_record.grade else 'F',
                    'indicators_count': len(trade_record.indicators_used),
                    'timeframe': trade_record.timeframe,
                    'symbol': trade_record.original_signal.symbol
                }
                
                performance_data.append(data_point)
        
        return sorted(performance_data, key=lambda x: x['timestamp'])
    
    def _analyze_agent_performance(self, 
                                 agent_name: str, 
                                 performance_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze agent performance and calculate key metrics."""
        if not performance_data:
            return {}
        
        # Separate in-sample and out-of-sample data
        in_sample_data = [d for d in performance_data if d['validation_type'] == 'IN_SAMPLE']
        out_of_sample_data = [d for d in performance_data if d['validation_type'] == 'OUT_OF_SAMPLE']
        live_data = [d for d in performance_data if d['validation_type'] == 'LIVE_TRADING']
        
        # Calculate basic performance metrics
        total_trades = len(performance_data)
        winning_trades = sum(1 for d in performance_data if d['success'])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        # Calculate PnL metrics
        total_pnl = sum(d['pnl'] for d in performance_data)
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0.0
        pnl_std = np.std([d['pnl'] for d in performance_data]) if total_trades > 1 else 0.0
        
        # Calculate Sharpe ratio (risk-adjusted return)
        sharpe_ratio = avg_pnl / (pnl_std + 1e-8) if pnl_std > 0 else 0.0
        
        # Calculate overfitting metrics
        in_sample_win_rate = (sum(1 for d in in_sample_data if d['success']) / len(in_sample_data)) if in_sample_data else 0.0
        out_of_sample_win_rate = (sum(1 for d in out_of_sample_data if d['success']) / len(out_of_sample_data)) if out_of_sample_data else 0.0
        
        overfitting_risk = 0.0
        if in_sample_win_rate > 0:
            overfitting_risk = max(0.0, (in_sample_win_rate - out_of_sample_win_rate) / in_sample_win_rate)
        
        # Calculate consistency metrics
        monthly_returns = self._calculate_monthly_returns(performance_data)
        consistency_score = 1.0 - (np.std(monthly_returns) if len(monthly_returns) > 1 else 0.0)
        
        # Calculate confidence correlation
        confidences = [d['confidence'] for d in performance_data]
        successes = [1.0 if d['success'] else 0.0 for d in performance_data]
        confidence_correlation = 0.0
        
        if len(set(confidences)) > 1 and len(set(successes)) > 1:
            correlation_matrix = np.corrcoef(confidences, successes)
            confidence_correlation = correlation_matrix[0, 1] if not np.isnan(correlation_matrix[0, 1]) else 0.0
        
        # Calculate complexity score (based on indicator usage)
        avg_indicators = np.mean([d['indicators_count'] for d in performance_data])
        complexity_score = min(1.0, avg_indicators / 20.0)  # Normalize to 0-1
        
        # Recent performance trend
        recent_performance_trend = self._calculate_performance_trend(performance_data)
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_pnl': avg_pnl,
            'total_pnl': total_pnl,
            'pnl_volatility': pnl_std,
            'sharpe_ratio': sharpe_ratio,
            'in_sample_win_rate': in_sample_win_rate,
            'out_of_sample_win_rate': out_of_sample_win_rate,
            'overfitting_risk': overfitting_risk,
            'consistency_score': consistency_score,
            'confidence_correlation': confidence_correlation,
            'complexity_score': complexity_score,
            'recent_trend': recent_performance_trend,
            'live_performance': (sum(1 for d in live_data if d['success']) / len(live_data)) if live_data else 0.0
        }
    
    def _calculate_monthly_returns(self, performance_data: List[Dict[str, Any]]) -> List[float]:
        """Calculate monthly returns for consistency analysis."""
        monthly_data = defaultdict(list)
        
        for data_point in performance_data:
            month_key = data_point['timestamp'].strftime('%Y-%m')
            monthly_data[month_key].append(data_point['pnl'])
        
        monthly_returns = []
        for month, pnls in monthly_data.items():
            monthly_return = sum(pnls)
            monthly_returns.append(monthly_return)
        
        return monthly_returns
    
    def _calculate_performance_trend(self, performance_data: List[Dict[str, Any]]) -> float:
        """Calculate performance trend over time."""
        if len(performance_data) < 10:
            return 0.0
        
        # Split data into time-based windows
        window_size = max(5, len(performance_data) // 4)
        windows = []
        
        for i in range(0, len(performance_data), window_size):
            window = performance_data[i:i + window_size]
            if len(window) >= 3:  # Minimum window size
                win_rate = sum(1 for d in window if d['success']) / len(window)
                windows.append(win_rate)
        
        if len(windows) < 2:
            return 0.0
        
        # Calculate linear trend
        x = np.arange(len(windows))
        slope, _ = np.polyfit(x, windows, 1)
        
        return float(slope)
    
    def _optimize_regularization(self, 
                               profile: AgentBehaviorProfile, 
                               metrics: Dict[str, float]) -> Dict[str, Any]:
        """Optimize regularization parameters to prevent overfitting."""
        current_strength = profile.regularization_strength
        overfitting_risk = metrics.get('overfitting_risk', 0.0)
        complexity_score = metrics.get('complexity_score', 0.5)
        
        # Calculate target regularization strength
        if overfitting_risk > self.overfitting_threshold:
            # Increase regularization to combat overfitting
            strength_multiplier = 1.0 + (overfitting_risk * 2.0)
        elif overfitting_risk < 0.1 and metrics.get('out_of_sample_win_rate', 0.0) > 0.6:
            # Reduce regularization for well-performing, stable agents
            strength_multiplier = 0.8
        else:
            # Adjust based on complexity
            strength_multiplier = 0.5 + complexity_score
        
        new_strength = current_strength * strength_multiplier
        new_strength = max(0.001, min(0.1, new_strength))  # Clamp to reasonable range
        
        # Adjust dropout rate based on overfitting risk
        if overfitting_risk > 0.3:
            new_dropout_rate = min(0.3, profile.dropout_rate * 1.5)
        elif overfitting_risk < 0.1:
            new_dropout_rate = max(0.05, profile.dropout_rate * 0.8)
        else:
            new_dropout_rate = profile.dropout_rate
        
        return {
            'old_strength': current_strength,
            'new_strength': new_strength,
            'old_dropout_rate': profile.dropout_rate,
            'new_dropout_rate': new_dropout_rate,
            'strength_change': abs(new_strength - current_strength),
            'overfitting_risk': overfitting_risk,
            'recommended': overfitting_risk > 0.2 or complexity_score > 0.7
        }
    
    def _optimize_learning_rate(self, 
                              profile: AgentBehaviorProfile, 
                              metrics: Dict[str, float]) -> Dict[str, Any]:
        """Optimize learning rate based on performance and market conditions."""
        current_lr = profile.current_learning_rate
        base_lr = profile.base_learning_rate
        
        # Performance-based adjustment
        recent_trend = metrics.get('recent_trend', 0.0)
        consistency_score = metrics.get('consistency_score', 0.5)
        overfitting_risk = metrics.get('overfitting_risk', 0.0)
        
        # Calculate learning rate adjustment factors
        trend_factor = 1.0
        if recent_trend > 0.1:  # Improving performance
            trend_factor = 1.1
        elif recent_trend < -0.1:  # Declining performance
            trend_factor = 0.9
        
        consistency_factor = 0.5 + consistency_score  # Range: 0.5 to 1.5
        overfitting_factor = 1.0 - overfitting_risk  # Reduce LR if overfitting
        
        # Market volatility adjustment (if available)
        volatility_factor = profile.volatility_adjustment_factor
        
        # Calculate new learning rate
        lr_multiplier = trend_factor * consistency_factor * overfitting_factor * volatility_factor
        new_lr = current_lr * lr_multiplier
        
        # Apply constraints
        new_lr = max(self.min_learning_rate, min(self.max_learning_rate, new_lr))
        
        # Ensure gradual changes (constrained learning rate)
        max_change = base_lr * 0.5  # Maximum 50% change from base
        if abs(new_lr - current_lr) > max_change:
            if new_lr > current_lr:
                new_lr = current_lr + max_change
            else:
                new_lr = current_lr - max_change
        
        return {
            'old_learning_rate': current_lr,
            'new_learning_rate': new_lr,
            'trend_factor': trend_factor,
            'consistency_factor': consistency_factor,
            'overfitting_factor': overfitting_factor,
            'volatility_factor': volatility_factor,
            'rate_change': abs(new_lr - current_lr),
            'recommended': abs(new_lr - current_lr) > current_lr * 0.1
        }
    
    def _optimize_performance_decay(self, 
                                  profile: AgentBehaviorProfile, 
                                  metrics: Dict[str, float]) -> Dict[str, Any]:
        """Optimize performance decay factor for recent data weighting."""
        current_decay = profile.performance_decay_factor
        recent_trend = metrics.get('recent_trend', 0.0)
        consistency_score = metrics.get('consistency_score', 0.5)
        
        # Adjust decay factor based on performance characteristics
        if recent_trend > 0.1 and consistency_score > 0.7:
            # Good recent performance and high consistency - weight recent data more
            new_decay = min(0.98, current_decay + 0.02)
        elif recent_trend < -0.1 or consistency_score < 0.3:
            # Poor recent performance or low consistency - weight historical data more
            new_decay = max(0.85, current_decay - 0.05)
        else:
            # Stable performance - maintain current decay
            new_decay = current_decay
        
        return {
            'old_decay_factor': current_decay,
            'new_decay_factor': new_decay,
            'decay_change': abs(new_decay - current_decay),
            'recent_trend': recent_trend,
            'consistency_score': consistency_score,
            'recommended': abs(new_decay - current_decay) > 0.01
        }
    
    def _optimize_ensemble_settings(self, 
                                  profile: AgentBehaviorProfile, 
                                  metrics: Dict[str, float]) -> Dict[str, Any]:
        """Optimize ensemble settings for robust decision making."""
        current_size = profile.ensemble_size
        current_threshold = profile.ensemble_agreement_threshold
        confidence_correlation = metrics.get('confidence_correlation', 0.0)
        consistency_score = metrics.get('consistency_score', 0.5)
        
        # Adjust ensemble size based on performance characteristics
        if consistency_score < 0.4 or confidence_correlation < 0.3:
            # Low consistency - increase ensemble size for stability
            new_size = min(10, current_size + 1)
        elif consistency_score > 0.8 and confidence_correlation > 0.7:
            # High consistency - can reduce ensemble size
            new_size = max(3, current_size - 1)
        else:
            new_size = current_size
        
        # Adjust agreement threshold based on performance
        if metrics.get('overfitting_risk', 0.0) > 0.3:
            # High overfitting risk - require higher agreement
            new_threshold = min(0.9, current_threshold + 0.1)
        elif consistency_score > 0.7:
            # Good consistency - can lower threshold
            new_threshold = max(0.5, current_threshold - 0.05)
        else:
            new_threshold = current_threshold
        
        # Update ensemble weights (uniform for simplicity)
        new_weights = [1.0 / new_size] * new_size
        
        return {
            'old_ensemble_size': current_size,
            'new_ensemble_size': new_size,
            'old_agreement_threshold': current_threshold,
            'new_agreement_threshold': new_threshold,
            'new_ensemble_weights': new_weights,
            'confidence_correlation': confidence_correlation,
            'consistency_score': consistency_score,
            'recommended': new_size != current_size or abs(new_threshold - current_threshold) > 0.05
        }
    
    def _validate_optimization_changes(self, 
                                     agent_name: str,
                                     profile: AgentBehaviorProfile, 
                                     metrics: Dict[str, float]) -> Dict[str, float]:
        """Validate optimization changes using walk-forward analysis."""
        try:
            # Create a test scenario with the proposed changes
            # This is a simplified validation - in practice, you'd run full backtests
            
            in_sample_performance = metrics.get('in_sample_win_rate', 0.0)
            out_of_sample_performance = metrics.get('out_of_sample_win_rate', 0.0)
            
            # Calculate performance degradation
            if in_sample_performance > 0:
                performance_degradation = (in_sample_performance - out_of_sample_performance) / in_sample_performance
            else:
                performance_degradation = 0.0
            
            return {
                'in_sample_performance': in_sample_performance,
                'out_of_sample_performance': out_of_sample_performance,
                'performance_degradation': performance_degradation,
                'validation_confidence': 0.8 if out_of_sample_performance > 0.5 else 0.5
            }
            
        except Exception as e:
            self.logger.warning(f"Validation failed for {agent_name}: {str(e)}")
            return {
                'in_sample_performance': 0.0,
                'out_of_sample_performance': 0.0,
                'performance_degradation': 1.0,
                'validation_confidence': 0.0
            }    
    def _create_optimization_result(self,
                                  agent_name: str,
                                  profile: AgentBehaviorProfile,
                                  metrics: Dict[str, float],
                                  regularization_result: Dict[str, Any],
                                  learning_rate_result: Dict[str, Any],
                                  decay_result: Dict[str, Any],
                                  ensemble_result: Dict[str, Any],
                                  validation_result: Dict[str, float]) -> OptimizationResult:
        """Create comprehensive optimization result."""
        
        # Calculate overall performance improvement
        performance_improvement = 0.0
        improvement_factors = []
        
        if regularization_result['recommended']:
            improvement_factors.append(0.1)  # Expected improvement from regularization
        
        if learning_rate_result['recommended']:
            improvement_factors.append(0.05)  # Expected improvement from LR optimization
        
        if decay_result['recommended']:
            improvement_factors.append(0.03)  # Expected improvement from decay optimization
        
        if ensemble_result['recommended']:
            improvement_factors.append(0.08)  # Expected improvement from ensemble optimization
        
        performance_improvement = sum(improvement_factors)
        
        # Calculate overfitting risk reduction
        overfitting_risk_reduction = max(0.0, regularization_result['overfitting_risk'] * 0.5)
        
        # Calculate consistency improvement
        consistency_improvement = 0.02 if ensemble_result['recommended'] else 0.0
        
        # Determine optimization success
        optimization_successful = (
            validation_result['out_of_sample_performance'] > 0.4 and
            validation_result['performance_degradation'] < 0.4 and
            performance_improvement > 0.0
        )
        
        # Generate recommendations
        recommendations = []
        risk_warnings = []
        
        if regularization_result['overfitting_risk'] > 0.3:
            recommendations.append("INCREASE_REGULARIZATION")
            recommendations.append("MONITOR_OUT_OF_SAMPLE_PERFORMANCE")
        
        if metrics.get('consistency_score', 0.5) < 0.4:
            recommendations.append("INCREASE_ENSEMBLE_SIZE")
            recommendations.append("REDUCE_LEARNING_RATE")
        
        if validation_result['performance_degradation'] > 0.5:
            risk_warnings.append("HIGH_OVERFITTING_RISK")
            risk_warnings.append("CONSIDER_SIMPLIFYING_STRATEGY")
        
        if metrics.get('recent_trend', 0.0) < -0.2:
            risk_warnings.append("DECLINING_PERFORMANCE_TREND")
        
        # Determine if intervention is required
        requires_intervention = (
            validation_result['performance_degradation'] > 0.6 or
            metrics.get('overfitting_risk', 0.0) > 0.5 or
            metrics.get('out_of_sample_win_rate', 0.0) < 0.3
        )
        
        return OptimizationResult(
            agent_name=agent_name,
            timestamp=datetime.utcnow(),
            old_learning_rate=learning_rate_result['old_learning_rate'],
            new_learning_rate=learning_rate_result['new_learning_rate'],
            regularization_adjustment=regularization_result['strength_change'],
            performance_improvement=performance_improvement,
            overfitting_risk_reduction=overfitting_risk_reduction,
            consistency_improvement=consistency_improvement,
            strategy_adjustments={
                'regularization_strength': regularization_result['new_strength'],
                'dropout_rate': regularization_result['new_dropout_rate'],
                'learning_rate': learning_rate_result['new_learning_rate'],
                'decay_factor': decay_result['new_decay_factor'],
                'ensemble_size': ensemble_result['new_ensemble_size'],
                'agreement_threshold': ensemble_result['new_agreement_threshold']
            },
            confidence_adjustments={
                'confidence_correlation': metrics.get('confidence_correlation', 0.0),
                'validation_confidence': validation_result['validation_confidence']
            },
            in_sample_performance=validation_result['in_sample_performance'],
            out_of_sample_performance=validation_result['out_of_sample_performance'],
            performance_degradation=validation_result['performance_degradation'],
            recommendations=recommendations,
            risk_warnings=risk_warnings,
            optimization_successful=optimization_successful,
            requires_intervention=requires_intervention
        )
    
    def _create_no_optimization_result(self, agent_name: str) -> OptimizationResult:
        """Create result for when no optimization is needed."""
        profile = self.agent_profiles[agent_name]
        
        return OptimizationResult(
            agent_name=agent_name,
            timestamp=datetime.utcnow(),
            old_learning_rate=profile.current_learning_rate,
            new_learning_rate=profile.current_learning_rate,
            regularization_adjustment=0.0,
            performance_improvement=0.0,
            overfitting_risk_reduction=0.0,
            consistency_improvement=0.0,
            strategy_adjustments={},
            confidence_adjustments={},
            in_sample_performance=0.0,
            out_of_sample_performance=0.0,
            performance_degradation=0.0,
            recommendations=["NO_OPTIMIZATION_NEEDED"],
            risk_warnings=[],
            optimization_successful=True,
            requires_intervention=False
        )
    
    def _apply_optimization_changes(self, 
                                  profile: AgentBehaviorProfile, 
                                  result: OptimizationResult):
        """Apply successful optimization changes to agent profile."""
        adjustments = result.strategy_adjustments
        
        if 'learning_rate' in adjustments:
            profile.current_learning_rate = adjustments['learning_rate']
        
        if 'regularization_strength' in adjustments:
            profile.regularization_strength = adjustments['regularization_strength']
        
        if 'dropout_rate' in adjustments:
            profile.dropout_rate = adjustments['dropout_rate']
        
        if 'decay_factor' in adjustments:
            profile.performance_decay_factor = adjustments['decay_factor']
        
        if 'ensemble_size' in adjustments:
            profile.ensemble_size = adjustments['ensemble_size']
            # Update ensemble weights
            profile.ensemble_weights = [1.0 / profile.ensemble_size] * profile.ensemble_size
        
        if 'agreement_threshold' in adjustments:
            profile.ensemble_agreement_threshold = adjustments['agreement_threshold']
        
        # Update consistency score
        profile.consistency_score = max(0.0, profile.consistency_score + result.consistency_improvement)
    
    def _save_agent_profile(self, profile: AgentBehaviorProfile):
        """Save agent profile to database."""
        try:
            # Using unified database manager instead of direct sqlite3
            with self.database.get_sync_session() as session:
                session.execute(text("""
                    INSERT OR REPLACE INTO agent_profiles (
                            agent_name, base_learning_rate, current_learning_rate,
                            learning_rate_schedule, regularization_type, regularization_strength,
                            dropout_rate, performance_decay_factor, consistency_score,
                            max_position_change_per_cycle, max_strategy_deviation, confidence_threshold,
                            ensemble_size, ensemble_weights, ensemble_agreement_threshold,
                            volatility_adjustment_factor, regime_adaptation_speed,
                            overfitting_detection_sensitivity, last_optimization, optimization_cycles,
                            updated_at
                        ) VALUES (:agent_name, :base_learning_rate, :current_learning_rate, :learning_rate_schedule, :regularization_type, :regularization_strength, :dropout_rate, :performance_decay_factor, :consistency_score, :max_position_change_per_cycle, :max_strategy_deviation, :confidence_threshold, :ensemble_size, :ensemble_weights, :ensemble_agreement_threshold, :volatility_adjustment_factor, :regime_adaptation_speed, :overfitting_detection_sensitivity, :last_optimization, :optimization_cycles, :updated_at)
                    """), {
                        'agent_name': profile.agent_name,
                        'base_learning_rate': profile.base_learning_rate,
                        'current_learning_rate': profile.current_learning_rate,
                        'learning_rate_schedule': profile.learning_rate_schedule.value,
                        'regularization_type': profile.regularization_type.value,
                        'regularization_strength': profile.regularization_strength,
                        'dropout_rate': profile.dropout_rate,
                        'performance_decay_factor': profile.performance_decay_factor,
                        'consistency_score': profile.consistency_score,
                        'max_position_change_per_cycle': profile.max_position_change_per_cycle,
                        'max_strategy_deviation': profile.max_strategy_deviation,
                        'confidence_threshold': profile.confidence_threshold,
                        'ensemble_size': profile.ensemble_size,
                        'ensemble_weights': json.dumps(profile.ensemble_weights),
                        'ensemble_agreement_threshold': profile.ensemble_agreement_threshold,
                        'volatility_adjustment_factor': profile.volatility_adjustment_factor,
                        'regime_adaptation_speed': profile.regime_adaptation_speed,
                        'overfitting_detection_sensitivity': profile.overfitting_detection_sensitivity,
                        'last_optimization': profile.last_optimization.isoformat() if profile.last_optimization else None,
                        'optimization_cycles': profile.optimization_cycles,
                        'updated_at': datetime.utcnow().isoformat()
                    })
                session.commit()
        except Exception as e:
            self.logger.error(f"Failed to save agent profile {profile.agent_name}: {str(e)}")
    
    def _record_optimization_result(self, result: OptimizationResult):
        """Record optimization result to database."""
        try:
            # Using unified database manager instead of direct sqlite3
            with self.database.get_sync_session() as session:
                session.execute(text("""
                    INSERT INTO optimization_results (
                            agent_name, timestamp, old_learning_rate, new_learning_rate,
                            regularization_adjustment, performance_improvement, overfitting_risk_reduction,
                            consistency_improvement, strategy_adjustments, confidence_adjustments,
                            in_sample_performance, out_of_sample_performance, performance_degradation,
                            recommendations, risk_warnings, optimization_successful, requires_intervention
                        ) VALUES (:agent_name, :timestamp, :old_learning_rate, :new_learning_rate, :regularization_adjustment, :performance_improvement, :overfitting_risk_reduction, :consistency_improvement, :strategy_adjustments, :confidence_adjustments, :in_sample_performance, :out_of_sample_performance, :performance_degradation, :recommendations, :risk_warnings, :optimization_successful, :requires_intervention)
                    """), {
                        'agent_name': result.agent_name,
                        'timestamp': result.timestamp.isoformat(),
                        'old_learning_rate': result.old_learning_rate,
                        'new_learning_rate': result.new_learning_rate,
                        'regularization_adjustment': result.regularization_adjustment,
                        'performance_improvement': result.performance_improvement,
                        'overfitting_risk_reduction': result.overfitting_risk_reduction,
                        'consistency_improvement': result.consistency_improvement,
                        'strategy_adjustments': json.dumps(result.strategy_adjustments),
                        'confidence_adjustments': json.dumps(result.confidence_adjustments),
                        'in_sample_performance': result.in_sample_performance,
                        'out_of_sample_performance': result.out_of_sample_performance,
                        'performance_degradation': result.performance_degradation,
                        'recommendations': json.dumps(result.recommendations),
                        'risk_warnings': json.dumps(result.risk_warnings),
                        'optimization_successful': result.optimization_successful,
                        'requires_intervention': result.requires_intervention
                    })
                session.commit()
                
                # Add to history
            self.optimization_history[result.agent_name].append(result)
            
        except Exception as e:
            self.logger.error(f"Failed to record optimization result: {str(e)}")
    
    def _initialize_agent_baselines(self, agent_name: str):
        """Initialize baseline performance metrics for an agent."""
        try:
            # Get initial performance data
            performance_data = self._collect_agent_performance_data(agent_name)
            
            if len(performance_data) >= 10:  # Minimum data for baseline
                metrics = self._analyze_agent_performance(agent_name, performance_data)
                
                baselines = {
                    'win_rate': metrics.get('win_rate', 0.0),
                    'avg_pnl': metrics.get('avg_pnl', 0.0),
                    'sharpe_ratio': metrics.get('sharpe_ratio', 0.0),
                    'consistency_score': metrics.get('consistency_score', 0.0),
                    'overfitting_risk': metrics.get('overfitting_risk', 0.0)
                }
                
                self.agent_baselines[agent_name] = baselines
                
                # Save to database
                # Using unified database manager instead of direct sqlite3
                with self.database.get_sync_session() as session:
                    for metric_name, baseline_value in baselines.items():
                        session.execute(text("""
                            INSERT OR REPLACE INTO agent_baselines (
                                agent_name, metric_name, baseline_value, measurement_date,
                                confidence_interval_lower, confidence_interval_upper
                            ) VALUES (:agent_name, :metric_name, :baseline_value, :measurement_date, :confidence_interval_lower, :confidence_interval_upper)
                        """), {
                            'agent_name': agent_name,
                            'metric_name': metric_name,
                            'baseline_value': baseline_value,
                            'measurement_date': datetime.utcnow().date(),
                            'confidence_interval_lower': baseline_value * 0.8,  # Simple confidence interval
                            'confidence_interval_upper': baseline_value * 1.2
                        })
                    session.commit()
                
                self.logger.info(f"Initialized baselines for agent {agent_name}")
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize baselines for {agent_name}: {str(e)}")
    
    def optimize_all_agents(self, force_optimization: bool = False) -> Dict[str, OptimizationResult]:
        """
        Optimize behavior for all registered agents.
        
        Args:
            force_optimization: Force optimization for all agents
            
        Returns:
            Dictionary mapping agent names to optimization results
        """
        results = {}
        
        self.logger.info(f"Starting optimization for {len(self.agent_profiles)} agents")
        
        for agent_name in self.agent_profiles.keys():
            try:
                result = self.optimize_agent_behavior(agent_name, force_optimization)
                results[agent_name] = result
            except Exception as e:
                self.logger.error(f"Failed to optimize agent {agent_name}: {str(e)}")
                # Create error result
                results[agent_name] = OptimizationResult(
                    agent_name=agent_name,
                    timestamp=datetime.utcnow(),
                    old_learning_rate=0.0,
                    new_learning_rate=0.0,
                    regularization_adjustment=0.0,
                    performance_improvement=0.0,
                    overfitting_risk_reduction=0.0,
                    consistency_improvement=0.0,
                    strategy_adjustments={},
                    confidence_adjustments={},
                    in_sample_performance=0.0,
                    out_of_sample_performance=0.0,
                    performance_degradation=1.0,
                    recommendations=[],
                    risk_warnings=[f"OPTIMIZATION_ERROR: {str(e)}"],
                    optimization_successful=False,
                    requires_intervention=True
                )
        
        # Log summary
        successful_optimizations = sum(1 for r in results.values() if r.optimization_successful)
        requiring_intervention = sum(1 for r in results.values() if r.requires_intervention)
        
        self.logger.info(
            f"Optimization complete: {successful_optimizations}/{len(results)} successful, "
            f"{requiring_intervention} require intervention"
        )
        
        return results
    
    def get_agent_optimization_summary(self, agent_name: str) -> Dict[str, Any]:
        """Get optimization summary for an agent."""
        if agent_name not in self.agent_profiles:
            return {'error': f'Agent {agent_name} not found'}
        
        profile = self.agent_profiles[agent_name]
        history = self.optimization_history.get(agent_name, [])
        baselines = self.agent_baselines.get(agent_name, {})
        
        # Calculate recent performance metrics
        recent_results = history[-5:] if history else []
        avg_improvement = np.mean([r.performance_improvement for r in recent_results]) if recent_results else 0.0
        
        return {
            'agent_name': agent_name,
            'profile': {
                'current_learning_rate': profile.current_learning_rate,
                'regularization_strength': profile.regularization_strength,
                'consistency_score': profile.consistency_score,
                'last_optimization': profile.last_optimization.isoformat() if profile.last_optimization else None,
                'optimization_cycles': profile.optimization_cycles
            },
            'recent_performance': {
                'average_improvement': avg_improvement,
                'optimizations_count': len(recent_results),
                'successful_optimizations': sum(1 for r in recent_results if r.optimization_successful),
                'interventions_required': sum(1 for r in recent_results if r.requires_intervention)
            },
            'baselines': baselines,
            'latest_result': history[-1].__dict__ if history else None
        }
    
    def get_system_optimization_overview(self) -> Dict[str, Any]:
        """Get system-wide optimization overview."""
        total_agents = len(self.agent_profiles)
        
        if total_agents == 0:
            return {
                'total_agents': 0,
                'optimization_summary': 'No agents registered',
                'recommendations': ['REGISTER_AGENTS_FOR_OPTIMIZATION']
            }
        
        # Calculate system-wide metrics
        recent_optimizations = []
        for agent_history in self.optimization_history.values():
            if agent_history:
                recent_optimizations.extend(agent_history[-3:])  # Last 3 per agent
        
        successful_rate = (sum(1 for r in recent_optimizations if r.optimization_successful) / 
                          len(recent_optimizations)) if recent_optimizations else 0.0
        
        avg_improvement = np.mean([r.performance_improvement for r in recent_optimizations]) if recent_optimizations else 0.0
        
        high_risk_agents = sum(1 for r in recent_optimizations if r.requires_intervention)
        
        # System recommendations
        recommendations = []
        if successful_rate < 0.6:
            recommendations.append('LOW_OPTIMIZATION_SUCCESS_RATE')
        if avg_improvement < 0.02:
            recommendations.append('MINIMAL_PERFORMANCE_IMPROVEMENTS')
        if high_risk_agents > total_agents * 0.3:
            recommendations.append('MULTIPLE_AGENTS_REQUIRE_INTERVENTION')
        
        if not recommendations:
            recommendations.append('SYSTEM_OPERATING_NORMALLY')
        
        return {
            'total_agents': total_agents,
            'optimization_summary': {
                'recent_optimizations': len(recent_optimizations),
                'successful_rate': successful_rate,
                'average_improvement': avg_improvement,
                'high_risk_agents': high_risk_agents
            },
            'agent_profiles_summary': {
                agent_name: {
                    'learning_rate': profile.current_learning_rate,
                    'regularization': profile.regularization_strength,
                    'last_optimization': profile.last_optimization
                }
                for agent_name, profile in self.agent_profiles.items()
            },
            'recommendations': recommendations,
            'last_updated': datetime.utcnow().isoformat()
        }