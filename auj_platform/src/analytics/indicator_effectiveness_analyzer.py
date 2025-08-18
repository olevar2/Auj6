"""
Indicator Effectiveness Analyzer for AUJ Platform.

This component is the skeptical guardian of indicator selection, using the WalkForwardValidator
as its core to score indicators based on composite metrics that heavily favor stable,
positive performance on out-of-sample data.

The analyzer is designed to be skeptical of indicators that perform well in training
but fail in validation, ensuring only robust indicators make it to the "Elite" lists.
"""

import asyncio
import time
import threading
import sqlite3
import contextlib
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import List, Optional, Dict, Any, Union, AsyncContextManager, ContextManager, Tuple
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import statistics
import json
from pathlib import Path
from sqlalchemy import text

import asyncpg
from sqlalchemy import create_engine, MetaData, event, Table, Column, Integer, String, DateTime, Float, Boolean
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool, StaticPool
from sqlalchemy.exc import SQLAlchemyError, DisconnectionError
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
import logging
import pandas as pd
import numpy as np

from ..core.data_contracts import (
    IndicatorResult, GradedDeal, TradeSignal, MarketRegime,
    ValidationResult, DealGrade
)
from ..core.exceptions import (
    ValidationError, InsufficientDataError,
    OptimizationError, AUJPlatformError
)
from ..core.unified_database_manager import get_unified_database, get_unified_database_sync
from ..validation.walk_forward_validator import WalkForwardValidator, ValidationPeriodType
from .performance_tracker import PerformanceTracker


class IndicatorQualityRating(str, Enum):
    """Quality ratings for indicators based on validation performance."""
    ELITE = "ELITE"                    # Top-tier, proven robust
    VALIDATED = "VALIDATED"            # Good performance, validated
    PROMISING = "PROMISING"            # Shows potential, needs more data
    QUESTIONABLE = "QUESTIONABLE"      # Mixed results, use with caution
    UNRELIABLE = "UNRELIABLE"         # Poor out-of-sample performance
    OVERFITTED = "OVERFITTED"         # Clear signs of overfitting


class IndicatorRiskProfile(str, Enum):
    """Risk profiles for indicator usage."""
    LOW_RISK = "LOW_RISK"           # Consistently stable
    MODERATE_RISK = "MODERATE_RISK" # Generally stable with some variance
    HIGH_RISK = "HIGH_RISK"         # High variance, use sparingly
    VERY_HIGH_RISK = "VERY_HIGH_RISK" # Unstable, avoid


@dataclass
class IndicatorPerformanceMetrics:
    """Comprehensive performance metrics for an indicator."""
    indicator_name: str
    total_usage_count: int

    # Basic effectiveness metrics
    success_rate_in_sample: float
    success_rate_out_of_sample: float
    success_rate_live: float

    # Advanced performance metrics
    avg_contribution_to_profit: Decimal
    profit_consistency_score: float
    risk_adjusted_effectiveness: float

    # Validation-specific metrics
    overfitting_score: float
    robustness_score: float
    stability_across_regimes: float

    # Quality assessment
    quality_rating: IndicatorQualityRating
    risk_profile: IndicatorRiskProfile
    confidence_score: float

    # Usage recommendations
    recommended_for_live: bool
    recommended_weight: float
    usage_constraints: List[str]

    # Temporal metrics
    recent_performance_trend: str
    performance_decay_rate: float

    # Market regime effectiveness
    regime_performance: Dict[str, float]

    # Metadata
    last_updated: datetime
    analysis_period_days: int
    sample_size_adequacy: str


class IndicatorEffectivenessAnalyzer:
    """
    Advanced Indicator Effectiveness Analyzer with Anti-Overfitting Focus.

    This component serves as the skeptical guardian of indicator selection,
    rigorously evaluating each indicator's true effectiveness using walk-forward
    validation and out-of-sample performance analysis.

    Key Philosophy:
    - Skeptical by design: Indicators are guilty of overfitting until proven innocent
    - Out-of-sample performance is weighted heavily over in-sample results
    - Consistency and stability are more important than peak performance
    - Recent performance trends are considered for adaptive recommendations
    """

    def __init__(self,
                 walk_forward_validator: WalkForwardValidator,
                 performance_tracker: PerformanceTracker,
                 database_path: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None):
        # Note: This class now requires config_manager parameter in __init__
        # self.config_manager = config_manager or UnifiedConfigManager()
        """
        Initialize the Indicator Effectiveness Analyzer.

        Args:
            walk_forward_validator: WalkForwardValidator instance for validation
            performance_tracker: PerformanceTracker for trade data
            database_path: Path to database for persistence
            config: Configuration parameters
        """
        self.walk_forward_validator = walk_forward_validator
        self.performance_tracker = performance_tracker
        # Use unified database abstraction instead of direct path
        self.database = get_unified_database_sync()
        self.config = config or {}

        # Analysis parameters - use proper config access
        if hasattr(self.config, 'get_int'):
            # If config is a config manager object
            self.min_usage_count = self.config.get_int('min_usage_count', 20)
            self.analysis_window_days = self.config.get_int('analysis_window_days', 90)
            self.overfitting_threshold = self.config.get_float('overfitting_threshold', 0.25)
            self.stability_threshold = self.config.get_float('stability_threshold', 0.7)
        else:
            # If config is a dictionary
            self.min_usage_count = self.config.get('min_usage_count', 20)
            self.analysis_window_days = self.config.get('analysis_window_days', 90)
            self.overfitting_threshold = self.config.get('overfitting_threshold', 0.25)
            self.stability_threshold = self.config.get('stability_threshold', 0.7)

        # Weighting factors for composite scoring
        self.weights = {
            'out_of_sample_performance': 0.4,
            'consistency': 0.25,
            'stability': 0.2,
            'recent_trend': 0.1,
            'regime_adaptability': 0.05
        }

        # Cache for performance analysis
        self.indicator_cache: Dict[str, IndicatorPerformanceMetrics] = {}
        self.cache_expiry: Dict[str, datetime] = {}
        self.cache_duration = timedelta(minutes=15)

        # Elite indicator tracking
        self.elite_indicators: Dict[str, List[str]] = {}  # By market regime
        self.last_elite_update: Optional[datetime] = None

        self.logger = logging.getLogger(__name__)
        self._initialize_database()

        self.logger.info("IndicatorEffectivenessAnalyzer initialized with unified database abstraction")

    def _initialize_database(self):
        """Initialize database schemas using unified database abstraction."""
        try:
            # Use unified database abstraction for schema creation
            with self.database.get_sync_session() as session:
                # Main indicator performance table
                session.execute(text("""
                    CREATE TABLE IF NOT EXISTS indicator_effectiveness (
                        indicator_name TEXT,
                        analysis_date DATE,
                        market_regime TEXT,

                        total_usage_count INTEGER,
                        success_rate_in_sample REAL,
                        success_rate_out_of_sample REAL,
                        success_rate_live REAL,

                        avg_contribution_to_profit DECIMAL,
                        profit_consistency_score REAL,
                        risk_adjusted_effectiveness REAL,

                        overfitting_score REAL,
                        robustness_score REAL,
                        stability_across_regimes REAL,

                        quality_rating TEXT,
                        risk_profile TEXT,
                        confidence_score REAL,

                        recommended_for_live BOOLEAN,
                        recommended_weight REAL,
                        usage_constraints TEXT,

                        recent_performance_trend TEXT,
                        performance_decay_rate REAL,

                        regime_performance TEXT,
                        analysis_period_days INTEGER,
                        sample_size_adequacy TEXT,

                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                        PRIMARY KEY (indicator_name, analysis_date, market_regime)
                    )
                """))

                # Elite indicator sets table
                session.execute(text("""
                    CREATE TABLE IF NOT EXISTS elite_indicator_sets (
                        regime TEXT,
                        indicator_name TEXT,
                        effectiveness_score REAL,
                        selection_date TIMESTAMP,
                        last_validation TIMESTAMP,
                        performance_metrics TEXT,

                        PRIMARY KEY (regime, indicator_name)
                    )
                """))

                # Historical effectiveness tracking
                session.execute(text("""
                    CREATE TABLE IF NOT EXISTS indicator_performance_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        indicator_name TEXT,
                        timestamp TIMESTAMP,
                        usage_context TEXT,
                        success BOOLEAN,
                        contribution_to_pnl DECIMAL,
                        market_regime TEXT,
                        validation_type TEXT,
                        trade_id TEXT,

                        FOREIGN KEY (trade_id) REFERENCES trades(trade_id)
                    )
                """))

                # Create indexes
                session.execute(text("CREATE INDEX IF NOT EXISTS idx_indicator_name ON indicator_effectiveness(indicator_name)"))
                session.execute(text("CREATE INDEX IF NOT EXISTS idx_analysis_date ON indicator_effectiveness(analysis_date)"))
                session.execute(text("CREATE INDEX IF NOT EXISTS idx_quality_rating ON indicator_effectiveness(quality_rating)"))

                session.commit()

            self.logger.info("Indicator effectiveness database initialized with unified abstraction")

        except Exception as e:
            self.logger.error(f"Failed to initialize indicator database: {str(e)}")
            raise

    def analyze_indicator_effectiveness(self,
                                     indicator_name: str,
                                     market_regime: Optional[MarketRegime] = None,
                                     force_refresh: bool = False) -> IndicatorPerformanceMetrics:
        """
        Perform comprehensive effectiveness analysis for a single indicator.

        Args:
            indicator_name: Name of the indicator to analyze
            market_regime: Specific market regime to analyze (None for all)
            force_refresh: Force refresh of cached results

        Returns:
            Comprehensive performance metrics for the indicator

        Raises:
            InsufficientDataError: If insufficient data for analysis
            ValidationError: If analysis cannot be completed
        """
        try:
            # Check cache first
            cache_key = f"{indicator_name}_{market_regime}_{self.analysis_window_days}"

            if (not force_refresh and
                cache_key in self.indicator_cache and
                cache_key in self.cache_expiry and
                datetime.utcnow() < self.cache_expiry[cache_key]):
                return self.indicator_cache[cache_key]

            self.logger.info(f"Analyzing effectiveness of indicator: {indicator_name}")

            # Get historical usage data
            usage_data = self._get_indicator_usage_data(indicator_name, market_regime)

            if len(usage_data) < self.min_usage_count:
                raise InsufficientDataError(
                    f"Insufficient usage data for {indicator_name}. "
                    f"Need {self.min_usage_count}, got {len(usage_data)}"
                )

            # Perform walk-forward validation analysis
            validation_results = self._perform_indicator_validation(indicator_name, usage_data)

            # Calculate comprehensive metrics
            metrics = self._calculate_indicator_metrics(indicator_name, usage_data, validation_results, market_regime)

            # Cache results
            self.indicator_cache[cache_key] = metrics
            self.cache_expiry[cache_key] = datetime.utcnow() + self.cache_duration

            # Save to database
            self._save_indicator_analysis(metrics, market_regime)

            self.logger.info(
                f"Analysis complete for {indicator_name}: "
                f"Quality={metrics.quality_rating.value}, "
                f"Out-of-sample success={metrics.success_rate_out_of_sample:.3f}"
            )

            return metrics

        except Exception as e:
            self.logger.error(f"Error analyzing indicator {indicator_name}: {str(e)}")
            raise ValidationError(f"Indicator analysis failed: {str(e)}") from e

    def _get_indicator_usage_data(self,
                                indicator_name: str,
                                market_regime: Optional[MarketRegime] = None) -> List[Dict[str, Any]]:
        """
        Retrieve historical usage data for an indicator.

        Args:
            indicator_name: Name of the indicator
            market_regime: Market regime filter (optional)

        Returns:
            List of usage instances with trade outcomes
        """
        cutoff_date = datetime.utcnow() - timedelta(days=self.analysis_window_days)
        usage_data = []

        # Get completed trades that used this indicator
        for trade_record in self.performance_tracker.completed_trades.values():
            if (trade_record.exit_time and
                trade_record.exit_time >= cutoff_date and
                indicator_name in trade_record.indicators_used):

                # Apply market regime filter if specified
                if market_regime and trade_record.market_regime != market_regime:
                    continue

                usage_instance = {
                    'trade_id': trade_record.trade_id,
                    'timestamp': trade_record.exit_time,
                    'validation_type': trade_record.validation_period_type.value,
                    'success': trade_record.pnl and trade_record.pnl > 0,
                    'pnl_contribution': trade_record.pnl or Decimal('0'),
                    'market_regime': trade_record.market_regime.value if trade_record.market_regime else 'UNKNOWN',
                    'confidence': trade_record.original_signal.confidence,
                    'grade': trade_record.grade.value if trade_record.grade else 'F',
                    'generating_agent': trade_record.generating_agent,
                    'timeframe': trade_record.timeframe,
                    'symbol': trade_record.original_signal.symbol
                }

                usage_data.append(usage_instance)

        return sorted(usage_data, key=lambda x: x['timestamp'])

    def _perform_indicator_validation(self,
                                    indicator_name: str,
                                    usage_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform walk-forward validation analysis on indicator usage.

        Args:
            indicator_name: Name of the indicator
            usage_data: Historical usage data

        Returns:
            Validation results and metrics
        """
        # Separate data by validation type
        in_sample_data = [d for d in usage_data if d['validation_type'] == 'IN_SAMPLE']
        out_of_sample_data = [d for d in usage_data if d['validation_type'] == 'OUT_OF_SAMPLE']
        live_data = [d for d in usage_data if d['validation_type'] == 'LIVE_TRADING']

        # Calculate success rates for each validation type
        validation_results = {
            'in_sample': self._calculate_validation_metrics(in_sample_data),
            'out_of_sample': self._calculate_validation_metrics(out_of_sample_data),
            'live_trading': self._calculate_validation_metrics(live_data)
        }

        # Calculate overfitting indicators
        if validation_results['in_sample']['sample_size'] > 0 and validation_results['out_of_sample']['sample_size'] > 0:
            in_sample_success = validation_results['in_sample']['success_rate']
            out_of_sample_success = validation_results['out_of_sample']['success_rate']

            if in_sample_success > 0:
                validation_results['performance_degradation'] = (in_sample_success - out_of_sample_success) / in_sample_success
            else:
                validation_results['performance_degradation'] = 0.0
        else:
            validation_results['performance_degradation'] = 0.0

        # Calculate consistency metrics
        validation_results['consistency'] = self._calculate_consistency_metrics(usage_data)

        # Calculate regime-specific performance
        validation_results['regime_performance'] = self._calculate_regime_performance(usage_data)

        return validation_results

    def _calculate_validation_metrics(self, data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate validation metrics for a dataset."""
        if not data:
            return {
                'sample_size': 0,
                'success_rate': 0.0,
                'avg_pnl_contribution': 0.0,
                'pnl_consistency': 0.0,
                'confidence_correlation': 0.0
            }

        successes = sum(1 for d in data if d['success'])
        success_rate = successes / len(data)

        pnl_values = [float(d['pnl_contribution']) for d in data]
        avg_pnl = np.mean(pnl_values)
        pnl_std = np.std(pnl_values) if len(pnl_values) > 1 else 0.0
        pnl_consistency = 1.0 / (1.0 + pnl_std) if pnl_std > 0 else 1.0

        # Calculate correlation between confidence and success
        confidences = [d['confidence'] for d in data]
        successes_binary = [1.0 if d['success'] else 0.0 for d in data]

        confidence_correlation = 0.0
        if len(set(confidences)) > 1 and len(set(successes_binary)) > 1:
            correlation_matrix = np.corrcoef(confidences, successes_binary)
            confidence_correlation = correlation_matrix[0, 1] if not np.isnan(correlation_matrix[0, 1]) else 0.0

        return {
            'sample_size': len(data),
            'success_rate': success_rate,
            'avg_pnl_contribution': avg_pnl,
            'pnl_consistency': pnl_consistency,
            'confidence_correlation': confidence_correlation
        }

    def _calculate_consistency_metrics(self, usage_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate consistency metrics across different dimensions."""
        if len(usage_data) < 10:
            return {'overall': 0.0, 'temporal': 0.0, 'cross_symbol': 0.0, 'cross_timeframe': 0.0}

        # Temporal consistency (performance over time)
        temporal_windows = self._split_data_by_time_windows(usage_data, window_days=14)
        temporal_success_rates = [
            sum(1 for d in window if d['success']) / len(window) if window else 0.0
            for window in temporal_windows
        ]
        temporal_consistency = 1.0 - (np.std(temporal_success_rates) if len(temporal_success_rates) > 1 else 0.0)

        # Cross-symbol consistency
        symbol_performance = defaultdict(list)
        for d in usage_data:
            symbol_performance[d['symbol']].append(d['success'])

        symbol_success_rates = [
            sum(successes) / len(successes) if successes else 0.0
            for successes in symbol_performance.values()
            if len(successes) >= 3  # Minimum trades per symbol
        ]
        cross_symbol_consistency = 1.0 - (np.std(symbol_success_rates) if len(symbol_success_rates) > 1 else 0.0)

        # Cross-timeframe consistency
        timeframe_performance = defaultdict(list)
        for d in usage_data:
            timeframe_performance[d['timeframe']].append(d['success'])

        timeframe_success_rates = [
            sum(successes) / len(successes) if successes else 0.0
            for successes in timeframe_performance.values()
            if len(successes) >= 3  # Minimum trades per timeframe
        ]
        cross_timeframe_consistency = 1.0 - (np.std(timeframe_success_rates) if len(timeframe_success_rates) > 1 else 0.0)

        # Overall consistency score
        overall_consistency = np.mean([temporal_consistency, cross_symbol_consistency, cross_timeframe_consistency])

        return {
            'overall': overall_consistency,
            'temporal': temporal_consistency,
            'cross_symbol': cross_symbol_consistency,
            'cross_timeframe': cross_timeframe_consistency
        }

    def _split_data_by_time_windows(self, data: List[Dict[str, Any]], window_days: int) -> List[List[Dict[str, Any]]]:
        """Split data into time-based windows."""
        if not data:
            return []

        sorted_data = sorted(data, key=lambda x: x['timestamp'])
        windows = []

        start_date = sorted_data[0]['timestamp']
        end_date = sorted_data[-1]['timestamp']

        current_start = start_date

        while current_start <= end_date:
            window_end = current_start + timedelta(days=window_days)

            window_data = [
                d for d in sorted_data
                if current_start <= d['timestamp'] < window_end
            ]

            if window_data:
                windows.append(window_data)

            current_start = window_end

        return windows

    def _calculate_regime_performance(self, usage_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate performance across different market regimes."""
        regime_performance = defaultdict(list)

        for d in usage_data:
            regime = d['market_regime']
            regime_performance[regime].append(d['success'])

        regime_success_rates = {}
        for regime, successes in regime_performance.items():
            if len(successes) >= 3:  # Minimum trades for statistical significance
                regime_success_rates[regime] = sum(successes) / len(successes)

        return regime_success_rates
    def _calculate_indicator_metrics(self,
                                   indicator_name: str,
                                   usage_data: List[Dict[str, Any]],
                                   validation_results: Dict[str, Any],
                                   market_regime: Optional[MarketRegime] = None) -> IndicatorPerformanceMetrics:
        """
        Calculate comprehensive indicator performance metrics.

        Args:
            indicator_name: Name of the indicator
            usage_data: Historical usage data
            validation_results: Validation analysis results
            market_regime: Market regime context

        Returns:
            Complete performance metrics
        """
        # Basic success rates
        success_rate_in_sample = validation_results['in_sample']['success_rate']
        success_rate_out_of_sample = validation_results['out_of_sample']['success_rate']
        success_rate_live = validation_results['live_trading']['success_rate']

        # Calculate overfitting score (0 = no overfitting, 1 = severe overfitting)
        overfitting_score = max(0.0, validation_results.get('performance_degradation', 0.0))

        # Robustness score (consistency across validation types)
        robustness_components = []
        if success_rate_out_of_sample > 0:
            robustness_components.append(success_rate_out_of_sample)
        if success_rate_live > 0:
            robustness_components.append(success_rate_live)

        robustness_score = np.mean(robustness_components) if robustness_components else 0.0

        # Stability across regimes
        regime_performances = list(validation_results['regime_performance'].values())
        if len(regime_performances) > 1:
            stability_across_regimes = 1.0 - np.std(regime_performances)
        else:
            stability_across_regimes = 1.0 if regime_performances and regime_performances[0] > 0.5 else 0.0

        # Calculate profit-related metrics
        total_pnl = sum(d['pnl_contribution'] for d in usage_data)
        avg_contribution = total_pnl / len(usage_data) if usage_data else Decimal('0')

        # Risk-adjusted effectiveness
        pnl_values = [float(d['pnl_contribution']) for d in usage_data]
        pnl_std = np.std(pnl_values) if len(pnl_values) > 1 else 1.0
        risk_adjusted_effectiveness = float(avg_contribution) / (pnl_std + 1e-8)

        # Consistency score
        consistency_score = validation_results['consistency']['overall']

        # Recent performance trend
        recent_trend = self._calculate_recent_trend(usage_data)

        # Performance decay rate
        decay_rate = self._calculate_performance_decay(usage_data)

        # Composite confidence score
        confidence_score = self._calculate_composite_confidence_score(
            overfitting_score, robustness_score, stability_across_regimes,
            consistency_score, len(usage_data)
        )

        # Quality rating
        quality_rating = self._determine_quality_rating(
            success_rate_out_of_sample, overfitting_score, robustness_score,
            confidence_score, len(usage_data)
        )

        # Risk profile
        risk_profile = self._determine_risk_profile(
            stability_across_regimes, consistency_score, pnl_std
        )

        # Usage recommendations
        recommended_for_live = self._should_recommend_for_live(
            quality_rating, overfitting_score, success_rate_out_of_sample
        )

        recommended_weight = self._calculate_recommended_weight(
            quality_rating, confidence_score, consistency_score
        )

        usage_constraints = self._generate_usage_constraints(
            quality_rating, risk_profile, validation_results
        )

        # Sample size adequacy
        sample_size_adequacy = self._assess_sample_size_adequacy(len(usage_data))

        return IndicatorPerformanceMetrics(
            indicator_name=indicator_name,
            total_usage_count=len(usage_data),
            success_rate_in_sample=success_rate_in_sample,
            success_rate_out_of_sample=success_rate_out_of_sample,
            success_rate_live=success_rate_live,
            avg_contribution_to_profit=avg_contribution,
            profit_consistency_score=validation_results['consistency']['overall'],
            risk_adjusted_effectiveness=risk_adjusted_effectiveness,
            overfitting_score=overfitting_score,
            robustness_score=robustness_score,
            stability_across_regimes=stability_across_regimes,
            quality_rating=quality_rating,
            risk_profile=risk_profile,
            confidence_score=confidence_score,
            recommended_for_live=recommended_for_live,
            recommended_weight=recommended_weight,
            usage_constraints=usage_constraints,
            recent_performance_trend=recent_trend,
            performance_decay_rate=decay_rate,
            regime_performance=validation_results['regime_performance'],
            last_updated=datetime.utcnow(),
            analysis_period_days=self.analysis_window_days,
            sample_size_adequacy=sample_size_adequacy
        )

    def _calculate_recent_trend(self, usage_data: List[Dict[str, Any]]) -> str:
        """Calculate recent performance trend."""
        if len(usage_data) < 10:
            return "INSUFFICIENT_DATA"

        # Split into recent and historical periods
        sorted_data = sorted(usage_data, key=lambda x: x['timestamp'])
        split_point = len(sorted_data) // 2

        historical_data = sorted_data[:split_point]
        recent_data = sorted_data[split_point:]

        historical_success = sum(1 for d in historical_data if d['success']) / len(historical_data)
        recent_success = sum(1 for d in recent_data if d['success']) / len(recent_data)

        improvement = recent_success - historical_success

        if improvement > 0.1:
            return "IMPROVING"
        elif improvement < -0.1:
            return "DECLINING"
        else:
            return "STABLE"

    def _calculate_performance_decay(self, usage_data: List[Dict[str, Any]]) -> float:
        """Calculate performance decay rate over time."""
        if len(usage_data) < 20:
            return 0.0

        # Calculate success rate in time windows
        time_windows = self._split_data_by_time_windows(usage_data, window_days=7)

        if len(time_windows) < 3:
            return 0.0

        success_rates = []
        for window in time_windows:
            if window:
                success_rate = sum(1 for d in window if d['success']) / len(window)
                success_rates.append(success_rate)

        if len(success_rates) < 3:
            return 0.0

        # Calculate linear trend (negative slope indicates decay)
        x = np.arange(len(success_rates))
        slope, _ = np.polyfit(x, success_rates, 1)

        # Convert to decay rate (0 = no decay, 1 = complete decay)
        decay_rate = max(0.0, -slope)

        return float(min(1.0, decay_rate))

    def _calculate_composite_confidence_score(self,
                                            overfitting_score: float,
                                            robustness_score: float,
                                            stability_score: float,
                                            consistency_score: float,
                                            sample_size: int) -> float:
        """Calculate composite confidence score."""
        # Base confidence from core metrics
        base_confidence = (
            (1.0 - overfitting_score) * 0.3 +
            robustness_score * 0.25 +
            stability_score * 0.25 +
            consistency_score * 0.2
        )

        # Sample size adjustment
        if sample_size < 20:
            sample_size_factor = 0.5
        elif sample_size < 50:
            sample_size_factor = 0.7
        elif sample_size < 100:
            sample_size_factor = 0.85
        else:
            sample_size_factor = 1.0

        confidence_score = base_confidence * sample_size_factor

        return float(min(1.0, max(0.0, confidence_score)))

    def _determine_quality_rating(self,
                                success_rate_oos: float,
                                overfitting_score: float,
                                robustness_score: float,
                                confidence_score: float,
                                sample_size: int) -> IndicatorQualityRating:
        """Determine quality rating based on multiple factors."""

        # Check for clear overfitting
        if overfitting_score > 0.4:
            return IndicatorQualityRating.OVERFITTED

        # Check sample size adequacy
        if sample_size < 20:
            return IndicatorQualityRating.PROMISING if success_rate_oos > 0.5 else IndicatorQualityRating.QUESTIONABLE

        # Main quality assessment
        if (success_rate_oos >= 0.65 and
            overfitting_score < 0.15 and
            robustness_score >= 0.7 and
            confidence_score >= 0.8):
            return IndicatorQualityRating.ELITE

        elif (success_rate_oos >= 0.55 and
              overfitting_score < 0.25 and
              robustness_score >= 0.6 and
              confidence_score >= 0.6):
            return IndicatorQualityRating.VALIDATED

        elif (success_rate_oos >= 0.45 and
              overfitting_score < 0.35 and
              confidence_score >= 0.4):
            return IndicatorQualityRating.PROMISING

        elif success_rate_oos >= 0.35:
            return IndicatorQualityRating.QUESTIONABLE

        else:
            return IndicatorQualityRating.UNRELIABLE

    def _determine_risk_profile(self,
                              stability_score: float,
                              consistency_score: float,
                              pnl_volatility: float) -> IndicatorRiskProfile:
        """Determine risk profile for indicator usage."""

        # Calculate overall stability metric
        overall_stability = (stability_score + consistency_score) / 2

        # Normalize PnL volatility (higher volatility = higher risk)
        volatility_risk = min(1.0, pnl_volatility / 100.0)  # Normalize to 0-1

        # Combined risk assessment
        risk_score = (1.0 - overall_stability) * 0.6 + volatility_risk * 0.4

        if risk_score <= 0.2:
            return IndicatorRiskProfile.LOW_RISK
        elif risk_score <= 0.4:
            return IndicatorRiskProfile.MODERATE_RISK
        elif risk_score <= 0.7:
            return IndicatorRiskProfile.HIGH_RISK
        else:
            return IndicatorRiskProfile.VERY_HIGH_RISK

    def _should_recommend_for_live(self,
                                 quality_rating: IndicatorQualityRating,
                                 overfitting_score: float,
                                 success_rate_oos: float) -> bool:
        """Determine if indicator should be recommended for live trading."""

        # Never recommend clearly problematic indicators
        if quality_rating in [IndicatorQualityRating.OVERFITTED, IndicatorQualityRating.UNRELIABLE]:
            return False

        # Conservative thresholds for live trading
        if (quality_rating in [IndicatorQualityRating.ELITE, IndicatorQualityRating.VALIDATED] and
            overfitting_score < 0.3 and
            success_rate_oos > 0.5):
            return True

        return False

    def _calculate_recommended_weight(self,
                                    quality_rating: IndicatorQualityRating,
                                    confidence_score: float,
                                    consistency_score: float) -> float:
        """Calculate recommended weight for indicator usage."""

        # Base weights by quality rating
        base_weights = {
            IndicatorQualityRating.ELITE: 1.0,
            IndicatorQualityRating.VALIDATED: 0.8,
            IndicatorQualityRating.PROMISING: 0.5,
            IndicatorQualityRating.QUESTIONABLE: 0.2,
            IndicatorQualityRating.UNRELIABLE: 0.0,
            IndicatorQualityRating.OVERFITTED: 0.0
        }

        base_weight = base_weights.get(quality_rating, 0.0)

        # Adjust by confidence and consistency
        adjusted_weight = base_weight * confidence_score * consistency_score

        return float(min(1.0, max(0.0, adjusted_weight)))

    def _generate_usage_constraints(self,
                                  quality_rating: IndicatorQualityRating,
                                  risk_profile: IndicatorRiskProfile,
                                  validation_results: Dict[str, Any]) -> List[str]:
        """Generate usage constraints and recommendations."""
        constraints = []

        # Quality-based constraints
        if quality_rating == IndicatorQualityRating.OVERFITTED:
            constraints.append("DO_NOT_USE_LIVE")
        elif quality_rating == IndicatorQualityRating.UNRELIABLE:
            constraints.append("USE_WITH_EXTREME_CAUTION")
        elif quality_rating == IndicatorQualityRating.QUESTIONABLE:
            constraints.append("REQUIRE_CONFIRMATION")
        elif quality_rating == IndicatorQualityRating.PROMISING:
            constraints.append("MONITOR_CLOSELY")

        # Risk-based constraints
        if risk_profile == IndicatorRiskProfile.VERY_HIGH_RISK:
            constraints.append("MAXIMUM_WEIGHT_10_PERCENT")
        elif risk_profile == IndicatorRiskProfile.HIGH_RISK:
            constraints.append("MAXIMUM_WEIGHT_25_PERCENT")
        elif risk_profile == IndicatorRiskProfile.MODERATE_RISK:
            constraints.append("MAXIMUM_WEIGHT_50_PERCENT")

        # Regime-specific constraints
        regime_performance = validation_results.get('regime_performance', {})
        poor_regimes = [regime for regime, perf in regime_performance.items() if perf < 0.4]

        if poor_regimes:
            constraints.append(f"AVOID_IN_REGIMES_{','.join(poor_regimes)}")

        # Sample size constraints
        in_sample_size = validation_results['in_sample']['sample_size']
        oos_size = validation_results['out_of_sample']['sample_size']

        if oos_size < 20:
            constraints.append("INSUFFICIENT_VALIDATION_DATA")
        elif oos_size < 50:
            constraints.append("LIMITED_VALIDATION_DATA")

        return constraints

    def _assess_sample_size_adequacy(self, sample_size: int) -> str:
        """Assess if sample size is adequate for reliable analysis."""
        if sample_size >= 100:
            return "EXCELLENT"
        elif sample_size >= 50:
            return "GOOD"
        elif sample_size >= 20:
            return "ADEQUATE"
        else:
            return "INSUFFICIENT"

    def _save_indicator_analysis(self,
                               metrics: IndicatorPerformanceMetrics,
                               market_regime: Optional[MarketRegime]):
        """Save indicator analysis using unified database abstraction."""
        try:
            with self.database.get_sync_session() as session:
                session.execute(text("""
                    INSERT OR REPLACE INTO indicator_effectiveness (
                        indicator_name, analysis_date, market_regime,
                        total_usage_count, success_rate_in_sample, success_rate_out_of_sample,
                        success_rate_live, avg_contribution_to_profit, profit_consistency_score,
                        risk_adjusted_effectiveness, overfitting_score, robustness_score,
                        stability_across_regimes, quality_rating, risk_profile,
                        confidence_score, recommended_for_live, recommended_weight,
                        usage_constraints, recent_performance_trend, performance_decay_rate,
                        regime_performance, analysis_period_days, sample_size_adequacy
                    ) VALUES (:indicator_name, :analysis_date, :market_regime, :total_usage_count,
                             :success_rate_in_sample, :success_rate_out_of_sample, :success_rate_live,
                             :avg_contribution_to_profit, :profit_consistency_score, :risk_adjusted_effectiveness,
                             :overfitting_score, :robustness_score, :stability_across_regimes,
                             :quality_rating, :risk_profile, :confidence_score, :recommended_for_live,
                             :recommended_weight, :usage_constraints, :recent_performance_trend,
                             :performance_decay_rate, :regime_performance, :analysis_period_days,
                             :sample_size_adequacy)
                """), {
                    'indicator_name': metrics.indicator_name,
                    'analysis_date': datetime.utcnow().date(),
                    'market_regime': market_regime.value if market_regime else 'ALL',
                    'total_usage_count': metrics.total_usage_count,
                    'success_rate_in_sample': metrics.success_rate_in_sample,
                    'success_rate_out_of_sample': metrics.success_rate_out_of_sample,
                    'success_rate_live': metrics.success_rate_live,
                    'avg_contribution_to_profit': float(metrics.avg_contribution_to_profit),
                    'profit_consistency_score': metrics.profit_consistency_score,
                    'risk_adjusted_effectiveness': metrics.risk_adjusted_effectiveness,
                    'overfitting_score': metrics.overfitting_score,
                    'robustness_score': metrics.robustness_score,
                    'stability_across_regimes': metrics.stability_across_regimes,
                    'quality_rating': metrics.quality_rating.value,
                    'risk_profile': metrics.risk_profile.value,
                    'confidence_score': metrics.confidence_score,
                    'recommended_for_live': metrics.recommended_for_live,
                    'recommended_weight': metrics.recommended_weight,
                    'usage_constraints': json.dumps(metrics.usage_constraints),
                    'recent_performance_trend': metrics.recent_performance_trend,
                    'performance_decay_rate': metrics.performance_decay_rate,
                    'regime_performance': json.dumps(metrics.regime_performance),
                    'analysis_period_days': metrics.analysis_period_days,
                    'sample_size_adequacy': metrics.sample_size_adequacy
                })
                session.commit()

        except Exception as e:
            self.logger.error(f"Failed to save indicator analysis: {str(e)}")

    def analyze_all_indicators(self,
                             market_regime: Optional[MarketRegime] = None,
                             min_usage_threshold: Optional[int] = None) -> Dict[str, IndicatorPerformanceMetrics]:
        """
        Analyze effectiveness of all indicators with sufficient usage data.

        Args:
            market_regime: Specific market regime to analyze
            min_usage_threshold: Minimum usage count threshold

        Returns:
            Dictionary mapping indicator names to their performance metrics
        """
        min_usage = min_usage_threshold or self.min_usage_count

        # Get all indicators that have been used
        indicator_usage_counts = defaultdict(int)
        cutoff_date = datetime.utcnow() - timedelta(days=self.analysis_window_days)

        for trade_record in self.performance_tracker.completed_trades.values():
            if trade_record.exit_time and trade_record.exit_time >= cutoff_date:
                if market_regime is None or trade_record.market_regime == market_regime:
                    for indicator in trade_record.indicators_used:
                        indicator_usage_counts[indicator] += 1

        # Filter indicators with sufficient usage
        qualified_indicators = [
            indicator for indicator, count in indicator_usage_counts.items()
            if count >= min_usage
        ]

        self.logger.info(f"Analyzing {len(qualified_indicators)} indicators with sufficient usage")

        # Analyze each qualified indicator
        results = {}
        for indicator in qualified_indicators:
            try:
                metrics = self.analyze_indicator_effectiveness(indicator, market_regime)
                results[indicator] = metrics
            except Exception as e:
                self.logger.warning(f"Failed to analyze indicator {indicator}: {str(e)}")

        return results

    def get_elite_indicators(self,
                           market_regime: MarketRegime,
                           max_count: int = 10,
                           force_refresh: bool = False) -> List[str]:
        """
        Get the elite indicator set for a specific market regime.

        Args:
            market_regime: Market regime to get indicators for
            max_count: Maximum number of indicators to return
            force_refresh: Force refresh of elite indicators

        Returns:
            List of elite indicator names
        """
        regime_key = market_regime.value

        # Check if we need to refresh
        if (force_refresh or
            regime_key not in self.elite_indicators or
            not self.last_elite_update or
            datetime.utcnow() - self.last_elite_update > timedelta(hours=1)):

            self._update_elite_indicators(market_regime)

        return self.elite_indicators.get(regime_key, [])[:max_count]

    def _update_elite_indicators(self, market_regime: MarketRegime):
        """Update elite indicator set for a market regime."""
        try:
            # Analyze all indicators for this regime
            all_indicators = self.analyze_all_indicators(market_regime)

            # Filter for elite quality indicators
            elite_candidates = [
                (name, metrics) for name, metrics in all_indicators.items()
                if metrics.quality_rating in [IndicatorQualityRating.ELITE, IndicatorQualityRating.VALIDATED]
                and metrics.recommended_for_live
            ]

            # Sort by composite effectiveness score
            elite_candidates.sort(key=lambda x: self._calculate_composite_effectiveness_score(x[1]), reverse=True)

            # Update elite indicators
            regime_key = market_regime.value
            self.elite_indicators[regime_key] = [name for name, _ in elite_candidates]
            self.last_elite_update = datetime.utcnow()

            # Save to database
            self._save_elite_indicators(regime_key, elite_candidates)

            self.logger.info(f"Updated elite indicators for {regime_key}: {len(elite_candidates)} indicators")

        except Exception as e:
            self.logger.error(f"Failed to update elite indicators for {market_regime}: {str(e)}")

    def _calculate_composite_effectiveness_score(self, metrics: IndicatorPerformanceMetrics) -> float:
        """Calculate composite effectiveness score for ranking."""
        score = (
            metrics.success_rate_out_of_sample * self.weights['out_of_sample_performance'] +
            metrics.profit_consistency_score * self.weights['consistency'] +
            metrics.stability_across_regimes * self.weights['stability'] +
            (1.0 if metrics.recent_performance_trend == "IMPROVING" else 0.8 if metrics.recent_performance_trend == "STABLE" else 0.5) * self.weights['recent_trend'] +
            (1.0 - metrics.performance_decay_rate) * 0.1  # Bonus for non-decaying performance
        )

        # Apply quality rating multiplier
        quality_multipliers = {
            IndicatorQualityRating.ELITE: 1.2,
            IndicatorQualityRating.VALIDATED: 1.0,
            IndicatorQualityRating.PROMISING: 0.8,
            IndicatorQualityRating.QUESTIONABLE: 0.5,
            IndicatorQualityRating.UNRELIABLE: 0.2,
            IndicatorQualityRating.OVERFITTED: 0.0
        }

        multiplier = quality_multipliers.get(metrics.quality_rating, 0.5)

        return score * multiplier

    def _save_elite_indicators(self, regime: str, elite_candidates: List[Tuple[str, IndicatorPerformanceMetrics]]):
        """Save elite indicators using unified database abstraction."""
        try:
            with self.database.get_sync_session() as session:
                # Clear existing elite indicators for this regime
                session.execute(text("DELETE FROM elite_indicator_sets WHERE regime = :regime"),
                              {'regime': regime})

                # Insert new elite indicators
                for name, metrics in elite_candidates:
                    effectiveness_score = self._calculate_composite_effectiveness_score(metrics)
                    performance_summary = {
                        'success_rate': metrics.success_rate_out_of_sample,
                        'overfitting_score': metrics.overfitting_score,
                        'quality_rating': metrics.quality_rating.value,
                        'confidence_score': metrics.confidence_score
                    }

                    session.execute(text("""
                        INSERT INTO elite_indicator_sets
                        (regime, indicator_name, effectiveness_score, selection_date, performance_metrics)
                        VALUES (:regime, :indicator_name, :effectiveness_score, :selection_date, :performance_metrics)
                    """), {
                        'regime': regime,
                        'indicator_name': name,
                        'effectiveness_score': effectiveness_score,
                        'selection_date': datetime.utcnow(),
                        'performance_metrics': json.dumps(performance_summary)
                    })

                session.commit()

        except Exception as e:
            self.logger.error(f"Failed to save elite indicators: {str(e)}")

    def get_effectiveness_summary(self) -> Dict[str, Any]:
        """Get overall effectiveness analysis summary."""
        try:
            # Analyze all indicators
            all_results = self.analyze_all_indicators()

            if not all_results:
                return {
                    'total_indicators_analyzed': 0,
                    'elite_count': 0,
                    'validated_count': 0,
                    'overfitted_count': 0,
                    'average_overfitting_score': 0.0,
                    'recommendations': ['NO_INDICATORS_ANALYZED']
                }

            # Calculate summary statistics
            quality_distribution = defaultdict(int)
            overfitting_scores = []

            for metrics in all_results.values():
                quality_distribution[metrics.quality_rating.value] += 1
                overfitting_scores.append(metrics.overfitting_score)

            # System-wide recommendations
            recommendations = []

            elite_count = quality_distribution.get('ELITE', 0)
            overfitted_count = quality_distribution.get('OVERFITTED', 0)
            total_count = len(all_results)

            if elite_count >= 5:
                recommendations.append('SUFFICIENT_ELITE_INDICATORS')
            elif elite_count >= 2:
                recommendations.append('MODERATE_ELITE_INDICATORS')
            else:
                recommendations.append('INSUFFICIENT_ELITE_INDICATORS')

            if overfitted_count > total_count * 0.3:
                recommendations.append('HIGH_OVERFITTING_DETECTED')

            avg_overfitting = np.mean(overfitting_scores) if overfitting_scores else 0.0

            if avg_overfitting > 0.4:
                recommendations.append('SYSTEM_WIDE_OVERFITTING_RISK')

            return {
                'total_indicators_analyzed': total_count,
                'quality_distribution': dict(quality_distribution),
                'elite_count': elite_count,
                'validated_count': quality_distribution.get('VALIDATED', 0),
                'overfitted_count': overfitted_count,
                'average_overfitting_score': avg_overfitting,
                'top_indicators': [
                    name for name, metrics in sorted(
                        all_results.items(),
                        key=lambda x: self._calculate_composite_effectiveness_score(x[1]),
                        reverse=True
                    )[:10]
                ],
                'recommendations': recommendations,
                'analysis_timestamp': datetime.utcnow().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Error generating effectiveness summary: {str(e)}")
            return {'error': str(e)}
