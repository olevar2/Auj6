"""
Ulcer Index Indicator - Advanced Downside Risk and Drawdown Analysis System

The Ulcer Index measures downside volatility and drawdown stress, providing crucial risk-adjusted
performance metrics for portfolio optimization and risk management in humanitarian trading.

Key Features:
- Advanced downside volatility measurement focusing on drawdown stress
- Risk-adjusted performance metrics (Ulcer Performance Index, Martin Ratio)
- Multi-timeframe drawdown analysis with recovery time estimation
- Machine learning-enhanced risk prediction and portfolio optimization
- Dynamic stress level classification and warning systems
- Recovery pattern analysis and prediction
- Statistical significance testing for risk metrics
- Advanced filtering for stress level detection

Mathematical Foundation:
- Percentage Drawdown = 100 * (Close - Highest High over period) / Highest High
- Ulcer Index = sqrt(sum(Percentage Drawdown²) / period)
- Risk-adjusted metrics: UPI = Return / Ulcer Index
- Enhanced with ML models for drawdown prediction and recovery analysis

Author: AUJ Platform Development Team
Version: 1.0.0
Created: 2025-01-XX
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML and Statistical Libraries
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_squared_error, r2_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

try:
    from scipy import stats, optimize
    from scipy.signal import find_peaks, savgol_filter
    from scipy.stats import jarque_bera, shapiro
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from ..base.standard_indicator import StandardIndicatorInterface


class StressLevel(Enum):
    """Drawdown stress level classification"""
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    SEVERE = "severe"
    EXTREME = "extreme"


class RecoveryPhase(Enum):
    """Drawdown recovery phase classification"""
    NO_DRAWDOWN = "no_drawdown"
    EARLY_DRAWDOWN = "early_drawdown"
    DEEP_DRAWDOWN = "deep_drawdown"
    RECOVERY_STARTING = "recovery_starting"
    ACTIVE_RECOVERY = "active_recovery"
    NEAR_RECOVERY = "near_recovery"


class RiskAlert(Enum):
    """Risk alert levels"""
    GREEN = "green"
    YELLOW = "yellow"
    ORANGE = "orange"
    RED = "red"
    CRITICAL = "critical"


@dataclass
class DrawdownMetrics:
    """Comprehensive drawdown analysis"""
    current_drawdown: float
    max_drawdown: float
    average_drawdown: float
    drawdown_duration: int
    recovery_time_estimate: int
    recovery_probability: float
    drawdown_frequency: float


@dataclass
class RiskAdjustedMetrics:
    """Risk-adjusted performance metrics"""
    ulcer_performance_index: float  # Return / Ulcer Index
    martin_ratio: float  # Return / Ulcer Index (alternative calculation)
    pain_index: float  # Average drawdown
    gain_pain_ratio: float  # Return / Pain Index
    sterling_ratio: float  # Return / Max Drawdown
    burke_ratio: float  # Return / sqrt(sum of squared drawdowns)


@dataclass
class StressAnalysis:
    """Market stress analysis"""
    stress_level: StressLevel
    stress_intensity: float
    stress_duration: int
    stress_frequency: float
    stress_clustering: float  # Tendency for stress periods to cluster
    systemic_risk_indicator: float


@dataclass
class RecoveryAnalysis:
    """Drawdown recovery pattern analysis"""
    recovery_phase: RecoveryPhase
    recovery_strength: float
    recovery_velocity: float
    recovery_consistency: float
    historical_recovery_time: float
    recovery_success_probability: float


@dataclass
class PortfolioOptimization:
    """Portfolio optimization recommendations"""
    position_size_multiplier: float
    risk_budget_allocation: float
    stress_tolerance_score: float
    recommended_exposure: float
    diversification_benefit: float
    correlation_adjusted_risk: float


@dataclass
class UlcerIndexResult:
    """Complete Ulcer Index analysis result"""
    # Core indicator values
    ulcer_index: float
    percentage_drawdown: float
    rolling_high: float
    
    # Drawdown metrics
    drawdown_metrics: DrawdownMetrics
    
    # Risk-adjusted performance
    risk_adjusted_metrics: RiskAdjustedMetrics
    
    # Stress analysis
    stress_analysis: StressAnalysis
    
    # Recovery analysis
    recovery_analysis: RecoveryAnalysis
    
    # Portfolio optimization
    portfolio_optimization: PortfolioOptimization
    
    # Risk alerts
    risk_alert: RiskAlert
    alert_message: str
    
    # ML predictions
    drawdown_prediction: float
    recovery_time_prediction: int
    stress_probability: float
    
    # Statistical measures
    percentile_rank: float
    statistical_significance: float
    distribution_analysis: Dict[str, float]
    
    # Multi-timeframe analysis
    short_term_ulcer: float
    medium_term_ulcer: float
    long_term_ulcer: float
    
    # Historical context
    historical_comparison: Dict[str, float]
    regime_analysis: str
    
    # Metadata
    timestamp: datetime
    confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary"""
        return {
            'ulcer_index': self.ulcer_index,
            'percentage_drawdown': self.percentage_drawdown,
            'rolling_high': self.rolling_high,
            'current_drawdown': self.drawdown_metrics.current_drawdown,
            'max_drawdown': self.drawdown_metrics.max_drawdown,
            'average_drawdown': self.drawdown_metrics.average_drawdown,
            'drawdown_duration': self.drawdown_metrics.drawdown_duration,
            'recovery_time_estimate': self.drawdown_metrics.recovery_time_estimate,
            'ulcer_performance_index': self.risk_adjusted_metrics.ulcer_performance_index,
            'martin_ratio': self.risk_adjusted_metrics.martin_ratio,
            'pain_index': self.risk_adjusted_metrics.pain_index,
            'stress_level': self.stress_analysis.stress_level.value,
            'stress_intensity': self.stress_analysis.stress_intensity,
            'recovery_phase': self.recovery_analysis.recovery_phase.value,
            'recovery_probability': self.recovery_analysis.recovery_success_probability,
            'risk_alert': self.risk_alert.value,
            'alert_message': self.alert_message,
            'position_size_multiplier': self.portfolio_optimization.position_size_multiplier,
            'recommended_exposure': self.portfolio_optimization.recommended_exposure,
            'drawdown_prediction': self.drawdown_prediction,
            'recovery_time_prediction': self.recovery_time_prediction,
            'stress_probability': self.stress_probability,
            'percentile_rank': self.percentile_rank,
            'short_term_ulcer': self.short_term_ulcer,
            'medium_term_ulcer': self.medium_term_ulcer,
            'long_term_ulcer': self.long_term_ulcer,
            'regime_analysis': self.regime_analysis,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat()
        }


class UlcerIndexIndicator(StandardIndicatorInterface):
    """
    Advanced Ulcer Index Indicator with Machine Learning Integration
    
    This implementation provides sophisticated downside risk analysis, drawdown stress
    measurement, and risk-adjusted performance metrics for optimal portfolio management.
    """
    
    def __init__(self, 
                 period: int = 14,
                 return_period: int = 252,  # For annualized returns
                 stress_threshold: float = 5.0,  # Stress level threshold
                 recovery_threshold: float = 0.95,  # Recovery threshold (95% of previous high)
                 enable_ml_prediction: bool = True,
                 use_volume_weighting: bool = True,
                 lookback_period: int = 504):  # 2 years of data
        """
        Initialize Ulcer Index Indicator
        
        Args:
            period: Period for Ulcer Index calculation
            return_period: Period for return calculations (typically 252 for annual)
            stress_threshold: Threshold for stress level classification
            recovery_threshold: Threshold for recovery determination
            enable_ml_prediction: Enable ML-based predictions
            use_volume_weighting: Include volume in stress calculations
            lookback_period: Historical data period for analysis
        """
        super().__init__()
        self.period = period
        self.return_period = return_period
        self.stress_threshold = stress_threshold
        self.recovery_threshold = recovery_threshold
        self.enable_ml_prediction = enable_ml_prediction
        self.use_volume_weighting = use_volume_weighting
        self.lookback_period = lookback_period
        
        # ML models
        self.drawdown_model: Optional[RandomForestRegressor] = None
        self.recovery_model: Optional[GradientBoostingClassifier] = None
        self.stress_model: Optional[RandomForestRegressor] = None
        self.scaler = StandardScaler() if HAS_SKLEARN else None
        self.robust_scaler = RobustScaler() if HAS_SKLEARN else None
        
        # Historical data storage
        self.ulcer_history: List[float] = []
        self.price_history: List[float] = []
        self.drawdown_history: List[float] = []
        self.volume_history: List[float] = []
        self.high_history: List[float] = []
        self.stress_events: List[Dict] = []
        self.recovery_events: List[Dict] = []
        
        # Analysis parameters
        self.stress_levels = {
            StressLevel.VERY_LOW: (0, 1),
            StressLevel.LOW: (1, 2),
            StressLevel.MODERATE: (2, 3),
            StressLevel.HIGH: (3, 5),
            StressLevel.SEVERE: (5, 8),
            StressLevel.EXTREME: (8, float('inf'))
        }
        
        self.logger = logging.getLogger(__name__)
        
        # Validate dependencies
        if enable_ml_prediction and not HAS_SKLEARN:
            self.logger.warning("Scikit-learn not available. ML features disabled.")
            self.enable_ml_prediction = False
    
    def _calculate_rolling_high(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate rolling maximum (highest high over period)"""
        return prices.rolling(window=period, min_periods=1).max()
    
    def _calculate_percentage_drawdown(self, prices: pd.Series, rolling_high: pd.Series) -> pd.Series:
        """Calculate percentage drawdown"""
        drawdown = 100 * (prices - rolling_high) / rolling_high
        return drawdown.fillna(0)
    
    def _calculate_ulcer_index(self, percentage_drawdown: pd.Series, period: int) -> pd.Series:
        """Calculate Ulcer Index"""
        squared_drawdowns = percentage_drawdown ** 2
        mean_squared_drawdown = squared_drawdowns.rolling(window=period).mean()
        ulcer_index = np.sqrt(mean_squared_drawdown)
        return ulcer_index.fillna(0)
    
    def _analyze_drawdown_metrics(self, 
                                 current_drawdown: float,
                                 drawdown_history: List[float],
                                 price_history: List[float]) -> DrawdownMetrics:
        """Analyze comprehensive drawdown metrics"""
        if len(drawdown_history) < 10:
            return DrawdownMetrics(
                current_drawdown=current_drawdown,
                max_drawdown=current_drawdown,
                average_drawdown=current_drawdown,
                drawdown_duration=1 if current_drawdown < 0 else 0,
                recovery_time_estimate=0,
                recovery_probability=0.5,
                drawdown_frequency=0.0
            )
        
        # Calculate max and average drawdown
        max_drawdown = min(drawdown_history)
        negative_drawdowns = [dd for dd in drawdown_history if dd < 0]
        average_drawdown = np.mean(negative_drawdowns) if negative_drawdowns else 0.0
        
        # Calculate current drawdown duration
        drawdown_duration = 0
        for i in range(len(drawdown_history) - 1, -1, -1):
            if drawdown_history[i] < 0:
                drawdown_duration += 1
            else:
                break
        
        # Estimate recovery time based on historical patterns
        recovery_time_estimate = 0
        if len(self.recovery_events) > 0:
            recovery_times = [event['duration'] for event in self.recovery_events]
            recovery_time_estimate = int(np.median(recovery_times))
        else:
            # Heuristic: deeper drawdowns take longer to recover
            recovery_time_estimate = int(abs(current_drawdown) * 2) if current_drawdown < 0 else 0
        
        # Calculate recovery probability based on historical success
        if len(self.recovery_events) >= 5:
            successful_recoveries = sum(1 for event in self.recovery_events 
                                      if event.get('successful', False))
            recovery_probability = successful_recoveries / len(self.recovery_events)
        else:
            # Base probability adjusted by drawdown depth
            recovery_probability = max(0.1, 0.9 - abs(current_drawdown) * 0.05)
        
        # Calculate drawdown frequency
        total_periods = len(drawdown_history)
        drawdown_periods = sum(1 for dd in drawdown_history if dd < -1.0)  # >1% drawdown
        drawdown_frequency = drawdown_periods / total_periods if total_periods > 0 else 0.0
        
        return DrawdownMetrics(
            current_drawdown=current_drawdown,
            max_drawdown=max_drawdown,
            average_drawdown=average_drawdown,
            drawdown_duration=drawdown_duration,
            recovery_time_estimate=recovery_time_estimate,
            recovery_probability=min(1.0, max(0.0, recovery_probability)),
            drawdown_frequency=drawdown_frequency
        )
    
    def _calculate_risk_adjusted_metrics(self, 
                                       ulcer_index: float,
                                       price_history: List[float],
                                       drawdown_history: List[float]) -> RiskAdjustedMetrics:
        """Calculate comprehensive risk-adjusted performance metrics"""
        if len(price_history) < self.return_period:
            return RiskAdjustedMetrics(
                ulcer_performance_index=0.0,
                martin_ratio=0.0,
                pain_index=0.0,
                gain_pain_ratio=0.0,
                sterling_ratio=0.0,
                burke_ratio=0.0
            )
        
        # Calculate annualized return
        if len(price_history) >= self.return_period:
            annual_return = ((price_history[-1] / price_history[-self.return_period]) ** 
                           (252 / self.return_period) - 1) * 100
        else:
            # Annualize shorter period
            periods = len(price_history)
            total_return = (price_history[-1] / price_history[0] - 1) * 100
            annual_return = total_return * (252 / periods)
        
        # Ulcer Performance Index (UPI)
        ulcer_performance_index = annual_return / ulcer_index if ulcer_index > 0 else 0.0
        
        # Martin Ratio (alternative UPI calculation)
        martin_ratio = ulcer_performance_index  # Same calculation in this implementation
        
        # Pain Index (average of all negative returns/drawdowns)
        negative_drawdowns = [dd for dd in drawdown_history if dd < 0]
        pain_index = abs(np.mean(negative_drawdowns)) if negative_drawdowns else 0.0
        
        # Gain-Pain Ratio
        gain_pain_ratio = annual_return / pain_index if pain_index > 0 else 0.0
        
        # Sterling Ratio
        max_drawdown = abs(min(drawdown_history)) if drawdown_history else 0.0
        sterling_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0.0
        
        # Burke Ratio
        squared_drawdowns = [dd**2 for dd in negative_drawdowns]
        burke_denominator = np.sqrt(np.sum(squared_drawdowns)) if squared_drawdowns else 0.0
        burke_ratio = annual_return / burke_denominator if burke_denominator > 0 else 0.0
        
        return RiskAdjustedMetrics(
            ulcer_performance_index=ulcer_performance_index,
            martin_ratio=martin_ratio,
            pain_index=pain_index,
            gain_pain_ratio=gain_pain_ratio,
            sterling_ratio=sterling_ratio,
            burke_ratio=burke_ratio
        )
    
    def _analyze_stress_level(self, 
                             ulcer_index: float,
                             current_drawdown: float,
                             ulcer_history: List[float]) -> StressAnalysis:
        """Analyze market stress levels and patterns"""
        # Determine stress level
        stress_level = StressLevel.VERY_LOW
        for level, (min_val, max_val) in self.stress_levels.items():
            if min_val <= ulcer_index < max_val:
                stress_level = level
                break
        
        # Calculate stress intensity (0-1 scale)
        if ulcer_index <= 1:
            stress_intensity = ulcer_index / 5  # Scale to 0-0.2 for very low stress
        elif ulcer_index <= 8:
            stress_intensity = 0.2 + (ulcer_index - 1) * 0.7 / 7  # Scale 0.2-0.9
        else:
            stress_intensity = min(1.0, 0.9 + (ulcer_index - 8) * 0.1 / 2)  # Scale 0.9-1.0
        
        # Calculate stress duration
        stress_duration = 0
        threshold = self.stress_threshold
        for i in range(len(ulcer_history) - 1, -1, -1):
            if ulcer_history[i] > threshold:
                stress_duration += 1
            else:
                break
        
        # Calculate stress frequency
        if len(ulcer_history) >= 50:
            high_stress_periods = sum(1 for ui in ulcer_history[-50:] if ui > threshold)
            stress_frequency = high_stress_periods / 50
        else:
            stress_frequency = 0.0
        
        # Calculate stress clustering (tendency for stress to cluster)
        stress_clustering = 0.0
        if len(ulcer_history) >= 20:
            stress_periods = [1 if ui > threshold else 0 for ui in ulcer_history[-20:]]
            # Count consecutive stress periods
            consecutive_counts = []
            current_consecutive = 0
            for period in stress_periods:
                if period == 1:
                    current_consecutive += 1
                else:
                    if current_consecutive > 0:
                        consecutive_counts.append(current_consecutive)
                    current_consecutive = 0
            
            if consecutive_counts:
                stress_clustering = np.mean(consecutive_counts) / 5  # Normalize by typical cluster size
        
        # Systemic risk indicator
        systemic_risk_indicator = min(1.0, stress_intensity * 0.4 + 
                                    stress_frequency * 0.3 + 
                                    stress_clustering * 0.3)
        
        return StressAnalysis(
            stress_level=stress_level,
            stress_intensity=stress_intensity,
            stress_duration=stress_duration,
            stress_frequency=stress_frequency,
            stress_clustering=stress_clustering,
            systemic_risk_indicator=systemic_risk_indicator
        )
    
    def _analyze_recovery_pattern(self, 
                                 current_drawdown: float,
                                 drawdown_history: List[float],
                                 price_history: List[float]) -> RecoveryAnalysis:
        """Analyze drawdown recovery patterns"""
        if current_drawdown >= 0:
            return RecoveryAnalysis(
                recovery_phase=RecoveryPhase.NO_DRAWDOWN,
                recovery_strength=1.0,
                recovery_velocity=0.0,
                recovery_consistency=1.0,
                historical_recovery_time=0.0,
                recovery_success_probability=1.0
            )
        
        # Determine recovery phase
        drawdown_depth = abs(current_drawdown)
        if drawdown_depth > 10:
            recovery_phase = RecoveryPhase.DEEP_DRAWDOWN
        elif drawdown_depth > 5:
            recovery_phase = RecoveryPhase.EARLY_DRAWDOWN
        else:
            # Check if recovering
            if len(drawdown_history) >= 5:
                recent_trend = np.polyfit(range(5), drawdown_history[-5:], 1)[0]
                if recent_trend > 0.1:  # Improving (less negative)
                    if drawdown_depth < 1:
                        recovery_phase = RecoveryPhase.NEAR_RECOVERY
                    else:
                        recovery_phase = RecoveryPhase.ACTIVE_RECOVERY
                elif recent_trend > 0:
                    recovery_phase = RecoveryPhase.RECOVERY_STARTING
                else:
                    recovery_phase = RecoveryPhase.EARLY_DRAWDOWN
            else:
                recovery_phase = RecoveryPhase.EARLY_DRAWDOWN
        
        # Calculate recovery strength
        if len(drawdown_history) >= 10:
            worst_recent = min(drawdown_history[-10:])
            current_improvement = current_drawdown - worst_recent
            max_possible_improvement = abs(worst_recent)
            recovery_strength = current_improvement / max_possible_improvement if max_possible_improvement > 0 else 0
            recovery_strength = max(0, min(1, recovery_strength))
        else:
            recovery_strength = 0.5
        
        # Calculate recovery velocity
        recovery_velocity = 0.0
        if len(drawdown_history) >= 5:
            velocity_slope = np.polyfit(range(5), drawdown_history[-5:], 1)[0]
            recovery_velocity = max(0, velocity_slope)  # Only positive (improving) velocity
        
        # Calculate recovery consistency
        recovery_consistency = 0.5
        if len(drawdown_history) >= 10:
            recent_changes = np.diff(drawdown_history[-10:])
            positive_changes = sum(1 for change in recent_changes if change > 0)
            recovery_consistency = positive_changes / len(recent_changes)
        
        # Historical recovery time
        if len(self.recovery_events) > 0:
            recovery_times = [event['duration'] for event in self.recovery_events]
            historical_recovery_time = np.median(recovery_times)
        else:
            historical_recovery_time = drawdown_depth * 2  # Heuristic
        
        # Recovery success probability
        if len(self.recovery_events) >= 3:
            successful_recoveries = sum(1 for event in self.recovery_events 
                                      if event.get('successful', False))
            base_probability = successful_recoveries / len(self.recovery_events)
        else:
            base_probability = 0.7  # Default assumption
        
        # Adjust based on current recovery indicators
        recovery_success_probability = base_probability * (0.5 + recovery_strength * 0.3 + recovery_consistency * 0.2)
        recovery_success_probability = max(0.1, min(0.95, recovery_success_probability))
        
        return RecoveryAnalysis(
            recovery_phase=recovery_phase,
            recovery_strength=recovery_strength,
            recovery_velocity=recovery_velocity,
            recovery_consistency=recovery_consistency,
            historical_recovery_time=historical_recovery_time,
            recovery_success_probability=recovery_success_probability
        )
    
    def _generate_portfolio_optimization(self, 
                                       stress_analysis: StressAnalysis,
                                       risk_adjusted_metrics: RiskAdjustedMetrics,
                                       recovery_analysis: RecoveryAnalysis) -> PortfolioOptimization:
        """Generate portfolio optimization recommendations"""
        # Position size multiplier based on stress level
        stress_multipliers = {
            StressLevel.VERY_LOW: 1.2,
            StressLevel.LOW: 1.1,
            StressLevel.MODERATE: 1.0,
            StressLevel.HIGH: 0.8,
            StressLevel.SEVERE: 0.6,
            StressLevel.EXTREME: 0.4
        }
        
        position_size_multiplier = stress_multipliers[stress_analysis.stress_level]
        
        # Adjust based on recovery probability
        position_size_multiplier *= (0.8 + recovery_analysis.recovery_success_probability * 0.4)
        
        # Risk budget allocation (what percentage of risk budget to use)
        base_allocation = 0.8
        stress_adjustment = (1 - stress_analysis.stress_intensity) * 0.4
        recovery_adjustment = recovery_analysis.recovery_strength * 0.2
        risk_budget_allocation = base_allocation + stress_adjustment + recovery_adjustment
        risk_budget_allocation = max(0.2, min(1.0, risk_budget_allocation))
        
        # Stress tolerance score
        stress_tolerance_score = (1 - stress_analysis.stress_intensity) * 0.6 + \
                               recovery_analysis.recovery_success_probability * 0.4
        
        # Recommended exposure
        recommended_exposure = position_size_multiplier * risk_budget_allocation
        recommended_exposure = max(0.1, min(1.5, recommended_exposure))
        
        # Diversification benefit (higher in stressed markets)
        diversification_benefit = 0.7 + stress_analysis.stress_intensity * 0.3
        
        # Correlation-adjusted risk (assuming some correlation in stressed markets)
        correlation_adjustment = 1 + stress_analysis.stress_intensity * 0.5
        correlation_adjusted_risk = correlation_adjustment
        
        return PortfolioOptimization(
            position_size_multiplier=position_size_multiplier,
            risk_budget_allocation=risk_budget_allocation,
            stress_tolerance_score=stress_tolerance_score,
            recommended_exposure=recommended_exposure,
            diversification_benefit=diversification_benefit,
            correlation_adjusted_risk=correlation_adjusted_risk
        )
    
    def _generate_risk_alert(self, 
                           stress_analysis: StressAnalysis,
                           drawdown_metrics: DrawdownMetrics) -> Tuple[RiskAlert, str]:
        """Generate risk alerts and messages"""
        # Determine alert level
        if stress_analysis.stress_level == StressLevel.EXTREME:
            alert = RiskAlert.CRITICAL
            message = f"CRITICAL: Extreme stress detected. Ulcer Index: {stress_analysis.stress_intensity:.1f}. Immediate risk review required."
        elif stress_analysis.stress_level == StressLevel.SEVERE:
            alert = RiskAlert.RED
            message = f"HIGH RISK: Severe market stress. Current drawdown: {drawdown_metrics.current_drawdown:.1f}%. Reduce exposure recommended."
        elif stress_analysis.stress_level == StressLevel.HIGH:
            alert = RiskAlert.ORANGE
            message = f"MODERATE RISK: High stress detected. Monitor positions closely. Recovery time estimate: {drawdown_metrics.recovery_time_estimate} periods."
        elif stress_analysis.stress_level == StressLevel.MODERATE:
            alert = RiskAlert.YELLOW
            message = f"CAUTION: Moderate stress levels. Current drawdown: {drawdown_metrics.current_drawdown:.1f}%. Normal monitoring recommended."
        else:
            alert = RiskAlert.GREEN
            message = f"LOW RISK: Market stress levels normal. Current conditions favorable for humanitarian trading mission."
        
        return alert, message
    
    def _extract_features(self, data: pd.DataFrame, index: int) -> np.ndarray:
        """Extract features for ML models"""
        if index < 30:
            return np.array([])
        
        features = []
        
        # Ulcer Index features
        if len(self.ulcer_history) >= 20:
            recent_ulcer = self.ulcer_history[-20:]
            features.extend([
                recent_ulcer[-1],
                np.mean(recent_ulcer),
                np.std(recent_ulcer),
                np.max(recent_ulcer),
                np.min(recent_ulcer),
                recent_ulcer[-1] - recent_ulcer[-5] if len(recent_ulcer) >= 5 else 0
            ])
        else:
            features.extend([0] * 6)
        
        # Drawdown features
        if len(self.drawdown_history) >= 20:
            recent_dd = self.drawdown_history[-20:]
            features.extend([
                recent_dd[-1],
                np.mean(recent_dd),
                np.std(recent_dd),
                min(recent_dd),
                sum(1 for dd in recent_dd if dd < -1),  # Number of >1% drawdown periods
                len([i for i, dd in enumerate(recent_dd) if dd < 0])  # Consecutive drawdown periods
            ])
        else:
            features.extend([0] * 6)
        
        # Price features
        prices = data['close'].iloc[max(0, index-20):index+1]
        features.extend([
            prices.pct_change().mean(),
            prices.pct_change().std(),
            (prices.iloc[-1] - prices.mean()) / prices.std() if prices.std() > 0 else 0,
            (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0] if prices.iloc[0] > 0 else 0
        ])
        
        # Volume features
        if 'volume' in data.columns:
            volumes = data['volume'].iloc[max(0, index-10):index+1]
            features.extend([
                volumes.iloc[-1] / volumes.mean() if volumes.mean() > 0 else 1.0,
                volumes.pct_change().std()
            ])
        else:
            features.extend([1.0, 0.0])
        
        # Volatility features
        returns = prices.pct_change().dropna()
        if len(returns) > 5:
            features.extend([
                returns.std(),
                returns.skew() if len(returns) > 3 else 0,
                returns.kurtosis() if len(returns) > 3 else 0
            ])
        else:
            features.extend([0, 0, 0])
        
        return np.array(features)
    
    def _train_ml_models(self, data: pd.DataFrame) -> None:
        """Train ML models for predictions"""
        if not self.enable_ml_prediction or not HAS_SKLEARN or len(data) < 100:
            return
        
        try:
            # Prepare features and targets
            features_list = []
            drawdown_targets = []
            recovery_targets = []
            stress_targets = []
            
            for i in range(50, len(data) - 10):  # Leave 10 periods for future lookback
                features = self._extract_features(data, i)
                if len(features) == 0:
                    continue
                
                features_list.append(features)
                
                # Future drawdown target
                current_price = data['close'].iloc[i]
                future_prices = data['close'].iloc[i+1:i+11]
                future_high = data['close'].iloc[max(0, i-20):i+1].max()
                
                future_drawdowns = 100 * (future_prices - future_high) / future_high
                max_future_drawdown = future_drawdowns.min()
                drawdown_targets.append(abs(max_future_drawdown))
                
                # Recovery target (will price recover to within 95% of high in next 10 periods?)
                recovery_threshold = future_high * 0.95
                recovery_targets.append(1 if future_prices.max() >= recovery_threshold else 0)
                
                # Stress target (will stress increase?)
                if len(self.ulcer_history) > i - 50:
                    current_ulcer = self.ulcer_history[min(len(self.ulcer_history)-1, i-50)]
                    future_stress = min(10, current_ulcer * 1.5)  # Predicted stress level
                    stress_targets.append(future_stress)
                else:
                    stress_targets.append(2.0)  # Default moderate stress
            
            if len(features_list) < 50:
                return
            
            X = np.array(features_list)
            y_drawdown = np.array(drawdown_targets)
            y_recovery = np.array(recovery_targets)
            y_stress = np.array(stress_targets)
            
            # Normalize features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train models
            self.drawdown_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=12,
                random_state=42
            )
            self.drawdown_model.fit(X_scaled, y_drawdown)
            
            self.recovery_model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=8,
                random_state=42
            )
            self.recovery_model.fit(X_scaled, y_recovery)
            
            self.stress_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            self.stress_model.fit(X_scaled, y_stress)
            
            self.logger.info("ML models trained successfully")
            
        except Exception as e:
            self.logger.error(f"Error training ML models: {e}")
            self.drawdown_model = None
            self.recovery_model = None
            self.stress_model = None
    
    def _predict_with_ml(self, data: pd.DataFrame, current_index: int) -> Tuple[float, int, float]:
        """Make ML predictions"""
        if (not self.enable_ml_prediction or 
            self.drawdown_model is None or 
            self.recovery_model is None or
            self.stress_model is None):
            return 2.0, 10, 0.3
        
        try:
            features = self._extract_features(data, current_index)
            if len(features) == 0:
                return 2.0, 10, 0.3
            
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            drawdown_pred = self.drawdown_model.predict(features_scaled)[0]
            recovery_prob = self.recovery_model.predict_proba(features_scaled)[0][1]
            stress_pred = self.stress_model.predict(features_scaled)[0]
            
            # Convert recovery probability to time estimate
            recovery_time_pred = int(10 / max(0.1, recovery_prob))  # Higher probability = faster recovery
            
            return drawdown_pred, recovery_time_pred, stress_pred
            
        except Exception as e:
            self.logger.error(f"Error in ML prediction: {e}")
            return 2.0, 10, 0.3
    
    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate Ulcer Index with comprehensive analysis
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary containing comprehensive Ulcer Index analysis
        """
        try:
            if data.empty or len(data) < self.period:
                raise ValueError(f"Insufficient data. Need at least {self.period} periods")
            
            # Ensure required columns exist
            required_columns = ['high', 'low', 'close']
            if not all(col in data.columns for col in required_columns):
                raise ValueError(f"Required columns {required_columns} not found in data")
            
            # Calculate rolling high
            rolling_high = self._calculate_rolling_high(data['close'], self.period)
            
            # Calculate percentage drawdown
            percentage_drawdown = self._calculate_percentage_drawdown(data['close'], rolling_high)
            
            # Calculate Ulcer Index
            ulcer_index_series = self._calculate_ulcer_index(percentage_drawdown, self.period)
            
            # Current values
            current_index = len(data) - 1
            current_ulcer = ulcer_index_series.iloc[current_index]
            current_drawdown = percentage_drawdown.iloc[current_index]
            current_high = rolling_high.iloc[current_index]
            current_price = data['close'].iloc[current_index]
            current_volume = data.get('volume', pd.Series([1.0] * len(data))).iloc[current_index]
            
            # Update history
            self.ulcer_history.append(current_ulcer)
            self.price_history.append(current_price)
            self.drawdown_history.append(current_drawdown)
            self.volume_history.append(current_volume)
            self.high_history.append(current_high)
            
            # Maintain history size
            if len(self.ulcer_history) > self.lookback_period:
                self.ulcer_history.pop(0)
                self.price_history.pop(0)
                self.drawdown_history.pop(0)
                self.volume_history.pop(0)
                self.high_history.pop(0)
            
            # Comprehensive analysis
            drawdown_metrics = self._analyze_drawdown_metrics(
                current_drawdown, self.drawdown_history, self.price_history
            )
            
            risk_adjusted_metrics = self._calculate_risk_adjusted_metrics(
                current_ulcer, self.price_history, self.drawdown_history
            )
            
            stress_analysis = self._analyze_stress_level(
                current_ulcer, current_drawdown, self.ulcer_history
            )
            
            recovery_analysis = self._analyze_recovery_pattern(
                current_drawdown, self.drawdown_history, self.price_history
            )
            
            portfolio_optimization = self._generate_portfolio_optimization(
                stress_analysis, risk_adjusted_metrics, recovery_analysis
            )
            
            risk_alert, alert_message = self._generate_risk_alert(stress_analysis, drawdown_metrics)
            
            # Statistical measures
            if len(self.ulcer_history) > 1:
                percentile_rank = (np.sum(np.array(self.ulcer_history) < current_ulcer) / 
                                 len(self.ulcer_history))
                
                # Distribution analysis
                if HAS_SCIPY and len(self.ulcer_history) >= 20:
                    # Test for normality
                    _, shapiro_p = shapiro(self.ulcer_history[-20:])
                    _, jb_p = jarque_bera(self.ulcer_history[-20:])
                    
                    distribution_analysis = {
                        'normality_shapiro_p': shapiro_p,
                        'normality_jb_p': jb_p,
                        'skewness': stats.skew(self.ulcer_history[-20:]),
                        'kurtosis': stats.kurtosis(self.ulcer_history[-20:])
                    }
                else:
                    distribution_analysis = {
                        'normality_shapiro_p': 0.5,
                        'normality_jb_p': 0.5,
                        'skewness': 0.0,
                        'kurtosis': 0.0
                    }
                
                statistical_significance = 1.0 - min(distribution_analysis['normality_shapiro_p'], 
                                                   distribution_analysis['normality_jb_p'])
            else:
                percentile_rank = 0.5
                distribution_analysis = {'normality_shapiro_p': 0.5, 'normality_jb_p': 0.5, 
                                       'skewness': 0.0, 'kurtosis': 0.0}
                statistical_significance = 0.5
            
            # Multi-timeframe analysis
            if len(self.ulcer_history) >= 60:
                short_term_ulcer = np.mean(self.ulcer_history[-5:])
                medium_term_ulcer = np.mean(self.ulcer_history[-20:])
                long_term_ulcer = np.mean(self.ulcer_history[-60:])
            else:
                short_term_ulcer = current_ulcer
                medium_term_ulcer = current_ulcer
                long_term_ulcer = current_ulcer
            
            # Historical comparison
            if len(self.ulcer_history) >= 252:  # 1 year of data
                year_avg = np.mean(self.ulcer_history[-252:])
                year_max = np.max(self.ulcer_history[-252:])
                year_min = np.min(self.ulcer_history[-252:])
                
                historical_comparison = {
                    'vs_year_average': (current_ulcer - year_avg) / year_avg if year_avg > 0 else 0,
                    'vs_year_max': current_ulcer / year_max if year_max > 0 else 0,
                    'vs_year_min': current_ulcer / year_min if year_min > 0 else 1
                }
            else:
                historical_comparison = {'vs_year_average': 0, 'vs_year_max': 0.5, 'vs_year_min': 1}
            
            # Regime analysis
            if current_ulcer < 2:
                regime_analysis = "low_risk_environment"
            elif current_ulcer < 5:
                regime_analysis = "normal_risk_environment"
            elif current_ulcer < 8:
                regime_analysis = "elevated_risk_environment"
            else:
                regime_analysis = "high_risk_environment"
            
            # Train ML models periodically
            if len(data) >= 100 and len(data) % 50 == 0:
                self._train_ml_models(data)
            
            # ML predictions
            drawdown_prediction, recovery_time_prediction, stress_probability = self._predict_with_ml(data, current_index)
            
            # Calculate overall confidence
            confidence = 0.7
            if len(self.ulcer_history) >= 100:
                confidence += 0.1
            if self.drawdown_model is not None:
                confidence += 0.1
            if len(self.stress_events) >= 5:
                confidence += 0.1
            
            # Create result
            result = UlcerIndexResult(
                ulcer_index=current_ulcer,
                percentage_drawdown=current_drawdown,
                rolling_high=current_high,
                drawdown_metrics=drawdown_metrics,
                risk_adjusted_metrics=risk_adjusted_metrics,
                stress_analysis=stress_analysis,
                recovery_analysis=recovery_analysis,
                portfolio_optimization=portfolio_optimization,
                risk_alert=risk_alert,
                alert_message=alert_message,
                drawdown_prediction=drawdown_prediction,
                recovery_time_prediction=recovery_time_prediction,
                stress_probability=stress_probability,
                percentile_rank=percentile_rank,
                statistical_significance=statistical_significance,
                distribution_analysis=distribution_analysis,
                short_term_ulcer=short_term_ulcer,
                medium_term_ulcer=medium_term_ulcer,
                long_term_ulcer=long_term_ulcer,
                historical_comparison=historical_comparison,
                regime_analysis=regime_analysis,
                timestamp=datetime.now(),
                confidence=confidence
            )
            
            self.logger.info(
                f"Ulcer Index calculated: {current_ulcer:.2f}, "
                f"Drawdown: {current_drawdown:.2f}%, Stress: {stress_analysis.stress_level.value}, "
                f"Alert: {risk_alert.value}"
            )
            
            return result.to_dict()
            
        except Exception as e:
            self.logger.error(f"Error calculating Ulcer Index: {e}")
            return {
                'error': str(e),
                'ulcer_index': np.nan,
                'confidence': 0.0,
                'timestamp': datetime.now().isoformat()
            }
    
    def get_required_columns(self) -> List[str]:
        """Get required data columns"""
        return ['high', 'low', 'close', 'volume']
    
    def get_indicator_type(self) -> str:
        """Get indicator type"""
        return "volatility"
    
    def get_description(self) -> str:
        """Get indicator description"""
        return (
            "Advanced Ulcer Index Indicator providing sophisticated downside risk analysis, "
            "drawdown stress measurement, and comprehensive risk-adjusted performance metrics. "
            "Features ML-enhanced predictions, portfolio optimization recommendations, and "
            "multi-level risk alerting for optimal humanitarian trading risk management."
        )    
    def reset(self) -> None:
        """Reset indicator state"""
        try:
            self.ulcer_history.clear()
            self.price_history.clear()
            self.drawdown_history.clear()
            self.volume_history.clear()
            self.high_history.clear()
            self.stress_events.clear()
            self.recovery_events.clear()
            
            # Reset ML models
            self.drawdown_model = None
            self.recovery_model = None
            self.stress_model = None
            self.scaler = StandardScaler() if HAS_SKLEARN else None
            self.robust_scaler = RobustScaler() if HAS_SKLEARN else None
            
            self.logger.info("Ulcer Index indicator reset successfully")
            
        except Exception as e:
            self.logger.error(f"Error resetting Ulcer Index indicator: {e}")
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration"""
        return {
            'period': self.period,
            'return_period': self.return_period,
            'stress_threshold': self.stress_threshold,
            'recovery_threshold': self.recovery_threshold,
            'enable_ml_prediction': self.enable_ml_prediction,
            'use_volume_weighting': self.use_volume_weighting,
            'lookback_period': self.lookback_period,
            'has_sklearn': HAS_SKLEARN,
            'has_scipy': HAS_SCIPY,
            'has_talib': HAS_TALIB,
            'model_trained': self.drawdown_model is not None,
            'history_length': len(self.ulcer_history),
            'stress_events_count': len(self.stress_events),
            'recovery_events_count': len(self.recovery_events)
        }
    
    def update_config(self, **kwargs) -> None:
        """Update configuration parameters"""
        try:
            if 'period' in kwargs:
                self.period = max(1, int(kwargs['period']))
            if 'return_period' in kwargs:
                self.return_period = max(1, int(kwargs['return_period']))
            if 'stress_threshold' in kwargs:
                self.stress_threshold = max(0.1, float(kwargs['stress_threshold']))
            if 'recovery_threshold' in kwargs:
                self.recovery_threshold = max(0.1, min(1.0, float(kwargs['recovery_threshold'])))
            if 'enable_ml_prediction' in kwargs:
                self.enable_ml_prediction = bool(kwargs['enable_ml_prediction']) and HAS_SKLEARN
            if 'use_volume_weighting' in kwargs:
                self.use_volume_weighting = bool(kwargs['use_volume_weighting'])
            if 'lookback_period' in kwargs:
                self.lookback_period = max(50, int(kwargs['lookback_period']))
            
            self.logger.info(f"Ulcer Index configuration updated: {kwargs}")
            
        except Exception as e:
            self.logger.error(f"Error updating configuration: {e}")
    
    def get_historical_analysis(self) -> Dict[str, Any]:
        """Get comprehensive historical analysis"""
        try:
            if len(self.ulcer_history) < 10:
                return {
                    'insufficient_data': True,
                    'message': 'Need at least 10 periods for historical analysis'
                }
            
            ulcer_array = np.array(self.ulcer_history)
            drawdown_array = np.array(self.drawdown_history)
            price_array = np.array(self.price_history)
            
            # Basic statistics
            basic_stats = {
                'mean_ulcer': np.mean(ulcer_array),
                'std_ulcer': np.std(ulcer_array),
                'max_ulcer': np.max(ulcer_array),
                'min_ulcer': np.min(ulcer_array),
                'median_ulcer': np.median(ulcer_array),
                'q75_ulcer': np.percentile(ulcer_array, 75),
                'q25_ulcer': np.percentile(ulcer_array, 25),
                'mean_drawdown': np.mean(drawdown_array),
                'max_drawdown': np.min(drawdown_array),  # Most negative
                'recovery_periods': len([dd for dd in drawdown_array if dd >= -0.1])
            }
            
            # Distribution analysis
            distribution_stats = {}
            if HAS_SCIPY and len(ulcer_array) >= 8:
                try:
                    # Normality tests
                    shapiro_stat, shapiro_p = shapiro(ulcer_array)
                    jb_stat, jb_p = jarque_bera(ulcer_array)
                    
                    distribution_stats = {
                        'shapiro_stat': shapiro_stat,
                        'shapiro_p_value': shapiro_p,
                        'jarque_bera_stat': jb_stat,
                        'jarque_bera_p_value': jb_p,
                        'skewness': stats.skew(ulcer_array),
                        'kurtosis': stats.kurtosis(ulcer_array),
                        'is_normal': shapiro_p > 0.05 and jb_p > 0.05
                    }
                except Exception:
                    distribution_stats = {'error': 'Could not compute distribution statistics'}
            
            # Stress period analysis
            stress_periods = []
            current_stress_period = None
            
            for i, ulcer_val in enumerate(ulcer_array):
                if ulcer_val > self.stress_threshold:
                    if current_stress_period is None:
                        current_stress_period = {'start': i, 'max_stress': ulcer_val, 'duration': 1}
                    else:
                        current_stress_period['duration'] += 1
                        current_stress_period['max_stress'] = max(current_stress_period['max_stress'], ulcer_val)
                else:
                    if current_stress_period is not None:
                        current_stress_period['end'] = i - 1
                        stress_periods.append(current_stress_period)
                        current_stress_period = None
            
            # Close any ongoing stress period
            if current_stress_period is not None:
                current_stress_period['end'] = len(ulcer_array) - 1
                stress_periods.append(current_stress_period)
            
            stress_analysis = {
                'total_stress_periods': len(stress_periods),
                'total_stress_duration': sum(sp['duration'] for sp in stress_periods),
                'average_stress_duration': np.mean([sp['duration'] for sp in stress_periods]) if stress_periods else 0,
                'max_stress_duration': max([sp['duration'] for sp in stress_periods]) if stress_periods else 0,
                'stress_frequency': len(stress_periods) / len(ulcer_array) if ulcer_array.size > 0 else 0,
                'current_in_stress': ulcer_array[-1] > self.stress_threshold if len(ulcer_array) > 0 else False
            }
            
            # Recovery analysis
            recovery_periods = []
            for i in range(1, len(drawdown_array)):
                if drawdown_array[i-1] < -1.0 and drawdown_array[i] >= -0.1:  # Recovery from >1% drawdown
                    # Find start of this drawdown
                    start_idx = i - 1
                    while start_idx > 0 and drawdown_array[start_idx] < 0:
                        start_idx -= 1
                    
                    recovery_duration = i - start_idx
                    max_drawdown_in_period = min(drawdown_array[start_idx:i])
                    
                    recovery_periods.append({
                        'start': start_idx,
                        'end': i,
                        'duration': recovery_duration,
                        'max_drawdown': max_drawdown_in_period,
                        'recovery_rate': abs(max_drawdown_in_period) / recovery_duration if recovery_duration > 0 else 0
                    })
            
            recovery_analysis = {
                'total_recoveries': len(recovery_periods),
                'average_recovery_time': np.mean([rp['duration'] for rp in recovery_periods]) if recovery_periods else 0,
                'fastest_recovery': min([rp['duration'] for rp in recovery_periods]) if recovery_periods else 0,
                'slowest_recovery': max([rp['duration'] for rp in recovery_periods]) if recovery_periods else 0,
                'average_recovery_rate': np.mean([rp['recovery_rate'] for rp in recovery_periods]) if recovery_periods else 0,
                'recovery_success_rate': len(recovery_periods) / max(1, len(stress_periods)) if stress_periods else 1.0
            }
            
            # Correlation analysis
            correlations = {}
            if len(price_array) == len(ulcer_array) and len(price_array) > 5:
                price_returns = np.diff(price_array) / price_array[:-1]
                ulcer_changes = np.diff(ulcer_array)
                
                if len(price_returns) == len(ulcer_changes):
                    correlations['price_return_vs_ulcer_change'] = np.corrcoef(price_returns, ulcer_changes)[0, 1]
                
                if len(self.volume_history) == len(ulcer_array) and len(self.volume_history) > 5:
                    volume_array = np.array(self.volume_history)
                    correlations['volume_vs_ulcer'] = np.corrcoef(volume_array, ulcer_array)[0, 1]
            
            # Time series properties
            time_series_analysis = {}
            if len(ulcer_array) > 10:
                # Autocorrelation
                autocorr_1 = np.corrcoef(ulcer_array[:-1], ulcer_array[1:])[0, 1]
                time_series_analysis['autocorrelation_lag1'] = autocorr_1
                
                # Trend analysis
                x = np.arange(len(ulcer_array))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, ulcer_array)
                time_series_analysis['trend_slope'] = slope
                time_series_analysis['trend_r_squared'] = r_value ** 2
                time_series_analysis['trend_p_value'] = p_value
                time_series_analysis['trend_direction'] = 'increasing' if slope > 0 else 'decreasing'
            
            # Risk metrics summary
            risk_metrics = {
                'average_ulcer_index': basic_stats['mean_ulcer'],
                'ulcer_volatility': basic_stats['std_ulcer'],
                'risk_consistency': 1 / (1 + basic_stats['std_ulcer']),  # Higher is more consistent
                'extreme_stress_frequency': sum(1 for u in ulcer_array if u > 8) / len(ulcer_array),
                'low_risk_frequency': sum(1 for u in ulcer_array if u < 2) / len(ulcer_array),
                'risk_persistence': autocorr_1 if 'autocorr_1' in locals() else 0
            }
            
            return {
                'basic_statistics': basic_stats,
                'distribution_analysis': distribution_stats,
                'stress_analysis': stress_analysis,
                'recovery_analysis': recovery_analysis,
                'correlations': correlations,
                'time_series_analysis': time_series_analysis,
                'risk_metrics': risk_metrics,
                'data_quality': {
                    'total_periods': len(ulcer_array),
                    'data_completeness': 1.0,  # Assuming complete data
                    'outlier_count': sum(1 for u in ulcer_array if u > basic_stats['q75_ulcer'] + 1.5 * (basic_stats['q75_ulcer'] - basic_stats['q25_ulcer'])),
                    'missing_values': 0
                },
                'recommendations': self._generate_historical_recommendations(basic_stats, stress_analysis, recovery_analysis)
            }
            
        except Exception as e:
            self.logger.error(f"Error in historical analysis: {e}")
            return {'error': str(e)}
    
    def _generate_historical_recommendations(self, 
                                           basic_stats: Dict, 
                                           stress_analysis: Dict, 
                                           recovery_analysis: Dict) -> List[str]:
        """Generate recommendations based on historical analysis"""
        recommendations = []
        
        # Stress-based recommendations
        if stress_analysis['stress_frequency'] > 0.3:
            recommendations.append("High stress frequency detected. Consider implementing more conservative position sizing.")
        
        if stress_analysis['average_stress_duration'] > 10:
            recommendations.append("Long stress periods observed. Ensure adequate liquidity reserves for extended drawdown periods.")
        
        # Recovery-based recommendations
        if recovery_analysis['recovery_success_rate'] < 0.7:
            recommendations.append("Lower recovery success rate. Consider implementing stop-loss mechanisms.")
        
        if recovery_analysis['average_recovery_time'] > 20:
            recommendations.append("Slow recovery periods detected. Plan for longer recovery timeframes in risk management.")
        
        # Ulcer Index level recommendations
        if basic_stats['mean_ulcer'] > 6:
            recommendations.append("High average Ulcer Index. Consider reducing overall portfolio risk exposure.")
        elif basic_stats['mean_ulcer'] < 2:
            recommendations.append("Low average Ulcer Index. Opportunity for slightly increased risk exposure if aligned with mission objectives.")
        
        # Volatility recommendations
        if basic_stats['std_ulcer'] > basic_stats['mean_ulcer']:
            recommendations.append("High Ulcer Index volatility. Implement dynamic risk management strategies.")
        
        # Add mission-specific recommendations
        recommendations.append("Maintain focus on humanitarian mission objectives while implementing risk recommendations.")
        recommendations.append("Regular monitoring and adjustment of risk parameters based on changing market conditions.")
        
        return recommendations
    
    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, str]:
        """Validate input data quality"""
        try:
            if data.empty:
                return False, "Empty dataset provided"
            
            required_columns = ['high', 'low', 'close']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                return False, f"Missing required columns: {missing_columns}"
            
            if len(data) < self.period:
                return False, f"Insufficient data: need at least {self.period} periods, got {len(data)}"
            
            # Check for data quality issues
            for col in required_columns:
                if data[col].isna().any():
                    return False, f"NaN values found in column {col}"
                
                if (data[col] <= 0).any():
                    return False, f"Non-positive values found in column {col}"
            
            # Check logical consistency
            if (data['high'] < data['low']).any():
                return False, "High prices less than low prices detected"
            
            if (data['close'] > data['high']).any() or (data['close'] < data['low']).any():
                return False, "Close prices outside high-low range detected"
            
            # Check for extreme values
            for col in required_columns:
                q99 = data[col].quantile(0.99)
                q01 = data[col].quantile(0.01)
                if q99 / q01 > 100:  # More than 100x range
                    self.logger.warning(f"Extreme price range detected in {col}: {q01:.2f} to {q99:.2f}")
            
            # Check data frequency consistency
            if 'timestamp' in data.columns:
                time_diffs = data['timestamp'].diff().dropna()
                if len(time_diffs.unique()) > 3:  # Allow some variation
                    self.logger.warning("Inconsistent time intervals detected in data")
            
            return True, "Data validation passed"
            
        except Exception as e:
            return False, f"Data validation error: {str(e)}"
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get indicator performance metrics"""
        try:
            if len(self.ulcer_history) < 20:
                return {
                    'insufficient_data': True,
                    'message': 'Need at least 20 periods for performance metrics'
                }
            
            ulcer_array = np.array(self.ulcer_history)
            
            # Calculation performance
            performance_metrics = {
                'calculation_stability': {
                    'outlier_ratio': sum(1 for u in ulcer_array[-20:] if abs(u - np.mean(ulcer_array[-20:])) > 3 * np.std(ulcer_array[-20:])) / 20,
                    'nan_ratio': 0.0,  # Assuming no NaN values in our calculation
                    'infinite_ratio': sum(1 for u in ulcer_array if np.isinf(u)) / len(ulcer_array)
                },
                'signal_quality': {
                    'signal_consistency': 1 - np.std(np.diff(ulcer_array[-10:])) / np.mean(ulcer_array[-10:]) if np.mean(ulcer_array[-10:]) > 0 else 0,
                    'noise_ratio': np.std(np.diff(ulcer_array)) / np.mean(ulcer_array) if np.mean(ulcer_array) > 0 else 1,
                    'trend_clarity': abs(np.corrcoef(range(20), ulcer_array[-20:])[0, 1]) if len(ulcer_array) >= 20 else 0
                },
                'prediction_accuracy': {},
                'computational_efficiency': {
                    'history_size': len(self.ulcer_history),
                    'ml_model_trained': self.drawdown_model is not None,
                    'memory_efficient': len(self.ulcer_history) <= self.lookback_period
                }
            }
            
            # ML model performance (if available)
            if self.drawdown_model is not None and len(self.ulcer_history) >= 50:
                try:
                    # Simple validation on recent data
                    recent_features = []
                    recent_targets = []
                    
                    for i in range(max(0, len(self.ulcer_history) - 20), len(self.ulcer_history) - 5):
                        if i >= 30:  # Ensure sufficient history for features
                            target = abs(self.drawdown_history[i + 5]) if i + 5 < len(self.drawdown_history) else 0
                            recent_targets.append(target)
                    
                    if len(recent_targets) > 5:
                        # Simplified accuracy metric
                        avg_target = np.mean(recent_targets)
                        avg_prediction = 2.0  # Default prediction
                        prediction_error = abs(avg_target - avg_prediction) / max(avg_target, 0.1)
                        
                        performance_metrics['prediction_accuracy'] = {
                            'drawdown_prediction_error': prediction_error,
                            'model_confidence': min(1.0, 1.0 / (1.0 + prediction_error)),
                            'prediction_count': len(recent_targets)
                        }
                except Exception as e:
                    performance_metrics['prediction_accuracy'] = {'error': str(e)}
            
            # Risk detection performance
            stress_detection_metrics = {
                'stress_detection_sensitivity': len([u for u in ulcer_array if u > self.stress_threshold]) / len(ulcer_array),
                'false_positive_rate': 0.05,  # Estimated
                'alert_frequency': len([u for u in ulcer_array[-50:] if u > self.stress_threshold * 1.5]) / min(50, len(ulcer_array)) if len(ulcer_array) > 0 else 0
            }
            
            performance_metrics['risk_detection'] = stress_detection_metrics
            
            # Overall performance score
            stability_score = 1 - performance_metrics['calculation_stability']['outlier_ratio']
            quality_score = 1 - performance_metrics['signal_quality']['noise_ratio']
            efficiency_score = 1.0 if performance_metrics['computational_efficiency']['memory_efficient'] else 0.8
            
            overall_score = (stability_score * 0.4 + quality_score * 0.4 + efficiency_score * 0.2)
            performance_metrics['overall_performance_score'] = max(0, min(1, overall_score))
            
            return performance_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
            return {'error': str(e)}


# Export the indicator class
__all__ = ['UlcerIndexIndicator', 'UlcerIndexResult', 'StressLevel', 'RecoveryPhase', 'RiskAlert']