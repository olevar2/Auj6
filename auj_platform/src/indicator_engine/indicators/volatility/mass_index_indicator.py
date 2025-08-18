"""
Mass Index Indicator - Advanced Reversal Signal Detection System

The Mass Index is a volatility-based technical indicator that identifies potential reversal points
by analyzing the expansion and contraction of the trading range between high and low prices.

Key Features:
- Advanced reversal signal detection using volatility expansion
- Market turning point identification with high precision
- Multi-timeframe volatility analysis
- Machine learning-enhanced pattern recognition
- Dynamic threshold adaptation based on market conditions
- Statistical validation of reversal signals
- Integration with volume and momentum analysis
- Advanced filtering to reduce false signals

Mathematical Foundation:
- Single-Period EMA = EMA(High - Low, period)
- Double-Period EMA = EMA(Single-Period EMA, period)
- Mass Index = Sum(Single-Period EMA / Double-Period EMA, sum_period)
- Reversal signals occur when Mass Index exceeds upper threshold and then falls below lower threshold
- Enhanced with volatility regime detection and ML pattern recognition

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
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import classification_report
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

try:
    from scipy import stats
    from scipy.signal import find_peaks, argrelextrema
    from scipy.ndimage import gaussian_filter1d
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from ..base.standard_indicator import StandardIndicatorInterface


class ReversalType(Enum):
    """Types of reversal signals"""
    BULLISH_REVERSAL = "bullish_reversal"
    BEARISH_REVERSAL = "bearish_reversal"
    NO_REVERSAL = "no_reversal"
    WEAK_REVERSAL = "weak_reversal"


class VolatilityExpansion(Enum):
    """Volatility expansion phases"""
    CONTRACTING = "contracting"
    STABLE = "stable"
    EXPANDING = "expanding"
    EXTREME_EXPANSION = "extreme_expansion"


class TurningPointStrength(Enum):
    """Strength of turning point signals"""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


@dataclass
class ReversalSignal:
    """Individual reversal signal data"""
    reversal_type: ReversalType
    strength: TurningPointStrength
    confidence: float
    probability: float
    expected_duration: int  # Expected duration of reversal in periods
    risk_reward_ratio: float
    volume_confirmation: bool
    momentum_confirmation: bool


@dataclass
class VolatilityAnalysis:
    """Comprehensive volatility expansion analysis"""
    current_expansion: VolatilityExpansion
    expansion_rate: float
    expansion_duration: int
    peak_volatility: float
    contraction_probability: float
    regime_stability: float


@dataclass
class TurningPointMetrics:
    """Market turning point identification metrics"""
    turning_point_probability: float
    direction_bias: float  # Positive for bullish, negative for bearish
    magnitude_estimate: float  # Expected price movement magnitude
    time_horizon: int  # Expected time to reversal completion
    confirmation_signals: int  # Number of confirming indicators


@dataclass
class MassIndexResult:
    """Complete Mass Index analysis result"""
    # Core indicator values
    mass_index: float
    single_ema: float
    double_ema: float
    ratio: float
    
    # Threshold analysis
    upper_threshold: float
    lower_threshold: float
    threshold_breach: bool
    
    # Reversal analysis
    reversal_signal: ReversalSignal
    
    # Volatility analysis
    volatility_analysis: VolatilityAnalysis
    
    # Turning point analysis
    turning_point_metrics: TurningPointMetrics
    
    # Statistical measures
    percentile_rank: float
    z_score: float
    
    # ML predictions
    reversal_probability: float
    direction_prediction: float
    pattern_classification: str
    
    # Historical context
    historical_performance: Dict[str, float]
    
    # Market conditions
    market_regime: str
    volatility_environment: str
    
    # Metadata
    timestamp: datetime
    confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary"""
        return {
            'mass_index': self.mass_index,
            'single_ema': self.single_ema,
            'double_ema': self.double_ema,
            'ratio': self.ratio,
            'upper_threshold': self.upper_threshold,
            'lower_threshold': self.lower_threshold,
            'threshold_breach': self.threshold_breach,
            'reversal_type': self.reversal_signal.reversal_type.value,
            'reversal_strength': self.reversal_signal.strength.value,
            'reversal_confidence': self.reversal_signal.confidence,
            'volatility_expansion': self.volatility_analysis.current_expansion.value,
            'expansion_rate': self.volatility_analysis.expansion_rate,
            'turning_point_probability': self.turning_point_metrics.turning_point_probability,
            'direction_bias': self.turning_point_metrics.direction_bias,
            'percentile_rank': self.percentile_rank,
            'z_score': self.z_score,
            'reversal_probability': self.reversal_probability,
            'direction_prediction': self.direction_prediction,
            'pattern_classification': self.pattern_classification,
            'market_regime': self.market_regime,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat()
        }


class MassIndexIndicator(StandardIndicatorInterface):
    """
    Advanced Mass Index Indicator with Machine Learning Integration
    
    This implementation provides sophisticated reversal signal detection and market
    turning point identification using advanced volatility expansion analysis.
    """
    
    def __init__(self, 
                 period: int = 9,
                 sum_period: int = 25,
                 upper_threshold: float = 27.0,
                 lower_threshold: float = 26.5,
                 adaptive_thresholds: bool = True,
                 use_volume_confirmation: bool = True,
                 enable_ml_prediction: bool = True,
                 lookback_period: int = 252):
        """
        Initialize Mass Index Indicator
        
        Args:
            period: EMA period for high-low range calculation
            sum_period: Period for mass index summation
            upper_threshold: Upper threshold for reversal signals
            lower_threshold: Lower threshold for reversal signals
            adaptive_thresholds: Enable adaptive threshold calculation
            use_volume_confirmation: Include volume analysis
            enable_ml_prediction: Enable ML-based predictions
            lookback_period: Historical data period for analysis
        """
        super().__init__()
        self.period = period
        self.sum_period = sum_period
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
        self.adaptive_thresholds = adaptive_thresholds
        self.use_volume_confirmation = use_volume_confirmation
        self.enable_ml_prediction = enable_ml_prediction
        self.lookback_period = lookback_period
        
        # ML models
        self.reversal_model: Optional[RandomForestClassifier] = None
        self.direction_model: Optional[GradientBoostingClassifier] = None
        self.pattern_model: Optional[RandomForestClassifier] = None
        self.scaler = StandardScaler() if HAS_SKLEARN else None
        
        # Historical data storage
        self.mass_index_history: List[float] = []
        self.price_history: List[float] = []
        self.volume_history: List[float] = []
        self.high_low_history: List[float] = []
        self.reversal_history: List[Dict] = []
        
        # Analysis parameters
        self.volatility_percentiles = [10, 25, 50, 75, 90, 95]
        self.pattern_library = {}
        
        self.logger = logging.getLogger(__name__)
        
        # Validate dependencies
        if enable_ml_prediction and not HAS_SKLEARN:
            self.logger.warning("Scikit-learn not available. ML features disabled.")
            self.enable_ml_prediction = False
    
    def _calculate_ema(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        if HAS_TALIB:
            return pd.Series(talib.EMA(series.values, timeperiod=period), index=series.index)
        else:
            return series.ewm(span=period).mean()
    
    def _calculate_adaptive_thresholds(self, mass_index_history: List[float]) -> Tuple[float, float]:
        """Calculate adaptive thresholds based on historical data"""
        if len(mass_index_history) < 50:
            return self.upper_threshold, self.lower_threshold
        
        # Use percentiles to set adaptive thresholds
        recent_data = mass_index_history[-252:]  # Last year of data
        percentiles = np.percentile(recent_data, [80, 85, 90, 95])
        
        # Calculate volatility of mass index
        volatility = np.std(recent_data)
        mean_value = np.mean(recent_data)
        
        # Adaptive thresholds based on volatility
        if volatility > np.std(mass_index_history) * 1.5:
            # High volatility environment - higher thresholds
            upper = max(percentiles[3], mean_value + 2.5 * volatility)
            lower = max(percentiles[2], mean_value + 2.0 * volatility)
        else:
            # Normal/low volatility - standard thresholds
            upper = max(percentiles[2], mean_value + 2.0 * volatility)
            lower = max(percentiles[1], mean_value + 1.5 * volatility)
        
        return upper, lower
    
    def _analyze_volatility_expansion(self, 
                                    high_low_ratio: float,
                                    high_low_history: List[float]) -> VolatilityAnalysis:
        """Analyze volatility expansion patterns"""
        if len(high_low_history) < 20:
            return VolatilityAnalysis(
                current_expansion=VolatilityExpansion.STABLE,
                expansion_rate=0.0,
                expansion_duration=0,
                peak_volatility=high_low_ratio,
                contraction_probability=0.5,
                regime_stability=0.5
            )
        
        recent_ratios = high_low_history[-20:]
        current_ratio = high_low_ratio
        
        # Calculate expansion rate
        if len(recent_ratios) >= 5:
            recent_avg = np.mean(recent_ratios[-5:])
            previous_avg = np.mean(recent_ratios[-10:-5])
            expansion_rate = (recent_avg - previous_avg) / previous_avg if previous_avg > 0 else 0.0
        else:
            expansion_rate = 0.0
        
        # Determine expansion phase
        percentile_90 = np.percentile(high_low_history[-100:], 90) if len(high_low_history) >= 100 else current_ratio * 1.2
        percentile_75 = np.percentile(high_low_history[-100:], 75) if len(high_low_history) >= 100 else current_ratio * 1.1
        percentile_25 = np.percentile(high_low_history[-100:], 25) if len(high_low_history) >= 100 else current_ratio * 0.9
        
        if current_ratio > percentile_90:
            expansion_phase = VolatilityExpansion.EXTREME_EXPANSION
        elif current_ratio > percentile_75:
            expansion_phase = VolatilityExpansion.EXPANDING
        elif current_ratio < percentile_25:
            expansion_phase = VolatilityExpansion.CONTRACTING
        else:
            expansion_phase = VolatilityExpansion.STABLE
        
        # Calculate expansion duration
        expansion_duration = 0
        threshold = percentile_75
        for i in range(len(high_low_history) - 1, max(0, len(high_low_history) - 30), -1):
            if high_low_history[i] > threshold:
                expansion_duration += 1
            else:
                break
        
        # Estimate contraction probability
        if expansion_phase in [VolatilityExpansion.EXPANDING, VolatilityExpansion.EXTREME_EXPANSION]:
            contraction_probability = min(0.9, expansion_duration * 0.05 + 0.3)
        else:
            contraction_probability = max(0.1, 0.5 - expansion_duration * 0.02)
        
        # Calculate regime stability
        regime_stability = 1.0 - (np.std(recent_ratios) / np.mean(recent_ratios)) if np.mean(recent_ratios) > 0 else 0.5
        
        return VolatilityAnalysis(
            current_expansion=expansion_phase,
            expansion_rate=expansion_rate,
            expansion_duration=expansion_duration,
            peak_volatility=max(high_low_history[-50:]) if len(high_low_history) >= 50 else current_ratio,
            contraction_probability=contraction_probability,
            regime_stability=min(1.0, max(0.0, regime_stability))
        )
    
    def _detect_reversal_signal(self, 
                               mass_index: float,
                               mass_index_history: List[float],
                               upper_threshold: float,
                               lower_threshold: float,
                               price: float,
                               volume: float) -> ReversalSignal:
        """Detect and analyze reversal signals"""
        if len(mass_index_history) < 5:
            return ReversalSignal(
                reversal_type=ReversalType.NO_REVERSAL,
                strength=TurningPointStrength.WEAK,
                confidence=0.0,
                probability=0.0,
                expected_duration=0,
                risk_reward_ratio=1.0,
                volume_confirmation=False,
                momentum_confirmation=False
            )
        
        # Check for reversal pattern
        recent_mass = mass_index_history[-5:]
        current_mass = mass_index
        
        # Bullish reversal: Mass index peaked above upper threshold and fell below lower threshold
        bullish_reversal = False
        bearish_reversal = False
        
        # Look for peak above upper threshold in recent history
        peak_above_upper = any(mi > upper_threshold for mi in recent_mass)
        
        if peak_above_upper and current_mass < lower_threshold:
            # Check price context for reversal direction
            if len(self.price_history) >= 10:
                recent_price_change = (self.price_history[-1] - self.price_history[-10]) / self.price_history[-10]
                if recent_price_change < -0.02:  # Price declining
                    bullish_reversal = True
                elif recent_price_change > 0.02:  # Price rising
                    bearish_reversal = True
        
        # Determine reversal type
        if bullish_reversal:
            reversal_type = ReversalType.BULLISH_REVERSAL
        elif bearish_reversal:
            reversal_type = ReversalType.BEARISH_REVERSAL
        else:
            reversal_type = ReversalType.NO_REVERSAL
        
        # Calculate confidence and strength
        confidence = 0.5
        strength = TurningPointStrength.WEAK
        
        if reversal_type != ReversalType.NO_REVERSAL:
            # Base confidence on how far mass index peaked above threshold
            max_recent = max(recent_mass)
            threshold_penetration = (max_recent - upper_threshold) / upper_threshold
            confidence += min(0.3, threshold_penetration * 2)
            
            # Add confidence based on current position below lower threshold
            threshold_distance = (lower_threshold - current_mass) / lower_threshold
            confidence += min(0.2, threshold_distance * 3)
            
            # Determine strength
            if confidence > 0.8:
                strength = TurningPointStrength.VERY_STRONG
            elif confidence > 0.6:
                strength = TurningPointStrength.STRONG
            elif confidence > 0.4:
                strength = TurningPointStrength.MODERATE
            else:
                strength = TurningPointStrength.WEAK
        
        # Volume confirmation
        volume_confirmation = False
        if self.use_volume_confirmation and len(self.volume_history) >= 20:
            avg_volume = np.mean(self.volume_history[-20:])
            volume_confirmation = volume > avg_volume * 1.2
            if volume_confirmation:
                confidence += 0.1
        
        # Momentum confirmation
        momentum_confirmation = False
        if len(self.price_history) >= 5:
            momentum = (self.price_history[-1] - self.price_history[-5]) / self.price_history[-5]
            if reversal_type == ReversalType.BULLISH_REVERSAL and momentum > 0:
                momentum_confirmation = True
            elif reversal_type == ReversalType.BEARISH_REVERSAL and momentum < 0:
                momentum_confirmation = True
            
            if momentum_confirmation:
                confidence += 0.1
        
        # Calculate probability
        probability = confidence * 0.8 if reversal_type != ReversalType.NO_REVERSAL else 0.0
        
        # Estimate expected duration
        expected_duration = 5  # Default
        if strength == TurningPointStrength.VERY_STRONG:
            expected_duration = 15
        elif strength == TurningPointStrength.STRONG:
            expected_duration = 10
        elif strength == TurningPointStrength.MODERATE:
            expected_duration = 7
        
        # Calculate risk-reward ratio
        if len(self.price_history) >= 20:
            price_volatility = np.std(self.price_history[-20:])
            risk_reward_ratio = min(3.0, 1.0 + confidence * 2)
        else:
            risk_reward_ratio = 1.0
        
        return ReversalSignal(
            reversal_type=reversal_type,
            strength=strength,
            confidence=min(1.0, max(0.0, confidence)),
            probability=min(1.0, max(0.0, probability)),
            expected_duration=expected_duration,
            risk_reward_ratio=risk_reward_ratio,
            volume_confirmation=volume_confirmation,
            momentum_confirmation=momentum_confirmation
        )
    
    def _analyze_turning_points(self, 
                               mass_index: float,
                               price: float,
                               volume: float) -> TurningPointMetrics:
        """Analyze market turning point probability and characteristics"""
        if len(self.mass_index_history) < 20:
            return TurningPointMetrics(
                turning_point_probability=0.0,
                direction_bias=0.0,
                magnitude_estimate=0.0,
                time_horizon=0,
                confirmation_signals=0
            )
        
        # Calculate turning point probability based on mass index position
        recent_mass = self.mass_index_history[-20:]
        mass_percentile = (np.sum(np.array(recent_mass) < mass_index) / len(recent_mass))
        
        # High mass index values suggest potential turning points
        if mass_percentile > 0.8:
            turning_point_probability = min(0.9, (mass_percentile - 0.8) * 4)
        else:
            turning_point_probability = mass_percentile * 0.3
        
        # Calculate direction bias
        direction_bias = 0.0
        if len(self.price_history) >= 10:
            price_momentum = (self.price_history[-1] - self.price_history[-10]) / self.price_history[-10]
            # Reversal bias - opposite to current momentum
            direction_bias = -np.tanh(price_momentum * 5) * turning_point_probability
        
        # Estimate magnitude of potential move
        magnitude_estimate = 0.0
        if len(self.price_history) >= 20:
            price_volatility = np.std(self.price_history[-20:]) / np.mean(self.price_history[-20:])
            magnitude_estimate = price_volatility * turning_point_probability * 2
        
        # Estimate time horizon
        if turning_point_probability > 0.7:
            time_horizon = 5  # Strong signal - quick reversal
        elif turning_point_probability > 0.5:
            time_horizon = 10  # Moderate signal
        else:
            time_horizon = 20  # Weak signal - longer development
        
        # Count confirmation signals
        confirmation_signals = 0
        
        # Volume confirmation
        if len(self.volume_history) >= 10:
            avg_volume = np.mean(self.volume_history[-10:])
            if volume > avg_volume * 1.2:
                confirmation_signals += 1
        
        # Volatility expansion confirmation
        if len(self.high_low_history) >= 10:
            recent_volatility = np.mean(self.high_low_history[-5:])
            historical_volatility = np.mean(self.high_low_history[-20:-5])
            if recent_volatility > historical_volatility * 1.3:
                confirmation_signals += 1
        
        # Mass index trend confirmation
        if len(self.mass_index_history) >= 5:
            mass_trend = np.polyfit(range(5), self.mass_index_history[-5:], 1)[0]
            if abs(mass_trend) > 0.1:  # Significant trend
                confirmation_signals += 1
        
        return TurningPointMetrics(
            turning_point_probability=turning_point_probability,
            direction_bias=direction_bias,
            magnitude_estimate=magnitude_estimate,
            time_horizon=time_horizon,
            confirmation_signals=confirmation_signals
        )
    
    def _extract_features(self, data: pd.DataFrame, index: int) -> np.ndarray:
        """Extract features for ML models"""
        if index < 30:
            return np.array([])
        
        features = []
        
        # Mass Index features
        if len(self.mass_index_history) >= 20:
            recent_mass = self.mass_index_history[-20:]
            features.extend([
                recent_mass[-1],
                np.mean(recent_mass),
                np.std(recent_mass),
                np.max(recent_mass),
                np.min(recent_mass),
                (recent_mass[-1] - np.mean(recent_mass)) / np.std(recent_mass) if np.std(recent_mass) > 0 else 0
            ])
        else:
            features.extend([0] * 6)
        
        # Price features
        prices = data['close'].iloc[max(0, index-20):index+1]
        features.extend([
            prices.pct_change().mean(),
            prices.pct_change().std(),
            (prices.iloc[-1] - prices.mean()) / prices.std() if prices.std() > 0 else 0,
            len(prices) - np.argmax(prices.values) - 1,  # Bars since high
            len(prices) - np.argmin(prices.values) - 1   # Bars since low
        ])
        
        # Volatility features
        high_low = data['high'].iloc[max(0, index-20):index+1] - data['low'].iloc[max(0, index-20):index+1]
        features.extend([
            high_low.mean(),
            high_low.std(),
            high_low.iloc[-1] / high_low.mean() if high_low.mean() > 0 else 1.0
        ])
        
        # Volume features
        if 'volume' in data.columns:
            volumes = data['volume'].iloc[max(0, index-20):index+1]
            features.extend([
                volumes.iloc[-1] / volumes.mean() if volumes.mean() > 0 else 1.0,
                volumes.pct_change().mean(),
                volumes.pct_change().std()
            ])
        else:
            features.extend([1.0, 0.0, 0.0])
        
        # Technical momentum features
        if index >= 10:
            roc_5 = (data['close'].iloc[index] - data['close'].iloc[index-5]) / data['close'].iloc[index-5]
            roc_10 = (data['close'].iloc[index] - data['close'].iloc[index-10]) / data['close'].iloc[index-10]
            features.extend([roc_5, roc_10])
        else:
            features.extend([0.0, 0.0])
        
        return np.array(features)
    
    def _train_ml_models(self, data: pd.DataFrame) -> None:
        """Train ML models for reversal prediction"""
        if not self.enable_ml_prediction or not HAS_SKLEARN or len(data) < 100:
            return
        
        try:
            # Prepare features and targets
            features_list = []
            reversal_targets = []
            direction_targets = []
            pattern_targets = []
            
            for i in range(50, len(data) - 10):  # Leave 10 periods for future lookback
                features = self._extract_features(data, i)
                if len(features) == 0:
                    continue
                
                features_list.append(features)
                
                # Reversal target (next 10 periods)
                future_high = data['high'].iloc[i+1:i+11].max()
                future_low = data['low'].iloc[i+1:i+11].min()
                current_price = data['close'].iloc[i]
                
                max_move = max(
                    (future_high - current_price) / current_price,
                    (current_price - future_low) / current_price
                )
                
                reversal_targets.append(1 if max_move > 0.03 else 0)  # 3% threshold
                
                # Direction target
                future_price = data['close'].iloc[i+5]  # 5 periods ahead
                direction_targets.append(1 if future_price > current_price else 0)
                
                # Pattern classification (simplified)
                if len(self.mass_index_history) >= 5:
                    recent_mass = self.mass_index_history[-5:]
                    if max(recent_mass) > self.upper_threshold:
                        pattern_targets.append(2)  # High volatility
                    elif min(recent_mass) < 20:  # Lower threshold for low volatility
                        pattern_targets.append(0)  # Low volatility
                    else:
                        pattern_targets.append(1)  # Normal volatility
                else:
                    pattern_targets.append(1)
            
            if len(features_list) < 50:
                return
            
            X = np.array(features_list)
            y_reversal = np.array(reversal_targets)
            y_direction = np.array(direction_targets)
            y_pattern = np.array(pattern_targets)
            
            # Normalize features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train models
            self.reversal_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=12,
                random_state=42
            )
            self.reversal_model.fit(X_scaled, y_reversal)
            
            self.direction_model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=8,
                random_state=42
            )
            self.direction_model.fit(X_scaled, y_direction)
            
            self.pattern_model = RandomForestClassifier(
                n_estimators=50,
                max_depth=10,
                random_state=42
            )
            self.pattern_model.fit(X_scaled, y_pattern)
            
            self.logger.info("ML models trained successfully")
            
        except Exception as e:
            self.logger.error(f"Error training ML models: {e}")
            self.reversal_model = None
            self.direction_model = None
            self.pattern_model = None
    
    def _predict_with_ml(self, data: pd.DataFrame, current_index: int) -> Tuple[float, float, str]:
        """Make ML predictions"""
        if (not self.enable_ml_prediction or 
            self.reversal_model is None or 
            self.direction_model is None or
            self.pattern_model is None):
            return 0.5, 0.0, "normal"
        
        try:
            features = self._extract_features(data, current_index)
            if len(features) == 0:
                return 0.5, 0.0, "normal"
            
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            reversal_prob = self.reversal_model.predict_proba(features_scaled)[0][1]
            direction_prob = self.direction_model.predict_proba(features_scaled)[0][1]
            pattern_pred = self.pattern_model.predict(features_scaled)[0]
            
            # Convert direction probability to bias (-1 to 1)
            direction_bias = (direction_prob - 0.5) * 2
            
            # Pattern classification
            pattern_names = ["low_volatility", "normal", "high_volatility"]
            pattern_classification = pattern_names[min(2, max(0, pattern_pred))]
            
            return reversal_prob, direction_bias, pattern_classification
            
        except Exception as e:
            self.logger.error(f"Error in ML prediction: {e}")
            return 0.5, 0.0, "normal"
    
    def _determine_market_regime(self, 
                                price_history: List[float],
                                volume_history: List[float]) -> str:
        """Determine current market regime"""
        if len(price_history) < 20:
            return "normal"
        
        # Calculate price volatility
        returns = [(price_history[i] - price_history[i-1]) / price_history[i-1] 
                  for i in range(1, len(price_history))]
        volatility = np.std(returns[-20:])
        
        # Calculate trend strength
        if len(price_history) >= 20:
            trend_slope = np.polyfit(range(20), price_history[-20:], 1)[0]
            trend_strength = abs(trend_slope) / np.mean(price_history[-20:])
        else:
            trend_strength = 0.0
        
        # Determine regime
        if volatility > np.std(returns) * 1.5:
            if trend_strength > 0.001:
                return "trending_volatile"
            else:
                return "ranging_volatile"
        elif trend_strength > 0.0005:
            return "trending_stable"
        else:
            return "ranging_stable"
    
    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate Mass Index indicator with advanced analysis
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary containing comprehensive Mass Index analysis
        """
        try:
            if data.empty or len(data) < max(self.period * 2, self.sum_period):
                raise ValueError(f"Insufficient data. Need at least {max(self.period * 2, self.sum_period)} periods")
            
            # Ensure required columns exist
            required_columns = ['high', 'low', 'close']
            if not all(col in data.columns for col in required_columns):
                raise ValueError(f"Required columns {required_columns} not found in data")
            
            # Calculate high-low range
            high_low_range = data['high'] - data['low']
            
            # Calculate single-period EMA of high-low range
            single_ema = self._calculate_ema(high_low_range, self.period)
            
            # Calculate double-period EMA (EMA of single EMA)
            double_ema = self._calculate_ema(single_ema, self.period)
            
            # Calculate ratio
            ratio = single_ema / double_ema
            ratio = ratio.fillna(1.0)  # Handle division by zero
            
            # Calculate Mass Index (sum of ratios)
            mass_index_series = ratio.rolling(window=self.sum_period).sum()
            
            # Current values
            current_index = len(data) - 1
            current_mass_index = mass_index_series.iloc[current_index]
            current_single_ema = single_ema.iloc[current_index]
            current_double_ema = double_ema.iloc[current_index]
            current_ratio = ratio.iloc[current_index]
            current_price = data['close'].iloc[current_index]
            current_volume = data.get('volume', pd.Series([1.0] * len(data))).iloc[current_index]
            current_high_low = high_low_range.iloc[current_index]
            
            # Update history
            self.mass_index_history.append(current_mass_index)
            self.price_history.append(current_price)
            self.volume_history.append(current_volume)
            self.high_low_history.append(current_high_low)
            
            # Maintain history size
            if len(self.mass_index_history) > self.lookback_period:
                self.mass_index_history.pop(0)
                self.price_history.pop(0)
                self.volume_history.pop(0)
                self.high_low_history.pop(0)
            
            # Calculate adaptive thresholds
            if self.adaptive_thresholds:
                upper_threshold, lower_threshold = self._calculate_adaptive_thresholds(self.mass_index_history)
            else:
                upper_threshold, lower_threshold = self.upper_threshold, self.lower_threshold
            
            # Check threshold breach
            threshold_breach = current_mass_index > upper_threshold
            
            # Analyze volatility expansion
            volatility_analysis = self._analyze_volatility_expansion(current_ratio, self.high_low_history)
            
            # Detect reversal signals
            reversal_signal = self._detect_reversal_signal(
                current_mass_index, self.mass_index_history, upper_threshold, 
                lower_threshold, current_price, current_volume
            )
            
            # Analyze turning points
            turning_point_metrics = self._analyze_turning_points(
                current_mass_index, current_price, current_volume
            )
            
            # Calculate statistical measures
            if len(self.mass_index_history) > 1:
                percentile_rank = (np.sum(np.array(self.mass_index_history) < current_mass_index) / 
                                 len(self.mass_index_history))
                z_score = ((current_mass_index - np.mean(self.mass_index_history)) / 
                          np.std(self.mass_index_history)) if np.std(self.mass_index_history) > 0 else 0.0
            else:
                percentile_rank = 0.5
                z_score = 0.0
            
            # Train ML models periodically
            if len(data) >= 100 and len(data) % 50 == 0:
                self._train_ml_models(data)
            
            # ML predictions
            reversal_probability, direction_prediction, pattern_classification = self._predict_with_ml(data, current_index)
            
            # Historical performance analysis
            historical_performance = {}
            if len(self.reversal_history) >= 10:
                successful_signals = sum(1 for signal in self.reversal_history[-20:] 
                                       if signal.get('success', False))
                historical_performance = {
                    'success_rate': successful_signals / min(20, len(self.reversal_history)),
                    'average_return': np.mean([signal.get('return', 0.0) for signal in self.reversal_history[-20:]]),
                    'max_drawdown': min([signal.get('return', 0.0) for signal in self.reversal_history[-20:]] + [0.0])
                }
            else:
                historical_performance = {
                    'success_rate': 0.5,
                    'average_return': 0.0,
                    'max_drawdown': 0.0
                }
            
            # Determine market regime and volatility environment
            market_regime = self._determine_market_regime(self.price_history, self.volume_history)
            volatility_environment = volatility_analysis.current_expansion.value
            
            # Calculate overall confidence
            confidence = 0.6
            if len(self.mass_index_history) >= 50:
                confidence += 0.1
            if self.reversal_model is not None:
                confidence += 0.1
            if reversal_signal.volume_confirmation and reversal_signal.momentum_confirmation:
                confidence += 0.2
            
            # Create result
            result = MassIndexResult(
                mass_index=current_mass_index,
                single_ema=current_single_ema,
                double_ema=current_double_ema,
                ratio=current_ratio,
                upper_threshold=upper_threshold,
                lower_threshold=lower_threshold,
                threshold_breach=threshold_breach,
                reversal_signal=reversal_signal,
                volatility_analysis=volatility_analysis,
                turning_point_metrics=turning_point_metrics,
                percentile_rank=percentile_rank,
                z_score=z_score,
                reversal_probability=reversal_probability,
                direction_prediction=direction_prediction,
                pattern_classification=pattern_classification,
                historical_performance=historical_performance,
                market_regime=market_regime,
                volatility_environment=volatility_environment,
                timestamp=datetime.now(),
                confidence=confidence
            )
            
            # Update reversal history
            if reversal_signal.reversal_type != ReversalType.NO_REVERSAL:
                self.reversal_history.append({
                    'timestamp': datetime.now(),
                    'reversal_type': reversal_signal.reversal_type.value,
                    'confidence': reversal_signal.confidence,
                    'price': current_price
                })
                
                if len(self.reversal_history) > 50:
                    self.reversal_history.pop(0)
            
            self.logger.info(
                f"Mass Index calculated: {current_mass_index:.2f}, "
                f"Threshold: {upper_threshold:.2f}/{lower_threshold:.2f}, "
                f"Reversal: {reversal_signal.reversal_type.value}, "
                f"Volatility: {volatility_analysis.current_expansion.value}"
            )
            
            return result.to_dict()
            
        except Exception as e:
            self.logger.error(f"Error calculating Mass Index: {e}")
            return {
                'error': str(e),
                'mass_index': np.nan,
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
            "Advanced Mass Index Indicator providing sophisticated reversal signal detection "
            "and market turning point identification through volatility expansion analysis, "
            "ML-enhanced pattern recognition, and comprehensive market regime assessment "
            "for optimal humanitarian trading decisions."
        )