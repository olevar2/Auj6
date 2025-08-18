"""
Relative Volatility Index (RVI) Indicator - Advanced Market Timing System

The Relative Volatility Index combines the power of RSI methodology with volatility analysis,
providing superior market timing signals by measuring the relative strength of volatility
rather than just price movements.

Key Features:
- Advanced RSI-based volatility analysis with adaptive periods
- Multi-timeframe volatility momentum detection
- Machine learning-enhanced signal generation
- Dynamic overbought/oversold level adaptation
- Volatility regime classification and analysis
- Volume-weighted volatility calculations
- Statistical significance testing for signals
- Advanced filtering to reduce false signals

Mathematical Foundation:
- Standard Deviation of Up/Down Moves
- RSI applied to volatility rather than price
- RVI = 100 * (Up Volatility RS) / (Up Volatility RS + 1)
- where RS = Average Up Volatility / Average Down Volatility
- Enhanced with ML models for pattern recognition

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
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_squared_error, classification_report
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
    from scipy.signal import find_peaks, butter, filtfilt
    from scipy.optimize import minimize_scalar
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from ..base.standard_indicator import StandardIndicatorInterface


class RVISignal(Enum):
    """RVI signal types"""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    NEUTRAL = "neutral"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


class VolatilityMomentum(Enum):
    """Volatility momentum phases"""
    ACCELERATING_UP = "accelerating_up"
    DECELERATING_UP = "decelerating_up"
    STABLE = "stable"
    DECELERATING_DOWN = "decelerating_down"
    ACCELERATING_DOWN = "accelerating_down"


class MarketCondition(Enum):
    """Market condition classification"""
    TRENDING_HIGH_VOL = "trending_high_vol"
    TRENDING_LOW_VOL = "trending_low_vol"
    RANGING_HIGH_VOL = "ranging_high_vol"
    RANGING_LOW_VOL = "ranging_low_vol"
    BREAKOUT_PENDING = "breakout_pending"


@dataclass
class VolatilityMetrics:
    """Comprehensive volatility analysis"""
    up_volatility: float
    down_volatility: float
    volatility_ratio: float
    volatility_momentum: VolatilityMomentum
    normalized_volatility: float
    volatility_percentile: float


@dataclass
class RVILevels:
    """Dynamic RVI overbought/oversold levels"""
    overbought: float
    oversold: float
    extreme_overbought: float
    extreme_oversold: float
    dynamic_adjustment: float


@dataclass
class DivergenceAnalysis:
    """Price-RVI divergence detection"""
    has_divergence: bool
    divergence_type: str  # "bullish", "bearish", "hidden_bullish", "hidden_bearish"
    strength: float
    duration: int
    probability: float


@dataclass
class RVIResult:
    """Complete RVI analysis result"""
    # Core indicator values
    rvi: float
    rvi_ema: float
    volatility_rs: float
    
    # Volatility components
    volatility_metrics: VolatilityMetrics
    
    # Signal analysis
    signal: RVISignal
    signal_strength: float
    signal_confidence: float
    
    # Dynamic levels
    levels: RVILevels
    
    # Divergence analysis
    divergence: DivergenceAnalysis
    
    # Market condition
    market_condition: MarketCondition
    trend_alignment: float
    
    # ML predictions
    price_direction_probability: float
    volatility_forecast: float
    pattern_classification: str
    
    # Statistical measures
    percentile_rank: float
    z_score: float
    statistical_significance: float
    
    # Multi-timeframe analysis
    short_term_rvi: float
    long_term_rvi: float
    timeframe_consistency: float
    
    # Risk metrics
    volatility_risk: float
    signal_reliability: float
    
    # Metadata
    timestamp: datetime
    confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary"""
        return {
            'rvi': self.rvi,
            'rvi_ema': self.rvi_ema,
            'volatility_rs': self.volatility_rs,
            'up_volatility': self.volatility_metrics.up_volatility,
            'down_volatility': self.volatility_metrics.down_volatility,
            'volatility_ratio': self.volatility_metrics.volatility_ratio,
            'volatility_momentum': self.volatility_metrics.volatility_momentum.value,
            'signal': self.signal.value,
            'signal_strength': self.signal_strength,
            'signal_confidence': self.signal_confidence,
            'overbought_level': self.levels.overbought,
            'oversold_level': self.levels.oversold,
            'has_divergence': self.divergence.has_divergence,
            'divergence_type': self.divergence.divergence_type,
            'market_condition': self.market_condition.value,
            'trend_alignment': self.trend_alignment,
            'price_direction_probability': self.price_direction_probability,
            'volatility_forecast': self.volatility_forecast,
            'pattern_classification': self.pattern_classification,
            'percentile_rank': self.percentile_rank,
            'z_score': self.z_score,
            'short_term_rvi': self.short_term_rvi,
            'long_term_rvi': self.long_term_rvi,
            'timeframe_consistency': self.timeframe_consistency,
            'volatility_risk': self.volatility_risk,
            'signal_reliability': self.signal_reliability,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat()
        }


class RelativeVolatilityIndexIndicator(StandardIndicatorInterface):
    """
    Advanced Relative Volatility Index Indicator with Machine Learning Integration
    
    This implementation combines RSI methodology with volatility analysis to provide
    superior market timing signals and volatility-based trend analysis.
    """
    
    def __init__(self, 
                 period: int = 14,
                 volatility_period: int = 10,
                 smoothing_period: int = 3,
                 overbought: float = 70.0,
                 oversold: float = 30.0,
                 adaptive_levels: bool = True,
                 use_volume_weighting: bool = True,
                 enable_ml_prediction: bool = True,
                 lookback_period: int = 252):
        """
        Initialize Relative Volatility Index Indicator
        
        Args:
            period: Period for RVI calculation
            volatility_period: Period for volatility calculation
            smoothing_period: Period for RVI smoothing
            overbought: Base overbought level
            oversold: Base oversold level
            adaptive_levels: Enable adaptive level calculation
            use_volume_weighting: Include volume in calculations
            enable_ml_prediction: Enable ML-based predictions
            lookback_period: Historical data period for analysis
        """
        super().__init__()
        self.period = period
        self.volatility_period = volatility_period
        self.smoothing_period = smoothing_period
        self.overbought = overbought
        self.oversold = oversold
        self.adaptive_levels = adaptive_levels
        self.use_volume_weighting = use_volume_weighting
        self.enable_ml_prediction = enable_ml_prediction
        self.lookback_period = lookback_period
        
        # ML models
        self.direction_model: Optional[GradientBoostingClassifier] = None
        self.volatility_model: Optional[RandomForestRegressor] = None
        self.pattern_model: Optional[RandomForestRegressor] = None
        self.scaler = StandardScaler() if HAS_SKLEARN else None
        self.volatility_scaler = MinMaxScaler() if HAS_SKLEARN else None
        
        # Historical data storage
        self.rvi_history: List[float] = []
        self.price_history: List[float] = []
        self.volume_history: List[float] = []
        self.volatility_history: List[float] = []
        self.up_volatility_history: List[float] = []
        self.down_volatility_history: List[float] = []
        
        # Analysis parameters
        self.extreme_levels = {
            'extreme_overbought': 80.0,
            'extreme_oversold': 20.0
        }
        
        self.logger = logging.getLogger(__name__)
        
        # Validate dependencies
        if enable_ml_prediction and not HAS_SKLEARN:
            self.logger.warning("Scikit-learn not available. ML features disabled.")
            self.enable_ml_prediction = False
    
    def _calculate_volatility(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate price volatility (standard deviation of returns)"""
        returns = data['close'].pct_change()
        
        if self.use_volume_weighting and 'volume' in data.columns:
            # Volume-weighted volatility
            weights = data['volume'] / data['volume'].rolling(window=period).sum()
            weighted_returns = returns * weights.fillna(0)
            volatility = weighted_returns.rolling(window=period).std()
        else:
            volatility = returns.rolling(window=period).std()
        
        return volatility.fillna(0)
    
    def _calculate_directional_volatility(self, data: pd.DataFrame, period: int) -> Tuple[pd.Series, pd.Series]:
        """Calculate up and down volatility separately"""
        returns = data['close'].pct_change()
        
        # Separate up and down moves
        up_moves = returns.where(returns > 0, 0)
        down_moves = abs(returns.where(returns < 0, 0))
        
        if self.use_volume_weighting and 'volume' in data.columns:
            # Volume weighting
            volume_weights = data['volume'] / data['volume'].rolling(window=period).sum()
            volume_weights = volume_weights.fillna(1.0 / period)
            
            up_volatility = (up_moves * volume_weights).rolling(window=period).std()
            down_volatility = (down_moves * volume_weights).rolling(window=period).std()
        else:
            up_volatility = up_moves.rolling(window=period).std()
            down_volatility = down_moves.rolling(window=period).std()
        
        return up_volatility.fillna(0), down_volatility.fillna(0)
    
    def _calculate_rvi(self, up_volatility: pd.Series, down_volatility: pd.Series, period: int) -> pd.Series:
        """Calculate Relative Volatility Index"""
        # Calculate average up and down volatility
        avg_up_vol = up_volatility.rolling(window=period).mean()
        avg_down_vol = down_volatility.rolling(window=period).mean()
        
        # Calculate RS (Relative Strength)
        rs = avg_up_vol / (avg_down_vol + 1e-10)  # Add small epsilon to avoid division by zero
        
        # Calculate RVI
        rvi = 100 * rs / (rs + 1)
        
        return rvi.fillna(50)  # Fill NaN with neutral value
    
    def _calculate_adaptive_levels(self, rvi_history: List[float]) -> RVILevels:
        """Calculate adaptive overbought/oversold levels"""
        if len(rvi_history) < 50:
            return RVILevels(
                overbought=self.overbought,
                oversold=self.oversold,
                extreme_overbought=self.extreme_levels['extreme_overbought'],
                extreme_oversold=self.extreme_levels['extreme_oversold'],
                dynamic_adjustment=0.0
            )
        
        # Calculate percentiles from recent history
        recent_data = rvi_history[-252:] if len(rvi_history) >= 252 else rvi_history
        percentiles = np.percentile(recent_data, [10, 20, 80, 90])
        
        # Calculate volatility of RVI
        rvi_volatility = np.std(recent_data)
        mean_rvi = np.mean(recent_data)
        
        # Dynamic adjustment based on volatility
        adjustment_factor = min(1.5, max(0.5, rvi_volatility / 10))
        
        # Adaptive levels
        oversold = max(15, min(40, percentiles[0] + adjustment_factor * 5))
        overbought = min(85, max(60, percentiles[3] - adjustment_factor * 5))
        extreme_oversold = max(5, oversold - 10)
        extreme_overbought = min(95, overbought + 10)
        
        dynamic_adjustment = adjustment_factor - 1.0
        
        return RVILevels(
            overbought=overbought,
            oversold=oversold,
            extreme_overbought=extreme_overbought,
            extreme_oversold=extreme_oversold,
            dynamic_adjustment=dynamic_adjustment
        )
    
    def _analyze_volatility_momentum(self, 
                                   volatility_history: List[float],
                                   current_volatility: float) -> VolatilityMomentum:
        """Analyze volatility momentum patterns"""
        if len(volatility_history) < 10:
            return VolatilityMomentum.STABLE
        
        # Calculate short and long term volatility trends
        short_term = volatility_history[-5:]
        medium_term = volatility_history[-10:]
        
        short_trend = np.polyfit(range(len(short_term)), short_term, 1)[0]
        medium_trend = np.polyfit(range(len(medium_term)), medium_term, 1)[0]
        
        # Determine momentum phase
        if short_trend > 0 and medium_trend > 0:
            if short_trend > medium_trend * 1.2:
                return VolatilityMomentum.ACCELERATING_UP
            else:
                return VolatilityMomentum.DECELERATING_UP
        elif short_trend < 0 and medium_trend < 0:
            if abs(short_trend) > abs(medium_trend) * 1.2:
                return VolatilityMomentum.ACCELERATING_DOWN
            else:
                return VolatilityMomentum.DECELERATING_DOWN
        else:
            return VolatilityMomentum.STABLE
    
    def _detect_divergence(self, 
                          price_history: List[float],
                          rvi_history: List[float]) -> DivergenceAnalysis:
        """Detect price-RVI divergences"""
        if len(price_history) < 20 or len(rvi_history) < 20:
            return DivergenceAnalysis(
                has_divergence=False,
                divergence_type="none",
                strength=0.0,
                duration=0,
                probability=0.0
            )
        
        # Look for divergence in last 20 periods
        lookback = min(20, len(price_history))
        recent_prices = price_history[-lookback:]
        recent_rvi = rvi_history[-lookback:]
        
        if HAS_SCIPY:
            # Find local extremes
            price_peaks, _ = find_peaks(recent_prices, distance=3)
            price_troughs, _ = find_peaks([-p for p in recent_prices], distance=3)
            price_troughs = [len(recent_prices) - 1 - t for t in price_troughs[::-1]]
            
            rvi_peaks, _ = find_peaks(recent_rvi, distance=3)
            rvi_troughs, _ = find_peaks([-r for r in recent_rvi], distance=3)
            rvi_troughs = [len(recent_rvi) - 1 - t for t in rvi_troughs[::-1]]
            
            # Check for divergence patterns
            has_divergence = False
            divergence_type = "none"
            strength = 0.0
            duration = 0
            
            # Bullish divergence: price makes lower lows, RVI makes higher lows
            if len(price_troughs) >= 2 and len(rvi_troughs) >= 2:
                last_price_trough = price_troughs[-1]
                prev_price_trough = price_troughs[-2]
                
                if (recent_prices[last_price_trough] < recent_prices[prev_price_trough] and
                    len(rvi_troughs) >= 2):
                    # Find corresponding RVI troughs
                    rvi_trough_candidates = [t for t in rvi_troughs if abs(t - last_price_trough) <= 3]
                    if rvi_trough_candidates:
                        last_rvi_trough = rvi_trough_candidates[-1]
                        prev_rvi_trough_candidates = [t for t in rvi_troughs if t < last_rvi_trough - 5]
                        
                        if prev_rvi_trough_candidates:
                            prev_rvi_trough = prev_rvi_trough_candidates[-1]
                            
                            if recent_rvi[last_rvi_trough] > recent_rvi[prev_rvi_trough]:
                                has_divergence = True
                                divergence_type = "bullish"
                                strength = (recent_rvi[last_rvi_trough] - recent_rvi[prev_rvi_trough]) / 10
                                duration = last_price_trough - prev_price_trough
            
            # Bearish divergence: price makes higher highs, RVI makes lower highs
            if len(price_peaks) >= 2 and len(rvi_peaks) >= 2 and not has_divergence:
                last_price_peak = price_peaks[-1]
                prev_price_peak = price_peaks[-2]
                
                if (recent_prices[last_price_peak] > recent_prices[prev_price_peak] and
                    len(rvi_peaks) >= 2):
                    # Find corresponding RVI peaks
                    rvi_peak_candidates = [p for p in rvi_peaks if abs(p - last_price_peak) <= 3]
                    if rvi_peak_candidates:
                        last_rvi_peak = rvi_peak_candidates[-1]
                        prev_rvi_peak_candidates = [p for p in rvi_peaks if p < last_rvi_peak - 5]
                        
                        if prev_rvi_peak_candidates:
                            prev_rvi_peak = prev_rvi_peak_candidates[-1]
                            
                            if recent_rvi[last_rvi_peak] < recent_rvi[prev_rvi_peak]:
                                has_divergence = True
                                divergence_type = "bearish"
                                strength = (recent_rvi[prev_rvi_peak] - recent_rvi[last_rvi_peak]) / 10
                                duration = last_price_peak - prev_price_peak
            
            # Calculate probability
            probability = min(0.9, strength * 0.5 + (duration / 20) * 0.3) if has_divergence else 0.0
            
            return DivergenceAnalysis(
                has_divergence=has_divergence,
                divergence_type=divergence_type,
                strength=min(1.0, max(0.0, strength)),
                duration=duration,
                probability=probability
            )
        
        # Fallback without scipy
        return DivergenceAnalysis(
            has_divergence=False,
            divergence_type="none",
            strength=0.0,
            duration=0,
            probability=0.0
        )
    
    def _classify_market_condition(self, 
                                  rvi: float,
                                  volatility: float,
                                  price_history: List[float]) -> MarketCondition:
        """Classify current market condition"""
        if len(price_history) < 20:
            return MarketCondition.RANGING_LOW_VOL
        
        # Calculate trend strength
        recent_prices = price_history[-20:]
        trend_slope = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
        price_range = max(recent_prices) - min(recent_prices)
        trend_strength = abs(trend_slope) / (price_range / len(recent_prices))
        
        # Determine if trending or ranging
        is_trending = trend_strength > 0.5
        
        # Determine volatility level
        if len(self.volatility_history) >= 50:
            vol_percentile = (np.sum(np.array(self.volatility_history[-50:]) < volatility) / 
                            len(self.volatility_history[-50:]))
        else:
            vol_percentile = 0.5
        
        is_high_vol = vol_percentile > 0.7
        
        # Check for breakout conditions
        is_near_extreme = rvi > 80 or rvi < 20
        
        if is_near_extreme and not is_trending:
            return MarketCondition.BREAKOUT_PENDING
        elif is_trending and is_high_vol:
            return MarketCondition.TRENDING_HIGH_VOL
        elif is_trending and not is_high_vol:
            return MarketCondition.TRENDING_LOW_VOL
        elif not is_trending and is_high_vol:
            return MarketCondition.RANGING_HIGH_VOL
        else:
            return MarketCondition.RANGING_LOW_VOL
    
    def _generate_signal(self, 
                        rvi: float,
                        rvi_ema: float,
                        levels: RVILevels,
                        divergence: DivergenceAnalysis,
                        market_condition: MarketCondition) -> Tuple[RVISignal, float, float]:
        """Generate trading signal with strength and confidence"""
        signal = RVISignal.NEUTRAL
        strength = 0.0
        confidence = 0.5
        
        # Base signal from RVI levels
        if rvi >= levels.extreme_overbought:
            signal = RVISignal.STRONG_SELL
            strength = min(1.0, (rvi - levels.extreme_overbought) / 10)
        elif rvi >= levels.overbought:
            signal = RVISignal.SELL
            strength = min(1.0, (rvi - levels.overbought) / 10)
        elif rvi <= levels.extreme_oversold:
            signal = RVISignal.STRONG_BUY
            strength = min(1.0, (levels.extreme_oversold - rvi) / 10)
        elif rvi <= levels.oversold:
            signal = RVISignal.BUY
            strength = min(1.0, (levels.oversold - rvi) / 10)
        
        # Adjust based on RVI momentum (EMA direction)
        rvi_momentum = 1 if rvi > rvi_ema else -1
        
        if signal in [RVISignal.BUY, RVISignal.STRONG_BUY] and rvi_momentum > 0:
            confidence += 0.2
        elif signal in [RVISignal.SELL, RVISignal.STRONG_SELL] and rvi_momentum < 0:
            confidence += 0.2
        elif signal != RVISignal.NEUTRAL:
            confidence -= 0.1
        
        # Adjust based on divergence
        if divergence.has_divergence:
            if (divergence.divergence_type == "bullish" and 
                signal in [RVISignal.BUY, RVISignal.STRONG_BUY]):
                confidence += divergence.probability * 0.3
                strength += divergence.strength * 0.2
            elif (divergence.divergence_type == "bearish" and 
                  signal in [RVISignal.SELL, RVISignal.STRONG_SELL]):
                confidence += divergence.probability * 0.3
                strength += divergence.strength * 0.2
        
        # Adjust based on market condition
        if market_condition == MarketCondition.BREAKOUT_PENDING:
            confidence += 0.15
        elif market_condition in [MarketCondition.TRENDING_HIGH_VOL, MarketCondition.TRENDING_LOW_VOL]:
            if signal != RVISignal.NEUTRAL:
                confidence += 0.1
        
        return signal, min(1.0, max(0.0, strength)), min(1.0, max(0.0, confidence))
    
    def _extract_features(self, data: pd.DataFrame, index: int) -> np.ndarray:
        """Extract features for ML models"""
        if index < 30:
            return np.array([])
        
        features = []
        
        # RVI features
        if len(self.rvi_history) >= 20:
            recent_rvi = self.rvi_history[-20:]
            features.extend([
                recent_rvi[-1],
                np.mean(recent_rvi),
                np.std(recent_rvi),
                np.max(recent_rvi),
                np.min(recent_rvi),
                recent_rvi[-1] - recent_rvi[-5] if len(recent_rvi) >= 5 else 0,
                (recent_rvi[-1] - np.mean(recent_rvi)) / np.std(recent_rvi) if np.std(recent_rvi) > 0 else 0
            ])
        else:
            features.extend([0] * 7)
        
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
        if len(self.volatility_history) >= 10:
            recent_vol = self.volatility_history[-10:]
            features.extend([
                recent_vol[-1],
                np.mean(recent_vol),
                recent_vol[-1] / np.mean(recent_vol) if np.mean(recent_vol) > 0 else 1.0
            ])
        else:
            features.extend([0] * 3)
        
        # Volume features
        if 'volume' in data.columns:
            volumes = data['volume'].iloc[max(0, index-10):index+1]
            features.extend([
                volumes.iloc[-1] / volumes.mean() if volumes.mean() > 0 else 1.0,
                volumes.pct_change().mean(),
                volumes.pct_change().std()
            ])
        else:
            features.extend([1.0, 0.0, 0.0])
        
        return np.array(features)
    
    def _train_ml_models(self, data: pd.DataFrame) -> None:
        """Train ML models for predictions"""
        if not self.enable_ml_prediction or not HAS_SKLEARN or len(data) < 100:
            return
        
        try:
            # Prepare features and targets
            features_list = []
            direction_targets = []
            volatility_targets = []
            pattern_targets = []
            
            for i in range(50, len(data) - 5):  # Leave 5 periods for future lookback
                features = self._extract_features(data, i)
                if len(features) == 0:
                    continue
                
                features_list.append(features)
                
                # Direction target (next 5 periods)
                current_price = data['close'].iloc[i]
                future_price = data['close'].iloc[i+5]
                direction_targets.append(1 if future_price > current_price else 0)
                
                # Volatility target
                future_volatility = data['close'].iloc[i+1:i+6].pct_change().std()
                volatility_targets.append(future_volatility)
                
                # Pattern target (RVI level)
                if len(self.rvi_history) > i - 50:
                    current_rvi = self.rvi_history[min(len(self.rvi_history)-1, i-50)]
                    if current_rvi > 70:
                        pattern_targets.append(2)  # Overbought
                    elif current_rvi < 30:
                        pattern_targets.append(0)  # Oversold
                    else:
                        pattern_targets.append(1)  # Neutral
                else:
                    pattern_targets.append(1)
            
            if len(features_list) < 50:
                return
            
            X = np.array(features_list)
            y_direction = np.array(direction_targets)
            y_volatility = np.array(volatility_targets)
            y_pattern = np.array(pattern_targets)
            
            # Normalize features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train models
            self.direction_model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=8,
                random_state=42
            )
            self.direction_model.fit(X_scaled, y_direction)
            
            self.volatility_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            # Normalize volatility targets
            y_volatility_scaled = self.volatility_scaler.fit_transform(y_volatility.reshape(-1, 1)).ravel()
            self.volatility_model.fit(X_scaled, y_volatility_scaled)
            
            self.pattern_model = RandomForestRegressor(
                n_estimators=50,
                max_depth=8,
                random_state=42
            )
            self.pattern_model.fit(X_scaled, y_pattern)
            
            self.logger.info("ML models trained successfully")
            
        except Exception as e:
            self.logger.error(f"Error training ML models: {e}")
            self.direction_model = None
            self.volatility_model = None
            self.pattern_model = None
    
    def _predict_with_ml(self, data: pd.DataFrame, current_index: int) -> Tuple[float, float, str]:
        """Make ML predictions"""
        if (not self.enable_ml_prediction or 
            self.direction_model is None or 
            self.volatility_model is None or
            self.pattern_model is None):
            return 0.5, 0.02, "neutral"
        
        try:
            features = self._extract_features(data, current_index)
            if len(features) == 0:
                return 0.5, 0.02, "neutral"
            
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            direction_prob = self.direction_model.predict_proba(features_scaled)[0][1]
            volatility_pred_scaled = self.volatility_model.predict(features_scaled)[0]
            volatility_pred = self.volatility_scaler.inverse_transform([[volatility_pred_scaled]])[0][0]
            pattern_pred = self.pattern_model.predict(features_scaled)[0]
            
            # Pattern classification
            if pattern_pred > 1.5:
                pattern_class = "overbought"
            elif pattern_pred < 0.5:
                pattern_class = "oversold"
            else:
                pattern_class = "neutral"
            
            return direction_prob, volatility_pred, pattern_class
            
        except Exception as e:
            self.logger.error(f"Error in ML prediction: {e}")
            return 0.5, 0.02, "neutral"
    
    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate Relative Volatility Index with advanced analysis
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary containing comprehensive RVI analysis
        """
        try:
            if data.empty or len(data) < max(self.period, self.volatility_period) * 2:
                raise ValueError(f"Insufficient data. Need at least {max(self.period, self.volatility_period) * 2} periods")
            
            # Ensure required columns exist
            required_columns = ['open', 'high', 'low', 'close']
            if not all(col in data.columns for col in required_columns):
                raise ValueError(f"Required columns {required_columns} not found in data")
            
            # Calculate directional volatility
            up_volatility, down_volatility = self._calculate_directional_volatility(data, self.volatility_period)
            
            # Calculate RVI
            rvi_series = self._calculate_rvi(up_volatility, down_volatility, self.period)
            
            # Calculate smoothed RVI
            rvi_ema = rvi_series.ewm(span=self.smoothing_period).mean()
            
            # Current values
            current_index = len(data) - 1
            current_rvi = rvi_series.iloc[current_index]
            current_rvi_ema = rvi_ema.iloc[current_index]
            current_price = data['close'].iloc[current_index]
            current_volume = data.get('volume', pd.Series([1.0] * len(data))).iloc[current_index]
            current_up_vol = up_volatility.iloc[current_index]
            current_down_vol = down_volatility.iloc[current_index]
            current_volatility = self._calculate_volatility(data, self.volatility_period).iloc[current_index]
            
            # Calculate volatility RS
            volatility_rs = current_up_vol / (current_down_vol + 1e-10)
            
            # Update history
            self.rvi_history.append(current_rvi)
            self.price_history.append(current_price)
            self.volume_history.append(current_volume)
            self.volatility_history.append(current_volatility)
            self.up_volatility_history.append(current_up_vol)
            self.down_volatility_history.append(current_down_vol)
            
            # Maintain history size
            if len(self.rvi_history) > self.lookback_period:
                self.rvi_history.pop(0)
                self.price_history.pop(0)
                self.volume_history.pop(0)
                self.volatility_history.pop(0)
                self.up_volatility_history.pop(0)
                self.down_volatility_history.pop(0)
            
            # Calculate adaptive levels
            if self.adaptive_levels:
                levels = self._calculate_adaptive_levels(self.rvi_history)
            else:
                levels = RVILevels(
                    overbought=self.overbought,
                    oversold=self.oversold,
                    extreme_overbought=self.extreme_levels['extreme_overbought'],
                    extreme_oversold=self.extreme_levels['extreme_oversold'],
                    dynamic_adjustment=0.0
                )
            
            # Volatility analysis
            volatility_ratio = current_up_vol / (current_down_vol + 1e-10)
            volatility_momentum = self._analyze_volatility_momentum(self.volatility_history, current_volatility)
            
            # Normalize volatility
            if len(self.volatility_history) > 1:
                volatility_percentile = (np.sum(np.array(self.volatility_history) < current_volatility) / 
                                       len(self.volatility_history))
                normalized_volatility = current_volatility / np.mean(self.volatility_history)
            else:
                volatility_percentile = 0.5
                normalized_volatility = 1.0
            
            volatility_metrics = VolatilityMetrics(
                up_volatility=current_up_vol,
                down_volatility=current_down_vol,
                volatility_ratio=volatility_ratio,
                volatility_momentum=volatility_momentum,
                normalized_volatility=normalized_volatility,
                volatility_percentile=volatility_percentile
            )
            
            # Divergence analysis
            divergence = self._detect_divergence(self.price_history, self.rvi_history)
            
            # Market condition classification
            market_condition = self._classify_market_condition(current_rvi, current_volatility, self.price_history)
            
            # Generate signal
            signal, signal_strength, signal_confidence = self._generate_signal(
                current_rvi, current_rvi_ema, levels, divergence, market_condition
            )
            
            # Calculate trend alignment
            if len(self.price_history) >= 10:
                price_trend = (self.price_history[-1] - self.price_history[-10]) / self.price_history[-10]
                rvi_trend = current_rvi - 50  # Distance from neutral
                trend_alignment = np.tanh(price_trend * rvi_trend / 100)
            else:
                trend_alignment = 0.0
            
            # Statistical measures
            if len(self.rvi_history) > 1:
                percentile_rank = (np.sum(np.array(self.rvi_history) < current_rvi) / 
                                 len(self.rvi_history))
                z_score = ((current_rvi - np.mean(self.rvi_history)) / 
                          np.std(self.rvi_history)) if np.std(self.rvi_history) > 0 else 0.0
                
                # Statistical significance test
                if len(self.rvi_history) >= 30:
                    # T-test for significance
                    recent_rvi = self.rvi_history[-10:]
                    historical_rvi = self.rvi_history[-30:-10]
                    
                    if HAS_SCIPY:
                        t_stat, p_value = stats.ttest_ind(recent_rvi, historical_rvi)
                        statistical_significance = 1.0 - p_value
                    else:
                        statistical_significance = 0.5
                else:
                    statistical_significance = 0.5
            else:
                percentile_rank = 0.5
                z_score = 0.0
                statistical_significance = 0.5
            
            # Multi-timeframe analysis
            if len(self.rvi_history) >= 20:
                short_term_rvi = np.mean(self.rvi_history[-5:])
                long_term_rvi = np.mean(self.rvi_history[-20:])
                timeframe_consistency = 1.0 - abs(short_term_rvi - long_term_rvi) / 50
            else:
                short_term_rvi = current_rvi
                long_term_rvi = current_rvi
                timeframe_consistency = 1.0
            
            # Train ML models periodically
            if len(data) >= 100 and len(data) % 50 == 0:
                self._train_ml_models(data)
            
            # ML predictions
            price_direction_probability, volatility_forecast, pattern_classification = self._predict_with_ml(data, current_index)
            
            # Risk metrics
            volatility_risk = min(1.0, current_volatility * 100)  # Scale volatility to 0-1 range
            
            # Signal reliability based on various factors
            signal_reliability = 0.5
            if divergence.has_divergence:
                signal_reliability += divergence.probability * 0.2
            if market_condition in [MarketCondition.TRENDING_HIGH_VOL, MarketCondition.TRENDING_LOW_VOL]:
                signal_reliability += 0.1
            if abs(z_score) > 1:
                signal_reliability += 0.1
            if timeframe_consistency > 0.8:
                signal_reliability += 0.1
            
            signal_reliability = min(1.0, max(0.0, signal_reliability))
            
            # Calculate overall confidence
            confidence = 0.6
            if len(self.rvi_history) >= 50:
                confidence += 0.1
            if self.direction_model is not None:
                confidence += 0.1
            if signal_confidence > 0.7:
                confidence += 0.1
            if statistical_significance > 0.95:
                confidence += 0.1
            
            # Create result
            result = RVIResult(
                rvi=current_rvi,
                rvi_ema=current_rvi_ema,
                volatility_rs=volatility_rs,
                volatility_metrics=volatility_metrics,
                signal=signal,
                signal_strength=signal_strength,
                signal_confidence=signal_confidence,
                levels=levels,
                divergence=divergence,
                market_condition=market_condition,
                trend_alignment=trend_alignment,
                price_direction_probability=price_direction_probability,
                volatility_forecast=volatility_forecast,
                pattern_classification=pattern_classification,
                percentile_rank=percentile_rank,
                z_score=z_score,
                statistical_significance=statistical_significance,
                short_term_rvi=short_term_rvi,
                long_term_rvi=long_term_rvi,
                timeframe_consistency=timeframe_consistency,
                volatility_risk=volatility_risk,
                signal_reliability=signal_reliability,
                timestamp=datetime.now(),
                confidence=confidence
            )
            
            self.logger.info(
                f"RVI calculated: {current_rvi:.2f}, Signal: {signal.value}, "
                f"Levels: {levels.overbought:.1f}/{levels.oversold:.1f}, "
                f"Market: {market_condition.value}"
            )
            
            return result.to_dict()
            
        except Exception as e:
            self.logger.error(f"Error calculating RVI: {e}")
            return {
                'error': str(e),
                'rvi': np.nan,
                'confidence': 0.0,
                'timestamp': datetime.now().isoformat()
            }
    
    def get_required_columns(self) -> List[str]:
        """Get required data columns"""
        return ['open', 'high', 'low', 'close', 'volume']
    
    def get_indicator_type(self) -> str:
        """Get indicator type"""
        return "volatility"
    
    def get_description(self) -> str:
        """Get indicator description"""
        return (
            "Advanced Relative Volatility Index combining RSI methodology with volatility analysis "
            "for superior market timing signals. Features adaptive levels, divergence detection, "
            "ML-enhanced predictions, and comprehensive market condition analysis for optimal "
            "humanitarian trading decisions."
        )