"""
Advanced Chaikin Volatility Indicator with Sophisticated Gap Analysis

This implementation features:
- High-low gap volatility measurement with multiple calculation methods
- Exponential smoothing with adaptive parameters
- Volatility regime detection and classification
- Trend analysis integration
- Gap expansion/contraction analysis
- Statistical significance testing
- Production-ready error handling

Author: AUJ Platform Development Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from enum import Enum
from scipy.stats import norm, jarque_bera
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

from ..base.standard_indicator import StandardIndicatorInterface
from ....core.exceptions import IndicatorCalculationError
from ....core.signal_type import SignalType


class VolatilityMethod(Enum):
    """Volatility calculation methods"""
    HIGH_LOW_GAP = "high_low_gap"
    TRUE_RANGE = "true_range"
    PERCENTAGE_RANGE = "percentage_range"
    LOG_RANGE = "log_range"
    ADAPTIVE_RANGE = "adaptive_range"


class VolatilityRegime(Enum):
    """Volatility regime classifications"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class ChaikinConfig:
    """Configuration for Chaikin Volatility calculation"""
    period: int = 10
    smoothing_period: int = 10
    method: VolatilityMethod = VolatilityMethod.HIGH_LOW_GAP
    adaptive_smoothing: bool = True
    regime_detection: bool = True
    trend_integration: bool = True
    statistical_validation: bool = True
    outlier_detection: bool = True
    min_periods: int = 20


@dataclass
class VolatilityMeasure:
    """Volatility measurement structure"""
    raw_volatility: float
    smoothed_volatility: float
    volatility_change: float
    expansion_rate: float
    regime: VolatilityRegime
    significance: float
    outlier_flag: bool


@dataclass
class ChaikinAnalysis:
    """Analysis results for Chaikin Volatility"""
    current_volatility: float
    volatility_trend: str
    regime_classification: VolatilityRegime
    expansion_signal: bool
    contraction_signal: bool
    breakout_probability: float
    mean_reversion_probability: float
    volatility_percentile: float


class ChaikinVolatilityIndicator(StandardIndicatorInterface):
    """
    Advanced Chaikin Volatility Indicator with Sophisticated Analysis
    
    Features:
    - Multiple volatility calculation methods
    - Adaptive exponential smoothing
    - Volatility regime detection
    - Statistical significance testing
    - Trend integration
    """
    
    def __init__(self, config: Optional[ChaikinConfig] = None):
        super().__init__()
        self.config = config or ChaikinConfig()
        self.logger = logging.getLogger(__name__)
        
        # Historical data storage
        self.volatility_history: List[float] = []
        self.smoothed_history: List[float] = []
        self.regime_history: List[VolatilityRegime] = []
        self.price_history: List[Tuple[float, float, float, float]] = []  # OHLC
        
        # Adaptive parameters
        self.adaptive_alpha: float = 2.0 / (self.config.smoothing_period + 1)
        self.current_ema: float = 0.0
        
        # Statistical tracking
        self.regime_clusters: Optional[KMeans] = None
        self.volatility_distribution: List[float] = []
        
        # Performance metrics
        self.calculation_count = 0
        self.error_count = 0
        
    def get_required_data_types(self) -> List[str]:
        """Return required data types"""
        return ["ohlcv"]
    
    def get_required_columns(self) -> List[str]:
        """Return required columns"""
        return ["open", "high", "low", "close", "volume"]
    
    def calculate(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Calculate advanced Chaikin Volatility with sophisticated analysis
        
        Args:
            data: Dictionary containing OHLCV data
            
        Returns:
            Dictionary containing Chaikin Volatility results
        """
        try:
            self.calculation_count += 1
            self.logger.debug(f"Calculating Chaikin Volatility (calculation #{self.calculation_count})")
            
            # Validate input data
            ohlcv_data = self._validate_input_data(data)
            
            if len(ohlcv_data) < self.config.min_periods:
                raise IndicatorCalculationError(
                    f"Insufficient data: need at least {self.config.min_periods} periods, got {len(ohlcv_data)}"
                )
            
            # Extract OHLCV data
            opens = ohlcv_data['open'].values
            highs = ohlcv_data['high'].values
            lows = ohlcv_data['low'].values
            closes = ohlcv_data['close'].values
            volumes = ohlcv_data['volume'].values
            
            # Calculate volatility measure
            volatility_measure = self._calculate_volatility_measure(opens, highs, lows, closes)
            
            # Perform comprehensive analysis
            analysis = self._analyze_volatility(volatility_measure, highs, lows, closes)
            
            # Update historical data
            self._update_history(opens[-1], highs[-1], lows[-1], closes[-1], volatility_measure)
            
            # Update regime classification if enabled
            if self.config.regime_detection:
                self._update_regime_classification()
            
            return self._format_output(volatility_measure, analysis, closes[-1])
            
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Error calculating Chaikin Volatility: {str(e)}")
            raise IndicatorCalculationError(f"Chaikin Volatility calculation failed: {str(e)}")
    
    def _validate_input_data(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Validate and extract OHLCV data"""
        if "ohlcv" not in data:
            raise IndicatorCalculationError("OHLCV data not found in input")
        
        ohlcv_data = data["ohlcv"]
        required_columns = self.get_required_columns()
        
        for col in required_columns:
            if col not in ohlcv_data.columns:
                raise IndicatorCalculationError(f"Required column '{col}' not found in data")
        
        # Check for NaN values
        if ohlcv_data[required_columns].isnull().any().any():
            self.logger.warning("NaN values detected in input data, forward filling...")
            ohlcv_data = ohlcv_data.fillna(method='ffill')
        
        return ohlcv_data
    
    def _calculate_volatility_measure(self, opens: np.ndarray, highs: np.ndarray, 
                                    lows: np.ndarray, closes: np.ndarray) -> VolatilityMeasure:
        """Calculate volatility measure using selected method"""
        
        # Calculate raw volatility based on method
        if self.config.method == VolatilityMethod.HIGH_LOW_GAP:
            raw_volatility = self._calculate_high_low_gap(highs, lows)
        elif self.config.method == VolatilityMethod.TRUE_RANGE:
            raw_volatility = self._calculate_true_range_volatility(highs, lows, closes)
        elif self.config.method == VolatilityMethod.PERCENTAGE_RANGE:
            raw_volatility = self._calculate_percentage_range(highs, lows, closes)
        elif self.config.method == VolatilityMethod.LOG_RANGE:
            raw_volatility = self._calculate_log_range(highs, lows)
        elif self.config.method == VolatilityMethod.ADAPTIVE_RANGE:
            raw_volatility = self._calculate_adaptive_range(opens, highs, lows, closes)
        else:
            raw_volatility = self._calculate_high_low_gap(highs, lows)
        
        # Apply smoothing
        smoothed_volatility = self._apply_smoothing(raw_volatility)
        
        # Calculate volatility change
        volatility_change = self._calculate_volatility_change(smoothed_volatility)
        
        # Calculate expansion rate
        expansion_rate = self._calculate_expansion_rate(raw_volatility, smoothed_volatility)
        
        # Detect regime
        regime = self._detect_volatility_regime(smoothed_volatility)
        
        # Calculate statistical significance
        significance = self._calculate_statistical_significance(raw_volatility)
        
        # Detect outliers
        outlier_flag = self._detect_outlier(raw_volatility)
        
        return VolatilityMeasure(
            raw_volatility=raw_volatility,
            smoothed_volatility=smoothed_volatility,
            volatility_change=volatility_change,
            expansion_rate=expansion_rate,
            regime=regime,
            significance=significance,
            outlier_flag=outlier_flag
        )
    
    def _calculate_high_low_gap(self, highs: np.ndarray, lows: np.ndarray) -> float:
        """Calculate traditional high-low gap volatility"""
        if len(highs) < self.config.period:
            if len(highs) > 0:
                return (highs[-1] - lows[-1]) / lows[-1] if lows[-1] > 0 else 0.0
            return 0.0
        
        # Calculate rolling high-low gaps
        gaps = []
        for i in range(len(highs) - self.config.period + 1, len(highs)):
            window_highs = highs[i - self.config.period + 1:i + 1]
            window_lows = lows[i - self.config.period + 1:i + 1]
            
            gap = np.mean(window_highs - window_lows)
            gaps.append(gap)
        
        return np.mean(gaps) if gaps else 0.0
    
    def _calculate_true_range_volatility(self, highs: np.ndarray, lows: np.ndarray, 
                                       closes: np.ndarray) -> float:
        """Calculate volatility using True Range"""
        if len(highs) < 2:
            return 0.0
        
        true_ranges = []
        for i in range(1, len(highs)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            true_ranges.append(tr)
        
        if len(true_ranges) < self.config.period:
            return np.mean(true_ranges) if true_ranges else 0.0
        
        return np.mean(true_ranges[-self.config.period:])
    
    def _calculate_percentage_range(self, highs: np.ndarray, lows: np.ndarray, 
                                  closes: np.ndarray) -> float:
        """Calculate percentage-based range volatility"""
        if len(highs) < self.config.period or closes[-1] <= 0:
            return 0.0
        
        period_end = len(highs)
        period_start = max(0, period_end - self.config.period)
        
        window_highs = highs[period_start:period_end]
        window_lows = lows[period_start:period_end]
        window_closes = closes[period_start:period_end]
        
        percentage_ranges = []
        for h, l, c in zip(window_highs, window_lows, window_closes):
            if c > 0:
                pct_range = ((h - l) / c) * 100
                percentage_ranges.append(pct_range)
        
        return np.mean(percentage_ranges) if percentage_ranges else 0.0
    
    def _calculate_log_range(self, highs: np.ndarray, lows: np.ndarray) -> float:
        """Calculate logarithmic range volatility"""
        if len(highs) < self.config.period:
            return 0.0
        
        period_end = len(highs)
        period_start = max(0, period_end - self.config.period)
        
        window_highs = highs[period_start:period_end]
        window_lows = lows[period_start:period_end]
        
        log_ranges = []
        for h, l in zip(window_highs, window_lows):
            if h > 0 and l > 0:
                log_range = np.log(h / l)
                log_ranges.append(log_range)
        
        return np.mean(log_ranges) if log_ranges else 0.0
    
    def _calculate_adaptive_range(self, opens: np.ndarray, highs: np.ndarray, 
                                lows: np.ndarray, closes: np.ndarray) -> float:
        """Calculate adaptive range based on market conditions"""
        if len(highs) < self.config.period:
            return 0.0
        
        # Combine multiple range measures
        hl_range = self._calculate_high_low_gap(highs, lows)
        tr_range = self._calculate_true_range_volatility(highs, lows, closes)
        pct_range = self._calculate_percentage_range(highs, lows, closes)
        
        # Calculate market efficiency ratio
        efficiency = self._calculate_efficiency_ratio(closes)
        
        # Weight different measures based on efficiency
        if efficiency > 0.7:  # Trending market
            adaptive_range = 0.6 * tr_range + 0.3 * hl_range + 0.1 * pct_range
        elif efficiency < 0.3:  # Ranging market
            adaptive_range = 0.3 * tr_range + 0.5 * hl_range + 0.2 * pct_range
        else:  # Mixed market
            adaptive_range = 0.4 * tr_range + 0.4 * hl_range + 0.2 * pct_range
        
        return adaptive_range
    
    def _calculate_efficiency_ratio(self, closes: np.ndarray) -> float:
        """Calculate market efficiency ratio"""
        if len(closes) < self.config.period:
            return 0.5
        
        period_end = len(closes)
        period_start = max(0, period_end - self.config.period)
        window_closes = closes[period_start:period_end]
        
        # Direction movement
        direction = abs(window_closes[-1] - window_closes[0])
        
        # Volatility (sum of absolute changes)
        volatility = np.sum(np.abs(np.diff(window_closes)))
        
        if volatility == 0:
            return 1.0
        
        efficiency = direction / volatility
        return min(1.0, efficiency)
    
    def _apply_smoothing(self, raw_volatility: float) -> float:
        """Apply exponential smoothing to volatility"""
        if self.current_ema == 0:
            self.current_ema = raw_volatility
            return raw_volatility
        
        # Adaptive alpha if enabled
        if self.config.adaptive_smoothing:
            alpha = self._calculate_adaptive_alpha(raw_volatility)
        else:
            alpha = self.adaptive_alpha
        
        # Exponential moving average
        self.current_ema = alpha * raw_volatility + (1 - alpha) * self.current_ema
        return self.current_ema
    
    def _calculate_adaptive_alpha(self, current_volatility: float) -> float:
        """Calculate adaptive smoothing factor"""
        if not self.volatility_history:
            return self.adaptive_alpha
        
        # Calculate recent volatility change
        recent_volatility = np.mean(self.volatility_history[-5:]) if len(self.volatility_history) >= 5 else self.volatility_history[-1]
        
        if recent_volatility == 0:
            return self.adaptive_alpha
        
        volatility_ratio = current_volatility / recent_volatility
        
        # Increase alpha for higher volatility changes
        if volatility_ratio > 1.5:
            alpha = min(0.9, self.adaptive_alpha * 2)
        elif volatility_ratio < 0.7:
            alpha = max(0.1, self.adaptive_alpha * 0.5)
        else:
            alpha = self.adaptive_alpha
        
        return alpha
    
    def _calculate_volatility_change(self, current_smoothed: float) -> float:
        """Calculate percentage change in volatility"""
        if not self.smoothed_history:
            return 0.0
        
        previous_smoothed = self.smoothed_history[-1]
        if previous_smoothed == 0:
            return 0.0
        
        return ((current_smoothed - previous_smoothed) / previous_smoothed) * 100
    
    def _calculate_expansion_rate(self, raw_volatility: float, smoothed_volatility: float) -> float:
        """Calculate volatility expansion rate"""
        if smoothed_volatility == 0:
            return 0.0
        
        return (raw_volatility - smoothed_volatility) / smoothed_volatility    
    def _detect_volatility_regime(self, smoothed_volatility: float) -> VolatilityRegime:
        """Detect current volatility regime"""
        if not self.volatility_distribution:
            return VolatilityRegime.NORMAL
        
        try:
            # Calculate percentiles of historical volatility
            percentiles = np.percentile(self.volatility_distribution, [25, 50, 75, 90])
            
            if smoothed_volatility <= percentiles[0]:
                return VolatilityRegime.LOW
            elif smoothed_volatility <= percentiles[2]:
                return VolatilityRegime.NORMAL
            elif smoothed_volatility <= percentiles[3]:
                return VolatilityRegime.HIGH
            else:
                return VolatilityRegime.EXTREME
                
        except Exception as e:
            self.logger.warning(f"Regime detection failed: {e}")
            return VolatilityRegime.NORMAL
    
    def _calculate_statistical_significance(self, raw_volatility: float) -> float:
        """Calculate statistical significance of current volatility"""
        if len(self.volatility_history) < 10:
            return 0.5
        
        try:
            recent_volatility = self.volatility_history[-10:]
            mean_vol = np.mean(recent_volatility)
            std_vol = np.std(recent_volatility)
            
            if std_vol == 0:
                return 0.5
            
            # Z-score calculation
            z_score = abs(raw_volatility - mean_vol) / std_vol
            
            # Convert to probability (two-tailed test)
            p_value = 2 * (1 - norm.cdf(abs(z_score)))
            significance = 1 - p_value
            
            return min(0.99, max(0.01, significance))
            
        except Exception as e:
            self.logger.warning(f"Statistical significance calculation failed: {e}")
            return 0.5
    
    def _detect_outlier(self, raw_volatility: float) -> bool:
        """Detect if current volatility is an outlier"""
        if len(self.volatility_history) < 20:
            return False
        
        try:
            recent_volatility = self.volatility_history[-20:]
            q1 = np.percentile(recent_volatility, 25)
            q3 = np.percentile(recent_volatility, 75)
            iqr = q3 - q1
            
            # Outlier detection using IQR method
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            return raw_volatility < lower_bound or raw_volatility > upper_bound
            
        except Exception:
            return False
    
    def _analyze_volatility(self, volatility_measure: VolatilityMeasure,
                          highs: np.ndarray, lows: np.ndarray, 
                          closes: np.ndarray) -> ChaikinAnalysis:
        """Perform comprehensive volatility analysis"""
        
        # Determine volatility trend
        volatility_trend = self._determine_volatility_trend(volatility_measure)
        
        # Detect expansion/contraction signals
        expansion_signal, contraction_signal = self._detect_expansion_contraction_signals(volatility_measure)
        
        # Calculate breakout probability
        breakout_prob = self._calculate_breakout_probability(volatility_measure, highs, lows, closes)
        
        # Calculate mean reversion probability
        mean_reversion_prob = self._calculate_mean_reversion_probability(volatility_measure)
        
        # Calculate volatility percentile
        volatility_percentile = self._calculate_volatility_percentile(volatility_measure.smoothed_volatility)
        
        return ChaikinAnalysis(
            current_volatility=volatility_measure.smoothed_volatility,
            volatility_trend=volatility_trend,
            regime_classification=volatility_measure.regime,
            expansion_signal=expansion_signal,
            contraction_signal=contraction_signal,
            breakout_probability=breakout_prob,
            mean_reversion_probability=mean_reversion_prob,
            volatility_percentile=volatility_percentile
        )
    
    def _determine_volatility_trend(self, volatility_measure: VolatilityMeasure) -> str:
        """Determine the current volatility trend"""
        if len(self.smoothed_history) < 3:
            return "neutral"
        
        recent_values = self.smoothed_history[-3:] + [volatility_measure.smoothed_volatility]
        
        # Calculate trend slope
        x = np.arange(len(recent_values))
        slope = np.polyfit(x, recent_values, 1)[0]
        
        # Determine trend strength
        if slope > 0.01:
            return "increasing"
        elif slope < -0.01:
            return "decreasing"
        else:
            return "neutral"
    
    def _detect_expansion_contraction_signals(self, volatility_measure: VolatilityMeasure) -> Tuple[bool, bool]:
        """Detect volatility expansion and contraction signals"""
        expansion_signal = False
        contraction_signal = False
        
        # Expansion signal criteria
        if (volatility_measure.expansion_rate > 0.2 and 
            volatility_measure.volatility_change > 15 and
            volatility_measure.regime in [VolatilityRegime.HIGH, VolatilityRegime.EXTREME]):
            expansion_signal = True
        
        # Contraction signal criteria
        if (volatility_measure.expansion_rate < -0.2 and 
            volatility_measure.volatility_change < -15 and
            volatility_measure.regime == VolatilityRegime.LOW):
            contraction_signal = True
        
        return expansion_signal, contraction_signal
    
    def _calculate_breakout_probability(self, volatility_measure: VolatilityMeasure,
                                      highs: np.ndarray, lows: np.ndarray, 
                                      closes: np.ndarray) -> float:
        """Calculate probability of price breakout based on volatility"""
        try:
            base_probability = 0.3
            
            # Increase probability for volatility expansion
            if volatility_measure.expansion_rate > 0.3:
                base_probability += 0.3
            
            # Increase probability for regime change
            if volatility_measure.regime == VolatilityRegime.EXTREME:
                base_probability += 0.2
            elif volatility_measure.regime == VolatilityRegime.LOW:
                base_probability -= 0.1
            
            # Adjust for significance
            base_probability += volatility_measure.significance * 0.2
            
            # Analyze price compression
            if len(closes) >= 10:
                recent_range = np.max(highs[-10:]) - np.min(lows[-10:])
                avg_range = np.mean(highs[-20:] - lows[-20:]) if len(closes) >= 20 else recent_range
                
                if avg_range > 0:
                    compression_ratio = recent_range / avg_range
                    if compression_ratio < 0.8:  # Price compression
                        base_probability += 0.2
            
            return min(0.9, max(0.1, base_probability))
            
        except Exception as e:
            self.logger.warning(f"Breakout probability calculation failed: {e}")
            return 0.3
    
    def _calculate_mean_reversion_probability(self, volatility_measure: VolatilityMeasure) -> float:
        """Calculate probability of volatility mean reversion"""
        try:
            base_probability = 0.5
            
            # Higher probability for extreme volatility
            if volatility_measure.regime == VolatilityRegime.EXTREME:
                base_probability = 0.8
            elif volatility_measure.regime == VolatilityRegime.LOW:
                base_probability = 0.7
            
            # Adjust for outliers
            if volatility_measure.outlier_flag:
                base_probability += 0.2
            
            # Adjust for expansion rate
            if abs(volatility_measure.expansion_rate) > 0.5:
                base_probability += 0.1
            
            return min(0.95, max(0.05, base_probability))
            
        except Exception:
            return 0.5
    
    def _calculate_volatility_percentile(self, current_volatility: float) -> float:
        """Calculate percentile rank of current volatility"""
        if not self.volatility_distribution:
            return 50.0
        
        try:
            # Calculate percentile rank
            sorted_volatility = np.sort(self.volatility_distribution)
            percentile = (np.searchsorted(sorted_volatility, current_volatility) / len(sorted_volatility)) * 100
            
            return min(100.0, max(0.0, percentile))
            
        except Exception:
            return 50.0
    
    def _update_history(self, open_price: float, high: float, low: float, 
                       close: float, volatility_measure: VolatilityMeasure):
        """Update historical data"""
        self.price_history.append((open_price, high, low, close))
        self.volatility_history.append(volatility_measure.raw_volatility)
        self.smoothed_history.append(volatility_measure.smoothed_volatility)
        self.regime_history.append(volatility_measure.regime)
        self.volatility_distribution.append(volatility_measure.smoothed_volatility)
        
        # Keep only recent history
        max_history = 500
        if len(self.volatility_history) > max_history:
            self.price_history = self.price_history[-max_history:]
            self.volatility_history = self.volatility_history[-max_history:]
            self.smoothed_history = self.smoothed_history[-max_history:]
            self.regime_history = self.regime_history[-max_history:]
            self.volatility_distribution = self.volatility_distribution[-max_history:]
    
    def _update_regime_classification(self):
        """Update volatility regime classification using clustering"""
        if len(self.volatility_distribution) < 50:
            return
        
        try:
            # Use K-means clustering to identify regimes
            volatility_data = np.array(self.volatility_distribution[-100:]).reshape(-1, 1)
            
            if self.regime_clusters is None:
                self.regime_clusters = KMeans(n_clusters=4, random_state=42, n_init=10)
            
            self.regime_clusters.fit(volatility_data)
            
            # Update regime boundaries based on cluster centers
            centers = sorted(self.regime_clusters.cluster_centers_.flatten())
            
            self.logger.debug(f"Updated regime centers: {centers}")
            
        except Exception as e:
            self.logger.warning(f"Regime classification update failed: {e}")
    
    def _format_output(self, volatility_measure: VolatilityMeasure, 
                      analysis: ChaikinAnalysis, current_price: float) -> Dict[str, Any]:
        """Format the output result"""
        
        # Determine signal type and strength
        signal_type = SignalType.NEUTRAL
        signal_strength = 0.5
        
        # Signal logic based on volatility analysis
        if analysis.expansion_signal:
            if analysis.breakout_probability > 0.7:
                signal_type = SignalType.BUY if analysis.volatility_trend == "increasing" else SignalType.NEUTRAL
                signal_strength = analysis.breakout_probability * 0.8
        elif analysis.contraction_signal:
            if analysis.mean_reversion_probability > 0.7:
                signal_type = SignalType.NEUTRAL  # Range-bound market expected
                signal_strength = 0.3
        
        # Adjust strength based on significance and regime
        signal_strength *= volatility_measure.significance
        
        if volatility_measure.regime == VolatilityRegime.EXTREME:
            signal_strength *= 1.2
        elif volatility_measure.regime == VolatilityRegime.LOW:
            signal_strength *= 0.7
        
        signal_strength = min(1.0, signal_strength)
        
        return {
            "signal_type": signal_type,
            "signal_strength": signal_strength,
            "values": {
                "raw_volatility": volatility_measure.raw_volatility,
                "smoothed_volatility": volatility_measure.smoothed_volatility,
                "volatility_change": volatility_measure.volatility_change,
                "expansion_rate": volatility_measure.expansion_rate,
                "current_price": current_price
            },
            "analysis": {
                "volatility_trend": analysis.volatility_trend,
                "regime_classification": analysis.regime_classification.value,
                "expansion_signal": analysis.expansion_signal,
                "contraction_signal": analysis.contraction_signal,
                "breakout_probability": analysis.breakout_probability,
                "mean_reversion_probability": analysis.mean_reversion_probability,
                "volatility_percentile": analysis.volatility_percentile
            },
            "metadata": {
                "method": self.config.method.value,
                "period": self.config.period,
                "smoothing_period": self.config.smoothing_period,
                "adaptive_smoothing": self.config.adaptive_smoothing,
                "significance": volatility_measure.significance,
                "outlier_flag": volatility_measure.outlier_flag,
                "calculation_count": self.calculation_count,
                "error_rate": self.error_count / max(1, self.calculation_count)
            },
            "regime_info": {
                "current_regime": volatility_measure.regime.value,
                "regime_history": [r.value for r in self.regime_history[-10:]]
            }
        }
    
    def get_signal_type(self, data: Dict[str, pd.DataFrame]) -> SignalType:
        """Get signal type based on Chaikin Volatility analysis"""
        try:
            result = self.calculate(data)
            return result["signal_type"]
        except Exception:
            return SignalType.NEUTRAL
    
    def get_signal_strength(self, data: Dict[str, pd.DataFrame]) -> float:
        """Get signal strength"""
        try:
            result = self.calculate(data)
            return result["signal_strength"]
        except Exception:
            return 0.0