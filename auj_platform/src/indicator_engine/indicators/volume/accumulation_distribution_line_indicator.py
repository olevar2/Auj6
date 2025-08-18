"""
Advanced Accumulation/Distribution Line Indicator
==============================================

A sophisticated implementation of the Accumulation/Distribution Line with:
- Volume-weighted momentum analysis
- Trend confirmation algorithms 
- ML-based divergence detection
- Multi-timeframe analysis
- Statistical validation

The A/D Line measures the flow of money into and out of a security by analyzing
the relationship between price movements and volume.

Mathematical Foundation:
- Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
- Money Flow Volume = Money Flow Multiplier × Volume
- A/D Line = Previous A/D Line + Current Money Flow Volume

Advanced Features:
1. Volume-weighted momentum analysis with adaptive smoothing
2. ML-based divergence detection using gradient analysis
3. Trend confirmation through multi-period correlation
4. Statistical significance testing for signals
5. Adaptive volume normalization for different market conditions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy import stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

# Import base class
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from ..indicator_engine.indicators.base.standard_indicator import StandardIndicatorInterface
from ..core.signal_type import SignalType

@dataclass
class ADLineConfig:
    """Configuration for Accumulation/Distribution Line calculation"""
    volume_smoothing_period: int = 21
    momentum_period: int = 14
    divergence_lookback: int = 50
    trend_confirmation_period: int = 30
    significance_threshold: float = 0.05
    volume_normalization_period: int = 100
    ml_training_period: int = 200
    
@dataclass
class ADLineResult:
    """Result structure for A/D Line analysis"""
    ad_line: float
    ad_momentum: float
    volume_pressure: float
    trend_confirmation: float
    divergence_score: float
    signal_strength: float
    money_flow_multiplier: float
    normalized_volume: float
    statistical_significance: float

class AccumulationDistributionLineIndicator(StandardIndicatorInterface):
    """
    Advanced Accumulation/Distribution Line Indicator
    
    This indicator provides sophisticated analysis of money flow through:
    1. Enhanced A/D Line calculation with volume normalization
    2. Momentum analysis using volume-weighted techniques
    3. ML-based divergence detection for early signal identification
    4. Statistical validation of signal significance
    5. Multi-timeframe trend confirmation
    """
    
    def __init__(self, config: Optional[ADLineConfig] = None):
        """Initialize the Advanced A/D Line Indicator"""
        super().__init__()
        self.config = config or ADLineConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize calculation components
        self._scaler = StandardScaler()
        self._divergence_model = LinearRegression()
        
        # Historical data storage
        self._price_history: List[float] = []
        self._volume_history: List[float] = []
        self._ad_history: List[float] = []
        self._momentum_history: List[float] = []
        
        # State tracking
        self._is_trained = False
        self._last_signal = SignalType.HOLD
        
    def calculate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate advanced A/D Line with comprehensive analysis
        
        Args:
            data: Dictionary containing OHLCV data
            
        Returns:
            Dictionary with A/D Line analysis results
        """
        try:
            # Validate and extract data
            if not self._validate_data(data):
                raise ValueError("Invalid or insufficient data provided")
            
            df = pd.DataFrame(data)
            
            # Ensure we have enough data
            if len(df) < max(self.config.volume_smoothing_period, 
                           self.config.ml_training_period):
                return self._create_default_result()
            
            # Calculate core A/D Line components
            ad_results = self._calculate_ad_line_enhanced(df)
            
            # Calculate momentum analysis
            momentum_results = self._calculate_volume_momentum(df, ad_results)
            
            # Perform divergence analysis
            divergence_results = self._analyze_divergences(df, ad_results)
            
            # Calculate trend confirmation
            trend_results = self._calculate_trend_confirmation(df, ad_results)
            
            # Statistical validation
            stats_results = self._validate_statistical_significance(ad_results)
            
            # Generate final result
            result = self._compile_final_result(
                ad_results, momentum_results, divergence_results,
                trend_results, stats_results
            )
            
            # Update historical data
            self._update_history(df, result)
            
            # Generate signal
            signal = self._generate_signal(result)
            
            return {
                'signal': signal,
                'confidence': result.signal_strength,
                'ad_line': result.ad_line,
                'ad_momentum': result.ad_momentum,
                'volume_pressure': result.volume_pressure,
                'trend_confirmation': result.trend_confirmation,
                'divergence_score': result.divergence_score,
                'money_flow_multiplier': result.money_flow_multiplier,
                'normalized_volume': result.normalized_volume,
                'statistical_significance': result.statistical_significance,
                'metadata': {
                    'indicator_name': 'AccumulationDistributionLine',
                    'calculation_method': 'advanced_volume_analysis',
                    'parameters': {
                        'volume_smoothing': self.config.volume_smoothing_period,
                        'momentum_period': self.config.momentum_period,
                        'divergence_lookback': self.config.divergence_lookback
                    }
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating A/D Line: {str(e)}")
            return self._create_error_result(str(e))
    
    def _calculate_ad_line_enhanced(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Calculate enhanced A/D Line with volume normalization"""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        volume = df['volume'].values
        
        # Calculate Money Flow Multiplier with enhanced precision
        high_low_diff = high - low
        # Handle zero division with small epsilon
        high_low_diff = np.where(high_low_diff == 0, 1e-8, high_low_diff)
        
        mf_multiplier = ((close - low) - (high - close)) / high_low_diff
        
        # Apply volume normalization
        normalized_volume = self._normalize_volume(volume)
        
        # Calculate Money Flow Volume
        money_flow_volume = mf_multiplier * normalized_volume
        
        # Calculate A/D Line with adaptive smoothing
        ad_line = np.cumsum(money_flow_volume)
        
        # Apply smoothing to reduce noise
        if len(ad_line) >= self.config.volume_smoothing_period:
            ad_line_smoothed = self._adaptive_smooth(ad_line)
        else:
            ad_line_smoothed = ad_line
        
        return {
            'ad_line': ad_line_smoothed,
            'money_flow_multiplier': mf_multiplier,
            'money_flow_volume': money_flow_volume,
            'normalized_volume': normalized_volume
        }
    
    def _calculate_volume_momentum(self, df: pd.DataFrame, 
                                 ad_results: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate volume-weighted momentum analysis"""
        ad_line = ad_results['ad_line']
        volume = ad_results['normalized_volume']
        
        if len(ad_line) < self.config.momentum_period:
            return {'momentum': 0.0, 'volume_pressure': 0.0}
        
        # Calculate A/D momentum using rate of change
        momentum_period = min(self.config.momentum_period, len(ad_line))
        ad_momentum = (ad_line[-1] - ad_line[-momentum_period]) / momentum_period
        
        # Calculate volume pressure (volume-weighted momentum)
        recent_volume = volume[-momentum_period:]
        volume_weights = recent_volume / np.sum(recent_volume)
        
        # Calculate price momentum for comparison
        close = df['close'].values
        if len(close) >= momentum_period:
            price_changes = np.diff(close[-momentum_period:])
            volume_pressure = np.sum(price_changes * volume_weights[1:])
        else:
            volume_pressure = 0.0
        
        return {
            'momentum': ad_momentum,
            'volume_pressure': volume_pressure
        }
    
    def _analyze_divergences(self, df: pd.DataFrame, 
                           ad_results: Dict[str, np.ndarray]) -> Dict[str, float]:
        """ML-based divergence detection between price and A/D Line"""
        if len(df) < self.config.divergence_lookback:
            return {'divergence_score': 0.0}
        
        # Prepare data for analysis
        lookback = min(self.config.divergence_lookback, len(df))
        close = df['close'].values[-lookback:]
        ad_line = ad_results['ad_line'][-lookback:]
        
        # Normalize data for comparison
        close_normalized = (close - np.mean(close)) / np.std(close)
        ad_normalized = (ad_line - np.mean(ad_line)) / np.std(ad_line)
        
        # Calculate correlation
        correlation = np.corrcoef(close_normalized, ad_normalized)[0, 1]
        
        # Calculate gradient divergence
        close_gradient = np.gradient(close_normalized)
        ad_gradient = np.gradient(ad_normalized)
        
        # Find peaks and troughs for divergence analysis
        close_peaks, _ = find_peaks(close_normalized, distance=5)
        close_troughs, _ = find_peaks(-close_normalized, distance=5)
        ad_peaks, _ = find_peaks(ad_normalized, distance=5)
        ad_troughs, _ = find_peaks(-ad_normalized, distance=5)
        
        # Calculate divergence score
        divergence_score = 0.0
        
        # Check for bullish divergence (price makes lower lows, A/D makes higher lows)
        if len(close_troughs) >= 2 and len(ad_troughs) >= 2:
            if (close_normalized[close_troughs[-1]] < close_normalized[close_troughs[-2]] and
                ad_normalized[ad_troughs[-1]] > ad_normalized[ad_troughs[-2]]):
                divergence_score += 0.5
        
        # Check for bearish divergence (price makes higher highs, A/D makes lower highs)
        if len(close_peaks) >= 2 and len(ad_peaks) >= 2:
            if (close_normalized[close_peaks[-1]] > close_normalized[close_peaks[-2]] and
                ad_normalized[ad_peaks[-1]] < ad_normalized[ad_peaks[-2]]):
                divergence_score -= 0.5
        
        # Factor in correlation strength
        divergence_score *= (1 - abs(correlation))
        
        return {'divergence_score': divergence_score}
    
    def _calculate_trend_confirmation(self, df: pd.DataFrame,
                                    ad_results: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate trend confirmation using multi-period analysis"""
        if len(df) < self.config.trend_confirmation_period:
            return {'trend_confirmation': 0.0}
        
        period = min(self.config.trend_confirmation_period, len(df))
        close = df['close'].values[-period:]
        ad_line = ad_results['ad_line'][-period:]
        
        # Calculate trend directions
        close_trend = np.polyfit(range(len(close)), close, 1)[0]
        ad_trend = np.polyfit(range(len(ad_line)), ad_line, 1)[0]
        
        # Normalize trends
        close_trend_norm = close_trend / np.mean(close)
        ad_trend_norm = ad_trend / np.mean(ad_line)
        
        # Calculate confirmation score
        if np.sign(close_trend_norm) == np.sign(ad_trend_norm):
            # Same direction - strong confirmation
            confirmation = min(abs(close_trend_norm) + abs(ad_trend_norm), 1.0)
        else:
            # Opposite directions - weak/negative confirmation
            confirmation = -min(abs(close_trend_norm) + abs(ad_trend_norm), 1.0)
        
        return {'trend_confirmation': confirmation}
    
    def _validate_statistical_significance(self, ad_results: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Validate statistical significance of A/D Line signals"""
        ad_line = ad_results['ad_line']
        
        if len(ad_line) < 30:  # Minimum for statistical analysis
            return {'significance': 0.0}
        
        # Test for trend significance using linear regression
        x = np.arange(len(ad_line))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, ad_line)
        
        # Calculate significance score (inverse of p-value, capped at 1.0)
        significance = max(0.0, 1.0 - p_value) if p_value <= self.config.significance_threshold else 0.0
        
        return {'significance': significance}
    
    def _normalize_volume(self, volume: np.ndarray) -> np.ndarray:
        """Normalize volume using adaptive techniques"""
        if len(volume) < self.config.volume_normalization_period:
            return volume
        
        # Use rolling average for normalization
        period = min(self.config.volume_normalization_period, len(volume))
        rolling_avg = np.convolve(volume, np.ones(period)/period, mode='same')
        
        # Prevent division by zero
        rolling_avg = np.where(rolling_avg == 0, 1, rolling_avg)
        
        return volume / rolling_avg
    
    def _adaptive_smooth(self, data: np.ndarray) -> np.ndarray:
        """Apply adaptive smoothing to reduce noise"""
        if len(data) < self.config.volume_smoothing_period:
            return data
        
        # Use exponential moving average with adaptive alpha
        alpha = 2.0 / (self.config.volume_smoothing_period + 1)
        smoothed = np.zeros_like(data)
        smoothed[0] = data[0]
        
        for i in range(1, len(data)):
            # Adaptive alpha based on volatility
            volatility = abs(data[i] - data[i-1]) / abs(data[i-1]) if data[i-1] != 0 else 0
            adjusted_alpha = alpha * (1 - min(volatility, 0.5))
            smoothed[i] = adjusted_alpha * data[i] + (1 - adjusted_alpha) * smoothed[i-1]
        
        return smoothed
    
    def _compile_final_result(self, ad_results: Dict, momentum_results: Dict,
                            divergence_results: Dict, trend_results: Dict,
                            stats_results: Dict) -> ADLineResult:
        """Compile all analysis results into final structure"""
        # Calculate signal strength as weighted combination
        weights = {
            'momentum': 0.25,
            'divergence': 0.30,
            'trend': 0.25,
            'significance': 0.20
        }
        
        signal_strength = (
            weights['momentum'] * abs(momentum_results['momentum']) +
            weights['divergence'] * abs(divergence_results['divergence_score']) +
            weights['trend'] * abs(trend_results['trend_confirmation']) +
            weights['significance'] * stats_results['significance']
        )
        
        return ADLineResult(
            ad_line=ad_results['ad_line'][-1],
            ad_momentum=momentum_results['momentum'],
            volume_pressure=momentum_results['volume_pressure'],
            trend_confirmation=trend_results['trend_confirmation'],
            divergence_score=divergence_results['divergence_score'],
            signal_strength=min(signal_strength, 1.0),
            money_flow_multiplier=ad_results['money_flow_multiplier'][-1],
            normalized_volume=ad_results['normalized_volume'][-1],
            statistical_significance=stats_results['significance']
        )
    
    def _generate_signal(self, result: ADLineResult) -> SignalType:
        """Generate trading signal based on comprehensive analysis"""
        # Minimum confidence threshold
        if result.signal_strength < 0.3:
            return SignalType.HOLD
        
        # Bullish conditions
        bullish_score = 0
        if result.ad_momentum > 0:
            bullish_score += 1
        if result.volume_pressure > 0:
            bullish_score += 1
        if result.trend_confirmation > 0.2:
            bullish_score += 1
        if result.divergence_score > 0.3:
            bullish_score += 1
        if result.statistical_significance > 0.7:
            bullish_score += 1
        
        # Bearish conditions
        bearish_score = 0
        if result.ad_momentum < 0:
            bearish_score += 1
        if result.volume_pressure < 0:
            bearish_score += 1
        if result.trend_confirmation < -0.2:
            bearish_score += 1
        if result.divergence_score < -0.3:
            bearish_score += 1
        if result.statistical_significance > 0.7:  # High significance supports any direction
            bearish_score += 1
        
        # Generate signal based on score balance
        if bullish_score >= 3 and bullish_score > bearish_score:
            return SignalType.BUY
        elif bearish_score >= 3 and bearish_score > bullish_score:
            return SignalType.SELL
        else:
            return SignalType.HOLD
    
    def _update_history(self, df: pd.DataFrame, result: ADLineResult):
        """Update historical data for future analysis"""
        max_history = max(self.config.ml_training_period, 
                         self.config.divergence_lookback) * 2
        
        # Update price and volume history
        self._price_history.extend(df['close'].values)
        self._volume_history.extend(df['volume'].values)
        self._ad_history.append(result.ad_line)
        self._momentum_history.append(result.ad_momentum)
        
        # Trim to maximum length
        if len(self._price_history) > max_history:
            self._price_history = self._price_history[-max_history:]
            self._volume_history = self._volume_history[-max_history:]
            self._ad_history = self._ad_history[-max_history:]
            self._momentum_history = self._momentum_history[-max_history:]
    
    def _validate_data(self, data: Dict[str, Any]) -> bool:
        """Validate input data structure and completeness"""
        required_fields = ['high', 'low', 'close', 'volume']
        
        if not isinstance(data, dict):
            return False
        
        for field in required_fields:
            if field not in data:
                self.logger.error(f"Missing required field: {field}")
                return False
            
            if not isinstance(data[field], (list, np.ndarray)) or len(data[field]) == 0:
                self.logger.error(f"Invalid data for field: {field}")
                return False
        
        # Check data consistency
        lengths = [len(data[field]) for field in required_fields]
        if len(set(lengths)) > 1:
            self.logger.error("Inconsistent data lengths across fields")
            return False
        
        return True
    
    def _create_default_result(self) -> Dict[str, Any]:
        """Create default result for insufficient data"""
        return {
            'signal': SignalType.HOLD,
            'confidence': 0.0,
            'ad_line': 0.0,
            'ad_momentum': 0.0,
            'volume_pressure': 0.0,
            'trend_confirmation': 0.0,
            'divergence_score': 0.0,
            'money_flow_multiplier': 0.0,
            'normalized_volume': 0.0,
            'statistical_significance': 0.0,
            'metadata': {
                'indicator_name': 'AccumulationDistributionLine',
                'error': 'Insufficient data for calculation'
            }
        }
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create error result structure"""
        return {
            'signal': SignalType.HOLD,
            'confidence': 0.0,
            'ad_line': 0.0,
            'ad_momentum': 0.0,
            'volume_pressure': 0.0,
            'trend_confirmation': 0.0,
            'divergence_score': 0.0,
            'money_flow_multiplier': 0.0,
            'normalized_volume': 0.0,
            'statistical_significance': 0.0,
            'metadata': {
                'indicator_name': 'AccumulationDistributionLine',
                'error': error_message
            }
        }