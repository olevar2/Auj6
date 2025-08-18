"""
Awesome Oscillator Indicator - Advanced Implementation

The Awesome Oscillator (AO) is a momentum indicator developed by Bill Williams. It measures
the difference between a 5-period and 34-period simple moving average of the median price (HL2).
The AO is designed to gauge market momentum and identify potential trend changes.

Key Features:
- Classic Awesome Oscillator calculation with HL2 median price
- Adaptive period optimization for different market conditions
- Volume-weighted AO for institutional flow analysis
- Advanced signal generation with saucer and twin peaks patterns
- Zero-line and color change analysis
- ML-enhanced pattern recognition for AO signals
- Multi-timeframe AO coherence analysis

Formula:
Median Price = (High + Low) / 2
AO = SMA(Median Price, 5) - SMA(Median Price, 34)

Bill Williams Signals:
- Saucer: Three consecutive bars above/below zero with middle bar lowest/highest
- Zero Line Cross: AO crossing above/below zero line
- Twin Peaks: Divergence pattern with two peaks/troughs
- Color Change: Change from red to green bars or vice versa
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from scipy import stats
from scipy.signal import find_peaks
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from ...base.indicator_base import BaseIndicator
from ....core.exceptions import IndicatorCalculationException
from ....data_providers.base_provider import DataRequirement, DataType
from ....core.signal_types import SignalType


class AwesomeOscillatorIndicator(BaseIndicator):
    """
    Advanced Awesome Oscillator Indicator Implementation
    
    This implementation provides comprehensive Awesome Oscillator analysis including:
    - Classic and adaptive AO calculation with HL2 median price
    - Volume-weighted analysis for institutional detection
    - Bill Williams pattern recognition (saucer, twin peaks)
    - Advanced signal generation with ML enhancement
    - Zero-line and momentum analysis
    - Multi-timeframe coherence detection
    """
    
    def __init__(self, fast_period: int = 5, slow_period: int = 34, **kwargs):
        """
        Initialize Awesome Oscillator Indicator
        
        Args:
            fast_period: Fast SMA period (default: 5)
            slow_period: Slow SMA period (default: 34)
            **kwargs: Additional parameters for advanced features
        """
        default_params = {
            'fast_period': fast_period,
            'slow_period': slow_period,
            'min_fast_period': 3,
            'max_fast_period': 8,
            'min_slow_period': 25,
            'max_slow_period': 50,
            'smoothing_period': 3,
            'volume_weighted': True,
            'adaptive_periods': True,
            'multi_timeframe': True,
            'divergence_analysis': True,
            'pattern_recognition': True,
            'smoothing_enabled': True,
            'ml_enhancement': True,
            'signal_sensitivity': 0.65,
            'ml_lookback': 50,
            'saucer_detection': True,
            'twin_peaks_detection': True,
            'zero_line_analysis': True,
            'color_analysis': True
        }
        default_params.update(kwargs)
        
        super().__init__(
            name="AwesomeOscillator",
            category="momentum",
            description="Advanced Awesome Oscillator with Bill Williams patterns",
            parameters=default_params
        )
        
        # ML models for enhanced analysis
        self.signal_classifier = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10, 
            random_state=42
        )
        self.pattern_classifier = RandomForestClassifier(
            n_estimators=80,
            max_depth=8,
            random_state=42
        )
        self.trend_predictor = RandomForestRegressor(
            n_estimators=100, 
            max_depth=8, 
            random_state=42
        )
        self.scaler = StandardScaler()
        self.models_trained = False
        
        # Historical data storage
        self.history = {
            'ao_values': [],
            'median_prices': [],
            'fast_sma': [],
            'slow_sma': [],
            'volume_weighted_ao': [],
            'smoothed_ao': [],
            'saucer_signals': [],
            'twin_peaks_signals': [],
            'zero_crossings': [],
            'color_changes': []
        }
    
    def get_data_requirements(self) -> DataRequirement:
        max_period = max(self.parameters['slow_period'], 
                        self.parameters['ml_lookback'])
        return DataRequirement(
            data_type=DataType.OHLCV,
            required_columns=['high', 'low', 'close', 'volume'],
            min_periods=max_period * 2 + 30,
            lookback_periods=150
        )
    
    def calculate_raw(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate Awesome Oscillator with comprehensive analysis"""
        try:
            # Optimize periods if enabled
            if self.parameters['adaptive_periods']:
                fast_period, slow_period = self._optimize_ao_periods(data)
            else:
                fast_period = self.parameters['fast_period']
                slow_period = self.parameters['slow_period']
            
            # Calculate median price (HL2)
            median_price = (data['high'] + data['low']) / 2
            
            # Calculate Awesome Oscillator
            ao = self._calculate_awesome_oscillator(median_price, fast_period, slow_period)
            
            # Calculate volume-weighted AO if enabled
            if self.parameters['volume_weighted']:
                vw_ao = self._calculate_volume_weighted_ao(data, fast_period, slow_period)
            else:
                vw_ao = ao
            
            # Apply smoothing if enabled
            if self.parameters['smoothing_enabled']:
                smoothed_ao = vw_ao.rolling(window=self.parameters['smoothing_period']).mean()
            else:
                smoothed_ao = vw_ao
            
            # Advanced analysis
            zero_line_analysis = self._analyze_zero_line_behavior(smoothed_ao)
            color_analysis = self._analyze_color_changes(smoothed_ao)
            
            # Bill Williams patterns
            saucer_analysis = self._detect_saucer_patterns(smoothed_ao) if self.parameters['saucer_detection'] else {'saucers': [], 'strength': 0.0}
            twin_peaks_analysis = self._detect_twin_peaks(smoothed_ao, data) if self.parameters['twin_peaks_detection'] else {'twin_peaks': [], 'strength': 0.0}
            
            # Additional analysis
            timeframe_analysis = self._analyze_multi_timeframe_ao(data)
            divergence_analysis = self._detect_ao_divergences(smoothed_ao, data)
            momentum_analysis = self._analyze_momentum_characteristics(smoothed_ao, data)
            
            # Train ML models if not already trained
            if not self.models_trained:
                self._train_ml_models(smoothed_ao, data)
            
            # Generate signal
            signal, confidence = self._generate_ao_signal(
                smoothed_ao, zero_line_analysis, color_analysis,
                saucer_analysis, twin_peaks_analysis, timeframe_analysis,
                divergence_analysis, momentum_analysis, data
            )
            
            # Update history
            current_ao = float(smoothed_ao.iloc[-1]) if not pd.isna(smoothed_ao.iloc[-1]) else 0.0
            current_median = float(median_price.iloc[-1]) if not pd.isna(median_price.iloc[-1]) else 0.0
            
            # Calculate SMAs for history
            fast_sma = median_price.rolling(window=fast_period).mean()
            slow_sma = median_price.rolling(window=slow_period).mean()
            
            self.history['ao_values'].append(current_ao)
            self.history['median_prices'].append(current_median)
            self.history['fast_sma'].append(float(fast_sma.iloc[-1]) if not pd.isna(fast_sma.iloc[-1]) else 0.0)
            self.history['slow_sma'].append(float(slow_sma.iloc[-1]) if not pd.isna(slow_sma.iloc[-1]) else 0.0)
            self.history['volume_weighted_ao'].append(float(vw_ao.iloc[-1]) if not pd.isna(vw_ao.iloc[-1]) else 0.0)
            
            # Keep history limited
            for key in self.history:
                if len(self.history[key]) > 100:
                    self.history[key] = self.history[key][-100:]
            
            result = {
                'awesome_oscillator': current_ao,
                'median_price': current_median,
                'fast_sma': float(fast_sma.iloc[-1]) if not pd.isna(fast_sma.iloc[-1]) else 0.0,
                'slow_sma': float(slow_sma.iloc[-1]) if not pd.isna(slow_sma.iloc[-1]) else 0.0,
                'volume_weighted_ao': float(vw_ao.iloc[-1]) if not pd.isna(vw_ao.iloc[-1]) else 0.0,
                'signal': signal,
                'confidence': confidence,
                'periods_used': {
                    'fast_period': fast_period,
                    'slow_period': slow_period
                },
                'zero_line_analysis': zero_line_analysis,
                'color_analysis': color_analysis,
                'saucer_analysis': saucer_analysis,
                'twin_peaks_analysis': twin_peaks_analysis,
                'timeframe_analysis': timeframe_analysis,
                'divergence_analysis': divergence_analysis,
                'momentum_analysis': momentum_analysis,
                'bill_williams_signals': self._get_bill_williams_signals(
                    saucer_analysis, twin_peaks_analysis, zero_line_analysis, color_analysis
                ),
                'trend_direction': 'bullish' if current_ao > 0 else 'bearish',
                'momentum_strength': min(abs(current_ao) / 1000, 1.0),  # Normalize to 0-1
                'zero_distance': abs(current_ao),
                'values_history': {
                    'awesome_oscillator': smoothed_ao.tail(20).tolist(),
                    'median_price': median_price.tail(20).tolist(),
                    'zero_line': [0] * 20
                }
            }
            
            return result
            
        except Exception as e:
            raise IndicatorCalculationException(
                indicator_name=self.name,
                calculation_step="raw_calculation",
                message=f"Failed to calculate Awesome Oscillator: {str(e)}",
                cause=e
            )
    
    def _calculate_awesome_oscillator(self, median_price: pd.Series, fast_period: int, slow_period: int) -> pd.Series:
        """Calculate classic Awesome Oscillator"""
        fast_sma = median_price.rolling(window=fast_period).mean()
        slow_sma = median_price.rolling(window=slow_period).mean()
        
        ao = fast_sma - slow_sma
        return ao.fillna(0)
    
    def _optimize_ao_periods(self, data: pd.DataFrame) -> Tuple[int, int]:
        """Optimize AO periods based on market conditions"""
        if len(data) < 60:
            return self.parameters['fast_period'], self.parameters['slow_period']
        
        # Calculate market characteristics
        median_price = (data['high'] + data['low']) / 2
        returns = median_price.pct_change().tail(40)
        volatility = returns.std()
        
        # Calculate trend persistence
        price_changes = median_price.diff().tail(30)
        trend_strength = abs(price_changes.sum()) / (price_changes.abs().sum() + 1e-8)
        
        base_fast = self.parameters['fast_period']
        base_slow = self.parameters['slow_period']
        
        # Adjust based on volatility
        if volatility > returns.rolling(window=20).std().mean() * 1.5:
            # High volatility - slightly longer periods for stability
            fast_period = min(self.parameters['max_fast_period'], base_fast + 1)
            slow_period = min(self.parameters['max_slow_period'], int(base_slow * 1.1))
        elif volatility < returns.rolling(window=20).std().mean() * 0.7:
            # Low volatility - shorter periods for sensitivity
            fast_period = max(self.parameters['min_fast_period'], base_fast - 1)
            slow_period = max(self.parameters['min_slow_period'], int(base_slow * 0.9))
        else:
            fast_period = base_fast
            slow_period = base_slow
        
        # Adjust based on trend strength
        if trend_strength > 0.8:
            # Strong trend - longer periods to avoid noise
            slow_period = min(self.parameters['max_slow_period'], slow_period + 3)
        
        return fast_period, slow_period
    
    def _calculate_volume_weighted_ao(self, data: pd.DataFrame, fast_period: int, slow_period: int) -> pd.Series:
        """Calculate volume-weighted Awesome Oscillator"""
        if 'volume' not in data.columns:
            median_price = (data['high'] + data['low']) / 2
            return self._calculate_awesome_oscillator(median_price, fast_period, slow_period)
        
        # Calculate volume-weighted median price
        median_price = (data['high'] + data['low']) / 2
        volume = data['volume']
        
        # Volume-weighted moving averages
        vw_fast_sum = (median_price * volume).rolling(window=fast_period).sum()
        volume_sum_fast = volume.rolling(window=fast_period).sum()
        vw_fast_sma = vw_fast_sum / (volume_sum_fast + 1e-8)
        
        vw_slow_sum = (median_price * volume).rolling(window=slow_period).sum()
        volume_sum_slow = volume.rolling(window=slow_period).sum()
        vw_slow_sma = vw_slow_sum / (volume_sum_slow + 1e-8)
        
        # Volume-weighted AO
        vw_ao = vw_fast_sma - vw_slow_sma
        
        # Regular AO for blending
        regular_ao = self._calculate_awesome_oscillator(median_price, fast_period, slow_period)
        
        # Blend based on volume significance
        volume_ratio = volume / volume.rolling(window=20).mean()
        volume_weight = np.clip((volume_ratio - 1) * 0.3, 0, 0.6)
        
        # Higher volume = more weight on volume-weighted AO
        blended_ao = regular_ao * (1 - volume_weight) + vw_ao * volume_weight
        
        return blended_ao.fillna(0)
    
    def _analyze_zero_line_behavior(self, ao: pd.Series) -> Dict[str, Any]:
        """Analyze AO behavior around zero line"""
        if len(ao) < 10:
            return {'zero_crossings': 0, 'above_zero_periods': 0, 'current_position': 'unknown'}
        
        recent_ao = ao.tail(30)
        current_value = recent_ao.iloc[-1]
        
        # Count zero crossings
        zero_crossings = 0
        for i in range(1, len(recent_ao)):
            if (recent_ao.iloc[i-1] >= 0 and recent_ao.iloc[i] < 0) or \
               (recent_ao.iloc[i-1] <= 0 and recent_ao.iloc[i] > 0):
                zero_crossings += 1
        
        # Time above/below zero
        above_zero = (recent_ao > 0).sum()
        below_zero = (recent_ao <= 0).sum()
        
        # Current position relative to zero
        if current_value > 0:
            position = 'above_zero'
        elif current_value < 0:
            position = 'below_zero'
        else:
            position = 'at_zero'
        
        # Distance from zero
        zero_distance = abs(current_value)
        
        # Recent momentum near zero
        zero_momentum = 'unknown'
        if len(recent_ao) >= 3:
            if current_value > 0 and recent_ao.iloc[-3] <= 0:
                zero_momentum = 'crossing_up'
            elif current_value < 0 and recent_ao.iloc[-3] >= 0:
                zero_momentum = 'crossing_down'
            elif current_value > recent_ao.iloc[-2]:
                zero_momentum = 'moving_up'
            elif current_value < recent_ao.iloc[-2]:
                zero_momentum = 'moving_down'
            else:
                zero_momentum = 'stable'
        
        return {
            'zero_crossings': zero_crossings,
            'above_zero_periods': above_zero,
            'below_zero_periods': below_zero,
            'current_position': position,
            'zero_distance': float(zero_distance),
            'zero_momentum': zero_momentum,
            'above_zero_ratio': float(above_zero / len(recent_ao))
        }
    
    def _analyze_color_changes(self, ao: pd.Series) -> Dict[str, Any]:
        """Analyze AO color changes (momentum direction changes)"""
        if len(ao) < 5:
            return {'recent_color_changes': 0, 'current_color': 'unknown', 'color_sequence': []}
        
        recent_ao = ao.tail(10)
        
        # Determine colors (green = rising, red = falling)
        colors = []
        color_changes = 0
        
        for i in range(1, len(recent_ao)):
            if recent_ao.iloc[i] > recent_ao.iloc[i-1]:
                current_color = 'green'
            elif recent_ao.iloc[i] < recent_ao.iloc[i-1]:
                current_color = 'red'
            else:
                current_color = 'neutral'
            
            colors.append(current_color)
            
            # Count color changes
            if i > 1 and colors[-1] != colors[-2] and colors[-1] != 'neutral' and colors[-2] != 'neutral':
                color_changes += 1
        
        current_color = colors[-1] if colors else 'unknown'
        
        # Analyze color sequences
        green_sequence = 0
        red_sequence = 0
        
        # Count current sequence
        for color in reversed(colors):
            if color == current_color and current_color != 'neutral':
                if current_color == 'green':
                    green_sequence += 1
                else:
                    red_sequence += 1
            else:
                break
        
        return {
            'recent_color_changes': color_changes,
            'current_color': current_color,
            'color_sequence': colors,
            'current_sequence_length': max(green_sequence, red_sequence),
            'green_bars_count': colors.count('green'),
            'red_bars_count': colors.count('red')
        }
    
    def _detect_saucer_patterns(self, ao: pd.Series) -> Dict[str, Any]:
        """Detect Bill Williams saucer patterns"""
        if len(ao) < 5:
            return {'saucers': [], 'strength': 0.0}
        
        saucers = []
        recent_ao = ao.tail(20)
        
        # Look for saucer patterns (3 consecutive bars)
        for i in range(2, len(recent_ao)):
            bar1 = recent_ao.iloc[i-2]
            bar2 = recent_ao.iloc[i-1]  # Middle bar
            bar3 = recent_ao.iloc[i]
            
            # Bullish saucer: all above zero, middle bar lowest
            if bar1 > 0 and bar2 > 0 and bar3 > 0:
                if bar2 < bar1 and bar2 < bar3:
                    saucer_strength = min((bar1 + bar3 - 2*bar2) / 1000, 1.0)
                    saucers.append({
                        'type': 'bullish_saucer',
                        'position': i,
                        'strength': saucer_strength,
                        'values': [bar1, bar2, bar3],
                        'description': 'Bullish saucer pattern detected'
                    })
            
            # Bearish saucer: all below zero, middle bar highest
            elif bar1 < 0 and bar2 < 0 and bar3 < 0:
                if bar2 > bar1 and bar2 > bar3:
                    saucer_strength = min((2*bar2 - bar1 - bar3) / 1000, 1.0)
                    saucers.append({
                        'type': 'bearish_saucer',
                        'position': i,
                        'strength': saucer_strength,
                        'values': [bar1, bar2, bar3],
                        'description': 'Bearish saucer pattern detected'
                    })
        
        # Calculate overall saucer strength
        total_strength = sum([s['strength'] for s in saucers])
        avg_strength = total_strength / len(saucers) if saucers else 0.0
        
        return {
            'saucers': saucers,
            'strength': float(avg_strength),
            'saucer_count': len(saucers),
            'recent_saucer': saucers[-1] if saucers else None
        }
    
    def _detect_twin_peaks(self, ao: pd.Series, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect Bill Williams twin peaks patterns"""
        if len(ao) < 20:
            return {'twin_peaks': [], 'strength': 0.0}
        
        twin_peaks = []
        recent_ao = ao.tail(30)
        recent_prices = data['close'].tail(30)
        
        # Find peaks and troughs in AO
        ao_peaks, _ = find_peaks(recent_ao.values, distance=5, prominence=50)
        ao_troughs, _ = find_peaks(-recent_ao.values, distance=5, prominence=50)
        
        # Find corresponding price peaks/troughs
        price_peaks, _ = find_peaks(recent_prices.values, distance=5)
        price_troughs, _ = find_peaks(-recent_prices.values, distance=5)
        
        # Bullish twin peaks: two troughs below zero, second higher than first, price makes lower low
        if len(ao_troughs) >= 2:
            for i in range(len(ao_troughs) - 1):
                trough1_idx = ao_troughs[i]
                trough2_idx = ao_troughs[i + 1]
                
                trough1_val = recent_ao.iloc[trough1_idx]
                trough2_val = recent_ao.iloc[trough2_idx]
                
                # Both troughs below zero, second higher
                if trough1_val < 0 and trough2_val < 0 and trough2_val > trough1_val:
                    # Find corresponding price troughs
                    price_trough1 = self._find_nearest_extreme(price_troughs, trough1_idx)
                    price_trough2 = self._find_nearest_extreme(price_troughs, trough2_idx)
                    
                    if price_trough1 is not None and price_trough2 is not None:
                        price1 = recent_prices.iloc[price_trough1]
                        price2 = recent_prices.iloc[price_trough2]
                        
                        # Price makes lower low while AO makes higher low
                        if price2 < price1:
                            divergence_strength = (trough2_val - trough1_val) / abs(trough1_val)
                            twin_peaks.append({
                                'type': 'bullish_twin_peaks',
                                'ao_trough1': trough1_val,
                                'ao_trough2': trough2_val,
                                'price1': price1,
                                'price2': price2,
                                'strength': min(divergence_strength, 1.0),
                                'position': trough2_idx,
                                'description': 'Bullish twin peaks divergence'
                            })
        
        # Bearish twin peaks: two peaks above zero, second lower than first, price makes higher high
        if len(ao_peaks) >= 2:
            for i in range(len(ao_peaks) - 1):
                peak1_idx = ao_peaks[i]
                peak2_idx = ao_peaks[i + 1]
                
                peak1_val = recent_ao.iloc[peak1_idx]
                peak2_val = recent_ao.iloc[peak2_idx]
                
                # Both peaks above zero, second lower
                if peak1_val > 0 and peak2_val > 0 and peak2_val < peak1_val:
                    # Find corresponding price peaks
                    price_peak1 = self._find_nearest_extreme(price_peaks, peak1_idx)
                    price_peak2 = self._find_nearest_extreme(price_peaks, peak2_idx)
                    
                    if price_peak1 is not None and price_peak2 is not None:
                        price1 = recent_prices.iloc[price_peak1]
                        price2 = recent_prices.iloc[price_peak2]
                        
                        # Price makes higher high while AO makes lower high
                        if price2 > price1:
                            divergence_strength = (peak1_val - peak2_val) / abs(peak1_val)
                            twin_peaks.append({
                                'type': 'bearish_twin_peaks',
                                'ao_peak1': peak1_val,
                                'ao_peak2': peak2_val,
                                'price1': price1,
                                'price2': price2,
                                'strength': min(divergence_strength, 1.0),
                                'position': peak2_idx,
                                'description': 'Bearish twin peaks divergence'
                            })
        
        # Calculate overall strength
        total_strength = sum([tp['strength'] for tp in twin_peaks])
        avg_strength = total_strength / len(twin_peaks) if twin_peaks else 0.0
        
        return {
            'twin_peaks': twin_peaks,
            'strength': float(avg_strength),
            'twin_peaks_count': len(twin_peaks),
            'recent_twin_peaks': twin_peaks[-1] if twin_peaks else None
        }
    
    def _find_nearest_extreme(self, extremes: np.ndarray, target_idx: int) -> Optional[int]:
        """Find the nearest extreme point to a target index"""
        if len(extremes) == 0:
            return None
        
        min_distance = float('inf')
        nearest_extreme = None
        
        for extreme_idx in extremes:
            distance = abs(extreme_idx - target_idx)
            if distance < min_distance and distance < 8:
                min_distance = distance
                nearest_extreme = extreme_idx
        
        return nearest_extreme
    
    def _analyze_multi_timeframe_ao(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze AO across multiple timeframes"""
        if not self.parameters['multi_timeframe'] or len(data) < 50:
            return {'coherence': 'unknown', 'timeframe_signals': []}
        
        timeframe_signals = []
        
        # Short-term AO (3, 21)
        try:
            median_price = (data['high'] + data['low']) / 2
            short_ao = self._calculate_awesome_oscillator(median_price, 3, 21)
            short_trend = 'bullish' if short_ao.iloc[-1] > 0 else 'bearish'
            timeframe_signals.append({
                'timeframe': 'short',
                'signal': short_trend,
                'value': float(short_ao.iloc[-1]),
                'periods': '3,21'
            })
        except:
            pass
        
        # Standard AO (5, 34)
        try:
            median_price = (data['high'] + data['low']) / 2
            std_ao = self._calculate_awesome_oscillator(median_price, 5, 34)
            std_trend = 'bullish' if std_ao.iloc[-1] > 0 else 'bearish'
            timeframe_signals.append({
                'timeframe': 'medium',
                'signal': std_trend,
                'value': float(std_ao.iloc[-1]),
                'periods': '5,34'
            })
        except:
            pass
        
        # Long-term AO (8, 50)
        if len(data) >= 50:
            try:
                median_price = (data['high'] + data['low']) / 2
                long_ao = self._calculate_awesome_oscillator(median_price, 8, 50)
                long_trend = 'bullish' if long_ao.iloc[-1] > 0 else 'bearish'
                timeframe_signals.append({
                    'timeframe': 'long',
                    'signal': long_trend,
                    'value': float(long_ao.iloc[-1]),
                    'periods': '8,50'
                })
            except:
                pass
        
        # Analyze coherence
        if timeframe_signals:
            signals = [signal['signal'] for signal in timeframe_signals]
            bullish_count = signals.count('bullish')
            bearish_count = signals.count('bearish')
            
            if bullish_count == len(signals):
                coherence = 'strong_bullish'
            elif bearish_count == len(signals):
                coherence = 'strong_bearish'
            elif bullish_count > bearish_count:
                coherence = 'mixed_bullish'
            elif bearish_count > bullish_count:
                coherence = 'mixed_bearish'
            else:
                coherence = 'neutral'
            
            # Calculate strength alignment
            values = [abs(signal['value']) for signal in timeframe_signals]
            strength_alignment = np.std(values) / (np.mean(values) + 1e-8)
            alignment_ratio = max(bullish_count, bearish_count) / len(signals)
        else:
            coherence = 'unknown'
            strength_alignment = 1.0
            alignment_ratio = 0.0
        
        return {
            'coherence': coherence,
            'timeframe_signals': timeframe_signals,
            'strength_alignment': float(strength_alignment),
            'alignment_ratio': float(alignment_ratio)
        }
    
    def _detect_ao_divergences(self, ao: pd.Series, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect AO divergences with price"""
        if not self.parameters['divergence_analysis'] or len(ao) < 30:
            return {'bullish_divergence': False, 'bearish_divergence': False, 'strength': 0.0}
        
        lookback = min(30, len(ao) - 5)
        recent_ao = ao.tail(lookback)
        recent_prices = data['close'].tail(lookback)
        
        # Find peaks and troughs
        ao_peaks, _ = find_peaks(recent_ao.values, distance=5)
        ao_troughs, _ = find_peaks(-recent_ao.values, distance=5)
        
        price_peaks, _ = find_peaks(recent_prices.values, distance=5)
        price_troughs, _ = find_peaks(-recent_prices.values, distance=5)
        
        bullish_divergence = False
        bearish_divergence = False
        divergence_strength = 0.0
        
        # Bullish divergence: price lower low, AO higher low
        if len(ao_troughs) >= 2 and len(price_troughs) >= 2:
            for i in range(len(ao_troughs) - 1):
                ao_trough_1 = ao_troughs[i]
                ao_trough_2 = ao_troughs[i + 1]
                
                price_trough_1 = self._find_nearest_extreme(price_troughs, ao_trough_1)
                price_trough_2 = self._find_nearest_extreme(price_troughs, ao_trough_2)
                
                if price_trough_1 is not None and price_trough_2 is not None:
                    price_1 = recent_prices.iloc[price_trough_1]
                    price_2 = recent_prices.iloc[price_trough_2]
                    ao_1 = recent_ao.iloc[ao_trough_1]
                    ao_2 = recent_ao.iloc[ao_trough_2]
                    
                    if price_2 < price_1 and ao_2 > ao_1:
                        price_decline = (price_1 - price_2) / price_1
                        ao_improvement = abs(ao_2 - ao_1) / abs(ao_1 + 1e-8)
                        
                        if price_decline > 0.02 and ao_improvement > 0.1:
                            bullish_divergence = True
                            divergence_strength = max(divergence_strength, price_decline + ao_improvement)
        
        # Bearish divergence: price higher high, AO lower high
        if len(ao_peaks) >= 2 and len(price_peaks) >= 2:
            for i in range(len(ao_peaks) - 1):
                ao_peak_1 = ao_peaks[i]
                ao_peak_2 = ao_peaks[i + 1]
                
                price_peak_1 = self._find_nearest_extreme(price_peaks, ao_peak_1)
                price_peak_2 = self._find_nearest_extreme(price_peaks, ao_peak_2)
                
                if price_peak_1 is not None and price_peak_2 is not None:
                    price_1 = recent_prices.iloc[price_peak_1]
                    price_2 = recent_prices.iloc[price_peak_2]
                    ao_1 = recent_ao.iloc[ao_peak_1]
                    ao_2 = recent_ao.iloc[ao_peak_2]
                    
                    if price_2 > price_1 and ao_2 < ao_1:
                        price_increase = (price_2 - price_1) / price_1
                        ao_decline = abs(ao_1 - ao_2) / abs(ao_1 + 1e-8)
                        
                        if price_increase > 0.02 and ao_decline > 0.1:
                            bearish_divergence = True
                            divergence_strength = max(divergence_strength, price_increase + ao_decline)
        
        return {
            'bullish_divergence': bullish_divergence,
            'bearish_divergence': bearish_divergence,
            'strength': float(divergence_strength),
            'ao_peaks': len(ao_peaks),
            'ao_troughs': len(ao_troughs)
        }
    
    def _analyze_momentum_characteristics(self, ao: pd.Series, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze momentum characteristics of AO"""
        if len(ao) < 10:
            return {'momentum_phase': 'unknown', 'acceleration': 0.0, 'trend_strength': 0.0}
        
        recent_ao = ao.tail(10)
        current_ao = recent_ao.iloc[-1]
        
        # Calculate momentum and acceleration
        momentum = recent_ao.diff().mean()
        acceleration = recent_ao.diff().diff().mean()
        
        # Determine momentum phase
        if current_ao > 0 and momentum > 0:
            momentum_phase = 'bullish_accelerating'
        elif current_ao > 0 and momentum < 0:
            momentum_phase = 'bullish_decelerating'
        elif current_ao < 0 and momentum < 0:
            momentum_phase = 'bearish_accelerating'
        elif current_ao < 0 and momentum > 0:
            momentum_phase = 'bearish_decelerating'
        else:
            momentum_phase = 'neutral'
        
        # Calculate trend strength
        ao_std = recent_ao.std()
        trend_strength = abs(momentum) / (ao_std + 1e-8)
        trend_strength = min(trend_strength, 1.0)
        
        # Oscillation analysis
        zero_crosses = ((recent_ao > 0) != (recent_ao.shift(1) > 0)).sum()
        oscillation_frequency = zero_crosses / len(recent_ao)
        
        return {
            'momentum_phase': momentum_phase,
            'momentum': float(momentum),
            'acceleration': float(acceleration),
            'trend_strength': float(trend_strength),
            'oscillation_frequency': float(oscillation_frequency),
            'current_ao': float(current_ao)
        }
    
    def _get_bill_williams_signals(self, saucer_analysis: Dict, twin_peaks_analysis: Dict,
                                  zero_line_analysis: Dict, color_analysis: Dict) -> Dict[str, Any]:
        """Compile Bill Williams specific signals"""
        signals = []
        
        # Saucer signals
        if saucer_analysis.get('recent_saucer'):
            signals.append({
                'type': 'saucer',
                'signal': saucer_analysis['recent_saucer']['type'],
                'strength': saucer_analysis['recent_saucer']['strength'],
                'description': saucer_analysis['recent_saucer']['description']
            })
        
        # Twin peaks signals
        if twin_peaks_analysis.get('recent_twin_peaks'):
            signals.append({
                'type': 'twin_peaks',
                'signal': twin_peaks_analysis['recent_twin_peaks']['type'],
                'strength': twin_peaks_analysis['recent_twin_peaks']['strength'],
                'description': twin_peaks_analysis['recent_twin_peaks']['description']
            })
        
        # Zero line cross signals
        if zero_line_analysis['zero_momentum'] in ['crossing_up', 'crossing_down']:
            signal_type = 'bullish_zero_cross' if 'up' in zero_line_analysis['zero_momentum'] else 'bearish_zero_cross'
            signals.append({
                'type': 'zero_line_cross',
                'signal': signal_type,
                'strength': min(zero_line_analysis['zero_distance'] / 1000, 1.0),
                'description': f"AO {zero_line_analysis['zero_momentum']}"
            })
        
        # Color change signals
        if color_analysis['current_color'] != 'neutral':
            signals.append({
                'type': 'color_change',
                'signal': f"{color_analysis['current_color']}_momentum",
                'strength': min(color_analysis['current_sequence_length'] / 5, 1.0),
                'description': f"AO showing {color_analysis['current_color']} momentum"
            })
        
        return {
            'bill_williams_signals': signals,
            'signal_count': len(signals),
            'combined_strength': np.mean([s['strength'] for s in signals]) if signals else 0.0
        }
    
    def _train_ml_models(self, ao: pd.Series, data: pd.DataFrame) -> bool:
        """Train ML models for AO signal enhancement"""
        if not self.parameters['ml_enhancement'] or len(ao) < self.parameters['ml_lookback'] * 2:
            return False
        
        try:
            features, signal_targets, pattern_targets, trend_targets = self._prepare_ml_data(ao, data)
            if len(features) > 40:
                self.scaler.fit(features)
                scaled_features = self.scaler.transform(features)
                
                # Train signal classifier
                self.signal_classifier.fit(scaled_features, signal_targets)
                
                # Train pattern classifier
                self.pattern_classifier.fit(scaled_features, pattern_targets)
                
                # Train trend predictor
                self.trend_predictor.fit(scaled_features, trend_targets)
                
                self.models_trained = True
                return True
        except Exception:
            pass
        return False
    
    def _prepare_ml_data(self, ao: pd.Series, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare ML training data"""
        features, signal_targets, pattern_targets, trend_targets = [], [], [], []
        lookback = 12
        
        for i in range(lookback, len(ao) - 8):
            # Feature windows
            ao_window = ao.iloc[i-lookback:i]
            price_window = data['close'].iloc[i-lookback:i]
            median_window = ((data['high'] + data['low']) / 2).iloc[i-lookback:i]
            
            # AO features
            ao_mean = ao_window.mean()
            ao_std = ao_window.std()
            ao_trend = np.polyfit(range(len(ao_window)), ao_window.values, 1)[0]
            ao_current = ao_window.iloc[-1]
            ao_max = ao_window.max()
            ao_min = ao_window.min()
            
            # Zero line features
            zero_crossings = ((ao_window > 0) != (ao_window.shift(1) > 0)).sum()
            time_above_zero = (ao_window > 0).sum()
            current_zero_distance = abs(ao_current)
            
            # Color analysis features
            ao_momentum = ao_window.diff().iloc[-1]
            color_changes = ((ao_window.diff() > 0) != (ao_window.diff().shift(1) > 0)).sum()
            
            # Price features
            price_returns = price_window.pct_change().dropna()
            price_volatility = price_returns.std() if len(price_returns) > 0 else 0.0
            price_trend = np.polyfit(range(len(price_window)), price_window.values, 1)[0]
            
            # Technical features
            momentum = ao_window.iloc[-1] - ao_window.iloc[-3] if len(ao_window) >= 3 else 0
            acceleration = momentum - (ao_window.iloc[-3] - ao_window.iloc[-6]) if len(ao_window) >= 6 else 0
            
            # Volume features
            if 'volume' in data.columns:
                volume_window = data['volume'].iloc[i-lookback:i]
                volume_trend = np.polyfit(range(len(volume_window)), volume_window.values, 1)[0]
                volume_ratio = volume_window.iloc[-1] / volume_window.mean()
            else:
                volume_trend = 0
                volume_ratio = 1
            
            feature_vector = [
                ao_mean, ao_std, ao_trend, ao_current, ao_max, ao_min,
                zero_crossings, time_above_zero, current_zero_distance,
                ao_momentum, color_changes, momentum, acceleration,
                price_volatility, price_trend, volume_trend, volume_ratio
            ]
            
            # Future targets
            future_ao = ao.iloc[i+1:i+9]
            
            if len(future_ao) >= 5:
                # Signal target
                future_ao_mean = future_ao.mean()
                if future_ao_mean > 100:
                    signal_target = 2  # Strong bullish
                elif future_ao_mean > 0:
                    signal_target = 1  # Bullish
                elif future_ao_mean < -100:
                    signal_target = 0  # Bearish
                else:
                    signal_target = 1  # Neutral
                
                # Pattern target (saucer or twin peaks likelihood)
                ao_variance = future_ao.var()
                if ao_variance < ao_window.var() * 0.5:
                    pattern_target = 1  # Pattern likely
                else:
                    pattern_target = 0  # No pattern
                
                # Trend target
                trend_target = future_ao.iloc[-1] - ao_current
            else:
                signal_target = 1
                pattern_target = 0
                trend_target = 0.0
            
            features.append(feature_vector)
            signal_targets.append(signal_target)
            pattern_targets.append(pattern_target)
            trend_targets.append(trend_target)
        
        return (np.array(features), np.array(signal_targets), 
                np.array(pattern_targets), np.array(trend_targets))
    
    def _generate_ao_signal(self, ao: pd.Series, zero_line_analysis: Dict, color_analysis: Dict,
                           saucer_analysis: Dict, twin_peaks_analysis: Dict,
                           timeframe_analysis: Dict, divergence_analysis: Dict,
                           momentum_analysis: Dict, data: pd.DataFrame) -> Tuple[SignalType, float]:
        """Generate comprehensive AO signal"""
        signal_components = []
        confidence_components = []
        
        current_ao = ao.iloc[-1] if not pd.isna(ao.iloc[-1]) else 0.0
        
        # Zero line signals (primary AO signals)
        if zero_line_analysis['zero_momentum'] == 'crossing_up':
            signal_components.append(1.0)  # Major buy signal
            confidence_components.append(0.95)
        elif zero_line_analysis['zero_momentum'] == 'crossing_down':
            signal_components.append(-0.9)
            confidence_components.append(0.9)
        elif zero_line_analysis['current_position'] == 'above_zero' and zero_line_analysis['zero_momentum'] == 'moving_up':
            signal_components.append(0.8)
            confidence_components.append(0.8)
        elif zero_line_analysis['current_position'] == 'below_zero' and zero_line_analysis['zero_momentum'] == 'moving_down':
            signal_components.append(-0.8)
            confidence_components.append(0.8)
        
        # Saucer pattern signals
        if saucer_analysis.get('recent_saucer'):
            saucer = saucer_analysis['recent_saucer']
            if saucer['type'] == 'bullish_saucer':
                signal_components.append(0.9 * saucer['strength'])
                confidence_components.append(0.9)
            elif saucer['type'] == 'bearish_saucer':
                signal_components.append(-0.9 * saucer['strength'])
                confidence_components.append(0.9)
        
        # Twin peaks signals
        if twin_peaks_analysis.get('recent_twin_peaks'):
            twin_peaks = twin_peaks_analysis['recent_twin_peaks']
            if twin_peaks['type'] == 'bullish_twin_peaks':
                signal_components.append(0.85 * twin_peaks['strength'])
                confidence_components.append(0.85)
            elif twin_peaks['type'] == 'bearish_twin_peaks':
                signal_components.append(-0.85 * twin_peaks['strength'])
                confidence_components.append(0.85)
        
        # Color change signals
        if color_analysis['current_color'] == 'green' and color_analysis['current_sequence_length'] >= 2:
            signal_components.append(0.7)
            confidence_components.append(0.7)
        elif color_analysis['current_color'] == 'red' and color_analysis['current_sequence_length'] >= 2:
            signal_components.append(-0.7)
            confidence_components.append(0.7)
        
        # Divergence signals
        if divergence_analysis['bullish_divergence']:
            strength = divergence_analysis['strength']
            signal_components.append(0.8 * strength)
            confidence_components.append(0.8)
        elif divergence_analysis['bearish_divergence']:
            strength = divergence_analysis['strength']
            signal_components.append(-0.8 * strength)
            confidence_components.append(0.8)
        
        # Multi-timeframe coherence
        coherence = timeframe_analysis['coherence']
        alignment_ratio = timeframe_analysis['alignment_ratio']
        
        if coherence == 'strong_bullish' and alignment_ratio > 0.8:
            signal_components.append(0.75)
            confidence_components.append(0.75)
        elif coherence == 'strong_bearish' and alignment_ratio > 0.8:
            signal_components.append(-0.75)
            confidence_components.append(0.75)
        
        # Momentum phase signals
        momentum_phase = momentum_analysis['momentum_phase']
        if momentum_phase == 'bullish_accelerating':
            signal_components.append(0.6)
            confidence_components.append(0.7)
        elif momentum_phase == 'bearish_accelerating':
            signal_components.append(-0.6)
            confidence_components.append(0.7)
        
        # ML enhancement
        if self.models_trained:
            try:
                recent_data = data.tail(15)
                recent_ao = ao.tail(15)
                
                if len(recent_data) >= 12 and len(recent_ao) >= 12:
                    features = self._extract_current_features(recent_ao, recent_data)
                    if features is not None:
                        scaled_features = self.scaler.transform([features])
                        
                        # Get ML predictions
                        signal_prob = self.signal_classifier.predict_proba(scaled_features)[0]
                        pattern_prob = self.pattern_classifier.predict_proba(scaled_features)[0]
                        trend_pred = self.trend_predictor.predict(scaled_features)[0]
                        
                        # Convert to signal
                        if signal_prob[2] > 0.7:  # Strong bullish
                            signal_components.append(0.7)
                            confidence_components.append(0.6)
                        elif signal_prob[0] > 0.7:  # Bearish
                            signal_components.append(-0.7)
                            confidence_components.append(0.6)
                        
                        # Pattern enhancement
                        if pattern_prob[1] > 0.6:  # Pattern likely
                            # Boost existing signals
                            if signal_components:
                                signal_components[-1] *= 1.1
                        
                        # Trend prediction
                        if abs(trend_pred) > 50:
                            direction = 0.5 if trend_pred > 0 else -0.5
                            signal_components.append(direction)
                            confidence_components.append(0.5)
            except:
                pass
        
        # Calculate final signal
        if signal_components and confidence_components:
            weighted_signal = np.average(signal_components, weights=confidence_components)
            avg_confidence = np.mean(confidence_components)
        else:
            weighted_signal = 0.0
            avg_confidence = 0.0
        
        # Apply sensitivity adjustment
        sensitivity = self.parameters['signal_sensitivity']
        threshold = 0.5 / sensitivity
        
        if weighted_signal > threshold:
            signal = SignalType.BUY
        elif weighted_signal < -threshold:
            signal = SignalType.SELL
        else:
            signal = SignalType.NEUTRAL
        
        return signal, min(avg_confidence, 1.0)
    
    def _extract_current_features(self, ao: pd.Series, data: pd.DataFrame) -> Optional[List[float]]:
        """Extract features for ML prediction"""
        try:
            if len(ao) < 12 or len(data) < 12:
                return None
            
            # AO features
            ao_mean = ao.mean()
            ao_std = ao.std()
            ao_trend = np.polyfit(range(len(ao)), ao.values, 1)[0]
            ao_current = ao.iloc[-1]
            ao_max = ao.max()
            ao_min = ao.min()
            
            # Zero line features
            zero_crossings = ((ao > 0) != (ao.shift(1) > 0)).sum()
            time_above_zero = (ao > 0).sum()
            current_zero_distance = abs(ao_current)
            
            # Color analysis features
            ao_momentum = ao.diff().iloc[-1]
            color_changes = ((ao.diff() > 0) != (ao.diff().shift(1) > 0)).sum()
            
            # Price features
            price_returns = data['close'].pct_change().dropna()
            price_volatility = price_returns.std() if len(price_returns) > 0 else 0.0
            price_trend = np.polyfit(range(len(data)), data['close'].values, 1)[0]
            
            # Technical features
            momentum = ao.iloc[-1] - ao.iloc[-3] if len(ao) >= 3 else 0
            acceleration = momentum - (ao.iloc[-3] - ao.iloc[-6]) if len(ao) >= 6 else 0
            
            # Volume features
            if 'volume' in data.columns:
                volume_trend = np.polyfit(range(len(data)), data['volume'].values, 1)[0]
                volume_ratio = data['volume'].iloc[-1] / data['volume'].mean()
            else:
                volume_trend = 0
                volume_ratio = 1
            
            return [
                ao_mean, ao_std, ao_trend, ao_current, ao_max, ao_min,
                zero_crossings, time_above_zero, current_zero_distance,
                ao_momentum, color_changes, momentum, acceleration,
                price_volatility, price_trend, volume_trend, volume_ratio
            ]
        except:
            return None
    
    def _generate_signal(self, value: Any, data: pd.DataFrame) -> Tuple[Optional[SignalType], float]:
        if isinstance(value, dict) and 'signal' in value and 'confidence' in value:
            return value['signal'], value['confidence']
        return SignalType.NEUTRAL, 0.0
    
    def _get_metadata(self, data: pd.DataFrame) -> Dict[str, Any]:
        base_metadata = super()._get_metadata(data)
        base_metadata.update({
            'indicator_type': 'momentum',
            'indicator_category': 'awesome_oscillator',
            'creator': 'Bill Williams',
            'models_trained': self.models_trained,
            'adaptive_periods': self.parameters['adaptive_periods'],
            'volume_weighted': self.parameters['volume_weighted'],
            'ml_enhancement': self.parameters['ml_enhancement'],
            'multi_timeframe': self.parameters['multi_timeframe'],
            'pattern_recognition': self.parameters['pattern_recognition'],
            'saucer_detection': self.parameters['saucer_detection'],
            'twin_peaks_detection': self.parameters['twin_peaks_detection'],
            'zero_line_analysis': self.parameters['zero_line_analysis'],
            'color_analysis': self.parameters['color_analysis'],
            'data_type': 'ohlcv',
            'complexity': 'advanced',
            'bill_williams_patterns': True
        })
        return base_metadata
