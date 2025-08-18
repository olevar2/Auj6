"""
Coppock Curve Indicator - Advanced Implementation

The Coppock Curve is a long-term momentum indicator created by Edwin Coppock. It was originally
designed for monthly data to identify major market bottoms and buying opportunities. The indicator
uses a weighted moving average of Rate of Change (ROC) values.

Key Features:
- Classic Coppock Curve calculation with ROC periods
- Adaptive period optimization for different timeframes
- Volume-weighted Coppock for institutional flow analysis
- Multi-timeframe Coppock coherence analysis
- Advanced signal generation with divergence detection
- ML-enhanced pattern recognition for Coppock signals
- Zero-line and directional change analysis

Formula:
ROC1 = ((Close - Close[11]) / Close[11]) * 100
ROC2 = ((Close - Close[14]) / Close[14]) * 100
Coppock = WMA(ROC1 + ROC2, 10)

Signals:
- Coppock turning up from below zero: Major buy signal
- Coppock crossing above zero: Bullish momentum
- Coppock peak and decline: Potential sell signal
- Divergences with price: Reversal warnings
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


class CoppockCurveIndicator(BaseIndicator):
    """
    Advanced Coppock Curve Indicator Implementation
    
    This implementation provides comprehensive Coppock Curve analysis including:
    - Classic and adaptive Coppock Curve calculation
    - Volume-weighted analysis for institutional detection
    - Multi-timeframe analysis and coherence detection
    - Advanced signal generation with ML enhancement
    - Divergence detection and pattern recognition
    - Long-term momentum and trend analysis
    """
    
    def __init__(self, roc_short: int = 11, roc_long: int = 14, wma_period: int = 10, **kwargs):
        """
        Initialize Coppock Curve Indicator
        
        Args:
            roc_short: Short-term ROC period (default: 11)
            roc_long: Long-term ROC period (default: 14)
            wma_period: Weighted moving average period (default: 10)
            **kwargs: Additional parameters for advanced features
        """
        default_params = {
            'roc_short': roc_short,
            'roc_long': roc_long,
            'wma_period': wma_period,
            'min_roc_short': 8,
            'max_roc_short': 15,
            'min_roc_long': 10,
            'max_roc_long': 20,
            'min_wma_period': 6,
            'max_wma_period': 15,
            'smoothing_period': 3,
            'volume_weighted': True,
            'adaptive_periods': True,
            'multi_timeframe': True,
            'divergence_analysis': True,
            'smoothing_enabled': True,
            'ml_enhancement': True,
            'signal_sensitivity': 0.7,
            'ml_lookback': 60,
            'pattern_detection': True,
            'zero_line_analysis': True
        }
        default_params.update(kwargs)
        
        super().__init__(
            name="CoppockCurve",
            category="momentum",
            description="Advanced Coppock Curve long-term momentum indicator",
            parameters=default_params
        )
        
        # ML models for enhanced analysis
        self.signal_classifier = RandomForestClassifier(
            n_estimators=100, 
            max_depth=12, 
            random_state=42
        )
        self.trend_predictor = RandomForestRegressor(
            n_estimators=100, 
            max_depth=10, 
            random_state=42
        )
        self.regime_classifier = RandomForestClassifier(
            n_estimators=80,
            max_depth=8,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.models_trained = False
        
        # Historical data storage
        self.history = {
            'coppock_values': [],
            'roc_short_values': [],
            'roc_long_values': [],
            'volume_weighted_coppock': [],
            'smoothed_coppock': [],
            'zero_crossings': [],
            'signals': []
        }
    
    def get_data_requirements(self) -> DataRequirement:
        max_period = max(self.parameters['roc_long'], 
                        self.parameters['wma_period'],
                        self.parameters['ml_lookback'])
        return DataRequirement(
            data_type=DataType.OHLCV,
            required_columns=['close', 'volume'],
            min_periods=max_period * 3 + 50,
            lookback_periods=200
        )
    
    def calculate_raw(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate Coppock Curve indicator with comprehensive analysis"""
        try:
            # Optimize periods if enabled
            if self.parameters['adaptive_periods']:
                roc_short, roc_long, wma_period = self._optimize_coppock_periods(data)
            else:
                roc_short = self.parameters['roc_short']
                roc_long = self.parameters['roc_long']
                wma_period = self.parameters['wma_period']
            
            # Calculate Rate of Change values
            roc_short_values = self._calculate_roc(data['close'], roc_short)
            roc_long_values = self._calculate_roc(data['close'], roc_long)
            
            # Calculate Coppock Curve
            coppock = self._calculate_coppock_curve(roc_short_values, roc_long_values, wma_period)
            
            # Calculate volume-weighted Coppock if enabled
            if self.parameters['volume_weighted']:
                vw_coppock = self._calculate_volume_weighted_coppock(data, roc_short, roc_long, wma_period)
            else:
                vw_coppock = coppock
            
            # Apply smoothing if enabled
            if self.parameters['smoothing_enabled']:
                smoothed_coppock = vw_coppock.rolling(window=self.parameters['smoothing_period']).mean()
            else:
                smoothed_coppock = vw_coppock
            
            # Advanced analysis
            zero_line_analysis = self._analyze_zero_line_behavior(smoothed_coppock)
            timeframe_analysis = self._analyze_multi_timeframe_coppock(data)
            divergence_analysis = self._detect_coppock_divergences(smoothed_coppock, data)
            pattern_analysis = self._detect_coppock_patterns(smoothed_coppock, data)
            momentum_regime = self._classify_momentum_regime(smoothed_coppock, data)
            
            # Train ML models if not already trained
            if not self.models_trained:
                self._train_ml_models(smoothed_coppock, data)
            
            # Generate signal
            signal, confidence = self._generate_coppock_signal(
                smoothed_coppock, zero_line_analysis, timeframe_analysis,
                divergence_analysis, pattern_analysis, momentum_regime, data
            )
            
            # Update history
            current_coppock = float(smoothed_coppock.iloc[-1]) if not pd.isna(smoothed_coppock.iloc[-1]) else 0.0
            current_roc_short = float(roc_short_values.iloc[-1]) if not pd.isna(roc_short_values.iloc[-1]) else 0.0
            current_roc_long = float(roc_long_values.iloc[-1]) if not pd.isna(roc_long_values.iloc[-1]) else 0.0
            
            self.history['coppock_values'].append(current_coppock)
            self.history['roc_short_values'].append(current_roc_short)
            self.history['roc_long_values'].append(current_roc_long)
            self.history['volume_weighted_coppock'].append(float(vw_coppock.iloc[-1]) if not pd.isna(vw_coppock.iloc[-1]) else 0.0)
            
            # Keep history limited
            for key in self.history:
                if len(self.history[key]) > 100:
                    self.history[key] = self.history[key][-100:]
            
            result = {
                'coppock': current_coppock,
                'roc_short': current_roc_short,
                'roc_long': current_roc_long,
                'volume_weighted_coppock': float(vw_coppock.iloc[-1]) if not pd.isna(vw_coppock.iloc[-1]) else 0.0,
                'signal': signal,
                'confidence': confidence,
                'periods_used': {
                    'roc_short': roc_short,
                    'roc_long': roc_long,
                    'wma_period': wma_period
                },
                'zero_line_analysis': zero_line_analysis,
                'timeframe_analysis': timeframe_analysis,
                'divergence_analysis': divergence_analysis,
                'pattern_analysis': pattern_analysis,
                'momentum_regime': momentum_regime,
                'trend_direction': 'bullish' if current_coppock > 0 else 'bearish',
                'momentum_strength': min(abs(current_coppock) / 10, 1.0),  # Normalize to 0-1
                'zero_distance': abs(current_coppock),
                'values_history': {
                    'coppock': smoothed_coppock.tail(20).tolist(),
                    'roc_short': roc_short_values.tail(20).tolist(),
                    'roc_long': roc_long_values.tail(20).tolist(),
                    'zero_line': [0] * 20
                }
            }
            
            return result
            
        except Exception as e:
            raise IndicatorCalculationException(
                indicator_name=self.name,
                calculation_step="raw_calculation",
                message=f"Failed to calculate Coppock Curve: {str(e)}",
                cause=e
            )
    
    def _calculate_roc(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Rate of Change"""
        roc = ((prices - prices.shift(period)) / prices.shift(period)) * 100
        return roc.fillna(0)
    
    def _calculate_weighted_moving_average(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Weighted Moving Average"""
        weights = np.arange(1, period + 1)
        
        def wma_calc(x):
            if len(x) < period:
                return np.nan
            return np.sum(weights * x[-period:]) / weights.sum()
        
        wma = series.rolling(window=period).apply(wma_calc, raw=False)
        return wma.fillna(0)
    
    def _calculate_coppock_curve(self, roc_short: pd.Series, roc_long: pd.Series, wma_period: int) -> pd.Series:
        """Calculate classic Coppock Curve"""
        # Sum of ROC values
        roc_sum = roc_short + roc_long
        
        # Apply Weighted Moving Average
        coppock = self._calculate_weighted_moving_average(roc_sum, wma_period)
        
        return coppock
    
    def _optimize_coppock_periods(self, data: pd.DataFrame) -> Tuple[int, int, int]:
        """Optimize Coppock Curve periods based on market conditions"""
        if len(data) < 60:
            return self.parameters['roc_short'], self.parameters['roc_long'], self.parameters['wma_period']
        
        # Calculate market characteristics
        returns = data['close'].pct_change().tail(50)
        volatility = returns.std()
        trend_strength = abs(returns.mean()) / (returns.std() + 1e-8)
        
        # Base periods
        base_roc_short = self.parameters['roc_short']
        base_roc_long = self.parameters['roc_long']
        base_wma = self.parameters['wma_period']
        
        # Adjust based on volatility
        if volatility > returns.rolling(window=30).std().mean() * 1.5:
            # High volatility - shorter periods for responsiveness
            roc_short = max(self.parameters['min_roc_short'], int(base_roc_short * 0.8))
            roc_long = max(self.parameters['min_roc_long'], int(base_roc_long * 0.8))
            wma_period = max(self.parameters['min_wma_period'], int(base_wma * 0.8))
        elif volatility < returns.rolling(window=30).std().mean() * 0.7:
            # Low volatility - longer periods for stability
            roc_short = min(self.parameters['max_roc_short'], int(base_roc_short * 1.2))
            roc_long = min(self.parameters['max_roc_long'], int(base_roc_long * 1.2))
            wma_period = min(self.parameters['max_wma_period'], int(base_wma * 1.2))
        else:
            roc_short = base_roc_short
            roc_long = base_roc_long
            wma_period = base_wma
        
        # Adjust based on trend strength
        if trend_strength > 2.0:
            # Strong trend - longer periods to avoid whipsaws
            roc_short = min(self.parameters['max_roc_short'], roc_short + 2)
            roc_long = min(self.parameters['max_roc_long'], roc_long + 2)
        
        return roc_short, roc_long, wma_period
    
    def _calculate_volume_weighted_coppock(self, data: pd.DataFrame, roc_short: int, 
                                          roc_long: int, wma_period: int) -> pd.Series:
        """Calculate volume-weighted Coppock Curve"""
        if 'volume' not in data.columns:
            roc_short_values = self._calculate_roc(data['close'], roc_short)
            roc_long_values = self._calculate_roc(data['close'], roc_long)
            return self._calculate_coppock_curve(roc_short_values, roc_long_values, wma_period)
        
        # Calculate volume-weighted price
        vwap = (data['close'] * data['volume']).rolling(window=20).sum() / data['volume'].rolling(window=20).sum()
        
        # Calculate ROC on volume-weighted price
        vw_roc_short = self._calculate_roc(vwap, roc_short)
        vw_roc_long = self._calculate_roc(vwap, roc_long)
        
        # Calculate volume-weighted Coppock
        vw_coppock = self._calculate_coppock_curve(vw_roc_short, vw_roc_long, wma_period)
        
        # Regular Coppock for blending
        regular_roc_short = self._calculate_roc(data['close'], roc_short)
        regular_roc_long = self._calculate_roc(data['close'], roc_long)
        regular_coppock = self._calculate_coppock_curve(regular_roc_short, regular_roc_long, wma_period)
        
        # Blend based on volume significance
        volume_ratio = data['volume'] / data['volume'].rolling(window=20).mean()
        volume_weight = np.clip((volume_ratio - 1) * 0.3, 0, 0.7)
        
        # Higher volume = more weight on volume-weighted Coppock
        blended_coppock = regular_coppock * (1 - volume_weight) + vw_coppock * volume_weight
        
        return blended_coppock.fillna(0)
    
    def _analyze_zero_line_behavior(self, coppock: pd.Series) -> Dict[str, Any]:
        """Analyze Coppock behavior around zero line"""
        if len(coppock) < 10:
            return {'zero_crossings': 0, 'above_zero_periods': 0, 'current_position': 'unknown'}
        
        recent_coppock = coppock.tail(30)
        current_value = recent_coppock.iloc[-1]
        
        # Count zero crossings
        zero_crossings = 0
        for i in range(1, len(recent_coppock)):
            if (recent_coppock.iloc[i-1] >= 0 and recent_coppock.iloc[i] < 0) or \
               (recent_coppock.iloc[i-1] <= 0 and recent_coppock.iloc[i] > 0):
                zero_crossings += 1
        
        # Time above/below zero
        above_zero = (recent_coppock > 0).sum()
        below_zero = (recent_coppock <= 0).sum()
        
        # Current position relative to zero
        if current_value > 2:
            position = 'strong_positive'
        elif current_value > 0:
            position = 'positive'
        elif current_value < -2:
            position = 'strong_negative'
        elif current_value < 0:
            position = 'negative'
        else:
            position = 'neutral'
        
        # Distance from zero
        zero_distance = abs(current_value)
        
        # Recent trend
        if len(recent_coppock) >= 5:
            recent_trend = recent_coppock.iloc[-1] - recent_coppock.iloc[-5]
            if recent_trend > 0.5:
                trend = 'rising'
            elif recent_trend < -0.5:
                trend = 'falling'
            else:
                trend = 'sideways'
        else:
            trend = 'unknown'
        
        return {
            'zero_crossings': zero_crossings,
            'above_zero_periods': above_zero,
            'below_zero_periods': below_zero,
            'current_position': position,
            'zero_distance': float(zero_distance),
            'recent_trend': trend,
            'above_zero_ratio': float(above_zero / len(recent_coppock))
        }
    
    def _analyze_multi_timeframe_coppock(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze Coppock across multiple timeframes"""
        if not self.parameters['multi_timeframe'] or len(data) < 60:
            return {'coherence': 'unknown', 'timeframe_signals': []}
        
        timeframe_signals = []
        
        # Short-term Coppock (8,10,6)
        try:
            short_roc1 = self._calculate_roc(data['close'], 8)
            short_roc2 = self._calculate_roc(data['close'], 10)
            short_coppock = self._calculate_coppock_curve(short_roc1, short_roc2, 6)
            short_trend = 'bullish' if short_coppock.iloc[-1] > 0 else 'bearish'
            timeframe_signals.append({
                'timeframe': 'short',
                'signal': short_trend,
                'value': float(short_coppock.iloc[-1]),
                'periods': '8,10,6'
            })
        except:
            pass
        
        # Standard Coppock (11,14,10)
        try:
            std_roc1 = self._calculate_roc(data['close'], 11)
            std_roc2 = self._calculate_roc(data['close'], 14)
            std_coppock = self._calculate_coppock_curve(std_roc1, std_roc2, 10)
            std_trend = 'bullish' if std_coppock.iloc[-1] > 0 else 'bearish'
            timeframe_signals.append({
                'timeframe': 'medium',
                'signal': std_trend,
                'value': float(std_coppock.iloc[-1]),
                'periods': '11,14,10'
            })
        except:
            pass
        
        # Long-term Coppock (14,18,12)
        if len(data) >= 30:
            try:
                long_roc1 = self._calculate_roc(data['close'], 14)
                long_roc2 = self._calculate_roc(data['close'], 18)
                long_coppock = self._calculate_coppock_curve(long_roc1, long_roc2, 12)
                long_trend = 'bullish' if long_coppock.iloc[-1] > 0 else 'bearish'
                timeframe_signals.append({
                    'timeframe': 'long',
                    'signal': long_trend,
                    'value': float(long_coppock.iloc[-1]),
                    'periods': '14,18,12'
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
    
    def _detect_coppock_divergences(self, coppock: pd.Series, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect Coppock divergences with price"""
        if not self.parameters['divergence_analysis'] or len(coppock) < 40:
            return {'bullish_divergence': False, 'bearish_divergence': False, 'strength': 0.0}
        
        lookback = min(40, len(coppock) - 5)
        recent_coppock = coppock.tail(lookback)
        recent_prices = data['close'].tail(lookback)
        
        # Find peaks and troughs
        coppock_peaks, _ = find_peaks(recent_coppock.values, distance=8)
        coppock_troughs, _ = find_peaks(-recent_coppock.values, distance=8)
        
        price_peaks, _ = find_peaks(recent_prices.values, distance=8)
        price_troughs, _ = find_peaks(-recent_prices.values, distance=8)
        
        bullish_divergence = False
        bearish_divergence = False
        divergence_strength = 0.0
        
        # Bullish divergence: price lower low, Coppock higher low
        if len(coppock_troughs) >= 2 and len(price_troughs) >= 2:
            for i in range(len(coppock_troughs) - 1):
                cop_trough_1 = coppock_troughs[i]
                cop_trough_2 = coppock_troughs[i + 1]
                
                price_trough_1 = self._find_nearest_extreme(price_troughs, cop_trough_1)
                price_trough_2 = self._find_nearest_extreme(price_troughs, cop_trough_2)
                
                if price_trough_1 is not None and price_trough_2 is not None:
                    price_1 = recent_prices.iloc[price_trough_1]
                    price_2 = recent_prices.iloc[price_trough_2]
                    cop_1 = recent_coppock.iloc[cop_trough_1]
                    cop_2 = recent_coppock.iloc[cop_trough_2]
                    
                    if price_2 < price_1 and cop_2 > cop_1:
                        price_decline = (price_1 - price_2) / price_1
                        cop_improvement = abs(cop_2 - cop_1) / abs(cop_1 + 1e-8)
                        
                        if price_decline > 0.02 and cop_improvement > 0.1:
                            bullish_divergence = True
                            divergence_strength = max(divergence_strength, price_decline + cop_improvement)
        
        # Bearish divergence: price higher high, Coppock lower high
        if len(coppock_peaks) >= 2 and len(price_peaks) >= 2:
            for i in range(len(coppock_peaks) - 1):
                cop_peak_1 = coppock_peaks[i]
                cop_peak_2 = coppock_peaks[i + 1]
                
                price_peak_1 = self._find_nearest_extreme(price_peaks, cop_peak_1)
                price_peak_2 = self._find_nearest_extreme(price_peaks, cop_peak_2)
                
                if price_peak_1 is not None and price_peak_2 is not None:
                    price_1 = recent_prices.iloc[price_peak_1]
                    price_2 = recent_prices.iloc[price_peak_2]
                    cop_1 = recent_coppock.iloc[cop_peak_1]
                    cop_2 = recent_coppock.iloc[cop_peak_2]
                    
                    if price_2 > price_1 and cop_2 < cop_1:
                        price_increase = (price_2 - price_1) / price_1
                        cop_decline = abs(cop_1 - cop_2) / abs(cop_1 + 1e-8)
                        
                        if price_increase > 0.02 and cop_decline > 0.1:
                            bearish_divergence = True
                            divergence_strength = max(divergence_strength, price_increase + cop_decline)
        
        return {
            'bullish_divergence': bullish_divergence,
            'bearish_divergence': bearish_divergence,
            'strength': float(divergence_strength),
            'coppock_peaks': len(coppock_peaks),
            'coppock_troughs': len(coppock_troughs)
        }
    
    def _find_nearest_extreme(self, extremes: np.ndarray, target_idx: int) -> Optional[int]:
        """Find the nearest extreme point to a target index"""
        if len(extremes) == 0:
            return None
        
        min_distance = float('inf')
        nearest_extreme = None
        
        for extreme_idx in extremes:
            distance = abs(extreme_idx - target_idx)
            if distance < min_distance and distance < 10:
                min_distance = distance
                nearest_extreme = extreme_idx
        
        return nearest_extreme
    
    def _detect_coppock_patterns(self, coppock: pd.Series, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect Coppock patterns and formations"""
        if not self.parameters['pattern_detection'] or len(coppock) < 20:
            return {'patterns': [], 'pattern_strength': 0.0}
        
        patterns = []
        recent_coppock = coppock.tail(20)
        
        # Zero-line crossover patterns
        crossovers = self._detect_zero_crossovers(recent_coppock)
        if crossovers:
            patterns.extend(crossovers)
        
        # Turning point patterns
        turning_points = self._detect_turning_points(recent_coppock)
        if turning_points:
            patterns.extend(turning_points)
        
        # Double bottom/top patterns
        double_patterns = self._detect_double_patterns(recent_coppock)
        if double_patterns:
            patterns.extend(double_patterns)
        
        # Momentum acceleration patterns
        acceleration_patterns = self._detect_acceleration_patterns(recent_coppock)
        if acceleration_patterns:
            patterns.extend(acceleration_patterns)
        
        # Calculate overall pattern strength
        pattern_strength = len(patterns) * 0.25
        pattern_strength = min(pattern_strength, 1.0)
        
        return {
            'patterns': patterns,
            'pattern_strength': float(pattern_strength),
            'pattern_count': len(patterns)
        }
    
    def _detect_zero_crossovers(self, coppock: pd.Series) -> List[Dict[str, Any]]:
        """Detect zero-line crossover patterns"""
        patterns = []
        
        for i in range(1, len(coppock)):
            prev_val = coppock.iloc[i-1]
            curr_val = coppock.iloc[i]
            
            # Bullish crossover (major Coppock signal)
            if prev_val <= 0 and curr_val > 0:
                patterns.append({
                    'pattern': 'bullish_zero_crossover',
                    'strength': min(abs(curr_val) / 2, 1.0),
                    'position': i,
                    'description': 'Coppock crossed above zero line - Major buy signal'
                })
            
            # Bearish crossover
            elif prev_val >= 0 and curr_val < 0:
                patterns.append({
                    'pattern': 'bearish_zero_crossover',
                    'strength': min(abs(curr_val) / 2, 1.0),
                    'position': i,
                    'description': 'Coppock crossed below zero line'
                })
        
        return patterns
    
    def _detect_turning_points(self, coppock: pd.Series) -> List[Dict[str, Any]]:
        """Detect turning point patterns"""
        patterns = []
        
        if len(coppock) < 5:
            return patterns
        
        # Find significant turning points
        for i in range(2, len(coppock) - 2):
            # Potential bottom (turning up)
            if (coppock.iloc[i] < coppock.iloc[i-1] and 
                coppock.iloc[i] < coppock.iloc[i-2] and
                coppock.iloc[i] < coppock.iloc[i+1] and
                coppock.iloc[i] < coppock.iloc[i+2]):
                
                # Especially significant if below zero
                strength = 0.7 if coppock.iloc[i] < 0 else 0.5
                patterns.append({
                    'pattern': 'coppock_bottom',
                    'strength': strength,
                    'position': i,
                    'description': 'Coppock turning up from bottom'
                })
            
            # Potential top (turning down)
            elif (coppock.iloc[i] > coppock.iloc[i-1] and 
                  coppock.iloc[i] > coppock.iloc[i-2] and
                  coppock.iloc[i] > coppock.iloc[i+1] and
                  coppock.iloc[i] > coppock.iloc[i+2]):
                
                # Especially significant if above zero
                strength = 0.7 if coppock.iloc[i] > 0 else 0.5
                patterns.append({
                    'pattern': 'coppock_top',
                    'strength': strength,
                    'position': i,
                    'description': 'Coppock turning down from peak'
                })
        
        return patterns
    
    def _detect_double_patterns(self, coppock: pd.Series) -> List[Dict[str, Any]]:
        """Detect double bottom/top patterns"""
        patterns = []
        
        if len(coppock) < 10:
            return patterns
        
        # Find peaks and troughs
        peaks, _ = find_peaks(coppock.values, distance=3)
        troughs, _ = find_peaks(-coppock.values, distance=3)
        
        # Check for double bottoms
        if len(troughs) >= 2:
            for i in range(len(troughs) - 1):
                trough1_val = coppock.iloc[troughs[i]]
                trough2_val = coppock.iloc[troughs[i + 1]]
                
                # Values should be similar (within 15%)
                if abs(trough1_val - trough2_val) / abs(trough1_val + 1e-8) < 0.15:
                    patterns.append({
                        'pattern': 'double_bottom',
                        'strength': min(abs(trough1_val) / 5, 1.0),
                        'position': troughs[i + 1],
                        'description': 'Coppock double bottom pattern'
                    })
        
        # Check for double tops
        if len(peaks) >= 2:
            for i in range(len(peaks) - 1):
                peak1_val = coppock.iloc[peaks[i]]
                peak2_val = coppock.iloc[peaks[i + 1]]
                
                # Values should be similar (within 15%)
                if abs(peak1_val - peak2_val) / abs(peak1_val + 1e-8) < 0.15:
                    patterns.append({
                        'pattern': 'double_top',
                        'strength': min(abs(peak1_val) / 5, 1.0),
                        'position': peaks[i + 1],
                        'description': 'Coppock double top pattern'
                    })
        
        return patterns
    
    def _detect_acceleration_patterns(self, coppock: pd.Series) -> List[Dict[str, Any]]:
        """Detect momentum acceleration patterns"""
        patterns = []
        
        if len(coppock) < 5:
            return patterns
        
        # Calculate acceleration (second derivative)
        velocity = coppock.diff()
        acceleration = velocity.diff()
        
        for i in range(2, len(acceleration)):
            # Positive acceleration (momentum building)
            if acceleration.iloc[i] > 0.5 and velocity.iloc[i] > 0:
                patterns.append({
                    'pattern': 'positive_acceleration',
                    'strength': min(acceleration.iloc[i] / 2, 1.0),
                    'position': i,
                    'description': 'Coppock momentum accelerating upward'
                })
            
            # Negative acceleration (momentum declining)
            elif acceleration.iloc[i] < -0.5 and velocity.iloc[i] < 0:
                patterns.append({
                    'pattern': 'negative_acceleration',
                    'strength': min(abs(acceleration.iloc[i]) / 2, 1.0),
                    'position': i,
                    'description': 'Coppock momentum accelerating downward'
                })
        
        return patterns
    
    def _classify_momentum_regime(self, coppock: pd.Series, data: pd.DataFrame) -> Dict[str, Any]:
        """Classify current momentum regime"""
        if len(coppock) < 20:
            return {'regime': 'unknown', 'phase': 'unclear', 'strength': 0.0}
        
        recent_coppock = coppock.tail(20)
        current_value = recent_coppock.iloc[-1]
        
        # Regime classification
        if current_value > 5:
            regime = 'strong_bullish'
        elif current_value > 0:
            regime = 'bullish'
        elif current_value < -5:
            regime = 'strong_bearish'
        elif current_value < 0:
            regime = 'bearish'
        else:
            regime = 'neutral'
        
        # Phase analysis
        if len(recent_coppock) >= 5:
            trend = recent_coppock.iloc[-1] - recent_coppock.iloc[-5]
            if trend > 1:
                phase = 'accelerating'
            elif trend > 0.2:
                phase = 'rising'
            elif trend < -1:
                phase = 'decelerating'
            elif trend < -0.2:
                phase = 'falling'
            else:
                phase = 'consolidating'
        else:
            phase = 'unknown'
        
        # Strength measurement
        strength = min(abs(current_value) / 10, 1.0)
        
        return {
            'regime': regime,
            'phase': phase,
            'strength': float(strength),
            'current_value': float(current_value),
            'zero_distance': abs(current_value)
        }
    
    def _train_ml_models(self, coppock: pd.Series, data: pd.DataFrame) -> bool:
        """Train ML models for Coppock signal enhancement"""
        if not self.parameters['ml_enhancement'] or len(coppock) < self.parameters['ml_lookback'] * 2:
            return False
        
        try:
            features, signal_targets, trend_targets, regime_targets = self._prepare_ml_data(coppock, data)
            if len(features) > 50:
                self.scaler.fit(features)
                scaled_features = self.scaler.transform(features)
                
                # Train signal classifier
                self.signal_classifier.fit(scaled_features, signal_targets)
                
                # Train trend predictor
                self.trend_predictor.fit(scaled_features, trend_targets)
                
                # Train regime classifier
                self.regime_classifier.fit(scaled_features, regime_targets)
                
                self.models_trained = True
                return True
        except Exception:
            pass
        return False
    
    def _prepare_ml_data(self, coppock: pd.Series, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare ML training data"""
        features, signal_targets, trend_targets, regime_targets = [], [], [], []
        lookback = 15
        
        for i in range(lookback, len(coppock) - 10):
            # Feature windows
            cop_window = coppock.iloc[i-lookback:i]
            price_window = data['close'].iloc[i-lookback:i]
            
            # Coppock features
            cop_mean = cop_window.mean()
            cop_std = cop_window.std()
            cop_trend = np.polyfit(range(len(cop_window)), cop_window.values, 1)[0]
            cop_current = cop_window.iloc[-1]
            cop_max = cop_window.max()
            cop_min = cop_window.min()
            
            # Zero line features
            zero_crossings = ((cop_window > 0) != (cop_window.shift(1) > 0)).sum()
            time_above_zero = (cop_window > 0).sum()
            current_zero_distance = abs(cop_current)
            
            # Price features
            price_returns = price_window.pct_change().dropna()
            price_volatility = price_returns.std()
            price_trend = np.polyfit(range(len(price_window)), price_window.values, 1)[0]
            
            # Technical features
            momentum = cop_window.iloc[-1] - cop_window.iloc[-5] if len(cop_window) >= 5 else 0
            acceleration = momentum - (cop_window.iloc[-5] - cop_window.iloc[-10]) if len(cop_window) >= 10 else 0
            
            # Volume features
            if 'volume' in data.columns:
                volume_window = data['volume'].iloc[i-lookback:i]
                volume_trend = np.polyfit(range(len(volume_window)), volume_window.values, 1)[0]
                volume_ratio = volume_window.iloc[-1] / volume_window.mean()
            else:
                volume_trend = 0
                volume_ratio = 1
            
            feature_vector = [
                cop_mean, cop_std, cop_trend, cop_current, cop_max, cop_min,
                zero_crossings, time_above_zero, current_zero_distance,
                price_volatility, price_trend, momentum, acceleration,
                volume_trend, volume_ratio
            ]
            
            # Future targets
            future_coppock = coppock.iloc[i+1:i+11]
            
            if len(future_coppock) >= 5:
                # Signal target
                future_cop_mean = future_coppock.mean()
                if future_cop_mean > 1:
                    signal_target = 2  # Strong bullish
                elif future_cop_mean > 0:
                    signal_target = 1  # Bullish
                elif future_cop_mean < -1:
                    signal_target = 0  # Bearish
                else:
                    signal_target = 1  # Neutral
                
                # Trend target
                trend_target = future_coppock.iloc[-1] - cop_current
                
                # Regime target
                if future_cop_mean > 2:
                    regime_target = 3  # Strong bullish
                elif future_cop_mean > 0:
                    regime_target = 2  # Bullish
                elif future_cop_mean < -2:
                    regime_target = 0  # Strong bearish
                elif future_cop_mean < 0:
                    regime_target = 1  # Bearish
                else:
                    regime_target = 2  # Neutral
            else:
                signal_target = 1
                trend_target = 0.0
                regime_target = 2
            
            features.append(feature_vector)
            signal_targets.append(signal_target)
            trend_targets.append(trend_target)
            regime_targets.append(regime_target)
        
        return (np.array(features), np.array(signal_targets), 
                np.array(trend_targets), np.array(regime_targets))
    
    def _generate_coppock_signal(self, coppock: pd.Series, zero_line_analysis: Dict,
                                timeframe_analysis: Dict, divergence_analysis: Dict,
                                pattern_analysis: Dict, momentum_regime: Dict,
                                data: pd.DataFrame) -> Tuple[SignalType, float]:
        """Generate comprehensive Coppock signal"""
        signal_components = []
        confidence_components = []
        
        current_coppock = coppock.iloc[-1] if not pd.isna(coppock.iloc[-1]) else 0.0
        
        # Zero-line based signals (primary Coppock signals)
        if zero_line_analysis['current_position'] == 'positive' and zero_line_analysis['recent_trend'] == 'rising':
            signal_components.append(0.9)  # Strong signal
            confidence_components.append(0.9)
        elif zero_line_analysis['current_position'] == 'negative' and zero_line_analysis['recent_trend'] == 'falling':
            signal_components.append(-0.9)
            confidence_components.append(0.9)
        
        # Zero crossover signals (classic Coppock)
        if len(coppock) > 1:
            prev_coppock = coppock.iloc[-2]
            if prev_coppock <= 0 and current_coppock > 0:
                signal_components.append(1.0)  # Major buy signal
                confidence_components.append(1.0)
            elif prev_coppock >= 0 and current_coppock < 0:
                signal_components.append(-0.8)
                confidence_components.append(0.8)
        
        # Turning point signals
        for pattern in pattern_analysis['patterns']:
            if pattern['pattern'] == 'coppock_bottom':
                signal_components.append(0.8 * pattern['strength'])
                confidence_components.append(0.8)
            elif pattern['pattern'] == 'coppock_top':
                signal_components.append(-0.7 * pattern['strength'])
                confidence_components.append(0.7)
            elif pattern['pattern'] == 'bullish_zero_crossover':
                signal_components.append(0.9 * pattern['strength'])
                confidence_components.append(0.9)
            elif pattern['pattern'] == 'bearish_zero_crossover':
                signal_components.append(-0.8 * pattern['strength'])
                confidence_components.append(0.8)
        
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
            signal_components.append(0.8)
            confidence_components.append(0.8)
        elif coherence == 'strong_bearish' and alignment_ratio > 0.8:
            signal_components.append(-0.8)
            confidence_components.append(0.8)
        
        # Momentum regime signals
        regime = momentum_regime['regime']
        phase = momentum_regime['phase']
        
        if regime == 'strong_bullish' and phase in ['accelerating', 'rising']:
            signal_components.append(0.7)
            confidence_components.append(0.7)
        elif regime == 'strong_bearish' and phase in ['decelerating', 'falling']:
            signal_components.append(-0.7)
            confidence_components.append(0.7)
        
        # ML enhancement
        if self.models_trained:
            try:
                recent_data = data.tail(20)
                recent_coppock = coppock.tail(20)
                
                if len(recent_data) >= 15 and len(recent_coppock) >= 15:
                    features = self._extract_current_features(recent_coppock, recent_data)
                    if features is not None:
                        scaled_features = self.scaler.transform([features])
                        
                        # Get ML predictions
                        signal_prob = self.signal_classifier.predict_proba(scaled_features)[0]
                        trend_pred = self.trend_predictor.predict(scaled_features)[0]
                        
                        # Convert to signal
                        if signal_prob[2] > 0.7:  # Strong bullish
                            signal_components.append(0.7)
                            confidence_components.append(0.6)
                        elif signal_prob[0] > 0.7:  # Bearish
                            signal_components.append(-0.7)
                            confidence_components.append(0.6)
                        
                        # Trend prediction signal
                        if abs(trend_pred) > 0.5:
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
    
    def _extract_current_features(self, coppock: pd.Series, data: pd.DataFrame) -> Optional[List[float]]:
        """Extract features for ML prediction"""
        try:
            if len(coppock) < 15 or len(data) < 15:
                return None
            
            # Coppock features
            cop_mean = coppock.mean()
            cop_std = coppock.std()
            cop_trend = np.polyfit(range(len(coppock)), coppock.values, 1)[0]
            cop_current = coppock.iloc[-1]
            cop_max = coppock.max()
            cop_min = coppock.min()
            
            # Zero line features
            zero_crossings = ((coppock > 0) != (coppock.shift(1) > 0)).sum()
            time_above_zero = (coppock > 0).sum()
            current_zero_distance = abs(cop_current)
            
            # Price features
            price_returns = data['close'].pct_change().dropna()
            price_volatility = price_returns.std() if len(price_returns) > 0 else 0.0
            price_trend = np.polyfit(range(len(data)), data['close'].values, 1)[0]
            
            # Technical features
            momentum = coppock.iloc[-1] - coppock.iloc[-5] if len(coppock) >= 5 else 0
            acceleration = momentum - (coppock.iloc[-5] - coppock.iloc[-10]) if len(coppock) >= 10 else 0
            
            # Volume features
            if 'volume' in data.columns:
                volume_trend = np.polyfit(range(len(data)), data['volume'].values, 1)[0]
                volume_ratio = data['volume'].iloc[-1] / data['volume'].mean()
            else:
                volume_trend = 0
                volume_ratio = 1
            
            return [
                cop_mean, cop_std, cop_trend, cop_current, cop_max, cop_min,
                zero_crossings, time_above_zero, current_zero_distance,
                price_volatility, price_trend, momentum, acceleration,
                volume_trend, volume_ratio
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
            'indicator_category': 'coppock_curve',
            'models_trained': self.models_trained,
            'adaptive_periods': self.parameters['adaptive_periods'],
            'volume_weighted': self.parameters['volume_weighted'],
            'ml_enhancement': self.parameters['ml_enhancement'],
            'multi_timeframe': self.parameters['multi_timeframe'],
            'divergence_analysis': self.parameters['divergence_analysis'],
            'pattern_detection': self.parameters['pattern_detection'],
            'zero_line_analysis': self.parameters['zero_line_analysis'],
            'data_type': 'ohlcv',
            'complexity': 'advanced',
            'timeframe_focus': 'long_term'
        })
        return base_metadata
