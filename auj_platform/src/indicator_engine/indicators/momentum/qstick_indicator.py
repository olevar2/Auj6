"""
QStick Indicator - Advanced Implementation

The QStick indicator measures the momentum and trend of candlestick bodies by analyzing
the difference between close and open prices over a specified period. It helps identify
market sentiment and the strength of buying/selling pressure.

Key Features:
- Classic QStick calculation with period optimization
- Volume-weighted QStick for institutional flow analysis
- Multi-timeframe QStick coherence analysis
- Advanced signal generation with divergence detection
- Adaptive smoothing and threshold detection
- ML-enhanced pattern recognition for QStick signals

Formula:
QStick = SMA(Close - Open, period)

Signals:
- QStick > 0: Bullish momentum (closes > opens)
- QStick < 0: Bearish momentum (closes < opens)
- QStick crossing zero line: Momentum shift
- Divergences with price: Potential reversal signals
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from scipy import stats
from scipy.signal import find_peaks
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from ...base.indicator_base import BaseIndicator
from ....core.exceptions import IndicatorCalculationException
from ....data_providers.base_provider import DataRequirement, DataType
from ....core.signal_types import SignalType


class QStickIndicator(BaseIndicator):
    """
    Advanced QStick Indicator Implementation
    
    This implementation provides comprehensive QStick analysis including:
    - Classic and volume-weighted QStick calculation
    - Multi-timeframe analysis and coherence detection
    - Advanced signal generation with ML enhancement
    - Divergence detection and pattern recognition
    - Adaptive parameters and optimization
    """
    
    def __init__(self, period: int = 14, **kwargs):
        """
        Initialize QStick Indicator
        
        Args:
            period: Period for QStick calculation (default: 14)
            **kwargs: Additional parameters for advanced features
        """
        default_params = {
            'period': period,
            'min_period': 5,
            'max_period': 30,
            'smoothing_period': 3,
            'threshold_percentile': 75,
            'volume_weighted': True,
            'adaptive_periods': True,
            'multi_timeframe': True,
            'divergence_analysis': True,
            'smoothing_enabled': True,
            'ml_enhancement': True,
            'signal_sensitivity': 0.6,
            'ml_lookback': 50,
            'pattern_detection': True
        }
        default_params.update(kwargs)
        
        super().__init__(
            name="QStick",
            category="momentum",
            description="Advanced QStick momentum indicator with ML enhancement",
            parameters=default_params
        )
        
        # ML models for enhanced analysis
        self.signal_classifier = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10, 
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
            'qstick_values': [],
            'volume_weighted_qstick': [],
            'smoothed_qstick': [],
            'thresholds': [],
            'divergences': [],
            'signals': []
        }
    
    def get_data_requirements(self) -> DataRequirement:
        max_period = max(self.parameters['period'], 
                        self.parameters['max_period'],
                        self.parameters['ml_lookback'])
        return DataRequirement(
            data_type=DataType.OHLCV,
            required_columns=['open', 'high', 'low', 'close', 'volume'],
            min_periods=max_period * 2 + 30,
            lookback_periods=150
        )
    
    def calculate_raw(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate QStick indicator with comprehensive analysis"""
        try:
            # Optimize period if enabled
            if self.parameters['adaptive_periods']:
                period = self._optimize_qstick_period(data)
            else:
                period = self.parameters['period']
            
            # Calculate primary QStick
            qstick = self._calculate_qstick(data, period)
            
            # Calculate volume-weighted QStick if enabled
            if self.parameters['volume_weighted']:
                vw_qstick = self._calculate_volume_weighted_qstick(data, period)
            else:
                vw_qstick = qstick
            
            # Apply smoothing if enabled
            if self.parameters['smoothing_enabled']:
                smoothed_qstick = vw_qstick.rolling(window=self.parameters['smoothing_period']).mean()
            else:
                smoothed_qstick = vw_qstick
            
            # Calculate dynamic thresholds
            upper_threshold, lower_threshold = self._calculate_dynamic_thresholds(smoothed_qstick)
            
            # Advanced analysis
            timeframe_analysis = self._analyze_multi_timeframe_qstick(data)
            divergence_analysis = self._detect_qstick_divergences(smoothed_qstick, data)
            pattern_analysis = self._detect_qstick_patterns(smoothed_qstick, data)
            
            # Train ML models if not already trained
            if not self.models_trained:
                self._train_ml_models(smoothed_qstick, data)
            
            # Generate signal
            signal, confidence = self._generate_qstick_signal(
                smoothed_qstick, upper_threshold, lower_threshold,
                timeframe_analysis, divergence_analysis, pattern_analysis, data
            )
            
            # Update history
            current_qstick = float(smoothed_qstick.iloc[-1]) if not pd.isna(smoothed_qstick.iloc[-1]) else 0.0
            current_vw_qstick = float(vw_qstick.iloc[-1]) if not pd.isna(vw_qstick.iloc[-1]) else 0.0
            
            self.history['qstick_values'].append(current_qstick)
            self.history['volume_weighted_qstick'].append(current_vw_qstick)
            self.history['smoothed_qstick'].append(current_qstick)
            self.history['thresholds'].append({
                'upper': upper_threshold,
                'lower': lower_threshold
            })
            
            # Keep history limited
            for key in self.history:
                if len(self.history[key]) > 100:
                    self.history[key] = self.history[key][-100:]
            
            result = {
                'qstick': current_qstick,
                'volume_weighted_qstick': current_vw_qstick,
                'upper_threshold': upper_threshold,
                'lower_threshold': lower_threshold,
                'signal': signal,
                'confidence': confidence,
                'period_used': period,
                'timeframe_analysis': timeframe_analysis,
                'divergence_analysis': divergence_analysis,
                'pattern_analysis': pattern_analysis,
                'market_sentiment': self._analyze_market_sentiment(current_qstick, smoothed_qstick),
                'qstick_strength': min(abs(current_qstick) * 100, 1.0),  # Normalize to 0-1
                'trend_direction': 'bullish' if current_qstick > 0 else 'bearish',
                'values_history': {
                    'qstick': smoothed_qstick.tail(20).tolist(),
                    'volume_weighted': vw_qstick.tail(20).tolist(),
                    'thresholds': {
                        'upper': [upper_threshold] * 20,
                        'lower': [lower_threshold] * 20
                    }
                }
            }
            
            return result
            
        except Exception as e:
            raise IndicatorCalculationException(
                indicator_name=self.name,
                calculation_step="raw_calculation",
                message=f"Failed to calculate QStick: {str(e)}",
                cause=e
            )
    
    def _calculate_qstick(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate classic QStick indicator"""
        qstick_values = data['close'] - data['open']
        qstick = qstick_values.rolling(window=period).mean()
        return qstick.fillna(0)
    
    def _optimize_qstick_period(self, data: pd.DataFrame) -> int:
        """Optimize QStick period based on market conditions"""
        if len(data) < 60:
            return self.parameters['period']
        
        # Calculate market volatility
        returns = data['close'].pct_change().tail(30)
        current_vol = returns.std()
        avg_vol = data['close'].pct_change().rolling(window=60).std().mean()
        vol_ratio = current_vol / (avg_vol + 1e-8)
        
        # Calculate trend strength using QStick at different periods
        base_period = self.parameters['period']
        test_periods = [int(base_period * 0.7), base_period, int(base_period * 1.3)]
        
        best_period = base_period
        best_score = 0
        
        for test_period in test_periods:
            if test_period < 5 or test_period > len(data) // 4:
                continue
            
            test_qstick = self._calculate_qstick(data, test_period)
            if len(test_qstick.dropna()) < 20:
                continue
            
            # Score based on signal clarity and trend consistency
            recent_qstick = test_qstick.tail(20).dropna()
            if len(recent_qstick) == 0:
                continue
            
            # Trend consistency score
            trend_changes = (recent_qstick * recent_qstick.shift(1) < 0).sum()
            consistency_score = 1 - (trend_changes / len(recent_qstick))
            
            # Signal strength score
            strength_score = abs(recent_qstick.mean()) / (recent_qstick.std() + 1e-8)
            
            # Combined score
            total_score = consistency_score * 0.6 + strength_score * 0.4
            
            if total_score > best_score:
                best_score = total_score
                best_period = test_period
        
        # Adjust for volatility
        if vol_ratio > 1.5:  # High volatility - shorter period
            best_period = int(best_period * 0.8)
        elif vol_ratio < 0.6:  # Low volatility - longer period
            best_period = int(best_period * 1.2)
        
        # Apply bounds
        return max(self.parameters['min_period'], 
                  min(best_period, self.parameters['max_period']))
    
    def _calculate_volume_weighted_qstick(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate volume-weighted QStick for institutional flow analysis"""
        if 'volume' not in data.columns:
            return self._calculate_qstick(data, period)
        
        qstick_values = data['close'] - data['open']
        volume = data['volume']
        
        # Volume-weighted QStick values
        weighted_qstick_values = qstick_values * volume
        
        # Calculate rolling sums
        weighted_sum = weighted_qstick_values.rolling(window=period).sum()
        volume_sum = volume.rolling(window=period).sum()
        
        # Volume-weighted QStick
        vw_qstick = weighted_sum / (volume_sum + 1e-8)
        
        return vw_qstick.fillna(0)
    
    def _calculate_dynamic_thresholds(self, qstick: pd.Series) -> Tuple[float, float]:
        """Calculate dynamic overbought/oversold thresholds"""
        if len(qstick) < 20:
            return 0.001, -0.001  # Default thresholds for QStick
        
        recent_qstick = qstick.tail(60).dropna()
        
        if len(recent_qstick) < 10:
            return 0.001, -0.001
        
        # Use percentile-based thresholds
        percentile = self.parameters['threshold_percentile']
        upper_threshold = np.percentile(recent_qstick, percentile)
        lower_threshold = np.percentile(recent_qstick, 100 - percentile)
        
        # Ensure minimum separation
        qstick_std = recent_qstick.std()
        min_separation = qstick_std * 0.5
        
        if upper_threshold - lower_threshold < min_separation:
            mid_point = recent_qstick.median()
            upper_threshold = mid_point + min_separation / 2
            lower_threshold = mid_point - min_separation / 2
        
        # Apply reasonable bounds for QStick (typically smaller values than other momentum indicators)
        upper_threshold = max(0.0005, min(upper_threshold, 0.01))
        lower_threshold = min(-0.0005, max(lower_threshold, -0.01))
        
        return float(upper_threshold), float(lower_threshold)
    
    def _analyze_multi_timeframe_qstick(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze QStick coherence across multiple timeframes"""
        if not self.parameters['multi_timeframe'] or len(data) < 50:
            return {'coherence': 'unknown', 'timeframe_signals': []}
        
        timeframe_signals = []
        
        # Short-term QStick (7-period)
        try:
            short_qstick = self._calculate_qstick(data, 7)
            short_trend = 'bullish' if short_qstick.iloc[-1] > 0 else 'bearish'
            timeframe_signals.append({
                'timeframe': 'short',
                'signal': short_trend,
                'value': float(short_qstick.iloc[-1]),
                'period': 7
            })
        except:
            pass
        
        # Medium-term QStick (14-period) - standard
        try:
            medium_qstick = self._calculate_qstick(data, 14)
            medium_trend = 'bullish' if medium_qstick.iloc[-1] > 0 else 'bearish'
            timeframe_signals.append({
                'timeframe': 'medium',
                'signal': medium_trend,
                'value': float(medium_qstick.iloc[-1]),
                'period': 14
            })
        except:
            pass
        
        # Long-term QStick (21-period)
        if len(data) >= 21:
            try:
                long_qstick = self._calculate_qstick(data, 21)
                long_trend = 'bullish' if long_qstick.iloc[-1] > 0 else 'bearish'
                timeframe_signals.append({
                    'timeframe': 'long',
                    'signal': long_trend,
                    'value': float(long_qstick.iloc[-1]),
                    'period': 21
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
            strength_alignment = np.std(values) / (np.mean(values) + 1e-8)  # Lower = more aligned
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
    
    def _detect_qstick_divergences(self, qstick: pd.Series, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect QStick divergences with price"""
        if not self.parameters['divergence_analysis'] or len(qstick) < 30:
            return {'bullish_divergence': False, 'bearish_divergence': False, 'strength': 0.0}
        
        lookback = min(30, len(qstick) - 5)
        recent_qstick = qstick.tail(lookback)
        recent_prices = data['close'].tail(lookback)
        
        # Find peaks and troughs
        qstick_peaks, _ = find_peaks(recent_qstick.values, distance=5)
        qstick_troughs, _ = find_peaks(-recent_qstick.values, distance=5)
        
        price_peaks, _ = find_peaks(recent_prices.values, distance=5)
        price_troughs, _ = find_peaks(-recent_prices.values, distance=5)
        
        bullish_divergence = False
        bearish_divergence = False
        divergence_strength = 0.0
        
        # Bullish divergence: price lower low, QStick higher low
        if len(qstick_troughs) >= 2 and len(price_troughs) >= 2:
            for i in range(len(qstick_troughs) - 1):
                qs_trough_1 = qstick_troughs[i]
                qs_trough_2 = qstick_troughs[i + 1]
                
                price_trough_1 = self._find_nearest_extreme(price_troughs, qs_trough_1)
                price_trough_2 = self._find_nearest_extreme(price_troughs, qs_trough_2)
                
                if price_trough_1 is not None and price_trough_2 is not None:
                    price_1 = recent_prices.iloc[price_trough_1]
                    price_2 = recent_prices.iloc[price_trough_2]
                    qs_1 = recent_qstick.iloc[qs_trough_1]
                    qs_2 = recent_qstick.iloc[qs_trough_2]
                    
                    if price_2 < price_1 and qs_2 > qs_1:
                        price_decline = (price_1 - price_2) / price_1
                        qs_improvement = abs(qs_2 - qs_1) / abs(qs_1 + 1e-8)
                        
                        if price_decline > 0.01 and qs_improvement > 0.1:
                            bullish_divergence = True
                            divergence_strength = max(divergence_strength, price_decline + qs_improvement)
        
        # Bearish divergence: price higher high, QStick lower high
        if len(qstick_peaks) >= 2 and len(price_peaks) >= 2:
            for i in range(len(qstick_peaks) - 1):
                qs_peak_1 = qstick_peaks[i]
                qs_peak_2 = qstick_peaks[i + 1]
                
                price_peak_1 = self._find_nearest_extreme(price_peaks, qs_peak_1)
                price_peak_2 = self._find_nearest_extreme(price_peaks, qs_peak_2)
                
                if price_peak_1 is not None and price_peak_2 is not None:
                    price_1 = recent_prices.iloc[price_peak_1]
                    price_2 = recent_prices.iloc[price_peak_2]
                    qs_1 = recent_qstick.iloc[qs_peak_1]
                    qs_2 = recent_qstick.iloc[qs_peak_2]
                    
                    if price_2 > price_1 and qs_2 < qs_1:
                        price_increase = (price_2 - price_1) / price_1
                        qs_decline = abs(qs_1 - qs_2) / abs(qs_1 + 1e-8)
                        
                        if price_increase > 0.01 and qs_decline > 0.1:
                            bearish_divergence = True
                            divergence_strength = max(divergence_strength, price_increase + qs_decline)
        
        return {
            'bullish_divergence': bullish_divergence,
            'bearish_divergence': bearish_divergence,
            'strength': float(divergence_strength),
            'qstick_peaks': len(qstick_peaks),
            'qstick_troughs': len(qstick_troughs)
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
    
    def _detect_qstick_patterns(self, qstick: pd.Series, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect QStick patterns and formations"""
        if not self.parameters['pattern_detection'] or len(qstick) < 20:
            return {'patterns': [], 'pattern_strength': 0.0}
        
        patterns = []
        recent_qstick = qstick.tail(20)
        
        # Zero-line crossover patterns
        crossovers = self._detect_zero_line_crossovers(recent_qstick)
        if crossovers:
            patterns.extend(crossovers)
        
        # QStick momentum patterns
        momentum_patterns = self._detect_momentum_patterns(recent_qstick)
        if momentum_patterns:
            patterns.extend(momentum_patterns)
        
        # Double top/bottom patterns
        double_patterns = self._detect_double_patterns(recent_qstick)
        if double_patterns:
            patterns.extend(double_patterns)
        
        # Calculate overall pattern strength
        pattern_strength = len(patterns) * 0.2  # Each pattern adds 0.2 to strength
        pattern_strength = min(pattern_strength, 1.0)
        
        return {
            'patterns': patterns,
            'pattern_strength': float(pattern_strength),
            'pattern_count': len(patterns)
        }
    
    def _detect_zero_line_crossovers(self, qstick: pd.Series) -> List[Dict[str, Any]]:
        """Detect zero-line crossover patterns"""
        patterns = []
        
        for i in range(1, len(qstick)):
            prev_val = qstick.iloc[i-1]
            curr_val = qstick.iloc[i]
            
            # Bullish crossover
            if prev_val <= 0 and curr_val > 0:
                patterns.append({
                    'pattern': 'bullish_crossover',
                    'strength': min(abs(curr_val) * 1000, 1.0),  # Normalize
                    'position': i,
                    'description': 'QStick crossed above zero line'
                })
            
            # Bearish crossover
            elif prev_val >= 0 and curr_val < 0:
                patterns.append({
                    'pattern': 'bearish_crossover',
                    'strength': min(abs(curr_val) * 1000, 1.0),  # Normalize
                    'position': i,
                    'description': 'QStick crossed below zero line'
                })
        
        return patterns
    
    def _detect_momentum_patterns(self, qstick: pd.Series) -> List[Dict[str, Any]]:
        """Detect momentum acceleration/deceleration patterns"""
        patterns = []
        
        if len(qstick) < 5:
            return patterns
        
        # Calculate momentum change
        momentum_change = qstick.diff()
        
        # Acceleration patterns
        for i in range(2, len(momentum_change)):
            if (momentum_change.iloc[i] > 0 and momentum_change.iloc[i-1] > 0 and 
                momentum_change.iloc[i] > momentum_change.iloc[i-1]):
                patterns.append({
                    'pattern': 'momentum_acceleration',
                    'strength': min(momentum_change.iloc[i] * 10000, 1.0),
                    'position': i,
                    'description': 'QStick momentum accelerating'
                })
            
            elif (momentum_change.iloc[i] < 0 and momentum_change.iloc[i-1] < 0 and 
                  momentum_change.iloc[i] < momentum_change.iloc[i-1]):
                patterns.append({
                    'pattern': 'momentum_deceleration',
                    'strength': min(abs(momentum_change.iloc[i]) * 10000, 1.0),
                    'position': i,
                    'description': 'QStick momentum decelerating'
                })
        
        return patterns
    
    def _detect_double_patterns(self, qstick: pd.Series) -> List[Dict[str, Any]]:
        """Detect double top/bottom patterns in QStick"""
        patterns = []
        
        if len(qstick) < 10:
            return patterns
        
        # Find peaks and troughs
        peaks, _ = find_peaks(qstick.values, distance=3)
        troughs, _ = find_peaks(-qstick.values, distance=3)
        
        # Check for double tops
        if len(peaks) >= 2:
            for i in range(len(peaks) - 1):
                peak1_val = qstick.iloc[peaks[i]]
                peak2_val = qstick.iloc[peaks[i + 1]]
                
                # Values should be similar (within 20%)
                if abs(peak1_val - peak2_val) / abs(peak1_val + 1e-8) < 0.2:
                    patterns.append({
                        'pattern': 'double_top',
                        'strength': min(abs(peak1_val) * 1000, 1.0),
                        'position': peaks[i + 1],
                        'description': 'QStick double top pattern'
                    })
        
        # Check for double bottoms
        if len(troughs) >= 2:
            for i in range(len(troughs) - 1):
                trough1_val = qstick.iloc[troughs[i]]
                trough2_val = qstick.iloc[troughs[i + 1]]
                
                # Values should be similar (within 20%)
                if abs(trough1_val - trough2_val) / abs(trough1_val + 1e-8) < 0.2:
                    patterns.append({
                        'pattern': 'double_bottom',
                        'strength': min(abs(trough1_val) * 1000, 1.0),
                        'position': troughs[i + 1],
                        'description': 'QStick double bottom pattern'
                    })
        
        return patterns
    
    def _train_ml_models(self, qstick: pd.Series, data: pd.DataFrame) -> bool:
        """Train ML models for QStick signal enhancement"""
        if not self.parameters['ml_enhancement'] or len(qstick) < self.parameters['ml_lookback'] * 2:
            return False
        
        try:
            features, signal_targets, trend_targets = self._prepare_ml_data(qstick, data)
            if len(features) > 30:
                self.scaler.fit(features)
                scaled_features = self.scaler.transform(features)
                
                # Train signal classifier
                self.signal_classifier.fit(scaled_features, signal_targets)
                
                # Train trend predictor
                self.trend_predictor.fit(scaled_features, trend_targets)
                
                self.models_trained = True
                return True
        except Exception:
            pass
        return False
    
    def _prepare_ml_data(self, qstick: pd.Series, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare ML training data"""
        features, signal_targets, trend_targets = [], [], []
        lookback = 10
        
        for i in range(lookback, len(qstick) - 5):
            # Feature windows
            qs_window = qstick.iloc[i-lookback:i]
            price_window = data['close'].iloc[i-lookback:i]
            
            # QStick features
            qs_mean = qs_window.mean()
            qs_std = qs_window.std()
            qs_trend = np.polyfit(range(len(qs_window)), qs_window.values, 1)[0]
            qs_current = qs_window.iloc[-1]
            qs_max = qs_window.max()
            qs_min = qs_window.min()
            
            # Price features
            price_returns = price_window.pct_change().dropna()
            price_volatility = price_returns.std()
            price_trend = np.polyfit(range(len(price_window)), price_window.values, 1)[0]
            
            # Technical features
            zero_crossings = ((qs_window > 0) != (qs_window.shift(1) > 0)).sum()
            momentum = qs_window.iloc[-1] - qs_window.iloc[-3] if len(qs_window) >= 3 else 0
            
            # Volume features
            if 'volume' in data.columns:
                volume_window = data['volume'].iloc[i-lookback:i]
                volume_trend = np.polyfit(range(len(volume_window)), volume_window.values, 1)[0]
                volume_ratio = volume_window.iloc[-1] / volume_window.mean()
            else:
                volume_trend = 0
                volume_ratio = 1
            
            feature_vector = [
                qs_mean, qs_std, qs_trend, qs_current, qs_max, qs_min,
                price_volatility, price_trend, zero_crossings, momentum,
                volume_trend, volume_ratio
            ]
            
            # Future targets
            future_qstick = qstick.iloc[i+1:i+6]
            future_price = data['close'].iloc[i+1:i+6]
            
            if len(future_qstick) >= 3:
                # Signal target
                future_qs_mean = future_qstick.mean()
                if future_qs_mean > 0.001:
                    signal_target = 2  # Strong bullish
                elif future_qs_mean > 0:
                    signal_target = 1  # Bullish
                elif future_qs_mean < -0.001:
                    signal_target = 0  # Bearish
                else:
                    signal_target = 1  # Neutral
                
                # Trend target
                trend_target = future_qstick.iloc[-1] - qs_current
            else:
                signal_target = 1
                trend_target = 0.0
            
            features.append(feature_vector)
            signal_targets.append(signal_target)
            trend_targets.append(trend_target)
        
        return np.array(features), np.array(signal_targets), np.array(trend_targets)
    
    def _analyze_market_sentiment(self, current_qstick: float, qstick_series: pd.Series) -> Dict[str, Any]:
        """Analyze market sentiment based on QStick values"""
        if len(qstick_series) < 10:
            return {'sentiment': 'unknown', 'strength': 0.0, 'consistency': 0.0}
        
        recent_qstick = qstick_series.tail(10)
        
        # Sentiment classification
        if current_qstick > 0.002:
            sentiment = 'strong_bullish'
        elif current_qstick > 0:
            sentiment = 'bullish'
        elif current_qstick < -0.002:
            sentiment = 'strong_bearish'
        elif current_qstick < 0:
            sentiment = 'bearish'
        else:
            sentiment = 'neutral'
        
        # Sentiment strength
        strength = min(abs(current_qstick) * 500, 1.0)  # Normalize to 0-1
        
        # Sentiment consistency
        positive_count = (recent_qstick > 0).sum()
        consistency = abs(positive_count - 5) / 5  # Distance from 50-50 split
        consistency = 1 - consistency  # Invert so higher = more consistent
        
        return {
            'sentiment': sentiment,
            'strength': float(strength),
            'consistency': float(consistency),
            'positive_ratio': float(positive_count / len(recent_qstick))
        }
    
    def _generate_qstick_signal(self, qstick: pd.Series, upper_threshold: float, lower_threshold: float,
                               timeframe_analysis: Dict, divergence_analysis: Dict,
                               pattern_analysis: Dict, data: pd.DataFrame) -> Tuple[SignalType, float]:
        """Generate comprehensive QStick signal"""
        signal_components = []
        confidence_components = []
        
        current_qstick = qstick.iloc[-1] if not pd.isna(qstick.iloc[-1]) else 0.0
        
        # Threshold-based signals
        if current_qstick > upper_threshold:
            signal_components.append(0.8)
            confidence_components.append(0.8)
        elif current_qstick < lower_threshold:
            signal_components.append(-0.8)
            confidence_components.append(0.8)
        
        # Zero-line signals
        if current_qstick > 0.0005:
            signal_components.append(0.6)
            confidence_components.append(0.7)
        elif current_qstick < -0.0005:
            signal_components.append(-0.6)
            confidence_components.append(0.7)
        
        # Zero-line crossover signals
        if len(qstick) > 1:
            prev_qstick = qstick.iloc[-2]
            if prev_qstick <= 0 and current_qstick > 0:
                signal_components.append(0.9)
                confidence_components.append(0.9)
            elif prev_qstick >= 0 and current_qstick < 0:
                signal_components.append(-0.9)
                confidence_components.append(0.9)
        
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
            signal_components.append(0.7)
            confidence_components.append(0.8)
        elif coherence == 'strong_bearish' and alignment_ratio > 0.8:
            signal_components.append(-0.7)
            confidence_components.append(0.8)
        elif coherence in ['mixed_bullish', 'mixed_bearish']:
            direction = 0.5 if 'bullish' in coherence else -0.5
            signal_components.append(direction * alignment_ratio)
            confidence_components.append(0.6)
        
        # Pattern signals
        pattern_strength = pattern_analysis['pattern_strength']
        for pattern in pattern_analysis['patterns']:
            if pattern['pattern'] in ['bullish_crossover', 'momentum_acceleration', 'double_bottom']:
                signal_components.append(0.6 * pattern['strength'])
                confidence_components.append(0.7)
            elif pattern['pattern'] in ['bearish_crossover', 'momentum_deceleration', 'double_top']:
                signal_components.append(-0.6 * pattern['strength'])
                confidence_components.append(0.7)
        
        # ML enhancement
        if self.models_trained:
            try:
                # Prepare current features
                recent_data = data.tail(15)
                recent_qstick = qstick.tail(15)
                
                if len(recent_data) >= 10 and len(recent_qstick) >= 10:
                    features = self._extract_current_features(recent_qstick, recent_data)
                    if features is not None:
                        scaled_features = self.scaler.transform([features])
                        
                        # Get ML predictions
                        signal_prob = self.signal_classifier.predict_proba(scaled_features)[0]
                        trend_pred = self.trend_predictor.predict(scaled_features)[0]
                        
                        # Convert to signal
                        if signal_prob[2] > 0.6:  # Strong bullish
                            signal_components.append(0.8)
                            confidence_components.append(0.7)
                        elif signal_prob[0] > 0.6:  # Bearish
                            signal_components.append(-0.8)
                            confidence_components.append(0.7)
                        
                        # Trend prediction signal
                        if abs(trend_pred) > 0.0005:
                            direction = 0.5 if trend_pred > 0 else -0.5
                            signal_components.append(direction)
                            confidence_components.append(0.6)
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
        threshold = 0.5 / sensitivity  # Higher sensitivity = lower threshold
        
        if weighted_signal > threshold:
            signal = SignalType.BUY
        elif weighted_signal < -threshold:
            signal = SignalType.SELL
        else:
            signal = SignalType.NEUTRAL
        
        return signal, min(avg_confidence, 1.0)
    
    def _extract_current_features(self, qstick: pd.Series, data: pd.DataFrame) -> Optional[List[float]]:
        """Extract features for ML prediction"""
        try:
            if len(qstick) < 10 or len(data) < 10:
                return None
            
            # QStick features
            qs_mean = qstick.mean()
            qs_std = qstick.std()
            qs_trend = np.polyfit(range(len(qstick)), qstick.values, 1)[0]
            qs_current = qstick.iloc[-1]
            qs_max = qstick.max()
            qs_min = qstick.min()
            
            # Price features
            price_returns = data['close'].pct_change().dropna()
            price_volatility = price_returns.std() if len(price_returns) > 0 else 0.0
            price_trend = np.polyfit(range(len(data)), data['close'].values, 1)[0]
            
            # Technical features
            zero_crossings = ((qstick > 0) != (qstick.shift(1) > 0)).sum()
            momentum = qstick.iloc[-1] - qstick.iloc[-3] if len(qstick) >= 3 else 0
            
            # Volume features
            if 'volume' in data.columns:
                volume_trend = np.polyfit(range(len(data)), data['volume'].values, 1)[0]
                volume_ratio = data['volume'].iloc[-1] / data['volume'].mean()
            else:
                volume_trend = 0
                volume_ratio = 1
            
            return [
                qs_mean, qs_std, qs_trend, qs_current, qs_max, qs_min,
                price_volatility, price_trend, zero_crossings, momentum,
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
            'indicator_category': 'qstick',
            'models_trained': self.models_trained,
            'adaptive_periods': self.parameters['adaptive_periods'],
            'volume_weighted': self.parameters['volume_weighted'],
            'ml_enhancement': self.parameters['ml_enhancement'],
            'multi_timeframe': self.parameters['multi_timeframe'],
            'divergence_analysis': self.parameters['divergence_analysis'],
            'pattern_detection': self.parameters['pattern_detection'],
            'data_type': 'ohlcv',
            'complexity': 'advanced'
        })
        return base_metadata
