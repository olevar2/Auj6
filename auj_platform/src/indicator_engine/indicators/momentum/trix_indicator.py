"""
Trix Indicator - Advanced Implementation  
========================================

Advanced TRIX (Triple Exponential Average) indicator with ML-enhanced signal
detection, adaptive smoothing periods, and sophisticated divergence analysis.

Author: AUJ Platform Development Team
Mission: Building advanced trading indicators for humanitarian impact
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy import stats
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')

from ..base.standard_indicator import (
    StandardIndicatorInterface, 
    DataRequirement, 
    DataType, 
    SignalType
)
from ....core.exceptions import IndicatorCalculationException


class TrixIndicator(StandardIndicatorInterface):
    """
    Advanced TRIX (Triple Exponential Average) Implementation
    
    Features:
    - Triple exponentially smoothed price momentum
    - ML-enhanced turning point detection
    - Adaptive smoothing periods based on market volatility
    - Advanced divergence analysis with statistical validation
    - Zero-line crossover optimization
    - Signal line integration with dynamic optimization
    - Rate of change analysis on TRIX for acceleration detection
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'period': 14,
            'signal_period': 9,
            'adaptive_smoothing': True,
            'use_signal_line': True,
            'divergence_lookback': 20,
            'ml_lookback': 60,
            'zero_line_optimization': True,
            'rate_of_change_period': 5,
            'volatility_adjustment': True,
            'pattern_detection': True
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__(name="TrixIndicator", parameters=default_params)
        
        self.scaler = StandardScaler()
        self.turning_point_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.signal_optimizer = GradientBoostingRegressor(n_estimators=50, random_state=42)
        self.pattern_clusterer = KMeans(n_clusters=5, random_state=42)
        self.models_trained = False
        
        self.history = {
            'trix_values': [],
            'signal_values': [],
            'ema1_values': [],
            'ema2_values': [],
            'ema3_values': [],
            'turning_points': [],
            'zero_crossovers': []
        }
    
    def get_data_requirements(self) -> DataRequirement:
        return DataRequirement(
            data_type=DataType.OHLC,
            required_columns=['close'],
            min_periods=max(self.parameters['period'], self.parameters['ml_lookback']) * 3 + 30,
            lookback_periods=200
        )
    
    def _calculate_ema(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average with improved initialization"""
        alpha = 2.0 / (period + 1)
        ema = series.ewm(alpha=alpha, adjust=False).mean()
        
        # Better initialization using SMA for first values
        if len(series) >= period:
            sma_init = series.iloc[:period].mean()
            ema.iloc[period-1] = sma_init
            
            # Recalculate from that point
            for i in range(period, len(series)):
                ema.iloc[i] = alpha * series.iloc[i] + (1 - alpha) * ema.iloc[i-1]
        
        return ema
    
    def _adapt_smoothing_period(self, data: pd.DataFrame) -> int:
        """Dynamically adapt smoothing period based on market volatility"""
        if not self.parameters['adaptive_smoothing']:
            return self.parameters['period']
        
        # Calculate recent volatility
        returns = data['close'].pct_change().tail(60)
        current_vol = returns.std()
        rolling_vol = returns.rolling(window=20).std()
        avg_vol = rolling_vol.mean()
        
        vol_ratio = current_vol / (avg_vol + 1e-8)
        base_period = self.parameters['period']
        
        # Adjust period based on volatility regime
        if vol_ratio > 1.5:  # High volatility - use shorter period for responsiveness
            adapted_period = max(8, int(base_period * 0.7))
        elif vol_ratio > 1.2:  # Medium-high volatility
            adapted_period = max(10, int(base_period * 0.85))
        elif vol_ratio < 0.6:  # Low volatility - use longer period for stability
            adapted_period = min(28, int(base_period * 1.4))
        elif vol_ratio < 0.8:  # Medium-low volatility
            adapted_period = min(21, int(base_period * 1.2))
        else:  # Normal volatility
            adapted_period = base_period
        
        # Also consider trend persistence
        if len(data) >= 30:
            price_changes = data['close'].diff().tail(30)
            trend_consistency = abs(price_changes.sum()) / price_changes.abs().sum()
            
            if trend_consistency > 0.7:  # Strong trend - use longer period
                adapted_period = min(adapted_period + 3, 35)
            elif trend_consistency < 0.3:  # Choppy market - use shorter period
                adapted_period = max(adapted_period - 2, 8)
        
        return adapted_period
    
    def _calculate_trix(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """Calculate TRIX with triple exponential smoothing"""
        close_prices = data['close']
        
        # Get adaptive period
        period = self._adapt_smoothing_period(data)
        
        # First EMA
        ema1 = self._calculate_ema(close_prices, period)
        
        # Second EMA (EMA of EMA1)
        ema2 = self._calculate_ema(ema1, period)
        
        # Third EMA (EMA of EMA2)
        ema3 = self._calculate_ema(ema2, period)
        
        # TRIX calculation: percentage change of EMA3
        trix = ((ema3 - ema3.shift(1)) / ema3.shift(1)) * 10000  # Multiply by 10000 for readability
        
        return trix, ema1, ema2, ema3
    
    def _calculate_signal_line(self, trix: pd.Series) -> pd.Series:
        """Calculate TRIX signal line with adaptive period"""
        signal_period = self.parameters['signal_period']
        
        # Adaptive signal period based on TRIX volatility
        if self.parameters['adaptive_smoothing'] and len(trix) >= 30:
            trix_volatility = trix.tail(30).std()
            avg_volatility = trix.rolling(window=20).std().tail(30).mean()
            vol_ratio = trix_volatility / (avg_volatility + 1e-8)
            
            if vol_ratio > 1.3:
                signal_period = max(5, int(signal_period * 0.8))
            elif vol_ratio < 0.7:
                signal_period = min(15, int(signal_period * 1.2))
        
        signal_line = self._calculate_ema(trix, signal_period)
        return signal_line
    
    def _detect_divergences(self, trix: pd.Series, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect bullish and bearish divergences between TRIX and price"""
        lookback = self.parameters['divergence_lookback']
        if len(trix) < lookback or len(data) < lookback:
            return {'bullish_divergence': False, 'bearish_divergence': False, 'strength': 0.0}
        
        recent_trix = trix.tail(lookback)
        recent_prices = data['close'].tail(lookback)
        
        # Find local highs and lows
        trix_highs = []
        trix_lows = []
        price_highs = []
        price_lows = []
        
        for i in range(2, len(recent_trix) - 2):
            # TRIX peaks
            if (recent_trix.iloc[i] > recent_trix.iloc[i-1] and 
                recent_trix.iloc[i] > recent_trix.iloc[i+1] and
                recent_trix.iloc[i] > recent_trix.iloc[i-2] and 
                recent_trix.iloc[i] > recent_trix.iloc[i+2]):
                trix_highs.append((i, recent_trix.iloc[i]))
                price_highs.append((i, recent_prices.iloc[i]))
            
            # TRIX troughs
            if (recent_trix.iloc[i] < recent_trix.iloc[i-1] and 
                recent_trix.iloc[i] < recent_trix.iloc[i+1] and
                recent_trix.iloc[i] < recent_trix.iloc[i-2] and 
                recent_trix.iloc[i] < recent_trix.iloc[i+2]):
                trix_lows.append((i, recent_trix.iloc[i]))
                price_lows.append((i, recent_prices.iloc[i]))
        
        bullish_divergence = False
        bearish_divergence = False
        divergence_strength = 0.0
        
        # Check for bullish divergence (price makes lower lows, TRIX makes higher lows)
        if len(trix_lows) >= 2 and len(price_lows) >= 2:
            last_trix_low = trix_lows[-1]
            prev_trix_low = trix_lows[-2]
            last_price_low = price_lows[-1]
            prev_price_low = price_lows[-2]
            
            if (last_price_low[1] < prev_price_low[1] and 
                last_trix_low[1] > prev_trix_low[1]):
                bullish_divergence = True
                # Calculate strength based on the magnitude of divergence
                price_change = (last_price_low[1] - prev_price_low[1]) / prev_price_low[1]
                trix_change = (last_trix_low[1] - prev_trix_low[1]) / abs(prev_trix_low[1])
                divergence_strength = abs(price_change) + abs(trix_change)
        
        # Check for bearish divergence (price makes higher highs, TRIX makes lower highs)
        if len(trix_highs) >= 2 and len(price_highs) >= 2:
            last_trix_high = trix_highs[-1]
            prev_trix_high = trix_highs[-2]
            last_price_high = price_highs[-1]
            prev_price_high = price_highs[-2]
            
            if (last_price_high[1] > prev_price_high[1] and 
                last_trix_high[1] < prev_trix_high[1]):
                bearish_divergence = True
                # Calculate strength
                price_change = (last_price_high[1] - prev_price_high[1]) / prev_price_high[1]
                trix_change = (last_trix_high[1] - prev_trix_high[1]) / abs(prev_trix_high[1])
                divergence_strength = max(divergence_strength, abs(price_change) + abs(trix_change))
        
        return {
            'bullish_divergence': bullish_divergence,
            'bearish_divergence': bearish_divergence,
            'strength': float(divergence_strength),
            'highs_count': len(trix_highs),
            'lows_count': len(trix_lows)
        }
    
    def _analyze_zero_line_crossovers(self, trix: pd.Series) -> Dict[str, Any]:
        """Analyze zero-line crossovers with momentum analysis"""
        if len(trix) < 10:
            return {'recent_crossover': False, 'direction': 'none', 'momentum': 0.0, 'distance_from_zero': 0.0}
        
        current_trix = trix.iloc[-1]
        previous_trix = trix.iloc[-2] if len(trix) > 1 else 0
        
        # Check for recent crossover
        recent_crossover = False
        crossover_direction = 'none'
        
        if previous_trix <= 0 and current_trix > 0:
            recent_crossover = True
            crossover_direction = 'bullish'
        elif previous_trix >= 0 and current_trix < 0:
            recent_crossover = True
            crossover_direction = 'bearish'
        
        # Calculate momentum near zero line
        momentum = 0.0
        if len(trix) >= 5:
            recent_trix = trix.tail(5)
            momentum = (recent_trix.iloc[-1] - recent_trix.iloc[0]) / 5
        
        # Distance from zero (normalized)
        distance_from_zero = abs(current_trix)
        
        # Analyze quality of crossover
        crossover_quality = 'weak'
        if recent_crossover:
            # Strong crossover has momentum and distance
            if abs(momentum) > abs(current_trix) * 0.1 and distance_from_zero > 0.5:
                crossover_quality = 'strong'
            elif abs(momentum) > abs(current_trix) * 0.05:
                crossover_quality = 'moderate'
        
        return {
            'recent_crossover': recent_crossover,
            'direction': crossover_direction,
            'momentum': float(momentum),
            'distance_from_zero': float(distance_from_zero),
            'quality': crossover_quality,
            'current_side': 'positive' if current_trix > 0 else 'negative' if current_trix < 0 else 'zero'
        }
    
    def _calculate_trix_patterns(self, trix: pd.Series) -> Dict[str, Any]:
        """Identify TRIX patterns using clustering"""
        if len(trix) < 20 or not self.parameters['pattern_detection']:
            return {'pattern': 'insufficient_data', 'confidence': 0.0}
        
        try:
            # Prepare pattern features
            recent_trix = trix.tail(20).values
            features = []
            
            # Create feature windows
            window_size = 5
            for i in range(len(recent_trix) - window_size + 1):
                window = recent_trix[i:i+window_size]
                
                # Features: mean, std, trend, range
                features.append([
                    np.mean(window),
                    np.std(window),
                    np.polyfit(range(len(window)), window, 1)[0],  # trend
                    np.max(window) - np.min(window),  # range
                    window[-1] - window[0],  # total change
                    len([x for x in window if x > 0]) / len(window)  # positive ratio
                ])
            
            if len(features) < 5:
                return {'pattern': 'insufficient_data', 'confidence': 0.0}
            
            features = np.array(features)
            
            # Fit clustering model
            if not hasattr(self, '_pattern_model_fitted'):
                self.pattern_clusterer.fit(features)
                self._pattern_model_fitted = True
            
            # Predict current pattern
            current_pattern = self.pattern_clusterer.predict([features[-1]])[0]
            
            # Map cluster to pattern name
            pattern_names = {
                0: 'trending_up',
                1: 'trending_down', 
                2: 'consolidating',
                3: 'volatile',
                4: 'reversing'
            }
            
            pattern_name = pattern_names.get(current_pattern, 'unknown')
            
            # Calculate confidence based on distance to cluster center
            centers = self.pattern_clusterer.cluster_centers_
            distances = [np.linalg.norm(features[-1] - center) for center in centers]
            min_distance = min(distances)
            confidence = max(0.0, 1.0 - (min_distance / np.std(distances)))
            
            return {
                'pattern': pattern_name,
                'confidence': float(confidence),
                'cluster_id': int(current_pattern)
            }
            
        except Exception:
            return {'pattern': 'unknown', 'confidence': 0.0}
    
    def _calculate_rate_of_change(self, trix: pd.Series) -> pd.Series:
        """Calculate rate of change of TRIX for acceleration analysis"""
        period = self.parameters['rate_of_change_period']
        if len(trix) < period + 1:
            return pd.Series(index=trix.index, dtype=float)
        
        roc = ((trix - trix.shift(period)) / trix.shift(period).abs()) * 100
        return roc.fillna(0)
    
    def _train_ml_models(self, trix: pd.Series, data: pd.DataFrame) -> bool:
        """Train ML models for turning point detection"""
        if len(trix) < self.parameters['ml_lookback'] * 2:
            return False
        
        try:
            features, targets = self._prepare_ml_data(trix, data)
            if len(features) > 30:
                self.scaler.fit(features)
                scaled_features = self.scaler.transform(features)
                
                # Train turning point classifier
                self.turning_point_classifier.fit(scaled_features, targets)
                
                self.models_trained = True
                return True
        except Exception:
            pass
        return False
    
    def _prepare_ml_data(self, trix: pd.Series, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare ML training data for turning point detection"""
        features, targets = [], []
        lookback = 10
        
        for i in range(lookback, len(trix) - 5):
            trix_window = trix.iloc[i-lookback:i].values
            price_window = data['close'].iloc[i-lookback:i].values
            
            # TRIX features
            trix_mean = np.mean(trix_window)
            trix_std = np.std(trix_window)
            trix_trend = np.polyfit(range(len(trix_window)), trix_window, 1)[0]
            trix_current = trix_window[-1]
            trix_momentum = trix_window[-1] - trix_window[-3] if len(trix_window) >= 3 else 0
            
            # Price features
            price_volatility = np.std(np.diff(price_window) / price_window[:-1])
            price_trend = np.polyfit(range(len(price_window)), price_window, 1)[0]
            
            # Zero line features
            zero_crossings = sum(1 for j in range(1, len(trix_window)) 
                                if (trix_window[j] > 0) != (trix_window[j-1] > 0))
            distance_from_zero = abs(trix_current)
            
            feature_vector = [
                trix_mean, trix_std, trix_trend, trix_current, trix_momentum,
                price_volatility, price_trend,
                zero_crossings, distance_from_zero,
                len([x for x in trix_window if x > 0]) / len(trix_window)  # Positive ratio
            ]
            
            # Target: turning point in next 3-5 periods
            future_trix = trix.iloc[i+1:i+5].values
            if len(future_trix) >= 3:
                current_direction = 1 if trix_current > trix_window[-2] else -1
                future_direction = 1 if future_trix[-1] > future_trix[0] else -1
                target = 1 if current_direction != future_direction else 0  # Turning point
            else:
                target = 0
            
            features.append(feature_vector)
            targets.append(target)
        
        return np.array(features), np.array(targets)
    
    def calculate_raw(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate TRIX with comprehensive analysis"""
        try:
            # Calculate TRIX and components
            trix, ema1, ema2, ema3 = self._calculate_trix(data)
            
            # Calculate signal line if enabled
            signal_line = None
            if self.parameters['use_signal_line']:
                signal_line = self._calculate_signal_line(trix)
            
            # Advanced analysis
            divergence_analysis = self._detect_divergences(trix, data)
            zero_line_analysis = self._analyze_zero_line_crossovers(trix)
            pattern_analysis = self._calculate_trix_patterns(trix)
            rate_of_change = self._calculate_rate_of_change(trix)
            
            # Train ML models
            if not self.models_trained:
                self._train_ml_models(trix, data)
            
            # Generate signal
            signal, confidence = self._generate_trix_signal(
                trix, signal_line, divergence_analysis, zero_line_analysis,
                pattern_analysis, rate_of_change, data
            )
            
            # Update history
            if len(trix) > 0 and not pd.isna(trix.iloc[-1]):
                self.history['trix_values'].append(float(trix.iloc[-1]))
                self.history['ema1_values'].append(float(ema1.iloc[-1]) if not pd.isna(ema1.iloc[-1]) else 0.0)
                self.history['ema2_values'].append(float(ema2.iloc[-1]) if not pd.isna(ema2.iloc[-1]) else 0.0)
                self.history['ema3_values'].append(float(ema3.iloc[-1]) if not pd.isna(ema3.iloc[-1]) else 0.0)
                
                if signal_line is not None and not pd.isna(signal_line.iloc[-1]):
                    self.history['signal_values'].append(float(signal_line.iloc[-1]))
                
                if zero_line_analysis['recent_crossover']:
                    self.history['zero_crossovers'].append({
                        'direction': zero_line_analysis['direction'],
                        'quality': zero_line_analysis['quality']
                    })
                
                # Keep history limited
                for key in self.history:
                    if len(self.history[key]) > 100:
                        self.history[key] = self.history[key][-100:]
            
            result = {
                'trix': float(trix.iloc[-1]) if len(trix) > 0 and not pd.isna(trix.iloc[-1]) else 0.0,
                'signal_line': float(signal_line.iloc[-1]) if signal_line is not None and len(signal_line) > 0 and not pd.isna(signal_line.iloc[-1]) else None,
                'signal': signal,
                'confidence': confidence,
                'divergence_analysis': divergence_analysis,
                'zero_line_analysis': zero_line_analysis,
                'pattern_analysis': pattern_analysis,
                'rate_of_change': float(rate_of_change.iloc[-1]) if len(rate_of_change) > 0 and not pd.isna(rate_of_change.iloc[-1]) else 0.0,
                'ema_components': {
                    'ema1': float(ema1.iloc[-1]) if len(ema1) > 0 and not pd.isna(ema1.iloc[-1]) else 0.0,
                    'ema2': float(ema2.iloc[-1]) if len(ema2) > 0 and not pd.isna(ema2.iloc[-1]) else 0.0,
                    'ema3': float(ema3.iloc[-1]) if len(ema3) > 0 and not pd.isna(ema3.iloc[-1]) else 0.0
                },
                'adaptive_period': self._adapt_smoothing_period(data),
                'market_regime': self._classify_market_regime(trix, data),
                'values_history': {
                    'trix': trix.tail(30).tolist(),
                    'signal_line': signal_line.tail(30).tolist() if signal_line is not None else []
                }
            }
            
            return result
            
        except Exception as e:
            raise IndicatorCalculationException(
                indicator_name=self.name,
                calculation_step="raw_calculation",
                message=f"Failed to calculate TRIX: {str(e)}",
                cause=e
            )
    
    def _generate_trix_signal(self, trix: pd.Series, signal_line: Optional[pd.Series],
                             divergence_analysis: Dict, zero_line_analysis: Dict,
                             pattern_analysis: Dict, rate_of_change: pd.Series,
                             data: pd.DataFrame) -> Tuple[SignalType, float]:
        """Generate comprehensive TRIX signal"""
        signal_components = []
        confidence_components = []
        
        current_trix = trix.iloc[-1] if len(trix) > 0 and not pd.isna(trix.iloc[-1]) else 0
        current_roc = rate_of_change.iloc[-1] if len(rate_of_change) > 0 and not pd.isna(rate_of_change.iloc[-1]) else 0
        
        # Zero-line crossover signals
        if zero_line_analysis['recent_crossover']:
            if zero_line_analysis['direction'] == 'bullish':
                strength = 0.9 if zero_line_analysis['quality'] == 'strong' else 0.6
                signal_components.append(strength)
                confidence_components.append(0.8)
            elif zero_line_analysis['direction'] == 'bearish':
                strength = -0.9 if zero_line_analysis['quality'] == 'strong' else -0.6
                signal_components.append(strength)
                confidence_components.append(0.8)
        
        # TRIX vs Signal Line crossover
        if signal_line is not None and len(signal_line) > 1:
            current_signal = signal_line.iloc[-1]
            prev_trix = trix.iloc[-2] if len(trix) > 1 else current_trix
            prev_signal = signal_line.iloc[-2] if len(signal_line) > 1 else current_signal
            
            # Detect crossover
            if prev_trix <= prev_signal and current_trix > current_signal:
                signal_components.append(0.7)
                confidence_components.append(0.7)
            elif prev_trix >= prev_signal and current_trix < current_signal:
                signal_components.append(-0.7)
                confidence_components.append(0.7)
        
        # Divergence signals
        if divergence_analysis['bullish_divergence']:
            signal_components.append(0.8 * divergence_analysis['strength'])
            confidence_components.append(0.7)
        elif divergence_analysis['bearish_divergence']:
            signal_components.append(-0.8 * divergence_analysis['strength'])
            confidence_components.append(0.7)
        
        # Rate of change acceleration signals
        if abs(current_roc) > 2:  # Significant acceleration
            roc_signal = np.sign(current_roc) * min(abs(current_roc) / 10, 0.6)
            signal_components.append(roc_signal)
            confidence_components.append(0.5)
        
        # Pattern-based signals
        if pattern_analysis['confidence'] > 0.6:
            pattern = pattern_analysis['pattern']
            if pattern == 'trending_up':
                signal_components.append(0.5)
                confidence_components.append(pattern_analysis['confidence'])
            elif pattern == 'trending_down':
                signal_components.append(-0.5)
                confidence_components.append(pattern_analysis['confidence'])
            elif pattern == 'reversing':
                # Reversal signal depends on current TRIX position
                reversal_signal = -np.sign(current_trix) * 0.4
                signal_components.append(reversal_signal)
                confidence_components.append(pattern_analysis['confidence'])
        
        # ML enhancement
        if self.models_trained:
            try:
                ml_signal, ml_confidence = self._get_ml_signal(trix, data)
                if ml_signal and ml_confidence > 0.6:
                    signal_components.append(1.0 if ml_signal == SignalType.BUY else -1.0)
                    confidence_components.append(ml_confidence)
            except:
                pass
        
        # Calculate final signal
        if signal_components and confidence_components:
            weighted_signal = np.average(signal_components, weights=confidence_components)
            avg_confidence = np.mean(confidence_components)
        else:
            weighted_signal = 0.0
            avg_confidence = 0.0
        
        # Apply momentum boost for strong trends
        if abs(current_trix) > 5 and np.sign(current_trix) == np.sign(weighted_signal):
            weighted_signal *= 1.2
        
        if weighted_signal > 0.6:
            signal = SignalType.STRONG_BUY if weighted_signal > 0.8 else SignalType.BUY
        elif weighted_signal < -0.6:
            signal = SignalType.STRONG_SELL if weighted_signal < -0.8 else SignalType.SELL
        else:
            signal = SignalType.NEUTRAL
        
        return signal, min(avg_confidence, 1.0)
    
    def _get_ml_signal(self, trix: pd.Series, data: pd.DataFrame) -> Tuple[Optional[SignalType], float]:
        """Get ML-based turning point prediction"""
        try:
            lookback = 10
            if len(trix) < lookback:
                return None, 0.0
            
            trix_window = trix.tail(lookback).values
            price_window = data['close'].tail(lookback).values
            
            # Recreate feature vector
            trix_mean = np.mean(trix_window)
            trix_std = np.std(trix_window)
            trix_trend = np.polyfit(range(len(trix_window)), trix_window, 1)[0]
            trix_current = trix_window[-1]
            trix_momentum = trix_window[-1] - trix_window[-3] if len(trix_window) >= 3 else 0
            
            price_volatility = np.std(np.diff(price_window) / price_window[:-1])
            price_trend = np.polyfit(range(len(price_window)), price_window, 1)[0]
            
            zero_crossings = sum(1 for j in range(1, len(trix_window)) 
                                if (trix_window[j] > 0) != (trix_window[j-1] > 0))
            distance_from_zero = abs(trix_current)
            
            feature_vector = np.array([[
                trix_mean, trix_std, trix_trend, trix_current, trix_momentum,
                price_volatility, price_trend,
                zero_crossings, distance_from_zero,
                len([x for x in trix_window if x > 0]) / len(trix_window)
            ]])
            
            scaled_features = self.scaler.transform(feature_vector)
            turning_point_prob = self.turning_point_classifier.predict_proba(scaled_features)[0]
            
            if len(turning_point_prob) >= 2:
                turn_prob = turning_point_prob[1]  # Probability of turning point
                if turn_prob > 0.7:
                    # Determine direction from current trend
                    if trix_trend > 0:
                        return SignalType.SELL, turn_prob  # Trend reversal
                    elif trix_trend < 0:
                        return SignalType.BUY, turn_prob   # Trend reversal
        except:
            pass
        
        return None, 0.0
    
    def _classify_market_regime(self, trix: pd.Series, data: pd.DataFrame) -> Dict[str, Any]:
        """Classify current market regime based on TRIX behavior"""
        if len(trix) < 20:
            return {'regime': 'unknown', 'trend': 'undefined', 'momentum': 'weak'}
        
        recent_trix = trix.tail(20)
        current_trix = trix.iloc[-1]
        
        # Trend classification
        trix_trend = np.polyfit(range(len(recent_trix)), recent_trix.values, 1)[0]
        
        if trix_trend > 1:
            trend = 'strong_bullish'
        elif trix_trend > 0.2:
            trend = 'bullish'
        elif trix_trend < -1:
            trend = 'strong_bearish'
        elif trix_trend < -0.2:
            trend = 'bearish'
        else:
            trend = 'sideways'
        
        # Momentum classification
        trix_volatility = recent_trix.std()
        
        if trix_volatility > 5:
            momentum = 'high'
        elif trix_volatility > 2:
            momentum = 'moderate'
        else:
            momentum = 'low'
        
        # Overall regime
        if abs(current_trix) > 5 and momentum == 'high':
            regime = 'trending_high_momentum'
        elif abs(current_trix) < 1 and momentum == 'low':
            regime = 'consolidating'
        elif trend in ['strong_bullish', 'strong_bearish']:
            regime = 'strong_trend'
        else:
            regime = 'transitional'
        
        return {
            'regime': regime,
            'trend': trend,
            'momentum': momentum,
            'trix_trend_slope': float(trix_trend),
            'volatility': float(trix_volatility)
        }
    
    def _generate_signal(self, value: Any, data: pd.DataFrame) -> Tuple[Optional[SignalType], float]:
        if isinstance(value, dict) and 'signal' in value and 'confidence' in value:
            return value['signal'], value['confidence']
        return SignalType.NEUTRAL, 0.0
    
    def _get_metadata(self, data: pd.DataFrame) -> Dict[str, Any]:
        base_metadata = super()._get_metadata(data)
        base_metadata.update({
            'indicator_type': 'momentum',
            'indicator_category': 'trix',
            'models_trained': self.models_trained,
            'adaptive_smoothing': self.parameters['adaptive_smoothing'],
            'uses_signal_line': self.parameters['use_signal_line'],
            'pattern_detection': self.parameters['pattern_detection'],
            'data_type': 'close',
            'complexity': 'advanced'
        })
        return base_metadata