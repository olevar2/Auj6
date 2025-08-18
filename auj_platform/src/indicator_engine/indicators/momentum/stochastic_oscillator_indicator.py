"""
Stochastic Oscillator Indicator - Advanced Implementation
========================================================

Advanced Stochastic Oscillator with ML-enhanced pattern recognition,
adaptive parameters, and sophisticated signal generation algorithms.

Author: AUJ Platform Development Team
Mission: Building advanced trading indicators for humanitarian impact
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from sklearn.ensemble import RandomForestRegressor, ExtraTreesClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import DBSCAN
from scipy import stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

from ..base.standard_indicator import (
    StandardIndicatorInterface, 
    DataRequirement, 
    DataType, 
    SignalType
)
from ....core.exceptions import IndicatorCalculationException


class StochasticOscillatorIndicator(StandardIndicatorInterface):
    """
    Advanced Stochastic Oscillator Implementation
    
    Features:
    - Fast and Slow Stochastic calculations with adaptive periods
    - ML-enhanced crossover signal validation
    - Advanced divergence detection with statistical significance
    - Market regime adaptation for threshold optimization
    - Volume-weighted stochastic variants
    - Pattern clustering for signal confirmation
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'k_period': 14,  # %K period
            'd_period': 3,   # %D period (smoothing)
            'smooth_k': 1,   # Smoothing for %K
            'overbought': 80,
            'oversold': 20,
            'adaptive_periods': True,
            'volume_weighted': False,
            'pattern_detection': True,
            'divergence_lookback': 25,
            'ml_lookback': 50
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__(name="StochasticOscillatorIndicator", parameters=default_params)
        
        self.scaler = RobustScaler()
        self.pattern_classifier = ExtraTreesClassifier(n_estimators=100, random_state=42)
        self.signal_predictor = RandomForestRegressor(n_estimators=50, random_state=42)
        self.pattern_clusterer = DBSCAN(eps=0.3, min_samples=5)
        self.models_trained = False
        
        self.history = {
            'k_values': [],
            'd_values': [],
            'crossovers': [],
            'patterns': [],
            'regimes': []
        }
    
    def get_data_requirements(self) -> DataRequirement:
        return DataRequirement(
            data_type=DataType.OHLCV,
            required_columns=['high', 'low', 'close', 'volume'],
            min_periods=max(self.parameters['k_period'], self.parameters['ml_lookback']) + 20,
            lookback_periods=150
        )
    
    def _calculate_stochastic_k(self, data: pd.DataFrame) -> pd.Series:
        """Calculate %K line with optional volume weighting"""
        k_period = self.parameters['k_period']
        
        if self.parameters['adaptive_periods']:
            # Adapt period based on volatility
            volatility = data['close'].pct_change().rolling(window=10).std()
            vol_ratio = volatility / volatility.rolling(window=30).mean()
            adaptive_k_period = max(5, int(k_period * vol_ratio.iloc[-1]))
        else:
            adaptive_k_period = k_period
        
        if self.parameters['volume_weighted']:
            # Volume-weighted high/low calculation
            volume = data['volume']
            high_vw = (data['high'] * volume).rolling(window=adaptive_k_period).sum() / volume.rolling(window=adaptive_k_period).sum()
            low_vw = (data['low'] * volume).rolling(window=adaptive_k_period).sum() / volume.rolling(window=adaptive_k_period).sum()
            
            lowest_low = low_vw.rolling(window=adaptive_k_period).min()
            highest_high = high_vw.rolling(window=adaptive_k_period).max()
        else:
            # Traditional calculation
            lowest_low = data['low'].rolling(window=adaptive_k_period).min()
            highest_high = data['high'].rolling(window=adaptive_k_period).max()
        
        # Calculate %K
        k_raw = 100 * (data['close'] - lowest_low) / (highest_high - lowest_low + 1e-8)
        
        # Smooth %K if required
        smooth_k = self.parameters['smooth_k']
        if smooth_k > 1:
            k_values = k_raw.rolling(window=smooth_k).mean()
        else:
            k_values = k_raw
        
        return k_values
    
    def _calculate_stochastic_d(self, k_values: pd.Series) -> pd.Series:
        """Calculate %D line (smoothed %K)"""
        d_period = self.parameters['d_period']
        d_values = k_values.rolling(window=d_period).mean()
        return d_values
    
    def _detect_crossovers(self, k_values: pd.Series, d_values: pd.Series) -> Dict[str, Any]:
        """Detect %K and %D crossovers with validation"""
        if len(k_values) < 2 or len(d_values) < 2:
            return {'type': 'none', 'strength': 0.0, 'validation': False, 'bars_ago': 0}
        
        current_k = k_values.iloc[-1]
        prev_k = k_values.iloc[-2]
        current_d = d_values.iloc[-1]
        prev_d = d_values.iloc[-2]
        
        crossover_type = 'none'
        strength = 0.0
        validation = False
        
        # Bullish crossover: %K crosses above %D
        if current_k > current_d and prev_k <= prev_d:
            crossover_type = 'bullish'
            strength = abs(current_k - current_d)
            
            # Validate with additional conditions
            if len(k_values) >= 5:
                # Check if %K is accelerating upward
                k_momentum = k_values.diff().tail(3).mean()
                # Check if crossover happens in oversold region for better signal
                in_oversold = current_d < self.parameters['oversold'] + 10
                validation = k_momentum > 0 and in_oversold
        
        # Bearish crossover: %K crosses below %D
        elif current_k < current_d and prev_k >= prev_d:
            crossover_type = 'bearish'
            strength = abs(current_k - current_d)
            
            # Validate with additional conditions
            if len(k_values) >= 5:
                # Check if %K is accelerating downward
                k_momentum = k_values.diff().tail(3).mean()
                # Check if crossover happens in overbought region for better signal
                in_overbought = current_d > self.parameters['overbought'] - 10
                validation = k_momentum < 0 and in_overbought
        
        return {
            'type': crossover_type,
            'strength': float(strength),
            'validation': validation,
            'bars_ago': 0 if crossover_type != 'none' else self._count_bars_since_last_crossover(k_values, d_values)
        }
    
    def _count_bars_since_last_crossover(self, k_values: pd.Series, d_values: pd.Series) -> int:
        """Count bars since last crossover"""
        for i in range(1, min(20, len(k_values))):
            if i >= len(k_values) or i >= len(d_values):
                break
            
            current_above = k_values.iloc[-i] > d_values.iloc[-i]
            prev_above = k_values.iloc[-i-1] > d_values.iloc[-i-1]
            
            if current_above != prev_above:
                return i
        
        return 20  # Max lookback
    
    def _detect_divergences(self, k_values: pd.Series, d_values: pd.Series, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect stochastic-price divergences"""
        lookback = self.parameters['divergence_lookback']
        if len(k_values) < lookback:
            return {'bullish': False, 'bearish': False, 'strength': 0.0, 'confidence': 0.0}
        
        recent_k = k_values.tail(lookback)
        recent_d = d_values.tail(lookback)
        recent_prices = data['close'].tail(lookback)
        
        # Use %D for divergence detection (smoother)
        stoch_values = recent_d
        
        # Find peaks and troughs
        stoch_peaks, _ = find_peaks(stoch_values.values, distance=3, prominence=5)
        stoch_troughs, _ = find_peaks(-stoch_values.values, distance=3, prominence=5)
        
        price_peaks, _ = find_peaks(recent_prices.values, distance=3)
        price_troughs, _ = find_peaks(-recent_prices.values, distance=3)
        
        bullish_divergence = False
        bearish_divergence = False
        divergence_strength = 0.0
        
        # Bullish divergence: price lower low, stochastic higher low
        if len(stoch_troughs) >= 2 and len(price_troughs) >= 2:
            last_stoch_trough = stoch_values.iloc[stoch_troughs[-1]]
            prev_stoch_trough = stoch_values.iloc[stoch_troughs[-2]]
            
            last_price_trough = recent_prices.iloc[price_troughs[-1]]
            prev_price_trough = recent_prices.iloc[price_troughs[-2]]
            
            if (last_price_trough < prev_price_trough and 
                last_stoch_trough > prev_stoch_trough and
                last_stoch_trough < 30):  # Confirm in oversold region
                bullish_divergence = True
                price_decline = abs(last_price_trough - prev_price_trough) / prev_price_trough
                stoch_improvement = abs(last_stoch_trough - prev_stoch_trough) / prev_stoch_trough if prev_stoch_trough != 0 else 0
                divergence_strength = min(stoch_improvement / (price_decline + 1e-8), 2.0)
        
        # Bearish divergence: price higher high, stochastic lower high
        if len(stoch_peaks) >= 2 and len(price_peaks) >= 2:
            last_stoch_peak = stoch_values.iloc[stoch_peaks[-1]]
            prev_stoch_peak = stoch_values.iloc[stoch_peaks[-2]]
            
            last_price_peak = recent_prices.iloc[price_peaks[-1]]
            prev_price_peak = recent_prices.iloc[price_peaks[-2]]
            
            if (last_price_peak > prev_price_peak and 
                last_stoch_peak < prev_stoch_peak and
                last_stoch_peak > 70):  # Confirm in overbought region
                bearish_divergence = True
                price_increase = abs(last_price_peak - prev_price_peak) / prev_price_peak
                stoch_decline = abs(last_stoch_peak - prev_stoch_peak) / prev_stoch_peak if prev_stoch_peak != 0 else 0
                divergence_strength = min(stoch_decline / (price_increase + 1e-8), 2.0)
        
        # Calculate confidence based on statistical significance
        if bullish_divergence or bearish_divergence:
            # Correlation test
            correlation = np.corrcoef(stoch_values.values, recent_prices.values)[0, 1]
            confidence = max(0, 1 - abs(correlation)) * min(divergence_strength, 1.0)
        else:
            confidence = 0.0
        
        return {
            'bullish': bullish_divergence,
            'bearish': bearish_divergence,
            'strength': float(divergence_strength),
            'confidence': float(confidence)
        }
    
    def _detect_patterns(self, k_values: pd.Series, d_values: pd.Series) -> Dict[str, Any]:
        """Detect stochastic patterns using clustering"""
        if not self.parameters['pattern_detection'] or len(k_values) < 15:
            return {'pattern': 'insufficient_data', 'confidence': 0.0, 'cluster': -1}
        
        try:
            # Prepare pattern features
            lookback = 10
            recent_k = k_values.tail(lookback).values
            recent_d = d_values.tail(lookback).values
            
            # Create pattern features
            k_trend = np.polyfit(range(len(recent_k)), recent_k, 1)[0]
            d_trend = np.polyfit(range(len(recent_d)), recent_d, 1)[0]
            k_volatility = np.std(recent_k)
            d_volatility = np.std(recent_d)
            crossover_count = len([i for i in range(1, len(recent_k)) 
                                 if (recent_k[i] > recent_d[i]) != (recent_k[i-1] > recent_d[i-1])])
            
            pattern_features = np.array([[
                k_trend, d_trend, k_volatility, d_volatility, crossover_count,
                recent_k[-1], recent_d[-1], np.mean(recent_k), np.mean(recent_d)
            ]])
            
            if hasattr(self, 'pattern_examples') and len(self.pattern_examples) > 10:
                # Fit clusterer if we have enough examples
                if not hasattr(self, 'pattern_clusterer_fitted'):
                    self.pattern_clusterer.fit(self.pattern_examples)
                    self.pattern_clusterer_fitted = True
                
                # Predict cluster
                cluster = self.pattern_clusterer.fit_predict(
                    np.vstack([self.pattern_examples, pattern_features])
                )[-1]
                
                # Interpret cluster
                if cluster == -1:
                    pattern_type = 'anomaly'
                    confidence = 0.3
                else:
                    # Simple pattern classification based on features
                    if k_trend > 0 and d_trend > 0:
                        pattern_type = 'bullish_momentum'
                    elif k_trend < 0 and d_trend < 0:
                        pattern_type = 'bearish_momentum'
                    elif crossover_count >= 2:
                        pattern_type = 'choppy_sideways'
                    else:
                        pattern_type = 'stable_trend'
                    
                    confidence = 0.7
                
                return {
                    'pattern': pattern_type,
                    'confidence': confidence,
                    'cluster': int(cluster)
                }
            else:
                # Initialize pattern examples
                if not hasattr(self, 'pattern_examples'):
                    self.pattern_examples = []
                self.pattern_examples.append(pattern_features[0])
                
                # Keep limited history
                if len(self.pattern_examples) > 100:
                    self.pattern_examples = self.pattern_examples[-100:]
        except:
            pass
        
        return {'pattern': 'unknown', 'confidence': 0.0, 'cluster': -1}
    
    def _calculate_adaptive_thresholds(self, k_values: pd.Series, d_values: pd.Series, 
                                     data: pd.DataFrame) -> Dict[str, float]:
        """Calculate adaptive overbought/oversold thresholds"""
        if len(k_values) < 50:
            return {
                'overbought': self.parameters['overbought'],
                'oversold': self.parameters['oversold']
            }
        
        # Market volatility factor
        volatility = data['close'].pct_change().rolling(window=20).std().iloc[-1]
        avg_volatility = data['close'].pct_change().rolling(window=100).std().mean()
        vol_ratio = volatility / (avg_volatility + 1e-8)
        
        # Stochastic distribution analysis
        recent_d = d_values.tail(50).dropna()
        d_percentiles = np.percentile(recent_d, [10, 90])
        
        # Adjust thresholds
        if vol_ratio > 1.5:  # High volatility - wider thresholds
            overbought = min(90, max(75, d_percentiles[1] + 5))
            oversold = max(10, min(25, d_percentiles[0] - 5))
        elif vol_ratio < 0.7:  # Low volatility - tighter thresholds
            overbought = min(85, max(70, d_percentiles[1]))
            oversold = max(15, min(30, d_percentiles[0]))
        else:  # Normal volatility
            overbought = self.parameters['overbought']
            oversold = self.parameters['oversold']
        
        return {
            'overbought': float(overbought),
            'oversold': float(oversold)
        }
    
    def _train_ml_models(self, k_values: pd.Series, d_values: pd.Series, data: pd.DataFrame) -> bool:
        """Train ML models for signal enhancement"""
        if len(k_values) < self.parameters['ml_lookback'] * 2:
            return False
        
        try:
            features, targets = self._prepare_ml_data(k_values, d_values, data)
            if len(features) > 40:
                # Train pattern classifier
                self.scaler.fit(features)
                scaled_features = self.scaler.transform(features)
                self.pattern_classifier.fit(scaled_features, targets)
                
                # Train signal predictor
                future_returns = data['close'].pct_change().shift(-5).tail(len(features)).dropna().values
                if len(future_returns) == len(scaled_features):
                    self.signal_predictor.fit(scaled_features, future_returns)
                
                self.models_trained = True
                return True
        except:
            pass
        return False
    
    def _prepare_ml_data(self, k_values: pd.Series, d_values: pd.Series, 
                        data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare ML training data"""
        features, targets = [], []
        lookback = 12
        
        for i in range(lookback, len(k_values) - 5):
            k_window = k_values.iloc[i-lookback:i].values
            d_window = d_values.iloc[i-lookback:i].values
            price_window = data['close'].iloc[i-lookback:i].values
            volume_window = data['volume'].iloc[i-lookback:i].values
            
            # Calculate features
            k_trend = np.polyfit(range(len(k_window)), k_window, 1)[0]
            d_trend = np.polyfit(range(len(d_window)), d_window, 1)[0]
            k_volatility = np.std(k_window)
            d_volatility = np.std(d_window)
            
            crossovers = len([j for j in range(1, len(k_window)) 
                            if (k_window[j] > d_window[j]) != (k_window[j-1] > d_window[j-1])])
            
            feature_vector = [
                np.mean(k_window), np.std(k_window), k_window[-1],
                np.mean(d_window), np.std(d_window), d_window[-1],
                k_trend, d_trend, k_volatility, d_volatility,
                crossovers,
                len([x for x in d_window if x > 80]) / len(d_window),  # Overbought ratio
                len([x for x in d_window if x < 20]) / len(d_window),  # Oversold ratio
                np.corrcoef(d_window, price_window)[0, 1] if len(set(d_window)) > 1 else 0,
                volume_window[-1] / np.mean(volume_window) if np.mean(volume_window) != 0 else 1
            ]
            
            # Target: future price movement classification
            future_return = (data['close'].iloc[i+5] - data['close'].iloc[i]) / data['close'].iloc[i]
            if future_return > 0.015:
                target = 2  # Strong bullish
            elif future_return > 0.005:
                target = 1  # Weak bullish
            elif future_return < -0.015:
                target = 0  # Strong bearish
            else:
                target = 1  # Neutral/weak
            
            features.append(feature_vector)
            targets.append(target)
        
        return np.array(features), np.array(targets)
    
    def calculate_raw(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate Stochastic Oscillator with advanced analysis"""
        try:
            # Calculate basic stochastic components
            k_values = self._calculate_stochastic_k(data)
            d_values = self._calculate_stochastic_d(k_values)
            
            # Advanced analysis
            crossovers = self._detect_crossovers(k_values, d_values)
            divergences = self._detect_divergences(k_values, d_values, data)
            patterns = self._detect_patterns(k_values, d_values)
            adaptive_thresholds = self._calculate_adaptive_thresholds(k_values, d_values, data)
            
            # Train ML models
            if not self.models_trained:
                self._train_ml_models(k_values, d_values, data)
            
            # Generate signal
            signal, confidence = self._generate_stochastic_signal(
                k_values, d_values, crossovers, divergences, patterns, 
                adaptive_thresholds, data
            )
            
            # Update history
            self.history['k_values'].append(float(k_values.iloc[-1]))
            self.history['d_values'].append(float(d_values.iloc[-1]))
            self.history['crossovers'].append(crossovers)
            self.history['patterns'].append(patterns)
            
            # Keep history limited
            for key in self.history:
                if len(self.history[key]) > 100:
                    self.history[key] = self.history[key][-100:]
            
            result = {
                'k_percent': float(k_values.iloc[-1]),
                'd_percent': float(d_values.iloc[-1]),
                'signal': signal,
                'confidence': confidence,
                'crossovers': crossovers,
                'divergences': divergences,
                'patterns': patterns,
                'adaptive_thresholds': adaptive_thresholds,
                'momentum': self._calculate_momentum(k_values, d_values),
                'extremes': self._analyze_extremes(k_values, d_values, adaptive_thresholds),
                'values_history': {
                    'k_percent': k_values.tail(30).tolist(),
                    'd_percent': d_values.tail(30).tolist()
                }
            }
            
            return result
            
        except Exception as e:
            raise IndicatorCalculationException(
                indicator_name=self.name,
                calculation_step="raw_calculation",
                message=f"Failed to calculate Stochastic Oscillator: {str(e)}",
                cause=e
            )
    
    def _generate_stochastic_signal(self, k_values: pd.Series, d_values: pd.Series,
                                  crossovers: Dict, divergences: Dict, patterns: Dict,
                                  thresholds: Dict, data: pd.DataFrame) -> Tuple[SignalType, float]:
        """Generate comprehensive stochastic signal"""
        signal_components = []
        confidence_components = []
        
        current_k = k_values.iloc[-1]
        current_d = d_values.iloc[-1]
        
        # Crossover signals
        if crossovers['type'] == 'bullish' and crossovers['validation']:
            signal_components.append(1.0)
            confidence_components.append(0.8)
        elif crossovers['type'] == 'bearish' and crossovers['validation']:
            signal_components.append(-1.0)
            confidence_components.append(0.8)
        elif crossovers['type'] != 'none':
            # Unvalidated crossover - lower confidence
            signal_strength = 1.0 if crossovers['type'] == 'bullish' else -1.0
            signal_components.append(signal_strength * 0.5)
            confidence_components.append(0.4)
        
        # Extreme level signals
        if current_d < thresholds['oversold']:
            signal_strength = (thresholds['oversold'] - current_d) / thresholds['oversold']
            signal_components.append(signal_strength)
            confidence_components.append(0.6)
        elif current_d > thresholds['overbought']:
            signal_strength = (current_d - thresholds['overbought']) / (100 - thresholds['overbought'])
            signal_components.append(-signal_strength)
            confidence_components.append(0.6)
        
        # Divergence signals
        if divergences['bullish']:
            signal_components.append(divergences['strength'])
            confidence_components.append(divergences['confidence'])
        elif divergences['bearish']:
            signal_components.append(-divergences['strength'])
            confidence_components.append(divergences['confidence'])
        
        # Pattern signals
        if patterns['pattern'] == 'bullish_momentum':
            signal_components.append(patterns['confidence'])
            confidence_components.append(patterns['confidence'])
        elif patterns['pattern'] == 'bearish_momentum':
            signal_components.append(-patterns['confidence'])
            confidence_components.append(patterns['confidence'])
        
        # ML enhancement
        if self.models_trained:
            try:
                ml_signal, ml_confidence = self._get_ml_signal(k_values, d_values, data)
                if ml_signal and ml_confidence > 0.6:
                    signal_components.append(1.0 if ml_signal == SignalType.BUY else -1.0)
                    confidence_components.append(ml_confidence)
            except:
                pass
        
        # Momentum consideration
        if len(k_values) >= 3:
            k_momentum = k_values.iloc[-1] - k_values.iloc[-3]
            d_momentum = d_values.iloc[-1] - d_values.iloc[-3]
            
            if k_momentum > 5 and d_momentum > 5:
                signal_components.append(0.4)
                confidence_components.append(0.3)
            elif k_momentum < -5 and d_momentum < -5:
                signal_components.append(-0.4)
                confidence_components.append(0.3)
        
        # Calculate final signal
        if signal_components and confidence_components:
            weighted_signal = np.average(signal_components, weights=confidence_components)
            avg_confidence = np.mean(confidence_components)
        else:
            weighted_signal = 0.0
            avg_confidence = 0.0
        
        if weighted_signal > 0.6:
            signal = SignalType.STRONG_BUY if weighted_signal > 0.8 else SignalType.BUY
        elif weighted_signal < -0.6:
            signal = SignalType.STRONG_SELL if weighted_signal < -0.8 else SignalType.SELL
        else:
            signal = SignalType.NEUTRAL
        
        return signal, min(avg_confidence, 1.0)
    
    def _get_ml_signal(self, k_values: pd.Series, d_values: pd.Series, 
                      data: pd.DataFrame) -> Tuple[Optional[SignalType], float]:
        """Get ML-based signal prediction"""
        try:
            lookback = 12
            if len(k_values) < lookback:
                return None, 0.0
            
            k_window = k_values.tail(lookback).values
            d_window = d_values.tail(lookback).values
            price_window = data['close'].tail(lookback).values
            volume_window = data['volume'].tail(lookback).values
            
            # Recreate feature vector
            k_trend = np.polyfit(range(len(k_window)), k_window, 1)[0]
            d_trend = np.polyfit(range(len(d_window)), d_window, 1)[0]
            k_volatility = np.std(k_window)
            d_volatility = np.std(d_window)
            
            crossovers = len([j for j in range(1, len(k_window)) 
                            if (k_window[j] > d_window[j]) != (k_window[j-1] > d_window[j-1])])
            
            feature_vector = np.array([[
                np.mean(k_window), np.std(k_window), k_window[-1],
                np.mean(d_window), np.std(d_window), d_window[-1],
                k_trend, d_trend, k_volatility, d_volatility,
                crossovers,
                len([x for x in d_window if x > 80]) / len(d_window),
                len([x for x in d_window if x < 20]) / len(d_window),
                np.corrcoef(d_window, price_window)[0, 1] if len(set(d_window)) > 1 else 0,
                volume_window[-1] / np.mean(volume_window) if np.mean(volume_window) != 0 else 1
            ]])
            
            scaled_features = self.scaler.transform(feature_vector)
            ml_proba = self.pattern_classifier.predict_proba(scaled_features)[0]
            
            if len(ml_proba) >= 3:
                max_prob_idx = np.argmax(ml_proba)
                max_prob = ml_proba[max_prob_idx]
                
                if max_prob > 0.7:
                    if max_prob_idx == 2:  # Strong bullish
                        return SignalType.BUY, max_prob
                    elif max_prob_idx == 0:  # Strong bearish
                        return SignalType.SELL, max_prob
        except:
            pass
        
        return None, 0.0
    
    def _calculate_momentum(self, k_values: pd.Series, d_values: pd.Series) -> Dict[str, Any]:
        """Calculate stochastic momentum characteristics"""
        if len(k_values) < 5:
            return {'k_momentum': 0.0, 'd_momentum': 0.0, 'momentum_agreement': False}
        
        k_momentum = k_values.diff().tail(3).mean()
        d_momentum = d_values.diff().tail(3).mean()
        
        momentum_agreement = (k_momentum > 0 and d_momentum > 0) or (k_momentum < 0 and d_momentum < 0)
        
        return {
            'k_momentum': float(k_momentum),
            'd_momentum': float(d_momentum),
            'momentum_agreement': momentum_agreement
        }
    
    def _analyze_extremes(self, k_values: pd.Series, d_values: pd.Series, 
                         thresholds: Dict) -> Dict[str, Any]:
        """Analyze extreme stochastic conditions"""
        current_k = k_values.iloc[-1]
        current_d = d_values.iloc[-1]
        
        # Determine extreme condition
        if current_d > thresholds['overbought']:
            extreme_type = 'overbought'
            intensity = (current_d - thresholds['overbought']) / (100 - thresholds['overbought'])
        elif current_d < thresholds['oversold']:
            extreme_type = 'oversold'
            intensity = (thresholds['oversold'] - current_d) / thresholds['oversold']
        else:
            extreme_type = 'none'
            intensity = 0.0
        
        # Count duration in extreme
        duration = 0
        if len(d_values) >= 10:
            recent_d = d_values.tail(10)
            if extreme_type == 'overbought':
                duration = len([x for x in recent_d if x > thresholds['overbought']])
            elif extreme_type == 'oversold':
                duration = len([x for x in recent_d if x < thresholds['oversold']])
        
        return {
            'type': extreme_type,
            'intensity': float(intensity),
            'duration': duration,
            'k_extreme': float(current_k),
            'd_extreme': float(current_d)
        }
    
    def _generate_signal(self, value: Any, data: pd.DataFrame) -> Tuple[Optional[SignalType], float]:
        if isinstance(value, dict) and 'signal' in value and 'confidence' in value:
            return value['signal'], value['confidence']
        return SignalType.NEUTRAL, 0.0
    
    def _get_metadata(self, data: pd.DataFrame) -> Dict[str, Any]:
        base_metadata = super()._get_metadata(data)
        base_metadata.update({
            'indicator_type': 'momentum',
            'indicator_category': 'stochastic_oscillator',
            'models_trained': self.models_trained,
            'adaptive_periods': self.parameters['adaptive_periods'],
            'volume_weighted': self.parameters['volume_weighted'],
            'data_type': 'ohlcv',
            'complexity': 'advanced'
        })
        return base_metadata