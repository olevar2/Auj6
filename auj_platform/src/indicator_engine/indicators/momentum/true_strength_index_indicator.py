"""
True Strength Index (TSI) Indicator - Advanced Implementation
============================================================

Advanced TSI with ML-enhanced momentum detection, adaptive smoothing,
and sophisticated signal generation for institutional-grade trading.

Author: AUJ Platform Development Team
Mission: Building advanced trading indicators for humanitarian impact
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from scipy import stats
from scipy.optimize import differential_evolution
import warnings
warnings.filterwarnings('ignore')

from ..base.standard_indicator import (
    StandardIndicatorInterface, 
    DataRequirement, 
    DataType, 
    SignalType
)
from ....core.exceptions import IndicatorCalculationException


class TrueStrengthIndexIndicator(StandardIndicatorInterface):
    """
    Advanced True Strength Index (TSI) Implementation
    
    Features:
    - Double-smoothed momentum with adaptive parameters
    - ML-enhanced signal detection and regime classification
    - Dynamic overbought/oversold threshold optimization
    - Multi-timeframe momentum convergence analysis
    - Volume-weighted momentum calculations
    - Advanced divergence detection with statistical validation
    - Signal line integration with optimized crossover detection
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'first_smoothing': 25,
            'second_smoothing': 13,
            'signal_period': 7,
            'overbought': 25,
            'oversold': -25,
            'adaptive_smoothing': True,
            'volume_weighted': True,
            'use_signal_line': True,
            'divergence_lookback': 20,
            'ml_lookback': 60,
            'optimization_enabled': True,
            'multi_timeframe': True,
            'pca_components': 3
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__(name="TrueStrengthIndexIndicator", parameters=default_params)
        
        self.scaler = StandardScaler()
        self.robust_scaler = RobustScaler()
        self.momentum_classifier = ExtraTreesClassifier(n_estimators=100, random_state=42)
        self.threshold_optimizer = GradientBoostingRegressor(n_estimators=50, random_state=42)
        self.pca = PCA(n_components=self.parameters['pca_components'])
        self.models_trained = False
        
        self.history = {
            'tsi_values': [],
            'signal_values': [],
            'momentum_values': [],
            'double_smooth_values': [],
            'threshold_values': [],
            'regime_states': []
        }
    
    def get_data_requirements(self) -> DataRequirement:
        max_period = max(self.parameters['first_smoothing'], 
                        self.parameters['second_smoothing'],
                        self.parameters['ml_lookback'])
        return DataRequirement(
            data_type=DataType.OHLCV,
            required_columns=['close', 'volume'],
            min_periods=max_period * 2 + 50,
            lookback_periods=200
        )
    
    def _calculate_ema(self, series: pd.Series, period: int, adjust: bool = False) -> pd.Series:
        """Enhanced EMA calculation with better initialization"""
        if len(series) == 0:
            return pd.Series(dtype=float)
        
        alpha = 2.0 / (period + 1)
        ema = series.ewm(alpha=alpha, adjust=adjust).mean()
        
        # Better initialization using SMA for robustness
        if len(series) >= period:
            sma_init = series.iloc[:period].mean()
            if not pd.isna(sma_init):
                ema.iloc[period-1] = sma_init
                
                # Recalculate from that point for consistency
                for i in range(period, len(series)):
                    if not pd.isna(series.iloc[i]) and not pd.isna(ema.iloc[i-1]):
                        ema.iloc[i] = alpha * series.iloc[i] + (1 - alpha) * ema.iloc[i-1]
        
        return ema
    
    def _adapt_smoothing_parameters(self, data: pd.DataFrame) -> Tuple[int, int]:
        """Dynamically adapt smoothing parameters based on market conditions"""
        if not self.parameters['adaptive_smoothing']:
            return self.parameters['first_smoothing'], self.parameters['second_smoothing']
        
        # Calculate market volatility regime
        returns = data['close'].pct_change().tail(60)
        current_vol = returns.std()
        rolling_vol = returns.rolling(window=20).std()
        avg_vol = rolling_vol.mean()
        vol_ratio = current_vol / (avg_vol + 1e-8)
        
        # Calculate trend persistence
        price_changes = data['close'].diff().tail(40)
        trend_consistency = abs(price_changes.sum()) / (price_changes.abs().sum() + 1e-8)
        
        # Calculate volume regime
        if 'volume' in data.columns:
            volume_ratio = data['volume'].tail(20).mean() / data['volume'].tail(60).mean()
        else:
            volume_ratio = 1.0
        
        base_first = self.parameters['first_smoothing']
        base_second = self.parameters['second_smoothing']
        
        # Adjust for volatility
        if vol_ratio > 1.5:  # High volatility - shorter periods for responsiveness
            first_smoothing = max(15, int(base_first * 0.6))
            second_smoothing = max(8, int(base_second * 0.7))
        elif vol_ratio < 0.6:  # Low volatility - longer periods for stability
            first_smoothing = min(40, int(base_first * 1.3))
            second_smoothing = min(20, int(base_second * 1.2))
        else:  # Normal volatility
            first_smoothing = base_first
            second_smoothing = base_second
        
        # Adjust for trend persistence
        if trend_consistency > 0.7:  # Strong trend - longer periods
            first_smoothing = min(first_smoothing + 5, 45)
            second_smoothing = min(second_smoothing + 2, 18)
        elif trend_consistency < 0.3:  # Choppy market - shorter periods
            first_smoothing = max(first_smoothing - 3, 12)
            second_smoothing = max(second_smoothing - 1, 6)
        
        # Adjust for volume regime
        if volume_ratio > 1.3:  # High volume - slightly shorter for sensitivity
            first_smoothing = max(first_smoothing - 2, 10)
            second_smoothing = max(second_smoothing - 1, 5)
        
        return first_smoothing, second_smoothing
    
    def _calculate_momentum(self, data: pd.DataFrame) -> pd.Series:
        """Calculate momentum with optional volume weighting"""
        close = data['close']
        momentum = close.diff()
        
        if self.parameters['volume_weighted'] and 'volume' in data.columns:
            volume = data['volume']
            
            # Normalize volume to avoid extreme values
            volume_norm = volume / volume.rolling(window=20).mean()
            volume_norm = volume_norm.fillna(1.0).clip(0.1, 3.0)
            
            # Weight momentum by normalized volume
            momentum = momentum * volume_norm
        
        return momentum
    
    def _calculate_tsi(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate TSI with double smoothing"""
        # Get adaptive parameters
        first_period, second_period = self._adapt_smoothing_parameters(data)
        
        # Calculate momentum
        momentum = self._calculate_momentum(data)
        abs_momentum = momentum.abs()
        
        # First smoothing
        smooth_momentum = self._calculate_ema(momentum, first_period)
        smooth_abs_momentum = self._calculate_ema(abs_momentum, first_period)
        
        # Second smoothing (double smoothing)
        double_smooth_momentum = self._calculate_ema(smooth_momentum, second_period)
        double_smooth_abs_momentum = self._calculate_ema(smooth_abs_momentum, second_period)
        
        # Calculate TSI
        tsi = 100 * (double_smooth_momentum / (double_smooth_abs_momentum + 1e-8))
        
        return tsi, double_smooth_momentum, double_smooth_abs_momentum
    
    def _calculate_signal_line(self, tsi: pd.Series) -> pd.Series:
        """Calculate TSI signal line with adaptive period"""
        signal_period = self.parameters['signal_period']
        
        # Adapt signal period based on TSI volatility
        if self.parameters['adaptive_smoothing'] and len(tsi) >= 30:
            tsi_volatility = tsi.tail(30).std()
            avg_volatility = tsi.rolling(window=20).std().tail(30).mean()
            vol_ratio = tsi_volatility / (avg_volatility + 1e-8)
            
            if vol_ratio > 1.3:
                signal_period = max(4, int(signal_period * 0.8))
            elif vol_ratio < 0.7:
                signal_period = min(12, int(signal_period * 1.2))
        
        signal_line = self._calculate_ema(tsi, signal_period)
        return signal_line
    
    def _optimize_thresholds(self, tsi: pd.Series, data: pd.DataFrame) -> Dict[str, float]:
        """Optimize overbought/oversold thresholds using genetic algorithm"""
        if not self.parameters['optimization_enabled'] or len(tsi) < 100:
            return {
                'overbought': self.parameters['overbought'],
                'oversold': self.parameters['oversold']
            }
        
        try:
            recent_tsi = tsi.tail(100).dropna()
            recent_returns = data['close'].pct_change().tail(len(recent_tsi)).dropna()
            
            if len(recent_tsi) != len(recent_returns):
                min_len = min(len(recent_tsi), len(recent_returns))
                recent_tsi = recent_tsi.tail(min_len)
                recent_returns = recent_returns.tail(min_len)
            
            def objective(params):
                overbought, oversold = params
                if overbought <= oversold or overbought > 50 or oversold < -50:
                    return 1e6
                
                signals = []
                for i in range(1, len(recent_tsi)):
                    if recent_tsi.iloc[i-1] > overbought and recent_tsi.iloc[i] <= overbought:
                        signals.append(-1)  # Sell signal
                    elif recent_tsi.iloc[i-1] < oversold and recent_tsi.iloc[i] >= oversold:
                        signals.append(1)   # Buy signal
                    else:
                        signals.append(0)   # Hold
                
                if not signals:
                    return 1e6
                
                returns_aligned = recent_returns.iloc[1:len(signals)+1].values
                signals = np.array(signals[:len(returns_aligned)])
                
                strategy_returns = signals * returns_aligned
                
                if len(strategy_returns) == 0 or np.std(strategy_returns) == 0:
                    return 1e6
                
                # Multi-objective: maximize return, minimize risk
                avg_return = np.mean(strategy_returns)
                volatility = np.std(strategy_returns)
                sharpe = avg_return / (volatility + 1e-8)
                
                # Penalty for extreme thresholds
                threshold_penalty = (abs(overbought) + abs(oversold)) / 100
                
                return -(sharpe - threshold_penalty)
            
            # Use differential evolution for global optimization
            bounds = [(10, 40), (-40, -10)]
            result = differential_evolution(objective, bounds, seed=42, maxiter=50)
            
            if result.success:
                optimized_overbought, optimized_oversold = result.x
                return {
                    'overbought': float(optimized_overbought),
                    'oversold': float(optimized_oversold)
                }
        except Exception:
            pass
        
        # Fallback to adaptive percentile-based thresholds
        recent_tsi = tsi.tail(60).dropna()
        if len(recent_tsi) > 10:
            return {
                'overbought': float(np.percentile(recent_tsi, 80)),
                'oversold': float(np.percentile(recent_tsi, 20))
            }
        
        return {
            'overbought': self.parameters['overbought'],
            'oversold': self.parameters['oversold']
        }
    
    def _detect_divergences(self, tsi: pd.Series, data: pd.DataFrame) -> Dict[str, Any]:
        """Advanced divergence detection with statistical validation"""
        lookback = self.parameters['divergence_lookback']
        if len(tsi) < lookback or len(data) < lookback:
            return {'bullish_divergence': False, 'bearish_divergence': False, 'strength': 0.0}
        
        recent_tsi = tsi.tail(lookback)
        recent_prices = data['close'].tail(lookback)
        
        # Find significant peaks and troughs using statistical approach
        tsi_peaks, tsi_troughs = self._find_statistical_extremes(recent_tsi)
        price_peaks, price_troughs = self._find_statistical_extremes(recent_prices)
        
        bullish_divergence = False
        bearish_divergence = False
        divergence_strength = 0.0
        
        # Check for bullish divergence (price lower lows, TSI higher lows)
        if len(tsi_troughs) >= 2 and len(price_troughs) >= 2:
            last_tsi_trough = tsi_troughs[-1]
            prev_tsi_trough = tsi_troughs[-2]
            last_price_trough = price_troughs[-1]
            prev_price_trough = price_troughs[-2]
            
            price_decline = (last_price_trough[1] - prev_price_trough[1]) / prev_price_trough[1]
            tsi_improvement = (last_tsi_trough[1] - prev_tsi_trough[1]) / abs(prev_tsi_trough[1])
            
            if price_decline < -0.01 and tsi_improvement > 0.05:  # Thresholds for significance
                bullish_divergence = True
                divergence_strength = abs(price_decline) + abs(tsi_improvement)
        
        # Check for bearish divergence (price higher highs, TSI lower highs)
        if len(tsi_peaks) >= 2 and len(price_peaks) >= 2:
            last_tsi_peak = tsi_peaks[-1]
            prev_tsi_peak = tsi_peaks[-2]
            last_price_peak = price_peaks[-1]
            prev_price_peak = price_peaks[-2]
            
            price_increase = (last_price_peak[1] - prev_price_peak[1]) / prev_price_peak[1]
            tsi_decline = (last_tsi_peak[1] - prev_tsi_peak[1]) / abs(prev_tsi_peak[1])
            
            if price_increase > 0.01 and tsi_decline < -0.05:
                bearish_divergence = True
                divergence_strength = max(divergence_strength, abs(price_increase) + abs(tsi_decline))
        
        return {
            'bullish_divergence': bullish_divergence,
            'bearish_divergence': bearish_divergence,
            'strength': float(divergence_strength),
            'tsi_peaks': len(tsi_peaks),
            'tsi_troughs': len(tsi_troughs),
            'price_peaks': len(price_peaks),
            'price_troughs': len(price_troughs)
        }
    
    def _find_statistical_extremes(self, series: pd.Series) -> Tuple[List[Tuple], List[Tuple]]:
        """Find statistical peaks and troughs using local extrema detection"""
        if len(series) < 5:
            return [], []
        
        values = series.values
        peaks = []
        troughs = []
        
        # Use statistical approach: local extrema with significance test
        window = 3
        threshold = series.std() * 0.5  # Significance threshold
        
        for i in range(window, len(values) - window):
            window_values = values[i-window:i+window+1]
            current_value = values[i]
            
            # Peak detection
            if current_value == max(window_values) and current_value > series.mean() + threshold:
                peaks.append((i, current_value))
            
            # Trough detection
            elif current_value == min(window_values) and current_value < series.mean() - threshold:
                troughs.append((i, current_value))
        
        return peaks, troughs
    
    def _analyze_multi_timeframe_momentum(self, tsi: pd.Series, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze momentum across multiple timeframes"""
        if not self.parameters['multi_timeframe'] or len(tsi) < 60:
            return {'convergence': False, 'short_term': 'neutral', 'medium_term': 'neutral', 'long_term': 'neutral'}
        
        # Short term (5-10 periods)
        short_tsi = tsi.tail(10)
        short_trend = np.polyfit(range(len(short_tsi)), short_tsi.values, 1)[0]
        
        # Medium term (15-25 periods)
        medium_tsi = tsi.tail(25)
        medium_trend = np.polyfit(range(len(medium_tsi)), medium_tsi.values, 1)[0]
        
        # Long term (40-60 periods)
        long_tsi = tsi.tail(60)
        long_trend = np.polyfit(range(len(long_tsi)), long_tsi.values, 1)[0]
        
        # Classify trends
        def classify_trend(slope):
            if slope > 0.5:
                return 'bullish'
            elif slope < -0.5:
                return 'bearish'
            else:
                return 'neutral'
        
        short_classification = classify_trend(short_trend)
        medium_classification = classify_trend(medium_trend)
        long_classification = classify_trend(long_trend)
        
        # Check for convergence (all timeframes agree)
        trends = [short_classification, medium_classification, long_classification]
        convergence = len(set(trends)) == 1 and trends[0] != 'neutral'
        
        return {
            'convergence': convergence,
            'short_term': short_classification,
            'medium_term': medium_classification,
            'long_term': long_classification,
            'trends': {
                'short': float(short_trend),
                'medium': float(medium_trend),
                'long': float(long_trend)
            }
        }
    
    def _train_ml_models(self, tsi: pd.Series, data: pd.DataFrame) -> bool:
        """Train ML models for momentum prediction and regime classification"""
        if len(tsi) < self.parameters['ml_lookback'] * 2:
            return False
        
        try:
            features, targets = self._prepare_ml_data(tsi, data)
            if len(features) > 50:
                # Scale features
                scaled_features = self.robust_scaler.fit_transform(features)
                
                # Apply PCA for dimensionality reduction
                pca_features = self.pca.fit_transform(scaled_features)
                
                # Train momentum classifier
                self.momentum_classifier.fit(pca_features, targets)
                
                # Train threshold optimizer
                future_volatility = data['close'].pct_change().rolling(5).std().shift(-5).tail(len(features)).dropna().values
                if len(future_volatility) == len(pca_features):
                    self.threshold_optimizer.fit(pca_features, future_volatility)
                
                self.models_trained = True
                return True
        except Exception:
            pass
        return False
    
    def _prepare_ml_data(self, tsi: pd.Series, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare comprehensive ML training data"""
        features, targets = [], []
        lookback = 15
        
        for i in range(lookback, len(tsi) - 10):
            tsi_window = tsi.iloc[i-lookback:i].values
            price_window = data['close'].iloc[i-lookback:i].values
            
            # TSI features
            tsi_mean = np.mean(tsi_window)
            tsi_std = np.std(tsi_window)
            tsi_trend = np.polyfit(range(len(tsi_window)), tsi_window, 1)[0]
            tsi_current = tsi_window[-1]
            tsi_momentum = np.mean(np.diff(tsi_window[-5:]))
            tsi_acceleration = np.mean(np.diff(np.diff(tsi_window[-7:])))
            
            # Statistical features
            tsi_skew = stats.skew(tsi_window)
            tsi_kurtosis = stats.kurtosis(tsi_window)
            
            # Price features
            price_returns = np.diff(price_window) / price_window[:-1]
            price_volatility = np.std(price_returns)
            price_trend = np.polyfit(range(len(price_window)), price_window, 1)[0]
            
            # Threshold features
            overbought_ratio = len([x for x in tsi_window if x > 20]) / len(tsi_window)
            oversold_ratio = len([x for x in tsi_window if x < -20]) / len(tsi_window)
            
            # Zero line features
            zero_crossings = sum(1 for j in range(1, len(tsi_window)) 
                                if (tsi_window[j] > 0) != (tsi_window[j-1] > 0))
            
            # Volume features (if available)
            if 'volume' in data.columns:
                volume_window = data['volume'].iloc[i-lookback:i].values
                volume_trend = np.polyfit(range(len(volume_window)), volume_window, 1)[0]
                volume_ratio = volume_window[-1] / np.mean(volume_window)
            else:
                volume_trend = 0
                volume_ratio = 1
            
            feature_vector = [
                tsi_mean, tsi_std, tsi_trend, tsi_current, tsi_momentum, tsi_acceleration,
                tsi_skew, tsi_kurtosis,
                price_volatility, price_trend,
                overbought_ratio, oversold_ratio, zero_crossings,
                volume_trend, volume_ratio
            ]
            
            # Target: future momentum direction
            future_tsi = tsi.iloc[i+5:i+10].values
            if len(future_tsi) > 0:
                future_momentum = np.mean(np.diff(future_tsi))
                if future_momentum > 0.5:
                    target = 2  # Strong bullish
                elif future_momentum > 0:
                    target = 1  # Bullish
                elif future_momentum < -0.5:
                    target = 0  # Strong bearish
                else:
                    target = 1  # Neutral (default to slight bullish)
            else:
                target = 1
            
            features.append(feature_vector)
            targets.append(target)
        
        return np.array(features), np.array(targets)
    
    def calculate_raw(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate TSI with comprehensive analysis"""
        try:
            # Calculate TSI and components
            tsi, double_smooth_momentum, double_smooth_abs_momentum = self._calculate_tsi(data)
            
            # Calculate signal line if enabled
            signal_line = None
            if self.parameters['use_signal_line']:
                signal_line = self._calculate_signal_line(tsi)
            
            # Advanced analysis
            optimized_thresholds = self._optimize_thresholds(tsi, data)
            divergence_analysis = self._detect_divergences(tsi, data)
            momentum_analysis = self._analyze_multi_timeframe_momentum(tsi, data)
            
            # Train ML models
            if not self.models_trained:
                self._train_ml_models(tsi, data)
            
            # Generate signal
            signal, confidence = self._generate_tsi_signal(
                tsi, signal_line, optimized_thresholds, divergence_analysis,
                momentum_analysis, data
            )
            
            # Update history
            if len(tsi) > 0 and not pd.isna(tsi.iloc[-1]):
                self.history['tsi_values'].append(float(tsi.iloc[-1]))
                
                if signal_line is not None and not pd.isna(signal_line.iloc[-1]):
                    self.history['signal_values'].append(float(signal_line.iloc[-1]))
                
                if not pd.isna(double_smooth_momentum.iloc[-1]):
                    self.history['momentum_values'].append(float(double_smooth_momentum.iloc[-1]))
                
                self.history['threshold_values'].append(optimized_thresholds)
                self.history['regime_states'].append(momentum_analysis)
                
                # Keep history limited
                for key in self.history:
                    if len(self.history[key]) > 100:
                        self.history[key] = self.history[key][-100:]
            
            result = {
                'tsi': float(tsi.iloc[-1]) if len(tsi) > 0 and not pd.isna(tsi.iloc[-1]) else 0.0,
                'signal_line': float(signal_line.iloc[-1]) if signal_line is not None and len(signal_line) > 0 and not pd.isna(signal_line.iloc[-1]) else None,
                'signal': signal,
                'confidence': confidence,
                'optimized_thresholds': optimized_thresholds,
                'divergence_analysis': divergence_analysis,
                'momentum_analysis': momentum_analysis,
                'double_smooth_components': {
                    'momentum': float(double_smooth_momentum.iloc[-1]) if not pd.isna(double_smooth_momentum.iloc[-1]) else 0.0,
                    'abs_momentum': float(double_smooth_abs_momentum.iloc[-1]) if not pd.isna(double_smooth_abs_momentum.iloc[-1]) else 0.0
                },
                'adaptive_parameters': self._adapt_smoothing_parameters(data),
                'market_regime': self._classify_market_regime(tsi, data),
                'values_history': {
                    'tsi': tsi.tail(30).tolist(),
                    'signal_line': signal_line.tail(30).tolist() if signal_line is not None else []
                }
            }
            
            return result
            
        except Exception as e:
            raise IndicatorCalculationException(
                indicator_name=self.name,
                calculation_step="raw_calculation",
                message=f"Failed to calculate True Strength Index: {str(e)}",
                cause=e
            )
    
    def _generate_tsi_signal(self, tsi: pd.Series, signal_line: Optional[pd.Series],
                            optimized_thresholds: Dict, divergence_analysis: Dict,
                            momentum_analysis: Dict, data: pd.DataFrame) -> Tuple[SignalType, float]:
        """Generate comprehensive TSI signal"""
        signal_components = []
        confidence_components = []
        
        current_tsi = tsi.iloc[-1] if len(tsi) > 0 and not pd.isna(tsi.iloc[-1]) else 0
        overbought = optimized_thresholds['overbought']
        oversold = optimized_thresholds['oversold']
        
        # Threshold-based signals
        if current_tsi > overbought:
            signal_components.append(-0.8)
            confidence_components.append(0.7)
        elif current_tsi < oversold:
            signal_components.append(0.8)
            confidence_components.append(0.7)
        
        # Zero-line crossover signals
        if len(tsi) > 1:
            prev_tsi = tsi.iloc[-2]
            if prev_tsi <= 0 and current_tsi > 0:
                signal_components.append(0.9)
                confidence_components.append(0.8)
            elif prev_tsi >= 0 and current_tsi < 0:
                signal_components.append(-0.9)
                confidence_components.append(0.8)
        
        # Signal line crossover
        if signal_line is not None and len(signal_line) > 1:
            current_signal = signal_line.iloc[-1]
            prev_tsi = tsi.iloc[-2] if len(tsi) > 1 else current_tsi
            prev_signal = signal_line.iloc[-2] if len(signal_line) > 1 else current_signal
            
            if prev_tsi <= prev_signal and current_tsi > current_signal:
                signal_components.append(0.7)
                confidence_components.append(0.7)
            elif prev_tsi >= prev_signal and current_tsi < current_signal:
                signal_components.append(-0.7)
                confidence_components.append(0.7)
        
        # Divergence signals
        if divergence_analysis['bullish_divergence']:
            signal_components.append(0.8 * divergence_analysis['strength'])
            confidence_components.append(0.8)
        elif divergence_analysis['bearish_divergence']:
            signal_components.append(-0.8 * divergence_analysis['strength'])
            confidence_components.append(0.8)
        
        # Multi-timeframe momentum signals
        if momentum_analysis['convergence']:
            if momentum_analysis['short_term'] == 'bullish':
                signal_components.append(0.6)
                confidence_components.append(0.7)
            elif momentum_analysis['short_term'] == 'bearish':
                signal_components.append(-0.6)
                confidence_components.append(0.7)
        
        # ML enhancement
        if self.models_trained:
            try:
                ml_signal, ml_confidence = self._get_ml_signal(tsi, data)
                if ml_signal and ml_confidence > 0.6:
                    signal_value = 1.0 if ml_signal in [SignalType.BUY, SignalType.STRONG_BUY] else -1.0
                    signal_components.append(signal_value)
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
        
        if weighted_signal > 0.6:
            signal = SignalType.STRONG_BUY if weighted_signal > 0.8 else SignalType.BUY
        elif weighted_signal < -0.6:
            signal = SignalType.STRONG_SELL if weighted_signal < -0.8 else SignalType.SELL
        else:
            signal = SignalType.NEUTRAL
        
        return signal, min(avg_confidence, 1.0)
    
    def _get_ml_signal(self, tsi: pd.Series, data: pd.DataFrame) -> Tuple[Optional[SignalType], float]:
        """Get ML-based momentum prediction"""
        try:
            lookback = 15
            if len(tsi) < lookback:
                return None, 0.0
            
            tsi_window = tsi.tail(lookback).values
            price_window = data['close'].tail(lookback).values
            
            # Recreate feature vector (simplified version)
            tsi_mean = np.mean(tsi_window)
            tsi_std = np.std(tsi_window)
            tsi_trend = np.polyfit(range(len(tsi_window)), tsi_window, 1)[0]
            tsi_current = tsi_window[-1]
            tsi_momentum = np.mean(np.diff(tsi_window[-5:]))
            tsi_acceleration = np.mean(np.diff(np.diff(tsi_window[-7:])))
            
            tsi_skew = stats.skew(tsi_window)
            tsi_kurtosis = stats.kurtosis(tsi_window)
            
            price_returns = np.diff(price_window) / price_window[:-1]
            price_volatility = np.std(price_returns)
            price_trend = np.polyfit(range(len(price_window)), price_window, 1)[0]
            
            overbought_ratio = len([x for x in tsi_window if x > 20]) / len(tsi_window)
            oversold_ratio = len([x for x in tsi_window if x < -20]) / len(tsi_window)
            zero_crossings = sum(1 for j in range(1, len(tsi_window)) 
                                if (tsi_window[j] > 0) != (tsi_window[j-1] > 0))
            
            # Volume features
            if 'volume' in data.columns:
                volume_window = data['volume'].tail(lookback).values
                volume_trend = np.polyfit(range(len(volume_window)), volume_window, 1)[0]
                volume_ratio = volume_window[-1] / np.mean(volume_window)
            else:
                volume_trend = 0
                volume_ratio = 1
            
            feature_vector = np.array([[
                tsi_mean, tsi_std, tsi_trend, tsi_current, tsi_momentum, tsi_acceleration,
                tsi_skew, tsi_kurtosis,
                price_volatility, price_trend,
                overbought_ratio, oversold_ratio, zero_crossings,
                volume_trend, volume_ratio
            ]])
            
            scaled_features = self.robust_scaler.transform(feature_vector)
            pca_features = self.pca.transform(scaled_features)
            
            momentum_proba = self.momentum_classifier.predict_proba(pca_features)[0]
            
            if len(momentum_proba) >= 3:
                strong_bearish_prob = momentum_proba[0]
                neutral_prob = momentum_proba[1]
                strong_bullish_prob = momentum_proba[2]
                
                max_prob = max(momentum_proba)
                if max_prob > 0.7:
                    if strong_bullish_prob == max_prob:
                        return SignalType.STRONG_BUY, max_prob
                    elif strong_bearish_prob == max_prob:
                        return SignalType.STRONG_SELL, max_prob
                    elif neutral_prob == max_prob:
                        return SignalType.NEUTRAL, max_prob
        except:
            pass
        
        return None, 0.0
    
    def _classify_market_regime(self, tsi: pd.Series, data: pd.DataFrame) -> Dict[str, Any]:
        """Classify current market regime based on TSI characteristics"""
        if len(tsi) < 30:
            return {'regime': 'unknown', 'momentum': 'undefined', 'volatility': 'normal'}
        
        recent_tsi = tsi.tail(30)
        current_tsi = tsi.iloc[-1]
        
        # Momentum classification
        tsi_trend = np.polyfit(range(len(recent_tsi)), recent_tsi.values, 1)[0]
        
        if tsi_trend > 1:
            momentum = 'strong_bullish'
        elif tsi_trend > 0.3:
            momentum = 'bullish'
        elif tsi_trend < -1:
            momentum = 'strong_bearish'
        elif tsi_trend < -0.3:
            momentum = 'bearish'
        else:
            momentum = 'neutral'
        
        # Volatility classification
        tsi_volatility = recent_tsi.std()
        
        if tsi_volatility > 15:
            volatility = 'high'
        elif tsi_volatility > 8:
            volatility = 'medium'
        else:
            volatility = 'low'
        
        # Overall regime classification
        if abs(current_tsi) > 25 and momentum in ['strong_bullish', 'strong_bearish']:
            regime = 'extreme_momentum'
        elif abs(current_tsi) < 5 and volatility == 'low':
            regime = 'consolidation'
        elif volatility == 'high':
            regime = 'volatile'
        elif momentum in ['strong_bullish', 'strong_bearish']:
            regime = 'trending'
        else:
            regime = 'transitional'
        
        return {
            'regime': regime,
            'momentum': momentum,
            'volatility': volatility,
            'tsi_trend_slope': float(tsi_trend),
            'tsi_volatility': float(tsi_volatility),
            'current_level': float(current_tsi)
        }
    
    def _generate_signal(self, value: Any, data: pd.DataFrame) -> Tuple[Optional[SignalType], float]:
        if isinstance(value, dict) and 'signal' in value and 'confidence' in value:
            return value['signal'], value['confidence']
        return SignalType.NEUTRAL, 0.0
    
    def _get_metadata(self, data: pd.DataFrame) -> Dict[str, Any]:
        base_metadata = super()._get_metadata(data)
        base_metadata.update({
            'indicator_type': 'momentum',
            'indicator_category': 'true_strength_index',
            'models_trained': self.models_trained,
            'adaptive_smoothing': self.parameters['adaptive_smoothing'],
            'volume_weighted': self.parameters['volume_weighted'],
            'multi_timeframe': self.parameters['multi_timeframe'],
            'optimization_enabled': self.parameters['optimization_enabled'],
            'data_type': 'ohlcv',
            'complexity': 'advanced'
        })
        return base_metadata