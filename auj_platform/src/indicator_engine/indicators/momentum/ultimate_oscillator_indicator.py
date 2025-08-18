"""
Ultimate Oscillator Indicator - Advanced Implementation
======================================================

Advanced Ultimate Oscillator with ML-enhanced pattern recognition,
adaptive timeframes, and sophisticated momentum convergence analysis.

Author: AUJ Platform Development Team
Mission: Building advanced trading indicators for humanitarian impact
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

from ..base.standard_indicator import (
    StandardIndicatorInterface, 
    DataRequirement, 
    DataType, 
    SignalType
)
from ....core.exceptions import IndicatorCalculationException


class UltimateOscillatorIndicator(StandardIndicatorInterface):
    """
    Advanced Ultimate Oscillator Implementation
    
    Features:
    - Multi-timeframe true range and buying pressure optimization
    - ML-enhanced momentum convergence detection
    - Adaptive timeframe selection based on market volatility
    - Advanced divergence analysis with statistical validation
    - Institutional flow detection through volume-weighted calculations
    - Dynamic threshold optimization using historical percentiles
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'short_period': 7,
            'medium_period': 14,
            'long_period': 28,
            'short_weight': 4,
            'medium_weight': 2,
            'long_weight': 1,
            'overbought': 70,
            'oversold': 30,
            'adaptive_timeframes': True,
            'volume_weighted': True,
            'divergence_lookback': 20,
            'ml_lookback': 60,
            'optimization_enabled': True
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__(name="UltimateOscillatorIndicator", parameters=default_params)
        
        self.scaler = StandardScaler()
        self.convergence_classifier = AdaBoostClassifier(n_estimators=100, random_state=42)
        self.threshold_optimizer = GradientBoostingRegressor(n_estimators=50, random_state=42)
        self.pattern_clusterer = AgglomerativeClustering(n_clusters=5)
        self.models_trained = False
        
        self.history = {
            'uo_values': [],
            'true_ranges': [],
            'buying_pressures': [],
            'convergence_states': [],
            'optimized_thresholds': []
        }
    
    def get_data_requirements(self) -> DataRequirement:
        max_period = max(self.parameters['short_period'], 
                        self.parameters['medium_period'], 
                        self.parameters['long_period'])
        return DataRequirement(
            data_type=DataType.OHLCV,
            required_columns=['high', 'low', 'close', 'volume'],
            min_periods=max(max_period, self.parameters['ml_lookback']) + 30,
            lookback_periods=150
        )
    
    def _calculate_true_range(self, data: pd.DataFrame) -> pd.Series:
        """Calculate True Range with volume weighting option"""
        high = data['high']
        low = data['low']
        prev_close = data['close'].shift(1)
        
        # Traditional True Range calculation
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        if self.parameters['volume_weighted']:
            # Weight true range by volume to capture institutional activity
            volume_factor = data['volume'] / data['volume'].rolling(window=20).mean()
            volume_factor = volume_factor.fillna(1.0).clip(0.1, 5.0)  # Reasonable bounds
            true_range = true_range * volume_factor
        
        return true_range
    
    def _calculate_buying_pressure(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Buying Pressure with institutional flow detection"""
        close = data['close']
        low = data['low']
        prev_close = close.shift(1)
        
        # Traditional buying pressure
        buying_pressure = close - pd.concat([low, prev_close], axis=1).min(axis=1)
        
        if self.parameters['volume_weighted']:
            # Enhance with volume-based institutional flow detection
            volume = data['volume']
            price_change = close.pct_change()
            
            # Detect accumulation vs distribution patterns
            # High volume + small price change = institutional accumulation
            volume_ratio = volume / volume.rolling(window=10).mean()
            price_volatility = price_change.abs()
            
            # Institutional factor: high volume, low volatility suggests smart money
            institutional_factor = (volume_ratio / (price_volatility + 1e-6)).clip(0.5, 2.0)
            buying_pressure = buying_pressure * institutional_factor
        
        return buying_pressure
    
    def _optimize_timeframes(self, data: pd.DataFrame) -> Tuple[int, int, int]:
        """Dynamically optimize timeframes based on market conditions"""
        if not self.parameters['adaptive_timeframes']:
            return (self.parameters['short_period'], 
                   self.parameters['medium_period'], 
                   self.parameters['long_period'])
        
        # Calculate market volatility regime
        returns = data['close'].pct_change().tail(60)
        volatility = returns.std()
        avg_volatility = returns.rolling(window=30).std().mean()
        vol_ratio = volatility / (avg_volatility + 1e-8)
        
        # Calculate trend persistence
        price_changes = data['close'].diff().tail(60)
        trend_consistency = len([x for x in price_changes if x * price_changes.iloc[-1] > 0]) / len(price_changes)
        
        # Adjust timeframes based on market regime
        base_short = self.parameters['short_period']
        base_medium = self.parameters['medium_period']
        base_long = self.parameters['long_period']
        
        if vol_ratio > 1.5:  # High volatility - use shorter periods
            short_period = max(3, int(base_short * 0.7))
            medium_period = max(7, int(base_medium * 0.8))
            long_period = max(14, int(base_long * 0.9))
        elif vol_ratio < 0.7:  # Low volatility - use longer periods
            short_period = min(14, int(base_short * 1.3))
            medium_period = min(21, int(base_medium * 1.2))
            long_period = min(42, int(base_long * 1.1))
        else:  # Normal volatility
            short_period = base_short
            medium_period = base_medium
            long_period = base_long
        
        # Adjust for trend persistence
        if trend_consistency > 0.7:  # Strong trend - use longer periods
            short_period = min(short_period + 2, 12)
            medium_period = min(medium_period + 3, 21)
            long_period = min(long_period + 5, 35)
        elif trend_consistency < 0.4:  # Choppy market - use shorter periods
            short_period = max(short_period - 1, 5)
            medium_period = max(medium_period - 2, 10)
            long_period = max(long_period - 3, 20)
        
        return short_period, medium_period, long_period
    
    def _calculate_ultimate_oscillator(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Ultimate Oscillator with optimized parameters"""
        # Get optimized timeframes
        short_period, medium_period, long_period = self._optimize_timeframes(data)
        
        # Calculate components
        true_range = self._calculate_true_range(data)
        buying_pressure = self._calculate_buying_pressure(data)
        
        # Calculate averages for each timeframe
        short_bp_sum = buying_pressure.rolling(window=short_period).sum()
        medium_bp_sum = buying_pressure.rolling(window=medium_period).sum()
        long_bp_sum = buying_pressure.rolling(window=long_period).sum()
        
        short_tr_sum = true_range.rolling(window=short_period).sum()
        medium_tr_sum = true_range.rolling(window=medium_period).sum()
        long_tr_sum = true_range.rolling(window=long_period).sum()
        
        # Calculate raw oscillator components
        short_raw = short_bp_sum / (short_tr_sum + 1e-8)
        medium_raw = medium_bp_sum / (medium_tr_sum + 1e-8)
        long_raw = long_bp_sum / (long_tr_sum + 1e-8)
        
        # Apply weights
        short_weight = self.parameters['short_weight']
        medium_weight = self.parameters['medium_weight']
        long_weight = self.parameters['long_weight']
        
        # Calculate weighted Ultimate Oscillator
        numerator = (short_weight * short_raw + 
                    medium_weight * medium_raw + 
                    long_weight * long_raw)
        denominator = short_weight + medium_weight + long_weight
        
        ultimate_oscillator = 100 * (numerator / denominator)
        
        return ultimate_oscillator
    
    def _detect_convergence_divergence(self, uo: pd.Series, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect multi-timeframe convergence and divergence patterns"""
        lookback = self.parameters['divergence_lookback']
        if len(uo) < lookback:
            return {'convergence': False, 'divergence': False, 'strength': 0.0, 'type': 'none'}
        
        recent_uo = uo.tail(lookback)
        recent_prices = data['close'].tail(lookback)
        
        # Calculate short, medium, long term UO trends
        short_uo = recent_uo.tail(7)
        medium_uo = recent_uo.tail(14)
        long_uo = recent_uo
        
        # Trend analysis
        short_trend = np.polyfit(range(len(short_uo)), short_uo.values, 1)[0]
        medium_trend = np.polyfit(range(len(medium_uo)), medium_uo.values, 1)[0]
        long_trend = np.polyfit(range(len(long_uo)), long_uo.values, 1)[0]
        price_trend = np.polyfit(range(len(recent_prices)), recent_prices.values, 1)[0]
        
        # Convergence detection: all UO timeframes agree
        uo_trends = [short_trend, medium_trend, long_trend]
        trend_signs = [np.sign(t) for t in uo_trends if t != 0]
        
        convergence = False
        if len(set(trend_signs)) == 1 and len(trend_signs) == 3:
            convergence = True
            convergence_strength = np.mean([abs(t) for t in uo_trends])
        else:
            convergence_strength = 0.0
        
        # Divergence detection: UO vs Price
        divergence = False
        divergence_type = 'none'
        
        if np.sign(price_trend) != np.sign(long_trend) and abs(price_trend) > 0.01 and abs(long_trend) > 0.1:
            divergence = True
            if price_trend > 0 and long_trend < 0:
                divergence_type = 'bearish'  # Price up, UO down
            elif price_trend < 0 and long_trend > 0:
                divergence_type = 'bullish'  # Price down, UO up
        
        # Calculate overall strength
        overall_strength = max(convergence_strength, abs(long_trend)) if convergence or divergence else 0.0
        
        return {
            'convergence': convergence,
            'divergence': divergence,
            'strength': float(overall_strength),
            'type': divergence_type,
            'trends': {
                'short': float(short_trend),
                'medium': float(medium_trend),
                'long': float(long_trend),
                'price': float(price_trend)
            }
        }
    
    def _optimize_thresholds(self, uo: pd.Series, data: pd.DataFrame) -> Dict[str, float]:
        """Dynamically optimize overbought/oversold thresholds"""
        if not self.parameters['optimization_enabled'] or len(uo) < 50:
            return {
                'overbought': self.parameters['overbought'],
                'oversold': self.parameters['oversold']
            }
        
        try:
            # Use optimization to find best thresholds
            recent_uo = uo.tail(100).dropna()
            recent_returns = data['close'].pct_change().tail(len(recent_uo)).dropna()
            
            if len(recent_uo) != len(recent_returns):
                min_len = min(len(recent_uo), len(recent_returns))
                recent_uo = recent_uo.tail(min_len)
                recent_returns = recent_returns.tail(min_len)
            
            def objective(thresholds):
                overbought, oversold = thresholds
                if overbought <= oversold or overbought > 95 or oversold < 5:
                    return 1e6  # Invalid thresholds
                
                # Calculate signals
                signals = []
                for i in range(1, len(recent_uo)):
                    if recent_uo.iloc[i-1] > overbought and recent_uo.iloc[i] <= overbought:
                        signals.append(-1)  # Sell signal
                    elif recent_uo.iloc[i-1] < oversold and recent_uo.iloc[i] >= oversold:
                        signals.append(1)   # Buy signal
                    else:
                        signals.append(0)   # No signal
                
                if not signals:
                    return 1e6
                
                # Calculate performance
                returns_aligned = recent_returns.iloc[1:len(signals)+1].values
                signals = np.array(signals[:len(returns_aligned)])
                
                # Strategy returns
                strategy_returns = signals * returns_aligned
                
                # Risk-adjusted return (Sharpe-like)
                if len(strategy_returns) == 0 or np.std(strategy_returns) == 0:
                    return 1e6
                
                sharpe = np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-8)
                return -sharpe  # Minimize negative Sharpe
            
            # Optimize thresholds
            result = minimize(
                objective, 
                x0=[self.parameters['overbought'], self.parameters['oversold']],
                bounds=[(60, 90), (10, 40)],
                method='L-BFGS-B'
            )
            
            if result.success:
                optimized_overbought, optimized_oversold = result.x
                return {
                    'overbought': float(optimized_overbought),
                    'oversold': float(optimized_oversold)
                }
        except:
            pass
        
        # Fallback to percentile-based thresholds
        recent_uo = uo.tail(60).dropna()
        return {
            'overbought': float(np.percentile(recent_uo, 85)),
            'oversold': float(np.percentile(recent_uo, 15))
        }
    
    def _analyze_momentum_phases(self, uo: pd.Series) -> Dict[str, Any]:
        """Analyze Ultimate Oscillator momentum phases"""
        if len(uo) < 20:
            return {'phase': 'unknown', 'strength': 0.0, 'duration': 0, 'acceleration': 0.0}
        
        recent_uo = uo.tail(20)
        current_uo = uo.iloc[-1]
        
        # Calculate momentum and acceleration
        uo_velocity = recent_uo.diff().tail(5).mean()
        uo_acceleration = recent_uo.diff().diff().tail(3).mean()
        
        # Determine phase
        if current_uo > 70:
            if uo_velocity > 0:
                phase = 'overbought_strengthening'
            else:
                phase = 'overbought_weakening'
        elif current_uo < 30:
            if uo_velocity > 0:
                phase = 'oversold_recovering'
            else:
                phase = 'oversold_deepening'
        elif current_uo > 50:
            if uo_velocity > 0:
                phase = 'bullish_momentum'
            else:
                phase = 'bullish_fading'
        else:
            if uo_velocity > 0:
                phase = 'bearish_recovering'
            else:
                phase = 'bearish_momentum'
        
        # Calculate phase duration
        duration = 0
        for i in range(1, min(15, len(recent_uo))):
            past_uo = recent_uo.iloc[-i-1]
            if (current_uo > 70) == (past_uo > 70) and (current_uo < 30) == (past_uo < 30):
                duration += 1
            else:
                break
        
        return {
            'phase': phase,
            'strength': float(abs(uo_velocity)),
            'duration': duration,
            'acceleration': float(uo_acceleration),
            'velocity': float(uo_velocity)
        }
    
    def _train_ml_models(self, uo: pd.Series, data: pd.DataFrame) -> bool:
        """Train ML models for convergence detection and threshold optimization"""
        if len(uo) < self.parameters['ml_lookback'] * 2:
            return False
        
        try:
            features, targets = self._prepare_ml_data(uo, data)
            if len(features) > 40:
                self.scaler.fit(features)
                scaled_features = self.scaler.transform(features)
                
                # Train convergence classifier
                self.convergence_classifier.fit(scaled_features, targets)
                
                # Train threshold optimizer
                future_volatility = data['close'].pct_change().rolling(5).std().shift(-5).tail(len(features)).dropna().values
                if len(future_volatility) == len(scaled_features):
                    self.threshold_optimizer.fit(scaled_features, future_volatility)
                
                self.models_trained = True
                return True
        except:
            pass
        return False
    
    def _prepare_ml_data(self, uo: pd.Series, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare ML training data"""
        features, targets = [], []
        lookback = 15
        
        for i in range(lookback, len(uo) - 10):
            uo_window = uo.iloc[i-lookback:i].values
            price_window = data['close'].iloc[i-lookback:i].values
            volume_window = data['volume'].iloc[i-lookback:i].values
            
            # UO features
            uo_mean = np.mean(uo_window)
            uo_std = np.std(uo_window)
            uo_trend = np.polyfit(range(len(uo_window)), uo_window, 1)[0]
            uo_current = uo_window[-1]
            
            # Price features
            price_volatility = np.std(np.diff(price_window) / price_window[:-1])
            price_trend = np.polyfit(range(len(price_window)), price_window, 1)[0]
            
            # Volume features
            volume_trend = np.polyfit(range(len(volume_window)), volume_window, 1)[0]
            volume_ratio = volume_window[-1] / np.mean(volume_window)
            
            feature_vector = [
                uo_mean, uo_std, uo_trend, uo_current,
                price_volatility, price_trend,
                volume_trend, volume_ratio,
                len([x for x in uo_window if x > 70]) / len(uo_window),  # Overbought ratio
                len([x for x in uo_window if x < 30]) / len(uo_window),  # Oversold ratio
                np.corrcoef(uo_window, price_window)[0, 1] if len(set(uo_window)) > 1 else 0
            ]
            
            # Target: convergence in next 5-10 periods
            future_uo = uo.iloc[i+5:i+10].values
            if len(future_uo) > 0:
                future_trend = np.polyfit(range(len(future_uo)), future_uo, 1)[0]
                target = 1 if abs(future_trend) > 0.5 else 0  # Convergence/strong trend
            else:
                target = 0
            
            features.append(feature_vector)
            targets.append(target)
        
        return np.array(features), np.array(targets)
    
    def calculate_raw(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate Ultimate Oscillator with advanced analysis"""
        try:
            # Calculate Ultimate Oscillator
            uo = self._calculate_ultimate_oscillator(data)
            
            # Advanced analysis
            convergence_analysis = self._detect_convergence_divergence(uo, data)
            optimized_thresholds = self._optimize_thresholds(uo, data)
            momentum_phases = self._analyze_momentum_phases(uo)
            
            # Train ML models
            if not self.models_trained:
                self._train_ml_models(uo, data)
            
            # Generate signal
            signal, confidence = self._generate_uo_signal(
                uo, convergence_analysis, optimized_thresholds, 
                momentum_phases, data
            )
            
            # Update history
            if len(uo) > 0 and not pd.isna(uo.iloc[-1]):
                self.history['uo_values'].append(float(uo.iloc[-1]))
                self.history['convergence_states'].append(convergence_analysis)
                self.history['optimized_thresholds'].append(optimized_thresholds)
                
                # Keep history limited
                for key in self.history:
                    if len(self.history[key]) > 100:
                        self.history[key] = self.history[key][-100:]
            
            result = {
                'ultimate_oscillator': float(uo.iloc[-1]) if len(uo) > 0 and not pd.isna(uo.iloc[-1]) else 0.0,
                'signal': signal,
                'confidence': confidence,
                'convergence_analysis': convergence_analysis,
                'optimized_thresholds': optimized_thresholds,
                'momentum_phases': momentum_phases,
                'timeframes': self._optimize_timeframes(data),
                'market_regime': self._classify_market_regime(uo, data),
                'values_history': {
                    'ultimate_oscillator': uo.tail(30).tolist()
                }
            }
            
            return result
            
        except Exception as e:
            raise IndicatorCalculationException(
                indicator_name=self.name,
                calculation_step="raw_calculation",
                message=f"Failed to calculate Ultimate Oscillator: {str(e)}",
                cause=e
            )
    
    def _generate_uo_signal(self, uo: pd.Series, convergence_analysis: Dict,
                           optimized_thresholds: Dict, momentum_phases: Dict,
                           data: pd.DataFrame) -> Tuple[SignalType, float]:
        """Generate comprehensive Ultimate Oscillator signal"""
        signal_components = []
        confidence_components = []
        
        current_uo = uo.iloc[-1] if len(uo) > 0 and not pd.isna(uo.iloc[-1]) else 50
        overbought = optimized_thresholds['overbought']
        oversold = optimized_thresholds['oversold']
        
        # Threshold-based signals
        if current_uo > overbought:
            signal_components.append(-0.8)
            confidence_components.append(0.7)
        elif current_uo < oversold:
            signal_components.append(0.8)
            confidence_components.append(0.7)
        
        # Convergence/Divergence signals
        if convergence_analysis['convergence']:
            trend_direction = np.sign(convergence_analysis['trends']['long'])
            signal_components.append(trend_direction * convergence_analysis['strength'])
            confidence_components.append(0.8)
        
        if convergence_analysis['divergence']:
            if convergence_analysis['type'] == 'bullish':
                signal_components.append(0.6)
                confidence_components.append(0.7)
            elif convergence_analysis['type'] == 'bearish':
                signal_components.append(-0.6)
                confidence_components.append(0.7)
        
        # Momentum phase signals
        phase = momentum_phases['phase']
        if phase in ['oversold_recovering', 'bearish_recovering']:
            signal_components.append(momentum_phases['strength'])
            confidence_components.append(0.6)
        elif phase in ['overbought_weakening', 'bullish_fading']:
            signal_components.append(-momentum_phases['strength'])
            confidence_components.append(0.6)
        
        # ML enhancement
        if self.models_trained:
            try:
                ml_signal, ml_confidence = self._get_ml_signal(uo, data)
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
        
        if weighted_signal > 0.6:
            signal = SignalType.STRONG_BUY if weighted_signal > 0.8 else SignalType.BUY
        elif weighted_signal < -0.6:
            signal = SignalType.STRONG_SELL if weighted_signal < -0.8 else SignalType.SELL
        else:
            signal = SignalType.NEUTRAL
        
        return signal, min(avg_confidence, 1.0)
    
    def _get_ml_signal(self, uo: pd.Series, data: pd.DataFrame) -> Tuple[Optional[SignalType], float]:
        """Get ML-based signal prediction"""
        try:
            lookback = 15
            if len(uo) < lookback:
                return None, 0.0
            
            uo_window = uo.tail(lookback).values
            price_window = data['close'].tail(lookback).values
            volume_window = data['volume'].tail(lookback).values
            
            # Recreate feature vector
            uo_mean = np.mean(uo_window)
            uo_std = np.std(uo_window)
            uo_trend = np.polyfit(range(len(uo_window)), uo_window, 1)[0]
            uo_current = uo_window[-1]
            
            price_volatility = np.std(np.diff(price_window) / price_window[:-1])
            price_trend = np.polyfit(range(len(price_window)), price_window, 1)[0]
            
            volume_trend = np.polyfit(range(len(volume_window)), volume_window, 1)[0]
            volume_ratio = volume_window[-1] / np.mean(volume_window)
            
            feature_vector = np.array([[
                uo_mean, uo_std, uo_trend, uo_current,
                price_volatility, price_trend,
                volume_trend, volume_ratio,
                len([x for x in uo_window if x > 70]) / len(uo_window),
                len([x for x in uo_window if x < 30]) / len(uo_window),
                np.corrcoef(uo_window, price_window)[0, 1] if len(set(uo_window)) > 1 else 0
            ]])
            
            scaled_features = self.scaler.transform(feature_vector)
            ml_proba = self.convergence_classifier.predict_proba(scaled_features)[0]
            
            if len(ml_proba) >= 2:
                convergence_prob = ml_proba[1]  # Probability of convergence/strong trend
                if convergence_prob > 0.7:
                    # Determine direction from current trend
                    if uo_trend > 0:
                        return SignalType.BUY, convergence_prob
                    elif uo_trend < 0:
                        return SignalType.SELL, convergence_prob
        except:
            pass
        
        return None, 0.0
    
    def _classify_market_regime(self, uo: pd.Series, data: pd.DataFrame) -> Dict[str, Any]:
        """Classify current market regime for UO interpretation"""
        if len(uo) < 30:
            return {'regime': 'unknown', 'volatility': 'normal', 'trend_strength': 'weak'}
        
        # Volatility regime
        returns = data['close'].pct_change().tail(30)
        current_vol = returns.std()
        avg_vol = returns.rolling(window=20).std().mean()
        vol_ratio = current_vol / (avg_vol + 1e-8)
        
        if vol_ratio > 1.5:
            volatility_regime = 'high'
        elif vol_ratio < 0.7:
            volatility_regime = 'low'
        else:
            volatility_regime = 'normal'
        
        # Trend strength from UO
        recent_uo = uo.tail(20)
        uo_trend = np.polyfit(range(len(recent_uo)), recent_uo.values, 1)[0]
        
        if abs(uo_trend) > 1.0:
            trend_strength = 'strong'
        elif abs(uo_trend) > 0.3:
            trend_strength = 'moderate'
        else:
            trend_strength = 'weak'
        
        # Overall regime
        if volatility_regime == 'high' and trend_strength == 'strong':
            regime = 'trending_volatile'
        elif volatility_regime == 'low' and trend_strength == 'weak':
            regime = 'range_bound'
        elif trend_strength == 'strong':
            regime = 'strong_trend'
        else:
            regime = 'transitional'
        
        return {
            'regime': regime,
            'volatility': volatility_regime,
            'trend_strength': trend_strength,
            'uo_trend': float(uo_trend)
        }
    
    def _generate_signal(self, value: Any, data: pd.DataFrame) -> Tuple[Optional[SignalType], float]:
        if isinstance(value, dict) and 'signal' in value and 'confidence' in value:
            return value['signal'], value['confidence']
        return SignalType.NEUTRAL, 0.0
    
    def _get_metadata(self, data: pd.DataFrame) -> Dict[str, Any]:
        base_metadata = super()._get_metadata(data)
        base_metadata.update({
            'indicator_type': 'momentum',
            'indicator_category': 'ultimate_oscillator',
            'models_trained': self.models_trained,
            'adaptive_timeframes': self.parameters['adaptive_timeframes'],
            'volume_weighted': self.parameters['volume_weighted'],
            'optimization_enabled': self.parameters['optimization_enabled'],
            'data_type': 'ohlcv',
            'complexity': 'advanced'
        })
        return base_metadata