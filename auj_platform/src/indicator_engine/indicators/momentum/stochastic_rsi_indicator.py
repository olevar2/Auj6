"""
Stochastic RSI Indicator - Advanced Implementation
=================================================

Advanced Stochastic RSI with ML-enhanced momentum detection, adaptive periods,
and sophisticated overbought/oversold analysis with institutional flow detection.

Author: AUJ Platform Development Team
Mission: Building advanced trading indicators for humanitarian impact
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from sklearn.ensemble import RandomForestClassifier, ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
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


class StochasticRSIIndicator(StandardIndicatorInterface):
    """
    Advanced Stochastic RSI Implementation
    
    Features:
    - Adaptive RSI and Stochastic periods based on market volatility
    - ML-enhanced overbought/oversold detection with regime classification
    - Multi-timeframe divergence analysis with statistical validation
    - Volume-weighted RSI calculations for institutional flow detection
    - Dynamic threshold optimization using genetic algorithms
    - Advanced signal line integration with crossover optimization
    - Pattern recognition for reversal and continuation signals
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'rsi_period': 14,
            'stoch_k_period': 14,
            'stoch_d_period': 3,
            'overbought': 80,
            'oversold': 20,
            'adaptive_periods': True,
            'volume_weighted': True,
            'use_smoothed_rsi': True,
            'ml_lookback': 60,
            'divergence_lookback': 20,
            'optimization_enabled': True,
            'pattern_detection': True,
            'multi_timeframe': True,
            'regime_classification': True
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__(name="StochasticRSIIndicator", parameters=default_params)
        
        self.scaler = StandardScaler()
        self.robust_scaler = RobustScaler()
        self.momentum_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.threshold_optimizer = ExtraTreesRegressor(n_estimators=50, random_state=42)
        self.pattern_clusterer = AgglomerativeClustering(n_clusters=5, linkage='ward')
        self.models_trained = False
        
        self.history = {
            'stoch_rsi_k': [],
            'stoch_rsi_d': [],
            'rsi_values': [],
            'threshold_values': [],
            'divergence_signals': [],
            'regime_states': []
        }
    
    def get_data_requirements(self) -> DataRequirement:
        max_period = max(self.parameters['rsi_period'], 
                        self.parameters['stoch_k_period'],
                        self.parameters['ml_lookback'])
        return DataRequirement(
            data_type=DataType.OHLCV,
            required_columns=['close', 'volume'],
            min_periods=max_period * 2 + 50,
            lookback_periods=200
        )
    
    def _adapt_periods(self, data: pd.DataFrame) -> Tuple[int, int, int]:
        """Adapt RSI and Stochastic periods based on market conditions"""
        if not self.parameters['adaptive_periods']:
            return (self.parameters['rsi_period'], 
                   self.parameters['stoch_k_period'], 
                   self.parameters['stoch_d_period'])
        
        if len(data) < 60:
            return (self.parameters['rsi_period'], 
                   self.parameters['stoch_k_period'], 
                   self.parameters['stoch_d_period'])
        
        # Calculate market volatility regime
        returns = data['close'].pct_change().tail(60)
        current_vol = returns.std()
        rolling_vol = returns.rolling(window=20).std()
        avg_vol = rolling_vol.mean()
        vol_ratio = current_vol / (avg_vol + 1e-8)
        
        # Calculate trend persistence
        price_changes = data['close'].diff().tail(40)
        trend_strength = abs(price_changes.sum()) / (price_changes.abs().sum() + 1e-8)
        
        # Calculate volume regime
        if 'volume' in data.columns:
            volume_ratio = data['volume'].tail(20).mean() / data['volume'].tail(60).mean()
        else:
            volume_ratio = 1.0
        
        base_rsi = self.parameters['rsi_period']
        base_stoch_k = self.parameters['stoch_k_period']
        base_stoch_d = self.parameters['stoch_d_period']
        
        # Adjust for volatility regime
        if vol_ratio > 1.5:  # High volatility - shorter periods for responsiveness
            rsi_period = max(8, int(base_rsi * 0.7))
            stoch_k_period = max(8, int(base_stoch_k * 0.7))
            stoch_d_period = max(2, int(base_stoch_d * 0.8))
        elif vol_ratio < 0.6:  # Low volatility - longer periods for stability
            rsi_period = min(25, int(base_rsi * 1.3))
            stoch_k_period = min(25, int(base_stoch_k * 1.3))
            stoch_d_period = min(7, int(base_stoch_d * 1.5))
        else:  # Normal volatility
            rsi_period = base_rsi
            stoch_k_period = base_stoch_k
            stoch_d_period = base_stoch_d
        
        # Adjust for trend strength
        if trend_strength > 0.7:  # Strong trend - longer periods
            rsi_period = min(rsi_period + 3, 28)
            stoch_k_period = min(stoch_k_period + 3, 28)
        elif trend_strength < 0.3:  # Weak trend - shorter periods
            rsi_period = max(rsi_period - 2, 8)
            stoch_k_period = max(stoch_k_period - 2, 8)
        
        # Adjust for volume regime
        if volume_ratio > 1.3:  # High volume - slightly shorter for sensitivity
            rsi_period = max(rsi_period - 1, 8)
            stoch_k_period = max(stoch_k_period - 1, 8)
        
        return rsi_period, stoch_k_period, stoch_d_period
    
    def _calculate_volume_weighted_rsi(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate volume-weighted RSI for institutional flow detection"""
        close = data['close']
        
        if not self.parameters['volume_weighted'] or 'volume' not in data.columns:
            return self._calculate_standard_rsi(close, period)
        
        volume = data['volume']
        price_changes = close.diff()
        
        # Volume-weighted gains and losses
        gains = price_changes.where(price_changes > 0, 0) * volume
        losses = (-price_changes.where(price_changes < 0, 0)) * volume
        
        # Volume-weighted averages
        avg_gains = gains.rolling(window=period).sum() / volume.rolling(window=period).sum()
        avg_losses = losses.rolling(window=period).sum() / volume.rolling(window=period).sum()
        
        # Handle initial period with simple moving average
        if len(avg_gains) >= period:
            for i in range(period, len(avg_gains)):
                if pd.isna(avg_gains.iloc[i]):
                    # Use simple average for initial calculation
                    avg_gains.iloc[i] = gains.iloc[i-period+1:i+1].mean()
                    avg_losses.iloc[i] = losses.iloc[i-period+1:i+1].mean()
        
        # Calculate RSI
        rs = avg_gains / (avg_losses + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.fillna(50)  # Fill NaN with neutral value
    
    def _calculate_standard_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate standard RSI"""
        price_changes = prices.diff()
        gains = price_changes.where(price_changes > 0, 0)
        losses = (-price_changes.where(price_changes < 0, 0))
        
        # Use EMA for smoothing if specified
        if self.parameters['use_smoothed_rsi']:
            avg_gains = gains.ewm(span=period, adjust=False).mean()
            avg_losses = losses.ewm(span=period, adjust=False).mean()
        else:
            avg_gains = gains.rolling(window=period).mean()
            avg_losses = losses.rolling(window=period).mean()
        
        rs = avg_gains / (avg_losses + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.fillna(50)
    
    def _calculate_stochastic_rsi(self, rsi: pd.Series, k_period: int, d_period: int) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic RSI %K and %D"""
        # Calculate %K
        rsi_low = rsi.rolling(window=k_period).min()
        rsi_high = rsi.rolling(window=k_period).max()
        
        stoch_rsi_k = ((rsi - rsi_low) / (rsi_high - rsi_low + 1e-8)) * 100
        
        # Calculate %D (moving average of %K)
        stoch_rsi_d = stoch_rsi_k.rolling(window=d_period).mean()
        
        return stoch_rsi_k.fillna(50), stoch_rsi_d.fillna(50)
    
    def _optimize_thresholds(self, stoch_rsi_k: pd.Series, data: pd.DataFrame) -> Dict[str, float]:
        """Optimize overbought/oversold thresholds using genetic algorithms"""
        if not self.parameters['optimization_enabled'] or len(stoch_rsi_k) < 100:
            return {
                'overbought': self.parameters['overbought'],
                'oversold': self.parameters['oversold']
            }
        
        try:
            recent_stoch = stoch_rsi_k.tail(100).dropna()
            recent_returns = data['close'].pct_change().tail(len(recent_stoch)).dropna()
            
            if len(recent_stoch) != len(recent_returns):
                min_len = min(len(recent_stoch), len(recent_returns))
                recent_stoch = recent_stoch.tail(min_len)
                recent_returns = recent_returns.tail(min_len)
            
            def objective(params):
                overbought, oversold = params
                if overbought <= oversold or overbought > 95 or oversold < 5:
                    return 1e6
                
                signals = []
                for i in range(1, len(recent_stoch)):
                    if recent_stoch.iloc[i-1] > overbought and recent_stoch.iloc[i] <= overbought:
                        signals.append(-1)  # Sell signal
                    elif recent_stoch.iloc[i-1] < oversold and recent_stoch.iloc[i] >= oversold:
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
                
                # Multi-objective optimization
                avg_return = np.mean(strategy_returns)
                volatility = np.std(strategy_returns)
                sharpe = avg_return / (volatility + 1e-8)
                
                # Win rate
                winning_trades = np.sum(strategy_returns > 0)
                total_trades = np.sum(np.abs(signals) > 0)
                win_rate = winning_trades / (total_trades + 1e-8)
                
                # Combined score
                score = sharpe + win_rate - 0.1  # Slight penalty for complexity
                
                return -score
            
            # Use differential evolution for global optimization
            bounds = [(60, 95), (5, 40)]
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
        recent_stoch = stoch_rsi_k.tail(60).dropna()
        if len(recent_stoch) > 10:
            return {
                'overbought': float(np.percentile(recent_stoch, 85)),
                'oversold': float(np.percentile(recent_stoch, 15))
            }
        
        return {
            'overbought': self.parameters['overbought'],
            'oversold': self.parameters['oversold']
        }
    
    def _detect_divergences(self, stoch_rsi_k: pd.Series, data: pd.DataFrame) -> Dict[str, Any]:
        """Advanced divergence detection with statistical validation"""
        lookback = self.parameters['divergence_lookback']
        if len(stoch_rsi_k) < lookback or len(data) < lookback:
            return {'bullish_divergence': False, 'bearish_divergence': False, 'strength': 0.0}
        
        recent_stoch = stoch_rsi_k.tail(lookback)
        recent_prices = data['close'].tail(lookback)
        
        # Find statistically significant peaks and troughs
        stoch_peaks, stoch_troughs = self._find_statistical_extremes(recent_stoch, 'stoch')
        price_peaks, price_troughs = self._find_statistical_extremes(recent_prices, 'price')
        
        bullish_divergence = False
        bearish_divergence = False
        divergence_strength = 0.0
        
        # Bullish divergence: price lower lows, Stoch RSI higher lows
        if len(stoch_troughs) >= 2 and len(price_troughs) >= 2:
            last_stoch_trough = stoch_troughs[-1]
            prev_stoch_trough = stoch_troughs[-2]
            last_price_trough = price_troughs[-1]
            prev_price_trough = price_troughs[-2]
            
            price_decline = (last_price_trough[1] - prev_price_trough[1]) / prev_price_trough[1]
            stoch_improvement = (last_stoch_trough[1] - prev_stoch_trough[1]) / 100
            
            # Statistical significance test
            if price_decline < -0.01 and stoch_improvement > 0.05:
                # Additional validation using correlation
                price_segment = recent_prices.iloc[prev_price_trough[0]:last_price_trough[0]+1]
                stoch_segment = recent_stoch.iloc[prev_stoch_trough[0]:last_stoch_trough[0]+1]
                
                if len(price_segment) > 3 and len(stoch_segment) > 3:
                    correlation = np.corrcoef(price_segment.values, stoch_segment.values)[0, 1]
                    if correlation < -0.3:  # Negative correlation confirms divergence
                        bullish_divergence = True
                        divergence_strength = abs(price_decline) + abs(stoch_improvement) + abs(correlation) * 0.5
        
        # Bearish divergence: price higher highs, Stoch RSI lower highs
        if len(stoch_peaks) >= 2 and len(price_peaks) >= 2:
            last_stoch_peak = stoch_peaks[-1]
            prev_stoch_peak = stoch_peaks[-2]
            last_price_peak = price_peaks[-1]
            prev_price_peak = price_peaks[-2]
            
            price_increase = (last_price_peak[1] - prev_price_peak[1]) / prev_price_peak[1]
            stoch_decline = (last_stoch_peak[1] - prev_stoch_peak[1]) / 100
            
            if price_increase > 0.01 and stoch_decline < -0.05:
                # Additional validation
                price_segment = recent_prices.iloc[prev_price_peak[0]:last_price_peak[0]+1]
                stoch_segment = recent_stoch.iloc[prev_stoch_peak[0]:last_stoch_peak[0]+1]
                
                if len(price_segment) > 3 and len(stoch_segment) > 3:
                    correlation = np.corrcoef(price_segment.values, stoch_segment.values)[0, 1]
                    if correlation < -0.3:
                        bearish_divergence = True
                        divergence_strength = max(divergence_strength, 
                                                abs(price_increase) + abs(stoch_decline) + abs(correlation) * 0.5)
        
        return {
            'bullish_divergence': bullish_divergence,
            'bearish_divergence': bearish_divergence,
            'strength': float(divergence_strength),
            'stoch_peaks': len(stoch_peaks),
            'stoch_troughs': len(stoch_troughs),
            'price_peaks': len(price_peaks),
            'price_troughs': len(price_troughs)
        }
    
    def _find_statistical_extremes(self, series: pd.Series, series_type: str) -> Tuple[List[Tuple], List[Tuple]]:
        """Find statistically significant peaks and troughs"""
        if len(series) < 10:
            return [], []
        
        values = series.values
        peaks = []
        troughs = []
        
        # Adaptive threshold based on series characteristics
        if series_type == 'stoch':
            # For Stochastic RSI, use fixed thresholds
            peak_threshold = 70
            trough_threshold = 30
            window = 3
        else:
            # For price, use statistical threshold
            threshold = series.std() * 0.8
            peak_threshold = series.mean() + threshold
            trough_threshold = series.mean() - threshold
            window = 4
        
        for i in range(window, len(values) - window):
            window_values = values[i-window:i+window+1]
            current_value = values[i]
            
            # Peak detection
            if (current_value == max(window_values) and 
                (series_type == 'price' or current_value > peak_threshold)):
                peaks.append((i, current_value))
            
            # Trough detection
            elif (current_value == min(window_values) and 
                  (series_type == 'price' or current_value < trough_threshold)):
                troughs.append((i, current_value))
        
        return peaks, troughs
    
    def _analyze_multi_timeframe_momentum(self, stoch_rsi_k: pd.Series, stoch_rsi_d: pd.Series) -> Dict[str, Any]:
        """Analyze momentum across multiple timeframes"""
        if not self.parameters['multi_timeframe'] or len(stoch_rsi_k) < 60:
            return {'convergence': False, 'short_trend': 'neutral', 'medium_trend': 'neutral', 'long_trend': 'neutral'}
        
        # Short term (5-10 periods)
        short_k = stoch_rsi_k.tail(10)
        short_trend_k = np.polyfit(range(len(short_k)), short_k.values, 1)[0]
        
        # Medium term (15-25 periods)
        medium_k = stoch_rsi_k.tail(25)
        medium_trend_k = np.polyfit(range(len(medium_k)), medium_k.values, 1)[0]
        
        # Long term (40-60 periods)
        long_k = stoch_rsi_k.tail(60)
        long_trend_k = np.polyfit(range(len(long_k)), long_k.values, 1)[0]
        
        # Classify trends
        def classify_trend(slope, threshold=2.0):
            if slope > threshold:
                return 'bullish'
            elif slope < -threshold:
                return 'bearish'
            else:
                return 'neutral'
        
        short_classification = classify_trend(short_trend_k)
        medium_classification = classify_trend(medium_trend_k, 1.5)
        long_classification = classify_trend(long_trend_k, 1.0)
        
        # Check for convergence
        trends = [short_classification, medium_classification, long_classification]
        convergence = len(set(trends)) == 1 and trends[0] != 'neutral'
        
        # Additional analysis: %K vs %D relationship
        current_k = stoch_rsi_k.iloc[-1]
        current_d = stoch_rsi_d.iloc[-1] if len(stoch_rsi_d) > 0 else current_k
        
        k_d_relationship = 'neutral'
        if current_k > current_d + 5:
            k_d_relationship = 'k_above_d'
        elif current_k < current_d - 5:
            k_d_relationship = 'k_below_d'
        
        return {
            'convergence': convergence,
            'short_trend': short_classification,
            'medium_trend': medium_classification,
            'long_trend': long_classification,
            'k_d_relationship': k_d_relationship,
            'trends': {
                'short': float(short_trend_k),
                'medium': float(medium_trend_k),
                'long': float(long_trend_k)
            }
        }
    
    def _classify_market_regime(self, stoch_rsi_k: pd.Series, rsi: pd.Series, data: pd.DataFrame) -> Dict[str, Any]:
        """Classify market regime using Stochastic RSI characteristics"""
        if not self.parameters['regime_classification'] or len(stoch_rsi_k) < 30:
            return {'regime': 'unknown', 'momentum_state': 'undefined', 'volatility': 'normal'}
        
        recent_stoch = stoch_rsi_k.tail(30)
        recent_rsi = rsi.tail(30)
        current_stoch = stoch_rsi_k.iloc[-1]
        current_rsi = rsi.iloc[-1]
        
        # Momentum state classification
        if current_stoch > 80 and current_rsi > 70:
            momentum_state = 'extremely_overbought'
        elif current_stoch > 60 and current_rsi > 60:
            momentum_state = 'overbought'
        elif current_stoch < 20 and current_rsi < 30:
            momentum_state = 'extremely_oversold'
        elif current_stoch < 40 and current_rsi < 40:
            momentum_state = 'oversold'
        else:
            momentum_state = 'neutral'
        
        # Volatility analysis
        stoch_volatility = recent_stoch.std()
        if stoch_volatility > 25:
            volatility = 'high'
        elif stoch_volatility > 15:
            volatility = 'medium'
        else:
            volatility = 'low'
        
        # Overall regime classification
        overbought_ratio = len(recent_stoch[recent_stoch > 80]) / len(recent_stoch)
        oversold_ratio = len(recent_stoch[recent_stoch < 20]) / len(recent_stoch)
        
        if overbought_ratio > 0.3:
            regime = 'persistent_overbought'
        elif oversold_ratio > 0.3:
            regime = 'persistent_oversold'
        elif volatility == 'high':
            regime = 'volatile_ranging'
        elif momentum_state in ['extremely_overbought', 'extremely_oversold']:
            regime = 'extreme_momentum'
        else:
            regime = 'normal_oscillation'
        
        return {
            'regime': regime,
            'momentum_state': momentum_state,
            'volatility': volatility,
            'stoch_volatility': float(stoch_volatility),
            'overbought_ratio': float(overbought_ratio),
            'oversold_ratio': float(oversold_ratio),
            'current_levels': {
                'stoch_rsi': float(current_stoch),
                'rsi': float(current_rsi)
            }
        }
    
    def _detect_patterns(self, stoch_rsi_k: pd.Series, stoch_rsi_d: pd.Series) -> Dict[str, Any]:
        """Detect chart patterns in Stochastic RSI"""
        if not self.parameters['pattern_detection'] or len(stoch_rsi_k) < 20:
            return {'pattern': 'insufficient_data', 'confidence': 0.0}
        
        try:
            recent_k = stoch_rsi_k.tail(20)
            recent_d = stoch_rsi_d.tail(20)
            
            # Create feature matrix for pattern recognition
            features = []
            window_size = 5
            
            for i in range(len(recent_k) - window_size + 1):
                k_window = recent_k.iloc[i:i+window_size].values
                d_window = recent_d.iloc[i:i+window_size].values
                
                # Statistical features
                k_mean = np.mean(k_window)
                k_std = np.std(k_window)
                k_trend = np.polyfit(range(len(k_window)), k_window, 1)[0]
                k_range = np.max(k_window) - np.min(k_window)
                
                # Crossover features
                crossovers = sum(1 for j in range(1, len(k_window)) 
                               if (k_window[j] > d_window[j]) != (k_window[j-1] > d_window[j-1]))
                
                # Position features
                overbought_time = sum(1 for val in k_window if val > 80)
                oversold_time = sum(1 for val in k_window if val < 20)
                
                features.append([k_mean, k_std, k_trend, k_range, crossovers, overbought_time, oversold_time])
            
            if len(features) < 5:
                return {'pattern': 'insufficient_data', 'confidence': 0.0}
            
            features = np.array(features)
            
            # Use clustering to identify patterns
            try:
                n_clusters = min(4, len(features))
                clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
                pattern_labels = clusterer.fit_predict(features)
                current_pattern = pattern_labels[-1]
                
                # Calculate silhouette score for confidence
                if len(set(pattern_labels)) > 1:
                    confidence = silhouette_score(features, pattern_labels)
                    confidence = max(0, min(1, (confidence + 1) / 2))  # Normalize to [0,1]
                else:
                    confidence = 0.0
                
                # Map clusters to pattern names based on current characteristics
                current_k = recent_k.iloc[-1]
                current_trend = np.polyfit(range(len(recent_k)), recent_k.values, 1)[0]
                
                if current_k > 80:
                    if current_trend > 0:
                        pattern_name = 'overbought_strengthening'
                    else:
                        pattern_name = 'overbought_weakening'
                elif current_k < 20:
                    if current_trend > 0:
                        pattern_name = 'oversold_recovering'
                    else:
                        pattern_name = 'oversold_deepening'
                elif abs(current_trend) > 2:
                    pattern_name = 'trending'
                else:
                    pattern_name = 'consolidating'
                
            except:
                pattern_name = 'unknown'
                confidence = 0.0
            
            return {
                'pattern': pattern_name,
                'confidence': float(confidence),
                'cluster_id': int(current_pattern) if 'current_pattern' in locals() else -1
            }
            
        except Exception:
            return {'pattern': 'error', 'confidence': 0.0}
    
    def _train_ml_models(self, stoch_rsi_k: pd.Series, data: pd.DataFrame) -> bool:
        """Train ML models for momentum prediction"""
        if len(stoch_rsi_k) < self.parameters['ml_lookback'] * 2:
            return False
        
        try:
            features, targets = self._prepare_ml_data(stoch_rsi_k, data)
            if len(features) > 50:
                # Scale features
                scaled_features = self.robust_scaler.fit_transform(features)
                
                # Train momentum classifier
                self.momentum_classifier.fit(scaled_features, targets)
                
                self.models_trained = True
                return True
        except Exception:
            pass
        return False
    
    def _prepare_ml_data(self, stoch_rsi_k: pd.Series, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare ML training data"""
        features, targets = [], []
        lookback = 15
        
        for i in range(lookback, len(stoch_rsi_k) - 10):
            stoch_window = stoch_rsi_k.iloc[i-lookback:i].values
            price_window = data['close'].iloc[i-lookback:i].values
            
            # Stochastic RSI features
            stoch_mean = np.mean(stoch_window)
            stoch_std = np.std(stoch_window)
            stoch_trend = np.polyfit(range(len(stoch_window)), stoch_window, 1)[0]
            stoch_current = stoch_window[-1]
            
            # Position features
            overbought_ratio = len([x for x in stoch_window if x > 80]) / len(stoch_window)
            oversold_ratio = len([x for x in stoch_window if x < 20]) / len(stoch_window)
            
            # Momentum features
            stoch_momentum = np.mean(np.diff(stoch_window[-5:]))
            
            # Price features
            price_returns = np.diff(price_window) / price_window[:-1]
            price_volatility = np.std(price_returns)
            price_trend = np.polyfit(range(len(price_window)), price_window, 1)[0]
            
            # Volume features (if available)
            if 'volume' in data.columns:
                volume_window = data['volume'].iloc[i-lookback:i].values
                volume_trend = np.polyfit(range(len(volume_window)), volume_window, 1)[0]
                volume_ratio = volume_window[-1] / (np.mean(volume_window) + 1e-8)
            else:
                volume_trend = 0
                volume_ratio = 1
            
            feature_vector = [
                stoch_mean, stoch_std, stoch_trend, stoch_current,
                overbought_ratio, oversold_ratio, stoch_momentum,
                price_volatility, price_trend,
                volume_trend, volume_ratio
            ]
            
            # Target: future momentum direction
            future_stoch = stoch_rsi_k.iloc[i+5:i+10].values
            if len(future_stoch) > 0:
                future_change = np.mean(future_stoch) - stoch_current
                if future_change > 5:
                    target = 2  # Strong bullish
                elif future_change > 1:
                    target = 1  # Bullish
                elif future_change < -5:
                    target = 0  # Bearish
                else:
                    target = 1  # Neutral (default)
            else:
                target = 1
            
            features.append(feature_vector)
            targets.append(target)
        
        return np.array(features), np.array(targets)
    
    def calculate_raw(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate Stochastic RSI with comprehensive analysis"""
        try:
            # Get adaptive periods
            rsi_period, stoch_k_period, stoch_d_period = self._adapt_periods(data)
            
            # Calculate RSI (volume-weighted if enabled)
            rsi = self._calculate_volume_weighted_rsi(data, rsi_period)
            
            # Calculate Stochastic RSI
            stoch_rsi_k, stoch_rsi_d = self._calculate_stochastic_rsi(rsi, stoch_k_period, stoch_d_period)
            
            # Advanced analysis
            optimized_thresholds = self._optimize_thresholds(stoch_rsi_k, data)
            divergence_analysis = self._detect_divergences(stoch_rsi_k, data)
            momentum_analysis = self._analyze_multi_timeframe_momentum(stoch_rsi_k, stoch_rsi_d)
            regime_classification = self._classify_market_regime(stoch_rsi_k, rsi, data)
            pattern_analysis = self._detect_patterns(stoch_rsi_k, stoch_rsi_d)
            
            # Train ML models
            if not self.models_trained:
                self._train_ml_models(stoch_rsi_k, data)
            
            # Generate signal
            signal, confidence = self._generate_stoch_rsi_signal(
                stoch_rsi_k, stoch_rsi_d, optimized_thresholds, divergence_analysis,
                momentum_analysis, pattern_analysis, data
            )
            
            # Update history
            if len(stoch_rsi_k) > 0 and not pd.isna(stoch_rsi_k.iloc[-1]):
                self.history['stoch_rsi_k'].append(float(stoch_rsi_k.iloc[-1]))
                self.history['stoch_rsi_d'].append(float(stoch_rsi_d.iloc[-1]) if not pd.isna(stoch_rsi_d.iloc[-1]) else 50.0)
                self.history['rsi_values'].append(float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0)
                self.history['threshold_values'].append(optimized_thresholds)
                self.history['regime_states'].append(regime_classification)
                
                # Keep history limited
                for key in self.history:
                    if len(self.history[key]) > 100:
                        self.history[key] = self.history[key][-100:]
            
            result = {
                'stoch_rsi_k': float(stoch_rsi_k.iloc[-1]) if len(stoch_rsi_k) > 0 and not pd.isna(stoch_rsi_k.iloc[-1]) else 50.0,
                'stoch_rsi_d': float(stoch_rsi_d.iloc[-1]) if len(stoch_rsi_d) > 0 and not pd.isna(stoch_rsi_d.iloc[-1]) else 50.0,
                'rsi': float(rsi.iloc[-1]) if len(rsi) > 0 and not pd.isna(rsi.iloc[-1]) else 50.0,
                'signal': signal,
                'confidence': confidence,
                'optimized_thresholds': optimized_thresholds,
                'divergence_analysis': divergence_analysis,
                'momentum_analysis': momentum_analysis,
                'regime_classification': regime_classification,
                'pattern_analysis': pattern_analysis,
                'adaptive_periods': {'rsi': rsi_period, 'stoch_k': stoch_k_period, 'stoch_d': stoch_d_period},
                'values_history': {
                    'stoch_rsi_k': stoch_rsi_k.tail(30).tolist(),
                    'stoch_rsi_d': stoch_rsi_d.tail(30).tolist(),
                    'rsi': rsi.tail(30).tolist()
                }
            }
            
            return result
            
        except Exception as e:
            raise IndicatorCalculationException(
                indicator_name=self.name,
                calculation_step="raw_calculation",
                message=f"Failed to calculate Stochastic RSI: {str(e)}",
                cause=e
            )
    
    def _generate_stoch_rsi_signal(self, stoch_rsi_k: pd.Series, stoch_rsi_d: pd.Series,
                                  optimized_thresholds: Dict, divergence_analysis: Dict,
                                  momentum_analysis: Dict, pattern_analysis: Dict,
                                  data: pd.DataFrame) -> Tuple[SignalType, float]:
        """Generate comprehensive Stochastic RSI signal"""
        signal_components = []
        confidence_components = []
        
        current_k = stoch_rsi_k.iloc[-1] if len(stoch_rsi_k) > 0 and not pd.isna(stoch_rsi_k.iloc[-1]) else 50
        current_d = stoch_rsi_d.iloc[-1] if len(stoch_rsi_d) > 0 and not pd.isna(stoch_rsi_d.iloc[-1]) else 50
        overbought = optimized_thresholds['overbought']
        oversold = optimized_thresholds['oversold']
        
        # Threshold-based signals
        if current_k > overbought:
            signal_components.append(-0.8)
            confidence_components.append(0.7)
        elif current_k < oversold:
            signal_components.append(0.8)
            confidence_components.append(0.7)
        
        # %K and %D crossover signals
        if len(stoch_rsi_k) > 1 and len(stoch_rsi_d) > 1:
            prev_k = stoch_rsi_k.iloc[-2]
            prev_d = stoch_rsi_d.iloc[-2]
            
            # Bullish crossover: %K crosses above %D
            if prev_k <= prev_d and current_k > current_d:
                # Stronger signal if in oversold territory
                strength = 0.9 if current_k < oversold + 10 else 0.6
                signal_components.append(strength)
                confidence_components.append(0.8)
            
            # Bearish crossover: %K crosses below %D
            elif prev_k >= prev_d and current_k < current_d:
                # Stronger signal if in overbought territory
                strength = -0.9 if current_k > overbought - 10 else -0.6
                signal_components.append(strength)
                confidence_components.append(0.8)
        
        # Divergence signals
        if divergence_analysis['bullish_divergence']:
            signal_components.append(0.8 * divergence_analysis['strength'])
            confidence_components.append(0.8)
        elif divergence_analysis['bearish_divergence']:
            signal_components.append(-0.8 * divergence_analysis['strength'])
            confidence_components.append(0.8)
        
        # Multi-timeframe momentum signals
        if momentum_analysis['convergence']:
            trend = momentum_analysis['short_trend']
            if trend == 'bullish':
                signal_components.append(0.6)
                confidence_components.append(0.7)
            elif trend == 'bearish':
                signal_components.append(-0.6)
                confidence_components.append(0.7)
        
        # Pattern-based signals
        if pattern_analysis['confidence'] > 0.6:
            pattern = pattern_analysis['pattern']
            if pattern == 'oversold_recovering':
                signal_components.append(0.7)
                confidence_components.append(pattern_analysis['confidence'])
            elif pattern == 'overbought_weakening':
                signal_components.append(-0.7)
                confidence_components.append(pattern_analysis['confidence'])
            elif pattern == 'trending' and current_k > 50:
                signal_components.append(0.5)
                confidence_components.append(pattern_analysis['confidence'])
            elif pattern == 'trending' and current_k < 50:
                signal_components.append(-0.5)
                confidence_components.append(pattern_analysis['confidence'])
        
        # ML enhancement
        if self.models_trained:
            try:
                ml_signal, ml_confidence = self._get_ml_signal(stoch_rsi_k, data)
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
    
    def _get_ml_signal(self, stoch_rsi_k: pd.Series, data: pd.DataFrame) -> Tuple[Optional[SignalType], float]:
        """Get ML-based momentum prediction"""
        try:
            lookback = 15
            if len(stoch_rsi_k) < lookback:
                return None, 0.0
            
            stoch_window = stoch_rsi_k.tail(lookback).values
            price_window = data['close'].tail(lookback).values
            
            # Recreate feature vector
            stoch_mean = np.mean(stoch_window)
            stoch_std = np.std(stoch_window)
            stoch_trend = np.polyfit(range(len(stoch_window)), stoch_window, 1)[0]
            stoch_current = stoch_window[-1]
            
            overbought_ratio = len([x for x in stoch_window if x > 80]) / len(stoch_window)
            oversold_ratio = len([x for x in stoch_window if x < 20]) / len(stoch_window)
            stoch_momentum = np.mean(np.diff(stoch_window[-5:]))
            
            price_returns = np.diff(price_window) / price_window[:-1]
            price_volatility = np.std(price_returns)
            price_trend = np.polyfit(range(len(price_window)), price_window, 1)[0]
            
            if 'volume' in data.columns:
                volume_window = data['volume'].tail(lookback).values
                volume_trend = np.polyfit(range(len(volume_window)), volume_window, 1)[0]
                volume_ratio = volume_window[-1] / (np.mean(volume_window) + 1e-8)
            else:
                volume_trend = 0
                volume_ratio = 1
            
            feature_vector = np.array([[
                stoch_mean, stoch_std, stoch_trend, stoch_current,
                overbought_ratio, oversold_ratio, stoch_momentum,
                price_volatility, price_trend,
                volume_trend, volume_ratio
            ]])
            
            scaled_features = self.robust_scaler.transform(feature_vector)
            momentum_proba = self.momentum_classifier.predict_proba(scaled_features)[0]
            
            if len(momentum_proba) >= 3:
                bearish_prob = momentum_proba[0]
                neutral_prob = momentum_proba[1]
                bullish_prob = momentum_proba[2]
                
                max_prob = max(momentum_proba)
                if max_prob > 0.7:
                    if bullish_prob == max_prob:
                        return SignalType.BUY, max_prob
                    elif bearish_prob == max_prob:
                        return SignalType.SELL, max_prob
        except:
            pass
        
        return None, 0.0
    
    def _generate_signal(self, value: Any, data: pd.DataFrame) -> Tuple[Optional[SignalType], float]:
        if isinstance(value, dict) and 'signal' in value and 'confidence' in value:
            return value['signal'], value['confidence']
        return SignalType.NEUTRAL, 0.0
    
    def _get_metadata(self, data: pd.DataFrame) -> Dict[str, Any]:
        base_metadata = super()._get_metadata(data)
        base_metadata.update({
            'indicator_type': 'momentum',
            'indicator_category': 'stochastic_rsi',
            'models_trained': self.models_trained,
            'adaptive_periods': self.parameters['adaptive_periods'],
            'volume_weighted': self.parameters['volume_weighted'],
            'multi_timeframe': self.parameters['multi_timeframe'],
            'regime_classification': self.parameters['regime_classification'],
            'optimization_enabled': self.parameters['optimization_enabled'],
            'data_type': 'ohlcv',
            'complexity': 'advanced'
        })
        return base_metadata