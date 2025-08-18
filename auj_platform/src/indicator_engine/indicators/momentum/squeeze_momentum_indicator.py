"""
Squeeze Momentum Indicator - Advanced Implementation
===================================================

Advanced Squeeze Momentum Indicator with ML-enhanced volatility detection,
adaptive Bollinger Bands and Keltner Channels, and sophisticated momentum analysis.

Author: AUJ Platform Development Team
Mission: Building advanced trading indicators for humanitarian impact
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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


class SqueezeMomentumIndicator(StandardIndicatorInterface):
    """
    Advanced Squeeze Momentum Indicator Implementation
    
    Features:
    - Adaptive Bollinger Bands and Keltner Channels
    - ML-enhanced squeeze detection and breakout prediction
    - Multi-timeframe momentum analysis
    - Volume-weighted momentum calculations
    - Dynamic threshold optimization
    - Advanced pattern recognition for squeeze phases
    - Institutional flow detection during squeeze periods
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'bb_period': 20,
            'bb_std_dev': 2.0,
            'kc_period': 20,
            'kc_multiplier': 1.5,
            'momentum_period': 12,
            'adaptive_periods': True,
            'volume_weighted': True,
            'ml_lookback': 60,
            'squeeze_sensitivity': 0.95,  # BB inside KC threshold
            'breakout_threshold': 1.05,   # KC outside BB threshold
            'momentum_smoothing': 3,
            'multi_timeframe': True,
            'pattern_detection': True
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__(name="SqueezeMomentumIndicator", parameters=default_params)
        
        self.scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        self.squeeze_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.momentum_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.pattern_clusterer = KMeans(n_clusters=4, random_state=42)
        self.models_trained = False
        
        self.history = {
            'momentum_values': [],
            'squeeze_states': [],
            'bb_upper': [],
            'bb_lower': [],
            'kc_upper': [],
            'kc_lower': [],
            'squeeze_signals': [],
            'breakout_signals': []
        }
    
    def get_data_requirements(self) -> DataRequirement:
        max_period = max(self.parameters['bb_period'], 
                        self.parameters['kc_period'],
                        self.parameters['ml_lookback'])
        return DataRequirement(
            data_type=DataType.OHLCV,
            required_columns=['high', 'low', 'close', 'volume'],
            min_periods=max_period + 50,
            lookback_periods=200
        )
    
    def _adapt_periods(self, data: pd.DataFrame) -> Tuple[int, int]:
        """Adapt BB and KC periods based on market volatility"""
        if not self.parameters['adaptive_periods']:
            return self.parameters['bb_period'], self.parameters['kc_period']
        
        # Calculate market volatility regime
        if len(data) < 60:
            return self.parameters['bb_period'], self.parameters['kc_period']
        
        returns = data['close'].pct_change().tail(60)
        current_vol = returns.std()
        rolling_vol = returns.rolling(window=20).std()
        avg_vol = rolling_vol.mean()
        vol_ratio = current_vol / (avg_vol + 1e-8)
        
        # Calculate trend persistence
        price_changes = data['close'].diff().tail(40)
        trend_consistency = abs(price_changes.sum()) / (price_changes.abs().sum() + 1e-8)
        
        base_bb = self.parameters['bb_period']
        base_kc = self.parameters['kc_period']
        
        # Adjust for volatility regime
        if vol_ratio > 1.5:  # High volatility - shorter periods
            bb_period = max(10, int(base_bb * 0.7))
            kc_period = max(10, int(base_kc * 0.7))
        elif vol_ratio < 0.6:  # Low volatility - longer periods
            bb_period = min(30, int(base_bb * 1.2))
            kc_period = min(30, int(base_kc * 1.2))
        else:  # Normal volatility
            bb_period = base_bb
            kc_period = base_kc
        
        # Adjust for trend persistence
        if trend_consistency > 0.7:  # Strong trend - longer periods
            bb_period = min(bb_period + 3, 35)
            kc_period = min(kc_period + 3, 35)
        elif trend_consistency < 0.3:  # Choppy market - shorter periods
            bb_period = max(bb_period - 2, 8)
            kc_period = max(kc_period - 2, 8)
        
        return bb_period, kc_period
    
    def _calculate_bollinger_bands(self, data: pd.DataFrame, period: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate adaptive Bollinger Bands"""
        close = data['close']
        
        # Base calculation
        sma = close.rolling(window=period).mean()
        std_dev = close.rolling(window=period).std()
        
        # Adaptive standard deviation multiplier
        if self.parameters['adaptive_periods'] and len(close) >= 60:
            # Calculate volatility regime for adaptive multiplier
            returns = close.pct_change().tail(60)
            vol_percentile = stats.percentileofscore(returns.rolling(20).std().dropna(), returns.std())
            
            # Adjust std_dev multiplier based on volatility percentile
            if vol_percentile > 80:  # High volatility - wider bands
                multiplier = self.parameters['bb_std_dev'] * 1.2
            elif vol_percentile < 20:  # Low volatility - narrower bands
                multiplier = self.parameters['bb_std_dev'] * 0.8
            else:
                multiplier = self.parameters['bb_std_dev']
        else:
            multiplier = self.parameters['bb_std_dev']
        
        upper_band = sma + (std_dev * multiplier)
        lower_band = sma - (std_dev * multiplier)
        
        return upper_band, lower_band, sma
    
    def _calculate_keltner_channels(self, data: pd.DataFrame, period: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate adaptive Keltner Channels"""
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Typical Price and EMA
        typical_price = (high + low + close) / 3
        ema = typical_price.ewm(span=period).mean()
        
        # True Range
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Average True Range
        atr = true_range.ewm(span=period).mean()
        
        # Adaptive multiplier based on volatility
        multiplier = self.parameters['kc_multiplier']
        if self.parameters['adaptive_periods'] and len(close) >= 60:
            vol_ratio = true_range.tail(20).mean() / true_range.tail(60).mean()
            if vol_ratio > 1.3:  # High recent volatility - wider channels
                multiplier *= 1.1
            elif vol_ratio < 0.7:  # Low recent volatility - narrower channels
                multiplier *= 0.9
        
        upper_channel = ema + (atr * multiplier)
        lower_channel = ema - (atr * multiplier)
        
        return upper_channel, lower_channel, ema
    
    def _calculate_momentum(self, data: pd.DataFrame, bb_mid: pd.Series, kc_mid: pd.Series) -> pd.Series:
        """Calculate volume-weighted momentum"""
        close = data['close']
        momentum_period = self.parameters['momentum_period']
        
        # Base momentum calculation
        if len(bb_mid) > 0 and len(kc_mid) > 0:
            # Use average of BB and KC midlines for momentum calculation
            midline = (bb_mid + kc_mid) / 2
            highest = data['high'].rolling(window=momentum_period).max()
            lowest = data['low'].rolling(window=momentum_period).min()
            momentum = close - (highest + lowest) / 2
        else:
            # Fallback to simple momentum
            momentum = close - close.shift(momentum_period)
        
        # Volume weighting if enabled
        if self.parameters['volume_weighted'] and 'volume' in data.columns:
            volume = data['volume']
            volume_sma = volume.rolling(window=20).mean()
            volume_ratio = volume / (volume_sma + 1e-8)
            volume_weight = np.tanh(volume_ratio - 1)  # Sigmoid weighting
            
            # Apply volume weight (institutional flow emphasis)
            momentum = momentum * (1 + volume_weight * 0.3)
        
        # Smooth momentum if specified
        if self.parameters['momentum_smoothing'] > 1:
            momentum = momentum.rolling(window=self.parameters['momentum_smoothing']).mean()
        
        return momentum
    
    def _detect_squeeze(self, bb_upper: pd.Series, bb_lower: pd.Series,
                       kc_upper: pd.Series, kc_lower: pd.Series) -> pd.Series:
        """Detect squeeze conditions (BB inside KC)"""
        # Squeeze occurs when BB is inside KC
        squeeze_condition = (bb_upper <= kc_upper * self.parameters['squeeze_sensitivity']) & \
                          (bb_lower >= kc_lower * self.parameters['squeeze_sensitivity'])
        
        return squeeze_condition
    
    def _detect_breakout(self, bb_upper: pd.Series, bb_lower: pd.Series,
                        kc_upper: pd.Series, kc_lower: pd.Series) -> pd.Series:
        """Detect breakout conditions (KC outside BB)"""
        # Breakout occurs when KC moves outside BB
        breakout_condition = (kc_upper >= bb_upper * self.parameters['breakout_threshold']) | \
                           (kc_lower <= bb_lower * self.parameters['breakout_threshold'])
        
        return breakout_condition
    
    def _analyze_squeeze_patterns(self, squeeze_states: pd.Series, momentum: pd.Series) -> Dict[str, Any]:
        """Analyze squeeze patterns and phases"""
        if len(squeeze_states) < 20 or not self.parameters['pattern_detection']:
            return {'phase': 'unknown', 'duration': 0, 'intensity': 0.0, 'pattern': 'insufficient_data'}
        
        try:
            recent_squeeze = squeeze_states.tail(20)
            recent_momentum = momentum.tail(20)
            
            # Current squeeze state
            current_squeeze = recent_squeeze.iloc[-1]
            
            # Calculate squeeze duration
            squeeze_duration = 0
            for i in range(len(recent_squeeze) - 1, -1, -1):
                if recent_squeeze.iloc[i] == current_squeeze:
                    squeeze_duration += 1
                else:
                    break
            
            # Momentum during squeeze
            if current_squeeze:
                # In squeeze - analyze momentum buildup
                squeeze_momentum = recent_momentum[recent_squeeze].tail(min(squeeze_duration, 10))
                
                if len(squeeze_momentum) > 0:
                    momentum_trend = np.polyfit(range(len(squeeze_momentum)), squeeze_momentum.values, 1)[0]
                    momentum_std = squeeze_momentum.std()
                    
                    # Classify squeeze intensity
                    if momentum_std < recent_momentum.std() * 0.5:
                        intensity = 'high'  # Low volatility = high squeeze
                    elif momentum_std < recent_momentum.std() * 0.8:
                        intensity = 'medium'
                    else:
                        intensity = 'low'
                    
                    # Determine phase
                    if momentum_trend > 0:
                        phase = 'building_bullish'
                    elif momentum_trend < 0:
                        phase = 'building_bearish'
                    else:
                        phase = 'consolidating'
                else:
                    intensity = 'unknown'
                    phase = 'consolidating'
            else:
                # Not in squeeze - analyze breakout
                intensity = 'none'
                recent_mom_trend = np.polyfit(range(len(recent_momentum)), recent_momentum.values, 1)[0]
                
                if recent_mom_trend > 0:
                    phase = 'breakout_bullish'
                elif recent_mom_trend < 0:
                    phase = 'breakout_bearish'
                else:
                    phase = 'post_breakout'
            
            # Pattern recognition using clustering
            try:
                if len(recent_momentum) >= 10:
                    # Create feature windows for pattern recognition
                    features = []
                    window_size = 5
                    
                    for i in range(len(recent_momentum) - window_size + 1):
                        window = recent_momentum.iloc[i:i+window_size].values
                        squeeze_window = recent_squeeze.iloc[i:i+window_size].values
                        
                        features.append([
                            np.mean(window),
                            np.std(window),
                            np.polyfit(range(len(window)), window, 1)[0],
                            np.sum(squeeze_window) / len(squeeze_window),
                            window[-1] - window[0]
                        ])
                    
                    if len(features) >= 4:
                        features = np.array(features)
                        patterns = self.pattern_clusterer.fit_predict(features)
                        current_pattern = patterns[-1]
                        
                        pattern_names = {
                            0: 'accumulation',
                            1: 'distribution', 
                            2: 'trending',
                            3: 'reversal'
                        }
                        pattern_name = pattern_names.get(current_pattern, 'unknown')
                    else:
                        pattern_name = 'insufficient_data'
                else:
                    pattern_name = 'insufficient_data'
                    
            except:
                pattern_name = 'unknown'
            
            return {
                'phase': phase,
                'duration': squeeze_duration,
                'intensity': intensity,
                'pattern': pattern_name,
                'current_squeeze': bool(current_squeeze)
            }
            
        except Exception:
            return {'phase': 'unknown', 'duration': 0, 'intensity': 0.0, 'pattern': 'error'}
    
    def _analyze_multi_timeframe_momentum(self, momentum: pd.Series) -> Dict[str, Any]:
        """Analyze momentum across multiple timeframes"""
        if not self.parameters['multi_timeframe'] or len(momentum) < 60:
            return {'short_trend': 'neutral', 'medium_trend': 'neutral', 'long_trend': 'neutral', 'convergence': False}
        
        # Short term (5-10 periods)
        short_momentum = momentum.tail(10)
        short_trend = np.polyfit(range(len(short_momentum)), short_momentum.values, 1)[0]
        
        # Medium term (15-25 periods)
        medium_momentum = momentum.tail(25)
        medium_trend = np.polyfit(range(len(medium_momentum)), medium_momentum.values, 1)[0]
        
        # Long term (40-60 periods)
        long_momentum = momentum.tail(60)
        long_trend = np.polyfit(range(len(long_momentum)), long_momentum.values, 1)[0]
        
        # Classify trends
        def classify_trend(slope, threshold=0.1):
            if slope > threshold:
                return 'bullish'
            elif slope < -threshold:
                return 'bearish'
            else:
                return 'neutral'
        
        # Adjust thresholds based on momentum volatility
        mom_std = momentum.tail(30).std()
        threshold = mom_std * 0.1
        
        short_classification = classify_trend(short_trend, threshold)
        medium_classification = classify_trend(medium_trend, threshold)
        long_classification = classify_trend(long_trend, threshold)
        
        # Check for convergence
        trends = [short_classification, medium_classification, long_classification]
        convergence = len(set(trends)) == 1 and trends[0] != 'neutral'
        
        return {
            'short_trend': short_classification,
            'medium_trend': medium_classification,
            'long_trend': long_classification,
            'convergence': convergence,
            'trend_slopes': {
                'short': float(short_trend),
                'medium': float(medium_trend),
                'long': float(long_trend)
            }
        }
    
    def _train_ml_models(self, squeeze_states: pd.Series, momentum: pd.Series, data: pd.DataFrame) -> bool:
        """Train ML models for squeeze prediction and momentum forecasting"""
        if len(squeeze_states) < self.parameters['ml_lookback'] * 2:
            return False
        
        try:
            features, squeeze_targets, momentum_targets = self._prepare_ml_data(squeeze_states, momentum, data)
            if len(features) > 40:
                self.scaler.fit(features)
                scaled_features = self.scaler.transform(features)
                
                # Train squeeze classifier
                self.squeeze_classifier.fit(scaled_features, squeeze_targets)
                
                # Train momentum predictor
                self.momentum_predictor.fit(scaled_features, momentum_targets)
                
                self.models_trained = True
                return True
        except Exception:
            pass
        return False
    
    def _prepare_ml_data(self, squeeze_states: pd.Series, momentum: pd.Series, 
                        data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare ML training data"""
        features, squeeze_targets, momentum_targets = [], [], []
        lookback = 15
        
        for i in range(lookback, len(squeeze_states) - 10):
            # Feature windows
            squeeze_window = squeeze_states.iloc[i-lookback:i].values
            momentum_window = momentum.iloc[i-lookback:i].values
            price_window = data['close'].iloc[i-lookback:i].values
            volume_window = data['volume'].iloc[i-lookback:i].values if 'volume' in data.columns else np.ones(lookback)
            
            # Squeeze features
            squeeze_ratio = np.sum(squeeze_window) / len(squeeze_window)
            squeeze_changes = np.sum(np.diff(squeeze_window.astype(int)))
            
            # Momentum features
            momentum_mean = np.mean(momentum_window)
            momentum_std = np.std(momentum_window)
            momentum_trend = np.polyfit(range(len(momentum_window)), momentum_window, 1)[0]
            momentum_current = momentum_window[-1]
            
            # Price features
            price_volatility = np.std(np.diff(price_window) / price_window[:-1])
            price_trend = np.polyfit(range(len(price_window)), price_window, 1)[0]
            
            # Volume features
            volume_trend = np.polyfit(range(len(volume_window)), volume_window, 1)[0]
            volume_ratio = volume_window[-1] / (np.mean(volume_window) + 1e-8)
            
            # Statistical features
            momentum_skew = stats.skew(momentum_window)
            momentum_kurtosis = stats.kurtosis(momentum_window)
            
            feature_vector = [
                squeeze_ratio, squeeze_changes,
                momentum_mean, momentum_std, momentum_trend, momentum_current,
                price_volatility, price_trend,
                volume_trend, volume_ratio,
                momentum_skew, momentum_kurtosis
            ]
            
            # Targets
            # Squeeze target: will there be a squeeze change in next 5 periods?
            future_squeeze = squeeze_states.iloc[i+1:i+6].values
            squeeze_target = 1 if np.any(future_squeeze != squeeze_states.iloc[i]) else 0
            
            # Momentum target: future momentum direction
            future_momentum = momentum.iloc[i+5:i+10].values
            if len(future_momentum) > 0:
                momentum_target = np.mean(future_momentum) - momentum_current
            else:
                momentum_target = 0.0
            
            features.append(feature_vector)
            squeeze_targets.append(squeeze_target)
            momentum_targets.append(momentum_target)
        
        return np.array(features), np.array(squeeze_targets), np.array(momentum_targets)
    
    def calculate_raw(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate Squeeze Momentum with comprehensive analysis"""
        try:
            # Get adaptive periods
            bb_period, kc_period = self._adapt_periods(data)
            
            # Calculate Bollinger Bands and Keltner Channels
            bb_upper, bb_lower, bb_mid = self._calculate_bollinger_bands(data, bb_period)
            kc_upper, kc_lower, kc_mid = self._calculate_keltner_channels(data, kc_period)
            
            # Calculate momentum
            momentum = self._calculate_momentum(data, bb_mid, kc_mid)
            
            # Detect squeeze and breakout conditions
            squeeze_states = self._detect_squeeze(bb_upper, bb_lower, kc_upper, kc_lower)
            breakout_states = self._detect_breakout(bb_upper, bb_lower, kc_upper, kc_lower)
            
            # Advanced analysis
            squeeze_patterns = self._analyze_squeeze_patterns(squeeze_states, momentum)
            momentum_analysis = self._analyze_multi_timeframe_momentum(momentum)
            
            # Train ML models
            if not self.models_trained:
                self._train_ml_models(squeeze_states, momentum, data)
            
            # Generate signal
            signal, confidence = self._generate_squeeze_signal(
                momentum, squeeze_states, squeeze_patterns, momentum_analysis, data
            )
            
            # Update history
            if len(momentum) > 0 and not pd.isna(momentum.iloc[-1]):
                self.history['momentum_values'].append(float(momentum.iloc[-1]))
                self.history['squeeze_states'].append(bool(squeeze_states.iloc[-1]))
                self.history['bb_upper'].append(float(bb_upper.iloc[-1]) if not pd.isna(bb_upper.iloc[-1]) else 0.0)
                self.history['bb_lower'].append(float(bb_lower.iloc[-1]) if not pd.isna(bb_lower.iloc[-1]) else 0.0)
                self.history['kc_upper'].append(float(kc_upper.iloc[-1]) if not pd.isna(kc_upper.iloc[-1]) else 0.0)
                self.history['kc_lower'].append(float(kc_lower.iloc[-1]) if not pd.isna(kc_lower.iloc[-1]) else 0.0)
                
                # Keep history limited
                for key in self.history:
                    if len(self.history[key]) > 100:
                        self.history[key] = self.history[key][-100:]
            
            result = {
                'momentum': float(momentum.iloc[-1]) if len(momentum) > 0 and not pd.isna(momentum.iloc[-1]) else 0.0,
                'squeeze': bool(squeeze_states.iloc[-1]) if len(squeeze_states) > 0 else False,
                'breakout': bool(breakout_states.iloc[-1]) if len(breakout_states) > 0 else False,
                'signal': signal,
                'confidence': confidence,
                'bollinger_bands': {
                    'upper': float(bb_upper.iloc[-1]) if len(bb_upper) > 0 and not pd.isna(bb_upper.iloc[-1]) else 0.0,
                    'lower': float(bb_lower.iloc[-1]) if len(bb_lower) > 0 and not pd.isna(bb_lower.iloc[-1]) else 0.0,
                    'middle': float(bb_mid.iloc[-1]) if len(bb_mid) > 0 and not pd.isna(bb_mid.iloc[-1]) else 0.0
                },
                'keltner_channels': {
                    'upper': float(kc_upper.iloc[-1]) if len(kc_upper) > 0 and not pd.isna(kc_upper.iloc[-1]) else 0.0,
                    'lower': float(kc_lower.iloc[-1]) if len(kc_lower) > 0 and not pd.isna(kc_lower.iloc[-1]) else 0.0,
                    'middle': float(kc_mid.iloc[-1]) if len(kc_mid) > 0 and not pd.isna(kc_mid.iloc[-1]) else 0.0
                },
                'squeeze_patterns': squeeze_patterns,
                'momentum_analysis': momentum_analysis,
                'adaptive_periods': {'bb_period': bb_period, 'kc_period': kc_period},
                'market_regime': self._classify_market_regime(squeeze_states, momentum, data),
                'values_history': {
                    'momentum': momentum.tail(30).tolist(),
                    'squeeze': squeeze_states.tail(30).tolist()
                }
            }
            
            return result
            
        except Exception as e:
            raise IndicatorCalculationException(
                indicator_name=self.name,
                calculation_step="raw_calculation",
                message=f"Failed to calculate Squeeze Momentum: {str(e)}",
                cause=e
            )
    
    def _generate_squeeze_signal(self, momentum: pd.Series, squeeze_states: pd.Series,
                                squeeze_patterns: Dict, momentum_analysis: Dict,
                                data: pd.DataFrame) -> Tuple[SignalType, float]:
        """Generate comprehensive squeeze momentum signal"""
        signal_components = []
        confidence_components = []
        
        current_momentum = momentum.iloc[-1] if len(momentum) > 0 and not pd.isna(momentum.iloc[-1]) else 0
        current_squeeze = squeeze_states.iloc[-1] if len(squeeze_states) > 0 else False
        
        # Momentum-based signals
        momentum_threshold = momentum.tail(30).std() * 0.5 if len(momentum) >= 30 else abs(current_momentum) * 0.5
        
        if current_momentum > momentum_threshold:
            signal_components.append(0.7)
            confidence_components.append(0.7)
        elif current_momentum < -momentum_threshold:
            signal_components.append(-0.7)
            confidence_components.append(0.7)
        
        # Squeeze state signals
        if current_squeeze:
            # In squeeze - look for momentum buildup
            squeeze_phase = squeeze_patterns.get('phase', 'unknown')
            if squeeze_phase == 'building_bullish':
                signal_components.append(0.6)
                confidence_components.append(0.6)
            elif squeeze_phase == 'building_bearish':
                signal_components.append(-0.6)
                confidence_components.append(0.6)
        else:
            # Not in squeeze - look for breakout direction
            if len(momentum) > 1:
                momentum_change = momentum.iloc[-1] - momentum.iloc[-2]
                if abs(momentum_change) > momentum_threshold * 0.5:
                    breakout_signal = np.sign(momentum_change) * 0.8
                    signal_components.append(breakout_signal)
                    confidence_components.append(0.8)
        
        # Squeeze pattern signals
        pattern = squeeze_patterns.get('pattern', 'unknown')
        if pattern == 'accumulation' and not current_squeeze:
            signal_components.append(0.5)
            confidence_components.append(0.6)
        elif pattern == 'distribution' and not current_squeeze:
            signal_components.append(-0.5)
            confidence_components.append(0.6)
        
        # Multi-timeframe momentum signals
        if momentum_analysis['convergence']:
            trend = momentum_analysis['short_trend']
            if trend == 'bullish':
                signal_components.append(0.6)
                confidence_components.append(0.7)
            elif trend == 'bearish':
                signal_components.append(-0.6)
                confidence_components.append(0.7)
        
        # Color change signals (momentum direction change)
        if len(momentum) > 2:
            prev_momentum = momentum.iloc[-2]
            if (prev_momentum <= 0 and current_momentum > 0) or \
               (prev_momentum < current_momentum and current_momentum > 0):
                signal_components.append(0.5)
                confidence_components.append(0.6)
            elif (prev_momentum >= 0 and current_momentum < 0) or \
                 (prev_momentum > current_momentum and current_momentum < 0):
                signal_components.append(-0.5)
                confidence_components.append(0.6)
        
        # ML enhancement
        if self.models_trained:
            try:
                ml_signal, ml_confidence = self._get_ml_signal(squeeze_states, momentum, data)
                if ml_signal and ml_confidence > 0.5:
                    signal_components.append(ml_signal)
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
        
        # Boost confidence during squeeze periods (higher reliability)
        if current_squeeze and avg_confidence > 0:
            avg_confidence *= 1.2
        
        if weighted_signal > 0.6:
            signal = SignalType.STRONG_BUY if weighted_signal > 0.8 else SignalType.BUY
        elif weighted_signal < -0.6:
            signal = SignalType.STRONG_SELL if weighted_signal < -0.8 else SignalType.SELL
        else:
            signal = SignalType.NEUTRAL
        
        return signal, min(avg_confidence, 1.0)
    
    def _get_ml_signal(self, squeeze_states: pd.Series, momentum: pd.Series, 
                      data: pd.DataFrame) -> Tuple[float, float]:
        """Get ML-based signal prediction"""
        try:
            lookback = 15
            if len(squeeze_states) < lookback or len(momentum) < lookback:
                return 0.0, 0.0
            
            squeeze_window = squeeze_states.tail(lookback).values
            momentum_window = momentum.tail(lookback).values
            price_window = data['close'].tail(lookback).values
            volume_window = data['volume'].tail(lookback).values if 'volume' in data.columns else np.ones(lookback)
            
            # Recreate feature vector
            squeeze_ratio = np.sum(squeeze_window) / len(squeeze_window)
            squeeze_changes = np.sum(np.diff(squeeze_window.astype(int)))
            
            momentum_mean = np.mean(momentum_window)
            momentum_std = np.std(momentum_window)
            momentum_trend = np.polyfit(range(len(momentum_window)), momentum_window, 1)[0]
            momentum_current = momentum_window[-1]
            
            price_volatility = np.std(np.diff(price_window) / price_window[:-1])
            price_trend = np.polyfit(range(len(price_window)), price_window, 1)[0]
            
            volume_trend = np.polyfit(range(len(volume_window)), volume_window, 1)[0]
            volume_ratio = volume_window[-1] / (np.mean(volume_window) + 1e-8)
            
            momentum_skew = stats.skew(momentum_window)
            momentum_kurtosis = stats.kurtosis(momentum_window)
            
            feature_vector = np.array([[
                squeeze_ratio, squeeze_changes,
                momentum_mean, momentum_std, momentum_trend, momentum_current,
                price_volatility, price_trend,
                volume_trend, volume_ratio,
                momentum_skew, momentum_kurtosis
            ]])
            
            scaled_features = self.scaler.transform(feature_vector)
            
            # Get predictions
            squeeze_change_prob = self.squeeze_classifier.predict_proba(scaled_features)[0]
            momentum_prediction = self.momentum_predictor.predict(scaled_features)[0]
            
            # Combine predictions
            if len(squeeze_change_prob) >= 2:
                change_prob = squeeze_change_prob[1]  # Probability of squeeze change
                
                # Signal strength based on momentum prediction and squeeze change probability
                signal_strength = np.tanh(momentum_prediction * 2)  # Normalize
                confidence = (change_prob + min(abs(momentum_prediction), 1.0)) / 2
                
                return signal_strength, confidence
                
        except:
            pass
        
        return 0.0, 0.0
    
    def _classify_market_regime(self, squeeze_states: pd.Series, momentum: pd.Series, 
                               data: pd.DataFrame) -> Dict[str, Any]:
        """Classify market regime based on squeeze characteristics"""
        if len(squeeze_states) < 20 or len(momentum) < 20:
            return {'regime': 'unknown', 'volatility': 'undefined', 'compression': 'unknown'}
        
        recent_squeeze = squeeze_states.tail(20)
        recent_momentum = momentum.tail(20)
        
        # Squeeze frequency
        squeeze_ratio = np.sum(recent_squeeze) / len(recent_squeeze)
        
        # Momentum characteristics
        momentum_volatility = recent_momentum.std()
        momentum_trend = np.polyfit(range(len(recent_momentum)), recent_momentum.values, 1)[0]
        
        # Volatility classification
        if len(data) >= 30:
            price_volatility = data['close'].pct_change().tail(30).std()
            if price_volatility > data['close'].pct_change().tail(60).std() * 1.5:
                volatility = 'high'
            elif price_volatility < data['close'].pct_change().tail(60).std() * 0.7:
                volatility = 'low'
            else:
                volatility = 'normal'
        else:
            volatility = 'unknown'
        
        # Compression state
        if squeeze_ratio > 0.7:
            compression = 'high_compression'
        elif squeeze_ratio > 0.3:
            compression = 'moderate_compression'
        else:
            compression = 'low_compression'
        
        # Overall regime
        if compression == 'high_compression' and momentum_volatility < recent_momentum.mean():
            regime = 'pre_breakout'
        elif compression == 'low_compression' and volatility == 'high':
            regime = 'active_breakout'
        elif squeeze_ratio < 0.2 and abs(momentum_trend) > momentum_volatility:
            regime = 'trending'
        else:
            regime = 'consolidation'
        
        return {
            'regime': regime,
            'volatility': volatility,
            'compression': compression,
            'squeeze_ratio': float(squeeze_ratio),
            'momentum_volatility': float(momentum_volatility),
            'momentum_trend': float(momentum_trend)
        }
    
    def _generate_signal(self, value: Any, data: pd.DataFrame) -> Tuple[Optional[SignalType], float]:
        if isinstance(value, dict) and 'signal' in value and 'confidence' in value:
            return value['signal'], value['confidence']
        return SignalType.NEUTRAL, 0.0
    
    def _get_metadata(self, data: pd.DataFrame) -> Dict[str, Any]:
        base_metadata = super()._get_metadata(data)
        base_metadata.update({
            'indicator_type': 'momentum',
            'indicator_category': 'squeeze_momentum',
            'models_trained': self.models_trained,
            'adaptive_periods': self.parameters['adaptive_periods'],
            'volume_weighted': self.parameters['volume_weighted'],
            'multi_timeframe': self.parameters['multi_timeframe'],
            'pattern_detection': self.parameters['pattern_detection'],
            'data_type': 'ohlcv',
            'complexity': 'advanced'
        })
        return base_metadata