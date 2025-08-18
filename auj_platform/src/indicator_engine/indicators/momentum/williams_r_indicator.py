"""
Williams %R Indicator - Advanced Implementation
==============================================

Advanced Williams %R with ML-enhanced pattern recognition,
adaptive parameters, and sophisticated momentum analysis.

Author: AUJ Platform Development Team
Mission: Building advanced trading indicators for humanitarian impact
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from ..base.standard_indicator import (
    StandardIndicatorInterface, 
    DataRequirement, 
    DataType, 
    SignalType
)
from ....core.exceptions import IndicatorCalculationException


class WilliamsRIndicator(StandardIndicatorInterface):
    """
    Advanced Williams %R Indicator Implementation
    
    Features:
    - Adaptive period calculation based on market volatility
    - ML-enhanced pattern recognition for reversal signals
    - Multi-timeframe analysis for trend confirmation
    - Volume-weighted %R variants
    - Advanced divergence detection with statistical validation
    - Momentum acceleration analysis
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'period': 14,
            'overbought': -20,  # Williams %R uses negative values
            'oversold': -80,
            'adaptive_period': True,
            'volume_weighted': False,
            'multi_timeframe': True,
            'divergence_lookback': 20,
            'ml_lookback': 50,
            'momentum_periods': [5, 10, 20]
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__(name="WilliamsRIndicator", parameters=default_params)
        
        self.scaler = StandardScaler()
        self.pattern_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.regime_classifier = KMeans(n_clusters=3, random_state=42)
        self.models_trained = False
        
        self.history = {
            'williams_r': [],
            'momentum': [],
            'regime': [],
            'patterns': []
        }
    
    def get_data_requirements(self) -> DataRequirement:
        return DataRequirement(
            data_type=DataType.OHLCV,
            required_columns=['high', 'low', 'close', 'volume'],
            min_periods=max(self.parameters['period'], self.parameters['ml_lookback']) + 20,
            lookback_periods=120
        )
    
    def _calculate_williams_r(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Williams %R with optional adaptations"""
        period = self.parameters['period']
        
        if self.parameters['adaptive_period']:
            # Adapt period based on volatility regime
            volatility = data['close'].pct_change().rolling(window=20).std()
            vol_ratio = volatility / volatility.rolling(window=60).mean()
            adaptive_period = max(5, int(period * vol_ratio.iloc[-1]))
        else:
            adaptive_period = period
        
        if self.parameters['volume_weighted']:
            # Volume-weighted high and low
            volume = data['volume']
            high_vw = (data['high'] * volume).rolling(window=adaptive_period).sum() / volume.rolling(window=adaptive_period).sum()
            low_vw = (data['low'] * volume).rolling(window=adaptive_period).sum() / volume.rolling(window=adaptive_period).sum()
            
            highest_high = high_vw.rolling(window=adaptive_period).max()
            lowest_low = low_vw.rolling(window=adaptive_period).min()
        else:
            # Traditional calculation
            highest_high = data['high'].rolling(window=adaptive_period).max()
            lowest_low = data['low'].rolling(window=adaptive_period).min()
        
        # Williams %R formula: (Highest High - Close) / (Highest High - Lowest Low) * -100
        williams_r = ((highest_high - data['close']) / (highest_high - lowest_low + 1e-8)) * -100
        
        return williams_r
    
    def _detect_extremes_and_reversals(self, williams_r: pd.Series) -> Dict[str, Any]:
        """Detect extreme levels and potential reversal signals"""
        current_wr = williams_r.iloc[-1]
        overbought = self.parameters['overbought']
        oversold = self.parameters['oversold']
        
        # Extreme level analysis
        if current_wr > overbought:
            extreme_type = 'overbought'
            extreme_intensity = (current_wr - overbought) / (0 - overbought)  # 0 is the upper bound
        elif current_wr < oversold:
            extreme_type = 'oversold'
            extreme_intensity = (oversold - current_wr) / (oversold - (-100))  # -100 is the lower bound
        else:
            extreme_type = 'none'
            extreme_intensity = 0.0
        
        # Reversal signal detection
        reversal_signals = []
        
        if len(williams_r) >= 5:
            recent_wr = williams_r.tail(5)
            
            # Bullish reversal: coming out of oversold
            if (current_wr > oversold and 
                recent_wr.iloc[-2] <= oversold and
                recent_wr.diff().tail(3).mean() > 0):
                reversal_signals.append({
                    'type': 'bullish_reversal',
                    'strength': min(abs(current_wr - oversold) / 20, 1.0),
                    'confidence': 0.7
                })
            
            # Bearish reversal: coming out of overbought
            if (current_wr < overbought and 
                recent_wr.iloc[-2] >= overbought and
                recent_wr.diff().tail(3).mean() < 0):
                reversal_signals.append({
                    'type': 'bearish_reversal',
                    'strength': min(abs(current_wr - overbought) / 20, 1.0),
                    'confidence': 0.7
                })
            
            # Momentum reversal patterns
            wr_momentum = recent_wr.diff()
            if len(wr_momentum.dropna()) >= 3:
                momentum_values = wr_momentum.dropna().tail(3).values
                
                # Bullish momentum shift (acceleration from negative to positive)
                if (momentum_values[-1] > momentum_values[-2] > momentum_values[-3] and
                    momentum_values[-1] > 0):
                    reversal_signals.append({
                        'type': 'bullish_acceleration',
                        'strength': min(momentum_values[-1] / 10, 1.0),
                        'confidence': 0.5
                    })
                
                # Bearish momentum shift
                elif (momentum_values[-1] < momentum_values[-2] < momentum_values[-3] and
                      momentum_values[-1] < 0):
                    reversal_signals.append({
                        'type': 'bearish_acceleration',
                        'strength': min(abs(momentum_values[-1]) / 10, 1.0),
                        'confidence': 0.5
                    })
        
        return {
            'extreme_type': extreme_type,
            'extreme_intensity': float(extreme_intensity),
            'reversal_signals': reversal_signals
        }
    
    def _multi_timeframe_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform multi-timeframe Williams %R analysis"""
        if not self.parameters['multi_timeframe']:
            return {'short': None, 'medium': None, 'long': None, 'consensus': 'neutral'}
        
        timeframes = [7, 14, 28]
        results = {}
        signals = []
        
        for i, period in enumerate(timeframes):
            try:
                # Temporarily adjust period
                original_period = self.parameters['period']
                self.parameters['period'] = period
                
                # Calculate Williams %R for this timeframe
                wr = self._calculate_williams_r(data)
                
                # Restore original period
                self.parameters['period'] = original_period
                
                if len(wr) > 0:
                    current_wr = wr.iloc[-1]
                    timeframe_name = ['short', 'medium', 'long'][i]
                    
                    # Determine signal
                    if current_wr > self.parameters['overbought']:
                        signal = 'overbought'
                        signals.append(-1)
                    elif current_wr < self.parameters['oversold']:
                        signal = 'oversold'
                        signals.append(1)
                    elif current_wr > -50:  # Above midline
                        signal = 'bullish_bias'
                        signals.append(0.5)
                    else:  # Below midline
                        signal = 'bearish_bias'
                        signals.append(-0.5)
                    
                    results[timeframe_name] = {
                        'williams_r': float(current_wr),
                        'signal': signal,
                        'period': period
                    }
            except:
                results[['short', 'medium', 'long'][i]] = None
        
        # Calculate consensus
        if signals:
            avg_signal = np.mean(signals)
            if avg_signal > 0.3:
                consensus = 'bullish'
            elif avg_signal < -0.3:
                consensus = 'bearish'
            else:
                consensus = 'neutral'
        else:
            consensus = 'neutral'
        
        results['consensus'] = consensus
        return results
    
    def _detect_divergences(self, williams_r: pd.Series, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect Williams %R - Price divergences"""
        lookback = self.parameters['divergence_lookback']
        if len(williams_r) < lookback:
            return {'bullish': False, 'bearish': False, 'strength': 0.0, 'confidence': 0.0}
        
        recent_wr = williams_r.tail(lookback)
        recent_prices = data['close'].tail(lookback)
        
        # Find local peaks and troughs
        from scipy.signal import find_peaks
        
        wr_peaks, _ = find_peaks(recent_wr.values, distance=3, prominence=5)
        wr_troughs, _ = find_peaks(-recent_wr.values, distance=3, prominence=5)
        
        price_peaks, _ = find_peaks(recent_prices.values, distance=3)
        price_troughs, _ = find_peaks(-recent_prices.values, distance=3)
        
        bullish_divergence = False
        bearish_divergence = False
        divergence_strength = 0.0
        
        # Bullish divergence: price lower low, Williams %R higher low
        if len(wr_troughs) >= 2 and len(price_troughs) >= 2:
            last_wr_trough = recent_wr.iloc[wr_troughs[-1]]
            prev_wr_trough = recent_wr.iloc[wr_troughs[-2]]
            
            last_price_trough = recent_prices.iloc[price_troughs[-1]]
            prev_price_trough = recent_prices.iloc[price_troughs[-2]]
            
            if (last_price_trough < prev_price_trough and 
                last_wr_trough > prev_wr_trough and
                last_wr_trough < -60):  # Confirm in oversold region
                bullish_divergence = True
                price_decline = abs(last_price_trough - prev_price_trough) / prev_price_trough
                wr_improvement = abs(last_wr_trough - prev_wr_trough) / abs(prev_wr_trough) if prev_wr_trough != 0 else 0
                divergence_strength = min(wr_improvement / (price_decline + 1e-8), 2.0)
        
        # Bearish divergence: price higher high, Williams %R lower high
        if len(wr_peaks) >= 2 and len(price_peaks) >= 2:
            last_wr_peak = recent_wr.iloc[wr_peaks[-1]]
            prev_wr_peak = recent_wr.iloc[wr_peaks[-2]]
            
            last_price_peak = recent_prices.iloc[price_peaks[-1]]
            prev_price_peak = recent_prices.iloc[price_peaks[-2]]
            
            if (last_price_peak > prev_price_peak and 
                last_wr_peak < prev_wr_peak and
                last_wr_peak > -40):  # Confirm in overbought region
                bearish_divergence = True
                price_increase = abs(last_price_peak - prev_price_peak) / prev_price_peak
                wr_decline = abs(last_wr_peak - prev_wr_peak) / abs(prev_wr_peak) if prev_wr_peak != 0 else 0
                divergence_strength = min(wr_decline / (price_increase + 1e-8), 2.0)
        
        # Statistical validation
        if bullish_divergence or bearish_divergence:
            correlation = np.corrcoef(recent_wr.values, recent_prices.values)[0, 1]
            confidence = max(0, 1 - abs(correlation)) * min(divergence_strength, 1.0)
        else:
            confidence = 0.0
        
        return {
            'bullish': bullish_divergence,
            'bearish': bearish_divergence,
            'strength': float(divergence_strength),
            'confidence': float(confidence)
        }
    
    def _analyze_momentum(self, williams_r: pd.Series) -> Dict[str, Any]:
        """Analyze Williams %R momentum across multiple timeframes"""
        momentum_analysis = {}
        
        for period in self.parameters['momentum_periods']:
            if len(williams_r) >= period:
                wr_momentum = williams_r.diff(period)
                current_momentum = wr_momentum.iloc[-1]
                
                # Classify momentum
                if current_momentum > 5:
                    momentum_direction = 'strong_bullish'
                elif current_momentum > 1:
                    momentum_direction = 'weak_bullish'
                elif current_momentum < -5:
                    momentum_direction = 'strong_bearish'
                elif current_momentum < -1:
                    momentum_direction = 'weak_bearish'
                else:
                    momentum_direction = 'neutral'
                
                momentum_analysis[f'momentum_{period}'] = {
                    'value': float(current_momentum),
                    'direction': momentum_direction
                }
        
        # Calculate momentum acceleration
        if len(williams_r) >= 6:
            short_momentum = williams_r.diff(2).tail(3).mean()
            long_momentum = williams_r.diff(5).tail(3).mean()
            
            if short_momentum > long_momentum + 2:
                acceleration = 'accelerating_bullish'
            elif short_momentum < long_momentum - 2:
                acceleration = 'accelerating_bearish'
            else:
                acceleration = 'stable'
            
            momentum_analysis['acceleration'] = acceleration
        
        return momentum_analysis
    
    def _classify_market_regime(self, williams_r: pd.Series, data: pd.DataFrame) -> Dict[str, Any]:
        """Classify market regime for Williams %R interpretation"""
        if len(williams_r) < 30:
            return {'regime': 'unknown', 'confidence': 0.0}
        
        try:
            # Prepare features for regime classification
            lookback = 30
            returns = data['close'].pct_change().tail(lookback)
            volatility = returns.rolling(window=5).std()
            wr_values = williams_r.tail(lookback)
            wr_volatility = wr_values.rolling(window=5).std()
            
            features = np.column_stack([
                returns.fillna(0).values,
                volatility.fillna(volatility.mean()).values,
                wr_values.fillna(-50).values,
                wr_volatility.fillna(wr_volatility.mean()).values
            ])
            
            # Fit regime classifier
            if not hasattr(self, 'regime_fitted'):
                if len(features) >= 15:
                    self.regime_classifier.fit(features)
                    self.regime_fitted = True
            
            if hasattr(self, 'regime_fitted') and self.regime_fitted:
                current_features = features[-1:].reshape(1, -1)
                regime_label = self.regime_classifier.predict(current_features)[0]
                
                # Interpret regime
                regime_names = ['trending_bullish', 'choppy_sideways', 'trending_bearish']
                regime = regime_names[regime_label % 3]
                
                # Calculate confidence
                cluster_center = self.regime_classifier.cluster_centers_[regime_label]
                distance = np.linalg.norm(current_features - cluster_center)
                max_distance = np.max([np.linalg.norm(center - cluster_center) 
                                     for center in self.regime_classifier.cluster_centers_])
                confidence = max(0, 1 - distance / (max_distance + 1e-8))
                
                return {
                    'regime': regime,
                    'confidence': float(confidence)
                }
        except:
            pass
        
        return {'regime': 'unknown', 'confidence': 0.0}
    
    def _train_ml_models(self, williams_r: pd.Series, data: pd.DataFrame) -> bool:
        """Train ML models for pattern recognition"""
        if len(williams_r) < self.parameters['ml_lookback'] * 2:
            return False
        
        try:
            features, targets = self._prepare_ml_data(williams_r, data)
            if len(features) > 30:
                self.scaler.fit(features)
                scaled_features = self.scaler.transform(features)
                self.pattern_classifier.fit(scaled_features, targets)
                self.models_trained = True
                return True
        except:
            pass
        return False
    
    def _prepare_ml_data(self, williams_r: pd.Series, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare ML training data"""
        features, targets = [], []
        lookback = 10
        
        for i in range(lookback, len(williams_r) - 5):
            wr_window = williams_r.iloc[i-lookback:i].values
            price_window = data['close'].iloc[i-lookback:i].values
            volume_window = data['volume'].iloc[i-lookback:i].values
            
            feature_vector = [
                np.mean(wr_window), np.std(wr_window), wr_window[-1],
                len([x for x in wr_window if x > -20]) / len(wr_window),  # Overbought ratio
                len([x for x in wr_window if x < -80]) / len(wr_window),  # Oversold ratio
                wr_window[-1] - wr_window[0],  # Williams %R trend
                np.corrcoef(wr_window, price_window)[0, 1] if len(set(wr_window)) > 1 else 0,
                volume_window[-1] / np.mean(volume_window) if np.mean(volume_window) != 0 else 1,
                np.max(wr_window) - np.min(wr_window),  # Williams %R range
                len([j for j in range(1, len(wr_window)) if wr_window[j] > wr_window[j-1]]) / (len(wr_window) - 1)  # Rising ratio
            ]
            
            # Target: future price movement
            future_return = (data['close'].iloc[i+5] - data['close'].iloc[i]) / data['close'].iloc[i]
            if future_return > 0.015:
                target = 2  # Strong bullish
            elif future_return > 0.005:
                target = 1  # Weak bullish
            elif future_return < -0.015:
                target = 0  # Strong bearish
            else:
                target = 1  # Neutral
            
            features.append(feature_vector)
            targets.append(target)
        
        return np.array(features), np.array(targets)
    
    def calculate_raw(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate Williams %R with advanced analysis"""
        try:
            # Calculate Williams %R
            williams_r = self._calculate_williams_r(data)
            
            # Advanced analysis
            extremes_reversals = self._detect_extremes_and_reversals(williams_r)
            multi_timeframe = self._multi_timeframe_analysis(data)
            divergences = self._detect_divergences(williams_r, data)
            momentum_analysis = self._analyze_momentum(williams_r)
            regime = self._classify_market_regime(williams_r, data)
            
            # Train ML models
            if not self.models_trained:
                self._train_ml_models(williams_r, data)
            
            # Generate signal
            signal, confidence = self._generate_williams_r_signal(
                williams_r, extremes_reversals, multi_timeframe, 
                divergences, momentum_analysis, regime, data
            )
            
            # Update history
            self.history['williams_r'].append(float(williams_r.iloc[-1]))
            self.history['momentum'].append(momentum_analysis)
            self.history['regime'].append(regime['regime'])
            
            # Keep history limited
            for key in self.history:
                if len(self.history[key]) > 100:
                    self.history[key] = self.history[key][-100:]
            
            result = {
                'williams_r': float(williams_r.iloc[-1]),
                'signal': signal,
                'confidence': confidence,
                'extremes_reversals': extremes_reversals,
                'multi_timeframe': multi_timeframe,
                'divergences': divergences,
                'momentum_analysis': momentum_analysis,
                'regime': regime,
                'position_strength': self._calculate_position_strength(williams_r),
                'values_history': {
                    'williams_r': williams_r.tail(30).tolist()
                }
            }
            
            return result
            
        except Exception as e:
            raise IndicatorCalculationException(
                indicator_name=self.name,
                calculation_step="raw_calculation",
                message=f"Failed to calculate Williams %R: {str(e)}",
                cause=e
            )
    
    def _generate_williams_r_signal(self, williams_r: pd.Series, extremes_reversals: Dict,
                                   multi_timeframe: Dict, divergences: Dict, momentum_analysis: Dict,
                                   regime: Dict, data: pd.DataFrame) -> Tuple[SignalType, float]:
        """Generate comprehensive Williams %R signal"""
        signal_components = []
        confidence_components = []
        
        current_wr = williams_r.iloc[-1]
        
        # Extreme level signals
        if extremes_reversals['extreme_type'] == 'oversold':
            signal_components.append(extremes_reversals['extreme_intensity'])
            confidence_components.append(0.6)
        elif extremes_reversals['extreme_type'] == 'overbought':
            signal_components.append(-extremes_reversals['extreme_intensity'])
            confidence_components.append(0.6)
        
        # Reversal signals
        for reversal in extremes_reversals['reversal_signals']:
            if reversal['type'] in ['bullish_reversal', 'bullish_acceleration']:
                signal_components.append(reversal['strength'])
                confidence_components.append(reversal['confidence'])
            elif reversal['type'] in ['bearish_reversal', 'bearish_acceleration']:
                signal_components.append(-reversal['strength'])
                confidence_components.append(reversal['confidence'])
        
        # Multi-timeframe consensus
        if multi_timeframe['consensus'] == 'bullish':
            signal_components.append(0.7)
            confidence_components.append(0.6)
        elif multi_timeframe['consensus'] == 'bearish':
            signal_components.append(-0.7)
            confidence_components.append(0.6)
        
        # Divergence signals
        if divergences['bullish']:
            signal_components.append(divergences['strength'])
            confidence_components.append(divergences['confidence'])
        elif divergences['bearish']:
            signal_components.append(-divergences['strength'])
            confidence_components.append(divergences['confidence'])
        
        # Momentum signals
        if 'momentum_5' in momentum_analysis:
            momentum_5 = momentum_analysis['momentum_5']
            if momentum_5['direction'] in ['strong_bullish', 'weak_bullish']:
                signal_components.append(0.4 if 'strong' in momentum_5['direction'] else 0.2)
                confidence_components.append(0.4)
            elif momentum_5['direction'] in ['strong_bearish', 'weak_bearish']:
                signal_components.append(-0.4 if 'strong' in momentum_5['direction'] else -0.2)
                confidence_components.append(0.4)
        
        # Regime-based adjustments
        if regime['regime'] == 'trending_bullish' and regime['confidence'] > 0.7:
            # In bullish trends, favor oversold bounces
            if current_wr < -60:
                signal_components.append(0.5)
                confidence_components.append(regime['confidence'])
        elif regime['regime'] == 'trending_bearish' and regime['confidence'] > 0.7:
            # In bearish trends, favor overbought rejections
            if current_wr > -40:
                signal_components.append(-0.5)
                confidence_components.append(regime['confidence'])
        
        # ML enhancement
        if self.models_trained:
            try:
                ml_signal, ml_confidence = self._get_ml_signal(williams_r, data)
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
    
    def _get_ml_signal(self, williams_r: pd.Series, data: pd.DataFrame) -> Tuple[Optional[SignalType], float]:
        """Get ML-based signal prediction"""
        try:
            lookback = 10
            if len(williams_r) < lookback:
                return None, 0.0
            
            wr_window = williams_r.tail(lookback).values
            price_window = data['close'].tail(lookback).values
            volume_window = data['volume'].tail(lookback).values
            
            feature_vector = np.array([[
                np.mean(wr_window), np.std(wr_window), wr_window[-1],
                len([x for x in wr_window if x > -20]) / len(wr_window),
                len([x for x in wr_window if x < -80]) / len(wr_window),
                wr_window[-1] - wr_window[0],
                np.corrcoef(wr_window, price_window)[0, 1] if len(set(wr_window)) > 1 else 0,
                volume_window[-1] / np.mean(volume_window) if np.mean(volume_window) != 0 else 1,
                np.max(wr_window) - np.min(wr_window),
                len([j for j in range(1, len(wr_window)) if wr_window[j] > wr_window[j-1]]) / (len(wr_window) - 1)
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
    
    def _calculate_position_strength(self, williams_r: pd.Series) -> Dict[str, Any]:
        """Calculate position strength based on Williams %R levels"""
        current_wr = williams_r.iloc[-1]
        
        # Position strength relative to range
        if current_wr > -20:
            position = 'extremely_overbought'
            strength = (current_wr + 20) / 20  # 0 to 1 scale
        elif current_wr > -50:
            position = 'bullish_zone'
            strength = (current_wr + 50) / 30
        elif current_wr > -80:
            position = 'bearish_zone'
            strength = (current_wr + 80) / 30
        else:
            position = 'extremely_oversold'
            strength = (80 + current_wr) / 20
        
        return {
            'position': position,
            'strength': float(max(0, min(1, strength))),
            'raw_value': float(current_wr)
        }
    
    def _generate_signal(self, value: Any, data: pd.DataFrame) -> Tuple[Optional[SignalType], float]:
        if isinstance(value, dict) and 'signal' in value and 'confidence' in value:
            return value['signal'], value['confidence']
        return SignalType.NEUTRAL, 0.0
    
    def _get_metadata(self, data: pd.DataFrame) -> Dict[str, Any]:
        base_metadata = super()._get_metadata(data)
        base_metadata.update({
            'indicator_type': 'momentum',
            'indicator_category': 'williams_r',
            'models_trained': self.models_trained,
            'adaptive_period': self.parameters['adaptive_period'],
            'volume_weighted': self.parameters['volume_weighted'],
            'data_type': 'ohlcv',
            'complexity': 'advanced'
        })
        return base_metadata