"""
Fisher Transform Indicator - Advanced Implementation
===================================================

Advanced Fisher Transform implementation with ML-enhanced pattern detection,
regime classification, and sophisticated price normalization algorithms.

Author: AUJ Platform Development Team
Mission: Building advanced trading indicators for humanitarian impact
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler
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


class FisherTransformIndicator(StandardIndicatorInterface):
    """
    Advanced Fisher Transform Indicator Implementation
    
    Features:
    - Sophisticated price normalization with adaptive parameters
    - ML-based pattern recognition and trend classification
    - Multi-period Fisher Transform analysis
    - Statistical regime detection using Gaussian Mixture Models
    - Advanced smoothing algorithms
    - Extrema detection with statistical significance
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'period': 10,
            'smooth_period': 3,
            'adaptive_normalization': True,
            'ml_lookback': 50,
            'regime_periods': 30,
            'significance_threshold': 1.5,
            'multi_period_analysis': True
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__(name="FisherTransformIndicator", parameters=default_params)
        
        self.scaler = RobustScaler()
        self.ml_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.regime_model = GaussianMixture(n_components=3, random_state=42)
        self.ml_trained = False
        self.regime_trained = False
        
        self.history = {
            'fisher': [],
            'trigger': [],
            'normalized_hl2': [],
            'extrema': []
        }
    
    def get_data_requirements(self) -> DataRequirement:
        return DataRequirement(
            data_type=DataType.OHLCV,
            required_columns=['high', 'low', 'close', 'volume'],
            min_periods=max(self.parameters['period'], self.parameters['ml_lookback']) + 20,
            lookback_periods=150
        )
    
    def _calculate_hl2_normalized(self, data: pd.DataFrame) -> pd.Series:
        """Calculate normalized HL2 with adaptive parameters"""
        high = data['high']
        low = data['low']
        hl2 = (high + low) / 2
        
        period = self.parameters['period']
        
        if self.parameters['adaptive_normalization']:
            # Adaptive period based on volatility
            volatility = data['close'].pct_change().rolling(window=10).std()
            vol_ratio = volatility / volatility.rolling(window=30).mean()
            adaptive_period = max(5, int(period * vol_ratio.iloc[-1]))
        else:
            adaptive_period = period
        
        # Calculate rolling min/max for normalization
        min_low = low.rolling(window=adaptive_period).min()
        max_high = high.rolling(window=adaptive_period).max()
        
        # Normalize to [-1, 1] range with epsilon to avoid division by zero
        range_hl = max_high - min_low
        normalized = 2 * ((hl2 - min_low) / (range_hl + 1e-8)) - 1
        
        # Clamp to valid range to avoid numerical issues
        normalized = np.clip(normalized, -0.999, 0.999)
        
        return normalized
    
    def _calculate_fisher_transform(self, normalized_series: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Calculate Fisher Transform and its trigger line"""
        # Apply Fisher Transform formula
        fisher_raw = 0.5 * np.log((1 + normalized_series) / (1 - normalized_series))
        
        # Smooth Fisher Transform
        smooth_period = self.parameters['smooth_period']
        fisher = fisher_raw.ewm(span=smooth_period).mean()
        
        # Calculate trigger line (previous Fisher value)
        trigger = fisher.shift(1)
        
        return fisher, trigger
    
    def _detect_extrema(self, fisher: pd.Series) -> Dict[str, Any]:
        """Detect statistical extrema in Fisher Transform"""
        if len(fisher) < 20:
            return {'type': 'none', 'significance': 0.0, 'level': 0.0}
        
        threshold = self.parameters['significance_threshold']
        current_value = fisher.iloc[-1]
        
        # Calculate statistical thresholds
        recent_fisher = fisher.tail(30)
        mean_level = recent_fisher.mean()
        std_level = recent_fisher.std()
        
        upper_threshold = mean_level + threshold * std_level
        lower_threshold = mean_level - threshold * std_level
        
        # Determine extrema type
        if current_value > upper_threshold:
            significance = min((current_value - upper_threshold) / std_level, 3.0)
            return {'type': 'overbought', 'significance': significance, 'level': current_value}
        elif current_value < lower_threshold:
            significance = min((lower_threshold - current_value) / std_level, 3.0)
            return {'type': 'oversold', 'significance': significance, 'level': current_value}
        else:
            return {'type': 'neutral', 'significance': 0.0, 'level': current_value}
    
    def _multi_period_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform multi-period Fisher Transform analysis"""
        if not self.parameters['multi_period_analysis']:
            return {'short': None, 'medium': None, 'long': None, 'consensus': 'neutral'}
        
        periods = [5, 15, 30]
        results = {}
        signals = []
        
        for i, period in enumerate(periods):
            try:
                # Temporarily adjust period
                original_period = self.parameters['period']
                self.parameters['period'] = period
                
                # Calculate Fisher Transform
                normalized = self._calculate_hl2_normalized(data)
                fisher, trigger = self._calculate_fisher_transform(normalized)
                
                # Determine signal
                if len(fisher) >= 2:
                    current_fisher = fisher.iloc[-1]
                    prev_fisher = fisher.iloc[-2]
                    current_trigger = trigger.iloc[-1] if not pd.isna(trigger.iloc[-1]) else 0
                    
                    if current_fisher > current_trigger and prev_fisher <= current_trigger:
                        signal = 'bullish'
                        signals.append(1)
                    elif current_fisher < current_trigger and prev_fisher >= current_trigger:
                        signal = 'bearish'
                        signals.append(-1)
                    else:
                        signal = 'neutral'
                        signals.append(0)
                    
                    results[['short', 'medium', 'long'][i]] = {
                        'signal': signal,
                        'fisher': current_fisher,
                        'trigger': current_trigger
                    }
                
                # Restore original period
                self.parameters['period'] = original_period
                
            except:
                results[['short', 'medium', 'long'][i]] = None
        
        # Determine consensus
        if signals:
            avg_signal = np.mean(signals)
            if avg_signal > 0.33:
                consensus = 'bullish'
            elif avg_signal < -0.33:
                consensus = 'bearish'
            else:
                consensus = 'neutral'
        else:
            consensus = 'neutral'
        
        results['consensus'] = consensus
        return results
    
    def _detect_regime(self, fisher: pd.Series, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect market regime using Gaussian Mixture Models"""
        if len(fisher) < self.parameters['regime_periods']:
            return {'regime': 'unknown', 'confidence': 0.0, 'characteristics': {}}
        
        try:
            # Prepare features for regime detection
            lookback = self.parameters['regime_periods']
            returns = data['close'].pct_change().tail(lookback)
            volatility = returns.rolling(window=5).std()
            volume_ratio = data['volume'] / data['volume'].rolling(window=10).mean()
            fisher_momentum = fisher.diff()
            
            features = np.column_stack([
                returns.fillna(0).values,
                volatility.fillna(volatility.mean()).values,
                volume_ratio.fillna(1).tail(lookback).values,
                fisher_momentum.fillna(0).tail(lookback).values
            ])
            
            if not self.regime_trained and len(features) >= 20:
                self.regime_model.fit(features)
                self.regime_trained = True
            
            if self.regime_trained:
                current_features = features[-1:].reshape(1, -1)
                regime_proba = self.regime_model.predict_proba(current_features)[0]
                regime_idx = np.argmax(regime_proba)
                confidence = regime_proba[regime_idx]
                
                regime_names = ['low_volatility', 'trending', 'high_volatility']
                regime = regime_names[regime_idx]
                
                # Calculate regime characteristics
                means = self.regime_model.means_[regime_idx]
                characteristics = {
                    'return_level': float(means[0]),
                    'volatility_level': float(means[1]),
                    'volume_activity': float(means[2]),
                    'fisher_momentum': float(means[3])
                }
                
                return {
                    'regime': regime,
                    'confidence': float(confidence),
                    'characteristics': characteristics
                }
        except:
            pass
        
        return {'regime': 'unknown', 'confidence': 0.0, 'characteristics': {}}
    
    def _train_ml_model(self, fisher: pd.Series, trigger: pd.Series, data: pd.DataFrame) -> bool:
        """Train ML model for pattern recognition"""
        if len(fisher) < self.parameters['ml_lookback'] * 2:
            return False
        
        try:
            features, targets = self._prepare_ml_data(fisher, trigger, data)
            if len(features) > 30:
                self.scaler.fit(features)
                scaled_features = self.scaler.transform(features)
                self.ml_model.fit(scaled_features, targets)
                self.ml_trained = True
                return True
        except:
            pass
        return False
    
    def _prepare_ml_data(self, fisher: pd.Series, trigger: pd.Series, 
                        data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare ML training data"""
        features, targets = [], []
        lookback = 10
        
        for i in range(lookback, len(fisher) - 5):
            fisher_window = fisher.iloc[i-lookback:i].values
            trigger_window = trigger.iloc[i-lookback:i].values
            price_window = data['close'].iloc[i-lookback:i].values
            volume_window = data['volume'].iloc[i-lookback:i].values
            
            # Remove NaN values
            valid_indices = ~(np.isnan(fisher_window) | np.isnan(trigger_window))
            if np.sum(valid_indices) < lookback // 2:
                continue
            
            fisher_clean = fisher_window[valid_indices]
            trigger_clean = trigger_window[valid_indices]
            
            feature_vector = [
                np.mean(fisher_clean), np.std(fisher_clean),
                np.mean(trigger_clean), np.std(trigger_clean),
                fisher_clean[-1] if len(fisher_clean) > 0 else 0,
                trigger_clean[-1] if len(trigger_clean) > 0 else 0,
                len([x for x in fisher_clean if x > 0]) / len(fisher_clean) if len(fisher_clean) > 0 else 0.5,
                np.corrcoef(fisher_clean, price_window[valid_indices])[0, 1] if len(set(fisher_clean)) > 1 else 0,
                volume_window[-1] / np.mean(volume_window) if np.mean(volume_window) != 0 else 1,
                np.max(fisher_clean) - np.min(fisher_clean) if len(fisher_clean) > 0 else 0
            ]
            
            # Target: future price movement
            future_return = (data['close'].iloc[i+5] - data['close'].iloc[i]) / data['close'].iloc[i]
            target = 2 if future_return > 0.015 else (0 if future_return < -0.015 else 1)
            
            features.append(feature_vector)
            targets.append(target)
        
        return np.array(features), np.array(targets)
    
    def calculate_raw(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate Fisher Transform with advanced analysis"""
        try:
            # Calculate normalized HL2
            normalized_hl2 = self._calculate_hl2_normalized(data)
            
            # Calculate Fisher Transform
            fisher, trigger = self._calculate_fisher_transform(normalized_hl2)
            
            # Detect extrema
            extrema = self._detect_extrema(fisher)
            
            # Multi-period analysis
            multi_period = self._multi_period_analysis(data)
            
            # Regime detection
            regime = self._detect_regime(fisher, data)
            
            # Train ML model
            if not self.ml_trained:
                self._train_ml_model(fisher, trigger, data)
            
            # Generate signal
            signal, confidence = self._generate_fisher_signal(
                fisher, trigger, extrema, multi_period, regime, data
            )
            
            # Update history
            self.history['fisher'].append(float(fisher.iloc[-1]))
            self.history['trigger'].append(float(trigger.iloc[-1]) if not pd.isna(trigger.iloc[-1]) else 0)
            self.history['normalized_hl2'].append(float(normalized_hl2.iloc[-1]))
            self.history['extrema'].append(extrema)
            
            # Keep history limited
            for key in self.history:
                if len(self.history[key]) > 100:
                    self.history[key] = self.history[key][-100:]
            
            result = {
                'fisher': float(fisher.iloc[-1]),
                'trigger': float(trigger.iloc[-1]) if not pd.isna(trigger.iloc[-1]) else 0,
                'normalized_hl2': float(normalized_hl2.iloc[-1]),
                'signal': signal,
                'confidence': confidence,
                'extrema': extrema,
                'multi_period': multi_period,
                'regime': regime,
                'crossover': self._detect_crossover(fisher, trigger),
                'momentum': self._calculate_momentum(fisher),
                'values_history': {
                    'fisher': fisher.tail(20).tolist(),
                    'trigger': trigger.tail(20).tolist()
                }
            }
            
            return result
            
        except Exception as e:
            raise IndicatorCalculationException(
                indicator_name=self.name,
                calculation_step="raw_calculation",
                message=f"Failed to calculate Fisher Transform: {str(e)}",
                cause=e
            )
    
    def _generate_fisher_signal(self, fisher: pd.Series, trigger: pd.Series,
                               extrema: Dict, multi_period: Dict, regime: Dict,
                               data: pd.DataFrame) -> Tuple[SignalType, float]:
        """Generate comprehensive Fisher Transform signal"""
        signal_components = []
        confidence_components = []
        
        # Basic crossover signal
        if len(fisher) >= 2 and not pd.isna(trigger.iloc[-1]):
            current_fisher = fisher.iloc[-1]
            prev_fisher = fisher.iloc[-2]
            current_trigger = trigger.iloc[-1]
            prev_trigger = trigger.iloc[-2] if not pd.isna(trigger.iloc[-2]) else current_trigger
            
            if current_fisher > current_trigger and prev_fisher <= prev_trigger:
                signal_components.append(1.0)
                confidence_components.append(0.7)
            elif current_fisher < current_trigger and prev_fisher >= prev_trigger:
                signal_components.append(-1.0)
                confidence_components.append(0.7)
        
        # Extrema reversal signals
        if extrema['type'] == 'oversold' and extrema['significance'] > 1.0:
            signal_components.append(extrema['significance'] / 3.0)
            confidence_components.append(min(extrema['significance'] / 2.0, 1.0))
        elif extrema['type'] == 'overbought' and extrema['significance'] > 1.0:
            signal_components.append(-extrema['significance'] / 3.0)
            confidence_components.append(min(extrema['significance'] / 2.0, 1.0))
        
        # Multi-period consensus
        if multi_period['consensus'] == 'bullish':
            signal_components.append(0.8)
            confidence_components.append(0.6)
        elif multi_period['consensus'] == 'bearish':
            signal_components.append(-0.8)
            confidence_components.append(0.6)
        
        # Regime-based adjustments
        if regime['regime'] == 'trending' and regime['confidence'] > 0.7:
            if regime['characteristics'].get('fisher_momentum', 0) > 0:
                signal_components.append(0.5)
                confidence_components.append(regime['confidence'])
            else:
                signal_components.append(-0.5)
                confidence_components.append(regime['confidence'])
        
        # ML enhancement
        if self.ml_trained:
            try:
                ml_signal, ml_confidence = self._get_ml_signal(fisher, trigger, data)
                if ml_signal:
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
        
        if weighted_signal > 0.5:
            signal = SignalType.STRONG_BUY if weighted_signal > 0.8 else SignalType.BUY
        elif weighted_signal < -0.5:
            signal = SignalType.STRONG_SELL if weighted_signal < -0.8 else SignalType.SELL
        else:
            signal = SignalType.NEUTRAL
        
        return signal, min(avg_confidence, 1.0)
    
    def _get_ml_signal(self, fisher: pd.Series, trigger: pd.Series, 
                      data: pd.DataFrame) -> Tuple[Optional[SignalType], float]:
        """Get ML-based signal"""
        try:
            lookback = 10
            if len(fisher) < lookback:
                return None, 0.0
            
            fisher_window = fisher.tail(lookback).values
            trigger_window = trigger.tail(lookback).values
            price_window = data['close'].tail(lookback).values
            volume_window = data['volume'].tail(lookback).values
            
            # Remove NaN values
            valid_indices = ~(np.isnan(fisher_window) | np.isnan(trigger_window))
            if np.sum(valid_indices) < lookback // 2:
                return None, 0.0
            
            fisher_clean = fisher_window[valid_indices]
            trigger_clean = trigger_window[valid_indices]
            
            feature_vector = np.array([[
                np.mean(fisher_clean), np.std(fisher_clean),
                np.mean(trigger_clean), np.std(trigger_clean),
                fisher_clean[-1] if len(fisher_clean) > 0 else 0,
                trigger_clean[-1] if len(trigger_clean) > 0 else 0,
                len([x for x in fisher_clean if x > 0]) / len(fisher_clean) if len(fisher_clean) > 0 else 0.5,
                np.corrcoef(fisher_clean, price_window[valid_indices])[0, 1] if len(set(fisher_clean)) > 1 else 0,
                volume_window[-1] / np.mean(volume_window) if np.mean(volume_window) != 0 else 1,
                np.max(fisher_clean) - np.min(fisher_clean) if len(fisher_clean) > 0 else 0
            ]])
            
            scaled_features = self.scaler.transform(feature_vector)
            ml_proba = self.ml_model.predict_proba(scaled_features)[0]
            
            if len(ml_proba) >= 3:
                if ml_proba[2] > 0.7:  # Strong bullish
                    return SignalType.BUY, ml_proba[2]
                elif ml_proba[0] > 0.7:  # Strong bearish
                    return SignalType.SELL, ml_proba[0]
        except:
            pass
        
        return None, 0.0
    
    def _detect_crossover(self, fisher: pd.Series, trigger: pd.Series) -> Dict[str, Any]:
        """Detect Fisher Transform crossovers"""
        if len(fisher) < 2 or pd.isna(trigger.iloc[-1]):
            return {'type': 'none', 'bars_ago': 0, 'strength': 0.0}
        
        current_fisher = fisher.iloc[-1]
        prev_fisher = fisher.iloc[-2]
        current_trigger = trigger.iloc[-1]
        prev_trigger = trigger.iloc[-2] if not pd.isna(trigger.iloc[-2]) else current_trigger
        
        if current_fisher > current_trigger and prev_fisher <= prev_trigger:
            strength = abs(current_fisher - current_trigger)
            return {'type': 'bullish', 'bars_ago': 0, 'strength': float(strength)}
        elif current_fisher < current_trigger and prev_fisher >= prev_trigger:
            strength = abs(current_fisher - current_trigger)
            return {'type': 'bearish', 'bars_ago': 0, 'strength': float(strength)}
        
        return {'type': 'none', 'bars_ago': 0, 'strength': 0.0}
    
    def _calculate_momentum(self, fisher: pd.Series) -> Dict[str, Any]:
        """Calculate Fisher Transform momentum"""
        if len(fisher) < 5:
            return {'direction': 'neutral', 'strength': 0.0, 'acceleration': 0.0}
        
        momentum = fisher.diff().tail(3).mean()
        acceleration = fisher.diff().diff().tail(2).mean()
        
        if momentum > 0.01:
            direction = 'bullish'
        elif momentum < -0.01:
            direction = 'bearish'
        else:
            direction = 'neutral'
        
        return {
            'direction': direction,
            'strength': float(abs(momentum)),
            'acceleration': float(acceleration)
        }
    
    def _generate_signal(self, value: Any, data: pd.DataFrame) -> Tuple[Optional[SignalType], float]:
        if isinstance(value, dict) and 'signal' in value and 'confidence' in value:
            return value['signal'], value['confidence']
        return SignalType.NEUTRAL, 0.0
    
    def _get_metadata(self, data: pd.DataFrame) -> Dict[str, Any]:
        base_metadata = super()._get_metadata(data)
        base_metadata.update({
            'indicator_type': 'momentum',
            'indicator_category': 'fisher_transform',
            'ml_trained': self.ml_trained,
            'regime_trained': self.regime_trained,
            'data_type': 'ohlcv',
            'complexity': 'advanced'
        })
        return base_metadata