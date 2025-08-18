"""
Rate of Change (ROC) Indicator - Advanced Implementation
=======================================================

Advanced ROC implementation with ML-enhanced trend analysis,
multi-period momentum detection, and sophisticated signal generation.

Author: AUJ Platform Development Team
Mission: Building advanced trading indicators for humanitarian impact
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from sklearn.ensemble import RandomForestRegressor, ExtraTreesClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from scipy import stats
from scipy.signal import butter, filtfilt
import warnings
warnings.filterwarnings('ignore')

from ..base.standard_indicator import (
    StandardIndicatorInterface, 
    DataRequirement, 
    DataType, 
    SignalType
)
from ....core.exceptions import IndicatorCalculationException


class RateOfChangeIndicator(StandardIndicatorInterface):
    """
    Advanced Rate of Change (ROC) Indicator Implementation
    
    Features:
    - Multi-period ROC analysis with adaptive periods
    - ML-enhanced momentum classification and trend prediction
    - Advanced signal filtering using digital signal processing
    - Momentum acceleration and deceleration detection
    - Volume-weighted ROC variants for institutional flow analysis
    - Statistical significance testing for ROC levels
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'period': 14,
            'periods': [5, 10, 14, 21, 30],  # Multi-period analysis
            'smoothing_period': 3,
            'volume_weighted': False,
            'adaptive_periods': True,
            'signal_filtering': True,
            'ml_lookback': 50,
            'significance_threshold': 2.0,  # Standard deviations for significance
            'momentum_acceleration': True
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__(name="RateOfChangeIndicator", parameters=default_params)
        
        self.scaler = MinMaxScaler()
        self.momentum_classifier = ExtraTreesClassifier(n_estimators=100, random_state=42)
        self.trend_predictor = RandomForestRegressor(n_estimators=50, random_state=42)
        self.pca = PCA(n_components=3)
        self.models_trained = False
        
        self.history = {
            'roc_values': {},
            'momentum_states': [],
            'acceleration': [],
            'significance_levels': []
        }
        
        # Initialize ROC history for each period
        for period in self.parameters['periods']:
            self.history['roc_values'][period] = []
    
    def get_data_requirements(self) -> DataRequirement:
        max_period = max(self.parameters['periods']) if self.parameters['periods'] else self.parameters['period']
        return DataRequirement(
            data_type=DataType.OHLCV,
            required_columns=['close', 'volume'],
            min_periods=max(max_period, self.parameters['ml_lookback']) + 30,
            lookback_periods=150
        )
    
    def _calculate_roc(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate ROC for a specific period with optional volume weighting"""
        if self.parameters['volume_weighted']:
            # Volume-weighted price calculation
            volume = data['volume']
            vwap = (data['close'] * volume).rolling(window=min(5, period)).sum() / volume.rolling(window=min(5, period)).sum()
            prices = vwap.fillna(data['close'])
        else:
            prices = data['close']
        
        if self.parameters['adaptive_periods']:
            # Adapt period based on volatility
            volatility = prices.pct_change().rolling(window=20).std()
            vol_ratio = volatility / volatility.rolling(window=60).mean()
            adaptive_period = max(3, int(period * vol_ratio.iloc[-1]) if not pd.isna(vol_ratio.iloc[-1]) else period)
        else:
            adaptive_period = period
        
        # ROC calculation: ((Current Price - Price n periods ago) / Price n periods ago) * 100
        roc = ((prices - prices.shift(adaptive_period)) / prices.shift(adaptive_period)) * 100
        
        # Apply smoothing if requested
        if self.parameters['smoothing_period'] > 1:
            roc = roc.rolling(window=self.parameters['smoothing_period']).mean()
        
        return roc
    
    def _apply_signal_filtering(self, roc_series: pd.Series) -> pd.Series:
        """Apply digital signal processing for noise reduction"""
        if not self.parameters['signal_filtering'] or len(roc_series) < 10:
            return roc_series
        
        try:
            # Design Butterworth lowpass filter
            nyquist = 0.5  # Assuming daily data
            normal_cutoff = 0.1  # Cutoff frequency
            b, a = butter(2, normal_cutoff, btype='low', analog=False)
            
            # Apply filter to ROC values (forward and backward to avoid phase shift)
            roc_values = roc_series.dropna().values
            if len(roc_values) >= 6:  # Minimum length for filtfilt
                filtered_values = filtfilt(b, a, roc_values)
                filtered_series = pd.Series(filtered_values, index=roc_series.dropna().index)
                return roc_series.fillna(method='bfill').combine_first(filtered_series)
        except:
            pass
        
        return roc_series
    
    def _analyze_momentum_acceleration(self, roc_values: Dict[int, pd.Series]) -> Dict[str, Any]:
        """Analyze momentum acceleration across different periods"""
        if not self.parameters['momentum_acceleration']:
            return {'acceleration': 'unknown', 'strength': 0.0, 'consistency': 0.0}
        
        accelerations = []
        
        for period in sorted(roc_values.keys()):
            roc = roc_values[period]
            if len(roc) >= 3:
                # Calculate acceleration (second derivative)
                velocity = roc.diff()
                acceleration = velocity.diff()
                
                if not pd.isna(acceleration.iloc[-1]):
                    accelerations.append({
                        'period': period,
                        'acceleration': acceleration.iloc[-1],
                        'velocity': velocity.iloc[-1],
                        'roc': roc.iloc[-1]
                    })
        
        if not accelerations:
            return {'acceleration': 'unknown', 'strength': 0.0, 'consistency': 0.0}
        
        # Analyze acceleration patterns
        acc_values = [a['acceleration'] for a in accelerations]
        vel_values = [a['velocity'] for a in accelerations]
        
        avg_acceleration = np.mean(acc_values)
        avg_velocity = np.mean(vel_values)
        
        # Consistency measure
        acc_signs = [np.sign(a) for a in acc_values if a != 0]
        consistency = len([s for s in acc_signs if s == acc_signs[0]]) / len(acc_signs) if acc_signs else 0
        
        # Classify acceleration state
        if avg_acceleration > 0.1 and avg_velocity > 0:
            acceleration_state = 'bullish_accelerating'
        elif avg_acceleration > 0.1 and avg_velocity < 0:
            acceleration_state = 'bearish_decelerating'
        elif avg_acceleration < -0.1 and avg_velocity > 0:
            acceleration_state = 'bullish_decelerating'
        elif avg_acceleration < -0.1 and avg_velocity < 0:
            acceleration_state = 'bearish_accelerating'
        else:
            acceleration_state = 'stable'
        
        return {
            'acceleration': acceleration_state,
            'strength': float(abs(avg_acceleration)),
            'consistency': float(consistency),
            'avg_velocity': float(avg_velocity),
            'details': accelerations
        }
    
    def _calculate_statistical_significance(self, roc_values: Dict[int, pd.Series]) -> Dict[str, Any]:
        """Calculate statistical significance of current ROC levels"""
        significance_results = {}
        
        for period, roc in roc_values.items():
            if len(roc) < 30:
                significance_results[period] = {'significant': False, 'z_score': 0.0, 'percentile': 50}
                continue
            
            current_roc = roc.iloc[-1]
            if pd.isna(current_roc):
                significance_results[period] = {'significant': False, 'z_score': 0.0, 'percentile': 50}
                continue
            
            # Calculate historical statistics
            historical_roc = roc.tail(60).dropna()  # Use last 60 periods
            mean_roc = historical_roc.mean()
            std_roc = historical_roc.std()
            
            # Z-score calculation
            z_score = (current_roc - mean_roc) / (std_roc + 1e-8)
            
            # Percentile calculation
            percentile = stats.percentileofscore(historical_roc.values, current_roc)
            
            # Significance test
            is_significant = abs(z_score) > self.parameters['significance_threshold']
            
            significance_results[period] = {
                'significant': is_significant,
                'z_score': float(z_score),
                'percentile': float(percentile),
                'current_roc': float(current_roc),
                'historical_mean': float(mean_roc),
                'historical_std': float(std_roc)
            }
        
        return significance_results
    
    def _detect_momentum_patterns(self, roc_values: Dict[int, pd.Series]) -> Dict[str, Any]:
        """Detect momentum patterns across timeframes"""
        patterns = {
            'trend_alignment': False,
            'momentum_divergence': False,
            'momentum_convergence': False,
            'pattern_strength': 0.0
        }
        
        if len(roc_values) < 2:
            return patterns
        
        # Get current ROC values for each period
        current_rocs = {}
        for period, roc in roc_values.items():
            if len(roc) > 0 and not pd.isna(roc.iloc[-1]):
                current_rocs[period] = roc.iloc[-1]
        
        if len(current_rocs) < 2:
            return patterns
        
        sorted_periods = sorted(current_rocs.keys())
        roc_signs = [np.sign(current_rocs[p]) for p in sorted_periods]
        
        # Trend alignment: all ROCs have same sign
        if len(set(roc_signs)) == 1 and roc_signs[0] != 0:
            patterns['trend_alignment'] = True
            patterns['pattern_strength'] += 0.3
        
        # Momentum patterns
        short_term_roc = current_rocs[sorted_periods[0]]
        long_term_roc = current_rocs[sorted_periods[-1]]
        
        # Divergence: short-term and long-term momentum in opposite directions
        if np.sign(short_term_roc) != np.sign(long_term_roc) and short_term_roc != 0 and long_term_roc != 0:
            patterns['momentum_divergence'] = True
            patterns['pattern_strength'] += 0.4
        
        # Convergence: momentum values are getting closer
        if len(sorted_periods) >= 3:
            mid_term_roc = current_rocs[sorted_periods[len(sorted_periods)//2]]
            
            # Check if values are converging
            range_current = max(current_rocs.values()) - min(current_rocs.values())
            
            # Historical range for comparison
            historical_ranges = []
            for i in range(1, min(6, len(roc_values[sorted_periods[0]]))):
                historical_values = {}
                for period in sorted_periods:
                    if len(roc_values[period]) > i and not pd.isna(roc_values[period].iloc[-i-1]):
                        historical_values[period] = roc_values[period].iloc[-i-1]
                
                if len(historical_values) == len(sorted_periods):
                    hist_range = max(historical_values.values()) - min(historical_values.values())
                    historical_ranges.append(hist_range)
            
            if historical_ranges and range_current < np.mean(historical_ranges) * 0.7:
                patterns['momentum_convergence'] = True
                patterns['pattern_strength'] += 0.3
        
        return patterns
    
    def _train_ml_models(self, roc_values: Dict[int, pd.Series], data: pd.DataFrame) -> bool:
        """Train ML models for momentum classification"""
        if not all(len(roc) >= self.parameters['ml_lookback'] for roc in roc_values.values()):
            return False
        
        try:
            features, targets = self._prepare_ml_data(roc_values, data)
            if len(features) > 40:
                # Fit PCA for dimensionality reduction
                if features.shape[1] > 3:
                    self.pca.fit(features)
                    pca_features = self.pca.transform(features)
                else:
                    pca_features = features
                
                # Train models
                self.scaler.fit(pca_features)
                scaled_features = self.scaler.transform(pca_features)
                
                self.momentum_classifier.fit(scaled_features, targets)
                
                # Train trend predictor
                future_returns = data['close'].pct_change().shift(-5).tail(len(features)).dropna().values
                if len(future_returns) == len(scaled_features):
                    self.trend_predictor.fit(scaled_features, future_returns)
                
                self.models_trained = True
                return True
        except:
            pass
        return False
    
    def _prepare_ml_data(self, roc_values: Dict[int, pd.Series], data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare ML training data"""
        features, targets = [], []
        lookback = 10
        
        # Find minimum length across all ROC series
        min_length = min(len(roc) for roc in roc_values.values())
        
        for i in range(lookback, min_length - 5):
            feature_vector = []
            
            # ROC features for each period
            for period in sorted(roc_values.keys()):
                roc = roc_values[period]
                roc_window = roc.iloc[i-lookback:i].values
                
                feature_vector.extend([
                    np.mean(roc_window), np.std(roc_window), roc_window[-1],
                    roc_window[-1] - roc_window[0],  # ROC trend
                    len([x for x in roc_window if x > 0]) / len(roc_window)  # Positive ratio
                ])
            
            # Price and volume features
            price_window = data['close'].iloc[i-lookback:i].values
            volume_window = data['volume'].iloc[i-lookback:i].values
            
            feature_vector.extend([
                np.mean(price_window), np.std(price_window),
                volume_window[-1] / np.mean(volume_window) if np.mean(volume_window) != 0 else 1
            ])
            
            # Target: future momentum class
            future_return = (data['close'].iloc[i+5] - data['close'].iloc[i]) / data['close'].iloc[i]
            if future_return > 0.02:
                target = 2  # Strong bullish
            elif future_return > 0.005:
                target = 1  # Weak bullish
            elif future_return < -0.02:
                target = 0  # Strong bearish
            else:
                target = 1  # Neutral
            
            features.append(feature_vector)
            targets.append(target)
        
        return np.array(features), np.array(targets)
    
    def calculate_raw(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate ROC with advanced analysis"""
        try:
            # Calculate ROC for all periods
            roc_values = {}
            for period in self.parameters['periods']:
                roc = self._calculate_roc(data, period)
                if self.parameters['signal_filtering']:
                    roc = self._apply_signal_filtering(roc)
                roc_values[period] = roc
            
            # Advanced analysis
            momentum_acceleration = self._analyze_momentum_acceleration(roc_values)
            statistical_significance = self._calculate_statistical_significance(roc_values)
            momentum_patterns = self._detect_momentum_patterns(roc_values)
            
            # Train ML models
            if not self.models_trained:
                self._train_ml_models(roc_values, data)
            
            # Generate signal
            signal, confidence = self._generate_roc_signal(
                roc_values, momentum_acceleration, statistical_significance,
                momentum_patterns, data
            )
            
            # Update history
            for period, roc in roc_values.items():
                if len(roc) > 0 and not pd.isna(roc.iloc[-1]):
                    self.history['roc_values'][period].append(float(roc.iloc[-1]))
                    # Keep history limited
                    if len(self.history['roc_values'][period]) > 100:
                        self.history['roc_values'][period] = self.history['roc_values'][period][-100:]
            
            self.history['momentum_states'].append(momentum_acceleration['acceleration'])
            self.history['acceleration'].append(momentum_acceleration['strength'])
            self.history['significance_levels'].append(statistical_significance)
            
            # Keep other history limited
            for key in ['momentum_states', 'acceleration', 'significance_levels']:
                if len(self.history[key]) > 100:
                    self.history[key] = self.history[key][-100:]
            
            # Prepare current ROC values for output
            current_roc_values = {}
            for period, roc in roc_values.items():
                if len(roc) > 0:
                    current_roc_values[f'roc_{period}'] = float(roc.iloc[-1]) if not pd.isna(roc.iloc[-1]) else 0.0
            
            result = {
                'roc_values': current_roc_values,
                'primary_roc': float(roc_values[self.parameters['period']].iloc[-1]) if len(roc_values[self.parameters['period']]) > 0 else 0.0,
                'signal': signal,
                'confidence': confidence,
                'momentum_acceleration': momentum_acceleration,
                'statistical_significance': statistical_significance,
                'momentum_patterns': momentum_patterns,
                'trend_strength': self._calculate_trend_strength(roc_values),
                'momentum_quality': self._assess_momentum_quality(roc_values, statistical_significance),
                'values_history': {
                    f'roc_{period}': roc.tail(20).tolist() for period, roc in roc_values.items()
                }
            }
            
            return result
            
        except Exception as e:
            raise IndicatorCalculationException(
                indicator_name=self.name,
                calculation_step="raw_calculation",
                message=f"Failed to calculate ROC: {str(e)}",
                cause=e
            )
    
    def _generate_roc_signal(self, roc_values: Dict[int, pd.Series], momentum_acceleration: Dict,
                           statistical_significance: Dict, momentum_patterns: Dict,
                           data: pd.DataFrame) -> Tuple[SignalType, float]:
        """Generate comprehensive ROC signal"""
        signal_components = []
        confidence_components = []
        
        # Primary ROC signal
        primary_period = self.parameters['period']
        if primary_period in roc_values and len(roc_values[primary_period]) > 0:
            primary_roc = roc_values[primary_period].iloc[-1]
            if not pd.isna(primary_roc):
                # Basic ROC signal
                if primary_roc > 2:
                    signal_components.append(1.0)
                    confidence_components.append(0.6)
                elif primary_roc < -2:
                    signal_components.append(-1.0)
                    confidence_components.append(0.6)
                elif primary_roc > 0:
                    signal_components.append(0.5)
                    confidence_components.append(0.3)
                elif primary_roc < 0:
                    signal_components.append(-0.5)
                    confidence_components.append(0.3)
        
        # Momentum acceleration signals
        if momentum_acceleration['acceleration'] == 'bullish_accelerating':
            signal_components.append(momentum_acceleration['strength'] * momentum_acceleration['consistency'])
            confidence_components.append(0.7)
        elif momentum_acceleration['acceleration'] == 'bearish_accelerating':
            signal_components.append(-momentum_acceleration['strength'] * momentum_acceleration['consistency'])
            confidence_components.append(0.7)
        elif momentum_acceleration['acceleration'] == 'bullish_decelerating':
            signal_components.append(-0.3)
            confidence_components.append(0.4)
        elif momentum_acceleration['acceleration'] == 'bearish_decelerating':
            signal_components.append(0.3)
            confidence_components.append(0.4)
        
        # Statistical significance signals
        significant_signals = []
        for period, sig in statistical_significance.items():
            if sig['significant']:
                if sig['z_score'] > self.parameters['significance_threshold']:
                    significant_signals.append(1.0)
                elif sig['z_score'] < -self.parameters['significance_threshold']:
                    significant_signals.append(-1.0)
        
        if significant_signals:
            avg_significant_signal = np.mean(significant_signals)
            signal_components.append(avg_significant_signal * 0.8)
            confidence_components.append(0.8)
        
        # Pattern signals
        if momentum_patterns['trend_alignment']:
            # Determine direction from primary ROC
            if primary_period in roc_values and len(roc_values[primary_period]) > 0:
                primary_roc = roc_values[primary_period].iloc[-1]
                if not pd.isna(primary_roc):
                    alignment_signal = np.sign(primary_roc) * momentum_patterns['pattern_strength']
                    signal_components.append(alignment_signal)
                    confidence_components.append(0.6)
        
        if momentum_patterns['momentum_divergence']:
            # Divergence often signals reversal
            if primary_period in roc_values and len(roc_values[primary_period]) > 0:
                primary_roc = roc_values[primary_period].iloc[-1]
                if not pd.isna(primary_roc):
                    # Counter-trend signal
                    divergence_signal = -np.sign(primary_roc) * 0.4
                    signal_components.append(divergence_signal)
                    confidence_components.append(0.5)
        
        # ML enhancement
        if self.models_trained:
            try:
                ml_signal, ml_confidence = self._get_ml_signal(roc_values, data)
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
    
    def _get_ml_signal(self, roc_values: Dict[int, pd.Series], data: pd.DataFrame) -> Tuple[Optional[SignalType], float]:
        """Get ML-based signal prediction"""
        try:
            lookback = 10
            feature_vector = []
            
            # ROC features
            for period in sorted(roc_values.keys()):
                roc = roc_values[period]
                if len(roc) >= lookback:
                    roc_window = roc.tail(lookback).values
                    feature_vector.extend([
                        np.mean(roc_window), np.std(roc_window), roc_window[-1],
                        roc_window[-1] - roc_window[0],
                        len([x for x in roc_window if x > 0]) / len(roc_window)
                    ])
                else:
                    return None, 0.0
            
            # Price and volume features
            if len(data) >= lookback:
                price_window = data['close'].tail(lookback).values
                volume_window = data['volume'].tail(lookback).values
                feature_vector.extend([
                    np.mean(price_window), np.std(price_window),
                    volume_window[-1] / np.mean(volume_window) if np.mean(volume_window) != 0 else 1
                ])
            else:
                return None, 0.0
            
            feature_array = np.array([feature_vector])
            
            # Apply PCA if fitted
            if hasattr(self, 'pca') and feature_array.shape[1] > 3:
                pca_features = self.pca.transform(feature_array)
            else:
                pca_features = feature_array
            
            scaled_features = self.scaler.transform(pca_features)
            ml_proba = self.momentum_classifier.predict_proba(scaled_features)[0]
            
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
    
    def _calculate_trend_strength(self, roc_values: Dict[int, pd.Series]) -> Dict[str, Any]:
        """Calculate overall trend strength from ROC values"""
        if not roc_values:
            return {'strength': 0.0, 'direction': 'neutral', 'consistency': 0.0}
        
        current_rocs = []
        for period, roc in roc_values.items():
            if len(roc) > 0 and not pd.isna(roc.iloc[-1]):
                current_rocs.append(roc.iloc[-1])
        
        if not current_rocs:
            return {'strength': 0.0, 'direction': 'neutral', 'consistency': 0.0}
        
        avg_roc = np.mean(current_rocs)
        roc_signs = [np.sign(r) for r in current_rocs if r != 0]
        
        # Consistency: how many ROCs agree on direction
        if roc_signs:
            most_common_sign = max(set(roc_signs), key=roc_signs.count)
            consistency = roc_signs.count(most_common_sign) / len(roc_signs)
        else:
            consistency = 0.0
        
        # Direction
        if avg_roc > 1:
            direction = 'bullish'
        elif avg_roc < -1:
            direction = 'bearish'
        else:
            direction = 'neutral'
        
        # Strength
        strength = min(abs(avg_roc) / 10, 1.0)
        
        return {
            'strength': float(strength),
            'direction': direction,
            'consistency': float(consistency),
            'average_roc': float(avg_roc)
        }
    
    def _assess_momentum_quality(self, roc_values: Dict[int, pd.Series], 
                                statistical_significance: Dict) -> Dict[str, Any]:
        """Assess the quality of current momentum"""
        quality_factors = []
        
        # Statistical significance factor
        significant_count = sum(1 for sig in statistical_significance.values() if sig['significant'])
        significance_ratio = significant_count / len(statistical_significance) if statistical_significance else 0
        quality_factors.append(significance_ratio)
        
        # Consistency across timeframes
        current_rocs = []
        for period, roc in roc_values.items():
            if len(roc) > 0 and not pd.isna(roc.iloc[-1]):
                current_rocs.append(roc.iloc[-1])
        
        if current_rocs:
            roc_signs = [np.sign(r) for r in current_rocs if r != 0]
            if roc_signs:
                most_common_sign = max(set(roc_signs), key=roc_signs.count)
                consistency = roc_signs.count(most_common_sign) / len(roc_signs)
                quality_factors.append(consistency)
        
        # Magnitude factor
        if current_rocs:
            avg_magnitude = np.mean([abs(r) for r in current_rocs])
            magnitude_score = min(avg_magnitude / 5, 1.0)  # Normalize to 0-1
            quality_factors.append(magnitude_score)
        
        overall_quality = np.mean(quality_factors) if quality_factors else 0.0
        
        if overall_quality > 0.7:
            quality_rating = 'high'
        elif overall_quality > 0.4:
            quality_rating = 'medium'
        else:
            quality_rating = 'low'
        
        return {
            'quality_score': float(overall_quality),
            'quality_rating': quality_rating,
            'factors': {
                'significance_ratio': float(significance_ratio),
                'consistency': float(consistency) if 'consistency' in locals() else 0.0,
                'magnitude_score': float(magnitude_score) if 'magnitude_score' in locals() else 0.0
            }
        }
    
    def _generate_signal(self, value: Any, data: pd.DataFrame) -> Tuple[Optional[SignalType], float]:
        if isinstance(value, dict) and 'signal' in value and 'confidence' in value:
            return value['signal'], value['confidence']
        return SignalType.NEUTRAL, 0.0
    
    def _get_metadata(self, data: pd.DataFrame) -> Dict[str, Any]:
        base_metadata = super()._get_metadata(data)
        base_metadata.update({
            'indicator_type': 'momentum',
            'indicator_category': 'rate_of_change',
            'models_trained': self.models_trained,
            'periods': self.parameters['periods'],
            'volume_weighted': self.parameters['volume_weighted'],
            'signal_filtering': self.parameters['signal_filtering'],
            'data_type': 'ohlcv',
            'complexity': 'advanced'
        })
        return base_metadata