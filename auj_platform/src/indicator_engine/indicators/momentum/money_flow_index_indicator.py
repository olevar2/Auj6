"""
Money Flow Index (MFI) Indicator - Advanced Implementation
=========================================================

Advanced MFI implementation with ML-enhanced divergence detection,
smart money flow analysis, and sophisticated volume-price relationship modeling.

Author: AUJ Platform Development Team
Mission: Building advanced trading indicators for humanitarian impact
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from sklearn.ensemble import IsolationForest, GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
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


class MoneyFlowIndexIndicator(StandardIndicatorInterface):
    """
    Advanced Money Flow Index (MFI) Indicator Implementation
    
    Features:
    - Sophisticated typical price and money flow calculations
    - ML-enhanced divergence detection and anomaly identification
    - Smart money vs retail money flow classification
    - Institutional flow pattern recognition
    - Dynamic overbought/oversold threshold adaptation
    - Volume-weighted momentum analysis
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'period': 14,
            'overbought_threshold': 80,
            'oversold_threshold': 20,
            'adaptive_thresholds': True,
            'smart_money_detection': True,
            'divergence_periods': 25,
            'volume_spike_threshold': 2.0,
            'ml_lookback': 50
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__(name="MoneyFlowIndexIndicator", parameters=default_params)
        
        self.scaler = MinMaxScaler()
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.flow_predictor = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.models_trained = False
        
        self.history = {
            'mfi': [],
            'money_flow': [],
            'typical_price': [],
            'volume_ratio': [],
            'flow_classification': []
        }
    
    def get_data_requirements(self) -> DataRequirement:
        return DataRequirement(
            data_type=DataType.OHLCV,
            required_columns=['high', 'low', 'close', 'volume'],
            min_periods=max(self.parameters['period'], self.parameters['ml_lookback']) + 30,
            lookback_periods=150
        )
    
    def _calculate_typical_price(self, data: pd.DataFrame) -> pd.Series:
        """Calculate typical price (HLC/3)"""
        return (data['high'] + data['low'] + data['close']) / 3
    
    def _calculate_raw_money_flow(self, data: pd.DataFrame, typical_price: pd.Series) -> pd.Series:
        """Calculate raw money flow"""
        return typical_price * data['volume']
    
    def _classify_money_flow(self, typical_price: pd.Series, raw_money_flow: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Classify money flow as positive or negative"""
        price_change = typical_price.diff()
        
        positive_flow = pd.Series(0.0, index=typical_price.index)
        negative_flow = pd.Series(0.0, index=typical_price.index)
        
        # Positive money flow when typical price increases
        positive_mask = price_change > 0
        positive_flow[positive_mask] = raw_money_flow[positive_mask]
        
        # Negative money flow when typical price decreases
        negative_mask = price_change < 0
        negative_flow[negative_mask] = raw_money_flow[negative_mask]
        
        return positive_flow, negative_flow
    
    def _calculate_money_flow_ratio(self, positive_flow: pd.Series, negative_flow: pd.Series) -> pd.Series:
        """Calculate money flow ratio"""
        period = self.parameters['period']
        
        positive_sum = positive_flow.rolling(window=period).sum()
        negative_sum = negative_flow.rolling(window=period).sum()
        
        # Avoid division by zero
        money_flow_ratio = positive_sum / (negative_sum + 1e-8)
        return money_flow_ratio
    
    def _calculate_mfi(self, money_flow_ratio: pd.Series) -> pd.Series:
        """Calculate Money Flow Index"""
        mfi = 100 - (100 / (1 + money_flow_ratio))
        return mfi
    
    def _detect_smart_money_flow(self, data: pd.DataFrame, raw_money_flow: pd.Series) -> Dict[str, Any]:
        """Detect smart money vs retail money flow patterns"""
        if not self.parameters['smart_money_detection'] or len(data) < 20:
            return {'smart_money_activity': 'unknown', 'confidence': 0.0, 'flow_ratio': 0.0}
        
        volume_spike_threshold = self.parameters['volume_spike_threshold']
        
        # Calculate volume statistics
        avg_volume = data['volume'].rolling(window=20).mean()
        volume_ratio = data['volume'] / avg_volume
        
        # Identify volume spikes
        volume_spikes = volume_ratio > volume_spike_threshold
        
        # Analyze price action during volume spikes
        price_change = data['close'].pct_change()
        
        recent_spikes = volume_spikes.tail(10)
        recent_price_changes = price_change.tail(10)
        recent_money_flow = raw_money_flow.tail(10)
        
        if recent_spikes.sum() == 0:
            return {'smart_money_activity': 'low_volume', 'confidence': 0.3, 'flow_ratio': 0.0}
        
        # Smart money characteristics:
        # 1. Large volume with small price movement (accumulation/distribution)
        # 2. Volume precedes price movement
        # 3. Counter-trend volume spikes
        
        spike_indices = recent_spikes[recent_spikes].index
        
        smart_money_signals = []
        for idx in spike_indices:
            try:
                idx_pos = recent_spikes.index.get_loc(idx)
                if idx_pos < len(recent_price_changes) - 1:
                    current_price_change = abs(recent_price_changes.iloc[idx_pos])
                    volume_spike_magnitude = volume_ratio.loc[idx]
                    
                    # Smart money: high volume, low price movement
                    if volume_spike_magnitude > volume_spike_threshold and current_price_change < 0.01:
                        smart_money_signals.append(1)
                    # Retail money: high volume, high price movement
                    elif volume_spike_magnitude > volume_spike_threshold and current_price_change > 0.02:
                        smart_money_signals.append(-1)
                    else:
                        smart_money_signals.append(0)
            except:
                continue
        
        if smart_money_signals:
            avg_signal = np.mean(smart_money_signals)
            confidence = abs(avg_signal)
            
            if avg_signal > 0.3:
                activity = 'smart_accumulation'
            elif avg_signal < -0.3:
                activity = 'retail_momentum'
            else:
                activity = 'mixed'
            
            flow_ratio = len([s for s in smart_money_signals if s > 0]) / len(smart_money_signals)
        else:
            activity = 'unknown'
            confidence = 0.0
            flow_ratio = 0.0
        
        return {
            'smart_money_activity': activity,
            'confidence': confidence,
            'flow_ratio': flow_ratio
        }
    
    def _detect_divergences(self, mfi: pd.Series, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect MFI-Price divergences"""
        if len(mfi) < self.parameters['divergence_periods']:
            return {'bullish': False, 'bearish': False, 'strength': 0.0, 'confidence': 0.0}
        
        period = self.parameters['divergence_periods']
        recent_mfi = mfi.tail(period)
        recent_prices = data['close'].tail(period)
        
        # Find peaks and troughs
        mfi_peaks, _ = find_peaks(recent_mfi.values, distance=3, prominence=5)
        mfi_troughs, _ = find_peaks(-recent_mfi.values, distance=3, prominence=5)
        
        price_peaks, _ = find_peaks(recent_prices.values, distance=3)
        price_troughs, _ = find_peaks(-recent_prices.values, distance=3)
        
        bullish_divergence = False
        bearish_divergence = False
        divergence_strength = 0.0
        
        # Bullish divergence: price lower low, MFI higher low
        if len(mfi_troughs) >= 2 and len(price_troughs) >= 2:
            last_mfi_trough = recent_mfi.iloc[mfi_troughs[-1]]
            prev_mfi_trough = recent_mfi.iloc[mfi_troughs[-2]]
            
            last_price_trough = recent_prices.iloc[price_troughs[-1]]
            prev_price_trough = recent_prices.iloc[price_troughs[-2]]
            
            if last_price_trough < prev_price_trough and last_mfi_trough > prev_mfi_trough:
                bullish_divergence = True
                price_diff = abs(last_price_trough - prev_price_trough) / prev_price_trough
                mfi_diff = abs(last_mfi_trough - prev_mfi_trough) / prev_mfi_trough if prev_mfi_trough != 0 else 0
                divergence_strength = min(mfi_diff / (price_diff + 1e-8), 2.0)
        
        # Bearish divergence: price higher high, MFI lower high
        if len(mfi_peaks) >= 2 and len(price_peaks) >= 2:
            last_mfi_peak = recent_mfi.iloc[mfi_peaks[-1]]
            prev_mfi_peak = recent_mfi.iloc[mfi_peaks[-2]]
            
            last_price_peak = recent_prices.iloc[price_peaks[-1]]
            prev_price_peak = recent_prices.iloc[price_peaks[-2]]
            
            if last_price_peak > prev_price_peak and last_mfi_peak < prev_mfi_peak:
                bearish_divergence = True
                price_diff = abs(last_price_peak - prev_price_peak) / prev_price_peak
                mfi_diff = abs(last_mfi_peak - prev_mfi_peak) / prev_mfi_peak if prev_mfi_peak != 0 else 0
                divergence_strength = min(mfi_diff / (price_diff + 1e-8), 2.0)
        
        confidence = min(divergence_strength, 1.0) if bullish_divergence or bearish_divergence else 0.0
        
        return {
            'bullish': bullish_divergence,
            'bearish': bearish_divergence,
            'strength': divergence_strength,
            'confidence': confidence
        }
    
    def _calculate_adaptive_thresholds(self, mfi: pd.Series) -> Dict[str, float]:
        """Calculate adaptive overbought/oversold thresholds"""
        if not self.parameters['adaptive_thresholds'] or len(mfi) < 50:
            return {
                'overbought': self.parameters['overbought_threshold'],
                'oversold': self.parameters['oversold_threshold']
            }
        
        # Use percentiles for adaptive thresholds
        recent_mfi = mfi.tail(50)
        
        # Calculate percentile-based thresholds
        upper_percentile = np.percentile(recent_mfi.dropna(), 85)
        lower_percentile = np.percentile(recent_mfi.dropna(), 15)
        
        # Ensure thresholds are reasonable
        overbought = max(70, min(90, upper_percentile))
        oversold = max(10, min(30, lower_percentile))
        
        return {
            'overbought': float(overbought),
            'oversold': float(oversold)
        }
    
    def _train_models(self, mfi: pd.Series, raw_money_flow: pd.Series, data: pd.DataFrame) -> bool:
        """Train ML models for flow analysis"""
        if len(mfi) < self.parameters['ml_lookback'] * 2:
            return False
        
        try:
            # Prepare features for anomaly detection
            features = self._prepare_features(mfi, raw_money_flow, data)
            if len(features) > 30:
                # Train anomaly detector
                self.anomaly_detector.fit(features)
                
                # Train flow predictor
                targets = mfi.tail(len(features) - 5).shift(-5).dropna().values
                flow_features = features[:-5]
                if len(targets) == len(flow_features):
                    self.flow_predictor.fit(flow_features, targets)
                
                self.models_trained = True
                return True
        except:
            pass
        return False
    
    def _prepare_features(self, mfi: pd.Series, raw_money_flow: pd.Series, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for ML models"""
        lookback = 10
        features = []
        
        for i in range(lookback, len(mfi)):
            mfi_window = mfi.iloc[i-lookback:i].values
            flow_window = raw_money_flow.iloc[i-lookback:i].values
            volume_window = data['volume'].iloc[i-lookback:i].values
            price_window = data['close'].iloc[i-lookback:i].values
            
            feature_vector = [
                np.mean(mfi_window), np.std(mfi_window), mfi_window[-1],
                np.mean(flow_window), np.std(flow_window), flow_window[-1],
                np.mean(volume_window), np.std(volume_window), volume_window[-1],
                np.mean(price_window), np.std(price_window), price_window[-1],
                np.corrcoef(mfi_window, price_window)[0, 1] if len(set(mfi_window)) > 1 else 0,
                len([x for x in mfi_window if x > 50]) / len(mfi_window),  # Above 50 ratio
                mfi_window[-1] - mfi_window[0],  # MFI change
            ]
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def _detect_anomalies(self, mfi: pd.Series, raw_money_flow: pd.Series, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect anomalies in money flow patterns"""
        if not self.models_trained:
            return {'anomaly_detected': False, 'anomaly_score': 0.0, 'anomaly_type': 'none'}
        
        try:
            current_features = self._prepare_features(mfi, raw_money_flow, data)
            if len(current_features) > 0:
                anomaly_score = self.anomaly_detector.decision_function([current_features[-1]])[0]
                is_anomaly = self.anomaly_detector.predict([current_features[-1]])[0] == -1
                
                # Determine anomaly type
                if is_anomaly:
                    current_mfi = mfi.iloc[-1]
                    if current_mfi > 80:
                        anomaly_type = 'extreme_overbought'
                    elif current_mfi < 20:
                        anomaly_type = 'extreme_oversold'
                    else:
                        anomaly_type = 'flow_pattern_anomaly'
                else:
                    anomaly_type = 'none'
                
                return {
                    'anomaly_detected': bool(is_anomaly),
                    'anomaly_score': float(anomaly_score),
                    'anomaly_type': anomaly_type
                }
        except:
            pass
        
        return {'anomaly_detected': False, 'anomaly_score': 0.0, 'anomaly_type': 'none'}
    
    def calculate_raw(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate MFI with advanced analysis"""
        try:
            # Calculate typical price and money flow
            typical_price = self._calculate_typical_price(data)
            raw_money_flow = self._calculate_raw_money_flow(data, typical_price)
            
            # Classify money flow
            positive_flow, negative_flow = self._classify_money_flow(typical_price, raw_money_flow)
            
            # Calculate money flow ratio and MFI
            money_flow_ratio = self._calculate_money_flow_ratio(positive_flow, negative_flow)
            mfi = self._calculate_mfi(money_flow_ratio)
            
            # Smart money flow detection
            smart_money = self._detect_smart_money_flow(data, raw_money_flow)
            
            # Divergence detection
            divergences = self._detect_divergences(mfi, data)
            
            # Adaptive thresholds
            thresholds = self._calculate_adaptive_thresholds(mfi)
            
            # Train models
            if not self.models_trained:
                self._train_models(mfi, raw_money_flow, data)
            
            # Anomaly detection
            anomalies = self._detect_anomalies(mfi, raw_money_flow, data)
            
            # Generate signal
            signal, confidence = self._generate_mfi_signal(
                mfi, thresholds, divergences, smart_money, anomalies, data
            )
            
            # Update history
            self.history['mfi'].append(float(mfi.iloc[-1]))
            self.history['money_flow'].append(float(raw_money_flow.iloc[-1]))
            self.history['typical_price'].append(float(typical_price.iloc[-1]))
            self.history['volume_ratio'].append(float(data['volume'].iloc[-1] / data['volume'].rolling(20).mean().iloc[-1]))
            self.history['flow_classification'].append(smart_money['smart_money_activity'])
            
            # Keep history limited
            for key in self.history:
                if len(self.history[key]) > 100:
                    self.history[key] = self.history[key][-100:]
            
            result = {
                'mfi': float(mfi.iloc[-1]),
                'money_flow_ratio': float(money_flow_ratio.iloc[-1]),
                'raw_money_flow': float(raw_money_flow.iloc[-1]),
                'signal': signal,
                'confidence': confidence,
                'thresholds': thresholds,
                'divergences': divergences,
                'smart_money_flow': smart_money,
                'anomalies': anomalies,
                'flow_analysis': self._analyze_flow_dynamics(positive_flow, negative_flow),
                'market_pressure': self._calculate_market_pressure(mfi, thresholds),
                'values_history': {
                    'mfi': mfi.tail(20).tolist(),
                    'positive_flow': positive_flow.tail(20).tolist(),
                    'negative_flow': negative_flow.tail(20).tolist()
                }
            }
            
            return result
            
        except Exception as e:
            raise IndicatorCalculationException(
                indicator_name=self.name,
                calculation_step="raw_calculation",
                message=f"Failed to calculate MFI: {str(e)}",
                cause=e
            )
    
    def _generate_mfi_signal(self, mfi: pd.Series, thresholds: Dict, divergences: Dict,
                            smart_money: Dict, anomalies: Dict, data: pd.DataFrame) -> Tuple[SignalType, float]:
        """Generate comprehensive MFI signal"""
        signal_components = []
        confidence_components = []
        
        current_mfi = mfi.iloc[-1]
        
        # Overbought/Oversold signals
        if current_mfi > thresholds['overbought']:
            signal_components.append(-0.8)
            confidence_components.append(0.7)
        elif current_mfi < thresholds['oversold']:
            signal_components.append(0.8)
            confidence_components.append(0.7)
        
        # Divergence signals
        if divergences['bullish']:
            signal_components.append(divergences['strength'])
            confidence_components.append(divergences['confidence'])
        elif divergences['bearish']:
            signal_components.append(-divergences['strength'])
            confidence_components.append(divergences['confidence'])
        
        # Smart money signals
        if smart_money['smart_money_activity'] == 'smart_accumulation':
            signal_components.append(smart_money['confidence'])
            confidence_components.append(smart_money['confidence'])
        elif smart_money['smart_money_activity'] == 'retail_momentum':
            # Counter-trend to retail momentum
            signal_components.append(-smart_money['confidence'] * 0.5)
            confidence_components.append(smart_money['confidence'])
        
        # Anomaly signals
        if anomalies['anomaly_detected']:
            if anomalies['anomaly_type'] == 'extreme_oversold':
                signal_components.append(0.6)
                confidence_components.append(0.5)
            elif anomalies['anomaly_type'] == 'extreme_overbought':
                signal_components.append(-0.6)
                confidence_components.append(0.5)
        
        # MFI momentum
        if len(mfi) >= 3:
            mfi_momentum = mfi.iloc[-1] - mfi.iloc[-3]
            if abs(mfi_momentum) > 5:
                momentum_signal = np.sign(mfi_momentum) * 0.3
                signal_components.append(momentum_signal)
                confidence_components.append(0.4)
        
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
    
    def _analyze_flow_dynamics(self, positive_flow: pd.Series, negative_flow: pd.Series) -> Dict[str, Any]:
        """Analyze money flow dynamics"""
        if len(positive_flow) < 10:
            return {'trend': 'unknown', 'strength': 0.0, 'balance': 0.5}
        
        recent_positive = positive_flow.tail(10).sum()
        recent_negative = negative_flow.tail(10).sum()
        
        total_flow = recent_positive + recent_negative
        if total_flow == 0:
            return {'trend': 'no_flow', 'strength': 0.0, 'balance': 0.5}
        
        positive_ratio = recent_positive / total_flow
        
        if positive_ratio > 0.6:
            trend = 'positive_dominant'
            strength = positive_ratio
        elif positive_ratio < 0.4:
            trend = 'negative_dominant'
            strength = 1 - positive_ratio
        else:
            trend = 'balanced'
            strength = 0.5
        
        return {
            'trend': trend,
            'strength': float(strength),
            'balance': float(positive_ratio)
        }
    
    def _calculate_market_pressure(self, mfi: pd.Series, thresholds: Dict) -> Dict[str, Any]:
        """Calculate market pressure based on MFI levels"""
        current_mfi = mfi.iloc[-1]
        
        if current_mfi > thresholds['overbought']:
            pressure_type = 'selling_pressure'
            intensity = (current_mfi - thresholds['overbought']) / (100 - thresholds['overbought'])
        elif current_mfi < thresholds['oversold']:
            pressure_type = 'buying_pressure'
            intensity = (thresholds['oversold'] - current_mfi) / thresholds['oversold']
        else:
            pressure_type = 'neutral'
            # Distance from neutral (50)
            intensity = abs(current_mfi - 50) / 50
        
        return {
            'type': pressure_type,
            'intensity': float(min(intensity, 1.0))
        }
    
    def _generate_signal(self, value: Any, data: pd.DataFrame) -> Tuple[Optional[SignalType], float]:
        if isinstance(value, dict) and 'signal' in value and 'confidence' in value:
            return value['signal'], value['confidence']
        return SignalType.NEUTRAL, 0.0
    
    def _get_metadata(self, data: pd.DataFrame) -> Dict[str, Any]:
        base_metadata = super()._get_metadata(data)
        base_metadata.update({
            'indicator_type': 'momentum',
            'indicator_category': 'money_flow_index',
            'models_trained': self.models_trained,
            'data_type': 'ohlcv',
            'complexity': 'advanced'
        })
        return base_metadata