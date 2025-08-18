"""
Accumulation Distribution Signal - AI-Enhanced Volume Flow Analysis
================================================================

Advanced implementation of Accumulation/Distribution with machine learning signal enhancement.
Uses sophisticated volume flow analysis with pattern recognition and regime detection.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import talib

from ..base.standard_indicator import StandardIndicatorInterface, DataRequirement, DataType, SignalType, IndicatorResult


class AccumulationDistributionSignal(StandardIndicatorInterface):
    """
    AI-Enhanced Accumulation/Distribution Signal with machine learning pattern recognition.
    
    Combines traditional A/D line calculation with:
    - Volume-weighted price analysis
    - Anomaly detection for institutional flow
    - Multi-timeframe signal convergence
    - Machine learning trend prediction
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'period': 20,
            'volume_threshold': 1.5,  # Volume spike threshold
            'ml_lookback': 50,  # ML feature window
            'anomaly_threshold': 0.1,  # Anomaly detection threshold
            'signal_smoothing': 5,  # Signal smoothing period
            'trend_strength_min': 0.6,  # Minimum trend strength for signals
        }
        if parameters:
            default_params.update(parameters)
        
        super().__init__("AccumulationDistributionSignal", default_params)
        
        # Initialize ML components
        self.scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(
            contamination=self.parameters['anomaly_threshold'],
            random_state=42
        )
        self.is_trained = False
    
    def get_data_requirements(self) -> DataRequirement:
        return DataRequirement(
            data_type=DataType.OHLCV,
            required_columns=['high', 'low', 'close', 'volume'],
            min_periods=self.parameters['ml_lookback'],
            lookback_periods=200
        )
    
    def calculate_raw(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate AI-enhanced Accumulation/Distribution signal.
        """
        try:
            # Extract OHLCV data
            high = data['high'].values
            low = data['low'].values
            close = data['close'].values
            volume = data['volume'].values
            
            # Calculate traditional A/D line
            ad_line = self._calculate_ad_line(high, low, close, volume)
            
            # Calculate volume-weighted features
            vwap = self._calculate_vwap(high, low, close, volume)
            volume_ratio = self._calculate_volume_ratio(volume, self.parameters['period'])
            
            # Generate ML features
            features = self._extract_ml_features(ad_line, close, volume, vwap)
            
            # Detect anomalies (institutional flow)
            anomaly_score = self._detect_anomalies(features)
            
            # Calculate trend strength
            trend_strength = self._calculate_trend_strength(ad_line, close)
            
            # Generate primary signal
            signal_strength = self._calculate_signal_strength(
                ad_line, volume_ratio, anomaly_score, trend_strength
            )
            
            # Apply multi-timeframe analysis
            mtf_confirmation = self._multi_timeframe_analysis(data)
            
            # Final signal with confidence
            final_signal = signal_strength * mtf_confirmation
            
            return {
                'ad_line': ad_line[-1],
                'ad_momentum': np.diff(ad_line[-10:]).mean() if len(ad_line) >= 10 else 0,
                'volume_ratio': volume_ratio[-1],
                'vwap': vwap[-1],
                'anomaly_score': anomaly_score,
                'trend_strength': trend_strength,
                'signal_strength': signal_strength,
                'mtf_confirmation': mtf_confirmation,
                'final_signal': final_signal,
                'institutional_flow': anomaly_score > 0.7,
                'volume_surge': volume_ratio[-1] > self.parameters['volume_threshold']
            }
            
        except Exception as e:
            raise Exception(f"AccumulationDistributionSignal calculation failed: {str(e)}")
    
    def _calculate_ad_line(self, high: np.ndarray, low: np.ndarray, 
                          close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Calculate traditional Accumulation/Distribution line with improvements."""
        # Money Flow Multiplier
        clv = np.where(
            high != low,
            ((close - low) - (high - close)) / (high - low),
            0
        )
        
        # Money Flow Volume
        mfv = clv * volume
        
        # Accumulation/Distribution Line
        ad_line = np.cumsum(mfv)
        
        return ad_line
    
    def _calculate_vwap(self, high: np.ndarray, low: np.ndarray,
                       close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Calculate Volume Weighted Average Price."""
        typical_price = (high + low + close) / 3
        vwap = np.cumsum(typical_price * volume) / np.cumsum(volume)
        return vwap
    
    def _calculate_volume_ratio(self, volume: np.ndarray, period: int) -> np.ndarray:
        """Calculate volume ratio vs moving average."""
        volume_ma = pd.Series(volume).rolling(window=period).mean().fillna(volume[0])
        volume_ratio = volume / volume_ma.values
        return volume_ratio
    
    def _extract_ml_features(self, ad_line: np.ndarray, close: np.ndarray,
                            volume: np.ndarray, vwap: np.ndarray) -> np.ndarray:
        """Extract machine learning features for pattern recognition."""
        lookback = min(self.parameters['ml_lookback'], len(ad_line))
        
        features = []
        for i in range(lookback, len(ad_line)):
            window_ad = ad_line[i-lookback:i]
            window_close = close[i-lookback:i]
            window_volume = volume[i-lookback:i]
            window_vwap = vwap[i-lookback:i]
            
            # Statistical features
            feature_vector = [
                np.mean(window_ad),
                np.std(window_ad),
                np.mean(np.diff(window_ad)),
                np.std(np.diff(window_ad)),
                np.corrcoef(window_ad, window_close)[0, 1] if len(window_ad) > 1 else 0,
                np.corrcoef(window_ad, window_volume)[0, 1] if len(window_ad) > 1 else 0,
                window_close[-1] / window_vwap[-1] - 1,  # Price vs VWAP deviation
                np.mean(window_volume[-5:]) / np.mean(window_volume[:-5]) - 1 if len(window_volume) > 5 else 0,
            ]
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def _detect_anomalies(self, features: np.ndarray) -> float:
        """Detect anomalies in volume flow patterns."""
        if len(features) == 0:
            return 0.0
        
        try:
            if not self.is_trained and len(features) >= 20:
                # Train anomaly detector
                scaled_features = self.scaler.fit_transform(features)
                self.anomaly_detector.fit(scaled_features)
                self.is_trained = True
            
            if self.is_trained:
                # Detect anomalies in latest data
                latest_features = features[-5:] if len(features) >= 5 else features
                scaled_latest = self.scaler.transform(latest_features)
                anomaly_scores = self.anomaly_detector.decision_function(scaled_latest)
                return float(np.mean(anomaly_scores))
            
        except Exception:
            pass
        
        return 0.0
    
    def _calculate_trend_strength(self, ad_line: np.ndarray, close: np.ndarray) -> float:
        """Calculate trend strength using A/D line and price correlation."""
        if len(ad_line) < 20:
            return 0.0
        
        # Calculate correlation between A/D line and price
        recent_ad = ad_line[-20:]
        recent_price = close[-20:]
        
        correlation = np.corrcoef(recent_ad, recent_price)[0, 1]
        
        # Calculate A/D line trend
        ad_slope = np.polyfit(range(len(recent_ad)), recent_ad, 1)[0]
        ad_trend = np.tanh(ad_slope / (np.std(recent_ad) + 1e-8))
        
        # Combine correlation and trend
        trend_strength = abs(correlation) * abs(ad_trend)
        
        return float(np.clip(trend_strength, 0, 1))
    
    def _calculate_signal_strength(self, ad_line: np.ndarray, volume_ratio: np.ndarray,
                                  anomaly_score: float, trend_strength: float) -> float:
        """Calculate overall signal strength."""
        if len(ad_line) < 10:
            return 0.0
        
        # A/D line momentum
        ad_momentum = np.diff(ad_line[-10:]).mean()
        ad_momentum_norm = np.tanh(ad_momentum / (np.std(ad_line[-20:]) + 1e-8))
        
        # Volume confirmation
        volume_conf = np.mean(volume_ratio[-5:]) - 1 if len(volume_ratio) >= 5 else 0
        volume_conf_norm = np.tanh(volume_conf)
        
        # Anomaly boost
        anomaly_boost = 1 + abs(anomaly_score) * 0.5
        
        # Combine signals
        signal = ad_momentum_norm * trend_strength * volume_conf_norm * anomaly_boost
        
        return float(np.clip(signal, -1, 1))
    
    def _multi_timeframe_analysis(self, data: pd.DataFrame) -> float:
        """Perform multi-timeframe confirmation analysis."""
        try:
            # Analyze different timeframes
            timeframes = [5, 10, 20]  # Different period lengths
            confirmations = []
            
            for tf in timeframes:
                if len(data) >= tf * 2:
                    # Calculate A/D for this timeframe
                    tf_data = data.iloc[::tf]  # Sample every tf periods
                    if len(tf_data) >= 10:
                        tf_high = tf_data['high'].values
                        tf_low = tf_data['low'].values
                        tf_close = tf_data['close'].values
                        tf_volume = tf_data['volume'].values
                        
                        tf_ad = self._calculate_ad_line(tf_high, tf_low, tf_close, tf_volume)
                        tf_trend = np.diff(tf_ad[-5:]).mean() if len(tf_ad) >= 5 else 0
                        confirmations.append(np.sign(tf_trend))
            
            if confirmations:
                # Calculate confirmation strength
                confirmation = np.mean(confirmations)
                return float(np.clip(abs(confirmation), 0, 1))
            
        except Exception:
            pass
        
        return 0.5  # Neutral if analysis fails
    
    def _generate_signal(self, value: Dict[str, Any], data: pd.DataFrame) -> Tuple[Optional[SignalType], float]:
        """Generate trading signal based on A/D analysis."""
        signal_strength = value['final_signal']
        trend_strength = value['trend_strength']
        
        # Require minimum trend strength
        if trend_strength < self.parameters['trend_strength_min']:
            return SignalType.NEUTRAL, 0.5
        
        # Generate signal based on strength
        confidence = min(abs(signal_strength), 1.0)
        
        if signal_strength > 0.7:
            return SignalType.STRONG_BUY, confidence
        elif signal_strength > 0.3:
            return SignalType.BUY, confidence
        elif signal_strength < -0.7:
            return SignalType.STRONG_SELL, confidence
        elif signal_strength < -0.3:
            return SignalType.SELL, confidence
        else:
            return SignalType.NEUTRAL, confidence
