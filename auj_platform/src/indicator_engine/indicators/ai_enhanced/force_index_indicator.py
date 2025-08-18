"""
Force Index AI-Enhanced Indicator
=================================

Advanced Force Index with machine learning enhancements, smart money detection,
and institutional flow analysis.
"""

from typing import Dict, Any, Union
import pandas as pd
import numpy as np
from scipy import stats
from scipy.signal import savgol_filter, find_peaks
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')

from ..base.standard_indicator import StandardIndicatorInterface, DataRequirement, DataType


class ForceIndexIndicator(StandardIndicatorInterface):
    """
    AI-Enhanced Force Index Indicator with institutional flow detection.
    
    Features:
    - Classic Force Index calculation with ML enhancements
    - Smart money flow detection using volume clustering
    - Institutional block trade identification
    - Divergence detection with price action
    - Adaptive smoothing based on market volatility
    - Anomaly detection for unusual force spikes
    - Multi-timeframe force analysis
    - Volume profile integration
    """
    
    def __init__(self, parameters: Dict[str, Any] = None):
        default_params = {
            'period': 13,
            'smooth_period': 5,
            'volume_threshold': 2.0,  # Std deviations for large volume
            'divergence_lookback': 20,
            'cluster_eps': 0.3,
            'cluster_min_samples': 3,
            'anomaly_contamination': 0.05,
            'adaptive_smoothing': True,
            'detect_divergences': True,
            'institutional_detection': True,
            'use_volume_profile': True
        }
        if parameters:
            default_params.update(parameters)
        super().__init__("ForceIndexIndicator", default_params)
        
        # Initialize ML components
        self.scaler = RobustScaler()
        self.isolation_forest = IsolationForest(
            contamination=self.parameters['anomaly_contamination'],
            random_state=42
        )
        self.dbscan = DBSCAN(
            eps=self.parameters['cluster_eps'],
            min_samples=self.parameters['cluster_min_samples']
        )
        
        # Historical data for pattern recognition
        self.force_history = []
        self.divergence_history = []
        self.institutional_flows = []
    
    def get_data_requirements(self) -> DataRequirement:
        return DataRequirement(
            data_type=DataType.OHLCV,
            required_columns=["open", "high", "low", "close", "volume"],
            min_periods=max(50, self.parameters['period'] * 3)
        )
    
    def _calculate_raw_force_index(self, data: pd.DataFrame) -> pd.Series:
        """Calculate raw Force Index."""
        price_change = data['close'].diff()
        force_index = price_change * data['volume']
        return force_index
    
    def _calculate_smoothed_force_index(self, force_index: pd.Series) -> pd.Series:
        """Apply adaptive smoothing to Force Index."""
        if not self.parameters['adaptive_smoothing']:
            return force_index.rolling(self.parameters['smooth_period']).mean()
        
        # Calculate market volatility for adaptive smoothing
        returns = force_index.pct_change().dropna()
        volatility = returns.rolling(20).std()
        
        # Adjust smoothing period based on volatility
        smoothed = pd.Series(index=force_index.index, dtype=float)
        for i in range(len(force_index)):
            if i < 20:
                period = self.parameters['smooth_period']
            else:
                vol_factor = volatility.iloc[i] / volatility.mean()
                period = max(2, int(self.parameters['smooth_period'] * (1 + vol_factor)))
            
            if i >= period:
                smoothed.iloc[i] = force_index.iloc[i-period+1:i+1].mean()
            else:
                smoothed.iloc[i] = force_index.iloc[:i+1].mean()
        
        return smoothed
    
    def _detect_institutional_flows(self, data: pd.DataFrame, force_index: pd.Series) -> Dict[str, Any]:
        """Detect institutional flows using volume and force clustering."""
        if not self.parameters['institutional_detection']:
            return {'institutional_signals': [], 'flow_strength': 0.0}
        
        try:
            # Calculate volume statistics
            volume_mean = data['volume'].rolling(50).mean()
            volume_std = data['volume'].rolling(50).std()
            volume_z_score = (data['volume'] - volume_mean) / volume_std
            
            # Identify high volume periods
            high_volume_mask = volume_z_score > self.parameters['volume_threshold']
            
            # Calculate force intensity during high volume
            force_intensity = np.abs(force_index) / (data['volume'] + 1e-6)
            
            # Prepare features for clustering
            features = pd.DataFrame({
                'volume_z': volume_z_score,
                'force_intensity': force_intensity,
                'price_change': data['close'].pct_change(),
                'force_index': force_index
            }).fillna(0)
            
            # Scale features
            features_scaled = self.scaler.fit_transform(features)
            
            # Perform clustering
            clusters = self.dbscan.fit_predict(features_scaled)
            
            # Analyze clusters for institutional patterns
            institutional_signals = []
            for cluster_id in set(clusters):
                if cluster_id == -1:  # Noise cluster
                    continue
                
                cluster_mask = clusters == cluster_id
                cluster_data = features[cluster_mask]
                
                # Check if cluster represents institutional flow
                avg_volume_z = cluster_data['volume_z'].mean()
                avg_force_intensity = cluster_data['force_intensity'].mean()
                
                if avg_volume_z > 1.5 and avg_force_intensity > cluster_data['force_intensity'].median():
                    institutional_signals.append({
                        'cluster_id': int(cluster_id),
                        'volume_strength': float(avg_volume_z),
                        'force_strength': float(avg_force_intensity),
                        'direction': 'buy' if cluster_data['force_index'].mean() > 0 else 'sell',
                        'confidence': float(len(cluster_data) / len(features))
                    })
            
            # Calculate overall flow strength
            flow_strength = 0.0
            if institutional_signals:
                flow_strength = np.mean([sig['force_strength'] for sig in institutional_signals])
            
            return {
                'institutional_signals': institutional_signals,
                'flow_strength': flow_strength,
                'total_clusters': len(set(clusters)) - (1 if -1 in clusters else 0)
            }
            
        except Exception as e:
            self.logger.error(f"Error in institutional flow detection: {e}")
            return {'institutional_signals': [], 'flow_strength': 0.0}
    
    def _detect_divergences(self, data: pd.DataFrame, force_index: pd.Series) -> Dict[str, Any]:
        """Detect price-force divergences."""
        if not self.parameters['detect_divergences']:
            return {'divergences': [], 'divergence_strength': 0.0}
        
        try:
            lookback = self.parameters['divergence_lookback']
            if len(data) < lookback * 2:
                return {'divergences': [], 'divergence_strength': 0.0}
            
            # Find peaks and troughs in price and force
            price_peaks, _ = find_peaks(data['close'], distance=lookback//2)
            price_troughs, _ = find_peaks(-data['close'], distance=lookback//2)
            force_peaks, _ = find_peaks(force_index, distance=lookback//2)
            force_troughs, _ = find_peaks(-force_index, distance=lookback//2)
            
            divergences = []
            
            # Check for bullish divergence (price lower lows, force higher lows)
            if len(price_troughs) >= 2 and len(force_troughs) >= 2:
                recent_price_trough = price_troughs[-1]
                prev_price_trough = price_troughs[-2]
                
                # Find corresponding force troughs
                force_trough_candidates = force_troughs[force_troughs <= recent_price_trough + lookback//4]
                if len(force_trough_candidates) > 0:
                    recent_force_trough = force_trough_candidates[-1]
                    
                    prev_force_candidates = force_troughs[force_troughs <= prev_price_trough + lookback//4]
                    if len(prev_force_candidates) > 0:
                        prev_force_trough = prev_force_candidates[-1]
                        
                        # Check for divergence
                        price_lower = data['close'].iloc[recent_price_trough] < data['close'].iloc[prev_price_trough]
                        force_higher = force_index.iloc[recent_force_trough] > force_index.iloc[prev_force_trough]
                        
                        if price_lower and force_higher:
                            strength = abs(force_index.iloc[recent_force_trough] - force_index.iloc[prev_force_trough])
                            divergences.append({
                                'type': 'bullish',
                                'strength': float(strength),
                                'recent_index': int(recent_price_trough),
                                'previous_index': int(prev_price_trough)
                            })
            
            # Check for bearish divergence (price higher highs, force lower highs)
            if len(price_peaks) >= 2 and len(force_peaks) >= 2:
                recent_price_peak = price_peaks[-1]
                prev_price_peak = price_peaks[-2]
                
                # Find corresponding force peaks
                force_peak_candidates = force_peaks[force_peaks <= recent_price_peak + lookback//4]
                if len(force_peak_candidates) > 0:
                    recent_force_peak = force_peak_candidates[-1]
                    
                    prev_force_candidates = force_peaks[force_peaks <= prev_price_peak + lookback//4]
                    if len(prev_force_candidates) > 0:
                        prev_force_peak = prev_force_candidates[-1]
                        
                        # Check for divergence
                        price_higher = data['close'].iloc[recent_price_peak] > data['close'].iloc[prev_price_peak]
                        force_lower = force_index.iloc[recent_force_peak] < force_index.iloc[prev_force_peak]
                        
                        if price_higher and force_lower:
                            strength = abs(force_index.iloc[recent_force_peak] - force_index.iloc[prev_force_peak])
                            divergences.append({
                                'type': 'bearish',
                                'strength': float(strength),
                                'recent_index': int(recent_price_peak),
                                'previous_index': int(prev_price_peak)
                            })
            
            # Calculate overall divergence strength
            divergence_strength = 0.0
            if divergences:
                divergence_strength = np.mean([div['strength'] for div in divergences])
            
            return {
                'divergences': divergences,
                'divergence_strength': float(divergence_strength),
                'total_divergences': len(divergences)
            }
            
        except Exception as e:
            self.logger.error(f"Error in divergence detection: {e}")
            return {'divergences': [], 'divergence_strength': 0.0}
    
    def _calculate_volume_profile_integration(self, data: pd.DataFrame, force_index: pd.Series) -> Dict[str, Any]:
        """Integrate volume profile analysis with force index."""
        if not self.parameters['use_volume_profile']:
            return {'volume_profile_signals': [], 'profile_strength': 0.0}
        
        try:
            # Create price-volume distribution
            price_levels = pd.cut(data['close'], bins=20, labels=False)
            volume_profile = data.groupby(price_levels)['volume'].sum()
            force_profile = data.groupby(price_levels)[force_index.name if force_index.name else 'force'].sum()
            
            # Find high volume nodes (HVN) and low volume nodes (LVN)
            volume_mean = volume_profile.mean()
            volume_std = volume_profile.std()
            
            hvn_levels = volume_profile[volume_profile > volume_mean + volume_std].index
            lvn_levels = volume_profile[volume_profile < volume_mean - volume_std].index
            
            # Analyze force at these levels
            signals = []
            for level in hvn_levels:
                if level in force_profile.index:
                    signals.append({
                        'level': int(level),
                        'type': 'HVN',
                        'volume': float(volume_profile[level]),
                        'force': float(force_profile[level]),
                        'significance': 'high'
                    })
            
            for level in lvn_levels:
                if level in force_profile.index:
                    signals.append({
                        'level': int(level),
                        'type': 'LVN',
                        'volume': float(volume_profile[level]),
                        'force': float(force_profile[level]),
                        'significance': 'low'
                    })
            
            # Calculate profile strength
            profile_strength = 0.0
            if signals:
                profile_strength = np.mean([abs(sig['force']) for sig in signals])
            
            return {
                'volume_profile_signals': signals,
                'profile_strength': float(profile_strength),
                'hvn_count': len(hvn_levels),
                'lvn_count': len(lvn_levels)
            }
            
        except Exception as e:
            self.logger.error(f"Error in volume profile integration: {e}")
            return {'volume_profile_signals': [], 'profile_strength': 0.0}
    
    def _detect_anomalies(self, force_index: pd.Series) -> pd.Series:
        """Detect anomalous force index values."""
        try:
            # Prepare features for anomaly detection
            features = pd.DataFrame({
                'force': force_index,
                'force_abs': np.abs(force_index),
                'force_change': force_index.diff(),
                'force_ma_ratio': force_index / force_index.rolling(20).mean()
            }).fillna(0)
            
            # Fit isolation forest
            anomaly_scores = self.isolation_forest.fit_predict(features)
            
            # Convert to pandas Series
            return pd.Series(anomaly_scores, index=force_index.index)
            
        except Exception as e:
            self.logger.error(f"Error in anomaly detection: {e}")
            return pd.Series(np.ones(len(force_index)), index=force_index.index)
    
    def calculate_raw(self, data: pd.DataFrame) -> Union[float, int, Dict[str, Any]]:
        """
        Calculate AI-enhanced Force Index with institutional flow analysis.
        
        Returns:
            Dict containing:
            - force_index: Current Force Index value
            - smoothed_force: Smoothed Force Index
            - institutional_flows: Detected institutional flow patterns
            - divergences: Price-force divergences
            - volume_profile: Volume profile integration
            - anomaly_score: Anomaly detection result
            - signal_strength: Overall signal strength
        """
        try:
            if len(data) < self.parameters['period']:
                return {'force_index': 0.0, 'signal_strength': 0.0}
            
            # Calculate raw Force Index
            raw_force = self._calculate_raw_force_index(data)
            
            if raw_force.isna().all():
                return {'force_index': 0.0, 'signal_strength': 0.0}
            
            # Apply smoothing
            smoothed_force = self._calculate_smoothed_force_index(raw_force)
            
            # Detect institutional flows
            institutional_analysis = self._detect_institutional_flows(data, smoothed_force)
            
            # Detect divergences
            divergence_analysis = self._detect_divergences(data, smoothed_force)
            
            # Volume profile integration
            volume_profile_analysis = self._calculate_volume_profile_integration(data, smoothed_force)
            
            # Anomaly detection
            anomaly_scores = self._detect_anomalies(smoothed_force)
            
            # Calculate current values
            current_force = float(smoothed_force.iloc[-1]) if not pd.isna(smoothed_force.iloc[-1]) else 0.0
            current_raw_force = float(raw_force.iloc[-1]) if not pd.isna(raw_force.iloc[-1]) else 0.0
            current_anomaly = int(anomaly_scores.iloc[-1]) if len(anomaly_scores) > 0 else 1
            
            # Calculate signal strength
            force_std = smoothed_force.rolling(50).std().iloc[-1]
            if pd.isna(force_std) or force_std == 0:
                normalized_force = 0.0
            else:
                normalized_force = current_force / force_std
            
            signal_strength = float(np.tanh(abs(normalized_force) / 2))  # Normalize to [0, 1]
            
            # Combine all analysis
            institutional_strength = institutional_analysis.get('flow_strength', 0.0)
            divergence_strength = divergence_analysis.get('divergence_strength', 0.0)
            profile_strength = volume_profile_analysis.get('profile_strength', 0.0)
            
            # Calculate composite signal strength
            composite_strength = (
                signal_strength * 0.4 +
                institutional_strength * 0.3 +
                divergence_strength * 0.2 +
                profile_strength * 0.1
            )
            
            # Update history
            self.force_history.append(current_force)
            if len(self.force_history) > 100:
                self.force_history = self.force_history[-100:]
            
            # Determine market sentiment
            sentiment = "neutral"
            if current_force > force_std:
                sentiment = "bullish"
            elif current_force < -force_std:
                sentiment = "bearish"
            
            return {
                'force_index': current_force,
                'raw_force_index': current_raw_force,
                'smoothed_force': current_force,
                'institutional_flows': institutional_analysis,
                'divergences': divergence_analysis,
                'volume_profile': volume_profile_analysis,
                'anomaly_score': float(current_anomaly),
                'signal_strength': float(np.clip(composite_strength, 0, 1)),
                'normalized_force': float(normalized_force),
                'market_sentiment': sentiment,
                'force_velocity': float(smoothed_force.diff().iloc[-1]) if len(smoothed_force) > 1 else 0.0,
                'force_acceleration': float(smoothed_force.diff().diff().iloc[-1]) if len(smoothed_force) > 2 else 0.0,
                'trend_strength': float(abs(normalized_force)),
                'is_anomaly': current_anomaly == -1
            }
            
        except Exception as e:
            self.logger.error(f"Error in ForceIndexIndicator calculation: {e}")
            return {
                'force_index': 0.0,
                'signal_strength': 0.0,
                'error': str(e)
            }
