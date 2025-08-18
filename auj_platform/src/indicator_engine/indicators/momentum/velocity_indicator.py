"""
Advanced Velocity Indicator with Machine Learning Integration

This module implements a sophisticated velocity analysis system with:
- Multi-dimensional velocity calculations across price, volume, and momentum
- Machine learning acceleration pattern recognition
- Advanced smoothing and adaptive parameter adjustment
- Regime detection and velocity regime classification
- Risk assessment and confidence scoring with position sizing

Part of the ASD trading platform's humanitarian mission.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import FastICA
from scipy import signal as scipy_signal
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

from ..base.standard_indicator import StandardIndicatorInterface


class VelocityIndicator(StandardIndicatorInterface):
    """
    Advanced Velocity Indicator with Machine Learning
    
    Features:
    - Multi-dimensional velocity analysis (price, volume, momentum, volatility)
    - ML-based acceleration pattern recognition and trend prediction
    - Advanced filtering and smoothing with adaptive parameters
    - Velocity regime detection and classification
    - Risk and confidence scoring with dynamic position sizing
    - Multi-timeframe velocity convergence analysis
    """
    
    def __init__(self, 
                 velocity_periods: List[int] = [5, 10, 20, 50],
                 smoothing_period: int = 14,
                 acceleration_period: int = 5,
                 ml_lookback: int = 252,
                 regime_periods: List[int] = [20, 50, 100],
                 confidence_threshold: float = 0.65):
        """
        Initialize Advanced Velocity Indicator
        
        Args:
            velocity_periods: Periods for multi-timeframe velocity calculation
            smoothing_period: Period for velocity smoothing
            acceleration_period: Period for acceleration calculation
            ml_lookback: Lookback period for ML training
            regime_periods: Periods for regime detection
            confidence_threshold: Minimum confidence for signal generation
        """
        super().__init__()
        
        self.velocity_periods = velocity_periods
        self.smoothing_period = smoothing_period
        self.acceleration_period = acceleration_period
        self.ml_lookback = ml_lookback
        self.regime_periods = regime_periods
        self.confidence_threshold = confidence_threshold
        
        # ML components for different aspects
        self.velocity_predictor = RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            random_state=42
        )
        self.acceleration_classifier = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=10,
            random_state=42
        )
        self.pattern_regressor = ExtraTreesRegressor(
            n_estimators=100,
            max_depth=8,
            random_state=42
        )
        self.momentum_predictor = MLPRegressor(
            hidden_layer_sizes=(64, 32, 16),
            max_iter=1000,
            random_state=42
        )
        
        # Scalers for different data types
        self.velocity_scaler = StandardScaler()
        self.feature_scaler = MinMaxScaler()
        
        # Clustering for regime detection
        self.regime_clusterer = DBSCAN(eps=0.3, min_samples=5)
        self.ica = FastICA(n_components=None, random_state=42)
        
        # State tracking
        self.is_trained = False
        self.feature_columns = []
        self.velocity_history = []
        
        # Advanced features
        self.adaptive_smoothing = True
        self.volume_weighting = True
        self.regime_awareness = True
        self.pattern_recognition = True
        
    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate advanced velocity analysis with ML integration
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary containing velocity analysis, signals, and metadata
        """
        try:
            if len(data) < max(self.velocity_periods + [self.ml_lookback // 2]):
                return self._create_error_result("Insufficient data for velocity calculation")
            
            # Calculate multi-dimensional velocity components
            velocity_data = self._calculate_multi_dimensional_velocity(data)
            
            # Calculate acceleration and jerk (third derivative)
            acceleration_data = self._calculate_acceleration_and_jerk(velocity_data)
            
            # Perform advanced filtering and smoothing
            filtered_velocity = self._apply_advanced_filtering(velocity_data, data)
            
            # Detect velocity regimes
            regime_analysis = self._detect_velocity_regimes(data, velocity_data)
            
            # Calculate adaptive parameters
            adaptive_params = self._calculate_adaptive_parameters(data, velocity_data, regime_analysis)
            
            # Generate base velocity signals
            base_signals = self._generate_base_velocity_signals(data, velocity_data, acceleration_data, adaptive_params)
            
            # Train or update ML models
            if not self.is_trained and len(data) >= self.ml_lookback:
                self._train_ml_models(data, velocity_data, acceleration_data)
            
            # Generate ML-enhanced predictions
            ml_predictions = self._generate_ml_predictions(data, velocity_data, acceleration_data)
            
            # Perform pattern recognition
            pattern_analysis = self._perform_pattern_recognition(data, velocity_data, acceleration_data)
            
            # Calculate signal strength and confidence
            signal_analysis = self._analyze_signal_strength(
                data, velocity_data, base_signals, ml_predictions, pattern_analysis
            )
            
            # Synthesize multi-timeframe signals
            synthesized_signals = self._synthesize_velocity_signals(
                data, velocity_data, signal_analysis, regime_analysis
            )
            
            # Calculate risk metrics
            risk_metrics = self._calculate_velocity_risk_metrics(data, velocity_data, synthesized_signals)
            
            # Generate final recommendations
            recommendations = self._generate_velocity_recommendations(
                synthesized_signals, risk_metrics, regime_analysis
            )
            
            return {
                'velocity_data': velocity_data,
                'acceleration_data': acceleration_data,
                'filtered_velocity': filtered_velocity,
                'base_signals': base_signals,
                'ml_predictions': ml_predictions,
                'pattern_analysis': pattern_analysis,
                'signal_analysis': signal_analysis,
                'synthesized_signals': synthesized_signals,
                'regime_analysis': regime_analysis,
                'risk_metrics': risk_metrics,
                'recommendations': recommendations,
                'adaptive_parameters': adaptive_params,
                'confidence_score': float(signal_analysis.get('overall_confidence', 0.0)),
                'signal_strength': float(signal_analysis.get('signal_strength', 0.0)),
                'metadata': self._generate_metadata(data, velocity_data, synthesized_signals)
            }
            
        except Exception as e:
            return self._create_error_result(f"Velocity indicator calculation error: {str(e)}")
    
    def _calculate_multi_dimensional_velocity(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate velocity across multiple dimensions"""
        velocity_data = pd.DataFrame(index=data.index)
        
        # Price velocity (rate of change)
        for period in self.velocity_periods:
            velocity_data[f'price_velocity_{period}'] = data['close'].pct_change(period)
            velocity_data[f'high_velocity_{period}'] = data['high'].pct_change(period)
            velocity_data[f'low_velocity_{period}'] = data['low'].pct_change(period)
        
        # Volume velocity (if available)
        if 'volume' in data.columns:
            for period in self.velocity_periods:
                velocity_data[f'volume_velocity_{period}'] = data['volume'].pct_change(period)
                # Volume-price velocity
                velocity_data[f'vp_velocity_{period}'] = (data['close'] * data['volume']).pct_change(period)
        
        # Range velocity (high-low expansion/contraction)
        price_range = (data['high'] - data['low']) / data['close']
        for period in self.velocity_periods:
            velocity_data[f'range_velocity_{period}'] = price_range.pct_change(period)
        
        # Volatility velocity
        returns = data['close'].pct_change()
        for period in self.velocity_periods:
            rolling_vol = returns.rolling(window=period).std()
            velocity_data[f'volatility_velocity_{period}'] = rolling_vol.pct_change(period)
        
        # Composite velocity
        velocity_data['composite_velocity'] = self._calculate_composite_velocity(velocity_data)
        
        # Smoothed velocities
        for period in self.velocity_periods:
            if f'price_velocity_{period}' in velocity_data.columns:
                velocity_data[f'smooth_velocity_{period}'] = (
                    velocity_data[f'price_velocity_{period}'].ewm(span=self.smoothing_period).mean()
                )
        
        # Relative velocity (compared to historical average)
        for period in self.velocity_periods:
            if f'price_velocity_{period}' in velocity_data.columns:
                historical_avg = velocity_data[f'price_velocity_{period}'].rolling(window=100).mean()
                velocity_data[f'relative_velocity_{period}'] = (
                    velocity_data[f'price_velocity_{period}'] / historical_avg - 1
                )
        
        return velocity_data.fillna(0)
    
    def _calculate_composite_velocity(self, velocity_data: pd.DataFrame) -> pd.Series:
        """Calculate composite velocity from multiple components"""
        # Weight different velocity components
        composite = pd.Series(0.0, index=velocity_data.index)
        
        weights = {
            'price': 0.4,
            'volume': 0.2,
            'range': 0.2,
            'volatility': 0.2
        }
        
        for period in self.velocity_periods:
            period_weight = 1.0 / len(self.velocity_periods)
            
            # Price component
            if f'price_velocity_{period}' in velocity_data.columns:
                composite += velocity_data[f'price_velocity_{period}'] * weights['price'] * period_weight
            
            # Volume component
            if f'volume_velocity_{period}' in velocity_data.columns:
                composite += velocity_data[f'volume_velocity_{period}'] * weights['volume'] * period_weight
            
            # Range component
            if f'range_velocity_{period}' in velocity_data.columns:
                composite += velocity_data[f'range_velocity_{period}'] * weights['range'] * period_weight
            
            # Volatility component
            if f'volatility_velocity_{period}' in velocity_data.columns:
                composite += velocity_data[f'volatility_velocity_{period}'] * weights['volatility'] * period_weight
        
        return composite
    
    def _calculate_acceleration_and_jerk(self, velocity_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate acceleration (second derivative) and jerk (third derivative)"""
        acceleration_data = pd.DataFrame(index=velocity_data.index)
        
        # Acceleration for each velocity component
        for col in velocity_data.columns:
            if 'velocity' in col:
                # Acceleration (first derivative of velocity)
                acceleration_data[col.replace('velocity', 'acceleration')] = velocity_data[col].diff()
                
                # Jerk (second derivative of velocity, third of price)
                acceleration_data[col.replace('velocity', 'jerk')] = velocity_data[col].diff().diff()
        
        # Composite acceleration and jerk
        if 'composite_velocity' in velocity_data.columns:
            acceleration_data['composite_acceleration'] = velocity_data['composite_velocity'].diff()
            acceleration_data['composite_jerk'] = acceleration_data['composite_acceleration'].diff()
        
        # Smoothed acceleration
        for period in [5, 10, 20]:
            if 'composite_acceleration' in acceleration_data.columns:
                acceleration_data[f'smooth_acceleration_{period}'] = (
                    acceleration_data['composite_acceleration'].ewm(span=period).mean()
                )
        
        # Acceleration momentum
        acceleration_data['acceleration_momentum'] = (
            acceleration_data.get('composite_acceleration', pd.Series(0, index=velocity_data.index)).rolling(window=5).mean()
        )
        
        return acceleration_data.fillna(0)
    
    def _apply_advanced_filtering(self, velocity_data: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Apply advanced filtering to velocity data"""
        filtered_data = pd.DataFrame(index=velocity_data.index)
        
        # Butterworth filter for noise reduction
        for col in velocity_data.columns:
            if 'velocity' in col and not velocity_data[col].isna().all():
                try:
                    # Design Butterworth filter
                    nyquist = 0.5
                    low_cutoff = 0.1
                    high_cutoff = 0.3
                    
                    # Low-pass filter
                    b_low, a_low = scipy_signal.butter(3, low_cutoff/nyquist, btype='low')
                    filtered_low = scipy_signal.filtfilt(b_low, a_low, velocity_data[col].fillna(0))
                    
                    # High-pass filter
                    b_high, a_high = scipy_signal.butter(3, high_cutoff/nyquist, btype='high')
                    filtered_high = scipy_signal.filtfilt(b_high, a_high, velocity_data[col].fillna(0))
                    
                    # Band-pass combination
                    filtered_data[f'{col}_filtered'] = pd.Series(filtered_low + filtered_high * 0.3, index=velocity_data.index)
                    
                except Exception:
                    # Fallback to simple exponential smoothing
                    filtered_data[f'{col}_filtered'] = velocity_data[col].ewm(span=10).mean()
        
        # Kalman-like adaptive filtering
        if self.adaptive_smoothing:
            filtered_data = self._apply_adaptive_filtering(filtered_data, data)
        
        return filtered_data.fillna(0)
    
    def _apply_adaptive_filtering(self, filtered_data: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Apply adaptive filtering based on market conditions"""
        # Market volatility for adaptive parameters
        returns = data['close'].pct_change()
        volatility = returns.rolling(window=20).std()
        vol_percentile = volatility.rolling(window=60).rank(pct=True)
        
        adaptive_filtered = filtered_data.copy()
        
        for col in filtered_data.columns:
            if not filtered_data[col].isna().all():
                # Adaptive smoothing based on volatility
                adaptive_span = 10 * (1 + (vol_percentile - 0.5) * 0.5)
                adaptive_span = adaptive_span.fillna(10).clip(lower=5, upper=30)
                
                # Apply adaptive exponential smoothing
                adaptive_series = pd.Series(index=filtered_data.index, dtype=float)
                adaptive_series.iloc[0] = filtered_data[col].iloc[0] if not pd.isna(filtered_data[col].iloc[0]) else 0
                
                for i in range(1, len(filtered_data)):
                    alpha = 2.0 / (adaptive_span.iloc[i] + 1) if not pd.isna(adaptive_span.iloc[i]) else 0.1
                    current_value = filtered_data[col].iloc[i] if not pd.isna(filtered_data[col].iloc[i]) else 0
                    adaptive_series.iloc[i] = (alpha * current_value + 
                                             (1 - alpha) * adaptive_series.iloc[i-1])
                
                adaptive_filtered[col] = adaptive_series
        
        return adaptive_filtered
    
    def _detect_velocity_regimes(self, data: pd.DataFrame, velocity_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect different velocity regimes in the market"""
        if not self.regime_awareness:
            return {'regime': 'normal', 'confidence': 0.5}
        
        # Prepare features for regime detection
        regime_features = []
        
        # Velocity statistics
        for period in self.velocity_periods[:3]:  # Use first 3 periods to avoid over-fitting
            if f'price_velocity_{period}' in velocity_data.columns:
                vel_col = velocity_data[f'price_velocity_{period}']
                regime_features.extend([
                    vel_col.rolling(window=20).mean(),
                    vel_col.rolling(window=20).std(),
                    vel_col.rolling(window=20).quantile(0.75) - vel_col.rolling(window=20).quantile(0.25)
                ])
        
        # Market characteristics
        returns = data['close'].pct_change()
        regime_features.extend([
            returns.rolling(window=20).mean(),
            returns.rolling(window=20).std(),
            (data['high'] - data['low']).rolling(window=20).mean() / data['close'].rolling(window=20).mean()
        ])
        
        # Volume characteristics (if available)
        if 'volume' in data.columns:
            regime_features.extend([
                data['volume'].rolling(window=20).mean(),
                data['volume'].rolling(window=20).std()
            ])
        
        # Create feature matrix
        feature_matrix = pd.DataFrame(regime_features).T
        feature_matrix = feature_matrix.fillna(feature_matrix.mean())
        
        # Normalize features
        if len(feature_matrix) > 50:
            try:
                features_scaled = self.velocity_scaler.fit_transform(feature_matrix.iloc[-100:])
                
                # Cluster analysis for regime detection
                clusters = self.regime_clusterer.fit_predict(features_scaled)
                
                # Analyze current regime
                current_cluster = clusters[-1] if len(clusters) > 0 else 0
                cluster_stability = np.sum(clusters[-20:] == current_cluster) / 20 if len(clusters) >= 20 else 0.5
                
                # Regime classification based on velocity characteristics
                current_velocity = velocity_data['composite_velocity'].iloc[-1] if len(velocity_data) > 0 else 0
                velocity_volatility = velocity_data['composite_velocity'].rolling(window=20).std().iloc[-1]
                
                if abs(current_velocity) > 0.02 and velocity_volatility < 0.01:
                    regime = 'trending'
                elif velocity_volatility > 0.03:
                    regime = 'volatile'
                elif abs(current_velocity) < 0.005:
                    regime = 'sideways'
                else:
                    regime = 'normal'
                
                return {
                    'regime': regime,
                    'confidence': float(cluster_stability),
                    'current_cluster': int(current_cluster),
                    'velocity_volatility': float(velocity_volatility) if not pd.isna(velocity_volatility) else 0.01,
                    'current_velocity': float(current_velocity) if not pd.isna(current_velocity) else 0.0
                }
                
            except Exception:
                pass
        
        # Fallback simple regime detection
        velocity_mean = velocity_data['composite_velocity'].rolling(window=20).mean().iloc[-1]
        velocity_std = velocity_data['composite_velocity'].rolling(window=20).std().iloc[-1]
        
        if pd.isna(velocity_mean) or pd.isna(velocity_std):
            return {'regime': 'normal', 'confidence': 0.5}
        
        if abs(velocity_mean) > 2 * velocity_std:
            regime = 'trending'
        elif velocity_std > abs(velocity_mean) * 3:
            regime = 'volatile'
        else:
            regime = 'normal'
        
        return {
            'regime': regime,
            'confidence': min(1.0, abs(velocity_mean) / (velocity_std + 1e-6)),
            'velocity_mean': float(velocity_mean),
            'velocity_std': float(velocity_std)
        }
    
    def _calculate_adaptive_parameters(self, data: pd.DataFrame, velocity_data: pd.DataFrame,
                                     regime_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate adaptive parameters based on velocity characteristics and regime"""
        # Base thresholds
        base_threshold = 0.01  # 1% velocity threshold
        
        # Regime-based adjustments
        regime_multipliers = {
            'trending': 1.5,
            'volatile': 0.7,
            'sideways': 2.0,
            'normal': 1.0
        }
        
        regime_multiplier = regime_multipliers.get(regime_analysis['regime'], 1.0)
        
        # Volatility-based adjustment
        velocity_volatility = regime_analysis.get('velocity_volatility', 0.01)
        volatility_adjustment = np.clip(velocity_volatility / 0.02, 0.5, 2.0)
        
        # Adaptive thresholds
        buy_threshold = base_threshold * regime_multiplier * volatility_adjustment
        sell_threshold = -base_threshold * regime_multiplier * volatility_adjustment
        
        # Signal sensitivity
        sensitivity = 1.0 / volatility_adjustment * regime_analysis.get('confidence', 0.5)
        
        return {
            'buy_threshold': buy_threshold,
            'sell_threshold': sell_threshold,
            'sensitivity': np.clip(sensitivity, 0.3, 3.0),
            'regime_multiplier': regime_multiplier,
            'volatility_adjustment': volatility_adjustment
        }
    
    def _generate_base_velocity_signals(self, data: pd.DataFrame, velocity_data: pd.DataFrame,
                                       acceleration_data: pd.DataFrame, adaptive_params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate base velocity signals"""
        buy_threshold = adaptive_params['buy_threshold']
        sell_threshold = adaptive_params['sell_threshold']
        sensitivity = adaptive_params['sensitivity']
        
        # Primary velocity signals
        primary_velocity = velocity_data.get('composite_velocity', pd.Series(0, index=data.index))
        
        # Basic velocity signals
        velocity_buy = primary_velocity > buy_threshold
        velocity_sell = primary_velocity < sell_threshold
        
        # Acceleration-based signals
        acceleration = acceleration_data.get('composite_acceleration', pd.Series(0, index=data.index))
        acceleration_buy = (primary_velocity > 0) & (acceleration > 0)
        acceleration_sell = (primary_velocity < 0) & (acceleration < 0)
        
        # Multi-timeframe velocity convergence
        mtf_convergence = self._calculate_velocity_convergence(velocity_data)
        convergence_buy = (mtf_convergence > 0.6) & velocity_buy
        convergence_sell = (mtf_convergence < -0.6) & velocity_sell
        
        # Velocity momentum signals
        velocity_momentum = primary_velocity.rolling(window=5).mean()
        momentum_buy = (velocity_momentum > buy_threshold * 0.5) & (velocity_momentum.diff() > 0)
        momentum_sell = (velocity_momentum < sell_threshold * 0.5) & (velocity_momentum.diff() < 0)
        
        # Signal strength calculation
        signal_strength = np.abs(primary_velocity) * sensitivity
        
        # Velocity extremes
        velocity_extreme_buy = primary_velocity > buy_threshold * 2
        velocity_extreme_sell = primary_velocity < sell_threshold * 2
        
        return {
            'velocity_buy': velocity_buy,
            'velocity_sell': velocity_sell,
            'acceleration_buy': acceleration_buy,
            'acceleration_sell': acceleration_sell,
            'convergence_buy': convergence_buy,
            'convergence_sell': convergence_sell,
            'momentum_buy': momentum_buy,
            'momentum_sell': momentum_sell,
            'velocity_extreme_buy': velocity_extreme_buy,
            'velocity_extreme_sell': velocity_extreme_sell,
            'signal_strength': signal_strength,
            'mtf_convergence': mtf_convergence,
            'primary_velocity': primary_velocity
        }
    
    def _calculate_velocity_convergence(self, velocity_data: pd.DataFrame) -> pd.Series:
        """Calculate multi-timeframe velocity convergence"""
        convergence = pd.Series(0.0, index=velocity_data.index)
        
        # Check convergence across different velocity periods
        velocity_directions = []
        
        for period in self.velocity_periods:
            if f'price_velocity_{period}' in velocity_data.columns:
                direction = np.sign(velocity_data[f'price_velocity_{period}'])
                velocity_directions.append(direction)
        
        if velocity_directions:
            # Calculate agreement across timeframes
            velocity_matrix = np.column_stack(velocity_directions)
            agreement = np.mean(velocity_matrix, axis=1)
            convergence = pd.Series(agreement, index=velocity_data.index)
        
        return convergence
    
    def _train_ml_models(self, data: pd.DataFrame, velocity_data: pd.DataFrame,
                        acceleration_data: pd.DataFrame) -> None:
        """Train ML models for velocity prediction and pattern recognition"""
        try:
            # Prepare features
            features = self._prepare_velocity_ml_features(data, velocity_data, acceleration_data)
            
            if len(features) < self.ml_lookback // 2:
                return
            
            # Prepare labels
            future_velocity = velocity_data['composite_velocity'].shift(-5)
            future_returns = data['close'].pct_change(5).shift(-5)
            
            # Classification labels for acceleration patterns
            acceleration_labels = np.where(
                acceleration_data.get('composite_acceleration', pd.Series(0)).shift(-3) > 0.001, 1,
                np.where(acceleration_data.get('composite_acceleration', pd.Series(0)).shift(-3) < -0.001, -1, 0)
            )
            
            # Remove NaN values
            valid_indices = ~(np.isnan(future_velocity) | np.any(np.isnan(features), axis=1) |
                            np.isnan(future_returns) | np.isnan(acceleration_labels))
            
            features_clean = features[valid_indices]
            velocity_labels_clean = future_velocity[valid_indices]
            return_labels_clean = future_returns[valid_indices]
            acceleration_labels_clean = acceleration_labels[valid_indices]
            
            if len(features_clean) < 50:
                return
            
            # Scale features
            features_scaled = self.feature_scaler.fit_transform(features_clean)
            
            # Train models
            self.velocity_predictor.fit(features_scaled, velocity_labels_clean)
            self.acceleration_classifier.fit(features_scaled, acceleration_labels_clean)
            self.pattern_regressor.fit(features_scaled, return_labels_clean)
            
            # Train momentum predictor
            momentum_features = features_scaled[:, :min(10, features_scaled.shape[1])]
            self.momentum_predictor.fit(momentum_features, velocity_labels_clean)
            
            self.is_trained = True
            self.feature_columns = [f'feature_{i}' for i in range(features_scaled.shape[1])]
            
        except Exception as e:
            print(f"ML training error: {e}")
    
    def _prepare_velocity_ml_features(self, data: pd.DataFrame, velocity_data: pd.DataFrame,
                                    acceleration_data: pd.DataFrame) -> np.ndarray:
        """Prepare features for ML models"""
        features = []
        
        # Velocity features
        features.append(velocity_data.get('composite_velocity', pd.Series(0)).values)
        
        # Multi-period velocities
        for period in self.velocity_periods[:4]:  # Limit to prevent overfitting
            if f'price_velocity_{period}' in velocity_data.columns:
                features.append(velocity_data[f'price_velocity_{period}'].values)
            if f'smooth_velocity_{period}' in velocity_data.columns:
                features.append(velocity_data[f'smooth_velocity_{period}'].values)
        
        # Acceleration features
        features.append(acceleration_data.get('composite_acceleration', pd.Series(0)).values)
        features.append(acceleration_data.get('composite_jerk', pd.Series(0)).values)
        features.append(acceleration_data.get('acceleration_momentum', pd.Series(0)).values)
        
        # Price features
        features.append(data['close'].pct_change().values)
        features.append(data['close'].pct_change(5).values)
        features.append(data['close'].rolling(window=10).mean().values)
        features.append(data['close'].rolling(window=20).std().values)
        
        # Range and volatility features
        price_range = (data['high'] - data['low']) / data['close']
        features.append(price_range.values)
        features.append(price_range.rolling(window=10).mean().values)
        
        # Volume features (if available)
        if 'volume' in data.columns:
            features.append(data['volume'].rolling(window=10).mean().values)
            features.append(data['volume'].pct_change().values)
            
            # Volume-price velocity
            for period in self.velocity_periods[:2]:
                if f'vp_velocity_{period}' in velocity_data.columns:
                    features.append(velocity_data[f'vp_velocity_{period}'].values)
        
        return np.column_stack(features)
    
    def _generate_ml_predictions(self, data: pd.DataFrame, velocity_data: pd.DataFrame,
                               acceleration_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate ML-enhanced predictions"""
        if not self.is_trained:
            return {
                'predicted_velocity': pd.Series(0, index=data.index),
                'predicted_acceleration': pd.Series(0, index=data.index),
                'pattern_prediction': pd.Series(0, index=data.index),
                'momentum_prediction': pd.Series(0, index=data.index),
                'ml_confidence': pd.Series(0.5, index=data.index)
            }
        
        try:
            # Prepare features
            features = self._prepare_velocity_ml_features(data, velocity_data, acceleration_data)
            features_scaled = self.feature_scaler.transform(features)
            
            # Generate predictions
            velocity_pred = self.velocity_predictor.predict(features_scaled)
            acceleration_pred = self.acceleration_classifier.predict(features_scaled)
            pattern_pred = self.pattern_regressor.predict(features_scaled)
            
            # Momentum prediction with reduced features
            momentum_features = features_scaled[:, :min(10, features_scaled.shape[1])]
            momentum_pred = self.momentum_predictor.predict(momentum_features)
            
            # Calculate confidence from ensemble agreement
            velocity_confidence = self._calculate_prediction_confidence(
                velocity_pred, acceleration_pred, pattern_pred, momentum_pred
            )
            
            return {
                'predicted_velocity': pd.Series(velocity_pred, index=data.index),
                'predicted_acceleration': pd.Series(acceleration_pred, index=data.index),
                'pattern_prediction': pd.Series(pattern_pred, index=data.index),
                'momentum_prediction': pd.Series(momentum_pred, index=data.index),
                'ml_confidence': pd.Series(velocity_confidence, index=data.index)
            }
            
        except Exception as e:
            print(f"ML prediction error: {e}")
            return {
                'predicted_velocity': pd.Series(0, index=data.index),
                'predicted_acceleration': pd.Series(0, index=data.index),
                'pattern_prediction': pd.Series(0, index=data.index),
                'momentum_prediction': pd.Series(0, index=data.index),
                'ml_confidence': pd.Series(0.5, index=data.index)
            }
    
    def _calculate_prediction_confidence(self, velocity_pred: np.ndarray, acceleration_pred: np.ndarray,
                                       pattern_pred: np.ndarray, momentum_pred: np.ndarray) -> np.ndarray:
        """Calculate confidence from prediction ensemble agreement"""
        # Normalize predictions to comparable ranges
        velocity_norm = np.clip(velocity_pred / 0.05, -1, 1)
        pattern_norm = np.clip(pattern_pred / 0.05, -1, 1)
        momentum_norm = np.clip(momentum_pred / 0.05, -1, 1)
        acceleration_norm = np.clip(acceleration_pred, -1, 1)
        
        # Calculate agreement between predictions
        agreements = []
        
        # Velocity-pattern agreement
        agreements.append(1 - abs(velocity_norm - pattern_norm) / 2)
        
        # Velocity-momentum agreement  
        agreements.append(1 - abs(velocity_norm - momentum_norm) / 2)
        
        # Pattern-momentum agreement
        agreements.append(1 - abs(pattern_norm - momentum_norm) / 2)
        
        # Velocity-acceleration consistency
        velocity_accel_consistency = 1 - abs(np.sign(velocity_norm) - acceleration_norm) / 2
        agreements.append(velocity_accel_consistency)
        
        # Overall confidence as mean agreement
        confidence = np.mean(agreements, axis=0)
        return np.clip(confidence, 0.1, 1.0)
    
    def _perform_pattern_recognition(self, data: pd.DataFrame, velocity_data: pd.DataFrame,
                                   acceleration_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform advanced pattern recognition on velocity data"""
        if not self.pattern_recognition:
            return {'patterns': {}, 'pattern_strength': pd.Series(0, index=data.index)}
        
        patterns = {}
        
        # Velocity reversal patterns
        patterns['velocity_reversal'] = self._detect_velocity_reversal_patterns(velocity_data)
        
        # Acceleration divergence patterns
        patterns['acceleration_divergence'] = self._detect_acceleration_divergence_patterns(
            data, velocity_data, acceleration_data
        )
        
        # Momentum exhaustion patterns
        patterns['momentum_exhaustion'] = self._detect_momentum_exhaustion_patterns(
            velocity_data, acceleration_data
        )
        
        # Velocity channel patterns
        patterns['velocity_channels'] = self._detect_velocity_channel_patterns(velocity_data)
        
        # Calculate overall pattern strength
        pattern_strength = self._calculate_pattern_strength(patterns)
        
        return {
            'patterns': patterns,
            'pattern_strength': pattern_strength,
            'pattern_count': sum(1 for p in patterns.values() if p.any() if hasattr(p, 'any') else False)
        }
    
    def _detect_velocity_reversal_patterns(self, velocity_data: pd.DataFrame) -> pd.Series:
        """Detect velocity reversal patterns"""
        velocity = velocity_data.get('composite_velocity', pd.Series(0, index=velocity_data.index))
        
        # Look for velocity sign changes with confirmation
        velocity_sign = np.sign(velocity)
        sign_changes = velocity_sign.diff() != 0
        
        # Confirm with acceleration
        acceleration = velocity.diff()
        acceleration_confirmation = (
            ((velocity > 0) & (acceleration < 0)) |  # Positive velocity slowing
            ((velocity < 0) & (acceleration > 0))    # Negative velocity slowing
        )
        
        reversal_patterns = sign_changes & acceleration_confirmation
        
        # Filter for significant reversals
        velocity_magnitude = abs(velocity)
        significant_reversals = reversal_patterns & (velocity_magnitude > velocity_magnitude.rolling(window=20).mean())
        
        return significant_reversals
    
    def _detect_acceleration_divergence_patterns(self, data: pd.DataFrame, velocity_data: pd.DataFrame,
                                               acceleration_data: pd.DataFrame) -> pd.Series:
        """Detect divergence between price and acceleration"""
        price_direction = np.sign(data['close'].diff())
        acceleration = acceleration_data.get('composite_acceleration', pd.Series(0, index=data.index))
        
        # Divergence: price and acceleration moving in opposite directions
        divergence = (
            ((price_direction > 0) & (acceleration < 0)) |  # Price up, acceleration down
            ((price_direction < 0) & (acceleration > 0))    # Price down, acceleration up
        )
        
        # Confirm with velocity magnitude
        velocity_magnitude = abs(velocity_data.get('composite_velocity', pd.Series(0, index=data.index)))
        significant_divergence = divergence & (velocity_magnitude > velocity_magnitude.rolling(window=10).quantile(0.7))
        
        return significant_divergence
    
    def _detect_momentum_exhaustion_patterns(self, velocity_data: pd.DataFrame,
                                          acceleration_data: pd.DataFrame) -> pd.Series:
        """Detect momentum exhaustion patterns"""
        velocity = velocity_data.get('composite_velocity', pd.Series(0, index=velocity_data.index))
        acceleration = acceleration_data.get('composite_acceleration', pd.Series(0, index=acceleration_data.index))
        jerk = acceleration_data.get('composite_jerk', pd.Series(0, index=acceleration_data.index))
        
        # Momentum exhaustion: high velocity, negative acceleration, positive jerk
        exhaustion_patterns = (
            (abs(velocity) > abs(velocity).rolling(window=20).quantile(0.8)) &  # High velocity
            ((velocity > 0) & (acceleration < 0) & (jerk > 0)) |  # Positive momentum exhausting
            ((velocity < 0) & (acceleration > 0) & (jerk < 0))    # Negative momentum exhausting
        )
        
        return exhaustion_patterns
    
    def _detect_velocity_channel_patterns(self, velocity_data: pd.DataFrame) -> pd.Series:
        """Detect velocity channel breakout patterns"""
        velocity = velocity_data.get('composite_velocity', pd.Series(0, index=velocity_data.index))
        
        # Calculate velocity channels
        velocity_upper = velocity.rolling(window=20).quantile(0.8)
        velocity_lower = velocity.rolling(window=20).quantile(0.2)
        velocity_mean = velocity.rolling(window=20).mean()
        
        # Channel breakouts
        upper_breakout = velocity > velocity_upper
        lower_breakout = velocity < velocity_lower
        
        # Confirm breakouts with volume if available
        channel_patterns = upper_breakout | lower_breakout
        
        return channel_patterns
    
    def _calculate_pattern_strength(self, patterns: Dict[str, Any]) -> pd.Series:
        """Calculate overall pattern strength"""
        if not patterns:
            return pd.Series(0, index=patterns.get('velocity_reversal', pd.Series()).index)
        
        pattern_strength = pd.Series(0.0, index=list(patterns.values())[0].index)
        
        weights = {
            'velocity_reversal': 0.3,
            'acceleration_divergence': 0.25,
            'momentum_exhaustion': 0.25,
            'velocity_channels': 0.2
        }
        
        for pattern_name, pattern_series in patterns.items():
            if hasattr(pattern_series, 'astype'):
                weight = weights.get(pattern_name, 0.2)
                pattern_strength += pattern_series.astype(float) * weight
        
        return pattern_strength
    
    def _analyze_signal_strength(self, data: pd.DataFrame, velocity_data: pd.DataFrame,
                                base_signals: Dict[str, Any], ml_predictions: Dict[str, Any],
                                pattern_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze overall signal strength and confidence"""
        # Combine signal sources
        base_strength = base_signals['signal_strength']
        ml_confidence = ml_predictions['ml_confidence']
        pattern_strength = pattern_analysis['pattern_strength']
        
        # Velocity signal combination
        velocity_signal = np.where(
            base_signals['velocity_buy'] | base_signals['convergence_buy'], 1,
            np.where(base_signals['velocity_sell'] | base_signals['convergence_sell'], -1, 0)
        )
        
        # ML signal combination
        ml_signal = np.where(
            ml_predictions['predicted_velocity'] > 0.01, 1,
            np.where(ml_predictions['predicted_velocity'] < -0.01, -1, 0)
        )
        
        # Combined signal
        combined_signal = (
            velocity_signal * 0.4 +
            ml_signal * 0.4 +
            np.sign(base_signals['mtf_convergence']) * 0.2
        )
        
        # Overall confidence
        overall_confidence = (
            base_strength * 0.3 +
            ml_confidence * 0.4 +
            pattern_strength * 0.3
        )
        
        # Signal strength
        signal_strength = np.abs(combined_signal) * overall_confidence
        
        return {
            'combined_signal': pd.Series(combined_signal, index=data.index),
            'overall_confidence': overall_confidence,
            'signal_strength': signal_strength,
            'velocity_contribution': base_strength * 0.3,
            'ml_contribution': ml_confidence * 0.4,
            'pattern_contribution': pattern_strength * 0.3
        }
    
    def _synthesize_velocity_signals(self, data: pd.DataFrame, velocity_data: pd.DataFrame,
                                   signal_analysis: Dict[str, Any], regime_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize final velocity signals"""
        combined_signal = signal_analysis['combined_signal']
        confidence = signal_analysis['overall_confidence']
        signal_strength = signal_analysis['signal_strength']
        
        # Regime-based signal adjustment
        regime_confidence = regime_analysis.get('confidence', 0.5)
        regime_adjusted_signal = combined_signal * (0.5 + regime_confidence * 0.5)
        
        # Generate final trading signals
        strong_buy = (regime_adjusted_signal > 0.7) & (confidence > self.confidence_threshold)
        buy = (regime_adjusted_signal > 0.4) & (confidence > self.confidence_threshold * 0.8)
        strong_sell = (regime_adjusted_signal < -0.7) & (confidence > self.confidence_threshold)
        sell = (regime_adjusted_signal < -0.4) & (confidence > self.confidence_threshold * 0.8)
        
        # Position sizing based on signal strength and regime
        base_position_size = signal_strength * confidence
        regime_multiplier = min(1.0, regime_confidence + 0.5)
        position_size = base_position_size * regime_multiplier
        
        # Signal quality
        signal_quality = confidence * signal_strength * regime_confidence
        
        return {
            'final_signal': regime_adjusted_signal,
            'strong_buy': strong_buy,
            'buy': buy,
            'strong_sell': strong_sell,
            'sell': sell,
            'position_size': position_size,
            'signal_quality': signal_quality,
            'regime_adjustment': regime_confidence
        }
    
    def _calculate_velocity_risk_metrics(self, data: pd.DataFrame, velocity_data: pd.DataFrame,
                                       synthesized_signals: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate velocity-specific risk metrics"""
        # Velocity volatility
        velocity_volatility = velocity_data.get('composite_velocity', pd.Series(0)).rolling(window=20).std()
        
        # Signal consistency
        signal_consistency = 1 - synthesized_signals['final_signal'].rolling(window=10).std().fillna(0.5)
        
        # Velocity extremes risk
        velocity_magnitude = abs(velocity_data.get('composite_velocity', pd.Series(0)))
        velocity_percentile = velocity_magnitude.rolling(window=100).rank(pct=True)
        extreme_velocity_risk = np.where(velocity_percentile > 0.95, 0.8, 0.2)
        
        # Overall velocity risk
        overall_risk = (
            np.clip(velocity_volatility.fillna(0.01) / 0.02, 0, 1) * 0.4 +
            (1 - signal_consistency) * 0.3 +
            extreme_velocity_risk * 0.3
        )
        
        return {
            'velocity_volatility': velocity_volatility.iloc[-1] if len(velocity_volatility) > 0 else 0.01,
            'signal_consistency': float(signal_consistency.iloc[-1]) if len(signal_consistency) > 0 else 0.5,
            'extreme_velocity_risk': pd.Series(extreme_velocity_risk, index=data.index),
            'overall_risk': pd.Series(overall_risk, index=data.index)
        }
    
    def _generate_velocity_recommendations(self, synthesized_signals: Dict[str, Any],
                                         risk_metrics: Dict[str, Any], regime_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final velocity-based trading recommendations"""
        final_signal = synthesized_signals['final_signal']
        risk_score = risk_metrics['overall_risk']
        position_size = synthesized_signals['position_size']
        
        # Risk-adjusted position sizing
        risk_adjustment = 1 - risk_score
        adjusted_position = position_size * risk_adjustment
        
        # Regime-based final adjustments
        regime = regime_analysis.get('regime', 'normal')
        regime_multipliers = {
            'trending': 1.2,
            'volatile': 0.6,
            'sideways': 0.8,
            'normal': 1.0
        }
        
        regime_multiplier = regime_multipliers.get(regime, 1.0)
        final_position = np.clip(adjusted_position * regime_multiplier, 0, 1)
        
        # Generate actions
        action = np.where(
            synthesized_signals['strong_buy'], 'STRONG_BUY',
            np.where(synthesized_signals['buy'], 'BUY',
                    np.where(synthesized_signals['strong_sell'], 'STRONG_SELL',
                            np.where(synthesized_signals['sell'], 'SELL', 'HOLD')))
        )
        
        # Entry and exit conditions
        entry_signals = synthesized_signals['strong_buy'] | synthesized_signals['strong_sell']
        exit_signals = (abs(final_signal) < 0.3) | (risk_score > 0.7)
        
        return {
            'action': action,
            'position_size': final_position,
            'entry_signals': entry_signals,
            'exit_signals': exit_signals,
            'risk_level': np.where(risk_score > 0.7, 'HIGH',
                                 np.where(risk_score > 0.4, 'MEDIUM', 'LOW')),
            'regime_factor': regime_multiplier,
            'confidence_level': synthesized_signals['signal_quality'],
            'velocity_direction': np.where(final_signal > 0, 'ACCELERATING_UP', 
                                         np.where(final_signal < 0, 'ACCELERATING_DOWN', 'STABLE'))
        }
    
    def _generate_metadata(self, data: pd.DataFrame, velocity_data: pd.DataFrame,
                          synthesized_signals: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive metadata"""
        return {
            'indicator_name': 'VelocityIndicator',
            'version': '1.0.0',
            'parameters': {
                'velocity_periods': self.velocity_periods,
                'smoothing_period': self.smoothing_period,
                'acceleration_period': self.acceleration_period,
                'ml_lookback': self.ml_lookback,
                'confidence_threshold': self.confidence_threshold
            },
            'data_points': len(data),
            'calculation_time': pd.Timestamp.now().isoformat(),
            'ml_model_trained': self.is_trained,
            'feature_count': len(self.feature_columns),
            'advanced_features': {
                'adaptive_smoothing': self.adaptive_smoothing,
                'volume_weighting': self.volume_weighting,
                'regime_awareness': self.regime_awareness,
                'pattern_recognition': self.pattern_recognition
            },
            'signal_distribution': {
                'strong_buy_count': int(synthesized_signals['strong_buy'].sum()),
                'buy_count': int(synthesized_signals['buy'].sum()),
                'strong_sell_count': int(synthesized_signals['strong_sell'].sum()),
                'sell_count': int(synthesized_signals['sell'].sum())
            },
            'performance_metrics': {
                'avg_signal_strength': float(synthesized_signals['signal_quality'].mean()),
                'max_signal_strength': float(synthesized_signals['signal_quality'].max()),
                'signal_consistency': float(synthesized_signals['signal_quality'].std()),
                'position_size_avg': float(synthesized_signals['position_size'].mean())
            },
            'velocity_statistics': {
                'current_velocity': float(velocity_data.get('composite_velocity', pd.Series(0)).iloc[-1]) if len(velocity_data) > 0 else 0.0,
                'velocity_range': float(velocity_data.get('composite_velocity', pd.Series(0)).max() - 
                                      velocity_data.get('composite_velocity', pd.Series(0)).min()) if len(velocity_data) > 0 else 0.0,
                'velocity_volatility': float(velocity_data.get('composite_velocity', pd.Series(0)).std()) if len(velocity_data) > 0 else 0.0
            }
        }
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error result"""
        return {
            'error': error_message,
            'velocity_data': pd.DataFrame(),
            'base_signals': {},
            'ml_predictions': {},
            'synthesized_signals': {},
            'recommendations': {},
            'confidence_score': 0.0,
            'signal_strength': 0.0,
            'metadata': {
                'indicator_name': 'VelocityIndicator',
                'error': error_message,
                'calculation_time': pd.Timestamp.now().isoformat()
            }
        }
