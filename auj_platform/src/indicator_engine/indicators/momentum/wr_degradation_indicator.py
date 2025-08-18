"""
Advanced Williams %R Degradation Indicator with Machine Learning Integration

This module implements a sophisticated Williams %R degradation analysis system with:
- Advanced degradation pattern detection and momentum analysis
- Machine learning degradation prediction and signal enhancement
- Multi-timeframe degradation convergence analysis
- Risk assessment and confidence scoring with position sizing
- Adaptive parameters and regime-aware filtering

Part of the ASD trading platform's humanitarian mission.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, IsolationForest
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import FastICA
from scipy import signal as scipy_signal
from scipy.stats import entropy, skew, kurtosis
import warnings
warnings.filterwarnings('ignore')

from ..base.standard_indicator import StandardIndicatorInterface


class WRDegradationIndicator(StandardIndicatorInterface):
    """
    Advanced Williams %R Degradation Indicator with Machine Learning
    
    Features:
    - Multi-dimensional degradation analysis (momentum, volatility, signal quality)
    - ML-based degradation pattern recognition and prediction
    - Advanced filtering and anomaly detection
    - Multi-timeframe degradation convergence analysis
    - Risk and confidence scoring with adaptive position sizing
    - Regime-aware degradation interpretation
    """
    
    def __init__(self, 
                 wr_periods: List[int] = [14, 21, 50],
                 degradation_window: int = 20,
                 smoothing_period: int = 5,
                 ml_lookback: int = 252,
                 anomaly_threshold: float = 0.1,
                 confidence_threshold: float = 0.6):
        """
        Initialize Advanced WR Degradation Indicator
        
        Args:
            wr_periods: Williams %R calculation periods
            degradation_window: Window for degradation analysis
            smoothing_period: Period for signal smoothing
            ml_lookback: Lookback period for ML training
            anomaly_threshold: Threshold for anomaly detection
            confidence_threshold: Minimum confidence for signal generation
        """
        super().__init__()
        
        self.wr_periods = wr_periods
        self.degradation_window = degradation_window
        self.smoothing_period = smoothing_period
        self.ml_lookback = ml_lookback
        self.anomaly_threshold = anomaly_threshold
        self.confidence_threshold = confidence_threshold
        
        # ML components for degradation analysis
        self.degradation_classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            random_state=42
        )
        self.quality_regressor = GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=8,
            random_state=42
        )
        self.pattern_detector = MLPClassifier(
            hidden_layer_sizes=(64, 32, 16),
            max_iter=1000,
            random_state=42
        )
        self.anomaly_detector = IsolationForest(
            contamination=anomaly_threshold,
            random_state=42
        )
        
        # Scalers
        self.feature_scaler = StandardScaler()
        self.degradation_scaler = MinMaxScaler()
        
        # Clustering for pattern recognition
        self.pattern_clusterer = DBSCAN(eps=0.3, min_samples=5)
        self.ica = FastICA(n_components=None, random_state=42)
        
        # State tracking
        self.is_trained = False
        self.feature_columns = []
        self.degradation_history = []
        
        # Advanced features
        self.adaptive_analysis = True
        self.anomaly_detection = True
        self.regime_awareness = True
        self.quality_assessment = True
        
    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate advanced WR degradation analysis with ML integration
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary containing degradation analysis, signals, and metadata
        """
        try:
            if len(data) < max(self.wr_periods + [self.ml_lookback // 2]):
                return self._create_error_result("Insufficient data for WR degradation calculation")
            
            # Calculate Williams %R components
            wr_data = self._calculate_multi_period_wr(data)
            
            # Analyze degradation patterns
            degradation_analysis = self._analyze_degradation_patterns(data, wr_data)
            
            # Perform quality assessment
            quality_analysis = self._assess_signal_quality(data, wr_data, degradation_analysis)
            
            # Detect anomalies and outliers
            anomaly_analysis = self._detect_anomalies(data, wr_data, degradation_analysis)
            
            # Calculate adaptive parameters
            adaptive_params = self._calculate_adaptive_parameters(data, wr_data, degradation_analysis)
            
            # Generate base degradation signals
            base_signals = self._generate_base_degradation_signals(
                data, wr_data, degradation_analysis, adaptive_params
            )
            
            # Train or update ML models
            if not self.is_trained and len(data) >= self.ml_lookback:
                self._train_ml_models(data, wr_data, degradation_analysis)
            
            # Generate ML-enhanced predictions
            ml_predictions = self._generate_ml_predictions(data, wr_data, degradation_analysis)
            
            # Perform regime analysis
            regime_analysis = self._analyze_degradation_regime(data, wr_data, degradation_analysis)
            
            # Calculate signal strength and confidence
            signal_analysis = self._analyze_signal_strength(
                data, wr_data, base_signals, ml_predictions, quality_analysis
            )
            
            # Synthesize degradation signals
            synthesized_signals = self._synthesize_degradation_signals(
                data, wr_data, signal_analysis, regime_analysis
            )
            
            # Calculate risk metrics
            risk_metrics = self._calculate_degradation_risk_metrics(
                data, wr_data, synthesized_signals, degradation_analysis
            )
            
            # Generate final recommendations
            recommendations = self._generate_degradation_recommendations(
                synthesized_signals, risk_metrics, regime_analysis
            )
            
            return {
                'wr_data': wr_data,
                'degradation_analysis': degradation_analysis,
                'quality_analysis': quality_analysis,
                'anomaly_analysis': anomaly_analysis,
                'base_signals': base_signals,
                'ml_predictions': ml_predictions,
                'signal_analysis': signal_analysis,
                'synthesized_signals': synthesized_signals,
                'regime_analysis': regime_analysis,
                'risk_metrics': risk_metrics,
                'recommendations': recommendations,
                'adaptive_parameters': adaptive_params,
                'confidence_score': float(signal_analysis.get('overall_confidence', 0.0)),
                'signal_strength': float(signal_analysis.get('signal_strength', 0.0)),
                'metadata': self._generate_metadata(data, wr_data, synthesized_signals)
            }
            
        except Exception as e:
            return self._create_error_result(f"WR degradation calculation error: {str(e)}")
    
    def _calculate_multi_period_wr(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Williams %R for multiple periods"""
        wr_data = pd.DataFrame(index=data.index)
        
        # Calculate Williams %R for each period
        for period in self.wr_periods:
            wr_data[f'wr_{period}'] = self._calculate_williams_r(data, period)
        
        # Primary Williams %R (first period)
        wr_data['wr_primary'] = wr_data[f'wr_{self.wr_periods[0]}']
        
        # Smoothed Williams %R
        for period in self.wr_periods:
            wr_data[f'wr_smooth_{period}'] = wr_data[f'wr_{period}'].ewm(span=self.smoothing_period).mean()
        
        # Williams %R momentum and acceleration
        wr_data['wr_momentum'] = wr_data['wr_primary'].diff()
        wr_data['wr_acceleration'] = wr_data['wr_momentum'].diff()
        wr_data['wr_jerk'] = wr_data['wr_acceleration'].diff()
        
        # Williams %R velocity (rate of change)
        for lookback in [3, 5, 10]:
            wr_data[f'wr_velocity_{lookback}'] = wr_data['wr_primary'].pct_change(lookback)
        
        # Multi-timeframe Williams %R relationships
        wr_data['wr_convergence'] = self._calculate_wr_convergence(wr_data)
        wr_data['wr_divergence'] = self._calculate_wr_divergence(wr_data)
        
        # Williams %R statistical measures
        wr_data['wr_volatility'] = wr_data['wr_primary'].rolling(window=20).std()
        wr_data['wr_skewness'] = wr_data['wr_primary'].rolling(window=20).apply(skew)
        wr_data['wr_kurtosis'] = wr_data['wr_primary'].rolling(window=20).apply(kurtosis)
        
        # Williams %R entropy (information content)
        wr_data['wr_entropy'] = self._calculate_rolling_entropy(wr_data['wr_primary'], 20)
        
        return wr_data.fillna(0)
    
    def _calculate_williams_r(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Williams %R"""
        highest_high = data['high'].rolling(window=period).max()
        lowest_low = data['low'].rolling(window=period).min()
        
        williams_r = -100 * (highest_high - data['close']) / (highest_high - lowest_low)
        return williams_r.fillna(-50)
    
    def _calculate_wr_convergence(self, wr_data: pd.DataFrame) -> pd.Series:
        """Calculate Williams %R convergence across timeframes"""
        convergence = pd.Series(0.0, index=wr_data.index)
        
        # Calculate agreement between different period Williams %R
        wr_directions = []
        for period in self.wr_periods:
            if f'wr_{period}' in wr_data.columns:
                # Normalize to direction (-1, 0, 1)
                direction = np.where(wr_data[f'wr_{period}'] > -50, 1,
                                   np.where(wr_data[f'wr_{period}'] < -50, -1, 0))
                wr_directions.append(direction)
        
        if wr_directions:
            # Calculate convergence as agreement across timeframes
            wr_matrix = np.column_stack(wr_directions)
            convergence = pd.Series(np.mean(wr_matrix, axis=1), index=wr_data.index)
        
        return convergence
    
    def _calculate_wr_divergence(self, wr_data: pd.DataFrame) -> pd.Series:
        """Calculate Williams %R divergence across timeframes"""
        divergence = pd.Series(0.0, index=wr_data.index)
        
        # Calculate standard deviation of Williams %R across periods
        wr_values = []
        for period in self.wr_periods:
            if f'wr_{period}' in wr_data.columns:
                # Normalize Williams %R values
                normalized_wr = (wr_data[f'wr_{period}'] + 100) / 100  # Scale to 0-1
                wr_values.append(normalized_wr)
        
        if wr_values:
            wr_matrix = np.column_stack(wr_values)
            divergence = pd.Series(np.std(wr_matrix, axis=1), index=wr_data.index)
        
        return divergence
    
    def _calculate_rolling_entropy(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate rolling entropy of a series"""
        entropy_values = pd.Series(index=series.index, dtype=float)
        
        for i in range(window, len(series)):
            window_data = series.iloc[i-window:i]
            
            # Create histogram
            hist, _ = np.histogram(window_data.dropna(), bins=10, density=True)
            hist = hist[hist > 0]  # Remove zero probabilities
            
            # Calculate entropy
            if len(hist) > 1:
                entropy_val = entropy(hist)
                entropy_values.iloc[i] = entropy_val
        
        return entropy_values.fillna(entropy_values.mean())
    
    def _analyze_degradation_patterns(self, data: pd.DataFrame, wr_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze Williams %R degradation patterns"""
        degradation_metrics = {}
        
        # Momentum degradation
        degradation_metrics['momentum_degradation'] = self._analyze_momentum_degradation(wr_data)
        
        # Signal quality degradation
        degradation_metrics['quality_degradation'] = self._analyze_signal_quality_degradation(data, wr_data)
        
        # Volatility degradation
        degradation_metrics['volatility_degradation'] = self._analyze_volatility_degradation(wr_data)
        
        # Entropy degradation (information loss)
        degradation_metrics['entropy_degradation'] = self._analyze_entropy_degradation(wr_data)
        
        # Convergence degradation
        degradation_metrics['convergence_degradation'] = self._analyze_convergence_degradation(wr_data)
        
        # Overall degradation score
        degradation_metrics['overall_degradation'] = self._calculate_overall_degradation(degradation_metrics)
        
        # Degradation trend
        degradation_metrics['degradation_trend'] = self._analyze_degradation_trend(degradation_metrics)
        
        return degradation_metrics
    
    def _analyze_momentum_degradation(self, wr_data: pd.DataFrame) -> pd.Series:
        """Analyze momentum degradation in Williams %R"""
        momentum = wr_data['wr_momentum']
        acceleration = wr_data['wr_acceleration']
        
        # Momentum consistency over time
        momentum_consistency = 1 - abs(momentum).rolling(window=self.degradation_window).std()
        
        # Acceleration degradation
        acceleration_degradation = abs(acceleration).rolling(window=self.degradation_window).mean()
        
        # Combined momentum degradation
        momentum_degradation = (momentum_consistency + acceleration_degradation) / 2
        
        return momentum_degradation.fillna(0.5)
    
    def _analyze_signal_quality_degradation(self, data: pd.DataFrame, wr_data: pd.DataFrame) -> pd.Series:
        """Analyze signal quality degradation"""
        # Signal-to-noise ratio degradation
        wr_signal = wr_data['wr_primary']
        price_signal = data['close'].pct_change()
        
        # Calculate correlation between WR and price changes
        correlation = self._calculate_rolling_correlation(wr_signal, price_signal, self.degradation_window)
        
        # Signal clarity degradation (based on smoothness)
        signal_noise = wr_signal.rolling(window=5).std()
        signal_clarity = 1 - (signal_noise / signal_noise.rolling(window=self.degradation_window).max())
        
        # Combined quality degradation
        quality_degradation = 1 - (abs(correlation) + signal_clarity) / 2
        
        return quality_degradation.fillna(0.5)
    
    def _analyze_volatility_degradation(self, wr_data: pd.DataFrame) -> pd.Series:
        """Analyze volatility degradation in Williams %R"""
        volatility = wr_data['wr_volatility']
        
        # Volatility consistency
        volatility_trend = volatility.rolling(window=self.degradation_window).apply(
            lambda x: abs(np.polyfit(range(len(x)), x, 1)[0]) if len(x) > 1 else 0
        )
        
        # Volatility clustering degradation
        volatility_clustering = volatility.rolling(window=self.degradation_window).apply(
            lambda x: x.autocorr(lag=1) if len(x) > 1 else 0
        )
        
        # Combined volatility degradation
        volatility_degradation = (volatility_trend + abs(volatility_clustering)) / 2
        
        return volatility_degradation.fillna(0.5)
    
    def _analyze_entropy_degradation(self, wr_data: pd.DataFrame) -> pd.Series:
        """Analyze entropy degradation (information loss)"""
        entropy_series = wr_data['wr_entropy']
        
        # Entropy trend degradation
        entropy_trend = entropy_series.rolling(window=self.degradation_window).apply(
            lambda x: abs(np.polyfit(range(len(x)), x, 1)[0]) if len(x) > 1 else 0
        )
        
        # Entropy variability
        entropy_variability = entropy_series.rolling(window=self.degradation_window).std()
        
        # Normalize and combine
        entropy_degradation = (entropy_trend + entropy_variability) / 2
        
        return entropy_degradation.fillna(0.5)
    
    def _analyze_convergence_degradation(self, wr_data: pd.DataFrame) -> pd.Series:
        """Analyze convergence degradation across timeframes"""
        convergence = wr_data['wr_convergence']
        divergence = wr_data['wr_divergence']
        
        # Convergence stability
        convergence_stability = 1 - abs(convergence).rolling(window=self.degradation_window).std()
        
        # Divergence increase
        divergence_increase = divergence.rolling(window=self.degradation_window).mean()
        
        # Combined convergence degradation
        convergence_degradation = (1 - convergence_stability + divergence_increase) / 2
        
        return convergence_degradation.fillna(0.5)
    
    def _calculate_overall_degradation(self, degradation_metrics: Dict[str, Any]) -> pd.Series:
        """Calculate overall degradation score"""
        weights = {
            'momentum_degradation': 0.25,
            'quality_degradation': 0.25,
            'volatility_degradation': 0.2,
            'entropy_degradation': 0.15,
            'convergence_degradation': 0.15
        }
        
        overall = pd.Series(0.0, index=list(degradation_metrics.values())[0].index)
        
        for metric_name, metric_values in degradation_metrics.items():
            if metric_name in weights and hasattr(metric_values, 'index'):
                overall += metric_values * weights[metric_name]
        
        return overall
    
    def _analyze_degradation_trend(self, degradation_metrics: Dict[str, Any]) -> pd.Series:
        """Analyze degradation trend direction"""
        overall_degradation = degradation_metrics['overall_degradation']
        
        # Calculate trend slope
        trend = overall_degradation.rolling(window=self.degradation_window).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
        )
        
        return trend.fillna(0)
    
    def _calculate_rolling_correlation(self, series1: pd.Series, series2: pd.Series, window: int) -> pd.Series:
        """Calculate rolling correlation between two series"""
        correlation = pd.Series(index=series1.index, dtype=float)
        
        for i in range(window, len(series1)):
            subset1 = series1.iloc[i-window:i].fillna(0)
            subset2 = series2.iloc[i-window:i].fillna(0)
            
            if len(subset1) > 1 and len(subset2) > 1:
                corr_val = subset1.corr(subset2)
                correlation.iloc[i] = corr_val if not pd.isna(corr_val) else 0
        
        return correlation.fillna(0)
    
    def _assess_signal_quality(self, data: pd.DataFrame, wr_data: pd.DataFrame,
                              degradation_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall signal quality"""
        if not self.quality_assessment:
            return {'quality_score': pd.Series(0.5, index=data.index)}
        
        # Signal strength assessment
        signal_strength = 1 - degradation_analysis['overall_degradation']
        
        # Signal reliability assessment
        wr_consistency = 1 - wr_data['wr_volatility'] / 30  # Normalize by typical WR range
        
        # Signal predictability assessment
        entropy_quality = 1 - wr_data['wr_entropy'] / wr_data['wr_entropy'].rolling(window=50).max()
        
        # Overall quality score
        quality_score = (signal_strength * 0.4 + wr_consistency * 0.3 + entropy_quality * 0.3)
        quality_score = quality_score.clip(lower=0, upper=1)
        
        return {
            'quality_score': quality_score,
            'signal_strength': signal_strength,
            'signal_reliability': wr_consistency,
            'signal_predictability': entropy_quality
        }
    
    def _detect_anomalies(self, data: pd.DataFrame, wr_data: pd.DataFrame,
                         degradation_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Detect anomalies in Williams %R and degradation patterns"""
        if not self.anomaly_detection:
            return {'anomalies': pd.Series(False, index=data.index)}
        
        try:
            # Prepare anomaly detection features
            anomaly_features = self._prepare_anomaly_features(data, wr_data, degradation_analysis)
            
            if len(anomaly_features) < 50:
                return {'anomalies': pd.Series(False, index=data.index)}
            
            # Detect anomalies
            anomaly_scores = self.anomaly_detector.fit_predict(anomaly_features)
            anomalies = pd.Series(anomaly_scores == -1, index=data.index)
            
            # Statistical anomaly detection
            statistical_anomalies = self._detect_statistical_anomalies(wr_data, degradation_analysis)
            
            # Combine anomaly detections
            combined_anomalies = anomalies | statistical_anomalies
            
            return {
                'anomalies': combined_anomalies,
                'ml_anomalies': anomalies,
                'statistical_anomalies': statistical_anomalies,
                'anomaly_strength': self._calculate_anomaly_strength(wr_data, combined_anomalies)
            }
            
        except Exception:
            return {'anomalies': pd.Series(False, index=data.index)}
    
    def _prepare_anomaly_features(self, data: pd.DataFrame, wr_data: pd.DataFrame,
                                 degradation_analysis: Dict[str, Any]) -> np.ndarray:
        """Prepare features for anomaly detection"""
        features = []
        
        # Williams %R features
        features.append(wr_data['wr_primary'].values)
        features.append(wr_data['wr_momentum'].values)
        features.append(wr_data['wr_acceleration'].values)
        features.append(wr_data['wr_volatility'].values)
        features.append(wr_data['wr_entropy'].values)
        
        # Degradation features
        features.append(degradation_analysis['overall_degradation'].values)
        features.append(degradation_analysis['momentum_degradation'].values)
        features.append(degradation_analysis['quality_degradation'].values)
        
        # Price features
        features.append(data['close'].pct_change().values)
        features.append(data['close'].rolling(window=10).std().values)
        
        return np.column_stack(features)
    
    def _detect_statistical_anomalies(self, wr_data: pd.DataFrame,
                                     degradation_analysis: Dict[str, Any]) -> pd.Series:
        """Detect statistical anomalies using z-score and IQR methods"""
        anomalies = pd.Series(False, index=wr_data.index)
        
        # Z-score anomalies
        wr_zscore = np.abs((wr_data['wr_primary'] - wr_data['wr_primary'].rolling(window=50).mean()) /
                          wr_data['wr_primary'].rolling(window=50).std())
        z_anomalies = wr_zscore > 3
        
        # IQR anomalies for degradation
        degradation = degradation_analysis['overall_degradation']
        q1 = degradation.rolling(window=50).quantile(0.25)
        q3 = degradation.rolling(window=50).quantile(0.75)
        iqr = q3 - q1
        
        degradation_anomalies = (degradation > q3 + 1.5 * iqr) | (degradation < q1 - 1.5 * iqr)
        
        # Combine anomalies
        anomalies = z_anomalies | degradation_anomalies
        
        return anomalies.fillna(False)
    
    def _calculate_anomaly_strength(self, wr_data: pd.DataFrame, anomalies: pd.Series) -> pd.Series:
        """Calculate strength of detected anomalies"""
        strength = pd.Series(0.0, index=wr_data.index)
        
        for i in range(len(anomalies)):
            if anomalies.iloc[i]:
                # Calculate deviation from normal behavior
                wr_deviation = abs(wr_data['wr_primary'].iloc[i] - 
                                 wr_data['wr_primary'].rolling(window=20).mean().iloc[i])
                wr_std = wr_data['wr_primary'].rolling(window=20).std().iloc[i]
                
                if wr_std > 0:
                    strength.iloc[i] = min(1.0, wr_deviation / (wr_std + 1e-6))
        
        return strength
    
    def _calculate_adaptive_parameters(self, data: pd.DataFrame, wr_data: pd.DataFrame,
                                     degradation_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate adaptive parameters based on degradation levels"""
        if not self.adaptive_analysis:
            return {'degradation_threshold': 0.7, 'quality_threshold': 0.3}
        
        # Current degradation level
        current_degradation = degradation_analysis['overall_degradation'].iloc[-1]
        
        # Adaptive degradation threshold
        degradation_percentile = degradation_analysis['overall_degradation'].rolling(window=100).rank(pct=True).iloc[-1]
        degradation_threshold = 0.5 + (degradation_percentile - 0.5) * 0.4
        
        # Adaptive quality threshold
        wr_volatility = wr_data['wr_volatility'].iloc[-1]
        vol_percentile = wr_data['wr_volatility'].rolling(window=60).rank(pct=True).iloc[-1]
        quality_threshold = 0.3 + (1 - vol_percentile) * 0.4
        
        return {
            'degradation_threshold': np.clip(degradation_threshold, 0.4, 0.9),
            'quality_threshold': np.clip(quality_threshold, 0.1, 0.7),
            'current_degradation': current_degradation if not pd.isna(current_degradation) else 0.5,
            'volatility_adjustment': vol_percentile if not pd.isna(vol_percentile) else 0.5
        }
    
    def _generate_base_degradation_signals(self, data: pd.DataFrame, wr_data: pd.DataFrame,
                                         degradation_analysis: Dict[str, Any],
                                         adaptive_params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate base degradation signals"""
        degradation = degradation_analysis['overall_degradation']
        degradation_trend = degradation_analysis['degradation_trend']
        
        degradation_threshold = adaptive_params['degradation_threshold']
        
        # High degradation signals (potential reversal)
        high_degradation = degradation > degradation_threshold
        degradation_increasing = degradation_trend > 0.01
        degradation_decreasing = degradation_trend < -0.01
        
        # Williams %R position signals
        wr_oversold = wr_data['wr_primary'] < -80
        wr_overbought = wr_data['wr_primary'] > -20
        
        # Combined degradation signals
        degradation_buy = high_degradation & wr_oversold & degradation_increasing
        degradation_sell = high_degradation & wr_overbought & degradation_increasing
        
        # Recovery signals (degradation decreasing)
        recovery_buy = degradation_decreasing & (wr_data['wr_primary'] > -70)
        recovery_sell = degradation_decreasing & (wr_data['wr_primary'] < -30)
        
        # Signal strength based on degradation level
        signal_strength = degradation * abs(degradation_trend)
        
        return {
            'degradation_buy': degradation_buy,
            'degradation_sell': degradation_sell,
            'recovery_buy': recovery_buy,
            'recovery_sell': recovery_sell,
            'high_degradation': high_degradation,
            'degradation_increasing': degradation_increasing,
            'degradation_decreasing': degradation_decreasing,
            'signal_strength': signal_strength,
            'degradation_level': degradation
        }
    
    def _train_ml_models(self, data: pd.DataFrame, wr_data: pd.DataFrame,
                        degradation_analysis: Dict[str, Any]) -> None:
        """Train ML models for degradation prediction"""
        try:
            # Prepare features
            features = self._prepare_degradation_ml_features(data, wr_data, degradation_analysis)
            
            if len(features) < self.ml_lookback // 2:
                return
            
            # Prepare labels
            future_degradation = degradation_analysis['overall_degradation'].shift(-5)
            future_returns = data['close'].pct_change(5).shift(-5)
            
            # Classification labels for degradation patterns
            degradation_labels = np.where(future_degradation > 0.7, 1,
                                        np.where(future_degradation < 0.3, -1, 0))
            
            # Quality labels
            quality_labels = 1 - future_degradation
            
            # Pattern labels
            pattern_labels = np.where(abs(future_returns) > 0.02, 1, 0)
            
            # Remove NaN values
            valid_indices = ~(np.isnan(degradation_labels) | np.any(np.isnan(features), axis=1) |
                            np.isnan(quality_labels) | np.isnan(pattern_labels))
            
            features_clean = features[valid_indices]
            degradation_labels_clean = degradation_labels[valid_indices]
            quality_labels_clean = quality_labels[valid_indices]
            pattern_labels_clean = pattern_labels[valid_indices]
            
            if len(features_clean) < 50:
                return
            
            # Scale features
            features_scaled = self.feature_scaler.fit_transform(features_clean)
            
            # Train models
            self.degradation_classifier.fit(features_scaled, degradation_labels_clean)
            self.quality_regressor.fit(features_scaled, quality_labels_clean)
            self.pattern_detector.fit(features_scaled, pattern_labels_clean)
            
            self.is_trained = True
            self.feature_columns = [f'feature_{i}' for i in range(features_scaled.shape[1])]
            
        except Exception as e:
            print(f"ML training error: {e}")
    
    def _prepare_degradation_ml_features(self, data: pd.DataFrame, wr_data: pd.DataFrame,
                                       degradation_analysis: Dict[str, Any]) -> np.ndarray:
        """Prepare features for ML models"""
        features = []
        
        # Williams %R features
        features.append(wr_data['wr_primary'].values)
        features.append(wr_data['wr_momentum'].values)
        features.append(wr_data['wr_acceleration'].values)
        features.append(wr_data['wr_volatility'].values)
        features.append(wr_data['wr_entropy'].values)
        features.append(wr_data['wr_convergence'].values)
        features.append(wr_data['wr_divergence'].values)
        
        # Multi-period Williams %R
        for period in self.wr_periods[:3]:  # Limit to prevent overfitting
            if f'wr_{period}' in wr_data.columns:
                features.append(wr_data[f'wr_{period}'].values)
        
        # Degradation features
        features.append(degradation_analysis['overall_degradation'].values)
        features.append(degradation_analysis['momentum_degradation'].values)
        features.append(degradation_analysis['quality_degradation'].values)
        features.append(degradation_analysis['volatility_degradation'].values)
        features.append(degradation_analysis['degradation_trend'].values)
        
        # Price features
        features.append(data['close'].pct_change().values)
        features.append(data['close'].pct_change(5).values)
        features.append(data['close'].rolling(window=10).mean().values)
        features.append(data['close'].rolling(window=20).std().values)
        
        # Volume features (if available)
        if 'volume' in data.columns:
            features.append(data['volume'].rolling(window=10).mean().values)
            features.append(data['volume'].pct_change().values)
        
        return np.column_stack(features)
    
    def _generate_ml_predictions(self, data: pd.DataFrame, wr_data: pd.DataFrame,
                               degradation_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate ML-enhanced predictions"""
        if not self.is_trained:
            return {
                'predicted_degradation': pd.Series(0.5, index=data.index),
                'predicted_quality': pd.Series(0.5, index=data.index),
                'pattern_prediction': pd.Series(0, index=data.index),
                'ml_confidence': pd.Series(0.5, index=data.index)
            }
        
        try:
            # Prepare features
            features = self._prepare_degradation_ml_features(data, wr_data, degradation_analysis)
            features_scaled = self.feature_scaler.transform(features)
            
            # Generate predictions
            degradation_pred = self.degradation_classifier.predict(features_scaled)
            degradation_proba = self.degradation_classifier.predict_proba(features_scaled)
            
            quality_pred = self.quality_regressor.predict(features_scaled)
            pattern_pred = self.pattern_detector.predict(features_scaled)
            
            # Calculate confidence
            confidence = np.max(degradation_proba, axis=1)
            
            return {
                'predicted_degradation': pd.Series(degradation_pred, index=data.index),
                'predicted_quality': pd.Series(quality_pred, index=data.index),
                'pattern_prediction': pd.Series(pattern_pred, index=data.index),
                'ml_confidence': pd.Series(confidence, index=data.index)
            }
            
        except Exception as e:
            print(f"ML prediction error: {e}")
            return {
                'predicted_degradation': pd.Series(0.5, index=data.index),
                'predicted_quality': pd.Series(0.5, index=data.index),
                'pattern_prediction': pd.Series(0, index=data.index),
                'ml_confidence': pd.Series(0.5, index=data.index)
            }
    
    def _analyze_degradation_regime(self, data: pd.DataFrame, wr_data: pd.DataFrame,
                                  degradation_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current degradation regime"""
        if not self.regime_awareness:
            return {'regime': 'normal', 'confidence': 0.5}
        
        # Current degradation characteristics
        current_degradation = degradation_analysis['overall_degradation'].iloc[-1]
        degradation_trend = degradation_analysis['degradation_trend'].iloc[-1]
        
        # Williams %R characteristics
        wr_volatility = wr_data['wr_volatility'].iloc[-1]
        wr_entropy = wr_data['wr_entropy'].iloc[-1]
        
        # Regime classification
        if current_degradation > 0.8 and degradation_trend > 0:
            regime = 'high_degradation'
            confidence = current_degradation
        elif current_degradation < 0.3 and degradation_trend < 0:
            regime = 'low_degradation'
            confidence = 1 - current_degradation
        elif abs(degradation_trend) > 0.02:
            regime = 'transitional'
            confidence = abs(degradation_trend) * 20  # Scale to 0-1
        elif wr_volatility > 20:
            regime = 'volatile'
            confidence = min(1.0, wr_volatility / 30)
        else:
            regime = 'stable'
            confidence = 1 - abs(degradation_trend) * 20
        
        return {
            'regime': regime,
            'confidence': float(confidence) if not pd.isna(confidence) else 0.5,
            'degradation_level': float(current_degradation) if not pd.isna(current_degradation) else 0.5,
            'degradation_trend': float(degradation_trend) if not pd.isna(degradation_trend) else 0.0,
            'volatility_level': float(wr_volatility) if not pd.isna(wr_volatility) else 10.0,
            'entropy_level': float(wr_entropy) if not pd.isna(wr_entropy) else 1.0
        }
    
    def _analyze_signal_strength(self, data: pd.DataFrame, wr_data: pd.DataFrame,
                                base_signals: Dict[str, Any], ml_predictions: Dict[str, Any],
                                quality_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze overall signal strength and confidence"""
        # Combine signal sources
        base_strength = base_signals['signal_strength']
        ml_confidence = ml_predictions['ml_confidence']
        quality_score = quality_analysis['quality_score']
        
        # Signal combination
        degradation_signal = np.where(
            base_signals['degradation_buy'] | base_signals['recovery_buy'], 1,
            np.where(base_signals['degradation_sell'] | base_signals['recovery_sell'], -1, 0)
        )
        
        ml_signal = ml_predictions['predicted_degradation']
        
        # Combined signal
        combined_signal = (
            degradation_signal * 0.4 +
            ml_signal * 0.4 +
            np.sign(base_signals['degradation_level'] - 0.5) * 0.2
        )
        
        # Overall confidence
        overall_confidence = (
            base_strength * 0.3 +
            ml_confidence * 0.4 +
            quality_score * 0.3
        )
        
        # Signal strength
        signal_strength = np.abs(combined_signal) * overall_confidence
        
        return {
            'combined_signal': pd.Series(combined_signal, index=data.index),
            'overall_confidence': overall_confidence,
            'signal_strength': signal_strength,
            'degradation_contribution': base_strength * 0.3,
            'ml_contribution': ml_confidence * 0.4,
            'quality_contribution': quality_score * 0.3
        }
    
    def _synthesize_degradation_signals(self, data: pd.DataFrame, wr_data: pd.DataFrame,
                                      signal_analysis: Dict[str, Any],
                                      regime_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize final degradation signals"""
        combined_signal = signal_analysis['combined_signal']
        confidence = signal_analysis['overall_confidence']
        signal_strength = signal_analysis['signal_strength']
        
        # Regime adjustment
        regime_confidence = regime_analysis.get('confidence', 0.5)
        regime_adjusted_signal = combined_signal * (0.5 + regime_confidence * 0.5)
        
        # Generate final trading signals
        strong_buy = (regime_adjusted_signal > 0.7) & (confidence > self.confidence_threshold)
        buy = (regime_adjusted_signal > 0.4) & (confidence > self.confidence_threshold * 0.8)
        strong_sell = (regime_adjusted_signal < -0.7) & (confidence > self.confidence_threshold)
        sell = (regime_adjusted_signal < -0.4) & (confidence > self.confidence_threshold * 0.8)
        
        # Position sizing based on signal quality and regime
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
    
    def _calculate_degradation_risk_metrics(self, data: pd.DataFrame, wr_data: pd.DataFrame,
                                          synthesized_signals: Dict[str, Any],
                                          degradation_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate degradation-specific risk metrics"""
        # Degradation risk
        degradation_risk = degradation_analysis['overall_degradation']
        
        # Signal consistency risk
        signal_consistency = 1 - synthesized_signals['final_signal'].rolling(window=10).std().fillna(0.5)
        
        # Williams %R extreme risk
        wr_extreme_risk = np.where(
            (wr_data['wr_primary'] > -10) | (wr_data['wr_primary'] < -90), 0.8, 0.3
        )
        
        # Entropy risk (unpredictability)
        entropy_risk = wr_data['wr_entropy'] / wr_data['wr_entropy'].rolling(window=50).max()
        
        # Overall risk score
        overall_risk = (
            degradation_risk * 0.4 +
            (1 - signal_consistency) * 0.25 +
            wr_extreme_risk * 0.2 +
            entropy_risk * 0.15
        )
        
        return {
            'degradation_risk': degradation_risk,
            'signal_consistency': signal_consistency,
            'wr_extreme_risk': pd.Series(wr_extreme_risk, index=data.index),
            'entropy_risk': entropy_risk,
            'overall_risk': overall_risk
        }
    
    def _generate_degradation_recommendations(self, synthesized_signals: Dict[str, Any],
                                            risk_metrics: Dict[str, Any],
                                            regime_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final degradation-based trading recommendations"""
        final_signal = synthesized_signals['final_signal']
        risk_score = risk_metrics['overall_risk']
        position_size = synthesized_signals['position_size']
        
        # Risk-adjusted position sizing
        risk_adjustment = 1 - risk_score
        adjusted_position = position_size * risk_adjustment
        
        # Regime-based adjustments
        regime = regime_analysis.get('regime', 'stable')
        regime_multipliers = {
            'high_degradation': 0.5,
            'low_degradation': 1.2,
            'transitional': 0.8,
            'volatile': 0.6,
            'stable': 1.0
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
            'degradation_status': regime_analysis.get('regime', 'stable')
        }
    
    def _generate_metadata(self, data: pd.DataFrame, wr_data: pd.DataFrame,
                          synthesized_signals: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive metadata"""
        return {
            'indicator_name': 'WRDegradationIndicator',
            'version': '1.0.0',
            'parameters': {
                'wr_periods': self.wr_periods,
                'degradation_window': self.degradation_window,
                'smoothing_period': self.smoothing_period,
                'ml_lookback': self.ml_lookback,
                'confidence_threshold': self.confidence_threshold
            },
            'data_points': len(data),
            'calculation_time': pd.Timestamp.now().isoformat(),
            'ml_model_trained': self.is_trained,
            'feature_count': len(self.feature_columns),
            'advanced_features': {
                'adaptive_analysis': self.adaptive_analysis,
                'anomaly_detection': self.anomaly_detection,
                'regime_awareness': self.regime_awareness,
                'quality_assessment': self.quality_assessment
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
            'degradation_statistics': {
                'current_wr': float(wr_data['wr_primary'].iloc[-1]) if len(wr_data) > 0 else -50.0,
                'wr_volatility': float(wr_data['wr_volatility'].iloc[-1]) if len(wr_data) > 0 else 10.0,
                'wr_entropy': float(wr_data['wr_entropy'].iloc[-1]) if len(wr_data) > 0 else 1.0
            }
        }
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error result"""
        return {
            'error': error_message,
            'wr_data': pd.DataFrame(),
            'degradation_analysis': {},
            'base_signals': {},
            'synthesized_signals': {},
            'recommendations': {},
            'confidence_score': 0.0,
            'signal_strength': 0.0,
            'metadata': {
                'indicator_name': 'WRDegradationIndicator',
                'error': error_message,
                'calculation_time': pd.Timestamp.now().isoformat()
            }
        }
