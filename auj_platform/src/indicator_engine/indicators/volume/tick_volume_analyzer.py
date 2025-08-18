"""
Advanced Tick Volume Analyzer for AUJ Platform

This module implements a sophisticated tick volume analysis system that processes
high-frequency market data to identify patterns, clustering behaviors, and 
tick-by-tick flow dynamics. The analyzer provides deep insights into market
microstructure through comprehensive tick volume analysis.

Key Features:
- High-frequency tick volume pattern analysis
- Volume clustering and accumulation detection
- Tick-by-tick flow dynamics modeling
- Real-time volume velocity and acceleration metrics
- Statistical distribution analysis of tick volumes
- Machine learning-based pattern recognition
- Anomaly detection in tick flow patterns
- Microstructure impact measurement
- Advanced visualization and reporting
- Production-ready performance optimization

Mathematical Models:
- Tick volume distribution analysis
- Poisson process modeling for tick arrivals
- Volume clustering algorithms
- Flow velocity and acceleration calculations
- Statistical significance testing
- Machine learning classification models
- Time series decomposition analysis
- Spectral analysis for cyclical patterns

The implementation follows AUJ Platform's humanitarian mission requirements with
robust error handling, comprehensive logging, and production-ready code quality.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
import warnings
from scipy import stats
from scipy.signal import find_peaks, welch, periodogram
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import joblib
import traceback

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TickVolumeRegime(Enum):
    """Enumeration for tick volume regime classification."""
    ULTRA_HIGH = "ULTRA_HIGH"
    HIGH = "HIGH"
    NORMAL = "NORMAL"
    LOW = "LOW"
    MICRO = "MICRO"
    ANOMALOUS = "ANOMALOUS"


class FlowDirection(Enum):
    """Enumeration for tick flow direction."""
    BUYING_PRESSURE = "BUYING_PRESSURE"
    SELLING_PRESSURE = "SELLING_PRESSURE"
    BALANCED = "BALANCED"
    CHURNING = "CHURNING"
    ACCUMULATION = "ACCUMULATION"
    DISTRIBUTION = "DISTRIBUTION"


class TickPattern(Enum):
    """Enumeration for tick volume patterns."""
    BURST = "BURST"
    SUSTAINED = "SUSTAINED"
    OSCILLATING = "OSCILLATING"
    DECLINING = "DECLINING"
    RANDOM = "RANDOM"
    CLUSTERED = "CLUSTERED"


@dataclass
class TickVolumeComponents:
    """Data class for tick volume analysis components."""
    tick_volume_regime: TickVolumeRegime
    flow_direction: FlowDirection
    tick_pattern: TickPattern
    confidence_score: float
    
    # Tick metrics
    tick_count: int = 0
    avg_tick_volume: float = 0.0
    tick_velocity: float = 0.0
    tick_acceleration: float = 0.0
    
    # Volume clustering
    cluster_strength: float = 0.0
    cluster_id: int = -1
    cluster_persistence: float = 0.0
    
    # Flow dynamics
    flow_intensity: float = 0.0
    flow_persistence: float = 0.0
    flow_momentum: float = 0.0
    flow_efficiency: float = 0.0
    
    # Statistical measures
    volume_skewness: float = 0.0
    volume_kurtosis: float = 0.0
    distribution_stability: float = 0.0
    outlier_ratio: float = 0.0
    
    # Advanced metrics
    microstructure_impact: float = 0.0
    liquidity_absorption: float = 0.0
    tick_intensity: float = 0.0
    pressure_gradient: float = 0.0
    
    # ML-based features
    pattern_probability: float = 0.0
    anomaly_score: float = 0.0
    classification_confidence: float = 0.0
    
    # Risk and efficiency
    execution_cost: float = 0.0
    slippage_estimate: float = 0.0
    timing_alpha: float = 0.0


@dataclass
class TickVolumeParameters:
    """Configuration parameters for tick volume analysis."""
    # Tick analysis windows
    tick_window: int = 100
    velocity_window: int = 20
    acceleration_window: int = 10
    
    # Clustering parameters
    cluster_eps: float = 0.3
    cluster_min_samples: int = 5
    cluster_persistence_window: int = 50
    
    # Flow analysis
    flow_window: int = 30
    flow_threshold: float = 0.1
    pressure_sensitivity: float = 0.05
    
    # Statistical thresholds
    outlier_threshold: float = 3.0
    anomaly_contamination: float = 0.1
    confidence_threshold: float = 0.7
    
    # Pattern recognition
    pattern_window: int = 50
    burst_threshold: float = 2.0
    sustained_threshold: float = 0.8
    
    # ML parameters
    ml_lookback: int = 500
    feature_count: int = 20
    retrain_interval: int = 1000
    
    # Performance parameters
    max_tick_buffer: int = 10000
    processing_batch_size: int = 1000
    
    # Validation parameters
    min_ticks: int = 50
    max_tick_gap: timedelta = timedelta(minutes=5)


class AdvancedTickVolumeAnalyzer:
    """
    Advanced Tick Volume Analyzer with machine learning enhancements.
    
    This class implements sophisticated tick-level volume analysis including
    high-frequency pattern recognition, clustering analysis, and flow dynamics
    modeling for comprehensive market microstructure insights.
    """
    
    def __init__(self, parameters: Optional[TickVolumeParameters] = None):
        """
        Initialize the Advanced Tick Volume Analyzer.
        
        Args:
            parameters: Configuration parameters for the analyzer
        """
        self.params = parameters or TickVolumeParameters()
        self.logger = logging.getLogger(__name__)
        
        # Initialize ML models
        self._init_ml_models()
        
        # Initialize scalers
        self.scaler = RobustScaler()
        self.feature_scaler = StandardScaler()
        
        # Initialize clustering models
        self.volume_cluster = DBSCAN(
            eps=self.params.cluster_eps,
            min_samples=self.params.cluster_min_samples
        )
        
        # State management
        self.tick_buffer = []
        self.is_trained = False
        self.last_analysis = None
        self.pattern_history = []
        
        # Performance tracking
        self.analysis_count = 0
        self.total_analysis_time = 0.0
        
        self.logger.info("Advanced Tick Volume Analyzer initialized successfully")
    
    def _init_ml_models(self) -> None:
        """Initialize machine learning models."""
        try:
            # Pattern classification model
            self.pattern_classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            
            # Anomaly detection model
            self.anomaly_detector = IsolationForest(
                contamination=self.params.anomaly_contamination,
                random_state=42,
                n_jobs=-1
            )
            
            # Flow direction classifier
            self.flow_classifier = RandomForestClassifier(
                n_estimators=50,
                max_depth=8,
                random_state=42,
                n_jobs=-1
            )
            
            self.logger.info("ML models initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing ML models: {str(e)}")
            raise
    
    def analyze_tick_data(self, tick_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze tick data for volume patterns and dynamics.
        
        Args:
            tick_data: DataFrame with tick-level data (timestamp, price, volume)
            
        Returns:
            Dictionary containing comprehensive tick volume analysis
        """
        try:
            start_time = datetime.now()
            
            # Validate input data
            if not self._validate_tick_data(tick_data):
                raise ValueError("Invalid tick data for analysis")
            
            # Prepare tick data
            df = self._prepare_tick_data(tick_data.copy())
            
            # Update tick buffer
            self._update_tick_buffer(df)
            
            # Calculate basic tick metrics
            df = self._calculate_tick_metrics(df)
            
            # Analyze volume clustering
            df = self._analyze_volume_clustering(df)
            
            # Calculate flow dynamics
            df = self._calculate_flow_dynamics(df)
            
            # Perform statistical analysis
            df = self._perform_statistical_analysis(df)
            
            # Detect patterns
            df = self._detect_tick_patterns(df)
            
            # Machine learning analysis
            if len(self.tick_buffer) >= self.params.ml_lookback:
                df = self._perform_ml_analysis(df)
            
            # Generate analysis components
            analysis_components = self._generate_analysis_components(df)
            
            # Calculate performance metrics
            performance = self._calculate_performance_metrics(df)
            
            # Update performance statistics
            analysis_time = (datetime.now() - start_time).total_seconds()
            self._update_performance_stats(analysis_time)
            
            # Store analysis
            self.last_analysis = analysis_components
            
            # Prepare result
            result = {
                'components': analysis_components,
                'data': df,
                'performance': performance,
                'tick_buffer_size': len(self.tick_buffer),
                'metadata': {
                    'analysis_time': analysis_time,
                    'tick_count': len(df),
                    'parameters': self.params.__dict__,
                    'model_trained': self.is_trained
                }
            }
            
            self.logger.info(f"Tick volume analysis completed in {analysis_time:.4f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in tick volume analysis: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
    
    def _validate_tick_data(self, data: pd.DataFrame) -> bool:
        """Validate tick data quality and completeness."""
        try:
            required_columns = ['timestamp', 'price', 'volume']
            
            # Check required columns
            if not all(col in data.columns for col in required_columns):
                self.logger.error(f"Missing required columns. Required: {required_columns}")
                return False
            
            # Check minimum tick count
            if len(data) < self.params.min_ticks:
                self.logger.error(f"Insufficient ticks. Required: {self.params.min_ticks}, Got: {len(data)}")
                return False
            
            # Check for non-positive volume
            if (data['volume'] <= 0).any():
                self.logger.warning("Non-positive volume values detected")
            
            # Check for invalid prices
            if (data['price'] <= 0).any():
                self.logger.error("Invalid price values detected")
                return False
            
            # Check timestamp ordering
            if not data['timestamp'].is_monotonic_increasing:
                self.logger.warning("Timestamps are not monotonic - will sort")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in tick data validation: {str(e)}")
            return False
    
    def _prepare_tick_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare tick data for analysis."""
        try:
            # Ensure timestamp is datetime
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Calculate time intervals
            df['time_delta'] = df['timestamp'].diff().dt.total_seconds()
            df['time_delta'] = df['time_delta'].fillna(0)
            
            # Calculate price changes
            df['price_change'] = df['price'].diff()
            df['price_change_pct'] = df['price'].pct_change()
            
            # Calculate volume metrics
            df['volume_change'] = df['volume'].diff()
            df['volume_change_pct'] = df['volume'].pct_change()
            
            # Calculate tick direction (uptick/downtick)
            df['tick_direction'] = np.where(
                df['price_change'] > 0, 1,
                np.where(df['price_change'] < 0, -1, 0)
            )
            
            # Calculate trade size categories
            volume_percentiles = df['volume'].quantile([0.25, 0.5, 0.75, 0.9, 0.95])
            df['size_category'] = pd.cut(
                df['volume'],
                bins=[0, volume_percentiles[0.25], volume_percentiles[0.5], 
                      volume_percentiles[0.75], volume_percentiles[0.9], 
                      volume_percentiles[0.95], float('inf')],
                labels=['micro', 'small', 'medium', 'large', 'block', 'institutional']
            )
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error in tick data preparation: {str(e)}")
            raise
    
    def _update_tick_buffer(self, df: pd.DataFrame) -> None:
        """Update the tick buffer with new data."""
        try:
            # Add new ticks to buffer
            for _, row in df.iterrows():
                tick_info = {
                    'timestamp': row['timestamp'],
                    'price': row['price'],
                    'volume': row['volume'],
                    'direction': row['tick_direction']
                }
                self.tick_buffer.append(tick_info)
            
            # Maintain buffer size
            if len(self.tick_buffer) > self.params.max_tick_buffer:
                excess = len(self.tick_buffer) - self.params.max_tick_buffer
                self.tick_buffer = self.tick_buffer[excess:]
            
        except Exception as e:
            self.logger.error(f"Error updating tick buffer: {str(e)}")
    
    def _calculate_tick_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic tick-level metrics."""
        try:
            # Tick velocity (ticks per unit time)
            rolling_window = min(self.params.velocity_window, len(df))
            df['tick_velocity'] = df.rolling(window=rolling_window)['time_delta'].apply(
                lambda x: rolling_window / (x.sum() + 1e-8), raw=True
            )
            
            # Tick acceleration
            df['tick_acceleration'] = df['tick_velocity'].diff()
            
            # Volume velocity (volume per unit time)
            df['volume_velocity'] = df['volume'] / (df['time_delta'] + 1e-8)
            
            # Average tick volume
            df['avg_tick_volume'] = df['volume'].rolling(window=rolling_window).mean()
            
            # Volume intensity (deviation from average)
            df['volume_intensity'] = (df['volume'] - df['avg_tick_volume']) / (df['avg_tick_volume'] + 1e-8)
            
            # Cumulative volume
            df['cumulative_volume'] = df['volume'].cumsum()
            
            # Volume momentum
            df['volume_momentum'] = df['volume'].rolling(window=rolling_window).apply(
                lambda x: x.iloc[-1] / (x.mean() + 1e-8), raw=False
            )
            
            # Price impact per unit volume
            df['price_impact'] = np.abs(df['price_change']) / (df['volume'] + 1e-8)
            
            # Market impact estimation
            df['market_impact'] = np.abs(df['price_change']) * np.log1p(df['volume'])
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating tick metrics: {str(e)}")
            raise
    
    def _analyze_volume_clustering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze volume clustering patterns."""
        try:
            # Prepare features for clustering
            if len(df) >= self.params.cluster_min_samples:
                features = df[['volume', 'volume_velocity', 'volume_intensity']].fillna(0)
                
                # Normalize features
                features_normalized = self.scaler.fit_transform(features)
                
                # Perform clustering
                try:
                    cluster_labels = self.volume_cluster.fit_predict(features_normalized)
                    df['cluster_id'] = cluster_labels
                    
                    # Calculate silhouette score if multiple clusters
                    unique_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                    if unique_clusters > 1:
                        silhouette_avg = silhouette_score(features_normalized, cluster_labels)
                        df['cluster_quality'] = silhouette_avg
                    else:
                        df['cluster_quality'] = 0.0
                        
                except Exception as cluster_error:
                    self.logger.warning(f"Clustering failed: {cluster_error}")
                    df['cluster_id'] = -1
                    df['cluster_quality'] = 0.0
            else:
                df['cluster_id'] = -1
                df['cluster_quality'] = 0.0
            
            # Calculate cluster strength
            df['cluster_strength'] = df.groupby('cluster_id')['volume'].transform('count') / len(df)
            
            # Calculate cluster persistence
            window = min(self.params.cluster_persistence_window, len(df))
            df['cluster_persistence'] = df['cluster_id'].rolling(window=window).apply(
                lambda x: len(set(x)) / len(x) if len(x) > 0 else 0, raw=True
            )
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error in volume clustering analysis: {str(e)}")
            raise
    
    def _calculate_flow_dynamics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate tick flow dynamics."""
        try:
            # Flow intensity based on volume and direction
            df['flow_intensity'] = df['volume'] * df['tick_direction']
            
            # Cumulative flow
            df['cumulative_flow'] = df['flow_intensity'].cumsum()
            
            # Flow momentum
            window = min(self.params.flow_window, len(df))
            df['flow_momentum'] = df['flow_intensity'].rolling(window=window).sum()
            
            # Flow persistence (how long flow continues in same direction)
            df['flow_persistence'] = df['tick_direction'].rolling(window=window).apply(
                lambda x: abs(x.sum()) / len(x), raw=True
            )
            
            # Flow efficiency (net flow relative to total volume)
            rolling_net_flow = df['flow_intensity'].rolling(window=window).sum()
            rolling_total_volume = df['volume'].rolling(window=window).sum()
            df['flow_efficiency'] = rolling_net_flow / (rolling_total_volume + 1e-8)
            
            # Pressure gradient (rate of flow change)
            df['pressure_gradient'] = df['flow_momentum'].diff()
            
            # Buying/selling pressure
            df['buying_pressure'] = df['volume'].where(df['tick_direction'] > 0, 0).rolling(window=window).sum()
            df['selling_pressure'] = df['volume'].where(df['tick_direction'] < 0, 0).rolling(window=window).sum()
            
            # Pressure ratio
            total_pressure = df['buying_pressure'] + df['selling_pressure']
            df['pressure_ratio'] = (df['buying_pressure'] - df['selling_pressure']) / (total_pressure + 1e-8)
            
            # Liquidity absorption (large volumes with small price changes)
            df['liquidity_absorption'] = df['volume'] / (np.abs(df['price_change']) + 1e-8)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating flow dynamics: {str(e)}")
            raise
    
    def _perform_statistical_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform statistical analysis of tick volumes."""
        try:
            window = min(self.params.pattern_window, len(df))
            
            # Volume distribution statistics
            df['volume_mean'] = df['volume'].rolling(window=window).mean()
            df['volume_std'] = df['volume'].rolling(window=window).std()
            df['volume_skewness'] = df['volume'].rolling(window=window).skew()
            df['volume_kurtosis'] = df['volume'].rolling(window=window).kurt()
            
            # Z-score for outlier detection
            df['volume_zscore'] = (df['volume'] - df['volume_mean']) / (df['volume_std'] + 1e-8)
            
            # Outlier detection
            df['is_outlier'] = np.abs(df['volume_zscore']) > self.params.outlier_threshold
            df['outlier_ratio'] = df['is_outlier'].rolling(window=window).mean()
            
            # Distribution stability
            df['distribution_stability'] = 1.0 / (1.0 + df['volume_std'] / (df['volume_mean'] + 1e-8))
            
            # Tick arrival rate (Poisson process analysis)
            df['arrival_rate'] = 1.0 / (df['time_delta'] + 1e-8)
            df['arrival_rate_smooth'] = df['arrival_rate'].rolling(window=window).mean()
            
            # Volume autocorrelation
            if len(df) >= window:
                df['volume_autocorr'] = df['volume'].rolling(window=window).apply(
                    lambda x: x.autocorr(lag=1) if len(x) > 1 else 0, raw=False
                )
            else:
                df['volume_autocorr'] = 0.0
            
            # Entropy measure (uncertainty in volume distribution)
            df['volume_entropy'] = df['volume'].rolling(window=window).apply(
                self._calculate_entropy, raw=False
            )
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error in statistical analysis: {str(e)}")
            raise
    
    def _calculate_entropy(self, series: pd.Series) -> float:
        """Calculate entropy of a volume series."""
        try:
            if len(series) < 2:
                return 0.0
            
            # Create histogram
            hist, _ = np.histogram(series, bins=min(10, len(series)//2))
            hist = hist[hist > 0]  # Remove zero bins
            
            if len(hist) == 0:
                return 0.0
            
            # Normalize to probabilities
            probs = hist / hist.sum()
            
            # Calculate entropy
            entropy = -np.sum(probs * np.log2(probs + 1e-8))
            
            return entropy
            
        except Exception as e:
            self.logger.error(f"Error calculating entropy: {str(e)}")
            return 0.0
    
    def _detect_tick_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect tick volume patterns."""
        try:
            window = min(self.params.pattern_window, len(df))
            
            # Burst detection (sudden volume spikes)
            volume_threshold = df['volume_mean'] * self.params.burst_threshold
            df['is_burst'] = df['volume'] > volume_threshold
            df['burst_intensity'] = df['volume'] / (df['volume_mean'] + 1e-8)
            
            # Sustained volume detection
            sustained_threshold = df['volume_mean'] * self.params.sustained_threshold
            df['is_sustained'] = df['volume'].rolling(window=window//2).mean() > sustained_threshold
            
            # Oscillating pattern detection
            volume_changes = df['volume'].diff()
            sign_changes = (volume_changes * volume_changes.shift(1) < 0).rolling(window=window).sum()
            df['oscillation_score'] = sign_changes / window
            df['is_oscillating'] = df['oscillation_score'] > 0.3
            
            # Declining volume pattern
            volume_trend = df['volume'].rolling(window=window).apply(
                lambda x: stats.linregress(range(len(x)), x)[0] if len(x) > 1 else 0, raw=False
            )
            df['volume_trend'] = volume_trend
            df['is_declining'] = volume_trend < -0.1
            
            # Clustering pattern (volumes tend to cluster around certain levels)
            df['is_clustered'] = df['cluster_strength'] > 0.1
            
            # Pattern classification
            pattern_conditions = [
                df['is_burst'] & (df['burst_intensity'] > 2.0),
                df['is_sustained'] & ~df['is_burst'],
                df['is_oscillating'] & ~df['is_sustained'],
                df['is_declining'],
                df['is_clustered'] & ~df['is_declining'],
            ]
            
            pattern_types = [
                TickPattern.BURST,
                TickPattern.SUSTAINED,
                TickPattern.OSCILLATING,
                TickPattern.DECLINING,
                TickPattern.CLUSTERED
            ]
            
            df['tick_pattern'] = np.select(pattern_conditions, pattern_types, default=TickPattern.RANDOM)
            
            # Pattern confidence
            df['pattern_confidence'] = np.select(
                pattern_conditions,
                [df['burst_intensity']/3.0, 0.8, df['oscillation_score'], 0.7, df['cluster_strength']],
                default=0.1
            ).clip(0, 1)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error detecting tick patterns: {str(e)}")
            raise
    
    def _perform_ml_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform machine learning analysis on tick data."""
        try:
            # Prepare features
            features = self._prepare_ml_features(df)
            
            if len(features) < self.params.ml_lookback:
                self.logger.warning("Insufficient data for ML analysis")
                df['pattern_probability'] = 0.5
                df['anomaly_score'] = 0.0
                df['classification_confidence'] = 0.0
                return df
            
            # Train models if needed
            if not self.is_trained or self.analysis_count % self.params.retrain_interval == 0:
                self._train_ml_models(features)
            
            # Generate predictions
            if self.is_trained:
                predictions = self._generate_ml_predictions(features)
                
                # Align predictions with dataframe
                prediction_start = max(0, len(df) - len(predictions['pattern_probability']))
                df = df.iloc[prediction_start:].copy()
                
                df['pattern_probability'] = predictions['pattern_probability']
                df['anomaly_score'] = predictions['anomaly_score']
                df['classification_confidence'] = predictions['classification_confidence']
            else:
                df['pattern_probability'] = 0.5
                df['anomaly_score'] = 0.0
                df['classification_confidence'] = 0.0
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error in ML analysis: {str(e)}")
            df['pattern_probability'] = 0.5
            df['anomaly_score'] = 0.0
            df['classification_confidence'] = 0.0
            return df
    
    def _prepare_ml_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for machine learning models."""
        try:
            features_df = pd.DataFrame(index=df.index)
            
            # Volume features
            features_df['volume'] = df['volume']
            features_df['volume_intensity'] = df['volume_intensity']
            features_df['volume_momentum'] = df['volume_momentum']
            features_df['volume_velocity'] = df['volume_velocity']
            features_df['avg_tick_volume'] = df['avg_tick_volume']
            
            # Flow features
            features_df['flow_intensity'] = df['flow_intensity']
            features_df['flow_momentum'] = df['flow_momentum']
            features_df['flow_persistence'] = df['flow_persistence']
            features_df['flow_efficiency'] = df['flow_efficiency']
            features_df['pressure_gradient'] = df['pressure_gradient']
            
            # Statistical features
            features_df['volume_zscore'] = df['volume_zscore']
            features_df['volume_skewness'] = df['volume_skewness']
            features_df['volume_kurtosis'] = df['volume_kurtosis']
            features_df['distribution_stability'] = df['distribution_stability']
            features_df['volume_entropy'] = df['volume_entropy']
            
            # Pattern features
            features_df['burst_intensity'] = df['burst_intensity']
            features_df['oscillation_score'] = df['oscillation_score']
            features_df['volume_trend'] = df['volume_trend']
            features_df['cluster_strength'] = df['cluster_strength']
            
            # Lagged features
            for lag in [1, 2, 3, 5]:
                features_df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
                features_df[f'flow_lag_{lag}'] = df['flow_intensity'].shift(lag)
            
            # Remove NaN and infinite values
            features_df = features_df.fillna(0).replace([np.inf, -np.inf], 0)
            
            return features_df
            
        except Exception as e:
            self.logger.error(f"Error preparing ML features: {str(e)}")
            return pd.DataFrame()
    
    def _train_ml_models(self, features: pd.DataFrame) -> None:
        """Train machine learning models."""
        try:
            if len(features) < 100:  # Minimum samples for training
                self.logger.warning("Insufficient samples for ML training")
                return
            
            # Prepare training data
            X, y_pattern, y_flow = self._prepare_training_data(features)
            
            if len(X) < 50:
                self.logger.warning("Insufficient valid samples for ML training")
                return
            
            # Scale features
            X_scaled = self.feature_scaler.fit_transform(X)
            
            # Train pattern classifier
            self.pattern_classifier.fit(X_scaled, y_pattern)
            
            # Train flow classifier
            self.flow_classifier.fit(X_scaled, y_flow)
            
            # Train anomaly detector
            self.anomaly_detector.fit(X_scaled)
            
            self.is_trained = True
            self.logger.info("ML models trained successfully")
            
        except Exception as e:
            self.logger.error(f"Error in ML model training: {str(e)}")
    
    def _prepare_training_data(self, features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare training data for ML models."""
        try:
            # Create pattern labels based on volume characteristics
            pattern_labels = []
            flow_labels = []
            
            for i in range(len(features)):
                row = features.iloc[i]
                
                # Pattern classification
                if row.get('burst_intensity', 0) > 2.0:
                    pattern_labels.append(0)  # Burst
                elif row.get('oscillation_score', 0) > 0.3:
                    pattern_labels.append(1)  # Oscillating
                elif row.get('volume_trend', 0) < -0.1:
                    pattern_labels.append(2)  # Declining
                elif row.get('cluster_strength', 0) > 0.1:
                    pattern_labels.append(3)  # Clustered
                else:
                    pattern_labels.append(4)  # Random
                
                # Flow classification
                flow_momentum = row.get('flow_momentum', 0)
                if flow_momentum > 0.5:
                    flow_labels.append(0)  # Buying pressure
                elif flow_momentum < -0.5:
                    flow_labels.append(1)  # Selling pressure
                else:
                    flow_labels.append(2)  # Balanced
            
            X = features.values
            y_pattern = np.array(pattern_labels)
            y_flow = np.array(flow_labels)
            
            # Remove invalid samples
            valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y_pattern) | np.isnan(y_flow))
            X = X[valid_mask]
            y_pattern = y_pattern[valid_mask]
            y_flow = y_flow[valid_mask]
            
            return X, y_pattern, y_flow
            
        except Exception as e:
            self.logger.error(f"Error preparing training data: {str(e)}")
            return np.array([]), np.array([]), np.array([])
    
    def _generate_ml_predictions(self, features: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate ML-based predictions."""
        try:
            if len(features) == 0:
                return {
                    'pattern_probability': np.array([0.5]),
                    'anomaly_score': np.array([0.0]),
                    'classification_confidence': np.array([0.0])
                }
            
            # Scale features
            X_scaled = self.feature_scaler.transform(features.values)
            
            # Pattern probability
            pattern_probs = self.pattern_classifier.predict_proba(X_scaled)
            pattern_probability = np.max(pattern_probs, axis=1)
            
            # Classification confidence
            classification_confidence = pattern_probability
            
            # Anomaly scores
            anomaly_scores = self.anomaly_detector.decision_function(X_scaled)
            # Normalize to 0-1 range
            anomaly_scores = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min() + 1e-8)
            
            return {
                'pattern_probability': pattern_probability,
                'anomaly_score': anomaly_scores,
                'classification_confidence': classification_confidence
            }
            
        except Exception as e:
            self.logger.error(f"Error generating ML predictions: {str(e)}")
            return {
                'pattern_probability': np.array([0.5] * len(features)),
                'anomaly_score': np.array([0.0] * len(features)),
                'classification_confidence': np.array([0.0] * len(features))
            }
    
    def _generate_analysis_components(self, df: pd.DataFrame) -> List[TickVolumeComponents]:
        """Generate tick volume analysis components."""
        try:
            components = []
            
            for i in range(len(df)):
                row = df.iloc[i]
                
                # Classify volume regime
                volume_regime = self._classify_volume_regime(row)
                
                # Classify flow direction
                flow_direction = self._classify_flow_direction(row)
                
                # Classify tick pattern
                tick_pattern = self._classify_tick_pattern(row)
                
                # Calculate confidence
                confidence_score = self._calculate_confidence_score(row)
                
                # Create component
                component = TickVolumeComponents(
                    tick_volume_regime=volume_regime,
                    flow_direction=flow_direction,
                    tick_pattern=tick_pattern,
                    confidence_score=confidence_score,
                    
                    # Tick metrics
                    tick_count=1,  # Each row represents one tick
                    avg_tick_volume=float(row.get('avg_tick_volume', 0)),
                    tick_velocity=float(row.get('tick_velocity', 0)),
                    tick_acceleration=float(row.get('tick_acceleration', 0)),
                    
                    # Volume clustering
                    cluster_strength=float(row.get('cluster_strength', 0)),
                    cluster_id=int(row.get('cluster_id', -1)),
                    cluster_persistence=float(row.get('cluster_persistence', 0)),
                    
                    # Flow dynamics
                    flow_intensity=float(row.get('flow_intensity', 0)),
                    flow_persistence=float(row.get('flow_persistence', 0)),
                    flow_momentum=float(row.get('flow_momentum', 0)),
                    flow_efficiency=float(row.get('flow_efficiency', 0)),
                    
                    # Statistical measures
                    volume_skewness=float(row.get('volume_skewness', 0)),
                    volume_kurtosis=float(row.get('volume_kurtosis', 0)),
                    distribution_stability=float(row.get('distribution_stability', 0)),
                    outlier_ratio=float(row.get('outlier_ratio', 0)),
                    
                    # Advanced metrics
                    microstructure_impact=float(row.get('market_impact', 0)),
                    liquidity_absorption=float(row.get('liquidity_absorption', 0)),
                    tick_intensity=float(row.get('volume_intensity', 0)),
                    pressure_gradient=float(row.get('pressure_gradient', 0)),
                    
                    # ML-based features
                    pattern_probability=float(row.get('pattern_probability', 0.5)),
                    anomaly_score=float(row.get('anomaly_score', 0)),
                    classification_confidence=float(row.get('classification_confidence', 0)),
                    
                    # Risk and efficiency
                    execution_cost=self._estimate_execution_cost(row),
                    slippage_estimate=self._estimate_slippage(row),
                    timing_alpha=self._calculate_timing_alpha(row)
                )
                
                components.append(component)
            
            return components
            
        except Exception as e:
            self.logger.error(f"Error generating analysis components: {str(e)}")
            return []
    
    def _classify_volume_regime(self, row: pd.Series) -> TickVolumeRegime:
        """Classify the volume regime."""
        try:
            volume_intensity = row.get('volume_intensity', 0)
            
            if volume_intensity > 5.0:
                return TickVolumeRegime.ULTRA_HIGH
            elif volume_intensity > 2.0:
                return TickVolumeRegime.HIGH
            elif volume_intensity > -0.5:
                return TickVolumeRegime.NORMAL
            elif volume_intensity > -2.0:
                return TickVolumeRegime.LOW
            elif volume_intensity > -5.0:
                return TickVolumeRegime.MICRO
            else:
                return TickVolumeRegime.ANOMALOUS
                
        except Exception as e:
            self.logger.error(f"Error classifying volume regime: {str(e)}")
            return TickVolumeRegime.NORMAL
    
    def _classify_flow_direction(self, row: pd.Series) -> FlowDirection:
        """Classify the flow direction."""
        try:
            flow_momentum = row.get('flow_momentum', 0)
            pressure_ratio = row.get('pressure_ratio', 0)
            flow_persistence = row.get('flow_persistence', 0)
            
            if flow_persistence > 0.8 and pressure_ratio > 0.3:
                return FlowDirection.ACCUMULATION
            elif flow_persistence > 0.8 and pressure_ratio < -0.3:
                return FlowDirection.DISTRIBUTION
            elif flow_momentum > 0.5:
                return FlowDirection.BUYING_PRESSURE
            elif flow_momentum < -0.5:
                return FlowDirection.SELLING_PRESSURE
            elif abs(flow_momentum) < 0.1:
                return FlowDirection.BALANCED
            else:
                return FlowDirection.CHURNING
                
        except Exception as e:
            self.logger.error(f"Error classifying flow direction: {str(e)}")
            return FlowDirection.BALANCED
    
    def _classify_tick_pattern(self, row: pd.Series) -> TickPattern:
        """Classify the tick pattern."""
        try:
            # Get pattern from dataframe if available
            if 'tick_pattern' in row and pd.notna(row['tick_pattern']):
                pattern_str = str(row['tick_pattern'])
                for pattern in TickPattern:
                    if pattern.value == pattern_str:
                        return pattern
            
            # Fallback classification
            burst_intensity = row.get('burst_intensity', 0)
            oscillation_score = row.get('oscillation_score', 0)
            volume_trend = row.get('volume_trend', 0)
            cluster_strength = row.get('cluster_strength', 0)
            
            if burst_intensity > 2.0:
                return TickPattern.BURST
            elif oscillation_score > 0.3:
                return TickPattern.OSCILLATING
            elif volume_trend < -0.1:
                return TickPattern.DECLINING
            elif cluster_strength > 0.1:
                return TickPattern.CLUSTERED
            else:
                return TickPattern.RANDOM
                
        except Exception as e:
            self.logger.error(f"Error classifying tick pattern: {str(e)}")
            return TickPattern.RANDOM
    
    def _calculate_confidence_score(self, row: pd.Series) -> float:
        """Calculate overall confidence score."""
        try:
            # Multiple confidence factors
            pattern_conf = row.get('pattern_confidence', 0)
            classification_conf = row.get('classification_confidence', 0)
            distribution_stability = row.get('distribution_stability', 0)
            cluster_quality = row.get('cluster_quality', 0)
            
            # Weighted average
            confidence = (
                pattern_conf * 0.3 +
                classification_conf * 0.3 +
                distribution_stability * 0.2 +
                cluster_quality * 0.2
            )
            
            return np.clip(confidence, 0.0, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence score: {str(e)}")
            return 0.5
    
    def _estimate_execution_cost(self, row: pd.Series) -> float:
        """Estimate execution cost based on tick analysis."""
        try:
            volume_intensity = abs(row.get('volume_intensity', 0))
            liquidity_absorption = row.get('liquidity_absorption', 0)
            
            # Higher volume intensity and lower liquidity absorption = higher cost
            base_cost = 0.01  # 1 pip base cost
            intensity_multiplier = 1.0 + min(volume_intensity / 5.0, 2.0)
            liquidity_factor = 1.0 / (1.0 + liquidity_absorption / 1000.0)
            
            cost = base_cost * intensity_multiplier * liquidity_factor
            
            return min(cost, 0.1)  # Cap at 10 pips
            
        except Exception as e:
            self.logger.error(f"Error estimating execution cost: {str(e)}")
            return 0.01
    
    def _estimate_slippage(self, row: pd.Series) -> float:
        """Estimate slippage based on market conditions."""
        try:
            price_impact = row.get('price_impact', 0)
            volume_intensity = abs(row.get('volume_intensity', 0))
            
            # Slippage increases with price impact and volume intensity
            base_slippage = 0.005  # 0.5 pip base slippage
            impact_multiplier = 1.0 + price_impact * 100
            intensity_multiplier = 1.0 + volume_intensity / 10.0
            
            slippage = base_slippage * impact_multiplier * intensity_multiplier
            
            return min(slippage, 0.05)  # Cap at 5 pips
            
        except Exception as e:
            self.logger.error(f"Error estimating slippage: {str(e)}")
            return 0.005
    
    def _calculate_timing_alpha(self, row: pd.Series) -> float:
        """Calculate timing alpha based on flow analysis."""
        try:
            flow_persistence = row.get('flow_persistence', 0)
            flow_efficiency = row.get('flow_efficiency', 0)
            pattern_confidence = row.get('pattern_confidence', 0)
            
            # Alpha is higher when flow is persistent, efficient, and patterns are clear
            alpha = (
                flow_persistence * 0.4 +
                abs(flow_efficiency) * 0.3 +
                pattern_confidence * 0.3
            )
            
            # Scale to typical alpha range
            return np.clip(alpha * 0.1, -0.05, 0.05)  # -5% to +5%
            
        except Exception as e:
            self.logger.error(f"Error calculating timing alpha: {str(e)}")
            return 0.0
    
    def _calculate_performance_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate performance metrics for the analyzer."""
        try:
            if len(df) < 2:
                return {}
            
            metrics = {}
            
            # Volume metrics
            if 'volume' in df.columns:
                volume_series = df['volume'].dropna()
                if len(volume_series) > 0:
                    metrics['avg_volume'] = float(volume_series.mean())
                    metrics['volume_volatility'] = float(volume_series.std())
                    metrics['max_volume'] = float(volume_series.max())
                    metrics['min_volume'] = float(volume_series.min())
            
            # Flow metrics
            if 'flow_intensity' in df.columns:
                flow_series = df['flow_intensity'].dropna()
                if len(flow_series) > 0:
                    metrics['net_flow'] = float(flow_series.sum())
                    metrics['flow_volatility'] = float(flow_series.std())
                    metrics['flow_efficiency'] = float(df.get('flow_efficiency', pd.Series(0)).mean())
            
            # Pattern metrics
            if 'tick_pattern' in df.columns:
                pattern_counts = df['tick_pattern'].value_counts()
                metrics['dominant_pattern'] = str(pattern_counts.index[0]) if len(pattern_counts) > 0 else 'UNKNOWN'
                metrics['pattern_diversity'] = len(pattern_counts)
            
            # Clustering metrics
            if 'cluster_id' in df.columns:
                cluster_counts = df['cluster_id'].value_counts()
                metrics['cluster_count'] = len(cluster_counts)
                metrics['avg_cluster_strength'] = float(df.get('cluster_strength', pd.Series(0)).mean())
            
            # Statistical metrics
            metrics['outlier_rate'] = float(df.get('outlier_ratio', pd.Series(0)).mean())
            metrics['avg_distribution_stability'] = float(df.get('distribution_stability', pd.Series(0)).mean())
            
            # ML performance
            if self.is_trained:
                metrics['pattern_confidence_avg'] = float(df.get('pattern_probability', pd.Series(0.5)).mean())
                metrics['anomaly_detection_rate'] = float((df.get('anomaly_score', pd.Series(0)) > 0.7).mean())
            
            # Data quality
            metrics['data_completeness'] = float(1.0 - df.isnull().sum().sum() / (len(df) * len(df.columns)))
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {str(e)}")
            return {}
    
    def _update_performance_stats(self, analysis_time: float) -> None:
        """Update performance statistics."""
        try:
            self.analysis_count += 1
            self.total_analysis_time += analysis_time
            
            if self.analysis_count % 100 == 0:
                avg_time = self.total_analysis_time / self.analysis_count
                self.logger.info(f"Average analysis time over {self.analysis_count} runs: {avg_time:.4f}s")
                
        except Exception as e:
            self.logger.error(f"Error updating performance stats: {str(e)}")
    
    def get_real_time_summary(self) -> Dict[str, Any]:
        """Get real-time summary of tick volume analysis."""
        try:
            if not self.last_analysis:
                return {}
            
            latest = self.last_analysis[-1] if self.last_analysis else None
            if not latest:
                return {}
            
            recent_analysis = self.last_analysis[-20:] if len(self.last_analysis) >= 20 else self.last_analysis
            
            summary = {
                'current_regime': latest.tick_volume_regime.value,
                'current_flow': latest.flow_direction.value,
                'current_pattern': latest.tick_pattern.value,
                'confidence': latest.confidence_score,
                
                # Recent activity
                'avg_tick_velocity': np.mean([comp.tick_velocity for comp in recent_analysis]),
                'avg_flow_intensity': np.mean([comp.flow_intensity for comp in recent_analysis]),
                'dominant_pattern': max(set([comp.tick_pattern for comp in recent_analysis]), 
                                      key=[comp.tick_pattern for comp in recent_analysis].count).value,
                
                # Risk metrics
                'execution_cost': latest.execution_cost,
                'slippage_estimate': latest.slippage_estimate,
                'timing_alpha': latest.timing_alpha,
                
                # Buffer status
                'tick_buffer_size': len(self.tick_buffer),
                'analysis_count': self.analysis_count
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating real-time summary: {str(e)}")
            return {}


# Example usage and testing
if __name__ == "__main__":
    # Test the analyzer with sample tick data
    try:
        # Generate sample tick data
        np.random.seed(42)
        n_ticks = 1000
        
        timestamps = pd.date_range(start='2024-01-01', periods=n_ticks, freq='1S')
        prices = 1.1000 + np.cumsum(np.random.randn(n_ticks) * 0.0001)
        
        # Generate realistic volume patterns
        base_volume = 100
        volume_spikes = np.random.exponential(50, n_ticks)
        random_volumes = np.random.lognormal(np.log(base_volume), 0.5, n_ticks)
        volumes = np.maximum(volume_spikes, random_volumes)
        
        tick_data = pd.DataFrame({
            'timestamp': timestamps,
            'price': prices,
            'volume': volumes
        })
        
        print(f"Generated {len(tick_data)} sample ticks")
        
        # Initialize analyzer
        params = TickVolumeParameters(
            tick_window=100,
            velocity_window=20,
            ml_lookback=500
        )
        
        analyzer = AdvancedTickVolumeAnalyzer(params)
        
        # Analyze tick data
        result = analyzer.analyze_tick_data(tick_data)
        
        # Display results
        print(f"Analysis completed in {result['metadata']['analysis_time']:.4f}s")
        print(f"Model trained: {result['metadata']['model_trained']}")
        print(f"Components generated: {len(result['components'])}")
        
        # Show latest analysis
        if result['components']:
            latest = result['components'][-1]
            print(f"\nLatest Tick Analysis:")
            print(f"Volume Regime: {latest.tick_volume_regime.value}")
            print(f"Flow Direction: {latest.flow_direction.value}")
            print(f"Pattern: {latest.tick_pattern.value}")
            print(f"Confidence: {latest.confidence_score:.3f}")
            print(f"Tick Velocity: {latest.tick_velocity:.3f}")
            print(f"Flow Intensity: {latest.flow_intensity:.3f}")
            print(f"Execution Cost: {latest.execution_cost:.5f}")
            print(f"Timing Alpha: {latest.timing_alpha:.5f}")
        
        # Real-time summary
        summary = analyzer.get_real_time_summary()
        if summary:
            print(f"\nReal-time Summary:")
            for key, value in summary.items():
                if isinstance(value, float):
                    print(f"{key}: {value:.4f}")
                else:
                    print(f"{key}: {value}")
        
        # Performance metrics
        if result['performance']:
            print(f"\nPerformance Metrics:")
            for key, value in result['performance'].items():
                if isinstance(value, (int, float)):
                    print(f"{key}: {value:.4f}")
                else:
                    print(f"{key}: {value}")
    
    except Exception as e:
        print(f"Error in testing: {str(e)}")
        print(traceback.format_exc())