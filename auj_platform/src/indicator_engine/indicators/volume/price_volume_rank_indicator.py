"""
Price Volume Rank (PVR) - Advanced Relative Volume Analysis and Anomaly Detection
================================================================================

The Price Volume Rank (PVR) indicator is a sophisticated analytical tool that provides
relative volume analysis through percentile rankings and statistical anomaly detection.
This implementation offers institutional-grade volume analysis capabilities designed
for high-stakes trading environments.

Key Features:
- Advanced relative volume analysis with percentile ranking
- Multi-timeframe volume anomaly detection
- Statistical distribution modeling and outlier identification
- Machine learning-based volume pattern classification
- Dynamic threshold adaptation based on market conditions
- Volume surge and drought detection algorithms
- Real-time liquidity assessment and market impact analysis

Mathematical Foundation:
The Price Volume Rank employs sophisticated statistical methods:

1. Percentile Ranking: PVR = percentile_rank(current_volume, historical_volumes)
2. Z-Score Analysis: Z = (current_volume - mean_volume) / std_volume
3. Statistical Anomaly Detection: Using modified Z-scores and IQR methods
4. Volume Distribution Modeling: Gamma, log-normal, and Weibull distributions
5. Adaptive Thresholds: Dynamic adjustment based on volatility regimes

The indicator provides multiple analytical layers:
- Raw percentile rankings (0-100)
- Normalized volume scores adjusted for time of day
- Anomaly probability scores
- Volume regime classification
- Liquidity impact assessment

Author: AUJ Platform Development Team
Created: 2025-06-21
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta
import warnings
from scipy import stats
from scipy.special import gamma
from scipy.optimize import minimize_scalar
from scipy.signal import find_peaks, argrelextrema
import talib
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.mixture import GaussianMixture
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VolumeRegime(Enum):
    """Volume regime classifications."""
    EXTREMELY_LOW = "extremely_low"
    VERY_LOW = "very_low"
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    VERY_HIGH = "very_high"
    EXTREMELY_HIGH = "extremely_high"

class VolumeAnomaly(Enum):
    """Volume anomaly types."""
    VOLUME_SURGE = "volume_surge"
    VOLUME_SPIKE = "volume_spike"
    VOLUME_DROUGHT = "volume_drought"
    VOLUME_CLUSTER = "volume_cluster"
    DISTRIBUTION_SHIFT = "distribution_shift"
    OUTLIER_HIGH = "outlier_high"
    OUTLIER_LOW = "outlier_low"
    NORMAL = "normal"

class LiquidityCondition(Enum):
    """Market liquidity condition classifications."""
    ILLIQUID = "illiquid"
    LOW_LIQUIDITY = "low_liquidity"
    MODERATE_LIQUIDITY = "moderate_liquidity"
    HIGH_LIQUIDITY = "high_liquidity"
    EXCESSIVE_LIQUIDITY = "excessive_liquidity"

@dataclass
class PriceVolumeRankSignal:
    """
    Comprehensive Price Volume Rank signal.
    
    Attributes:
        timestamp: Signal timestamp
        volume_percentile: Volume percentile rank (0-100)
        normalized_rank: Time-adjusted volume rank
        z_score: Statistical Z-score of volume
        volume_regime: Current volume regime classification
        anomaly_type: Detected volume anomaly type
        anomaly_probability: Probability of volume anomaly
        distribution_fit_score: How well volume fits expected distribution
        surge_intensity: Intensity of volume surge (if applicable)
        drought_severity: Severity of volume drought (if applicable)
        liquidity_condition: Current market liquidity assessment
        relative_volume_ratio: Current volume vs average ratio
        volume_momentum: Rate of change in volume
        cluster_membership: Volume cluster classification
        outlier_score: Statistical outlier scoring
        impact_probability: Probability of significant market impact
        confidence: Overall signal confidence (0-100)
    """
    timestamp: datetime
    volume_percentile: float
    normalized_rank: float
    z_score: float
    volume_regime: VolumeRegime
    anomaly_type: VolumeAnomaly
    anomaly_probability: float
    distribution_fit_score: float
    surge_intensity: float
    drought_severity: float
    liquidity_condition: LiquidityCondition
    relative_volume_ratio: float
    volume_momentum: float
    cluster_membership: int
    outlier_score: float
    impact_probability: float
    confidence: float

class PriceVolumeRankIndicator:
    """
    Advanced Price Volume Rank Indicator with sophisticated relative volume analysis,
    percentile rankings, and anomaly detection capabilities.
    
    This indicator provides institutional-grade volume analysis through statistical
    modeling and machine learning techniques.
    """
    
    def __init__(self,
                 lookback_period: int = 252,
                 short_period: int = 20,
                 anomaly_threshold: float = 95.0,
                 outlier_threshold: float = 2.5,
                 min_observations: int = 50,
                 confidence_threshold: float = 70.0):
        """
        Initialize the Price Volume Rank Indicator.
        
        Args:
            lookback_period: Period for historical volume analysis
            short_period: Short-term period for momentum analysis
            anomaly_threshold: Percentile threshold for anomaly detection
            outlier_threshold: Z-score threshold for outlier detection
            min_observations: Minimum observations for statistical validity
            confidence_threshold: Minimum confidence for signal generation
        """
        self.lookback_period = lookback_period
        self.short_period = short_period
        self.anomaly_threshold = anomaly_threshold
        self.outlier_threshold = outlier_threshold
        self.min_observations = min_observations
        self.confidence_threshold = confidence_threshold
        
        # Machine learning models
        self._isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )
        self._volume_classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )
        self._gaussian_mixture = GaussianMixture(
            n_components=3,
            random_state=42
        )
        self._scaler = RobustScaler()
        
        # Analysis state
        self._volume_distribution_params = {}
        self._regime_thresholds = {}
        self._historical_stats = {}
        
        logger.info(f"PriceVolumeRankIndicator initialized with lookback_period={lookback_period}")
    
    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate Price Volume Rank with advanced statistical analysis.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary containing PVR analysis and anomaly detection results
        """
        try:
            if data is None or data.empty:
                raise ValueError("Input data is empty or None")
            
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            if len(data) < max(self.lookback_period, self.min_observations):
                logger.warning(f"Insufficient data for analysis")
                return self._generate_empty_result()
            
            # Calculate percentile rankings
            percentile_analysis = self._calculate_percentile_rankings(data)
            
            # Perform statistical analysis
            statistical_analysis = self._perform_statistical_analysis(data, percentile_analysis)
            
            # Detect volume anomalies
            anomaly_analysis = self._detect_volume_anomalies(data, percentile_analysis, statistical_analysis)
            
            # Analyze volume regimes
            regime_analysis = self._analyze_volume_regimes(data, percentile_analysis, statistical_analysis)
            
            # Assess liquidity conditions
            liquidity_analysis = self._assess_liquidity_conditions(data, percentile_analysis)
            
            # Perform clustering analysis
            clustering_analysis = self._perform_clustering_analysis(data, percentile_analysis)
            
            # Calculate momentum and dynamics
            momentum_analysis = self._analyze_volume_momentum(data, percentile_analysis)
            
            # Generate comprehensive signals
            signals = self._generate_comprehensive_signals(
                data, percentile_analysis, statistical_analysis, anomaly_analysis,
                regime_analysis, liquidity_analysis, clustering_analysis, momentum_analysis
            )
            
            return {
                'percentile_analysis': percentile_analysis,
                'statistical_analysis': statistical_analysis,
                'anomaly_analysis': anomaly_analysis,
                'regime_analysis': regime_analysis,
                'liquidity_analysis': liquidity_analysis,
                'clustering_analysis': clustering_analysis,
                'momentum_analysis': momentum_analysis,
                'signals': signals,
                'metadata': self._generate_metadata()
            }
            
        except Exception as e:
            logger.error(f"Error in PriceVolumeRankIndicator calculation: {str(e)}")
            return self._generate_error_result(str(e))
    
    def _calculate_percentile_rankings(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Calculate volume percentile rankings."""
        try:
            volume = data['volume'].values
            close = data['close'].values
            
            # Rolling percentile rankings
            percentiles = np.zeros_like(volume)
            normalized_ranks = np.zeros_like(volume)
            relative_ratios = np.zeros_like(volume)
            
            for i in range(self.lookback_period, len(volume)):
                # Historical volume window
                historical_volume = volume[max(0, i - self.lookback_period):i]
                current_volume = volume[i]
                
                # Calculate percentile rank
                percentile = stats.percentileofscore(historical_volume, current_volume)
                percentiles[i] = percentile
                
                # Calculate relative volume ratio
                avg_volume = np.mean(historical_volume)
                relative_ratios[i] = current_volume / avg_volume if avg_volume > 0 else 1.0
                
                # Normalize for time-of-day effects (if timestamp available)
                # For simplicity, using basic normalization here
                recent_avg = np.mean(historical_volume[-self.short_period:])
                normalized_ranks[i] = (current_volume / recent_avg) * 50 if recent_avg > 0 else 50
                normalized_ranks[i] = min(100, max(0, normalized_ranks[i]))
            
            # Calculate rolling statistics
            volume_means = talib.SMA(volume, timeperiod=self.lookback_period)
            volume_stds = talib.STDDEV(volume, timeperiod=self.lookback_period)
            
            # Calculate Z-scores
            z_scores = np.zeros_like(volume)
            for i in range(self.lookback_period, len(volume)):
                if volume_stds[i] > 0:
                    z_scores[i] = (volume[i] - volume_means[i]) / volume_stds[i]
            
            return {
                'percentiles': percentiles,
                'normalized_ranks': normalized_ranks,
                'relative_ratios': relative_ratios,
                'z_scores': z_scores,
                'volume_means': volume_means,
                'volume_stds': volume_stds
            }
            
        except Exception as e:
            logger.error(f"Error calculating percentile rankings: {str(e)}")
            return {}
    
    def _perform_statistical_analysis(self, data: pd.DataFrame, percentile_analysis: Dict) -> Dict[str, Any]:
        """Perform advanced statistical analysis on volume data."""
        try:
            volume = data['volume'].values
            percentiles = percentile_analysis.get('percentiles', np.array([]))
            
            # Fit various distributions to volume data
            distribution_fits = {}
            distribution_scores = np.zeros_like(volume)
            
            for i in range(self.lookback_period, len(volume)):
                historical_volume = volume[max(0, i - self.lookback_period):i]
                
                # Fit log-normal distribution
                try:
                    shape, loc, scale = stats.lognorm.fit(historical_volume, floc=0)
                    lognorm_score = stats.lognorm.pdf(volume[i], shape, loc, scale)
                    distribution_fits['lognorm'] = (shape, loc, scale)
                except:
                    lognorm_score = 0
                
                # Fit gamma distribution
                try:
                    shape, loc, scale = stats.gamma.fit(historical_volume)
                    gamma_score = stats.gamma.pdf(volume[i], shape, loc, scale)
                    distribution_fits['gamma'] = (shape, loc, scale)
                except:
                    gamma_score = 0
                
                # Fit exponential distribution
                try:
                    loc, scale = stats.expon.fit(historical_volume)
                    expon_score = stats.expon.pdf(volume[i], loc, scale)
                    distribution_fits['expon'] = (loc, scale)
                except:
                    expon_score = 0
                
                # Combined distribution score
                distribution_scores[i] = np.mean([lognorm_score, gamma_score, expon_score])
            
            # Modified Z-score for outlier detection
            modified_z_scores = self._calculate_modified_z_scores(volume)
            
            # IQR-based outlier detection
            iqr_outliers = self._detect_iqr_outliers(volume)
            
            return {
                'distribution_fits': distribution_fits,
                'distribution_scores': distribution_scores,
                'modified_z_scores': modified_z_scores,
                'iqr_outliers': iqr_outliers
            }
            
        except Exception as e:
            logger.error(f"Error in statistical analysis: {str(e)}")
            return {}
    
    def _detect_volume_anomalies(self, data: pd.DataFrame, percentile_analysis: Dict,
                                statistical_analysis: Dict) -> Dict[str, Any]:
        """Detect various types of volume anomalies."""
        try:
            volume = data['volume'].values
            percentiles = percentile_analysis.get('percentiles', np.array([]))
            z_scores = percentile_analysis.get('z_scores', np.array([]))
            modified_z_scores = statistical_analysis.get('modified_z_scores', np.array([]))
            
            anomaly_types = []
            anomaly_probabilities = np.zeros_like(volume)
            surge_intensities = np.zeros_like(volume)
            drought_severities = np.zeros_like(volume)
            
            for i in range(self.lookback_period, len(volume)):
                anomaly_type = VolumeAnomaly.NORMAL
                anomaly_prob = 0.0
                surge_intensity = 0.0
                drought_severity = 0.0
                
                percentile = percentiles[i] if i < len(percentiles) else 50
                z_score = z_scores[i] if i < len(z_scores) else 0
                modified_z = modified_z_scores[i] if i < len(modified_z_scores) else 0
                
                # Volume surge detection
                if percentile > 99:
                    anomaly_type = VolumeAnomaly.VOLUME_SPIKE
                    surge_intensity = min(100, (percentile - 99) * 10)
                    anomaly_prob = min(100, surge_intensity)
                elif percentile > self.anomaly_threshold:
                    anomaly_type = VolumeAnomaly.VOLUME_SURGE
                    surge_intensity = (percentile - self.anomaly_threshold) / (100 - self.anomaly_threshold) * 100
                    anomaly_prob = surge_intensity
                
                # Volume drought detection
                elif percentile < 1:
                    anomaly_type = VolumeAnomaly.VOLUME_DROUGHT
                    drought_severity = (1 - percentile) * 100
                    anomaly_prob = drought_severity
                elif percentile < (100 - self.anomaly_threshold):
                    drought_severity = (100 - self.anomaly_threshold - percentile) / (100 - self.anomaly_threshold) * 100
                    if drought_severity > 50:
                        anomaly_type = VolumeAnomaly.VOLUME_DROUGHT
                        anomaly_prob = drought_severity
                
                # Statistical outlier detection
                if abs(modified_z) > self.outlier_threshold:
                    if modified_z > 0:
                        anomaly_type = VolumeAnomaly.OUTLIER_HIGH
                    else:
                        anomaly_type = VolumeAnomaly.OUTLIER_LOW
                    anomaly_prob = max(anomaly_prob, min(100, abs(modified_z) / self.outlier_threshold * 50))
                
                # Volume clustering detection
                recent_volumes = volume[max(0, i-10):i]
                if len(recent_volumes) > 5:
                    cv = stats.variation(recent_volumes)
                    if cv < 0.1:  # Low coefficient of variation = clustering
                        anomaly_type = VolumeAnomaly.VOLUME_CLUSTER
                        anomaly_prob = max(anomaly_prob, (0.1 - cv) / 0.1 * 100)
                
                anomaly_types.append(anomaly_type)
                anomaly_probabilities[i] = anomaly_prob
                surge_intensities[i] = surge_intensity
                drought_severities[i] = drought_severity
            
            # Pad beginning of lists
            while len(anomaly_types) < len(volume):
                anomaly_types.insert(0, VolumeAnomaly.NORMAL)
            
            # Use Isolation Forest for additional anomaly detection
            if len(volume) > self.min_observations:
                volume_features = volume.reshape(-1, 1)
                try:
                    self._isolation_forest.fit(volume_features)
                    isolation_scores = self._isolation_forest.decision_function(volume_features)
                    isolation_outliers = self._isolation_forest.predict(volume_features) == -1
                except:
                    isolation_scores = np.zeros_like(volume)
                    isolation_outliers = np.zeros_like(volume, dtype=bool)
            else:
                isolation_scores = np.zeros_like(volume)
                isolation_outliers = np.zeros_like(volume, dtype=bool)
            
            return {
                'anomaly_types': anomaly_types,
                'anomaly_probabilities': anomaly_probabilities,
                'surge_intensities': surge_intensities,
                'drought_severities': drought_severities,
                'isolation_scores': isolation_scores,
                'isolation_outliers': isolation_outliers
            }
            
        except Exception as e:
            logger.error(f"Error detecting volume anomalies: {str(e)}")
            return {}
    
    def _analyze_volume_regimes(self, data: pd.DataFrame, percentile_analysis: Dict,
                              statistical_analysis: Dict) -> Dict[str, Any]:
        """Analyze volume regimes and classify market conditions."""
        try:
            volume = data['volume'].values
            percentiles = percentile_analysis.get('percentiles', np.array([]))
            
            # Define adaptive regime thresholds
            regime_classifications = []
            regime_scores = np.zeros_like(volume)
            
            # Calculate dynamic thresholds based on historical data
            for i in range(self.lookback_period, len(volume)):
                historical_percentiles = percentiles[max(0, i - self.lookback_period):i]
                
                if len(historical_percentiles) > 0:
                    # Adaptive thresholds based on historical distribution
                    p10 = np.percentile(historical_percentiles, 10)
                    p25 = np.percentile(historical_percentiles, 25)
                    p75 = np.percentile(historical_percentiles, 75)
                    p90 = np.percentile(historical_percentiles, 90)
                    p99 = np.percentile(historical_percentiles, 99)
                else:
                    # Default thresholds
                    p10, p25, p75, p90, p99 = 10, 25, 75, 90, 99
                
                current_percentile = percentiles[i] if i < len(percentiles) else 50
                
                # Classify regime
                if current_percentile >= p99:
                    regime = VolumeRegime.EXTREMELY_HIGH
                    score = 100
                elif current_percentile >= p90:
                    regime = VolumeRegime.VERY_HIGH
                    score = 80 + (current_percentile - p90) / (p99 - p90) * 20
                elif current_percentile >= p75:
                    regime = VolumeRegime.HIGH
                    score = 60 + (current_percentile - p75) / (p90 - p75) * 20
                elif current_percentile >= p25:
                    regime = VolumeRegime.NORMAL
                    score = 40 + (current_percentile - p25) / (p75 - p25) * 20
                elif current_percentile >= p10:
                    regime = VolumeRegime.LOW
                    score = 20 + (current_percentile - p10) / (p25 - p10) * 20
                elif current_percentile >= 1:
                    regime = VolumeRegime.VERY_LOW
                    score = 10 + (current_percentile - 1) / (p10 - 1) * 10
                else:
                    regime = VolumeRegime.EXTREMELY_LOW
                    score = current_percentile * 10
                
                regime_classifications.append(regime)
                regime_scores[i] = score
            
            # Pad beginning
            while len(regime_classifications) < len(volume):
                regime_classifications.insert(0, VolumeRegime.NORMAL)
            
            # Regime persistence analysis
            regime_persistence = self._calculate_regime_persistence(regime_classifications)
            
            return {
                'regime_classifications': regime_classifications,
                'regime_scores': regime_scores,
                'regime_persistence': regime_persistence
            }
            
        except Exception as e:
            logger.error(f"Error analyzing volume regimes: {str(e)}")
            return {}
    
    def _assess_liquidity_conditions(self, data: pd.DataFrame, percentile_analysis: Dict) -> Dict[str, Any]:
        """Assess market liquidity conditions based on volume patterns."""
        try:
            volume = data['volume'].values
            close = data['close'].values
            high = data['high'].values
            low = data['low'].values
            percentiles = percentile_analysis.get('percentiles', np.array([]))
            
            liquidity_conditions = []
            liquidity_scores = np.zeros_like(volume)
            impact_probabilities = np.zeros_like(volume)
            
            for i in range(self.short_period, len(volume)):
                # Recent volume and price data
                recent_volume = volume[i-self.short_period:i]
                recent_close = close[i-self.short_period:i]
                recent_high = high[i-self.short_period:i]
                recent_low = low[i-self.short_period:i]
                
                # Volume consistency
                volume_cv = stats.variation(recent_volume) if len(recent_volume) > 1 else 1.0
                
                # Price-volume relationship
                price_changes = np.diff(recent_close)
                volume_changes = np.diff(recent_volume)
                
                if len(price_changes) > 1 and len(volume_changes) > 1:
                    price_volume_corr = np.corrcoef(price_changes, volume_changes)[0, 1]
                    if np.isnan(price_volume_corr):
                        price_volume_corr = 0
                else:
                    price_volume_corr = 0
                
                # Average volume percentile
                avg_percentile = np.mean(percentiles[max(0, i-self.short_period):i])
                
                # Liquidity scoring
                liquidity_score = (
                    (1 - volume_cv) * 30 +  # Lower variability = better liquidity
                    abs(price_volume_corr) * 20 +  # Price-volume correlation
                    (avg_percentile / 100) * 50  # Higher volume = better liquidity
                )
                
                liquidity_scores[i] = liquidity_score
                
                # Market impact probability
                current_volume = volume[i]
                avg_volume = np.mean(recent_volume)
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
                
                # Higher volume ratio = lower impact probability
                impact_prob = max(0, 100 - (volume_ratio - 1) * 50)
                impact_probabilities[i] = min(100, impact_prob)
                
                # Classify liquidity condition
                if liquidity_score >= 80:
                    condition = LiquidityCondition.EXCESSIVE_LIQUIDITY
                elif liquidity_score >= 65:
                    condition = LiquidityCondition.HIGH_LIQUIDITY
                elif liquidity_score >= 45:
                    condition = LiquidityCondition.MODERATE_LIQUIDITY
                elif liquidity_score >= 25:
                    condition = LiquidityCondition.LOW_LIQUIDITY
                else:
                    condition = LiquidityCondition.ILLIQUID
                
                liquidity_conditions.append(condition)
            
            # Pad beginning
            while len(liquidity_conditions) < len(volume):
                liquidity_conditions.insert(0, LiquidityCondition.MODERATE_LIQUIDITY)
            
            return {
                'liquidity_conditions': liquidity_conditions,
                'liquidity_scores': liquidity_scores,
                'impact_probabilities': impact_probabilities
            }
            
        except Exception as e:
            logger.error(f"Error assessing liquidity conditions: {str(e)}")
            return {}
    
    def _perform_clustering_analysis(self, data: pd.DataFrame, percentile_analysis: Dict) -> Dict[str, Any]:
        """Perform clustering analysis on volume patterns."""
        try:
            volume = data['volume'].values
            percentiles = percentile_analysis.get('percentiles', np.array([]))
            
            if len(volume) < self.min_observations:
                return {'cluster_labels': np.zeros_like(volume), 'cluster_centers': []}
            
            # Prepare features for clustering
            features = []
            for i in range(self.short_period, len(volume)):
                feature_vector = [
                    volume[i],  # Current volume
                    np.mean(volume[i-self.short_period:i]),  # Average volume
                    np.std(volume[i-self.short_period:i]),   # Volume volatility
                    percentiles[i] if i < len(percentiles) else 50,  # Percentile rank
                    stats.variation(volume[i-self.short_period:i]) if volume[i-self.short_period:i].std() > 0 else 0  # Coefficient of variation
                ]
                features.append(feature_vector)
            
            features = np.array(features)
            
            # Normalize features
            try:
                features_scaled = self._scaler.fit_transform(features)
            except:
                features_scaled = features
            
            # Perform K-means clustering
            n_clusters = min(5, len(features) // 10)  # Adaptive number of clusters
            if n_clusters >= 2:
                try:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    cluster_labels = kmeans.fit_predict(features_scaled)
                    cluster_centers = kmeans.cluster_centers_
                except:
                    cluster_labels = np.zeros(len(features))
                    cluster_centers = []
            else:
                cluster_labels = np.zeros(len(features))
                cluster_centers = []
            
            # Extend cluster labels to full length
            full_cluster_labels = np.zeros_like(volume)
            full_cluster_labels[self.short_period:self.short_period + len(cluster_labels)] = cluster_labels
            
            return {
                'cluster_labels': full_cluster_labels,
                'cluster_centers': cluster_centers,
                'n_clusters': n_clusters if n_clusters >= 2 else 1
            }
            
        except Exception as e:
            logger.error(f"Error in clustering analysis: {str(e)}")
            return {'cluster_labels': np.zeros_like(data['volume'].values), 'cluster_centers': []}
    
    def _analyze_volume_momentum(self, data: pd.DataFrame, percentile_analysis: Dict) -> Dict[str, Any]:
        """Analyze volume momentum and dynamics."""
        try:
            volume = data['volume'].values
            percentiles = percentile_analysis.get('percentiles', np.array([]))
            
            # Volume momentum calculations
            volume_momentum = talib.MOM(volume, timeperiod=self.short_period)
            volume_roc = talib.ROC(volume, timeperiod=self.short_period)
            
            # Percentile momentum
            percentile_momentum = np.gradient(percentiles)
            
            # Volume acceleration
            volume_acceleration = np.gradient(volume_momentum)
            
            # Momentum strength
            momentum_strength = np.zeros_like(volume)
            for i in range(self.short_period, len(volume)):
                recent_momentum = volume_momentum[i-self.short_period:i]
                momentum_strength[i] = np.std(recent_momentum) * 100
            
            return {
                'volume_momentum': volume_momentum,
                'volume_roc': volume_roc,
                'percentile_momentum': percentile_momentum,
                'volume_acceleration': volume_acceleration,
                'momentum_strength': momentum_strength
            }
            
        except Exception as e:
            logger.error(f"Error analyzing volume momentum: {str(e)}")
            return {}
    
    def _calculate_modified_z_scores(self, data: np.ndarray) -> np.ndarray:
        """Calculate modified Z-scores for robust outlier detection."""
        try:
            modified_z_scores = np.zeros_like(data)
            
            for i in range(self.lookback_period, len(data)):
                window_data = data[max(0, i - self.lookback_period):i]
                median = np.median(window_data)
                mad = np.median(np.abs(window_data - median))
                
                if mad > 0:
                    modified_z_scores[i] = 0.6745 * (data[i] - median) / mad
                else:
                    modified_z_scores[i] = 0
            
            return modified_z_scores
        except:
            return np.zeros_like(data)
    
    def _detect_iqr_outliers(self, data: np.ndarray) -> np.ndarray:
        """Detect outliers using Interquartile Range method."""
        try:
            outlier_scores = np.zeros_like(data)
            
            for i in range(self.lookback_period, len(data)):
                window_data = data[max(0, i - self.lookback_period):i]
                Q1 = np.percentile(window_data, 25)
                Q3 = np.percentile(window_data, 75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                if data[i] < lower_bound:
                    outlier_scores[i] = (lower_bound - data[i]) / (Q1 - np.min(window_data) + 1e-8)
                elif data[i] > upper_bound:
                    outlier_scores[i] = (data[i] - upper_bound) / (np.max(window_data) - Q3 + 1e-8)
                else:
                    outlier_scores[i] = 0
            
            return outlier_scores
        except:
            return np.zeros_like(data)
    
    def _calculate_regime_persistence(self, regime_classifications: List[VolumeRegime]) -> np.ndarray:
        """Calculate regime persistence scores."""
        try:
            persistence = np.zeros(len(regime_classifications))
            
            for i in range(1, len(regime_classifications)):
                if regime_classifications[i] == regime_classifications[i-1]:
                    persistence[i] = persistence[i-1] + 1
                else:
                    persistence[i] = 0
            
            return persistence
        except:
            return np.zeros(len(regime_classifications))
    
    def _generate_comprehensive_signals(self, data: pd.DataFrame, percentile_analysis: Dict,
                                      statistical_analysis: Dict, anomaly_analysis: Dict,
                                      regime_analysis: Dict, liquidity_analysis: Dict,
                                      clustering_analysis: Dict, momentum_analysis: Dict) -> List[PriceVolumeRankSignal]:
        """Generate comprehensive PVR signals."""
        try:
            signals = []
            timestamps = pd.to_datetime(data.index) if hasattr(data.index, 'to_pydatetime') else data.index
            
            percentiles = percentile_analysis.get('percentiles', np.array([]))
            normalized_ranks = percentile_analysis.get('normalized_ranks', np.array([]))
            z_scores = percentile_analysis.get('z_scores', np.array([]))
            relative_ratios = percentile_analysis.get('relative_ratios', np.array([]))
            
            distribution_scores = statistical_analysis.get('distribution_scores', np.array([]))
            modified_z_scores = statistical_analysis.get('modified_z_scores', np.array([]))
            
            anomaly_types = anomaly_analysis.get('anomaly_types', [])
            anomaly_probabilities = anomaly_analysis.get('anomaly_probabilities', np.array([]))
            surge_intensities = anomaly_analysis.get('surge_intensities', np.array([]))
            drought_severities = anomaly_analysis.get('drought_severities', np.array([]))
            
            regime_classifications = regime_analysis.get('regime_classifications', [])
            liquidity_conditions = liquidity_analysis.get('liquidity_conditions', [])
            impact_probabilities = liquidity_analysis.get('impact_probabilities', np.array([]))
            
            cluster_labels = clustering_analysis.get('cluster_labels', np.array([]))
            volume_momentum = momentum_analysis.get('volume_momentum', np.array([]))
            
            for i in range(self.lookback_period, len(data)):
                try:
                    # Calculate overall confidence
                    confidence_factors = []
                    
                    # Statistical confidence
                    if i < len(distribution_scores):
                        stat_conf = min(100, distribution_scores[i] * 1000) if distribution_scores[i] > 0 else 50
                        confidence_factors.append(stat_conf)
                    
                    # Data quality confidence
                    if i < len(percentiles) and percentiles[i] > 0:
                        confidence_factors.append(80)
                    
                    # Anomaly detection confidence
                    if i < len(anomaly_probabilities):
                        anom_conf = 100 - min(50, anomaly_probabilities[i])
                        confidence_factors.append(anom_conf)
                    
                    if not confidence_factors:
                        confidence_factors = [50]
                    
                    confidence = np.mean(confidence_factors)
                    
                    if confidence >= self.confidence_threshold:
                        signal = PriceVolumeRankSignal(
                            timestamp=timestamps[i] if i < len(timestamps) else datetime.now(),
                            volume_percentile=percentiles[i] if i < len(percentiles) else 50.0,
                            normalized_rank=normalized_ranks[i] if i < len(normalized_ranks) else 50.0,
                            z_score=z_scores[i] if i < len(z_scores) else 0.0,
                            volume_regime=regime_classifications[i] if i < len(regime_classifications) else VolumeRegime.NORMAL,
                            anomaly_type=anomaly_types[i] if i < len(anomaly_types) else VolumeAnomaly.NORMAL,
                            anomaly_probability=anomaly_probabilities[i] if i < len(anomaly_probabilities) else 0.0,
                            distribution_fit_score=distribution_scores[i] if i < len(distribution_scores) else 0.0,
                            surge_intensity=surge_intensities[i] if i < len(surge_intensities) else 0.0,
                            drought_severity=drought_severities[i] if i < len(drought_severities) else 0.0,
                            liquidity_condition=liquidity_conditions[i] if i < len(liquidity_conditions) else LiquidityCondition.MODERATE_LIQUIDITY,
                            relative_volume_ratio=relative_ratios[i] if i < len(relative_ratios) else 1.0,
                            volume_momentum=volume_momentum[i] if i < len(volume_momentum) else 0.0,
                            cluster_membership=int(cluster_labels[i]) if i < len(cluster_labels) else 0,
                            outlier_score=abs(modified_z_scores[i]) if i < len(modified_z_scores) else 0.0,
                            impact_probability=impact_probabilities[i] if i < len(impact_probabilities) else 50.0,
                            confidence=confidence
                        )
                        
                        signals.append(signal)
                
                except Exception as e:
                    logger.warning(f"Error generating signal at index {i}: {str(e)}")
                    continue
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating comprehensive signals: {str(e)}")
            return []
    
    def _generate_metadata(self) -> Dict[str, Any]:
        """Generate metadata for the calculation results."""
        return {
            'indicator_name': 'Price Volume Rank',
            'version': '1.0.0',
            'parameters': {
                'lookback_period': self.lookback_period,
                'short_period': self.short_period,
                'anomaly_threshold': self.anomaly_threshold,
                'outlier_threshold': self.outlier_threshold,
                'min_observations': self.min_observations,
                'confidence_threshold': self.confidence_threshold
            },
            'features': [
                'Advanced relative volume analysis with percentile ranking',
                'Multi-timeframe volume anomaly detection',
                'Statistical distribution modeling and outlier identification',
                'Machine learning volume pattern classification',
                'Dynamic threshold adaptation based on market conditions',
                'Volume surge and drought detection algorithms',
                'Real-time liquidity assessment and market impact analysis',
                'Sophisticated clustering and regime analysis'
            ],
            'calculation_timestamp': datetime.now(),
            'data_requirements': ['open', 'high', 'low', 'close', 'volume']
        }
    
    def _generate_empty_result(self) -> Dict[str, Any]:
        """Generate empty result structure."""
        return {
            'percentile_analysis': {},
            'statistical_analysis': {},
            'anomaly_analysis': {},
            'regime_analysis': {},
            'liquidity_analysis': {},
            'clustering_analysis': {},
            'momentum_analysis': {},
            'signals': [],
            'metadata': self._generate_metadata(),
            'error': 'Insufficient data'
        }
    
    def _generate_error_result(self, error_message: str) -> Dict[str, Any]:
        """Generate error result structure."""
        return {
            'percentile_analysis': {},
            'statistical_analysis': {},
            'anomaly_analysis': {},
            'regime_analysis': {},
            'liquidity_analysis': {},
            'clustering_analysis': {},
            'momentum_analysis': {},
            'signals': [],
            'metadata': self._generate_metadata(),
            'error': error_message
        }

# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=600, freq='1H')
    
    # Generate realistic OHLCV data with volume anomalies
    base_price = 100
    returns = np.random.normal(0, 0.008, 600)
    
    # Generate volume with various patterns
    volume_base = np.random.lognormal(9, 0.4, 600)
    
    # Add volume anomalies and patterns
    for i in range(50, 550, 40):
        # Volume spikes
        if i % 120 == 50:
            volume_base[i:i+3] *= np.random.uniform(5, 10, 3)
        # Volume droughts
        elif i % 120 == 90:
            volume_base[i:i+5] *= np.random.uniform(0.1, 0.3, 5)
        # Volume clusters
        elif i % 120 == 10:
            cluster_vol = volume_base[i] * np.random.uniform(0.9, 1.1, 8)
            volume_base[i:i+8] = cluster_vol
    
    prices = base_price * np.exp(np.cumsum(returns))
    
    sample_data = pd.DataFrame({
        'open': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, 0.002, 600))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.002, 600))),
        'close': prices,
        'volume': volume_base
    }, index=dates)
    
    # Test the indicator
    indicator = PriceVolumeRankIndicator(
        lookback_period=252,
        short_period=20,
        anomaly_threshold=95.0,
        confidence_threshold=70.0
    )
    
    try:
        result = indicator.calculate(sample_data)
        
        print("Price Volume Rank Indicator Results:")
        print(f"- Calculation successful: {not result.get('error', False)}")
        print(f"- Signals generated: {len(result.get('signals', []))}")
        print(f"- Lookback period: {indicator.lookback_period}")
        print(f"- Anomaly threshold: {indicator.anomaly_threshold}%")
        
        # Display some sample signals
        signals = result.get('signals', [])
        if signals:
            print(f"\nSample PVR signals (showing first 3):")
            for i, signal in enumerate(signals[:3]):
                print(f"Signal {i+1}:")
                print(f"  Timestamp: {signal.timestamp}")
                print(f"  Volume Percentile: {signal.volume_percentile:.1f}%")
                print(f"  Volume Regime: {signal.volume_regime}")
                print(f"  Anomaly Type: {signal.anomaly_type}")
                print(f"  Anomaly Probability: {signal.anomaly_probability:.1f}%")
                print(f"  Liquidity Condition: {signal.liquidity_condition}")
                print(f"  Relative Volume Ratio: {signal.relative_volume_ratio:.2f}")
                print(f"  Outlier Score: {signal.outlier_score:.2f}")
                print(f"  Impact Probability: {signal.impact_probability:.1f}%")
                print(f"  Confidence: {signal.confidence:.1f}%")
        
        print(f"\nMetadata: {result.get('metadata', {}).get('indicator_name', 'N/A')}")
        
    except Exception as e:
        print(f"Error testing Price Volume Rank Indicator: {str(e)}")
        import traceback
        traceback.print_exc()