"""
Advanced Timeframe Configuration System for AUJ Platform

This module implements a sophisticated timeframe management system with:
- Multi-timeframe analysis and synchronization
- Adaptive timeframe selection based on market conditions
- Advanced correlation analysis across timeframes
- Dynamic timeframe optimization algorithms
- Machine learning-enhanced timeframe recommendations
- Real-time timeframe quality assessment
- Complex market regime-based timeframe adaptation

Author: AUJ Platform AI Enhanced Indicators
Version: 7.0 - Production Grade Implementation
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import warnings
from collections import defaultdict, deque
from scipy import stats, signal
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class TimeframeType(Enum):
    """Enumeration of supported timeframe types."""
    TICK = "tick"
    SECOND = "second"
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"

class MarketRegime(Enum):
    """Market regime classifications for timeframe adaptation."""
    TRENDING = "trending"
    RANGING = "ranging"
    VOLATILE = "volatile"
    QUIET = "quiet"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"

class SynchronizationMethod(Enum):
    """Methods for timeframe synchronization."""
    LINEAR = "linear"
    LOGARITHMIC = "logarithmic"
    FIBONACCI = "fibonacci"
    ADAPTIVE = "adaptive"
    HARMONIC = "harmonic"

@dataclass
class TimeframeMetrics:
    """Comprehensive metrics for a specific timeframe."""
    timeframe: str
    reliability_score: float = 0.0
    noise_level: float = 0.0
    trend_strength: float = 0.0
    volatility: float = 0.0
    data_quality: float = 0.0
    correlation_strength: float = 0.0
    prediction_accuracy: float = 0.0
    information_content: float = 0.0
    liquidity_measure: float = 0.0
    efficiency_ratio: float = 0.0
    fractal_dimension: float = 0.0
    hurst_exponent: float = 0.0
    entropy: float = 0.0
    signal_to_noise_ratio: float = 0.0
    adaptive_weight: float = 0.0
    regime_stability: float = 0.0
    execution_suitability: float = 0.0
    risk_level: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class TimeframeSyncConfig:
    """Configuration for timeframe synchronization."""
    primary_timeframe: str
    secondary_timeframes: List[str]
    sync_method: SynchronizationMethod
    correlation_threshold: float = 0.7
    quality_threshold: float = 0.6
    max_lag: int = 5
    smoothing_factor: float = 0.2
    adaptive_alpha: float = 0.1
    regime_sensitivity: float = 0.15

@dataclass
class MultiTimeframeSignal:
    """Signal aggregated across multiple timeframes."""
    primary_signal: float
    secondary_signals: Dict[str, float]
    confidence_score: float
    timeframe_weights: Dict[str, float]
    regime_context: MarketRegime
    execution_priority: int
    risk_assessment: float
    signal_quality: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

class TimeframeConfigIndicator:
    """
    Advanced Timeframe Configuration System
    
    Provides sophisticated multi-timeframe analysis with:
    - Adaptive timeframe selection
    - Cross-timeframe correlation analysis  
    - Dynamic synchronization algorithms
    - Machine learning optimization
    - Real-time quality assessment
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the timeframe configuration system."""
        self.logger = logging.getLogger(__name__)
        
        # Default configuration
        self.config = {
            'supported_timeframes': ['1m', '5m', '15m', '1h', '4h', '1d', '1w'],
            'primary_timeframe': '1h',
            'analysis_window': 100,
            'correlation_window': 50,
            'min_data_points': 20,
            'quality_threshold': 0.6,
            'correlation_threshold': 0.7,
            'volatility_threshold': 0.02,
            'trend_threshold': 0.3,
            'max_timeframes': 5,
            'optimization_interval': 24,  # hours
            'ml_training_window': 500,
            'regime_detection_window': 200,
            'synchronization_tolerance': 0.1,
            'adaptive_learning_rate': 0.01,
            'quality_decay_factor': 0.95,
            'correlation_decay_factor': 0.9,
            'regime_transition_sensitivity': 0.2
        }
        
        if config:
            self.config.update(config)
        
        # Initialize data structures
        self.timeframe_data: Dict[str, deque] = {}
        self.timeframe_metrics: Dict[str, TimeframeMetrics] = {}
        self.correlation_matrix: pd.DataFrame = pd.DataFrame()
        self.active_timeframes: Set[str] = set(self.config['supported_timeframes'])
        self.current_regime: MarketRegime = MarketRegime.RANGING
        self.regime_history: deque = deque(maxlen=50)
        self.signal_history: deque = deque(maxlen=100)
        self.quality_history: deque = deque(maxlen=200)
        
        # Synchronization configuration
        self.sync_configs: Dict[str, TimeframeSyncConfig] = {}
        self.sync_weights: Dict[str, float] = {}
        
        # Machine learning components
        self.scaler = StandardScaler()
        self.regime_classifier = None
        self.quality_predictor = None
        self.optimization_model = None
        
        # Performance tracking
        self.prediction_accuracy: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        self.execution_performance: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Threading and async support
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.update_lock = threading.Lock()
        
        # Initialize system
        self._initialize_timeframes()
        self._setup_synchronization()
        
        self.logger.info("TimeframeConfigIndicator initialized successfully")
    
    def _initialize_timeframes(self):
        """Initialize timeframe data structures and metrics."""
        try:
            for tf in self.config['supported_timeframes']:
                # Initialize data deque with max length
                max_len = max(self.config['analysis_window'], self.config['ml_training_window'])
                self.timeframe_data[tf] = deque(maxlen=max_len)
                
                # Initialize metrics
                self.timeframe_metrics[tf] = TimeframeMetrics(
                    timeframe=tf,
                    reliability_score=0.5,
                    data_quality=0.5,
                    adaptive_weight=1.0 / len(self.config['supported_timeframes'])
                )
                
                # Initialize prediction accuracy tracking
                self.prediction_accuracy[tf] = deque(maxlen=50)
                self.execution_performance[tf] = deque(maxlen=100)
            
            self.logger.info(f"Initialized {len(self.config['supported_timeframes'])} timeframes")
            
        except Exception as e:
            self.logger.error(f"Error initializing timeframes: {e}")
    
    def _setup_synchronization(self):
        """Setup timeframe synchronization configurations."""
        try:
            primary_tf = self.config['primary_timeframe']
            secondary_tfs = [tf for tf in self.config['supported_timeframes'] if tf != primary_tf]
            
            # Create synchronization config
            self.sync_configs[primary_tf] = TimeframeSyncConfig(
                primary_timeframe=primary_tf,
                secondary_timeframes=secondary_tfs,
                sync_method=SynchronizationMethod.ADAPTIVE,
                correlation_threshold=self.config['correlation_threshold'],
                quality_threshold=self.config['quality_threshold']
            )
            
            # Initialize synchronization weights
            total_timeframes = len(self.config['supported_timeframes'])
            for tf in self.config['supported_timeframes']:
                if tf == primary_tf:
                    self.sync_weights[tf] = 0.4  # Primary gets higher weight
                else:
                    self.sync_weights[tf] = 0.6 / (total_timeframes - 1)
            
            self.logger.info("Timeframe synchronization setup completed")
            
        except Exception as e:
            self.logger.error(f"Error setting up synchronization: {e}")
    
    def calculate(self, data_point: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main calculation method for timeframe configuration analysis.
        
        Args:
            data_point: Dictionary containing market data
            
        Returns:
            Dictionary with timeframe analysis results
        """
        try:
            # Update timeframe data
            timeframe = data_point.get('timeframe', self.config['primary_timeframe'])
            self._update_timeframe_data(timeframe, data_point)
            
            # Perform comprehensive analysis
            analysis_results = self._perform_comprehensive_analysis()
            
            # Generate multi-timeframe signals
            signals = self._generate_multi_timeframe_signals()
            
            # Optimize timeframe selection
            optimal_config = self._optimize_timeframe_selection()
            
            # Update regime detection
            current_regime = self._detect_market_regime()
            
            # Adapt timeframes based on regime
            adapted_config = self._adapt_timeframes_to_regime(current_regime)
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics()
            
            # Generate recommendations
            recommendations = self._generate_timeframe_recommendations()
            
            return {
                'timeframe_analysis': analysis_results,
                'multi_timeframe_signals': signals,
                'optimal_configuration': optimal_config,
                'current_regime': current_regime.value,
                'adapted_configuration': adapted_config,
                'quality_metrics': quality_metrics,
                'recommendations': recommendations,
                'active_timeframes': list(self.active_timeframes),
                'correlation_matrix': self._format_correlation_matrix(),
                'performance_summary': self._get_performance_summary(),
                'system_health': self._assess_system_health(),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error in timeframe calculation: {e}")
            return self._create_error_response(str(e))
    
    def _update_timeframe_data(self, timeframe: str, data_point: Dict[str, Any]):
        """Update data for a specific timeframe."""
        try:
            with self.update_lock:
                if timeframe not in self.timeframe_data:
                    max_len = max(self.config['analysis_window'], self.config['ml_training_window'])
                    self.timeframe_data[timeframe] = deque(maxlen=max_len)
                    self.timeframe_metrics[timeframe] = TimeframeMetrics(timeframe=timeframe)
                
                # Process and clean data point
                processed_data = self._process_data_point(data_point)
                
                # Add to timeframe data
                self.timeframe_data[timeframe].append(processed_data)
                
                # Update metrics
                self._update_timeframe_metrics(timeframe, processed_data)
            
        except Exception as e:
            self.logger.error(f"Error updating timeframe data: {e}")
    
    def _process_data_point(self, data_point: Dict[str, Any]) -> Dict[str, Any]:
        """Process and validate a data point."""
        try:
            # Extract essential fields
            processed = {
                'timestamp': data_point.get('timestamp', datetime.now()),
                'open': float(data_point.get('open', data_point.get('price', 0))),
                'high': float(data_point.get('high', data_point.get('price', 0))),
                'low': float(data_point.get('low', data_point.get('price', 0))),
                'close': float(data_point.get('close', data_point.get('price', 0))),
                'volume': float(data_point.get('volume', 0)),
                'price': float(data_point.get('price', data_point.get('close', 0)))
            }
            
            # Calculate derived metrics
            if processed['high'] > 0 and processed['low'] > 0:
                processed['range'] = processed['high'] - processed['low']
                processed['mid_price'] = (processed['high'] + processed['low']) / 2
                processed['price_position'] = (processed['close'] - processed['low']) / (processed['high'] - processed['low']) if processed['range'] > 0 else 0.5
            else:
                processed['range'] = 0
                processed['mid_price'] = processed['price']
                processed['price_position'] = 0.5
            
            # Calculate returns if possible
            processed['return'] = 0.0
            processed['log_return'] = 0.0
            
            return processed
            
        except Exception as e:
            self.logger.error(f"Error processing data point: {e}")
            return {}
    
    def _update_timeframe_metrics(self, timeframe: str, data_point: Dict[str, Any]):
        """Update metrics for a specific timeframe."""
        try:
            if timeframe not in self.timeframe_metrics:
                return
            
            metrics = self.timeframe_metrics[timeframe]
            tf_data = list(self.timeframe_data[timeframe])
            
            if len(tf_data) < self.config['min_data_points']:
                return
            
            # Extract price series
            prices = [dp.get('close', dp.get('price', 0)) for dp in tf_data[-50:]]
            volumes = [dp.get('volume', 0) for dp in tf_data[-50:]]
            
            if not prices or all(p == 0 for p in prices):
                return
            
            # Calculate returns
            returns = np.diff(np.log(prices)) if len(prices) > 1 else [0]
            
            # Update basic metrics
            metrics.volatility = np.std(returns) if len(returns) > 1 else 0.0
            metrics.noise_level = self._calculate_noise_level(prices)
            metrics.trend_strength = self._calculate_trend_strength(prices)
            metrics.data_quality = self._assess_data_quality(tf_data)
            metrics.liquidity_measure = np.mean(volumes) if volumes else 0.0
            metrics.efficiency_ratio = self._calculate_efficiency_ratio(prices)
            
            # Calculate advanced metrics
            metrics.fractal_dimension = self._calculate_fractal_dimension(prices)
            metrics.hurst_exponent = self._calculate_hurst_exponent(returns)
            metrics.entropy = self._calculate_information_entropy(returns)
            metrics.signal_to_noise_ratio = self._calculate_snr(prices)
            
            # Update adaptive metrics
            self._update_adaptive_metrics(timeframe, metrics)
            
            # Update timestamp
            metrics.last_updated = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Error updating timeframe metrics: {e}")
    
    def _calculate_noise_level(self, prices: List[float]) -> float:
        """Calculate noise level in price series."""
        try:
            if len(prices) < 3:
                return 0.0
            
            # Calculate price differences
            diffs = np.diff(prices)
            
            # Calculate second differences (acceleration)
            second_diffs = np.diff(diffs)
            
            # Noise is the standard deviation of second differences
            noise_level = np.std(second_diffs) / np.mean(np.abs(prices)) if len(second_diffs) > 0 else 0.0
            
            return min(1.0, noise_level * 100)  # Normalize to 0-1 range
            
        except Exception:
            return 0.0
    
    def _calculate_trend_strength(self, prices: List[float]) -> float:
        """Calculate trend strength using linear regression."""
        try:
            if len(prices) < 3:
                return 0.0
            
            x = np.arange(len(prices))
            
            # Linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, prices)
            
            # Trend strength based on R-squared and slope significance
            r_squared = r_value ** 2
            slope_normalized = abs(slope) / np.mean(prices) if np.mean(prices) > 0 else 0
            
            trend_strength = r_squared * min(1.0, slope_normalized * 100)
            
            return trend_strength
            
        except Exception:
            return 0.0
    
    def _assess_data_quality(self, data_points: List[Dict[str, Any]]) -> float:
        """Assess the quality of data for a timeframe."""
        try:
            if not data_points:
                return 0.0
            
            quality_factors = []
            
            # Check data completeness
            complete_points = sum(1 for dp in data_points if all(k in dp for k in ['open', 'high', 'low', 'close']))
            completeness = complete_points / len(data_points)
            quality_factors.append(completeness)
            
            # Check data consistency
            prices = [dp.get('close', 0) for dp in data_points]
            price_consistency = 1.0 - (np.std(prices) / np.mean(prices)) if np.mean(prices) > 0 else 0.0
            quality_factors.append(max(0.0, min(1.0, price_consistency)))
            
            # Check temporal consistency
            timestamps = [dp.get('timestamp') for dp in data_points if dp.get('timestamp')]
            if len(timestamps) > 1:
                time_gaps = [(timestamps[i+1] - timestamps[i]).total_seconds() for i in range(len(timestamps)-1) if isinstance(timestamps[i], datetime) and isinstance(timestamps[i+1], datetime)]
                if time_gaps:
                    gap_consistency = 1.0 - (np.std(time_gaps) / np.mean(time_gaps)) if np.mean(time_gaps) > 0 else 0.0
                    quality_factors.append(max(0.0, min(1.0, gap_consistency)))
            
            # Check for outliers
            if prices:
                z_scores = np.abs(stats.zscore(prices))
                outlier_ratio = np.sum(z_scores > 3) / len(prices)
                outlier_quality = 1.0 - outlier_ratio
                quality_factors.append(outlier_quality)
            
            return np.mean(quality_factors) if quality_factors else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_efficiency_ratio(self, prices: List[float]) -> float:
        """Calculate Kaufman's Efficiency Ratio."""
        try:
            if len(prices) < 2:
                return 0.0
            
            # Direction (net change)
            direction = abs(prices[-1] - prices[0])
            
            # Volatility (sum of absolute changes)
            volatility = sum(abs(prices[i] - prices[i-1]) for i in range(1, len(prices)))
            
            # Efficiency ratio
            efficiency = direction / volatility if volatility > 0 else 0.0
            
            return min(1.0, efficiency)
            
        except Exception:
            return 0.0
    
    def _calculate_fractal_dimension(self, prices: List[float]) -> float:
        """Calculate fractal dimension using Higuchi's method."""
        try:
            if len(prices) < 10:
                return 1.5  # Default value
            
            n = len(prices)
            k_max = min(10, n // 4)
            
            lk_values = []
            k_values = []
            
            for k in range(1, k_max + 1):
                lk = 0
                for m in range(1, k + 1):
                    ll = 0
                    indices = list(range(m - 1, n, k))
                    if len(indices) > 1:
                        for i in range(1, len(indices)):
                            ll += abs(prices[indices[i]] - prices[indices[i-1]])
                        ll = ll * (n - 1) / (k * len(indices))
                        lk += ll
                
                if lk > 0:
                    lk_values.append(np.log(lk))
                    k_values.append(np.log(1.0 / k))
            
            if len(lk_values) > 1:
                slope, _, _, _, _ = stats.linregress(k_values, lk_values)
                fractal_dimension = slope
                return max(1.0, min(3.0, fractal_dimension))
            else:
                return 1.5
                
        except Exception:
            return 1.5
    
    def _calculate_hurst_exponent(self, returns: List[float]) -> float:
        """Calculate Hurst exponent using R/S analysis."""
        try:
            if len(returns) < 10:
                return 0.5  # Default value for random walk
            
            returns = np.array(returns)
            n = len(returns)
            
            # Calculate cumulative deviations
            mean_return = np.mean(returns)
            cumulative_deviations = np.cumsum(returns - mean_return)
            
            # Calculate range
            R = np.max(cumulative_deviations) - np.min(cumulative_deviations)
            
            # Calculate standard deviation
            S = np.std(returns)
            
            # R/S ratio
            rs_ratio = R / S if S > 0 else 1.0
            
            # Hurst exponent approximation
            hurst = np.log(rs_ratio) / np.log(n) if n > 1 else 0.5
            
            return max(0.0, min(1.0, hurst))
            
        except Exception:
            return 0.5
    
    def _calculate_information_entropy(self, returns: List[float]) -> float:
        """Calculate information entropy of returns."""
        try:
            if len(returns) < 5:
                return 0.0
            
            # Discretize returns into bins
            n_bins = min(10, len(returns) // 2)
            hist, _ = np.histogram(returns, bins=n_bins)
            
            # Calculate probabilities
            probabilities = hist / np.sum(hist)
            probabilities = probabilities[probabilities > 0]  # Remove zero probabilities
            
            # Calculate entropy
            entropy = -np.sum(probabilities * np.log2(probabilities))
            
            # Normalize to 0-1 range
            max_entropy = np.log2(n_bins)
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
            
            return normalized_entropy
            
        except Exception:
            return 0.0
    
    def _calculate_snr(self, prices: List[float]) -> float:
        """Calculate signal-to-noise ratio."""
        try:
            if len(prices) < 5:
                return 1.0
            
            # Calculate trend (signal)
            x = np.arange(len(prices))
            slope, intercept, _, _, _ = stats.linregress(x, prices)
            trend = slope * x + intercept
            
            # Calculate residuals (noise)
            residuals = np.array(prices) - trend
            
            # SNR calculation
            signal_power = np.var(trend)
            noise_power = np.var(residuals)
            
            snr = signal_power / noise_power if noise_power > 0 else float('inf')
            
            # Convert to dB and normalize
            snr_db = 10 * np.log10(snr) if snr > 0 else 0
            
            # Normalize to 0-1 range (assuming typical SNR range of 0-40 dB)
            normalized_snr = min(1.0, max(0.0, snr_db / 40.0))
            
            return normalized_snr
            
        except Exception:
            return 1.0
    
    def _update_adaptive_metrics(self, timeframe: str, metrics: TimeframeMetrics):
        """Update adaptive metrics based on performance."""
        try:
            # Get recent performance data
            recent_accuracy = list(self.prediction_accuracy[timeframe])
            recent_execution = list(self.execution_performance[timeframe])
            
            # Calculate performance-based adjustments
            if recent_accuracy:
                accuracy_factor = np.mean(recent_accuracy)
                metrics.prediction_accuracy = accuracy_factor
            
            if recent_execution:
                execution_factor = np.mean(recent_execution)
                metrics.execution_suitability = execution_factor
            
            # Update reliability score
            reliability_factors = [
                metrics.data_quality,
                metrics.prediction_accuracy,
                metrics.execution_suitability,
                1.0 - metrics.noise_level,
                metrics.signal_to_noise_ratio
            ]
            
            metrics.reliability_score = np.mean([f for f in reliability_factors if f is not None])
            
            # Update adaptive weight
            total_reliability = sum(m.reliability_score for m in self.timeframe_metrics.values())
            if total_reliability > 0:
                metrics.adaptive_weight = metrics.reliability_score / total_reliability
            
            # Calculate correlation strength with other timeframes
            correlation_scores = []
            for other_tf, other_metrics in self.timeframe_metrics.items():
                if other_tf != timeframe:
                    corr = self._calculate_timeframe_correlation(timeframe, other_tf)
                    if corr is not None:
                        correlation_scores.append(abs(corr))
            
            metrics.correlation_strength = np.mean(correlation_scores) if correlation_scores else 0.0
            
            # Update information content
            metrics.information_content = (
                metrics.trend_strength * 0.3 +
                metrics.signal_to_noise_ratio * 0.3 +
                (1.0 - metrics.noise_level) * 0.2 +
                metrics.efficiency_ratio * 0.2
            )
            
            # Calculate regime stability
            metrics.regime_stability = self._calculate_regime_stability(timeframe)
            
            # Update risk level
            risk_factors = [
                metrics.volatility,
                metrics.noise_level,
                1.0 - metrics.data_quality,
                1.0 - metrics.regime_stability
            ]
            
            metrics.risk_level = np.mean(risk_factors)
            
        except Exception as e:
            self.logger.error(f"Error updating adaptive metrics: {e}")
    
    def _calculate_timeframe_correlation(self, tf1: str, tf2: str) -> Optional[float]:
        """Calculate correlation between two timeframes."""
        try:
            if tf1 not in self.timeframe_data or tf2 not in self.timeframe_data:
                return None
            
            data1 = list(self.timeframe_data[tf1])
            data2 = list(self.timeframe_data[tf2])
            
            if len(data1) < self.config['min_data_points'] or len(data2) < self.config['min_data_points']:
                return None
            
            # Extract price series
            prices1 = [dp.get('close', dp.get('price', 0)) for dp in data1[-self.config['correlation_window']:]]
            prices2 = [dp.get('close', dp.get('price', 0)) for dp in data2[-self.config['correlation_window']:]]
            
            # Align lengths
            min_len = min(len(prices1), len(prices2))
            if min_len < 5:
                return None
            
            prices1 = prices1[-min_len:]
            prices2 = prices2[-min_len:]
            
            # Calculate correlation
            correlation, _ = stats.pearsonr(prices1, prices2)
            
            return correlation if not np.isnan(correlation) else None
            
        except Exception:
            return None
    
    def _calculate_regime_stability(self, timeframe: str) -> float:
        """Calculate regime stability for a timeframe."""
        try:
            if len(self.regime_history) < 5:
                return 0.5
            
            # Count regime changes in recent history
            recent_regimes = list(self.regime_history)[-10:]
            regime_changes = sum(1 for i in range(1, len(recent_regimes)) 
                               if recent_regimes[i] != recent_regimes[i-1])
            
            # Stability is inverse of change frequency
            stability = 1.0 - (regime_changes / (len(recent_regimes) - 1)) if len(recent_regimes) > 1 else 0.5
            
            return max(0.0, min(1.0, stability))
            
        except Exception:
            return 0.5    
    def _perform_comprehensive_analysis(self) -> Dict[str, Any]:
        """Perform comprehensive multi-timeframe analysis."""
        try:
            analysis_results = {}
            
            # Update correlation matrix
            self._update_correlation_matrix()
            
            # Analyze timeframe relationships
            relationships = self._analyze_timeframe_relationships()
            analysis_results['timeframe_relationships'] = relationships
            
            # Perform cluster analysis
            clusters = self._perform_timeframe_clustering()
            analysis_results['timeframe_clusters'] = clusters
            
            # Analyze temporal patterns
            temporal_patterns = self._analyze_temporal_patterns()
            analysis_results['temporal_patterns'] = temporal_patterns
            
            # Calculate synchronization metrics
            sync_metrics = self._calculate_synchronization_metrics()
            analysis_results['synchronization_metrics'] = sync_metrics
            
            # Analyze regime transitions
            regime_analysis = self._analyze_regime_transitions()
            analysis_results['regime_analysis'] = regime_analysis
            
            # Calculate optimal weights
            optimal_weights = self._calculate_optimal_weights()
            analysis_results['optimal_weights'] = optimal_weights
            
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive analysis: {e}")
            return {}
    
    def _update_correlation_matrix(self):
        """Update the correlation matrix between timeframes."""
        try:
            timeframes = list(self.active_timeframes)
            n_timeframes = len(timeframes)
            
            if n_timeframes < 2:
                return
            
            # Initialize correlation matrix
            correlation_data = np.eye(n_timeframes)
            
            # Calculate pairwise correlations
            for i, tf1 in enumerate(timeframes):
                for j, tf2 in enumerate(timeframes):
                    if i != j:
                        corr = self._calculate_timeframe_correlation(tf1, tf2)
                        if corr is not None:
                            correlation_data[i, j] = corr
            
            # Create DataFrame
            self.correlation_matrix = pd.DataFrame(
                correlation_data,
                index=timeframes,
                columns=timeframes
            )
            
        except Exception as e:
            self.logger.error(f"Error updating correlation matrix: {e}")
    
    def _analyze_timeframe_relationships(self) -> Dict[str, Any]:
        """Analyze relationships between different timeframes."""
        try:
            relationships = {}
            
            if self.correlation_matrix.empty:
                return relationships
            
            # Find strongest correlations
            corr_matrix = self.correlation_matrix.values
            np.fill_diagonal(corr_matrix, 0)  # Remove self-correlations
            
            # Strongest positive correlation
            max_pos_idx = np.unravel_index(np.argmax(corr_matrix), corr_matrix.shape)
            max_pos_corr = corr_matrix[max_pos_idx]
            
            # Strongest negative correlation
            min_neg_idx = np.unravel_index(np.argmin(corr_matrix), corr_matrix.shape)
            min_neg_corr = corr_matrix[min_neg_idx]
            
            relationships['strongest_positive'] = {
                'timeframes': (self.correlation_matrix.index[max_pos_idx[0]], 
                              self.correlation_matrix.columns[max_pos_idx[1]]),
                'correlation': max_pos_corr
            }
            
            relationships['strongest_negative'] = {
                'timeframes': (self.correlation_matrix.index[min_neg_idx[0]], 
                              self.correlation_matrix.columns[min_neg_idx[1]]),
                'correlation': min_neg_corr
            }
            
            # Average correlation strength
            relationships['average_correlation'] = np.mean(np.abs(corr_matrix))
            
            # Correlation stability
            relationships['correlation_stability'] = 1.0 - np.std(corr_matrix)
            
            # Identify correlation clusters
            from scipy.cluster.hierarchy import linkage, fcluster
            from scipy.spatial.distance import squareform
            
            # Convert correlation to distance
            distance_matrix = 1 - np.abs(corr_matrix)
            condensed_distances = squareform(distance_matrix)
            
            # Hierarchical clustering
            linkage_matrix = linkage(condensed_distances, method='ward')
            cluster_labels = fcluster(linkage_matrix, t=0.5, criterion='distance')
            
            relationships['correlation_clusters'] = {
                'labels': cluster_labels.tolist(),
                'n_clusters': len(set(cluster_labels))
            }
            
            return relationships
            
        except Exception as e:
            self.logger.error(f"Error analyzing timeframe relationships: {e}")
            return {}
    
    def _perform_timeframe_clustering(self) -> Dict[str, Any]:
        """Perform clustering analysis on timeframe characteristics."""
        try:
            if not self.timeframe_metrics:
                return {}
            
            # Extract features for clustering
            features = []
            timeframe_names = []
            
            for tf, metrics in self.timeframe_metrics.items():
                if len(self.timeframe_data.get(tf, [])) >= self.config['min_data_points']:
                    feature_vector = [
                        metrics.volatility,
                        metrics.trend_strength,
                        metrics.noise_level,
                        metrics.data_quality,
                        metrics.signal_to_noise_ratio,
                        metrics.efficiency_ratio,
                        metrics.information_content,
                        metrics.reliability_score
                    ]
                    features.append(feature_vector)
                    timeframe_names.append(tf)
            
            if len(features) < 2:
                return {}
            
            # Standardize features
            features_scaled = self.scaler.fit_transform(features)
            
            # Determine optimal number of clusters
            optimal_k = self._determine_optimal_clusters(features_scaled)
            
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features_scaled)
            
            # Calculate cluster quality
            if optimal_k > 1:
                silhouette_avg = silhouette_score(features_scaled, cluster_labels)
            else:
                silhouette_avg = 0.0
            
            # Organize results
            clusters = {}
            for i, tf in enumerate(timeframe_names):
                cluster_id = int(cluster_labels[i])
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(tf)
            
            return {
                'clusters': clusters,
                'n_clusters': optimal_k,
                'silhouette_score': silhouette_avg,
                'cluster_centers': kmeans.cluster_centers_.tolist(),
                'timeframe_labels': dict(zip(timeframe_names, cluster_labels))
            }
            
        except Exception as e:
            self.logger.error(f"Error in timeframe clustering: {e}")
            return {}
    
    def _determine_optimal_clusters(self, features: np.ndarray) -> int:
        """Determine optimal number of clusters using elbow method."""
        try:
            max_k = min(len(features) - 1, 5)
            if max_k < 2:
                return 1
            
            inertias = []
            silhouette_scores = []
            
            for k in range(2, max_k + 1):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(features)
                
                inertias.append(kmeans.inertia_)
                silhouette_scores.append(silhouette_score(features, cluster_labels))
            
            # Find elbow point
            if len(inertias) >= 2:
                # Calculate rate of change
                rate_of_change = np.diff(inertias)
                
                # Find the point where rate of change is minimized
                optimal_k_idx = np.argmin(rate_of_change) + 2  # +2 because we start from k=2
                
                # Validate with silhouette score
                best_silhouette_idx = np.argmax(silhouette_scores) + 2
                
                # Choose based on combined criteria
                if abs(optimal_k_idx - best_silhouette_idx) <= 1:
                    return optimal_k_idx
                else:
                    return best_silhouette_idx
            else:
                return 2
                
        except Exception:
            return 2
    
    def _analyze_temporal_patterns(self) -> Dict[str, Any]:
        """Analyze temporal patterns across timeframes."""
        try:
            patterns = {}
            
            # Analyze cyclical patterns
            cyclical_analysis = self._detect_cyclical_patterns()
            patterns['cyclical_patterns'] = cyclical_analysis
            
            # Analyze lead-lag relationships
            lead_lag_analysis = self._analyze_lead_lag_relationships()
            patterns['lead_lag_relationships'] = lead_lag_analysis
            
            # Analyze seasonal patterns
            seasonal_analysis = self._detect_seasonal_patterns()
            patterns['seasonal_patterns'] = seasonal_analysis
            
            # Analyze momentum patterns
            momentum_analysis = self._analyze_momentum_patterns()
            patterns['momentum_patterns'] = momentum_analysis
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error analyzing temporal patterns: {e}")
            return {}
    
    def _detect_cyclical_patterns(self) -> Dict[str, Any]:
        """Detect cyclical patterns in timeframe data."""
        try:
            cyclical_patterns = {}
            
            for tf, data in self.timeframe_data.items():
                if len(data) < 20:
                    continue
                
                # Extract price series
                prices = [dp.get('close', dp.get('price', 0)) for dp in data]
                
                if not prices:
                    continue
                
                # Use FFT to detect dominant frequencies
                fft_result = np.fft.fft(prices)
                frequencies = np.fft.fftfreq(len(prices))
                
                # Find dominant frequencies (excluding DC component)
                magnitude = np.abs(fft_result[1:len(fft_result)//2])
                freq_positive = frequencies[1:len(frequencies)//2]
                
                if len(magnitude) > 0:
                    # Find peaks in frequency domain
                    peaks, _ = signal.find_peaks(magnitude, height=np.max(magnitude) * 0.1)
                    
                    dominant_cycles = []
                    for peak in peaks[:3]:  # Top 3 cycles
                        cycle_length = 1.0 / abs(freq_positive[peak]) if freq_positive[peak] != 0 else 0
                        cycle_strength = magnitude[peak] / np.max(magnitude)
                        
                        dominant_cycles.append({
                            'cycle_length': cycle_length,
                            'strength': cycle_strength,
                            'frequency': freq_positive[peak]
                        })
                    
                    cyclical_patterns[tf] = {
                        'dominant_cycles': dominant_cycles,
                        'spectral_entropy': self._calculate_spectral_entropy(magnitude)
                    }
            
            return cyclical_patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting cyclical patterns: {e}")
            return {}
    
    def _calculate_spectral_entropy(self, magnitude: np.ndarray) -> float:
        """Calculate spectral entropy of frequency domain data."""
        try:
            # Normalize magnitude to probabilities
            power = magnitude ** 2
            prob_density = power / np.sum(power)
            
            # Calculate entropy
            prob_density = prob_density[prob_density > 0]  # Remove zeros
            entropy = -np.sum(prob_density * np.log2(prob_density))
            
            # Normalize
            max_entropy = np.log2(len(magnitude))
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
            
            return normalized_entropy
            
        except Exception:
            return 0.0
    
    def _analyze_lead_lag_relationships(self) -> Dict[str, Any]:
        """Analyze lead-lag relationships between timeframes."""
        try:
            lead_lag_results = {}
            timeframes = list(self.active_timeframes)
            
            for i, tf1 in enumerate(timeframes):
                for j, tf2 in enumerate(timeframes):
                    if i != j:
                        lead_lag = self._calculate_lead_lag(tf1, tf2)
                        if lead_lag is not None:
                            lead_lag_results[f"{tf1}_vs_{tf2}"] = lead_lag
            
            return lead_lag_results
            
        except Exception as e:
            self.logger.error(f"Error analyzing lead-lag relationships: {e}")
            return {}
    
    def _calculate_lead_lag(self, tf1: str, tf2: str) -> Optional[Dict[str, Any]]:
        """Calculate lead-lag relationship between two timeframes."""
        try:
            if tf1 not in self.timeframe_data or tf2 not in self.timeframe_data:
                return None
            
            data1 = list(self.timeframe_data[tf1])
            data2 = list(self.timeframe_data[tf2])
            
            if len(data1) < 20 or len(data2) < 20:
                return None
            
            # Extract returns
            prices1 = [dp.get('close', dp.get('price', 0)) for dp in data1[-50:]]
            prices2 = [dp.get('close', dp.get('price', 0)) for dp in data2[-50:]]
            
            returns1 = np.diff(np.log(prices1)) if len(prices1) > 1 else []
            returns2 = np.diff(np.log(prices2)) if len(prices2) > 1 else []
            
            if len(returns1) < 10 or len(returns2) < 10:
                return None
            
            # Align lengths
            min_len = min(len(returns1), len(returns2))
            returns1 = returns1[-min_len:]
            returns2 = returns2[-min_len:]
            
            # Calculate cross-correlation
            max_lags = min(5, min_len // 4)
            correlations = []
            lags = []
            
            for lag in range(-max_lags, max_lags + 1):
                if lag == 0:
                    corr, _ = stats.pearsonr(returns1, returns2)
                elif lag > 0:
                    if len(returns1) > lag:
                        corr, _ = stats.pearsonr(returns1[:-lag], returns2[lag:])
                    else:
                        continue
                else:  # lag < 0
                    if len(returns2) > abs(lag):
                        corr, _ = stats.pearsonr(returns1[abs(lag):], returns2[:lag])
                    else:
                        continue
                
                if not np.isnan(corr):
                    correlations.append(corr)
                    lags.append(lag)
            
            if not correlations:
                return None
            
            # Find optimal lag
            max_corr_idx = np.argmax(np.abs(correlations))
            optimal_lag = lags[max_corr_idx]
            max_correlation = correlations[max_corr_idx]
            
            return {
                'optimal_lag': optimal_lag,
                'max_correlation': max_correlation,
                'all_correlations': list(zip(lags, correlations)),
                'lead_timeframe': tf1 if optimal_lag > 0 else tf2,
                'lag_strength': abs(max_correlation)
            }
            
        except Exception:
            return None
    
    def _detect_seasonal_patterns(self) -> Dict[str, Any]:
        """Detect seasonal patterns in timeframe data."""
        try:
            seasonal_patterns = {}
            
            for tf, data in self.timeframe_data.items():
                if len(data) < 50:
                    continue
                
                # Extract timestamps and prices
                timestamps = [dp.get('timestamp') for dp in data if dp.get('timestamp')]
                prices = [dp.get('close', dp.get('price', 0)) for dp in data]
                
                if len(timestamps) != len(prices) or len(timestamps) < 50:
                    continue
                
                # Analyze by hour of day
                hourly_returns = defaultdict(list)
                for i in range(1, len(timestamps)):
                    if isinstance(timestamps[i], datetime) and isinstance(timestamps[i-1], datetime):
                        hour = timestamps[i].hour
                        ret = np.log(prices[i] / prices[i-1]) if prices[i-1] > 0 else 0
                        hourly_returns[hour].append(ret)
                
                # Calculate hourly statistics
                hourly_stats = {}
                for hour, returns in hourly_returns.items():
                    if len(returns) > 2:
                        hourly_stats[hour] = {
                            'mean_return': np.mean(returns),
                            'volatility': np.std(returns),
                            'count': len(returns)
                        }
                
                seasonal_patterns[tf] = {
                    'hourly_patterns': hourly_stats,
                    'pattern_strength': self._calculate_seasonal_strength(hourly_stats)
                }
            
            return seasonal_patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting seasonal patterns: {e}")
            return {}
    
    def _calculate_seasonal_strength(self, hourly_stats: Dict[int, Dict[str, float]]) -> float:
        """Calculate strength of seasonal patterns."""
        try:
            if len(hourly_stats) < 3:
                return 0.0
            
            # Extract mean returns by hour
            mean_returns = [stats['mean_return'] for stats in hourly_stats.values()]
            
            # Calculate variance of hourly means
            pattern_variance = np.var(mean_returns)
            
            # Normalize (typical variance range)
            strength = min(1.0, pattern_variance * 10000)  # Scale factor for typical FX returns
            
            return strength
            
        except Exception:
            return 0.0
    
    def _analyze_momentum_patterns(self) -> Dict[str, Any]:
        """Analyze momentum patterns across timeframes."""
        try:
            momentum_patterns = {}
            
            for tf, data in self.timeframe_data.items():
                if len(data) < 20:
                    continue
                
                prices = [dp.get('close', dp.get('price', 0)) for dp in data[-30:]]
                
                if len(prices) < 10:
                    continue
                
                # Calculate different momentum indicators
                momentum_indicators = {}
                
                # Rate of change
                if len(prices) > 10:
                    roc = (prices[-1] - prices[-10]) / prices[-10] if prices[-10] > 0 else 0
                    momentum_indicators['rate_of_change_10'] = roc
                
                # Momentum acceleration
                if len(prices) > 5:
                    recent_slope = (prices[-1] - prices[-5]) / 5
                    older_slope = (prices[-5] - prices[-10]) / 5 if len(prices) > 10 else recent_slope
                    acceleration = recent_slope - older_slope
                    momentum_indicators['momentum_acceleration'] = acceleration
                
                # Trend persistence
                returns = np.diff(np.log(prices)) if len(prices) > 1 else []
                if len(returns) > 5:
                    # Count consecutive periods with same sign
                    signs = np.sign(returns)
                    persistence = 0
                    current_streak = 1
                    max_streak = 1
                    
                    for i in range(1, len(signs)):
                        if signs[i] == signs[i-1]:
                            current_streak += 1
                            max_streak = max(max_streak, current_streak)
                        else:
                            current_streak = 1
                    
                    persistence = max_streak / len(signs)
                    momentum_indicators['trend_persistence'] = persistence
                
                momentum_patterns[tf] = momentum_indicators
            
            return momentum_patterns
            
        except Exception as e:
            self.logger.error(f"Error analyzing momentum patterns: {e}")
            return {}
    
    def _calculate_synchronization_metrics(self) -> Dict[str, Any]:
        """Calculate synchronization metrics between timeframes."""
        try:
            sync_metrics = {}
            
            # Calculate phase synchronization
            phase_sync = self._calculate_phase_synchronization()
            sync_metrics['phase_synchronization'] = phase_sync
            
            # Calculate amplitude synchronization
            amplitude_sync = self._calculate_amplitude_synchronization()
            sync_metrics['amplitude_synchronization'] = amplitude_sync
            
            # Calculate frequency synchronization
            frequency_sync = self._calculate_frequency_synchronization()
            sync_metrics['frequency_synchronization'] = frequency_sync
            
            # Calculate overall synchronization index
            sync_indices = [phase_sync, amplitude_sync, frequency_sync]
            valid_indices = [idx for idx in sync_indices if idx is not None]
            
            if valid_indices:
                overall_sync = np.mean(valid_indices)
                sync_metrics['overall_synchronization'] = overall_sync
            else:
                sync_metrics['overall_synchronization'] = 0.0
            
            return sync_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating synchronization metrics: {e}")
            return {}
    
    def _calculate_phase_synchronization(self) -> Optional[float]:
        """Calculate phase synchronization between timeframes."""
        try:
            if len(self.active_timeframes) < 2:
                return None
            
            phase_differences = []
            timeframes = list(self.active_timeframes)
            
            for i in range(len(timeframes)):
                for j in range(i + 1, len(timeframes)):
                    tf1, tf2 = timeframes[i], timeframes[j]
                    
                    if tf1 in self.timeframe_data and tf2 in self.timeframe_data:
                        data1 = list(self.timeframe_data[tf1])
                        data2 = list(self.timeframe_data[tf2])
                        
                        if len(data1) >= 20 and len(data2) >= 20:
                            prices1 = [dp.get('close', dp.get('price', 0)) for dp in data1[-20:]]
                            prices2 = [dp.get('close', dp.get('price', 0)) for dp in data2[-20:]]
                            
                            # Calculate instantaneous phase using Hilbert transform
                            analytic1 = signal.hilbert(prices1)
                            analytic2 = signal.hilbert(prices2)
                            
                            phase1 = np.angle(analytic1)
                            phase2 = np.angle(analytic2)
                            
                            # Phase difference
                            phase_diff = phase1 - phase2
                            
                            # Calculate phase locking value
                            phase_locking = np.abs(np.mean(np.exp(1j * phase_diff)))
                            phase_differences.append(phase_locking)
            
            if phase_differences:
                return np.mean(phase_differences)
            else:
                return None
                
        except Exception:
            return None
    
    def _calculate_amplitude_synchronization(self) -> Optional[float]:
        """Calculate amplitude synchronization between timeframes."""
        try:
            if len(self.active_timeframes) < 2:
                return None
            
            amplitude_correlations = []
            timeframes = list(self.active_timeframes)
            
            for i in range(len(timeframes)):
                for j in range(i + 1, len(timeframes)):
                    tf1, tf2 = timeframes[i], timeframes[j]
                    
                    if tf1 in self.timeframe_data and tf2 in self.timeframe_data:
                        data1 = list(self.timeframe_data[tf1])
                        data2 = list(self.timeframe_data[tf2])
                        
                        if len(data1) >= 20 and len(data2) >= 20:
                            prices1 = [dp.get('close', dp.get('price', 0)) for dp in data1[-20:]]
                            prices2 = [dp.get('close', dp.get('price', 0)) for dp in data2[-20:]]
                            
                            # Calculate instantaneous amplitude using Hilbert transform
                            analytic1 = signal.hilbert(prices1)
                            analytic2 = signal.hilbert(prices2)
                            
                            amplitude1 = np.abs(analytic1)
                            amplitude2 = np.abs(analytic2)
                            
                            # Calculate correlation between amplitudes
                            if len(amplitude1) == len(amplitude2) and len(amplitude1) > 1:
                                corr, _ = stats.pearsonr(amplitude1, amplitude2)
                                if not np.isnan(corr):
                                    amplitude_correlations.append(abs(corr))
            
            if amplitude_correlations:
                return np.mean(amplitude_correlations)
            else:
                return None
                
        except Exception:
            return None
    
    def _calculate_frequency_synchronization(self) -> Optional[float]:
        """Calculate frequency synchronization between timeframes."""
        try:
            if len(self.active_timeframes) < 2:
                return None
            
            frequency_similarities = []
            timeframes = list(self.active_timeframes)
            
            for i in range(len(timeframes)):
                for j in range(i + 1, len(timeframes)):
                    tf1, tf2 = timeframes[i], timeframes[j]
                    
                    if tf1 in self.timeframe_data and tf2 in self.timeframe_data:
                        data1 = list(self.timeframe_data[tf1])
                        data2 = list(self.timeframe_data[tf2])
                        
                        if len(data1) >= 30 and len(data2) >= 30:
                            prices1 = [dp.get('close', dp.get('price', 0)) for dp in data1[-30:]]
                            prices2 = [dp.get('close', dp.get('price', 0)) for dp in data2[-30:]]
                            
                            # Calculate dominant frequencies using FFT
                            fft1 = np.fft.fft(prices1)
                            fft2 = np.fft.fft(prices2)
                            
                            # Get frequency magnitudes
                            mag1 = np.abs(fft1[1:len(fft1)//2])
                            mag2 = np.abs(fft2[1:len(fft2)//2])
                            
                            # Normalize
                            mag1 = mag1 / np.sum(mag1) if np.sum(mag1) > 0 else mag1
                            mag2 = mag2 / np.sum(mag2) if np.sum(mag2) > 0 else mag2
                            
                            # Calculate similarity (using correlation)
                            min_len = min(len(mag1), len(mag2))
                            if min_len > 2:
                                corr, _ = stats.pearsonr(mag1[:min_len], mag2[:min_len])
                                if not np.isnan(corr):
                                    frequency_similarities.append(abs(corr))
            
            if frequency_similarities:
                return np.mean(frequency_similarities)
            else:
                return None
                
        except Exception:
            return None    
    def _analyze_regime_transitions(self) -> Dict[str, Any]:
        """Analyze market regime transitions across timeframes."""
        try:
            regime_analysis = {}
            
            # Detect current regime for each timeframe
            timeframe_regimes = {}
            for tf in self.active_timeframes:
                regime = self._detect_timeframe_regime(tf)
                if regime:
                    timeframe_regimes[tf] = regime
            
            regime_analysis['timeframe_regimes'] = timeframe_regimes
            
            # Analyze regime consistency
            if timeframe_regimes:
                regime_values = list(timeframe_regimes.values())
                regime_counts = {regime.value: regime_values.count(regime) for regime in set(regime_values)}
                
                # Most common regime
                dominant_regime = max(regime_counts, key=regime_counts.get)
                regime_consistency = regime_counts[dominant_regime] / len(regime_values)
                
                regime_analysis['dominant_regime'] = dominant_regime
                regime_analysis['regime_consistency'] = regime_consistency
                regime_analysis['regime_distribution'] = regime_counts
            
            # Analyze regime transition patterns
            transition_patterns = self._analyze_transition_patterns()
            regime_analysis['transition_patterns'] = transition_patterns
            
            # Calculate regime stability
            regime_stability = self._calculate_regime_stability()
            regime_analysis['regime_stability'] = regime_stability
            
            return regime_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing regime transitions: {e}")
            return {}
    
    def _detect_timeframe_regime(self, timeframe: str) -> Optional[MarketRegime]:
        """Detect market regime for a specific timeframe."""
        try:
            if timeframe not in self.timeframe_data:
                return None
            
            data = list(self.timeframe_data[timeframe])
            if len(data) < 20:
                return None
            
            # Extract recent price data
            prices = [dp.get('close', dp.get('price', 0)) for dp in data[-20:]]
            
            if not prices or all(p == 0 for p in prices):
                return None
            
            # Calculate regime indicators
            returns = np.diff(np.log(prices)) if len(prices) > 1 else [0]
            
            # Volatility measure
            volatility = np.std(returns) if len(returns) > 1 else 0
            
            # Trend strength
            x = np.arange(len(prices))
            slope, _, r_value, _, _ = stats.linregress(x, prices)
            trend_strength = abs(r_value)
            trend_direction = np.sign(slope)
            
            # Range-bound measure
            price_range = (max(prices) - min(prices)) / np.mean(prices) if np.mean(prices) > 0 else 0
            
            # Classify regime
            if volatility > 0.02:  # High volatility threshold
                if trend_strength > 0.7:
                    return MarketRegime.VOLATILE if price_range > 0.05 else MarketRegime.TRENDING
                else:
                    return MarketRegime.VOLATILE
            elif trend_strength > 0.6:
                # Check for breakout
                recent_change = abs(prices[-1] - prices[-5]) / prices[-5] if len(prices) >= 5 and prices[-5] > 0 else 0
                if recent_change > 0.03:
                    return MarketRegime.BREAKOUT
                else:
                    return MarketRegime.TRENDING
            elif price_range < 0.02:
                return MarketRegime.QUIET
            else:
                return MarketRegime.RANGING
                
        except Exception:
            return None
    
    def _analyze_transition_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in regime transitions."""
        try:
            if len(self.regime_history) < 10:
                return {}
            
            transitions = []
            recent_regimes = list(self.regime_history)
            
            # Find transitions
            for i in range(1, len(recent_regimes)):
                if recent_regimes[i] != recent_regimes[i-1]:
                    transitions.append({
                        'from': recent_regimes[i-1].value,
                        'to': recent_regimes[i].value,
                        'index': i
                    })
            
            if not transitions:
                return {'transition_frequency': 0, 'common_transitions': {}}
            
            # Analyze transition frequency
            transition_frequency = len(transitions) / len(recent_regimes)
            
            # Find common transition patterns
            transition_pairs = [(t['from'], t['to']) for t in transitions]
            transition_counts = {}
            for pair in transition_pairs:
                transition_counts[f"{pair[0]}->{pair[1]}"] = transition_counts.get(f"{pair[0]}->{pair[1]}", 0) + 1
            
            # Most common transitions
            common_transitions = dict(sorted(transition_counts.items(), key=lambda x: x[1], reverse=True)[:5])
            
            return {
                'transition_frequency': transition_frequency,
                'total_transitions': len(transitions),
                'common_transitions': common_transitions,
                'transition_details': transitions[-5:]  # Last 5 transitions
            }
            
        except Exception:
            return {}
    
    def _calculate_regime_stability(self) -> float:
        """Calculate overall regime stability."""
        try:
            if len(self.regime_history) < 5:
                return 0.5
            
            # Calculate how long regimes persist on average
            recent_regimes = list(self.regime_history)
            regime_durations = []
            current_regime = recent_regimes[0]
            current_duration = 1
            
            for i in range(1, len(recent_regimes)):
                if recent_regimes[i] == current_regime:
                    current_duration += 1
                else:
                    regime_durations.append(current_duration)
                    current_regime = recent_regimes[i]
                    current_duration = 1
            
            # Add final duration
            regime_durations.append(current_duration)
            
            # Average duration as stability measure
            if regime_durations:
                avg_duration = np.mean(regime_durations)
                # Normalize to 0-1 scale (assuming max duration of 20)
                stability = min(1.0, avg_duration / 20.0)
                return stability
            else:
                return 0.5
                
        except Exception:
            return 0.5
    
    def _calculate_optimal_weights(self) -> Dict[str, float]:
        """Calculate optimal weights for timeframe combination."""
        try:
            if not self.timeframe_metrics:
                return {}
            
            # Extract metrics for optimization
            timeframes = []
            reliability_scores = []
            information_content = []
            signal_to_noise = []
            
            for tf, metrics in self.timeframe_metrics.items():
                if len(self.timeframe_data.get(tf, [])) >= self.config['min_data_points']:
                    timeframes.append(tf)
                    reliability_scores.append(metrics.reliability_score)
                    information_content.append(metrics.information_content)
                    signal_to_noise.append(metrics.signal_to_noise_ratio)
            
            if not timeframes:
                return {}
            
            # Multi-criteria optimization
            # Combine different factors
            composite_scores = []
            for i in range(len(timeframes)):
                composite = (
                    reliability_scores[i] * 0.4 +
                    information_content[i] * 0.3 +
                    signal_to_noise[i] * 0.3
                )
                composite_scores.append(composite)
            
            # Normalize to sum to 1
            total_score = sum(composite_scores)
            if total_score > 0:
                optimal_weights = {tf: score / total_score for tf, score in zip(timeframes, composite_scores)}
            else:
                # Equal weights as fallback
                equal_weight = 1.0 / len(timeframes)
                optimal_weights = {tf: equal_weight for tf in timeframes}
            
            return optimal_weights
            
        except Exception as e:
            self.logger.error(f"Error calculating optimal weights: {e}")
            return {}
    
    def _generate_multi_timeframe_signals(self) -> Dict[str, Any]:
        """Generate signals based on multi-timeframe analysis."""
        try:
            signals = {}
            
            # Generate individual timeframe signals
            individual_signals = {}
            for tf in self.active_timeframes:
                signal = self._generate_timeframe_signal(tf)
                if signal is not None:
                    individual_signals[tf] = signal
            
            signals['individual_signals'] = individual_signals
            
            # Combine signals using optimal weights
            combined_signal = self._combine_timeframe_signals(individual_signals)
            signals['combined_signal'] = combined_signal
            
            # Generate regime-based signals
            regime_signals = self._generate_regime_signals()
            signals['regime_signals'] = regime_signals
            
            # Generate synchronization signals
            sync_signals = self._generate_synchronization_signals()
            signals['synchronization_signals'] = sync_signals
            
            # Generate confluence signals
            confluence_signals = self._generate_confluence_signals(individual_signals)
            signals['confluence_signals'] = confluence_signals
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating multi-timeframe signals: {e}")
            return {}
    
    def _generate_timeframe_signal(self, timeframe: str) -> Optional[Dict[str, Any]]:
        """Generate signal for a specific timeframe."""
        try:
            if timeframe not in self.timeframe_data:
                return None
            
            data = list(self.timeframe_data[timeframe])
            if len(data) < 10:
                return None
            
            metrics = self.timeframe_metrics.get(timeframe)
            if not metrics:
                return None
            
            # Extract recent prices
            prices = [dp.get('close', dp.get('price', 0)) for dp in data[-10:]]
            
            if not prices:
                return None
            
            # Calculate signal components
            trend_signal = self._calculate_trend_signal(prices)
            momentum_signal = self._calculate_momentum_signal(prices)
            volatility_signal = self._calculate_volatility_signal(prices)
            quality_factor = metrics.data_quality
            
            # Combine signals
            raw_signal = (
                trend_signal * 0.4 +
                momentum_signal * 0.4 +
                volatility_signal * 0.2
            )
            
            # Apply quality adjustment
            adjusted_signal = raw_signal * quality_factor
            
            # Calculate confidence
            confidence = (
                metrics.reliability_score * 0.4 +
                metrics.signal_to_noise_ratio * 0.3 +
                quality_factor * 0.3
            )
            
            return {
                'signal': adjusted_signal,
                'confidence': confidence,
                'trend_component': trend_signal,
                'momentum_component': momentum_signal,
                'volatility_component': volatility_signal,
                'quality_factor': quality_factor,
                'timeframe': timeframe
            }
            
        except Exception:
            return None
    
    def _calculate_trend_signal(self, prices: List[float]) -> float:
        """Calculate trend-based signal component."""
        try:
            if len(prices) < 3:
                return 0.0
            
            # Linear regression trend
            x = np.arange(len(prices))
            slope, _, r_value, _, _ = stats.linregress(x, prices)
            
            # Normalize slope
            price_mean = np.mean(prices)
            normalized_slope = slope / price_mean if price_mean > 0 else 0
            
            # Trend signal with strength weighting
            trend_signal = normalized_slope * (r_value ** 2)
            
            # Cap signal to [-1, 1] range
            return max(-1.0, min(1.0, trend_signal * 100))
            
        except Exception:
            return 0.0
    
    def _calculate_momentum_signal(self, prices: List[float]) -> float:
        """Calculate momentum-based signal component."""
        try:
            if len(prices) < 5:
                return 0.0
            
            # Rate of change
            roc = (prices[-1] - prices[-5]) / prices[-5] if prices[-5] > 0 else 0
            
            # Momentum acceleration
            mid_point = len(prices) // 2
            recent_slope = (prices[-1] - prices[mid_point]) / (len(prices) - mid_point)
            older_slope = (prices[mid_point] - prices[0]) / mid_point
            
            acceleration = recent_slope - older_slope
            
            # Combine ROC and acceleration
            momentum_signal = roc * 0.7 + acceleration * 0.3
            
            # Normalize and cap
            return max(-1.0, min(1.0, momentum_signal * 50))
            
        except Exception:
            return 0.0
    
    def _calculate_volatility_signal(self, prices: List[float]) -> float:
        """Calculate volatility-based signal component."""
        try:
            if len(prices) < 3:
                return 0.0
            
            # Calculate returns
            returns = np.diff(np.log(prices)) if len(prices) > 1 else [0]
            
            if not returns:
                return 0.0
            
            # Current volatility
            current_vol = np.std(returns[-3:]) if len(returns) >= 3 else np.std(returns)
            
            # Historical volatility
            hist_vol = np.std(returns)
            
            # Volatility signal (negative when vol is high)
            if hist_vol > 0:
                vol_ratio = current_vol / hist_vol
                vol_signal = 1.0 - min(2.0, vol_ratio)  # Penalize high volatility
            else:
                vol_signal = 0.0
            
            return max(-1.0, min(1.0, vol_signal))
            
        except Exception:
            return 0.0
    
    def _combine_timeframe_signals(self, individual_signals: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Combine individual timeframe signals using optimal weights."""
        try:
            if not individual_signals:
                return {'signal': 0.0, 'confidence': 0.0}
            
            # Get optimal weights
            optimal_weights = self._calculate_optimal_weights()
            
            # Calculate weighted signal
            weighted_signal = 0.0
            weighted_confidence = 0.0
            total_weight = 0.0
            
            signal_details = {}
            
            for tf, signal_data in individual_signals.items():
                weight = optimal_weights.get(tf, 1.0 / len(individual_signals))
                signal = signal_data.get('signal', 0.0)
                confidence = signal_data.get('confidence', 0.0)
                
                weighted_signal += signal * weight * confidence  # Weight by confidence too
                weighted_confidence += confidence * weight
                total_weight += weight
                
                signal_details[tf] = {
                    'signal': signal,
                    'confidence': confidence,
                    'weight': weight,
                    'contribution': signal * weight
                }
            
            # Normalize
            if total_weight > 0:
                final_signal = weighted_signal / total_weight
                final_confidence = weighted_confidence / total_weight
            else:
                final_signal = 0.0
                final_confidence = 0.0
            
            return {
                'signal': final_signal,
                'confidence': final_confidence,
                'component_signals': signal_details,
                'total_weight': total_weight,
                'signal_count': len(individual_signals)
            }
            
        except Exception as e:
            self.logger.error(f"Error combining timeframe signals: {e}")
            return {'signal': 0.0, 'confidence': 0.0}
    
    def _generate_regime_signals(self) -> Dict[str, Any]:
        """Generate signals based on current market regime."""
        try:
            current_regime = self.current_regime
            regime_consistency = 0.0
            
            # Calculate regime consistency across timeframes
            timeframe_regimes = []
            for tf in self.active_timeframes:
                regime = self._detect_timeframe_regime(tf)
                if regime:
                    timeframe_regimes.append(regime)
            
            if timeframe_regimes:
                regime_counts = {regime: timeframe_regimes.count(regime) for regime in set(timeframe_regimes)}
                dominant_regime = max(regime_counts, key=regime_counts.get)
                regime_consistency = regime_counts[dominant_regime] / len(timeframe_regimes)
                current_regime = dominant_regime
            
            # Generate regime-specific signals
            regime_signal = 0.0
            regime_confidence = regime_consistency
            
            if current_regime == MarketRegime.TRENDING:
                regime_signal = 0.8  # Strong positive signal for trending
            elif current_regime == MarketRegime.BREAKOUT:
                regime_signal = 0.6  # Moderate positive signal for breakout
            elif current_regime == MarketRegime.RANGING:
                regime_signal = 0.0  # Neutral for ranging
            elif current_regime == MarketRegime.VOLATILE:
                regime_signal = -0.3  # Slight negative for high volatility
            elif current_regime == MarketRegime.QUIET:
                regime_signal = 0.2  # Slight positive for quiet periods
            elif current_regime == MarketRegime.REVERSAL:
                regime_signal = -0.5  # Negative for reversal
            
            return {
                'regime': current_regime.value,
                'signal': regime_signal,
                'confidence': regime_confidence,
                'regime_consistency': regime_consistency,
                'timeframe_regimes': [r.value for r in timeframe_regimes]
            }
            
        except Exception as e:
            self.logger.error(f"Error generating regime signals: {e}")
            return {'regime': 'unknown', 'signal': 0.0, 'confidence': 0.0}
    
    def _generate_synchronization_signals(self) -> Dict[str, Any]:
        """Generate signals based on timeframe synchronization."""
        try:
            sync_metrics = self._calculate_synchronization_metrics()
            
            if not sync_metrics:
                return {'signal': 0.0, 'confidence': 0.0}
            
            # Extract synchronization measures
            overall_sync = sync_metrics.get('overall_synchronization', 0.0)
            phase_sync = sync_metrics.get('phase_synchronization', 0.0)
            amplitude_sync = sync_metrics.get('amplitude_synchronization', 0.0)
            frequency_sync = sync_metrics.get('frequency_synchronization', 0.0)
            
            # Generate synchronization signal
            # High synchronization suggests strong consensus
            sync_signal = overall_sync * 0.8  # Positive signal when synchronized
            sync_confidence = overall_sync  # Confidence based on sync level
            
            return {
                'signal': sync_signal,
                'confidence': sync_confidence,
                'overall_synchronization': overall_sync,
                'phase_synchronization': phase_sync,
                'amplitude_synchronization': amplitude_sync,
                'frequency_synchronization': frequency_sync
            }
            
        except Exception as e:
            self.logger.error(f"Error generating synchronization signals: {e}")
            return {'signal': 0.0, 'confidence': 0.0}
    
    def _generate_confluence_signals(self, individual_signals: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate signals based on confluence between timeframes."""
        try:
            if len(individual_signals) < 2:
                return {'signal': 0.0, 'confidence': 0.0, 'confluence_count': 0}
            
            # Extract signals and classify
            signals = [data.get('signal', 0.0) for data in individual_signals.values()]
            confidences = [data.get('confidence', 0.0) for data in individual_signals.values()]
            
            # Count confluences (signals in same direction)
            positive_signals = [s for s in signals if s > 0.1]
            negative_signals = [s for s in signals if s < -0.1]
            neutral_signals = [s for s in signals if -0.1 <= s <= 0.1]
            
            # Calculate confluence strength
            max_confluence_count = max(len(positive_signals), len(negative_signals))
            confluence_ratio = max_confluence_count / len(signals)
            
            # Determine confluence signal
            if len(positive_signals) > len(negative_signals):
                confluence_signal = confluence_ratio * np.mean(positive_signals)
            elif len(negative_signals) > len(positive_signals):
                confluence_signal = confluence_ratio * np.mean(negative_signals)
            else:
                confluence_signal = 0.0
            
            # Confluence confidence based on agreement and individual confidences
            confluence_confidence = confluence_ratio * np.mean(confidences)
            
            return {
                'signal': confluence_signal,
                'confidence': confluence_confidence,
                'confluence_ratio': confluence_ratio,
                'confluence_count': max_confluence_count,
                'positive_count': len(positive_signals),
                'negative_count': len(negative_signals),
                'neutral_count': len(neutral_signals),
                'total_timeframes': len(signals)
            }
            
        except Exception as e:
            self.logger.error(f"Error generating confluence signals: {e}")
            return {'signal': 0.0, 'confidence': 0.0, 'confluence_count': 0}    
    def _optimize_timeframe_selection(self) -> Dict[str, Any]:
        """Optimize timeframe selection for current market conditions."""
        try:
            optimization_results = {}
            
            # Score all timeframes
            timeframe_scores = {}
            for tf in self.active_timeframes:
                score = self._score_timeframe(tf)
                if score is not None:
                    timeframe_scores[tf] = score
            
            if not timeframe_scores:
                return optimization_results
            
            # Rank timeframes
            ranked_timeframes = sorted(timeframe_scores.items(), key=lambda x: x[1], reverse=True)
            optimization_results['ranked_timeframes'] = ranked_timeframes
            
            # Select optimal set
            optimal_set = self._select_optimal_timeframe_set(timeframe_scores)
            optimization_results['optimal_set'] = optimal_set
            
            # Calculate efficiency metrics
            efficiency_metrics = self._calculate_selection_efficiency(optimal_set)
            optimization_results['efficiency_metrics'] = efficiency_metrics
            
            # Optimization recommendations
            recommendations = self._generate_optimization_recommendations(ranked_timeframes)
            optimization_results['recommendations'] = recommendations
            
            return optimization_results
            
        except Exception as e:
            self.logger.error(f"Error in timeframe optimization: {e}")
            return {}
    
    def _score_timeframe(self, timeframe: str) -> Optional[float]:
        """Score a timeframe based on multiple criteria."""
        try:
            if timeframe not in self.timeframe_metrics:
                return None
            
            metrics = self.timeframe_metrics[timeframe]
            
            # Scoring components
            reliability_score = metrics.reliability_score * 0.25
            information_score = metrics.information_content * 0.20
            quality_score = metrics.data_quality * 0.15
            snr_score = metrics.signal_to_noise_ratio * 0.15
            efficiency_score = metrics.efficiency_ratio * 0.10
            trend_score = metrics.trend_strength * 0.10
            low_noise_score = (1.0 - metrics.noise_level) * 0.05
            
            # Composite score
            total_score = (
                reliability_score + information_score + quality_score +
                snr_score + efficiency_score + trend_score + low_noise_score
            )
            
            return min(1.0, max(0.0, total_score))
            
        except Exception:
            return None
    
    def _select_optimal_timeframe_set(self, timeframe_scores: Dict[str, float]) -> List[str]:
        """Select optimal set of timeframes avoiding redundancy."""
        try:
            if not timeframe_scores:
                return []
            
            # Start with highest scoring timeframe
            sorted_timeframes = sorted(timeframe_scores.items(), key=lambda x: x[1], reverse=True)
            optimal_set = [sorted_timeframes[0][0]]
            
            # Add timeframes that provide additional value
            for tf, score in sorted_timeframes[1:]:
                if len(optimal_set) >= self.config['max_timeframes']:
                    break
                
                # Check correlation with existing timeframes
                add_timeframe = True
                for existing_tf in optimal_set:
                    correlation = self._calculate_timeframe_correlation(tf, existing_tf)
                    if correlation is not None and abs(correlation) > 0.8:
                        add_timeframe = False
                        break
                
                # Add if not too correlated and score is decent
                if add_timeframe and score > 0.3:
                    optimal_set.append(tf)
            
            return optimal_set
            
        except Exception:
            return []
    
    def _calculate_selection_efficiency(self, optimal_set: List[str]) -> Dict[str, float]:
        """Calculate efficiency metrics for timeframe selection."""
        try:
            if not optimal_set:
                return {}
            
            # Calculate total information content
            total_information = 0.0
            total_reliability = 0.0
            
            for tf in optimal_set:
                if tf in self.timeframe_metrics:
                    metrics = self.timeframe_metrics[tf]
                    total_information += metrics.information_content
                    total_reliability += metrics.reliability_score
            
            # Efficiency metrics
            avg_information = total_information / len(optimal_set)
            avg_reliability = total_reliability / len(optimal_set)
            
            # Diversity measure (based on correlations)
            correlations = []
            for i, tf1 in enumerate(optimal_set):
                for j, tf2 in enumerate(optimal_set[i+1:], i+1):
                    corr = self._calculate_timeframe_correlation(tf1, tf2)
                    if corr is not None:
                        correlations.append(abs(corr))
            
            diversity = 1.0 - np.mean(correlations) if correlations else 1.0
            
            return {
                'total_information': total_information,
                'average_information': avg_information,
                'total_reliability': total_reliability,
                'average_reliability': avg_reliability,
                'diversity_measure': diversity,
                'efficiency_ratio': (avg_information * avg_reliability * diversity)
            }
            
        except Exception:
            return {}
    
    def _generate_optimization_recommendations(self, ranked_timeframes: List[Tuple[str, float]]) -> List[str]:
        """Generate optimization recommendations."""
        try:
            recommendations = []
            
            if not ranked_timeframes:
                recommendations.append("No timeframe data available for optimization")
                return recommendations
            
            # Analyze top performers
            top_performers = ranked_timeframes[:3]
            bottom_performers = ranked_timeframes[-2:] if len(ranked_timeframes) > 2 else []
            
            # Recommendations based on performance
            if top_performers:
                best_tf, best_score = top_performers[0]
                recommendations.append(f"Primary timeframe recommendation: {best_tf} (score: {best_score:.3f})")
                
                if len(top_performers) > 1:
                    second_tf, second_score = top_performers[1]
                    recommendations.append(f"Secondary timeframe: {second_tf} (score: {second_score:.3f})")
            
            # Identify weak timeframes
            if bottom_performers:
                for tf, score in bottom_performers:
                    if score < 0.3:
                        recommendations.append(f"Consider removing {tf} (low score: {score:.3f})")
            
            # Regime-specific recommendations
            if self.current_regime == MarketRegime.VOLATILE:
                recommendations.append("High volatility detected: favor longer timeframes for stability")
            elif self.current_regime == MarketRegime.TRENDING:
                recommendations.append("Trending market: multiple timeframes showing confluence recommended")
            elif self.current_regime == MarketRegime.RANGING:
                recommendations.append("Range-bound market: shorter timeframes may provide better signals")
            
            return recommendations
            
        except Exception:
            return ["Error generating recommendations"]
    
    def _adapt_timeframes_to_regime(self, regime: MarketRegime) -> Dict[str, Any]:
        """Adapt timeframe configuration based on market regime."""
        try:
            adapted_config = {
                'regime': regime.value,
                'primary_timeframe': self.config['primary_timeframe'],
                'active_timeframes': list(self.active_timeframes),
                'weights': {},
                'adaptations': []
            }
            
            # Regime-specific adaptations
            if regime == MarketRegime.VOLATILE:
                # Favor longer timeframes in volatile conditions
                adapted_config['adaptations'].append("Increasing weights for longer timeframes")
                for tf in self.active_timeframes:
                    if 'h' in tf or 'd' in tf or 'w' in tf:  # Hour, day, week timeframes
                        adapted_config['weights'][tf] = 1.5
                    else:
                        adapted_config['weights'][tf] = 0.7
                        
            elif regime == MarketRegime.TRENDING:
                # Balanced approach for trending markets
                adapted_config['adaptations'].append("Balanced multi-timeframe approach for trending")
                for tf in self.active_timeframes:
                    adapted_config['weights'][tf] = 1.0
                    
            elif regime == MarketRegime.RANGING:
                # Favor shorter timeframes in ranging markets
                adapted_config['adaptations'].append("Increasing weights for shorter timeframes")
                for tf in self.active_timeframes:
                    if 'm' in tf and not 'h' in tf:  # Minute timeframes
                        adapted_config['weights'][tf] = 1.3
                    else:
                        adapted_config['weights'][tf] = 0.8
                        
            elif regime == MarketRegime.BREAKOUT:
                # Focus on medium timeframes for breakouts
                adapted_config['adaptations'].append("Focus on medium timeframes for breakout confirmation")
                for tf in self.active_timeframes:
                    if 'h' in tf:  # Hour timeframes
                        adapted_config['weights'][tf] = 1.4
                    else:
                        adapted_config['weights'][tf] = 0.9
                        
            elif regime == MarketRegime.QUIET:
                # Reduce overall activity in quiet markets
                adapted_config['adaptations'].append("Reduced activity for quiet market conditions")
                for tf in self.active_timeframes:
                    adapted_config['weights'][tf] = 0.6
                    
            else:  # REVERSAL or unknown
                # Conservative approach
                adapted_config['adaptations'].append("Conservative approach for uncertain conditions")
                for tf in self.active_timeframes:
                    adapted_config['weights'][tf] = 0.8
            
            # Normalize weights
            total_weight = sum(adapted_config['weights'].values())
            if total_weight > 0:
                adapted_config['weights'] = {tf: w / total_weight for tf, w in adapted_config['weights'].items()}
            
            return adapted_config
            
        except Exception as e:
            self.logger.error(f"Error adapting timeframes to regime: {e}")
            return {}
    
    def _detect_market_regime(self) -> MarketRegime:
        """Detect current market regime based on all timeframes."""
        try:
            # Collect regime votes from all timeframes
            regime_votes = []
            for tf in self.active_timeframes:
                regime = self._detect_timeframe_regime(tf)
                if regime:
                    regime_votes.append(regime)
            
            if not regime_votes:
                return self.current_regime  # Keep current if no data
            
            # Count votes
            regime_counts = {regime: regime_votes.count(regime) for regime in set(regime_votes)}
            
            # Select regime with most votes
            detected_regime = max(regime_counts, key=regime_counts.get)
            
            # Update regime history
            self.regime_history.append(detected_regime)
            self.current_regime = detected_regime
            
            return detected_regime
            
        except Exception:
            return self.current_regime
    
    def _calculate_quality_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive quality metrics."""
        try:
            quality_metrics = {}
            
            # Overall system quality
            timeframe_qualities = [metrics.data_quality for metrics in self.timeframe_metrics.values()]
            overall_quality = np.mean(timeframe_qualities) if timeframe_qualities else 0.0
            
            quality_metrics['overall_quality'] = overall_quality
            quality_metrics['timeframe_count'] = len(self.active_timeframes)
            quality_metrics['data_completeness'] = self._assess_data_completeness()
            
            # Quality distribution
            if timeframe_qualities:
                quality_metrics['quality_std'] = np.std(timeframe_qualities)
                quality_metrics['min_quality'] = min(timeframe_qualities)
                quality_metrics['max_quality'] = max(timeframe_qualities)
            
            # Recent quality trend
            if len(self.quality_history) > 5:
                recent_qualities = list(self.quality_history)[-10:]
                quality_trend = np.polyfit(range(len(recent_qualities)), recent_qualities, 1)[0]
                quality_metrics['quality_trend'] = quality_trend
            
            # Add current quality to history
            self.quality_history.append(overall_quality)
            
            return quality_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating quality metrics: {e}")
            return {}
    
    def _assess_data_completeness(self) -> float:
        """Assess data completeness across timeframes."""
        try:
            completeness_scores = []
            
            for tf in self.active_timeframes:
                if tf in self.timeframe_data:
                    expected_points = self.config['analysis_window']
                    actual_points = len(self.timeframe_data[tf])
                    completeness = min(1.0, actual_points / expected_points)
                    completeness_scores.append(completeness)
            
            return np.mean(completeness_scores) if completeness_scores else 0.0
            
        except Exception:
            return 0.0
    
    def _generate_timeframe_recommendations(self) -> List[str]:
        """Generate comprehensive timeframe recommendations."""
        try:
            recommendations = []
            
            # Analyze current configuration
            if not self.active_timeframes:
                recommendations.append("No active timeframes - add timeframes to begin analysis")
                return recommendations
            
            # Quality-based recommendations
            quality_metrics = self._calculate_quality_metrics()
            overall_quality = quality_metrics.get('overall_quality', 0.0)
            
            if overall_quality < 0.5:
                recommendations.append(f"Low data quality detected ({overall_quality:.2f}) - consider data source improvements")
            elif overall_quality > 0.8:
                recommendations.append(f"Excellent data quality ({overall_quality:.2f}) - optimal for analysis")
            
            # Performance-based recommendations
            best_performers = []
            weak_performers = []
            
            for tf, metrics in self.timeframe_metrics.items():
                score = self._score_timeframe(tf)
                if score is not None:
                    if score > 0.7:
                        best_performers.append((tf, score))
                    elif score < 0.3:
                        weak_performers.append((tf, score))
            
            if best_performers:
                best_tf = max(best_performers, key=lambda x: x[1])
                recommendations.append(f"Top performing timeframe: {best_tf[0]} (score: {best_tf[1]:.3f})")
            
            if weak_performers:
                for tf, score in weak_performers:
                    recommendations.append(f"Consider reviewing {tf} timeframe (low score: {score:.3f})")
            
            # Regime-specific recommendations
            regime_recommendations = self._get_regime_recommendations()
            recommendations.extend(regime_recommendations)
            
            # Synchronization recommendations
            sync_metrics = self._calculate_synchronization_metrics()
            overall_sync = sync_metrics.get('overall_synchronization', 0.0)
            
            if overall_sync < 0.3:
                recommendations.append("Low timeframe synchronization - signals may be conflicting")
            elif overall_sync > 0.7:
                recommendations.append("High timeframe synchronization - strong signal consensus")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return ["Error generating recommendations"]
    
    def _get_regime_recommendations(self) -> List[str]:
        """Get regime-specific recommendations."""
        try:
            recommendations = []
            regime = self.current_regime
            
            if regime == MarketRegime.VOLATILE:
                recommendations.append("High volatility regime: Use wider stops and longer timeframes")
                recommendations.append("Consider reducing position sizes in volatile conditions")
                
            elif regime == MarketRegime.TRENDING:
                recommendations.append("Trending regime: Multi-timeframe confluence favorable")
                recommendations.append("Trend-following strategies recommended")
                
            elif regime == MarketRegime.RANGING:
                recommendations.append("Range-bound regime: Mean reversion strategies preferred")
                recommendations.append("Focus on shorter timeframes for range trading")
                
            elif regime == MarketRegime.BREAKOUT:
                recommendations.append("Breakout regime: Monitor for confirmation across timeframes")
                recommendations.append("Volume confirmation important for breakout validation")
                
            elif regime == MarketRegime.QUIET:
                recommendations.append("Quiet market regime: Reduced trading activity recommended")
                recommendations.append("Wait for increased volume and volatility")
                
            elif regime == MarketRegime.REVERSAL:
                recommendations.append("Reversal regime: Exercise caution with trend trades")
                recommendations.append("Look for confirmation before position changes")
            
            return recommendations
            
        except Exception:
            return []
    
    def _format_correlation_matrix(self) -> Dict[str, Any]:
        """Format correlation matrix for output."""
        try:
            if self.correlation_matrix.empty:
                return {}
            
            return {
                'matrix': self.correlation_matrix.to_dict(),
                'strong_correlations': self._find_strong_correlations(),
                'average_correlation': float(np.mean(np.abs(self.correlation_matrix.values)))
            }
            
        except Exception:
            return {}
    
    def _find_strong_correlations(self) -> List[Dict[str, Any]]:
        """Find strong correlations in the matrix."""
        try:
            strong_correlations = []
            matrix = self.correlation_matrix.values
            
            for i in range(len(matrix)):
                for j in range(i + 1, len(matrix)):
                    corr_value = matrix[i, j]
                    if abs(corr_value) > 0.7:
                        strong_correlations.append({
                            'timeframe1': self.correlation_matrix.index[i],
                            'timeframe2': self.correlation_matrix.columns[j],
                            'correlation': float(corr_value),
                            'strength': 'strong' if abs(corr_value) > 0.8 else 'moderate'
                        })
            
            return sorted(strong_correlations, key=lambda x: abs(x['correlation']), reverse=True)
            
        except Exception:
            return []
    
    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary across timeframes."""
        try:
            summary = {}
            
            # Accuracy summary
            accuracy_data = {}
            for tf, accuracies in self.prediction_accuracy.items():
                if accuracies:
                    accuracy_data[tf] = {
                        'mean_accuracy': float(np.mean(accuracies)),
                        'std_accuracy': float(np.std(accuracies)),
                        'sample_count': len(accuracies)
                    }
            
            summary['prediction_accuracy'] = accuracy_data
            
            # Execution performance
            execution_data = {}
            for tf, performance in self.execution_performance.items():
                if performance:
                    execution_data[tf] = {
                        'mean_performance': float(np.mean(performance)),
                        'std_performance': float(np.std(performance)),
                        'sample_count': len(performance)
                    }
            
            summary['execution_performance'] = execution_data
            
            # Overall metrics
            all_accuracies = [acc for accs in self.prediction_accuracy.values() for acc in accs]
            all_execution = [perf for perfs in self.execution_performance.values() for perf in perfs]
            
            if all_accuracies:
                summary['overall_accuracy'] = float(np.mean(all_accuracies))
            if all_execution:
                summary['overall_execution'] = float(np.mean(all_execution))
            
            return summary
            
        except Exception:
            return {}
    
    def _assess_system_health(self) -> Dict[str, Any]:
        """Assess overall system health."""
        try:
            health_metrics = {}
            
            # Data health
            data_completeness = self._assess_data_completeness()
            health_metrics['data_health'] = 'good' if data_completeness > 0.8 else 'fair' if data_completeness > 0.5 else 'poor'
            
            # Timeframe health
            active_count = len(self.active_timeframes)
            recommended_count = min(5, len(self.config['supported_timeframes']))
            tf_health = active_count / recommended_count
            health_metrics['timeframe_health'] = 'good' if tf_health > 0.8 else 'fair' if tf_health > 0.5 else 'poor'
            
            # Quality health
            quality_metrics = self._calculate_quality_metrics()
            overall_quality = quality_metrics.get('overall_quality', 0.0)
            health_metrics['quality_health'] = 'good' if overall_quality > 0.7 else 'fair' if overall_quality > 0.4 else 'poor'
            
            # Synchronization health
            sync_metrics = self._calculate_synchronization_metrics()
            overall_sync = sync_metrics.get('overall_synchronization', 0.0)
            health_metrics['sync_health'] = 'good' if overall_sync > 0.6 else 'fair' if overall_sync > 0.3 else 'poor'
            
            # Overall health score
            health_scores = {'good': 1.0, 'fair': 0.5, 'poor': 0.0}
            overall_score = np.mean([health_scores[h] for h in health_metrics.values() if h in health_scores])
            
            if overall_score > 0.75:
                health_metrics['overall_health'] = 'excellent'
            elif overall_score > 0.5:
                health_metrics['overall_health'] = 'good'
            elif overall_score > 0.25:
                health_metrics['overall_health'] = 'fair'
            else:
                health_metrics['overall_health'] = 'poor'
            
            health_metrics['health_score'] = float(overall_score)
            
            return health_metrics
            
        except Exception:
            return {'overall_health': 'unknown', 'health_score': 0.0}
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error response."""
        return {
            'error': True,
            'message': error_message,
            'timeframe_analysis': {},
            'multi_timeframe_signals': {},
            'optimal_configuration': {},
            'current_regime': 'unknown',
            'timestamp': datetime.now().isoformat()
        }
    
    # Public utility methods
    def add_timeframe(self, timeframe: str) -> bool:
        """Add a new timeframe to active analysis."""
        try:
            if timeframe not in self.config['supported_timeframes']:
                self.config['supported_timeframes'].append(timeframe)
            
            self.active_timeframes.add(timeframe)
            
            # Initialize data structures
            max_len = max(self.config['analysis_window'], self.config['ml_training_window'])
            self.timeframe_data[timeframe] = deque(maxlen=max_len)
            self.timeframe_metrics[timeframe] = TimeframeMetrics(timeframe=timeframe)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding timeframe {timeframe}: {e}")
            return False
    
    def remove_timeframe(self, timeframe: str) -> bool:
        """Remove a timeframe from active analysis."""
        try:
            if timeframe in self.active_timeframes:
                self.active_timeframes.remove(timeframe)
            
            # Clean up data structures
            if timeframe in self.timeframe_data:
                del self.timeframe_data[timeframe]
            if timeframe in self.timeframe_metrics:
                del self.timeframe_metrics[timeframe]
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error removing timeframe {timeframe}: {e}")
            return False
    
    def get_timeframe_status(self) -> Dict[str, Any]:
        """Get current status of all timeframes."""
        try:
            status = {}
            
            for tf in self.active_timeframes:
                metrics = self.timeframe_metrics.get(tf)
                data_count = len(self.timeframe_data.get(tf, []))
                
                status[tf] = {
                    'active': True,
                    'data_points': data_count,
                    'reliability_score': metrics.reliability_score if metrics else 0.0,
                    'data_quality': metrics.data_quality if metrics else 0.0,
                    'last_updated': metrics.last_updated.isoformat() if metrics and metrics.last_updated else None
                }
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error getting timeframe status: {e}")
            return {}
    
    def reset_system(self):
        """Reset the entire timeframe system."""
        try:
            # Clear all data
            self.timeframe_data.clear()
            self.timeframe_metrics.clear()
            self.correlation_matrix = pd.DataFrame()
            self.regime_history.clear()
            self.signal_history.clear()
            self.quality_history.clear()
            
            # Reset to initial state
            self.current_regime = MarketRegime.RANGING
            self.active_timeframes = set(self.config['supported_timeframes'])
            
            # Reinitialize
            self._initialize_timeframes()
            self._setup_synchronization()
            
            self.logger.info("Timeframe system reset completed")
            
        except Exception as e:
            self.logger.error(f"Error resetting system: {e}")


# Example usage and testing
if __name__ == "__main__":
    # Initialize the timeframe configuration system
    timeframe_config = TimeframeConfigIndicator()
    
    # Generate sample multi-timeframe data
    np.random.seed(42)
    timeframes = ['1m', '5m', '15m', '1h', '4h']
    
    # Simulate 100 data points
    for i in range(100):
        timestamp = datetime.now() - timedelta(minutes=100-i)
        base_price = 1.2000 + np.cumsum(np.random.randn(1) * 0.0001)[0]
        
        for tf in timeframes:
            # Add some timeframe-specific noise and trends
            tf_multiplier = {'1m': 1.0, '5m': 0.8, '15m': 0.6, '1h': 0.4, '4h': 0.2}
            noise = np.random.randn() * 0.0005 * tf_multiplier.get(tf, 1.0)
            
            data_point = {
                'timestamp': timestamp,
                'timeframe': tf,
                'open': base_price - noise,
                'high': base_price + abs(noise) * 1.5,
                'low': base_price - abs(noise) * 1.5,
                'close': base_price + noise,
                'price': base_price + noise,
                'volume': np.random.lognormal(8, 0.5)
            }
            
            result = timeframe_config.calculate(data_point)
            
            # Print sample results every 20 iterations
            if i % 20 == 0 and tf == '1h':
                print(f"Step {i}, Timeframe {tf}:")
                print(f"  Current Regime: {result.get('current_regime', 'unknown')}")
                print(f"  Active Timeframes: {len(result.get('active_timeframes', []))}")
                print(f"  System Health: {result.get('system_health', {}).get('overall_health', 'unknown')}")
                print()
    
    # Get final system status
    final_status = timeframe_config.get_timeframe_status()
    print("Final Timeframe Status:")
    for tf, status in final_status.items():
        print(f"  {tf}: {status['data_points']} points, quality: {status['data_quality']:.3f}")
    
    print(f"\nSystem Health: {timeframe_config._assess_system_health()}")
    print(f"Current Regime: {timeframe_config.current_regime.value}")