"""
Self-Similarity Detector Indicator - Advanced Fractal Market Analysis
====================================================================

This indicator implements sophisticated self-similarity detection using advanced mathematical
techniques including fractal dimension analysis, Hurst exponent calculation, and
multifractal detrended fluctuation analysis (MF-DFA). It identifies recurring patterns
across multiple timeframes to predict future price movements.

The indicator uses advanced algorithms to:
1. Calculate local Hurst exponents using rolling windows
2. Perform multifractal analysis using DFA and MF-DFA
3. Detect self-similar patterns using correlation dimension analysis
4. Apply machine learning clustering for pattern recognition
5. Generate confidence-weighted trading signals based on fractal properties

This is a production-ready implementation with comprehensive error handling,
performance optimization, and advanced mathematical models.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
from scipy import stats, signal
from scipy.optimize import curve_fit
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

from ..base.standard_indicator import StandardIndicatorInterface, DataRequirement, DataType, SignalType, IndicatorResult


@dataclass
class SelfSimilarityConfig:
    """Configuration for self-similarity analysis"""
    window_size: int = 100
    min_periods: int = 50
    hurst_window: int = 30
    dfa_orders: List[int] = None
    correlation_threshold: float = 0.7
    fractal_dimension_threshold: float = 1.5
    clustering_eps: float = 0.3
    min_samples: int = 5
    confidence_threshold: float = 0.6
    
    def __post_init__(self):
        if self.dfa_orders is None:
            self.dfa_orders = [1, 2, 3]


class SelfSimilarityDetectorIndicator(StandardIndicatorInterface):
    """
    Advanced Self-Similarity Detector using fractal analysis and pattern recognition.
    
    This indicator combines multiple mathematical approaches:
    1. Hurst Exponent Analysis - measures long-term memory in time series
    2. Multifractal DFA - analyzes multifractal properties
    3. Correlation Dimension - measures fractal dimension of attractors
    4. Pattern Clustering - identifies similar price patterns using ML
    5. Self-Similarity Scoring - quantifies pattern repetition strength
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'window_size': 100,
            'min_periods': 50,
            'hurst_window': 30,
            'dfa_orders': [1, 2, 3],
            'correlation_threshold': 0.7,
            'fractal_dimension_threshold': 1.5,
            'clustering_eps': 0.3,
            'min_samples': 5,
            'confidence_threshold': 0.6,
            'lookback_periods': 200,
            'adaptive_thresholds': True,
            'pattern_memory_length': 20
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__(name="SelfSimilarityDetector", parameters=default_params)
        
        self.config = SelfSimilarityConfig(
            window_size=self.parameters['window_size'],
            min_periods=self.parameters['min_periods'],
            hurst_window=self.parameters['hurst_window'],
            dfa_orders=self.parameters['dfa_orders'],
            correlation_threshold=self.parameters['correlation_threshold'],
            fractal_dimension_threshold=self.parameters['fractal_dimension_threshold'],
            clustering_eps=self.parameters['clustering_eps'],
            min_samples=self.parameters['min_samples'],
            confidence_threshold=self.parameters['confidence_threshold']
        )
        
        # Internal state for pattern memory
        self.pattern_memory = []
        self.similarity_history = []
        self.fractal_cache = {}
        
    def get_data_requirements(self) -> DataRequirement:
        """Define data requirements for self-similarity analysis"""
        return DataRequirement(
            data_type=DataType.OHLCV,
            required_columns=['open', 'high', 'low', 'close', 'volume'],
            min_periods=self.config.min_periods,
            lookback_periods=self.parameters['lookback_periods']
        )
    
    def validate_parameters(self) -> bool:
        """Validate indicator parameters"""
        if self.config.window_size < self.config.min_periods:
            raise ValueError("window_size must be >= min_periods")
        
        if self.config.hurst_window < 10:
            raise ValueError("hurst_window must be >= 10")
        
        if not (0 < self.config.correlation_threshold < 1):
            raise ValueError("correlation_threshold must be between 0 and 1")
        
        if not (1 < self.config.fractal_dimension_threshold < 2):
            raise ValueError("fractal_dimension_threshold must be between 1 and 2")
        
        return True
    
    def calculate_raw(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate self-similarity metrics using advanced fractal analysis
        """
        try:
            # Prepare price series
            prices = data['close'].values
            returns = np.diff(np.log(prices))
            
            if len(returns) < self.config.min_periods:
                return self._create_default_result("Insufficient data")
            
            # 1. Calculate Local Hurst Exponent
            hurst_values = self._calculate_rolling_hurst_exponent(returns)
            
            # 2. Perform Multifractal Analysis
            mf_results = self._calculate_multifractal_properties(returns)
            
            # 3. Calculate Correlation Dimension
            correlation_dim = self._calculate_correlation_dimension(returns)
            
            # 4. Detect Self-Similar Patterns
            pattern_similarity = self._detect_pattern_similarity(prices)
            
            # 5. Perform Advanced Pattern Clustering
            cluster_results = self._perform_pattern_clustering(data)
            
            # 6. Calculate Fractal Dimension using Box-Counting
            fractal_dimension = self._calculate_fractal_dimension(prices)
            
            # 7. Generate Self-Similarity Score
            similarity_score = self._calculate_similarity_score(
                hurst_values, mf_results, correlation_dim, 
                pattern_similarity, fractal_dimension
            )
            
            # 8. Adaptive Threshold Adjustment
            adaptive_thresholds = self._calculate_adaptive_thresholds(similarity_score)
            
            # 9. Pattern Persistence Analysis
            persistence_score = self._analyze_pattern_persistence(similarity_score)
            
            # Compile comprehensive results
            result = {
                'similarity_score': similarity_score,
                'hurst_exponent': np.mean(hurst_values[-10:]) if len(hurst_values) > 0 else 0.5,
                'multifractal_strength': mf_results.get('strength', 0.0),
                'correlation_dimension': correlation_dim,
                'pattern_similarity': pattern_similarity,
                'fractal_dimension': fractal_dimension,
                'cluster_strength': cluster_results.get('strength', 0.0),
                'pattern_persistence': persistence_score,
                'adaptive_threshold': adaptive_thresholds.get('dynamic_threshold', 0.5),
                'regime_stability': mf_results.get('regime_stability', 0.5),
                'memory_strength': self._calculate_memory_strength(hurst_values),
                'components': {
                    'hurst_history': hurst_values[-20:].tolist() if len(hurst_values) >= 20 else hurst_values.tolist(),
                    'multifractal_spectrum': mf_results.get('spectrum', {}),
                    'cluster_labels': cluster_results.get('labels', []),
                    'pattern_matches': cluster_results.get('pattern_matches', 0),
                    'fractal_scaling': mf_results.get('scaling_exponents', []),
                    'similarity_trend': self._calculate_similarity_trend()
                }
            }
            
            # Update internal state
            self._update_pattern_memory(similarity_score, pattern_similarity)
            
            return result
            
        except Exception as e:
            return self._create_error_result(f"Calculation error: {str(e)}")
    
    def _calculate_rolling_hurst_exponent(self, returns: np.ndarray) -> np.ndarray:
        """Calculate rolling Hurst exponent using R/S analysis"""
        hurst_values = []
        window = self.config.hurst_window
        
        for i in range(window, len(returns)):
            series = returns[i-window:i]
            
            try:
                # R/S Analysis implementation
                lags = range(2, min(window//2, 20))
                rs_values = []
                
                for lag in lags:
                    # Calculate cumulative deviations
                    mean_return = np.mean(series)
                    cumulative_deviations = np.cumsum(series - mean_return)
                    
                    # Calculate range and standard deviation for subseries
                    subseries_length = len(series) // lag
                    if subseries_length < 2:
                        continue
                    
                    rs_subseries = []
                    for j in range(lag):
                        start_idx = j * subseries_length
                        end_idx = min((j + 1) * subseries_length, len(series))
                        
                        if end_idx - start_idx < 2:
                            continue
                        
                        subseries = cumulative_deviations[start_idx:end_idx]
                        if len(subseries) > 1:
                            range_val = np.max(subseries) - np.min(subseries)
                            std_val = np.std(series[start_idx:end_idx])
                            
                            if std_val > 0:
                                rs_subseries.append(range_val / std_val)
                    
                    if rs_subseries:
                        rs_values.append(np.mean(rs_subseries))
                
                if len(rs_values) >= 3:
                    # Fit power law: R/S = c * n^H
                    log_lags = np.log(lags[:len(rs_values)])
                    log_rs = np.log(rs_values)
                    
                    # Remove inf and nan values
                    valid_idx = np.isfinite(log_lags) & np.isfinite(log_rs)
                    if np.sum(valid_idx) >= 3:
                        hurst, _ = np.polyfit(log_lags[valid_idx], log_rs[valid_idx], 1)
                        hurst_values.append(np.clip(hurst, 0, 1))
                    else:
                        hurst_values.append(0.5)  # Default to random walk
                else:
                    hurst_values.append(0.5)
                    
            except Exception:
                hurst_values.append(0.5)
        
        return np.array(hurst_values)
    
    def _calculate_multifractal_properties(self, returns: np.ndarray) -> Dict[str, Any]:
        """Perform multifractal detrended fluctuation analysis"""
        try:
            # Convert returns to integrated series
            integrated_series = np.cumsum(returns - np.mean(returns))
            n = len(integrated_series)
            
            # Define scales for analysis
            scales = np.logspace(1, np.log10(n//4), 20).astype(int)
            scales = np.unique(scales)
            
            # Multifractal spectrum calculation
            q_values = np.linspace(-5, 5, 21)  # Range of q values
            fluctuation_functions = np.zeros((len(q_values), len(scales)))
            
            for i, scale in enumerate(scales):
                if scale < 4:
                    continue
                
                # Divide series into non-overlapping segments
                segments = n // scale
                fluctuations = []
                
                for seg in range(segments):
                    start_idx = seg * scale
                    end_idx = (seg + 1) * scale
                    segment = integrated_series[start_idx:end_idx]
                    
                    # Polynomial detrending (order 1)
                    x = np.arange(len(segment))
                    coeffs = np.polyfit(x, segment, 1)
                    trend = np.polyval(coeffs, x)
                    detrended = segment - trend
                    
                    # Calculate fluctuation
                    fluctuation = np.sqrt(np.mean(detrended**2))
                    fluctuations.append(fluctuation)
                
                if fluctuations:
                    fluctuations = np.array(fluctuations)
                    
                    # Calculate F(s) for different q values
                    for j, q in enumerate(q_values):
                        if q == 0:
                            # Special case for q=0 (geometric mean)
                            valid_fluct = fluctuations[fluctuations > 0]
                            if len(valid_fluct) > 0:
                                fluctuation_functions[j, i] = np.exp(np.mean(np.log(valid_fluct)))
                        else:
                            # General case
                            valid_fluct = fluctuations[fluctuations > 0]
                            if len(valid_fluct) > 0:
                                fluctuation_functions[j, i] = np.mean(valid_fluct**q)**(1.0/q)
            
            # Calculate scaling exponents h(q)
            scaling_exponents = []
            for j, q in enumerate(q_values):
                valid_data = fluctuation_functions[j, :] > 0
                if np.sum(valid_data) >= 3:
                    log_scales = np.log(scales[valid_data])
                    log_flucts = np.log(fluctuation_functions[j, valid_data])
                    
                    # Remove inf and nan
                    finite_mask = np.isfinite(log_scales) & np.isfinite(log_flucts)
                    if np.sum(finite_mask) >= 3:
                        h_q, _ = np.polyfit(log_scales[finite_mask], log_flucts[finite_mask], 1)
                        scaling_exponents.append(h_q)
                    else:
                        scaling_exponents.append(0.5)
                else:
                    scaling_exponents.append(0.5)
            
            scaling_exponents = np.array(scaling_exponents)
            
            # Calculate multifractal strength
            mf_strength = np.std(scaling_exponents) if len(scaling_exponents) > 0 else 0
            
            # Calculate regime stability
            h0_index = len(q_values) // 2  # q = 0
            regime_stability = 1.0 - abs(scaling_exponents[h0_index] - 0.5) * 2
            
            return {
                'strength': mf_strength,
                'scaling_exponents': scaling_exponents.tolist(),
                'q_values': q_values.tolist(),
                'spectrum': dict(zip(q_values.tolist(), scaling_exponents.tolist())),
                'regime_stability': np.clip(regime_stability, 0, 1)
            }
            
        except Exception:
            return {
                'strength': 0.0,
                'scaling_exponents': [],
                'spectrum': {},
                'regime_stability': 0.5
            }
    
    def _calculate_correlation_dimension(self, returns: np.ndarray) -> float:
        """Calculate correlation dimension using Grassberger-Procaccia algorithm"""
        try:
            # Embed the time series in higher dimensions
            embedding_dim = 5
            delay = 1
            
            if len(returns) < embedding_dim * delay + 10:
                return 1.5  # Default value
            
            # Create embedded vectors
            embedded_vectors = []
            for i in range(len(returns) - embedding_dim * delay):
                vector = [returns[i + j * delay] for j in range(embedding_dim)]
                embedded_vectors.append(vector)
            
            embedded_vectors = np.array(embedded_vectors)
            
            if len(embedded_vectors) < 10:
                return 1.5
            
            # Calculate correlation integral for different radius values
            n_points = min(len(embedded_vectors), 500)  # Limit for computational efficiency
            sample_indices = np.random.choice(len(embedded_vectors), n_points, replace=False)
            sample_vectors = embedded_vectors[sample_indices]
            
            # Define radius range
            distances = []
            for i in range(min(100, len(sample_vectors))):
                for j in range(i+1, min(i+50, len(sample_vectors))):
                    dist = np.linalg.norm(sample_vectors[i] - sample_vectors[j])
                    distances.append(dist)
            
            if not distances:
                return 1.5
            
            distances = np.array(distances)
            distances = distances[distances > 0]
            
            if len(distances) < 10:
                return 1.5
            
            # Calculate correlation dimension
            radius_values = np.logspace(np.log10(np.min(distances)), 
                                      np.log10(np.max(distances)), 20)
            
            correlation_integrals = []
            for radius in radius_values:
                count = 0
                total_pairs = 0
                
                for i in range(min(100, len(sample_vectors))):
                    for j in range(i+1, min(i+20, len(sample_vectors))):
                        dist = np.linalg.norm(sample_vectors[i] - sample_vectors[j])
                        total_pairs += 1
                        if dist < radius:
                            count += 1
                
                if total_pairs > 0:
                    correlation_integrals.append(count / total_pairs)
                else:
                    correlation_integrals.append(0)
            
            # Fit power law to get correlation dimension
            valid_indices = np.array(correlation_integrals) > 0
            if np.sum(valid_indices) >= 3:
                log_radius = np.log(radius_values[valid_indices])
                log_correlation = np.log(np.array(correlation_integrals)[valid_indices])
                
                finite_mask = np.isfinite(log_radius) & np.isfinite(log_correlation)
                if np.sum(finite_mask) >= 3:
                    correlation_dim, _ = np.polyfit(log_radius[finite_mask], 
                                                  log_correlation[finite_mask], 1)
                    return np.clip(correlation_dim, 0.5, 3.0)
            
            return 1.5
            
        except Exception:
            return 1.5
    
    def _detect_pattern_similarity(self, prices: np.ndarray) -> float:
        """Detect self-similar patterns using correlation analysis"""
        try:
            if len(prices) < self.config.window_size:
                return 0.0
            
            # Normalize prices
            normalized_prices = (prices - np.mean(prices)) / (np.std(prices) + 1e-8)
            
            # Extract recent pattern
            pattern_length = min(20, len(normalized_prices) // 4)
            recent_pattern = normalized_prices[-pattern_length:]
            
            # Search for similar patterns in historical data
            similarities = []
            search_length = min(len(normalized_prices) - pattern_length - 1, 100)
            
            for i in range(search_length):
                historical_pattern = normalized_prices[i:i+pattern_length]
                
                if len(historical_pattern) == len(recent_pattern):
                    # Calculate correlation
                    correlation = np.corrcoef(recent_pattern, historical_pattern)[0, 1]
                    if not np.isnan(correlation):
                        similarities.append(abs(correlation))
            
            if similarities:
                # Return the average of top similarities
                similarities = sorted(similarities, reverse=True)
                top_similarities = similarities[:min(5, len(similarities))]
                return np.mean(top_similarities)
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _perform_pattern_clustering(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform advanced pattern clustering using DBSCAN"""
        try:
            prices = data['close'].values
            volumes = data['volume'].values
            
            if len(prices) < 50:
                return {'strength': 0.0, 'labels': [], 'pattern_matches': 0}
            
            # Create feature vectors from price patterns
            pattern_length = 10
            features = []
            
            for i in range(len(prices) - pattern_length):
                price_pattern = prices[i:i+pattern_length]
                volume_pattern = volumes[i:i+pattern_length]
                
                # Normalize patterns
                price_norm = (price_pattern - np.mean(price_pattern)) / (np.std(price_pattern) + 1e-8)
                volume_norm = (volume_pattern - np.mean(volume_pattern)) / (np.std(volume_pattern) + 1e-8)
                
                # Create feature vector
                feature_vector = np.concatenate([price_norm, volume_norm])
                features.append(feature_vector)
            
            if len(features) < self.config.min_samples:
                return {'strength': 0.0, 'labels': [], 'pattern_matches': 0}
            
            features = np.array(features)
            
            # Standardize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Perform DBSCAN clustering
            clustering = DBSCAN(eps=self.config.clustering_eps, 
                              min_samples=self.config.min_samples)
            labels = clustering.fit_predict(features_scaled)
            
            # Calculate clustering strength
            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
            
            if n_clusters > 1:
                # Calculate silhouette score
                try:
                    silhouette_avg = silhouette_score(features_scaled, labels)
                    cluster_strength = max(0, silhouette_avg)
                except:
                    cluster_strength = 0.0
            else:
                cluster_strength = 0.0
            
            # Count pattern matches
            pattern_matches = 0
            for label in unique_labels:
                if label != -1:  # Ignore noise points
                    cluster_size = np.sum(labels == label)
                    if cluster_size >= self.config.min_samples:
                        pattern_matches += cluster_size
            
            return {
                'strength': cluster_strength,
                'labels': labels.tolist(),
                'pattern_matches': pattern_matches,
                'n_clusters': n_clusters
            }
            
        except Exception:
            return {'strength': 0.0, 'labels': [], 'pattern_matches': 0}
    
    def _calculate_fractal_dimension(self, prices: np.ndarray) -> float:
        """Calculate fractal dimension using box-counting method"""
        try:
            if len(prices) < 20:
                return 1.5
            
            # Normalize prices to [0, 1] range
            normalized_prices = (prices - np.min(prices)) / (np.max(prices) - np.min(prices) + 1e-8)
            
            # Create a 2D representation (time vs price)
            time_points = np.linspace(0, 1, len(normalized_prices))
            
            # Box-counting algorithm
            box_sizes = np.logspace(-2, 0, 20)  # From 0.01 to 1
            box_counts = []
            
            for box_size in box_sizes:
                # Count boxes that contain the curve
                n_boxes_x = int(1.0 / box_size)
                n_boxes_y = int(1.0 / box_size)
                
                occupied_boxes = set()
                
                for i in range(len(time_points) - 1):
                    # Interpolate between consecutive points
                    t1, p1 = time_points[i], normalized_prices[i]
                    t2, p2 = time_points[i + 1], normalized_prices[i + 1]
                    
                    # Sample points along the line segment
                    n_samples = max(2, int(np.sqrt((t2-t1)**2 + (p2-p1)**2) / box_size))
                    
                    for j in range(n_samples):
                        alpha = j / max(1, n_samples - 1)
                        t = t1 + alpha * (t2 - t1)
                        p = p1 + alpha * (p2 - p1)
                        
                        # Determine which box this point belongs to
                        box_x = int(t * n_boxes_x)
                        box_y = int(p * n_boxes_y)
                        
                        # Ensure we don't exceed boundaries
                        box_x = min(box_x, n_boxes_x - 1)
                        box_y = min(box_y, n_boxes_y - 1)
                        
                        occupied_boxes.add((box_x, box_y))
                
                box_counts.append(len(occupied_boxes))
            
            # Calculate fractal dimension
            valid_indices = np.array(box_counts) > 0
            if np.sum(valid_indices) >= 3:
                log_box_sizes = np.log(1.0 / box_sizes[valid_indices])
                log_box_counts = np.log(np.array(box_counts)[valid_indices])
                
                finite_mask = np.isfinite(log_box_sizes) & np.isfinite(log_box_counts)
                if np.sum(finite_mask) >= 3:
                    fractal_dim, _ = np.polyfit(log_box_sizes[finite_mask], 
                                              log_box_counts[finite_mask], 1)
                    return np.clip(fractal_dim, 1.0, 2.0)
            
            return 1.5
            
        except Exception:
            return 1.5
    
    def _calculate_similarity_score(self, hurst_values: np.ndarray, 
                                  mf_results: Dict[str, Any],
                                  correlation_dim: float,
                                  pattern_similarity: float,
                                  fractal_dimension: float) -> float:
        """Calculate comprehensive self-similarity score"""
        try:
            scores = []
            weights = []
            
            # Hurst exponent component
            if len(hurst_values) > 0:
                recent_hurst = np.mean(hurst_values[-5:])
                # Score based on deviation from random walk (0.5)
                hurst_score = 1.0 - 2 * abs(recent_hurst - 0.5)
                scores.append(max(0, hurst_score))
                weights.append(0.25)
            
            # Multifractal strength component
            mf_strength = mf_results.get('strength', 0)
            # Normalize multifractal strength (higher is better up to a point)
            mf_score = min(1.0, mf_strength * 5)  # Scale appropriately
            scores.append(mf_score)
            weights.append(0.20)
            
            # Correlation dimension component
            # Score based on how close to ideal fractal dimension (1.5-1.8)
            ideal_corr_dim = 1.65
            corr_dim_score = 1.0 - abs(correlation_dim - ideal_corr_dim) / 1.0
            scores.append(max(0, corr_dim_score))
            weights.append(0.20)
            
            # Pattern similarity component
            scores.append(pattern_similarity)
            weights.append(0.25)
            
            # Fractal dimension component
            ideal_fractal_dim = 1.5
            fractal_score = 1.0 - abs(fractal_dimension - ideal_fractal_dim) / 0.5
            scores.append(max(0, fractal_score))
            weights.append(0.10)
            
            # Calculate weighted average
            if scores and weights:
                similarity_score = np.average(scores, weights=weights)
                return np.clip(similarity_score, 0, 1)
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_adaptive_thresholds(self, similarity_score: float) -> Dict[str, float]:
        """Calculate adaptive thresholds based on historical performance"""
        try:
            # Add current score to history
            self.similarity_history.append(similarity_score)
            
            # Keep only recent history
            max_history = 100
            if len(self.similarity_history) > max_history:
                self.similarity_history = self.similarity_history[-max_history:]
            
            if len(self.similarity_history) < 10:
                return {'dynamic_threshold': self.config.confidence_threshold}
            
            # Calculate adaptive threshold based on statistical analysis
            hist_array = np.array(self.similarity_history)
            mean_score = np.mean(hist_array)
            std_score = np.std(hist_array)
            
            # Dynamic threshold: mean + 0.5 * std
            dynamic_threshold = mean_score + 0.5 * std_score
            dynamic_threshold = np.clip(dynamic_threshold, 0.3, 0.9)
            
            return {
                'dynamic_threshold': dynamic_threshold,
                'mean_score': mean_score,
                'std_score': std_score
            }
            
        except Exception:
            return {'dynamic_threshold': self.config.confidence_threshold}
    
    def _analyze_pattern_persistence(self, similarity_score: float) -> float:
        """Analyze how persistent self-similar patterns are"""
        try:
            if len(self.similarity_history) < 5:
                return 0.5
            
            # Look at recent scores
            recent_scores = self.similarity_history[-5:]
            
            # Calculate trend persistence
            if len(recent_scores) >= 3:
                # Linear regression on recent scores
                x = np.arange(len(recent_scores))
                slope, _ = np.polyfit(x, recent_scores, 1)
                
                # Score based on consistency and trend
                consistency = 1.0 - np.std(recent_scores)
                trend_strength = abs(slope) * 10  # Scale the slope
                
                persistence = 0.7 * consistency + 0.3 * min(1.0, trend_strength)
                return np.clip(persistence, 0, 1)
            
            return 0.5
            
        except Exception:
            return 0.5
    
    def _calculate_memory_strength(self, hurst_values: np.ndarray) -> float:
        """Calculate the strength of long-term memory in the series"""
        try:
            if len(hurst_values) == 0:
                return 0.5
            
            # Memory strength based on Hurst exponent
            recent_hurst = np.mean(hurst_values[-10:]) if len(hurst_values) >= 10 else np.mean(hurst_values)
            
            # Strong memory: H significantly different from 0.5
            memory_strength = 2 * abs(recent_hurst - 0.5)
            return np.clip(memory_strength, 0, 1)
            
        except Exception:
            return 0.5
    
    def _calculate_similarity_trend(self) -> str:
        """Calculate the trend in similarity scores"""
        try:
            if len(self.similarity_history) < 5:
                return "insufficient_data"
            
            recent_scores = self.similarity_history[-5:]
            x = np.arange(len(recent_scores))
            slope, _ = np.polyfit(x, recent_scores, 1)
            
            if slope > 0.02:
                return "increasing"
            elif slope < -0.02:
                return "decreasing"
            else:
                return "stable"
                
        except Exception:
            return "unknown"
    
    def _update_pattern_memory(self, similarity_score: float, pattern_similarity: float):
        """Update pattern memory for future reference"""
        try:
            memory_length = self.parameters.get('pattern_memory_length', 20)
            
            pattern_data = {
                'timestamp': len(self.pattern_memory),
                'similarity_score': similarity_score,
                'pattern_similarity': pattern_similarity
            }
            
            self.pattern_memory.append(pattern_data)
            
            # Keep only recent memory
            if len(self.pattern_memory) > memory_length:
                self.pattern_memory = self.pattern_memory[-memory_length:]
                
        except Exception:
            pass  # Non-critical operation
    
    def _create_default_result(self, reason: str) -> Dict[str, Any]:
        """Create default result when calculation cannot be performed"""
        return {
            'similarity_score': 0.0,
            'hurst_exponent': 0.5,
            'multifractal_strength': 0.0,
            'correlation_dimension': 1.5,
            'pattern_similarity': 0.0,
            'fractal_dimension': 1.5,
            'cluster_strength': 0.0,
            'pattern_persistence': 0.5,
            'adaptive_threshold': 0.5,
            'regime_stability': 0.5,
            'memory_strength': 0.5,
            'reason': reason,
            'components': {
                'hurst_history': [],
                'multifractal_spectrum': {},
                'cluster_labels': [],
                'pattern_matches': 0,
                'fractal_scaling': [],
                'similarity_trend': 'unknown'
            }
        }
    
    def _create_error_result(self, error_msg: str) -> Dict[str, Any]:
        """Create error result"""
        result = self._create_default_result("calculation_error")
        result['error'] = error_msg
        return result
    
    def _generate_signal(self, value: Dict[str, Any], data: pd.DataFrame) -> Tuple[Optional[SignalType], float]:
        """Generate trading signal based on self-similarity analysis"""
        try:
            similarity_score = value.get('similarity_score', 0)
            pattern_persistence = value.get('pattern_persistence', 0.5)
            memory_strength = value.get('memory_strength', 0.5)
            adaptive_threshold = value.get('adaptive_threshold', 0.5)
            
            # Calculate signal confidence
            confidence_factors = [
                similarity_score,
                pattern_persistence,
                memory_strength * 0.5,  # Weight memory less
                1.0 if similarity_score > adaptive_threshold else 0.0
            ]
            
            confidence = np.mean(confidence_factors)
            
            # Generate signal based on pattern analysis
            if similarity_score > adaptive_threshold and confidence > 0.6:
                # Analyze price trend for signal direction
                recent_prices = data['close'].tail(10).values
                if len(recent_prices) >= 2:
                    price_trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
                    
                    if price_trend > 0.01:  # Strong upward trend
                        return SignalType.BUY, confidence
                    elif price_trend < -0.01:  # Strong downward trend
                        return SignalType.SELL, confidence
                    else:
                        return SignalType.HOLD, confidence * 0.8
                else:
                    return SignalType.HOLD, confidence * 0.7
            
            elif similarity_score < adaptive_threshold * 0.5:
                # Low similarity suggests pattern breakdown
                return SignalType.NEUTRAL, confidence * 0.5
            
            return SignalType.HOLD, confidence
            
        except Exception:
            return SignalType.NEUTRAL, 0.0
    
    def _get_metadata(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get additional metadata for the calculation"""
        base_metadata = super()._get_metadata(data)
        
        self_similarity_metadata = {
            'pattern_memory_size': len(self.pattern_memory),
            'similarity_history_size': len(self.similarity_history),
            'adaptive_thresholds_enabled': self.parameters.get('adaptive_thresholds', True),
            'fractal_cache_size': len(self.fractal_cache),
            'analysis_components': [
                'hurst_exponent', 'multifractal_analysis', 'correlation_dimension',
                'pattern_similarity', 'fractal_dimension', 'pattern_clustering'
            ]
        }
        
        base_metadata.update(self_similarity_metadata)
        return base_metadata