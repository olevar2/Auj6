"""
Chaos Fractal Dimension Indicator - Advanced Implementation

This indicator implements sophisticated fractal dimension calculation using multiple methods:
- Box-counting method for geometric fractal dimension
- Correlation dimension using Grassberger-Procaccia algorithm
- Information dimension for entropy-based analysis
- Advanced chaos theory metrics for market state assessment

Mission: Supporting humanitarian trading platform for poor and sick children through
maximum profitability via advanced fractal geometry analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.linear_model import LinearRegression
import warnings

from ..base.standard_indicator import StandardIndicatorInterface
from ....core.exceptions import IndicatorCalculationError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FractalDimensionResult:
    """Results container for fractal dimension calculations"""
    box_counting_dimension: float
    correlation_dimension: float
    information_dimension: float
    chaos_level: float
    market_regime: str
    confidence_score: float
    scaling_exponents: List[float]
    error_metrics: Dict[str, float]

class ChaosFractalDimensionIndicator(StandardIndicatorInterface):
    """
    Advanced Chaos Fractal Dimension Indicator
    
    Implements multiple fractal dimension calculation methods for comprehensive
    market chaos and complexity analysis. Uses sophisticated mathematical models
    from chaos theory and fractal geometry.
    """
    
    def __init__(self, 
                 lookback_period: int = 200,
                 embedding_dimension: int = 6,
                 min_box_size: int = 2,
                 max_box_size: int = 50,
                 correlation_radius_min: float = 0.001,
                 correlation_radius_max: float = 0.1,
                 chaos_threshold: float = 0.7,
                 confidence_threshold: float = 0.8):
        """
        Initialize the Chaos Fractal Dimension Indicator
        
        Args:
            lookback_period: Historical data window for analysis
            embedding_dimension: Phase space reconstruction dimension
            min_box_size: Minimum box size for box-counting method
            max_box_size: Maximum box size for box-counting method
            correlation_radius_min: Minimum radius for correlation dimension
            correlation_radius_max: Maximum radius for correlation dimension
            chaos_threshold: Threshold for chaos detection
            confidence_threshold: Minimum confidence for reliable signals
        """
        super().__init__()
        self.lookback_period = lookback_period
        self.embedding_dimension = embedding_dimension
        self.min_box_size = min_box_size
        self.max_box_size = max_box_size
        self.correlation_radius_min = correlation_radius_min
        self.correlation_radius_max = correlation_radius_max
        self.chaos_threshold = chaos_threshold
        self.confidence_threshold = confidence_threshold
        
        # Initialize calculation cache
        self._calculation_cache = {}
        self._previous_results = []
        
        logger.info(f"Initialized ChaosFractalDimensionIndicator with lookback={lookback_period}")

    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate comprehensive fractal dimension analysis
        
        Args:
            data: OHLCV DataFrame with required columns
            
        Returns:
            Dictionary containing fractal dimension metrics and analysis
        """
        try:
            # Validate input data
            self._validate_data(data)
            
            # Prepare price series for analysis
            prices = self._prepare_price_series(data)
            
            if len(prices) < self.lookback_period:
                logger.warning(f"Insufficient data: {len(prices)} < {self.lookback_period}")
                return self._create_default_result()
            
            # Use most recent data window
            recent_prices = prices[-self.lookback_period:].values
            
            # Calculate multiple fractal dimensions
            box_dimension = self._calculate_box_counting_dimension(recent_prices)
            correlation_dimension = self._calculate_correlation_dimension(recent_prices)
            information_dimension = self._calculate_information_dimension(recent_prices)
            
            # Assess market chaos and regime
            chaos_metrics = self._analyze_chaos_metrics(recent_prices)
            market_regime = self._classify_market_regime(box_dimension, correlation_dimension, chaos_metrics)
            
            # Calculate confidence and error metrics
            confidence_score = self._calculate_confidence_score(box_dimension, correlation_dimension)
            error_metrics = self._calculate_error_metrics(recent_prices)
            
            # Create comprehensive result
            result = FractalDimensionResult(
                box_counting_dimension=box_dimension,
                correlation_dimension=correlation_dimension,
                information_dimension=information_dimension,
                chaos_level=chaos_metrics['chaos_level'],
                market_regime=market_regime,
                confidence_score=confidence_score,
                scaling_exponents=chaos_metrics['scaling_exponents'],
                error_metrics=error_metrics
            )
            
            # Store for trend analysis
            self._previous_results.append(result)
            if len(self._previous_results) > 50:
                self._previous_results.pop(0)
            
            return self._format_output(result, data.index[-1])
            
        except Exception as e:
            logger.error(f"Error in chaos fractal dimension calculation: {e}")
            raise IndicatorCalculationError(f"ChaosFractalDimensionIndicator calculation failed: {e}")

    def _calculate_box_counting_dimension(self, prices: np.ndarray) -> float:
        """
        Calculate fractal dimension using box-counting method
        
        Args:
            prices: Price time series
            
        Returns:
            Box-counting fractal dimension
        """
        try:
            # Normalize prices to [0,1] range
            normalized_prices = (prices - prices.min()) / (prices.max() - prices.min())
            
            box_sizes = np.logspace(np.log10(self.min_box_size), 
                                  np.log10(min(self.max_box_size, len(prices)//4)), 
                                  num=20, dtype=int)
            box_sizes = np.unique(box_sizes)
            
            box_counts = []
            
            for box_size in box_sizes:
                # Create 2D embedding for box counting
                if len(normalized_prices) < box_size * 2:
                    continue
                    
                # Phase space reconstruction
                embedded = self._create_phase_space_embedding(normalized_prices, box_size)
                
                # Count boxes containing data points
                count = self._count_occupied_boxes(embedded, box_size)
                box_counts.append(count)
            
            if len(box_counts) < 3:
                return 1.5  # Default dimension for insufficient data
            
            # Linear regression in log-log space
            log_box_sizes = np.log(box_sizes[:len(box_counts)])
            log_box_counts = np.log(box_counts)
            
            # Remove any infinite or NaN values
            valid_mask = np.isfinite(log_box_sizes) & np.isfinite(log_box_counts)
            if not np.any(valid_mask):
                return 1.5
                
            slope, _, r_value, _, _ = stats.linregress(log_box_sizes[valid_mask], 
                                                     log_box_counts[valid_mask])
            
            # Fractal dimension is negative slope
            fractal_dimension = -slope
            
            # Ensure reasonable bounds
            fractal_dimension = np.clip(fractal_dimension, 1.0, 2.0)
            
            return fractal_dimension
            
        except Exception as e:
            logger.warning(f"Box counting calculation error: {e}")
            return 1.5

    def _calculate_correlation_dimension(self, prices: np.ndarray) -> float:
        """
        Calculate correlation dimension using Grassberger-Procaccia algorithm
        
        Args:
            prices: Price time series
            
        Returns:
            Correlation dimension
        """
        try:
            # Create phase space embedding
            embedded = self._create_phase_space_embedding(prices, self.embedding_dimension)
            
            if embedded.shape[0] < 50:  # Need sufficient points
                return 1.5
            
            # Calculate pairwise distances
            distances = pdist(embedded)
            
            # Define radius range
            radius_range = np.logspace(np.log10(self.correlation_radius_min),
                                     np.log10(self.correlation_radius_max),
                                     num=30)
            
            correlations = []
            
            for radius in radius_range:
                # Count pairs within radius
                count = np.sum(distances < radius)
                correlation = count / len(distances) if len(distances) > 0 else 0
                correlations.append(max(correlation, 1e-10))  # Avoid log(0)
            
            # Linear regression in log-log space
            log_radius = np.log(radius_range)
            log_correlations = np.log(correlations)
            
            # Find linear region (typically middle range)
            valid_mask = np.isfinite(log_radius) & np.isfinite(log_correlations)
            if not np.any(valid_mask):
                return 1.5
                
            # Use middle 60% of data for stable estimation
            start_idx = int(0.2 * len(log_radius))
            end_idx = int(0.8 * len(log_radius))
            
            if end_idx <= start_idx:
                return 1.5
                
            slope, _, _, _, _ = stats.linregress(log_radius[start_idx:end_idx], 
                                               log_correlations[start_idx:end_idx])
            
            correlation_dimension = slope
            
            # Ensure reasonable bounds
            correlation_dimension = np.clip(correlation_dimension, 0.5, 3.0)
            
            return correlation_dimension
            
        except Exception as e:
            logger.warning(f"Correlation dimension calculation error: {e}")
            return 1.5

    def _calculate_information_dimension(self, prices: np.ndarray) -> float:
        """
        Calculate information dimension based on entropy
        
        Args:
            prices: Price time series
            
        Returns:
            Information dimension
        """
        try:
            # Create phase space embedding
            embedded = self._create_phase_space_embedding(prices, 3)
            
            box_sizes = np.logspace(np.log10(self.min_box_size), 
                                  np.log10(min(20, len(prices)//10)), 
                                  num=10, dtype=int)
            
            entropies = []
            
            for box_size in box_sizes:
                # Calculate information content for each box size
                entropy = self._calculate_information_entropy(embedded, box_size)
                entropies.append(entropy)
            
            if len(entropies) < 3:
                return 1.5
            
            # Linear regression
            log_box_sizes = np.log(box_sizes)
            
            valid_mask = np.isfinite(log_box_sizes) & np.isfinite(entropies)
            if not np.any(valid_mask):
                return 1.5
                
            slope, _, _, _, _ = stats.linregress(log_box_sizes[valid_mask], 
                                               np.array(entropies)[valid_mask])
            
            information_dimension = -slope
            
            # Ensure reasonable bounds
            information_dimension = np.clip(information_dimension, 0.5, 3.0)
            
            return information_dimension
            
        except Exception as e:
            logger.warning(f"Information dimension calculation error: {e}")
            return 1.5    def _create_phase_space_embedding(self, data: np.ndarray, dimension: int, delay: int = 1) -> np.ndarray:
        """
        Create phase space embedding using Takens' theorem
        
        Args:
            data: Input time series
            dimension: Embedding dimension
            delay: Time delay for embedding
            
        Returns:
            Embedded phase space vectors
        """
        n = len(data)
        m = n - (dimension - 1) * delay
        
        if m <= 0:
            return np.array([[]])
        
        embedded = np.zeros((m, dimension))
        for i in range(dimension):
            embedded[:, i] = data[i * delay:i * delay + m]
            
        return embedded

    def _count_occupied_boxes(self, embedded: np.ndarray, box_size: int) -> int:
        """
        Count number of boxes containing data points
        
        Args:
            embedded: Phase space embedded data
            box_size: Size of boxes for counting
            
        Returns:
            Number of occupied boxes
        """
        if embedded.size == 0:
            return 1
            
        # Normalize to [0,1] range
        min_vals = embedded.min(axis=0)
        max_vals = embedded.max(axis=0)
        range_vals = max_vals - min_vals
        
        # Avoid division by zero
        range_vals[range_vals == 0] = 1.0
        
        normalized = (embedded - min_vals) / range_vals
        
        # Discretize into boxes
        discretized = np.floor(normalized * box_size).astype(int)
        discretized = np.clip(discretized, 0, box_size - 1)
        
        # Count unique boxes
        unique_boxes = set()
        for point in discretized:
            unique_boxes.add(tuple(point))
            
        return len(unique_boxes)

    def _calculate_information_entropy(self, embedded: np.ndarray, box_size: int) -> float:
        """
        Calculate information entropy for given box size
        
        Args:
            embedded: Phase space embedded data
            box_size: Size of boxes for entropy calculation
            
        Returns:
            Information entropy
        """
        if embedded.size == 0:
            return 0.0
            
        # Discretize data into boxes
        min_vals = embedded.min(axis=0)
        max_vals = embedded.max(axis=0)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1.0
        
        normalized = (embedded - min_vals) / range_vals
        discretized = np.floor(normalized * box_size).astype(int)
        discretized = np.clip(discretized, 0, box_size - 1)
        
        # Count occurrences in each box
        box_counts = {}
        for point in discretized:
            box_key = tuple(point)
            box_counts[box_key] = box_counts.get(box_key, 0) + 1
        
        # Calculate probabilities and entropy
        total_points = len(discretized)
        entropy = 0.0
        
        for count in box_counts.values():
            probability = count / total_points
            if probability > 0:
                entropy -= probability * np.log(probability)
                
        return entropy

    def _analyze_chaos_metrics(self, prices: np.ndarray) -> Dict[str, Any]:
        """
        Analyze comprehensive chaos theory metrics
        
        Args:
            prices: Price time series
            
        Returns:
            Dictionary containing chaos analysis results
        """
        try:
            # Calculate Lyapunov exponent approximation
            lyapunov = self._estimate_lyapunov_exponent(prices)
            
            # Calculate scaling exponents
            scaling_exponents = self._calculate_scaling_exponents(prices)
            
            # Estimate chaos level
            chaos_level = self._estimate_chaos_level(lyapunov, scaling_exponents)
            
            # Calculate entropy measures
            entropy_metrics = self._calculate_entropy_measures(prices)
            
            return {
                'lyapunov_exponent': lyapunov,
                'scaling_exponents': scaling_exponents,
                'chaos_level': chaos_level,
                'entropy_metrics': entropy_metrics,
                'divergence_rate': max(0, lyapunov),
                'predictability_horizon': 1.0 / max(0.01, abs(lyapunov))
            }
            
        except Exception as e:
            logger.warning(f"Chaos metrics calculation error: {e}")
            return {
                'lyapunov_exponent': 0.0,
                'scaling_exponents': [1.0],
                'chaos_level': 0.5,
                'entropy_metrics': {'sample_entropy': 1.0},
                'divergence_rate': 0.0,
                'predictability_horizon': 100.0
            }

    def _estimate_lyapunov_exponent(self, prices: np.ndarray) -> float:
        """
        Estimate largest Lyapunov exponent for chaos detection
        
        Args:
            prices: Price time series
            
        Returns:
            Estimated Lyapunov exponent
        """
        try:
            # Create phase space embedding
            embedded = self._create_phase_space_embedding(prices, 3, delay=1)
            
            if embedded.shape[0] < 10:
                return 0.0
            
            # Calculate divergence rates
            divergences = []
            
            for i in range(min(embedded.shape[0] - 10, 50)):
                # Find nearest neighbor
                distances = np.linalg.norm(embedded[i+1:] - embedded[i], axis=1)
                if len(distances) == 0:
                    continue
                    
                nearest_idx = np.argmin(distances) + i + 1
                if nearest_idx >= embedded.shape[0] - 5:
                    continue
                
                # Track divergence over time
                initial_distance = distances[nearest_idx - i - 1]
                if initial_distance == 0:
                    continue
                
                divergence_series = []
                for t in range(1, min(10, embedded.shape[0] - nearest_idx)):
                    current_distance = np.linalg.norm(embedded[i + t] - embedded[nearest_idx + t])
                    if current_distance > 0 and initial_distance > 0:
                        divergence_series.append(np.log(current_distance / initial_distance) / t)
                
                if divergence_series:
                    divergences.extend(divergence_series)
            
            if not divergences:
                return 0.0
            
            # Return median divergence rate as Lyapunov estimate
            return np.median(divergences)
            
        except Exception as e:
            logger.warning(f"Lyapunov exponent estimation error: {e}")
            return 0.0

    def _calculate_scaling_exponents(self, prices: np.ndarray) -> List[float]:
        """
        Calculate multifractal scaling exponents
        
        Args:
            prices: Price time series
            
        Returns:
            List of scaling exponents
        """
        try:
            # Calculate returns
            returns = np.diff(np.log(prices + 1e-10))
            
            if len(returns) < 20:
                return [1.0]
            
            # Define q values for multifractal analysis
            q_values = np.linspace(-3, 3, 7)
            scaling_exponents = []
            
            # Define scale range
            scales = np.logspace(1, min(3, np.log10(len(returns)//4)), num=10, dtype=int)
            
            for q in q_values:
                fluctuations = []
                
                for scale in scales:
                    if scale >= len(returns):
                        continue
                        
                    # Partition series into non-overlapping segments
                    n_segments = len(returns) // scale
                    segments = returns[:n_segments * scale].reshape(n_segments, scale)
                    
                    # Calculate local fluctuations
                    segment_fluctuations = []
                    for segment in segments:
                        # Detrend segment
                        x = np.arange(scale)
                        coeffs = np.polyfit(x, segment, 1)
                        trend = np.polyval(coeffs, x)
                        fluctuation = np.sqrt(np.mean((segment - trend) ** 2))
                        segment_fluctuations.append(fluctuation)
                    
                    if segment_fluctuations:
                        if q == 0:
                            # Special case for q=0 (log averaging)
                            avg_fluctuation = np.exp(np.mean(np.log(np.array(segment_fluctuations) + 1e-10)))
                        else:
                            # General case
                            avg_fluctuation = np.mean(np.array(segment_fluctuations) ** q) ** (1.0 / q) if q != 0 else np.exp(np.mean(np.log(segment_fluctuations + 1e-10)))
                        
                        fluctuations.append(avg_fluctuation)
                
                if len(fluctuations) >= 3:
                    # Linear regression to find scaling exponent
                    log_scales = np.log(scales[:len(fluctuations)])
                    log_fluctuations = np.log(np.array(fluctuations) + 1e-10)
                    
                    slope, _, _, _, _ = stats.linregress(log_scales, log_fluctuations)
                    scaling_exponents.append(slope)
            
            return scaling_exponents if scaling_exponents else [1.0]
            
        except Exception as e:
            logger.warning(f"Scaling exponents calculation error: {e}")
            return [1.0]

    def _estimate_chaos_level(self, lyapunov: float, scaling_exponents: List[float]) -> float:
        """
        Estimate overall chaos level from various metrics
        
        Args:
            lyapunov: Lyapunov exponent
            scaling_exponents: Multifractal scaling exponents
            
        Returns:
            Chaos level between 0 and 1
        """
        try:
            # Positive Lyapunov indicates chaos
            lyapunov_component = max(0, min(1, lyapunov / 0.1))
            
            # Scaling exponent spread indicates multifractality
            if len(scaling_exponents) > 1:
                exponent_spread = np.std(scaling_exponents)
                multifractal_component = min(1, exponent_spread / 0.5)
            else:
                multifractal_component = 0.0
            
            # Combine components
            chaos_level = 0.6 * lyapunov_component + 0.4 * multifractal_component
            
            return np.clip(chaos_level, 0.0, 1.0)
            
        except Exception as e:
            logger.warning(f"Chaos level estimation error: {e}")
            return 0.5

    def _calculate_entropy_measures(self, prices: np.ndarray) -> Dict[str, float]:
        """
        Calculate various entropy measures
        
        Args:
            prices: Price time series
            
        Returns:
            Dictionary of entropy measures
        """
        try:
            returns = np.diff(np.log(prices + 1e-10))
            
            # Sample entropy
            sample_entropy = self._calculate_sample_entropy(returns)
            
            # Approximate entropy
            approximate_entropy = self._calculate_approximate_entropy(returns)
            
            # Shannon entropy of returns distribution
            shannon_entropy = self._calculate_shannon_entropy(returns)
            
            return {
                'sample_entropy': sample_entropy,
                'approximate_entropy': approximate_entropy,
                'shannon_entropy': shannon_entropy
            }
            
        except Exception as e:
            logger.warning(f"Entropy measures calculation error: {e}")
            return {
                'sample_entropy': 1.0,
                'approximate_entropy': 1.0,
                'shannon_entropy': 1.0
            }

    def _calculate_sample_entropy(self, data: np.ndarray, m: int = 2, r: float = None) -> float:
        """
        Calculate sample entropy
        
        Args:
            data: Input time series
            m: Pattern length
            r: Tolerance for matching
            
        Returns:
            Sample entropy value
        """
        try:
            if r is None:
                r = 0.2 * np.std(data)
            
            if len(data) < m + 1:
                return 1.0
            
            def _maxdist(xi, xj, m):
                return max([abs(ua - va) for ua, va in zip(xi, xj)])
            
            def _phi(m):
                patterns = np.array([data[i:i + m] for i in range(len(data) - m + 1)])
                C = np.zeros(len(patterns))
                
                for i in range(len(patterns)):
                    template = patterns[i]
                    for j in range(len(patterns)):
                        if _maxdist(template, patterns[j], m) <= r:
                            C[i] += 1.0
                
                phi = np.mean(np.log(C / len(patterns)))
                return phi
            
            return _phi(m) - _phi(m + 1)
            
        except Exception as e:
            logger.warning(f"Sample entropy calculation error: {e}")
            return 1.0

    def _calculate_approximate_entropy(self, data: np.ndarray, m: int = 2, r: float = None) -> float:
        """
        Calculate approximate entropy
        
        Args:
            data: Input time series
            m: Pattern length
            r: Tolerance for matching
            
        Returns:
            Approximate entropy value
        """
        try:
            if r is None:
                r = 0.2 * np.std(data)
            
            N = len(data)
            
            def _compute_phi(m):
                patterns = [data[i:i + m] for i in range(N - m + 1)]
                C = []
                
                for i in range(len(patterns)):
                    template = patterns[i]
                    matches = 0
                    for j in range(len(patterns)):
                        if max(abs(a - b) for a, b in zip(template, patterns[j])) <= r:
                            matches += 1
                    C.append(matches / len(patterns))
                
                return sum(np.log(c) for c in C if c > 0) / len(C)
            
            return _compute_phi(m) - _compute_phi(m + 1)
            
        except Exception as e:
            logger.warning(f"Approximate entropy calculation error: {e}")
            return 1.0

    def _calculate_shannon_entropy(self, data: np.ndarray, bins: int = 50) -> float:
        """
        Calculate Shannon entropy of data distribution
        
        Args:
            data: Input time series
            bins: Number of bins for histogram
            
        Returns:
            Shannon entropy value
        """
        try:
            hist, _ = np.histogram(data, bins=bins)
            hist = hist[hist > 0]  # Remove empty bins
            
            # Normalize to get probabilities
            probabilities = hist / np.sum(hist)
            
            # Calculate Shannon entropy
            entropy = -np.sum(probabilities * np.log2(probabilities))
            
            return entropy
            
        except Exception as e:
            logger.warning(f"Shannon entropy calculation error: {e}")
            return 1.0

    def _classify_market_regime(self, box_dimension: float, correlation_dimension: float, 
                              chaos_metrics: Dict[str, Any]) -> str:
        """
        Classify market regime based on fractal dimensions and chaos metrics
        
        Args:
            box_dimension: Box-counting fractal dimension
            correlation_dimension: Correlation dimension
            chaos_metrics: Chaos analysis results
            
        Returns:
            Market regime classification
        """
        try:
            chaos_level = chaos_metrics['chaos_level']
            lyapunov = chaos_metrics['lyapunov_exponent']
            
            # Define regime classification rules
            if chaos_level > self.chaos_threshold:
                if lyapunov > 0.05:
                    return "CHAOTIC_TURBULENT"
                elif box_dimension > 1.7:
                    return "CHAOTIC_COMPLEX"
                else:
                    return "CHAOTIC_SIMPLE"
            elif box_dimension > 1.6:
                if correlation_dimension > 2.0:
                    return "TRENDING_STRONG"
                else:
                    return "TRENDING_MODERATE"
            elif box_dimension < 1.3:
                return "RANGING_STABLE"
            else:
                return "TRANSITIONAL"
                
        except Exception as e:
            logger.warning(f"Market regime classification error: {e}")
            return "UNKNOWN"

    def _calculate_confidence_score(self, box_dimension: float, correlation_dimension: float) -> float:
        """
        Calculate confidence score for the analysis
        
        Args:
            box_dimension: Box-counting dimension
            correlation_dimension: Correlation dimension
            
        Returns:
            Confidence score between 0 and 1
        """
        try:
            # Check consistency between different dimension measures
            dimension_consistency = 1.0 - abs(box_dimension - correlation_dimension) / 2.0
            
            # Check if dimensions are in reasonable ranges
            range_validity = 1.0 if (1.0 <= box_dimension <= 2.0 and 0.5 <= correlation_dimension <= 3.0) else 0.5
            
            # Combine factors
            confidence = 0.7 * dimension_consistency + 0.3 * range_validity
            
            return np.clip(confidence, 0.0, 1.0)
            
        except Exception as e:
            logger.warning(f"Confidence calculation error: {e}")
            return 0.5

    def _calculate_error_metrics(self, prices: np.ndarray) -> Dict[str, float]:
        """
        Calculate error and quality metrics for the analysis
        
        Args:
            prices: Price time series
            
        Returns:
            Dictionary of error metrics
        """
        try:
            returns = np.diff(np.log(prices + 1e-10))
            
            # Calculate various error metrics
            error_metrics = {
                'data_quality': self._assess_data_quality(prices),
                'stationarity_score': self._assess_stationarity(returns),
                'noise_level': self._estimate_noise_level(returns),
                'autocorrelation': self._calculate_autocorrelation(returns)
            }
            
            return error_metrics
            
        except Exception as e:
            logger.warning(f"Error metrics calculation error: {e}")
            return {
                'data_quality': 0.8,
                'stationarity_score': 0.5,
                'noise_level': 0.3,
                'autocorrelation': 0.1
            }

    def _assess_data_quality(self, prices: np.ndarray) -> float:
        """
        Assess quality of input data
        
        Args:
            prices: Price time series
            
        Returns:
            Data quality score between 0 and 1
        """
        try:
            # Check for missing values, outliers, etc.
            quality_score = 1.0
            
            # Penalize for constant values
            if np.std(prices) < 1e-10:
                quality_score *= 0.1
            
            # Penalize for extreme outliers
            z_scores = np.abs(stats.zscore(prices))
            outlier_ratio = np.sum(z_scores > 5) / len(prices)
            quality_score *= (1.0 - outlier_ratio)
            
            # Penalize for irregular gaps
            price_diffs = np.diff(prices)
            if len(price_diffs) > 0:
                diff_outliers = np.sum(np.abs(stats.zscore(price_diffs)) > 4) / len(price_diffs)
                quality_score *= (1.0 - diff_outliers)
            
            return np.clip(quality_score, 0.0, 1.0)
            
        except Exception as e:
            logger.warning(f"Data quality assessment error: {e}")
            return 0.8

    def _assess_stationarity(self, returns: np.ndarray) -> float:
        """
        Assess stationarity of returns series
        
        Args:
            returns: Return time series
            
        Returns:
            Stationarity score between 0 and 1
        """
        try:
            if len(returns) < 20:
                return 0.5
            
            # Split into segments and compare statistics
            n_segments = min(4, len(returns) // 10)
            segment_size = len(returns) // n_segments
            
            segment_means = []
            segment_stds = []
            
            for i in range(n_segments):
                start_idx = i * segment_size
                end_idx = (i + 1) * segment_size if i < n_segments - 1 else len(returns)
                segment = returns[start_idx:end_idx]
                
                segment_means.append(np.mean(segment))
                segment_stds.append(np.std(segment))
            
            # Calculate stability of statistics
            mean_stability = 1.0 - np.std(segment_means) / (np.mean(np.abs(segment_means)) + 1e-10)
            std_stability = 1.0 - np.std(segment_stds) / (np.mean(segment_stds) + 1e-10)
            
            stationarity_score = 0.5 * mean_stability + 0.5 * std_stability
            
            return np.clip(stationarity_score, 0.0, 1.0)
            
        except Exception as e:
            logger.warning(f"Stationarity assessment error: {e}")
            return 0.5

    def _estimate_noise_level(self, returns: np.ndarray) -> float:
        """
        Estimate noise level in returns
        
        Args:
            returns: Return time series
            
        Returns:
            Estimated noise level
        """
        try:
            if len(returns) < 10:
                return 0.5
            
            # Use first-order differencing to estimate noise
            diff_returns = np.diff(returns)
            noise_estimate = np.std(diff_returns) / (np.std(returns) + 1e-10)
            
            return np.clip(noise_estimate, 0.0, 1.0)
            
        except Exception as e:
            logger.warning(f"Noise level estimation error: {e}")
            return 0.3

    def _calculate_autocorrelation(self, returns: np.ndarray, lag: int = 1) -> float:
        """
        Calculate autocorrelation of returns
        
        Args:
            returns: Return time series
            lag: Lag for autocorrelation
            
        Returns:
            Autocorrelation coefficient
        """
        try:
            if len(returns) < lag + 10:
                return 0.0
            
            # Calculate autocorrelation
            correlation = np.corrcoef(returns[:-lag], returns[lag:])[0, 1]
            
            return correlation if not np.isnan(correlation) else 0.0
            
        except Exception as e:
            logger.warning(f"Autocorrelation calculation error: {e}")
            return 0.0

    def _format_output(self, result: FractalDimensionResult, timestamp) -> Dict[str, Any]:
        """
        Format the calculation results for output
        
        Args:
            result: Fractal dimension calculation results
            timestamp: Current timestamp
            
        Returns:
            Formatted output dictionary
        """
        return {
            'timestamp': timestamp,
            'indicator_name': 'ChaosFractalDimension',
            
            # Primary metrics
            'box_counting_dimension': round(result.box_counting_dimension, 4),
            'correlation_dimension': round(result.correlation_dimension, 4),
            'information_dimension': round(result.information_dimension, 4),
            'chaos_level': round(result.chaos_level, 4),
            'market_regime': result.market_regime,
            'confidence_score': round(result.confidence_score, 4),
            
            # Additional analysis
            'scaling_exponents': [round(exp, 4) for exp in result.scaling_exponents],
            'fractal_complexity': round((result.box_counting_dimension + result.correlation_dimension) / 2, 4),
            'dimension_spread': round(abs(result.box_counting_dimension - result.correlation_dimension), 4),
            
            # Quality metrics
            'error_metrics': {k: round(v, 4) for k, v in result.error_metrics.items()},
            
            # Trading signals
            'chaos_signal': 'HIGH' if result.chaos_level > self.chaos_threshold else 'LOW',
            'regime_stability': 'STABLE' if result.confidence_score > self.confidence_threshold else 'UNSTABLE',
            'multifractal_strength': 'STRONG' if len(result.scaling_exponents) > 3 and np.std(result.scaling_exponents) > 0.3 else 'WEAK'
        }

    def _validate_data(self, data: pd.DataFrame) -> None:
        """
        Validate input data for required columns and quality
        
        Args:
            data: Input DataFrame to validate
            
        Raises:
            IndicatorCalculationError: If data validation fails
        """
        required_columns = ['high', 'low', 'close']
        
        if not all(col in data.columns for col in required_columns):
            raise IndicatorCalculationError(f"Missing required columns: {required_columns}")
        
        if len(data) < 50:
            raise IndicatorCalculationError("Insufficient data: minimum 50 periods required")
        
        # Check for invalid values
        for col in required_columns:
            if data[col].isnull().any():
                raise IndicatorCalculationError(f"Null values found in {col}")
            if (data[col] <= 0).any():
                raise IndicatorCalculationError(f"Non-positive values found in {col}")

    def _prepare_price_series(self, data: pd.DataFrame) -> pd.Series:
        """
        Prepare price series for analysis
        
        Args:
            data: Input OHLCV data
            
        Returns:
            Processed price series
        """
        # Use typical price for more stable analysis
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        return typical_price

    def _create_default_result(self) -> Dict[str, Any]:
        """
        Create default result for insufficient data cases
        
        Returns:
            Default result dictionary
        """
        return {
            'timestamp': pd.Timestamp.now(),
            'indicator_name': 'ChaosFractalDimension',
            'box_counting_dimension': 1.5,
            'correlation_dimension': 1.5,
            'information_dimension': 1.5,
            'chaos_level': 0.5,
            'market_regime': 'INSUFFICIENT_DATA',
            'confidence_score': 0.0,
            'scaling_exponents': [1.0],
            'fractal_complexity': 1.5,
            'dimension_spread': 0.0,
            'error_metrics': {
                'data_quality': 0.0,
                'stationarity_score': 0.0,
                'noise_level': 1.0,
                'autocorrelation': 0.0
            },
            'chaos_signal': 'UNKNOWN',
            'regime_stability': 'UNSTABLE',
            'multifractal_strength': 'UNKNOWN'
        }

    def get_required_columns(self) -> List[str]:
        """
        Get list of required data columns
        
        Returns:
            List of required column names
        """
        return ['high', 'low', 'close']

    def get_indicator_name(self) -> str:
        """
        Get the indicator name
        
        Returns:
            Indicator name string
        """
        return "ChaosFractalDimension"