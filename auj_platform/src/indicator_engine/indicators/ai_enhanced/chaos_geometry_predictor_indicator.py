"""
Chaos Geometry Predictor - Advanced Nonlinear Dynamics Trading System
===================================================================

Implements chaos theory, fractal geometry, and nonlinear dynamics for 
market prediction using phase space reconstruction and attractor analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from scipy.spatial.distance import pdist
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

from ..base.standard_indicator import StandardIndicatorInterface, DataRequirement, DataType, SignalType


class ChaosGeometryPredictor(StandardIndicatorInterface):
    """
    AI-Enhanced Chaos Geometry Predictor for market dynamics.
    
    Features:
    - Phase space reconstruction using time delay embedding
    - Lyapunov exponent calculation for chaos detection
    - Fractal dimension analysis (correlation and box-counting)
    - Strange attractor identification and classification
    - Recurrence quantification analysis (RQA)
    - Nonlinear prediction using chaos theory
    - Market regime transition detection
    - Entropy-based complexity measures
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'embedding_dimension': 5,      # Phase space embedding dimension
            'time_delay': 3,               # Time delay for embedding
            'min_periods': 100,            # Minimum periods for analysis
            'max_lyapunov_iterations': 50, # Maximum iterations for Lyapunov calculation
            'recurrence_threshold': 0.1,   # Recurrence threshold for RQA
            'prediction_horizon': 5,       # Steps ahead for prediction
            'fractal_scales': [2, 4, 8, 16, 32],  # Scales for fractal analysis
            'neighbor_count': 10,          # Neighbors for phase space analysis
            'entropy_bins': 20,            # Bins for entropy calculation
            'regime_sensitivity': 0.05,    # Sensitivity for regime detection
        }
        if parameters:
            default_params.update(parameters)
        
        super().__init__("ChaosGeometryPredictor", default_params)
        
        # Analysis components
        self.phase_space_history = []
        self.attractor_history = []
        self.regime_transitions = []
        
    def get_data_requirements(self) -> DataRequirement:
        return DataRequirement(
            data_type=DataType.OHLCV,
            required_columns=['close', 'high', 'low', 'volume'],
            min_periods=self.parameters['min_periods'],
            lookback_periods=300
        )
    
    def calculate_raw(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate comprehensive chaos geometry analysis.
        """
        try:
            closes = data['close'].values
            highs = data['high'].values
            lows = data['low'].values
            volumes = data['volume'].values
            
            # Calculate log returns for better stationarity
            log_returns = np.diff(np.log(closes + 1e-8))
            
            if len(log_returns) < self.parameters['min_periods']:
                return self._default_result()
            
            # Phase space reconstruction
            phase_space = self._reconstruct_phase_space(log_returns)
            
            # Chaos indicators
            chaos_analysis = self._analyze_chaos_indicators(log_returns, phase_space)
            
            # Fractal analysis
            fractal_analysis = self._analyze_fractal_dimensions(closes, log_returns)
            
            # Attractor analysis
            attractor_analysis = self._analyze_strange_attractors(phase_space)
            
            # Recurrence quantification
            rqa_analysis = self._recurrence_quantification_analysis(phase_space)
            
            # Nonlinear prediction
            prediction_analysis = self._nonlinear_prediction(phase_space, log_returns)
            
            # Market regime analysis
            regime_analysis = self._analyze_market_regimes(chaos_analysis, fractal_analysis, attractor_analysis)
            
            # Entropy and complexity measures
            complexity_analysis = self._analyze_complexity_measures(log_returns, phase_space)
            
            # Signal generation
            signal_strength = self._calculate_signal_strength(
                chaos_analysis, fractal_analysis, attractor_analysis,
                prediction_analysis, regime_analysis, complexity_analysis
            )
            
            return {
                'chaos_analysis': chaos_analysis,
                'fractal_analysis': fractal_analysis,
                'attractor_analysis': attractor_analysis,
                'rqa_analysis': rqa_analysis,
                'prediction_analysis': prediction_analysis,
                'regime_analysis': regime_analysis,
                'complexity_analysis': complexity_analysis,
                'signal_strength': signal_strength,
                'phase_space_dimension': phase_space.shape[1] if len(phase_space.shape) > 1 else 0,
                'chaos_confidence': self._calculate_chaos_confidence(chaos_analysis, fractal_analysis)
            }
            
        except Exception as e:
            raise Exception(f"ChaosGeometryPredictor calculation failed: {str(e)}")
    
    def _default_result(self) -> Dict[str, Any]:
        """Return default result for insufficient data."""
        return {
            'chaos_analysis': {},
            'fractal_analysis': {},
            'attractor_analysis': {},
            'rqa_analysis': {},
            'prediction_analysis': {},
            'regime_analysis': {},
            'complexity_analysis': {},
            'signal_strength': 0.0,
            'phase_space_dimension': 0,
            'chaos_confidence': 0.0
        }
    
    def _reconstruct_phase_space(self, time_series: np.ndarray) -> np.ndarray:
        """Reconstruct phase space using time delay embedding (Takens' theorem)."""
        n = len(time_series)
        m = self.parameters['embedding_dimension']
        tau = self.parameters['time_delay']
        
        # Number of embedded vectors
        N = n - (m - 1) * tau
        
        if N <= 0:
            return np.array([])
        
        # Create embedded matrix
        embedded = np.zeros((N, m))
        
        for i in range(m):
            start_idx = i * tau
            end_idx = start_idx + N
            embedded[:, i] = time_series[start_idx:end_idx]
        
        return embedded
    
    def _analyze_chaos_indicators(self, time_series: np.ndarray, phase_space: np.ndarray) -> Dict[str, Any]:
        """Analyze various chaos indicators."""
        if len(phase_space) == 0:
            return {}
        
        # Lyapunov exponent estimation
        lyapunov_exponent = self._estimate_lyapunov_exponent(phase_space)
        
        # Kolmogorov entropy (approximation)
        kolmogorov_entropy = self._estimate_kolmogorov_entropy(phase_space)
        
        # Correlation dimension
        correlation_dimension = self._estimate_correlation_dimension(phase_space)
        
        # Predictability measure
        predictability = self._estimate_predictability(time_series)
        
        # Chaos classification
        chaos_type = self._classify_chaos_type(lyapunov_exponent, correlation_dimension, predictability)
        
        return {
            'lyapunov_exponent': float(lyapunov_exponent),
            'kolmogorov_entropy': float(kolmogorov_entropy),
            'correlation_dimension': float(correlation_dimension),
            'predictability': float(predictability),
            'chaos_type': chaos_type,
            'is_chaotic': lyapunov_exponent > 0,
            'chaos_strength': float(max(0, lyapunov_exponent))
        }
    
    def _estimate_lyapunov_exponent(self, phase_space: np.ndarray) -> float:
        """Estimate largest Lyapunov exponent using Wolf's algorithm."""
        if len(phase_space) < 10:
            return 0.0
        
        try:
            n_points = min(len(phase_space), 1000)  # Limit for computational efficiency
            phase_subset = phase_space[:n_points]
            
            # Find nearest neighbors
            nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(phase_subset)
            distances, indices = nbrs.kneighbors(phase_subset)
            
            # Track evolution of nearby trajectories
            divergences = []
            max_iterations = min(self.parameters['max_lyapunov_iterations'], n_points - 10)
            
            for i in range(max_iterations):
                if i + 1 >= len(phase_subset):
                    break
                
                # Current point and its nearest neighbor
                current_distance = distances[i, 1]  # Distance to nearest neighbor
                neighbor_idx = indices[i, 1]
                
                if current_distance > 0 and i + 1 < len(distances) and neighbor_idx + 1 < len(phase_subset):
                    # Distance after one time step
                    next_distance = np.linalg.norm(
                        phase_subset[i + 1] - phase_subset[neighbor_idx + 1]
                    )
                    
                    if next_distance > 0:
                        divergence = np.log(next_distance / current_distance)
                        divergences.append(divergence)
            
            if divergences:
                return float(np.mean(divergences))
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _estimate_kolmogorov_entropy(self, phase_space: np.ndarray) -> float:
        """Estimate Kolmogorov entropy (approximation using correlation sum)."""
        if len(phase_space) < 10:
            return 0.0
        
        try:
            n_points = min(len(phase_space), 500)
            phase_subset = phase_space[:n_points]
            
            # Calculate pairwise distances
            distances = pdist(phase_subset)
            
            if len(distances) == 0:
                return 0.0
            
            # Use correlation sum to estimate entropy
            radius = np.median(distances) * 0.1
            correlation_sum = np.sum(distances < radius) / len(distances)
            
            if correlation_sum > 0:
                entropy_estimate = -np.log(correlation_sum) / len(phase_subset[0])
                return float(max(0, entropy_estimate))
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _estimate_correlation_dimension(self, phase_space: np.ndarray) -> float:
        """Estimate correlation dimension using Grassberger-Procaccia algorithm."""
        if len(phase_space) < 20:
            return 0.0
        
        try:
            n_points = min(len(phase_space), 300)
            phase_subset = phase_space[:n_points]
            
            # Calculate correlation sum for different radii
            distances = pdist(phase_subset)
            
            if len(distances) == 0:
                return 0.0
            
            # Use multiple radius values
            min_dist = np.min(distances[distances > 0]) if len(distances[distances > 0]) > 0 else 1e-6
            max_dist = np.max(distances)
            
            if max_dist <= min_dist:
                return 0.0
            
            radii = np.logspace(np.log10(min_dist), np.log10(max_dist), 10)
            correlation_sums = []
            
            for radius in radii:
                correlation_sum = np.sum(distances < radius) / len(distances)
                if correlation_sum > 0:
                    correlation_sums.append(correlation_sum)
                else:
                    correlation_sums.append(1e-10)
            
            # Estimate dimension from slope of log(C) vs log(r)
            log_radii = np.log(radii)
            log_correlation = np.log(correlation_sums)
            
            # Linear regression to find slope
            valid_indices = np.isfinite(log_radii) & np.isfinite(log_correlation)
            if np.sum(valid_indices) > 3:
                slope, _ = np.polyfit(
                    log_radii[valid_indices], 
                    log_correlation[valid_indices], 
                    1
                )
                return float(max(0, slope))
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _estimate_predictability(self, time_series: np.ndarray) -> float:
        """Estimate predictability using approximate entropy."""
        if len(time_series) < 30:
            return 0.0
        
        try:
            # Approximate entropy calculation
            m = 2  # Pattern length
            r = 0.2 * np.std(time_series)  # Tolerance
            
            def _maxdist(xi, xj, m):
                return max([abs(ua - va) for ua, va in zip(xi, xj)])
            
            def _phi(m):
                patterns = np.array([time_series[i:i + m] for i in range(len(time_series) - m + 1)])
                C = np.zeros(len(patterns))
                
                for i in range(len(patterns)):
                    template = patterns[i]
                    matches = sum([1 for pattern in patterns if _maxdist(template, pattern, m) <= r])
                    C[i] = matches / float(len(patterns))
                
                phi = sum([np.log(c) for c in C if c > 0]) / float(len(patterns))
                return phi
            
            approximate_entropy = _phi(m) - _phi(m + 1)
            
            # Convert to predictability (inverse of entropy)
            predictability = 1.0 / (1.0 + abs(approximate_entropy))
            return float(predictability)
            
        except Exception:
            return 0.5  # Default moderate predictability
    
    def _classify_chaos_type(self, lyapunov: float, correlation_dim: float, predictability: float) -> str:
        """Classify the type of chaotic behavior."""
        if lyapunov <= 0:
            if predictability > 0.7:
                return 'periodic'
            else:
                return 'quasi_periodic'
        
        if lyapunov > 0 and lyapunov < 0.1:
            return 'weak_chaos'
        elif lyapunov >= 0.1 and lyapunov < 0.5:
            return 'moderate_chaos'
        elif lyapunov >= 0.5:
            return 'strong_chaos'
        
        return 'unknown'
    
    def _analyze_fractal_dimensions(self, prices: np.ndarray, returns: np.ndarray) -> Dict[str, Any]:
        """Analyze fractal dimensions using multiple methods."""
        if len(prices) < 50:
            return {}
        
        # Box-counting dimension
        box_dimension = self._box_counting_dimension(prices)
        
        # Hurst exponent
        hurst_exponent = self._calculate_hurst_exponent(returns)
        
        # Fractal dimension from Hurst
        fractal_dimension = 2 - hurst_exponent
        
        # Multi-fractal analysis
        multifractal_spectrum = self._multifractal_analysis(returns)
        
        # Persistence analysis
        persistence = self._analyze_persistence(returns)
        
        return {
            'box_dimension': float(box_dimension),
            'hurst_exponent': float(hurst_exponent),
            'fractal_dimension': float(fractal_dimension),
            'multifractal_spectrum': multifractal_spectrum,
            'persistence': persistence,
            'market_efficiency': float(abs(hurst_exponent - 0.5)),  # Deviation from random walk
            'trend_strength': float(max(0, hurst_exponent - 0.5) * 2)  # Trending vs mean-reverting
        }
    
    def _box_counting_dimension(self, prices: np.ndarray) -> float:
        """Calculate box-counting fractal dimension."""
        try:
            # Normalize prices to [0, 1]
            normalized_prices = (prices - np.min(prices)) / (np.max(prices) - np.min(prices) + 1e-8)
            
            # Create grid and count boxes
            scales = self.parameters['fractal_scales']
            box_counts = []
            
            for scale in scales:
                if scale >= len(prices):
                    continue
                
                # Grid size
                grid_size = 1.0 / scale
                
                # Count occupied boxes
                x_bins = np.floor(np.arange(len(normalized_prices)) * scale / len(normalized_prices))
                y_bins = np.floor(normalized_prices * scale)
                
                # Unique boxes
                boxes = set(zip(x_bins, y_bins))
                box_counts.append(len(boxes))
            
            if len(box_counts) < 3:
                return 1.0
            
            # Linear regression on log scale
            log_scales = np.log(scales[:len(box_counts)])
            log_counts = np.log(box_counts)
            
            # Calculate slope (negative of fractal dimension)
            slope, _ = np.polyfit(log_scales, log_counts, 1)
            
            return float(-slope)
            
        except Exception:
            return 1.0
    
    def _calculate_hurst_exponent(self, returns: np.ndarray) -> float:
        """Calculate Hurst exponent using R/S analysis."""
        try:
            n = len(returns)
            if n < 20:
                return 0.5
            
            # Calculate R/S statistic for different lags
            lags = np.logspace(1, np.log10(n//4), 10).astype(int)
            rs_values = []
            
            for lag in lags:
                if lag >= n:
                    continue
                
                # Divide series into chunks
                chunks = n // lag
                rs_chunk = []
                
                for i in range(chunks):
                    start_idx = i * lag
                    end_idx = (i + 1) * lag
                    chunk = returns[start_idx:end_idx]
                    
                    if len(chunk) < 2:
                        continue
                    
                    # Calculate mean
                    mean_chunk = np.mean(chunk)
                    
                    # Calculate deviations from mean
                    deviations = chunk - mean_chunk
                    
                    # Calculate cumulative deviations
                    cumulative_deviations = np.cumsum(deviations)
                    
                    # Calculate range
                    R = np.max(cumulative_deviations) - np.min(cumulative_deviations)
                    
                    # Calculate standard deviation
                    S = np.std(chunk)
                    
                    # R/S ratio
                    if S > 0:
                        rs_chunk.append(R / S)
                
                if rs_chunk:
                    rs_values.append(np.mean(rs_chunk))
            
            if len(rs_values) < 3:
                return 0.5
            
            # Linear regression on log-log plot
            log_lags = np.log(lags[:len(rs_values)])
            log_rs = np.log(rs_values)
            
            # Hurst exponent is the slope
            hurst, _ = np.polyfit(log_lags, log_rs, 1)
            
            # Clamp to reasonable range
            return float(np.clip(hurst, 0.0, 1.0))
            
        except Exception:
            return 0.5
    
    def _multifractal_analysis(self, returns: np.ndarray) -> Dict[str, Any]:
        """Perform multifractal analysis."""
        try:
            if len(returns) < 100:
                return {}
            
            # Multifractal detrended fluctuation analysis (simplified)
            q_values = np.arange(-2, 3, 0.5)  # Moment orders
            
            # Scale range
            scales = np.logspace(1, np.log10(len(returns)//4), 15).astype(int)
            
            fluctuation_functions = {}
            
            for q in q_values:
                fluctuations = []
                
                for scale in scales:
                    if scale >= len(returns):
                        continue
                    
                    # Calculate local fluctuations
                    n_segments = len(returns) // scale
                    segment_fluctuations = []
                    
                    for i in range(n_segments):
                        start_idx = i * scale
                        end_idx = (i + 1) * scale
                        segment = returns[start_idx:end_idx]
                        
                        if len(segment) > 1:
                            # Detrend segment
                            x = np.arange(len(segment))
                            coeffs = np.polyfit(x, segment, 1)
                            trend = np.polyval(coeffs, x)
                            detrended = segment - trend
                            
                            # Calculate fluctuation
                            fluctuation = np.sqrt(np.mean(detrended**2))
                            segment_fluctuations.append(fluctuation)
                    
                    if segment_fluctuations:
                        if q == 0:
                            # Special case for q=0 (logarithmic average)
                            valid_fluctuations = [f for f in segment_fluctuations if f > 0]
                            if valid_fluctuations:
                                avg_fluctuation = np.exp(np.mean(np.log(valid_fluctuations)))
                            else:
                                avg_fluctuation = 0
                        else:
                            # General case
                            if q > 0:
                                avg_fluctuation = (np.mean([f**q for f in segment_fluctuations]))**(1/q)
                            else:
                                # For negative q, use only non-zero fluctuations
                                valid_fluctuations = [f for f in segment_fluctuations if f > 0]
                                if valid_fluctuations:
                                    avg_fluctuation = (np.mean([f**q for f in valid_fluctuations]))**(1/q)
                                else:
                                    avg_fluctuation = 0
                        
                        fluctuations.append(avg_fluctuation)
                
                fluctuation_functions[q] = fluctuations
            
            # Calculate generalized Hurst exponents
            hurst_exponents = {}
            for q, fluctuations in fluctuation_functions.items():
                if len(fluctuations) >= 3:
                    valid_scales = scales[:len(fluctuations)]
                    log_scales = np.log(valid_scales)
                    log_fluctuations = np.log([f for f in fluctuations if f > 0])
                    
                    if len(log_fluctuations) >= 3:
                        slope, _ = np.polyfit(log_scales[:len(log_fluctuations)], log_fluctuations, 1)
                        hurst_exponents[q] = slope
            
            # Multifractal spectrum
            spectrum_width = 0
            if len(hurst_exponents) > 1:
                h_values = list(hurst_exponents.values())
                spectrum_width = max(h_values) - min(h_values)
            
            return {
                'hurst_exponents': hurst_exponents,
                'spectrum_width': float(spectrum_width),
                'is_multifractal': spectrum_width > 0.1,
                'complexity_measure': float(spectrum_width)
            }
            
        except Exception:
            return {}
    
    def _analyze_persistence(self, returns: np.ndarray) -> Dict[str, Any]:
        """Analyze persistence and anti-persistence patterns."""
        if len(returns) < 20:
            return {}
        
        try:
            # Calculate autocorrelation at different lags
            max_lag = min(20, len(returns) // 4)
            autocorrelations = []
            
            for lag in range(1, max_lag + 1):
                if lag >= len(returns):
                    break
                
                # Calculate autocorrelation
                corr = np.corrcoef(returns[:-lag], returns[lag:])[0, 1]
                if not np.isnan(corr):
                    autocorrelations.append(corr)
            
            if not autocorrelations:
                return {}
            
            # Persistence measures
            first_lag_corr = autocorrelations[0] if autocorrelations else 0
            avg_correlation = np.mean(autocorrelations)
            
            # Classify persistence
            if first_lag_corr > 0.1:
                persistence_type = 'persistent'
            elif first_lag_corr < -0.1:
                persistence_type = 'anti_persistent'
            else:
                persistence_type = 'random_walk'
            
            return {
                'first_lag_correlation': float(first_lag_corr),
                'average_correlation': float(avg_correlation),
                'persistence_type': persistence_type,
                'persistence_strength': float(abs(first_lag_corr)),
                'autocorrelations': [float(x) for x in autocorrelations[:5]]  # First 5 lags
            }
            
        except Exception:
            return {}
    
    def _analyze_strange_attractors(self, phase_space: np.ndarray) -> Dict[str, Any]:
        """Analyze strange attractor properties."""
        if len(phase_space) < 20:
            return {}
        
        try:
            # PCA analysis for dimensionality
            pca = PCA()
            pca.fit(phase_space)
            
            # Effective dimension (number of components explaining 95% variance)
            cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
            effective_dim = np.argmax(cumulative_variance >= 0.95) + 1
            
            # Attractor shape analysis
            shape_analysis = self._analyze_attractor_shape(phase_space)
            
            # Density analysis
            density_analysis = self._analyze_attractor_density(phase_space)
            
            # Stability analysis
            stability_analysis = self._analyze_attractor_stability(phase_space)
            
            return {
                'effective_dimension': int(effective_dim),
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'total_variance_explained': float(cumulative_variance[-1]),
                'shape_analysis': shape_analysis,
                'density_analysis': density_analysis,
                'stability_analysis': stability_analysis,
                'attractor_type': self._classify_attractor_type(effective_dim, shape_analysis, stability_analysis)
            }
            
        except Exception:
            return {}
    
    def _analyze_attractor_shape(self, phase_space: np.ndarray) -> Dict[str, Any]:
        """Analyze the geometric shape of the attractor."""
        try:
            # Calculate moments for shape characterization
            center = np.mean(phase_space, axis=0)
            centered_data = phase_space - center
            
            # Calculate covariance matrix
            cov_matrix = np.cov(centered_data.T)
            
            # Eigenvalue analysis
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            eigenvalues = np.sort(eigenvalues)[::-1]  # Sort in descending order
            
            # Shape measures
            if len(eigenvalues) >= 2:
                eccentricity = np.sqrt(1 - (eigenvalues[1] / eigenvalues[0])) if eigenvalues[0] > 0 else 0
                flatness = eigenvalues[-1] / eigenvalues[0] if eigenvalues[0] > 0 else 0
            else:
                eccentricity = 0
                flatness = 0
            
            return {
                'eccentricity': float(eccentricity),
                'flatness': float(flatness),
                'eigenvalues': eigenvalues.tolist(),
                'volume': float(np.prod(eigenvalues)**0.5) if len(eigenvalues) > 0 else 0
            }
            
        except Exception:
            return {}
    
    def _analyze_attractor_density(self, phase_space: np.ndarray) -> Dict[str, Any]:
        """Analyze density distribution in the attractor."""
        try:
            # Use k-nearest neighbors to estimate density
            k = min(10, len(phase_space) // 2)
            if k < 2:
                return {}
            
            nbrs = NearestNeighbors(n_neighbors=k).fit(phase_space)
            distances, _ = nbrs.kneighbors(phase_space)
            
            # Average distance to k-th nearest neighbor
            kth_distances = distances[:, -1]
            
            # Density measures
            avg_density = 1.0 / (np.mean(kth_distances) + 1e-8)
            density_variance = np.var(1.0 / (kth_distances + 1e-8))
            
            # Identify high and low density regions
            density_threshold = np.percentile(kth_distances, 25)  # 25th percentile
            high_density_fraction = np.sum(kth_distances < density_threshold) / len(kth_distances)
            
            return {
                'average_density': float(avg_density),
                'density_variance': float(density_variance),
                'high_density_fraction': float(high_density_fraction),
                'density_homogeneity': float(1.0 / (1.0 + density_variance))
            }
            
        except Exception:
            return {}
    
    def _analyze_attractor_stability(self, phase_space: np.ndarray) -> Dict[str, Any]:
        """Analyze stability properties of the attractor."""
        try:
            if len(phase_space) < 30:
                return {}
            
            # Split data into segments for stability analysis
            n_segments = 3
            segment_size = len(phase_space) // n_segments
            
            segment_centers = []
            segment_spreads = []
            
            for i in range(n_segments):
                start_idx = i * segment_size
                end_idx = (i + 1) * segment_size if i < n_segments - 1 else len(phase_space)
                segment = phase_space[start_idx:end_idx]
                
                if len(segment) > 0:
                    center = np.mean(segment, axis=0)
                    spread = np.mean(np.linalg.norm(segment - center, axis=1))
                    
                    segment_centers.append(center)
                    segment_spreads.append(spread)
            
            if len(segment_centers) < 2:
                return {}
            
            # Calculate center drift
            center_distances = []
            for i in range(1, len(segment_centers)):
                distance = np.linalg.norm(segment_centers[i] - segment_centers[i-1])
                center_distances.append(distance)
            
            avg_center_drift = np.mean(center_distances) if center_distances else 0
            
            # Calculate spread stability
            spread_variance = np.var(segment_spreads) if len(segment_spreads) > 1 else 0
            avg_spread = np.mean(segment_spreads)
            
            # Stability score
            stability_score = 1.0 / (1.0 + avg_center_drift + spread_variance)
            
            return {
                'center_drift': float(avg_center_drift),
                'spread_variance': float(spread_variance),
                'average_spread': float(avg_spread),
                'stability_score': float(stability_score),
                'is_stable': stability_score > 0.7
            }
            
        except Exception:
            return {}
    
    def _classify_attractor_type(self, effective_dim: int, shape_analysis: Dict, stability_analysis: Dict) -> str:
        """Classify the type of attractor."""
        if effective_dim <= 1:
            return 'point_attractor'
        elif effective_dim == 2:
            if shape_analysis.get('eccentricity', 0) > 0.8:
                return 'limit_cycle'
            else:
                return 'quasi_periodic'
        elif effective_dim >= 3:
            if stability_analysis.get('is_stable', False):
                return 'strange_attractor_stable'
            else:
                return 'strange_attractor_chaotic'
        else:
            return 'unknown'
    
    def _recurrence_quantification_analysis(self, phase_space: np.ndarray) -> Dict[str, Any]:
        """Perform Recurrence Quantification Analysis (RQA)."""
        if len(phase_space) < 50:
            return {}
        
        try:
            # Subsample for computational efficiency
            n_points = min(len(phase_space), 200)
            phase_subset = phase_space[:n_points]
            
            # Calculate recurrence matrix
            threshold = self.parameters['recurrence_threshold']
            recurrence_matrix = self._calculate_recurrence_matrix(phase_subset, threshold)
            
            # RQA measures
            rqa_measures = self._calculate_rqa_measures(recurrence_matrix)
            
            return rqa_measures
            
        except Exception:
            return {}
    
    def _calculate_recurrence_matrix(self, phase_space: np.ndarray, threshold: float) -> np.ndarray:
        """Calculate recurrence matrix."""
        n = len(phase_space)
        recurrence_matrix = np.zeros((n, n))
        
        # Calculate pairwise distances
        for i in range(n):
            for j in range(i, n):
                distance = np.linalg.norm(phase_space[i] - phase_space[j])
                if distance < threshold:
                    recurrence_matrix[i, j] = 1
                    recurrence_matrix[j, i] = 1
        
        return recurrence_matrix
    
    def _calculate_rqa_measures(self, recurrence_matrix: np.ndarray) -> Dict[str, Any]:
        """Calculate RQA measures from recurrence matrix."""
        n = len(recurrence_matrix)
        
        # Recurrence rate
        recurrence_rate = np.sum(recurrence_matrix) / (n * n)
        
        # Determinism (percentage of recurrent points forming diagonal lines)
        diagonal_lines = self._find_diagonal_lines(recurrence_matrix)
        total_diagonal_points = sum(len(line) for line in diagonal_lines if len(line) >= 2)
        determinism = total_diagonal_points / max(np.sum(recurrence_matrix), 1)
        
        # Average diagonal line length
        diagonal_lengths = [len(line) for line in diagonal_lines if len(line) >= 2]
        avg_diagonal_length = np.mean(diagonal_lengths) if diagonal_lengths else 0
        
        # Laminarity (percentage of recurrent points forming vertical lines)
        vertical_lines = self._find_vertical_lines(recurrence_matrix)
        total_vertical_points = sum(len(line) for line in vertical_lines if len(line) >= 2)
        laminarity = total_vertical_points / max(np.sum(recurrence_matrix), 1)
        
        return {
            'recurrence_rate': float(recurrence_rate),
            'determinism': float(determinism),
            'average_diagonal_length': float(avg_diagonal_length),
            'laminarity': float(laminarity),
            'complexity': float(determinism * (1 - laminarity))  # Measure of system complexity
        }
    
    def _find_diagonal_lines(self, matrix: np.ndarray) -> List[List[Tuple[int, int]]]:
        """Find diagonal lines in recurrence matrix."""
        n = len(matrix)
        lines = []
        visited = set()
        
        for i in range(n):
            for j in range(n):
                if matrix[i, j] == 1 and (i, j) not in visited:
                    # Start a new diagonal line
                    line = [(i, j)]
                    visited.add((i, j))
                    
                    # Extend diagonally
                    k = 1
                    while i + k < n and j + k < n and matrix[i + k, j + k] == 1:
                        line.append((i + k, j + k))
                        visited.add((i + k, j + k))
                        k += 1
                    
                    if len(line) > 1:
                        lines.append(line)
        
        return lines
    
    def _find_vertical_lines(self, matrix: np.ndarray) -> List[List[Tuple[int, int]]]:
        """Find vertical lines in recurrence matrix."""
        n = len(matrix)
        lines = []
        visited = set()
        
        for j in range(n):
            for i in range(n):
                if matrix[i, j] == 1 and (i, j) not in visited:
                    # Start a new vertical line
                    line = [(i, j)]
                    visited.add((i, j))
                    
                    # Extend vertically
                    k = 1
                    while i + k < n and matrix[i + k, j] == 1:
                        line.append((i + k, j))
                        visited.add((i + k, j))
                        k += 1
                    
                    if len(line) > 1:
                        lines.append(line)
        
        return lines
    
    def _nonlinear_prediction(self, phase_space: np.ndarray, time_series: np.ndarray) -> Dict[str, Any]:
        """Perform nonlinear prediction using chaos theory."""
        if len(phase_space) < 20:
            return {}
        
        try:
            horizon = self.parameters['prediction_horizon']
            n_neighbors = min(self.parameters['neighbor_count'], len(phase_space) // 4)
            
            if n_neighbors < 2:
                return {}
            
            # Use last point for prediction
            query_point = phase_space[-1].reshape(1, -1)
            
            # Find nearest neighbors
            nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(phase_space[:-horizon])
            distances, indices = nbrs.kneighbors(query_point)
            
            # Predict based on evolution of nearest neighbors
            predictions = []
            
            for i, neighbor_idx in enumerate(indices[0]):
                if neighbor_idx + horizon < len(time_series):
                    # Weight by inverse distance
                    weight = 1.0 / (distances[0][i] + 1e-8)
                    future_value = time_series[neighbor_idx + horizon]
                    predictions.append((future_value, weight))
            
            if not predictions:
                return {}
            
            # Weighted average prediction
            total_weight = sum(weight for _, weight in predictions)
            if total_weight > 0:
                predicted_return = sum(value * weight for value, weight in predictions) / total_weight
            else:
                predicted_return = 0
            
            # Prediction confidence based on neighbor agreement
            prediction_values = [value for value, _ in predictions]
            prediction_std = np.std(prediction_values) if len(prediction_values) > 1 else 0
            confidence = 1.0 / (1.0 + prediction_std)
            
            return {
                'predicted_return': float(predicted_return),
                'prediction_confidence': float(confidence),
                'prediction_std': float(prediction_std),
                'neighbor_count': len(predictions),
                'prediction_horizon': horizon
            }
            
        except Exception:
            return {}
    
    def _analyze_market_regimes(self, chaos_analysis: Dict, fractal_analysis: Dict, 
                               attractor_analysis: Dict) -> Dict[str, Any]:
        """Analyze market regime transitions."""
        try:
            # Regime indicators
            regime_indicators = []
            
            # Chaos-based regime
            if chaos_analysis:
                chaos_type = chaos_analysis.get('chaos_type', 'unknown')
                lyapunov = chaos_analysis.get('lyapunov_exponent', 0)
                
                if chaos_type in ['periodic', 'quasi_periodic']:
                    regime_indicators.append('stable')
                elif chaos_type in ['weak_chaos', 'moderate_chaos']:
                    regime_indicators.append('transitional')
                elif chaos_type == 'strong_chaos':
                    regime_indicators.append('chaotic')
            
            # Fractal-based regime
            if fractal_analysis:
                hurst = fractal_analysis.get('hurst_exponent', 0.5)
                
                if hurst > 0.6:
                    regime_indicators.append('trending')
                elif hurst < 0.4:
                    regime_indicators.append('mean_reverting')
                else:
                    regime_indicators.append('random_walk')
            
            # Attractor-based regime
            if attractor_analysis:
                attractor_type = attractor_analysis.get('attractor_type', 'unknown')
                
                if 'stable' in attractor_type:
                    regime_indicators.append('stable')
                elif 'chaotic' in attractor_type:
                    regime_indicators.append('chaotic')
            
            # Determine dominant regime
            if regime_indicators:
                regime_counts = {}
                for regime in regime_indicators:
                    regime_counts[regime] = regime_counts.get(regime, 0) + 1
                
                dominant_regime = max(regime_counts, key=regime_counts.get)
                regime_confidence = regime_counts[dominant_regime] / len(regime_indicators)
            else:
                dominant_regime = 'unknown'
                regime_confidence = 0
            
            # Regime transition probability
            transition_probability = self._estimate_regime_transition_probability(
                chaos_analysis, fractal_analysis
            )
            
            return {
                'dominant_regime': dominant_regime,
                'regime_confidence': float(regime_confidence),
                'regime_indicators': regime_indicators,
                'transition_probability': float(transition_probability),
                'regime_stability': float(1 - transition_probability)
            }
            
        except Exception:
            return {}
    
    def _estimate_regime_transition_probability(self, chaos_analysis: Dict, fractal_analysis: Dict) -> float:
        """Estimate probability of regime transition."""
        try:
            transition_factors = []
            
            # Chaos factor
            if chaos_analysis:
                lyapunov = chaos_analysis.get('lyapunov_exponent', 0)
                # Higher Lyapunov exponent suggests higher probability of transition
                transition_factors.append(min(abs(lyapunov), 1.0))
            
            # Fractal factor
            if fractal_analysis:
                hurst = fractal_analysis.get('hurst_exponent', 0.5)
                # Deviation from 0.5 suggests stability, closer to 0.5 suggests potential transition
                hurst_stability = abs(hurst - 0.5)
                transition_factors.append(1.0 - (hurst_stability * 2))
            
            if transition_factors:
                return np.mean(transition_factors)
            
            return 0.5  # Default neutral probability
            
        except Exception:
            return 0.5
    
    def _analyze_complexity_measures(self, time_series: np.ndarray, phase_space: np.ndarray) -> Dict[str, Any]:
        """Analyze various complexity measures."""
        try:
            measures = {}
            
            # Sample entropy
            measures['sample_entropy'] = self._calculate_sample_entropy(time_series)
            
            # Permutation entropy
            measures['permutation_entropy'] = self._calculate_permutation_entropy(time_series)
            
            # Phase space entropy
            if len(phase_space) > 0:
                measures['phase_space_entropy'] = self._calculate_phase_space_entropy(phase_space)
            
            # Complexity index (combination of measures)
            entropy_values = [v for v in measures.values() if not np.isnan(v) and v > 0]
            if entropy_values:
                measures['complexity_index'] = np.mean(entropy_values)
            else:
                measures['complexity_index'] = 0.5
            
            return measures
            
        except Exception:
            return {}
    
    def _calculate_sample_entropy(self, time_series: np.ndarray) -> float:
        """Calculate sample entropy."""
        try:
            if len(time_series) < 20:
                return 0.5
            
            m = 2  # Pattern length
            r = 0.2 * np.std(time_series)  # Tolerance
            
            def _maxdist(xi, xj):
                return max([abs(ua - va) for ua, va in zip(xi, xj)])
            
            def _phi(m):
                patterns = [time_series[i:i + m] for i in range(len(time_series) - m + 1)]
                C = []
                
                for i, pattern_i in enumerate(patterns):
                    matches = 0
                    for j, pattern_j in enumerate(patterns):
                        if i != j and _maxdist(pattern_i, pattern_j) <= r:
                            matches += 1
                    C.append(matches)
                
                return sum(C)
            
            phi_m = _phi(m)
            phi_m1 = _phi(m + 1)
            
            if phi_m > 0 and phi_m1 > 0:
                return -np.log(phi_m1 / phi_m)
            
            return 0.5
            
        except Exception:
            return 0.5
    
    def _calculate_permutation_entropy(self, time_series: np.ndarray) -> float:
        """Calculate permutation entropy."""
        try:
            if len(time_series) < 10:
                return 0.5
            
            m = 3  # Embedding dimension
            
            # Create permutation patterns
            patterns = []
            for i in range(len(time_series) - m + 1):
                segment = time_series[i:i + m]
                # Get permutation pattern
                pattern = tuple(np.argsort(segment))
                patterns.append(pattern)
            
            if not patterns:
                return 0.5
            
            # Calculate relative frequencies
            unique_patterns, counts = np.unique(patterns, return_counts=True, axis=0)
            probabilities = counts / len(patterns)
            
            # Calculate entropy
            entropy_value = -np.sum(probabilities * np.log(probabilities))
            
            # Normalize by maximum possible entropy
            max_entropy = np.log(np.math.factorial(m))
            
            return entropy_value / max_entropy if max_entropy > 0 else 0.5
            
        except Exception:
            return 0.5
    
    def _calculate_phase_space_entropy(self, phase_space: np.ndarray) -> float:
        """Calculate entropy in phase space."""
        try:
            if len(phase_space) < 20:
                return 0.5
            
            # Discretize phase space
            n_bins = self.parameters['entropy_bins']
            
            # Find bounds for each dimension
            mins = np.min(phase_space, axis=0)
            maxs = np.max(phase_space, axis=0)
            
            # Create grid
            bin_edges = []
            for i in range(phase_space.shape[1]):
                edges = np.linspace(mins[i], maxs[i], n_bins + 1)
                bin_edges.append(edges)
            
            # Assign points to bins
            bin_indices = []
            for point in phase_space:
                bin_index = []
                for i, coord in enumerate(point):
                    # Find which bin this coordinate falls into
                    bin_idx = np.digitize(coord, bin_edges[i]) - 1
                    bin_idx = np.clip(bin_idx, 0, n_bins - 1)  # Ensure within bounds
                    bin_index.append(bin_idx)
                bin_indices.append(tuple(bin_index))
            
            # Calculate entropy
            unique_bins, counts = np.unique(bin_indices, return_counts=True, axis=0)
            probabilities = counts / len(bin_indices)
            
            entropy_value = -np.sum(probabilities * np.log(probabilities))
            
            # Normalize by maximum possible entropy
            max_entropy = np.log(n_bins ** phase_space.shape[1])
            
            return entropy_value / max_entropy if max_entropy > 0 else 0.5
            
        except Exception:
            return 0.5
    
    def _calculate_chaos_confidence(self, chaos_analysis: Dict, fractal_analysis: Dict) -> float:
        """Calculate confidence in chaos analysis."""
        confidence_factors = []
        
        # Lyapunov exponent confidence
        if chaos_analysis and 'lyapunov_exponent' in chaos_analysis:
            lyapunov = abs(chaos_analysis['lyapunov_exponent'])
            # Higher absolute value suggests more confidence
            confidence_factors.append(min(lyapunov, 1.0))
        
        # Fractal dimension confidence
        if fractal_analysis and 'fractal_dimension' in fractal_analysis:
            fractal_dim = fractal_analysis['fractal_dimension']
            # Deviation from integer values suggests fractal nature
            deviation_from_integer = min(abs(fractal_dim - round(fractal_dim)), 0.5) * 2
            confidence_factors.append(deviation_from_integer)
        
        # Hurst exponent confidence
        if fractal_analysis and 'hurst_exponent' in fractal_analysis:
            hurst = fractal_analysis['hurst_exponent']
            # Deviation from 0.5 (random walk) suggests structure
            hurst_confidence = abs(hurst - 0.5) * 2
            confidence_factors.append(hurst_confidence)
        
        if confidence_factors:
            return float(np.mean(confidence_factors))
        
        return 0.0
    
    def _calculate_signal_strength(self, chaos_analysis: Dict, fractal_analysis: Dict,
                                 attractor_analysis: Dict, prediction_analysis: Dict,
                                 regime_analysis: Dict, complexity_analysis: Dict) -> float:
        """Calculate trading signal strength based on chaos geometry analysis."""
        signal_components = []
        
        # Chaos-based signal
        if chaos_analysis:
            chaos_type = chaos_analysis.get('chaos_type', 'unknown')
            lyapunov = chaos_analysis.get('lyapunov_exponent', 0)
            
            if chaos_type == 'periodic':
                # Periodic behavior suggests continuation
                signal_components.append(0.3)
            elif chaos_type in ['weak_chaos', 'moderate_chaos']:
                # Moderate chaos can be traded
                signal_components.append(0.1 * np.sign(lyapunov))
            elif chaos_type == 'strong_chaos':
                # Strong chaos suggests caution (contrarian)
                signal_components.append(-0.2)
        
        # Fractal-based signal
        if fractal_analysis:
            hurst = fractal_analysis.get('hurst_exponent', 0.5)
            trend_strength = fractal_analysis.get('trend_strength', 0)
            
            if hurst > 0.6:
                # Trending market - follow momentum
                signal_components.append(trend_strength * 0.4)
            elif hurst < 0.4:
                # Mean-reverting market - contrarian signal
                signal_components.append(-trend_strength * 0.3)
        
        # Prediction-based signal
        if prediction_analysis:
            predicted_return = prediction_analysis.get('predicted_return', 0)
            confidence = prediction_analysis.get('prediction_confidence', 0)
            
            # Use prediction with confidence weighting
            prediction_signal = predicted_return * confidence * 2  # Scale up
            signal_components.append(np.clip(prediction_signal, -0.5, 0.5))
        
        # Regime-based signal
        if regime_analysis:
            regime = regime_analysis.get('dominant_regime', 'unknown')
            regime_confidence = regime_analysis.get('regime_confidence', 0)
            transition_prob = regime_analysis.get('transition_probability', 0.5)
            
            if regime == 'stable' and transition_prob < 0.3:
                # Stable regime with low transition probability
                signal_components.append(0.2 * regime_confidence)
            elif regime == 'chaotic' and transition_prob > 0.7:
                # Chaotic regime with high transition probability
                signal_components.append(-0.3 * regime_confidence)
        
        # Complexity-based signal
        if complexity_analysis:
            complexity_index = complexity_analysis.get('complexity_index', 0.5)
            
            # Moderate complexity is tradeable, extreme complexity suggests caution
            if 0.3 <= complexity_index <= 0.7:
                signal_components.append(0.1)
            else:
                signal_components.append(-0.1)
        
        # Combine signals
        if signal_components:
            total_signal = np.sum(signal_components)
            return float(np.clip(total_signal, -1, 1))
        
        return 0.0
    
    def _generate_signal(self, value: Dict[str, Any], data: pd.DataFrame) -> Tuple[Optional[SignalType], float]:
        """Generate trading signal based on chaos geometry analysis."""
        signal_strength = value['signal_strength']
        chaos_confidence = value['chaos_confidence']
        
        # Use chaos confidence to adjust signal strength
        confidence = min(abs(signal_strength) * (1 + chaos_confidence), 1.0)
        
        # Apply minimum confidence threshold
        if confidence < 0.3:
            return SignalType.NEUTRAL, confidence
        
        if signal_strength > 0.6:
            return SignalType.STRONG_BUY, confidence
        elif signal_strength > 0.3:
            return SignalType.BUY, confidence
        elif signal_strength < -0.6:
            return SignalType.STRONG_SELL, confidence
        elif signal_strength < -0.3:
            return SignalType.SELL, confidence
        else:
            return SignalType.NEUTRAL, confidence
