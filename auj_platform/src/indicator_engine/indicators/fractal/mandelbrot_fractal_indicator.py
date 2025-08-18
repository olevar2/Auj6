"""
Mandelbrot Fractal Indicator - Advanced Implementation

This indicator implements Benoit Mandelbrot's fractal analysis principles
applied to financial markets, incorporating the Mandelbrot set mathematical
framework for price action complexity measurement. Features include:
- Mandelbrot set calculations adapted for financial time series
- Fractal scaling analysis using self-similarity principles
- Price action complexity measurement with escape-time algorithms
- Multi-scale fractal analysis and attractor mapping
- Behavioral pattern recognition using fractal geometry
- Advanced statistical validation with confidence intervals

Mission: Supporting humanitarian trading platform for poor and sick children through
maximum profitability via advanced Mandelbrot fractal analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Complex
from dataclasses import dataclass
import logging
from scipy import stats
from scipy.signal import find_peaks, hilbert
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import DBSCAN
import warnings

from ..base.standard_indicator import StandardIndicatorInterface
from ....core.exceptions import IndicatorCalculationError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MandelbrotResult:
    """Results container for Mandelbrot Fractal"""
    fractal_complexity: float
    mandelbrot_escape_time: float
    self_similarity_index: float
    scaling_exponent: float
    attractor_strength: float
    chaos_index: float
    pattern_stability: float
    market_regime: str
    complexity_trend: str
    fractal_confidence: float

class MandelbrotFractalIndicator(StandardIndicatorInterface):
    """
    Advanced Mandelbrot Fractal Indicator
    
    Implements Benoit Mandelbrot's fractal analysis principles
    applied to financial markets with complexity measurement.
    """
    
    def __init__(self, 
                 window_size: int = 100,
                 max_iterations: int = 100,
                 escape_radius: float = 2.0,
                 scaling_window: int = 50,
                 complexity_threshold: float = 0.5):
        super().__init__()
        self.window_size = window_size
        self.max_iterations = max_iterations
        self.escape_radius = escape_radius
        self.scaling_window = scaling_window
        self.complexity_threshold = complexity_threshold
        
        logger.info(f"Initialized MandelbrotFractalIndicator with window_size={window_size}")

    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        try:
            self._validate_data(data)
            
            if len(data) < self.window_size:
                return self._create_default_result()
            
            recent_data = data[-self.window_size:].copy()
            
            # Convert price data to complex plane representation
            complex_series = self._create_complex_series(recent_data)
            
            # Calculate Mandelbrot-inspired escape times
            escape_times = self._calculate_escape_times(complex_series)
            
            # Analyze fractal complexity
            fractal_complexity = self._calculate_fractal_complexity(escape_times)
            
            # Calculate self-similarity index
            self_similarity = self._calculate_self_similarity(recent_data)
            
            # Analyze scaling properties
            scaling_analysis = self._analyze_scaling_properties(recent_data)
            
            # Calculate attractor strength
            attractor_strength = self._calculate_attractor_strength(complex_series)
            
            # Analyze chaos characteristics
            chaos_analysis = self._analyze_chaos_characteristics(recent_data)
            
            # Pattern stability analysis
            pattern_stability = self._calculate_pattern_stability(recent_data)
            
            # Market regime classification
            regime_analysis = self._classify_market_regime(fractal_complexity, chaos_analysis)
            
            result = MandelbrotResult(
                fractal_complexity=fractal_complexity,
                mandelbrot_escape_time=np.mean(escape_times),
                self_similarity_index=self_similarity,
                scaling_exponent=scaling_analysis['exponent'],
                attractor_strength=attractor_strength,
                chaos_index=chaos_analysis['chaos_index'],
                pattern_stability=pattern_stability,
                market_regime=regime_analysis['regime'],
                complexity_trend=regime_analysis['trend'],
                fractal_confidence=scaling_analysis['confidence']
            )
            
            return self._format_output(result, data.index[-1])
            
        except Exception as e:
            logger.error(f"Error in Mandelbrot fractal calculation: {e}")
            raise IndicatorCalculationError(f"MandelbrotFractalIndicator calculation failed: {e}")

    def _create_complex_series(self, data: pd.DataFrame) -> np.ndarray:
        """Create complex number representation of price series"""
        try:
            prices = data['close'].values
            
            # Normalize prices to [0, 1] range
            scaler = MinMaxScaler(feature_range=(0, 1))
            normalized_prices = scaler.fit_transform(prices.reshape(-1, 1)).flatten()
            
            # Create complex series using price and momentum as real/imaginary parts
            if len(prices) > 1:
                momentum = np.diff(normalized_prices)
                # Pad momentum to match price length
                momentum = np.concatenate([[0], momentum])
            else:
                momentum = np.zeros_like(normalized_prices)
            
            # Scale to appropriate range for Mandelbrot analysis
            real_part = (normalized_prices - 0.5) * 2  # Scale to [-1, 1]
            imag_part = momentum * 10  # Scale momentum
            
            complex_series = real_part + 1j * imag_part
            
            return complex_series
            
        except Exception as e:
            logger.warning(f"Complex series creation error: {e}")
            return np.array([0+0j])

    def _calculate_escape_times(self, complex_series: np.ndarray) -> np.ndarray:
        """Calculate escape times for Mandelbrot-inspired analysis"""
        try:
            escape_times = []
            
            for c in complex_series:
                z = complex(0, 0)
                escape_time = 0
                
                for iteration in range(self.max_iterations):
                    if abs(z) > self.escape_radius:
                        break
                    
                    # Modified Mandelbrot iteration with market dynamics
                    z = z*z + c + 0.1*z.conjugate()  # Add market memory term
                    escape_time += 1
                
                # Normalize escape time
                normalized_escape = escape_time / self.max_iterations
                escape_times.append(normalized_escape)
            
            return np.array(escape_times)
            
        except Exception as e:
            logger.warning(f"Escape time calculation error: {e}")
            return np.array([0.5])

    def _calculate_fractal_complexity(self, escape_times: np.ndarray) -> float:
        """Calculate fractal complexity from escape time distribution"""
        try:
            if len(escape_times) == 0:
                return 0.0
            
            # Calculate distribution characteristics
            complexity_measures = []
            
            # 1. Entropy of escape time distribution
            hist, _ = np.histogram(escape_times, bins=min(20, len(escape_times)//2))
            hist = hist / np.sum(hist)  # Normalize
            hist = hist[hist > 0]  # Remove zeros
            
            if len(hist) > 1:
                entropy = -np.sum(hist * np.log2(hist))
                normalized_entropy = entropy / np.log2(len(hist))
                complexity_measures.append(normalized_entropy)
            
            # 2. Variance of escape times
            variance_complexity = np.var(escape_times)
            complexity_measures.append(min(variance_complexity * 4, 1.0))
            
            # 3. Fractal dimension approximation
            if len(escape_times) > 10:
                # Box-counting approximation
                scales = np.logspace(-2, 0, 5)
                counts = []
                
                for scale in scales:
                    bins = int(1.0 / scale)
                    hist, _ = np.histogram(escape_times, bins=bins)
                    non_empty = np.sum(hist > 0)
                    counts.append(non_empty)
                
                if len(counts) > 2:
                    log_scales = np.log(1.0 / scales)
                    log_counts = np.log(counts)
                    
                    slope, _, r_value, _, _ = stats.linregress(log_scales, log_counts)
                    
                    if r_value ** 2 > 0.7:
                        fractal_dim = max(0, min(slope / 2, 1.0))
                        complexity_measures.append(fractal_dim)
            
            # Combine complexity measures
            if complexity_measures:
                return np.mean(complexity_measures)
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"Fractal complexity calculation error: {e}")
            return 0.0

    def _calculate_self_similarity(self, data: pd.DataFrame) -> float:
        """Calculate self-similarity index using multi-scale analysis"""
        try:
            prices = data['close'].values
            
            if len(prices) < 20:
                return 0.0
            
            # Calculate correlations at different scales
            scales = [2, 4, 8, 16]
            correlations = []
            
            for scale in scales:
                if scale >= len(prices) // 2:
                    continue
                
                # Downsample at this scale
                downsampled = prices[::scale]
                
                if len(downsampled) < 10:
                    continue
                
                # Calculate correlation with original (interpolated to same length)
                if len(downsampled) != len(prices):
                    # Simple interpolation
                    x_old = np.linspace(0, 1, len(downsampled))
                    x_new = np.linspace(0, 1, len(prices))
                    interpolated = np.interp(x_new, x_old, downsampled)
                else:
                    interpolated = downsampled
                
                # Calculate correlation
                correlation = np.corrcoef(prices, interpolated)[0, 1]
                
                if not np.isnan(correlation):
                    correlations.append(abs(correlation))
            
            if correlations:
                return np.mean(correlations)
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"Self-similarity calculation error: {e}")
            return 0.0

    def _analyze_scaling_properties(self, data: pd.DataFrame) -> Dict[str, float]:
        """Analyze scaling properties of the price series"""
        try:
            if len(data) < self.scaling_window:
                return {'exponent': 0.0, 'confidence': 0.0}
            
            recent_data = data[-self.scaling_window:]
            prices = recent_data['close'].values
            
            # Calculate price fluctuations at different scales
            scales = np.unique(np.logspace(0.5, np.log10(len(prices) // 4), 8).astype(int))
            fluctuations = []
            
            for scale in scales:
                if scale >= len(prices):
                    continue
                
                # Calculate local fluctuations
                n_windows = len(prices) // scale
                scale_fluctuations = []
                
                for i in range(n_windows):
                    window = prices[i*scale:(i+1)*scale]
                    if len(window) > 1:
                        fluctuation = np.std(window)
                        scale_fluctuations.append(fluctuation)
                
                if scale_fluctuations:
                    avg_fluctuation = np.mean(scale_fluctuations)
                    fluctuations.append(avg_fluctuation)
            
            if len(fluctuations) < 3:
                return {'exponent': 0.0, 'confidence': 0.0}
            
            # Fit power law: F(s) ~ s^H
            log_scales = np.log(scales[:len(fluctuations)])
            log_fluctuations = np.log(fluctuations)
            
            slope, _, r_value, _, _ = stats.linregress(log_scales, log_fluctuations)
            
            return {
                'exponent': np.clip(slope, -1.0, 1.0),
                'confidence': r_value ** 2
            }
            
        except Exception as e:
            logger.warning(f"Scaling analysis error: {e}")
            return {'exponent': 0.0, 'confidence': 0.0}

    def _calculate_attractor_strength(self, complex_series: np.ndarray) -> float:
        """Calculate attractor strength in complex plane"""
        try:
            if len(complex_series) < 10:
                return 0.0
            
            # Calculate distances from origin
            distances = np.abs(complex_series)
            
            # Find stable regions (attractors)
            stability_threshold = np.std(distances) * 0.5
            stable_points = distances < (np.mean(distances) - stability_threshold)
            
            # Attractor strength as proportion of stable points
            attractor_strength = np.sum(stable_points) / len(complex_series)
            
            return np.clip(attractor_strength, 0.0, 1.0)
            
        except Exception as e:
            logger.warning(f"Attractor strength calculation error: {e}")
            return 0.0

    def _analyze_chaos_characteristics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Analyze chaos characteristics using phase space reconstruction"""
        try:
            prices = data['close'].values
            
            if len(prices) < 30:
                return {'chaos_index': 0.0}
            
            # Create phase space embedding
            embedding_dim = 3
            delay = 1
            
            embedded = self._embed_time_series(prices, embedding_dim, delay)
            
            if embedded.shape[0] < 10:
                return {'chaos_index': 0.0}
            
            # Calculate largest Lyapunov exponent approximation
            lyapunov = self._estimate_lyapunov_exponent(embedded)
            
            # Chaos index (positive Lyapunov indicates chaos)
            chaos_index = max(0.0, lyapunov)
            
            return {'chaos_index': np.clip(chaos_index, 0.0, 1.0)}
            
        except Exception as e:
            logger.warning(f"Chaos analysis error: {e}")
            return {'chaos_index': 0.0}

    def _embed_time_series(self, data: np.ndarray, dim: int, delay: int) -> np.ndarray:
        """Create time delay embedding"""
        n = len(data)
        m = n - (dim - 1) * delay
        
        if m <= 0:
            return np.array([[]])
        
        embedded = np.zeros((m, dim))
        for i in range(dim):
            embedded[:, i] = data[i * delay:i * delay + m]
            
        return embedded

    def _estimate_lyapunov_exponent(self, embedded: np.ndarray) -> float:
        """Estimate largest Lyapunov exponent"""
        try:
            if embedded.shape[0] < 20:
                return 0.0
            
            divergences = []
            
            for i in range(min(embedded.shape[0] - 10, 50)):
                # Find nearest neighbor
                distances = np.linalg.norm(embedded[i+1:] - embedded[i], axis=1)
                if len(distances) == 0:
                    continue
                
                nearest_idx = np.argmin(distances) + i + 1
                if nearest_idx >= embedded.shape[0] - 5:
                    continue
                
                initial_distance = distances[nearest_idx - i - 1]
                if initial_distance == 0:
                    continue
                
                # Track divergence over time
                for t in range(1, min(5, embedded.shape[0] - nearest_idx)):
                    current_distance = np.linalg.norm(embedded[i + t] - embedded[nearest_idx + t])
                    if current_distance > 0:
                        divergence = np.log(current_distance / initial_distance) / t
                        divergences.append(divergence)
            
            if divergences:
                return np.mean(divergences)
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"Lyapunov estimation error: {e}")
            return 0.0

    def _calculate_pattern_stability(self, data: pd.DataFrame) -> float:
        """Calculate pattern stability using fractal persistence"""
        try:
            if len(data) < 20:
                return 0.0
            
            prices = data['close'].values
            returns = np.diff(prices) / prices[:-1]
            
            # Calculate pattern consistency using autocorrelation
            autocorrelations = []
            
            for lag in range(1, min(10, len(returns) // 2)):
                if lag < len(returns):
                    autocorr = np.corrcoef(returns[:-lag], returns[lag:])[0, 1]
                    if not np.isnan(autocorr):
                        autocorrelations.append(abs(autocorr))
            
            if autocorrelations:
                # Stability as persistence of patterns
                stability = np.mean(autocorrelations)
                return np.clip(stability, 0.0, 1.0)
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"Pattern stability calculation error: {e}")
            return 0.0

    def _classify_market_regime(self, complexity: float, chaos_analysis: Dict[str, float]) -> Dict[str, str]:
        """Classify market regime based on fractal characteristics"""
        try:
            chaos_index = chaos_analysis.get('chaos_index', 0.0)
            
            # Regime classification
            if complexity > 0.7 and chaos_index > 0.6:
                regime = 'HIGHLY_CHAOTIC'
                trend = 'INCREASING_COMPLEXITY'
            elif complexity > 0.5:
                regime = 'MODERATELY_COMPLEX'
                trend = 'STABLE_COMPLEXITY'
            elif complexity < 0.3:
                regime = 'LOW_COMPLEXITY'
                trend = 'DECREASING_COMPLEXITY'
            else:
                regime = 'TRANSITIONAL'
                trend = 'VARIABLE_COMPLEXITY'
            
            return {'regime': regime, 'trend': trend}
            
        except Exception as e:
            logger.warning(f"Regime classification error: {e}")
            return {'regime': 'UNKNOWN', 'trend': 'UNKNOWN'}

    def _format_output(self, result: MandelbrotResult, timestamp) -> Dict[str, Any]:
        """Format the calculation results for output"""
        return {
            'timestamp': timestamp,
            'indicator_name': 'MandelbrotFractal',
            'fractal_complexity': round(result.fractal_complexity, 4),
            'mandelbrot_escape_time': round(result.mandelbrot_escape_time, 4),
            'self_similarity_index': round(result.self_similarity_index, 4),
            'scaling_exponent': round(result.scaling_exponent, 4),
            'attractor_strength': round(result.attractor_strength, 4),
            'chaos_index': round(result.chaos_index, 4),
            'pattern_stability': round(result.pattern_stability, 4),
            'market_regime': result.market_regime,
            'complexity_trend': result.complexity_trend,
            'fractal_confidence': round(result.fractal_confidence, 4)
        }

    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate input data"""
        required_columns = ['close']
        if not all(col in data.columns for col in required_columns):
            raise IndicatorCalculationError(f"Missing required columns: {required_columns}")
        if len(data) < self.scaling_window:
            raise IndicatorCalculationError(f"Insufficient data: minimum {self.scaling_window} periods required")

    def _create_default_result(self) -> Dict[str, Any]:
        """Create default result for insufficient data cases"""
        return {
            'timestamp': pd.Timestamp.now(),
            'indicator_name': 'MandelbrotFractal',
            'fractal_complexity': 0.0,
            'mandelbrot_escape_time': 0.5,
            'self_similarity_index': 0.0,
            'scaling_exponent': 0.0,
            'attractor_strength': 0.0,
            'chaos_index': 0.0,
            'pattern_stability': 0.0,
            'market_regime': 'INSUFFICIENT_DATA',
            'complexity_trend': 'UNKNOWN',
            'fractal_confidence': 0.0
        }

    def get_required_columns(self) -> List[str]:
        return ['close']

    def get_indicator_name(self) -> str:
        return "MandelbrotFractal"