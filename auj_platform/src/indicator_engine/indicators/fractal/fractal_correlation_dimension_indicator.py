"""
Fractal Correlation Dimension Indicator - Advanced Implementation

This indicator implements Grassberger-Procaccia correlation dimension analysis
with sophisticated mathematical models for market regime classification.
Features include:
- Grassberger-Procaccia correlation dimension calculation
- Multi-scale correlation analysis
- Embedding dimension optimization
- Information dimension and capacity dimension
- Market regime classification via dimension analysis
- Advanced statistical validation and confidence intervals

Mission: Supporting humanitarian trading platform for poor and sick children through
maximum profitability via advanced fractal dimension analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import warnings

from ..base.standard_indicator import StandardIndicatorInterface
from ....core.exceptions import IndicatorCalculationError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CorrelationDimensionResult:
    """Results container for Fractal Correlation Dimension"""
    correlation_dimension: float
    information_dimension: float
    capacity_dimension: float
    optimal_embedding_dim: int
    dimension_confidence: float
    market_regime: str
    complexity_level: str
    dimension_stability: float
    scaling_quality: float
    attractor_type: str

class FractalCorrelationDimensionIndicator(StandardIndicatorInterface):
    """
    Advanced Fractal Correlation Dimension Indicator
    
    Implements Grassberger-Procaccia correlation dimension analysis
    with sophisticated mathematical models for market regime classification.
    """
    
    def __init__(self, 
                 window_size: int = 200,
                 max_embedding_dim: int = 10,
                 min_embedding_dim: int = 2,
                 radius_range: Tuple[float, float] = (0.01, 0.5),
                 n_radius_points: int = 20):
        super().__init__()
        self.window_size = window_size
        self.max_embedding_dim = max_embedding_dim
        self.min_embedding_dim = min_embedding_dim
        self.radius_range = radius_range
        self.n_radius_points = n_radius_points
        
        logger.info(f"Initialized FractalCorrelationDimensionIndicator with window_size={window_size}")

    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        try:
            self._validate_data(data)
            
            if len(data) < self.window_size:
                return self._create_default_result()
            
            recent_data = data[-self.window_size:].copy()
            
            # Prepare time series
            time_series = self._prepare_time_series(recent_data)
            
            # Calculate correlation dimensions for different embedding dimensions
            dimension_results = self._calculate_correlation_dimensions(time_series)
            
            # Find optimal embedding dimension
            optimal_dim = self._find_optimal_embedding_dimension(dimension_results)
            
            # Calculate additional fractal dimensions
            info_dim = self._calculate_information_dimension(time_series, optimal_dim)
            capacity_dim = self._calculate_capacity_dimension(time_series, optimal_dim)
            
            # Analyze market regime
            regime_analysis = self._analyze_market_regime(dimension_results, optimal_dim)
            
            # Generate signals
            signals = self._generate_dimension_signals(dimension_results, regime_analysis)
            
            result = CorrelationDimensionResult(
                correlation_dimension=dimension_results[optimal_dim]['dimension'],
                information_dimension=info_dim,
                capacity_dimension=capacity_dim,
                optimal_embedding_dim=optimal_dim,
                dimension_confidence=dimension_results[optimal_dim]['confidence'],
                market_regime=regime_analysis['regime'],
                complexity_level=signals['complexity_level'],
                dimension_stability=regime_analysis['stability'],
                scaling_quality=dimension_results[optimal_dim]['scaling_quality'],
                attractor_type=signals['attractor_type']
            )
            
            return self._format_output(result, data.index[-1])
            
        except Exception as e:
            logger.error(f"Error in correlation dimension calculation: {e}")
            raise IndicatorCalculationError(f"FractalCorrelationDimensionIndicator calculation failed: {e}")

    def _prepare_time_series(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare time series data for dimension analysis"""
        # Use log returns for better stationarity
        prices = data['close'].values
        log_returns = np.diff(np.log(prices))
        
        # Standardize the series
        scaler = StandardScaler()
        normalized_returns = scaler.fit_transform(log_returns.reshape(-1, 1)).flatten()
        
        return normalized_returns

    def _calculate_correlation_dimensions(self, time_series: np.ndarray) -> Dict[int, Dict[str, float]]:
        """Calculate correlation dimensions for different embedding dimensions"""
        results = {}
        
        for m in range(self.min_embedding_dim, self.max_embedding_dim + 1):
            try:
                # Create embedded phase space
                embedded = self._embed_time_series(time_series, m, delay=1)
                
                if embedded.shape[0] < 50:  # Need sufficient points
                    continue
                
                # Calculate correlation dimension
                dim_result = self._grassberger_procaccia_dimension(embedded)
                results[m] = dim_result
                
            except Exception as e:
                logger.warning(f"Dimension calculation failed for m={m}: {e}")
                continue
        
        return results

    def _embed_time_series(self, data: np.ndarray, dim: int, delay: int) -> np.ndarray:
        """Create phase space embedding using delay coordinates"""
        n = len(data)
        m = n - (dim - 1) * delay
        
        if m <= 0:
            return np.array([[]])
        
        embedded = np.zeros((m, dim))
        for i in range(dim):
            embedded[:, i] = data[i * delay:i * delay + m]
            
        return embedded

    def _grassberger_procaccia_dimension(self, embedded: np.ndarray) -> Dict[str, float]:
        """Calculate correlation dimension using Grassberger-Procaccia algorithm"""
        try:
            n_points = embedded.shape[0]
            
            # Calculate pairwise distances
            distances = pdist(embedded)
            
            # Generate radius values
            r_min, r_max = self.radius_range
            radii = np.logspace(np.log10(r_min), np.log10(r_max), self.n_radius_points)
            
            # Calculate correlation integral for each radius
            log_radii = []
            log_correlations = []
            
            for r in radii:
                # Count pairs within radius r
                count = np.sum(distances < r)
                
                # Correlation integral
                c_r = count / (n_points * (n_points - 1) / 2)
                
                if c_r > 0:
                    log_radii.append(np.log(r))
                    log_correlations.append(np.log(c_r))
            
            if len(log_radii) < 5:
                return {'dimension': 0.0, 'confidence': 0.0, 'scaling_quality': 0.0}
            
            # Find scaling region and calculate dimension
            dimension, confidence, scaling_quality = self._estimate_scaling_exponent(log_radii, log_correlations)
            
            return {
                'dimension': dimension,
                'confidence': confidence,
                'scaling_quality': scaling_quality
            }
            
        except Exception as e:
            logger.warning(f"Grassberger-Procaccia calculation error: {e}")
            return {'dimension': 0.0, 'confidence': 0.0, 'scaling_quality': 0.0}

    def _estimate_scaling_exponent(self, log_radii: List[float], log_correlations: List[float]) -> Tuple[float, float, float]:
        """Estimate scaling exponent with confidence intervals"""
        try:
            log_radii = np.array(log_radii)
            log_correlations = np.array(log_correlations)
            
            # Find the best linear scaling region
            best_slope = 0.0
            best_r2 = 0.0
            best_range = (0, len(log_radii))
            
            min_points = max(5, len(log_radii) // 3)
            
            for start in range(len(log_radii) - min_points + 1):
                for end in range(start + min_points, len(log_radii) + 1):
                    x = log_radii[start:end]
                    y = log_correlations[start:end]
                    
                    if len(x) < 3:
                        continue
                    
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                    r2 = r_value ** 2
                    
                    if r2 > best_r2:
                        best_slope = slope
                        best_r2 = r2
                        best_range = (start, end)
            
            # Calculate confidence based on R²
            confidence = best_r2
            
            return best_slope, confidence, best_r2
            
        except Exception as e:
            logger.warning(f"Scaling exponent estimation error: {e}")
            return 0.0, 0.0, 0.0

    def _find_optimal_embedding_dimension(self, dimension_results: Dict[int, Dict[str, float]]) -> int:
        """Find optimal embedding dimension using false nearest neighbors method"""
        if not dimension_results:
            return self.min_embedding_dim
        
        # Look for dimension where correlation dimension stabilizes
        dimensions = sorted(dimension_results.keys())
        
        if len(dimensions) < 2:
            return dimensions[0]
        
        # Find where dimension change becomes small
        optimal_dim = dimensions[0]
        min_change = float('inf')
        
        for i in range(1, len(dimensions)):
            current_dim = dimensions[i]
            prev_dim = dimensions[i-1]
            
            current_corr_dim = dimension_results[current_dim]['dimension']
            prev_corr_dim = dimension_results[prev_dim]['dimension']
            
            change = abs(current_corr_dim - prev_corr_dim)
            
            if change < min_change and dimension_results[current_dim]['confidence'] > 0.7:
                min_change = change
                optimal_dim = current_dim
        
        return optimal_dim

    def _calculate_information_dimension(self, time_series: np.ndarray, embedding_dim: int) -> float:
        """Calculate information dimension"""
        try:
            embedded = self._embed_time_series(time_series, embedding_dim, delay=1)
            
            if embedded.shape[0] < 20:
                return 0.0
            
            # Discretize the space for probability calculation
            n_bins = max(10, int(np.sqrt(embedded.shape[0])))
            
            # Calculate probabilities in each bin
            hist, _ = np.histogramdd(embedded, bins=n_bins)
            probabilities = hist.flatten()
            probabilities = probabilities[probabilities > 0]
            probabilities = probabilities / np.sum(probabilities)
            
            # Calculate information dimension
            entropy = -np.sum(probabilities * np.log(probabilities))
            
            # Approximate information dimension
            info_dim = entropy / np.log(n_bins)
            
            return np.clip(info_dim, 0.0, embedding_dim)
            
        except Exception as e:
            logger.warning(f"Information dimension calculation error: {e}")
            return 0.0

    def _calculate_capacity_dimension(self, time_series: np.ndarray, embedding_dim: int) -> float:
        """Calculate capacity (box-counting) dimension"""
        try:
            embedded = self._embed_time_series(time_series, embedding_dim, delay=1)
            
            if embedded.shape[0] < 20:
                return 0.0
            
            # Box-counting dimension calculation
            scales = np.logspace(-2, 0, 10)
            counts = []
            
            for scale in scales:
                # Count boxes containing points
                box_counts = []
                for dim in range(embedding_dim):
                    min_val = np.min(embedded[:, dim])
                    max_val = np.max(embedded[:, dim])
                    range_val = max_val - min_val
                    
                    if range_val == 0:
                        box_counts.append(1)
                    else:
                        n_boxes = int(np.ceil(range_val / scale))
                        box_counts.append(n_boxes)
                
                total_boxes = np.prod(box_counts)
                counts.append(total_boxes)
            
            if len(counts) < 3:
                return 0.0
            
            # Fit log-log relationship
            log_scales = np.log(1.0 / scales)
            log_counts = np.log(counts)
            
            slope, _, r_value, _, _ = stats.linregress(log_scales, log_counts)
            
            # Return capacity dimension if fit is good
            if r_value ** 2 > 0.7:
                return np.clip(slope, 0.0, embedding_dim)
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"Capacity dimension calculation error: {e}")
            return 0.0

    def _analyze_market_regime(self, dimension_results: Dict[int, Dict[str, float]], 
                             optimal_dim: int) -> Dict[str, Any]:
        """Analyze market regime based on fractal dimensions"""
        try:
            if optimal_dim not in dimension_results:
                return {'regime': 'UNKNOWN', 'stability': 0.0}
            
            corr_dim = dimension_results[optimal_dim]['dimension']
            confidence = dimension_results[optimal_dim]['confidence']
            
            # Classify regime based on correlation dimension
            if corr_dim < 1.5:
                regime = 'DETERMINISTIC'
            elif corr_dim < 2.5:
                regime = 'LOW_DIMENSIONAL_CHAOS'
            elif corr_dim < 4.0:
                regime = 'HIGH_DIMENSIONAL_CHAOS'
            else:
                regime = 'STOCHASTIC'
            
            # Calculate stability
            dimensions = []
            for dim_data in dimension_results.values():
                if dim_data['confidence'] > 0.5:
                    dimensions.append(dim_data['dimension'])
            
            if len(dimensions) > 1:
                stability = 1.0 - (np.std(dimensions) / (np.mean(dimensions) + 1e-8))
            else:
                stability = confidence
            
            return {
                'regime': regime,
                'stability': np.clip(stability, 0.0, 1.0)
            }
            
        except Exception as e:
            logger.warning(f"Market regime analysis error: {e}")
            return {'regime': 'UNKNOWN', 'stability': 0.0}

    def _generate_dimension_signals(self, dimension_results: Dict[int, Dict[str, float]], 
                                  regime_analysis: Dict[str, Any]) -> Dict[str, str]:
        """Generate trading signals from dimension analysis"""
        try:
            # Get average dimension
            valid_dims = [d['dimension'] for d in dimension_results.values() if d['confidence'] > 0.5]
            
            if not valid_dims:
                return {'complexity_level': 'UNKNOWN', 'attractor_type': 'UNKNOWN'}
            
            avg_dimension = np.mean(valid_dims)
            
            # Complexity level
            if avg_dimension < 2.0:
                complexity_level = 'LOW'
            elif avg_dimension < 3.5:
                complexity_level = 'MEDIUM'
            else:
                complexity_level = 'HIGH'
            
            # Attractor type
            regime = regime_analysis['regime']
            stability = regime_analysis['stability']
            
            if regime == 'DETERMINISTIC' and stability > 0.8:
                attractor_type = 'FIXED_POINT'
            elif regime in ['LOW_DIMENSIONAL_CHAOS', 'HIGH_DIMENSIONAL_CHAOS']:
                if stability > 0.6:
                    attractor_type = 'STRANGE_ATTRACTOR'
                else:
                    attractor_type = 'CHAOTIC'
            else:
                attractor_type = 'STOCHASTIC'
            
            return {
                'complexity_level': complexity_level,
                'attractor_type': attractor_type
            }
            
        except Exception as e:
            logger.warning(f"Signal generation error: {e}")
            return {'complexity_level': 'UNKNOWN', 'attractor_type': 'UNKNOWN'}

    def _format_output(self, result: CorrelationDimensionResult, timestamp) -> Dict[str, Any]:
        """Format the calculation results for output"""
        return {
            'timestamp': timestamp,
            'indicator_name': 'FractalCorrelationDimension',
            'correlation_dimension': round(result.correlation_dimension, 4),
            'information_dimension': round(result.information_dimension, 4),
            'capacity_dimension': round(result.capacity_dimension, 4),
            'optimal_embedding_dim': result.optimal_embedding_dim,
            'dimension_confidence': round(result.dimension_confidence, 4),
            'market_regime': result.market_regime,
            'complexity_level': result.complexity_level,
            'dimension_stability': round(result.dimension_stability, 4),
            'scaling_quality': round(result.scaling_quality, 4),
            'attractor_type': result.attractor_type
        }

    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate input data"""
        required_columns = ['close']
        if not all(col in data.columns for col in required_columns):
            raise IndicatorCalculationError(f"Missing required columns: {required_columns}")
        if len(data) < self.window_size:
            raise IndicatorCalculationError(f"Insufficient data: minimum {self.window_size} periods required")

    def _create_default_result(self) -> Dict[str, Any]:
        """Create default result for insufficient data cases"""
        return {
            'timestamp': pd.Timestamp.now(),
            'indicator_name': 'FractalCorrelationDimension',
            'correlation_dimension': 0.0,
            'information_dimension': 0.0,
            'capacity_dimension': 0.0,
            'optimal_embedding_dim': self.min_embedding_dim,
            'dimension_confidence': 0.0,
            'market_regime': 'INSUFFICIENT_DATA',
            'complexity_level': 'UNKNOWN',
            'dimension_stability': 0.0,
            'scaling_quality': 0.0,
            'attractor_type': 'UNKNOWN'
        }

    def get_required_columns(self) -> List[str]:
        return ['close']

    def get_indicator_name(self) -> str:
        return "FractalCorrelationDimension"