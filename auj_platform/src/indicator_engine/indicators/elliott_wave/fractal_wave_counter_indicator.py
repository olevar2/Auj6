"""
Fractal Wave Counter Indicator

Advanced implementation with sophisticated fractal analysis to count and classify
Elliott Wave structures using advanced mathematical models for the humanitarian
trading platform.

This indicator uses fractal geometry and chaos theory to identify and count
Elliott Wave patterns with high precision and mathematical rigor.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy import signal
from scipy.signal import find_peaks, argrelextrema
from scipy.stats import linregress
import warnings
warnings.filterwarnings('ignore')

from ..base.standard_indicator import StandardIndicatorInterface
from ....core.signal_type import SignalType
from ....core.exceptions import IndicatorCalculationError


@dataclass
class FractalConfig:
    """Configuration for Fractal Wave Counter."""
    fractal_dimension_window: int = 50
    wave_validation_window: int = 20
    min_wave_length: int = 5
    max_wave_length: int = 100
    fractal_threshold: float = 1.5
    confidence_threshold: float = 0.75
    clustering_eps: float = 0.1
    min_samples: int = 3


@dataclass
class WaveStructure:
    """Structure representing an Elliott Wave."""
    start_idx: int
    end_idx: int
    wave_type: str  # 'impulse' or 'corrective'
    wave_number: int  # 1-5 for impulse, A-C for corrective
    fractal_dimension: float
    strength: float
    fibonacci_ratio: float
    price_start: float
    price_end: float
    volume_profile: float


@dataclass
class FractalAnalysis:
    """Results of fractal analysis."""
    fractal_dimension: float
    hurst_exponent: float
    self_similarity: float
    chaos_factor: float
    predictability_index: float


class FractalWaveCounterIndicator(StandardIndicatorInterface):
    """
    Advanced Fractal Wave Counter with mathematical precision.
    
    This indicator combines fractal geometry, chaos theory, and Elliott Wave
    principles to identify and classify wave structures with high accuracy.
    """
    
    def __init__(self, config: Optional[FractalConfig] = None):
        super().__init__()
        self.config = config or FractalConfig()
        self.logger = logging.getLogger(__name__)
        
        # Wave storage
        self.identified_waves: List[WaveStructure] = []
        self.fractal_history: List[FractalAnalysis] = []
        
        # Mathematical constants
        self.golden_ratio = 1.618033988749
        self.fibonacci_sequence = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
        self.elliott_ratios = [0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618, 2.618]
        
        # Clustering for pattern recognition
        self.scaler = StandardScaler()
        self.dbscan = DBSCAN(eps=self.config.clustering_eps, min_samples=self.config.min_samples)
        
    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate fractal wave count with advanced mathematical analysis.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary containing wave count and fractal analysis
        """
        try:
            if len(data) < self.config.fractal_dimension_window:
                raise IndicatorCalculationError(
                    f"Insufficient data: need at least {self.config.fractal_dimension_window} periods"
                )
            
            # Perform fractal analysis
            fractal_analysis = self._perform_fractal_analysis(data)
            
            # Identify potential wave turning points
            turning_points = self._identify_turning_points(data)
            
            # Classify and count waves
            wave_structures = self._classify_waves(data, turning_points, fractal_analysis)
            
            # Validate wave patterns
            validated_waves = self._validate_wave_patterns(wave_structures, data)
            
            # Calculate wave statistics
            wave_statistics = self._calculate_wave_statistics(validated_waves)
            
            # Perform clustering analysis
            cluster_analysis = self._perform_cluster_analysis(validated_waves)
            
            # Generate predictions
            predictions = self._generate_wave_predictions(validated_waves, fractal_analysis, data)
            
            result = {
                'total_wave_count': len(validated_waves),
                'impulse_wave_count': sum(1 for w in validated_waves if w.wave_type == 'impulse'),
                'corrective_wave_count': sum(1 for w in validated_waves if w.wave_type == 'corrective'),
                'current_wave_position': self._determine_current_position(validated_waves, len(data)),
                'fractal_dimension': fractal_analysis.fractal_dimension,
                'hurst_exponent': fractal_analysis.hurst_exponent,
                'self_similarity': fractal_analysis.self_similarity,
                'chaos_factor': fractal_analysis.chaos_factor,
                'predictability_index': fractal_analysis.predictability_index,
                'wave_strength_avg': wave_statistics['avg_strength'],
                'wave_length_avg': wave_statistics['avg_length'],
                'fibonacci_compliance': wave_statistics['fibonacci_compliance'],
                'pattern_confidence': wave_statistics['pattern_confidence'],
                'cluster_count': cluster_analysis['cluster_count'],
                'dominant_pattern': cluster_analysis['dominant_pattern'],
                'next_wave_prediction': predictions['next_wave'],
                'completion_probability': predictions['completion_probability'],
                'target_levels': predictions['target_levels'],
                'support_resistance': self._calculate_support_resistance(validated_waves, data),
                'signal_type': self._determine_signal_type(validated_waves, fractal_analysis),
                'raw_data': {
                    'turning_points': [tp for tp in turning_points],
                    'wave_structures': [w.__dict__ for w in validated_waves],
                    'fractal_analysis': fractal_analysis.__dict__
                }
            }
            
            # Update historical data
            self._update_historical_data(validated_waves, fractal_analysis)
            
            self.logger.info(f"Fractal Wave Counter calculated: {len(validated_waves)} waves identified")
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating Fractal Wave Counter: {str(e)}")
            raise IndicatorCalculationError(f"Fractal Wave Counter calculation failed: {str(e)}")
    
    def _perform_fractal_analysis(self, data: pd.DataFrame) -> FractalAnalysis:
        """Perform comprehensive fractal analysis."""
        try:
            prices = data['close'].values
            log_prices = np.log(prices)
            
            # Calculate fractal dimension using box-counting method
            fractal_dimension = self._calculate_fractal_dimension(prices)
            
            # Calculate Hurst exponent
            hurst_exponent = self._calculate_hurst_exponent(log_prices)
            
            # Calculate self-similarity measure
            self_similarity = self._calculate_self_similarity(prices)
            
            # Calculate chaos factor
            chaos_factor = self._calculate_chaos_factor(prices)
            
            # Calculate predictability index
            predictability_index = self._calculate_predictability_index(prices)
            
            return FractalAnalysis(
                fractal_dimension=fractal_dimension,
                hurst_exponent=hurst_exponent,
                self_similarity=self_similarity,
                chaos_factor=chaos_factor,
                predictability_index=predictability_index
            )
            
        except Exception as e:
            self.logger.warning(f"Fractal analysis failed: {str(e)}")
            return FractalAnalysis(1.5, 0.5, 0.5, 0.5, 0.5)
    
    def _calculate_fractal_dimension(self, prices: np.ndarray) -> float:
        """Calculate fractal dimension using box-counting method."""
        try:
            # Normalize prices to [0, 1]
            normalized_prices = (prices - prices.min()) / (prices.max() - prices.min() + 1e-8)
            
            # Box sizes (powers of 2)
            box_sizes = np.array([2**i for i in range(1, 8)])
            box_counts = []
            
            for box_size in box_sizes:
                # Count boxes needed to cover the curve
                grid_size = 1.0 / box_size
                boxes = set()
                
                for i in range(len(normalized_prices) - 1):
                    x1, y1 = i / len(normalized_prices), normalized_prices[i]
                    x2, y2 = (i + 1) / len(normalized_prices), normalized_prices[i + 1]
                    
                    # Line drawing algorithm to find all boxes the line passes through
                    steps = max(abs(int(x2/grid_size) - int(x1/grid_size)), 
                               abs(int(y2/grid_size) - int(y1/grid_size))) + 1
                    
                    for step in range(steps):
                        t = step / steps if steps > 0 else 0
                        x = x1 + t * (x2 - x1)
                        y = y1 + t * (y2 - y1)
                        box_x = int(x / grid_size)
                        box_y = int(y / grid_size)
                        boxes.add((box_x, box_y))
                
                box_counts.append(len(boxes))
            
            # Fit log-log plot to get fractal dimension
            if len(box_counts) > 2 and all(count > 0 for count in box_counts):
                log_sizes = np.log(1.0 / box_sizes)
                log_counts = np.log(box_counts)
                slope, _, r_value, _, _ = linregress(log_sizes, log_counts)
                
                # Fractal dimension is the negative slope
                fractal_dim = abs(slope)
                
                # Ensure reasonable bounds
                return max(1.0, min(2.0, fractal_dim))
            
            return 1.5  # Default fractal dimension
            
        except Exception as e:
            self.logger.warning(f"Fractal dimension calculation failed: {str(e)}")
            return 1.5
    
    def _calculate_hurst_exponent(self, log_prices: np.ndarray) -> float:
        """Calculate Hurst exponent using R/S analysis."""
        try:
            n = len(log_prices)
            if n < 20:
                return 0.5
            
            # Calculate returns
            returns = np.diff(log_prices)
            
            # Range of window sizes
            window_sizes = np.unique(np.logspace(1, np.log10(n//4), 10).astype(int))
            rs_values = []
            
            for window_size in window_sizes:
                if window_size >= n:
                    continue
                    
                # Split into non-overlapping windows
                num_windows = n // window_size
                rs_window_values = []
                
                for i in range(num_windows):
                    start_idx = i * window_size
                    end_idx = start_idx + window_size
                    window_returns = returns[start_idx:end_idx]
                    
                    if len(window_returns) == 0:
                        continue
                    
                    # Calculate mean and cumulative deviations
                    mean_return = np.mean(window_returns)
                    cumulative_deviations = np.cumsum(window_returns - mean_return)
                    
                    # Calculate range and standard deviation
                    range_val = np.max(cumulative_deviations) - np.min(cumulative_deviations)
                    std_val = np.std(window_returns)
                    
                    if std_val > 0:
                        rs_window_values.append(range_val / std_val)
                
                if rs_window_values:
                    rs_values.append(np.mean(rs_window_values))
            
            # Fit log-log plot
            if len(rs_values) > 2 and len(window_sizes[:len(rs_values)]) > 2:
                log_window_sizes = np.log(window_sizes[:len(rs_values)])
                log_rs_values = np.log(rs_values)
                
                # Remove any infinite or NaN values
                valid_mask = np.isfinite(log_window_sizes) & np.isfinite(log_rs_values)
                if np.sum(valid_mask) > 2:
                    slope, _, _, _, _ = linregress(log_window_sizes[valid_mask], log_rs_values[valid_mask])
                    return max(0.0, min(1.0, slope))
            
            return 0.5  # Random walk default
            
        except Exception as e:
            self.logger.warning(f"Hurst exponent calculation failed: {str(e)}")
            return 0.5
    
    def _calculate_self_similarity(self, prices: np.ndarray) -> float:
        """Calculate self-similarity measure."""
        try:
            # Compare patterns at different scales
            scales = [5, 10, 20, 40]
            correlations = []
            
            for scale in scales:
                if scale * 2 >= len(prices):
                    continue
                
                # Downsample at different scales
                downsampled1 = prices[::scale]
                downsampled2 = prices[scale//2::scale]
                
                min_len = min(len(downsampled1), len(downsampled2))
                if min_len > 10:
                    correlation = np.corrcoef(
                        downsampled1[:min_len], 
                        downsampled2[:min_len]
                    )[0, 1]
                    
                    if not np.isnan(correlation):
                        correlations.append(abs(correlation))
            
            return np.mean(correlations) if correlations else 0.5
            
        except Exception as e:
            self.logger.warning(f"Self-similarity calculation failed: {str(e)}")
            return 0.5
    
    def _calculate_chaos_factor(self, prices: np.ndarray) -> float:
        """Calculate chaos factor using largest Lyapunov exponent approximation."""
        try:
            # Simple approximation of largest Lyapunov exponent
            returns = np.diff(np.log(prices))
            
            # Embed the time series
            embedding_dim = 3
            if len(returns) < embedding_dim * 2:
                return 0.5
            
            embedded = []
            for i in range(len(returns) - embedding_dim + 1):
                embedded.append(returns[i:i + embedding_dim])
            
            embedded = np.array(embedded)
            
            # Calculate average divergence
            divergences = []
            for i in range(len(embedded) - 1):
                for j in range(i + 1, min(i + 20, len(embedded))):
                    initial_distance = np.linalg.norm(embedded[i] - embedded[j])
                    if initial_distance > 0:
                        final_distance = np.linalg.norm(embedded[i + 1] - embedded[j + 1]) if j + 1 < len(embedded) else initial_distance
                        if final_distance > 0:
                            divergence = np.log(final_distance / initial_distance)
                            divergences.append(divergence)
            
            chaos_factor = np.mean(divergences) if divergences else 0.0
            return max(0.0, min(1.0, (chaos_factor + 1) / 2))  # Normalize to [0, 1]
            
        except Exception as e:
            self.logger.warning(f"Chaos factor calculation failed: {str(e)}")
            return 0.5
    
    def _calculate_predictability_index(self, prices: np.ndarray) -> float:
        """Calculate predictability index."""
        try:
            # Use autocorrelation to measure predictability
            returns = np.diff(np.log(prices))
            autocorrelations = []
            
            max_lag = min(20, len(returns) // 4)
            for lag in range(1, max_lag + 1):
                if len(returns) > lag:
                    correlation = np.corrcoef(returns[:-lag], returns[lag:])[0, 1]
                    if not np.isnan(correlation):
                        autocorrelations.append(abs(correlation))
            
            # Predictability is related to persistence of autocorrelations
            if autocorrelations:
                predictability = np.sum(autocorrelations) / len(autocorrelations)
                return max(0.0, min(1.0, predictability))
            
            return 0.5
            
        except Exception as e:
            self.logger.warning(f"Predictability index calculation failed: {str(e)}")
            return 0.5
    
    def _identify_turning_points(self, data: pd.DataFrame) -> List[int]:
        """Identify potential wave turning points using fractal analysis."""
        try:
            highs = data['high'].values
            lows = data['low'].values
            
            # Find local maxima and minima
            high_peaks, _ = find_peaks(highs, distance=self.config.min_wave_length)
            low_peaks, _ = find_peaks(-lows, distance=self.config.min_wave_length)
            
            # Combine and sort turning points
            turning_points = np.concatenate([high_peaks, low_peaks])
            turning_points = np.sort(turning_points)
            
            # Filter based on significance
            filtered_points = self._filter_significant_points(turning_points, data)
            
            return filtered_points.tolist()
            
        except Exception as e:
            self.logger.warning(f"Turning point identification failed: {str(e)}")
            return []
    
    def _filter_significant_points(self, turning_points: np.ndarray, data: pd.DataFrame) -> np.ndarray:
        """Filter turning points based on significance."""
        if len(turning_points) < 3:
            return turning_points
        
        filtered_points = [turning_points[0]]  # Always keep first point
        
        for i in range(1, len(turning_points) - 1):
            current_idx = turning_points[i]
            prev_idx = filtered_points[-1]
            next_idx = turning_points[i + 1]
            
            # Calculate price change significance
            prev_price = data['close'].iloc[prev_idx]
            current_price = data['close'].iloc[current_idx]
            next_price = data['close'].iloc[next_idx]
            
            # Minimum price change threshold (relative to volatility)
            volatility = data['close'].rolling(20).std().iloc[current_idx]
            min_change = volatility * self.config.fractal_threshold
            
            prev_change = abs(current_price - prev_price)
            next_change = abs(next_price - current_price)
            
            if prev_change > min_change or next_change > min_change:
                filtered_points.append(current_idx)
        
        if len(turning_points) > 0:
            filtered_points.append(turning_points[-1])  # Always keep last point
        
        return np.array(filtered_points)
    
    def _classify_waves(self, data: pd.DataFrame, turning_points: List[int], 
                       fractal_analysis: FractalAnalysis) -> List[WaveStructure]:
        """Classify waves using Elliott Wave theory and fractal analysis."""
        waves = []
        
        if len(turning_points) < 2:
            return waves
        
        for i in range(len(turning_points) - 1):
            start_idx = turning_points[i]
            end_idx = turning_points[i + 1]
            
            # Calculate wave properties
            wave_length = end_idx - start_idx
            if wave_length < self.config.min_wave_length or wave_length > self.config.max_wave_length:
                continue
            
            price_start = data['close'].iloc[start_idx]
            price_end = data['close'].iloc[end_idx]
            price_change = abs(price_end - price_start)
            
            # Calculate wave strength
            wave_data = data.iloc[start_idx:end_idx + 1]
            strength = self._calculate_wave_strength(wave_data, fractal_analysis)
            
            # Calculate Fibonacci ratio
            fibonacci_ratio = self._calculate_fibonacci_ratio(price_change, waves)
            
            # Determine wave type and number
            wave_type, wave_number = self._determine_wave_type(i, len(turning_points), strength)
            
            # Calculate volume profile
            volume_profile = wave_data['volume'].mean() / data['volume'].mean()
            
            wave = WaveStructure(
                start_idx=start_idx,
                end_idx=end_idx,
                wave_type=wave_type,
                wave_number=wave_number,
                fractal_dimension=fractal_analysis.fractal_dimension,
                strength=strength,
                fibonacci_ratio=fibonacci_ratio,
                price_start=price_start,
                price_end=price_end,
                volume_profile=volume_profile
            )
            
            waves.append(wave)
        
        return waves    
    def _calculate_wave_strength(self, wave_data: pd.DataFrame, fractal_analysis: FractalAnalysis) -> float:
        """Calculate wave strength based on multiple factors."""
        try:
            # Price momentum component
            price_momentum = abs(wave_data['close'].iloc[-1] - wave_data['close'].iloc[0]) / wave_data['close'].iloc[0]
            
            # Volume component
            volume_strength = wave_data['volume'].mean() / wave_data['volume'].std() if wave_data['volume'].std() > 0 else 1.0
            
            # Fractal component
            fractal_strength = fractal_analysis.fractal_dimension / 2.0  # Normalize to ~0.5-1.0
            
            # Combine components
            strength = (price_momentum * 0.5 + 
                       min(volume_strength, 2.0) * 0.3 + 
                       fractal_strength * 0.2)
            
            return max(0.0, min(2.0, strength))
            
        except Exception as e:
            self.logger.warning(f"Wave strength calculation failed: {str(e)}")
            return 1.0
    
    def _calculate_fibonacci_ratio(self, price_change: float, existing_waves: List[WaveStructure]) -> float:
        """Calculate Fibonacci ratio relative to previous waves."""
        if not existing_waves:
            return 1.0
        
        # Find the most recent significant wave
        recent_wave = existing_waves[-1] if existing_waves else None
        if not recent_wave:
            return 1.0
        
        recent_change = abs(recent_wave.price_end - recent_wave.price_start)
        if recent_change == 0:
            return 1.0
        
        ratio = price_change / recent_change
        
        # Find closest Fibonacci ratio
        closest_fib = min(self.elliott_ratios, key=lambda x: abs(x - ratio))
        return float(closest_fib)
    
    def _determine_wave_type(self, wave_index: int, total_waves: int, strength: float) -> Tuple[str, int]:
        """Determine wave type and number using Elliott Wave rules."""
        # Simplified Elliott Wave classification
        cycle_position = wave_index % 8  # 5 impulse + 3 corrective
        
        if cycle_position < 5:
            # Impulse wave (1, 2, 3, 4, 5)
            wave_number = cycle_position + 1
            wave_type = "impulse"
        else:
            # Corrective wave (A, B, C)
            wave_number = cycle_position - 4  # Maps to 1, 2, 3 (A, B, C)
            wave_type = "corrective"
        
        return wave_type, wave_number
    
    def _validate_wave_patterns(self, waves: List[WaveStructure], data: pd.DataFrame) -> List[WaveStructure]:
        """Validate wave patterns using Elliott Wave rules."""
        validated_waves = []
        
        for i, wave in enumerate(waves):
            # Basic validation checks
            if self._is_valid_wave(wave, waves[:i], data):
                validated_waves.append(wave)
        
        return validated_waves
    
    def _is_valid_wave(self, wave: WaveStructure, previous_waves: List[WaveStructure], 
                      data: pd.DataFrame) -> bool:
        """Check if wave conforms to Elliott Wave rules."""
        try:
            # Minimum strength requirement
            if wave.strength < 0.1:
                return False
            
            # Length validation
            wave_length = wave.end_idx - wave.start_idx
            if wave_length < self.config.min_wave_length:
                return False
            
            # Elliott Wave rule validations
            if wave.wave_type == "impulse":
                return self._validate_impulse_wave(wave, previous_waves)
            else:
                return self._validate_corrective_wave(wave, previous_waves)
                
        except Exception:
            return False
    
    def _validate_impulse_wave(self, wave: WaveStructure, previous_waves: List[WaveStructure]) -> bool:
        """Validate impulse wave according to Elliott Wave rules."""
        # Wave 3 should not be the shortest
        if wave.wave_number == 3 and len(previous_waves) >= 2:
            wave1_length = abs(previous_waves[0].price_end - previous_waves[0].price_start)
            wave3_length = abs(wave.price_end - wave.price_start)
            
            if len(previous_waves) >= 4:
                wave5_length = abs(previous_waves[4].price_end - previous_waves[4].price_start)
                if wave3_length < wave1_length and wave3_length < wave5_length:
                    return False
        
        # Wave 4 should not overlap with wave 1 (in most cases)
        if wave.wave_number == 4 and len(previous_waves) >= 1:
            wave1 = previous_waves[0]
            wave4_low = min(wave.price_start, wave.price_end)
            wave1_high = max(wave1.price_start, wave1.price_end)
            
            # Some overlap allowed in complex corrections
            if wave4_low < wave1_high * 0.9:  # Allow 10% tolerance
                return False
        
        return True
    
    def _validate_corrective_wave(self, wave: WaveStructure, previous_waves: List[WaveStructure]) -> bool:
        """Validate corrective wave according to Elliott Wave rules."""
        # Corrective waves should retrace reasonable portion of impulse
        if len(previous_waves) >= 1:
            last_impulse = None
            for prev_wave in reversed(previous_waves):
                if prev_wave.wave_type == "impulse":
                    last_impulse = prev_wave
                    break
            
            if last_impulse:
                impulse_range = abs(last_impulse.price_end - last_impulse.price_start)
                correction_range = abs(wave.price_end - wave.price_start)
                
                # Typical retracement should be 23.6% to 78.6%
                retracement_ratio = correction_range / impulse_range if impulse_range > 0 else 0
                if retracement_ratio < 0.236 or retracement_ratio > 0.786:
                    return False
        
        return True
    
    def _calculate_wave_statistics(self, waves: List[WaveStructure]) -> Dict[str, float]:
        """Calculate statistical measures of wave patterns."""
        if not waves:
            return {
                'avg_strength': 0.0,
                'avg_length': 0.0,
                'fibonacci_compliance': 0.0,
                'pattern_confidence': 0.0
            }
        
        # Average strength
        avg_strength = np.mean([w.strength for w in waves])
        
        # Average length
        avg_length = np.mean([w.end_idx - w.start_idx for w in waves])
        
        # Fibonacci compliance
        fib_compliance = self._calculate_fibonacci_compliance(waves)
        
        # Pattern confidence
        pattern_confidence = self._calculate_pattern_confidence(waves)
        
        return {
            'avg_strength': float(avg_strength),
            'avg_length': float(avg_length),
            'fibonacci_compliance': float(fib_compliance),
            'pattern_confidence': float(pattern_confidence)
        }
    
    def _calculate_fibonacci_compliance(self, waves: List[WaveStructure]) -> float:
        """Calculate how well waves comply with Fibonacci ratios."""
        if not waves:
            return 0.0
        
        compliant_count = 0
        total_count = 0
        
        for i, wave in enumerate(waves[1:], 1):  # Skip first wave
            if i < len(waves):
                prev_wave = waves[i-1]
                current_length = abs(wave.price_end - wave.price_start)
                prev_length = abs(prev_wave.price_end - prev_wave.price_start)
                
                if prev_length > 0:
                    ratio = current_length / prev_length
                    
                    # Check if ratio is close to any Fibonacci ratio
                    min_distance = min([abs(ratio - fib_ratio) for fib_ratio in self.elliott_ratios])
                    if min_distance < 0.1:  # Within 10% tolerance
                        compliant_count += 1
                    total_count += 1
        
        return compliant_count / total_count if total_count > 0 else 0.0
    
    def _calculate_pattern_confidence(self, waves: List[WaveStructure]) -> float:
        """Calculate overall pattern confidence."""
        if not waves:
            return 0.0
        
        # Factors contributing to confidence
        factors = []
        
        # Strength consistency
        strengths = [w.strength for w in waves]
        strength_consistency = 1.0 - (np.std(strengths) / np.mean(strengths)) if np.mean(strengths) > 0 else 0.0
        factors.append(max(0.0, min(1.0, strength_consistency)))
        
        # Wave count appropriateness (prefer 5 or 8 waves)
        wave_count = len(waves)
        count_score = 1.0 if wave_count in [5, 8] else max(0.5, 1.0 - abs(wave_count - 5) * 0.1)
        factors.append(count_score)
        
        # Alternation in corrective waves
        corrective_waves = [w for w in waves if w.wave_type == "corrective"]
        if len(corrective_waves) >= 2:
            alternation_score = self._calculate_alternation_score(corrective_waves)
            factors.append(alternation_score)
        else:
            factors.append(0.5)
        
        return np.mean(factors)
    
    def _calculate_alternation_score(self, corrective_waves: List[WaveStructure]) -> float:
        """Calculate alternation score for corrective waves."""
        if len(corrective_waves) < 2:
            return 0.5
        
        # Elliott Wave principle of alternation
        alternations = 0
        comparisons = 0
        
        for i in range(len(corrective_waves) - 1):
            wave1 = corrective_waves[i]
            wave2 = corrective_waves[i + 1]
            
            # Compare complexity (simple vs complex)
            length1 = wave1.end_idx - wave1.start_idx
            length2 = wave2.end_idx - wave2.start_idx
            
            # Different lengths suggest alternation
            if abs(length1 - length2) / max(length1, length2) > 0.3:
                alternations += 1
            
            comparisons += 1
        
        return alternations / comparisons if comparisons > 0 else 0.5
    
    def _perform_cluster_analysis(self, waves: List[WaveStructure]) -> Dict[str, Any]:
        """Perform clustering analysis on wave patterns."""
        if len(waves) < 3:
            return {'cluster_count': 0, 'dominant_pattern': 'insufficient_data'}
        
        try:
            # Prepare features for clustering
            features = []
            for wave in waves:
                wave_features = [
                    wave.strength,
                    wave.end_idx - wave.start_idx,  # length
                    abs(wave.price_end - wave.price_start),  # amplitude
                    wave.fibonacci_ratio,
                    wave.volume_profile
                ]
                features.append(wave_features)
            
            features_array = np.array(features)
            
            # Normalize features
            features_scaled = self.scaler.fit_transform(features_array)
            
            # Perform clustering
            cluster_labels = self.dbscan.fit_predict(features_scaled)
            
            # Analyze clusters
            unique_labels = set(cluster_labels)
            cluster_count = len(unique_labels) - (1 if -1 in unique_labels else 0)  # Exclude noise
            
            # Find dominant pattern
            if cluster_count > 0:
                label_counts = {label: list(cluster_labels).count(label) for label in unique_labels if label != -1}
                dominant_label = max(label_counts, key=label_counts.get)
                
                # Characterize dominant pattern
                dominant_waves = [waves[i] for i, label in enumerate(cluster_labels) if label == dominant_label]
                dominant_pattern = self._characterize_pattern(dominant_waves)
            else:
                dominant_pattern = 'no_clear_pattern'
            
            return {
                'cluster_count': cluster_count,
                'dominant_pattern': dominant_pattern,
                'cluster_labels': cluster_labels.tolist(),
                'noise_points': list(cluster_labels).count(-1)
            }
            
        except Exception as e:
            self.logger.warning(f"Cluster analysis failed: {str(e)}")
            return {'cluster_count': 0, 'dominant_pattern': 'analysis_failed'}
    
    def _characterize_pattern(self, waves: List[WaveStructure]) -> str:
        """Characterize a pattern based on wave properties."""
        if not waves:
            return 'empty'
        
        avg_strength = np.mean([w.strength for w in waves])
        avg_length = np.mean([w.end_idx - w.start_idx for w in waves])
        impulse_ratio = sum(1 for w in waves if w.wave_type == 'impulse') / len(waves)
        
        # Pattern classification
        if avg_strength > 1.5 and impulse_ratio > 0.6:
            return 'strong_trending'
        elif avg_strength < 0.5 and avg_length < 10:
            return 'choppy_sideways'
        elif impulse_ratio < 0.4:
            return 'corrective_complex'
        else:
            return 'normal_trending'
    
    def _generate_wave_predictions(self, waves: List[WaveStructure], 
                                  fractal_analysis: FractalAnalysis, 
                                  data: pd.DataFrame) -> Dict[str, Any]:
        """Generate predictions for next wave movements."""
        if not waves:
            return {
                'next_wave': 'unknown',
                'completion_probability': 0.0,
                'target_levels': []
            }
        
        last_wave = waves[-1]
        current_position = len(data) - 1
        
        # Determine next expected wave
        next_wave_type, next_wave_number = self._predict_next_wave(waves)
        
        # Calculate completion probability
        completion_prob = self._calculate_completion_probability(last_wave, current_position, fractal_analysis)
        
        # Calculate target levels
        target_levels = self._calculate_target_levels(waves, data)
        
        return {
            'next_wave': f"{next_wave_type}_{next_wave_number}",
            'completion_probability': completion_prob,
            'target_levels': target_levels
        }
    
    def _predict_next_wave(self, waves: List[WaveStructure]) -> Tuple[str, int]:
        """Predict the next wave type and number."""
        if not waves:
            return "impulse", 1
        
        last_wave = waves[-1]
        
        if last_wave.wave_type == "impulse":
            if last_wave.wave_number < 5:
                # Continue impulse sequence
                if last_wave.wave_number % 2 == 1:  # Odd wave (1, 3, 5) -> corrective
                    return "corrective", 1  # A wave
                else:  # Even wave (2, 4) -> impulse
                    return "impulse", last_wave.wave_number + 1
            else:
                # After wave 5, start corrective
                return "corrective", 1  # A wave
        else:  # corrective
            if last_wave.wave_number < 3:
                # Continue corrective sequence
                return "corrective", last_wave.wave_number + 1
            else:
                # After C wave, start new impulse
                return "impulse", 1
    
    def _calculate_completion_probability(self, last_wave: WaveStructure, 
                                        current_position: int, 
                                        fractal_analysis: FractalAnalysis) -> float:
        """Calculate probability that current wave is complete."""
        try:
            # Time component
            wave_length = last_wave.end_idx - last_wave.start_idx
            time_since_end = current_position - last_wave.end_idx
            time_ratio = time_since_end / wave_length if wave_length > 0 else 0
            
            # Minimum time requirement
            time_score = min(1.0, time_ratio / 0.5)  # 50% of wave length
            
            # Fractal stability
            stability_score = 1.0 - fractal_analysis.chaos_factor
            
            # Predictability component
            predict_score = fractal_analysis.predictability_index
            
            # Combined probability
            completion_prob = (time_score * 0.4 + stability_score * 0.3 + predict_score * 0.3)
            
            return max(0.0, min(1.0, completion_prob))
            
        except Exception:
            return 0.5
    
    def _calculate_target_levels(self, waves: List[WaveStructure], data: pd.DataFrame) -> List[float]:
        """Calculate potential target levels based on wave analysis."""
        if not waves:
            return []
        
        current_price = data['close'].iloc[-1]
        targets = []
        
        # Fibonacci extensions and retracements
        if len(waves) >= 2:
            last_wave = waves[-1]
            prev_wave = waves[-2]
            
            wave_range = abs(last_wave.price_end - last_wave.price_start)
            
            # Calculate Fibonacci targets
            for ratio in self.elliott_ratios:
                if last_wave.price_end > last_wave.price_start:  # Upward wave
                    target = last_wave.price_end + wave_range * ratio
                else:  # Downward wave
                    target = last_wave.price_end - wave_range * ratio
                
                targets.append(float(target))
        
        # Remove targets too close to current price
        min_distance = current_price * 0.01  # 1% minimum distance
        filtered_targets = [t for t in targets if abs(t - current_price) > min_distance]
        
        # Sort and return top 5 targets
        filtered_targets.sort(key=lambda x: abs(x - current_price))
        return filtered_targets[:5]
    
    def _calculate_support_resistance(self, waves: List[WaveStructure], data: pd.DataFrame) -> Dict[str, List[float]]:
        """Calculate support and resistance levels from wave analysis."""
        support_levels = []
        resistance_levels = []
        current_price = data['close'].iloc[-1]
        
        for wave in waves:
            wave_high = max(wave.price_start, wave.price_end)
            wave_low = min(wave.price_start, wave.price_end)
            
            if wave_high < current_price:
                support_levels.append(wave_high)
            elif wave_low > current_price:
                resistance_levels.append(wave_low)
        
        # Remove duplicates and sort
        support_levels = sorted(list(set(support_levels)), reverse=True)[:5]
        resistance_levels = sorted(list(set(resistance_levels)))[:5]
        
        return {
            'support': support_levels,
            'resistance': resistance_levels
        }
    
    def _determine_current_position(self, waves: List[WaveStructure], data_length: int) -> Dict[str, Any]:
        """Determine current position in Elliott Wave cycle."""
        if not waves:
            return {'wave': 'unknown', 'cycle_position': 0, 'progress': 0.0}
        
        last_wave = waves[-1]
        current_idx = data_length - 1
        
        # Calculate progress within current wave
        wave_start = last_wave.end_idx
        progress = (current_idx - wave_start) / max(1, last_wave.end_idx - last_wave.start_idx)
        
        return {
            'wave': f"{last_wave.wave_type}_{last_wave.wave_number}",
            'cycle_position': len(waves) % 8,  # Position in 8-wave cycle
            'progress': min(1.0, max(0.0, progress))
        }
    
    def _determine_signal_type(self, waves: List[WaveStructure], fractal_analysis: FractalAnalysis) -> SignalType:
        """Determine signal type based on wave analysis."""
        if not waves:
            return SignalType.NEUTRAL
        
        last_wave = waves[-1]
        trend_strength = sum(w.strength for w in waves[-3:]) / 3 if len(waves) >= 3 else last_wave.strength
        
        # Strong signals for high-confidence patterns
        if (last_wave.wave_type == "impulse" and 
            last_wave.wave_number in [1, 3, 5] and 
            trend_strength > 1.5 and
            fractal_analysis.predictability_index > 0.7):
            
            if last_wave.price_end > last_wave.price_start:
                return SignalType.STRONG_BUY
            else:
                return SignalType.STRONG_SELL
        
        # Regular signals
        if last_wave.wave_type == "impulse" and trend_strength > 1.0:
            if last_wave.price_end > last_wave.price_start:
                return SignalType.BUY
            else:
                return SignalType.SELL
        
        return SignalType.NEUTRAL
    
    def _update_historical_data(self, waves: List[WaveStructure], fractal_analysis: FractalAnalysis):
        """Update historical data for future analysis."""
        self.identified_waves = waves
        self.fractal_history.append(fractal_analysis)
        
        # Keep only recent history
        max_history = 100
        if len(self.fractal_history) > max_history:
            self.fractal_history = self.fractal_history[-max_history:]
    
    def get_signal_type(self) -> SignalType:
        """Get the current signal type."""
        return getattr(self, '_last_signal_type', SignalType.NEUTRAL)
    
    def get_signal_strength(self) -> float:
        """Get the current signal strength."""
        if self.identified_waves:
            return self.identified_waves[-1].strength
        return 0.0