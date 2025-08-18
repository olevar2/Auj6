"""
Wave Point Indicator

Advanced implementation to identify critical wave turning points and pivot levels
using sophisticated mathematical algorithms for the humanitarian trading platform.

This indicator employs multiple analytical techniques including differential geometry,
signal processing, machine learning, and chaos theory to identify high-probability
reversal points in Elliott Wave structures.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy import signal, optimize
from scipy.signal import find_peaks, argrelextrema, savgol_filter, hilbert
from scipy.interpolate import CubicSpline, UnivariateSpline
from scipy.stats import zscore
import warnings
warnings.filterwarnings('ignore')

from ..base.standard_indicator import StandardIndicatorInterface
from ....core.signal_type import SignalType
from ....core.exceptions import IndicatorCalculationError


@dataclass
class WavePointConfig:
    """Configuration for Wave Point Indicator."""
    analysis_window: int = 100
    pivot_sensitivity: float = 0.01
    confirmation_periods: int = 3
    min_point_distance: int = 5
    curvature_threshold: float = 0.001
    volume_weight: float = 0.3
    fibonacci_tolerance: float = 0.05
    clustering_eps: float = 0.02


@dataclass
class WavePoint:
    """Represents a critical wave turning point."""
    index: int
    price: float
    timestamp: Any
    point_type: str  # 'peak', 'trough', 'inflection'
    strength: float
    confidence: float
    volume_confirmation: float
    curvature: float
    fibonacci_level: Optional[float]
    technical_confluence: int
    elliott_wave_position: Optional[str]


@dataclass
class PivotLevel:
    """Represents a significant pivot level."""
    price: float
    strength: float
    touches: int
    first_touch: int
    last_touch: int
    level_type: str  # 'support', 'resistance', 'pivot'
    fibonacci_ratio: Optional[float]


@dataclass
class GeometricAnalysis:
    """Geometric analysis of price structure."""
    curvature: np.ndarray
    torsion: np.ndarray
    gradient: np.ndarray
    laplacian: np.ndarray
    critical_points: List[int]


class WavePointIndicator(StandardIndicatorInterface):
    """
    Advanced Wave Point Indicator for identifying critical turning points.
    
    This indicator combines multiple analytical approaches to identify high-probability
    wave reversal points with sophisticated mathematical precision.
    """
    
    def __init__(self, config: Optional[WavePointConfig] = None):
        super().__init__()
        self.config = config or WavePointConfig()
        self.logger = logging.getLogger(__name__)
        
        # Analysis components
        self.scaler = StandardScaler()
        self.clustering = DBSCAN(eps=self.config.clustering_eps, min_samples=3)
        
        # Historical data
        self.identified_points: List[WavePoint] = []
        self.pivot_levels: List[PivotLevel] = []
        self.geometric_history: List[GeometricAnalysis] = []
        
        # Fibonacci levels for retracement analysis
        self.fibonacci_levels = [0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618, 2.618]
        
        # Technical indicators cache
        self.technical_cache: Dict[str, np.ndarray] = {}
        
    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Identify critical wave turning points and pivot levels.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary containing wave points and analysis results
        """
        try:
            if len(data) < self.config.analysis_window:
                raise IndicatorCalculationError(
                    f"Insufficient data: need at least {self.config.analysis_window} periods"
                )
            
            # Perform geometric analysis
            geometric_analysis = self._perform_geometric_analysis(data)
            
            # Identify candidate turning points
            candidate_points = self._identify_candidate_points(data, geometric_analysis)
            
            # Apply strength and confidence scoring
            scored_points = self._score_turning_points(candidate_points, data, geometric_analysis)
            
            # Filter and validate points
            validated_points = self._validate_turning_points(scored_points, data)
            
            # Identify pivot levels
            pivot_levels = self._identify_pivot_levels(data, validated_points)
            
            # Perform confluence analysis
            confluence_analysis = self._analyze_technical_confluence(data, validated_points)
            
            # Generate predictions
            predictions = self._generate_turning_point_predictions(validated_points, data)
            
            # Calculate Elliott Wave positioning
            elliott_positioning = self._analyze_elliott_wave_positioning(validated_points, data)
            
            result = {
                'turning_points': [point.__dict__ for point in validated_points],
                'pivot_levels': [level.__dict__ for level in pivot_levels],
                'active_points_count': len(validated_points),
                'strong_points_count': len([p for p in validated_points if p.strength > 0.7]),
                'geometric_curvature': geometric_analysis.curvature.tolist()[-10:],  # Recent curvature
                'critical_points': geometric_analysis.critical_points,
                'confluence_zones': confluence_analysis['zones'],
                'confluence_strength': confluence_analysis['strength'],
                'next_probable_turning_point': predictions['next_point'],
                'turning_point_probability': predictions['probability'],
                'support_levels': [level for level in pivot_levels if level.level_type == 'support'],
                'resistance_levels': [level for level in pivot_levels if level.level_type == 'resistance'],
                'fibonacci_confluences': self._find_fibonacci_confluences(validated_points, data),
                'wave_structure_health': self._assess_wave_structure_health(validated_points),
                'elliott_wave_count': elliott_positioning['wave_count'],
                'current_wave_position': elliott_positioning['current_position'],
                'wave_completion_ratio': elliott_positioning['completion_ratio'],
                'signal_type': self._determine_signal_type(validated_points, pivot_levels, predictions),
                'raw_data': {
                    'geometric_analysis': {
                        'curvature': geometric_analysis.curvature.tolist(),
                        'gradient': geometric_analysis.gradient.tolist(),
                        'critical_points': geometric_analysis.critical_points
                    },
                    'technical_confluence': confluence_analysis
                }
            }
            
            # Update historical data
            self._update_historical_data(validated_points, pivot_levels, geometric_analysis)
            
            self.logger.info(f"Wave Point analysis completed - {len(validated_points)} turning points identified")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in Wave Point Indicator: {str(e)}")
            raise IndicatorCalculationError(f"Wave Point analysis failed: {str(e)}")
    
    def _perform_geometric_analysis(self, data: pd.DataFrame) -> GeometricAnalysis:
        """Perform sophisticated geometric analysis of price structure."""
        try:
            recent_data = data.tail(self.config.analysis_window)
            prices = recent_data['close'].values
            
            # Smooth the data to reduce noise
            smoothed_prices = savgol_filter(prices, window_length=min(11, len(prices)//2*2+1), polyorder=3)
            
            # Calculate first derivative (gradient/slope)
            gradient = np.gradient(smoothed_prices)
            
            # Calculate second derivative (curvature)
            curvature = np.gradient(gradient)
            
            # Calculate Laplacian (divergence of gradient)
            laplacian = np.gradient(curvature)
            
            # Calculate torsion (for 3D analysis, using volume as third dimension)
            if 'volume' in recent_data.columns:
                volume_normalized = (recent_data['volume'].values - recent_data['volume'].mean()) / recent_data['volume'].std()
                torsion = self._calculate_torsion(smoothed_prices, gradient, volume_normalized)
            else:
                torsion = np.zeros_like(curvature)
            
            # Identify critical points (where gradient changes sign significantly)
            critical_points = self._find_critical_points(gradient, curvature)
            
            return GeometricAnalysis(
                curvature=curvature,
                torsion=torsion,
                gradient=gradient,
                laplacian=laplacian,
                critical_points=critical_points
            )
            
        except Exception as e:
            self.logger.warning(f"Geometric analysis failed: {str(e)}")
            return GeometricAnalysis(
                curvature=np.zeros(len(data.tail(self.config.analysis_window))),
                torsion=np.zeros(len(data.tail(self.config.analysis_window))),
                gradient=np.zeros(len(data.tail(self.config.analysis_window))),
                laplacian=np.zeros(len(data.tail(self.config.analysis_window))),
                critical_points=[]
            )
    
    def _calculate_torsion(self, prices: np.ndarray, gradient: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Calculate torsion for 3D price-volume analysis."""
        try:
            # Create 3D curve: (time, price, volume)
            time_coords = np.arange(len(prices))
            
            # Calculate second derivatives
            d2price_dt2 = np.gradient(gradient)
            d2volume_dt2 = np.gradient(np.gradient(volume))
            
            # Cross product for torsion calculation
            cross_product = gradient * d2volume_dt2 - np.gradient(volume) * d2price_dt2
            
            # Normalize torsion
            velocity_magnitude = np.sqrt(gradient**2 + np.gradient(volume)**2) + 1e-8
            torsion = cross_product / (velocity_magnitude**3)
            
            return torsion
            
        except Exception:
            return np.zeros_like(prices)
    
    def _find_critical_points(self, gradient: np.ndarray, curvature: np.ndarray) -> List[int]:
        """Find critical points where significant directional changes occur."""
        critical_points = []
        
        # Points where gradient is near zero (local extrema)
        gradient_threshold = np.std(gradient) * 0.5
        near_zero_gradient = np.where(np.abs(gradient) < gradient_threshold)[0]
        
        # Points where curvature is significant (inflection points)
        curvature_threshold = np.std(curvature) * 1.5
        high_curvature = np.where(np.abs(curvature) > curvature_threshold)[0]
        
        # Combine and deduplicate
        all_candidates = np.concatenate([near_zero_gradient, high_curvature])
        
        # Remove points too close to each other
        if len(all_candidates) > 0:
            sorted_candidates = np.sort(all_candidates)
            filtered_points = [sorted_candidates[0]]
            
            for point in sorted_candidates[1:]:
                if point - filtered_points[-1] >= self.config.min_point_distance:
                    filtered_points.append(point)
            
            critical_points = filtered_points
        
        return critical_points
    
    def _identify_candidate_points(self, data: pd.DataFrame, geometric_analysis: GeometricAnalysis) -> List[WavePoint]:
        """Identify candidate turning points using multiple methods."""
        candidate_points = []
        recent_data = data.tail(self.config.analysis_window)
        
        # Method 1: Peak/Trough detection
        peaks_troughs = self._find_peaks_and_troughs(recent_data)
        candidate_points.extend(peaks_troughs)
        
        # Method 2: Geometric critical points
        geometric_points = self._convert_geometric_points(geometric_analysis, recent_data)
        candidate_points.extend(geometric_points)
        
        # Method 3: Volume-price divergence points
        divergence_points = self._find_divergence_points(recent_data)
        candidate_points.extend(divergence_points)
        
        # Method 4: Technical indicator confluence points
        technical_points = self._find_technical_confluence_points(recent_data)
        candidate_points.extend(technical_points)
        
        # Remove duplicates and sort by index
        unique_points = self._deduplicate_points(candidate_points)
        
        return unique_points
    
    def _find_peaks_and_troughs(self, data: pd.DataFrame) -> List[WavePoint]:
        """Find peaks and troughs using advanced signal processing."""
        points = []
        
        high_prices = data['high'].values
        low_prices = data['low'].values
        close_prices = data['close'].values
        
        # Find peaks in high prices
        peaks, peak_properties = find_peaks(
            high_prices, 
            distance=self.config.min_point_distance,
            prominence=np.std(high_prices) * self.config.pivot_sensitivity
        )
        
        # Find troughs in low prices
        troughs, trough_properties = find_peaks(
            -low_prices,
            distance=self.config.min_point_distance,
            prominence=np.std(low_prices) * self.config.pivot_sensitivity
        )
        
        # Create WavePoint objects for peaks
        for peak_idx in peaks:
            if peak_idx < len(data):
                point = WavePoint(
                    index=peak_idx,
                    price=high_prices[peak_idx],
                    timestamp=data.index[peak_idx] if hasattr(data.index, '__getitem__') else peak_idx,
                    point_type='peak',
                    strength=0.0,  # Will be calculated later
                    confidence=0.0,  # Will be calculated later
                    volume_confirmation=0.0,
                    curvature=0.0,
                    fibonacci_level=None,
                    technical_confluence=0,
                    elliott_wave_position=None
                )
                points.append(point)
        
        # Create WavePoint objects for troughs
        for trough_idx in troughs:
            if trough_idx < len(data):
                point = WavePoint(
                    index=trough_idx,
                    price=low_prices[trough_idx],
                    timestamp=data.index[trough_idx] if hasattr(data.index, '__getitem__') else trough_idx,
                    point_type='trough',
                    strength=0.0,  # Will be calculated later
                    confidence=0.0,  # Will be calculated later
                    volume_confirmation=0.0,
                    curvature=0.0,
                    fibonacci_level=None,
                    technical_confluence=0,
                    elliott_wave_position=None
                )
                points.append(point)
        
        return points
    
    def _convert_geometric_points(self, geometric_analysis: GeometricAnalysis, data: pd.DataFrame) -> List[WavePoint]:
        """Convert geometric critical points to WavePoint objects."""
        points = []
        
        for critical_idx in geometric_analysis.critical_points:
            if 0 <= critical_idx < len(data):
                # Determine point type based on curvature
                curvature_val = geometric_analysis.curvature[critical_idx] if critical_idx < len(geometric_analysis.curvature) else 0.0
                
                if curvature_val > self.config.curvature_threshold:
                    point_type = 'peak'
                elif curvature_val < -self.config.curvature_threshold:
                    point_type = 'trough'
                else:
                    point_type = 'inflection'
                
                point = WavePoint(
                    index=critical_idx,
                    price=data['close'].iloc[critical_idx],
                    timestamp=data.index[critical_idx] if hasattr(data.index, '__getitem__') else critical_idx,
                    point_type=point_type,
                    strength=0.0,
                    confidence=0.0,
                    volume_confirmation=0.0,
                    curvature=float(curvature_val),
                    fibonacci_level=None,
                    technical_confluence=0,
                    elliott_wave_position=None
                )
                points.append(point)
        
        return points
    
    def _find_divergence_points(self, data: pd.DataFrame) -> List[WavePoint]:
        """Find points where price and volume diverge significantly."""
        points = []
        
        if 'volume' not in data.columns:
            return points
        
        try:
            # Calculate price and volume momentum
            price_momentum = data['close'].pct_change().rolling(window=5).mean()
            volume_momentum = data['volume'].pct_change().rolling(window=5).mean()
            
            # Find divergence points
            for i in range(5, len(data) - 5):
                price_trend = price_momentum.iloc[i-2:i+3].mean()
                volume_trend = volume_momentum.iloc[i-2:i+3].mean()
                
                # Significant divergence
                if abs(price_trend) > 0.001 and abs(volume_trend) > 0.1:
                    if (price_trend > 0 and volume_trend < 0) or (price_trend < 0 and volume_trend > 0):
                        point = WavePoint(
                            index=i,
                            price=data['close'].iloc[i],
                            timestamp=data.index[i] if hasattr(data.index, '__getitem__') else i,
                            point_type='divergence',
                            strength=0.0,
                            confidence=0.0,
                            volume_confirmation=abs(volume_trend),
                            curvature=0.0,
                            fibonacci_level=None,
                            technical_confluence=0,
                            elliott_wave_position=None
                        )
                        points.append(point)
        
        except Exception as e:
            self.logger.warning(f"Divergence point detection failed: {str(e)}")
        
        return points
    
    def _find_technical_confluence_points(self, data: pd.DataFrame) -> List[WavePoint]:
        """Find points where multiple technical indicators converge."""
        points = []
        
        try:
            # Calculate technical indicators
            rsi = self._calculate_rsi(data['close'])
            macd, macd_signal = self._calculate_macd(data['close'])
            bb_upper, bb_lower = self._calculate_bollinger_bands(data['close'])
            
            # Find confluence points
            for i in range(20, len(data)):
                confluence_count = 0
                
                # RSI overbought/oversold
                if rsi[i] > 70 or rsi[i] < 30:
                    confluence_count += 1
                
                # MACD signal crossover
                if i > 0 and ((macd[i] > macd_signal[i] and macd[i-1] <= macd_signal[i-1]) or
                             (macd[i] < macd_signal[i] and macd[i-1] >= macd_signal[i-1])):
                    confluence_count += 1
                
                # Bollinger Band touch
                if data['close'].iloc[i] >= bb_upper[i] or data['close'].iloc[i] <= bb_lower[i]:
                    confluence_count += 1
                
                # Require at least 2 confluences
                if confluence_count >= 2:
                    point = WavePoint(
                        index=i,
                        price=data['close'].iloc[i],
                        timestamp=data.index[i] if hasattr(data.index, '__getitem__') else i,
                        point_type='confluence',
                        strength=0.0,
                        confidence=0.0,
                        volume_confirmation=0.0,
                        curvature=0.0,
                        fibonacci_level=None,
                        technical_confluence=confluence_count,
                        elliott_wave_position=None
                    )
                    points.append(point)
        
        except Exception as e:
            self.logger.warning(f"Technical confluence detection failed: {str(e)}")
        
        return points
    
    def _deduplicate_points(self, points: List[WavePoint]) -> List[WavePoint]:
        """Remove duplicate points that are too close to each other."""
        if not points:
            return []
        
        # Sort by index
        sorted_points = sorted(points, key=lambda p: p.index)
        
        deduplicated = [sorted_points[0]]
        
        for point in sorted_points[1:]:
            # Check if point is far enough from the last added point
            if point.index - deduplicated[-1].index >= self.config.min_point_distance:
                deduplicated.append(point)
            else:
                # Keep the point with higher confluence or strength
                if (point.technical_confluence > deduplicated[-1].technical_confluence or
                    point.strength > deduplicated[-1].strength):
                    deduplicated[-1] = point
        
        return deduplicated    
    def _score_turning_points(self, candidate_points: List[WavePoint], data: pd.DataFrame, 
                             geometric_analysis: GeometricAnalysis) -> List[WavePoint]:
        """Score turning points based on multiple criteria."""
        scored_points = []
        
        for point in candidate_points:
            try:
                # Calculate strength score
                strength = self._calculate_point_strength(point, data, geometric_analysis)
                
                # Calculate confidence score
                confidence = self._calculate_point_confidence(point, data)
                
                # Calculate volume confirmation
                volume_confirmation = self._calculate_volume_confirmation(point, data)
                
                # Update point with scores
                point.strength = strength
                point.confidence = confidence
                point.volume_confirmation = volume_confirmation
                
                # Calculate Fibonacci level if applicable
                point.fibonacci_level = self._calculate_fibonacci_level(point, data)
                
                scored_points.append(point)
                
            except Exception as e:
                self.logger.warning(f"Point scoring failed for index {point.index}: {str(e)}")
                scored_points.append(point)  # Keep point with default scores
        
        return scored_points
    
    def _calculate_point_strength(self, point: WavePoint, data: pd.DataFrame, 
                                 geometric_analysis: GeometricAnalysis) -> float:
        """Calculate the strength of a turning point."""
        try:
            strength_factors = []
            
            # Price volatility around the point
            window_start = max(0, point.index - 5)
            window_end = min(len(data), point.index + 6)
            local_data = data.iloc[window_start:window_end]
            
            if len(local_data) > 1:
                local_volatility = local_data['close'].std()
                avg_volatility = data['close'].rolling(window=20).std().mean()
                volatility_factor = local_volatility / (avg_volatility + 1e-8)
                strength_factors.append(min(2.0, volatility_factor))
            
            # Curvature strength
            if point.index < len(geometric_analysis.curvature):
                curvature_strength = abs(geometric_analysis.curvature[point.index])
                normalized_curvature = curvature_strength / (np.std(geometric_analysis.curvature) + 1e-8)
                strength_factors.append(min(2.0, normalized_curvature))
            
            # Volume spike
            if 'volume' in data.columns and point.index < len(data):
                local_volume = data['volume'].iloc[point.index]
                avg_volume = data['volume'].rolling(window=20).mean().iloc[point.index]
                volume_factor = local_volume / (avg_volume + 1e-8)
                strength_factors.append(min(2.0, volume_factor * self.config.volume_weight))
            
            # Technical confluence contribution
            confluence_factor = point.technical_confluence / 5.0  # Normalize to max 5 confluences
            strength_factors.append(confluence_factor)
            
            # Calculate weighted average
            if strength_factors:
                return float(np.mean(strength_factors))
            else:
                return 0.5
                
        except Exception:
            return 0.5
    
    def _calculate_point_confidence(self, point: WavePoint, data: pd.DataFrame) -> float:
        """Calculate confidence in the turning point identification."""
        try:
            confidence_factors = []
            
            # Time since point (older points have more confirmation)
            current_index = len(data) - 1
            time_factor = min(1.0, (current_index - point.index) / self.config.confirmation_periods)
            confidence_factors.append(time_factor)
            
            # Price movement confirmation
            if point.index < len(data) - 1:
                if point.point_type == 'peak':
                    # Check if price has moved down from peak
                    post_peak_low = data['low'].iloc[point.index:].min()
                    confirmation = (point.price - post_peak_low) / point.price
                elif point.point_type == 'trough':
                    # Check if price has moved up from trough
                    post_trough_high = data['high'].iloc[point.index:].max()
                    confirmation = (post_trough_high - point.price) / point.price
                else:
                    confirmation = 0.5
                
                confidence_factors.append(min(1.0, confirmation * 10))  # Scale confirmation
            
            # Geometric consistency
            if abs(point.curvature) > self.config.curvature_threshold:
                curvature_confidence = min(1.0, abs(point.curvature) / self.config.curvature_threshold)
                confidence_factors.append(curvature_confidence)
            
            return float(np.mean(confidence_factors)) if confidence_factors else 0.5
            
        except Exception:
            return 0.5
    
    def _calculate_volume_confirmation(self, point: WavePoint, data: pd.DataFrame) -> float:
        """Calculate volume confirmation for the turning point."""
        try:
            if 'volume' not in data.columns or point.index >= len(data):
                return 0.5
            
            # Volume at the turning point
            point_volume = data['volume'].iloc[point.index]
            
            # Average volume around the point
            window_start = max(0, point.index - 5)
            window_end = min(len(data), point.index + 6)
            avg_volume = data['volume'].iloc[window_start:window_end].mean()
            
            # Volume confirmation (higher volume = higher confirmation)
            if avg_volume > 0:
                volume_ratio = point_volume / avg_volume
                return float(min(1.0, volume_ratio / 2.0))  # Normalize to [0, 1]
            
            return 0.5
            
        except Exception:
            return 0.5
    
    def _calculate_fibonacci_level(self, point: WavePoint, data: pd.DataFrame) -> Optional[float]:
        """Calculate if point aligns with Fibonacci retracement levels."""
        try:
            # Find recent significant high and low
            lookback = min(50, len(data))
            recent_data = data.tail(lookback)
            
            recent_high = recent_data['high'].max()
            recent_low = recent_data['low'].min()
            
            if recent_high == recent_low:
                return None
            
            # Calculate retracement level
            retracement = (point.price - recent_low) / (recent_high - recent_low)
            
            # Find closest Fibonacci level
            closest_fib = None
            min_distance = float('inf')
            
            for fib_level in self.fibonacci_levels:
                distance = abs(retracement - fib_level)
                if distance < min_distance and distance < self.config.fibonacci_tolerance:
                    min_distance = distance
                    closest_fib = fib_level
            
            return closest_fib
            
        except Exception:
            return None
    
    def _validate_turning_points(self, scored_points: List[WavePoint], data: pd.DataFrame) -> List[WavePoint]:
        """Validate and filter turning points based on quality criteria."""
        validated_points = []
        
        for point in scored_points:
            # Minimum strength threshold
            if point.strength < 0.3:
                continue
            
            # Minimum confidence threshold
            if point.confidence < 0.3:
                continue
            
            # Ensure point is not too close to data boundary
            if point.index < 5 or point.index > len(data) - 5:
                continue
            
            validated_points.append(point)
        
        return validated_points
    
    def _identify_pivot_levels(self, data: pd.DataFrame, turning_points: List[WavePoint]) -> List[PivotLevel]:
        """Identify significant pivot levels from turning points."""
        pivot_levels = []
        
        if not turning_points:
            return pivot_levels
        
        # Group points by price levels using clustering
        try:
            prices = np.array([point.price for point in turning_points]).reshape(-1, 1)
            
            if len(prices) >= 3:
                # Normalize prices for clustering
                normalized_prices = self.scaler.fit_transform(prices)
                
                # Perform clustering
                clusters = self.clustering.fit_predict(normalized_prices)
                
                # Create pivot levels from clusters
                unique_clusters = set(clusters)
                for cluster_id in unique_clusters:
                    if cluster_id == -1:  # Skip noise points
                        continue
                    
                    cluster_points = [turning_points[i] for i, c in enumerate(clusters) if c == cluster_id]
                    
                    if len(cluster_points) >= 2:  # Require at least 2 touches
                        cluster_prices = [p.price for p in cluster_points]
                        avg_price = np.mean(cluster_prices)
                        strength = np.mean([p.strength for p in cluster_points])
                        
                        # Determine level type
                        peak_count = sum(1 for p in cluster_points if p.point_type == 'peak')
                        trough_count = sum(1 for p in cluster_points if p.point_type == 'trough')
                        
                        if peak_count > trough_count:
                            level_type = 'resistance'
                        elif trough_count > peak_count:
                            level_type = 'support'
                        else:
                            level_type = 'pivot'
                        
                        # Calculate touches and timing
                        touches = len(cluster_points)
                        indices = [p.index for p in cluster_points]
                        first_touch = min(indices)
                        last_touch = max(indices)
                        
                        # Check for Fibonacci ratio
                        fibonacci_ratio = self._find_fibonacci_ratio_for_level(avg_price, data)
                        
                        pivot_level = PivotLevel(
                            price=float(avg_price),
                            strength=float(strength),
                            touches=touches,
                            first_touch=first_touch,
                            last_touch=last_touch,
                            level_type=level_type,
                            fibonacci_ratio=fibonacci_ratio
                        )
                        
                        pivot_levels.append(pivot_level)
        
        except Exception as e:
            self.logger.warning(f"Pivot level identification failed: {str(e)}")
        
        return pivot_levels
    
    def _find_fibonacci_ratio_for_level(self, price: float, data: pd.DataFrame) -> Optional[float]:
        """Find Fibonacci ratio for a price level."""
        try:
            # Find significant swing high and low
            lookback = min(100, len(data))
            recent_data = data.tail(lookback)
            
            swing_high = recent_data['high'].max()
            swing_low = recent_data['low'].min()
            
            if swing_high == swing_low:
                return None
            
            # Calculate ratio
            ratio = (price - swing_low) / (swing_high - swing_low)
            
            # Find closest Fibonacci level
            for fib_level in self.fibonacci_levels:
                if abs(ratio - fib_level) < self.config.fibonacci_tolerance:
                    return fib_level
            
            return None
            
        except Exception:
            return None
    
    def _analyze_technical_confluence(self, data: pd.DataFrame, turning_points: List[WavePoint]) -> Dict[str, Any]:
        """Analyze technical confluence around turning points."""
        try:
            confluence_zones = []
            
            # Calculate technical indicators
            rsi = self._calculate_rsi(data['close'])
            sma20 = data['close'].rolling(window=20).mean()
            sma50 = data['close'].rolling(window=50).mean()
            
            for point in turning_points:
                if point.index < len(data):
                    confluence_count = 0
                    confluence_details = []
                    
                    # RSI confluence
                    if point.index < len(rsi):
                        if (point.point_type == 'peak' and rsi[point.index] > 70) or \
                           (point.point_type == 'trough' and rsi[point.index] < 30):
                            confluence_count += 1
                            confluence_details.append('RSI_extremes')
                    
                    # Moving average confluence
                    if point.index < len(sma20) and point.index < len(sma50):
                        price = data['close'].iloc[point.index]
                        sma20_val = sma20.iloc[point.index]
                        sma50_val = sma50.iloc[point.index]
                        
                        if abs(price - sma20_val) / price < 0.01:  # Within 1% of SMA20
                            confluence_count += 1
                            confluence_details.append('SMA20_confluence')
                        
                        if abs(price - sma50_val) / price < 0.01:  # Within 1% of SMA50
                            confluence_count += 1
                            confluence_details.append('SMA50_confluence')
                    
                    # Fibonacci confluence
                    if point.fibonacci_level is not None:
                        confluence_count += 1
                        confluence_details.append(f'Fibonacci_{point.fibonacci_level}')
                    
                    if confluence_count >= 2:  # Significant confluence
                        confluence_zones.append({
                            'index': point.index,
                            'price': point.price,
                            'confluence_count': confluence_count,
                            'details': confluence_details,
                            'strength': confluence_count / 5.0  # Normalize
                        })
            
            # Overall confluence strength
            avg_strength = np.mean([zone['strength'] for zone in confluence_zones]) if confluence_zones else 0.0
            
            return {
                'zones': confluence_zones,
                'strength': float(avg_strength),
                'zone_count': len(confluence_zones)
            }
            
        except Exception as e:
            self.logger.warning(f"Technical confluence analysis failed: {str(e)}")
            return {'zones': [], 'strength': 0.0, 'zone_count': 0}
    
    def _generate_turning_point_predictions(self, turning_points: List[WavePoint], 
                                          data: pd.DataFrame) -> Dict[str, Any]:
        """Generate predictions for next probable turning points."""
        try:
            if len(turning_points) < 3:
                return {'next_point': None, 'probability': 0.0}
            
            # Analyze patterns in turning points
            recent_points = turning_points[-5:]  # Last 5 turning points
            
            # Calculate average distance between turning points
            distances = []
            for i in range(1, len(recent_points)):
                distance = recent_points[i].index - recent_points[i-1].index
                distances.append(distance)
            
            avg_distance = np.mean(distances) if distances else 20
            
            # Predict next turning point location
            last_point = recent_points[-1]
            predicted_index = last_point.index + int(avg_distance)
            
            # Estimate probability based on pattern consistency
            distance_consistency = 1.0 - (np.std(distances) / (avg_distance + 1e-8)) if distances else 0.5
            strength_consistency = np.std([p.strength for p in recent_points])
            
            probability = (distance_consistency + (1.0 - strength_consistency)) / 2
            probability = max(0.0, min(1.0, probability))
            
            # Predict price level
            if predicted_index < len(data):
                predicted_price = data['close'].iloc[-1]  # Use current price as base
            else:
                predicted_price = last_point.price
            
            # Predict point type (alternating pattern)
            predicted_type = 'trough' if last_point.point_type == 'peak' else 'peak'
            
            next_point = {
                'predicted_index': int(predicted_index),
                'predicted_price': float(predicted_price),
                'predicted_type': predicted_type,
                'confidence': float(probability)
            }
            
            return {
                'next_point': next_point,
                'probability': float(probability)
            }
            
        except Exception as e:
            self.logger.warning(f"Turning point prediction failed: {str(e)}")
            return {'next_point': None, 'probability': 0.0}
    
    def _analyze_elliott_wave_positioning(self, turning_points: List[WavePoint], 
                                        data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze Elliott Wave positioning of turning points."""
        try:
            if len(turning_points) < 5:
                return {
                    'wave_count': len(turning_points),
                    'current_position': 'insufficient_data',
                    'completion_ratio': 0.0
                }
            
            # Take last 8 points for complete Elliott cycle analysis
            recent_points = turning_points[-8:] if len(turning_points) >= 8 else turning_points
            
            # Assign Elliott Wave labels
            for i, point in enumerate(recent_points):
                cycle_position = i % 8  # 5 impulse + 3 corrective
                
                if cycle_position < 5:
                    # Impulse waves (1, 2, 3, 4, 5)
                    wave_number = cycle_position + 1
                    point.elliott_wave_position = f"Wave_{wave_number}"
                else:
                    # Corrective waves (A, B, C)
                    corrective_labels = ['A', 'B', 'C']
                    wave_label = corrective_labels[cycle_position - 5]
                    point.elliott_wave_position = f"Wave_{wave_label}"
            
            # Analyze current position
            if recent_points:
                last_point = recent_points[-1]
                current_position = last_point.elliott_wave_position
                
                # Calculate completion ratio for current cycle
                cycle_progress = (len(recent_points) % 8) / 8.0
                completion_ratio = cycle_progress
            else:
                current_position = 'unknown'
                completion_ratio = 0.0
            
            return {
                'wave_count': len(turning_points),
                'current_position': current_position,
                'completion_ratio': float(completion_ratio)
            }
            
        except Exception as e:
            self.logger.warning(f"Elliott Wave analysis failed: {str(e)}")
            return {
                'wave_count': len(turning_points),
                'current_position': 'analysis_failed',
                'completion_ratio': 0.0
            }
    
    def _find_fibonacci_confluences(self, turning_points: List[WavePoint], data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Find Fibonacci confluences among turning points."""
        confluences = []
        
        try:
            # Group points by Fibonacci levels
            fib_groups = {}
            for point in turning_points:
                if point.fibonacci_level is not None:
                    if point.fibonacci_level not in fib_groups:
                        fib_groups[point.fibonacci_level] = []
                    fib_groups[point.fibonacci_level].append(point)
            
            # Identify confluences (multiple points at same Fibonacci level)
            for fib_level, points in fib_groups.items():
                if len(points) >= 2:
                    avg_price = np.mean([p.price for p in points])
                    avg_strength = np.mean([p.strength for p in points])
                    
                    confluence = {
                        'fibonacci_level': float(fib_level),
                        'price_level': float(avg_price),
                        'point_count': len(points),
                        'strength': float(avg_strength),
                        'indices': [p.index for p in points]
                    }
                    confluences.append(confluence)
            
            # Sort by strength
            confluences.sort(key=lambda x: x['strength'], reverse=True)
            
        except Exception as e:
            self.logger.warning(f"Fibonacci confluence analysis failed: {str(e)}")
        
        return confluences
    
    def _assess_wave_structure_health(self, turning_points: List[WavePoint]) -> Dict[str, float]:
        """Assess the health/quality of wave structure."""
        try:
            if len(turning_points) < 3:
                return {'overall_health': 0.5, 'strength_consistency': 0.5, 'spacing_regularity': 0.5}
            
            # Strength consistency
            strengths = [p.strength for p in turning_points]
            strength_consistency = 1.0 - (np.std(strengths) / (np.mean(strengths) + 1e-8))
            strength_consistency = max(0.0, min(1.0, strength_consistency))
            
            # Spacing regularity
            indices = [p.index for p in turning_points]
            spacing = np.diff(indices)
            spacing_regularity = 1.0 - (np.std(spacing) / (np.mean(spacing) + 1e-8))
            spacing_regularity = max(0.0, min(1.0, spacing_regularity))
            
            # Overall health
            overall_health = (strength_consistency + spacing_regularity) / 2
            
            return {
                'overall_health': float(overall_health),
                'strength_consistency': float(strength_consistency),
                'spacing_regularity': float(spacing_regularity)
            }
            
        except Exception:
            return {'overall_health': 0.5, 'strength_consistency': 0.5, 'spacing_regularity': 0.5}
    
    def _determine_signal_type(self, turning_points: List[WavePoint], pivot_levels: List[PivotLevel], 
                              predictions: Dict[str, Any]) -> SignalType:
        """Determine signal type based on wave point analysis."""
        try:
            if not turning_points:
                return SignalType.NEUTRAL
            
            last_point = turning_points[-1]
            
            # Strong signals for high-quality turning points
            if last_point.strength > 0.8 and last_point.confidence > 0.8:
                if last_point.point_type == 'trough':
                    return SignalType.STRONG_BUY
                elif last_point.point_type == 'peak':
                    return SignalType.STRONG_SELL
            
            # Regular signals for moderate-quality points
            elif last_point.strength > 0.6 and last_point.confidence > 0.6:
                if last_point.point_type == 'trough':
                    return SignalType.BUY
                elif last_point.point_type == 'peak':
                    return SignalType.SELL
            
            # Consider prediction probability
            if predictions.get('probability', 0) > 0.7:
                predicted_type = predictions.get('next_point', {}).get('predicted_type')
                if predicted_type == 'trough':
                    return SignalType.BUY
                elif predicted_type == 'peak':
                    return SignalType.SELL
            
            return SignalType.NEUTRAL
            
        except Exception:
            return SignalType.NEUTRAL
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> np.ndarray:
        """Calculate RSI indicator."""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi.fillna(50).values
        except Exception:
            return np.full(len(prices), 50.0)
    
    def _calculate_macd(self, prices: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate MACD indicator."""
        try:
            ema12 = prices.ewm(span=12).mean()
            ema26 = prices.ewm(span=26).mean()
            macd = ema12 - ema26
            signal_line = macd.ewm(span=9).mean()
            
            return macd.fillna(0).values, signal_line.fillna(0).values
        except Exception:
            return np.zeros(len(prices)), np.zeros(len(prices))
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate Bollinger Bands."""
        try:
            sma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            
            upper_band = sma + 2 * std
            lower_band = sma - 2 * std
            
            return upper_band.fillna(prices).values, lower_band.fillna(prices).values
        except Exception:
            return prices.values, prices.values
    
    def _update_historical_data(self, turning_points: List[WavePoint], pivot_levels: List[PivotLevel], 
                               geometric_analysis: GeometricAnalysis):
        """Update historical data for future analysis."""
        self.identified_points = turning_points
        self.pivot_levels = pivot_levels
        self.geometric_history.append(geometric_analysis)
        
        # Keep only recent history
        max_history = 50
        if len(self.geometric_history) > max_history:
            self.geometric_history = self.geometric_history[-max_history:]
    
    def get_signal_type(self) -> SignalType:
        """Get the current signal type."""
        return getattr(self, '_last_signal_type', SignalType.NEUTRAL)
    
    def get_signal_strength(self) -> float:
        """Get the current signal strength."""
        if self.identified_points:
            return self.identified_points[-1].strength
        return 0.0
    
    def get_latest_turning_points(self, count: int = 5) -> List[WavePoint]:
        """Get the latest identified turning points."""
        return self.identified_points[-count:] if len(self.identified_points) >= count else self.identified_points
    
    def get_active_pivot_levels(self) -> List[PivotLevel]:
        """Get currently active pivot levels."""
        return self.pivot_levels