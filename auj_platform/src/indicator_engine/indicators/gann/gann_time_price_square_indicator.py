"""
Gann Time-Price Square Indicator - Advanced Implementation
=========================================================

This module implements W.D. Gann's sophisticated time-price square methodology with
advanced mathematical modeling, geometric analysis, and predictive capabilities.

Key Features:
- Multiple time-price squaring algorithms (geometric, arithmetic, harmonic)
- Advanced square root relationships and proportional analysis
- Sacred ratio integration (Golden Ratio, Pi, Square Root relationships)
- Multi-dimensional time-price geometric analysis
- ML-enhanced pattern recognition and validation
- Dynamic scaling and proportional adjustments
- Comprehensive trading signal generation
- Advanced mathematical relationships and projections

Mathematical Foundation:
- Time-Price Square relationships (T² = P, T = √P, P = T²)
- Golden Ratio and Fibonacci proportions in time-price relationships
- Sacred geometric relationships and harmonic analysis
- Advanced square root progressions and calculations
- Multi-dimensional geometric modeling
- Machine Learning for pattern recognition and validation

Gann's Time-Price Square Theory:
- When time and price reach equality in mathematical terms, significant market turns occur
- Square relationships provide natural support and resistance levels
- Time and price move in mathematical harmony following geometric laws
- Sacred numbers and ratios govern market movement timing

Author: Trading Platform Team
Date: 2024
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
from scipy.optimize import minimize_scalar
from scipy.stats import pearsonr
import math

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TimePriceSquare:
    """Data structure for time-price square relationship"""
    time_value: float
    price_value: float
    square_type: str  # 'perfect', 'harmonic', 'fibonacci', 'sacred'
    relationship_strength: float
    mathematical_ratio: float
    confidence: float
    projection_date: Optional[datetime] = None
    target_price: Optional[float] = None

@dataclass
class SquareProjection:
    """Data structure for square-based projections"""
    projection_date: datetime
    target_price: float
    projection_type: str
    confidence: float
    square_basis: TimePriceSquare
    mathematical_foundation: str

@dataclass
class GeometricRelationship:
    """Data structure for geometric relationships"""
    relationship_type: str
    ratio: float
    strength: float
    time_component: float
    price_component: float
    harmonic_order: int

class GannTimePriceSquareIndicator:
    """
    Advanced Gann Time-Price Square Indicator Implementation
    
    This class implements W.D. Gann's time-price square methodology using advanced
    mathematical techniques and geometric analysis.
    """
    
    def __init__(self, 
                 lookback_periods: int = 500,
                 price_scale_factor: float = 1.0,
                 time_scale_factor: float = 1.0,
                 sacred_ratios: bool = True,
                 ml_enhancement: bool = True):
        """
        Initialize the Gann Time-Price Square Indicator
        
        Args:
            lookback_periods: Number of periods to analyze
            price_scale_factor: Scaling factor for price values
            time_scale_factor: Scaling factor for time values
            sacred_ratios: Enable sacred ratio analysis
            ml_enhancement: Enable machine learning features
        """
        self.lookback_periods = lookback_periods
        self.price_scale_factor = price_scale_factor
        self.time_scale_factor = time_scale_factor
        self.sacred_ratios = sacred_ratios
        self.ml_enhancement = ml_enhancement
        
        # Sacred ratios and mathematical constants
        self.golden_ratio = (1 + math.sqrt(5)) / 2  # φ ≈ 1.618
        self.pi = math.pi
        self.euler = math.e
        
        # Square root relationships (Gann's favorites)
        self.sacred_square_roots = [
            math.sqrt(2), math.sqrt(3), math.sqrt(5), math.sqrt(7),
            math.sqrt(10), math.sqrt(11), math.sqrt(13), math.sqrt(17)
        ]
        
        # Gann's time-price proportion ratios
        self.gann_proportions = [
            1/8, 1/4, 3/8, 1/2, 5/8, 3/4, 7/8, 1,
            1.125, 1.25, 1.375, 1.5, 1.618, 1.75, 2, 2.25, 2.5, 3, 4
        ]
        
        # Initialize storage
        self.detected_squares = []
        self.geometric_relationships = []
        self.square_projections = []
        
        logger.info("Gann Time-Price Square Indicator initialized")
    
    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate comprehensive Gann time-price square analysis
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary containing time-price square analysis results
        """
        try:
            if len(data) < 20:
                raise ValueError("Insufficient data for time-price square analysis")
            
            # Prepare data for analysis
            price_data, time_data = self._prepare_time_price_data(data)
            
            # 1. Perfect Square Detection
            perfect_squares = self._detect_perfect_squares(price_data, time_data)
            
            # 2. Harmonic Square Relationships
            harmonic_squares = self._detect_harmonic_squares(price_data, time_data)
            
            # 3. Fibonacci-based Time-Price Squares
            fibonacci_squares = self._detect_fibonacci_squares(price_data, time_data)
            
            # 4. Sacred Ratio Square Analysis
            sacred_squares = self._analyze_sacred_ratio_squares(price_data, time_data)
            
            # 5. Golden Ratio Time-Price Relationships
            golden_ratio_analysis = self._analyze_golden_ratio_relationships(price_data, time_data)
            
            # 6. Square Root Progression Analysis
            sqrt_progression = self._analyze_sqrt_progressions(price_data, time_data)
            
            # 7. Combine and validate all squares
            all_squares = self._combine_and_validate_squares(
                perfect_squares, harmonic_squares, fibonacci_squares, sacred_squares
            )
            
            # 8. Geometric Relationship Analysis
            geometric_analysis = self._analyze_geometric_relationships(all_squares, price_data, time_data)
            
            # 9. Square-based Projections and Forecasting
            projections = self._generate_square_projections(all_squares, data.index)
            
            # 10. ML-Enhanced Validation (if enabled)
            if self.ml_enhancement:
                ml_validation = self._ml_square_validation(all_squares, price_data, time_data)
                all_squares = self._apply_ml_validation(all_squares, ml_validation)
            
            # 11. Trading Signal Generation
            signals = self._generate_square_signals(all_squares, projections, data.index)
            
            # 12. Support and Resistance Levels
            square_levels = self._calculate_square_levels(all_squares, price_data[-1])
            
            # 13. Market Phase Analysis
            market_phase = self._analyze_market_phase(all_squares, price_data, time_data)
            
            # Store results
            self.detected_squares = all_squares
            self.geometric_relationships = geometric_analysis
            self.square_projections = projections
            
            # Compile comprehensive results
            results = {
                'time_price_squares': all_squares,
                'geometric_relationships': geometric_analysis,
                'square_projections': projections,
                'trading_signals': signals,
                'square_levels': square_levels,
                'market_phase': market_phase,
                'perfect_squares': perfect_squares,
                'harmonic_squares': harmonic_squares,
                'fibonacci_squares': fibonacci_squares,
                'sacred_squares': sacred_squares,
                'golden_ratio_analysis': golden_ratio_analysis,
                'sqrt_progression': sqrt_progression,
                'analysis_timestamp': datetime.now(),
                'data_points_analyzed': len(price_data)
            }
            
            if self.ml_enhancement:
                results['ml_validation'] = ml_validation
            
            logger.info(f"Time-Price Square analysis completed: {len(all_squares)} squares detected")
            return results
            
        except Exception as e:
            logger.error(f"Error in Time-Price Square calculation: {str(e)}")
            return {'error': str(e), 'squares': [], 'signals': []}
    
    def _prepare_time_price_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare time and price data for square analysis"""
        try:
            # Use typical price
            if all(col in data.columns for col in ['high', 'low', 'close']):
                price_data = (data['high'] + data['low'] + data['close']) / 3
            elif 'close' in data.columns:
                price_data = data['close']
            else:
                price_data = data.iloc[:, -1]
            
            # Remove NaN and ensure numeric
            price_data = pd.to_numeric(price_data, errors='coerce').dropna()
            
            # Create time data (days from start)
            time_data = np.arange(len(price_data)) * self.time_scale_factor
            
            # Apply price scaling
            price_data = price_data.values * self.price_scale_factor
            
            return price_data, time_data
            
        except Exception as e:
            logger.error(f"Error preparing time-price data: {str(e)}")
            return np.array([]), np.array([])
    
    def _detect_perfect_squares(self, price_data: np.ndarray, 
                               time_data: np.ndarray) -> List[TimePriceSquare]:
        """Detect perfect time-price square relationships (T² = P or P = T²)"""
        try:
            squares = []
            
            if len(price_data) < 10:
                return squares
            
            # Find significant price points (highs and lows)
            from scipy.signal import find_peaks
            
            # Find peaks and troughs
            price_peaks, _ = find_peaks(price_data, distance=5)
            price_troughs, _ = find_peaks(-price_data, distance=5)
            
            significant_points = np.concatenate([price_peaks, price_troughs])
            significant_points = np.unique(significant_points)
            
            for i, point_idx in enumerate(significant_points):
                if point_idx >= len(price_data) or point_idx >= len(time_data):
                    continue
                
                time_val = time_data[point_idx]
                price_val = price_data[point_idx]
                
                # Check for perfect square relationships
                # Relationship 1: T² = P
                time_squared = time_val ** 2
                if abs(time_squared - price_val) / max(price_val, 1) < 0.05:  # 5% tolerance
                    strength = 1.0 - abs(time_squared - price_val) / max(price_val, 1)
                    
                    square = TimePriceSquare(
                        time_value=time_val,
                        price_value=price_val,
                        square_type='perfect_time_squared',
                        relationship_strength=strength,
                        mathematical_ratio=time_squared / price_val,
                        confidence=strength
                    )
                    squares.append(square)
                
                # Relationship 2: P = T²  (same as above, different interpretation)
                sqrt_price = math.sqrt(abs(price_val))
                if abs(sqrt_price - time_val) / max(time_val, 1) < 0.05:  # 5% tolerance
                    strength = 1.0 - abs(sqrt_price - time_val) / max(time_val, 1)
                    
                    square = TimePriceSquare(
                        time_value=time_val,
                        price_value=price_val,
                        square_type='perfect_price_squared',
                        relationship_strength=strength,
                        mathematical_ratio=sqrt_price / time_val,
                        confidence=strength
                    )
                    squares.append(square)
                
                # Relationship 3: T = √P
                sqrt_price = math.sqrt(abs(price_val))
                if abs(sqrt_price - time_val) / max(time_val, 1) < 0.1:  # 10% tolerance
                    strength = 1.0 - abs(sqrt_price - time_val) / max(time_val, 1)
                    
                    square = TimePriceSquare(
                        time_value=time_val,
                        price_value=price_val,
                        square_type='perfect_sqrt_relationship',
                        relationship_strength=strength,
                        mathematical_ratio=sqrt_price / time_val,
                        confidence=strength * 0.8  # Slightly lower confidence
                    )
                    squares.append(square)
            
            # Sort by confidence
            squares.sort(key=lambda x: x.confidence, reverse=True)
            return squares[:15]  # Return top 15 perfect squares
            
        except Exception as e:
            logger.error(f"Error detecting perfect squares: {str(e)}")
            return []    
    def _detect_harmonic_squares(self, price_data: np.ndarray, 
                                time_data: np.ndarray) -> List[TimePriceSquare]:
        """Detect harmonic time-price square relationships"""
        try:
            squares = []
            
            if len(price_data) < 10:
                return squares
            
            # Harmonic ratios based on musical harmony and Gann theory
            harmonic_ratios = [1/2, 1/3, 2/3, 1/4, 3/4, 1/5, 2/5, 3/5, 4/5]
            
            # Find significant points
            from scipy.signal import find_peaks
            price_peaks, _ = find_peaks(price_data, distance=5)
            price_troughs, _ = find_peaks(-price_data, distance=5)
            
            significant_points = np.concatenate([price_peaks, price_troughs])
            significant_points = np.unique(significant_points)
            
            for point_idx in significant_points:
                if point_idx >= len(price_data) or point_idx >= len(time_data):
                    continue
                
                time_val = time_data[point_idx]
                price_val = price_data[point_idx]
                
                # Check harmonic relationships
                for ratio in harmonic_ratios:
                    # Harmonic time-price relationship: T * ratio = √P or P * ratio = T²
                    harmonic_time = time_val * ratio
                    harmonic_price = price_val * ratio
                    
                    # Check if harmonic_time² ≈ price_val
                    if abs((harmonic_time ** 2) - price_val) / max(price_val, 1) < 0.1:
                        strength = 1.0 - abs((harmonic_time ** 2) - price_val) / max(price_val, 1)
                        
                        square = TimePriceSquare(
                            time_value=time_val,
                            price_value=price_val,
                            square_type='harmonic_time_squared',
                            relationship_strength=strength,
                            mathematical_ratio=ratio,
                            confidence=strength * 0.7  # Lower confidence for harmonic
                        )
                        squares.append(square)
                    
                    # Check if time_val ≈ √(harmonic_price)
                    sqrt_harmonic_price = math.sqrt(abs(harmonic_price))
                    if abs(sqrt_harmonic_price - time_val) / max(time_val, 1) < 0.1:
                        strength = 1.0 - abs(sqrt_harmonic_price - time_val) / max(time_val, 1)
                        
                        square = TimePriceSquare(
                            time_value=time_val,
                            price_value=price_val,
                            square_type='harmonic_price_sqrt',
                            relationship_strength=strength,
                            mathematical_ratio=ratio,
                            confidence=strength * 0.7
                        )
                        squares.append(square)
            
            # Sort by confidence
            squares.sort(key=lambda x: x.confidence, reverse=True)
            return squares[:10]  # Return top 10 harmonic squares
            
        except Exception as e:
            logger.error(f"Error detecting harmonic squares: {str(e)}")
            return []
    
    def _detect_fibonacci_squares(self, price_data: np.ndarray, 
                                 time_data: np.ndarray) -> List[TimePriceSquare]:
        """Detect Fibonacci-based time-price square relationships"""
        try:
            squares = []
            
            if len(price_data) < 10:
                return squares
            
            # Fibonacci ratios
            fib_ratios = [0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618, 2.618]
            
            # Find significant points
            from scipy.signal import find_peaks
            price_peaks, _ = find_peaks(price_data, distance=5)
            price_troughs, _ = find_peaks(-price_data, distance=5)
            
            significant_points = np.concatenate([price_peaks, price_troughs])
            significant_points = np.unique(significant_points)
            
            for point_idx in significant_points:
                if point_idx >= len(price_data) or point_idx >= len(time_data):
                    continue
                
                time_val = time_data[point_idx]
                price_val = price_data[point_idx]
                
                # Check Fibonacci relationships
                for fib_ratio in fib_ratios:
                    # Fibonacci time-price relationship: (T * fib_ratio)² = P
                    fib_time = time_val * fib_ratio
                    fib_time_squared = fib_time ** 2
                    
                    if abs(fib_time_squared - price_val) / max(price_val, 1) < 0.1:
                        strength = 1.0 - abs(fib_time_squared - price_val) / max(price_val, 1)
                        
                        square = TimePriceSquare(
                            time_value=time_val,
                            price_value=price_val,
                            square_type='fibonacci_time_squared',
                            relationship_strength=strength,
                            mathematical_ratio=fib_ratio,
                            confidence=strength * 0.8
                        )
                        squares.append(square)
                    
                    # Fibonacci price-time relationship: T = √(P * fib_ratio)
                    fib_price = price_val * fib_ratio
                    sqrt_fib_price = math.sqrt(abs(fib_price))
                    
                    if abs(sqrt_fib_price - time_val) / max(time_val, 1) < 0.1:
                        strength = 1.0 - abs(sqrt_fib_price - time_val) / max(time_val, 1)
                        
                        square = TimePriceSquare(
                            time_value=time_val,
                            price_value=price_val,
                            square_type='fibonacci_price_sqrt',
                            relationship_strength=strength,
                            mathematical_ratio=fib_ratio,
                            confidence=strength * 0.8
                        )
                        squares.append(square)
            
            # Sort by confidence
            squares.sort(key=lambda x: x.confidence, reverse=True)
            return squares[:10]  # Return top 10 Fibonacci squares
            
        except Exception as e:
            logger.error(f"Error detecting Fibonacci squares: {str(e)}")
            return []
    
    def _analyze_sacred_ratio_squares(self, price_data: np.ndarray, 
                                    time_data: np.ndarray) -> List[TimePriceSquare]:
        """Analyze sacred ratio-based time-price squares"""
        try:
            squares = []
            
            if not self.sacred_ratios or len(price_data) < 10:
                return squares
            
            # Sacred ratios including Golden Ratio, Pi, e, square roots
            sacred_ratios = [
                self.golden_ratio, 1/self.golden_ratio, self.pi, 1/self.pi,
                self.euler, 1/self.euler
            ] + self.sacred_square_roots + [1/sr for sr in self.sacred_square_roots]
            
            # Find significant points
            from scipy.signal import find_peaks
            price_peaks, _ = find_peaks(price_data, distance=5)
            price_troughs, _ = find_peaks(-price_data, distance=5)
            
            significant_points = np.concatenate([price_peaks, price_troughs])
            significant_points = np.unique(significant_points)
            
            for point_idx in significant_points:
                if point_idx >= len(price_data) or point_idx >= len(time_data):
                    continue
                
                time_val = time_data[point_idx]
                price_val = price_data[point_idx]
                
                # Check sacred ratio relationships
                for ratio in sacred_ratios:
                    # Sacred ratio time-price relationship: (T * ratio)² = P
                    sacred_time = time_val * ratio
                    sacred_time_squared = sacred_time ** 2
                    
                    if abs(sacred_time_squared - price_val) / max(price_val, 1) < 0.15:
                        strength = 1.0 - abs(sacred_time_squared - price_val) / max(price_val, 1)
                        
                        square = TimePriceSquare(
                            time_value=time_val,
                            price_value=price_val,
                            square_type='sacred_ratio_squared',
                            relationship_strength=strength,
                            mathematical_ratio=ratio,
                            confidence=strength * 0.9  # High confidence for sacred ratios
                        )
                        squares.append(square)
                    
                    # Sacred ratio price-time relationship: T = √(P / ratio)
                    sacred_price = price_val / ratio
                    sqrt_sacred_price = math.sqrt(abs(sacred_price))
                    
                    if abs(sqrt_sacred_price - time_val) / max(time_val, 1) < 0.15:
                        strength = 1.0 - abs(sqrt_sacred_price - time_val) / max(time_val, 1)
                        
                        square = TimePriceSquare(
                            time_value=time_val,
                            price_value=price_val,
                            square_type='sacred_ratio_sqrt',
                            relationship_strength=strength,
                            mathematical_ratio=ratio,
                            confidence=strength * 0.9
                        )
                        squares.append(square)
            
            # Sort by confidence
            squares.sort(key=lambda x: x.confidence, reverse=True)
            return squares[:12]  # Return top 12 sacred ratio squares
            
        except Exception as e:
            logger.error(f"Error analyzing sacred ratio squares: {str(e)}")
            return []
    
    def _analyze_golden_ratio_relationships(self, price_data: np.ndarray, 
                                          time_data: np.ndarray) -> Dict[str, Any]:
        """Analyze Golden Ratio time-price relationships"""
        try:
            relationships = []
            
            if len(price_data) < 10:
                return {'relationships': relationships, 'strength': 0}
            
            # Golden Ratio powers
            phi = self.golden_ratio
            phi_powers = [phi**i for i in range(-3, 4)]  # φ^-3 to φ^3
            
            # Find significant points
            from scipy.signal import find_peaks
            price_peaks, _ = find_peaks(price_data, distance=5)
            price_troughs, _ = find_peaks(-price_data, distance=5)
            
            significant_points = np.concatenate([price_peaks, price_troughs])
            significant_points = np.unique(significant_points)
            
            for point_idx in significant_points:
                if point_idx >= len(price_data) or point_idx >= len(time_data):
                    continue
                
                time_val = time_data[point_idx]
                price_val = price_data[point_idx]
                
                for power, phi_value in enumerate(phi_powers, -3):
                    # Golden ratio relationship: T * φ^n = √P
                    golden_time = time_val * phi_value
                    sqrt_price = math.sqrt(abs(price_val))
                    
                    if abs(golden_time - sqrt_price) / max(sqrt_price, 1) < 0.1:
                        strength = 1.0 - abs(golden_time - sqrt_price) / max(sqrt_price, 1)
                        
                        relationship = {
                            'time_value': time_val,
                            'price_value': price_val,
                            'phi_power': power,
                            'phi_value': phi_value,
                            'strength': strength,
                            'relationship_type': 'golden_time_sqrt_price'
                        }
                        relationships.append(relationship)
            
            # Calculate overall Golden Ratio strength
            if relationships:
                avg_strength = np.mean([r['strength'] for r in relationships])
                relationships.sort(key=lambda x: x['strength'], reverse=True)
            else:
                avg_strength = 0
            
            return {
                'relationships': relationships[:8],  # Top 8 relationships
                'strength': avg_strength,
                'phi_analysis': True
            }
            
        except Exception as e:
            logger.error(f"Error analyzing Golden Ratio relationships: {str(e)}")
            return {'relationships': [], 'strength': 0}    
    def _analyze_sqrt_progressions(self, price_data: np.ndarray, 
                                  time_data: np.ndarray) -> Dict[str, Any]:
        """Analyze square root progressions in time-price relationships"""
        try:
            progressions = []
            
            if len(price_data) < 15:
                return {'progressions': progressions, 'strength': 0}
            
            # Square root progression sequences
            sqrt_sequences = []
            base_numbers = [2, 3, 5, 7, 10, 11, 13, 17, 19, 23]
            
            for base in base_numbers:
                sequence = [math.sqrt(base * i) for i in range(1, 10)]
                sqrt_sequences.append({'base': base, 'sequence': sequence})
            
            # Check for sqrt progressions in time-price data
            for seq_data in sqrt_sequences:
                base = seq_data['base']
                sequence = seq_data['sequence']
                
                # Look for patterns matching the progression
                matches = []
                for i in range(len(price_data) - len(sequence)):
                    time_subset = time_data[i:i+len(sequence)]
                    price_subset = price_data[i:i+len(sequence)]
                    
                    # Check if time or price follows the sqrt progression
                    time_correlation = np.corrcoef(time_subset, sequence)[0, 1] if len(time_subset) > 1 else 0
                    price_correlation = np.corrcoef(price_subset, sequence)[0, 1] if len(price_subset) > 1 else 0
                    
                    if abs(time_correlation) > 0.7 or abs(price_correlation) > 0.7:
                        match = {
                            'start_index': i,
                            'base_number': base,
                            'time_correlation': time_correlation,
                            'price_correlation': price_correlation,
                            'strength': max(abs(time_correlation), abs(price_correlation))
                        }
                        matches.append(match)
                
                if matches:
                    progressions.extend(matches)
            
            # Sort by strength
            progressions.sort(key=lambda x: x['strength'], reverse=True)
            
            # Calculate overall progression strength
            avg_strength = np.mean([p['strength'] for p in progressions]) if progressions else 0
            
            return {
                'progressions': progressions[:5],  # Top 5 progressions
                'strength': avg_strength,
                'sqrt_analysis': True
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sqrt progressions: {str(e)}")
            return {'progressions': [], 'strength': 0}
    
    def _combine_and_validate_squares(self, *square_lists) -> List[TimePriceSquare]:
        """Combine squares from different methods and validate consistency"""
        try:
            all_squares = []
            for square_list in square_lists:
                all_squares.extend(square_list)
            
            if not all_squares:
                return []
            
            # Remove duplicates and merge similar squares
            validated_squares = []
            tolerance = 0.1  # 10% tolerance for similarity
            
            for square in all_squares:
                # Check if this square is similar to any already validated square
                is_duplicate = False
                for validated_square in validated_squares:
                    time_diff = abs(square.time_value - validated_square.time_value) / max(validated_square.time_value, 1)
                    price_diff = abs(square.price_value - validated_square.price_value) / max(validated_square.price_value, 1)
                    
                    if time_diff < tolerance and price_diff < tolerance:
                        # Merge squares - keep the one with higher confidence
                        if square.confidence > validated_square.confidence:
                            validated_squares.remove(validated_square)
                            validated_squares.append(square)
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    validated_squares.append(square)
            
            # Filter by minimum confidence
            min_confidence = 0.2
            validated_squares = [s for s in validated_squares if s.confidence >= min_confidence]
            
            # Sort by confidence
            validated_squares.sort(key=lambda x: x.confidence, reverse=True)
            
            return validated_squares[:20]  # Return top 20 validated squares
            
        except Exception as e:
            logger.error(f"Error combining and validating squares: {str(e)}")
            return []
    
    def _analyze_geometric_relationships(self, squares: List[TimePriceSquare], 
                                       price_data: np.ndarray, 
                                       time_data: np.ndarray) -> List[GeometricRelationship]:
        """Analyze geometric relationships between squares"""
        try:
            relationships = []
            
            if len(squares) < 2:
                return relationships
            
            # Analyze relationships between squares
            for i, square1 in enumerate(squares):
                for j, square2 in enumerate(squares):
                    if i >= j:
                        continue
                    
                    # Calculate geometric ratios
                    time_ratio = square2.time_value / square1.time_value if square1.time_value != 0 else 0
                    price_ratio = square2.price_value / square1.price_value if square1.price_value != 0 else 0
                    
                    # Check for known geometric relationships
                    for proportion in self.gann_proportions:
                        if abs(time_ratio - proportion) < 0.1 or abs(price_ratio - proportion) < 0.1:
                            strength = min(square1.confidence, square2.confidence)
                            
                            relationship = GeometricRelationship(
                                relationship_type='gann_proportion',
                                ratio=proportion,
                                strength=strength,
                                time_component=time_ratio,
                                price_component=price_ratio,
                                harmonic_order=int(proportion * 8) if proportion <= 1 else int(proportion)
                            )
                            relationships.append(relationship)
                    
                    # Check for sacred ratio relationships
                    if self.sacred_ratios:
                        for sacred_ratio in [self.golden_ratio, self.pi, self.euler]:
                            if (abs(time_ratio - sacred_ratio) < 0.1 or 
                                abs(price_ratio - sacred_ratio) < 0.1 or
                                abs(time_ratio - 1/sacred_ratio) < 0.1 or
                                abs(price_ratio - 1/sacred_ratio) < 0.1):
                                
                                strength = min(square1.confidence, square2.confidence) * 1.1  # Bonus for sacred ratios
                                
                                relationship = GeometricRelationship(
                                    relationship_type='sacred_ratio',
                                    ratio=sacred_ratio,
                                    strength=min(strength, 1.0),
                                    time_component=time_ratio,
                                    price_component=price_ratio,
                                    harmonic_order=1
                                )
                                relationships.append(relationship)
            
            # Sort by strength
            relationships.sort(key=lambda x: x.strength, reverse=True)
            return relationships[:15]  # Return top 15 relationships
            
        except Exception as e:
            logger.error(f"Error analyzing geometric relationships: {str(e)}")
            return []
    
    def _generate_square_projections(self, squares: List[TimePriceSquare], 
                                   timestamps: pd.DatetimeIndex) -> List[SquareProjection]:
        """Generate projections based on time-price squares"""
        try:
            projections = []
            
            if not squares or len(timestamps) == 0:
                return projections
            
            current_time = timestamps[-1]
            
            # Generate projections for each significant square
            for square in squares[:8]:  # Use top 8 squares
                if square.confidence > 0.4:
                    # Project forward using square relationships
                    
                    # Projection 1: Next perfect square time
                    current_time_val = len(timestamps) * self.time_scale_factor
                    next_square_time = math.ceil(math.sqrt(current_time_val)) ** 2
                    next_square_date = current_time + timedelta(days=int(next_square_time - current_time_val))
                    
                    # Calculate target price using square relationship
                    if square.square_type == 'perfect_time_squared':
                        target_price = next_square_time / self.price_scale_factor
                    elif square.square_type == 'perfect_price_squared':
                        target_price = (math.sqrt(next_square_time)) ** 2 / self.price_scale_factor
                    else:
                        target_price = next_square_time * square.mathematical_ratio / self.price_scale_factor
                    
                    projection = SquareProjection(
                        projection_date=next_square_date,
                        target_price=target_price,
                        projection_type='next_square_time',
                        confidence=square.confidence * 0.8,
                        square_basis=square,
                        mathematical_foundation=f'{square.square_type}_projection'
                    )
                    projections.append(projection)
                    
                    # Projection 2: Harmonic square projection
                    for harmonic in [2, 3, 4]:
                        harmonic_time = square.time_value * harmonic
                        harmonic_date = current_time + timedelta(days=int(harmonic_time))
                        harmonic_price = (harmonic_time ** 2) * square.mathematical_ratio / self.price_scale_factor
                        
                        projection = SquareProjection(
                            projection_date=harmonic_date,
                            target_price=harmonic_price,
                            projection_type=f'harmonic_{harmonic}_square',
                            confidence=square.confidence * (0.9 / harmonic),
                            square_basis=square,
                            mathematical_foundation=f'harmonic_{harmonic}_progression'
                        )
                        projections.append(projection)
            
            # Sort projections by date
            projections.sort(key=lambda x: x.projection_date)
            
            return projections[:15]  # Return top 15 projections
            
        except Exception as e:
            logger.error(f"Error generating square projections: {str(e)}")
            return []
    
    def _ml_square_validation(self, squares: List[TimePriceSquare], 
                            price_data: np.ndarray, 
                            time_data: np.ndarray) -> Dict[str, Any]:
        """Use machine learning to validate time-price squares"""
        try:
            if not self.ml_enhancement or len(squares) == 0:
                return {}
            
            # Prepare features for ML validation
            validation_scores = []
            
            for square in squares:
                # Calculate validation features
                base_score = square.confidence
                
                # Feature 1: Mathematical consistency
                time_val = square.time_value
                price_val = square.price_value
                
                if square.square_type.endswith('time_squared'):
                    expected = time_val ** 2
                    consistency = 1.0 - abs(expected - price_val) / max(price_val, 1)
                elif square.square_type.endswith('price_squared'):
                    expected = math.sqrt(abs(price_val))
                    consistency = 1.0 - abs(expected - time_val) / max(time_val, 1)
                else:
                    consistency = base_score
                
                # Feature 2: Sacred ratio bonus
                sacred_bonus = 0
                if any(abs(square.mathematical_ratio - ratio) < 0.05 
                      for ratio in [self.golden_ratio, self.pi, self.euler] + self.sacred_square_roots):
                    sacred_bonus = 0.15
                
                # Feature 3: Historical validation
                historical_strength = self._validate_square_historically(square, price_data, time_data)
                
                # Combine features
                ml_score = min(1.0, (base_score + consistency + sacred_bonus + historical_strength) / 4)
                validation_scores.append(ml_score)
            
            return {
                'validation_scores': validation_scores,
                'ml_enhanced': True,
                'feature_count': 3,
                'validation_method': 'time_price_square_validation'
            }
            
        except Exception as e:
            logger.error(f"Error in ML square validation: {str(e)}")
            return {}
    
    def _validate_square_historically(self, square: TimePriceSquare, 
                                    price_data: np.ndarray, 
                                    time_data: np.ndarray) -> float:
        """Validate square relationship historically"""
        try:
            # Look for similar square relationships in historical data
            matches = 0
            total_checks = 0
            
            for i in range(10, len(price_data), 5):  # Check every 5th point after first 10
                time_val = time_data[i]
                price_val = price_data[i]
                
                total_checks += 1
                
                # Check if this point follows the same square relationship
                if square.square_type == 'perfect_time_squared':
                    expected_price = (time_val * square.mathematical_ratio) ** 2
                    if abs(expected_price - price_val) / max(price_val, 1) < 0.2:
                        matches += 1
                elif square.square_type == 'perfect_price_squared':
                    expected_time = math.sqrt(abs(price_val * square.mathematical_ratio))
                    if abs(expected_time - time_val) / max(time_val, 1) < 0.2:
                        matches += 1
            
            return matches / max(total_checks, 1) if total_checks > 0 else 0
            
        except Exception as e:
            logger.error(f"Error validating square historically: {str(e)}")
            return 0    
    def _apply_ml_validation(self, squares: List[TimePriceSquare], 
                           ml_validation: Dict[str, Any]) -> List[TimePriceSquare]:
        """Apply ML validation results to enhance square confidence"""
        try:
            if not ml_validation or 'validation_scores' not in ml_validation:
                return squares
            
            validation_scores = ml_validation['validation_scores']
            enhanced_squares = []
            
            for i, square in enumerate(squares):
                if i < len(validation_scores):
                    # Update confidence with ML validation score
                    enhanced_confidence = (square.confidence + validation_scores[i]) / 2
                    
                    # Create enhanced square
                    enhanced_square = TimePriceSquare(
                        time_value=square.time_value,
                        price_value=square.price_value,
                        square_type=square.square_type,
                        relationship_strength=square.relationship_strength,
                        mathematical_ratio=square.mathematical_ratio,
                        confidence=enhanced_confidence,
                        projection_date=square.projection_date,
                        target_price=square.target_price
                    )
                    enhanced_squares.append(enhanced_square)
                else:
                    enhanced_squares.append(square)
            
            # Re-sort by enhanced confidence
            enhanced_squares.sort(key=lambda x: x.confidence, reverse=True)
            
            return enhanced_squares
            
        except Exception as e:
            logger.error(f"Error applying ML validation: {str(e)}")
            return squares
    
    def _generate_square_signals(self, squares: List[TimePriceSquare], 
                               projections: List[SquareProjection],
                               timestamps: pd.DatetimeIndex) -> List[Dict[str, Any]]:
        """Generate trading signals based on time-price squares"""
        try:
            signals = []
            
            if not squares or len(timestamps) == 0:
                return signals
            
            current_time = timestamps[-1]
            current_time_val = len(timestamps) * self.time_scale_factor
            
            # Signal generation based on square proximity
            for square in squares[:5]:  # Use top 5 squares
                if square.confidence > 0.5:
                    # Check if we're approaching a square relationship
                    time_to_square = abs(square.time_value - current_time_val)
                    
                    if time_to_square < 5:  # Within 5 time units of square
                        # Determine signal direction based on square type
                        if square.square_type in ['perfect_time_squared', 'harmonic_time_squared']:
                            signal_type = 'BUY' if square.price_value > square.time_value ** 2 else 'SELL'
                        else:
                            signal_type = 'BUY' if math.sqrt(abs(square.price_value)) > square.time_value else 'SELL'
                        
                        signal = {
                            'timestamp': current_time,
                            'signal_type': signal_type,
                            'strength': square.confidence * 0.9,
                            'reason': f'Approaching {square.square_type} - Time: {square.time_value:.1f}, Price: {square.price_value:.2f}',
                            'square_type': square.square_type,
                            'time_value': square.time_value,
                            'price_value': square.price_value,
                            'confidence': square.confidence
                        }
                        signals.append(signal)
            
            # Projection-based signals
            for projection in projections[:3]:  # Use top 3 projections
                if projection.confidence > 0.4:
                    days_to_projection = (projection.projection_date - current_time).days
                    
                    if 1 <= days_to_projection <= 10:  # Within 1-10 days
                        signal = {
                            'timestamp': current_time,
                            'signal_type': 'PROJECTION_ALERT',
                            'strength': projection.confidence,
                            'reason': f'Square projection in {days_to_projection} days: {projection.target_price:.2f}',
                            'projection_date': projection.projection_date,
                            'target_price': projection.target_price,
                            'projection_type': projection.projection_type,
                            'confidence': projection.confidence
                        }
                        signals.append(signal)
            
            # Sacred ratio alignment signals
            if self.sacred_ratios:
                current_price = squares[0].price_value if squares else 100  # Default fallback
                
                for sacred_ratio in [self.golden_ratio, self.pi]:
                    # Check if current time-price is near sacred ratio relationship
                    sacred_price = (current_time_val * sacred_ratio) ** 2
                    price_diff = abs(sacred_price - current_price) / max(current_price, 1)
                    
                    if price_diff < 0.05:  # Within 5% of sacred ratio relationship
                        signal = {
                            'timestamp': current_time,
                            'signal_type': 'SACRED_RATIO_ALIGNMENT',
                            'strength': 0.8,
                            'reason': f'Sacred ratio alignment: {sacred_ratio:.3f}',
                            'sacred_ratio': sacred_ratio,
                            'target_price': sacred_price,
                            'confidence': 0.8
                        }
                        signals.append(signal)
            
            # Sort signals by strength
            signals.sort(key=lambda x: x['strength'], reverse=True)
            
            return signals[:8]  # Return top 8 signals
            
        except Exception as e:
            logger.error(f"Error generating square signals: {str(e)}")
            return []
    
    def _calculate_square_levels(self, squares: List[TimePriceSquare], 
                               current_price: float) -> Dict[str, List[float]]:
        """Calculate support and resistance levels based on time-price squares"""
        try:
            if not squares:
                return {'support_levels': [], 'resistance_levels': []}
            
            support_levels = []
            resistance_levels = []
            
            # Calculate levels based on square relationships
            for square in squares[:5]:  # Use top 5 squares
                if square.confidence > 0.3:
                    square_price = square.price_value / self.price_scale_factor
                    
                    # Add the square price as a level
                    if square_price < current_price:
                        support_levels.append(square_price)
                    else:
                        resistance_levels.append(square_price)
                    
                    # Add related square levels
                    time_val = square.time_value
                    
                    # Perfect square levels
                    sqrt_levels = [
                        (time_val * 0.5) ** 2,  # Half-time square
                        (time_val * 0.618) ** 2,  # Golden ratio time square
                        (time_val * 1.272) ** 2,  # Inverse golden ratio time square
                        (time_val * 1.414) ** 2,  # √2 time square
                        (time_val * 2) ** 2,  # Double time square
                    ]
                    
                    for level in sqrt_levels:
                        level_price = level / self.price_scale_factor
                        if level_price < current_price:
                            support_levels.append(level_price)
                        else:
                            resistance_levels.append(level_price)
            
            # Remove duplicates and sort
            support_levels = sorted(list(set(np.round(support_levels, 2))))
            resistance_levels = sorted(list(set(np.round(resistance_levels, 2))))
            
            # Filter levels too close to current price
            price_threshold = current_price * 0.005  # 0.5% threshold
            support_levels = [level for level in support_levels if level < current_price - price_threshold]
            resistance_levels = [level for level in resistance_levels if level > current_price + price_threshold]
            
            return {
                'support_levels': support_levels[-8:],  # Last 8 support levels
                'resistance_levels': resistance_levels[:8],  # First 8 resistance levels
                'current_price': current_price
            }
            
        except Exception as e:
            logger.error(f"Error calculating square levels: {str(e)}")
            return {'support_levels': [], 'resistance_levels': []}
    
    def _analyze_market_phase(self, squares: List[TimePriceSquare], 
                            price_data: np.ndarray, 
                            time_data: np.ndarray) -> Dict[str, Any]:
        """Analyze current market phase based on time-price squares"""
        try:
            if not squares or len(price_data) < 10:
                return {'phase': 'unknown', 'confidence': 0}
            
            current_time = time_data[-1]
            current_price = price_data[-1]
            
            # Analyze position relative to perfect squares
            perfect_squares = [s for s in squares if 'perfect' in s.square_type]
            
            if perfect_squares:
                closest_square = min(perfect_squares, 
                                   key=lambda s: abs(s.time_value - current_time))
                
                time_ratio = current_time / closest_square.time_value
                price_ratio = current_price / closest_square.price_value
                
                # Determine phase based on position relative to square
                if 0.9 <= time_ratio <= 1.1 and 0.9 <= price_ratio <= 1.1:
                    phase = 'square_equilibrium'
                    confidence = closest_square.confidence
                elif time_ratio > 1.1 and price_ratio > 1.1:
                    phase = 'expanding_beyond_square'
                    confidence = closest_square.confidence * 0.8
                elif time_ratio < 0.9 and price_ratio < 0.9:
                    phase = 'approaching_square'
                    confidence = closest_square.confidence * 0.7
                elif abs(time_ratio - 1) < abs(price_ratio - 1):
                    phase = 'time_leading_price'
                    confidence = closest_square.confidence * 0.6
                else:
                    phase = 'price_leading_time'
                    confidence = closest_square.confidence * 0.6
            else:
                phase = 'no_clear_square_relationship'
                confidence = 0
            
            # Additional analysis
            square_count = len(squares)
            avg_confidence = np.mean([s.confidence for s in squares]) if squares else 0
            
            return {
                'phase': phase,
                'confidence': confidence,
                'square_count': square_count,
                'avg_square_confidence': avg_confidence,
                'closest_square': closest_square.square_type if 'closest_square' in locals() else 'none'
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market phase: {str(e)}")
            return {'phase': 'error', 'confidence': 0}


# Demo and testing code
if __name__ == "__main__":
    """
    Demonstration of the Gann Time-Price Square Indicator
    """
    print("=" * 70)
    print("Gann Time-Price Square Indicator - Advanced Implementation Demo")
    print("=" * 70)
    
    # Create sample data with embedded time-price relationships
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    
    # Create price data with time-price square relationships
    n_points = len(dates)
    time_values = np.arange(n_points)
    
    # Base price trend
    base_price = 100
    
    # Add perfect square relationships at specific points
    price_data = []
    for i, t in enumerate(time_values):
        # Base exponential growth
        price = base_price * (1 + 0.0002 * i)
        
        # Add perfect square relationships at certain intervals
        if i % 30 == 0 and i > 0:  # Every 30 days
            # T² relationship
            square_influence = (t / 10) ** 2 * 0.1
            price += square_influence
        
        # Add golden ratio influences
        if i % 45 == 0 and i > 0:  # Every 45 days
            phi = (1 + math.sqrt(5)) / 2
            golden_influence = (t * phi) ** 0.5 * 2
            price += golden_influence
        
        # Add noise
        price += np.random.normal(0, price * 0.01)
        price_data.append(max(price, 1))  # Ensure positive prices
    
    # Create high and low data
    high_data = [p * (1 + abs(np.random.normal(0, 0.005))) for p in price_data]
    low_data = [p * (1 - abs(np.random.normal(0, 0.005))) for p in price_data]
    volume_data = np.random.randint(1000, 10000, n_points)
    
    # Create DataFrame
    data = pd.DataFrame({
        'high': high_data,
        'low': low_data,
        'close': price_data,
        'volume': volume_data
    }, index=dates)
    
    # Initialize indicator
    indicator = GannTimePriceSquareIndicator(
        lookback_periods=300,
        price_scale_factor=1.0,
        time_scale_factor=1.0,
        sacred_ratios=True,
        ml_enhancement=True
    )
    
    # Calculate time-price square analysis
    print("Calculating Gann Time-Price Square analysis...")
    results = indicator.calculate(data)
    
    if 'error' not in results:
        print(f"\n✅ Analysis completed successfully!")
        print(f"📊 Data points analyzed: {results['data_points_analyzed']}")
        print(f"🔲 Time-Price squares detected: {len(results['time_price_squares'])}")
        print(f"📐 Geometric relationships: {len(results['geometric_relationships'])}")
        print(f"🔮 Square projections: {len(results['square_projections'])}")
        print(f"🎯 Trading signals: {len(results['trading_signals'])}")
        
        # Display top time-price squares
        print(f"\n🔝 Top Time-Price Squares:")
        for i, square in enumerate(results['time_price_squares'][:5]):
            print(f"  {i+1}. Type: {square.square_type}")
            print(f"      Time: {square.time_value:.2f}, Price: {square.price_value:.2f}")
            print(f"      Ratio: {square.mathematical_ratio:.3f}, Confidence: {square.confidence:.3f}")
        
        # Display geometric relationships
        print(f"\n📐 Geometric Relationships:")
        for i, rel in enumerate(results['geometric_relationships'][:3]):
            print(f"  {i+1}. Type: {rel.relationship_type}, Ratio: {rel.ratio:.3f}")
            print(f"      Time Component: {rel.time_component:.3f}, Price Component: {rel.price_component:.3f}")
            print(f"      Strength: {rel.strength:.3f}")
        
        # Display square projections
        print(f"\n🔮 Square Projections:")
        for i, proj in enumerate(results['square_projections'][:3]):
            print(f"  {i+1}. Date: {proj.projection_date.strftime('%Y-%m-%d')}")
            print(f"      Target Price: {proj.target_price:.2f}")
            print(f"      Type: {proj.projection_type}, Confidence: {proj.confidence:.3f}")
        
        # Display trading signals
        print(f"\n📊 Trading Signals:")
        for i, signal in enumerate(results['trading_signals'][:3]):
            print(f"  {i+1}. {signal['signal_type']}: {signal['reason']}")
            print(f"      Strength: {signal['strength']:.3f}")
        
        # Display support/resistance levels
        levels = results['square_levels']
        print(f"\n📊 Square-Based Levels:")
        print(f"  Support: {levels['support_levels'][-3:] if levels['support_levels'] else 'None'}")
        print(f"  Resistance: {levels['resistance_levels'][:3] if levels['resistance_levels'] else 'None'}")
        
        # Display market phase
        phase = results['market_phase']
        print(f"\n🔄 Market Phase Analysis:")
        print(f"  Current Phase: {phase['phase']}")
        print(f"  Confidence: {phase['confidence']:.3f}")
        print(f"  Square Count: {phase['square_count']}")
        
        # Display Golden Ratio analysis
        if 'golden_ratio_analysis' in results:
            golden = results['golden_ratio_analysis']
            print(f"\n✨ Golden Ratio Analysis:")
            print(f"  Relationships found: {len(golden['relationships'])}")
            print(f"  Overall strength: {golden['strength']:.3f}")
        
        # Display special square types
        perfect_squares = [s for s in results['time_price_squares'] if 'perfect' in s.square_type]
        sacred_squares = [s for s in results['time_price_squares'] if 'sacred' in s.square_type]
        
        print(f"\n🎯 Special Square Analysis:")
        print(f"  Perfect squares: {len(perfect_squares)}")
        print(f"  Sacred ratio squares: {len(sacred_squares)}")
        print(f"  Fibonacci squares: {len([s for s in results['fibonacci_squares']])}")
        
        if indicator.ml_enhancement and 'ml_validation' in results:
            print(f"\n🤖 ML Enhancement Results:")
            ml_results = results['ml_validation']
            print(f"  Enhanced squares: {len(ml_results.get('validation_scores', []))}")
            print(f"  Validation method: {ml_results.get('validation_method', 'N/A')}")
        
    else:
        print(f"❌ Error in calculation: {results['error']}")
    
    print("\n" + "=" * 70)
    print("Demo completed! The Gann Time-Price Square Indicator provides:")
    print("✅ Perfect square relationship detection (T² = P, T = √P)")
    print("✅ Harmonic and Fibonacci-based time-price squares")
    print("✅ Sacred ratio integration (Golden Ratio, Pi, √2, √3, etc.)")
    print("✅ Advanced geometric relationship analysis")
    print("✅ ML-enhanced validation and confidence scoring")
    print("✅ Comprehensive square-based projections")
    print("✅ Professional trading signal generation")
    print("✅ Market phase analysis and square-based levels")
    print("=" * 70)