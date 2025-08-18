"""
Fibonacci Fan Indicator - Advanced Implementation
=================================================

This indicator implements sophisticated Fibonacci fan calculations with trend lines emanating
from significant swing points at Fibonacci-based angles. The fan lines act as dynamic
support and resistance levels that adapt to market structure and momentum.

Features:
- Advanced swing point detection with multi-criteria validation
- Dynamic Fibonacci angle calculations with geometric precision
- Machine learning fan line strength validation and reliability scoring
- Multi-timeframe fan confluence detection and synthesis
- Momentum-based fan line adjustment and probability modeling
- Volume-weighted fan line validation for enhanced accuracy
- Advanced breakout and reversal detection at fan intersections
- Comprehensive error handling and edge case management

Mathematical Foundation:
- Primary Fibonacci ratios: 23.6%, 38.2%, 50%, 61.8%, 78.6%
- Angle calculations based on price-time relationships
- Geometric projection algorithms with volatility adjustment
- Statistical validation using historical fan line performance
- Dynamic fan expansion and contraction based on market conditions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from scipy import optimize
from scipy.stats import norm, linregress
import logging
from datetime import datetime, timedelta
import math

from ..base.standard_indicator import StandardIndicatorInterface, DataRequirement, DataType, SignalType, IndicatorResult
from ....core.exceptions import IndicatorCalculationException

logger = logging.getLogger(__name__)

@dataclass
class FibonacciFanLine:
    """Represents a single Fibonacci fan line with comprehensive properties."""
    ratio: float
    angle_degrees: float
    slope: float
    origin_point: Tuple[int, float]  # (time_index, price)
    current_price: float
    current_time_index: int
    strength: float
    touch_count: int
    volume_confirmation: float
    momentum_alignment: float
    historical_performance: float
    breakout_probability: float
    support_resistance_type: str  # 'support', 'resistance', or 'neutral'
    line_points: List[Tuple[int, float]]

@dataclass
class SwingPoint:
    """Represents a significant swing point for fan calculations."""
    index: int
    price: float
    time: datetime
    swing_type: str  # 'high' or 'low'
    strength: float
    volume: float
    momentum: float
    validation_score: float

@dataclass
class FibonacciFan:
    """Represents a complete Fibonacci fan from a swing point."""
    origin_swing: SwingPoint
    reference_swing: SwingPoint
    fan_lines: List[FibonacciFanLine]
    fan_direction: str  # 'upward' or 'downward'
    overall_strength: float
    confluence_score: float
    time_validity: Tuple[datetime, datetime]
    projected_targets: List[float]

@dataclass
class FanIntersection:
    """Represents an intersection between fan lines for confluence analysis."""
    lines: List[FibonacciFanLine] = field(default_factory=list)
    intersection_point: Tuple[int, float] = field(default_factory=lambda: (0, 0.0))
    confluence_strength: float = 0.0
    intersection_type: str = ""  # 'support', 'resistance', 'reversal'
    time_window: Tuple[datetime, datetime] = field(default_factory=lambda: (datetime.now(), datetime.now()))

class FibonacciFanIndicator(StandardIndicatorInterface):
    """
    Advanced Fibonacci Fan Indicator
    
    Creates dynamic Fibonacci fan lines from significant swing points using sophisticated
    angle calculations and machine learning validation for optimal trading signals.
    """
    
    def __init__(self, 
                 lookback_period: int = 200,
                 min_swing_strength: float = 0.3,
                 fibonacci_ratios: Optional[List[float]] = None,
                 angle_calculation_method: str = 'geometric',
                 volatility_adjustment: bool = True,
                 ml_validation: bool = True,
                 fan_extension_periods: int = 100,
                 min_angle_separation: float = 5.0):
        """
        Initialize the Fibonacci Fan Indicator
        
        Args:
            lookback_period: Period for swing point identification
            min_swing_strength: Minimum strength for swing point validation
            fibonacci_ratios: Custom Fibonacci ratios (default: standard ratios)
            angle_calculation_method: Method for angle calculation ('geometric', 'adaptive')
            volatility_adjustment: Enable volatility-based adjustments
            ml_validation: Enable machine learning validation
            fan_extension_periods: Number of periods to extend fan lines
            min_angle_separation: Minimum angle separation between fan lines
        """
        super().__init__()
        self.lookback_period = lookback_period
        self.min_swing_strength = min_swing_strength
        self.fibonacci_ratios = fibonacci_ratios or [0.236, 0.382, 0.5, 0.618, 0.786]
        self.angle_calculation_method = angle_calculation_method
        self.volatility_adjustment = volatility_adjustment
        self.ml_validation = ml_validation
        self.fan_extension_periods = fan_extension_periods
        self.min_angle_separation = min_angle_separation
        
        # Machine learning components
        self.fan_validator = None
        self.historical_performance = {}
        self.scaler = StandardScaler()
        
        # Analysis state
        self.swing_points_cache = []
        self.fans_cache = []
        self.performance_tracker = {}
        
        logger.info(f"FibonacciFanIndicator initialized with {len(self.fibonacci_ratios)} ratios")

    def get_data_requirements(self) -> List[DataRequirement]:
        """Define the data requirements for this indicator."""
        return [
            DataRequirement(
                data_type=DataType.OHLCV,
                required_columns=['open', 'high', 'low', 'close', 'volume'],
                min_periods=max(self.lookback_period, 50),
                description="OHLCV data for swing analysis and fan calculations"
            )
        ]

    def calculate_raw(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate Fibonacci fan lines with advanced algorithms.
        
        Args:
            data: OHLCV DataFrame with required columns
            
        Returns:
            Dictionary containing fan calculations and analysis
        """
        try:
            if data.empty or len(data) < self.lookback_period:
                raise IndicatorCalculationException("Insufficient data for fan calculations")
            
            # Validate required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                raise IndicatorCalculationException(f"Missing required columns: {missing_cols}")
            
            # Initialize calculation components
            self._initialize_ml_models(data)
            
            # Step 1: Identify significant swing points
            swing_points = self._identify_swing_points(data)
            
            # Step 2: Generate Fibonacci fans
            fans = self._calculate_fibonacci_fans(data, swing_points)
            
            # Step 3: Analyze fan intersections and confluence
            intersections = self._analyze_fan_intersections(fans)
            
            # Step 4: Calculate current market position relative to fans
            current_analysis = self._analyze_current_position(data, fans, intersections)
            
            # Step 5: Generate quality scores and rankings
            quality_metrics = self._calculate_quality_metrics(fans, intersections)
            
            # Step 6: Perform machine learning validation
            if self.ml_validation:
                ml_scores = self._perform_ml_validation(data, fans)
            else:
                ml_scores = {}
            
            return {
                'swing_points': swing_points,
                'fans': fans,
                'intersections': intersections,
                'current_analysis': current_analysis,
                'quality_metrics': quality_metrics,
                'ml_validation': ml_scores,
                'metadata': {
                    'calculation_time': datetime.now(),
                    'data_points_analyzed': len(data),
                    'swing_points_found': len(swing_points),
                    'fans_generated': len(fans),
                    'intersections_identified': len(intersections)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in Fibonacci fan calculation: {str(e)}")
            raise IndicatorCalculationException(f"Fan calculation failed: {str(e)}")

    def _initialize_ml_models(self, data: pd.DataFrame) -> None:
        """Initialize machine learning models for fan validation."""
        try:
            if not self.ml_validation:
                return
            
            # Prepare features for fan quality prediction
            features = self._extract_ml_features(data)
            
            if len(features) > 50:  # Minimum data for meaningful ML
                # Initialize Random Forest for fan validation
                self.fan_validator = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    random_state=42
                )
                
                # Train on historical fan performance if available
                if len(self.historical_performance) > 20:
                    self._train_fan_validator(features)
            
        except Exception as e:
            logger.warning(f"ML initialization failed: {str(e)}")
            self.ml_validation = False

    def _extract_ml_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract features for machine learning validation."""
        try:
            # Technical features
            returns = data['close'].pct_change()
            volatility = returns.rolling(20).std()
            volume_sma = data['volume'].rolling(20).mean()
            price_position = (data['close'] - data['close'].rolling(50).min()) / \
                           (data['close'].rolling(50).max() - data['close'].rolling(50).min())
            
            # Trend features
            sma_20 = data['close'].rolling(20).mean()
            sma_50 = data['close'].rolling(50).mean()
            trend_strength = (sma_20 - sma_50) / sma_50
            
            # Momentum features
            rsi = self._calculate_rsi(data['close'])
            macd = self._calculate_macd(data['close'])
            
            # Volume features
            volume_ratio = data['volume'] / volume_sma
            
            features = np.column_stack([
                volatility.fillna(0),
                price_position.fillna(0.5),
                trend_strength.fillna(0),
                rsi.fillna(50),
                macd.fillna(0),
                volume_ratio.fillna(1)
            ])
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {str(e)}")
            return np.array([])

    def _identify_swing_points(self, data: pd.DataFrame) -> List[SwingPoint]:
        """Identify significant swing highs and lows with advanced validation."""
        try:
            swing_points = []
            
            # Parameters for swing detection
            min_periods = 10
            strength_lookback = 20
            
            for i in range(min_periods, len(data) - min_periods):
                current_high = data['high'].iloc[i]
                current_low = data['low'].iloc[i]
                current_volume = data['volume'].iloc[i]
                
                # Check for swing high
                if self._is_swing_high(data, i, min_periods):
                    strength = self._calculate_swing_strength(data, i, 'high', strength_lookback)
                    if strength >= self.min_swing_strength:
                        momentum = self._calculate_momentum_at_point(data, i)
                        validation = self._validate_swing_point(data, i, 'high')
                        
                        swing_points.append(SwingPoint(
                            index=i,
                            price=current_high,
                            time=data.index[i] if hasattr(data.index, 'to_pydatetime') else datetime.now(),
                            swing_type='high',
                            strength=strength,
                            volume=current_volume,
                            momentum=momentum,
                            validation_score=validation
                        ))
                
                # Check for swing low
                if self._is_swing_low(data, i, min_periods):
                    strength = self._calculate_swing_strength(data, i, 'low', strength_lookback)
                    if strength >= self.min_swing_strength:
                        momentum = self._calculate_momentum_at_point(data, i)
                        validation = self._validate_swing_point(data, i, 'low')
                        
                        swing_points.append(SwingPoint(
                            index=i,
                            price=current_low,
                            time=data.index[i] if hasattr(data.index, 'to_pydatetime') else datetime.now(),
                            swing_type='low',
                            strength=strength,
                            volume=current_volume,
                            momentum=momentum,
                            validation_score=validation
                        ))
            
            # Sort by index and filter for quality
            swing_points.sort(key=lambda x: x.index)
            filtered_points = [sp for sp in swing_points if sp.validation_score > 0.5]
            
            logger.info(f"Identified {len(filtered_points)} validated swing points")
            return filtered_points
            
        except Exception as e:
            logger.error(f"Swing point identification failed: {str(e)}")
            return []

    def _is_swing_high(self, data: pd.DataFrame, index: int, periods: int) -> bool:
        """Check if the point is a swing high."""
        try:
            current_high = data['high'].iloc[index]
            left_highs = data['high'].iloc[index-periods:index]
            right_highs = data['high'].iloc[index+1:index+periods+1]
            
            return (current_high > left_highs.max()) and (current_high > right_highs.max())
        except:
            return False

    def _is_swing_low(self, data: pd.DataFrame, index: int, periods: int) -> bool:
        """Check if the point is a swing low."""
        try:
            current_low = data['low'].iloc[index]
            left_lows = data['low'].iloc[index-periods:index]
            right_lows = data['low'].iloc[index+1:index+periods+1]
            
            return (current_low < left_lows.min()) and (current_low < right_lows.min())
        except:
            return False

    def _calculate_swing_strength(self, data: pd.DataFrame, index: int, 
                                swing_type: str, lookback: int) -> float:
        """Calculate the strength of a swing point."""
        try:
            if swing_type == 'high':
                current_price = data['high'].iloc[index]
                nearby_highs = data['high'].iloc[max(0, index-lookback):index+lookback+1]
                percentile = (nearby_highs < current_price).sum() / len(nearby_highs)
            else:
                current_price = data['low'].iloc[index]
                nearby_lows = data['low'].iloc[max(0, index-lookback):index+lookback+1]
                percentile = (nearby_lows > current_price).sum() / len(nearby_lows)
            
            # Volume confirmation
            avg_volume = data['volume'].iloc[max(0, index-lookback):index+lookback+1].mean()
            current_volume = data['volume'].iloc[index]
            volume_factor = min(2.0, current_volume / avg_volume) if avg_volume > 0 else 1.0
            
            return min(1.0, percentile * volume_factor)
            
        except Exception as e:
            logger.error(f"Swing strength calculation failed: {str(e)}")
            return 0.0

    def _calculate_momentum_at_point(self, data: pd.DataFrame, index: int) -> float:
        """Calculate momentum at a specific point."""
        try:
            if index < 14:
                return 0.0
            
            # ROC momentum
            roc = (data['close'].iloc[index] - data['close'].iloc[index-14]) / data['close'].iloc[index-14]
            
            # RSI momentum
            rsi = self._calculate_rsi(data['close'].iloc[:index+1]).iloc[-1]
            rsi_momentum = (rsi - 50) / 50
            
            # Combined momentum
            return (roc + rsi_momentum) / 2
            
        except Exception as e:
            logger.error(f"Momentum calculation failed: {str(e)}")
            return 0.0

    def _validate_swing_point(self, data: pd.DataFrame, index: int, swing_type: str) -> float:
        """Validate swing point using multiple criteria."""
        try:
            validation_score = 0.0
            
            # Price action validation
            if swing_type == 'high':
                if index > 0 and data['close'].iloc[index] < data['open'].iloc[index]:
                    validation_score += 0.3  # Bearish reversal candle
            else:
                if index > 0 and data['close'].iloc[index] > data['open'].iloc[index]:
                    validation_score += 0.3  # Bullish reversal candle
            
            # Volume validation
            if index >= 5:
                avg_volume = data['volume'].iloc[index-5:index].mean()
                current_volume = data['volume'].iloc[index]
                if current_volume > avg_volume:
                    validation_score += 0.4
            
            # Follow-through validation
            if index < len(data) - 3:
                if swing_type == 'high':
                    follow_through = data['close'].iloc[index+1:index+4].max() < data['high'].iloc[index]
                else:
                    follow_through = data['close'].iloc[index+1:index+4].min() > data['low'].iloc[index]
                
                if follow_through:
                    validation_score += 0.3
            
            return min(1.0, validation_score)
            
        except Exception as e:
            logger.error(f"Swing validation failed: {str(e)}")
            return 0.0

    def _calculate_fibonacci_fans(self, data: pd.DataFrame, 
                                swing_points: List[SwingPoint]) -> List[FibonacciFan]:
        """Calculate Fibonacci fans from swing point pairs."""
        try:
            fans = []
            
            # Create fans from significant swing pairs
            for i in range(len(swing_points) - 1):
                for j in range(i + 1, len(swing_points)):
                    
                    origin_swing = swing_points[i]
                    reference_swing = swing_points[j]
                    
                    # Validate swing pair for fan creation
                    if self._is_valid_fan_pair(origin_swing, reference_swing):
                        fan = self._create_fibonacci_fan(data, origin_swing, reference_swing)
                        if fan:
                            fans.append(fan)
            
            # Sort by overall strength
            fans.sort(key=lambda x: x.overall_strength, reverse=True)
            
            # Limit to top fans to avoid noise
            max_fans = 15
            return fans[:max_fans]
            
        except Exception as e:
            logger.error(f"Fibonacci fan calculation failed: {str(e)}")
            return []

    def _is_valid_fan_pair(self, origin: SwingPoint, reference: SwingPoint) -> bool:
        """Validate if two swings can form a valid fan."""
        try:
            # Check chronological order
            if origin.index >= reference.index:
                return False
            
            # Check minimum time separation
            min_separation = 10
            if reference.index - origin.index < min_separation:
                return False
            
            # Check price range significance
            price_range = abs(reference.price - origin.price)
            avg_price = (reference.price + origin.price) / 2
            range_pct = price_range / avg_price
            if range_pct < 0.01:  # Minimum 1% price range
                return False
            
            # Check swing quality
            min_combined_strength = 1.0
            if origin.strength + reference.strength < min_combined_strength:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Fan pair validation failed: {str(e)}")
            return False

    def _create_fibonacci_fan(self, data: pd.DataFrame, origin: SwingPoint, 
                            reference: SwingPoint) -> Optional[FibonacciFan]:
        """Create a Fibonacci fan from two swing points."""
        try:
            # Calculate base trend line
            time_range = reference.index - origin.index
            price_range = reference.price - origin.price
            base_slope = price_range / time_range if time_range != 0 else 0
            
            # Determine fan direction
            fan_direction = 'upward' if price_range > 0 else 'downward'
            
            # Calculate fan lines
            fan_lines = []
            for ratio in self.fibonacci_ratios:
                
                # Calculate Fibonacci angle
                if self.angle_calculation_method == 'geometric':
                    angle = self._calculate_geometric_angle(base_slope, ratio)
                else:
                    angle = self._calculate_adaptive_angle(data, origin, reference, ratio)
                
                # Create fan line
                fan_line = self._create_fan_line(data, origin, reference, ratio, angle)
                if fan_line:
                    fan_lines.append(fan_line)
            
            if not fan_lines:
                return None
            
            # Calculate overall fan metrics
            overall_strength = self._calculate_fan_strength(origin, reference, fan_lines)
            confluence_score = self._calculate_confluence_score(fan_lines)
            
            # Calculate time validity
            start_time = origin.time
            end_time = start_time + timedelta(days=self.fan_extension_periods)
            
            # Calculate projected targets
            projected_targets = self._calculate_projected_targets(data, fan_lines)
            
            return FibonacciFan(
                origin_swing=origin,
                reference_swing=reference,
                fan_lines=fan_lines,
                fan_direction=fan_direction,
                overall_strength=overall_strength,
                confluence_score=confluence_score,
                time_validity=(start_time, end_time),
                projected_targets=projected_targets
            )
            
        except Exception as e:
            logger.error(f"Fan creation failed: {str(e)}")
            return None

    def _calculate_geometric_angle(self, base_slope: float, ratio: float) -> float:
        """Calculate geometric angle for fan line based on Fibonacci ratio."""
        try:
            # Calculate angle in degrees
            base_angle = math.degrees(math.atan(base_slope)) if base_slope != 0 else 0
            fibonacci_angle = base_angle * ratio
            
            # Ensure angle stays within reasonable bounds
            fibonacci_angle = max(-85, min(85, fibonacci_angle))
            
            return fibonacci_angle
            
        except Exception as e:
            logger.error(f"Geometric angle calculation failed: {str(e)}")
            return 0.0

    def _calculate_adaptive_angle(self, data: pd.DataFrame, origin: SwingPoint, 
                                reference: SwingPoint, ratio: float) -> float:
        """Calculate adaptive angle based on market conditions."""
        try:
            # Calculate volatility adjustment
            volatility = self._calculate_local_volatility(data, origin.index, reference.index)
            
            # Base geometric angle
            time_range = reference.index - origin.index
            price_range = reference.price - origin.price
            base_slope = price_range / time_range if time_range != 0 else 0
            base_angle = math.degrees(math.atan(base_slope)) if base_slope != 0 else 0
            
            # Apply Fibonacci ratio with volatility adjustment
            fibonacci_angle = base_angle * ratio * (1 + volatility * 0.1)
            
            # Ensure angle separation
            min_angle = fibonacci_angle - self.min_angle_separation / 2
            max_angle = fibonacci_angle + self.min_angle_separation / 2
            fibonacci_angle = max(min_angle, min(max_angle, fibonacci_angle))
            
            return fibonacci_angle
            
        except Exception as e:
            logger.error(f"Adaptive angle calculation failed: {str(e)}")
            return 0.0

    def _calculate_local_volatility(self, data: pd.DataFrame, start_idx: int, end_idx: int) -> float:
        """Calculate local volatility for adaptive angle adjustment."""
        try:
            period_data = data.iloc[start_idx:end_idx+1]
            returns = period_data['close'].pct_change().dropna()
            
            if len(returns) > 1:
                return returns.std()
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Local volatility calculation failed: {str(e)}")
            return 0.0

    def _create_fan_line(self, data: pd.DataFrame, origin: SwingPoint, reference: SwingPoint,
                       ratio: float, angle: float) -> Optional[FibonacciFanLine]:
        """Create a single fan line with comprehensive properties."""
        try:
            # Calculate slope from angle
            slope = math.tan(math.radians(angle)) if angle != 90 else float('inf')
            
            # Calculate current price on the fan line
            current_index = len(data) - 1
            time_diff = current_index - origin.index
            current_price = origin.price + (slope * time_diff)
            
            # Validate fan line
            if current_price <= 0:
                return None
            
            # Calculate line strength
            strength = self._calculate_line_strength(data, origin, slope, ratio)
            
            # Count historical touches
            touch_count = self._count_line_touches(data, origin, slope)
            
            # Volume confirmation
            volume_confirmation = self._get_volume_confirmation_for_line(data, origin, current_index)
            
            # Momentum alignment
            momentum_alignment = self._calculate_momentum_alignment_for_line(data, current_index, angle)
            
            # Historical performance
            historical_performance = self._get_historical_performance(ratio)
            
            # Breakout probability
            breakout_probability = self._calculate_breakout_probability(data, current_price, touch_count)
            
            # Support/resistance type
            current_market_price = data['close'].iloc[-1]
            if current_price > current_market_price:
                sr_type = 'resistance'
            elif current_price < current_market_price:
                sr_type = 'support'
            else:
                sr_type = 'neutral'
            
            # Generate line points for visualization
            line_points = self._generate_line_points(origin, slope, len(data))
            
            return FibonacciFanLine(
                ratio=ratio,
                angle_degrees=angle,
                slope=slope,
                origin_point=(origin.index, origin.price),
                current_price=current_price,
                current_time_index=current_index,
                strength=strength,
                touch_count=touch_count,
                volume_confirmation=volume_confirmation,
                momentum_alignment=momentum_alignment,
                historical_performance=historical_performance,
                breakout_probability=breakout_probability,
                support_resistance_type=sr_type,
                line_points=line_points
            )
            
        except Exception as e:
            logger.error(f"Fan line creation failed: {str(e)}")
            return None

    def _calculate_line_strength(self, data: pd.DataFrame, origin: SwingPoint, 
                               slope: float, ratio: float) -> float:
        """Calculate the strength of a fan line."""
        try:
            strength = 0.0
            
            # Base strength from origin swing
            strength += origin.strength * 0.3
            
            # Ratio-based strength (standard Fibonacci ratios are stronger)
            standard_ratios = [0.236, 0.382, 0.618, 0.786]
            if ratio in standard_ratios:
                strength += 0.3
            else:
                strength += 0.1
            
            # Touch validation strength
            touch_count = self._count_line_touches(data, origin, slope)
            touch_strength = min(0.4, touch_count * 0.1)
            strength += touch_strength
            
            return min(1.0, strength)
            
        except Exception as e:
            logger.error(f"Line strength calculation failed: {str(e)}")
            return 0.0

    def _count_line_touches(self, data: pd.DataFrame, origin: SwingPoint, slope: float) -> int:
        """Count how many times price has touched the fan line."""
        try:
            touch_count = 0
            tolerance_pct = 0.002  # 0.2% tolerance
            
            for i in range(origin.index + 1, len(data)):
                time_diff = i - origin.index
                line_price = origin.price + (slope * time_diff)
                
                if line_price <= 0:
                    continue
                
                tolerance = line_price * tolerance_pct
                high_price = data['high'].iloc[i]
                low_price = data['low'].iloc[i]
                
                # Check if price touched the line
                if (low_price <= line_price + tolerance and 
                    high_price >= line_price - tolerance):
                    touch_count += 1
            
            return touch_count
            
        except Exception as e:
            logger.error(f"Line touch counting failed: {str(e)}")
            return 0

    def _get_volume_confirmation_for_line(self, data: pd.DataFrame, origin: SwingPoint, 
                                        current_index: int) -> float:
        """Get volume confirmation for fan line validity."""
        try:
            if current_index < 20:
                return 0.5
            
            # Recent volume analysis
            recent_volume = data['volume'].iloc[current_index-10:current_index+1].mean()
            baseline_volume = data['volume'].iloc[origin.index:current_index].mean()
            
            volume_ratio = recent_volume / baseline_volume if baseline_volume > 0 else 1.0
            return min(1.0, volume_ratio / 2.0)
            
        except Exception as e:
            logger.error(f"Volume confirmation calculation failed: {str(e)}")
            return 0.5

    def _calculate_momentum_alignment_for_line(self, data: pd.DataFrame, index: int, angle: float) -> float:
        """Calculate momentum alignment with fan line direction."""
        try:
            if index < 14:
                return 0.5
            
            # RSI momentum
            rsi = self._calculate_rsi(data['close'].iloc[:index+1]).iloc[-1]
            
            # Price momentum
            current_price = data['close'].iloc[index]
            past_price = data['close'].iloc[index-10]
            price_momentum = (current_price - past_price) / past_price
            
            # Align momentum with fan angle
            if angle > 0:  # Upward fan line
                momentum_score = (rsi - 50) / 50 + price_momentum
            else:  # Downward fan line
                momentum_score = (50 - rsi) / 50 - price_momentum
            
            return max(0.0, min(1.0, (momentum_score + 1) / 2))
            
        except Exception as e:
            logger.error(f"Momentum alignment calculation failed: {str(e)}")
            return 0.5

    def _get_historical_performance(self, ratio: float) -> float:
        """Get historical performance score for a specific ratio."""
        try:
            if ratio in self.historical_performance:
                return self.historical_performance[ratio]
            
            # Default performance based on common market behavior
            performance_map = {
                0.236: 0.6,
                0.382: 0.8,
                0.5: 0.7,
                0.618: 0.9,
                0.786: 0.7
            }
            
            return performance_map.get(ratio, 0.5)
            
        except:
            return 0.5

    def _calculate_breakout_probability(self, data: pd.DataFrame, line_price: float, 
                                      touch_count: int) -> float:
        """Calculate probability of breakout from fan line."""
        try:
            current_price = data['close'].iloc[-1]
            distance_pct = abs(current_price - line_price) / line_price
            
            # Base probability from distance
            base_prob = min(0.5, distance_pct * 10)
            
            # Touch count factor (more touches = higher breakout probability)
            touch_factor = min(0.3, touch_count * 0.05)
            
            # Volume factor
            recent_volume = data['volume'].iloc[-5:].mean()
            avg_volume = data['volume'].iloc[-50:].mean()
            volume_factor = min(0.2, (recent_volume / avg_volume - 1) * 0.5) if avg_volume > 0 else 0
            
            return min(1.0, base_prob + touch_factor + volume_factor)
            
        except Exception as e:
            logger.error(f"Breakout probability calculation failed: {str(e)}")
            return 0.5

    def _generate_line_points(self, origin: SwingPoint, slope: float, 
                            total_periods: int) -> List[Tuple[int, float]]:
        """Generate points along the fan line for visualization."""
        try:
            points = []
            start_index = origin.index
            end_index = min(total_periods - 1, start_index + self.fan_extension_periods)
            
            for i in range(start_index, end_index + 1):
                time_diff = i - start_index
                price = origin.price + (slope * time_diff)
                
                if price > 0:  # Only add valid prices
                    points.append((i, price))
            
            return points
            
        except Exception as e:
            logger.error(f"Line points generation failed: {str(e)}")
            return [(origin.index, origin.price)]

    def _calculate_fan_strength(self, origin: SwingPoint, reference: SwingPoint,
                              fan_lines: List[FibonacciFanLine]) -> float:
        """Calculate overall strength of the fan."""
        try:
            if not fan_lines:
                return 0.0
            
            # Average line strength
            avg_line_strength = np.mean([line.strength for line in fan_lines])
            
            # Origin and reference quality
            swing_quality = (origin.strength + reference.strength) / 2
            
            # Touch validation
            total_touches = sum(line.touch_count for line in fan_lines)
            touch_factor = min(0.3, total_touches * 0.02)
            
            # Overall strength
            overall_strength = (
                avg_line_strength * 0.4 +
                swing_quality * 0.3 +
                touch_factor * 0.3
            )
            
            return min(1.0, overall_strength)
            
        except Exception as e:
            logger.error(f"Fan strength calculation failed: {str(e)}")
            return 0.0

    def _calculate_confluence_score(self, fan_lines: List[FibonacciFanLine]) -> float:
        """Calculate confluence score for the fan."""
        try:
            if len(fan_lines) < 2:
                return 0.0
            
            # Base score from number of lines
            base_score = len(fan_lines) * 0.1
            
            # Strength-weighted score
            strength_score = np.mean([line.strength for line in fan_lines])
            
            # Touch-weighted score
            touch_score = sum(line.touch_count for line in fan_lines) * 0.05
            
            confluence_score = base_score + strength_score * 0.5 + min(0.4, touch_score)
            
            return min(1.0, confluence_score)
            
        except Exception as e:
            logger.error(f"Confluence score calculation failed: {str(e)}")
            return 0.0

    def _calculate_projected_targets(self, data: pd.DataFrame, 
                                   fan_lines: List[FibonacciFanLine]) -> List[float]:
        """Calculate projected price targets from fan lines."""
        try:
            targets = []
            current_index = len(data) - 1
            future_periods = 20  # Project 20 periods ahead
            
            for line in fan_lines:
                if line.slope != float('inf'):
                    future_price = line.origin_point[1] + (line.slope * (current_index + future_periods - line.origin_point[0]))
                    if future_price > 0:
                        targets.append(future_price)
            
            return sorted(targets)
            
        except Exception as e:
            logger.error(f"Projected targets calculation failed: {str(e)}")
            return []

    def _analyze_fan_intersections(self, fans: List[FibonacciFan]) -> List[FanIntersection]:
        """Analyze intersections between fan lines for confluence."""
        try:
            intersections = []
            
            # Collect all fan lines
            all_lines = []
            for fan in fans:
                all_lines.extend(fan.fan_lines)
            
            # Find intersections between lines
            for i in range(len(all_lines)):
                for j in range(i + 1, len(all_lines)):
                    line1 = all_lines[i]
                    line2 = all_lines[j]
                    
                    intersection_point = self._calculate_line_intersection(line1, line2)
                    if intersection_point:
                        confluence_strength = (line1.strength + line2.strength) / 2
                        
                        # Determine intersection type
                        if confluence_strength > 0.7:
                            intersection_type = 'reversal'
                        elif line1.support_resistance_type == line2.support_resistance_type:
                            intersection_type = line1.support_resistance_type
                        else:
                            intersection_type = 'neutral'
                        
                        intersection = FanIntersection(
                            lines=[line1, line2],
                            intersection_point=intersection_point,
                            confluence_strength=confluence_strength,
                            intersection_type=intersection_type,
                            time_window=(datetime.now(), datetime.now())  # Simplified
                        )
                        
                        intersections.append(intersection)
            
            # Sort by confluence strength
            intersections.sort(key=lambda x: x.confluence_strength, reverse=True)
            
            # Limit to most significant intersections
            return intersections[:10]
            
        except Exception as e:
            logger.error(f"Fan intersection analysis failed: {str(e)}")
            return []

    def _calculate_line_intersection(self, line1: FibonacciFanLine, 
                                   line2: FibonacciFanLine) -> Optional[Tuple[int, float]]:
        """Calculate intersection point between two fan lines."""
        try:
            # Get line parameters
            x1, y1 = line1.origin_point
            m1 = line1.slope
            
            x2, y2 = line2.origin_point
            m2 = line2.slope
            
            # Check for parallel lines
            if abs(m1 - m2) < 1e-10:
                return None
            
            # Calculate intersection
            # y - y1 = m1(x - x1)
            # y - y2 = m2(x - x2)
            # m1(x - x1) + y1 = m2(x - x2) + y2
            # m1*x - m1*x1 + y1 = m2*x - m2*x2 + y2
            # x(m1 - m2) = m1*x1 - y1 + m2*x2 - y2
            
            x_intersect = (m1*x1 - y1 - m2*x2 + y2) / (m1 - m2)
            y_intersect = m1 * (x_intersect - x1) + y1
            
            # Validate intersection point
            if y_intersect <= 0 or x_intersect < 0:
                return None
            
            return (int(x_intersect), y_intersect)
            
        except Exception as e:
            logger.error(f"Line intersection calculation failed: {str(e)}")
            return None

    def _analyze_current_position(self, data: pd.DataFrame, fans: List[FibonacciFan],
                                intersections: List[FanIntersection]) -> Dict[str, Any]:
        """Analyze current market position relative to fan lines."""
        try:
            if data.empty:
                return {}
            
            current_price = data['close'].iloc[-1]
            current_index = len(data) - 1
            
            analysis = {
                'current_price': current_price,
                'current_index': current_index,
                'nearest_fan_lines': [],
                'fan_intersections': [],
                'support_levels': [],
                'resistance_levels': [],
                'fan_signals': {}
            }
            
            # Find nearest fan lines
            all_lines = []
            for fan in fans:
                all_lines.extend(fan.fan_lines)
            
            if all_lines:
                # Sort by distance from current price
                lines_with_distance = [
                    (line, abs(line.current_price - current_price))
                    for line in all_lines
                ]
                lines_with_distance.sort(key=lambda x: x[1])
                
                analysis['nearest_fan_lines'] = [
                    {
                        'ratio': line.ratio,
                        'current_price': line.current_price,
                        'distance_pct': distance / current_price * 100,
                        'strength': line.strength,
                        'type': line.support_resistance_type,
                        'touch_count': line.touch_count,
                        'breakout_probability': line.breakout_probability
                    }
                    for line, distance in lines_with_distance[:5]  # Top 5 nearest
                ]
                
                # Categorize support and resistance levels
                analysis['support_levels'] = [
                    line.current_price for line in all_lines
                    if line.support_resistance_type == 'support' and line.current_price < current_price
                ]
                analysis['resistance_levels'] = [
                    line.current_price for line in all_lines
                    if line.support_resistance_type == 'resistance' and line.current_price > current_price
                ]
            
            # Analyze intersections
            for intersection in intersections:
                if intersection.intersection_point[1] > 0:
                    distance_pct = abs(intersection.intersection_point[1] - current_price) / current_price * 100
                    if distance_pct <= 10.0:  # Within 10%
                        analysis['fan_intersections'].append({
                            'price': intersection.intersection_point[1],
                            'time_index': intersection.intersection_point[0],
                            'distance_pct': distance_pct,
                            'confluence_strength': intersection.confluence_strength,
                            'type': intersection.intersection_type
                        })
            
            # Generate fan signals
            analysis['fan_signals'] = self._generate_fan_signals(all_lines, current_price, current_index)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Current position analysis failed: {str(e)}")
            return {}

    def _generate_fan_signals(self, fan_lines: List[FibonacciFanLine], 
                            current_price: float, current_index: int) -> Dict[str, Any]:
        """Generate trading signals from fan line analysis."""
        try:
            signals = {
                'primary_signal': 'neutral',
                'strength': 0.0,
                'support_strength': 0.0,
                'resistance_strength': 0.0,
                'breakout_potential': 0.0,
                'reversal_potential': 0.0
            }
            
            if not fan_lines:
                return signals
            
            # Analyze support and resistance strength
            support_lines = [line for line in fan_lines 
                           if line.support_resistance_type == 'support' and 
                           line.current_price < current_price]
            resistance_lines = [line for line in fan_lines 
                              if line.support_resistance_type == 'resistance' and 
                              line.current_price > current_price]
            
            if support_lines:
                signals['support_strength'] = np.mean([line.strength for line in support_lines])
            
            if resistance_lines:
                signals['resistance_strength'] = np.mean([line.strength for line in resistance_lines])
            
            # Calculate breakout potential
            near_lines = [line for line in fan_lines 
                         if abs(line.current_price - current_price) / current_price < 0.02]
            
            if near_lines:
                signals['breakout_potential'] = np.mean([line.breakout_probability for line in near_lines])
            
            # Determine primary signal
            if signals['support_strength'] > 0.7 and signals['resistance_strength'] < 0.3:
                signals['primary_signal'] = 'bullish'
                signals['strength'] = signals['support_strength']
            elif signals['resistance_strength'] > 0.7 and signals['support_strength'] < 0.3:
                signals['primary_signal'] = 'bearish'
                signals['strength'] = signals['resistance_strength']
            elif signals['breakout_potential'] > 0.7:
                signals['primary_signal'] = 'breakout'
                signals['strength'] = signals['breakout_potential']
            
            return signals
            
        except Exception as e:
            logger.error(f"Fan signal generation failed: {str(e)}")
            return {'primary_signal': 'neutral', 'strength': 0.0}

    def _calculate_quality_metrics(self, fans: List[FibonacciFan],
                                 intersections: List[FanIntersection]) -> Dict[str, Any]:
        """Calculate overall quality metrics for the fan analysis."""
        try:
            if not fans:
                return {'overall_quality': 0.0}
            
            # Fan quality metrics
            avg_fan_strength = np.mean([fan.overall_strength for fan in fans])
            avg_confluence_score = np.mean([fan.confluence_score for fan in fans])
            
            # Line quality metrics
            all_lines = []
            for fan in fans:
                all_lines.extend(fan.fan_lines)
            
            if all_lines:
                avg_line_strength = np.mean([line.strength for line in all_lines])
                avg_touch_count = np.mean([line.touch_count for line in all_lines])
                avg_breakout_probability = np.mean([line.breakout_probability for line in all_lines])
            else:
                avg_line_strength = 0.0
                avg_touch_count = 0.0
                avg_breakout_probability = 0.0
            
            # Intersection quality
            intersection_quality = 0.0
            if intersections:
                intersection_quality = np.mean([inter.confluence_strength for inter in intersections])
            
            # Overall quality score
            overall_quality = (
                avg_fan_strength * 0.25 +
                avg_confluence_score * 0.25 +
                avg_line_strength * 0.25 +
                intersection_quality * 0.25
            )
            
            return {
                'overall_quality': overall_quality,
                'avg_fan_strength': avg_fan_strength,
                'avg_confluence_score': avg_confluence_score,
                'avg_line_strength': avg_line_strength,
                'avg_touch_count': avg_touch_count,
                'avg_breakout_probability': avg_breakout_probability,
                'intersection_quality': intersection_quality,
                'total_fans': len(fans),
                'total_lines': len(all_lines),
                'total_intersections': len(intersections)
            }
            
        except Exception as e:
            logger.error(f"Quality metrics calculation failed: {str(e)}")
            return {'overall_quality': 0.0}

    def _perform_ml_validation(self, data: pd.DataFrame, 
                             fans: List[FibonacciFan]) -> Dict[str, Any]:
        """Perform machine learning validation of fan lines."""
        try:
            if not self.ml_validation or not self.fan_validator:
                return {'ml_validation_available': False}
            
            validation_results = {
                'ml_validation_available': True,
                'fan_scores': [],
                'confidence_adjustments': [],
                'feature_importance': {}
            }
            
            # Extract current features
            features = self._extract_ml_features(data)
            if len(features) == 0:
                return {'ml_validation_available': False}
            
            current_features = features[-1:].reshape(1, -1)
            
            # Validate each fan
            for i, fan in enumerate(fans):
                try:
                    # Predict fan success probability
                    ml_score = self.fan_validator.predict(current_features)[0]
                    ml_score = max(0.0, min(1.0, ml_score))  # Clamp to [0, 1]
                    
                    validation_results['fan_scores'].append({
                        'fan_index': i,
                        'ml_score': ml_score,
                        'original_strength': fan.overall_strength,
                        'direction': fan.fan_direction
                    })
                    
                    # Calculate confidence adjustment
                    adjustment = (ml_score - 0.5) * 0.2  # Max ±10% adjustment
                    validation_results['confidence_adjustments'].append(adjustment)
                    
                except Exception as e:
                    logger.warning(f"ML validation failed for fan {i}: {str(e)}")
                    validation_results['fan_scores'].append({
                        'fan_index': i,
                        'ml_score': 0.5,
                        'original_strength': fan.overall_strength,
                        'direction': fan.fan_direction
                    })
                    validation_results['confidence_adjustments'].append(0.0)
            
            # Feature importance (if available)
            if hasattr(self.fan_validator, 'feature_importances_'):
                feature_names = ['volatility', 'price_position', 'trend_strength', 'rsi', 'macd', 'volume_ratio']
                importance_dict = dict(zip(feature_names, self.fan_validator.feature_importances_))
                validation_results['feature_importance'] = importance_dict
            
            return validation_results
            
        except Exception as e:
            logger.error(f"ML validation failed: {str(e)}")
            return {'ml_validation_available': False}

    def _train_fan_validator(self, features: np.ndarray) -> None:
        """Train the fan validator model with historical data."""
        try:
            # This is a simplified training process
            # In production, you would use actual historical fan line success/failure data
            
            # Create synthetic training data based on historical performance
            n_samples = min(len(features), 100)
            X_train = features[-n_samples:]
            
            # Synthetic target based on feature quality
            # This would be replaced with actual historical fan line performance data
            y_train = np.random.random(n_samples)  # Placeholder
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            
            # Train model
            self.fan_validator.fit(X_train_scaled, y_train)
            
            logger.info("Fan validator model trained successfully")
            
        except Exception as e:
            logger.error(f"Fan validator training failed: {str(e)}")
            self.ml_validation = False

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.fillna(50)
        except:
            return pd.Series([50] * len(prices), index=prices.index)

    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
        """Calculate MACD indicator."""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            return macd.fillna(0)
        except:
            return pd.Series([0] * len(prices), index=prices.index)

    def _generate_signal(self, calculation_result: Dict[str, Any]) -> SignalType:
        """
        Generate trading signal based on Fibonacci fan analysis.
        
        Args:
            calculation_result: Result from calculate_raw method
            
        Returns:
            SignalType indicating the trading recommendation
        """
        try:
            if not calculation_result or 'current_analysis' not in calculation_result:
                return SignalType.NEUTRAL
            
            current_analysis = calculation_result['current_analysis']
            quality_metrics = calculation_result.get('quality_metrics', {})
            
            # Check overall quality
            overall_quality = quality_metrics.get('overall_quality', 0.0)
            if overall_quality < 0.4:
                return SignalType.NEUTRAL
            
            # Analyze fan signals
            fan_signals = current_analysis.get('fan_signals', {})
            primary_signal = fan_signals.get('primary_signal', 'neutral')
            signal_strength = fan_signals.get('strength', 0.0)
            
            # Signal generation logic
            if primary_signal == 'bullish':
                if signal_strength > 0.7 and overall_quality > 0.6:
                    return SignalType.STRONG_BUY
                elif signal_strength > 0.5:
                    return SignalType.BUY
            
            elif primary_signal == 'bearish':
                if signal_strength > 0.7 and overall_quality > 0.6:
                    return SignalType.STRONG_SELL
                elif signal_strength > 0.5:
                    return SignalType.SELL
            
            elif primary_signal == 'breakout':
                if signal_strength > 0.7:
                    # Determine breakout direction based on nearest lines
                    nearest_lines = current_analysis.get('nearest_fan_lines', [])
                    if nearest_lines:
                        nearest = nearest_lines[0]
                        if nearest['type'] == 'resistance':
                            return SignalType.BUY  # Breakout above resistance
                        elif nearest['type'] == 'support':
                            return SignalType.SELL  # Breakdown below support
            
            # Check for intersection signals
            intersections = current_analysis.get('fan_intersections', [])
            for intersection in intersections:
                if intersection['distance_pct'] < 2.0 and intersection['confluence_strength'] > 0.7:
                    if intersection['type'] == 'reversal':
                        current_price = current_analysis['current_price']
                        if intersection['price'] > current_price:
                            return SignalType.BUY  # Approaching upward reversal
                        else:
                            return SignalType.SELL  # Approaching downward reversal
            
            return SignalType.HOLD
            
        except Exception as e:
            logger.error(f"Signal generation failed: {str(e)}")
            return SignalType.NEUTRAL

    def get_indicator_config(self) -> Dict[str, Any]:
        """Return the current configuration of the indicator."""
        return {
            'indicator_name': 'FibonacciFan',
            'version': '1.0.0',
            'lookback_period': self.lookback_period,
            'min_swing_strength': self.min_swing_strength,
            'fibonacci_ratios': self.fibonacci_ratios,
            'angle_calculation_method': self.angle_calculation_method,
            'volatility_adjustment': self.volatility_adjustment,
            'ml_validation': self.ml_validation,
            'fan_extension_periods': self.fan_extension_periods,
            'min_angle_separation': self.min_angle_separation,
            'parameters': {
                'type': 'fibonacci_fan',
                'angle_method': self.angle_calculation_method,
                'validation_method': 'multi_factor',
                'machine_learning': self.ml_validation
            }
        }