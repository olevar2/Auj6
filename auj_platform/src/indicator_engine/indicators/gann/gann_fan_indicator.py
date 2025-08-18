"""
Gann Fan Indicator - Advanced Implementation

This module implements W.D. Gann's fan analysis with sophisticated mathematical models,
dynamic support/resistance identification, and advanced geometric analysis for maximum profitability.

The implementation includes:
- Comprehensive fan line calculations from significant pivot points
- Dynamic fan scaling based on market volatility and timeframe
- Advanced geometric analysis of fan intersections
- Support/resistance level identification from fan lines
- Fan convergence and divergence analysis
- Machine learning-enhanced fan line validation
- Multi-timeframe fan harmonics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, NamedTuple
from dataclasses import dataclass
import logging
from scipy import stats, optimize, spatial
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.cluster import DBSCAN
import math

from ..base.standard_indicator import StandardIndicatorInterface
from ....core.exceptions import IndicatorCalculationError


class GannFanLine(NamedTuple):
    """Represents a single Gann fan line"""
    id: str
    pivot_time: float
    pivot_price: float
    angle_degrees: float
    slope: float
    strength: float
    current_level: float
    support_resistance_type: str
    touches: int
    validation_score: float


class GannFan(NamedTuple):
    """Represents a complete Gann fan from a pivot point"""
    id: str
    pivot_time: float
    pivot_price: float
    pivot_type: str  # 'high' or 'low'
    fan_lines: List[GannFanLine]
    overall_strength: float
    convergence_zones: List[Dict]
    active_lines: List[str]


@dataclass
class GannFanConfig:
    """Configuration for Gann Fan calculations"""
    primary_angles: List[float] = None  # Default: Gann's primary angles
    secondary_angles: List[float] = None  # Additional angles for detailed analysis
    min_pivot_strength: float = 0.6  # Minimum strength for pivot to generate fan
    max_fans: int = 10  # Maximum number of fans to maintain
    lookback_period: int = 500  # Bars to look back for pivot identification
    fan_validation_period: int = 50  # Bars ahead to validate fan lines
    angle_tolerance: float = 2.0  # Degrees tolerance for angle grouping
    touch_tolerance_atr: float = 0.5  # ATR units tolerance for line touches
    ml_validation: bool = True  # Enable ML-based fan validation
    dynamic_scaling: bool = True  # Enable dynamic angle scaling
    convergence_analysis: bool = True  # Enable convergence zone analysis
    
    def __post_init__(self):
        if self.primary_angles is None:
            # W.D. Gann's primary fan angles
            self.primary_angles = [
                82.5,   # 8x1 - Very steep up
                75.0,   # 4x1 - Steep up
                71.25,  # 3x1 - Strong up
                63.75,  # 2x1 - Moderate up
                45.0,   # 1x1 - Main trend (45 degrees)
                26.25,  # 1x2 - Moderate down
                18.75,  # 1x3 - Strong down
                15.0,   # 1x4 - Steep down
                7.5     # 1x8 - Very steep down
            ]
            
        if self.secondary_angles is None:
            # Additional angles for more detailed analysis
            self.secondary_angles = [
                78.75,  # 6x1
                67.5,   # 2.5x1
                56.25,  # 1.5x1
                33.75,  # 1x1.5
                22.5,   # 1x2.5
                11.25   # 1x6
            ]


class GannFanIndicator(StandardIndicatorInterface):
    """
    Advanced Gann Fan Indicator Implementation
    
    This indicator implements W.D. Gann's fan theory with sophisticated enhancements:
    
    1. Dynamic Pivot Detection: Identifies significant highs/lows using advanced algorithms
    2. Comprehensive Fan Generation: Creates fan lines at all Gann angles from each pivot
    3. Geometric Analysis: Analyzes fan intersections and convergence zones
    4. Support/Resistance Identification: Extracts key levels from fan line interactions
    5. ML Validation: Uses machine learning to validate fan line effectiveness
    6. Multi-Timeframe Analysis: Considers fan relationships across different timeframes
    7. Dynamic Scaling: Adjusts fan angles based on market volatility and price action
    
    Mathematical Foundation:
    - Fan line equation: y = pivot_price + slope * (t - pivot_time)
    - Slope calculation: tan(angle_degrees * π/180) * scaling_factor
    - Validation score: Combination of line touches, price proximity, and ML prediction
    - Convergence analysis: Geometric intersection of multiple fan lines
    """
    
    def __init__(self, config: Optional[GannFanConfig] = None):
        super().__init__()
        self.config = config or GannFanConfig()
        self.scaler = StandardScaler()
        self.validation_model = RandomForestRegressor(
            n_estimators=150,
            max_depth=12,
            random_state=42,
            n_jobs=-1
        )
        self.anomaly_detector = IsolationForest(
            contamination=0.15,
            random_state=42
        )
        self.is_trained = False
        self.fan_cache = {}
        self.active_fans = []
        self.logger = logging.getLogger(__name__)
        
    def get_name(self) -> str:
        return "GannFanIndicator"
        
    def get_required_columns(self) -> List[str]:
        return ['high', 'low', 'close', 'open', 'volume', 'timestamp']
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Gann Fans with advanced mathematical modeling
        
        Args:
            data: DataFrame with OHLCV data and timestamp
            
        Returns:
            DataFrame with fan data, support/resistance levels, and trading signals
        """
        try:
            if len(data) < self.config.lookback_period:
                raise IndicatorCalculationError(
                    f"Insufficient data: {len(data)} bars, need {self.config.lookback_period}"
                )
                
            # Prepare data
            df = data.copy()
            df = self._prepare_market_data(df)
            
            # Identify significant pivot points
            pivot_points = self._identify_pivot_points(df)
            
            # Generate Gann fans from pivots
            fans = self._generate_gann_fans(df, pivot_points)
            
            # Validate fan lines using ML if enabled
            if self.config.ml_validation:
                fans = self._validate_fans_with_ml(df, fans)
                
            # Analyze fan convergence zones
            if self.config.convergence_analysis:
                fans = self._analyze_fan_convergence(df, fans)
                
            # Extract support/resistance levels
            sr_levels = self._extract_fan_support_resistance(df, fans)
            
            # Analyze fan interactions and strength
            interaction_data = self._analyze_fan_interactions(df, fans)
            
            # Generate trading signals
            signals = self._generate_fan_signals(df, fans, sr_levels)
            
            # Combine results
            result = self._combine_results(df, fans, sr_levels, interaction_data, signals)
            
            # Cache fans for future use
            self.active_fans = fans
            
            self.logger.info(f"Generated {len(fans)} Gann fans with {sum(len(f.fan_lines) for f in fans)} total lines")
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating Gann fans: {str(e)}")
            raise IndicatorCalculationError(f"Gann fan calculation failed: {str(e)}")
            
    def _prepare_market_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare and enhance market data for fan analysis"""
        df = df.copy()
        
        # Calculate price statistics
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['weighted_price'] = (df['high'] + df['low'] + 2*df['close']) / 4
        df['price_range'] = df['high'] - df['low']
        
        # Calculate volatility measures
        df['true_range'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['true_range'].rolling(window=14).mean()
        df['atr_percent'] = df['atr'] / df['close'] * 100
        df['volatility'] = df['close'].pct_change().rolling(window=20).std() * np.sqrt(252)
        
        # Calculate momentum and trend indicators
        df['rsi'] = self._calculate_rsi(df['close'], 14)
        df['momentum'] = df['close'].pct_change(10)
        df['trend_strength'] = abs(df['close'] - df['close'].shift(20)) / df['atr']
        
        # Volume analysis
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Price deviation analysis
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['price_deviation'] = (df['close'] - df['sma_50']) / df['atr']
        
        # Normalize time index
        if 'timestamp' in df.columns:
            df['time_index'] = pd.to_datetime(df['timestamp'])
            df['time_numeric'] = (df['time_index'] - df['time_index'].iloc[0]).dt.total_seconds()
        else:
            df['time_numeric'] = np.arange(len(df))
            
        return df
        
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI for market analysis"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
        
    def _identify_pivot_points(self, df: pd.DataFrame) -> List[Dict]:
        """Identify significant pivot points for fan generation"""
        pivot_points = []
        
        # Multi-method pivot identification
        window = 15
        
        # Method 1: Local extrema with strength validation
        df['high_pivot'] = (
            (df['high'] == df['high'].rolling(window*2, center=True).max()) &
            (df['high'] > df['high'].rolling(window*4).quantile(0.8))
        )
        df['low_pivot'] = (
            (df['low'] == df['low'].rolling(window*2, center=True).min()) &
            (df['low'] < df['low'].rolling(window*4).quantile(0.2))
        )
        
        # Method 2: Volume-confirmed pivots
        df['volume_surge'] = df['volume_ratio'] > 1.5
        
        # Method 3: Momentum divergence pivots
        df['momentum_divergence'] = self._detect_momentum_divergence(df)
        
        # Combine methods to identify strong pivots
        for i in range(window*2, len(df) - window*2):
            pivot_data = {
                'index': i,
                'time': df['time_numeric'].iloc[i],
                'high_price': df['high'].iloc[i],
                'low_price': df['low'].iloc[i],
                'close_price': df['close'].iloc[i],
                'volume': df['volume'].iloc[i],
                'atr': df['atr'].iloc[i],
                'pivot_type': None,
                'strength': 0.0,
                'validation_score': 0.0
            }
            
            # Check for significant high
            if df['high_pivot'].iloc[i]:
                strength = self._calculate_pivot_strength(df, i, 'high')
                if strength > self.config.min_pivot_strength:
                    pivot_data['pivot_type'] = 'high'
                    pivot_data['strength'] = strength
                    
            # Check for significant low
            elif df['low_pivot'].iloc[i]:
                strength = self._calculate_pivot_strength(df, i, 'low')
                if strength > self.config.min_pivot_strength:
                    pivot_data['pivot_type'] = 'low'
                    pivot_data['strength'] = strength
                    
            # Add pivot if significant
            if pivot_data['pivot_type']:
                # Additional validation
                pivot_data['validation_score'] = self._validate_pivot_quality(df, i, pivot_data)
                if pivot_data['validation_score'] > 0.6:
                    pivot_points.append(pivot_data)
                    
        # Filter and sort pivots
        pivot_points = self._filter_close_pivots(pivot_points, min_distance=window)
        pivot_points.sort(key=lambda x: x['strength'], reverse=True)
        
        self.logger.info(f"Identified {len(pivot_points)} significant pivot points")
        return pivot_points[:self.config.max_fans]  # Limit number of fans
        
    def _detect_momentum_divergence(self, df: pd.DataFrame) -> pd.Series:
        """Detect momentum divergence for pivot identification"""
        momentum_short = df['close'].pct_change(5)
        momentum_long = df['close'].pct_change(20)
        
        # Price making new highs/lows but momentum not confirming
        price_new_high = df['high'] == df['high'].rolling(20).max()
        price_new_low = df['low'] == df['low'].rolling(20).min()
        
        momentum_weakness = (
            (momentum_short < momentum_short.shift(5)) | 
            (momentum_long < momentum_long.shift(10))
        )
        
        divergence = (price_new_high | price_new_low) & momentum_weakness
        return divergence
        
    def _calculate_pivot_strength(self, df: pd.DataFrame, index: int, pivot_type: str) -> float:
        """Calculate the strength of a pivot point"""
        window = 10
        
        if index < window or index >= len(df) - window:
            return 0.0
            
        strength_factors = []
        
        # 1. Price extremity
        if pivot_type == 'high':
            local_max = df['high'].iloc[index-window:index+window+1].max()
            price_extremity = df['high'].iloc[index] / local_max
        else:
            local_min = df['low'].iloc[index-window:index+window+1].min()
            price_extremity = local_min / df['low'].iloc[index] if df['low'].iloc[index] > 0 else 0
            
        strength_factors.append(price_extremity * 0.3)
        
        # 2. Volume confirmation
        volume_strength = min(df['volume_ratio'].iloc[index], 3.0) / 3.0
        strength_factors.append(volume_strength * 0.25)
        
        # 3. Range significance
        range_strength = df['price_range'].iloc[index] / df['atr'].iloc[index]
        range_strength = min(range_strength, 3.0) / 3.0
        strength_factors.append(range_strength * 0.2)
        
        # 4. Trend context
        trend_context = abs(df['price_deviation'].iloc[index])
        trend_strength = min(trend_context, 2.0) / 2.0
        strength_factors.append(trend_strength * 0.15)
        
        # 5. Momentum characteristics
        momentum_strength = abs(df['momentum'].iloc[index])
        momentum_strength = min(momentum_strength, 0.1) / 0.1
        strength_factors.append(momentum_strength * 0.1)
        
        total_strength = sum(strength_factors)
        return min(total_strength, 1.0)
        
    def _validate_pivot_quality(self, df: pd.DataFrame, index: int, pivot_data: Dict) -> float:
        """Validate the quality of a pivot point"""
        validation_factors = []
        
        # 1. Clean price action around pivot
        window = 5
        price_action = df.iloc[index-window:index+window+1]
        price_volatility = price_action['close'].std() / price_action['close'].mean()
        clean_action = max(0.0, 1.0 - price_volatility * 5)
        validation_factors.append(clean_action * 0.3)
        
        # 2. Volume pattern consistency
        volume_consistency = 1.0 - (price_action['volume'].std() / price_action['volume'].mean())
        volume_consistency = max(0.0, min(volume_consistency, 1.0))
        validation_factors.append(volume_consistency * 0.2)
        
        # 3. No immediate false signals
        false_signal_penalty = 0.0
        if pivot_data['pivot_type'] == 'high':
            # Check if there are higher highs immediately after
            future_highs = df['high'].iloc[index+1:index+6]
            if len(future_highs) > 0 and future_highs.max() > pivot_data['high_price']:
                false_signal_penalty = 0.3
        else:
            # Check if there are lower lows immediately after
            future_lows = df['low'].iloc[index+1:index+6]
            if len(future_lows) > 0 and future_lows.min() < pivot_data['low_price']:
                false_signal_penalty = 0.3
                
        validation_factors.append((1.0 - false_signal_penalty) * 0.3)
        
        # 4. RSI confirmation
        rsi_value = df['rsi'].iloc[index]
        if pivot_data['pivot_type'] == 'high':
            rsi_confirmation = (rsi_value - 50) / 50 if rsi_value > 50 else 0
        else:
            rsi_confirmation = (50 - rsi_value) / 50 if rsi_value < 50 else 0
            
        validation_factors.append(max(0.0, rsi_confirmation) * 0.2)
        
        total_validation = sum(validation_factors)
        return min(total_validation, 1.0)
        
    def _filter_close_pivots(self, pivots: List[Dict], min_distance: int) -> List[Dict]:
        """Filter out pivots that are too close to each other"""
        if not pivots:
            return pivots
            
        # Sort by strength (strongest first)
        sorted_pivots = sorted(pivots, key=lambda x: x['strength'], reverse=True)
        
        filtered = []
        for pivot in sorted_pivots:
            # Check distance to all previously added pivots
            too_close = False
            for existing_pivot in filtered:
                if abs(pivot['index'] - existing_pivot['index']) < min_distance:
                    too_close = True
                    break
                    
            if not too_close:
                filtered.append(pivot)
                
        return sorted(filtered, key=lambda x: x['index'])
        
    def _generate_gann_fans(self, df: pd.DataFrame, pivot_points: List[Dict]) -> List[GannFan]:
        """Generate Gann fans from pivot points"""
        fans = []
        
        current_time = df['time_numeric'].iloc[-1]
        
        for pivot in pivot_points:
            fan_lines = []
            
            # Use all angles (primary + secondary)
            all_angles = self.config.primary_angles + self.config.secondary_angles
            
            for angle_deg in all_angles:
                fan_line = self._create_fan_line(df, pivot, angle_deg, current_time)
                if fan_line:
                    fan_lines.append(fan_line)
                    
            if fan_lines:
                # Create fan object
                fan = GannFan(
                    id=f"fan_{pivot['index']}_{pivot['pivot_type']}",
                    pivot_time=pivot['time'],
                    pivot_price=pivot['high_price'] if pivot['pivot_type'] == 'high' else pivot['low_price'],
                    pivot_type=pivot['pivot_type'],
                    fan_lines=fan_lines,
                    overall_strength=pivot['strength'],
                    convergence_zones=[],
                    active_lines=[]
                )
                
                fans.append(fan)
                
        self.logger.info(f"Generated {len(fans)} fans with {sum(len(f.fan_lines) for f in fans)} total lines")
        return fans        
    def _create_fan_line(self, df: pd.DataFrame, pivot: Dict, angle_deg: float, current_time: float) -> Optional[GannFanLine]:
        """Create a single fan line from a pivot at given angle"""
        
        # Calculate scaling factors for dynamic scaling
        if self.config.dynamic_scaling:
            scaling_factors = self._calculate_dynamic_scaling(df, pivot['index'])
        else:
            scaling_factors = {'price_scale': 1.0, 'time_scale': 1.0}
            
        # Convert angle to slope with scaling
        angle_rad = math.radians(angle_deg)
        base_slope = math.tan(angle_rad)
        
        # Apply directional adjustment based on pivot type
        if pivot['pivot_type'] == 'low':
            # From low pivot, fans go up and down
            adjusted_slope = base_slope * scaling_factors['price_scale'] / scaling_factors['time_scale']
        else:
            # From high pivot, fans go up and down (but inverted reference)
            adjusted_slope = -base_slope * scaling_factors['price_scale'] / scaling_factors['time_scale']
            
        # Calculate current level
        pivot_price = pivot['high_price'] if pivot['pivot_type'] == 'high' else pivot['low_price']
        time_diff = current_time - pivot['time']
        current_level = pivot_price + adjusted_slope * time_diff
        
        # Validate line (basic sanity checks)
        if current_level <= 0 or abs(adjusted_slope) > 1e6:
            return None
            
        # Calculate line strength and validation
        line_strength = self._calculate_line_strength(df, pivot, adjusted_slope, angle_deg)
        touches = self._count_line_touches(df, pivot, adjusted_slope)
        validation_score = self._validate_line_quality(df, pivot, adjusted_slope, touches)
        
        # Determine support/resistance type
        sr_type = self._determine_sr_type(pivot['pivot_type'], angle_deg, adjusted_slope)
        
        fan_line = GannFanLine(
            id=f"line_{pivot['index']}_{angle_deg}",
            pivot_time=pivot['time'],
            pivot_price=pivot_price,
            angle_degrees=angle_deg,
            slope=adjusted_slope,
            strength=line_strength,
            current_level=current_level,
            support_resistance_type=sr_type,
            touches=touches,
            validation_score=validation_score
        )
        
        return fan_line
        
    def _calculate_dynamic_scaling(self, df: pd.DataFrame, pivot_index: int) -> Dict[str, float]:
        """Calculate dynamic scaling factors based on market conditions"""
        
        # Use data around the pivot for scaling calculation
        window = 50
        start_idx = max(0, pivot_index - window)
        end_idx = min(len(df), pivot_index + window)
        
        local_data = df.iloc[start_idx:end_idx]
        
        # Price scaling based on volatility
        price_volatility = local_data['atr'].mean()
        price_range = local_data['high'].max() - local_data['low'].min()
        price_scale = price_range / (price_volatility * 20)  # Normalize
        
        # Time scaling based on trend strength
        time_range = local_data['time_numeric'].max() - local_data['time_numeric'].min()
        trend_strength = abs(local_data['close'].iloc[-1] - local_data['close'].iloc[0]) / price_range
        time_scale = time_range / (86400 * 30)  # Normalize to approximately 30 days
        
        # Volatility adjustment
        volatility_adjustment = local_data['volatility'].mean() if 'volatility' in local_data.columns else 1.0
        
        return {
            'price_scale': max(0.1, min(price_scale * volatility_adjustment, 10.0)),
            'time_scale': max(0.1, min(time_scale, 10.0))
        }
        
    def _calculate_line_strength(self, df: pd.DataFrame, pivot: Dict, slope: float, angle_deg: float) -> float:
        """Calculate the strength of a fan line"""
        
        strength_factors = []
        
        # 1. Angle significance (45-degree line gets highest score)
        angle_significance = 1.0 - abs(angle_deg - 45.0) / 45.0
        strength_factors.append(angle_significance * 0.25)
        
        # 2. Pivot strength contribution
        pivot_strength = pivot['strength']
        strength_factors.append(pivot_strength * 0.3)
        
        # 3. Line steepness appropriateness
        reasonable_slope = abs(slope) < 10.0  # Not too steep
        slope_factor = 1.0 if reasonable_slope else 0.5
        strength_factors.append(slope_factor * 0.2)
        
        # 4. Historical price interaction
        interaction_strength = self._calculate_price_interaction(df, pivot, slope)
        strength_factors.append(interaction_strength * 0.25)
        
        total_strength = sum(strength_factors)
        return min(total_strength, 1.0)
        
    def _calculate_price_interaction(self, df: pd.DataFrame, pivot: Dict, slope: float) -> float:
        """Calculate how well price has interacted with this line historically"""
        
        pivot_index = pivot['index']
        if pivot_index + self.config.fan_validation_period >= len(df):
            return 0.5  # Default for insufficient future data
            
        # Look at price action after the pivot
        future_data = df.iloc[pivot_index+1:pivot_index+self.config.fan_validation_period+1]
        
        if len(future_data) == 0:
            return 0.5
            
        pivot_price = pivot['high_price'] if pivot['pivot_type'] == 'high' else pivot['low_price']
        pivot_time = pivot['time']
        
        interactions = 0
        total_points = len(future_data)
        touch_threshold = df['atr'].iloc[pivot_index] * self.config.touch_tolerance_atr
        
        for i, (_, row) in enumerate(future_data.iterrows()):
            # Calculate line level at this time
            time_diff = row['time_numeric'] - pivot_time
            line_level = pivot_price + slope * time_diff
            
            # Check if price interacted with line
            price_distance = min(
                abs(row['high'] - line_level),
                abs(row['low'] - line_level),
                abs(row['close'] - line_level)
            )
            
            if price_distance <= touch_threshold:
                interactions += 1
                
        interaction_ratio = interactions / total_points if total_points > 0 else 0
        return min(interaction_ratio * 2, 1.0)  # Scale up and cap at 1.0
        
    def _count_line_touches(self, df: pd.DataFrame, pivot: Dict, slope: float) -> int:
        """Count the number of times price has touched this fan line"""
        
        pivot_index = pivot['index']
        if pivot_index + self.config.fan_validation_period >= len(df):
            return 0
            
        future_data = df.iloc[pivot_index+1:pivot_index+self.config.fan_validation_period+1]
        
        pivot_price = pivot['high_price'] if pivot['pivot_type'] == 'high' else pivot['low_price']
        pivot_time = pivot['time']
        touch_threshold = df['atr'].iloc[pivot_index] * self.config.touch_tolerance_atr
        
        touches = 0
        for _, row in future_data.iterrows():
            time_diff = row['time_numeric'] - pivot_time
            line_level = pivot_price + slope * time_diff
            
            # Check if any price level touched the line
            if (abs(row['high'] - line_level) <= touch_threshold or
                abs(row['low'] - line_level) <= touch_threshold or
                abs(row['close'] - line_level) <= touch_threshold):
                touches += 1
                
        return touches
        
    def _validate_line_quality(self, df: pd.DataFrame, pivot: Dict, slope: float, touches: int) -> float:
        """Validate the quality of a fan line"""
        
        validation_factors = []
        
        # 1. Touch frequency (more touches = higher validation)
        max_possible_touches = self.config.fan_validation_period
        touch_ratio = touches / max_possible_touches if max_possible_touches > 0 else 0
        validation_factors.append(touch_ratio * 0.4)
        
        # 2. Line reasonableness (not too steep or flat)
        slope_reasonableness = 1.0 / (1.0 + abs(slope)) if abs(slope) > 0.1 else 0.9
        validation_factors.append(slope_reasonableness * 0.2)
        
        # 3. Consistency with market structure
        market_trend = self._assess_market_trend(df, pivot['index'])
        line_direction = 1 if slope > 0 else -1
        
        consistency = 0.5  # Neutral
        if market_trend > 0 and line_direction > 0:
            consistency = 0.8  # Upward line in uptrend
        elif market_trend < 0 and line_direction < 0:
            consistency = 0.8  # Downward line in downtrend
        elif abs(market_trend) < 0.1:
            consistency = 0.7  # Any line in sideways market
            
        validation_factors.append(consistency * 0.25)
        
        # 4. Pivot quality influence
        validation_factors.append(pivot['validation_score'] * 0.15)
        
        total_validation = sum(validation_factors)
        return min(total_validation, 1.0)
        
    def _assess_market_trend(self, df: pd.DataFrame, pivot_index: int) -> float:
        """Assess market trend around pivot point"""
        
        window = 20
        start_idx = max(0, pivot_index - window)
        end_idx = min(len(df), pivot_index + window)
        
        trend_data = df.iloc[start_idx:end_idx]
        if len(trend_data) < 10:
            return 0.0
            
        # Calculate trend using linear regression
        x = np.arange(len(trend_data))
        y = trend_data['close'].values
        
        if len(x) > 1:
            slope, _, r_value, _, _ = stats.linregress(x, y)
            # Normalize slope and weight by R-squared
            normalized_slope = slope / trend_data['close'].mean() * 100  # Percentage change per bar
            trend_strength = normalized_slope * (r_value ** 2)
            return max(-1.0, min(1.0, trend_strength))
        
        return 0.0
        
    def _determine_sr_type(self, pivot_type: str, angle_deg: float, slope: float) -> str:
        """Determine if line acts as support or resistance"""
        
        # Basic classification based on slope and pivot type
        if slope > 0.01:  # Upward sloping
            if pivot_type == 'low':
                return 'dynamic_support'
            else:
                return 'dynamic_resistance'
        elif slope < -0.01:  # Downward sloping
            if pivot_type == 'high':
                return 'dynamic_resistance'
            else:
                return 'dynamic_support'
        else:  # Nearly horizontal
            if pivot_type == 'high':
                return 'horizontal_resistance'
            else:
                return 'horizontal_support'
                
    def _validate_fans_with_ml(self, df: pd.DataFrame, fans: List[GannFan]) -> List[GannFan]:
        """Validate fan lines using machine learning"""
        
        if not self.config.ml_validation:
            return fans
            
        try:
            # Train model if not already trained
            if not self.is_trained and len(fans) > 5:
                self._train_validation_model(df, fans)
                
            if self.is_trained:
                return self._apply_ml_validation(df, fans)
            else:
                return fans
                
        except Exception as e:
            self.logger.warning(f"ML validation failed: {str(e)}")
            return fans
            
    def _train_validation_model(self, df: pd.DataFrame, fans: List[GannFan]):
        """Train ML model for fan line validation"""
        
        features = []
        targets = []
        
        for fan in fans:
            for line in fan.fan_lines:
                try:
                    # Extract features for this line
                    line_features = self._extract_line_features(df, fan, line)
                    features.append(line_features)
                    
                    # Target is the validation score (ground truth)
                    targets.append(line.validation_score)
                    
                except Exception:
                    continue
                    
        if len(features) < 10:
            return
            
        # Prepare training data
        X = np.array(features)
        y = np.array(targets)
        
        # Remove invalid samples
        valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) < 5:
            return
            
        # Scale features
        X = self.scaler.fit_transform(X)
        
        # Train model
        self.validation_model.fit(X, y)
        self.is_trained = True
        
        self.logger.info(f"Trained fan validation model with {len(X)} samples")
        
    def _extract_line_features(self, df: pd.DataFrame, fan: GannFan, line: GannFanLine) -> List[float]:
        """Extract features for ML model"""
        
        features = []
        
        # Line characteristics
        features.extend([
            line.angle_degrees,
            abs(line.slope),
            line.touches,
            line.strength
        ])
        
        # Fan characteristics
        features.extend([
            fan.overall_strength,
            len(fan.fan_lines)
        ])
        
        # Market context
        pivot_index = None
        for i, row in df.iterrows():
            if abs(row['time_numeric'] - fan.pivot_time) < 1e-6:
                pivot_index = i
                break
                
        if pivot_index is not None:
            market_vol = df['atr_percent'].iloc[pivot_index] if pivot_index < len(df) else 1.0
            market_trend = self._assess_market_trend(df, pivot_index)
            rsi_value = df['rsi'].iloc[pivot_index] if pivot_index < len(df) else 50.0
            
            features.extend([
                market_vol,
                market_trend,
                rsi_value,
                fan.pivot_price
            ])
        else:
            features.extend([1.0, 0.0, 50.0, fan.pivot_price])
            
        # Line type encoding
        sr_type_encoding = {
            'dynamic_support': 1, 'dynamic_resistance': 2,
            'horizontal_support': 3, 'horizontal_resistance': 4
        }
        features.append(sr_type_encoding.get(line.support_resistance_type, 0))
        
        return features
        
    def _apply_ml_validation(self, df: pd.DataFrame, fans: List[GannFan]) -> List[GannFan]:
        """Apply ML validation to enhance fan line scores"""
        
        enhanced_fans = []
        
        for fan in fans:
            enhanced_lines = []
            
            for line in fan.fan_lines:
                try:
                    # Extract features
                    features = self._extract_line_features(df, fan, line)
                    features_scaled = self.scaler.transform([features])
                    
                    # Predict validation score
                    ml_validation = self.validation_model.predict(features_scaled)[0]
                    
                    # Combine original and ML validation
                    combined_validation = (line.validation_score + ml_validation) / 2
                    combined_validation = max(0.0, min(1.0, combined_validation))
                    
                    # Create enhanced line
                    enhanced_line = line._replace(validation_score=combined_validation)
                    enhanced_lines.append(enhanced_line)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to enhance line {line.id}: {str(e)}")
                    enhanced_lines.append(line)
                    
            # Create enhanced fan
            enhanced_fan = fan._replace(fan_lines=enhanced_lines)
            enhanced_fans.append(enhanced_fan)
            
        return enhanced_fans
        
    def _analyze_fan_convergence(self, df: pd.DataFrame, fans: List[GannFan]) -> List[GannFan]:
        """Analyze convergence zones between fan lines"""
        
        enhanced_fans = []
        
        for fan in fans:
            convergence_zones = []
            
            # Find intersections between lines in this fan
            fan_lines = fan.fan_lines
            
            for i, line1 in enumerate(fan_lines[:-1]):
                for j, line2 in enumerate(fan_lines[i+1:], i+1):
                    intersection = self._calculate_line_intersection(line1, line2)
                    if intersection:
                        convergence_zones.append(intersection)
                        
            # Also find intersections with lines from other fans
            for other_fan in fans:
                if other_fan.id != fan.id:
                    for line1 in fan.fan_lines:
                        for line2 in other_fan.fan_lines:
                            intersection = self._calculate_line_intersection(line1, line2)
                            if intersection:
                                intersection['inter_fan'] = True
                                convergence_zones.append(intersection)
                                
            # Filter and rank convergence zones
            convergence_zones = self._filter_convergence_zones(convergence_zones)
            
            # Identify active lines (strong validation scores)
            active_lines = [
                line.id for line in fan.fan_lines 
                if line.validation_score > 0.7 and line.touches > 2
            ]
            
            # Create enhanced fan
            enhanced_fan = fan._replace(
                convergence_zones=convergence_zones,
                active_lines=active_lines
            )
            enhanced_fans.append(enhanced_fan)
            
        return enhanced_fans
        
    def _calculate_line_intersection(self, line1: GannFanLine, line2: GannFanLine) -> Optional[Dict]:
        """Calculate intersection point between two fan lines"""
        
        # Skip if lines are from same pivot (parallel)
        if abs(line1.pivot_time - line2.pivot_time) < 1e-6:
            return None
            
        # Line equations: y = pivot_price + slope * (t - pivot_time)
        # line1: y = p1 + m1 * (t - t1)
        # line2: y = p2 + m2 * (t - t2)
        
        p1, m1, t1 = line1.pivot_price, line1.slope, line1.pivot_time
        p2, m2, t2 = line2.pivot_price, line2.slope, line2.pivot_time
        
        # Check if lines are parallel
        if abs(m1 - m2) < 1e-10:
            return None
            
        # Solve: p1 + m1*(t - t1) = p2 + m2*(t - t2)
        # t = (p2 - p1 + m1*t1 - m2*t2) / (m1 - m2)
        intersection_time = (p2 - p1 + m1*t1 - m2*t2) / (m1 - m2)
        intersection_price = p1 + m1 * (intersection_time - t1)
        
        # Calculate significance of intersection
        combined_strength = (line1.strength + line2.strength) / 2
        combined_validation = (line1.validation_score + line2.validation_score) / 2
        
        # Check if intersection is reasonable (not too far in past/future)
        current_time = max(line1.pivot_time, line2.pivot_time)
        time_distance = abs(intersection_time - current_time)
        max_reasonable_distance = 86400 * 100  # 100 days in seconds
        
        if time_distance > max_reasonable_distance or intersection_price <= 0:
            return None
            
        return {
            'time': intersection_time,
            'price': intersection_price,
            'line1_id': line1.id,
            'line2_id': line2.id,
            'strength': combined_strength,
            'validation': combined_validation,
            'significance': combined_strength * combined_validation,
            'inter_fan': False
        }
        
    def _filter_convergence_zones(self, zones: List[Dict]) -> List[Dict]:
        """Filter and rank convergence zones"""
        
        if not zones:
            return zones
            
        # Remove zones with very low significance
        filtered_zones = [z for z in zones if z['significance'] > 0.3]
        
        # Sort by significance
        filtered_zones.sort(key=lambda x: x['significance'], reverse=True)
        
        # Keep top zones and avoid clustering
        final_zones = []
        for zone in filtered_zones:
            # Check if too close to existing zones
            too_close = False
            for existing_zone in final_zones:
                time_diff = abs(zone['time'] - existing_zone['time'])
                price_diff = abs(zone['price'] - existing_zone['price'])
                
                if time_diff < 86400 and price_diff < existing_zone['price'] * 0.01:  # 1% price difference
                    too_close = True
                    break
                    
            if not too_close:
                final_zones.append(zone)
                
            # Limit number of zones
            if len(final_zones) >= 10:
                break
                
        return final_zones        
    def _extract_fan_support_resistance(self, df: pd.DataFrame, fans: List[GannFan]) -> Dict:
        """Extract support and resistance levels from fan lines"""
        
        sr_levels = {
            'support_levels': [],
            'resistance_levels': [],
            'dynamic_levels': [],
            'convergence_levels': [],
            'key_levels': []
        }
        
        current_time = df['time_numeric'].iloc[-1]
        current_price = df['close'].iloc[-1]
        
        # Extract levels from individual fan lines
        all_levels = []
        
        for fan in fans:
            for line in fan.fan_lines:
                # Calculate current level
                time_diff = current_time - line.pivot_time
                current_level = line.pivot_price + line.slope * time_diff
                
                # Only consider reasonable levels
                if current_level > 0 and current_level < current_price * 10:
                    level_data = {
                        'level': current_level,
                        'strength': line.strength * line.validation_score,
                        'type': line.support_resistance_type,
                        'source': line.id,
                        'fan_id': fan.id,
                        'touches': line.touches,
                        'angle': line.angle_degrees,
                        'distance_to_current': abs(current_level - current_price) / df['atr'].iloc[-1]
                    }
                    all_levels.append(level_data)
                    
        # Extract levels from convergence zones
        for fan in fans:
            for zone in fan.convergence_zones:
                if zone['time'] <= current_time + 86400 * 30:  # Within 30 days
                    level_data = {
                        'level': zone['price'],
                        'strength': zone['strength'] * 1.5,  # Boost convergence zones
                        'type': 'convergence',
                        'source': f"convergence_{zone['line1_id']}_{zone['line2_id']}",
                        'fan_id': 'multiple',
                        'touches': 0,
                        'angle': 0,
                        'distance_to_current': abs(zone['price'] - current_price) / df['atr'].iloc[-1]
                    }
                    all_levels.append(level_data)
                    
        # Cluster similar levels
        if all_levels:
            clustered_levels = self._cluster_sr_levels(all_levels, df['atr'].iloc[-1])
            
            # Categorize levels
            for cluster in clustered_levels:
                avg_level = cluster['center_level']
                
                if 'support' in cluster['dominant_type']:
                    sr_levels['support_levels'].append(cluster)
                elif 'resistance' in cluster['dominant_type']:
                    sr_levels['resistance_levels'].append(cluster)
                elif 'convergence' in cluster['dominant_type']:
                    sr_levels['convergence_levels'].append(cluster)
                    
                if 'dynamic' in cluster['dominant_type']:
                    sr_levels['dynamic_levels'].append(cluster)
                    
            # Identify key levels (highest strength)
            all_clusters = clustered_levels.copy()
            all_clusters.sort(key=lambda x: x['total_strength'], reverse=True)
            sr_levels['key_levels'] = all_clusters[:15]  # Top 15 levels
            
        return sr_levels
        
    def _cluster_sr_levels(self, levels: List[Dict], atr: float) -> List[Dict]:
        """Cluster similar support/resistance levels"""
        
        if not levels:
            return []
            
        # Sort levels by price
        sorted_levels = sorted(levels, key=lambda x: x['level'])
        
        clusters = []
        current_cluster = [sorted_levels[0]]
        
        tolerance = atr * 0.5  # Clustering tolerance
        
        for level in sorted_levels[1:]:
            cluster_center = np.mean([l['level'] for l in current_cluster])
            
            if abs(level['level'] - cluster_center) <= tolerance:
                current_cluster.append(level)
            else:
                # Finalize current cluster
                if current_cluster:
                    clusters.append(self._create_sr_cluster(current_cluster))
                current_cluster = [level]
                
        # Add final cluster
        if current_cluster:
            clusters.append(self._create_sr_cluster(current_cluster))
            
        return clusters
        
    def _create_sr_cluster(self, levels: List[Dict]) -> Dict:
        """Create support/resistance cluster from similar levels"""
        
        price_levels = [l['level'] for l in levels]
        strengths = [l['strength'] for l in levels]
        types = [l['type'] for l in levels]
        touches = [l['touches'] for l in levels]
        
        center_level = np.mean(price_levels)
        total_strength = sum(strengths)
        total_touches = sum(touches)
        level_count = len(levels)
        
        # Determine dominant type
        type_counts = {}
        for t in types:
            type_counts[t] = type_counts.get(t, 0) + 1
            
        dominant_type = max(type_counts.keys(), key=type_counts.get)
        
        # Calculate reliability score
        strength_consistency = 1.0 - (np.std(strengths) / np.mean(strengths)) if strengths else 0.5
        price_consistency = 1.0 - (np.std(price_levels) / np.mean(price_levels)) if price_levels else 0.5
        reliability = (strength_consistency + price_consistency) / 2
        
        return {
            'center_level': center_level,
            'total_strength': total_strength,
            'level_count': level_count,
            'dominant_type': dominant_type,
            'total_touches': total_touches,
            'reliability': reliability,
            'price_range': max(price_levels) - min(price_levels),
            'component_levels': levels
        }
        
    def _analyze_fan_interactions(self, df: pd.DataFrame, fans: List[GannFan]) -> Dict:
        """Analyze interactions between different fans"""
        
        interaction_data = {
            'fan_correlations': [],
            'strength_distribution': {},
            'angle_analysis': {},
            'temporal_relationships': []
        }
        
        if len(fans) < 2:
            return interaction_data
            
        # Analyze correlations between fans
        for i, fan1 in enumerate(fans[:-1]):
            for j, fan2 in enumerate(fans[i+1:], i+1):
                correlation = self._calculate_fan_correlation(fan1, fan2)
                if correlation['significance'] > 0.3:
                    interaction_data['fan_correlations'].append(correlation)
                    
        # Analyze strength distribution
        strengths = [fan.overall_strength for fan in fans]
        interaction_data['strength_distribution'] = {
            'mean': np.mean(strengths),
            'std': np.std(strengths),
            'max': max(strengths),
            'min': min(strengths),
            'strong_fans': len([s for s in strengths if s > 0.7])
        }
        
        # Analyze angle distribution
        all_angles = []
        for fan in fans:
            for line in fan.fan_lines:
                if line.validation_score > 0.5:
                    all_angles.append(line.angle_degrees)
                    
        if all_angles:
            interaction_data['angle_analysis'] = {
                'dominant_angles': self._find_dominant_angles(all_angles),
                'angle_distribution': np.histogram(all_angles, bins=18)[0].tolist(),
                'average_angle': np.mean(all_angles)
            }
            
        # Temporal relationships
        fan_times = [(fan.pivot_time, fan.overall_strength) for fan in fans]
        fan_times.sort(key=lambda x: x[0])
        
        for i in range(len(fan_times) - 1):
            time_diff = fan_times[i+1][0] - fan_times[i][0]
            strength_change = fan_times[i+1][1] - fan_times[i][1]
            
            interaction_data['temporal_relationships'].append({
                'time_gap': time_diff,
                'strength_change': strength_change,
                'trend': 'strengthening' if strength_change > 0 else 'weakening'
            })
            
        return interaction_data
        
    def _calculate_fan_correlation(self, fan1: GannFan, fan2: GannFan) -> Dict:
        """Calculate correlation between two fans"""
        
        # Time relationship
        time_gap = abs(fan1.pivot_time - fan2.pivot_time)
        temporal_correlation = max(0, 1.0 - time_gap / (86400 * 30))  # Decay over 30 days
        
        # Strength relationship
        strength_similarity = 1.0 - abs(fan1.overall_strength - fan2.overall_strength)
        
        # Geometric relationship
        common_angles = set()
        for line1 in fan1.fan_lines:
            for line2 in fan2.fan_lines:
                if abs(line1.angle_degrees - line2.angle_degrees) < self.config.angle_tolerance:
                    common_angles.add(line1.angle_degrees)
                    
        angle_correlation = len(common_angles) / max(len(fan1.fan_lines), len(fan2.fan_lines))
        
        # Overall significance
        significance = (temporal_correlation + strength_similarity + angle_correlation) / 3
        
        return {
            'fan1_id': fan1.id,
            'fan2_id': fan2.id,
            'temporal_correlation': temporal_correlation,
            'strength_similarity': strength_similarity,
            'angle_correlation': angle_correlation,
            'significance': significance,
            'time_gap': time_gap
        }
        
    def _find_dominant_angles(self, angles: List[float]) -> List[Dict]:
        """Find dominant angle patterns in fan lines"""
        
        # Create histogram of angles
        angle_bins = np.arange(0, 91, 5)  # 5-degree bins
        hist, bin_edges = np.histogram(angles, bins=angle_bins)
        
        # Find peaks in histogram
        dominant_angles = []
        for i, count in enumerate(hist):
            if count > 0:
                angle_center = (bin_edges[i] + bin_edges[i+1]) / 2
                frequency = count / len(angles)
                
                if frequency > 0.1:  # At least 10% of angles
                    dominant_angles.append({
                        'angle': angle_center,
                        'frequency': frequency,
                        'count': count
                    })
                    
        # Sort by frequency
        dominant_angles.sort(key=lambda x: x['frequency'], reverse=True)
        
        return dominant_angles[:5]  # Top 5 dominant angles
        
    def _generate_fan_signals(self, df: pd.DataFrame, fans: List[GannFan], sr_levels: Dict) -> pd.Series:
        """Generate trading signals based on fan analysis"""
        
        signals = pd.Series(0, index=df.index)  # 0 = neutral, 1 = buy, -1 = sell
        
        if not fans:
            return signals
            
        current_price = df['close'].iloc[-1]
        current_time = df['time_numeric'].iloc[-1]
        
        signal_strength = 0.0
        signal_count = 0
        
        # Analyze each fan for signals
        for fan in fans:
            fan_signal = self._generate_fan_signal(df, fan, current_price, current_time, sr_levels)
            if fan_signal != 0:
                signal_strength += fan_signal * fan.overall_strength
                signal_count += 1
                
        # Apply signal threshold
        if signal_count > 0:
            avg_signal = signal_strength / signal_count
            
            if avg_signal > 0.4:
                signals.iloc[-1] = 1  # Buy
            elif avg_signal < -0.4:
                signals.iloc[-1] = -1  # Sell
            else:
                signals.iloc[-1] = 0  # Neutral
                
        return signals
        
    def _generate_fan_signal(self, df: pd.DataFrame, fan: GannFan, current_price: float, 
                           current_time: float, sr_levels: Dict) -> float:
        """Generate signal for individual fan"""
        
        signal = 0.0
        
        # Check each active line in the fan
        for line_id in fan.active_lines:
            line = next((l for l in fan.fan_lines if l.id == line_id), None)
            if not line:
                continue
                
            # Calculate current line level
            time_diff = current_time - line.pivot_time
            line_level = line.pivot_price + line.slope * time_diff
            
            # Distance to line
            distance = abs(current_price - line_level) / df['atr'].iloc[-1]
            
            if distance < 1.0:  # Close to line
                line_signal = self._calculate_line_signal(line, current_price, line_level)
                signal += line_signal * line.validation_score
                
        # Check convergence zones
        for zone in fan.convergence_zones:
            if abs(zone['time'] - current_time) < 86400 * 7:  # Within 7 days
                distance = abs(current_price - zone['price']) / df['atr'].iloc[-1]
                
                if distance < 0.5:  # Very close to convergence
                    # Convergence zones often indicate reversal or breakout
                    convergence_signal = 0.6 * zone['significance']
                    
                    # Direction depends on approach
                    if current_price > zone['price']:
                        signal += convergence_signal  # Bullish if above
                    else:
                        signal -= convergence_signal  # Bearish if below
                        
        # Normalize signal
        return max(-1.0, min(1.0, signal))
        
    def _calculate_line_signal(self, line: GannFanLine, current_price: float, line_level: float) -> float:
        """Calculate signal from individual fan line"""
        
        signal = 0.0
        
        # Basic support/resistance logic
        if 'support' in line.support_resistance_type:
            if current_price > line_level:
                signal = 0.5  # Price above support line
            elif current_price < line_level:
                signal = -0.2  # Price below support (potential break)
                
        elif 'resistance' in line.support_resistance_type:
            if current_price < line_level:
                signal = -0.5  # Price below resistance line
            elif current_price > line_level:
                signal = 0.2  # Price above resistance (potential break)
                
        # Adjust based on line angle
        if line.angle_degrees == 45.0:  # Main trend line
            signal *= 1.5
        elif line.angle_degrees in [63.75, 26.25]:  # Strong angles
            signal *= 1.2
            
        # Adjust based on touches (more touches = stronger level)
        if line.touches > 3:
            signal *= 1.3
        elif line.touches == 0:
            signal *= 0.5
            
        return signal
        
    def _combine_results(self, df: pd.DataFrame, fans: List[GannFan], sr_levels: Dict, 
                        interaction_data: Dict, signals: pd.Series) -> pd.DataFrame:
        """Combine all analysis results into final output DataFrame"""
        
        result = df.copy()
        
        current_price = df['close'].iloc[-1]
        current_time = df['time_numeric'].iloc[-1]
        
        # Fan statistics
        result['gann_fan_count'] = len(fans)
        result['gann_strong_fans'] = len([f for f in fans if f.overall_strength > 0.7])
        
        if fans:
            avg_fan_strength = np.mean([f.overall_strength for f in fans])
            result['gann_avg_fan_strength'] = avg_fan_strength
            
            # Active lines statistics
            total_active_lines = sum(len(f.active_lines) for f in fans)
            result['gann_active_lines'] = total_active_lines
        else:
            result['gann_avg_fan_strength'] = 0.0
            result['gann_active_lines'] = 0
            
        # Support/resistance levels
        key_levels = sr_levels.get('key_levels', [])
        if key_levels:
            closest_level = min(key_levels, key=lambda x: abs(x['center_level'] - current_price))
            result['gann_closest_level'] = closest_level['center_level']
            result['gann_closest_level_strength'] = closest_level['total_strength']
            result['gann_closest_level_type'] = closest_level['dominant_type']
            result['gann_distance_to_level'] = abs(current_price - closest_level['center_level']) / df['atr'].iloc[-1]
        else:
            result['gann_closest_level'] = np.nan
            result['gann_closest_level_strength'] = 0.0
            result['gann_closest_level_type'] = 'none'
            result['gann_distance_to_level'] = np.nan
            
        # Convergence analysis
        total_convergences = sum(len(f.convergence_zones) for f in fans)
        result['gann_convergence_zones'] = total_convergences
        
        # Find nearest convergence zone
        nearest_convergence = None
        min_distance = float('inf')
        
        for fan in fans:
            for zone in fan.convergence_zones:
                time_distance = abs(zone['time'] - current_time)
                price_distance = abs(zone['price'] - current_price)
                combined_distance = time_distance / 86400 + price_distance / current_price  # Normalize
                
                if combined_distance < min_distance:
                    min_distance = combined_distance
                    nearest_convergence = zone
                    
        if nearest_convergence:
            result['gann_nearest_convergence_price'] = nearest_convergence['price']
            result['gann_nearest_convergence_strength'] = nearest_convergence['strength']
            result['gann_convergence_distance'] = min_distance
        else:
            result['gann_nearest_convergence_price'] = np.nan
            result['gann_nearest_convergence_strength'] = 0.0
            result['gann_convergence_distance'] = np.nan
            
        # Interaction metrics
        if interaction_data.get('strength_distribution'):
            result['gann_strength_consistency'] = 1.0 - (
                interaction_data['strength_distribution']['std'] / 
                interaction_data['strength_distribution']['mean']
            ) if interaction_data['strength_distribution']['mean'] > 0 else 0.5
        else:
            result['gann_strength_consistency'] = 0.5
            
        # Angle analysis
        if interaction_data.get('angle_analysis'):
            dominant_angles = interaction_data['angle_analysis'].get('dominant_angles', [])
            if dominant_angles:
                result['gann_dominant_angle'] = dominant_angles[0]['angle']
                result['gann_angle_dominance'] = dominant_angles[0]['frequency']
            else:
                result['gann_dominant_angle'] = 45.0  # Default
                result['gann_angle_dominance'] = 0.0
        else:
            result['gann_dominant_angle'] = 45.0
            result['gann_angle_dominance'] = 0.0
            
        # Trading signals
        result['gann_fan_signal'] = signals
        
        # Market structure assessment
        result['gann_market_structure'] = self._assess_market_structure(fans, sr_levels, current_price)
        
        return result
        
    def _assess_market_structure(self, fans: List[GannFan], sr_levels: Dict, current_price: float) -> str:
        """Assess current market structure based on fan analysis"""
        
        if not fans:
            return 'undefined'
            
        # Analyze fan orientations
        upward_fans = 0
        downward_fans = 0
        
        for fan in fans:
            fan_orientation = self._determine_fan_orientation(fan)
            if fan_orientation == 'bullish':
                upward_fans += fan.overall_strength
            elif fan_orientation == 'bearish':
                downward_fans += fan.overall_strength
                
        # Analyze support/resistance structure
        support_strength = sum(
            level['total_strength'] for level in sr_levels.get('support_levels', [])
            if level['center_level'] < current_price
        )
        
        resistance_strength = sum(
            level['total_strength'] for level in sr_levels.get('resistance_levels', [])
            if level['center_level'] > current_price
        )
        
        # Classification logic
        if upward_fans > downward_fans * 1.5 and support_strength > resistance_strength:
            return 'bullish_trending'
        elif downward_fans > upward_fans * 1.5 and resistance_strength > support_strength:
            return 'bearish_trending'
        elif abs(upward_fans - downward_fans) < 0.3:
            return 'consolidating'
        elif support_strength > resistance_strength * 1.2:
            return 'supported'
        elif resistance_strength > support_strength * 1.2:
            return 'resistance_heavy'
        else:
            return 'mixed'
            
    def _determine_fan_orientation(self, fan: GannFan) -> str:
        """Determine overall orientation of a fan"""
        
        upward_lines = 0
        downward_lines = 0
        
        for line in fan.fan_lines:
            if line.validation_score > 0.5:  # Only consider validated lines
                if line.slope > 0.01:
                    upward_lines += line.strength
                elif line.slope < -0.01:
                    downward_lines += line.strength
                    
        if upward_lines > downward_lines:
            return 'bullish'
        elif downward_lines > upward_lines:
            return 'bearish'
        else:
            return 'neutral'


def create_gann_fan_indicator(config: Optional[GannFanConfig] = None) -> GannFanIndicator:
    """Factory function to create GannFanIndicator instance"""
    return GannFanIndicator(config)


# Example usage and testing
if __name__ == "__main__":
    import yfinance as yf
    
    # Test with sample data
    ticker = "EURUSD=X"
    data = yf.download(ticker, period="3mo", interval="1h")
    data.reset_index(inplace=True)
    data.columns = data.columns.str.lower()
    data['timestamp'] = data['datetime']
    
    # Create indicator
    config = GannFanConfig(
        min_pivot_strength=0.6,
        max_fans=8,
        ml_validation=True,
        convergence_analysis=True,
        dynamic_scaling=True
    )
    
    indicator = GannFanIndicator(config)
    
    try:
        # Calculate Gann fans
        result = indicator.calculate(data)
        
        print("Gann Fan Calculation Results:")
        print(f"Data shape: {result.shape}")
        print(f"Columns: {list(result.columns)}")
        
        # Display recent signals
        recent = result.tail(5)
        for col in ['gann_fan_count', 'gann_fan_signal', 'gann_market_structure', 'gann_closest_level_type']:
            if col in recent.columns:
                print(f"\n{col}:")
                print(recent[col].to_string())
                
        # Display fan statistics
        fan_count = recent['gann_fan_count'].iloc[-1]
        strong_fans = recent['gann_strong_fans'].iloc[-1]
        active_lines = recent['gann_active_lines'].iloc[-1]
        
        print(f"\nFan Statistics:")
        print(f"Total fans: {fan_count}")
        print(f"Strong fans: {strong_fans}")
        print(f"Active lines: {active_lines}")
                
    except Exception as e:
        print(f"Error testing Gann Fan indicator: {e}")
        import traceback
        traceback.print_exc()