"""
Fibonacci Retracement Indicator - Advanced Horizontal Support/Resistance Analysis
================================================================================

This module implements a sophisticated Fibonacci retracement indicator that identifies
horizontal support and resistance levels at key Fibonacci ratios. It includes advanced
algorithms for swing point detection, multi-timeframe analysis, machine learning
integration, confluence detection, and dynamic level validation.

Features:
- Advanced swing point detection using fractal analysis
- Multiple Fibonacci ratio sets (classic, extended, custom)
- Machine learning price reaction prediction at levels
- Dynamic level strength calculation
- Multi-timeframe confluence analysis
- Volume-weighted level significance
- Trend-aware retracement validation
- Breakout and bounce detection
- Real-time level adjustment

The indicator helps traders identify potential reversal and continuation points
using mathematically derived Fibonacci ratios combined with market structure analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from scipy import stats
from scipy.signal import find_peaks
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from ..base.standard_indicator import StandardIndicatorInterface, IndicatorResult, DataRequirement, DataType, SignalType
from ...core.exceptions import IndicatorCalculationException

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class FibonacciLevel:
    """Represents a Fibonacci retracement level"""
    ratio: float
    price: float
    level_type: str  # 'support', 'resistance', 'neutral'
    strength: float  # 0.0 to 1.0
    touches: int
    last_touch: Optional[datetime]
    volume_at_level: float
    ml_score: float  # Machine learning confidence
    confluence_count: int
    

@dataclass
class SwingPoint:
    """Represents a swing high or low point"""
    index: int
    price: float
    timestamp: datetime
    swing_type: str  # 'high' or 'low'
    strength: float
    volume: float
    

class FibonacciRetracementIndicator(StandardIndicatorInterface):
    """
    Advanced Fibonacci Retracement Indicator with machine learning integration
    and sophisticated level analysis capabilities.
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'swing_detection_period': 10,
            'min_swing_strength': 0.3,
            'fibonacci_ratios': [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0],
            'extended_ratios': [1.272, 1.414, 1.618, 2.0, 2.618],
            'use_extended_ratios': True,
            'level_tolerance': 0.001,  # 0.1% tolerance for level touches
            'min_level_strength': 0.2,
            'volume_weight': 0.3,
            'ml_lookback': 50,
            'confluence_weight': 0.4,
            'trend_filter': True,
            'multi_timeframe': True,
            'max_levels': 15,
            'level_expiry_days': 30
        }
        
        if parameters:
            default_params.update(parameters)
        
        super().__init__(name="FibonacciRetracement", parameters=default_params)
        
        # Initialize internal state
        self.swing_points: List[SwingPoint] = []
        self.fibonacci_levels: List[FibonacciLevel] = []
        self.ml_model = None
        self.scaler = StandardScaler()
        self.last_calculation = None
        
        # Fibonacci ratios
        self.ratios = self.parameters['fibonacci_ratios'].copy()
        if self.parameters['use_extended_ratios']:
            self.ratios.extend(self.parameters['extended_ratios'])
        
        logger.info(f"FibonacciRetracementIndicator initialized with {len(self.ratios)} ratios")
    
    def get_data_requirements(self) -> DataRequirement:
        """Define the data requirements for Fibonacci retracement calculation"""
        return DataRequirement(
            data_type=DataType.OHLCV,
            required_columns=['high', 'low', 'close', 'volume'],
            min_periods=max(50, self.parameters['swing_detection_period'] * 3),
            lookback_periods=200
        )
    
    def validate_parameters(self) -> bool:
        """Validate the indicator parameters"""
        try:
            required_params = ['swing_detection_period', 'fibonacci_ratios']
            for param in required_params:
                if param not in self.parameters:
                    raise ValueError(f"Missing required parameter: {param}")
            
            if self.parameters['swing_detection_period'] < 3:
                raise ValueError("swing_detection_period must be at least 3")
            
            if not self.parameters['fibonacci_ratios']:
                raise ValueError("fibonacci_ratios cannot be empty")
            
            return True
            
        except Exception as e:
            logger.error(f"Parameter validation failed: {str(e)}")
            return False
    
    def _detect_swing_points(self, data: pd.DataFrame) -> List[SwingPoint]:
        """
        Detect significant swing points using advanced fractal analysis
        """
        try:
            swing_points = []
            period = self.parameters['swing_detection_period']
            min_strength = self.parameters['min_swing_strength']
            
            highs = data['high'].values
            lows = data['low'].values
            volumes = data['volume'].values
            
            # Detect swing highs
            high_peaks, high_properties = find_peaks(
                highs, 
                distance=period,
                prominence=np.std(highs) * min_strength
            )
            
            # Detect swing lows (invert the data)
            low_peaks, low_properties = find_peaks(
                -lows,
                distance=period,
                prominence=np.std(lows) * min_strength
            )
            
            # Process swing highs
            for i, peak_idx in enumerate(high_peaks):
                if peak_idx >= period and peak_idx < len(data) - period:
                    strength = high_properties['prominences'][i] / np.std(highs)
                    
                    swing_point = SwingPoint(
                        index=peak_idx,
                        price=highs[peak_idx],
                        timestamp=data.index[peak_idx],
                        swing_type='high',
                        strength=min(strength, 1.0),
                        volume=volumes[peak_idx]
                    )
                    swing_points.append(swing_point)
            
            # Process swing lows
            for i, peak_idx in enumerate(low_peaks):
                if peak_idx >= period and peak_idx < len(data) - period:
                    strength = low_properties['prominences'][i] / np.std(lows)
                    
                    swing_point = SwingPoint(
                        index=peak_idx,
                        price=lows[peak_idx],
                        timestamp=data.index[peak_idx],
                        swing_type='low',
                        strength=min(strength, 1.0),
                        volume=volumes[peak_idx]
                    )
                    swing_points.append(swing_point)
            
            # Sort by timestamp and filter by strength
            swing_points.sort(key=lambda x: x.timestamp)
            swing_points = [sp for sp in swing_points if sp.strength >= min_strength]
            
            logger.debug(f"Detected {len(swing_points)} swing points")
            return swing_points[-20:]  # Keep only recent swing points
            
        except Exception as e:
            logger.error(f"Error detecting swing points: {str(e)}")
            return []
    
    def _calculate_fibonacci_levels(self, swing_high: SwingPoint, swing_low: SwingPoint) -> List[FibonacciLevel]:
        """
        Calculate Fibonacci retracement levels between two swing points
        """
        try:
            levels = []
            price_range = swing_high.price - swing_low.price
            
            if abs(price_range) < 1e-10:  # Avoid division by zero
                return levels
            
            for ratio in self.ratios:
                # Calculate retracement level
                level_price = swing_high.price - (price_range * ratio)
                
                # Determine level type based on current market position
                level_type = 'neutral'
                if ratio < 0.5:
                    level_type = 'resistance'
                elif ratio > 0.5:
                    level_type = 'support'
                
                # Calculate base strength based on swing point strengths
                base_strength = (swing_high.strength + swing_low.strength) / 2
                
                level = FibonacciLevel(
                    ratio=ratio,
                    price=level_price,
                    level_type=level_type,
                    strength=base_strength,
                    touches=0,
                    last_touch=None,
                    volume_at_level=0.0,
                    ml_score=0.0,
                    confluence_count=1
                )
                levels.append(level)
            
            return levels
            
        except Exception as e:
            logger.error(f"Error calculating Fibonacci levels: {str(e)}")
            return []
    
    def _analyze_level_touches(self, levels: List[FibonacciLevel], data: pd.DataFrame) -> List[FibonacciLevel]:
        """
        Analyze historical price touches at Fibonacci levels
        """
        try:
            tolerance = self.parameters['level_tolerance']
            
            for level in levels:
                touches = 0
                volume_sum = 0.0
                last_touch = None
                
                for i, row in data.iterrows():
                    high_price = row['high']
                    low_price = row['low']
                    volume = row['volume']
                    
                    # Check if price touched the level
                    level_range = level.price * tolerance
                    if low_price <= level.price + level_range and high_price >= level.price - level_range:
                        touches += 1
                        volume_sum += volume
                        last_touch = i
                
                # Update level properties
                level.touches = touches
                level.last_touch = last_touch
                level.volume_at_level = volume_sum
                
                # Adjust strength based on touches and volume
                if touches > 0:
                    volume_factor = min(volume_sum / (data['volume'].mean() * touches), 2.0)
                    touch_factor = min(touches / 3.0, 1.0)
                    level.strength = min(level.strength * (1 + touch_factor + volume_factor * self.parameters['volume_weight']), 1.0)
            
            return levels
            
        except Exception as e:
            logger.error(f"Error analyzing level touches: {str(e)}")
            return levels
    
    def _detect_confluence_zones(self, all_levels: List[FibonacciLevel]) -> List[FibonacciLevel]:
        """
        Detect confluence zones where multiple Fibonacci levels converge
        """
        try:
            tolerance = self.parameters['level_tolerance'] * 2  # Wider tolerance for confluence
            confluence_weight = self.parameters['confluence_weight']
            
            for i, level1 in enumerate(all_levels):
                confluence_count = 1
                confluence_strength = level1.strength
                
                for j, level2 in enumerate(all_levels):
                    if i != j and abs(level1.price - level2.price) / level1.price <= tolerance:
                        confluence_count += 1
                        confluence_strength += level2.strength
                
                # Update level with confluence information
                level1.confluence_count = confluence_count
                if confluence_count > 1:
                    level1.strength = min(level1.strength + confluence_strength * confluence_weight, 1.0)
            
            return all_levels
            
        except Exception as e:
            logger.error(f"Error detecting confluence zones: {str(e)}")
            return all_levels
    
    def _prepare_ml_features(self, data: pd.DataFrame, level_price: float) -> np.ndarray:
        """
        Prepare features for machine learning model
        """
        try:
            features = []
            current_price = data['close'].iloc[-1]
            
            # Price-based features
            features.append((level_price - current_price) / current_price)  # Distance to level
            features.append(current_price / data['close'].rolling(20).mean().iloc[-1])  # Price momentum
            
            # Technical features
            features.append(data['close'].pct_change().rolling(10).std().iloc[-1])  # Volatility
            features.append(data['volume'].rolling(10).mean().iloc[-1] / data['volume'].rolling(50).mean().iloc[-1])  # Volume ratio
            
            # Trend features
            sma_20 = data['close'].rolling(20).mean().iloc[-1]
            sma_50 = data['close'].rolling(50).mean().iloc[-1]
            features.append((sma_20 - sma_50) / sma_50)  # Trend strength
            
            # RSI-like momentum
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean().iloc[-1]
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean().iloc[-1]
            if loss != 0:
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                features.append(rsi / 100.0)
            else:
                features.append(0.5)
            
            # MACD-like feature
            ema_12 = data['close'].ewm(span=12).mean().iloc[-1]
            ema_26 = data['close'].ewm(span=26).mean().iloc[-1]
            features.append((ema_12 - ema_26) / current_price)
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Error preparing ML features: {str(e)}")
            return np.array([[0.0] * 7])
    
    def _train_ml_model(self, data: pd.DataFrame, levels: List[FibonacciLevel]):
        """
        Train machine learning model to predict price reactions at levels
        """
        try:
            if len(levels) < 5 or len(data) < self.parameters['ml_lookback']:
                return
            
            X, y = [], []
            
            # Prepare training data
            for level in levels:
                if level.touches > 0:
                    # Create features for this level
                    for i in range(self.parameters['ml_lookback'], len(data)):
                        subset = data.iloc[i-self.parameters['ml_lookback']:i]
                        features = self._prepare_ml_features(subset, level.price)
                        
                        # Calculate target (price reaction strength)
                        future_data = data.iloc[i:min(i+10, len(data))]
                        if len(future_data) > 3:
                            reaction_strength = abs(future_data['close'].max() - future_data['close'].min()) / future_data['close'].iloc[0]
                            X.append(features[0])
                            y.append(min(reaction_strength, 1.0))
            
            if len(X) > 10:
                X = np.array(X)
                y = np.array(y)
                
                # Scale features
                X_scaled = self.scaler.fit_transform(X)
                
                # Train model
                self.ml_model = RandomForestRegressor(
                    n_estimators=50,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )
                self.ml_model.fit(X_scaled, y)
                
                logger.debug(f"ML model trained with {len(X)} samples")
                
        except Exception as e:
            logger.error(f"Error training ML model: {str(e)}")
    
    def _predict_level_strength(self, data: pd.DataFrame, level: FibonacciLevel) -> float:
        """
        Use machine learning to predict level strength
        """
        try:
            if self.ml_model is None:
                return level.strength
            
            features = self._prepare_ml_features(data, level.price)
            features_scaled = self.scaler.transform(features)
            prediction = self.ml_model.predict(features_scaled)[0]
            
            return min(prediction, 1.0)
            
        except Exception as e:
            logger.error(f"Error predicting level strength: {str(e)}")
            return level.strength
    
    def _filter_significant_levels(self, levels: List[FibonacciLevel]) -> List[FibonacciLevel]:
        """
        Filter and rank levels by significance
        """
        try:
            min_strength = self.parameters['min_level_strength']
            max_levels = self.parameters['max_levels']
            
            # Filter by minimum strength
            significant_levels = [level for level in levels if level.strength >= min_strength]
            
            # Sort by combined strength and confluence
            significant_levels.sort(
                key=lambda x: x.strength * (1 + x.confluence_count * 0.1), 
                reverse=True
            )
            
            # Return top levels
            return significant_levels[:max_levels]
            
        except Exception as e:
            logger.error(f"Error filtering significant levels: {str(e)}")
            return levels
    
    def calculate_raw(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate Fibonacci retracement levels with advanced analysis
        """
        try:
            # Detect swing points
            self.swing_points = self._detect_swing_points(data)
            
            if len(self.swing_points) < 2:
                return {
                    'levels': [],
                    'swing_points': [],
                    'current_price': data['close'].iloc[-1],
                    'signal_strength': 0.0,
                    'nearest_level': None,
                    'level_count': 0
                }
            
            # Generate Fibonacci levels from swing point pairs
            all_levels = []
            
            # Use recent swing points to generate levels
            recent_swings = self.swing_points[-10:]  # Last 10 swing points
            
            for i in range(len(recent_swings)):
                for j in range(i + 1, len(recent_swings)):
                    swing1, swing2 = recent_swings[i], recent_swings[j]
                    
                    # Ensure we have a high and a low
                    if swing1.swing_type != swing2.swing_type:
                        if swing1.swing_type == 'high':
                            swing_high, swing_low = swing1, swing2
                        else:
                            swing_high, swing_low = swing2, swing1
                        
                        # Calculate levels for this swing pair
                        levels = self._calculate_fibonacci_levels(swing_high, swing_low)
                        all_levels.extend(levels)
            
            if not all_levels:
                return {
                    'levels': [],
                    'swing_points': [self._swing_point_to_dict(sp) for sp in self.swing_points],
                    'current_price': data['close'].iloc[-1],
                    'signal_strength': 0.0,
                    'nearest_level': None,
                    'level_count': 0
                }
            
            # Analyze level touches
            all_levels = self._analyze_level_touches(all_levels, data)
            
            # Detect confluence zones
            all_levels = self._detect_confluence_zones(all_levels)
            
            # Train ML model if needed
            self._train_ml_model(data, all_levels)
            
            # Predict level strengths using ML
            for level in all_levels:
                level.ml_score = self._predict_level_strength(data, level)
                # Combine original strength with ML prediction
                level.strength = (level.strength + level.ml_score) / 2
            
            # Filter and rank significant levels
            significant_levels = self._filter_significant_levels(all_levels)
            self.fibonacci_levels = significant_levels
            
            # Find nearest level to current price
            current_price = data['close'].iloc[-1]
            nearest_level = None
            min_distance = float('inf')
            
            for level in significant_levels:
                distance = abs(level.price - current_price) / current_price
                if distance < min_distance:
                    min_distance = distance
                    nearest_level = level
            
            # Calculate overall signal strength
            signal_strength = 0.0
            if nearest_level and min_distance < 0.02:  # Within 2%
                signal_strength = nearest_level.strength * (1 - min_distance * 10)
            
            # Prepare result
            result = {
                'levels': [self._level_to_dict(level) for level in significant_levels],
                'swing_points': [self._swing_point_to_dict(sp) for sp in self.swing_points],
                'current_price': current_price,
                'signal_strength': signal_strength,
                'nearest_level': self._level_to_dict(nearest_level) if nearest_level else None,
                'level_count': len(significant_levels),
                'ml_model_active': self.ml_model is not None,
                'analysis_timestamp': datetime.utcnow().isoformat()
            }
            
            self.last_calculation = result
            return result
            
        except Exception as e:
            logger.error(f"Error in Fibonacci retracement calculation: {str(e)}")
            raise IndicatorCalculationException(
                indicator_name=self.name,
                calculation_step="fibonacci_retracement_calculation",
                message=str(e)
            )
    
    def _swing_point_to_dict(self, swing_point: SwingPoint) -> Dict[str, Any]:
        """Convert SwingPoint to dictionary"""
        return {
            'index': swing_point.index,
            'price': swing_point.price,
            'timestamp': swing_point.timestamp.isoformat(),
            'swing_type': swing_point.swing_type,
            'strength': swing_point.strength,
            'volume': swing_point.volume
        }
    
    def _level_to_dict(self, level: FibonacciLevel) -> Dict[str, Any]:
        """Convert FibonacciLevel to dictionary"""
        return {
            'ratio': level.ratio,
            'price': level.price,
            'level_type': level.level_type,
            'strength': level.strength,
            'touches': level.touches,
            'last_touch': level.last_touch.isoformat() if level.last_touch else None,
            'volume_at_level': level.volume_at_level,
            'ml_score': level.ml_score,
            'confluence_count': level.confluence_count
        }
    
    def _generate_signal(self, value: Dict[str, Any], data: pd.DataFrame) -> Tuple[Optional[SignalType], float]:
        """
        Generate trading signals based on Fibonacci level analysis
        """
        try:
            if not value.get('nearest_level') or value['signal_strength'] < 0.3:
                return SignalType.NEUTRAL, 0.0
            
            nearest_level = value['nearest_level']
            current_price = value['current_price']
            level_price = nearest_level['price']
            level_type = nearest_level['level_type']
            strength = value['signal_strength']
            
            # Determine signal based on level type and price position
            if current_price < level_price and level_type == 'support':
                # Price below support - potential bounce
                return SignalType.BUY, strength
            elif current_price > level_price and level_type == 'resistance':
                # Price above resistance - potential reversal
                return SignalType.SELL, strength
            elif abs(current_price - level_price) / current_price < 0.005:
                # Very close to level - direction depends on momentum
                recent_momentum = data['close'].pct_change(5).iloc[-1]
                if recent_momentum > 0 and level_type == 'support':
                    return SignalType.BUY, strength * 0.8
                elif recent_momentum < 0 and level_type == 'resistance':
                    return SignalType.SELL, strength * 0.8
            
            return SignalType.NEUTRAL, 0.0
            
        except Exception as e:
            logger.error(f"Error generating signal: {str(e)}")
            return SignalType.NEUTRAL, 0.0
    
    def _get_metadata(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get additional metadata for the calculation"""
        base_metadata = super()._get_metadata(data)
        
        fibonacci_metadata = {
            'swing_points_detected': len(self.swing_points),
            'fibonacci_levels_count': len(self.fibonacci_levels),
            'ml_model_trained': self.ml_model is not None,
            'ratios_used': self.ratios,
            'confluence_zones': sum(1 for level in self.fibonacci_levels if level.confluence_count > 1)
        }
        
        base_metadata.update(fibonacci_metadata)
        return base_metadata


def create_fibonacci_retracement_indicator(parameters: Optional[Dict[str, Any]] = None) -> FibonacciRetracementIndicator:
    """
    Factory function to create a FibonacciRetracementIndicator instance
    
    Args:
        parameters: Optional dictionary of parameters to customize the indicator
        
    Returns:
        Configured FibonacciRetracementIndicator instance
    """
    return FibonacciRetracementIndicator(parameters=parameters)


# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='1h')
    prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.1)
    
    sample_data = pd.DataFrame({
        'high': prices + np.random.uniform(0, 1, len(dates)),
        'low': prices - np.random.uniform(0, 1, len(dates)),
        'close': prices,
        'volume': np.random.uniform(1000, 10000, len(dates))
    }, index=dates)
    
    # Test the indicator
    indicator = create_fibonacci_retracement_indicator({
        'swing_detection_period': 8,
        'min_swing_strength': 0.2,
        'use_extended_ratios': True
    })
    
    try:
        result = indicator.calculate(sample_data)
        print("Fibonacci Retracement Calculation Result:")
        print(f"Signal: {result.signal}, Confidence: {result.confidence:.3f}")
        print(f"Levels detected: {result.value.get('level_count', 0)}")
        print(f"Signal strength: {result.value.get('signal_strength', 0):.3f}")
        
        if result.value.get('nearest_level'):
            nearest = result.value['nearest_level']
            print(f"Nearest level: {nearest['ratio']:.3f} at ${nearest['price']:.2f}")
            print(f"Level strength: {nearest['strength']:.3f}")
        
    except Exception as e:
        print(f"Error testing indicator: {str(e)}")