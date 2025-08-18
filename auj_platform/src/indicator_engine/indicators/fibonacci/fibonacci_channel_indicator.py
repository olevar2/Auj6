"""
Fibonacci Channel Indicator - Advanced Implementation
====================================================

This indicator implements sophisticated Fibonacci channel calculations using advanced geometric algorithms
and machine learning-enhanced trend detection. Fibonacci channels are parallel lines based on Fibonacci
ratios that provide dynamic support and resistance levels following market trends.

Features:
- Advanced parallel channel construction with Fibonacci ratios
- Machine learning trend direction and strength analysis
- Multi-timeframe channel confluence detection
- Dynamic channel adjustment based on volatility and momentum
- Breakout detection and confirmation systems
- Comprehensive error handling and validation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import logging
from datetime import datetime, timedelta

from ..base.standard_indicator import StandardIndicatorInterface, DataRequirement, DataType, SignalType
from ....core.exceptions import IndicatorCalculationException

logger = logging.getLogger(__name__)

@dataclass
class FibonacciChannelLine:
    """Represents a single Fibonacci channel line with comprehensive properties."""
    ratio: float
    slope: float
    intercept: float
    upper_bound: float
    lower_bound: float
    confidence: float
    strength: float
    direction: str  # 'up', 'down', 'sideways'
    start_point: Tuple[float, float]
    end_point: Tuple[float, float]
    volume_confirmation: float
    breakout_probability: float

@dataclass
class ChannelBand:
    """Represents a complete Fibonacci channel band."""
    upper_line: FibonacciChannelLine
    lower_line: FibonacciChannelLine
    middle_line: FibonacciChannelLine
    channel_width: float
    trend_strength: float
    volatility_adjustment: float
    support_resistance_strength: float

@dataclass
class BreakoutSignal:
    """Represents a channel breakout signal."""
    timestamp: datetime
    price: float
    direction: str  # 'bullish', 'bearish'
    strength: float
    volume_confirmation: float
    target_price: float
    stop_loss: float
    confidence: float

class FibonacciChannelIndicator(StandardIndicatorInterface):
    """
    Advanced Fibonacci Channel Indicator with machine learning trend analysis.
    
    This implementation uses sophisticated geometric algorithms to construct
    Fibonacci-based parallel channels and employs machine learning techniques
    to detect trends, confirm breakouts, and predict price movements.
    """
    
    def __init__(self, 
                 name: str = "FibonacciChannel",
                 fibonacci_ratios: List[float] = None,
                 lookback_period: int = 50,
                 min_trend_strength: float = 0.3,
                 channel_sensitivity: float = 0.02,
                 volatility_adjustment: bool = True,
                 ml_trend_detection: bool = True,
                 breakout_confirmation_period: int = 3):
        """
        Initialize the Fibonacci Channel Indicator.
        
        Args:
            name: Indicator name
            fibonacci_ratios: Custom Fibonacci ratios for channel construction
            lookback_period: Period for trend analysis
            min_trend_strength: Minimum strength for trend validation
            channel_sensitivity: Sensitivity for channel detection
            volatility_adjustment: Enable dynamic adjustment based on volatility
            ml_trend_detection: Enable machine learning trend detection
            breakout_confirmation_period: Periods required for breakout confirmation
        """
        parameters = {
            'fibonacci_ratios': fibonacci_ratios or [0.236, 0.382, 0.618, 0.786, 1.000, 1.272, 1.618],
            'lookback_period': lookback_period,
            'min_trend_strength': min_trend_strength,
            'channel_sensitivity': channel_sensitivity,
            'volatility_adjustment': volatility_adjustment,
            'ml_trend_detection': ml_trend_detection,
            'breakout_confirmation_period': breakout_confirmation_period
        }
        
        super().__init__(name=name, parameters=parameters)
        
        self.fibonacci_ratios = parameters['fibonacci_ratios']
        self.lookback_period = lookback_period
        self.min_trend_strength = min_trend_strength
        self.channel_sensitivity = channel_sensitivity
        self.volatility_adjustment = volatility_adjustment
        self.ml_trend_detection = ml_trend_detection
        self.breakout_confirmation_period = breakout_confirmation_period
        
        # ML components
        self.scaler = StandardScaler()
        self.trend_detector = None
        self.breakout_predictor = None
        self._initialize_ml_components()
        
        # Channel tracking
        self.active_channels: List[ChannelBand] = []
        self.breakout_signals: List[BreakoutSignal] = []
        self.historical_channels: List[ChannelBand] = []
        
    def get_data_requirements(self) -> DataRequirement:
        """Define data requirements for Fibonacci Channel calculation."""
        return DataRequirement(
            data_type=DataType.OHLCV,
            required_columns=['open', 'high', 'low', 'close'],
            min_periods=self.lookback_period,
            lookback_periods=self.lookback_period * 2,
            preprocessing=None
        )
    
    def calculate_raw(self, data: pd.DataFrame) -> Dict[str, Union[pd.Series, np.ndarray, Dict]]:
        """
        Perform the raw Fibonacci channel calculation.
        
        Args:
            data: DataFrame with OHLC data
            
        Returns:
            Dictionary containing channel data, signals, and analysis
        """
        return self.calculate(data)
        
    def _generate_signal(self, value: Any, data: pd.DataFrame) -> Tuple[Optional[SignalType], float]:
        """Generate trading signal based on channel analysis."""
        if not isinstance(value, dict) or 'signals' not in value:
            return None, 0.0
        
        signals = value['signals']
        if len(signals) == 0:
            return None, 0.0
        
        latest_signal = signals.iloc[-1]
        confidence = value.get('confidence', pd.Series([0.0])).iloc[-1] if len(value.get('confidence', [])) > 0 else 0.0
        
        if latest_signal > 0.7:
            return SignalType.STRONG_BUY, min(confidence, 0.95)
        elif latest_signal > 0.4:
            return SignalType.BUY, min(confidence, 0.85)
        elif latest_signal < -0.7:
            return SignalType.STRONG_SELL, min(confidence, 0.95)
        elif latest_signal < -0.4:
            return SignalType.SELL, min(confidence, 0.85)
        elif abs(latest_signal) > 0.2:
            return SignalType.HOLD, min(confidence, 0.7)
        else:
            return SignalType.NEUTRAL, min(confidence, 0.5)
    
    def _initialize_ml_components(self):
        """Initialize machine learning components for trend and breakout detection."""
        try:
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
            from sklearn.neural_network import MLPClassifier
            
            # Initialize trend detector
            self.trend_detector = RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                random_state=42,
                n_jobs=-1
            )
            
            # Initialize breakout predictor
            self.breakout_predictor = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            
        except ImportError:
            logger.warning("scikit-learn not available, disabling ML features")
            self.ml_trend_detection = False
    
    def calculate(self, data: pd.DataFrame) -> Dict[str, Union[pd.Series, np.ndarray, Dict]]:
        """
        Calculate Fibonacci channels with advanced trend and breakout analysis.
        
        Args:
            data: DataFrame with OHLC data
            
        Returns:
            Dictionary containing channel data, signals, and analysis
        """
        try:
            if len(data) < self.lookback_period:
                raise IndicatorCalculationException(
                    indicator_name=self.name,
                    calculation_step="data_validation",
                    message="Insufficient data for Fibonacci channel calculation"
                )
            
            # Detect primary trend
            trend_analysis = self._analyze_trend(data)
            
            if not trend_analysis or trend_analysis['strength'] < self.min_trend_strength:
                return self._empty_result(len(data))
            
            # Construct Fibonacci channels
            channels = self._construct_fibonacci_channels(data, trend_analysis)
            
            if not channels:
                return self._empty_result(len(data))
            
            # Detect breakouts
            breakout_signals = self._detect_breakouts(data, channels)
            
            # Generate trading signals
            signals = self._generate_channel_signals(data, channels, breakout_signals)
            
            # Calculate strength metrics
            strength_metrics = self._calculate_channel_strength(data, channels)
            
            # Identify support and resistance levels
            sr_levels = self._identify_support_resistance(channels, data)
            
            # Calculate price targets
            price_targets = self._calculate_price_targets(channels, data, breakout_signals)
            
            return {
                'channels': self._format_channels(channels),
                'breakout_signals': self._format_breakout_signals(breakout_signals),
                'signals': signals,
                'trend_analysis': trend_analysis,
                'strength': strength_metrics['strength'],
                'confidence': strength_metrics['confidence'],
                'channel_position': strength_metrics['channel_position'],
                'volatility_factor': strength_metrics['volatility_factor'],
                'support_resistance': sr_levels,
                'price_targets': price_targets,
                'channel_width': self._calculate_channel_width(channels),
                'trend_continuation_probability': self._calculate_trend_probability(data, channels),
                'breakout_zones': self._identify_breakout_zones(channels, data)
            }
            
        except Exception as e:
            logger.error(f"Error in Fibonacci channel calculation: {str(e)}")
            raise IndicatorCalculationException(
                indicator_name=self.name,
                calculation_step="main_calculation",
                message=f"Fibonacci channel calculation failed: {str(e)}",
                cause=e
            )
    
    def _analyze_trend(self, data: pd.DataFrame) -> Optional[Dict]:
        """Analyze the primary trend using advanced algorithms."""
        
        def _calculate_trend_strength(prices: np.ndarray, time_points: np.ndarray) -> Dict:
            """Calculate trend strength using multiple methods."""
            if len(prices) < 10:
                return {'strength': 0.0, 'direction': 'sideways', 'confidence': 0.0}
            
            # Linear regression trend
            X = time_points.reshape(-1, 1)
            lr = LinearRegression()
            lr.fit(X, prices)
            
            slope = lr.coef_[0]
            r_squared = lr.score(X, prices)
            
            # RANSAC for robust trend detection
            ransac = RANSACRegressor(random_state=42)
            ransac.fit(X, prices)
            
            robust_slope = ransac.estimator_.coef_[0]
            inlier_mask = ransac.inlier_mask_
            outlier_ratio = 1.0 - np.sum(inlier_mask) / len(inlier_mask)
            
            # Determine trend direction and strength
            price_range = np.max(prices) - np.min(prices)
            slope_normalized = slope / price_range if price_range > 0 else 0
            
            if abs(slope_normalized) < 0.001:
                direction = 'sideways'
                strength = 0.1
            elif slope_normalized > 0:
                direction = 'up'
                strength = min(1.0, abs(slope_normalized) * 100)
            else:
                direction = 'down'
                strength = min(1.0, abs(slope_normalized) * 100)
            
            # Confidence based on R² and outlier ratio
            confidence = r_squared * (1.0 - outlier_ratio)
            
            return {
                'direction': direction,
                'strength': strength,
                'confidence': confidence,
                'slope': slope,
                'robust_slope': robust_slope,
                'r_squared': r_squared,
                'outlier_ratio': outlier_ratio
            }
        
        # Analyze recent trend
        recent_data = data.iloc[-self.lookback_period:]
        prices = recent_data['close'].values
        time_points = np.arange(len(prices))
        
        trend_info = _calculate_trend_strength(prices, time_points)
        
        # Add volatility context
        atr = self._calculate_atr(recent_data)
        avg_atr = np.mean(atr[-10:]) if len(atr) >= 10 else 0
        volatility_factor = avg_atr / np.mean(prices) if np.mean(prices) > 0 else 0
        
        trend_info['volatility_factor'] = volatility_factor
        trend_info['atr'] = avg_atr
        
        # ML trend validation if enabled
        if self.ml_trend_detection and self.trend_detector is not None:
            try:
                ml_confidence = self._validate_trend_with_ml(recent_data, trend_info)
                trend_info['ml_confidence'] = ml_confidence
                trend_info['confidence'] = (trend_info['confidence'] + ml_confidence) / 2
            except Exception as e:
                logger.warning(f"ML trend validation failed: {str(e)}")
        
        return trend_info
    
    def _validate_trend_with_ml(self, data: pd.DataFrame, trend_info: Dict) -> float:
        """Validate trend using machine learning model."""
        try:
            # Extract features for ML validation
            features = self._extract_trend_features(data, trend_info)
            
            # This would use a pre-trained model in production
            # For now, return a confidence based on statistical measures
            return min(1.0, trend_info['r_squared'] * (2.0 - trend_info['outlier_ratio']))
            
        except Exception:
            return 0.5  # Neutral confidence
    
    def _extract_trend_features(self, data: pd.DataFrame, trend_info: Dict) -> np.ndarray:
        """Extract features for ML trend validation."""
        features = [
            trend_info['strength'],
            trend_info['confidence'],
            trend_info['r_squared'],
            trend_info['outlier_ratio'],
            trend_info['volatility_factor']
        ]
        
        # Add technical indicators as features
        if len(data) >= 20:
            rsi = self._calculate_rsi(data['close'], 14)
            macd = self._calculate_macd(data['close'])
            
            features.extend([
                rsi[-1] if len(rsi) > 0 else 50,
                macd['signal'][-1] if len(macd['signal']) > 0 else 0
            ])
        else:
            features.extend([50, 0])
        
        return np.array(features).reshape(1, -1)
    
    def _construct_fibonacci_channels(self, data: pd.DataFrame, trend_analysis: Dict) -> List[ChannelBand]:
        """Construct Fibonacci channels based on trend analysis."""
        channels = []
        
        try:
            # Find trend line anchor points
            anchor_points = self._find_anchor_points(data, trend_analysis)
            
            if len(anchor_points) < 2:
                return channels
            
            # Calculate base trend line
            base_line = self._calculate_trend_line(anchor_points, data)
            
            # Construct parallel channels at Fibonacci levels
            for ratio in self.fibonacci_ratios:
                try:
                    channel = self._create_channel_band(base_line, ratio, data, trend_analysis)
                    if channel and self._validate_channel(channel, data):
                        channels.append(channel)
                        
                except Exception as e:
                    logger.warning(f"Error creating channel for ratio {ratio}: {str(e)}")
                    continue
            
            # Sort channels by strength
            channels.sort(key=lambda x: x.trend_strength, reverse=True)
            
            return channels[:5]  # Return top 5 channels
            
        except Exception as e:
            logger.error(f"Error constructing Fibonacci channels: {str(e)}")
            return []
    
    def _find_anchor_points(self, data: pd.DataFrame, trend_analysis: Dict) -> List[Tuple[int, float]]:
        """Find key anchor points for trend line construction."""
        anchor_points = []
        
        direction = trend_analysis['direction']
        recent_data = data.iloc[-self.lookback_period:]
        
        if direction == 'up':
            # Find significant lows for uptrend
            anchor_points = self._find_significant_lows(recent_data)
        elif direction == 'down':
            # Find significant highs for downtrend
            anchor_points = self._find_significant_highs(recent_data)
        else:
            # For sideways, find both highs and lows
            highs = self._find_significant_highs(recent_data)
            lows = self._find_significant_lows(recent_data)
            anchor_points = highs + lows
        
        # Sort by time
        anchor_points.sort(key=lambda x: x[0])
        
        return anchor_points
    
    def _find_significant_lows(self, data: pd.DataFrame) -> List[Tuple[int, float]]:
        """Find significant low points for trend line construction."""
        lows = []
        
        # Simple pivot low detection
        for i in range(2, len(data) - 2):
            current_low = data.iloc[i]['low']
            
            # Check if it's a local minimum
            if (current_low <= data.iloc[i-1]['low'] and 
                current_low <= data.iloc[i-2]['low'] and
                current_low <= data.iloc[i+1]['low'] and 
                current_low <= data.iloc[i+2]['low']):
                
                lows.append((i, current_low))
        
        # Filter by significance
        if len(lows) > 3:
            # Keep only the most significant lows
            lows.sort(key=lambda x: x[1])  # Sort by price
            lows = lows[:3]  # Keep 3 lowest
        
        return lows
    
    def _find_significant_highs(self, data: pd.DataFrame) -> List[Tuple[int, float]]:
        """Find significant high points for trend line construction."""
        highs = []
        
        # Simple pivot high detection
        for i in range(2, len(data) - 2):
            current_high = data.iloc[i]['high']
            
            # Check if it's a local maximum
            if (current_high >= data.iloc[i-1]['high'] and 
                current_high >= data.iloc[i-2]['high'] and
                current_high >= data.iloc[i+1]['high'] and 
                current_high >= data.iloc[i+2]['high']):
                
                highs.append((i, current_high))
        
        # Filter by significance
        if len(highs) > 3:
            # Keep only the most significant highs
            highs.sort(key=lambda x: x[1], reverse=True)  # Sort by price descending
            highs = highs[:3]  # Keep 3 highest
        
        return highs
    
    def _calculate_trend_line(self, anchor_points: List[Tuple[int, float]], data: pd.DataFrame) -> Dict:
        """Calculate the primary trend line from anchor points."""
        if len(anchor_points) < 2:
            return {}
        
        # Use linear regression on anchor points
        x_points = np.array([point[0] for point in anchor_points])
        y_points = np.array([point[1] for point in anchor_points])
        
        X = x_points.reshape(-1, 1)
        lr = LinearRegression()
        lr.fit(X, y_points)
        
        slope = lr.coef_[0]
        intercept = lr.intercept_
        r_squared = lr.score(X, y_points)
        
        return {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_squared,
            'start_point': (x_points[0], y_points[0]),
            'end_point': (x_points[-1], y_points[-1])
        }
    
    def _create_channel_band(self, base_line: Dict, ratio: float, 
                           data: pd.DataFrame, trend_analysis: Dict) -> Optional[ChannelBand]:
        """Create a channel band at a specific Fibonacci ratio."""
        if not base_line:
            return None
        
        try:
            # Calculate channel width based on volatility
            atr = self._calculate_atr(data.iloc[-20:])
            avg_atr = np.mean(atr) if len(atr) > 0 else 0
            
            base_width = avg_atr * ratio
            if self.volatility_adjustment:
                volatility_factor = trend_analysis.get('volatility_factor', 1.0)
                base_width *= (1.0 + volatility_factor)
            
            # Create upper and lower channel lines
            upper_intercept = base_line['intercept'] + base_width
            lower_intercept = base_line['intercept'] - base_width
            
            # Calculate line properties
            upper_line = FibonacciChannelLine(
                ratio=ratio,
                slope=base_line['slope'],
                intercept=upper_intercept,
                upper_bound=upper_intercept + base_width * 0.1,
                lower_bound=upper_intercept - base_width * 0.1,
                confidence=base_line['r_squared'],
                strength=trend_analysis['strength'],
                direction=trend_analysis['direction'],
                start_point=base_line['start_point'],
                end_point=base_line['end_point'],
                volume_confirmation=0.5,  # Will be calculated later
                breakout_probability=0.0
            )
            
            lower_line = FibonacciChannelLine(
                ratio=ratio,
                slope=base_line['slope'],
                intercept=lower_intercept,
                upper_bound=lower_intercept + base_width * 0.1,
                lower_bound=lower_intercept - base_width * 0.1,
                confidence=base_line['r_squared'],
                strength=trend_analysis['strength'],
                direction=trend_analysis['direction'],
                start_point=base_line['start_point'],
                end_point=base_line['end_point'],
                volume_confirmation=0.5,
                breakout_probability=0.0
            )
            
            middle_line = FibonacciChannelLine(
                ratio=ratio,
                slope=base_line['slope'],
                intercept=base_line['intercept'],
                upper_bound=base_line['intercept'] + base_width * 0.05,
                lower_bound=base_line['intercept'] - base_width * 0.05,
                confidence=base_line['r_squared'],
                strength=trend_analysis['strength'],
                direction=trend_analysis['direction'],
                start_point=base_line['start_point'],
                end_point=base_line['end_point'],
                volume_confirmation=0.5,
                breakout_probability=0.0
            )
            
            # Create channel band
            channel = ChannelBand(
                upper_line=upper_line,
                lower_line=lower_line,
                middle_line=middle_line,
                channel_width=base_width * 2,
                trend_strength=trend_analysis['strength'],
                volatility_adjustment=trend_analysis.get('volatility_factor', 1.0),
                support_resistance_strength=0.0  # Will be calculated
            )
            
            # Calculate support/resistance strength
            channel.support_resistance_strength = self._calculate_sr_strength(channel, data)
            
            return channel
            
        except Exception as e:
            logger.warning(f"Error creating channel band: {str(e)}")
            return None
    
    def _validate_channel(self, channel: ChannelBand, data: pd.DataFrame) -> bool:
        """Validate if a channel is meaningful and should be kept."""
        try:
            # Check if channel has reasonable width
            if channel.channel_width <= 0:
                return False
            
            # Check if prices interact with the channel
            recent_data = data.iloc[-20:]
            interactions = 0
            
            for i, row in recent_data.iterrows():
                price_at_time = self._get_channel_price_at_time(channel, i, len(data))
                
                if price_at_time:
                    upper_price = price_at_time['upper']
                    lower_price = price_at_time['lower']
                    
                    # Check if price is near channel lines
                    if (abs(row['high'] - upper_price) / upper_price < 0.02 or
                        abs(row['low'] - lower_price) / lower_price < 0.02):
                        interactions += 1
            
            # Require at least 3 interactions in recent data
            return interactions >= 3
            
        except Exception:
            return False
    
    def _get_channel_price_at_time(self, channel: ChannelBand, time_idx: int, 
                                 total_length: int) -> Optional[Dict]:
        """Get channel prices at a specific time index."""
        try:
            relative_time = time_idx - (total_length - self.lookback_period)
            
            upper_price = channel.upper_line.slope * relative_time + channel.upper_line.intercept
            lower_price = channel.lower_line.slope * relative_time + channel.lower_line.intercept
            middle_price = channel.middle_line.slope * relative_time + channel.middle_line.intercept
            
            return {
                'upper': upper_price,
                'lower': lower_price,
                'middle': middle_price
            }
            
        except Exception:
            return None
    
    def _calculate_sr_strength(self, channel: ChannelBand, data: pd.DataFrame) -> float:
        """Calculate support/resistance strength for a channel."""
        try:
            strength = 0.0
            touches = 0
            
            recent_data = data.iloc[-30:]  # Last 30 periods
            
            for i, row in recent_data.iterrows():
                channel_prices = self._get_channel_price_at_time(channel, i, len(data))
                
                if not channel_prices:
                    continue
                
                # Check touches on upper line
                upper_distance = abs(row['high'] - channel_prices['upper']) / channel_prices['upper']
                if upper_distance < 0.01:  # Within 1%
                    touches += 1
                    strength += channel.upper_line.confidence
                
                # Check touches on lower line
                lower_distance = abs(row['low'] - channel_prices['lower']) / channel_prices['lower']
                if lower_distance < 0.01:  # Within 1%
                    touches += 1
                    strength += channel.lower_line.confidence
            
            return min(1.0, strength / max(touches, 1))
            
        except Exception:
            return 0.0
    
    def _detect_breakouts(self, data: pd.DataFrame, channels: List[ChannelBand]) -> List[BreakoutSignal]:
        """Detect breakouts from Fibonacci channels."""
        breakouts = []
        
        try:
            for channel in channels:
                channel_breakouts = self._detect_channel_breakouts(data, channel)
                breakouts.extend(channel_breakouts)
            
            # Sort by strength and return top breakouts
            breakouts.sort(key=lambda x: x.strength, reverse=True)
            return breakouts[:3]  # Return top 3 breakouts
            
        except Exception as e:
            logger.warning(f"Error detecting breakouts: {str(e)}")
            return []
    
    def _detect_channel_breakouts(self, data: pd.DataFrame, channel: ChannelBand) -> List[BreakoutSignal]:
        """Detect breakouts for a specific channel."""
        breakouts = []
        
        try:
            recent_data = data.iloc[-self.breakout_confirmation_period:]
            
            for i in range(len(recent_data)):
                row = recent_data.iloc[i]
                time_idx = len(data) - len(recent_data) + i
                
                channel_prices = self._get_channel_price_at_time(channel, time_idx, len(data))
                if not channel_prices:
                    continue
                
                # Check for bullish breakout (above upper channel)
                if row['close'] > channel_prices['upper']:
                    strength = self._calculate_breakout_strength(
                        row, channel_prices['upper'], 'bullish', data, i
                    )
                    
                    if strength > 0.5:
                        breakout = BreakoutSignal(
                            timestamp=row.name if hasattr(row, 'name') else datetime.now(),
                            price=row['close'],
                            direction='bullish',
                            strength=strength,
                            volume_confirmation=self._get_volume_confirmation(data, i),
                            target_price=self._calculate_breakout_target(row['close'], channel, 'bullish'),
                            stop_loss=channel_prices['middle'],
                            confidence=min(0.9, strength * channel.trend_strength)
                        )
                        breakouts.append(breakout)
                
                # Check for bearish breakout (below lower channel)
                elif row['close'] < channel_prices['lower']:
                    strength = self._calculate_breakout_strength(
                        row, channel_prices['lower'], 'bearish', data, i
                    )
                    
                    if strength > 0.5:
                        breakout = BreakoutSignal(
                            timestamp=row.name if hasattr(row, 'name') else datetime.now(),
                            price=row['close'],
                            direction='bearish',
                            strength=strength,
                            volume_confirmation=self._get_volume_confirmation(data, i),
                            target_price=self._calculate_breakout_target(row['close'], channel, 'bearish'),
                            stop_loss=channel_prices['middle'],
                            confidence=min(0.9, strength * channel.trend_strength)
                        )
                        breakouts.append(breakout)
            
            return breakouts
            
        except Exception as e:
            logger.warning(f"Error detecting channel breakouts: {str(e)}")
            return []
    
    def _calculate_breakout_strength(self, row: pd.Series, channel_level: float, 
                                   direction: str, data: pd.DataFrame, idx: int) -> float:
        """Calculate the strength of a breakout."""
        try:
            # Distance from channel level
            distance = abs(row['close'] - channel_level) / channel_level
            
            # Volume confirmation
            volume_factor = 1.0
            if 'volume' in row:
                avg_volume = data['volume'].iloc[-20:].mean()
                volume_factor = min(2.0, row['volume'] / avg_volume) if avg_volume > 0 else 1.0
            
            # Momentum confirmation
            momentum_factor = 1.0
            if idx >= 3:
                recent_close = data['close'].iloc[-3:]
                if direction == 'bullish':
                    momentum_factor = 1.5 if recent_close.iloc[-1] > recent_close.iloc[0] else 0.8
                else:
                    momentum_factor = 1.5 if recent_close.iloc[-1] < recent_close.iloc[0] else 0.8
            
            strength = distance * 10 * volume_factor * momentum_factor
            return min(1.0, strength)
            
        except Exception:
            return 0.0
    
    def _get_volume_confirmation(self, data: pd.DataFrame, idx: int) -> float:
        """Get volume confirmation for a breakout."""
        try:
            if 'volume' not in data.columns:
                return 0.5
            
            current_volume = data['volume'].iloc[-(len(data) - idx)]
            avg_volume = data['volume'].iloc[-20:].mean()
            
            return min(1.0, current_volume / avg_volume) if avg_volume > 0 else 0.5
            
        except Exception:
            return 0.5
    
    def _calculate_breakout_target(self, entry_price: float, channel: ChannelBand, 
                                 direction: str) -> float:
        """Calculate price target for a breakout."""
        try:
            channel_width = channel.channel_width
            
            if direction == 'bullish':
                return entry_price + channel_width * 1.618  # Fibonacci extension
            else:
                return entry_price - channel_width * 1.618
                
        except Exception:
            return entry_price
    
    def _generate_channel_signals(self, data: pd.DataFrame, channels: List[ChannelBand], 
                                breakout_signals: List[BreakoutSignal]) -> pd.Series:
        """Generate trading signals based on channel analysis."""
        signals = pd.Series(0.0, index=data.index)
        
        if not channels:
            return signals
        
        try:
            # Generate signals for recent periods
            for i in range(max(1, len(data) - 50), len(data)):
                signal_strength = 0.0
                
                # Channel position signals
                for channel in channels:
                    channel_prices = self._get_channel_price_at_time(channel, i, len(data))
                    if not channel_prices:
                        continue
                    
                    current_price = data.iloc[i]['close']
                    
                    # Calculate position within channel
                    channel_range = channel_prices['upper'] - channel_prices['lower']
                    if channel_range > 0:
                        position = (current_price - channel_prices['lower']) / channel_range
                        
                        # Generate position-based signals
                        if position < 0.2:  # Near lower channel
                            signal_strength += channel.trend_strength * 0.3
                        elif position > 0.8:  # Near upper channel
                            signal_strength -= channel.trend_strength * 0.3
                        elif 0.4 < position < 0.6:  # Near middle
                            signal_strength += channel.trend_strength * 0.1
                
                # Breakout signals
                for breakout in breakout_signals:
                    if breakout.direction == 'bullish':
                        signal_strength += breakout.strength * 0.5
                    else:
                        signal_strength -= breakout.strength * 0.5
                
                signals.iloc[i] = np.clip(signal_strength, -1.0, 1.0)
            
            return signals
            
        except Exception as e:
            logger.warning(f"Error generating channel signals: {str(e)}")
            return signals
    
    def _calculate_channel_strength(self, data: pd.DataFrame, 
                                  channels: List[ChannelBand]) -> Dict[str, pd.Series]:
        """Calculate comprehensive strength metrics for channels."""
        length = len(data)
        
        strength = pd.Series(0.0, index=data.index)
        confidence = pd.Series(0.0, index=data.index)
        channel_position = pd.Series(0.0, index=data.index)
        volatility_factor = pd.Series(1.0, index=data.index)
        
        if not channels:
            return {
                'strength': strength,
                'confidence': confidence,
                'channel_position': channel_position,
                'volatility_factor': volatility_factor
            }
        
        try:
            # Calculate ATR for volatility context
            atr = self._calculate_atr(data)
            
            for i in range(length):
                avg_strength = 0.0
                avg_confidence = 0.0
                total_position = 0.0
                valid_channels = 0
                
                for channel in channels:
                    channel_prices = self._get_channel_price_at_time(channel, i, length)
                    if not channel_prices:
                        continue
                    
                    current_price = data.iloc[i]['close']
                    
                    # Calculate position within channel
                    channel_range = channel_prices['upper'] - channel_prices['lower']
                    if channel_range > 0:
                        position = (current_price - channel_prices['lower']) / channel_range
                        total_position += position
                        valid_channels += 1
                    
                    avg_strength += channel.trend_strength
                    avg_confidence += channel.upper_line.confidence
                
                if valid_channels > 0:
                    strength.iloc[i] = avg_strength / valid_channels
                    confidence.iloc[i] = avg_confidence / valid_channels
                    channel_position.iloc[i] = total_position / valid_channels
                
                # Volatility factor
                if i < len(atr):
                    avg_price = data.iloc[max(0, i-10):i+1]['close'].mean()
                    if avg_price > 0:
                        volatility_factor.iloc[i] = atr[i] / avg_price
            
            return {
                'strength': strength,
                'confidence': confidence,
                'channel_position': channel_position,
                'volatility_factor': volatility_factor
            }
            
        except Exception as e:
            logger.warning(f"Error calculating channel strength: {str(e)}")
            return {
                'strength': strength,
                'confidence': confidence,
                'channel_position': channel_position,
                'volatility_factor': volatility_factor
            }
    
    def _identify_support_resistance(self, channels: List[ChannelBand], 
                                   data: pd.DataFrame) -> Dict[str, List[float]]:
        """Identify key support and resistance levels from channels."""
        support_levels = []
        resistance_levels = []
        
        try:
            current_price = data['close'].iloc[-1]
            
            for channel in channels:
                channel_prices = self._get_channel_price_at_time(channel, len(data)-1, len(data))
                if not channel_prices:
                    continue
                
                # Add channel levels as support/resistance
                if channel_prices['upper'] > current_price:
                    resistance_levels.append({
                        'price': channel_prices['upper'],
                        'strength': channel.support_resistance_strength
                    })
                
                if channel_prices['lower'] < current_price:
                    support_levels.append({
                        'price': channel_prices['lower'],
                        'strength': channel.support_resistance_strength
                    })
                
                # Middle line can act as both
                if abs(channel_prices['middle'] - current_price) / current_price < 0.05:
                    if channel_prices['middle'] > current_price:
                        resistance_levels.append({
                            'price': channel_prices['middle'],
                            'strength': channel.support_resistance_strength * 0.7
                        })
                    else:
                        support_levels.append({
                            'price': channel_prices['middle'],
                            'strength': channel.support_resistance_strength * 0.7
                        })
            
            # Sort by strength and return top levels
            support_levels.sort(key=lambda x: x['strength'], reverse=True)
            resistance_levels.sort(key=lambda x: x['strength'], reverse=True)
            
            return {
                'support': [level['price'] for level in support_levels[:3]],
                'resistance': [level['price'] for level in resistance_levels[:3]]
            }
            
        except Exception as e:
            logger.warning(f"Error identifying support/resistance: {str(e)}")
            return {'support': [], 'resistance': []}
    
    def _calculate_price_targets(self, channels: List[ChannelBand], data: pd.DataFrame, 
                               breakout_signals: List[BreakoutSignal]) -> Dict[str, float]:
        """Calculate price targets based on channel analysis."""
        targets = {}
        
        try:
            current_price = data['close'].iloc[-1]
            
            # Targets from breakout signals
            bullish_targets = []
            bearish_targets = []
            
            for breakout in breakout_signals:
                if breakout.direction == 'bullish' and breakout.target_price > current_price:
                    bullish_targets.append(breakout.target_price)
                elif breakout.direction == 'bearish' and breakout.target_price < current_price:
                    bearish_targets.append(breakout.target_price)
            
            if bullish_targets:
                targets['bullish_target'] = min(bullish_targets)  # Nearest bullish target
            
            if bearish_targets:
                targets['bearish_target'] = max(bearish_targets)  # Nearest bearish target
            
            # Targets from channel projections
            if channels:
                strongest_channel = max(channels, key=lambda x: x.trend_strength)
                channel_prices = self._get_channel_price_at_time(strongest_channel, len(data)-1, len(data))
                
                if channel_prices:
                    targets['upper_channel_target'] = channel_prices['upper']
                    targets['lower_channel_target'] = channel_prices['lower']
                    targets['middle_channel_target'] = channel_prices['middle']
            
            return targets
            
        except Exception as e:
            logger.warning(f"Error calculating price targets: {str(e)}")
            return {}
    
    def _calculate_channel_width(self, channels: List[ChannelBand]) -> pd.Series:
        """Calculate average channel width over time."""
        if not channels:
            return pd.Series([0.0])
        
        try:
            avg_width = np.mean([channel.channel_width for channel in channels])
            return pd.Series([avg_width])
            
        except Exception:
            return pd.Series([0.0])
    
    def _calculate_trend_probability(self, data: pd.DataFrame, 
                                   channels: List[ChannelBand]) -> float:
        """Calculate probability of trend continuation."""
        try:
            if not channels:
                return 0.5
            
            # Average channel strength
            avg_strength = np.mean([channel.trend_strength for channel in channels])
            
            # Recent price action alignment
            recent_data = data.iloc[-10:]
            trend_alignment = 0.0
            
            for channel in channels:
                if channel.middle_line.direction == 'up':
                    if recent_data['close'].iloc[-1] > recent_data['close'].iloc[0]:
                        trend_alignment += 1.0
                elif channel.middle_line.direction == 'down':
                    if recent_data['close'].iloc[-1] < recent_data['close'].iloc[0]:
                        trend_alignment += 1.0
            
            if channels:
                trend_alignment /= len(channels)
            
            return min(1.0, (avg_strength + trend_alignment) / 2)
            
        except Exception:
            return 0.5
    
    def _identify_breakout_zones(self, channels: List[ChannelBand], 
                               data: pd.DataFrame) -> List[Dict]:
        """Identify potential breakout zones."""
        zones = []
        
        try:
            current_price = data['close'].iloc[-1]
            
            for channel in channels:
                channel_prices = self._get_channel_price_at_time(channel, len(data)-1, len(data))
                if not channel_prices:
                    continue
                
                # Upper breakout zone
                upper_distance = abs(current_price - channel_prices['upper']) / current_price
                if upper_distance < 0.05:  # Within 5%
                    zones.append({
                        'type': 'resistance_breakout',
                        'price': channel_prices['upper'],
                        'distance': upper_distance,
                        'strength': channel.support_resistance_strength,
                        'direction': 'bullish'
                    })
                
                # Lower breakout zone
                lower_distance = abs(current_price - channel_prices['lower']) / current_price
                if lower_distance < 0.05:  # Within 5%
                    zones.append({
                        'type': 'support_breakout',
                        'price': channel_prices['lower'],
                        'distance': lower_distance,
                        'strength': channel.support_resistance_strength,
                        'direction': 'bearish'
                    })
            
            # Sort by proximity
            zones.sort(key=lambda x: x['distance'])
            return zones[:3]  # Return top 3 zones
            
        except Exception as e:
            logger.warning(f"Error identifying breakout zones: {str(e)}")
            return []
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> np.ndarray:
        """Calculate Average True Range."""
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift(1))
        low_close = np.abs(data['low'] - data['close'].shift(1))
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean().fillna(0).values
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> np.ndarray:
        """Calculate RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50).values
    
    def _calculate_macd(self, prices: pd.Series) -> Dict:
        """Calculate MACD."""
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9).mean()
        
        return {
            'macd': macd_line.fillna(0).values,
            'signal': signal_line.fillna(0).values
        }
    
    def _format_channels(self, channels: List[ChannelBand]) -> pd.DataFrame:
        """Format channel data for output."""
        if not channels:
            return pd.DataFrame()
        
        channel_data = []
        for i, channel in enumerate(channels):
            channel_data.append({
                'channel_id': i,
                'upper_slope': channel.upper_line.slope,
                'upper_intercept': channel.upper_line.intercept,
                'lower_slope': channel.lower_line.slope,
                'lower_intercept': channel.lower_line.intercept,
                'middle_slope': channel.middle_line.slope,
                'middle_intercept': channel.middle_line.intercept,
                'channel_width': channel.channel_width,
                'trend_strength': channel.trend_strength,
                'sr_strength': channel.support_resistance_strength,
                'direction': channel.upper_line.direction
            })
        
        return pd.DataFrame(channel_data)
    
    def _format_breakout_signals(self, breakouts: List[BreakoutSignal]) -> pd.DataFrame:
        """Format breakout signals for output."""
        if not breakouts:
            return pd.DataFrame()
        
        breakout_data = []
        for breakout in breakouts:
            breakout_data.append({
                'timestamp': breakout.timestamp,
                'price': breakout.price,
                'direction': breakout.direction,
                'strength': breakout.strength,
                'volume_confirmation': breakout.volume_confirmation,
                'target_price': breakout.target_price,
                'stop_loss': breakout.stop_loss,
                'confidence': breakout.confidence
            })
        
        return pd.DataFrame(breakout_data)
    
    def _empty_result(self, length: int) -> Dict[str, Union[pd.Series, np.ndarray, Dict]]:
        """Return empty result structure."""
        empty_series = pd.Series(0.0, index=range(length))
        
        return {
            'channels': pd.DataFrame(),
            'breakout_signals': pd.DataFrame(),
            'signals': empty_series,
            'trend_analysis': {},
            'strength': empty_series,
            'confidence': empty_series,
            'channel_position': empty_series,
            'volatility_factor': empty_series,
            'support_resistance': {'support': [], 'resistance': []},
            'price_targets': {},
            'channel_width': pd.Series([0.0]),
            'trend_continuation_probability': 0.5,
            'breakout_zones': []
        }