"""
Gann Box Indicator - Advanced Implementation

This module implements W.D. Gann's box analysis with sophisticated mathematical models,
automatic box generation, and advanced pattern recognition capabilities for maximum profitability.

The implementation includes:
- Automated significant high/low identification
- Dynamic price-time box generation and scaling
- Advanced geometric box analysis
- Pattern recognition within boxes
- Support/resistance level extraction from box boundaries
- Box breakout prediction using machine learning
- Multi-timeframe box harmonics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, NamedTuple
from dataclasses import dataclass
import logging
from scipy import stats, optimize, spatial
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.cluster import DBSCAN
import math

from ..base.standard_indicator import StandardIndicatorInterface
from ....core.exceptions import IndicatorCalculationError


class GannBox(NamedTuple):
    """Represents a Gann box with all its properties"""
    id: str
    start_time: float
    end_time: float
    start_price: float
    end_price: float
    high_price: float
    low_price: float
    width_time: float
    height_price: float
    volume_profile: Dict
    pattern_type: str
    strength: float
    breakout_direction: Optional[str]
    
    
@dataclass
class GannBoxConfig:
    """Configuration for Gann Box calculations"""
    min_box_height_atr: float = 2.0  # Minimum box height in ATR units
    min_box_width_bars: int = 10     # Minimum box width in bars
    max_box_width_bars: int = 100    # Maximum box width in bars
    lookback_period: int = 500       # Bars to look back for box identification
    significance_threshold: float = 0.7  # Threshold for significant highs/lows
    volume_confirmation: bool = True  # Require volume confirmation for boxes
    pattern_recognition: bool = True  # Enable pattern recognition within boxes
    ml_breakout_prediction: bool = True  # Enable ML-based breakout prediction
    box_overlap_tolerance: float = 0.1  # Tolerance for overlapping boxes
    harmonic_analysis: bool = True    # Enable harmonic relationships between boxes


class GannBoxIndicator(StandardIndicatorInterface):
    """
    Advanced Gann Box Indicator Implementation
    
    This indicator implements W.D. Gann's box theory with sophisticated enhancements:
    
    1. Automated Box Detection: Identifies significant price-time boxes using advanced algorithms
    2. Dynamic Scaling: Adjusts box dimensions based on market volatility and timeframe
    3. Pattern Recognition: Detects patterns within boxes (accumulation, distribution, etc.)
    4. Volume Analysis: Incorporates volume profile analysis within each box
    5. Breakout Prediction: Uses ML to predict box breakout direction and timing
    6. Harmonic Analysis: Analyzes harmonic relationships between multiple boxes
    7. Support/Resistance: Extracts key levels from box boundaries
    
    Mathematical Foundation:
    - Box significance: Z-score of price range relative to historical ranges
    - Volume confirmation: Volume ratio compared to historical average
    - Pattern classification: Statistical analysis of price action within box
    - Breakout probability: ML model trained on historical breakout patterns
    """
    
    def __init__(self, config: Optional[GannBoxConfig] = None):
        super().__init__()
        self.config = config or GannBoxConfig()
        self.scaler = StandardScaler()
        self.breakout_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        )
        self.anomaly_detector = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        self.is_trained = False
        self.box_cache = {}
        self.logger = logging.getLogger(__name__)
        
    def get_name(self) -> str:
        return "GannBoxIndicator"
        
    def get_required_columns(self) -> List[str]:
        return ['high', 'low', 'close', 'open', 'volume', 'timestamp']
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Gann Boxes with advanced mathematical modeling
        
        Args:
            data: DataFrame with OHLCV data and timestamp
            
        Returns:
            DataFrame with box data, patterns, and trading signals
        """
        try:
            if len(data) < self.config.lookback_period:
                raise IndicatorCalculationError(
                    f"Insufficient data: {len(data)} bars, need {self.config.lookback_period}"
                )
                
            # Prepare data
            df = data.copy()
            df = self._prepare_market_data(df)
            
            # Identify significant highs and lows
            significant_points = self._identify_significant_points(df)
            
            # Generate Gann boxes
            boxes = self._generate_gann_boxes(df, significant_points)
            
            # Analyze patterns within boxes
            if self.config.pattern_recognition:
                boxes = self._analyze_box_patterns(df, boxes)
                
            # Calculate volume profiles for each box
            if self.config.volume_confirmation:
                boxes = self._calculate_volume_profiles(df, boxes)
                
            # Predict breakouts using machine learning
            if self.config.ml_breakout_prediction:
                boxes = self._predict_breakouts(df, boxes)
                
            # Perform harmonic analysis
            if self.config.harmonic_analysis:
                harmonic_data = self._perform_harmonic_analysis(boxes)
            else:
                harmonic_data = {}
                
            # Extract support/resistance levels
            sr_levels = self._extract_support_resistance(boxes)
            
            # Generate trading signals
            signals = self._generate_trading_signals(df, boxes, sr_levels)
            
            # Combine results
            result = self._combine_results(df, boxes, sr_levels, harmonic_data, signals)
            
            self.logger.info(f"Generated {len(boxes)} Gann boxes for {len(result)} periods")
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating Gann boxes: {str(e)}")
            raise IndicatorCalculationError(f"Gann box calculation failed: {str(e)}")
            
    def _prepare_market_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare and enhance market data for box analysis"""
        df = df.copy()
        
        # Calculate price statistics
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['median_price'] = (df['high'] + df['low']) / 2
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
        
        # Calculate volume statistics
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        df['volume_std'] = df['volume'].rolling(window=20).std()
        
        # Calculate momentum indicators
        df['rsi'] = self._calculate_rsi(df['close'], 14)
        df['momentum'] = df['close'].pct_change(10)
        df['price_velocity'] = df['close'].diff().rolling(3).mean()
        
        # Normalize time index
        if 'timestamp' in df.columns:
            df['time_index'] = pd.to_datetime(df['timestamp'])
            df['time_numeric'] = (df['time_index'] - df['time_index'].iloc[0]).dt.total_seconds()
        else:
            df['time_numeric'] = np.arange(len(df))
            
        # Calculate price deviation from trend
        df['price_trend'] = df['close'].rolling(window=50).mean()
        df['price_deviation'] = (df['close'] - df['price_trend']) / df['atr']
        
        return df
        
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI for momentum analysis"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
        
    def _identify_significant_points(self, df: pd.DataFrame) -> List[Dict]:
        """Identify significant highs and lows for box generation"""
        significant_points = []
        
        # Method 1: Statistical significance using Z-scores
        window = 20
        
        # Calculate rolling statistics
        df['high_zscore'] = (df['high'] - df['high'].rolling(window).mean()) / df['high'].rolling(window).std()
        df['low_zscore'] = (df['low'] - df['low'].rolling(window).mean()) / df['low'].rolling(window).std()
        
        # Method 2: Local extrema with volume confirmation
        df['high_extrema'] = (
            (df['high'] == df['high'].rolling(window, center=True).max()) &
            (df['high_zscore'] > 1.5)
        )
        df['low_extrema'] = (
            (df['low'] == df['low'].rolling(window, center=True).min()) &
            (df['low_zscore'] < -1.5)
        )
        
        # Method 3: Volatility-adjusted significance
        df['high_volatility_adj'] = df['high'] / (df['atr'] + 1e-8)
        df['low_volatility_adj'] = df['low'] / (df['atr'] + 1e-8)
        
        volatility_threshold = df['high_volatility_adj'].quantile(0.85)
        
        # Combine all methods to identify significant points
        for i in range(window, len(df) - window):
            point_data = {
                'index': i,
                'time': df['time_numeric'].iloc[i],
                'high_price': df['high'].iloc[i],
                'low_price': df['low'].iloc[i],
                'close_price': df['close'].iloc[i],
                'volume': df['volume'].iloc[i],
                'atr': df['atr'].iloc[i],
                'is_high_significant': False,
                'is_low_significant': False,
                'significance_score': 0.0
            }
            
            # Check for significant high
            if (df['high_extrema'].iloc[i] or 
                df['high_volatility_adj'].iloc[i] > volatility_threshold):
                
                # Volume confirmation
                volume_confirmation = df['volume_ratio'].iloc[i] > 1.2
                
                # Calculate significance score
                price_significance = df['high_zscore'].iloc[i]
                volume_significance = df['volume_ratio'].iloc[i] - 1.0
                
                significance_score = (price_significance + volume_significance) / 2
                
                if significance_score > self.config.significance_threshold:
                    point_data['is_high_significant'] = True
                    point_data['significance_score'] = significance_score
                    
            # Check for significant low
            if (df['low_extrema'].iloc[i] or 
                df['low_volatility_adj'].iloc[i] > volatility_threshold):
                
                # Volume confirmation
                volume_confirmation = df['volume_ratio'].iloc[i] > 1.2
                
                # Calculate significance score
                price_significance = abs(df['low_zscore'].iloc[i])
                volume_significance = df['volume_ratio'].iloc[i] - 1.0
                
                significance_score = (price_significance + volume_significance) / 2
                
                if significance_score > self.config.significance_threshold:
                    point_data['is_low_significant'] = True
                    point_data['significance_score'] = max(
                        point_data['significance_score'], significance_score
                    )
                    
            # Add point if it's significant
            if point_data['is_high_significant'] or point_data['is_low_significant']:
                significant_points.append(point_data)
                
        # Filter points by minimum distance
        significant_points = self._filter_close_points(
            significant_points, 
            min_distance=self.config.min_box_width_bars // 2
        )
        
        self.logger.info(f"Identified {len(significant_points)} significant points")
        return significant_points
        
    def _filter_close_points(self, points: List[Dict], min_distance: int) -> List[Dict]:
        """Filter out points that are too close to each other"""
        if not points:
            return points
            
        # Sort by significance score
        points.sort(key=lambda x: x['significance_score'], reverse=True)
        
        filtered = []
        for point in points:
            # Check distance to all previously added points
            too_close = False
            for existing_point in filtered:
                if abs(point['index'] - existing_point['index']) < min_distance:
                    too_close = True
                    break
                    
            if not too_close:
                filtered.append(point)
                
        return sorted(filtered, key=lambda x: x['index'])
        
    def _generate_gann_boxes(self, df: pd.DataFrame, significant_points: List[Dict]) -> List[GannBox]:
        """Generate Gann boxes from significant points"""
        boxes = []
        
        # Generate boxes by pairing significant points
        for i, start_point in enumerate(significant_points[:-1]):
            for j, end_point in enumerate(significant_points[i+1:], i+1):
                
                # Check time distance constraints
                time_distance = end_point['index'] - start_point['index']
                if (time_distance < self.config.min_box_width_bars or 
                    time_distance > self.config.max_box_width_bars):
                    continue
                    
                # Extract price range between points
                start_idx = start_point['index']
                end_idx = end_point['index']
                
                box_data = df.iloc[start_idx:end_idx+1]
                box_high = box_data['high'].max()
                box_low = box_data['low'].min()
                box_height = box_high - box_low
                
                # Check minimum height constraint
                avg_atr = box_data['atr'].mean()
                if box_height < self.config.min_box_height_atr * avg_atr:
                    continue
                    
                # Create box
                box = GannBox(
                    id=f"box_{start_idx}_{end_idx}",
                    start_time=start_point['time'],
                    end_time=end_point['time'],
                    start_price=start_point['close_price'],
                    end_price=end_point['close_price'],
                    high_price=box_high,
                    low_price=box_low,
                    width_time=end_point['time'] - start_point['time'],
                    height_price=box_height,
                    volume_profile={},
                    pattern_type="unknown",
                    strength=self._calculate_box_strength(df, start_idx, end_idx),
                    breakout_direction=None
                )
                
                # Validate box quality
                if self._validate_box_quality(df, box, start_idx, end_idx):
                    boxes.append(box)
                    
                # Limit number of boxes to prevent excessive computation
                if len(boxes) >= 50:
                    break
                    
            if len(boxes) >= 50:
                break
                
        # Remove overlapping boxes (keep stronger ones)
        boxes = self._remove_overlapping_boxes(boxes)
        
        # Sort by strength
        boxes.sort(key=lambda x: x.strength, reverse=True)
        
        self.logger.info(f"Generated {len(boxes)} valid Gann boxes")
        return boxes[:20]  # Keep top 20 boxes
        
    def _calculate_box_strength(self, df: pd.DataFrame, start_idx: int, end_idx: int) -> float:
        """Calculate the strength/quality of a box"""
        box_data = df.iloc[start_idx:end_idx+1]
        
        strength_factors = []
        
        # 1. Price range significance
        box_range = box_data['high'].max() - box_data['low'].min()
        avg_range = df['price_range'].iloc[start_idx-20:start_idx].mean()
        range_significance = min(box_range / avg_range, 3.0) / 3.0
        strength_factors.append(range_significance * 0.3)
        
        # 2. Volume confirmation
        avg_volume = box_data['volume'].mean()
        historical_volume = df['volume'].iloc[start_idx-50:start_idx].mean()
        volume_factor = min(avg_volume / historical_volume, 2.0) / 2.0
        strength_factors.append(volume_factor * 0.2)
        
        # 3. Time duration appropriateness
        box_width = end_idx - start_idx
        optimal_width = (self.config.min_box_width_bars + self.config.max_box_width_bars) / 2
        width_factor = 1.0 - abs(box_width - optimal_width) / optimal_width
        strength_factors.append(max(width_factor, 0.0) * 0.15)
        
        # 4. Price action quality (consolidation vs trending)
        price_volatility = box_data['close'].std() / box_data['close'].mean()
        consolidation_factor = max(0.0, 1.0 - price_volatility * 10)
        strength_factors.append(consolidation_factor * 0.2)
        
        # 5. Breakout potential (momentum at boundaries)
        start_momentum = abs(df['momentum'].iloc[start_idx])
        end_momentum = abs(df['momentum'].iloc[end_idx])
        momentum_factor = (start_momentum + end_momentum) / 2
        strength_factors.append(min(momentum_factor, 1.0) * 0.15)
        
        total_strength = sum(strength_factors)
        return min(total_strength, 1.0)
        
    def _validate_box_quality(self, df: pd.DataFrame, box: GannBox, 
                             start_idx: int, end_idx: int) -> bool:
        """Validate if a box meets quality criteria"""
        
        # Check minimum strength
        if box.strength < 0.4:
            return False
            
        # Check for reasonable price action within box
        box_data = df.iloc[start_idx:end_idx+1]
        
        # Should have some price movement but not excessive
        price_range_ratio = box.height_price / box_data['close'].mean()
        if price_range_ratio < 0.01 or price_range_ratio > 0.5:
            return False
            
        # Check volume consistency
        volume_cv = box_data['volume'].std() / box_data['volume'].mean()
        if volume_cv > 2.0:  # Too much volume variation
            return False
            
        # Check for data quality
        if box_data['close'].isna().any() or box_data['volume'].isna().any():
            return False
            
        return True
        
    def _remove_overlapping_boxes(self, boxes: List[GannBox]) -> List[GannBox]:
        """Remove overlapping boxes, keeping stronger ones"""
        if len(boxes) <= 1:
            return boxes
            
        # Sort by strength (strongest first)
        sorted_boxes = sorted(boxes, key=lambda x: x.strength, reverse=True)
        
        filtered_boxes = []
        
        for box in sorted_boxes:
            # Check for overlap with existing boxes
            overlaps = False
            
            for existing_box in filtered_boxes:
                overlap_ratio = self._calculate_box_overlap(box, existing_box)
                if overlap_ratio > self.config.box_overlap_tolerance:
                    overlaps = True
                    break
                    
            if not overlaps:
                filtered_boxes.append(box)
                
        return filtered_boxes
        
    def _calculate_box_overlap(self, box1: GannBox, box2: GannBox) -> float:
        """Calculate overlap ratio between two boxes"""
        
        # Time overlap
        time_overlap = max(0, min(box1.end_time, box2.end_time) - max(box1.start_time, box2.start_time))
        time_union = max(box1.end_time, box2.end_time) - min(box1.start_time, box2.start_time)
        
        # Price overlap
        price_overlap = max(0, min(box1.high_price, box2.high_price) - max(box1.low_price, box2.low_price))
        price_union = max(box1.high_price, box2.high_price) - min(box1.low_price, box2.low_price)
        
        # Combined overlap ratio
        if time_union > 0 and price_union > 0:
            time_overlap_ratio = time_overlap / time_union
            price_overlap_ratio = price_overlap / price_union
            return (time_overlap_ratio + price_overlap_ratio) / 2
        else:
            return 0.0        
    def _analyze_box_patterns(self, df: pd.DataFrame, boxes: List[GannBox]) -> List[GannBox]:
        """Analyze patterns within each box"""
        enhanced_boxes = []
        
        for box in boxes:
            # Find box data indices
            start_idx = df[df['time_numeric'] >= box.start_time].index[0]
            end_idx = df[df['time_numeric'] <= box.end_time].index[-1]
            
            box_data = df.iloc[start_idx:end_idx+1]
            
            if len(box_data) < 5:  # Need minimum data for pattern analysis
                enhanced_boxes.append(box)
                continue
                
            # Analyze price action pattern
            pattern_type = self._classify_box_pattern(box_data, box)
            
            # Create enhanced box
            enhanced_box = box._replace(pattern_type=pattern_type)
            enhanced_boxes.append(enhanced_box)
            
        return enhanced_boxes
        
    def _classify_box_pattern(self, box_data: pd.DataFrame, box: GannBox) -> str:
        """Classify the pattern type within a box"""
        
        # Calculate pattern metrics
        price_trend = np.polyfit(range(len(box_data)), box_data['close'], 1)[0]
        volume_trend = np.polyfit(range(len(box_data)), box_data['volume'], 1)[0]
        
        # Price distribution analysis
        price_quartiles = box_data['close'].quantile([0.25, 0.5, 0.75])
        price_range = box.high_price - box.low_price
        
        # Volume distribution analysis
        volume_ratio = box_data['volume'].mean() / box_data['volume'].median()
        
        # Pattern classification logic
        if abs(price_trend) < price_range * 0.1:  # Minimal trend
            if volume_ratio > 1.5:
                return "accumulation"  # High volume, low price movement
            else:
                return "consolidation"  # Low volume, low price movement
                
        elif price_trend > 0:  # Upward trend
            if volume_trend > 0:
                return "markup"  # Rising price with rising volume
            else:
                return "weak_markup"  # Rising price with falling volume
                
        else:  # Downward trend
            if volume_trend > 0:
                return "distribution"  # Falling price with rising volume
            else:
                return "markdown"  # Falling price with falling volume
                
    def _calculate_volume_profiles(self, df: pd.DataFrame, boxes: List[GannBox]) -> List[GannBox]:
        """Calculate volume profiles for each box"""
        enhanced_boxes = []
        
        for box in boxes:
            # Find box data indices
            start_idx = df[df['time_numeric'] >= box.start_time].index[0]
            end_idx = df[df['time_numeric'] <= box.end_time].index[-1]
            
            box_data = df.iloc[start_idx:end_idx+1]
            
            # Create price bins for volume profile
            price_bins = np.linspace(box.low_price, box.high_price, 20)
            volume_profile = {}
            
            for i in range(len(price_bins) - 1):
                price_level = (price_bins[i] + price_bins[i+1]) / 2
                
                # Find bars within this price range
                mask = (
                    (box_data['low'] <= price_bins[i+1]) & 
                    (box_data['high'] >= price_bins[i])
                )
                
                # Calculate volume at this price level
                level_volume = box_data[mask]['volume'].sum()
                volume_profile[price_level] = level_volume
                
            # Find key volume levels
            max_volume_price = max(volume_profile.keys(), key=volume_profile.get)
            total_volume = sum(volume_profile.values())
            
            # Calculate value area (70% of volume)
            sorted_levels = sorted(volume_profile.items(), key=lambda x: x[1], reverse=True)
            cumulative_volume = 0
            value_area_levels = []
            
            for price, volume in sorted_levels:
                cumulative_volume += volume
                value_area_levels.append(price)
                if cumulative_volume >= total_volume * 0.7:
                    break
                    
            value_area_high = max(value_area_levels) if value_area_levels else box.high_price
            value_area_low = min(value_area_levels) if value_area_levels else box.low_price
            
            # Enhanced volume profile data
            enhanced_volume_profile = {
                'raw_profile': volume_profile,
                'poc': max_volume_price,  # Point of Control
                'value_area_high': value_area_high,
                'value_area_low': value_area_low,
                'total_volume': total_volume,
                'volume_distribution': self._analyze_volume_distribution(volume_profile)
            }
            
            # Create enhanced box
            enhanced_box = box._replace(volume_profile=enhanced_volume_profile)
            enhanced_boxes.append(enhanced_box)
            
        return enhanced_boxes
        
    def _analyze_volume_distribution(self, volume_profile: Dict) -> Dict:
        """Analyze volume distribution characteristics"""
        if not volume_profile:
            return {}
            
        volumes = list(volume_profile.values())
        prices = list(volume_profile.keys())
        
        # Calculate distribution statistics
        volume_mean = np.mean(volumes)
        volume_std = np.std(volumes)
        volume_skew = stats.skew(volumes) if len(volumes) > 2 else 0
        
        # Find volume concentration
        max_volume = max(volumes)
        concentration_ratio = max_volume / volume_mean if volume_mean > 0 else 0
        
        # Identify volume clusters
        volume_threshold = volume_mean + volume_std
        high_volume_levels = [p for p, v in volume_profile.items() if v > volume_threshold]
        
        return {
            'concentration_ratio': concentration_ratio,
            'skewness': volume_skew,
            'high_volume_levels': high_volume_levels,
            'distribution_type': 'concentrated' if concentration_ratio > 2.0 else 'distributed'
        }
        
    def _predict_breakouts(self, df: pd.DataFrame, boxes: List[GannBox]) -> List[GannBox]:
        """Predict breakout direction using machine learning"""
        if not self.config.ml_breakout_prediction:
            return boxes
            
        try:
            # Prepare training data from historical boxes
            if not self.is_trained and len(boxes) > 10:
                self._train_breakout_model(df, boxes)
                
            if self.is_trained:
                return self._apply_breakout_predictions(df, boxes)
            else:
                return boxes
                
        except Exception as e:
            self.logger.warning(f"Breakout prediction failed: {str(e)}")
            return boxes
            
    def _train_breakout_model(self, df: pd.DataFrame, boxes: List[GannBox]):
        """Train the breakout prediction model"""
        features = []
        targets = []
        
        for box in boxes:
            try:
                # Find box data
                start_idx = df[df['time_numeric'] >= box.start_time].index[0]
                end_idx = df[df['time_numeric'] <= box.end_time].index[-1]
                
                # Need data after box for labeling
                if end_idx + 20 >= len(df):
                    continue
                    
                box_data = df.iloc[start_idx:end_idx+1]
                post_box_data = df.iloc[end_idx+1:end_idx+21]
                
                # Extract features
                box_features = self._extract_box_features(box_data, box)
                features.append(box_features)
                
                # Determine breakout direction (target)
                breakout_direction = self._determine_breakout_direction(box, post_box_data)
                targets.append(breakout_direction)
                
            except Exception:
                continue
                
        if len(features) < 10:
            return
            
        # Prepare data for training
        X = np.array(features)
        y = np.array(targets)
        
        # Remove invalid samples
        valid_mask = ~np.isnan(X).any(axis=1)
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) < 5:
            return
            
        # Scale features
        X = self.scaler.fit_transform(X)
        
        # Train model
        self.breakout_model.fit(X, y)
        self.is_trained = True
        
        self.logger.info(f"Trained breakout model with {len(X)} samples")
        
    def _extract_box_features(self, box_data: pd.DataFrame, box: GannBox) -> List[float]:
        """Extract features from box data for ML model"""
        features = []
        
        # Box geometry features
        aspect_ratio = box.width_time / box.height_price if box.height_price > 0 else 0
        features.extend([
            box.height_price,
            box.width_time,
            aspect_ratio,
            box.strength
        ])
        
        # Price action features
        price_trend = np.polyfit(range(len(box_data)), box_data['close'], 1)[0]
        price_volatility = box_data['close'].std()
        price_range_ratio = (box_data['high'].max() - box_data['low'].min()) / box_data['close'].mean()
        
        features.extend([
            price_trend,
            price_volatility,
            price_range_ratio
        ])
        
        # Volume features
        volume_trend = np.polyfit(range(len(box_data)), box_data['volume'], 1)[0]
        volume_ratio = box_data['volume'].mean() / box_data['volume'].median()
        
        features.extend([
            volume_trend,
            volume_ratio,
            box_data['volume'].std()
        ])
        
        # Technical indicators
        rsi_avg = box_data['rsi'].mean()
        momentum_avg = box_data['momentum'].mean()
        
        features.extend([
            rsi_avg,
            momentum_avg
        ])
        
        # Pattern type encoding
        pattern_encoding = {
            'accumulation': 1, 'distribution': 2, 'consolidation': 3,
            'markup': 4, 'markdown': 5, 'weak_markup': 6, 'unknown': 0
        }
        features.append(pattern_encoding.get(box.pattern_type, 0))
        
        return features
        
    def _determine_breakout_direction(self, box: GannBox, post_data: pd.DataFrame) -> int:
        """Determine actual breakout direction from post-box data"""
        if len(post_data) == 0:
            return 0  # Neutral
            
        # Check if price broke above or below box
        max_price_after = post_data['high'].max()
        min_price_after = post_data['low'].min()
        
        box_height = box.high_price - box.low_price
        breakout_threshold = box_height * 0.1  # 10% of box height
        
        upward_breakout = max_price_after > (box.high_price + breakout_threshold)
        downward_breakout = min_price_after < (box.low_price - breakout_threshold)
        
        if upward_breakout and not downward_breakout:
            return 1  # Bullish breakout
        elif downward_breakout and not upward_breakout:
            return -1  # Bearish breakout
        else:
            return 0  # No clear breakout or both directions
            
    def _apply_breakout_predictions(self, df: pd.DataFrame, boxes: List[GannBox]) -> List[GannBox]:
        """Apply breakout predictions to current boxes"""
        enhanced_boxes = []
        
        for box in boxes:
            try:
                # Find box data
                start_idx = df[df['time_numeric'] >= box.start_time].index[0]
                end_idx = df[df['time_numeric'] <= box.end_time].index[-1]
                box_data = df.iloc[start_idx:end_idx+1]
                
                # Extract features
                features = self._extract_box_features(box_data, box)
                features_scaled = self.scaler.transform([features])
                
                # Predict breakout direction
                prediction = self.breakout_model.predict(features_scaled)[0]
                prediction_proba = self.breakout_model.predict_proba(features_scaled)[0]
                
                # Map prediction to direction
                direction_map = {-1: 'bearish', 0: 'neutral', 1: 'bullish'}
                breakout_direction = direction_map.get(prediction, 'neutral')
                
                # Add confidence score
                confidence = max(prediction_proba)
                
                # Create enhanced box with prediction
                enhanced_box = box._replace(breakout_direction=breakout_direction)
                enhanced_boxes.append(enhanced_box)
                
            except Exception as e:
                self.logger.warning(f"Failed to predict breakout for box {box.id}: {str(e)}")
                enhanced_boxes.append(box)
                
        return enhanced_boxes
        
    def _perform_harmonic_analysis(self, boxes: List[GannBox]) -> Dict:
        """Perform harmonic analysis between multiple boxes"""
        harmonic_data = {
            'harmonic_relationships': [],
            'geometric_patterns': [],
            'fibonacci_alignments': [],
            'time_relationships': [],
            'price_relationships': []
        }
        
        if len(boxes) < 2:
            return harmonic_data
            
        # Fibonacci ratios for harmonic analysis
        fib_ratios = [0.236, 0.382, 0.618, 0.764, 1.0, 1.618, 2.618]
        
        # Analyze relationships between box pairs
        for i, box1 in enumerate(boxes[:-1]):
            for j, box2 in enumerate(boxes[i+1:], i+1):
                
                # Time relationships
                time_ratio = box2.width_time / box1.width_time if box1.width_time > 0 else 0
                
                # Price relationships
                price_ratio = box2.height_price / box1.height_price if box1.height_price > 0 else 0
                
                # Check for Fibonacci relationships
                for fib_ratio in fib_ratios:
                    # Time Fibonacci
                    if abs(time_ratio - fib_ratio) < 0.05:
                        harmonic_data['time_relationships'].append({
                            'box1_id': box1.id,
                            'box2_id': box2.id,
                            'ratio': time_ratio,
                            'fibonacci_ratio': fib_ratio,
                            'type': 'time',
                            'strength': (box1.strength + box2.strength) / 2
                        })
                        
                    # Price Fibonacci
                    if abs(price_ratio - fib_ratio) < 0.05:
                        harmonic_data['price_relationships'].append({
                            'box1_id': box1.id,
                            'box2_id': box2.id,
                            'ratio': price_ratio,
                            'fibonacci_ratio': fib_ratio,
                            'type': 'price',
                            'strength': (box1.strength + box2.strength) / 2
                        })
                        
                # Geometric patterns (squares, rectangles, etc.)
                geometric_similarity = self._calculate_geometric_similarity(box1, box2)
                if geometric_similarity > 0.7:
                    harmonic_data['geometric_patterns'].append({
                        'box1_id': box1.id,
                        'box2_id': box2.id,
                        'similarity': geometric_similarity,
                        'pattern_type': self._classify_geometric_pattern(box1, box2)
                    })
                    
        return harmonic_data
        
    def _calculate_geometric_similarity(self, box1: GannBox, box2: GannBox) -> float:
        """Calculate geometric similarity between two boxes"""
        
        # Aspect ratio similarity
        aspect1 = box1.width_time / box1.height_price if box1.height_price > 0 else 0
        aspect2 = box2.width_time / box2.height_price if box2.height_price > 0 else 0
        
        if aspect1 > 0 and aspect2 > 0:
            aspect_similarity = 1.0 - abs(aspect1 - aspect2) / max(aspect1, aspect2)
        else:
            aspect_similarity = 0.0
            
        # Size similarity
        size1 = box1.width_time * box1.height_price
        size2 = box2.width_time * box2.height_price
        
        if size1 > 0 and size2 > 0:
            size_similarity = min(size1, size2) / max(size1, size2)
        else:
            size_similarity = 0.0
            
        # Combined similarity
        return (aspect_similarity + size_similarity) / 2
        
    def _classify_geometric_pattern(self, box1: GannBox, box2: GannBox) -> str:
        """Classify the geometric pattern between two boxes"""
        
        # Calculate aspect ratios
        aspect1 = box1.width_time / box1.height_price if box1.height_price > 0 else 0
        aspect2 = box2.width_time / box2.height_price if box2.height_price > 0 else 0
        
        avg_aspect = (aspect1 + aspect2) / 2
        
        if abs(avg_aspect - 1.0) < 0.2:
            return "square_pattern"
        elif avg_aspect > 2.0:
            return "horizontal_rectangle"
        elif avg_aspect < 0.5:
            return "vertical_rectangle"
        else:
            return "rectangle_pattern"
            
    def _extract_support_resistance(self, boxes: List[GannBox]) -> Dict:
        """Extract support and resistance levels from box boundaries"""
        
        sr_levels = {
            'support_levels': [],
            'resistance_levels': [],
            'key_levels': [],
            'level_clusters': []
        }
        
        # Collect all box boundaries
        all_levels = []
        
        for box in boxes:
            # Box boundaries
            all_levels.extend([
                {'level': box.high_price, 'type': 'resistance', 'strength': box.strength, 'source': box.id},
                {'level': box.low_price, 'type': 'support', 'strength': box.strength, 'source': box.id}
            ])
            
            # Volume POC levels if available
            if box.volume_profile and 'poc' in box.volume_profile:
                poc_level = box.volume_profile['poc']
                all_levels.append({
                    'level': poc_level, 
                    'type': 'poc', 
                    'strength': box.strength * 1.2,  # Boost POC importance
                    'source': f"{box.id}_poc"
                })
                
            # Value area boundaries
            if box.volume_profile and 'value_area_high' in box.volume_profile:
                va_high = box.volume_profile['value_area_high']
                va_low = box.volume_profile['value_area_low']
                
                all_levels.extend([
                    {'level': va_high, 'type': 'resistance', 'strength': box.strength * 0.8, 'source': f"{box.id}_va_high"},
                    {'level': va_low, 'type': 'support', 'strength': box.strength * 0.8, 'source': f"{box.id}_va_low"}
                ])
                
        # Cluster similar levels
        if all_levels:
            level_clusters = self._cluster_similar_levels(all_levels)
            sr_levels['level_clusters'] = level_clusters
            
            # Separate into support and resistance
            for cluster in level_clusters:
                avg_level = cluster['center_level']
                total_strength = cluster['total_strength']
                
                if cluster['dominant_type'] == 'support':
                    sr_levels['support_levels'].append({
                        'level': avg_level,
                        'strength': total_strength,
                        'count': cluster['level_count'],
                        'sources': cluster['sources']
                    })
                elif cluster['dominant_type'] == 'resistance':
                    sr_levels['resistance_levels'].append({
                        'level': avg_level,
                        'strength': total_strength,
                        'count': cluster['level_count'],
                        'sources': cluster['sources']
                    })
                    
            # Identify key levels (highest strength)
            all_clusters = level_clusters.copy()
            all_clusters.sort(key=lambda x: x['total_strength'], reverse=True)
            sr_levels['key_levels'] = all_clusters[:10]
            
        return sr_levels
        
    def _cluster_similar_levels(self, levels: List[Dict]) -> List[Dict]:
        """Cluster similar price levels together"""
        if not levels:
            return []
            
        # Sort levels by price
        sorted_levels = sorted(levels, key=lambda x: x['level'])
        
        clusters = []
        
        # Simple clustering based on price proximity
        current_cluster = [sorted_levels[0]]
        
        for level in sorted_levels[1:]:
            # Check if level is close to current cluster
            cluster_center = np.mean([l['level'] for l in current_cluster])
            cluster_range = max([l['level'] for l in current_cluster]) - min([l['level'] for l in current_cluster])
            
            # Adaptive tolerance based on cluster size
            tolerance = max(cluster_range * 0.1, cluster_center * 0.001)  # 0.1% of price or 10% of cluster range
            
            if abs(level['level'] - cluster_center) <= tolerance:
                current_cluster.append(level)
            else:
                # Finalize current cluster
                if len(current_cluster) > 0:
                    clusters.append(self._create_level_cluster(current_cluster))
                current_cluster = [level]
                
        # Add final cluster
        if len(current_cluster) > 0:
            clusters.append(self._create_level_cluster(current_cluster))
            
        return clusters
        
    def _create_level_cluster(self, levels: List[Dict]) -> Dict:
        """Create a level cluster from a group of similar levels"""
        
        # Calculate cluster statistics
        price_levels = [l['level'] for l in levels]
        strengths = [l['strength'] for l in levels]
        types = [l['type'] for l in levels]
        sources = [l['source'] for l in levels]
        
        center_level = np.mean(price_levels)
        total_strength = sum(strengths)
        level_count = len(levels)
        
        # Determine dominant type
        type_counts = {}
        for t in types:
            type_counts[t] = type_counts.get(t, 0) + 1
            
        dominant_type = max(type_counts.keys(), key=type_counts.get)
        
        return {
            'center_level': center_level,
            'total_strength': total_strength,
            'level_count': level_count,
            'dominant_type': dominant_type,
            'sources': sources,
            'price_range': max(price_levels) - min(price_levels)
        }        
    def _generate_trading_signals(self, df: pd.DataFrame, boxes: List[GannBox], 
                                sr_levels: Dict) -> pd.Series:
        """Generate trading signals based on box analysis"""
        
        signals = pd.Series(0, index=df.index)  # 0 = neutral, 1 = buy, -1 = sell
        
        if not boxes:
            return signals
            
        current_price = df['close'].iloc[-1]
        current_time = df['time_numeric'].iloc[-1]
        
        # Find active boxes (boxes that are recent or contain current price)
        active_boxes = []
        for box in boxes:
            # Check if current price is within box range
            if (box.low_price <= current_price <= box.high_price and 
                box.start_time <= current_time <= box.end_time):
                active_boxes.append(box)
            # Check if box is recent (within last 20% of data)
            elif box.end_time > current_time - (current_time - df['time_numeric'].iloc[0]) * 0.2:
                active_boxes.append(box)
                
        # Signal generation logic
        signal_strength = 0.0
        signal_count = 0
        
        for box in active_boxes:
            box_signal = self._generate_box_signal(df, box, current_price, sr_levels)
            if box_signal != 0:
                signal_strength += box_signal * box.strength
                signal_count += 1
                
        # Combine signals with confidence weighting
        if signal_count > 0:
            avg_signal = signal_strength / signal_count
            
            # Apply signal threshold
            if avg_signal > 0.3:
                signals.iloc[-1] = 1  # Buy
            elif avg_signal < -0.3:
                signals.iloc[-1] = -1  # Sell
            else:
                signals.iloc[-1] = 0  # Neutral
                
        return signals
        
    def _generate_box_signal(self, df: pd.DataFrame, box: GannBox, 
                           current_price: float, sr_levels: Dict) -> float:
        """Generate signal for individual box"""
        
        signal = 0.0
        
        # 1. Breakout signals
        if box.breakout_direction:
            if box.breakout_direction == 'bullish':
                # Check if price is breaking above box
                if current_price > box.high_price:
                    signal += 0.8
                # Check if price is approaching upper boundary
                elif current_price > (box.high_price - box.height_price * 0.1):
                    signal += 0.4
                    
            elif box.breakout_direction == 'bearish':
                # Check if price is breaking below box
                if current_price < box.low_price:
                    signal -= 0.8
                # Check if price is approaching lower boundary
                elif current_price < (box.low_price + box.height_price * 0.1):
                    signal -= 0.4
                    
        # 2. Pattern-based signals
        if box.pattern_type == 'accumulation':
            # Bullish signal for accumulation patterns
            if current_price > (box.low_price + box.high_price) / 2:
                signal += 0.6
                
        elif box.pattern_type == 'distribution':
            # Bearish signal for distribution patterns
            if current_price < (box.low_price + box.high_price) / 2:
                signal -= 0.6
                
        elif box.pattern_type == 'markup':
            # Continuation bullish signal
            signal += 0.4
            
        elif box.pattern_type == 'markdown':
            # Continuation bearish signal
            signal -= 0.4
            
        # 3. Volume profile signals
        if box.volume_profile and 'poc' in box.volume_profile:
            poc_level = box.volume_profile['poc']
            
            # POC as support/resistance
            distance_to_poc = abs(current_price - poc_level) / box.height_price
            
            if distance_to_poc < 0.05:  # Very close to POC
                if current_price > poc_level:
                    signal += 0.3  # POC acting as support
                else:
                    signal -= 0.3  # POC acting as resistance
                    
        # 4. Support/resistance level signals
        for level_cluster in sr_levels.get('level_clusters', []):
            level = level_cluster['center_level']
            strength = level_cluster['total_strength']
            
            distance_to_level = abs(current_price - level) / df['atr'].iloc[-1]
            
            if distance_to_level < 0.5:  # Close to S/R level
                if level_cluster['dominant_type'] == 'support' and current_price > level:
                    signal += 0.2 * strength  # Bounce off support
                elif level_cluster['dominant_type'] == 'resistance' and current_price < level:
                    signal -= 0.2 * strength  # Rejection at resistance
                    
        # Normalize signal to [-1, 1] range
        return max(-1.0, min(1.0, signal))
        
    def _combine_results(self, df: pd.DataFrame, boxes: List[GannBox], 
                        sr_levels: Dict, harmonic_data: Dict, signals: pd.Series) -> pd.DataFrame:
        """Combine all analysis results into final output DataFrame"""
        
        result = df.copy()
        
        # Add box-based indicators
        current_time = df['time_numeric'].iloc[-1]
        current_price = df['close'].iloc[-1]
        
        # Find current active box
        active_box = None
        for box in boxes:
            if (box.low_price <= current_price <= box.high_price and 
                box.start_time <= current_time <= box.end_time):
                active_box = box
                break
                
        # Box position indicators
        if active_box:
            box_position = (current_price - active_box.low_price) / active_box.height_price
            result['gann_box_position'] = box_position
            result['gann_box_strength'] = active_box.strength
            result['gann_box_pattern'] = active_box.pattern_type
            result['gann_box_breakout_direction'] = active_box.breakout_direction
            
            # Distance to box boundaries
            result['gann_distance_to_top'] = (active_box.high_price - current_price) / df['atr'].iloc[-1]
            result['gann_distance_to_bottom'] = (current_price - active_box.low_price) / df['atr'].iloc[-1]
        else:
            result['gann_box_position'] = np.nan
            result['gann_box_strength'] = 0.0
            result['gann_box_pattern'] = 'none'
            result['gann_box_breakout_direction'] = 'none'
            result['gann_distance_to_top'] = np.nan
            result['gann_distance_to_bottom'] = np.nan
            
        # Support/resistance levels
        key_levels = sr_levels.get('key_levels', [])
        if key_levels:
            closest_level = min(key_levels, key=lambda x: abs(x['center_level'] - current_price))
            result['gann_closest_sr_level'] = closest_level['center_level']
            result['gann_closest_sr_strength'] = closest_level['total_strength']
            result['gann_closest_sr_type'] = closest_level['dominant_type']
            result['gann_distance_to_sr'] = abs(current_price - closest_level['center_level']) / df['atr'].iloc[-1]
        else:
            result['gann_closest_sr_level'] = np.nan
            result['gann_closest_sr_strength'] = 0.0
            result['gann_closest_sr_type'] = 'none'
            result['gann_distance_to_sr'] = np.nan
            
        # Harmonic analysis results
        result['gann_harmonic_strength'] = len(harmonic_data.get('harmonic_relationships', []))
        result['gann_geometric_patterns'] = len(harmonic_data.get('geometric_patterns', []))
        result['gann_fibonacci_alignments'] = len(harmonic_data.get('fibonacci_alignments', []))
        
        # Box statistics
        result['gann_total_boxes'] = len(boxes)
        result['gann_strong_boxes'] = len([b for b in boxes if b.strength > 0.7])
        
        if boxes:
            avg_box_strength = np.mean([b.strength for b in boxes])
            result['gann_avg_box_strength'] = avg_box_strength
        else:
            result['gann_avg_box_strength'] = 0.0
            
        # Trading signals
        result['gann_box_signal'] = signals
        
        # Additional derived indicators
        result['gann_market_structure'] = self._classify_market_structure(boxes, current_price)
        result['gann_consolidation_score'] = self._calculate_consolidation_score(boxes, current_price)
        
        return result
        
    def _classify_market_structure(self, boxes: List[GannBox], current_price: float) -> str:
        """Classify current market structure based on box analysis"""
        
        if not boxes:
            return 'undefined'
            
        # Analyze box patterns
        pattern_counts = {}
        for box in boxes[-5:]:  # Last 5 boxes
            pattern = box.pattern_type
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            
        dominant_pattern = max(pattern_counts.keys(), key=pattern_counts.get) if pattern_counts else 'unknown'
        
        # Analyze price position relative to boxes
        above_boxes = 0
        below_boxes = 0
        within_boxes = 0
        
        for box in boxes[-10:]:  # Last 10 boxes
            if current_price > box.high_price:
                above_boxes += 1
            elif current_price < box.low_price:
                below_boxes += 1
            else:
                within_boxes += 1
                
        # Classification logic
        if dominant_pattern == 'accumulation' and above_boxes > below_boxes:
            return 'bullish_accumulation'
        elif dominant_pattern == 'distribution' and below_boxes > above_boxes:
            return 'bearish_distribution'
        elif within_boxes > (above_boxes + below_boxes):
            return 'consolidation'
        elif dominant_pattern in ['markup', 'weak_markup']:
            return 'bullish_trend'
        elif dominant_pattern == 'markdown':
            return 'bearish_trend'
        else:
            return 'transitional'
            
    def _calculate_consolidation_score(self, boxes: List[GannBox], current_price: float) -> float:
        """Calculate consolidation score based on box overlap and price containment"""
        
        if not boxes:
            return 0.0
            
        # Find overlapping boxes
        overlapping_boxes = 0
        total_pairs = 0
        
        for i, box1 in enumerate(boxes[-10:]):
            for j, box2 in enumerate(boxes[-10:][i+1:], i+1):
                total_pairs += 1
                overlap_ratio = self._calculate_box_overlap(box1, box2)
                if overlap_ratio > 0.3:  # Significant overlap
                    overlapping_boxes += 1
                    
        overlap_score = overlapping_boxes / total_pairs if total_pairs > 0 else 0.0
        
        # Check price containment within recent boxes
        contained_count = 0
        recent_boxes = boxes[-5:]
        
        for box in recent_boxes:
            if box.low_price <= current_price <= box.high_price:
                contained_count += 1
                
        containment_score = contained_count / len(recent_boxes) if recent_boxes else 0.0
        
        # Combined consolidation score
        consolidation_score = (overlap_score + containment_score) / 2
        return min(consolidation_score, 1.0)


def create_gann_box_indicator(config: Optional[GannBoxConfig] = None) -> GannBoxIndicator:
    """Factory function to create GannBoxIndicator instance"""
    return GannBoxIndicator(config)


# Example usage and testing
if __name__ == "__main__":
    import yfinance as yf
    
    # Test with sample data
    ticker = "EURUSD=X"
    data = yf.download(ticker, period="6mo", interval="1h")
    data.reset_index(inplace=True)
    data.columns = data.columns.str.lower()
    data['timestamp'] = data['datetime']
    
    # Create indicator
    config = GannBoxConfig(
        min_box_height_atr=1.5,
        min_box_width_bars=8,
        pattern_recognition=True,
        ml_breakout_prediction=True,
        harmonic_analysis=True
    )
    
    indicator = GannBoxIndicator(config)
    
    try:
        # Calculate Gann boxes
        result = indicator.calculate(data)
        
        print("Gann Box Calculation Results:")
        print(f"Data shape: {result.shape}")
        print(f"Columns: {list(result.columns)}")
        
        # Display recent signals
        recent = result.tail(5)
        for col in ['gann_box_position', 'gann_box_pattern', 'gann_box_signal', 'gann_market_structure']:
            if col in recent.columns:
                print(f"\n{col}:")
                print(recent[col].to_string())
                
        # Display box statistics
        total_boxes = recent['gann_total_boxes'].iloc[-1]
        strong_boxes = recent['gann_strong_boxes'].iloc[-1]
        avg_strength = recent['gann_avg_box_strength'].iloc[-1]
        
        print(f"\nBox Statistics:")
        print(f"Total boxes: {total_boxes}")
        print(f"Strong boxes: {strong_boxes}")
        print(f"Average box strength: {avg_strength:.3f}")
                
    except Exception as e:
        print(f"Error testing Gann Box indicator: {e}")
        import traceback
        traceback.print_exc()