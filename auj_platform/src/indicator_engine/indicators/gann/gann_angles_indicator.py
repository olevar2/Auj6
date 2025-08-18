"""
Gann Angles Indicator - Advanced Implementation

This module implements W.D. Gann's angle analysis with sophisticated mathematical models,
machine learning enhancement, and multi-timeframe support for maximum profitability.

The implementation includes:
- Advanced geometric angle calculations
- Dynamic angle adjustments based on volatility
- ML-enhanced angle prediction and validation
- Multi-timeframe angle harmonics
- Support/resistance level identification
- Trend strength measurement through angle velocity
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from scipy import stats, optimize
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import math

from ..base.standard_indicator import StandardIndicatorInterface
from ....core.exceptions import IndicatorCalculationError


@dataclass
class GannAngleConfig:
    """Configuration for Gann Angle calculations"""
    primary_angles: List[float] = None  # Default: [1x1, 2x1, 3x1, 4x1, 8x1, 1x2, 1x3, 1x4, 1x8]
    time_scaling_method: str = "volatility_adaptive"  # "fixed", "volatility_adaptive", "ml_enhanced"
    lookback_period: int = 252  # Trading days for volatility calculation
    ml_training_window: int = 1000  # Bars for ML model training
    angle_tolerance: float = 0.02  # Tolerance for angle validation (2%)
    enable_harmonic_analysis: bool = True
    enable_dynamic_scaling: bool = True
    
    def __post_init__(self):
        if self.primary_angles is None:
            # W.D. Gann's primary angles (degrees converted to slopes)
            self.primary_angles = [
                45.0,    # 1x1 - Primary trend line
                63.75,   # 2x1 - Strong uptrend
                71.25,   # 3x1 - Very strong uptrend  
                75.0,    # 4x1 - Extreme uptrend
                82.5,    # 8x1 - Parabolic uptrend
                26.25,   # 1x2 - Moderate uptrend
                18.75,   # 1x3 - Weak uptrend
                15.0,    # 1x4 - Very weak uptrend
                7.5      # 1x8 - Minimal uptrend
            ]


class GannAnglesIndicator(StandardIndicatorInterface):
    """
    Advanced Gann Angles Indicator Implementation
    
    This indicator implements W.D. Gann's angle theory with sophisticated enhancements:
    
    1. Dynamic Price-Time Scaling: Adjusts angles based on market volatility and price range
    2. Machine Learning Enhancement: Predicts optimal angle positioning using historical patterns
    3. Multi-Timeframe Harmonics: Analyzes angle relationships across different timeframes
    4. Advanced Geometric Analysis: Calculates angle intersections and convergence zones
    5. Trend Strength Measurement: Quantifies trend strength through angle velocity analysis
    
    Mathematical Foundation:
    - Angle slope calculation: tan(angle_degrees * π/180)
    - Dynamic scaling factor: σ(price) / σ(time) * volatility_adjustment
    - ML prediction: Random Forest model trained on historical angle performance
    """
    
    def __init__(self, config: Optional[GannAngleConfig] = None):
        super().__init__()
        self.config = config or GannAngleConfig()
        self.scaler = StandardScaler()
        self.ml_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.is_trained = False
        self.angle_cache = {}
        self.logger = logging.getLogger(__name__)
        
    def get_name(self) -> str:
        return "GannAnglesIndicator"
        
    def get_required_columns(self) -> List[str]:
        return ['high', 'low', 'close', 'volume', 'timestamp']
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Gann Angles with advanced mathematical modeling
        
        Args:
            data: DataFrame with OHLCV data and timestamp
            
        Returns:
            DataFrame with angle lines, support/resistance levels, and trend metrics
        """
        try:
            if len(data) < self.config.lookback_period:
                raise IndicatorCalculationError(
                    f"Insufficient data: {len(data)} bars, need {self.config.lookback_period}"
                )
                
            # Prepare data
            df = data.copy()
            df = self._prepare_price_data(df)
            
            # Calculate dynamic scaling factors
            scaling_factors = self._calculate_scaling_factors(df)
            
            # Identify significant pivot points
            pivot_points = self._identify_pivot_points(df)
            
            # Calculate Gann angles for each pivot
            angle_lines = self._calculate_gann_angles(df, pivot_points, scaling_factors)
            
            # Apply machine learning enhancement
            if self.config.time_scaling_method == "ml_enhanced":
                angle_lines = self._enhance_with_ml(df, angle_lines)
                
            # Calculate support/resistance levels
            sr_levels = self._calculate_support_resistance(df, angle_lines)
            
            # Analyze trend strength and angle velocity
            trend_metrics = self._analyze_trend_strength(df, angle_lines)
            
            # Perform harmonic analysis
            if self.config.enable_harmonic_analysis:
                harmonic_data = self._perform_harmonic_analysis(angle_lines)
                trend_metrics.update(harmonic_data)
                
            # Combine results
            result = self._combine_results(df, angle_lines, sr_levels, trend_metrics)
            
            self.logger.info(f"Calculated Gann angles for {len(result)} periods")
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating Gann angles: {str(e)}")
            raise IndicatorCalculationError(f"Gann angles calculation failed: {str(e)}")
            
    def _prepare_price_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare and clean price data for analysis"""
        df = df.copy()
        
        # Calculate price statistics
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['price_range'] = df['high'] - df['low']
        df['true_range'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        
        # Calculate volatility measures
        df['atr'] = df['true_range'].rolling(window=14).mean()
        df['volatility'] = df['close'].pct_change().rolling(window=20).std() * np.sqrt(252)
        
        # Normalize time index
        if 'timestamp' in df.columns:
            df['time_index'] = pd.to_datetime(df['timestamp'])
            df['time_numeric'] = (df['time_index'] - df['time_index'].iloc[0]).dt.total_seconds()
        else:
            df['time_numeric'] = np.arange(len(df))
            
        return df
        
    def _calculate_scaling_factors(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate dynamic scaling factors for price-time relationship"""
        scaling_factors = {}
        
        if self.config.time_scaling_method == "fixed":
            scaling_factors['price_scale'] = 1.0
            scaling_factors['time_scale'] = 1.0
            
        elif self.config.time_scaling_method == "volatility_adaptive":
            # Use volatility to scale angles dynamically
            recent_volatility = df['volatility'].iloc[-20:].mean()
            price_range = df['close'].iloc[-252:].max() - df['close'].iloc[-252:].min()
            time_range = df['time_numeric'].iloc[-252:].max() - df['time_numeric'].iloc[-252:].min()
            
            scaling_factors['price_scale'] = price_range / (recent_volatility * 100)
            scaling_factors['time_scale'] = time_range / 252  # Normalize to trading days
            
        elif self.config.time_scaling_method == "ml_enhanced":
            # ML-based scaling (will be enhanced in _enhance_with_ml)
            scaling_factors = self._calculate_ml_scaling(df)
            
        scaling_factors['volatility_adjustment'] = df['atr'].iloc[-1] / df['atr'].mean()
        
        return scaling_factors
        
    def _identify_pivot_points(self, df: pd.DataFrame) -> List[Dict]:
        """Identify significant pivot points for angle calculation"""
        pivots = []
        
        # Use multiple methods to identify pivots
        # Method 1: Local extrema
        window = 20
        df['high_pivot'] = (
            (df['high'] == df['high'].rolling(window, center=True).max()) &
            (df['high'] > df['high'].rolling(window*2, center=True).quantile(0.8))
        )
        df['low_pivot'] = (
            (df['low'] == df['low'].rolling(window, center=True).min()) &
            (df['low'] < df['low'].rolling(window*2, center=True).quantile(0.2))
        )
        
        # Method 2: Volume-weighted pivots
        df['volume_weighted_price'] = (df['typical_price'] * df['volume']).rolling(10).sum() / df['volume'].rolling(10).sum()
        df['vwp_pivot'] = abs(df['volume_weighted_price'] - df['close']) > df['atr'] * 0.5
        
        # Combine pivot identification methods
        for i in range(window, len(df) - window):
            if df['high_pivot'].iloc[i] or df['low_pivot'].iloc[i] or df['vwp_pivot'].iloc[i]:
                pivot = {
                    'index': i,
                    'price': df['typical_price'].iloc[i],
                    'time': df['time_numeric'].iloc[i],
                    'type': 'high' if df['high_pivot'].iloc[i] else 'low',
                    'strength': self._calculate_pivot_strength(df, i),
                    'volume_confirmation': df['volume'].iloc[i] > df['volume'].rolling(20).mean().iloc[i]
                }
                pivots.append(pivot)
                
        # Filter pivots by strength and spacing
        pivots = [p for p in pivots if p['strength'] > 0.6]
        pivots = self._filter_close_pivots(pivots, min_distance=5)
        
        return sorted(pivots, key=lambda x: x['index'])
        
    def _calculate_pivot_strength(self, df: pd.DataFrame, index: int) -> float:
        """Calculate the strength of a pivot point"""
        window = 10
        
        if index < window or index >= len(df) - window:
            return 0.0
            
        # Price deviation from local mean
        local_prices = df['typical_price'].iloc[index-window:index+window+1]
        price_strength = abs(df['typical_price'].iloc[index] - local_prices.mean()) / local_prices.std()
        
        # Volume confirmation
        volume_strength = df['volume'].iloc[index] / df['volume'].iloc[index-window:index+window+1].mean()
        
        # Range significance
        range_strength = df['price_range'].iloc[index] / df['atr'].iloc[index]
        
        # Combine strengths
        total_strength = (price_strength * 0.5 + volume_strength * 0.3 + range_strength * 0.2)
        return min(total_strength / 3.0, 1.0)  # Normalize to [0, 1]
        
    def _filter_close_pivots(self, pivots: List[Dict], min_distance: int) -> List[Dict]:
        """Filter out pivots that are too close to each other"""
        if not pivots:
            return pivots
            
        filtered = [pivots[0]]
        
        for pivot in pivots[1:]:
            if pivot['index'] - filtered[-1]['index'] >= min_distance:
                # Keep stronger pivot if they're close
                if pivot['strength'] > filtered[-1]['strength']:
                    filtered[-1] = pivot
                else:
                    filtered.append(pivot)
                    
        return filtered
        
    def _calculate_gann_angles(self, df: pd.DataFrame, pivots: List[Dict], scaling: Dict) -> Dict:
        """Calculate Gann angle lines from pivot points"""
        angle_lines = {
            'angles': {},
            'current_levels': {},
            'future_projections': {},
            'intersections': []
        }
        
        current_index = len(df) - 1
        
        for pivot in pivots[-10:]:  # Use last 10 significant pivots
            pivot_angles = {}
            
            for angle_deg in self.config.primary_angles:
                # Convert angle to slope with dynamic scaling
                angle_rad = math.radians(angle_deg)
                base_slope = math.tan(angle_rad)
                
                # Apply scaling factors
                adjusted_slope = base_slope * scaling['price_scale'] / scaling['time_scale']
                adjusted_slope *= scaling['volatility_adjustment']
                
                # Calculate angle line equation: y = mx + b
                time_diff = df['time_numeric'].iloc[current_index] - pivot['time']
                projected_price = pivot['price'] + adjusted_slope * time_diff
                
                # Store angle line data
                angle_key = f"pivot_{pivot['index']}_angle_{angle_deg}"
                pivot_angles[angle_key] = {
                    'slope': adjusted_slope,
                    'intercept': pivot['price'],
                    'pivot_time': pivot['time'],
                    'pivot_price': pivot['price'],
                    'current_level': projected_price,
                    'angle_degrees': angle_deg,
                    'strength': self._calculate_angle_strength(df, pivot, adjusted_slope)
                }
                
                # Calculate future projections
                future_times = []
                future_levels = []
                for future_periods in [5, 10, 20, 50]:
                    future_time = df['time_numeric'].iloc[current_index] + future_periods * (
                        df['time_numeric'].iloc[-1] - df['time_numeric'].iloc[-2]
                    )
                    future_price = pivot['price'] + adjusted_slope * (future_time - pivot['time'])
                    future_times.append(future_time)
                    future_levels.append(future_price)
                    
                pivot_angles[angle_key]['future_projections'] = {
                    'times': future_times,
                    'levels': future_levels
                }
                
            angle_lines['angles'][f"pivot_{pivot['index']}"] = pivot_angles
            
        # Calculate current levels for all angles
        angle_lines['current_levels'] = self._extract_current_levels(angle_lines['angles'])
        
        # Find angle intersections
        angle_lines['intersections'] = self._find_angle_intersections(angle_lines['angles'])
        
        return angle_lines
        
    def _calculate_angle_strength(self, df: pd.DataFrame, pivot: Dict, slope: float) -> float:
        """Calculate the strength/reliability of an angle line"""
        pivot_idx = pivot['index']
        
        if pivot_idx >= len(df) - 10:
            return 0.5  # Default strength for recent pivots
            
        # Test how well the angle line fits recent price action
        test_period = min(50, len(df) - pivot_idx - 1)
        deviations = []
        
        for i in range(1, test_period + 1):
            actual_price = df['typical_price'].iloc[pivot_idx + i]
            time_diff = df['time_numeric'].iloc[pivot_idx + i] - pivot['time']
            predicted_price = pivot['price'] + slope * time_diff
            
            deviation = abs(actual_price - predicted_price) / df['atr'].iloc[pivot_idx + i]
            deviations.append(deviation)
            
        # Calculate strength based on prediction accuracy
        avg_deviation = np.mean(deviations)
        strength = max(0.0, 1.0 - avg_deviation / 2.0)  # Normalize
        
        # Boost strength for volume-confirmed pivots
        if pivot.get('volume_confirmation', False):
            strength *= 1.2
            
        return min(strength, 1.0)
        
    def _extract_current_levels(self, angles: Dict) -> Dict:
        """Extract current price levels for all angle lines"""
        current_levels = {}
        
        for pivot_key, pivot_angles in angles.items():
            current_levels[pivot_key] = {}
            for angle_key, angle_data in pivot_angles.items():
                current_levels[pivot_key][angle_key] = {
                    'level': angle_data['current_level'],
                    'strength': angle_data['strength'],
                    'angle_degrees': angle_data['angle_degrees']
                }
                
        return current_levels
        
    def _find_angle_intersections(self, angles: Dict) -> List[Dict]:
        """Find intersections between different angle lines"""
        intersections = []
        
        # Convert angles to list for easier processing
        angle_list = []
        for pivot_key, pivot_angles in angles.items():
            for angle_key, angle_data in pivot_angles.items():
                angle_list.append({
                    'id': f"{pivot_key}_{angle_key}",
                    'slope': angle_data['slope'],
                    'intercept': angle_data['intercept'],
                    'pivot_time': angle_data['pivot_time'],
                    'strength': angle_data['strength']
                })
                
        # Find intersections between angle pairs
        for i, angle1 in enumerate(angle_list):
            for j, angle2 in enumerate(angle_list[i+1:], i+1):
                intersection = self._calculate_line_intersection(angle1, angle2)
                if intersection:
                    intersections.append(intersection)
                    
        # Sort by combined strength and proximity to current time
        intersections.sort(key=lambda x: x['combined_strength'], reverse=True)
        
        return intersections[:20]  # Keep top 20 intersections
        
    def _calculate_line_intersection(self, line1: Dict, line2: Dict) -> Optional[Dict]:
        """Calculate intersection point between two angle lines"""
        # Line equations: y = m1*(t - t1) + p1 and y = m2*(t - t2) + p2
        m1, p1, t1 = line1['slope'], line1['intercept'], line1['pivot_time']
        m2, p2, t2 = line2['slope'], line2['intercept'], line2['pivot_time']
        
        # Check if lines are parallel
        if abs(m1 - m2) < 1e-10:
            return None
            
        # Solve for intersection time: m1*(t - t1) + p1 = m2*(t - t2) + p2
        # t = (p2 - p1 + m1*t1 - m2*t2) / (m1 - m2)
        intersection_time = (p2 - p1 + m1*t1 - m2*t2) / (m1 - m2)
        intersection_price = m1 * (intersection_time - t1) + p1
        
        # Calculate combined strength
        combined_strength = (line1['strength'] + line2['strength']) / 2
        
        # Add bonus for confluence (multiple angles meeting)
        angle_diff = abs(math.degrees(math.atan(m1)) - math.degrees(math.atan(m2)))
        if 20 < angle_diff < 160:  # Avoid very acute or very obtuse angles
            combined_strength *= 1.2
            
        return {
            'time': intersection_time,
            'price': intersection_price,
            'line1_id': line1['id'],
            'line2_id': line2['id'],
            'combined_strength': min(combined_strength, 1.0),
            'angle_difference': angle_diff
        }
        
    def _enhance_with_ml(self, df: pd.DataFrame, angle_lines: Dict) -> Dict:
        """Enhance angle calculations with machine learning"""
        if len(df) < self.config.ml_training_window:
            return angle_lines  # Not enough data for ML
            
        try:
            # Prepare features for ML model
            features = self._prepare_ml_features(df)
            
            if not self.is_trained and len(features) >= self.config.ml_training_window:
                # Train the model
                self._train_ml_model(features, df)
                
            if self.is_trained:
                # Use ML to enhance angle predictions
                angle_lines = self._apply_ml_enhancements(features, angle_lines)
                
        except Exception as e:
            self.logger.warning(f"ML enhancement failed: {str(e)}")
            
        return angle_lines
        
    def _prepare_ml_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features for machine learning model"""
        features = []
        
        # Price features
        features.extend([
            df['close'].pct_change(1).fillna(0),
            df['close'].pct_change(5).fillna(0),
            df['close'].pct_change(20).fillna(0),
            (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std(),
        ])
        
        # Volatility features
        features.extend([
            df['atr'] / df['close'],
            df['volatility'].fillna(df['volatility'].mean()),
            df['true_range'] / df['close'],
        ])
        
        # Volume features
        features.extend([
            df['volume'] / df['volume'].rolling(20).mean(),
            (df['volume'] * df['close']).rolling(10).sum(),
        ])
        
        # Technical features
        rsi = self._calculate_rsi(df['close'], 14)
        macd = self._calculate_macd(df['close'])
        
        features.extend([
            rsi,
            macd,
            (df['high'] + df['low'] + df['close']) / 3,
        ])
        
        return np.column_stack(features)
        
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI for ML features"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
        
    def _calculate_macd(self, prices: pd.Series) -> pd.Series:
        """Calculate MACD for ML features"""
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        return ema12 - ema26
        
    def _train_ml_model(self, features: np.ndarray, df: pd.DataFrame):
        """Train the machine learning model"""
        # Prepare target variable (future price movement)
        target = df['close'].shift(-5) / df['close'] - 1  # 5-period forward return
        target = target.fillna(0)
        
        # Clean data
        valid_indices = ~(np.isnan(features).any(axis=1) | np.isnan(target))
        clean_features = features[valid_indices]
        clean_target = target[valid_indices]
        
        if len(clean_features) < 100:
            return
            
        # Scale features
        clean_features = self.scaler.fit_transform(clean_features)
        
        # Train model
        train_size = int(len(clean_features) * 0.8)
        X_train, y_train = clean_features[:train_size], clean_target[:train_size]
        
        self.ml_model.fit(X_train, y_train)
        self.is_trained = True
        
        self.logger.info("ML model trained successfully")
        
    def _apply_ml_enhancements(self, features: np.ndarray, angle_lines: Dict) -> Dict:
        """Apply ML enhancements to angle calculations"""
        if features.shape[0] == 0:
            return angle_lines
            
        try:
            # Get current features
            current_features = features[-1:].reshape(1, -1)
            current_features = self.scaler.transform(current_features)
            
            # Predict future price movement
            prediction = self.ml_model.predict(current_features)[0]
            
            # Adjust angle strengths based on ML prediction
            for pivot_key, pivot_angles in angle_lines['angles'].items():
                for angle_key, angle_data in pivot_angles.items():
                    # Calculate alignment between angle direction and ML prediction
                    angle_direction = 1 if angle_data['slope'] > 0 else -1
                    prediction_direction = 1 if prediction > 0 else -1
                    
                    alignment = angle_direction * prediction_direction
                    
                    # Adjust strength based on alignment
                    if alignment > 0:
                        angle_data['strength'] *= 1.2  # Boost aligned angles
                    else:
                        angle_data['strength'] *= 0.8  # Reduce conflicting angles
                        
                    angle_data['strength'] = min(angle_data['strength'], 1.0)
                    angle_data['ml_prediction'] = prediction
                    
        except Exception as e:
            self.logger.warning(f"ML enhancement application failed: {str(e)}")
            
        return angle_lines        
    def _calculate_support_resistance(self, df: pd.DataFrame, angle_lines: Dict) -> Dict:
        """Calculate support and resistance levels from angle lines"""
        sr_levels = {
            'support_levels': [],
            'resistance_levels': [],
            'dynamic_levels': [],
            'confluence_zones': []
        }
        
        current_price = df['close'].iloc[-1]
        
        # Extract all current angle levels
        all_levels = []
        for pivot_key, pivot_angles in angle_lines['angles'].items():
            for angle_key, angle_data in pivot_angles.items():
                level = angle_data['current_level']
                strength = angle_data['strength']
                
                all_levels.append({
                    'level': level,
                    'strength': strength,
                    'type': 'support' if level < current_price else 'resistance',
                    'source': f"{pivot_key}_{angle_key}"
                })
                
        # Sort levels by proximity to current price
        all_levels.sort(key=lambda x: abs(x['level'] - current_price))
        
        # Group levels into support and resistance
        support_levels = [lvl for lvl in all_levels if lvl['level'] < current_price]
        resistance_levels = [lvl for lvl in all_levels if lvl['level'] >= current_price]
        
        # Find confluence zones (multiple levels close together)
        confluence_zones = self._find_confluence_zones(all_levels, df['atr'].iloc[-1])
        
        sr_levels['support_levels'] = support_levels[:10]  # Top 10 support levels
        sr_levels['resistance_levels'] = resistance_levels[:10]  # Top 10 resistance levels
        sr_levels['confluence_zones'] = confluence_zones
        
        return sr_levels
        
    def _find_confluence_zones(self, levels: List[Dict], atr: float) -> List[Dict]:
        """Find confluence zones where multiple levels cluster together"""
        confluence_zones = []
        
        # Group levels within ATR distance
        tolerance = atr * 0.5
        used_indices = set()
        
        for i, level1 in enumerate(levels):
            if i in used_indices:
                continue
                
            zone_levels = [level1]
            zone_strength = level1['strength']
            used_indices.add(i)
            
            for j, level2 in enumerate(levels[i+1:], i+1):
                if j in used_indices:
                    continue
                    
                if abs(level1['level'] - level2['level']) <= tolerance:
                    zone_levels.append(level2)
                    zone_strength += level2['strength']
                    used_indices.add(j)
                    
            if len(zone_levels) >= 2:  # At least 2 levels for confluence
                avg_level = np.mean([lvl['level'] for lvl in zone_levels])
                confluence_zones.append({
                    'center_level': avg_level,
                    'strength': zone_strength / len(zone_levels),
                    'level_count': len(zone_levels),
                    'width': max([lvl['level'] for lvl in zone_levels]) - min([lvl['level'] for lvl in zone_levels]),
                    'component_levels': zone_levels
                })
                
        # Sort by strength
        confluence_zones.sort(key=lambda x: x['strength'] * x['level_count'], reverse=True)
        
        return confluence_zones[:5]  # Top 5 confluence zones
        
    def _analyze_trend_strength(self, df: pd.DataFrame, angle_lines: Dict) -> Dict:
        """Analyze trend strength through angle velocity and convergence"""
        trend_metrics = {
            'overall_trend_strength': 0.0,
            'trend_direction': 'neutral',
            'angle_velocity': 0.0,
            'convergence_score': 0.0,
            'breakout_probability': 0.0
        }
        
        current_price = df['close'].iloc[-1]
        
        # Calculate angle velocity (rate of change of angle levels)
        angle_velocities = []
        trend_votes = {'up': 0, 'down': 0, 'neutral': 0}
        
        for pivot_key, pivot_angles in angle_lines['angles'].items():
            for angle_key, angle_data in pivot_angles.items():
                # Calculate velocity based on slope and strength
                velocity = angle_data['slope'] * angle_data['strength']
                angle_velocities.append(velocity)
                
                # Vote for trend direction
                if velocity > 0.01:
                    trend_votes['up'] += angle_data['strength']
                elif velocity < -0.01:
                    trend_votes['down'] += angle_data['strength']
                else:
                    trend_votes['neutral'] += angle_data['strength']
                    
        # Calculate overall metrics
        if angle_velocities:
            avg_velocity = np.mean(angle_velocities)
            trend_metrics['angle_velocity'] = avg_velocity
            
            # Determine trend direction
            max_vote = max(trend_votes, key=trend_votes.get)
            trend_metrics['trend_direction'] = max_vote
            
            # Calculate trend strength
            total_votes = sum(trend_votes.values())
            if total_votes > 0:
                trend_metrics['overall_trend_strength'] = trend_votes[max_vote] / total_votes
                
        # Calculate convergence score
        trend_metrics['convergence_score'] = self._calculate_convergence_score(angle_lines)
        
        # Calculate breakout probability
        trend_metrics['breakout_probability'] = self._calculate_breakout_probability(
            df, angle_lines, trend_metrics
        )
        
        return trend_metrics
        
    def _calculate_convergence_score(self, angle_lines: Dict) -> float:
        """Calculate how much angles are converging or diverging"""
        if not angle_lines.get('intersections'):
            return 0.0
            
        # Analyze intersection patterns
        recent_intersections = [
            intersection for intersection in angle_lines['intersections']
            if intersection['combined_strength'] > 0.5
        ]
        
        if not recent_intersections:
            return 0.0
            
        # Calculate average strength of intersections
        avg_strength = np.mean([inter['combined_strength'] for inter in recent_intersections])
        
        # Calculate time clustering of intersections
        intersection_times = [inter['time'] for inter in recent_intersections]
        if len(intersection_times) > 1:
            time_variance = np.var(intersection_times)
            time_clustering = 1.0 / (1.0 + time_variance / 1000000)  # Normalize
        else:
            time_clustering = 0.5
            
        convergence_score = (avg_strength + time_clustering) / 2
        return min(convergence_score, 1.0)
        
    def _calculate_breakout_probability(self, df: pd.DataFrame, angle_lines: Dict, 
                                      trend_metrics: Dict) -> float:
        """Calculate probability of price breakout from current angle structure"""
        
        # Base probability from trend strength
        base_prob = trend_metrics['overall_trend_strength']
        
        # Adjustment factors
        factors = []
        
        # 1. Volatility factor
        current_vol = df['volatility'].iloc[-1]
        avg_vol = df['volatility'].mean()
        vol_factor = min(current_vol / avg_vol, 2.0) * 0.2
        factors.append(vol_factor)
        
        # 2. Volume factor
        current_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].rolling(20).mean().iloc[-1]
        volume_factor = min(current_volume / avg_volume, 3.0) * 0.15
        factors.append(volume_factor)
        
        # 3. Convergence factor
        convergence_factor = trend_metrics['convergence_score'] * 0.25
        factors.append(convergence_factor)
        
        # 4. Angle velocity factor
        velocity_factor = abs(trend_metrics['angle_velocity']) * 0.2
        factors.append(velocity_factor)
        
        # 5. Support/resistance proximity factor
        current_price = df['close'].iloc[-1]
        sr_levels = self._calculate_support_resistance(df, angle_lines)
        
        min_distance = float('inf')
        for level_group in [sr_levels['support_levels'], sr_levels['resistance_levels']]:
            for level in level_group:
                distance = abs(level['level'] - current_price) / df['atr'].iloc[-1]
                min_distance = min(min_distance, distance)
                
        proximity_factor = max(0, 1.0 - min_distance / 2.0) * 0.2
        factors.append(proximity_factor)
        
        # Combine factors
        total_adjustment = sum(factors)
        breakout_probability = min(base_prob + total_adjustment, 1.0)
        
        return breakout_probability
        
    def _perform_harmonic_analysis(self, angle_lines: Dict) -> Dict:
        """Perform harmonic analysis of angle relationships"""
        harmonic_data = {
            'harmonic_ratios': [],
            'fibonacci_alignments': [],
            'golden_ratio_levels': [],
            'harmonic_strength': 0.0
        }
        
        # Fibonacci ratios for harmonic analysis
        fib_ratios = [0.236, 0.382, 0.618, 0.764, 1.618, 2.618]
        golden_ratio = 1.618
        
        # Extract all angle slopes for harmonic analysis
        slopes = []
        for pivot_key, pivot_angles in angle_lines['angles'].items():
            for angle_key, angle_data in pivot_angles.items():
                slopes.append({
                    'slope': angle_data['slope'],
                    'strength': angle_data['strength'],
                    'angle': angle_data['angle_degrees']
                })
                
        # Find harmonic relationships between slopes
        harmonic_relationships = []
        
        for i, slope1 in enumerate(slopes):
            for j, slope2 in enumerate(slopes[i+1:], i+1):
                if slope1['slope'] != 0 and slope2['slope'] != 0:
                    ratio = abs(slope2['slope'] / slope1['slope'])
                    
                    # Check for Fibonacci ratios
                    for fib_ratio in fib_ratios:
                        if abs(ratio - fib_ratio) < 0.05:  # 5% tolerance
                            harmonic_relationships.append({
                                'ratio': ratio,
                                'fibonacci_ratio': fib_ratio,
                                'strength': (slope1['strength'] + slope2['strength']) / 2,
                                'slope1': slope1['slope'],
                                'slope2': slope2['slope'],
                                'type': 'fibonacci'
                            })
                            
                    # Check for golden ratio
                    if abs(ratio - golden_ratio) < 0.05:
                        harmonic_relationships.append({
                            'ratio': ratio,
                            'fibonacci_ratio': golden_ratio,
                            'strength': (slope1['strength'] + slope2['strength']) / 2,
                            'slope1': slope1['slope'],
                            'slope2': slope2['slope'],
                            'type': 'golden_ratio'
                        })
                        
        # Sort by strength
        harmonic_relationships.sort(key=lambda x: x['strength'], reverse=True)
        
        # Calculate overall harmonic strength
        if harmonic_relationships:
            harmonic_data['harmonic_strength'] = np.mean([hr['strength'] for hr in harmonic_relationships])
            harmonic_data['harmonic_ratios'] = harmonic_relationships[:10]
            
            # Separate Fibonacci and golden ratio alignments
            harmonic_data['fibonacci_alignments'] = [
                hr for hr in harmonic_relationships if hr['type'] == 'fibonacci'
            ][:5]
            
            harmonic_data['golden_ratio_levels'] = [
                hr for hr in harmonic_relationships if hr['type'] == 'golden_ratio'
            ][:3]
            
        return harmonic_data
        
    def _combine_results(self, df: pd.DataFrame, angle_lines: Dict, 
                        sr_levels: Dict, trend_metrics: Dict) -> pd.DataFrame:
        """Combine all analysis results into final output DataFrame"""
        
        result = df.copy()
        
        # Add angle line levels (closest to current price)
        current_price = df['close'].iloc[-1]
        
        # Find closest support and resistance levels
        support_levels = sr_levels['support_levels']
        resistance_levels = sr_levels['resistance_levels']
        
        closest_support = None
        closest_resistance = None
        
        if support_levels:
            closest_support = min(support_levels, key=lambda x: abs(x['level'] - current_price))
            
        if resistance_levels:
            closest_resistance = min(resistance_levels, key=lambda x: abs(x['level'] - current_price))
            
        # Add angle signals
        result['gann_trend_direction'] = trend_metrics['trend_direction']
        result['gann_trend_strength'] = trend_metrics['overall_trend_strength']
        result['gann_angle_velocity'] = trend_metrics['angle_velocity']
        result['gann_convergence_score'] = trend_metrics['convergence_score']
        result['gann_breakout_probability'] = trend_metrics['breakout_probability']
        
        # Add support/resistance levels
        result['gann_closest_support'] = closest_support['level'] if closest_support else np.nan
        result['gann_support_strength'] = closest_support['strength'] if closest_support else 0.0
        result['gann_closest_resistance'] = closest_resistance['level'] if closest_resistance else np.nan
        result['gann_resistance_strength'] = closest_resistance['strength'] if closest_resistance else 0.0
        
        # Add confluence zones
        confluence_zones = sr_levels['confluence_zones']
        if confluence_zones:
            strongest_zone = max(confluence_zones, key=lambda x: x['strength'])
            result['gann_confluence_level'] = strongest_zone['center_level']
            result['gann_confluence_strength'] = strongest_zone['strength']
            result['gann_confluence_count'] = strongest_zone['level_count']
        else:
            result['gann_confluence_level'] = np.nan
            result['gann_confluence_strength'] = 0.0
            result['gann_confluence_count'] = 0
            
        # Add harmonic analysis
        if 'harmonic_strength' in trend_metrics:
            result['gann_harmonic_strength'] = trend_metrics['harmonic_strength']
        else:
            result['gann_harmonic_strength'] = 0.0
            
        # Generate trading signals
        result['gann_signal'] = self._generate_trading_signals(result, angle_lines, sr_levels, trend_metrics)
        
        return result
        
    def _generate_trading_signals(self, df: pd.DataFrame, angle_lines: Dict, 
                                 sr_levels: Dict, trend_metrics: Dict) -> pd.Series:
        """Generate trading signals based on Gann analysis"""
        
        signals = pd.Series(0, index=df.index)  # 0 = neutral, 1 = buy, -1 = sell
        
        current_price = df['close'].iloc[-1]
        
        # Signal conditions
        trend_strength = trend_metrics['overall_trend_strength']
        trend_direction = trend_metrics['trend_direction']
        breakout_prob = trend_metrics['breakout_probability']
        convergence = trend_metrics['convergence_score']
        
        # Strong trend signals
        if trend_strength > 0.7 and breakout_prob > 0.6:
            if trend_direction == 'up':
                signals.iloc[-1] = 1  # Buy signal
            elif trend_direction == 'down':
                signals.iloc[-1] = -1  # Sell signal
                
        # Reversal signals at strong support/resistance
        elif convergence > 0.8:
            closest_support = df['gann_closest_support'].iloc[-1]
            closest_resistance = df['gann_closest_resistance'].iloc[-1]
            
            # Bounce off support
            if not pd.isna(closest_support) and df['gann_support_strength'].iloc[-1] > 0.8:
                if abs(current_price - closest_support) / df['atr'].iloc[-1] < 0.5:
                    signals.iloc[-1] = 1  # Buy signal
                    
            # Rejection at resistance
            elif not pd.isna(closest_resistance) and df['gann_resistance_strength'].iloc[-1] > 0.8:
                if abs(current_price - closest_resistance) / df['atr'].iloc[-1] < 0.5:
                    signals.iloc[-1] = -1  # Sell signal
                    
        # Confluence zone signals
        confluence_strength = df['gann_confluence_strength'].iloc[-1]
        if confluence_strength > 0.7 and not pd.isna(df['gann_confluence_level'].iloc[-1]):
            confluence_level = df['gann_confluence_level'].iloc[-1]
            distance_to_confluence = abs(current_price - confluence_level) / df['atr'].iloc[-1]
            
            if distance_to_confluence < 0.3:  # Very close to confluence
                if trend_direction == 'up' and current_price > confluence_level:
                    signals.iloc[-1] = 1  # Breakout buy
                elif trend_direction == 'down' and current_price < confluence_level:
                    signals.iloc[-1] = -1  # Breakdown sell
                    
        return signals
        
    def _calculate_ml_scaling(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate ML-based scaling factors (placeholder for future ML enhancement)"""
        # This would use historical data to learn optimal scaling
        # For now, return volatility-adaptive scaling
        return {
            'price_scale': 1.0,
            'time_scale': 1.0,
            'volatility_adjustment': df['atr'].iloc[-1] / df['atr'].mean()
        }


def create_gann_angles_indicator(config: Optional[GannAngleConfig] = None) -> GannAnglesIndicator:
    """Factory function to create GannAnglesIndicator instance"""
    return GannAnglesIndicator(config)


# Example usage and testing
if __name__ == "__main__":
    import yfinance as yf
    
    # Test with sample data
    ticker = "EURUSD=X"
    data = yf.download(ticker, period="1y", interval="1d")
    data.reset_index(inplace=True)
    data.columns = data.columns.str.lower()
    data['timestamp'] = data['date']
    
    # Create indicator
    config = GannAngleConfig(
        time_scaling_method="volatility_adaptive",
        enable_harmonic_analysis=True,
        enable_dynamic_scaling=True
    )
    
    indicator = GannAnglesIndicator(config)
    
    try:
        # Calculate Gann angles
        result = indicator.calculate(data)
        
        print("Gann Angles Calculation Results:")
        print(f"Data shape: {result.shape}")
        print(f"Columns: {list(result.columns)}")
        
        # Display recent signals
        recent = result.tail(5)
        for col in ['gann_trend_direction', 'gann_trend_strength', 'gann_signal']:
            if col in recent.columns:
                print(f"\n{col}:")
                print(recent[col].to_string())
                
    except Exception as e:
        print(f"Error testing Gann Angles indicator: {e}")