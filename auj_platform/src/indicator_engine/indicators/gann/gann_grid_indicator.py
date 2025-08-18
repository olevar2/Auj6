"""
Gann Grid Indicator Implementation
Humanitarian Trading Platform - Advanced Indicator Suite

This module implements a sophisticated Gann Grid Indicator with advanced grid overlay system,
price-time square calculations, harmonic analysis capabilities, dynamic grid generation,
ML-enhanced pattern recognition, and comprehensive trading signal generation.

Grid Features:
- Advanced grid overlay system with dynamic spacing
- Price-time square calculations and geometric relationships
- Multi-dimensional harmonic analysis
- ML-enhanced pattern recognition and classification
- Support/resistance level identification
- Grid confluence analysis and strength assessment
- Advanced trading signal generation
- Comprehensive market structure analysis

Author: AUJ Platform Development Team
Mission: To help poor families and sick children through advanced trading technology
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.ensemble import RandomForestRegressor, IsolationForest
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.metrics import mean_squared_error, r2_score
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

try:
    import scipy.stats as stats
    from scipy.signal import find_peaks, savgol_filter
    from scipy.optimize import minimize, differential_evolution
    from scipy.spatial.distance import euclidean
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GannGridConfig:
    """Configuration for Gann Grid calculations"""
    
    # Grid parameters
    min_grid_size: int = 20
    max_grid_size: int = 100
    grid_spacing_factor: float = 1.0
    dynamic_spacing: bool = True
    
    # Price-time square parameters
    square_calculation_method: str = "geometric"  # geometric, arithmetic, fibonacci
    harmonic_analysis: bool = True
    price_time_ratio: float = 1.0
    
    # Grid validation
    min_confluence_strength: float = 0.6
    max_grid_levels: int = 50
    level_clustering_tolerance: float = 0.8
    
    # ML enhancement parameters
    ml_validation: bool = ML_AVAILABLE
    anomaly_detection: bool = True
    pattern_recognition: bool = True
    prediction_horizon: int = 20
    
    # Signal generation
    signal_threshold: float = 0.65
    multi_timeframe_analysis: bool = True
    confluence_analysis: bool = True
    
    # Performance optimization
    max_lookback_periods: int = 1000
    calculation_precision: int = 6
    memory_optimization: bool = True


@dataclass
class GannGridSquare:
    """Represents a Gann grid square"""
    
    id: str
    center_time: float
    center_price: float
    width_time: float
    height_price: float
    grid_level: int
    
    # Square properties
    corners: List[Tuple[float, float]] = field(default_factory=list)
    harmonic_frequency: float = 0.0
    energy_level: float = 0.0
    confluence_count: int = 0
    
    # Market interaction
    price_touches: int = 0
    support_resistance_type: str = "neutral"
    validation_score: float = 0.0
    strength: float = 0.0
    
    # Classification
    pattern_type: str = "standard"
    significance: float = 0.0
    active: bool = True


@dataclass
class GannGridLevel:
    """Represents a grid level"""
    
    id: str
    level_type: str  # horizontal, vertical, diagonal
    level_value: float
    orientation: float  # angle in degrees
    
    # Level properties
    grid_squares: List[str] = field(default_factory=list)
    confluence_points: List[Tuple[float, float]] = field(default_factory=list)
    strength: float = 0.0
    touches: int = 0
    
    # Support/resistance analysis
    support_resistance_type: str = "neutral"
    validation_score: float = 0.0
    active_timeframe: float = 0.0
    
    # Statistical properties
    reliability: float = 0.0
    historical_accuracy: float = 0.0


@dataclass
class GannGrid:
    """Main Gann grid structure"""
    
    id: str
    origin_time: float
    origin_price: float
    grid_angle: float
    
    # Grid components
    grid_squares: List[GannGridSquare] = field(default_factory=list)
    grid_levels: List[GannGridLevel] = field(default_factory=list)
    confluence_zones: List[Dict] = field(default_factory=list)
    
    # Grid properties
    total_squares: int = 0
    active_squares: int = 0
    overall_strength: float = 0.0
    harmonic_resonance: float = 0.0
    
    # Market interaction
    price_interaction_count: int = 0
    support_resistance_efficiency: float = 0.0
    prediction_accuracy: float = 0.0
    
    # Classification
    grid_type: str = "standard"
    market_phase: str = "undefined"
    significance_score: float = 0.0


class GannGridIndicator:
    """
    Advanced Gann Grid Indicator with sophisticated grid overlay system,
    price-time square calculations, and harmonic analysis capabilities.
    """
    
    def __init__(self, config: Optional[GannGridConfig] = None):
        self.config = config or GannGridConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize ML models if available
        self.ml_models = {}
        self.scalers = {}
        
        if self.config.ml_validation and ML_AVAILABLE:
            self._initialize_ml_models()
            
        # Grid analysis state
        self.current_grids: List[GannGrid] = []
        self.historical_grids: List[GannGrid] = []
        self.performance_metrics = {}
        
        self.logger.info("Advanced Gann Grid Indicator initialized")
        
    def _initialize_ml_models(self):
        """Initialize machine learning models for enhanced analysis"""
        
        try:
            # Grid pattern recognition model
            self.ml_models['pattern_classifier'] = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            # Strength prediction model
            self.ml_models['strength_predictor'] = RandomForestRegressor(
                n_estimators=50,
                max_depth=8,
                random_state=42
            )
            
            # Anomaly detection for grid validation
            if self.config.anomaly_detection:
                self.ml_models['anomaly_detector'] = IsolationForest(
                    contamination=0.1,
                    random_state=42
                )
            
            # Feature scalers
            self.scalers['standard'] = StandardScaler()
            self.scalers['minmax'] = MinMaxScaler()
            
            self.logger.info("ML models initialized successfully")
            
        except Exception as e:
            self.logger.warning(f"ML model initialization failed: {e}")
            self.config.ml_validation = False
            
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Gann Grid indicator values
        
        Args:
            df: DataFrame with OHLCV data and timestamp
            
        Returns:
            DataFrame with Gann Grid analysis results
        """
        
        try:
            # Validate input data
            df = self._validate_input_data(df)
            
            # Prepare data for grid analysis
            prepared_data = self._prepare_grid_data(df)
            
            # Identify grid origins and pivot points
            grid_origins = self._identify_grid_origins(prepared_data)
            
            # Generate Gann grids
            grids = self._generate_gann_grids(prepared_data, grid_origins)
            
            # Calculate price-time squares
            enhanced_grids = self._calculate_price_time_squares(prepared_data, grids)
            
            # Perform harmonic analysis
            harmonic_analysis = self._perform_harmonic_analysis(prepared_data, enhanced_grids)
            
            # Generate confluence zones
            confluence_zones = self._generate_confluence_zones(prepared_data, enhanced_grids)
            
            # Validate grids with ML if enabled
            if self.config.ml_validation:
                validated_grids = self._ml_validate_grids(prepared_data, enhanced_grids)
            else:
                validated_grids = enhanced_grids
                
            # Extract support/resistance levels
            sr_levels = self._extract_grid_support_resistance(prepared_data, validated_grids)
            
            # Analyze grid interactions
            interaction_analysis = self._analyze_grid_interactions(prepared_data, validated_grids)
            
            # Generate trading signals
            signals = self._generate_grid_signals(prepared_data, validated_grids, sr_levels)
            
            # Combine all results
            result = self._combine_grid_results(
                df, validated_grids, sr_levels, confluence_zones, 
                harmonic_analysis, interaction_analysis, signals
            )
            
            # Update state
            self.current_grids = validated_grids
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in Gann Grid calculation: {e}")
            return self._create_error_result(df)
            
    def _validate_input_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and prepare input data"""
        
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Check required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        # Ensure timestamp is available
        if 'timestamp' not in df.columns:
            if df.index.name == 'timestamp' or 'datetime' in str(type(df.index[0])):
                df = df.reset_index()
                df['timestamp'] = df['index'] if 'index' in df.columns else df.index
            else:
                df['timestamp'] = pd.date_range(start='2023-01-01', periods=len(df), freq='1H')
                
        # Convert timestamp to numeric for calculations
        if not pd.api.types.is_numeric_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['time_numeric'] = df['timestamp'].astype('int64') // 10**9
        else:
            df['time_numeric'] = df['timestamp']
            
        # Calculate additional technical indicators needed
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['price_range'] = df['high'] - df['low']
        df['true_range'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['true_range'].rolling(window=14).mean()
        
        # Volatility measures
        df['price_volatility'] = df['close'].pct_change().rolling(window=20).std()
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        
        return df.dropna()
        
    def _prepare_grid_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data specifically for grid analysis"""
        
        data = df.copy()
        
        # Calculate pivot points for grid origin identification
        data['pivot_high'] = data['high'].rolling(window=5, center=True).max() == data['high']
        data['pivot_low'] = data['low'].rolling(window=5, center=True).min() == data['low']
        
        # Market structure analysis
        data['trend_strength'] = self._calculate_trend_strength(data)
        data['market_volatility'] = data['atr'] / data['close']
        
        # Price-time relationship indicators
        data['price_momentum'] = data['close'].pct_change(20)
        data['time_momentum'] = np.arange(len(data)) / len(data)
        
        # Harmonic components
        if len(data) >= 50:
            data['harmonic_component'] = self._calculate_harmonic_component(data['close'])
        else:
            data['harmonic_component'] = 0.0
            
        # Grid spacing calculations
        data['dynamic_grid_spacing'] = self._calculate_dynamic_spacing(data)
        
        return data
        
    def _calculate_trend_strength(self, df: pd.DataFrame) -> pd.Series:
        """Calculate trend strength for grid analysis"""
        
        # Multiple timeframe trend analysis
        short_ma = df['close'].rolling(window=10).mean()
        medium_ma = df['close'].rolling(window=20).mean()
        long_ma = df['close'].rolling(window=50).mean()
        
        # Trend alignment score
        trend_alignment = 0.0
        if len(df) >= 50:
            trend_alignment = np.where(
                (short_ma > medium_ma) & (medium_ma > long_ma), 1.0,
                np.where((short_ma < medium_ma) & (medium_ma < long_ma), -1.0, 0.0)
            )
            
        # ADX-like calculation for trend strength
        high_low = df['high'] - df['low']
        high_close_prev = abs(df['high'] - df['close'].shift(1))
        low_close_prev = abs(df['low'] - df['close'].shift(1))
        
        true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
        
        dm_plus = np.where((df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']),
                          np.maximum(df['high'] - df['high'].shift(1), 0), 0)
        dm_minus = np.where((df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)),
                           np.maximum(df['low'].shift(1) - df['low'], 0), 0)
        
        tr_ma = pd.Series(true_range).rolling(window=14).mean()
        dm_plus_ma = pd.Series(dm_plus).rolling(window=14).mean()
        dm_minus_ma = pd.Series(dm_minus).rolling(window=14).mean()
        
        di_plus = 100 * dm_plus_ma / tr_ma
        di_minus = 100 * dm_minus_ma / tr_ma
        
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.rolling(window=14).mean()
        
        return adx.fillna(0) / 100.0  # Normalize to 0-1
        
    def _calculate_harmonic_component(self, price_series: pd.Series) -> pd.Series:
        """Calculate harmonic component for grid analysis"""
        
        if not SCIPY_AVAILABLE or len(price_series) < 50:
            return pd.Series(0.0, index=price_series.index)
            
        try:
            # Apply FFT to identify dominant frequencies
            prices = price_series.values
            fft = np.fft.fft(prices)
            frequencies = np.fft.fftfreq(len(prices))
            
            # Find dominant frequency
            dominant_freq_idx = np.argmax(np.abs(fft[1:len(fft)//2])) + 1
            dominant_frequency = frequencies[dominant_freq_idx]
            
            # Generate harmonic component
            time_index = np.arange(len(prices))
            harmonic = np.sin(2 * np.pi * dominant_frequency * time_index)
            
            return pd.Series(harmonic, index=price_series.index)
            
        except Exception as e:
            self.logger.warning(f"Harmonic component calculation failed: {e}")
            return pd.Series(0.0, index=price_series.index)
            
    def _calculate_dynamic_spacing(self, df: pd.DataFrame) -> pd.Series:
        """Calculate dynamic grid spacing based on market conditions"""
        
        # Base spacing on volatility and range
        base_spacing = df['atr'] * self.config.grid_spacing_factor
        
        # Adjust for trend strength
        trend_adjustment = 1.0 + (df['trend_strength'] * 0.5)
        
        # Adjust for volume
        volume_ratio = df['volume'] / df['volume_ma']
        volume_adjustment = np.clip(volume_ratio, 0.5, 2.0)
        
        # Calculate final spacing
        dynamic_spacing = base_spacing * trend_adjustment * volume_adjustment
        
        return dynamic_spacing
        
    def _identify_grid_origins(self, df: pd.DataFrame) -> List[Dict]:
        """Identify potential grid origin points"""
        
        origins = []
        
        # Significant pivot points as potential origins
        pivot_highs = df[df['pivot_high']].copy()
        pivot_lows = df[df['pivot_low']].copy()
        
        # Combine and filter pivots
        all_pivots = []
        
        for idx, row in pivot_highs.iterrows():
            all_pivots.append({
                'time': row['time_numeric'],
                'price': row['high'],
                'type': 'high',
                'index': idx,
                'strength': row['trend_strength'],
                'volume': row['volume']
            })
            
        for idx, row in pivot_lows.iterrows():
            all_pivots.append({
                'time': row['time_numeric'],
                'price': row['low'],
                'type': 'low',
                'index': idx,
                'strength': row['trend_strength'],
                'volume': row['volume']
            })
            
        # Sort by time
        all_pivots.sort(key=lambda x: x['time'])
        
        # Filter significant pivots
        significant_pivots = []
        for pivot in all_pivots:
            # Check significance criteria
            if (pivot['strength'] > 0.3 and 
                pivot['volume'] > df['volume_ma'].iloc[pivot['index']]):
                significant_pivots.append(pivot)
                
        # Limit number of origins
        max_origins = min(len(significant_pivots), 10)
        origins = significant_pivots[:max_origins]
        
        self.logger.info(f"Identified {len(origins)} grid origins")
        return origins
        
    def _generate_gann_grids(self, df: pd.DataFrame, origins: List[Dict]) -> List[GannGrid]:
        """Generate Gann grids from identified origins"""
        
        grids = []
        
        for i, origin in enumerate(origins):
            try:
                grid = self._create_single_grid(df, origin, f"grid_{i}")
                if grid and grid.total_squares > 0:
                    grids.append(grid)
                    
            except Exception as e:
                self.logger.warning(f"Failed to create grid {i}: {e}")
                continue
                
        self.logger.info(f"Generated {len(grids)} Gann grids")
        return grids
        
    def _create_single_grid(self, df: pd.DataFrame, origin: Dict, grid_id: str) -> Optional[GannGrid]:
        """Create a single Gann grid from origin point"""
        
        origin_time = origin['time']
        origin_price = origin['price']
        
        # Determine grid angle based on market conditions
        grid_angle = self._calculate_grid_angle(df, origin)
        
        # Create grid structure
        grid = GannGrid(
            id=grid_id,
            origin_time=origin_time,
            origin_price=origin_price,
            grid_angle=grid_angle
        )
        
        # Generate grid squares
        grid_squares = self._generate_grid_squares(df, origin, grid_angle)
        
        # Generate grid levels
        grid_levels = self._generate_grid_levels(df, origin, grid_angle, grid_squares)
        
        # Assign to grid
        grid.grid_squares = grid_squares
        grid.grid_levels = grid_levels
        grid.total_squares = len(grid_squares)
        grid.active_squares = len([sq for sq in grid_squares if sq.active])
        
        # Calculate initial grid properties
        grid.overall_strength = self._calculate_grid_strength(df, grid)
        grid.harmonic_resonance = self._calculate_harmonic_resonance(df, grid)
        
        return grid
        
    def _calculate_grid_angle(self, df: pd.DataFrame, origin: Dict) -> float:
        """Calculate optimal grid angle based on market structure"""
        
        # Default Gann angles to test
        gann_angles = [0, 15, 30, 45, 60, 75, 90]
        
        origin_idx = origin['index']
        
        # Test each angle and find best fit
        best_angle = 45.0  # Default
        best_score = 0.0
        
        for angle in gann_angles:
            score = self._evaluate_grid_angle(df, origin, angle)
            if score > best_score:
                best_score = score
                best_angle = angle
                
        return best_angle
        
    def _evaluate_grid_angle(self, df: pd.DataFrame, origin: Dict, angle: float) -> float:
        """Evaluate how well a grid angle fits the market data"""
        
        origin_idx = origin['index']
        origin_time = origin['time']
        origin_price = origin['price']
        
        # Look ahead and behind for validation
        lookback = min(50, origin_idx)
        lookahead = min(50, len(df) - origin_idx - 1)
        
        if lookback < 10 or lookahead < 10:
            return 0.0
            
        # Calculate expected grid lines
        angle_rad = np.radians(angle)
        slope = np.tan(angle_rad) if angle != 90 else float('inf')
        
        score = 0.0
        test_count = 0
        
        # Test price interactions with grid lines
        for i in range(max(0, origin_idx - lookback), min(len(df), origin_idx + lookahead)):
            if i == origin_idx:
                continue
                
            row = df.iloc[i]
            time_diff = row['time_numeric'] - origin_time
            
            if angle == 90:  # Vertical line
                expected_time = origin_time
                if abs(time_diff) < df['atr'].iloc[i] * 86400:  # Within ATR in time
                    score += 1.0
            else:
                # Calculate expected price on grid line
                expected_price = origin_price + slope * time_diff
                
                # Check if market price is close to grid line
                price_diff = abs(row['typical_price'] - expected_price)
                tolerance = df['atr'].iloc[i] * 0.5
                
                if price_diff < tolerance:
                    score += 1.0 / (1.0 + price_diff / tolerance)
                    
            test_count += 1
            
        return score / test_count if test_count > 0 else 0.0        
    def _generate_grid_squares(self, df: pd.DataFrame, origin: Dict, grid_angle: float) -> List[GannGridSquare]:
        """Generate grid squares from origin point"""
        
        squares = []
        origin_time = origin['time']
        origin_price = origin['price']
        
        # Calculate dynamic grid spacing
        origin_idx = origin['index']
        grid_spacing = df['dynamic_grid_spacing'].iloc[origin_idx]
        
        # Grid dimensions
        grid_size = self.config.min_grid_size
        max_squares = min(self.config.max_grid_levels, grid_size * grid_size)
        
        # Generate squares in a grid pattern
        for level in range(1, int(np.sqrt(max_squares)) + 1):
            for x_offset in range(-level, level + 1):
                for y_offset in range(-level, level + 1):
                    if abs(x_offset) != level and abs(y_offset) != level:
                        continue  # Only generate perimeter squares
                        
                    # Calculate square position
                    time_offset = x_offset * grid_spacing * 86400  # Convert to seconds
                    price_offset = y_offset * grid_spacing
                    
                    square_time = origin_time + time_offset
                    square_price = origin_price + price_offset
                    
                    # Create square
                    square = GannGridSquare(
                        id=f"{origin['type']}_{level}_{x_offset}_{y_offset}",
                        center_time=square_time,
                        center_price=square_price,
                        width_time=grid_spacing * 86400,
                        height_price=grid_spacing,
                        grid_level=level
                    )
                    
                    # Calculate square corners
                    half_width = square.width_time / 2
                    half_height = square.height_price / 2
                    
                    square.corners = [
                        (square_time - half_width, square_price - half_height),  # Bottom-left
                        (square_time + half_width, square_price - half_height),  # Bottom-right
                        (square_time + half_width, square_price + half_height),  # Top-right
                        (square_time - half_width, square_price + half_height)   # Top-left
                    ]
                    
                    # Calculate harmonic frequency
                    square.harmonic_frequency = self._calculate_square_harmonic_frequency(square, grid_angle)
                    
                    # Initial validation
                    square.validation_score = self._validate_grid_square(df, square)
                    
                    if square.validation_score > 0.3:  # Only keep valid squares
                        squares.append(square)
                        
        self.logger.debug(f"Generated {len(squares)} grid squares for origin {origin['type']}")
        return squares
        
    def _calculate_square_harmonic_frequency(self, square: GannGridSquare, grid_angle: float) -> float:
        """Calculate harmonic frequency for a grid square"""
        
        # Base frequency calculation
        time_component = 1.0 / (square.width_time / 86400)  # Daily frequency
        price_component = square.center_price / square.height_price
        
        # Angle influence
        angle_factor = np.sin(np.radians(grid_angle + 45))  # Phase shift
        
        # Harmonic frequency
        frequency = np.sqrt(time_component * price_component) * angle_factor
        
        return abs(frequency)
        
    def _validate_grid_square(self, df: pd.DataFrame, square: GannGridSquare) -> float:
        """Validate grid square against market data"""
        
        validation_score = 0.0
        
        # Find data points within square timeframe
        mask = (df['time_numeric'] >= square.center_time - square.width_time/2) & \
               (df['time_numeric'] <= square.center_time + square.width_time/2)
        
        square_data = df[mask]
        
        if len(square_data) == 0:
            return 0.0
            
        # Check price interactions
        price_interactions = 0
        total_points = len(square_data)
        
        for _, row in square_data.iterrows():
            price_range = [row['low'], row['high']]
            square_price_range = [
                square.center_price - square.height_price/2,
                square.center_price + square.height_price/2
            ]
            
            # Check for overlap
            if (price_range[1] >= square_price_range[0] and 
                price_range[0] <= square_price_range[1]):
                price_interactions += 1
                
        # Calculate interaction ratio
        interaction_ratio = price_interactions / total_points if total_points > 0 else 0.0
        
        # Base validation score
        validation_score = interaction_ratio
        
        # Bonus for corner touches (support/resistance)
        for corner in square.corners:
            corner_time, corner_price = corner
            
            # Find closest data point
            time_distances = abs(df['time_numeric'] - corner_time)
            closest_idx = time_distances.idxmin()
            
            if closest_idx in df.index:
                closest_row = df.loc[closest_idx]
                price_distance = min(
                    abs(closest_row['high'] - corner_price),
                    abs(closest_row['low'] - corner_price),
                    abs(closest_row['close'] - corner_price)
                )
                
                tolerance = closest_row['atr'] * 0.3
                if price_distance < tolerance:
                    validation_score += 0.1  # Bonus for corner interaction
                    
        return min(validation_score, 1.0)
        
    def _generate_grid_levels(self, df: pd.DataFrame, origin: Dict, grid_angle: float, 
                            squares: List[GannGridSquare]) -> List[GannGridLevel]:
        """Generate grid levels from squares"""
        
        levels = []
        
        # Extract horizontal levels
        horizontal_levels = self._extract_horizontal_levels(squares)
        
        # Extract vertical levels
        vertical_levels = self._extract_vertical_levels(squares)
        
        # Extract diagonal levels
        diagonal_levels = self._extract_diagonal_levels(squares, grid_angle)
        
        # Combine all levels
        all_levels = horizontal_levels + vertical_levels + diagonal_levels
        
        # Validate and strengthen levels
        for level in all_levels:
            level.strength = self._calculate_level_strength(df, level, squares)
            level.validation_score = self._validate_grid_level(df, level)
            
            if level.validation_score > 0.4:
                levels.append(level)
                
        self.logger.debug(f"Generated {len(levels)} grid levels")
        return levels
        
    def _extract_horizontal_levels(self, squares: List[GannGridSquare]) -> List[GannGridLevel]:
        """Extract horizontal grid levels"""
        
        levels = []
        
        # Group squares by price level
        price_groups = {}
        for square in squares:
            price_key = round(square.center_price, self.config.calculation_precision)
            if price_key not in price_groups:
                price_groups[price_key] = []
            price_groups[price_key].append(square.id)
            
        # Create horizontal levels
        for price, square_ids in price_groups.items():
            if len(square_ids) >= 2:  # Need at least 2 squares for a level
                level = GannGridLevel(
                    id=f"horizontal_{price}",
                    level_type="horizontal",
                    level_value=price,
                    orientation=0.0,
                    grid_squares=square_ids
                )
                levels.append(level)
                
        return levels
        
    def _extract_vertical_levels(self, squares: List[GannGridSquare]) -> List[GannGridLevel]:
        """Extract vertical grid levels"""
        
        levels = []
        
        # Group squares by time level
        time_groups = {}
        for square in squares:
            time_key = round(square.center_time, 0)  # Round to nearest second
            if time_key not in time_groups:
                time_groups[time_key] = []
            time_groups[time_key].append(square.id)
            
        # Create vertical levels
        for time_val, square_ids in time_groups.items():
            if len(square_ids) >= 2:  # Need at least 2 squares for a level
                level = GannGridLevel(
                    id=f"vertical_{time_val}",
                    level_type="vertical",
                    level_value=time_val,
                    orientation=90.0,
                    grid_squares=square_ids
                )
                levels.append(level)
                
        return levels
        
    def _extract_diagonal_levels(self, squares: List[GannGridSquare], grid_angle: float) -> List[GannGridLevel]:
        """Extract diagonal grid levels"""
        
        levels = []
        
        # Define diagonal angles to extract
        diagonal_angles = [grid_angle, grid_angle + 90, grid_angle + 45, grid_angle - 45]
        
        for angle in diagonal_angles:
            if angle < 0:
                angle += 180
            if angle >= 180:
                angle -= 180
                
            # Group squares along diagonal lines
            diagonal_groups = self._group_squares_by_diagonal(squares, angle)
            
            # Create diagonal levels
            for group_id, square_ids in diagonal_groups.items():
                if len(square_ids) >= 2:
                    level = GannGridLevel(
                        id=f"diagonal_{angle}_{group_id}",
                        level_type="diagonal",
                        level_value=group_id,
                        orientation=angle,
                        grid_squares=square_ids
                    )
                    levels.append(level)
                    
        return levels
        
    def _group_squares_by_diagonal(self, squares: List[GannGridSquare], angle: float) -> Dict[str, List[str]]:
        """Group squares that lie on diagonal lines"""
        
        groups = {}
        angle_rad = np.radians(angle)
        
        # Calculate slope and intercept for each square
        for square in squares:
            if angle == 90:  # Vertical line
                line_key = f"vertical_{square.center_time}"
            else:
                slope = np.tan(angle_rad)
                # Calculate line equation: y = mx + b
                intercept = square.center_price - slope * square.center_time
                line_key = f"diagonal_{round(intercept, 2)}"
                
            if line_key not in groups:
                groups[line_key] = []
            groups[line_key].append(square.id)
            
        return groups
        
    def _calculate_level_strength(self, df: pd.DataFrame, level: GannGridLevel, 
                                squares: List[GannGridSquare]) -> float:
        """Calculate strength of a grid level"""
        
        strength = 0.0
        
        # Base strength from number of squares
        strength += len(level.grid_squares) * 0.1
        
        # Strength from square validation scores
        for square_id in level.grid_squares:
            square = next((sq for sq in squares if sq.id == square_id), None)
            if square:
                strength += square.validation_score * 0.2
                
        # Market interaction strength
        interaction_strength = self._calculate_level_market_interaction(df, level)
        strength += interaction_strength * 0.5
        
        # Level type bonuses
        if level.level_type == "horizontal":
            strength *= 1.2  # Horizontal levels often stronger for S/R
        elif level.level_type == "diagonal" and abs(level.orientation - 45) < 5:
            strength *= 1.1  # 45-degree angles are significant in Gann theory
            
        return min(strength, 1.0)
        
    def _calculate_level_market_interaction(self, df: pd.DataFrame, level: GannGridLevel) -> float:
        """Calculate how much the market interacts with a grid level"""
        
        interaction_score = 0.0
        total_tests = 0
        
        if level.level_type == "horizontal":
            # Test price touches for horizontal levels
            price_level = level.level_value
            
            for _, row in df.iterrows():
                tolerance = row['atr'] * 0.5
                
                # Check if price touched the level
                if (row['low'] <= price_level + tolerance and 
                    row['high'] >= price_level - tolerance):
                    interaction_score += 1.0
                    
                    # Bonus for exact touches
                    exact_distances = [
                        abs(row['high'] - price_level),
                        abs(row['low'] - price_level),
                        abs(row['close'] - price_level),
                        abs(row['open'] - price_level)
                    ]
                    
                    min_distance = min(exact_distances)
                    if min_distance < tolerance * 0.3:
                        interaction_score += 0.5
                        
                total_tests += 1
                
        elif level.level_type == "vertical":
            # Test time-based interactions for vertical levels
            time_level = level.level_value
            
            # Find data points close to this time
            time_distances = abs(df['time_numeric'] - time_level)
            closest_indices = time_distances.nsmallest(5).index
            
            for idx in closest_indices:
                if idx in df.index:
                    row = df.loc[idx]
                    time_diff = abs(row['time_numeric'] - time_level)
                    
                    # Time tolerance based on data frequency
                    time_tolerance = 3600  # 1 hour in seconds
                    
                    if time_diff < time_tolerance:
                        interaction_score += 1.0 / (1.0 + time_diff / time_tolerance)
                        total_tests += 1
                        
        elif level.level_type == "diagonal":
            # Test diagonal line interactions
            angle_rad = np.radians(level.orientation)
            
            if level.orientation == 90:
                # Vertical diagonal
                interaction_score = self._calculate_level_market_interaction(
                    df, GannGridLevel("temp", "vertical", level.level_value, 90.0)
                )
            else:
                slope = np.tan(angle_rad)
                
                for _, row in df.iterrows():
                    # Calculate expected price on diagonal line at this time
                    expected_price = level.level_value + slope * row['time_numeric']
                    
                    # Check distance to actual prices
                    tolerance = row['atr'] * 0.7  # Diagonal levels need more tolerance
                    
                    actual_prices = [row['high'], row['low'], row['close'], row['open']]
                    min_distance = min(abs(p - expected_price) for p in actual_prices)
                    
                    if min_distance < tolerance:
                        interaction_score += 1.0 / (1.0 + min_distance / tolerance)
                        
                    total_tests += 1
                    
        return interaction_score / total_tests if total_tests > 0 else 0.0
        
    def _validate_grid_level(self, df: pd.DataFrame, level: GannGridLevel) -> float:
        """Validate grid level against market data"""
        
        # Basic validation from market interaction
        market_interaction = self._calculate_level_market_interaction(df, level)
        
        # Length validation (longer levels are generally stronger)
        length_score = min(len(level.grid_squares) / 5.0, 1.0)
        
        # Consistency validation
        consistency_score = 1.0  # Default
        
        if level.level_type == "horizontal":
            # Check price consistency across squares
            prices = []
            for square_id in level.grid_squares:
                # Extract price from square_id or use level value
                prices.append(level.level_value)
                
            if len(prices) > 1:
                price_std = np.std(prices)
                avg_price = np.mean(prices)
                consistency_score = 1.0 / (1.0 + price_std / avg_price) if avg_price > 0 else 0.5
                
        # Combine validation components
        validation_score = (market_interaction * 0.6 + 
                          length_score * 0.2 + 
                          consistency_score * 0.2)
        
        return min(validation_score, 1.0)
        
    def _calculate_price_time_squares(self, df: pd.DataFrame, grids: List[GannGrid]) -> List[GannGrid]:
        """Calculate price-time squares for enhanced grid analysis"""
        
        enhanced_grids = []
        
        for grid in grids:
            enhanced_grid = self._enhance_grid_with_price_time_squares(df, grid)
            enhanced_grids.append(enhanced_grid)
            
        return enhanced_grids
        
    def _enhance_grid_with_price_time_squares(self, df: pd.DataFrame, grid: GannGrid) -> GannGrid:
        """Enhance a single grid with price-time square calculations"""
        
        enhanced_grid = grid
        
        # Calculate price-time relationships for each square
        for square in enhanced_grid.grid_squares:
            # Price-time ratio calculation
            time_span = square.width_time / 86400  # Convert to days
            price_span = square.height_price
            
            if time_span > 0 and price_span > 0:
                price_time_ratio = price_span / time_span
                
                # Geometric mean for price-time square
                if self.config.square_calculation_method == "geometric":
                    square.energy_level = np.sqrt(price_span * time_span)
                elif self.config.square_calculation_method == "arithmetic":
                    square.energy_level = (price_span + time_span) / 2
                elif self.config.square_calculation_method == "fibonacci":
                    square.energy_level = price_span * 1.618 + time_span * 0.618
                else:
                    square.energy_level = price_span * time_span
                    
                # Adjust energy level by harmonic frequency
                square.energy_level *= (1.0 + square.harmonic_frequency * 0.1)
                
            # Market interaction analysis
            square.price_touches = self._count_square_price_touches(df, square)
            square.support_resistance_type = self._classify_square_sr_type(df, square)
            
            # Update square strength
            square.strength = self._calculate_square_strength(df, square)
            
        # Update grid-level metrics
        enhanced_grid.harmonic_resonance = self._calculate_harmonic_resonance(df, enhanced_grid)
        enhanced_grid.overall_strength = self._calculate_grid_strength(df, enhanced_grid)
        
        return enhanced_grid
        
    def _count_square_price_touches(self, df: pd.DataFrame, square: GannGridSquare) -> int:
        """Count how many times price touched the square boundaries"""
        
        touches = 0
        
        # Define square boundaries
        time_start = square.center_time - square.width_time / 2
        time_end = square.center_time + square.width_time / 2
        price_bottom = square.center_price - square.height_price / 2
        price_top = square.center_price + square.height_price / 2
        
        # Find data within time range
        mask = (df['time_numeric'] >= time_start) & (df['time_numeric'] <= time_end)
        square_data = df[mask]
        
        for _, row in square_data.iterrows():
            # Check boundary touches
            tolerance = row['atr'] * 0.2
            
            # Top boundary
            if abs(row['high'] - price_top) < tolerance:
                touches += 1
            # Bottom boundary  
            elif abs(row['low'] - price_bottom) < tolerance:
                touches += 1
            # Side boundaries (time-based)
            elif (abs(row['time_numeric'] - time_start) < 1800 or  # 30 minutes
                  abs(row['time_numeric'] - time_end) < 1800):
                if price_bottom <= row['typical_price'] <= price_top:
                    touches += 1
                    
        return touches
        
    def _classify_square_sr_type(self, df: pd.DataFrame, square: GannGridSquare) -> str:
        """Classify square as support, resistance, or neutral"""
        
        # Get data around square timeframe
        time_buffer = square.width_time
        time_start = square.center_time - time_buffer
        time_end = square.center_time + time_buffer
        
        mask = (df['time_numeric'] >= time_start) & (df['time_numeric'] <= time_end)
        data = df[mask]
        
        if len(data) < 5:
            return "neutral"
            
        # Analyze price behavior relative to square center
        center_price = square.center_price
        
        prices_above = (data['close'] > center_price).sum()
        prices_below = (data['close'] < center_price).sum()
        total_prices = len(data)
        
        # Classification thresholds
        if prices_below / total_prices > 0.7:
            return "resistance"
        elif prices_above / total_prices > 0.7:
            return "support"
        else:
            return "neutral"
            
    def _calculate_square_strength(self, df: pd.DataFrame, square: GannGridSquare) -> float:
        """Calculate overall strength of a grid square"""
        
        strength = 0.0
        
        # Base strength from validation score
        strength += square.validation_score * 0.3
        
        # Strength from price touches
        touch_strength = min(square.price_touches / 5.0, 1.0)
        strength += touch_strength * 0.3
        
        # Strength from energy level (normalized)
        if square.energy_level > 0:
            energy_strength = min(square.energy_level / 100.0, 1.0)
            strength += energy_strength * 0.2
            
        # Strength from harmonic frequency
        harmonic_strength = min(square.harmonic_frequency, 1.0)
        strength += harmonic_strength * 0.2
        
        return min(strength, 1.0)
        
    def _calculate_harmonic_resonance(self, df: pd.DataFrame, grid: GannGrid) -> float:
        """Calculate harmonic resonance of the entire grid"""
        
        if not grid.grid_squares:
            return 0.0
            
        # Collect harmonic frequencies from all squares
        frequencies = [sq.harmonic_frequency for sq in grid.grid_squares if sq.active]
        
        if not frequencies:
            return 0.0
            
        # Calculate resonance metrics
        mean_frequency = np.mean(frequencies)
        frequency_std = np.std(frequencies)
        
        # Resonance is higher when frequencies are similar (low std)
        frequency_consistency = 1.0 / (1.0 + frequency_std) if frequency_std > 0 else 1.0
        
        # Overall energy from frequencies
        total_energy = sum(frequencies)
        normalized_energy = min(total_energy / len(frequencies), 1.0)
        
        # Combine metrics
        harmonic_resonance = (frequency_consistency * 0.6 + normalized_energy * 0.4)
        
        return harmonic_resonance
        
    def _calculate_grid_strength(self, df: pd.DataFrame, grid: GannGrid) -> float:
        """Calculate overall strength of a grid"""
        
        if not grid.grid_squares:
            return 0.0
            
        # Average square strength
        square_strengths = [sq.strength for sq in grid.grid_squares]
        avg_square_strength = np.mean(square_strengths) if square_strengths else 0.0
        
        # Level strength contribution
        if grid.grid_levels:
            level_strengths = [level.strength for level in grid.grid_levels]
            avg_level_strength = np.mean(level_strengths)
        else:
            avg_level_strength = 0.0
            
        # Harmonic contribution
        harmonic_contribution = grid.harmonic_resonance
        
        # Market interaction efficiency
        total_touches = sum(sq.price_touches for sq in grid.grid_squares)
        interaction_efficiency = min(total_touches / len(grid.grid_squares), 1.0)
        
        # Combine all strength components
        overall_strength = (avg_square_strength * 0.3 +
                          avg_level_strength * 0.3 +
                          harmonic_contribution * 0.2 +
                          interaction_efficiency * 0.2)
        
        return min(overall_strength, 1.0)        
    def _perform_harmonic_analysis(self, df: pd.DataFrame, grids: List[GannGrid]) -> Dict:
        """Perform comprehensive harmonic analysis on grids"""
        
        harmonic_data = {
            'grid_harmonics': [],
            'harmonic_patterns': [],
            'resonance_zones': [],
            'harmonic_strength': 0.0,
            'dominant_frequencies': []
        }
        
        if not grids:
            return harmonic_data
            
        # Analyze each grid's harmonic properties
        all_frequencies = []
        grid_harmonic_data = []
        
        for grid in grids:
            grid_harmonics = self._analyze_grid_harmonics(df, grid)
            grid_harmonic_data.append(grid_harmonics)
            
            # Collect frequencies
            frequencies = grid_harmonics.get('frequencies', [])
            all_frequencies.extend(frequencies)
            
        harmonic_data['grid_harmonics'] = grid_harmonic_data
        
        # Find harmonic patterns across grids
        if len(grids) > 1:
            harmonic_patterns = self._find_harmonic_patterns(grids)
            harmonic_data['harmonic_patterns'] = harmonic_patterns
            
        # Identify resonance zones
        resonance_zones = self._identify_resonance_zones(df, grids, all_frequencies)
        harmonic_data['resonance_zones'] = resonance_zones
        
        # Calculate overall harmonic strength
        if all_frequencies:
            harmonic_data['harmonic_strength'] = self._calculate_overall_harmonic_strength(all_frequencies)
            harmonic_data['dominant_frequencies'] = self._find_dominant_frequencies(all_frequencies)
            
        return harmonic_data
        
    def _analyze_grid_harmonics(self, df: pd.DataFrame, grid: GannGrid) -> Dict:
        """Analyze harmonic properties of a single grid"""
        
        harmonics = {
            'grid_id': grid.id,
            'frequencies': [],
            'harmonic_ratios': [],
            'resonance_strength': 0.0,
            'harmonic_phase': 0.0
        }
        
        # Collect frequencies from squares
        frequencies = []
        for square in grid.grid_squares:
            if square.active and square.harmonic_frequency > 0:
                frequencies.append(square.harmonic_frequency)
                
        harmonics['frequencies'] = frequencies
        
        if len(frequencies) < 2:
            return harmonics
            
        # Calculate harmonic ratios
        harmonic_ratios = []
        for i in range(len(frequencies)):
            for j in range(i + 1, len(frequencies)):
                ratio = frequencies[i] / frequencies[j] if frequencies[j] != 0 else 0
                harmonic_ratios.append(ratio)
                
        harmonics['harmonic_ratios'] = harmonic_ratios
        
        # Calculate resonance strength
        freq_mean = np.mean(frequencies)
        freq_std = np.std(frequencies)
        
        # Resonance is stronger when frequencies are harmonically related
        resonance_strength = freq_mean / (1.0 + freq_std) if freq_std > 0 else freq_mean
        harmonics['resonance_strength'] = min(resonance_strength, 1.0)
        
        # Calculate harmonic phase
        if SCIPY_AVAILABLE and len(frequencies) > 5:
            try:
                # FFT analysis for phase calculation
                fft_result = np.fft.fft(frequencies)
                phase = np.angle(fft_result[1])  # Phase of fundamental frequency
                harmonics['harmonic_phase'] = phase
            except:
                harmonics['harmonic_phase'] = 0.0
                
        return harmonics
        
    def _find_harmonic_patterns(self, grids: List[GannGrid]) -> List[Dict]:
        """Find harmonic patterns across multiple grids"""
        
        patterns = []
        
        # Compare grids pairwise for harmonic relationships
        for i, grid1 in enumerate(grids[:-1]):
            for j, grid2 in enumerate(grids[i+1:], i+1):
                pattern = self._analyze_grid_harmonic_relationship(grid1, grid2)
                if pattern['significance'] > 0.5:
                    patterns.append(pattern)
                    
        # Find multi-grid harmonic clusters
        if len(grids) > 2:
            cluster_patterns = self._find_harmonic_clusters(grids)
            patterns.extend(cluster_patterns)
            
        return patterns
        
    def _analyze_grid_harmonic_relationship(self, grid1: GannGrid, grid2: GannGrid) -> Dict:
        """Analyze harmonic relationship between two grids"""
        
        relationship = {
            'grid1_id': grid1.id,
            'grid2_id': grid2.id,
            'harmonic_correlation': 0.0,
            'frequency_ratio': 0.0,
            'phase_relationship': 0.0,
            'significance': 0.0
        }
        
        # Get primary frequencies
        freq1 = grid1.harmonic_resonance if grid1.harmonic_resonance > 0 else 0.1
        freq2 = grid2.harmonic_resonance if grid2.harmonic_resonance > 0 else 0.1
        
        # Calculate frequency ratio
        frequency_ratio = freq1 / freq2 if freq2 != 0 else 0
        relationship['frequency_ratio'] = frequency_ratio
        
        # Check for harmonic relationships (octaves, fifths, etc.)
        harmonic_ratios = [0.5, 2.0, 1.5, 0.667, 1.25, 0.8, 1.618, 0.618]  # Common harmonic ratios
        
        harmonic_correlation = 0.0
        for target_ratio in harmonic_ratios:
            if abs(frequency_ratio - target_ratio) < 0.1:
                harmonic_correlation = 1.0 - abs(frequency_ratio - target_ratio) / 0.1
                break
                
        relationship['harmonic_correlation'] = harmonic_correlation
        
        # Phase relationship (if both grids have phase information)
        # This would require more complex analysis, simplified here
        phase_relationship = abs(grid1.overall_strength - grid2.overall_strength)
        relationship['phase_relationship'] = 1.0 - phase_relationship
        
        # Overall significance
        significance = (harmonic_correlation * 0.6 + 
                       relationship['phase_relationship'] * 0.4)
        relationship['significance'] = significance
        
        return relationship
        
    def _find_harmonic_clusters(self, grids: List[GannGrid]) -> List[Dict]:
        """Find clusters of harmonically related grids"""
        
        clusters = []
        
        if not ML_AVAILABLE or len(grids) < 3:
            return clusters
            
        try:
            # Extract features for clustering
            features = []
            grid_ids = []
            
            for grid in grids:
                if grid.harmonic_resonance > 0:
                    features.append([
                        grid.harmonic_resonance,
                        grid.overall_strength,
                        len(grid.grid_squares),
                        grid.grid_angle
                    ])
                    grid_ids.append(grid.id)
                    
            if len(features) < 3:
                return clusters
                
            # Perform clustering
            features_array = np.array(features)
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features_array)
            
            # Use DBSCAN for clustering
            clustering = DBSCAN(eps=0.5, min_samples=2)
            cluster_labels = clustering.fit_predict(scaled_features)
            
            # Create cluster information
            unique_labels = set(cluster_labels)
            for label in unique_labels:
                if label != -1:  # -1 is noise in DBSCAN
                    cluster_indices = np.where(cluster_labels == label)[0]
                    cluster_grids = [grid_ids[i] for i in cluster_indices]
                    
                    if len(cluster_grids) >= 2:
                        cluster = {
                            'cluster_id': f"harmonic_cluster_{label}",
                            'grid_ids': cluster_grids,
                            'cluster_size': len(cluster_grids),
                            'harmonic_coherence': self._calculate_cluster_coherence(
                                [grids[i] for i in cluster_indices]
                            )
                        }
                        clusters.append(cluster)
                        
        except Exception as e:
            self.logger.warning(f"Harmonic clustering failed: {e}")
            
        return clusters
        
    def _calculate_cluster_coherence(self, cluster_grids: List[GannGrid]) -> float:
        """Calculate harmonic coherence of a grid cluster"""
        
        if len(cluster_grids) < 2:
            return 0.0
            
        # Calculate pairwise harmonic correlations
        correlations = []
        
        for i, grid1 in enumerate(cluster_grids[:-1]):
            for j, grid2 in enumerate(cluster_grids[i+1:], i+1):
                relationship = self._analyze_grid_harmonic_relationship(grid1, grid2)
                correlations.append(relationship['harmonic_correlation'])
                
        return np.mean(correlations) if correlations else 0.0
        
    def _identify_resonance_zones(self, df: pd.DataFrame, grids: List[GannGrid], 
                                frequencies: List[float]) -> List[Dict]:
        """Identify zones of harmonic resonance"""
        
        resonance_zones = []
        
        if not frequencies:
            return resonance_zones
            
        # Find frequency clusters
        if len(frequencies) > 5 and ML_AVAILABLE:
            try:
                # Cluster similar frequencies
                freq_array = np.array(frequencies).reshape(-1, 1)
                kmeans = KMeans(n_clusters=min(5, len(frequencies)//2), random_state=42)
                freq_clusters = kmeans.fit_predict(freq_array)
                
                # Create resonance zones from clusters
                unique_clusters = set(freq_clusters)
                for cluster_id in unique_clusters:
                    cluster_indices = np.where(freq_clusters == cluster_id)[0]
                    cluster_frequencies = [frequencies[i] for i in cluster_indices]
                    
                    if len(cluster_frequencies) >= 2:
                        zone = {
                            'zone_id': f"resonance_zone_{cluster_id}",
                            'frequencies': cluster_frequencies,
                            'center_frequency': np.mean(cluster_frequencies),
                            'frequency_bandwidth': max(cluster_frequencies) - min(cluster_frequencies),
                            'strength': len(cluster_frequencies) / len(frequencies),
                            'grid_contributors': self._find_zone_contributors(grids, cluster_frequencies)
                        }
                        resonance_zones.append(zone)
                        
            except Exception as e:
                self.logger.warning(f"Resonance zone identification failed: {e}")
                
        return resonance_zones
        
    def _find_zone_contributors(self, grids: List[GannGrid], zone_frequencies: List[float]) -> List[str]:
        """Find which grids contribute to a resonance zone"""
        
        contributors = []
        tolerance = 0.1
        
        for grid in grids:
            for square in grid.grid_squares:
                if square.active and square.harmonic_frequency > 0:
                    for zone_freq in zone_frequencies:
                        if abs(square.harmonic_frequency - zone_freq) < tolerance:
                            if grid.id not in contributors:
                                contributors.append(grid.id)
                            break
                            
        return contributors
        
    def _calculate_overall_harmonic_strength(self, frequencies: List[float]) -> float:
        """Calculate overall harmonic strength across all grids"""
        
        if not frequencies:
            return 0.0
            
        # Frequency distribution analysis
        freq_mean = np.mean(frequencies)
        freq_std = np.std(frequencies)
        
        # Harmonic strength is higher when frequencies are well-distributed
        # but not too scattered
        distribution_score = freq_mean / (1.0 + freq_std) if freq_std > 0 else freq_mean
        
        # Number of frequencies contributes to strength
        count_score = min(len(frequencies) / 20.0, 1.0)
        
        # Overall harmonic strength
        harmonic_strength = (distribution_score * 0.7 + count_score * 0.3)
        
        return min(harmonic_strength, 1.0)
        
    def _find_dominant_frequencies(self, frequencies: List[float]) -> List[Dict]:
        """Find dominant frequencies in the harmonic spectrum"""
        
        if not frequencies:
            return []
            
        # Create frequency histogram
        freq_bins = np.linspace(min(frequencies), max(frequencies), 20)
        hist, bin_edges = np.histogram(frequencies, bins=freq_bins)
        
        # Find peaks in histogram
        dominant_frequencies = []
        
        if SCIPY_AVAILABLE:
            try:
                peaks, properties = find_peaks(hist, height=1, distance=2)
                
                for peak_idx in peaks:
                    freq_center = (bin_edges[peak_idx] + bin_edges[peak_idx + 1]) / 2
                    frequency_count = hist[peak_idx]
                    
                    dominant_frequencies.append({
                        'frequency': freq_center,
                        'count': int(frequency_count),
                        'dominance': frequency_count / len(frequencies),
                        'bin_index': peak_idx
                    })
                    
                # Sort by dominance
                dominant_frequencies.sort(key=lambda x: x['dominance'], reverse=True)
                
            except Exception as e:
                self.logger.warning(f"Dominant frequency analysis failed: {e}")
                
        return dominant_frequencies[:5]  # Return top 5
        
    def _generate_confluence_zones(self, df: pd.DataFrame, grids: List[GannGrid]) -> List[Dict]:
        """Generate confluence zones where multiple grid elements intersect"""
        
        confluence_zones = []
        
        if len(grids) < 2:
            return confluence_zones
            
        # Collect all grid levels from all grids
        all_levels = []
        for grid in grids:
            for level in grid.grid_levels:
                if level.validation_score > 0.5:
                    all_levels.append({
                        'level': level,
                        'grid_id': grid.id,
                        'level_type': level.level_type,
                        'level_value': level.level_value,
                        'orientation': level.orientation,
                        'strength': level.strength
                    })
                    
        # Find intersections between levels
        intersections = self._find_level_intersections(all_levels)
        
        # Convert intersections to confluence zones
        for intersection in intersections:
            if intersection['participant_count'] >= 2:
                zone = {
                    'zone_id': f"confluence_{len(confluence_zones)}",
                    'center_time': intersection['intersection_time'],
                    'center_price': intersection['intersection_price'],
                    'participant_levels': intersection['participants'],
                    'participant_count': intersection['participant_count'],
                    'strength': intersection['combined_strength'],
                    'zone_type': intersection['zone_type'],
                    'reliability': self._calculate_confluence_reliability(df, intersection)
                }
                
                if zone['reliability'] > 0.4:
                    confluence_zones.append(zone)
                    
        # Sort by strength
        confluence_zones.sort(key=lambda x: x['strength'], reverse=True)
        
        # Limit number of zones
        return confluence_zones[:self.config.max_grid_levels]
        
    def _find_level_intersections(self, levels: List[Dict]) -> List[Dict]:
        """Find intersections between grid levels"""
        
        intersections = []
        
        # Test all pairs of levels for intersections
        for i, level1 in enumerate(levels[:-1]):
            for j, level2 in enumerate(levels[i+1:], i+1):
                intersection = self._calculate_level_intersection(level1, level2)
                if intersection:
                    intersections.append(intersection)
                    
        # Group nearby intersections
        grouped_intersections = self._group_nearby_intersections(intersections)
        
        return grouped_intersections
        
    def _calculate_level_intersection(self, level1: Dict, level2: Dict) -> Optional[Dict]:
        """Calculate intersection between two grid levels"""
        
        l1 = level1['level']
        l2 = level2['level']
        
        # Skip if same level
        if l1.id == l2.id:
            return None
            
        intersection = None
        
        # Horizontal-Vertical intersection
        if (l1.level_type == "horizontal" and l2.level_type == "vertical") or \
           (l1.level_type == "vertical" and l2.level_type == "horizontal"):
            
            h_level = l1 if l1.level_type == "horizontal" else l2
            v_level = l1 if l1.level_type == "vertical" else l2
            
            intersection = {
                'intersection_time': v_level.level_value,
                'intersection_price': h_level.level_value,
                'participants': [level1, level2],
                'participant_count': 2,
                'combined_strength': l1.strength + l2.strength,
                'zone_type': 'horizontal_vertical'
            }
            
        # Diagonal intersections (more complex)
        elif l1.level_type == "diagonal" or l2.level_type == "diagonal":
            diagonal_intersection = self._calculate_diagonal_intersection(level1, level2)
            if diagonal_intersection:
                intersection = diagonal_intersection
                
        return intersection
        
    def _calculate_diagonal_intersection(self, level1: Dict, level2: Dict) -> Optional[Dict]:
        """Calculate intersection involving diagonal levels"""
        
        l1 = level1['level']
        l2 = level2['level']
        
        # Handle diagonal-horizontal intersections
        if l1.level_type == "diagonal" and l2.level_type == "horizontal":
            return self._diagonal_horizontal_intersection(level1, level2)
        elif l1.level_type == "horizontal" and l2.level_type == "diagonal":
            return self._diagonal_horizontal_intersection(level2, level1)
            
        # Handle diagonal-vertical intersections
        elif l1.level_type == "diagonal" and l2.level_type == "vertical":
            return self._diagonal_vertical_intersection(level1, level2)
        elif l1.level_type == "vertical" and l2.level_type == "diagonal":
            return self._diagonal_vertical_intersection(level2, level1)
            
        # Handle diagonal-diagonal intersections
        elif l1.level_type == "diagonal" and l2.level_type == "diagonal":
            return self._diagonal_diagonal_intersection(level1, level2)
            
        return None
        
    def _diagonal_horizontal_intersection(self, diag_level: Dict, horiz_level: Dict) -> Optional[Dict]:
        """Calculate intersection between diagonal and horizontal levels"""
        
        diag = diag_level['level']
        horiz = horiz_level['level']
        
        if diag.orientation == 90:  # Vertical diagonal
            return None
            
        # Diagonal line equation: price = level_value + slope * time
        angle_rad = np.radians(diag.orientation)
        slope = np.tan(angle_rad)
        
        # Horizontal line: price = horiz.level_value
        # Intersection: horiz.level_value = diag.level_value + slope * time
        # Solve for time: time = (horiz.level_value - diag.level_value) / slope
        
        if slope != 0:
            intersection_time = (horiz.level_value - diag.level_value) / slope
            intersection_price = horiz.level_value
            
            return {
                'intersection_time': intersection_time,
                'intersection_price': intersection_price,
                'participants': [diag_level, horiz_level],
                'participant_count': 2,
                'combined_strength': diag.strength + horiz.strength,
                'zone_type': 'diagonal_horizontal'
            }
            
        return None
        
    def _diagonal_vertical_intersection(self, diag_level: Dict, vert_level: Dict) -> Optional[Dict]:
        """Calculate intersection between diagonal and vertical levels"""
        
        diag = diag_level['level']
        vert = vert_level['level']
        
        if diag.orientation == 90:  # Vertical diagonal, same as vertical level
            return None
            
        # Diagonal line equation: price = level_value + slope * time
        angle_rad = np.radians(diag.orientation)
        slope = np.tan(angle_rad)
        
        # Vertical line: time = vert.level_value
        # Intersection price: price = diag.level_value + slope * vert.level_value
        
        intersection_time = vert.level_value
        intersection_price = diag.level_value + slope * vert.level_value
        
        return {
            'intersection_time': intersection_time,
            'intersection_price': intersection_price,
            'participants': [diag_level, vert_level],
            'participant_count': 2,
            'combined_strength': diag.strength + vert.strength,
            'zone_type': 'diagonal_vertical'
        }
        
    def _diagonal_diagonal_intersection(self, diag1_level: Dict, diag2_level: Dict) -> Optional[Dict]:
        """Calculate intersection between two diagonal levels"""
        
        diag1 = diag1_level['level']
        diag2 = diag2_level['level']
        
        # Skip if same orientation
        if abs(diag1.orientation - diag2.orientation) < 1:
            return None
            
        # Handle vertical diagonals
        if diag1.orientation == 90 and diag2.orientation == 90:
            return None
            
        # Line equations
        angle1_rad = np.radians(diag1.orientation)
        angle2_rad = np.radians(diag2.orientation)
        
        if diag1.orientation == 90:
            # diag1 is vertical: time = diag1.level_value
            slope2 = np.tan(angle2_rad)
            intersection_time = diag1.level_value
            intersection_price = diag2.level_value + slope2 * intersection_time
        elif diag2.orientation == 90:
            # diag2 is vertical: time = diag2.level_value
            slope1 = np.tan(angle1_rad)
            intersection_time = diag2.level_value
            intersection_price = diag1.level_value + slope1 * intersection_time
        else:
            # Both diagonal
            slope1 = np.tan(angle1_rad)
            slope2 = np.tan(angle2_rad)
            
            # Solve system of equations:
            # price = diag1.level_value + slope1 * time
            # price = diag2.level_value + slope2 * time
            
            if abs(slope1 - slope2) < 1e-10:  # Parallel lines
                return None
                
            # Intersection time: (diag2.level_value - diag1.level_value) / (slope1 - slope2)
            intersection_time = (diag2.level_value - diag1.level_value) / (slope1 - slope2)
            intersection_price = diag1.level_value + slope1 * intersection_time
            
        return {
            'intersection_time': intersection_time,
            'intersection_price': intersection_price,
            'participants': [diag1_level, diag2_level],
            'participant_count': 2,
            'combined_strength': diag1.strength + diag2.strength,
            'zone_type': 'diagonal_diagonal'
        }        
    def _group_nearby_intersections(self, intersections: List[Dict]) -> List[Dict]:
        """Group intersections that are close together"""
        
        if not intersections:
            return []
            
        grouped = []
        used_indices = set()
        
        for i, intersection in enumerate(intersections):
            if i in used_indices:
                continue
                
            # Start a new group
            group = {
                'intersection_time': intersection['intersection_time'],
                'intersection_price': intersection['intersection_price'],
                'participants': intersection['participants'].copy(),
                'participant_count': intersection['participant_count'],
                'combined_strength': intersection['combined_strength'],
                'zone_type': intersection['zone_type']
            }
            
            used_indices.add(i)
            
            # Find nearby intersections to group
            for j, other_intersection in enumerate(intersections):
                if j in used_indices or j <= i:
                    continue
                    
                # Calculate distance
                time_diff = abs(other_intersection['intersection_time'] - group['intersection_time'])
                price_diff = abs(other_intersection['intersection_price'] - group['intersection_price'])
                
                # Grouping tolerance
                time_tolerance = 86400 * 2  # 2 days
                price_tolerance = 100  # 100 price units (adjust based on instrument)
                
                if time_diff < time_tolerance and price_diff < price_tolerance:
                    # Add to group
                    group['participants'].extend(other_intersection['participants'])
                    group['participant_count'] += other_intersection['participant_count']
                    group['combined_strength'] += other_intersection['combined_strength']
                    
                    # Update center (weighted average)
                    total_strength = group['combined_strength']
                    group['intersection_time'] = (
                        (group['intersection_time'] * group['combined_strength'] +
                         other_intersection['intersection_time'] * other_intersection['combined_strength']) /
                        total_strength
                    )
                    group['intersection_price'] = (
                        (group['intersection_price'] * group['combined_strength'] +
                         other_intersection['intersection_price'] * other_intersection['combined_strength']) /
                        total_strength
                    )
                    
                    used_indices.add(j)
                    
            grouped.append(group)
            
        return grouped
        
    def _calculate_confluence_reliability(self, df: pd.DataFrame, intersection: Dict) -> float:
        """Calculate reliability of a confluence zone"""
        
        reliability = 0.0
        
        # Base reliability from participant count
        participant_bonus = min(intersection['participant_count'] / 5.0, 1.0)
        reliability += participant_bonus * 0.3
        
        # Strength-based reliability
        strength_score = min(intersection['combined_strength'], 1.0)
        reliability += strength_score * 0.4
        
        # Market validation (if intersection is in historical data)
        intersection_time = intersection['intersection_time']
        intersection_price = intersection['intersection_price']
        
        # Find closest data point
        if len(df) > 0:
            time_distances = abs(df['time_numeric'] - intersection_time)
            
            if len(time_distances) > 0:
                closest_idx = time_distances.idxmin()
                
                if closest_idx in df.index:
                    closest_row = df.loc[closest_idx]
                    
                    # Check if price was close to intersection
                    price_tolerance = closest_row['atr'] * 0.5
                    actual_prices = [closest_row['high'], closest_row['low'], 
                                   closest_row['close'], closest_row['open']]
                    
                    min_distance = min(abs(p - intersection_price) for p in actual_prices)
                    
                    if min_distance < price_tolerance:
                        market_validation = 1.0 - (min_distance / price_tolerance)
                        reliability += market_validation * 0.3
                        
        return min(reliability, 1.0)
        
    def _ml_validate_grids(self, df: pd.DataFrame, grids: List[GannGrid]) -> List[GannGrid]:
        """Validate grids using machine learning techniques"""
        
        if not self.config.ml_validation or not ML_AVAILABLE:
            return grids
            
        validated_grids = []
        
        try:
            # Prepare features for ML validation
            features = self._extract_grid_features(df, grids)
            
            if len(features) < 5:  # Need minimum data for ML
                return grids
                
            # Anomaly detection
            if self.config.anomaly_detection and 'anomaly_detector' in self.ml_models:
                anomaly_scores = self._detect_grid_anomalies(features)
            else:
                anomaly_scores = [1.0] * len(grids)
                
            # Strength prediction
            if 'strength_predictor' in self.ml_models:
                predicted_strengths = self._predict_grid_strengths(features)
            else:
                predicted_strengths = [grid.overall_strength for grid in grids]
                
            # Apply ML validation results
            for i, grid in enumerate(grids):
                if i < len(anomaly_scores) and i < len(predicted_strengths):
                    # Adjust grid strength based on ML predictions
                    ml_adjustment = (anomaly_scores[i] + predicted_strengths[i]) / 2
                    grid.overall_strength = (grid.overall_strength + ml_adjustment) / 2
                    
                    # Mark grid as validated if it passes thresholds
                    if (anomaly_scores[i] > 0.3 and 
                        predicted_strengths[i] > 0.3 and 
                        grid.overall_strength > 0.4):
                        validated_grids.append(grid)
                        
        except Exception as e:
            self.logger.warning(f"ML validation failed: {e}")
            return grids
            
        return validated_grids if validated_grids else grids
        
    def _extract_grid_features(self, df: pd.DataFrame, grids: List[GannGrid]) -> np.ndarray:
        """Extract features from grids for ML analysis"""
        
        features = []
        
        for grid in grids:
            grid_features = [
                grid.overall_strength,
                grid.harmonic_resonance,
                len(grid.grid_squares),
                len(grid.grid_levels),
                grid.total_squares,
                grid.active_squares,
                grid.grid_angle,
                grid.price_interaction_count,
                grid.support_resistance_efficiency
            ]
            
            # Add square-based features
            if grid.grid_squares:
                square_strengths = [sq.strength for sq in grid.grid_squares]
                square_validations = [sq.validation_score for sq in grid.grid_squares]
                square_touches = [sq.price_touches for sq in grid.grid_squares]
                
                grid_features.extend([
                    np.mean(square_strengths),
                    np.std(square_strengths),
                    np.mean(square_validations),
                    np.std(square_validations),
                    np.mean(square_touches),
                    max(square_touches) if square_touches else 0
                ])
            else:
                grid_features.extend([0.0] * 6)
                
            # Add level-based features
            if grid.grid_levels:
                level_strengths = [level.strength for level in grid.grid_levels]
                level_touches = [level.touches for level in grid.grid_levels]
                
                grid_features.extend([
                    np.mean(level_strengths),
                    np.std(level_strengths),
                    np.mean(level_touches),
                    max(level_touches) if level_touches else 0
                ])
            else:
                grid_features.extend([0.0] * 4)
                
            features.append(grid_features)
            
        return np.array(features)
        
    def _detect_grid_anomalies(self, features: np.ndarray) -> List[float]:
        """Detect anomalous grids using isolation forest"""
        
        try:
            anomaly_detector = self.ml_models['anomaly_detector']
            
            # Scale features
            scaled_features = self.scalers['standard'].fit_transform(features)
            
            # Predict anomalies (-1 for anomalies, 1 for normal)
            anomaly_predictions = anomaly_detector.fit_predict(scaled_features)
            
            # Convert to scores (0-1, higher is better)
            anomaly_scores = [(pred + 1) / 2 for pred in anomaly_predictions]
            
            return anomaly_scores
            
        except Exception as e:
            self.logger.warning(f"Anomaly detection failed: {e}")
            return [1.0] * len(features)
            
    def _predict_grid_strengths(self, features: np.ndarray) -> List[float]:
        """Predict grid strengths using ML model"""
        
        try:
            # Use current strength as target for training (simplified)
            current_strengths = features[:, 0]  # First feature is overall_strength
            
            if len(features) < 5:
                return current_strengths.tolist()
                
            # Scale features
            scaled_features = self.scalers['minmax'].fit_transform(features)
            
            # Simple prediction based on feature combination
            # In a real implementation, this would use historical training data
            feature_weights = np.array([0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05])
            
            if scaled_features.shape[1] >= len(feature_weights):
                predicted_strengths = np.dot(scaled_features[:, :len(feature_weights)], feature_weights)
                # Normalize to 0-1 range
                predicted_strengths = np.clip(predicted_strengths, 0, 1)
                return predicted_strengths.tolist()
            else:
                return current_strengths.tolist()
                
        except Exception as e:
            self.logger.warning(f"Strength prediction failed: {e}")
            return [0.5] * len(features)
            
    def _extract_grid_support_resistance(self, df: pd.DataFrame, grids: List[GannGrid]) -> Dict:
        """Extract support and resistance levels from grids"""
        
        sr_levels = {
            'support_levels': [],
            'resistance_levels': [],
            'dynamic_levels': [],
            'confluence_levels': [],
            'key_levels': []
        }
        
        current_price = df['close'].iloc[-1]
        
        # Extract levels from all grids
        all_levels = []
        
        for grid in grids:
            for level in grid.grid_levels:
                if level.validation_score > 0.5:
                    level_data = {
                        'level': level.level_value,
                        'strength': level.strength,
                        'type': level.level_type,
                        'orientation': level.orientation,
                        'source': f"grid_{grid.id}",
                        'touches': level.touches,
                        'reliability': level.reliability,
                        'distance_to_current': abs(level.level_value - current_price) if level.level_type == "horizontal" else np.inf
                    }
                    all_levels.append(level_data)
                    
        # Cluster similar levels
        if all_levels:
            clustered_levels = self._cluster_grid_sr_levels(all_levels, df['atr'].iloc[-1])
            
            # Categorize levels
            for cluster in clustered_levels:
                center_level = cluster['center_level']
                level_type = cluster['dominant_type']
                
                if level_type == "horizontal":
                    if center_level < current_price:
                        sr_levels['support_levels'].append(cluster)
                    else:
                        sr_levels['resistance_levels'].append(cluster)
                elif level_type in ["vertical", "diagonal"]:
                    sr_levels['dynamic_levels'].append(cluster)
                    
            # Identify key levels (highest strength)
            all_clusters = clustered_levels.copy()
            all_clusters.sort(key=lambda x: x['total_strength'], reverse=True)
            sr_levels['key_levels'] = all_clusters[:20]  # Top 20 levels
            
        return sr_levels
        
    def _cluster_grid_sr_levels(self, levels: List[Dict], atr: float) -> List[Dict]:
        """Cluster similar support/resistance levels from grids"""
        
        if not levels:
            return []
            
        # Group by level type first
        horizontal_levels = [l for l in levels if l['type'] == 'horizontal']
        other_levels = [l for l in levels if l['type'] != 'horizontal']
        
        clusters = []
        
        # Cluster horizontal levels by price
        if horizontal_levels:
            sorted_levels = sorted(horizontal_levels, key=lambda x: x['level'])
            current_cluster = [sorted_levels[0]]
            tolerance = atr * 0.6
            
            for level in sorted_levels[1:]:
                cluster_center = np.mean([l['level'] for l in current_cluster])
                
                if abs(level['level'] - cluster_center) <= tolerance:
                    current_cluster.append(level)
                else:
                    if current_cluster:
                        clusters.append(self._create_grid_sr_cluster(current_cluster))
                    current_cluster = [level]
                    
            if current_cluster:
                clusters.append(self._create_grid_sr_cluster(current_cluster))
                
        # Add other levels as individual clusters
        for level in other_levels:
            clusters.append(self._create_grid_sr_cluster([level]))
            
        return clusters
        
    def _create_grid_sr_cluster(self, levels: List[Dict]) -> Dict:
        """Create support/resistance cluster from grid levels"""
        
        level_values = [l['level'] for l in levels]
        strengths = [l['strength'] for l in levels]
        types = [l['type'] for l in levels]
        touches = [l['touches'] for l in levels]
        reliabilities = [l['reliability'] for l in levels]
        
        center_level = np.mean(level_values)
        total_strength = sum(strengths)
        total_touches = sum(touches)
        level_count = len(levels)
        
        # Determine dominant type
        type_counts = {}
        for t in types:
            type_counts[t] = type_counts.get(t, 0) + 1
            
        dominant_type = max(type_counts.keys(), key=type_counts.get)
        
        # Calculate cluster reliability
        avg_reliability = np.mean(reliabilities) if reliabilities else 0.5
        strength_consistency = 1.0 - (np.std(strengths) / np.mean(strengths)) if strengths and np.mean(strengths) > 0 else 0.5
        
        overall_reliability = (avg_reliability + strength_consistency) / 2
        
        return {
            'center_level': center_level,
            'total_strength': total_strength,
            'level_count': level_count,
            'dominant_type': dominant_type,
            'total_touches': total_touches,
            'reliability': overall_reliability,
            'level_range': max(level_values) - min(level_values) if len(level_values) > 1 else 0,
            'component_levels': levels
        }
        
    def _analyze_grid_interactions(self, df: pd.DataFrame, grids: List[GannGrid]) -> Dict:
        """Analyze interactions between different grids"""
        
        interaction_data = {
            'grid_correlations': [],
            'strength_distribution': {},
            'harmonic_relationships': [],
            'grid_efficiency': 0.0
        }
        
        if len(grids) < 2:
            return interaction_data
            
        # Analyze correlations between grids
        for i, grid1 in enumerate(grids[:-1]):
            for j, grid2 in enumerate(grids[i+1:], i+1):
                correlation = self._calculate_grid_correlation(grid1, grid2)
                if correlation['significance'] > 0.3:
                    interaction_data['grid_correlations'].append(correlation)
                    
        # Analyze strength distribution
        strengths = [grid.overall_strength for grid in grids]
        harmonic_resonances = [grid.harmonic_resonance for grid in grids]
        
        interaction_data['strength_distribution'] = {
            'mean_strength': np.mean(strengths),
            'std_strength': np.std(strengths),
            'max_strength': max(strengths),
            'min_strength': min(strengths),
            'strong_grids': len([s for s in strengths if s > 0.7]),
            'mean_harmonic': np.mean(harmonic_resonances),
            'harmonic_consistency': 1.0 - (np.std(harmonic_resonances) / np.mean(harmonic_resonances)) if np.mean(harmonic_resonances) > 0 else 0.5
        }
        
        # Calculate overall grid efficiency
        total_squares = sum(grid.total_squares for grid in grids)
        active_squares = sum(grid.active_squares for grid in grids)
        
        if total_squares > 0:
            interaction_data['grid_efficiency'] = active_squares / total_squares
        else:
            interaction_data['grid_efficiency'] = 0.0
            
        return interaction_data
        
    def _calculate_grid_correlation(self, grid1: GannGrid, grid2: GannGrid) -> Dict:
        """Calculate correlation between two grids"""
        
        # Strength correlation
        strength_similarity = 1.0 - abs(grid1.overall_strength - grid2.overall_strength)
        
        # Harmonic correlation
        harmonic_similarity = 1.0 - abs(grid1.harmonic_resonance - grid2.harmonic_resonance)
        
        # Geometric correlation (angle similarity)
        angle_diff = abs(grid1.grid_angle - grid2.grid_angle)
        angle_similarity = 1.0 - min(angle_diff, 180 - angle_diff) / 90.0
        
        # Spatial correlation (distance between origins)
        time_distance = abs(grid1.origin_time - grid2.origin_time)
        price_distance = abs(grid1.origin_price - grid2.origin_price)
        
        # Normalize spatial distance (simplified)
        max_time_distance = 86400 * 30  # 30 days
        max_price_distance = grid1.origin_price * 0.1  # 10% of price
        
        time_proximity = 1.0 - min(time_distance / max_time_distance, 1.0)
        price_proximity = 1.0 - min(price_distance / max_price_distance, 1.0) if max_price_distance > 0 else 0.5
        
        spatial_correlation = (time_proximity + price_proximity) / 2
        
        # Overall significance
        significance = (strength_similarity * 0.3 + 
                       harmonic_similarity * 0.3 + 
                       angle_similarity * 0.2 + 
                       spatial_correlation * 0.2)
        
        return {
            'grid1_id': grid1.id,
            'grid2_id': grid2.id,
            'strength_similarity': strength_similarity,
            'harmonic_similarity': harmonic_similarity,
            'angle_similarity': angle_similarity,
            'spatial_correlation': spatial_correlation,
            'significance': significance
        }
        
    def _generate_grid_signals(self, df: pd.DataFrame, grids: List[GannGrid], sr_levels: Dict) -> pd.Series:
        """Generate trading signals based on grid analysis"""
        
        signals = pd.Series(0, index=df.index)  # 0 = neutral, 1 = buy, -1 = sell
        
        if not grids:
            return signals
            
        current_price = df['close'].iloc[-1]
        current_time = df['time_numeric'].iloc[-1]
        
        signal_strength = 0.0
        signal_count = 0
        
        # Analyze each grid for signals
        for grid in grids:
            grid_signal = self._generate_grid_signal(df, grid, current_price, current_time, sr_levels)
            if grid_signal != 0:
                signal_strength += grid_signal * grid.overall_strength
                signal_count += 1
                
        # Confluence zone signals
        confluence_signal = self._generate_confluence_signal(df, sr_levels, current_price, current_time)
        if confluence_signal != 0:
            signal_strength += confluence_signal * 1.5  # Boost confluence signals
            signal_count += 1
            
        # Apply signal threshold
        if signal_count > 0:
            avg_signal = signal_strength / signal_count
            
            if avg_signal > self.config.signal_threshold:
                signals.iloc[-1] = 1  # Buy
            elif avg_signal < -self.config.signal_threshold:
                signals.iloc[-1] = -1  # Sell
            else:
                signals.iloc[-1] = 0  # Neutral
                
        return signals
        
    def _generate_grid_signal(self, df: pd.DataFrame, grid: GannGrid, current_price: float, 
                            current_time: float, sr_levels: Dict) -> float:
        """Generate signal for individual grid"""
        
        signal = 0.0
        
        # Check grid levels for signal generation
        for level in grid.grid_levels:
            if level.validation_score < 0.5:
                continue
                
            level_signal = self._calculate_level_signal(level, current_price, current_time)
            signal += level_signal * level.strength
            
        # Check grid squares for pattern signals
        for square in grid.grid_squares:
            if not square.active or square.validation_score < 0.4:
                continue
                
            square_signal = self._calculate_square_signal(square, current_price, current_time)
            signal += square_signal * square.strength * 0.5  # Lower weight for squares
            
        # Normalize signal
        return max(-1.0, min(1.0, signal))
        
    def _calculate_level_signal(self, level: GannGridLevel, current_price: float, current_time: float) -> float:
        """Calculate signal from individual grid level"""
        
        signal = 0.0
        
        if level.level_type == "horizontal":
            distance = abs(current_price - level.level_value)
            
            # Signal based on support/resistance logic
            if current_price > level.level_value:
                # Above level (potential support)
                signal = 0.3 if distance < level.level_value * 0.01 else 0.1
            else:
                # Below level (potential resistance)
                signal = -0.3 if distance < level.level_value * 0.01 else -0.1
                
        elif level.level_type == "vertical":
            time_distance = abs(current_time - level.level_value)
            
            # Time-based signals (reversal or continuation points)
            if time_distance < 3600:  # Within 1 hour
                signal = 0.2  # Neutral time signal
                
        elif level.level_type == "diagonal":
            # Calculate position relative to diagonal line
            if level.orientation != 90:
                angle_rad = np.radians(level.orientation)
                slope = np.tan(angle_rad)
                expected_price = level.level_value + slope * current_time
                
                if current_price > expected_price:
                    signal = 0.2  # Above diagonal (potential support)
                else:
                    signal = -0.2  # Below diagonal (potential resistance)
                    
        # Adjust based on level strength
        signal *= level.strength
        
        # Adjust based on touches (more touches = stronger level)
        if level.touches > 3:
            signal *= 1.2
        elif level.touches == 0:
            signal *= 0.5
            
        return signal
        
    def _calculate_square_signal(self, square: GannGridSquare, current_price: float, current_time: float) -> float:
        """Calculate signal from grid square"""
        
        signal = 0.0
        
        # Check if current price/time is within square
        time_in_square = (square.center_time - square.width_time/2 <= current_time <= 
                         square.center_time + square.width_time/2)
        price_in_square = (square.center_price - square.height_price/2 <= current_price <= 
                          square.center_price + square.height_price/2)
        
        if time_in_square and price_in_square:
            # Inside square - signal based on square type and energy
            energy_signal = min(square.energy_level / 50.0, 1.0)  # Normalize energy
            
            if square.support_resistance_type == "support":
                signal = energy_signal * 0.4
            elif square.support_resistance_type == "resistance":
                signal = -energy_signal * 0.4
            else:
                signal = energy_signal * 0.1  # Neutral energy signal
                
        else:
            # Outside square - check proximity
            time_distance = abs(current_time - square.center_time) / square.width_time
            price_distance = abs(current_price - square.center_price) / square.height_price
            
            total_distance = np.sqrt(time_distance**2 + price_distance**2)
            
            if total_distance < 0.5:  # Close to square
                proximity_signal = (0.5 - total_distance) * 2  # 0 to 1
                signal = proximity_signal * 0.1  # Low weight for proximity
                
        return signal
        
    def _generate_confluence_signal(self, df: pd.DataFrame, sr_levels: Dict, 
                                  current_price: float, current_time: float) -> float:
        """Generate signal from confluence zones"""
        
        signal = 0.0
        
        confluence_levels = sr_levels.get('confluence_levels', [])
        
        for confluence in confluence_levels:
            if confluence['reliability'] < 0.5:
                continue
                
            # Check proximity to confluence zone
            if confluence['dominant_type'] == "horizontal":
                distance = abs(current_price - confluence['center_level'])
                tolerance = confluence['center_level'] * 0.005  # 0.5%
                
                if distance < tolerance:
                    # Close to confluence level
                    confluence_strength = confluence['total_strength'] * confluence['reliability']
                    
                    if current_price > confluence['center_level']:
                        signal += confluence_strength * 0.5  # Support signal
                    else:
                        signal -= confluence_strength * 0.5  # Resistance signal
                        
        return max(-1.0, min(1.0, signal))
        
    def _combine_grid_results(self, df: pd.DataFrame, grids: List[GannGrid], sr_levels: Dict,
                            confluence_zones: List[Dict], harmonic_analysis: Dict,
                            interaction_analysis: Dict, signals: pd.Series) -> pd.DataFrame:
        """Combine all grid analysis results into final output DataFrame"""
        
        result = df.copy()
        
        current_price = df['close'].iloc[-1]
        
        # Grid statistics
        result['gann_grid_count'] = len(grids)
        result['gann_strong_grids'] = len([g for g in grids if g.overall_strength > 0.7])
        
        if grids:
            avg_grid_strength = np.mean([g.overall_strength for g in grids])
            avg_harmonic_resonance = np.mean([g.harmonic_resonance for g in grids])
            total_squares = sum(g.total_squares for g in grids)
            active_squares = sum(g.active_squares for g in grids)
            
            result['gann_avg_grid_strength'] = avg_grid_strength
            result['gann_avg_harmonic_resonance'] = avg_harmonic_resonance
            result['gann_total_squares'] = total_squares
            result['gann_active_squares'] = active_squares
            result['gann_square_efficiency'] = active_squares / total_squares if total_squares > 0 else 0.0
        else:
            result['gann_avg_grid_strength'] = 0.0
            result['gann_avg_harmonic_resonance'] = 0.0
            result['gann_total_squares'] = 0
            result['gann_active_squares'] = 0
            result['gann_square_efficiency'] = 0.0
            
        # Support/resistance levels
        key_levels = sr_levels.get('key_levels', [])
        if key_levels:
            # Find closest horizontal level
            horizontal_levels = [l for l in key_levels if l['dominant_type'] == 'horizontal']
            if horizontal_levels:
                closest_level = min(horizontal_levels, key=lambda x: abs(x['center_level'] - current_price))
                result['gann_closest_level'] = closest_level['center_level']
                result['gann_closest_level_strength'] = closest_level['total_strength']
                result['gann_closest_level_type'] = closest_level['dominant_type']
                result['gann_distance_to_level'] = abs(current_price - closest_level['center_level']) / df['atr'].iloc[-1]
            else:
                result['gann_closest_level'] = np.nan
                result['gann_closest_level_strength'] = 0.0
                result['gann_closest_level_type'] = 'none'
                result['gann_distance_to_level'] = np.nan
        else:
            result['gann_closest_level'] = np.nan
            result['gann_closest_level_strength'] = 0.0
            result['gann_closest_level_type'] = 'none'
            result['gann_distance_to_level'] = np.nan
            
        # Confluence analysis
        result['gann_confluence_zones'] = len(confluence_zones)
        
        if confluence_zones:
            strongest_confluence = max(confluence_zones, key=lambda x: x['strength'])
            result['gann_strongest_confluence_strength'] = strongest_confluence['strength']
            result['gann_confluence_participant_count'] = strongest_confluence['participant_count']
        else:
            result['gann_strongest_confluence_strength'] = 0.0
            result['gann_confluence_participant_count'] = 0
            
        # Harmonic analysis
        result['gann_harmonic_strength'] = harmonic_analysis.get('harmonic_strength', 0.0)
        result['gann_resonance_zones'] = len(harmonic_analysis.get('resonance_zones', []))
        
        dominant_frequencies = harmonic_analysis.get('dominant_frequencies', [])
        if dominant_frequencies:
            result['gann_dominant_frequency'] = dominant_frequencies[0]['frequency']
            result['gann_frequency_dominance'] = dominant_frequencies[0]['dominance']
        else:
            result['gann_dominant_frequency'] = 0.0
            result['gann_frequency_dominance'] = 0.0
            
        # Interaction analysis
        grid_efficiency = interaction_analysis.get('grid_efficiency', 0.0)
        result['gann_grid_efficiency'] = grid_efficiency
        
        strength_dist = interaction_analysis.get('strength_distribution', {})
        result['gann_strength_consistency'] = 1.0 - (strength_dist.get('std_strength', 1.0) / 
                                                    strength_dist.get('mean_strength', 1.0)) if strength_dist.get('mean_strength', 0) > 0 else 0.5
        
        # Trading signals
        result['gann_grid_signal'] = signals
        
        # Market structure assessment
        result['gann_market_structure'] = self._assess_grid_market_structure(grids, sr_levels, current_price)
        
        return result
        
    def _assess_grid_market_structure(self, grids: List[GannGrid], sr_levels: Dict, current_price: float) -> str:
        """Assess market structure based on grid analysis"""
        
        if not grids:
            return 'undefined'
            
        # Analyze grid orientations and strengths
        strong_grids = [g for g in grids if g.overall_strength > 0.6]
        
        if not strong_grids:
            return 'weak_structure'
            
        # Analyze support/resistance balance
        support_strength = sum(
            level['total_strength'] for level in sr_levels.get('support_levels', [])
        )
        resistance_strength = sum(
            level['total_strength'] for level in sr_levels.get('resistance_levels', [])
        )
        
        # Analyze harmonic coherence
        harmonic_coherences = [g.harmonic_resonance for g in strong_grids]
        avg_harmonic_coherence = np.mean(harmonic_coherences) if harmonic_coherences else 0.0
        
        # Classification logic
        if avg_harmonic_coherence > 0.7:
            if support_strength > resistance_strength * 1.5:
                return 'harmonically_supported'
            elif resistance_strength > support_strength * 1.5:
                return 'harmonically_resisted'
            else:
                return 'harmonic_equilibrium'
        elif support_strength > resistance_strength * 1.2:
            return 'grid_supported'
        elif resistance_strength > support_strength * 1.2:
            return 'grid_resisted'
        elif len(strong_grids) >= 3:
            return 'multi_grid_structure'
        else:
            return 'simple_grid_structure'
            
    def _create_error_result(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create error result DataFrame with default values"""
        
        result = df.copy()
        
        # Set all grid indicators to default/error values
        grid_columns = [
            'gann_grid_count', 'gann_strong_grids', 'gann_avg_grid_strength',
            'gann_avg_harmonic_resonance', 'gann_total_squares', 'gann_active_squares',
            'gann_square_efficiency', 'gann_closest_level', 'gann_closest_level_strength',
            'gann_closest_level_type', 'gann_distance_to_level', 'gann_confluence_zones',
            'gann_strongest_confluence_strength', 'gann_confluence_participant_count',
            'gann_harmonic_strength', 'gann_resonance_zones', 'gann_dominant_frequency',
            'gann_frequency_dominance', 'gann_grid_efficiency', 'gann_strength_consistency',
            'gann_grid_signal', 'gann_market_structure'
        ]
        
        for col in grid_columns:
            if col in ['gann_closest_level_type', 'gann_market_structure']:
                result[col] = 'error'
            elif col == 'gann_grid_signal':
                result[col] = 0
            elif col in ['gann_closest_level', 'gann_distance_to_level']:
                result[col] = np.nan
            else:
                result[col] = 0.0
                
        return result


def create_gann_grid_indicator(config: Optional[GannGridConfig] = None) -> GannGridIndicator:
    """Factory function to create GannGridIndicator instance"""
    return GannGridIndicator(config)


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
    config = GannGridConfig(
        min_grid_size=15,
        max_grid_size=50,
        dynamic_spacing=True,
        harmonic_analysis=True,
        ml_validation=True,
        confluence_analysis=True
    )
    
    indicator = GannGridIndicator(config)
    
    try:
        # Calculate Gann grids
        result = indicator.calculate(data)
        
        print("Gann Grid Calculation Results:")
        print(f"Data shape: {result.shape}")
        print(f"Columns: {list(result.columns)}")
        
        # Display recent signals
        recent = result.tail(5)
        for col in ['gann_grid_count', 'gann_grid_signal', 'gann_market_structure', 'gann_closest_level_type']:
            if col in recent.columns:
                print(f"\n{col}:")
                print(recent[col].to_string())
                
        # Display grid statistics
        grid_count = recent['gann_grid_count'].iloc[-1]
        strong_grids = recent['gann_strong_grids'].iloc[-1]
        grid_efficiency = recent['gann_grid_efficiency'].iloc[-1]
        
        print(f"\nGrid Statistics:")
        print(f"Total grids: {grid_count}")
        print(f"Strong grids: {strong_grids}")
        print(f"Grid efficiency: {grid_efficiency:.3f}")
                
    except Exception as e:
        print(f"Error testing Gann Grid indicator: {e}")
        import traceback
        traceback.print_exc()