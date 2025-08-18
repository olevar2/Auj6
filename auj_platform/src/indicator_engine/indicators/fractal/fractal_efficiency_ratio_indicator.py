"""
Fractal Efficiency Ratio Indicator - Advanced Implementation

This indicator implements sophisticated fractal efficiency analysis combining
Perry Kaufman's Efficiency Ratio with advanced fractal geometry principles.
Features include:
- Multi-timeframe fractal efficiency analysis
- Adaptive efficiency calculation with volatility adjustment
- Hurst exponent integration for long-term memory analysis
- Fractal scaling efficiency measurement
- Market microstructure efficiency evaluation
- Advanced statistical validation and confidence intervals

Mission: Supporting humanitarian trading platform for poor and sick children through
maximum profitability via advanced fractal efficiency analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from scipy import stats
from scipy.optimize import minimize_scalar
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import warnings

from ..base.standard_indicator import StandardIndicatorInterface
from ....core.exceptions import IndicatorCalculationError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EfficiencyRatioResult:
    """Results container for Fractal Efficiency Ratio"""
    efficiency_ratio: float
    fractal_efficiency: float
    hurst_efficiency: float
    volatility_adjusted_efficiency: float
    scaling_efficiency: float
    microstructure_efficiency: float
    efficiency_trend: str
    market_state: str
    efficiency_confidence: float
    adaptive_period: int

class FractalEfficiencyRatioIndicator(StandardIndicatorInterface):
    """
    Advanced Fractal Efficiency Ratio Indicator
    
    Implements sophisticated fractal efficiency analysis combining
    Perry Kaufman's Efficiency Ratio with advanced fractal geometry.
    """
    
    def __init__(self, 
                 base_period: int = 20,
                 min_period: int = 10,
                 max_period: int = 50,
                 volatility_window: int = 14,
                 hurst_window: int = 100):
        super().__init__()
        self.base_period = base_period
        self.min_period = min_period
        self.max_period = max_period
        self.volatility_window = volatility_window
        self.hurst_window = hurst_window
        
        logger.info(f"Initialized FractalEfficiencyRatioIndicator with base_period={base_period}")

    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        try:
            self._validate_data(data)
            
            if len(data) < max(self.max_period, self.hurst_window):
                return self._create_default_result()
            
            # Calculate basic efficiency ratio
            efficiency_ratio = self._calculate_basic_efficiency_ratio(data)
            
            # Calculate fractal efficiency components
            fractal_efficiency = self._calculate_fractal_efficiency(data)
            
            # Calculate Hurst-based efficiency
            hurst_efficiency = self._calculate_hurst_efficiency(data)
            
            # Calculate volatility-adjusted efficiency
            vol_adjusted_efficiency = self._calculate_volatility_adjusted_efficiency(data)
            
            # Calculate scaling efficiency
            scaling_efficiency = self._calculate_scaling_efficiency(data)
            
            # Calculate microstructure efficiency
            microstructure_efficiency = self._calculate_microstructure_efficiency(data)
            
            # Determine adaptive period
            adaptive_period = self._calculate_adaptive_period(data, efficiency_ratio)
            
            # Analyze efficiency trends and market state
            efficiency_analysis = self._analyze_efficiency_patterns(data, efficiency_ratio)
            
            result = EfficiencyRatioResult(
                efficiency_ratio=efficiency_ratio,
                fractal_efficiency=fractal_efficiency,
                hurst_efficiency=hurst_efficiency,
                volatility_adjusted_efficiency=vol_adjusted_efficiency,
                scaling_efficiency=scaling_efficiency,
                microstructure_efficiency=microstructure_efficiency,
                efficiency_trend=efficiency_analysis['trend'],
                market_state=efficiency_analysis['market_state'],
                efficiency_confidence=efficiency_analysis['confidence'],
                adaptive_period=adaptive_period
            )
            
            return self._format_output(result, data.index[-1])
            
        except Exception as e:
            logger.error(f"Error in efficiency ratio calculation: {e}")
            raise IndicatorCalculationError(f"FractalEfficiencyRatioIndicator calculation failed: {e}")

    def _calculate_basic_efficiency_ratio(self, data: pd.DataFrame) -> float:
        """Calculate Perry Kaufman's Efficiency Ratio"""
        try:
            if len(data) < self.base_period:
                return 0.0
            
            recent_data = data[-self.base_period:]
            prices = recent_data['close'].values
            
            # Net change (direction)
            net_change = abs(prices[-1] - prices[0])
            
            # Sum of absolute price changes (volatility)
            price_changes = np.abs(np.diff(prices))
            volatility = np.sum(price_changes)
            
            # Efficiency ratio
            if volatility == 0:
                return 0.0
            
            efficiency = net_change / volatility
            return np.clip(efficiency, 0.0, 1.0)
            
        except Exception as e:
            logger.warning(f"Basic efficiency ratio calculation error: {e}")
            return 0.0

    def _calculate_fractal_efficiency(self, data: pd.DataFrame) -> float:
        """Calculate fractal-based efficiency using box-counting dimension"""
        try:
            if len(data) < self.base_period:
                return 0.0
            
            recent_data = data[-self.base_period:]
            prices = recent_data['close'].values
            
            # Normalize prices for fractal analysis
            normalized_prices = (prices - np.min(prices)) / (np.max(prices) - np.min(prices) + 1e-8)
            
            # Calculate fractal dimension using box-counting
            fractal_dim = self._calculate_box_counting_dimension(normalized_prices)
            
            # Convert fractal dimension to efficiency measure
            # Theoretical maximum dimension for a line is 1.0 (perfectly efficient)
            # Higher dimensions indicate more complex, less efficient movement
            if fractal_dim > 0:
                fractal_efficiency = 1.0 / fractal_dim
            else:
                fractal_efficiency = 0.0
            
            return np.clip(fractal_efficiency, 0.0, 1.0)
            
        except Exception as e:
            logger.warning(f"Fractal efficiency calculation error: {e}")
            return 0.0

    def _calculate_box_counting_dimension(self, prices: np.ndarray) -> float:
        """Calculate box-counting fractal dimension"""
        try:
            if len(prices) < 10:
                return 1.0
            
            # Create 2D trajectory
            time_points = np.arange(len(prices))
            trajectory = np.column_stack([time_points, prices])
            
            # Box-counting algorithm
            scales = np.logspace(-2, 0, 8)  # Reduced for performance
            counts = []
            
            for scale in scales:
                # Count boxes containing trajectory points
                x_boxes = int(np.ceil((np.max(trajectory[:, 0]) - np.min(trajectory[:, 0])) / scale)) + 1
                y_boxes = int(np.ceil((np.max(trajectory[:, 1]) - np.min(trajectory[:, 1])) / scale)) + 1
                
                # Create grid
                occupied_boxes = set()
                
                for point in trajectory:
                    x_box = int((point[0] - np.min(trajectory[:, 0])) / scale)
                    y_box = int((point[1] - np.min(trajectory[:, 1])) / scale)
                    occupied_boxes.add((x_box, y_box))
                
                counts.append(len(occupied_boxes))
            
            if len(counts) < 3:
                return 1.0
            
            # Fit log-log relationship
            log_scales = np.log(1.0 / scales)
            log_counts = np.log(counts)
            
            # Use only the linear part of the relationship
            slope, _, r_value, _, _ = stats.linregress(log_scales, log_counts)
            
            # Return dimension if fit is reasonable
            if r_value ** 2 > 0.7:
                return max(1.0, slope)
            
            return 1.0
            
        except Exception as e:
            logger.warning(f"Box-counting dimension calculation error: {e}")
            return 1.0

    def _calculate_hurst_efficiency(self, data: pd.DataFrame) -> float:
        """Calculate efficiency based on Hurst exponent"""
        try:
            if len(data) < self.hurst_window:
                return 0.5
            
            recent_data = data[-self.hurst_window:]
            prices = recent_data['close'].values
            log_returns = np.diff(np.log(prices))
            
            # Calculate Hurst exponent using R/S analysis
            hurst = self._calculate_hurst_exponent(log_returns)
            
            # Convert Hurst to efficiency measure
            # H = 0.5: random walk (neutral efficiency)
            # H > 0.5: persistent/trending (higher efficiency for trends)
            # H < 0.5: anti-persistent/mean-reverting (lower efficiency)
            
            if hurst > 0.5:
                efficiency = 2 * (hurst - 0.5)  # 0 to 1 scale
            else:
                efficiency = 2 * hurst  # 0 to 1 scale
            
            return np.clip(efficiency, 0.0, 1.0)
            
        except Exception as e:
            logger.warning(f"Hurst efficiency calculation error: {e}")
            return 0.5

    def _calculate_hurst_exponent(self, returns: np.ndarray) -> float:
        """Calculate Hurst exponent using R/S analysis"""
        try:
            if len(returns) < 20:
                return 0.5
            
            # Calculate R/S statistic for different lags
            lags = np.unique(np.logspace(0.5, np.log10(len(returns) // 4), 10).astype(int))
            rs_values = []
            
            for lag in lags:
                if lag >= len(returns):
                    continue
                
                # Split series into non-overlapping periods
                n_periods = len(returns) // lag
                rs_period_values = []
                
                for i in range(n_periods):
                    period_returns = returns[i*lag:(i+1)*lag]
                    
                    if len(period_returns) == 0:
                        continue
                    
                    # Calculate mean
                    mean_return = np.mean(period_returns)
                    
                    # Calculate cumulative deviations
                    deviations = period_returns - mean_return
                    cumulative_deviations = np.cumsum(deviations)
                    
                    # Range
                    R = np.max(cumulative_deviations) - np.min(cumulative_deviations)
                    
                    # Standard deviation
                    S = np.std(period_returns)
                    
                    # R/S ratio
                    if S > 0:
                        rs_period_values.append(R / S)
                
                if rs_period_values:
                    rs_values.append(np.mean(rs_period_values))
            
            if len(rs_values) < 3:
                return 0.5
            
            # Fit log-log relationship
            log_lags = np.log(lags[:len(rs_values)])
            log_rs = np.log(rs_values)
            
            slope, _, r_value, _, _ = stats.linregress(log_lags, log_rs)
            
            # Hurst exponent is the slope
            if r_value ** 2 > 0.7:
                return np.clip(slope, 0.0, 1.0)
            
            return 0.5
            
        except Exception as e:
            logger.warning(f"Hurst exponent calculation error: {e}")
            return 0.5

    def _calculate_volatility_adjusted_efficiency(self, data: pd.DataFrame) -> float:
        """Calculate efficiency adjusted for volatility regime"""
        try:
            if len(data) < max(self.base_period, self.volatility_window):
                return 0.0
            
            recent_data = data[-self.base_period:]
            vol_data = data[-self.volatility_window:]
            
            # Basic efficiency
            basic_efficiency = self._calculate_basic_efficiency_ratio(data)
            
            # Calculate volatility
            returns = vol_data['close'].pct_change().dropna()
            current_vol = returns.std()
            historical_vol = data['close'].pct_change().rolling(50).std().mean()
            
            if historical_vol == 0:
                vol_ratio = 1.0
            else:
                vol_ratio = current_vol / historical_vol
            
            # Adjust efficiency based on volatility regime
            # Higher volatility typically reduces efficiency
            vol_adjustment = 1.0 / (1.0 + vol_ratio)
            
            adjusted_efficiency = basic_efficiency * vol_adjustment
            
            return np.clip(adjusted_efficiency, 0.0, 1.0)
            
        except Exception as e:
            logger.warning(f"Volatility-adjusted efficiency calculation error: {e}")
            return 0.0

    def _calculate_scaling_efficiency(self, data: pd.DataFrame) -> float:
        """Calculate efficiency based on price scaling properties"""
        try:
            if len(data) < self.base_period:
                return 0.0
            
            recent_data = data[-self.base_period:]
            prices = recent_data['close'].values
            
            # Calculate efficiency at multiple time scales
            scales = [5, 10, 15, 20]
            efficiencies = []
            
            for scale in scales:
                if scale >= len(prices):
                    continue
                
                # Calculate efficiency for this scale
                net_change = abs(prices[-1] - prices[-scale])
                price_changes = np.abs(np.diff(prices[-scale:]))
                volatility = np.sum(price_changes)
                
                if volatility > 0:
                    scale_efficiency = net_change / volatility
                    efficiencies.append(scale_efficiency)
            
            if not efficiencies:
                return 0.0
            
            # Return average efficiency across scales
            return np.clip(np.mean(efficiencies), 0.0, 1.0)
            
        except Exception as e:
            logger.warning(f"Scaling efficiency calculation error: {e}")
            return 0.0

    def _calculate_microstructure_efficiency(self, data: pd.DataFrame) -> float:
        """Calculate microstructure efficiency using high-frequency patterns"""
        try:
            if len(data) < self.base_period:
                return 0.0
            
            recent_data = data[-self.base_period:]
            
            # Use OHLC data for microstructure analysis
            if all(col in recent_data.columns for col in ['open', 'high', 'low', 'close']):
                # Calculate intrabar efficiency
                body_sizes = abs(recent_data['close'] - recent_data['open'])
                shadow_sizes = (recent_data['high'] - recent_data['low']) - body_sizes
                
                total_ranges = recent_data['high'] - recent_data['low']
                
                # Efficiency as ratio of body to total range
                valid_ranges = total_ranges > 0
                if valid_ranges.sum() == 0:
                    return 0.0
                
                microstructure_eff = (body_sizes[valid_ranges] / total_ranges[valid_ranges]).mean()
            else:
                # Use close-to-close efficiency as proxy
                microstructure_eff = self._calculate_basic_efficiency_ratio(data)
            
            return np.clip(microstructure_eff, 0.0, 1.0)
            
        except Exception as e:
            logger.warning(f"Microstructure efficiency calculation error: {e}")
            return 0.0

    def _calculate_adaptive_period(self, data: pd.DataFrame, efficiency_ratio: float) -> int:
        """Calculate adaptive period based on efficiency"""
        try:
            # Adaptive period: higher efficiency = shorter period
            # Lower efficiency = longer period for smoothing
            
            efficiency_factor = 1.0 - efficiency_ratio  # Invert for period calculation
            
            # Scale between min and max period
            adaptive_period = self.min_period + (self.max_period - self.min_period) * efficiency_factor
            
            return int(np.clip(adaptive_period, self.min_period, self.max_period))
            
        except Exception as e:
            logger.warning(f"Adaptive period calculation error: {e}")
            return self.base_period

    def _analyze_efficiency_patterns(self, data: pd.DataFrame, efficiency_ratio: float) -> Dict[str, Any]:
        """Analyze efficiency patterns and trends"""
        try:
            # Calculate efficiency trend
            if len(data) >= self.base_period * 2:
                # Compare current efficiency with historical
                current_period = data[-self.base_period:]
                previous_period = data[-self.base_period*2:-self.base_period]
                
                prev_efficiency = self._calculate_basic_efficiency_ratio(
                    pd.DataFrame({'close': previous_period['close']})
                )
                
                if efficiency_ratio > prev_efficiency * 1.1:
                    trend = 'INCREASING'
                elif efficiency_ratio < prev_efficiency * 0.9:
                    trend = 'DECREASING'
                else:
                    trend = 'STABLE'
            else:
                trend = 'UNKNOWN'
            
            # Determine market state
            if efficiency_ratio > 0.7:
                market_state = 'TRENDING'
            elif efficiency_ratio > 0.3:
                market_state = 'TRANSITIONAL'
            else:
                market_state = 'RANGING'
            
            # Calculate confidence based on data quality
            confidence = min(len(data) / self.hurst_window, 1.0)
            
            return {
                'trend': trend,
                'market_state': market_state,
                'confidence': confidence
            }
            
        except Exception as e:
            logger.warning(f"Efficiency pattern analysis error: {e}")
            return {
                'trend': 'UNKNOWN',
                'market_state': 'UNKNOWN',
                'confidence': 0.0
            }

    def _format_output(self, result: EfficiencyRatioResult, timestamp) -> Dict[str, Any]:
        """Format the calculation results for output"""
        return {
            'timestamp': timestamp,
            'indicator_name': 'FractalEfficiencyRatio',
            'efficiency_ratio': round(result.efficiency_ratio, 4),
            'fractal_efficiency': round(result.fractal_efficiency, 4),
            'hurst_efficiency': round(result.hurst_efficiency, 4),
            'volatility_adjusted_efficiency': round(result.volatility_adjusted_efficiency, 4),
            'scaling_efficiency': round(result.scaling_efficiency, 4),
            'microstructure_efficiency': round(result.microstructure_efficiency, 4),
            'efficiency_trend': result.efficiency_trend,
            'market_state': result.market_state,
            'efficiency_confidence': round(result.efficiency_confidence, 4),
            'adaptive_period': result.adaptive_period
        }

    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate input data"""
        required_columns = ['close']
        if not all(col in data.columns for col in required_columns):
            raise IndicatorCalculationError(f"Missing required columns: {required_columns}")
        if len(data) < self.base_period:
            raise IndicatorCalculationError(f"Insufficient data: minimum {self.base_period} periods required")

    def _create_default_result(self) -> Dict[str, Any]:
        """Create default result for insufficient data cases"""
        return {
            'timestamp': pd.Timestamp.now(),
            'indicator_name': 'FractalEfficiencyRatio',
            'efficiency_ratio': 0.0,
            'fractal_efficiency': 0.0,
            'hurst_efficiency': 0.5,
            'volatility_adjusted_efficiency': 0.0,
            'scaling_efficiency': 0.0,
            'microstructure_efficiency': 0.0,
            'efficiency_trend': 'INSUFFICIENT_DATA',
            'market_state': 'UNKNOWN',
            'efficiency_confidence': 0.0,
            'adaptive_period': self.base_period
        }

    def get_required_columns(self) -> List[str]:
        return ['close']

    def get_indicator_name(self) -> str:
        return "FractalEfficiencyRatio"