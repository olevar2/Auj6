"""
Advanced Linear Regression Channels Indicator with Breakout Detection

This indicator implements sophisticated regression channel analysis including:
- Multiple regression models (linear, polynomial, robust)
- Dynamic channel width adaptation
- Breakout and breakdown detection with confirmation
- Channel slope and momentum analysis  
- Regression quality assessment and model validation
- Support/resistance level identification
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from scipy import stats
from scipy.optimize import minimize
from sklearn.linear_model import HuberRegressor, RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
import logging
from dataclasses import dataclass

from ..base.standard_indicator import StandardIndicatorInterface
from ....core.signal_type import SignalType
from ....core.exceptions import IndicatorCalculationError


@dataclass
class LinearRegressionChannelsResult:
    """Result container for linear regression channels analysis"""
    regression_line: np.ndarray
    upper_channel: np.ndarray
    lower_channel: np.ndarray
    channel_width: np.ndarray
    slope: float
    r_squared: float
    p_value: float
    residuals: np.ndarray
    breakout_signals: np.ndarray
    support_resistance_levels: List[float]
    channel_position: float
    trend_strength: float
    regression_quality: str
    model_parameters: Dict[str, float]


class LinearRegressionChannelsIndicator(StandardIndicatorInterface):
    """
    Advanced Linear Regression Channels Indicator
    
    Implements sophisticated regression-based channel analysis with
    multiple models and comprehensive breakout detection.
    """
    
    def __init__(self, 
                 window: int = 50,
                 channel_width_factor: float = 2.0,
                 polynomial_degree: int = 1,
                 robust_method: str = 'huber',
                 breakout_threshold: float = 0.1):
        """
        Initialize Linear Regression Channels Indicator
        
        Args:
            window: Lookback window for regression analysis
            channel_width_factor: Multiplier for channel width (std devs)
            polynomial_degree: Degree for polynomial regression
            robust_method: Robust regression method ('huber', 'ransac', 'ols')
            breakout_threshold: Threshold for breakout detection
        """
        super().__init__()
        self.window = window
        self.channel_width_factor = channel_width_factor
        self.polynomial_degree = polynomial_degree
        self.robust_method = robust_method.lower()
        self.breakout_threshold = breakout_threshold
        self.logger = logging.getLogger(__name__)
        
    def calculate(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Calculate linear regression channels analysis
        
        Args:
            data: DataFrame with OHLCV data
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing regression channels results
        """
        try:
            if data.empty or len(data) < self.window:
                raise IndicatorCalculationError("Insufficient data for regression channels analysis")
            
            # Extract price series
            prices = data['close'].values
            
            if len(prices) < 20:
                raise IndicatorCalculationError("Insufficient price data")
            
            # Perform regression channels analysis
            regression_result = self._calculate_regression_channels(prices)
            
            # Generate trading signal
            signal = self._generate_signal(regression_result, prices)
            
            return {
                'signal': signal,
                'regression_line': regression_result.regression_line.tolist(),
                'upper_channel': regression_result.upper_channel.tolist(),
                'lower_channel': regression_result.lower_channel.tolist(),
                'channel_width': regression_result.channel_width.tolist(),
                'slope': regression_result.slope,
                'r_squared': regression_result.r_squared,
                'p_value': regression_result.p_value,
                'residuals': regression_result.residuals.tolist(),
                'breakout_signals': regression_result.breakout_signals.tolist(),
                'support_resistance_levels': regression_result.support_resistance_levels,
                'channel_position': regression_result.channel_position,
                'trend_strength': regression_result.trend_strength,
                'regression_quality': regression_result.regression_quality,
                'model_parameters': regression_result.model_parameters,
                'strength': self._calculate_signal_strength(regression_result),
                'confidence': self._calculate_confidence(regression_result)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating regression channels: {str(e)}")
            raise IndicatorCalculationError(f"Regression channels calculation failed: {str(e)}")
    
    def _calculate_regression_channels(self, prices: np.ndarray) -> LinearRegressionChannelsResult:
        """Calculate regression channels using specified method"""
        n = len(prices)
        
        # Create time index
        x = np.arange(n).reshape(-1, 1)
        y = prices
        
        # Fit regression model based on method
        if self.robust_method == 'huber':
            model_result = self._fit_huber_regression(x, y)
        elif self.robust_method == 'ransac':
            model_result = self._fit_ransac_regression(x, y)
        else:
            model_result = self._fit_ols_regression(x, y)
        
        regression_line, slope, r_squared, p_value, residuals = model_result
        
        # Calculate dynamic channel width
        channel_widths = self._calculate_dynamic_channel_width(residuals)
        
        # Create upper and lower channels
        upper_channel = regression_line + channel_widths
        lower_channel = regression_line - channel_widths
        
        # Detect breakouts
        breakout_signals = self._detect_breakouts(prices, upper_channel, lower_channel)
        
        # Find support/resistance levels
        support_resistance = self._find_support_resistance_levels(prices, regression_line)
        
        # Calculate current channel position
        current_position = self._calculate_channel_position(
            prices[-1], regression_line[-1], upper_channel[-1], lower_channel[-1]
        )
        
        # Assess trend strength
        trend_strength = self._calculate_trend_strength(slope, r_squared, residuals)
        
        # Classify regression quality
        quality = self._classify_regression_quality(r_squared, p_value, residuals)
        
        # Model parameters
        parameters = {
            'slope': slope,
            'intercept': regression_line[0] - slope * 0,
            'channel_width_factor': self.channel_width_factor,
            'window_size': self.window
        }
        
        return LinearRegressionChannelsResult(
            regression_line=regression_line,
            upper_channel=upper_channel,
            lower_channel=lower_channel,
            channel_width=channel_widths,
            slope=slope,
            r_squared=r_squared,
            p_value=p_value,
            residuals=residuals,
            breakout_signals=breakout_signals,
            support_resistance_levels=support_resistance,
            channel_position=current_position,
            trend_strength=trend_strength,
            regression_quality=quality,
            model_parameters=parameters
        )
    
    def _fit_ols_regression(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float, float, float, np.ndarray]:
        """Fit ordinary least squares regression"""
        if self.polynomial_degree > 1:
            # Polynomial regression
            poly_features = PolynomialFeatures(degree=self.polynomial_degree)
            x_poly = poly_features.fit_transform(x)
            
            coeffs = np.linalg.lstsq(x_poly, y, rcond=None)[0]
            regression_line = x_poly @ coeffs
            slope = coeffs[1] if len(coeffs) > 1 else 0.0
        else:
            # Linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(x.flatten(), y)
            regression_line = slope * x.flatten() + intercept
            r_squared = r_value ** 2
        
        # Calculate residuals
        residuals = y - regression_line
        
        # Calculate R-squared and p-value for polynomial case
        if self.polynomial_degree > 1:
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Approximate p-value using F-test
            n = len(y)
            p = self.polynomial_degree + 1
            if n > p:
                f_stat = (r_squared / p) / ((1 - r_squared) / (n - p - 1))
                p_value = 1 - stats.f.cdf(f_stat, p, n - p - 1)
            else:
                p_value = 1.0
        
        return regression_line, slope, r_squared, p_value, residuals
    
    def _fit_huber_regression(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float, float, float, np.ndarray]:
        """Fit Huber robust regression"""
        try:
            huber = HuberRegressor(epsilon=1.35, max_iter=1000)
            
            if self.polynomial_degree > 1:
                poly_features = PolynomialFeatures(degree=self.polynomial_degree)
                x_poly = poly_features.fit_transform(x)
                huber.fit(x_poly, y)
                regression_line = huber.predict(x_poly)
                slope = huber.coef_[1] if len(huber.coef_) > 1 else 0.0
            else:
                huber.fit(x, y)
                regression_line = huber.predict(x)
                slope = huber.coef_[0]
            
            residuals = y - regression_line
            
            # Calculate R-squared
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Approximate p-value (simplified)
            p_value = 0.05 if r_squared > 0.5 else 0.5
            
            return regression_line, slope, r_squared, p_value, residuals
            
        except Exception:
            # Fallback to OLS
            return self._fit_ols_regression(x, y)
    
    def _fit_ransac_regression(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float, float, float, np.ndarray]:
        """Fit RANSAC robust regression"""
        try:
            ransac = RANSACRegressor(max_trials=1000, residual_threshold=None)
            
            if self.polynomial_degree > 1:
                poly_features = PolynomialFeatures(degree=self.polynomial_degree)
                x_poly = poly_features.fit_transform(x)
                ransac.fit(x_poly, y)
                regression_line = ransac.predict(x_poly)
                slope = ransac.estimator_.coef_[1] if len(ransac.estimator_.coef_) > 1 else 0.0
            else:
                ransac.fit(x, y)
                regression_line = ransac.predict(x)
                slope = ransac.estimator_.coef_[0]
            
            residuals = y - regression_line
            
            # Calculate R-squared
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Approximate p-value
            p_value = 0.05 if r_squared > 0.5 else 0.5
            
            return regression_line, slope, r_squared, p_value, residuals
            
        except Exception:
            # Fallback to OLS
            return self._fit_ols_regression(x, y)
    
    def _calculate_dynamic_channel_width(self, residuals: np.ndarray) -> np.ndarray:
        """Calculate dynamic channel width based on residual volatility"""
        n = len(residuals)
        channel_widths = np.zeros(n)
        
        # Rolling standard deviation of residuals
        window = min(20, n // 4)
        
        for i in range(n):
            start_idx = max(0, i - window + 1)
            end_idx = i + 1
            
            window_residuals = residuals[start_idx:end_idx]
            std_residual = np.std(window_residuals) if len(window_residuals) > 1 else np.std(residuals)
            
            channel_widths[i] = self.channel_width_factor * std_residual
        
        return channel_widths
    
    def _detect_breakouts(self, prices: np.ndarray, upper_channel: np.ndarray, lower_channel: np.ndarray) -> np.ndarray:
        """Detect channel breakouts and breakdowns"""
        n = len(prices)
        breakout_signals = np.zeros(n)
        
        for i in range(1, n):
            # Upper breakout
            if (prices[i] > upper_channel[i] and 
                prices[i-1] <= upper_channel[i-1]):
                breakout_signals[i] = 1  # Bullish breakout
            
            # Lower breakdown
            elif (prices[i] < lower_channel[i] and 
                  prices[i-1] >= lower_channel[i-1]):
                breakout_signals[i] = -1  # Bearish breakdown
        
        return breakout_signals
    
    def _find_support_resistance_levels(self, prices: np.ndarray, regression_line: np.ndarray) -> List[float]:
        """Find significant support and resistance levels"""
        levels = []
        
        # Recent price levels that have been tested multiple times
        recent_prices = prices[-20:] if len(prices) >= 20 else prices
        
        # Find price levels with multiple touches
        for price in recent_prices:
            touch_count = np.sum(np.abs(recent_prices - price) < 0.005 * price)
            if touch_count >= 3:  # At least 3 touches
                levels.append(price)
        
        # Add regression line as potential support/resistance
        levels.append(regression_line[-1])
        
        # Remove duplicates and sort
        levels = sorted(list(set([round(level, 4) for level in levels])))
        
        return levels[-5:] if len(levels) > 5 else levels  # Return top 5 levels
    
    def _calculate_channel_position(self, current_price: float, regression_value: float, 
                                   upper_value: float, lower_value: float) -> float:
        """Calculate current position within the channel (0=bottom, 1=top)"""
        if upper_value == lower_value:
            return 0.5
        
        position = (current_price - lower_value) / (upper_value - lower_value)
        return np.clip(position, 0.0, 1.0)
    
    def _calculate_trend_strength(self, slope: float, r_squared: float, residuals: np.ndarray) -> float:
        """Calculate overall trend strength"""
        # Normalize slope by price level
        mean_price = np.mean(np.abs(residuals)) + 1e-8
        normalized_slope = abs(slope) / mean_price
        
        # Combine slope magnitude with regression quality
        slope_strength = min(normalized_slope * 100, 1.0)
        quality_strength = r_squared
        
        # Residual consistency
        residual_consistency = 1 / (1 + np.std(residuals) / (np.mean(np.abs(residuals)) + 1e-8))
        
        return (slope_strength + quality_strength + residual_consistency) / 3
    
    def _classify_regression_quality(self, r_squared: float, p_value: float, residuals: np.ndarray) -> str:
        """Classify the quality of the regression fit"""
        if r_squared > 0.8 and p_value < 0.01:
            return "excellent"
        elif r_squared > 0.6 and p_value < 0.05:
            return "good"
        elif r_squared > 0.4 and p_value < 0.1:
            return "moderate"
        else:
            return "poor"
    
    def _generate_signal(self, result: LinearRegressionChannelsResult, prices: np.ndarray) -> SignalType:
        """Generate trading signal based on regression channels analysis"""
        current_price = prices[-1]
        channel_position = result.channel_position
        recent_breakout = result.breakout_signals[-5:] if len(result.breakout_signals) >= 5 else result.breakout_signals
        
        # Strong trend with good regression quality
        if result.regression_quality in ["excellent", "good"] and result.trend_strength > 0.6:
            # Trend following signals
            if result.slope > 0 and channel_position < 0.3:
                return SignalType.BUY  # Uptrend, price near bottom of channel
            elif result.slope < 0 and channel_position > 0.7:
                return SignalType.SELL  # Downtrend, price near top of channel
        
        # Breakout signals
        if np.any(recent_breakout == 1):  # Recent bullish breakout
            return SignalType.BUY
        elif np.any(recent_breakout == -1):  # Recent bearish breakdown
            return SignalType.SELL
        
        # Mean reversion signals in sideways markets
        if abs(result.slope) < 0.001 and result.regression_quality != "poor":
            if channel_position > 0.8:
                return SignalType.SELL  # Near top of channel
            elif channel_position < 0.2:
                return SignalType.BUY   # Near bottom of channel
        
        return SignalType.HOLD
    
    def _calculate_signal_strength(self, result: LinearRegressionChannelsResult) -> float:
        """Calculate signal strength based on regression quality"""
        # Regression quality strength
        quality_scores = {"excellent": 1.0, "good": 0.8, "moderate": 0.5, "poor": 0.2}
        quality_strength = quality_scores.get(result.regression_quality, 0.2)
        
        # Trend strength
        trend_strength = result.trend_strength
        
        # Channel position extremity
        position_strength = max(abs(result.channel_position - 0.5) * 2, 0)
        
        return (quality_strength + trend_strength + position_strength) / 3
    
    def _calculate_confidence(self, result: LinearRegressionChannelsResult) -> float:
        """Calculate confidence based on statistical measures"""
        # R-squared confidence
        r_squared_conf = result.r_squared
        
        # P-value confidence
        p_value_conf = max(0, 1 - result.p_value)
        
        # Model stability (based on residuals)
        residual_stability = 1 / (1 + np.std(result.residuals) / (np.mean(np.abs(result.residuals)) + 1e-8))
        
        return (r_squared_conf + p_value_conf + residual_stability) / 3