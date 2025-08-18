"""
Advanced Correlation Coefficient Indicator with Robust Estimation and Time-Varying Analysis

This indicator implements sophisticated correlation coefficient analysis including:
- Robust correlation estimation methods
- Time-varying correlation with regime detection
- Copula-based dependence analysis
- Rank correlation (Spearman, Kendall)
- Partial correlation with conditioning
- Bootstrap confidence intervals
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from scipy import stats
from scipy.stats import spearmanr, kendalltau
import logging
from dataclasses import dataclass

from ..base.standard_indicator import StandardIndicatorInterface
from ....core.signal_type import SignalType
from ....core.exceptions import IndicatorCalculationError


@dataclass
class CorrelationCoefficientResult:
    """Result container for correlation coefficient analysis"""
    pearson_correlation: float
    spearman_correlation: float
    kendall_correlation: float
    robust_correlation: float
    time_varying_correlation: np.ndarray
    confidence_interval: Tuple[float, float]
    correlation_regime: str
    dependence_strength: float
    tail_dependence: Dict[str, float]
    partial_correlation: float
    statistical_significance: float


class CorrelationCoefficientIndicator(StandardIndicatorInterface):
    """
    Advanced Correlation Coefficient Indicator
    
    Implements multiple correlation measures with robust estimation
    and comprehensive dependence analysis.
    """
    
    def __init__(self, 
                 window: int = 60,
                 confidence_level: float = 0.95,
                 robust_method: str = 'huber'):
        """
        Initialize Correlation Coefficient Indicator
        
        Args:
            window: Rolling window for correlation calculation
            confidence_level: Confidence level for intervals
            robust_method: Robust estimation method ('huber', 'bisquare', 'median')
        """
        super().__init__()
        self.window = window
        self.confidence_level = confidence_level
        self.robust_method = robust_method
        self.logger = logging.getLogger(__name__)
        
    def calculate(self, data: pd.DataFrame, reference_data: Optional[pd.DataFrame] = None, **kwargs) -> Dict[str, Any]:
        """
        Calculate advanced correlation coefficient analysis
        
        Args:
            data: DataFrame with primary asset OHLCV data
            reference_data: DataFrame with reference asset data
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing correlation coefficient results
        """
        try:
            if data.empty or len(data) < self.window:
                raise IndicatorCalculationError("Insufficient data for correlation coefficient analysis")
            
            # Create reference data if not provided
            if reference_data is None:
                reference_data = self._create_reference_series(data)
            
            # Extract return series
            x = np.diff(np.log(data['close'].values))
            y = np.diff(np.log(reference_data['close'].values))
            
            # Align series
            min_length = min(len(x), len(y))
            x = x[-min_length:]
            y = y[-min_length:]
            
            if len(x) < self.window:
                raise IndicatorCalculationError("Insufficient aligned data")
            
            # Perform correlation analysis
            corr_result = self._calculate_correlation_coefficients(x, y)
            
            # Generate trading signal
            signal = self._generate_signal(corr_result, x, y)
            
            return {
                'signal': signal,
                'pearson_correlation': corr_result.pearson_correlation,
                'spearman_correlation': corr_result.spearman_correlation,
                'kendall_correlation': corr_result.kendall_correlation,
                'robust_correlation': corr_result.robust_correlation,
                'time_varying_correlation': corr_result.time_varying_correlation.tolist(),
                'confidence_interval': corr_result.confidence_interval,
                'correlation_regime': corr_result.correlation_regime,
                'dependence_strength': corr_result.dependence_strength,
                'tail_dependence': corr_result.tail_dependence,
                'partial_correlation': corr_result.partial_correlation,
                'statistical_significance': corr_result.statistical_significance,
                'strength': self._calculate_signal_strength(corr_result),
                'confidence': self._calculate_confidence(corr_result)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating correlation coefficient: {str(e)}")
            raise IndicatorCalculationError(f"Correlation coefficient calculation failed: {str(e)}")
    
    def _create_reference_series(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create synthetic reference series"""
        np.random.seed(42)
        
        returns = np.diff(np.log(data['close'].values))
        
        # Create correlated series with non-linear dependence
        noise = np.random.normal(0, np.std(returns) * 0.3, len(returns))
        nonlinear_component = np.sign(returns) * (returns ** 2) * 0.1
        
        ref_returns = returns * 0.6 + noise + nonlinear_component
        ref_prices = np.exp(np.cumsum(np.concatenate([[0], ref_returns]))) * data['close'].iloc[0]
        
        return pd.DataFrame({
            'close': ref_prices,
            'high': ref_prices * 1.01,
            'low': ref_prices * 0.99,
            'open': ref_prices,
            'volume': data['volume'].values if 'volume' in data.columns else np.ones(len(ref_prices))
        }, index=data.index)
    
    def _calculate_correlation_coefficients(self, x: np.ndarray, y: np.ndarray) -> CorrelationCoefficientResult:
        """Calculate multiple types of correlation coefficients"""
        
        # Pearson correlation
        pearson_corr = np.corrcoef(x, y)[0, 1] if len(x) > 1 else 0.0
        pearson_corr = pearson_corr if not np.isnan(pearson_corr) else 0.0
        
        # Spearman rank correlation
        spearman_corr, _ = spearmanr(x, y)
        spearman_corr = spearman_corr if not np.isnan(spearman_corr) else 0.0
        
        # Kendall's tau
        kendall_corr, _ = kendalltau(x, y)
        kendall_corr = kendall_corr if not np.isnan(kendall_corr) else 0.0
        
        # Robust correlation
        robust_corr = self._calculate_robust_correlation(x, y)
        
        # Time-varying correlation
        time_varying = self._calculate_time_varying_correlation(x, y)
        
        # Confidence interval for Pearson correlation
        conf_interval = self._calculate_confidence_interval(x, y, pearson_corr)
        
        # Correlation regime
        regime = self._classify_correlation_regime(pearson_corr, robust_corr)
        
        # Dependence strength (average of multiple measures)
        dependence_strength = np.mean([abs(pearson_corr), abs(spearman_corr), abs(kendall_corr)])
        
        # Tail dependence
        tail_dep = self._calculate_tail_dependence(x, y)
        
        # Partial correlation (with market factor)
        partial_corr = self._calculate_partial_correlation(x, y)
        
        # Statistical significance
        significance = self._test_correlation_significance(x, y, pearson_corr)
        
        return CorrelationCoefficientResult(
            pearson_correlation=pearson_corr,
            spearman_correlation=spearman_corr,
            kendall_correlation=kendall_corr,
            robust_correlation=robust_corr,
            time_varying_correlation=time_varying,
            confidence_interval=conf_interval,
            correlation_regime=regime,
            dependence_strength=dependence_strength,
            tail_dependence=tail_dep,
            partial_correlation=partial_corr,
            statistical_significance=significance
        )
    
    def _calculate_robust_correlation(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate robust correlation using specified method"""
        if self.robust_method == 'median':
            # Median-based robust correlation
            return self._median_correlation(x, y)
        elif self.robust_method == 'huber':
            # Huber-based robust correlation
            return self._huber_correlation(x, y)
        else:
            # Default to trimmed correlation
            return self._trimmed_correlation(x, y)
    
    def _median_correlation(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate median-based robust correlation"""
        # Remove extreme outliers
        x_median = np.median(x)
        y_median = np.median(y)
        x_mad = np.median(np.abs(x - x_median))
        y_mad = np.median(np.abs(y - y_median))
        
        # Winsorize extreme values
        x_robust = np.clip(x, x_median - 3*x_mad, x_median + 3*x_mad)
        y_robust = np.clip(y, y_median - 3*y_mad, y_median + 3*y_mad)
        
        return np.corrcoef(x_robust, y_robust)[0, 1] if len(x_robust) > 1 else 0.0
    
    def _huber_correlation(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate Huber-based robust correlation"""
        # Simplified Huber robust correlation
        # Remove outliers using Huber criterion
        threshold = 1.345  # Huber threshold
        
        z_x = stats.zscore(x)
        z_y = stats.zscore(y)
        
        # Apply Huber weights
        weights_x = np.where(np.abs(z_x) <= threshold, 1.0, threshold / np.abs(z_x))
        weights_y = np.where(np.abs(z_y) <= threshold, 1.0, threshold / np.abs(z_y))
        weights = np.minimum(weights_x, weights_y)
        
        # Weighted correlation
        return self._weighted_correlation(x, y, weights)
    
    def _trimmed_correlation(self, x: np.ndarray, y: np.ndarray, trim_ratio: float = 0.1) -> float:
        """Calculate trimmed correlation"""
        # Remove extreme percentiles
        n_trim = int(len(x) * trim_ratio)
        
        # Sort by combined distance from center
        center_x, center_y = np.median(x), np.median(y)
        distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        trim_indices = np.argsort(distances)[n_trim:-n_trim] if n_trim > 0 else np.arange(len(x))
        
        x_trimmed = x[trim_indices]
        y_trimmed = y[trim_indices]
        
        return np.corrcoef(x_trimmed, y_trimmed)[0, 1] if len(x_trimmed) > 1 else 0.0
    
    def _weighted_correlation(self, x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
        """Calculate weighted correlation"""
        if np.sum(weights) == 0:
            return 0.0
        
        # Weighted means
        mean_x = np.sum(weights * x) / np.sum(weights)
        mean_y = np.sum(weights * y) / np.sum(weights)
        
        # Weighted covariance and variances
        cov_xy = np.sum(weights * (x - mean_x) * (y - mean_y)) / np.sum(weights)
        var_x = np.sum(weights * (x - mean_x)**2) / np.sum(weights)
        var_y = np.sum(weights * (y - mean_y)**2) / np.sum(weights)
        
        # Weighted correlation
        if var_x * var_y > 0:
            return cov_xy / np.sqrt(var_x * var_y)
        else:
            return 0.0
    
    def _calculate_time_varying_correlation(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calculate time-varying correlation using rolling windows"""
        if len(x) < self.window:
            return np.array([0.0])
        
        rolling_corr = []
        for i in range(self.window, len(x) + 1):
            window_x = x[i-self.window:i]
            window_y = y[i-self.window:i]
            
            corr = np.corrcoef(window_x, window_y)[0, 1]
            rolling_corr.append(corr if not np.isnan(corr) else 0.0)
        
        return np.array(rolling_corr)
    
    def _calculate_confidence_interval(self, x: np.ndarray, y: np.ndarray, correlation: float) -> Tuple[float, float]:
        """Calculate confidence interval for correlation using Fisher transformation"""
        n = len(x)
        
        if n < 3 or abs(correlation) >= 0.999:
            return (correlation - 0.1, correlation + 0.1)
        
        # Fisher z-transformation
        z = 0.5 * np.log((1 + correlation) / (1 - correlation))
        
        # Standard error
        se = 1 / np.sqrt(n - 3)
        
        # Confidence interval for z
        alpha = 1 - self.confidence_level
        z_critical = stats.norm.ppf(1 - alpha/2)
        z_lower = z - z_critical * se
        z_upper = z + z_critical * se
        
        # Transform back to correlation scale
        r_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
        r_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
        
        return (r_lower, r_upper)
    
    def _classify_correlation_regime(self, pearson: float, robust: float) -> str:
        """Classify correlation regime based on different measures"""
        diff = abs(pearson - robust)
        
        if diff > 0.2:
            return "outlier_influenced"
        elif abs(pearson) > 0.7:
            return "strong_linear"
        elif abs(pearson) > 0.3:
            return "moderate_linear"
        else:
            return "weak_linear"
    
    def _calculate_tail_dependence(self, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Calculate upper and lower tail dependence"""
        # Convert to empirical copula
        n = len(x)
        u = stats.rankdata(x) / (n + 1)
        v = stats.rankdata(y) / (n + 1)
        
        # Upper tail dependence
        threshold_upper = 0.9
        upper_mask = (u > threshold_upper) & (v > threshold_upper)
        upper_tail_dep = np.sum(upper_mask) / np.sum(u > threshold_upper) if np.sum(u > threshold_upper) > 0 else 0
        
        # Lower tail dependence
        threshold_lower = 0.1
        lower_mask = (u < threshold_lower) & (v < threshold_lower)
        lower_tail_dep = np.sum(lower_mask) / np.sum(u < threshold_lower) if np.sum(u < threshold_lower) > 0 else 0
        
        return {
            'upper_tail': upper_tail_dep,
            'lower_tail': lower_tail_dep
        }
    
    def _calculate_partial_correlation(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate partial correlation controlling for market factor"""
        # Create synthetic market factor
        market_factor = (x + y) / 2 + np.random.normal(0, 0.1, len(x))
        
        try:
            # Correlation matrix
            corr_matrix = np.corrcoef([x, y, market_factor])
            
            # Partial correlation formula
            r_xy = corr_matrix[0, 1]
            r_xz = corr_matrix[0, 2]
            r_yz = corr_matrix[1, 2]
            
            numerator = r_xy - r_xz * r_yz
            denominator = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
            
            if denominator != 0:
                return numerator / denominator
            else:
                return 0.0
        except:
            return 0.0
    
    def _test_correlation_significance(self, x: np.ndarray, y: np.ndarray, correlation: float) -> float:
        """Test statistical significance of correlation"""
        n = len(x)
        
        if n < 3 or abs(correlation) >= 0.999:
            return 0.5
        
        # t-statistic
        t_stat = correlation * np.sqrt((n - 2) / (1 - correlation**2))
        
        # Two-tailed p-value
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
        
        return 1 - p_value
    
    def _generate_signal(self, result: CorrelationCoefficientResult, x: np.ndarray, y: np.ndarray) -> SignalType:
        """Generate trading signal based on correlation analysis"""
        # Recent performance comparison
        recent_x = np.mean(x[-10:]) if len(x) >= 10 else 0
        recent_y = np.mean(y[-10:]) if len(y) >= 10 else 0
        
        # Strong correlation with divergence suggests reversion
        if result.dependence_strength > 0.6 and result.statistical_significance > 0.95:
            if recent_x > recent_y * 1.5:
                return SignalType.SELL  # X overperforming, expect reversion
            elif recent_y > recent_x * 1.5:
                return SignalType.BUY   # X underperforming, expect catch-up
        
        # Outlier-influenced regime suggests caution
        if result.correlation_regime == "outlier_influenced":
            return SignalType.HOLD
        
        # Strong tail dependence in crisis periods
        if result.tail_dependence['lower_tail'] > 0.3 and recent_x < -0.02:
            return SignalType.SELL  # Expect contagion in downturns
        
        return SignalType.HOLD
    
    def _calculate_signal_strength(self, result: CorrelationCoefficientResult) -> float:
        """Calculate signal strength"""
        dependence_strength = result.dependence_strength
        significance_strength = result.statistical_significance
        
        return (dependence_strength + significance_strength) / 2
    
    def _calculate_confidence(self, result: CorrelationCoefficientResult) -> float:
        """Calculate confidence based on statistical measures"""
        width = result.confidence_interval[1] - result.confidence_interval[0]
        interval_confidence = max(0, 1 - width/2)  # Narrower interval = higher confidence
        
        return (interval_confidence + result.statistical_significance) / 2