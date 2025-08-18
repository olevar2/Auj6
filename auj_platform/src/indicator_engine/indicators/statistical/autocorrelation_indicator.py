"""
Advanced Autocorrelation Indicator with Ljung-Box Test and Multi-lag Analysis

This indicator implements sophisticated autocorrelation analysis including:
- Multiple lag autocorrelation computation
- Ljung-Box test for white noise hypothesis
- Partial autocorrelation function (PACF)
- Durbin-Watson test for serial correlation
- Advanced statistical significance testing
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from scipy import stats
from scipy.stats import chi2
import logging
from dataclasses import dataclass

from ..base.standard_indicator import StandardIndicatorInterface
from ....core.signal_type import SignalType
from ....core.exceptions import IndicatorCalculationError


@dataclass
class AutocorrelationResult:
    """Result container for autocorrelation analysis"""
    autocorrelations: np.ndarray
    partial_autocorrelations: np.ndarray
    ljung_box_statistic: float
    ljung_box_p_value: float
    durbin_watson_statistic: float
    significance_bounds: Tuple[float, float]
    significant_lags: List[int]
    white_noise_hypothesis: bool


class AutocorrelationIndicator(StandardIndicatorInterface):
    """
    Advanced Autocorrelation Indicator
    
    Implements comprehensive autocorrelation analysis with statistical testing
    for detecting serial correlation patterns in price series.
    """
    
    def __init__(self, max_lags: int = 20, significance_level: float = 0.05):
        """
        Initialize Autocorrelation Indicator
        
        Args:
            max_lags: Maximum number of lags to analyze
            significance_level: Statistical significance threshold
        """
        super().__init__()
        self.max_lags = max_lags
        self.significance_level = significance_level
        self.logger = logging.getLogger(__name__)
        
    def calculate(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Calculate advanced autocorrelation analysis
        
        Args:
            data: DataFrame with OHLCV data
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing autocorrelation results
        """
        try:
            if data.empty or len(data) < self.max_lags * 2:
                raise IndicatorCalculationError("Insufficient data for autocorrelation analysis")
                
            # Use close prices for analysis
            prices = data['close'].values
            returns = np.diff(np.log(prices))  # Log returns
            
            # Calculate autocorrelations
            autocorr_result = self._calculate_autocorrelation(returns)
            
            # Generate trading signal based on autocorrelation patterns
            signal = self._generate_signal(autocorr_result)
            
            return {
                'signal': signal,
                'autocorrelations': autocorr_result.autocorrelations.tolist(),
                'partial_autocorrelations': autocorr_result.partial_autocorrelations.tolist(),
                'ljung_box_statistic': autocorr_result.ljung_box_statistic,
                'ljung_box_p_value': autocorr_result.ljung_box_p_value,
                'durbin_watson_statistic': autocorr_result.durbin_watson_statistic,
                'significance_bounds': autocorr_result.significance_bounds,
                'significant_lags': autocorr_result.significant_lags,
                'white_noise_hypothesis': autocorr_result.white_noise_hypothesis,
                'strength': self._calculate_signal_strength(autocorr_result),
                'confidence': self._calculate_confidence(autocorr_result)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating autocorrelation: {str(e)}")
            raise IndicatorCalculationError(f"Autocorrelation calculation failed: {str(e)}")
    
    def _calculate_autocorrelation(self, returns: np.ndarray) -> AutocorrelationResult:
        """Calculate comprehensive autocorrelation analysis"""
        n = len(returns)
        
        # Calculate autocorrelations for different lags
        autocorrs = np.zeros(self.max_lags + 1)
        for lag in range(self.max_lags + 1):
            if lag == 0:
                autocorrs[lag] = 1.0
            else:
                autocorrs[lag] = self._lag_correlation(returns, lag)
        
        # Calculate partial autocorrelations using Yule-Walker equations
        pacf = self._calculate_pacf(autocorrs)
        
        # Ljung-Box test for white noise
        ljung_box_stat, ljung_box_p = self._ljung_box_test(returns, self.max_lags)
        
        # Durbin-Watson test
        durbin_watson_stat = self._durbin_watson_test(returns)
        
        # Significance bounds (95% confidence interval)
        bounds = self._calculate_significance_bounds(n)
        
        # Find significant lags
        significant_lags = [
            lag for lag in range(1, len(autocorrs))
            if abs(autocorrs[lag]) > bounds[1]
        ]
        
        # White noise hypothesis
        white_noise = ljung_box_p > self.significance_level
        
        return AutocorrelationResult(
            autocorrelations=autocorrs,
            partial_autocorrelations=pacf,
            ljung_box_statistic=ljung_box_stat,
            ljung_box_p_value=ljung_box_p,
            durbin_watson_statistic=durbin_watson_stat,
            significance_bounds=bounds,
            significant_lags=significant_lags,
            white_noise_hypothesis=white_noise
        )
    
    def _lag_correlation(self, series: np.ndarray, lag: int) -> float:
        """Calculate correlation at specific lag"""
        if lag >= len(series):
            return 0.0
            
        y1 = series[:-lag] if lag > 0 else series
        y2 = series[lag:] if lag > 0 else series
        
        if len(y1) == 0 or len(y2) == 0:
            return 0.0
            
        return np.corrcoef(y1, y2)[0, 1] if not np.isnan(np.corrcoef(y1, y2)[0, 1]) else 0.0
    
    def _calculate_pacf(self, autocorrs: np.ndarray) -> np.ndarray:
        """Calculate Partial Autocorrelation Function using Yule-Walker equations"""
        pacf = np.zeros_like(autocorrs)
        pacf[0] = 1.0
        
        if len(autocorrs) > 1:
            pacf[1] = autocorrs[1]
        
        for k in range(2, len(autocorrs)):
            # Solve Yule-Walker equations
            gamma_matrix = np.array([
                [autocorrs[abs(i-j)] for j in range(k)]
                for i in range(k)
            ])
            
            gamma_vector = autocorrs[1:k+1]
            
            try:
                phi = np.linalg.solve(gamma_matrix, gamma_vector)
                pacf[k] = phi[-1]
            except np.linalg.LinAlgError:
                pacf[k] = 0.0
        
        return pacf
    
    def _ljung_box_test(self, returns: np.ndarray, lags: int) -> Tuple[float, float]:
        """Ljung-Box test for serial correlation"""
        n = len(returns)
        autocorrs = np.array([self._lag_correlation(returns, lag) for lag in range(1, lags + 1)])
        
        # Ljung-Box statistic
        lags_array = np.arange(1, lags + 1)
        lb_stat = n * (n + 2) * np.sum((autocorrs ** 2) / (n - lags_array))
        
        # P-value from chi-squared distribution
        p_value = 1 - chi2.cdf(lb_stat, lags)
        
        return lb_stat, p_value
    
    def _durbin_watson_test(self, returns: np.ndarray) -> float:
        """Durbin-Watson test for first-order serial correlation"""
        residuals = returns - np.mean(returns)
        diff_residuals = np.diff(residuals)
        
        numerator = np.sum(diff_residuals ** 2)
        denominator = np.sum(residuals ** 2)
        
        return numerator / denominator if denominator != 0 else 2.0
    
    def _calculate_significance_bounds(self, n: int) -> Tuple[float, float]:
        """Calculate significance bounds for autocorrelation"""
        # Approximate 95% confidence bounds
        bound = 1.96 / np.sqrt(n)
        return (-bound, bound)
    
    def _generate_signal(self, result: AutocorrelationResult) -> SignalType:
        """Generate trading signal based on autocorrelation analysis"""
        # Strong positive autocorrelation suggests trend continuation
        if len(result.significant_lags) > 0:
            recent_autocorr = result.autocorrelations[1:6]  # First 5 lags
            avg_autocorr = np.mean(recent_autocorr)
            
            if avg_autocorr > 0.1:
                return SignalType.BUY  # Trend continuation expected
            elif avg_autocorr < -0.1:
                return SignalType.SELL  # Mean reversion expected
        
        # If white noise hypothesis is accepted, no clear signal
        if result.white_noise_hypothesis:
            return SignalType.HOLD
        
        # Default to hold if patterns are unclear
        return SignalType.HOLD
    
    def _calculate_signal_strength(self, result: AutocorrelationResult) -> float:
        """Calculate signal strength based on autocorrelation magnitude"""
        if len(result.significant_lags) == 0:
            return 0.0
            
        # Use maximum absolute autocorrelation as strength measure
        max_autocorr = np.max(np.abs(result.autocorrelations[1:]))
        return min(max_autocorr * 2, 1.0)  # Scale to [0, 1]
    
    def _calculate_confidence(self, result: AutocorrelationResult) -> float:
        """Calculate confidence based on statistical tests"""
        # Lower p-value means higher confidence in non-random pattern
        ljung_box_confidence = 1 - result.ljung_box_p_value
        
        # More significant lags increase confidence
        lag_confidence = len(result.significant_lags) / self.max_lags
        
        return (ljung_box_confidence + lag_confidence) / 2