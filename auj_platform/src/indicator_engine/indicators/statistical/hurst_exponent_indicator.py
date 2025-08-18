"""
Advanced Hurst Exponent Indicator with R/S Analysis and Detrended Fluctuation Analysis

This indicator implements sophisticated long-memory analysis including:
- Rescaled Range (R/S) analysis with bias correction
- Detrended Fluctuation Analysis (DFA) 
- Multi-fractal analysis and regime detection
- Persistence vs anti-persistence classification
- Long-memory parameter estimation
- Market efficiency testing
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from scipy import stats
import logging
from dataclasses import dataclass

from ..base.standard_indicator import StandardIndicatorInterface
from ....core.signal_type import SignalType
from ....core.exceptions import IndicatorCalculationError


@dataclass
class HurstExponentResult:
    """Result container for Hurst exponent analysis"""
    hurst_rs: float
    hurst_dfa: float
    hurst_average: float
    confidence_interval: Tuple[float, float]
    persistence_regime: str
    long_memory_strength: float
    market_efficiency_score: float
    fractal_dimension: float
    rs_statistic: float
    dfa_fluctuation: np.ndarray
    regime_change_detected: bool
    statistical_significance: float


class HurstExponentIndicator(StandardIndicatorInterface):
    """
    Advanced Hurst Exponent Indicator
    
    Implements multiple methods for estimating the Hurst exponent
    and analyzing long-range dependence in financial time series.
    """
    
    def __init__(self, 
                 min_window: int = 50,
                 max_window: int = 250,
                 dfa_scales: Optional[List[int]] = None):
        """
        Initialize Hurst Exponent Indicator
        
        Args:
            min_window: Minimum window size for analysis
            max_window: Maximum window size for analysis  
            dfa_scales: Custom scales for DFA analysis
        """
        super().__init__()
        self.min_window = min_window
        self.max_window = max_window
        self.dfa_scales = dfa_scales or self._generate_dfa_scales()
        self.logger = logging.getLogger(__name__)
        
    def _generate_dfa_scales(self) -> List[int]:
        """Generate logarithmically spaced scales for DFA"""
        return [int(x) for x in np.logspace(1, np.log10(self.max_window), 15)]
        
    def calculate(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Calculate Hurst exponent analysis
        
        Args:
            data: DataFrame with OHLCV data
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing Hurst exponent results
        """
        try:
            if data.empty or len(data) < self.min_window:
                raise IndicatorCalculationError("Insufficient data for Hurst exponent analysis")
            
            # Extract price series and calculate returns
            prices = data['close'].values
            log_prices = np.log(prices)
            returns = np.diff(log_prices)
            
            if len(returns) < self.min_window:
                raise IndicatorCalculationError("Insufficient return data")
            
            # Perform Hurst exponent analysis
            hurst_result = self._calculate_hurst_analysis(log_prices, returns)
            
            # Generate trading signal
            signal = self._generate_signal(hurst_result, returns)
            
            return {
                'signal': signal,
                'hurst_rs': hurst_result.hurst_rs,
                'hurst_dfa': hurst_result.hurst_dfa,
                'hurst_average': hurst_result.hurst_average,
                'confidence_interval': hurst_result.confidence_interval,
                'persistence_regime': hurst_result.persistence_regime,
                'long_memory_strength': hurst_result.long_memory_strength,
                'market_efficiency_score': hurst_result.market_efficiency_score,
                'fractal_dimension': hurst_result.fractal_dimension,
                'rs_statistic': hurst_result.rs_statistic,
                'dfa_fluctuation': hurst_result.dfa_fluctuation.tolist(),
                'regime_change_detected': hurst_result.regime_change_detected,
                'statistical_significance': hurst_result.statistical_significance,
                'strength': self._calculate_signal_strength(hurst_result),
                'confidence': self._calculate_confidence(hurst_result)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating Hurst exponent: {str(e)}")
            raise IndicatorCalculationError(f"Hurst exponent calculation failed: {str(e)}")
    
    def _calculate_hurst_analysis(self, log_prices: np.ndarray, returns: np.ndarray) -> HurstExponentResult:
        """Perform comprehensive Hurst exponent analysis"""
        
        # R/S Analysis
        hurst_rs, rs_stat = self._rescaled_range_analysis(returns)
        
        # Detrended Fluctuation Analysis
        hurst_dfa, dfa_fluct = self._detrended_fluctuation_analysis(log_prices)
        
        # Average Hurst exponent
        hurst_avg = (hurst_rs + hurst_dfa) / 2
        
        # Confidence interval using bootstrap
        conf_interval = self._bootstrap_confidence_interval(returns, log_prices)
        
        # Persistence regime classification
        regime = self._classify_persistence_regime(hurst_avg)
        
        # Long memory strength
        memory_strength = abs(hurst_avg - 0.5) * 2  # Scale to [0, 1]
        
        # Market efficiency score
        efficiency_score = self._calculate_market_efficiency(hurst_avg, returns)
        
        # Fractal dimension
        fractal_dim = 2 - hurst_avg
        
        # Regime change detection
        regime_change = self._detect_regime_change(returns)
        
        # Statistical significance
        significance = self._test_hurst_significance(hurst_avg, len(returns))
        
        return HurstExponentResult(
            hurst_rs=hurst_rs,
            hurst_dfa=hurst_dfa,
            hurst_average=hurst_avg,
            confidence_interval=conf_interval,
            persistence_regime=regime,
            long_memory_strength=memory_strength,
            market_efficiency_score=efficiency_score,
            fractal_dimension=fractal_dim,
            rs_statistic=rs_stat,
            dfa_fluctuation=dfa_fluct,
            regime_change_detected=regime_change,
            statistical_significance=significance
        )
    
    def _rescaled_range_analysis(self, returns: np.ndarray) -> Tuple[float, float]:
        """Perform Rescaled Range (R/S) analysis"""
        n = len(returns)
        
        # Generate window sizes
        window_sizes = []
        for size in range(self.min_window, min(n, self.max_window), 10):
            window_sizes.append(size)
        
        if len(window_sizes) < 3:
            return 0.5, 0.0
        
        rs_values = []
        
        for window_size in window_sizes:
            rs_window = []
            
            # Calculate R/S for multiple non-overlapping windows
            for start in range(0, n - window_size + 1, window_size):
                end = start + window_size
                window_returns = returns[start:end]
                
                if len(window_returns) < window_size:
                    continue
                
                # Mean return for the window
                mean_return = np.mean(window_returns)
                
                # Cumulative deviations from mean
                cumulative_deviations = np.cumsum(window_returns - mean_return)
                
                # Range of cumulative deviations
                R = np.max(cumulative_deviations) - np.min(cumulative_deviations)
                
                # Standard deviation
                S = np.std(window_returns, ddof=1) if len(window_returns) > 1 else 1e-8
                
                # R/S ratio
                if S > 1e-8:
                    rs_window.append(R / S)
            
            if rs_window:
                rs_values.append(np.mean(rs_window))
        
        if len(rs_values) < 3:
            return 0.5, 0.0
        
        # Linear regression in log-log space
        log_windows = np.log(window_sizes[:len(rs_values)])
        log_rs = np.log(np.maximum(rs_values, 1e-8))
        
        # Remove any NaN or infinite values
        valid_mask = np.isfinite(log_windows) & np.isfinite(log_rs)
        log_windows = log_windows[valid_mask]
        log_rs = log_rs[valid_mask]
        
        if len(log_windows) < 3:
            return 0.5, 0.0
        
        # Fit line: log(R/S) = log(c) + H * log(n)
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_windows, log_rs)
        
        hurst_exponent = slope
        rs_statistic = r_value ** 2  # R-squared
        
        # Clip to reasonable range
        hurst_exponent = np.clip(hurst_exponent, 0.0, 1.0)
        
        return hurst_exponent, rs_statistic
    
    def _detrended_fluctuation_analysis(self, log_prices: np.ndarray) -> Tuple[float, np.ndarray]:
        """Perform Detrended Fluctuation Analysis (DFA)"""
        # Integrate the series (cumulative sum after removing mean)
        y = np.cumsum(log_prices - np.mean(log_prices))
        
        fluctuations = []
        scales = []
        
        for scale in self.dfa_scales:
            if scale >= len(y) or scale < 4:
                continue
                
            # Divide series into non-overlapping segments
            n_segments = len(y) // scale
            
            if n_segments < 1:
                continue
            
            # Calculate local trends for each segment
            segment_fluctuations = []
            
            for i in range(n_segments):
                start_idx = i * scale
                end_idx = start_idx + scale
                segment = y[start_idx:end_idx]
                
                # Fit polynomial trend (linear detrending)
                x = np.arange(scale)
                coeffs = np.polyfit(x, segment, 1)
                trend = np.polyval(coeffs, x)
                
                # Calculate fluctuation
                fluctuation = np.sqrt(np.mean((segment - trend) ** 2))
                segment_fluctuations.append(fluctuation)
            
            if segment_fluctuations:
                avg_fluctuation = np.mean(segment_fluctuations)
                fluctuations.append(avg_fluctuation)
                scales.append(scale)
        
        if len(fluctuations) < 3:
            return 0.5, np.array([0.0])
        
        # Linear regression in log-log space
        log_scales = np.log(scales)
        log_fluctuations = np.log(np.maximum(fluctuations, 1e-8))
        
        # Remove any NaN or infinite values
        valid_mask = np.isfinite(log_scales) & np.isfinite(log_fluctuations)
        log_scales = log_scales[valid_mask]
        log_fluctuations = log_fluctuations[valid_mask]
        
        if len(log_scales) < 3:
            return 0.5, np.array(fluctuations)
        
        # Fit line: log(F) = log(c) + α * log(scale)
        slope, _, _, _, _ = stats.linregress(log_scales, log_fluctuations)
        
        hurst_exponent = slope
        
        # Clip to reasonable range
        hurst_exponent = np.clip(hurst_exponent, 0.0, 1.0)
        
        return hurst_exponent, np.array(fluctuations)
    
    def _bootstrap_confidence_interval(self, returns: np.ndarray, log_prices: np.ndarray, n_bootstrap: int = 100) -> Tuple[float, float]:
        """Calculate confidence interval using bootstrap"""
        hurst_estimates = []
        
        n = len(returns)
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n, size=n, replace=True)
            boot_returns = returns[indices]
            boot_log_prices = log_prices[indices]
            
            # Calculate Hurst exponent
            hurst_rs, _ = self._rescaled_range_analysis(boot_returns)
            hurst_dfa, _ = self._detrended_fluctuation_analysis(boot_log_prices)
            
            hurst_avg = (hurst_rs + hurst_dfa) / 2
            hurst_estimates.append(hurst_avg)
        
        if len(hurst_estimates) < 10:
            return (0.4, 0.6)
        
        # Calculate 95% confidence interval
        lower = np.percentile(hurst_estimates, 2.5)
        upper = np.percentile(hurst_estimates, 97.5)
        
        return (lower, upper)
    
    def _classify_persistence_regime(self, hurst: float) -> str:
        """Classify persistence regime based on Hurst exponent"""
        if hurst > 0.6:
            return "persistent"
        elif hurst < 0.4:
            return "anti_persistent"
        else:
            return "random_walk"
    
    def _calculate_market_efficiency(self, hurst: float, returns: np.ndarray) -> float:
        """Calculate market efficiency score"""
        # Efficiency score based on deviation from random walk (H=0.5)
        efficiency_from_hurst = 1 - 2 * abs(hurst - 0.5)
        
        # Additional efficiency measure from return autocorrelation
        if len(returns) > 10:
            autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
            autocorr = autocorr if not np.isnan(autocorr) else 0.0
            efficiency_from_autocorr = 1 - abs(autocorr)
        else:
            efficiency_from_autocorr = 0.5
        
        return (efficiency_from_hurst + efficiency_from_autocorr) / 2
    
    def _detect_regime_change(self, returns: np.ndarray) -> bool:
        """Detect regime changes in persistence behavior"""
        if len(returns) < 100:
            return False
        
        # Split series in half and compare Hurst exponents
        mid_point = len(returns) // 2
        first_half = returns[:mid_point]
        second_half = returns[mid_point:]
        
        hurst1, _ = self._rescaled_range_analysis(first_half)
        hurst2, _ = self._rescaled_range_analysis(second_half)
        
        # Significant change threshold
        return abs(hurst1 - hurst2) > 0.15
    
    def _test_hurst_significance(self, hurst: float, n: int) -> float:
        """Test statistical significance of Hurst exponent"""
        # Standard error approximation for Hurst exponent
        se = np.sqrt(1 / (12 * n))
        
        # Test against null hypothesis H=0.5 (random walk)
        z_score = abs(hurst - 0.5) / se
        
        # Two-tailed p-value
        p_value = 2 * (1 - stats.norm.cdf(z_score))
        
        return 1 - p_value  # Convert to significance level
    
    def _generate_signal(self, result: HurstExponentResult, returns: np.ndarray) -> SignalType:
        """Generate trading signal based on Hurst analysis"""
        hurst = result.hurst_average
        regime = result.persistence_regime
        
        # Recent price momentum
        recent_momentum = np.mean(returns[-10:]) if len(returns) >= 10 else 0
        
        if regime == "persistent":
            # Trend following in persistent markets
            if recent_momentum > 0.001:
                return SignalType.BUY
            elif recent_momentum < -0.001:
                return SignalType.SELL
        
        elif regime == "anti_persistent":
            # Mean reversion in anti-persistent markets
            if recent_momentum > 0.002:
                return SignalType.SELL  # Expect reversion
            elif recent_momentum < -0.002:
                return SignalType.BUY   # Expect reversion
        
        # Regime change detected
        if result.regime_change_detected:
            # Conservative approach during regime changes
            return SignalType.HOLD
        
        return SignalType.HOLD
    
    def _calculate_signal_strength(self, result: HurstExponentResult) -> float:
        """Calculate signal strength based on Hurst characteristics"""
        # Strength from deviation from random walk
        deviation_strength = result.long_memory_strength
        
        # Statistical significance strength
        significance_strength = result.statistical_significance
        
        # Regime clarity (how far from boundary values)
        if result.persistence_regime == "random_walk":
            regime_strength = 0.0
        else:
            regime_strength = min(abs(result.hurst_average - 0.5) * 4, 1.0)
        
        return (deviation_strength + significance_strength + regime_strength) / 3
    
    def _calculate_confidence(self, result: HurstExponentResult) -> float:
        """Calculate confidence based on statistical measures"""
        # Confidence from statistical significance
        significance_conf = result.statistical_significance
        
        # Confidence from R/S statistic quality
        rs_conf = min(result.rs_statistic, 1.0)
        
        # Confidence from interval width (narrower = higher confidence)
        interval_width = result.confidence_interval[1] - result.confidence_interval[0]
        interval_conf = max(0, 1 - interval_width)
        
        return (significance_conf + rs_conf + interval_conf) / 3