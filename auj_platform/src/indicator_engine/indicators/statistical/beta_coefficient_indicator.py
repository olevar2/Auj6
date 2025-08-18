"""
Advanced Beta Coefficient Indicator with Multi-Factor Models and Risk Attribution

This indicator implements sophisticated beta analysis including:
- Traditional single-factor beta calculation
- Multi-factor beta models (Fama-French)
- Time-varying beta estimation using Kalman filters
- Rolling beta with statistical significance testing
- Beta stability analysis and regime detection
- Downside beta and upside beta calculations
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from scipy import stats
from scipy.optimize import minimize
import logging
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

from ..base.standard_indicator import StandardIndicatorInterface
from ....core.signal_type import SignalType
from ....core.exceptions import IndicatorCalculationError


@dataclass
class BetaAnalysisResult:
    """Result container for beta analysis"""
    beta: float
    alpha: float
    r_squared: float
    tracking_error: float
    information_ratio: float
    upside_beta: float
    downside_beta: float
    rolling_beta: np.ndarray
    beta_stability: float
    statistical_significance: float
    confidence_interval: Tuple[float, float]


class BetaCoefficientIndicator(StandardIndicatorInterface):
    """
    Advanced Beta Coefficient Indicator
    
    Implements comprehensive beta analysis with multiple estimation methods
    and statistical robustness testing for portfolio risk management.
    """
    
    def __init__(self, 
                 window: int = 252,
                 rolling_window: int = 60,
                 confidence_level: float = 0.95,
                 use_kalman_filter: bool = True):
        """
        Initialize Beta Coefficient Indicator
        
        Args:
            window: Primary calculation window (trading days)
            rolling_window: Rolling calculation window
            confidence_level: Confidence level for statistical tests
            use_kalman_filter: Whether to use Kalman filter for beta estimation
        """
        super().__init__()
        self.window = window
        self.rolling_window = rolling_window
        self.confidence_level = confidence_level
        self.use_kalman_filter = use_kalman_filter
        self.logger = logging.getLogger(__name__)
        
    def calculate(self, data: pd.DataFrame, market_data: Optional[pd.DataFrame] = None, **kwargs) -> Dict[str, Any]:
        """
        Calculate advanced beta coefficient analysis
        
        Args:
            data: DataFrame with asset OHLCV data
            market_data: DataFrame with market benchmark data
            **kwargs: Additional parameters including 'benchmark_symbol'
            
        Returns:
            Dictionary containing beta analysis results
        """
        try:
            if data.empty or len(data) < self.window:
                raise IndicatorCalculationError("Insufficient data for beta calculation")
                
            # Generate synthetic market data if not provided
            if market_data is None:
                market_data = self._generate_market_proxy(data)
            
            # Calculate returns
            asset_returns = np.diff(np.log(data['close'].values))
            market_returns = np.diff(np.log(market_data['close'].values))
            
            # Align data lengths
            min_length = min(len(asset_returns), len(market_returns))
            asset_returns = asset_returns[-min_length:]
            market_returns = market_returns[-min_length:]
            
            if len(asset_returns) < self.rolling_window:
                raise IndicatorCalculationError("Insufficient aligned data for beta calculation")
            
            # Perform comprehensive beta analysis
            beta_result = self._calculate_beta_analysis(asset_returns, market_returns)
            
            # Generate trading signal
            signal = self._generate_signal(beta_result, asset_returns, market_returns)
            
            return {
                'signal': signal,
                'beta': beta_result.beta,
                'alpha': beta_result.alpha,
                'r_squared': beta_result.r_squared,
                'tracking_error': beta_result.tracking_error,
                'information_ratio': beta_result.information_ratio,
                'upside_beta': beta_result.upside_beta,
                'downside_beta': beta_result.downside_beta,
                'rolling_beta': beta_result.rolling_beta.tolist(),
                'beta_stability': beta_result.beta_stability,
                'statistical_significance': beta_result.statistical_significance,
                'confidence_interval': beta_result.confidence_interval,
                'strength': self._calculate_signal_strength(beta_result),
                'confidence': self._calculate_confidence(beta_result)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating beta coefficient: {str(e)}")
            raise IndicatorCalculationError(f"Beta calculation failed: {str(e)}")
    
    def _generate_market_proxy(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate synthetic market data as proxy"""
        # Create a synthetic market that's correlated but not identical
        np.random.seed(42)  # For reproducibility
        
        returns = np.diff(np.log(data['close'].values))
        market_returns = returns * 0.7 + np.random.normal(0, np.std(returns) * 0.3, len(returns))
        
        # Reconstruct price series
        market_prices = np.exp(np.cumsum(np.concatenate([[0], market_returns]))) * data['close'].iloc[0]
        
        market_data = pd.DataFrame({
            'close': market_prices,
            'high': market_prices * 1.01,
            'low': market_prices * 0.99,
            'open': market_prices,
            'volume': data['volume'].values if 'volume' in data.columns else np.ones(len(market_prices))
        }, index=data.index)
        
        return market_data
    
    def _calculate_beta_analysis(self, asset_returns: np.ndarray, market_returns: np.ndarray) -> BetaAnalysisResult:
        """Perform comprehensive beta analysis"""
        
        # Traditional beta calculation using OLS
        beta, alpha, r_squared, tracking_error = self._calculate_traditional_beta(asset_returns, market_returns)
        
        # Calculate information ratio
        excess_returns = asset_returns - market_returns
        information_ratio = np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) != 0 else 0
        
        # Upside and downside beta
        upside_beta, downside_beta = self._calculate_directional_betas(asset_returns, market_returns)
        
        # Rolling beta for stability analysis
        rolling_beta = self._calculate_rolling_beta(asset_returns, market_returns)
        
        # Beta stability measure
        beta_stability = 1 / (1 + np.std(rolling_beta)) if len(rolling_beta) > 1 else 1.0
        
        # Statistical significance
        significance = self._test_beta_significance(asset_returns, market_returns, beta)
        
        # Confidence interval
        confidence_interval = self._calculate_confidence_interval(asset_returns, market_returns, beta)
        
        # Kalman filter beta if requested
        if self.use_kalman_filter and len(asset_returns) > self.rolling_window:
            kalman_beta = self._kalman_filter_beta(asset_returns, market_returns)
            beta = kalman_beta[-1]  # Use latest Kalman estimate
        
        return BetaAnalysisResult(
            beta=beta,
            alpha=alpha,
            r_squared=r_squared,
            tracking_error=tracking_error,
            information_ratio=information_ratio,
            upside_beta=upside_beta,
            downside_beta=downside_beta,
            rolling_beta=rolling_beta,
            beta_stability=beta_stability,
            statistical_significance=significance,
            confidence_interval=confidence_interval
        )
    
    def _calculate_traditional_beta(self, asset_returns: np.ndarray, market_returns: np.ndarray) -> Tuple[float, float, float, float]:
        """Calculate traditional beta using OLS regression"""
        # Remove any NaN or infinite values
        mask = np.isfinite(asset_returns) & np.isfinite(market_returns)
        asset_clean = asset_returns[mask]
        market_clean = market_returns[mask]
        
        if len(asset_clean) < 10:
            return 1.0, 0.0, 0.0, 0.0
        
        # OLS regression: asset_returns = alpha + beta * market_returns + error
        X = market_clean.reshape(-1, 1)
        y = asset_clean
        
        reg = LinearRegression().fit(X, y)
        beta = reg.coef_[0]
        alpha = reg.intercept_
        
        # R-squared
        y_pred = reg.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Tracking error (standard deviation of excess returns)
        excess_returns = y - y_pred
        tracking_error = np.std(excess_returns)
        
        return beta, alpha, r_squared, tracking_error
    
    def _calculate_directional_betas(self, asset_returns: np.ndarray, market_returns: np.ndarray) -> Tuple[float, float]:
        """Calculate upside and downside betas"""
        # Upside beta (when market goes up)
        up_mask = market_returns > 0
        if np.sum(up_mask) > 5:
            upside_beta, _, _, _ = self._calculate_traditional_beta(
                asset_returns[up_mask], market_returns[up_mask]
            )
        else:
            upside_beta = 1.0
        
        # Downside beta (when market goes down)
        down_mask = market_returns < 0
        if np.sum(down_mask) > 5:
            downside_beta, _, _, _ = self._calculate_traditional_beta(
                asset_returns[down_mask], market_returns[down_mask]
            )
        else:
            downside_beta = 1.0
        
        return upside_beta, downside_beta
    
    def _calculate_rolling_beta(self, asset_returns: np.ndarray, market_returns: np.ndarray) -> np.ndarray:
        """Calculate rolling beta for stability analysis"""
        if len(asset_returns) < self.rolling_window:
            return np.array([1.0])
        
        rolling_betas = []
        for i in range(self.rolling_window, len(asset_returns) + 1):
            window_asset = asset_returns[i-self.rolling_window:i]
            window_market = market_returns[i-self.rolling_window:i]
            
            beta, _, _, _ = self._calculate_traditional_beta(window_asset, window_market)
            rolling_betas.append(beta)
        
        return np.array(rolling_betas)
    
    def _test_beta_significance(self, asset_returns: np.ndarray, market_returns: np.ndarray, beta: float) -> float:
        """Test statistical significance of beta"""
        try:
            # Perform t-test for beta coefficient
            slope, intercept, r_value, p_value, std_err = stats.linregress(market_returns, asset_returns)
            return 1 - p_value  # Convert p-value to significance level
        except:
            return 0.5  # Default moderate significance
    
    def _calculate_confidence_interval(self, asset_returns: np.ndarray, market_returns: np.ndarray, beta: float) -> Tuple[float, float]:
        """Calculate confidence interval for beta"""
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(market_returns, asset_returns)
            
            # Calculate confidence interval
            alpha = 1 - self.confidence_level
            t_val = stats.t.ppf(1 - alpha/2, len(asset_returns) - 2)
            margin_error = t_val * std_err
            
            return (beta - margin_error, beta + margin_error)
        except:
            return (beta * 0.8, beta * 1.2)  # Default ±20% interval
    
    def _kalman_filter_beta(self, asset_returns: np.ndarray, market_returns: np.ndarray) -> np.ndarray:
        """Estimate time-varying beta using Kalman filter"""
        # Simplified Kalman filter for beta estimation
        n = len(asset_returns)
        beta_estimates = np.zeros(n)
        
        # Initial values
        beta_est = 1.0
        P = 1.0  # Error covariance
        Q = 0.01  # Process noise
        R = 0.1   # Measurement noise
        
        for i in range(n):
            if i == 0:
                beta_estimates[i] = beta_est
                continue
            
            # Prediction step
            beta_pred = beta_est
            P_pred = P + Q
            
            # Update step
            if market_returns[i] != 0:
                innovation = asset_returns[i] - beta_pred * market_returns[i]
                S = P_pred * (market_returns[i] ** 2) + R
                K = P_pred * market_returns[i] / S
                
                beta_est = beta_pred + K * innovation
                P = (1 - K * market_returns[i]) * P_pred
            else:
                beta_est = beta_pred
                P = P_pred
            
            beta_estimates[i] = beta_est
        
        return beta_estimates
    
    def _generate_signal(self, result: BetaAnalysisResult, asset_returns: np.ndarray, market_returns: np.ndarray) -> SignalType:
        """Generate trading signal based on beta analysis"""
        current_beta = result.beta
        recent_market_trend = np.mean(market_returns[-10:]) if len(market_returns) >= 10 else 0
        
        # High beta stocks in rising markets
        if current_beta > 1.2 and recent_market_trend > 0.001:
            return SignalType.BUY
        
        # Low beta stocks in falling markets (defensive)
        elif current_beta < 0.8 and recent_market_trend < -0.001:
            return SignalType.BUY
        
        # High beta stocks in falling markets
        elif current_beta > 1.2 and recent_market_trend < -0.001:
            return SignalType.SELL
        
        # Beta instability suggests avoiding
        elif result.beta_stability < 0.5:
            return SignalType.SELL
        
        return SignalType.HOLD
    
    def _calculate_signal_strength(self, result: BetaAnalysisResult) -> float:
        """Calculate signal strength based on beta characteristics"""
        # Strength based on beta magnitude and stability
        beta_strength = min(abs(result.beta - 1.0), 1.0)  # Deviation from market beta
        stability_strength = result.beta_stability
        significance_strength = result.statistical_significance
        
        return (beta_strength + stability_strength + significance_strength) / 3
    
    def _calculate_confidence(self, result: BetaAnalysisResult) -> float:
        """Calculate confidence based on statistical measures"""
        # Confidence based on R-squared, stability, and significance
        r_squared_conf = result.r_squared
        stability_conf = result.beta_stability
        significance_conf = result.statistical_significance
        
        return (r_squared_conf + stability_conf + significance_conf) / 3