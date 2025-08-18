"""
Advanced Standard Deviation Indicator with Multi-Scale Analysis and Risk Assessment

This indicator implements comprehensive standard deviation analysis including:
- Multi-timeframe standard deviation calculation
- Conditional and unconditional volatility measures
- Risk-adjusted volatility metrics
- Volatility clustering detection
- Regime-specific standard deviation analysis
- Dynamic volatility forecasting
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from scipy import stats
from sklearn.preprocessing import StandardScaler
import logging
from dataclasses import dataclass

from ..base.standard_indicator import StandardIndicatorInterface
from ....core.signal_type import SignalType
from ....core.exceptions import IndicatorCalculationError


@dataclass
class StandardDeviationResult:
    """Result container for standard deviation analysis"""
    standard_deviation: float
    rolling_std: np.ndarray
    exponential_std: np.ndarray
    conditional_std: np.ndarray
    garch_std: np.ndarray
    realized_volatility: np.ndarray
    volatility_of_volatility: float
    volatility_clustering: np.ndarray
    regime_specific_std: Dict[str, float]
    percentile_ranks: np.ndarray
    z_scores: np.ndarray
    volatility_forecast: np.ndarray
    risk_metrics: Dict[str, float]
    volatility_smile: Dict[str, float]
    autocorrelation_structure: np.ndarray


class StandardDeviationIndicator(StandardIndicatorInterface):
    """
    Advanced Standard Deviation Indicator
    
    Implements comprehensive volatility analysis with multiple
    standard deviation measures and advanced risk assessment.
    """
    
    def __init__(self, 
                 window: int = 20,
                 short_window: int = 5,
                 long_window: int = 60,
                 lambda_param: float = 0.94):
        """
        Initialize Standard Deviation Indicator
        
        Args:
            window: Main window for standard deviation calculation
            short_window: Short-term window for comparison
            long_window: Long-term window for comparison
            lambda_param: Decay parameter for exponential weighting
        """
        super().__init__()
        self.window = window
        self.short_window = short_window
        self.long_window = long_window
        self.lambda_param = lambda_param
        self.logger = logging.getLogger(__name__)
        
    def calculate(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Calculate standard deviation analysis
        
        Args:
            data: DataFrame with OHLCV data
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing standard deviation analysis results
        """
        try:
            if data.empty or len(data) < self.window:
                raise IndicatorCalculationError("Insufficient data for standard deviation analysis")
            
            # Calculate returns
            prices = data['close'].values
            returns = np.diff(np.log(prices))
            
            if len(returns) < 10:
                raise IndicatorCalculationError("Insufficient return data")
            
            # Perform standard deviation analysis
            std_result = self._calculate_standard_deviation_analysis(returns, data)
            
            # Generate trading signal
            signal = self._generate_signal(std_result, returns, prices)
            
            return {
                'signal': signal,
                'standard_deviation': std_result.standard_deviation,
                'rolling_std': std_result.rolling_std.tolist(),
                'exponential_std': std_result.exponential_std.tolist(),
                'conditional_std': std_result.conditional_std.tolist(),
                'garch_std': std_result.garch_std.tolist(),
                'realized_volatility': std_result.realized_volatility.tolist(),
                'volatility_of_volatility': std_result.volatility_of_volatility,
                'volatility_clustering': std_result.volatility_clustering.tolist(),
                'regime_specific_std': std_result.regime_specific_std,
                'percentile_ranks': std_result.percentile_ranks.tolist(),
                'z_scores': std_result.z_scores.tolist(),
                'volatility_forecast': std_result.volatility_forecast.tolist(),
                'risk_metrics': std_result.risk_metrics,
                'volatility_smile': std_result.volatility_smile,
                'autocorrelation_structure': std_result.autocorrelation_structure.tolist(),
                'strength': self._calculate_signal_strength(std_result),
                'confidence': self._calculate_confidence(std_result)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating standard deviation: {str(e)}")
            raise IndicatorCalculationError(f"Standard deviation calculation failed: {str(e)}")
    
    def _calculate_standard_deviation_analysis(self, returns: np.ndarray, data: pd.DataFrame) -> StandardDeviationResult:
        """Perform comprehensive standard deviation analysis"""
        
        # Basic standard deviation
        standard_deviation = np.std(returns, ddof=1)
        
        # Rolling standard deviation
        rolling_std = self._calculate_rolling_std(returns)
        
        # Exponentially weighted standard deviation
        exponential_std = self._calculate_exponential_std(returns)
        
        # Conditional standard deviation (GARCH-like)
        conditional_std = self._calculate_conditional_std(returns)
        
        # GARCH(1,1) volatility
        garch_std = self._calculate_garch_volatility(returns)
        
        # Realized volatility
        realized_vol = self._calculate_realized_volatility(data)
        
        # Volatility of volatility
        vol_of_vol = self._calculate_volatility_of_volatility(rolling_std)
        
        # Volatility clustering
        vol_clustering = self._detect_volatility_clustering(returns)
        
        # Regime-specific standard deviations
        regime_std = self._calculate_regime_specific_std(returns)
        
        # Percentile ranks
        percentile_ranks = self._calculate_percentile_ranks(rolling_std)
        
        # Z-scores
        z_scores = self._calculate_z_scores(returns, rolling_std)
        
        # Volatility forecasting
        vol_forecast = self._forecast_volatility(returns, rolling_std)
        
        # Risk metrics
        risk_metrics = self._calculate_risk_metrics(returns, rolling_std)
        
        # Volatility smile
        vol_smile = self._calculate_volatility_smile(returns)
        
        # Autocorrelation structure
        autocorr = self._calculate_autocorrelation_structure(returns)
        
        return StandardDeviationResult(
            standard_deviation=standard_deviation,
            rolling_std=rolling_std,
            exponential_std=exponential_std,
            conditional_std=conditional_std,
            garch_std=garch_std,
            realized_volatility=realized_vol,
            volatility_of_volatility=vol_of_vol,
            volatility_clustering=vol_clustering,
            regime_specific_std=regime_std,
            percentile_ranks=percentile_ranks,
            z_scores=z_scores,
            volatility_forecast=vol_forecast,
            risk_metrics=risk_metrics,
            volatility_smile=vol_smile,
            autocorrelation_structure=autocorr
        )
    
    def _calculate_rolling_std(self, returns: np.ndarray) -> np.ndarray:
        """Calculate rolling standard deviation"""
        n = len(returns)
        rolling_std = np.zeros(n)
        
        for i in range(self.window, n):
            window_returns = returns[i-self.window:i]
            rolling_std[i] = np.std(window_returns, ddof=1)
        
        # Fill initial values
        if self.window < n:
            initial_std = np.std(returns[:self.window], ddof=1)
            rolling_std[:self.window] = initial_std
        
        return rolling_std
    
    def _calculate_exponential_std(self, returns: np.ndarray) -> np.ndarray:
        """Calculate exponentially weighted standard deviation"""
        n = len(returns)
        exp_std = np.zeros(n)
        
        # Initialize with first return
        exp_var = returns[0] ** 2 if n > 0 else 0
        
        for i in range(n):
            # Update exponential variance
            exp_var = self.lambda_param * exp_var + (1 - self.lambda_param) * returns[i] ** 2
            exp_std[i] = np.sqrt(exp_var)
        
        return exp_std
    
    def _calculate_conditional_std(self, returns: np.ndarray) -> np.ndarray:
        """Calculate conditional standard deviation (simplified GARCH)"""
        n = len(returns)
        cond_std = np.zeros(n)
        
        # GARCH(1,1) parameters (simplified)
        alpha0 = 0.01  # Long-run variance
        alpha1 = 0.05  # ARCH parameter
        beta1 = 0.9    # GARCH parameter
        
        # Initialize
        h = np.var(returns[:min(10, n)]) if n >= 10 else 0.01
        
        for i in range(n):
            # Update conditional variance
            if i > 0:
                h = alpha0 + alpha1 * returns[i-1] ** 2 + beta1 * h
            
            cond_std[i] = np.sqrt(h)
        
        return cond_std
    
    def _calculate_garch_volatility(self, returns: np.ndarray) -> np.ndarray:
        """Calculate GARCH(1,1) volatility with MLE estimation"""
        n = len(returns)
        
        # Simplified GARCH implementation
        # In practice, would use maximum likelihood estimation
        
        # Initial parameters
        omega = 0.01
        alpha = 0.05
        beta = 0.9
        
        # Initialize variance
        h = np.zeros(n)
        h[0] = np.var(returns) if n > 1 else 0.01
        
        for t in range(1, n):
            h[t] = omega + alpha * returns[t-1] ** 2 + beta * h[t-1]
        
        return np.sqrt(h)
    
    def _calculate_realized_volatility(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate realized volatility from high-frequency data simulation"""
        n = len(data)
        realized_vol = np.zeros(n)
        
        # Simulate intraday returns using OHLC data
        for i in range(1, n):
            # Parkinson estimator using high-low
            if 'high' in data.columns and 'low' in data.columns:
                high = data['high'].iloc[i]
                low = data['low'].iloc[i]
                
                if high > 0 and low > 0 and high >= low:
                    realized_vol[i] = np.sqrt(0.361 * (np.log(high/low)) ** 2)
                else:
                    realized_vol[i] = realized_vol[i-1] if i > 1 else 0.01
            else:
                # Fallback to close-to-close volatility
                if i >= self.window:
                    window_returns = np.diff(np.log(data['close'].iloc[i-self.window:i+1]))
                    realized_vol[i] = np.std(window_returns, ddof=1)
                else:
                    realized_vol[i] = 0.01
        
        return realized_vol
    
    def _calculate_volatility_of_volatility(self, rolling_std: np.ndarray) -> float:
        """Calculate volatility of volatility"""
        if len(rolling_std) < 2:
            return 0.0
        
        # Log volatility changes
        log_vol = np.log(rolling_std + 1e-10)
        vol_changes = np.diff(log_vol)
        
        return np.std(vol_changes, ddof=1)
    
    def _detect_volatility_clustering(self, returns: np.ndarray) -> np.ndarray:
        """Detect volatility clustering using squared returns autocorrelation"""
        n = len(returns)
        clustering = np.zeros(n)
        
        squared_returns = returns ** 2
        
        # Rolling autocorrelation of squared returns
        for i in range(20, n):
            window_sq_ret = squared_returns[i-20:i]
            
            # Autocorrelation at lag 1
            if len(window_sq_ret) > 1:
                autocorr = np.corrcoef(window_sq_ret[:-1], window_sq_ret[1:])[0, 1]
                clustering[i] = autocorr if not np.isnan(autocorr) else 0
        
        # Fill initial values
        clustering[:20] = clustering[20] if n > 20 else 0
        
        return clustering
    
    def _calculate_regime_specific_std(self, returns: np.ndarray) -> Dict[str, float]:
        """Calculate standard deviations for different market regimes"""
        if len(returns) < 30:
            return {'low_vol': 0.01, 'normal_vol': 0.02, 'high_vol': 0.04}
        
        # Simple regime classification based on rolling volatility
        rolling_vol = np.zeros(len(returns))
        for i in range(10, len(returns)):
            rolling_vol[i] = np.std(returns[i-10:i], ddof=1)
        
        # Percentile-based regime classification
        vol_33 = np.percentile(rolling_vol[10:], 33)
        vol_67 = np.percentile(rolling_vol[10:], 67)
        
        low_vol_returns = returns[rolling_vol <= vol_33]
        normal_vol_returns = returns[(rolling_vol > vol_33) & (rolling_vol <= vol_67)]
        high_vol_returns = returns[rolling_vol > vol_67]
        
        return {
            'low_vol': np.std(low_vol_returns, ddof=1) if len(low_vol_returns) > 1 else 0.01,
            'normal_vol': np.std(normal_vol_returns, ddof=1) if len(normal_vol_returns) > 1 else 0.02,
            'high_vol': np.std(high_vol_returns, ddof=1) if len(high_vol_returns) > 1 else 0.04
        }
    
    def _calculate_percentile_ranks(self, rolling_std: np.ndarray) -> np.ndarray:
        """Calculate percentile ranks of current volatility"""
        n = len(rolling_std)
        percentiles = np.zeros(n)
        
        for i in range(20, n):
            historical_vol = rolling_std[:i]
            current_vol = rolling_std[i]
            
            # Percentile rank
            rank = np.sum(historical_vol <= current_vol) / len(historical_vol)
            percentiles[i] = rank
        
        # Fill initial values
        percentiles[:20] = 0.5
        
        return percentiles
    
    def _calculate_z_scores(self, returns: np.ndarray, rolling_std: np.ndarray) -> np.ndarray:
        """Calculate standardized returns (z-scores)"""
        n = len(returns)
        z_scores = np.zeros(n)
        
        # Calculate rolling mean
        rolling_mean = np.zeros(n)
        for i in range(self.window, n):
            rolling_mean[i] = np.mean(returns[i-self.window:i])
        
        # Z-scores
        for i in range(n):
            if rolling_std[i] > 0:
                z_scores[i] = (returns[i] - rolling_mean[i]) / rolling_std[i]
        
        return z_scores
    
    def _forecast_volatility(self, returns: np.ndarray, rolling_std: np.ndarray) -> np.ndarray:
        """Forecast future volatility using simple models"""
        n = len(returns)
        forecast = np.zeros(n)
        
        for i in range(30, n):
            # Historical volatility
            hist_vol = rolling_std[i]
            
            # Trend in volatility
            recent_vol = rolling_std[max(0, i-10):i]
            if len(recent_vol) >= 2:
                vol_trend = np.polyfit(range(len(recent_vol)), recent_vol, 1)[0]
            else:
                vol_trend = 0
            
            # Mean reversion component
            long_term_vol = np.mean(rolling_std[max(0, i-60):i])
            mean_reversion = (long_term_vol - hist_vol) * 0.1
            
            # Forecast
            forecast[i] = hist_vol + vol_trend + mean_reversion
            forecast[i] = max(forecast[i], 0.001)  # Ensure positive
        
        # Fill initial values
        forecast[:30] = rolling_std[:30]
        
        return forecast
    
    def _calculate_risk_metrics(self, returns: np.ndarray, rolling_std: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive risk metrics"""
        # Value at Risk
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        # Expected Shortfall
        es_95 = np.mean(returns[returns <= var_95]) if np.any(returns <= var_95) else var_95
        
        # Sharpe ratio (assuming zero risk-free rate)
        sharpe = np.mean(returns) / np.std(returns, ddof=1) if np.std(returns) > 0 else 0
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns, ddof=1) if len(downside_returns) > 1 else np.std(returns, ddof=1)
        sortino = np.mean(returns) / downside_std if downside_std > 0 else 0
        
        # Maximum drawdown
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = cumulative - running_max
        max_drawdown = np.min(drawdowns)
        
        # Calmar ratio
        annualized_return = np.mean(returns) * 252
        calmar = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'var_95': var_95,
            'var_99': var_99,
            'expected_shortfall_95': es_95,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar
        }
    
    def _calculate_volatility_smile(self, returns: np.ndarray) -> Dict[str, float]:
        """Calculate volatility smile characteristics"""
        # Quantile-based volatility analysis
        quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
        quantile_vols = {}
        
        for q in quantiles:
            threshold = np.percentile(returns, q * 100)
            extreme_returns = returns[returns <= threshold] if q < 0.5 else returns[returns >= threshold]
            
            if len(extreme_returns) > 1:
                quantile_vols[f'vol_{int(q*100)}'] = np.std(extreme_returns, ddof=1)
            else:
                quantile_vols[f'vol_{int(q*100)}'] = np.std(returns, ddof=1)
        
        # Skewness and kurtosis
        skewness = stats.skew(returns, bias=False)
        kurtosis = stats.kurtosis(returns, bias=False)
        
        quantile_vols.update({
            'skewness': skewness,
            'kurtosis': kurtosis
        })
        
        return quantile_vols
    
    def _calculate_autocorrelation_structure(self, returns: np.ndarray) -> np.ndarray:
        """Calculate autocorrelation structure of returns and squared returns"""
        max_lag = min(20, len(returns) // 4)
        
        autocorr = np.zeros(max_lag)
        squared_returns = returns ** 2
        
        for lag in range(1, max_lag + 1):
            if len(returns) > lag:
                # Autocorrelation of squared returns (volatility clustering)
                corr = np.corrcoef(squared_returns[:-lag], squared_returns[lag:])[0, 1]
                autocorr[lag-1] = corr if not np.isnan(corr) else 0
        
        return autocorr
    
    def _generate_signal(self, result: StandardDeviationResult, returns: np.ndarray, prices: np.ndarray) -> SignalType:
        """Generate trading signal based on volatility analysis"""
        if len(returns) < 10:
            return SignalType.HOLD
        
        current_std = result.rolling_std[-1]
        exp_std = result.exponential_std[-1]
        vol_percentile = result.percentile_ranks[-1]
        z_score = result.z_scores[-1]
        vol_forecast = result.volatility_forecast[-1]
        clustering = result.volatility_clustering[-1]
        
        # Volatility regime analysis
        regime_std = result.regime_specific_std
        
        # Low volatility regime - potential for trend following
        if vol_percentile < 0.2:  # Bottom 20% of volatility
            # Strong momentum signal in low volatility
            if abs(z_score) > 1.5:
                return SignalType.BUY if z_score > 0 else SignalType.SELL
        
        # High volatility regime - mean reversion signals
        elif vol_percentile > 0.8:  # Top 20% of volatility
            # Mean reversion in high volatility
            if z_score > 2:
                return SignalType.SELL  # Overextended upside
            elif z_score < -2:
                return SignalType.BUY   # Overextended downside
        
        # Volatility breakout signals
        vol_expansion = current_std / exp_std if exp_std > 0 else 1
        if vol_expansion > 1.5:  # Volatility expansion
            # Direction based on recent price action
            recent_return = returns[-1] if len(returns) > 0 else 0
            if recent_return > 0:
                return SignalType.BUY   # Bullish breakout
            else:
                return SignalType.SELL  # Bearish breakdown
        
        # Volatility compression signals
        elif vol_expansion < 0.7:  # Volatility compression
            # Expect breakout, use trend direction
            if len(prices) >= 10:
                trend = (prices[-1] - prices[-10]) / prices[-10]
                if trend > 0.02:  # Uptrend
                    return SignalType.BUY
                elif trend < -0.02:  # Downtrend
                    return SignalType.SELL
        
        # Volatility clustering signals
        if clustering > 0.5:  # Strong clustering
            # Continue current volatility direction
            vol_change = current_std - result.rolling_std[-2] if len(result.rolling_std) > 1 else 0
            if vol_change > 0 and vol_percentile > 0.6:
                return SignalType.SELL  # Increasing volatility, potential stress
        
        return SignalType.HOLD
    
    def _calculate_signal_strength(self, result: StandardDeviationResult) -> float:
        """Calculate signal strength based on volatility characteristics"""
        vol_percentile = result.percentile_ranks[-1]
        z_score = abs(result.z_scores[-1]) if len(result.z_scores) > 0 else 0
        clustering = abs(result.volatility_clustering[-1])
        
        # Extremeness strength
        extremeness = max(vol_percentile, 1 - vol_percentile) * 2
        
        # Z-score strength
        z_strength = min(z_score / 3, 1)  # Normalize by 3 standard deviations
        
        # Clustering strength
        cluster_strength = clustering
        
        # Forecast accuracy (simplified)
        forecast_strength = 0.7  # Placeholder for forecast accuracy
        
        return (extremeness + z_strength + cluster_strength + forecast_strength) / 4
    
    def _calculate_confidence(self, result: StandardDeviationResult) -> float:
        """Calculate confidence based on volatility model quality"""
        # Volatility persistence
        autocorr_strength = np.mean(np.abs(result.autocorrelation_structure))
        
        # Regime stability
        regime_consistency = 1 / (1 + result.volatility_of_volatility)
        
        # Model consistency (GARCH vs rolling)
        if len(result.rolling_std) > 0 and len(result.garch_std) > 0:
            model_consistency = np.corrcoef(result.rolling_std[-20:], result.garch_std[-20:])[0, 1]
            model_consistency = abs(model_consistency) if not np.isnan(model_consistency) else 0.5
        else:
            model_consistency = 0.5
        
        # Risk metric reliability
        risk_reliability = min(abs(result.risk_metrics['sharpe_ratio']) / 2, 1)
        
        return (autocorr_strength + regime_consistency + model_consistency + risk_reliability) / 4