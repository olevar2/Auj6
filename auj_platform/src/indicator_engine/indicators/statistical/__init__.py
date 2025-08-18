"""
Statistical Indicators Package

This package provides advanced statistical indicators for quantitative trading
and market analysis. Each indicator implements sophisticated statistical methods
with comprehensive error handling and signal generation.

Available Indicators:
- AutocorrelationIndicator: Multi-lag autocorrelation analysis with statistical testing
- BetaCoefficientIndicator: Multi-factor beta analysis with confidence intervals  
- CointegrationIndicator: Johansen and Engle-Granger cointegration testing
- CorrelationAnalysisIndicator: Dynamic conditional correlation with breakpoint detection
- CorrelationCoefficientIndicator: Multiple correlation measures with bootstrap confidence
- GarchVolatilityModelIndicator: GARCH family models with volatility forecasting
- HurstExponentIndicator: Long memory analysis with regime detection
- KalmanFilterIndicator: Multi-dimensional state estimation with uncertainty quantification
- LinearRegressionChannelsIndicator: Robust regression channels with dynamic boundaries
- MarketRegimeDetectionIndicator: Hidden Markov model regime switching
- RSquaredIndicator: Goodness-of-fit analysis with model validation
- SkewnessIndicator: Distribution asymmetry analysis with tail risk assessment
- StandardDeviationChannelsIndicator: Dynamic volatility channels with breakout detection
- StandardDeviationIndicator: Multi-scale volatility analysis with risk metrics
- VarianceRatioIndicator: Market efficiency testing with autocorrelation analysis
- ZScoreIndicator: Multi-scale statistical outlier detection with regime awareness

All indicators follow the StandardIndicatorInterface and provide:
- Advanced mathematical implementations (no placeholders)
- Comprehensive error handling and validation
- Signal generation with strength and confidence metrics
- Production-ready code quality
- Statistical rigor and robustness
"""

from .autocorrelation_indicator import AutocorrelationIndicator
from .beta_coefficient_indicator import BetaCoefficientIndicator
from .cointegration_indicator import CointegrationIndicator
from .correlation_analysis_indicator import CorrelationAnalysisIndicator
from .correlation_coefficient_indicator import CorrelationCoefficientIndicator
from .garch_volatility_model_indicator import GarchVolatilityModelIndicator
from .hurst_exponent_indicator import HurstExponentIndicator
from .kalman_filter_indicator import KalmanFilterIndicator
from .linear_regression_channels_indicator import LinearRegressionChannelsIndicator
from .market_regime_detection_indicator import MarketRegimeDetectionIndicator
from .rsquared_indicator import RSquaredIndicator
from .skewness_indicator import SkewnessIndicator
from .standard_deviation_channels_indicator import StandardDeviationChannelsIndicator
from .standard_deviation_indicator import StandardDeviationIndicator
from .variance_ratio_indicator import VarianceRatioIndicator
from .zscore_indicator import ZScoreIndicator

__all__ = [
    'AutocorrelationIndicator',
    'BetaCoefficientIndicator', 
    'CointegrationIndicator',
    'CorrelationAnalysisIndicator',
    'CorrelationCoefficientIndicator',
    'GarchVolatilityModelIndicator',
    'HurstExponentIndicator',
    'KalmanFilterIndicator',
    'LinearRegressionChannelsIndicator',
    'MarketRegimeDetectionIndicator',
    'RSquaredIndicator',
    'SkewnessIndicator',
    'StandardDeviationChannelsIndicator',
    'StandardDeviationIndicator',
    'VarianceRatioIndicator',
    'ZScoreIndicator'
]

# Statistical indicator categories for easy organization
AUTOCORRELATION_INDICATORS = [
    'AutocorrelationIndicator',
    'VarianceRatioIndicator'
]

REGRESSION_INDICATORS = [
    'BetaCoefficientIndicator',
    'LinearRegressionChannelsIndicator', 
    'RSquaredIndicator'
]

CORRELATION_INDICATORS = [
    'CointegrationIndicator',
    'CorrelationAnalysisIndicator',
    'CorrelationCoefficientIndicator'
]

VOLATILITY_INDICATORS = [
    'GarchVolatilityModelIndicator',
    'StandardDeviationChannelsIndicator',
    'StandardDeviationIndicator'
]

DISTRIBUTION_INDICATORS = [
    'SkewnessIndicator',
    'ZScoreIndicator'
]

REGIME_INDICATORS = [
    'HurstExponentIndicator',
    'KalmanFilterIndicator',
    'MarketRegimeDetectionIndicator'
]

# Version information
__version__ = '1.0.0'
__author__ = 'Advanced Statistical Indicators Team'
__description__ = 'Professional statistical indicators for quantitative trading'