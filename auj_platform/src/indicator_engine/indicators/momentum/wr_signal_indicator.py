"""
Williams %R Signal Indicator with Advanced ML and Multi-Dimensional Analysis

This indicator extends the classic Williams %R oscillator with sophisticated signal generation,
machine learning integration, multi-timeframe analysis, and advanced pattern recognition.
Designed for maximum profitability in support of the humanitarian mission.

Features:
- Advanced Williams %R calculation with multiple lookback periods
- Machine learning for signal classification and strength prediction
- Multi-timeframe analysis and regime detection
- Divergence detection and confluence analysis
- Dynamic threshold adaptation based on market conditions
- Risk-adjusted signal scoring with confidence intervals
- Pattern recognition for reversal and continuation signals
- Volatility-adjusted signal generation
- Real-time signal filtering and validation

Author: AUJ Platform Development Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import stats
from scipy.signal import find_peaks, savgol_filter
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

from ..base.standard_indicator import StandardIndicatorInterface
from ....core.unified_config import get_unified_config


class WRSignalType(Enum):
    """Williams %R Signal Types"""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    WEAK_BUY = "WEAK_BUY"
    NEUTRAL = "NEUTRAL"
    WEAK_SELL = "WEAK_SELL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


class MarketRegime(Enum):
    """Market Regime Classification"""
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    VOLATILE = "VOLATILE"
    BREAKOUT = "BREAKOUT"


@dataclass
class WRSignalResult:
    """Williams %R Signal Analysis Result"""
    signal_type: WRSignalType
    signal_strength: float
    confidence_score: float
    risk_score: float
    wr_value: float
    wr_smoothed: float
    signal_quality: float
    divergence_detected: bool
    regime: MarketRegime
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    timeframe_confluence: Dict[str, float] = None
    pattern_signals: List[str] = None


class WRSignalIndicator(StandardIndicatorInterface):
    """
    Advanced Williams %R Signal Indicator with ML Integration
    
    This indicator provides sophisticated signal generation based on Williams %R
    with machine learning enhancement and multi-dimensional analysis.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Williams %R Signal Indicator
        
        Args:
            config: Configuration dictionary with parameters
        """
        super().__init__(config)
        self.config_manager = get_unified_config()
        
        # Core parameters
        self.primary_period = self.config_manager.get_int('primary_period', 14)
        self.secondary_periods = self.config_manager.get_dict('secondary_periods', [7, 21, 50])
        self.smoothing_period = self.config_manager.get_int('smoothing_period', 3)
        self.signal_threshold = self.config_manager.get_float('signal_threshold', 20.0)
        
        # Advanced parameters
        self.lookback_window = self.config_manager.get_int('lookback_window', 100)
        self.min_samples = self.config_manager.get_int('min_samples', 50)
        self.confidence_threshold = self.config_manager.get_float('confidence_threshold', 0.7)
        self.risk_threshold = self.config_manager.get_float('risk_threshold', 0.3)
        
        # Multi-timeframe settings
        self.timeframes = self.config_manager.get_dict('timeframes', ['1H', '4H', '1D'])
        self.timeframe_weights = self.config_manager.get_dict('timeframe_weights', [0.3, 0.4, 0.3])
        
        # ML models
        self.signal_classifier = None
        self.strength_predictor = None
        self.confidence_model = None
        self.scaler = StandardScaler()
        
        # Adaptive thresholds
        self.adaptive_thresholds = True
        self.threshold_periods = self.config_manager.get_int('threshold_periods', 50)
        
        # Pattern recognition
        self.pattern_detection = True
        self.divergence_periods = self.config_manager.get_int('divergence_periods', 20)
        
        # Risk management
        self.risk_adjustment = True
        self.volatility_window = self.config_manager.get_int('volatility_window', 20)
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # Cache for performance
        self._cache = {}
        self._last_update = None
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Williams %R signals with advanced analysis
        
        Args:
            data: OHLCV data with columns ['open', 'high', 'low', 'close', 'volume']
            
        Returns:
            DataFrame with Williams %R signal analysis
        """
        try:
            if len(data) < self.min_samples:
                raise ValueError(f"Insufficient data: need {self.min_samples}, got {len(data)}")
            
            # Validate required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Initialize results
            results = data.copy()
            
            # Calculate core Williams %R values
            wr_data = self._calculate_williams_r(data)
            results = pd.concat([results, wr_data], axis=1)
            
            # Calculate multi-period analysis
            multi_period_data = self._calculate_multi_period_analysis(data)
            results = pd.concat([results, multi_period_data], axis=1)
            
            # Perform regime detection
            regime_data = self._detect_market_regime(data, results)
            results = pd.concat([results, regime_data], axis=1)
            
            # Calculate adaptive thresholds
            if self.adaptive_thresholds:
                threshold_data = self._calculate_adaptive_thresholds(results)
                results = pd.concat([results, threshold_data], axis=1)
            
            # Detect patterns and divergences
            if self.pattern_detection:
                pattern_data = self._detect_patterns_and_divergences(data, results)
                results = pd.concat([results, pattern_data], axis=1)
            
            # Calculate signal features
            features_data = self._calculate_signal_features(data, results)
            results = pd.concat([results, features_data], axis=1)
            
            # Train/update ML models
            if len(results) >= self.min_samples:
                self._update_ml_models(results)
            
            # Generate signals
            signals_data = self._generate_signals(results)
            results = pd.concat([results, signals_data], axis=1)
            
            # Calculate multi-timeframe confluence
            confluence_data = self._calculate_timeframe_confluence(results)
            results = pd.concat([results, confluence_data], axis=1)
            
            # Perform risk analysis
            risk_data = self._calculate_risk_metrics(data, results)
            results = pd.concat([results, risk_data], axis=1)
            
            # Generate final signal recommendations
            final_signals = self._generate_final_signals(results)
            results = pd.concat([results, final_signals], axis=1)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in WR Signal calculation: {str(e)}")
            raise
    
    def _calculate_williams_r(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Williams %R with multiple periods"""
        try:
            results = pd.DataFrame(index=data.index)
            
            # Primary Williams %R
            highest_high = data['high'].rolling(window=self.primary_period).max()
            lowest_low = data['low'].rolling(window=self.primary_period).min()
            results['wr_primary'] = -100 * (highest_high - data['close']) / (highest_high - lowest_low)
            
            # Smoothed Williams %R
            results['wr_smoothed'] = results['wr_primary'].rolling(window=self.smoothing_period).mean()
            
            # Williams %R momentum
            results['wr_momentum'] = results['wr_primary'].diff(self.smoothing_period)
            
            # Williams %R velocity (rate of change of momentum)
            results['wr_velocity'] = results['wr_momentum'].diff()
            
            # Williams %R acceleration
            results['wr_acceleration'] = results['wr_velocity'].diff()
            
            # Normalized Williams %R (0-1 scale)
            results['wr_normalized'] = (results['wr_primary'] + 100) / 100
            
            # Williams %R standard deviation
            results['wr_std'] = results['wr_primary'].rolling(window=self.primary_period).std()
            
            # Williams %R z-score
            wr_mean = results['wr_primary'].rolling(window=self.primary_period).mean()
            results['wr_zscore'] = (results['wr_primary'] - wr_mean) / results['wr_std']
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error calculating Williams %R: {str(e)}")
            raise
    
    def _calculate_multi_period_analysis(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Williams %R for multiple periods"""
        try:
            results = pd.DataFrame(index=data.index)
            
            for i, period in enumerate(self.secondary_periods):
                highest_high = data['high'].rolling(window=period).max()
                lowest_low = data['low'].rolling(window=period).min()
                wr_value = -100 * (highest_high - data['close']) / (highest_high - lowest_low)
                
                results[f'wr_{period}'] = wr_value
                results[f'wr_{period}_smoothed'] = wr_value.rolling(window=self.smoothing_period).mean()
                
                # Calculate momentum for each period
                results[f'wr_{period}_momentum'] = wr_value.diff(self.smoothing_period)
                
                # Calculate relative strength between periods
                if i > 0:
                    prev_period = self.secondary_periods[i-1]
                    results[f'wr_strength_{prev_period}_{period}'] = (
                        results[f'wr_{prev_period}'] - results[f'wr_{period}']
                    )
            
            # Calculate period consensus
            wr_columns = [f'wr_{period}' for period in self.secondary_periods]
            results['wr_consensus'] = results[wr_columns].mean(axis=1)
            results['wr_consensus_std'] = results[wr_columns].std(axis=1)
            
            # Calculate period divergence
            results['wr_period_divergence'] = results['wr_consensus_std'] / abs(results['wr_consensus'])
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in multi-period analysis: {str(e)}")
            raise
    
    def _detect_market_regime(self, data: pd.DataFrame, wr_data: pd.DataFrame) -> pd.DataFrame:
        """Detect current market regime"""
        try:
            results = pd.DataFrame(index=data.index)
            
            # Calculate trend indicators
            sma_short = data['close'].rolling(window=20).mean()
            sma_long = data['close'].rolling(window=50).mean()
            trend_direction = np.where(sma_short > sma_long, 1, -1)
            
            # Calculate volatility
            returns = data['close'].pct_change()
            volatility = returns.rolling(window=20).std() * np.sqrt(252)
            vol_percentile = volatility.rolling(window=100).rank(pct=True)
            
            # Calculate momentum
            momentum = data['close'].pct_change(20)
            momentum_percentile = momentum.rolling(window=100).rank(pct=True)
            
            # Regime classification
            regime_scores = pd.DataFrame(index=data.index)
            
            # Trending up conditions
            regime_scores['trending_up'] = (
                (trend_direction > 0) * 0.4 +
                (momentum_percentile > 0.6) * 0.3 +
                (vol_percentile < 0.7) * 0.3
            )
            
            # Trending down conditions
            regime_scores['trending_down'] = (
                (trend_direction < 0) * 0.4 +
                (momentum_percentile < 0.4) * 0.3 +
                (vol_percentile < 0.7) * 0.3
            )
            
            # Ranging conditions
            range_indicator = abs(momentum_percentile - 0.5) < 0.2
            regime_scores['ranging'] = (
                range_indicator * 0.5 +
                (vol_percentile < 0.5) * 0.5
            )
            
            # Volatile conditions
            regime_scores['volatile'] = vol_percentile > 0.8
            
            # Breakout conditions
            breakout_indicator = (abs(momentum_percentile - 0.5) > 0.4) & (vol_percentile > 0.6)
            regime_scores['breakout'] = breakout_indicator.astype(float)
            
            # Determine primary regime
            regime_columns = ['trending_up', 'trending_down', 'ranging', 'volatile', 'breakout']
            results['regime_primary'] = regime_scores[regime_columns].idxmax(axis=1)
            results['regime_confidence'] = regime_scores[regime_columns].max(axis=1)
            
            # Add regime scores
            for col in regime_columns:
                results[f'regime_score_{col}'] = regime_scores[col]
            
            # Calculate regime persistence
            results['regime_persistence'] = (
                results['regime_primary'] == results['regime_primary'].shift(1)
            ).rolling(window=10).mean()
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in regime detection: {str(e)}")
            raise
    
    def _calculate_adaptive_thresholds(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate adaptive thresholds based on market conditions"""
        try:
            results = pd.DataFrame(index=data.index)
            
            # Calculate rolling percentiles for adaptive thresholds
            wr_values = data['wr_primary'].rolling(window=self.threshold_periods)
            
            results['wr_threshold_upper'] = wr_values.quantile(0.8)
            results['wr_threshold_lower'] = wr_values.quantile(0.2)
            results['wr_threshold_extreme_upper'] = wr_values.quantile(0.95)
            results['wr_threshold_extreme_lower'] = wr_values.quantile(0.05)
            
            # Volatility-adjusted thresholds
            volatility = data['wr_std'].rolling(window=20).mean()
            vol_adjustment = np.clip(volatility / volatility.rolling(window=50).mean(), 0.5, 2.0)
            
            base_threshold = self.signal_threshold
            results['wr_threshold_buy'] = -100 + base_threshold * vol_adjustment
            results['wr_threshold_sell'] = -base_threshold * vol_adjustment
            
            # Regime-adjusted thresholds
            regime_multiplier = pd.Series(1.0, index=data.index)
            
            if 'regime_primary' in data.columns:
                regime_multiplier.loc[data['regime_primary'] == 'volatile'] *= 1.5
                regime_multiplier.loc[data['regime_primary'] == 'ranging'] *= 0.8
                regime_multiplier.loc[data['regime_primary'] == 'breakout'] *= 1.2
            
            results['wr_threshold_buy_adjusted'] = results['wr_threshold_buy'] * regime_multiplier
            results['wr_threshold_sell_adjusted'] = results['wr_threshold_sell'] * regime_multiplier
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error calculating adaptive thresholds: {str(e)}")
            raise
    
    def _detect_patterns_and_divergences(self, price_data: pd.DataFrame, wr_data: pd.DataFrame) -> pd.DataFrame:
        """Detect patterns and divergences in Williams %R"""
        try:
            results = pd.DataFrame(index=price_data.index)
            
            # Bullish divergence detection
            price_lows = self._find_local_extrema(price_data['low'], 'min')
            wr_lows = self._find_local_extrema(wr_data['wr_primary'], 'min')
            
            bullish_divergence = self._detect_divergence(
                price_lows, wr_lows, 'bullish', self.divergence_periods
            )
            results['bullish_divergence'] = bullish_divergence
            
            # Bearish divergence detection
            price_highs = self._find_local_extrema(price_data['high'], 'max')
            wr_highs = self._find_local_extrema(wr_data['wr_primary'], 'max')
            
            bearish_divergence = self._detect_divergence(
                price_highs, wr_highs, 'bearish', self.divergence_periods
            )
            results['bearish_divergence'] = bearish_divergence
            
            # Pattern recognition
            results['wr_double_top'] = self._detect_double_pattern(wr_data['wr_primary'], 'top')
            results['wr_double_bottom'] = self._detect_double_pattern(wr_data['wr_primary'], 'bottom')
            
            # Momentum patterns
            results['wr_momentum_increasing'] = (
                wr_data['wr_momentum'] > wr_data['wr_momentum'].shift(1)
            ).rolling(window=3).sum() >= 2
            
            results['wr_momentum_decreasing'] = (
                wr_data['wr_momentum'] < wr_data['wr_momentum'].shift(1)
            ).rolling(window=3).sum() >= 2
            
            # Oversold/Overbought patterns
            if 'wr_threshold_buy_adjusted' in wr_data.columns:
                results['wr_oversold'] = wr_data['wr_primary'] < wr_data['wr_threshold_buy_adjusted']
                results['wr_overbought'] = wr_data['wr_primary'] > wr_data['wr_threshold_sell_adjusted']
            else:
                results['wr_oversold'] = wr_data['wr_primary'] < -80
                results['wr_overbought'] = wr_data['wr_primary'] > -20
            
            # Pattern strength
            pattern_columns = ['bullish_divergence', 'bearish_divergence', 'wr_double_top', 'wr_double_bottom']
            results['pattern_strength'] = results[pattern_columns].sum(axis=1)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in pattern detection: {str(e)}")
            raise
    
    def _find_local_extrema(self, series: pd.Series, extrema_type: str, distance: int = 5) -> pd.Series:
        """Find local extrema in a time series"""
        try:
            if extrema_type == 'min':
                peaks, _ = find_peaks(-series.values, distance=distance)
            else:  # max
                peaks, _ = find_peaks(series.values, distance=distance)
            
            extrema = pd.Series(False, index=series.index)
            if len(peaks) > 0:
                extrema.iloc[peaks] = True
            
            return extrema
            
        except Exception as e:
            self.logger.error(f"Error finding local extrema: {str(e)}")
            return pd.Series(False, index=series.index)
    
    def _detect_divergence(self, price_extrema: pd.Series, indicator_extrema: pd.Series, 
                          div_type: str, lookback: int) -> pd.Series:
        """Detect price-indicator divergence"""
        try:
            divergence = pd.Series(False, index=price_extrema.index)
            
            for i in range(lookback, len(price_extrema)):
                current_idx = price_extrema.index[i]
                window_start = max(0, i - lookback)
                
                # Find previous extrema in lookback window
                price_window = price_extrema.iloc[window_start:i]
                indicator_window = indicator_extrema.iloc[window_start:i]
                
                if price_window.any() and indicator_window.any():
                    prev_price_peak = price_window.idxmax() if div_type == 'bearish' else price_window.idxmin()
                    prev_indicator_peak = indicator_window.idxmax() if div_type == 'bearish' else indicator_window.idxmin()
                    
                    if price_extrema.iloc[i] and indicator_extrema.iloc[i]:
                        if div_type == 'bullish':
                            # Price makes lower low, indicator makes higher low
                            price_condition = price_extrema.loc[current_idx] < price_extrema.loc[prev_price_peak]
                            indicator_condition = indicator_extrema.loc[current_idx] > indicator_extrema.loc[prev_indicator_peak]
                        else:  # bearish
                            # Price makes higher high, indicator makes lower high
                            price_condition = price_extrema.loc[current_idx] > price_extrema.loc[prev_price_peak]
                            indicator_condition = indicator_extrema.loc[current_idx] < indicator_extrema.loc[prev_indicator_peak]
                        
                        if price_condition and indicator_condition:
                            divergence.iloc[i] = True
            
            return divergence
            
        except Exception as e:
            self.logger.error(f"Error detecting divergence: {str(e)}")
            return pd.Series(False, index=price_extrema.index)
    
    def _detect_double_pattern(self, series: pd.Series, pattern_type: str, tolerance: float = 0.02) -> pd.Series:
        """Detect double top/bottom patterns"""
        try:
            pattern = pd.Series(False, index=series.index)
            
            if pattern_type == 'top':
                extrema = self._find_local_extrema(series, 'max')
            else:
                extrema = self._find_local_extrema(series, 'min')
            
            extrema_indices = extrema[extrema].index.tolist()
            
            for i in range(1, len(extrema_indices)):
                current_idx = extrema_indices[i]
                prev_idx = extrema_indices[i-1]
                
                current_value = series.loc[current_idx]
                prev_value = series.loc[prev_idx]
                
                # Check if values are similar (within tolerance)
                if abs(current_value - prev_value) / abs(prev_value) <= tolerance:
                    pattern.loc[current_idx] = True
            
            return pattern
            
        except Exception as e:
            self.logger.error(f"Error detecting double pattern: {str(e)}")
            return pd.Series(False, index=series.index)
    
    def _calculate_signal_features(self, price_data: pd.DataFrame, wr_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate features for ML signal generation"""
        try:
            results = pd.DataFrame(index=price_data.index)
            
            # Price-based features
            returns = price_data['close'].pct_change()
            results['return_1d'] = returns
            results['return_5d'] = price_data['close'].pct_change(5)
            results['return_volatility'] = returns.rolling(window=20).std()
            
            # Volume features
            results['volume_ratio'] = (
                price_data['volume'] / price_data['volume'].rolling(window=20).mean()
            )
            results['price_volume_correlation'] = (
                returns.rolling(window=20).corr(price_data['volume'].pct_change())
            )
            
            # Williams %R features
            results['wr_position'] = (wr_data['wr_primary'] + 100) / 100  # Normalized position
            results['wr_momentum_strength'] = abs(wr_data['wr_momentum'])
            results['wr_velocity_strength'] = abs(wr_data['wr_velocity'])
            
            # Cross-period features
            if 'wr_7' in wr_data.columns and 'wr_21' in wr_data.columns:
                results['wr_cross_7_21'] = wr_data['wr_7'] - wr_data['wr_21']
                results['wr_cross_signal'] = (
                    (wr_data['wr_7'] > wr_data['wr_21']) & 
                    (wr_data['wr_7'].shift(1) <= wr_data['wr_21'].shift(1))
                ).astype(float)
            
            # Regime features
            if 'regime_confidence' in wr_data.columns:
                results['regime_confidence'] = wr_data['regime_confidence']
                results['regime_persistence'] = wr_data['regime_persistence']
            
            # Pattern features
            pattern_cols = [col for col in wr_data.columns if 'divergence' in col or 'pattern' in col]
            if pattern_cols:
                results['pattern_count'] = wr_data[pattern_cols].sum(axis=1)
            
            # Technical features
            results['wr_percentile'] = wr_data['wr_primary'].rolling(window=50).rank(pct=True)
            results['wr_distance_from_mean'] = (
                wr_data['wr_primary'] - wr_data['wr_primary'].rolling(window=20).mean()
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error calculating signal features: {str(e)}")
            raise
    
    def _update_ml_models(self, data: pd.DataFrame) -> None:
        """Update ML models with latest data"""
        try:
            if len(data) < self.min_samples:
                return
            
            # Prepare features
            feature_columns = [
                'return_1d', 'return_5d', 'return_volatility', 'volume_ratio',
                'wr_position', 'wr_momentum_strength', 'wr_velocity_strength',
                'wr_percentile', 'wr_distance_from_mean'
            ]
            
            # Add optional features if available
            optional_features = [
                'wr_cross_7_21', 'regime_confidence', 'pattern_count',
                'price_volume_correlation'
            ]
            
            available_features = [col for col in feature_columns + optional_features if col in data.columns]
            
            if len(available_features) < 5:
                self.logger.warning("Insufficient features for ML model training")
                return
            
            # Prepare training data
            X = data[available_features].dropna()
            
            if len(X) < self.min_samples:
                return
            
            # Create target variables
            future_returns = data['close'].shift(-5) / data['close'] - 1
            
            # Signal classification (buy/sell/hold)
            y_signal = pd.cut(future_returns, bins=[-np.inf, -0.02, 0.02, np.inf], 
                            labels=['SELL', 'HOLD', 'BUY'])
            
            # Signal strength (absolute future return)
            y_strength = abs(future_returns)
            
            # Align data
            valid_idx = X.index.intersection(y_signal.dropna().index)
            X_valid = X.loc[valid_idx]
            y_signal_valid = y_signal.loc[valid_idx]
            y_strength_valid = y_strength.loc[valid_idx]
            
            if len(X_valid) < self.min_samples:
                return
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X_valid)
            
            # Train signal classifier
            self.signal_classifier = RandomForestClassifier(
                n_estimators=100, random_state=42, max_depth=10
            )
            self.signal_classifier.fit(X_scaled, y_signal_valid)
            
            # Train strength predictor
            self.strength_predictor = GradientBoostingRegressor(
                n_estimators=100, random_state=42, max_depth=6
            )
            self.strength_predictor.fit(X_scaled, y_strength_valid)
            
            # Calculate model performance
            signal_score = cross_val_score(self.signal_classifier, X_scaled, y_signal_valid, cv=3).mean()
            strength_score = cross_val_score(self.strength_predictor, X_scaled, y_strength_valid, cv=3).mean()
            
            self.logger.info(f"ML models updated - Signal accuracy: {signal_score:.3f}, Strength R²: {strength_score:.3f}")
            
        except Exception as e:
            self.logger.error(f"Error updating ML models: {str(e)}")
    
    def _generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate Williams %R signals"""
        try:
            results = pd.DataFrame(index=data.index)
            
            # Initialize signals
            results['signal_type'] = WRSignalType.NEUTRAL.value
            results['signal_strength'] = 0.0
            results['signal_confidence'] = 0.0
            
            # Rule-based signals
            if 'wr_threshold_buy_adjusted' in data.columns:
                buy_threshold = data['wr_threshold_buy_adjusted']
                sell_threshold = data['wr_threshold_sell_adjusted']
            else:
                buy_threshold = -80
                sell_threshold = -20
            
            # Basic oversold/overbought signals
            oversold = data['wr_primary'] < buy_threshold
            overbought = data['wr_primary'] > sell_threshold
            
            # Enhanced signal conditions
            bullish_momentum = data['wr_momentum'] > 0
            bearish_momentum = data['wr_momentum'] < 0
            
            # Pattern confirmation
            bullish_patterns = data.get('bullish_divergence', False) | data.get('wr_double_bottom', False)
            bearish_patterns = data.get('bearish_divergence', False) | data.get('wr_double_top', False)
            
            # Generate base signals
            strong_buy = oversold & bullish_momentum & bullish_patterns
            buy = oversold & bullish_momentum
            weak_buy = oversold & ~bearish_momentum
            
            strong_sell = overbought & bearish_momentum & bearish_patterns
            sell = overbought & bearish_momentum
            weak_sell = overbought & ~bullish_momentum
            
            # Apply signals
            results.loc[strong_buy, 'signal_type'] = WRSignalType.STRONG_BUY.value
            results.loc[buy, 'signal_type'] = WRSignalType.BUY.value
            results.loc[weak_buy, 'signal_type'] = WRSignalType.WEAK_BUY.value
            results.loc[strong_sell, 'signal_type'] = WRSignalType.STRONG_SELL.value
            results.loc[sell, 'signal_type'] = WRSignalType.SELL.value
            results.loc[weak_sell, 'signal_type'] = WRSignalType.WEAK_SELL.value
            
            # Calculate signal strength
            wr_extreme = np.maximum(abs(data['wr_primary'] + 100), abs(data['wr_primary']))
            momentum_strength = abs(data['wr_momentum']) / (abs(data['wr_momentum']).rolling(window=20).mean() + 1e-6)
            
            results['signal_strength'] = np.clip(
                (wr_extreme / 100) * momentum_strength * 
                (1 + data.get('pattern_strength', 0) * 0.2), 0, 1
            )
            
            # ML enhancement if models are available
            if self.signal_classifier is not None and self.strength_predictor is not None:
                ml_signals = self._apply_ml_signals(data, results)
                results = pd.concat([results, ml_signals], axis=1)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {str(e)}")
            raise
    
    def _apply_ml_signals(self, data: pd.DataFrame, base_signals: pd.DataFrame) -> pd.DataFrame:
        """Apply ML models to enhance signals"""
        try:
            results = pd.DataFrame(index=data.index)
            
            # Prepare features for ML prediction
            feature_columns = [
                'return_1d', 'return_5d', 'return_volatility', 'volume_ratio',
                'wr_position', 'wr_momentum_strength', 'wr_velocity_strength',
                'wr_percentile', 'wr_distance_from_mean'
            ]
            
            optional_features = [
                'wr_cross_7_21', 'regime_confidence', 'pattern_count',
                'price_volume_correlation'
            ]
            
            available_features = [col for col in feature_columns + optional_features if col in data.columns]
            
            if len(available_features) < 5:
                results['ml_signal_confidence'] = 0.0
                results['ml_signal_strength'] = base_signals['signal_strength']
                return results
            
            X = data[available_features].fillna(0)
            X_scaled = self.scaler.transform(X)
            
            # Get ML predictions
            ml_signal_probs = self.signal_classifier.predict_proba(X_scaled)
            ml_strength = self.strength_predictor.predict(X_scaled)
            
            # Extract probabilities for each class
            classes = self.signal_classifier.classes_
            prob_df = pd.DataFrame(ml_signal_probs, columns=classes, index=data.index)
            
            # Calculate ML confidence
            max_prob = prob_df.max(axis=1)
            results['ml_signal_confidence'] = max_prob
            
            # Get ML signal type
            ml_signal_type = pd.Series(classes[ml_signal_probs.argmax(axis=1)], index=data.index)
            results['ml_signal_type'] = ml_signal_type
            
            # Combine ML strength with base strength
            results['ml_signal_strength'] = np.clip(
                base_signals['signal_strength'] * 0.6 + 
                np.clip(ml_strength, 0, 1) * 0.4, 0, 1
            )
            
            # Update confidence based on agreement between rule-based and ML
            signal_agreement = (base_signals['signal_type'] == ml_signal_type).astype(float)
            results['combined_confidence'] = (
                base_signals['signal_confidence'] * 0.4 +
                max_prob * 0.4 +
                signal_agreement * 0.2
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error applying ML signals: {str(e)}")
            return pd.DataFrame(index=data.index)
    
    def _calculate_timeframe_confluence(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate multi-timeframe signal confluence"""
        try:
            results = pd.DataFrame(index=data.index)
            
            # Simulate multi-timeframe analysis by using different periods
            timeframe_periods = {'1H': 24, '4H': 96, '1D': 240}
            confluence_scores = []
            
            for tf, period in timeframe_periods.items():
                if len(data) < period:
                    continue
                
                # Resample to timeframe (simplified approach)
                tf_wr = data['wr_primary'].rolling(window=period//4).mean()
                tf_momentum = data['wr_momentum'].rolling(window=period//4).mean()
                
                # Calculate timeframe signal
                tf_signal = np.where(
                    (tf_wr < -70) & (tf_momentum > 0), 1,  # Bullish
                    np.where((tf_wr > -30) & (tf_momentum < 0), -1, 0)  # Bearish
                )
                
                confluence_scores.append(pd.Series(tf_signal, index=data.index, name=f'tf_{tf}_signal'))
            
            if confluence_scores:
                confluence_df = pd.concat(confluence_scores, axis=1)
                
                # Calculate weighted confluence
                weights = self.timeframe_weights[:len(confluence_scores)]
                weights = np.array(weights) / sum(weights)  # Normalize weights
                
                results['timeframe_confluence'] = (confluence_df * weights).sum(axis=1)
                results['timeframe_agreement'] = (confluence_df != 0).sum(axis=1) / len(confluence_scores)
                
                # Individual timeframe signals
                for col in confluence_df.columns:
                    results[col] = confluence_df[col]
            else:
                results['timeframe_confluence'] = 0.0
                results['timeframe_agreement'] = 0.0
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error calculating timeframe confluence: {str(e)}")
            raise
    
    def _calculate_risk_metrics(self, price_data: pd.DataFrame, signal_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate risk metrics for signals"""
        try:
            results = pd.DataFrame(index=price_data.index)
            
            # Volatility-based risk
            returns = price_data['close'].pct_change()
            volatility = returns.rolling(window=self.volatility_window).std() * np.sqrt(252)
            vol_percentile = volatility.rolling(window=100).rank(pct=True)
            
            # Base risk score from volatility
            results['volatility_risk'] = vol_percentile
            
            # Williams %R position risk
            wr_extreme = abs(signal_data['wr_primary']) / 100
            results['position_risk'] = 1 - wr_extreme  # Higher risk when not at extremes
            
            # Momentum risk
            momentum_consistency = (
                signal_data['wr_momentum'].rolling(window=5).std() / 
                (abs(signal_data['wr_momentum'].rolling(window=5).mean()) + 1e-6)
            )
            results['momentum_risk'] = np.clip(momentum_consistency, 0, 1)
            
            # Pattern reliability risk
            if 'pattern_strength' in signal_data.columns:
                pattern_risk = 1 / (1 + signal_data['pattern_strength'])
                results['pattern_risk'] = pattern_risk
            else:
                results['pattern_risk'] = 0.5
            
            # Regime risk
            if 'regime_confidence' in signal_data.columns:
                results['regime_risk'] = 1 - signal_data['regime_confidence']
            else:
                results['regime_risk'] = 0.5
            
            # Combined risk score
            risk_components = ['volatility_risk', 'position_risk', 'momentum_risk', 'pattern_risk', 'regime_risk']
            results['combined_risk'] = results[risk_components].mean(axis=1)
            
            # Risk-adjusted signal strength
            if 'signal_strength' in signal_data.columns:
                results['risk_adjusted_strength'] = (
                    signal_data['signal_strength'] * (1 - results['combined_risk'])
                )
            
            # Stop loss and take profit levels
            atr = self._calculate_atr(price_data, period=14)
            current_price = price_data['close']
            
            # Dynamic stop loss based on ATR and volatility
            stop_multiplier = 1.5 + vol_percentile  # Wider stops in volatile markets
            results['stop_loss_distance'] = atr * stop_multiplier
            
            # Take profit levels
            tp_multiplier = 2.0 + vol_percentile * 0.5
            results['take_profit_distance'] = atr * tp_multiplier
            
            # Calculate actual levels based on signal direction
            signal_direction = np.where(
                signal_data['signal_type'].isin([WRSignalType.BUY.value, WRSignalType.STRONG_BUY.value, WRSignalType.WEAK_BUY.value]), 1,
                np.where(signal_data['signal_type'].isin([WRSignalType.SELL.value, WRSignalType.STRONG_SELL.value, WRSignalType.WEAK_SELL.value]), -1, 0)
            )
            
            results['stop_loss_level'] = np.where(
                signal_direction > 0,
                current_price - results['stop_loss_distance'],
                np.where(signal_direction < 0, current_price + results['stop_loss_distance'], np.nan)
            )
            
            results['take_profit_level'] = np.where(
                signal_direction > 0,
                current_price + results['take_profit_distance'],
                np.where(signal_direction < 0, current_price - results['take_profit_distance'], np.nan)
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {str(e)}")
            raise
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        try:
            high_low = data['high'] - data['low']
            high_close = abs(data['high'] - data['close'].shift(1))
            low_close = abs(data['low'] - data['close'].shift(1))
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean()
            
            return atr
            
        except Exception as e:
            self.logger.error(f"Error calculating ATR: {str(e)}")
            return pd.Series(index=data.index)
    
    def _generate_final_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate final signal recommendations with all analysis combined"""
        try:
            results = pd.DataFrame(index=data.index)
            
            # Combine all signal components
            base_strength = data.get('signal_strength', 0)
            ml_strength = data.get('ml_signal_strength', base_strength)
            risk_adjusted = data.get('risk_adjusted_strength', ml_strength)
            
            # Apply confluence weighting
            confluence_weight = abs(data.get('timeframe_confluence', 0))
            agreement_weight = data.get('timeframe_agreement', 0.5)
            
            # Final signal strength
            results['final_signal_strength'] = np.clip(
                risk_adjusted * (0.7 + confluence_weight * 0.2 + agreement_weight * 0.1), 0, 1
            )
            
            # Final confidence score
            base_confidence = data.get('signal_confidence', 0.5)
            ml_confidence = data.get('ml_signal_confidence', base_confidence)
            combined_confidence = data.get('combined_confidence', ml_confidence)
            
            results['final_confidence'] = np.clip(
                combined_confidence * (0.8 + agreement_weight * 0.2), 0, 1
            )
            
            # Quality score
            pattern_boost = 1 + data.get('pattern_strength', 0) * 0.1
            regime_boost = 1 + data.get('regime_confidence', 0.5) * 0.1
            
            results['signal_quality'] = np.clip(
                results['final_signal_strength'] * pattern_boost * regime_boost, 0, 1
            )
            
            # Final signal classification
            strength_threshold = 0.6
            confidence_threshold = self.confidence_threshold
            
            strong_signal = (results['final_signal_strength'] > strength_threshold) & \
                          (results['final_confidence'] > confidence_threshold)
            
            medium_signal = (results['final_signal_strength'] > 0.4) & \
                          (results['final_confidence'] > 0.5)
            
            # Apply final signal types
            final_signal_type = data.get('signal_type', WRSignalType.NEUTRAL.value).copy()
            
            # Upgrade strong signals
            buy_signals = final_signal_type.isin([WRSignalType.BUY.value, WRSignalType.WEAK_BUY.value])
            sell_signals = final_signal_type.isin([WRSignalType.SELL.value, WRSignalType.WEAK_SELL.value])
            
            final_signal_type.loc[buy_signals & strong_signal] = WRSignalType.STRONG_BUY.value
            final_signal_type.loc[sell_signals & strong_signal] = WRSignalType.STRONG_SELL.value
            
            # Downgrade weak signals
            weak_confidence = results['final_confidence'] < 0.4
            final_signal_type.loc[buy_signals & weak_confidence] = WRSignalType.WEAK_BUY.value
            final_signal_type.loc[sell_signals & weak_confidence] = WRSignalType.WEAK_SELL.value
            
            # Filter out very low quality signals
            low_quality = results['signal_quality'] < 0.3
            final_signal_type.loc[low_quality] = WRSignalType.NEUTRAL.value
            
            results['final_signal_type'] = final_signal_type
            
            # Trading recommendations
            results['recommended_entry'] = data['close']
            results['recommended_stop_loss'] = data.get('stop_loss_level', np.nan)
            results['recommended_take_profit'] = data.get('take_profit_level', np.nan)
            
            # Position sizing recommendation based on confidence and risk
            risk_score = data.get('combined_risk', 0.5)
            confidence_factor = results['final_confidence']
            
            # Conservative position sizing: higher confidence and lower risk = larger position
            results['position_size_factor'] = np.clip(
                confidence_factor * (1 - risk_score) * results['signal_quality'], 0.1, 1.0
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error generating final signals: {str(e)}")
            raise
    
    def get_signal_summary(self, data: pd.DataFrame) -> WRSignalResult:
        """Get the latest signal summary"""
        try:
            if len(data) == 0:
                raise ValueError("No data available for signal summary")
            
            latest = data.iloc[-1]
            
            # Extract signal information
            signal_type_str = latest.get('final_signal_type', WRSignalType.NEUTRAL.value)
            signal_type = WRSignalType(signal_type_str)
            
            # Extract regime
            regime_str = latest.get('regime_primary', MarketRegime.RANGING.value)
            try:
                regime = MarketRegime(regime_str.upper())
            except ValueError:
                regime = MarketRegime.RANGING
            
            # Extract pattern signals
            pattern_signals = []
            if latest.get('bullish_divergence', False):
                pattern_signals.append('Bullish Divergence')
            if latest.get('bearish_divergence', False):
                pattern_signals.append('Bearish Divergence')
            if latest.get('wr_double_top', False):
                pattern_signals.append('Double Top')
            if latest.get('wr_double_bottom', False):
                pattern_signals.append('Double Bottom')
            
            # Create timeframe confluence dictionary
            timeframe_confluence = {}
            for col in data.columns:
                if col.startswith('tf_') and col.endswith('_signal'):
                    tf_name = col.replace('tf_', '').replace('_signal', '')
                    timeframe_confluence[tf_name] = latest.get(col, 0.0)
            
            return WRSignalResult(
                signal_type=signal_type,
                signal_strength=latest.get('final_signal_strength', 0.0),
                confidence_score=latest.get('final_confidence', 0.0),
                risk_score=latest.get('combined_risk', 0.5),
                wr_value=latest.get('wr_primary', 0.0),
                wr_smoothed=latest.get('wr_smoothed', 0.0),
                signal_quality=latest.get('signal_quality', 0.0),
                divergence_detected=latest.get('bullish_divergence', False) or latest.get('bearish_divergence', False),
                regime=regime,
                entry_price=latest.get('recommended_entry'),
                stop_loss=latest.get('recommended_stop_loss'),
                take_profit=latest.get('recommended_take_profit'),
                timeframe_confluence=timeframe_confluence,
                pattern_signals=pattern_signals
            )
            
        except Exception as e:
            self.logger.error(f"Error creating signal summary: {str(e)}")
            raise
    
    def get_required_columns(self) -> List[str]:
        """Get the required data columns for this indicator"""
        return ['open', 'high', 'low', 'close', 'volume']
    
    def get_configuration(self) -> Dict[str, Any]:
        """Get current indicator configuration"""
        return {
            'primary_period': self.primary_period,
            'secondary_periods': self.secondary_periods,
            'smoothing_period': self.smoothing_period,
            'signal_threshold': self.signal_threshold,
            'lookback_window': self.lookback_window,
            'min_samples': self.min_samples,
            'confidence_threshold': self.confidence_threshold,
            'risk_threshold': self.risk_threshold,
            'timeframes': self.timeframes,
            'timeframe_weights': self.timeframe_weights,
            'adaptive_thresholds': self.adaptive_thresholds,
            'pattern_detection': self.pattern_detection,
            'risk_adjustment': self.risk_adjustment
        }
