"""
Relative Vigor Index (RVI) Indicator - Advanced Implementation

This indicator implements an advanced version of the Relative Vigor Index
with machine learning enhancements and adaptive parameters for superior
momentum analysis and market timing.

Author: Humanitarian Trading Platform
License: MIT
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import logging
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import FastICA
from sklearn.neural_network import MLPRegressor
import talib
import warnings
warnings.filterwarnings('ignore')

from ..base.standard_indicator import StandardIndicatorInterface

logger = logging.getLogger(__name__)


@dataclass
class RVIConfig:
    """Configuration for Relative Vigor Index Indicator"""
    period: int = 14
    signal_period: int = 4
    smooth_period: int = 3
    volatility_period: int = 20
    trend_period: int = 20
    ml_lookback: int = 60
    adaptive_scaling: bool = True
    divergence_periods: int = 20
    overbought_threshold: float = 0.8
    oversold_threshold: float = -0.8
    confidence_threshold: float = 0.65
    noise_filter_enabled: bool = True


class RelativeVigorIndexIndicator(StandardIndicatorInterface):
    """
    Advanced Relative Vigor Index (RVI) Indicator Implementation
    
    The RVI compares the closing price to the trading range, similar to Stochastic,
    but uses the relationship between closing and opening prices to gauge momentum.
    Enhanced with ML for regime detection and adaptive thresholds.
    """
    
    def __init__(self, config: Optional[RVIConfig] = None):
        """Initialize Relative Vigor Index indicator"""
        self.config = config or RVIConfig()
        self.scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        self.ml_model = ExtraTreesRegressor(n_estimators=150, random_state=42)
        self.regime_model = MLPRegressor(hidden_layer_sizes=(50, 30), random_state=42)
        self.clustering_model = DBSCAN(eps=0.3, min_samples=5)
        self.ica = FastICA(n_components=4, random_state=42)
        self.is_fitted = False
        self.regime_fitted = False
        
    def calculate(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Calculate Relative Vigor Index with advanced ML integration
        
        Args:
            data: DataFrame with OHLCV data
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing RVI signals and metadata
        """
        try:
            if len(data) < self.config.period:
                raise ValueError("Insufficient data for RVI calculation")
                
            # Extract OHLCV data
            open_prices = data['open'].values
            high = data['high'].values
            low = data['low'].values
            close = data['close'].values
            volume = data['volume'].values if 'volume' in data else np.ones(len(close))
            
            # Calculate core RVI components
            rvi_components = self._calculate_rvi_components(open_prices, high, low, close)
            
            # Calculate traditional RVI
            traditional_rvi = self._calculate_traditional_rvi(rvi_components)
            
            # Calculate enhanced RVI with adaptive parameters
            enhanced_rvi = self._calculate_enhanced_rvi(rvi_components, close, volume)
            
            # Market regime detection
            regime_analysis = self._detect_market_regimes(close, high, low, enhanced_rvi)
            
            # Volatility-based adaptive thresholds
            adaptive_thresholds = self._calculate_adaptive_thresholds(close, enhanced_rvi)
            
            # Divergence analysis
            divergence_signals = self._analyze_divergences(close, enhanced_rvi)
            
            # Volume-weighted RVI
            volume_weighted_rvi = self._calculate_volume_weighted_rvi(rvi_components, volume)
            
            # Create feature matrix for ML
            features = self._create_feature_matrix(
                traditional_rvi, enhanced_rvi, volume_weighted_rvi,
                regime_analysis, adaptive_thresholds, close, volume
            )
            
            # Generate ML-enhanced signals
            ml_signals = self._generate_ml_signals(features, enhanced_rvi)
            
            # Multi-timeframe analysis
            mtf_analysis = self._multi_timeframe_analysis(close, enhanced_rvi)
            
            # Generate comprehensive trading signals
            trading_signals = self._generate_trading_signals(
                enhanced_rvi, regime_analysis, adaptive_thresholds,
                divergence_signals, ml_signals, mtf_analysis
            )
            
            # Calculate confidence scores
            confidence_scores = self._calculate_confidence_scores(
                trading_signals, features, enhanced_rvi, regime_analysis
            )
            
            # Risk-adjusted signals
            risk_adjusted_signals = self._apply_risk_adjustments(
                trading_signals, confidence_scores, regime_analysis
            )
            
            # Performance analytics
            performance_metrics = self._calculate_performance_metrics(
                close, enhanced_rvi, risk_adjusted_signals
            )
            
            # Prepare metadata
            metadata = self._prepare_metadata(
                traditional_rvi, enhanced_rvi, regime_analysis,
                performance_metrics, confidence_scores
            )
            
            return {
                'signal': risk_adjusted_signals['composite_signal'].iloc[-1] if len(risk_adjusted_signals) > 0 else 0.0,
                'rvi': enhanced_rvi['rvi'].iloc[-1] if len(enhanced_rvi) > 0 else 0.0,
                'rvi_signal': enhanced_rvi['rvi_signal'].iloc[-1] if len(enhanced_rvi) > 0 else 0.0,
                'traditional_rvi': traditional_rvi,
                'enhanced_rvi': enhanced_rvi,
                'volume_weighted_rvi': volume_weighted_rvi,
                'regime_analysis': regime_analysis,
                'adaptive_thresholds': adaptive_thresholds,
                'divergence_signals': divergence_signals,
                'ml_signals': ml_signals,
                'trading_signals': trading_signals,
                'risk_adjusted_signals': risk_adjusted_signals,
                'confidence': confidence_scores.iloc[-1] if len(confidence_scores) > 0 else 0.5,
                'performance_metrics': performance_metrics,
                'mtf_analysis': mtf_analysis,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Error calculating RVI: {str(e)}")
            return self._get_error_result(str(e))
    
    def _calculate_rvi_components(self, open_prices: np.ndarray, high: np.ndarray,
                                 low: np.ndarray, close: np.ndarray) -> pd.DataFrame:
        """Calculate core RVI components"""
        try:
            # Numerator: Closing price relative to opening price
            numerator = close - open_prices
            
            # Denominator: High-Low range
            denominator = high - low
            denominator = np.where(denominator == 0, 0.0001, denominator)  # Avoid division by zero
            
            # Basic ratio
            basic_ratio = numerator / denominator
            
            # Smoothed numerator and denominator for traditional RVI
            smooth_num = pd.Series(numerator).rolling(window=self.config.period).mean()
            smooth_den = pd.Series(denominator).rolling(window=self.config.period).mean()
            smooth_den = smooth_den.replace(0, 0.0001)
            
            # Additional components for enhanced analysis
            range_ratio = (high - low) / close  # Volatility indicator
            body_ratio = abs(close - open_prices) / (high - low)  # Body to range ratio
            
            # Momentum components
            price_momentum = pd.Series(close).diff(periods=self.config.trend_period)
            range_momentum = pd.Series(high - low).diff(periods=self.config.trend_period)
            
            return pd.DataFrame({
                'numerator': numerator,
                'denominator': denominator,
                'basic_ratio': basic_ratio,
                'smooth_numerator': smooth_num,
                'smooth_denominator': smooth_den,
                'range_ratio': range_ratio,
                'body_ratio': body_ratio,
                'price_momentum': price_momentum,
                'range_momentum': range_momentum
            })
            
        except Exception as e:
            logger.error(f"Error calculating RVI components: {str(e)}")
            return pd.DataFrame({'numerator': [0], 'denominator': [1], 'basic_ratio': [0]})
    
    def _calculate_traditional_rvi(self, rvi_components: pd.DataFrame) -> pd.DataFrame:
        """Calculate traditional RVI and signal line"""
        try:
            # Traditional RVI calculation
            rvi = rvi_components['smooth_numerator'] / rvi_components['smooth_denominator']
            
            # RVI signal line (moving average of RVI)
            rvi_signal = rvi.rolling(window=self.config.signal_period).mean()
            
            # RVI histogram (difference between RVI and signal)
            rvi_histogram = rvi - rvi_signal
            
            # RVI momentum
            rvi_momentum = rvi.diff(periods=self.config.smooth_period)
            
            # Normalize RVI to [-1, 1] range
            rvi_normalized = np.tanh(rvi * 2)
            
            return pd.DataFrame({
                'rvi': rvi,
                'rvi_signal': rvi_signal,
                'rvi_histogram': rvi_histogram,
                'rvi_momentum': rvi_momentum,
                'rvi_normalized': rvi_normalized
            })
            
        except Exception as e:
            logger.error(f"Error calculating traditional RVI: {str(e)}")
            return pd.DataFrame({'rvi': [0], 'rvi_signal': [0], 'rvi_histogram': [0]})
    
    def _calculate_enhanced_rvi(self, rvi_components: pd.DataFrame, close: np.ndarray, volume: np.ndarray) -> pd.DataFrame:
        """Calculate enhanced RVI with adaptive features"""
        try:
            # Traditional RVI
            traditional_rvi = self._calculate_traditional_rvi(rvi_components)
            
            # Adaptive smoothing based on volatility
            volatility = pd.Series(close).pct_change().rolling(window=self.config.volatility_period).std()
            
            if self.config.adaptive_scaling:
                # Dynamic smoothing factor
                smooth_factor = np.clip(3 + volatility * 100, 2, 10).astype(int)
                
                # Adaptive RVI
                adaptive_rvi = pd.Series(index=traditional_rvi.index)
                for i in range(len(traditional_rvi)):
                    if i >= smooth_factor.iloc[i]:
                        window = int(smooth_factor.iloc[i])
                        adaptive_rvi.iloc[i] = traditional_rvi['rvi'].iloc[max(0, i-window):i+1].mean()
                    else:
                        adaptive_rvi.iloc[i] = traditional_rvi['rvi'].iloc[i]
            else:
                adaptive_rvi = traditional_rvi['rvi']
            
            # Volume-adjusted RVI
            volume_factor = volume / pd.Series(volume).rolling(window=20).mean()
            volume_adjusted_rvi = traditional_rvi['rvi'] * np.tanh(volume_factor - 1)
            
            # Trend-adjusted RVI
            price_trend = pd.Series(close).rolling(window=self.config.trend_period).apply(
                lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1
            )
            trend_adjusted_rvi = traditional_rvi['rvi'] * price_trend
            
            # Noise filtering
            if self.config.noise_filter_enabled:
                # Apply Kalman-like filtering
                filtered_rvi = self._apply_noise_filter(adaptive_rvi)
            else:
                filtered_rvi = adaptive_rvi
            
            # Enhanced signal line
            enhanced_signal = filtered_rvi.rolling(window=self.config.signal_period).mean()
            
            # Multi-layer histogram
            histogram_1 = filtered_rvi - enhanced_signal
            histogram_2 = histogram_1.diff()
            
            return pd.DataFrame({
                'rvi': filtered_rvi,
                'rvi_signal': enhanced_signal,
                'rvi_histogram': histogram_1,
                'rvi_histogram_2': histogram_2,
                'adaptive_rvi': adaptive_rvi,
                'volume_adjusted_rvi': volume_adjusted_rvi,
                'trend_adjusted_rvi': trend_adjusted_rvi,
                'volatility_factor': volatility
            })
            
        except Exception as e:
            logger.error(f"Error calculating enhanced RVI: {str(e)}")
            return pd.DataFrame({'rvi': [0], 'rvi_signal': [0], 'rvi_histogram': [0]})
    
    def _apply_noise_filter(self, series: pd.Series) -> pd.Series:
        """Apply noise filtering to RVI series"""
        try:
            # Simple exponential smoothing with adaptive alpha
            filtered_series = series.copy()
            alpha = 0.3  # Base smoothing factor
            
            for i in range(1, len(series)):
                if not pd.isna(series.iloc[i]) and not pd.isna(filtered_series.iloc[i-1]):
                    # Adaptive alpha based on volatility
                    recent_vol = abs(series.iloc[i] - series.iloc[i-1])
                    adaptive_alpha = alpha * (1 + recent_vol)
                    adaptive_alpha = min(adaptive_alpha, 0.8)
                    
                    filtered_series.iloc[i] = (adaptive_alpha * series.iloc[i] + 
                                             (1 - adaptive_alpha) * filtered_series.iloc[i-1])
            
            return filtered_series
            
        except Exception as e:
            logger.error(f"Error applying noise filter: {str(e)}")
            return series
    
    def _detect_market_regimes(self, close: np.ndarray, high: np.ndarray, 
                              low: np.ndarray, enhanced_rvi: pd.DataFrame) -> pd.DataFrame:
        """Detect market regimes using price action and RVI"""
        try:
            # Volatility regime
            returns = pd.Series(close).pct_change()
            volatility = returns.rolling(window=20).std()
            vol_regime = pd.cut(volatility, bins=3, labels=['low_vol', 'normal_vol', 'high_vol'])
            
            # Trend regime
            ma_short = pd.Series(close).rolling(window=10).mean()
            ma_long = pd.Series(close).rolling(window=30).mean()
            trend_regime = np.where(ma_short > ma_long, 'uptrend', 'downtrend')
            
            # RVI regime
            rvi_mean = enhanced_rvi['rvi'].rolling(window=50).mean()
            rvi_std = enhanced_rvi['rvi'].rolling(window=50).std()
            
            rvi_regime = np.where(
                enhanced_rvi['rvi'] > rvi_mean + rvi_std, 'strong_bullish',
                np.where(
                    enhanced_rvi['rvi'] < rvi_mean - rvi_std, 'strong_bearish',
                    np.where(enhanced_rvi['rvi'] > 0, 'bullish', 'bearish')
                )
            )
            
            # Market cycle phase
            cycle_phase = self._determine_cycle_phase(close, enhanced_rvi['rvi'])
            
            # Momentum regime
            momentum = enhanced_rvi['rvi_histogram']
            momentum_regime = np.where(
                momentum > momentum.quantile(0.75), 'strong_momentum',
                np.where(
                    momentum < momentum.quantile(0.25), 'weak_momentum', 'normal_momentum'
                )
            )
            
            return pd.DataFrame({
                'volatility_regime': vol_regime,
                'trend_regime': trend_regime,
                'rvi_regime': rvi_regime,
                'cycle_phase': cycle_phase,
                'momentum_regime': momentum_regime
            })
            
        except Exception as e:
            logger.error(f"Error detecting market regimes: {str(e)}")
            return pd.DataFrame({'volatility_regime': ['normal_vol'], 'trend_regime': ['uptrend'], 'rvi_regime': ['bullish']})
    
    def _determine_cycle_phase(self, prices: np.ndarray, rvi: pd.Series) -> List[str]:
        """Determine market cycle phase"""
        try:
            # Price momentum
            price_momentum = pd.Series(prices).pct_change(periods=20)
            
            # RVI momentum
            rvi_momentum = rvi.diff(periods=10)
            
            cycle_phases = []
            for i in range(len(prices)):
                pm = price_momentum.iloc[i] if not pd.isna(price_momentum.iloc[i]) else 0
                rm = rvi_momentum.iloc[i] if not pd.isna(rvi_momentum.iloc[i]) else 0
                
                if pm > 0 and rm > 0:
                    cycle_phases.append('acceleration')
                elif pm > 0 and rm <= 0:
                    cycle_phases.append('deceleration')
                elif pm <= 0 and rm <= 0:
                    cycle_phases.append('decline')
                else:
                    cycle_phases.append('recovery')
            
            return cycle_phases
            
        except Exception as e:
            logger.error(f"Error determining cycle phase: {str(e)}")
            return ['neutral'] * len(prices)
    
    def _calculate_adaptive_thresholds(self, close: np.ndarray, enhanced_rvi: pd.DataFrame) -> pd.DataFrame:
        """Calculate adaptive overbought/oversold thresholds"""
        try:
            # Rolling statistics for adaptive thresholds
            rvi_mean = enhanced_rvi['rvi'].rolling(window=50).mean()
            rvi_std = enhanced_rvi['rvi'].rolling(window=50).std()
            
            # Volatility-adjusted thresholds
            volatility = pd.Series(close).pct_change().rolling(window=20).std()
            vol_multiplier = 1 + volatility * 5
            
            # Adaptive thresholds
            upper_threshold = rvi_mean + (rvi_std * vol_multiplier * 2)
            lower_threshold = rvi_mean - (rvi_std * vol_multiplier * 2)
            
            # Extreme thresholds
            extreme_upper = rvi_mean + (rvi_std * vol_multiplier * 3)
            extreme_lower = rvi_mean - (rvi_std * vol_multiplier * 3)
            
            # Dynamic zones
            overbought_zone = enhanced_rvi['rvi'] > upper_threshold
            oversold_zone = enhanced_rvi['rvi'] < lower_threshold
            
            return pd.DataFrame({
                'upper_threshold': upper_threshold,
                'lower_threshold': lower_threshold,
                'extreme_upper': extreme_upper,
                'extreme_lower': extreme_lower,
                'overbought_zone': overbought_zone.astype(int),
                'oversold_zone': oversold_zone.astype(int),
                'volatility_multiplier': vol_multiplier
            })
            
        except Exception as e:
            logger.error(f"Error calculating adaptive thresholds: {str(e)}")
            return pd.DataFrame({'upper_threshold': [0.8], 'lower_threshold': [-0.8], 'overbought_zone': [0], 'oversold_zone': [0]})
    
    def _analyze_divergences(self, close: np.ndarray, enhanced_rvi: pd.DataFrame) -> pd.DataFrame:
        """Analyze price-RVI divergences"""
        try:
            periods = self.config.divergence_periods
            
            # Price highs and lows
            price_highs = pd.Series(close).rolling(window=periods).max()
            price_lows = pd.Series(close).rolling(window=periods).min()
            
            # RVI highs and lows
            rvi_highs = enhanced_rvi['rvi'].rolling(window=periods).max()
            rvi_lows = enhanced_rvi['rvi'].rolling(window=periods).min()
            
            # Bullish divergence: price makes lower lows, RVI makes higher lows
            bullish_divergence = (
                (close < price_lows.shift(1)) & 
                (enhanced_rvi['rvi'] > rvi_lows.shift(1))
            )
            
            # Bearish divergence: price makes higher highs, RVI makes lower highs
            bearish_divergence = (
                (close > price_highs.shift(1)) & 
                (enhanced_rvi['rvi'] < rvi_highs.shift(1))
            )
            
            # Hidden divergences
            hidden_bullish = (
                (close > price_lows.shift(1)) & 
                (enhanced_rvi['rvi'] < rvi_lows.shift(1))
            )
            
            hidden_bearish = (
                (close < price_highs.shift(1)) & 
                (enhanced_rvi['rvi'] > rvi_highs.shift(1))
            )
            
            # Divergence strength
            price_slope = (close - close.shift(periods)) / periods
            rvi_slope = (enhanced_rvi['rvi'] - enhanced_rvi['rvi'].shift(periods)) / periods
            
            divergence_strength = abs(price_slope - rvi_slope)
            
            return pd.DataFrame({
                'bullish_divergence': bullish_divergence.astype(int),
                'bearish_divergence': bearish_divergence.astype(int),
                'hidden_bullish': hidden_bullish.astype(int),
                'hidden_bearish': hidden_bearish.astype(int),
                'divergence_strength': divergence_strength
            })
            
        except Exception as e:
            logger.error(f"Error analyzing divergences: {str(e)}")
            return pd.DataFrame({'bullish_divergence': [0], 'bearish_divergence': [0], 'divergence_strength': [0]})
    
    def _calculate_volume_weighted_rvi(self, rvi_components: pd.DataFrame, volume: np.ndarray) -> pd.DataFrame:
        """Calculate volume-weighted RVI"""
        try:
            # Volume-weighted numerator and denominator
            volume_weight = volume / pd.Series(volume).rolling(window=20).mean()
            
            weighted_numerator = rvi_components['numerator'] * volume_weight
            weighted_denominator = rvi_components['denominator'] * volume_weight
            
            # Volume-weighted RVI
            vw_numerator = pd.Series(weighted_numerator).rolling(window=self.config.period).mean()
            vw_denominator = pd.Series(weighted_denominator).rolling(window=self.config.period).mean()
            vw_denominator = vw_denominator.replace(0, 0.0001)
            
            volume_weighted_rvi = vw_numerator / vw_denominator
            
            # Volume-weighted signal
            vw_signal = volume_weighted_rvi.rolling(window=self.config.signal_period).mean()
            
            # Volume confirmation
            volume_confirmation = np.where(volume_weight > 1.2, 1, 0)
            
            return pd.DataFrame({
                'volume_weighted_rvi': volume_weighted_rvi,
                'vw_signal': vw_signal,
                'volume_weight': volume_weight,
                'volume_confirmation': volume_confirmation
            })
            
        except Exception as e:
            logger.error(f"Error calculating volume-weighted RVI: {str(e)}")
            return pd.DataFrame({'volume_weighted_rvi': [0], 'vw_signal': [0], 'volume_confirmation': [0]})
    
    def _create_feature_matrix(self, traditional_rvi: pd.DataFrame, enhanced_rvi: pd.DataFrame,
                              volume_weighted_rvi: pd.DataFrame, regime_analysis: pd.DataFrame,
                              adaptive_thresholds: pd.DataFrame, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Create feature matrix for ML models"""
        try:
            features = []
            
            # RVI features
            features.extend([
                enhanced_rvi['rvi'].values,
                enhanced_rvi['rvi_signal'].values,
                enhanced_rvi['rvi_histogram'].values,
                traditional_rvi['rvi_momentum'].values
            ])
            
            # Volume features
            features.extend([
                volume_weighted_rvi['volume_weighted_rvi'].values,
                volume_weighted_rvi['volume_weight'].values
            ])
            
            # Threshold features
            features.extend([
                adaptive_thresholds['overbought_zone'].values,
                adaptive_thresholds['oversold_zone'].values,
                adaptive_thresholds['volatility_multiplier'].values
            ])
            
            # Price features
            price_roc = pd.Series(close).pct_change(periods=5).values
            price_volatility = pd.Series(close).pct_change().rolling(window=10).std().values
            features.extend([price_roc, price_volatility])
            
            # Volume features
            volume_roc = pd.Series(volume).pct_change(periods=5).values
            features.append(volume_roc)
            
            return np.column_stack(features)
            
        except Exception as e:
            logger.error(f"Error creating feature matrix: {str(e)}")
            return np.zeros((len(enhanced_rvi), 12))
    
    def _generate_ml_signals(self, features: np.ndarray, enhanced_rvi: pd.DataFrame) -> pd.DataFrame:
        """Generate ML-enhanced signals"""
        try:
            if len(features) < self.config.ml_lookback:
                return pd.DataFrame({'ml_prediction': np.zeros(len(features)), 'ml_confidence': np.zeros(len(features))})
            
            # Prepare target variable (future RVI change)
            target = enhanced_rvi['rvi'].shift(-3) - enhanced_rvi['rvi']
            
            # Clean data
            features_clean = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
            target_clean = target.fillna(0).values
            
            # Ensure same length
            min_len = min(len(features_clean), len(target_clean))
            features_clean = features_clean[:min_len]
            target_clean = target_clean[:min_len]
            
            # Train model if enough data
            if not self.is_fitted and len(features_clean) >= self.config.ml_lookback:
                try:
                    # Scale features
                    features_scaled = self.scaler.fit_transform(features_clean)
                    
                    # Apply ICA for feature extraction
                    features_ica = self.ica.fit_transform(features_scaled)
                    
                    # Train model
                    split_point = int(len(features_ica) * 0.8)
                    X_train, y_train = features_ica[:split_point], target_clean[:split_point]
                    
                    if len(X_train) > 10:
                        self.ml_model.fit(X_train, y_train)
                        self.is_fitted = True
                        
                except Exception as e:
                    logger.warning(f"ML model training failed: {str(e)}")
                    self.is_fitted = False
            
            # Generate predictions
            if self.is_fitted:
                try:
                    features_scaled = self.scaler.transform(features_clean)
                    features_ica = self.ica.transform(features_scaled)
                    
                    predictions = self.ml_model.predict(features_ica)
                    
                    # Calculate confidence based on prediction consistency
                    prediction_std = np.std(predictions[-10:]) if len(predictions) > 10 else np.std(predictions)
                    confidence = 1 / (1 + prediction_std * 10) if prediction_std > 0 else 0.5
                    
                    return pd.DataFrame({
                        'ml_prediction': np.tanh(predictions),  # Normalize predictions
                        'ml_confidence': np.full(len(predictions), confidence)
                    })
                except Exception as e:
                    logger.warning(f"ML prediction failed: {str(e)}")
            
            # Fallback
            return pd.DataFrame({
                'ml_prediction': np.zeros(len(features)),
                'ml_confidence': np.full(len(features), 0.5)
            })
            
        except Exception as e:
            logger.error(f"Error generating ML signals: {str(e)}")
            return pd.DataFrame({'ml_prediction': [0], 'ml_confidence': [0.5]})
    
    def _multi_timeframe_analysis(self, close: np.ndarray, enhanced_rvi: pd.DataFrame) -> pd.DataFrame:
        """Perform multi-timeframe RVI analysis"""
        try:
            timeframes = [5, 10, 20, 50]
            mtf_signals = pd.DataFrame(index=enhanced_rvi.index)
            
            for tf in timeframes:
                # Calculate RVI for different timeframes
                tf_close = pd.Series(close).rolling(window=tf).mean()
                tf_rvi = enhanced_rvi['rvi'].rolling(window=tf).mean()
                
                # Generate signals for this timeframe
                tf_signal = np.where(tf_rvi > 0, 1, -1)
                
                mtf_signals[f'tf_{tf}_signal'] = tf_signal
                mtf_signals[f'tf_{tf}_rvi'] = tf_rvi
            
            # Consensus signal
            signal_columns = [col for col in mtf_signals.columns if 'signal' in col]
            consensus_signal = mtf_signals[signal_columns].mean(axis=1)
            
            mtf_signals['consensus_signal'] = consensus_signal
            mtf_signals['signal_agreement'] = (mtf_signals[signal_columns] > 0).sum(axis=1) / len(signal_columns)
            
            return mtf_signals
            
        except Exception as e:
            logger.error(f"Error in multi-timeframe analysis: {str(e)}")
            return pd.DataFrame({'consensus_signal': [0], 'signal_agreement': [0.5]})
    
    def _generate_trading_signals(self, enhanced_rvi: pd.DataFrame, regime_analysis: pd.DataFrame,
                                 adaptive_thresholds: pd.DataFrame, divergence_signals: pd.DataFrame,
                                 ml_signals: pd.DataFrame, mtf_analysis: pd.DataFrame) -> pd.DataFrame:
        """Generate comprehensive trading signals"""
        try:
            signals = pd.DataFrame(index=enhanced_rvi.index)
            
            # Base RVI signals
            rvi_cross_signal = np.where(
                enhanced_rvi['rvi'] > enhanced_rvi['rvi_signal'], 1, -1
            )
            
            # Threshold-based signals
            threshold_signal = np.where(
                enhanced_rvi['rvi'] > adaptive_thresholds['upper_threshold'], -1,  # Overbought
                np.where(enhanced_rvi['rvi'] < adaptive_thresholds['lower_threshold'], 1, 0)  # Oversold
            )
            
            # Divergence signals
            divergence_signal = (
                divergence_signals['bullish_divergence'] * 1 + 
                divergence_signals['bearish_divergence'] * -1
            )
            
            # ML enhancement
            ml_enhancement = ml_signals['ml_prediction'] * ml_signals['ml_confidence']
            
            # Multi-timeframe signal
            mtf_signal = mtf_analysis['consensus_signal']
            
            # Composite signal
            composite_signal = (
                rvi_cross_signal * 0.3 +
                threshold_signal * 0.2 +
                divergence_signal * 0.2 +
                ml_enhancement * 0.15 +
                mtf_signal * 0.15
            )
            
            # Generate buy/sell signals
            buy_signal = (composite_signal > 0.5) & (composite_signal.shift(1) <= 0.5)
            sell_signal = (composite_signal < -0.5) & (composite_signal.shift(1) >= -0.5)
            
            signals['composite_signal'] = composite_signal
            signals['rvi_cross_signal'] = rvi_cross_signal
            signals['threshold_signal'] = threshold_signal
            signals['divergence_signal'] = divergence_signal
            signals['buy_signal'] = buy_signal.astype(int)
            signals['sell_signal'] = sell_signal.astype(int)
            signals['signal_strength'] = np.abs(composite_signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating trading signals: {str(e)}")
            return pd.DataFrame({'composite_signal': [0], 'buy_signal': [0], 'sell_signal': [0]})
    
    def _calculate_confidence_scores(self, signals: pd.DataFrame, features: np.ndarray,
                                    enhanced_rvi: pd.DataFrame, regime_analysis: pd.DataFrame) -> pd.Series:
        """Calculate confidence scores for signals"""
        try:
            # Signal strength confidence
            signal_confidence = signals['signal_strength']
            
            # Feature consistency
            feature_std = np.std(features, axis=1) if len(features.shape) > 1 else np.array([0.5] * len(signals))
            consistency_confidence = 1 / (1 + feature_std * 2)
            
            # RVI momentum confidence
            rvi_momentum = enhanced_rvi['rvi_histogram'].abs()
            momentum_confidence = np.tanh(rvi_momentum)
            
            # Regime-based confidence adjustments
            regime_confidence = np.where(
                regime_analysis['volatility_regime'] == 'high_vol', 0.7,
                np.where(regime_analysis['volatility_regime'] == 'low_vol', 1.0, 0.85)
            )
            
            # Combined confidence
            combined_confidence = (
                signal_confidence * 0.3 +
                consistency_confidence * 0.25 +
                momentum_confidence * 0.25 +
                regime_confidence * 0.2
            )
            
            return pd.Series(combined_confidence, index=signals.index)
            
        except Exception as e:
            logger.error(f"Error calculating confidence scores: {str(e)}")
            return pd.Series([0.5] * len(signals), index=signals.index)
    
    def _apply_risk_adjustments(self, signals: pd.DataFrame, confidence_scores: pd.Series,
                               regime_analysis: pd.DataFrame) -> pd.DataFrame:
        """Apply risk-based adjustments to signals"""
        try:
            risk_adjusted = signals.copy()
            
            # Confidence threshold filtering
            high_confidence_mask = confidence_scores >= self.config.confidence_threshold
            
            # Regime-based risk adjustments
            high_vol_mask = regime_analysis['volatility_regime'] == 'high_vol'
            
            # Adjust signals based on confidence and regime
            risk_adjusted['buy_signal'] = (
                signals['buy_signal'] & high_confidence_mask & ~high_vol_mask
            ).astype(int)
            
            risk_adjusted['sell_signal'] = (
                signals['sell_signal'] & high_confidence_mask & ~high_vol_mask
            ).astype(int)
            
            # Scale composite signal by confidence
            risk_adjusted['composite_signal'] = signals['composite_signal'] * confidence_scores
            
            # Risk score
            risk_score = np.where(high_vol_mask, 0.8, 0.2) + (1 - confidence_scores)
            risk_adjusted['risk_score'] = risk_score
            
            return risk_adjusted
            
        except Exception as e:
            logger.error(f"Error applying risk adjustments: {str(e)}")
            return signals
    
    def _calculate_performance_metrics(self, close: np.ndarray, enhanced_rvi: pd.DataFrame,
                                     signals: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance metrics"""
        try:
            # Price returns
            price_returns = pd.Series(close).pct_change()
            
            # Signal-based returns
            signal_returns = price_returns * signals['composite_signal'].shift(1)
            
            # Performance metrics
            total_return = (1 + signal_returns.fillna(0)).prod() - 1
            volatility = signal_returns.std() * np.sqrt(252)
            sharpe_ratio = signal_returns.mean() / signal_returns.std() * np.sqrt(252) if signal_returns.std() > 0 else 0
            
            # Maximum drawdown
            cumulative_returns = (1 + signal_returns.fillna(0)).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            # RVI-specific metrics
            rvi_range = enhanced_rvi['rvi'].max() - enhanced_rvi['rvi'].min()
            rvi_efficiency = abs(enhanced_rvi['rvi'].iloc[-1] - enhanced_rvi['rvi'].iloc[0]) / rvi_range if rvi_range > 0 else 0
            
            return {
                'total_return': float(total_return),
                'volatility': float(volatility),
                'sharpe_ratio': float(sharpe_ratio),
                'max_drawdown': float(max_drawdown),
                'rvi_range': float(rvi_range),
                'rvi_efficiency': float(rvi_efficiency),
                'avg_rvi': float(enhanced_rvi['rvi'].mean()),
                'rvi_volatility': float(enhanced_rvi['rvi'].std())
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            return {'total_return': 0.0, 'sharpe_ratio': 0.0, 'max_drawdown': 0.0}
    
    def _prepare_metadata(self, traditional_rvi: pd.DataFrame, enhanced_rvi: pd.DataFrame,
                         regime_analysis: pd.DataFrame, performance_metrics: Dict[str, float],
                         confidence_scores: pd.Series) -> Dict[str, Any]:
        """Prepare comprehensive metadata"""
        try:
            return {
                'indicator_name': 'Relative Vigor Index',
                'version': '1.0.0',
                'data_points': len(enhanced_rvi),
                'config': {
                    'period': self.config.period,
                    'signal_period': self.config.signal_period,
                    'adaptive_scaling': self.config.adaptive_scaling,
                    'noise_filter_enabled': self.config.noise_filter_enabled
                },
                'current_values': {
                    'rvi': float(enhanced_rvi['rvi'].iloc[-1]) if len(enhanced_rvi) > 0 else 0.0,
                    'rvi_signal': float(enhanced_rvi['rvi_signal'].iloc[-1]) if len(enhanced_rvi) > 0 else 0.0,
                    'rvi_histogram': float(enhanced_rvi['rvi_histogram'].iloc[-1]) if len(enhanced_rvi) > 0 else 0.0,
                    'confidence': float(confidence_scores.iloc[-1]) if len(confidence_scores) > 0 else 0.5
                },
                'statistics': {
                    'avg_rvi': float(enhanced_rvi['rvi'].mean()) if len(enhanced_rvi) > 0 else 0.0,
                    'rvi_std': float(enhanced_rvi['rvi'].std()) if len(enhanced_rvi) > 0 else 0.0,
                    'max_rvi': float(enhanced_rvi['rvi'].max()) if len(enhanced_rvi) > 0 else 0.0,
                    'min_rvi': float(enhanced_rvi['rvi'].min()) if len(enhanced_rvi) > 0 else 0.0,
                    'avg_confidence': float(confidence_scores.mean()) if len(confidence_scores) > 0 else 0.5
                },
                'performance_metrics': performance_metrics,
                'regime_info': {
                    'current_volatility_regime': regime_analysis['volatility_regime'].iloc[-1] if len(regime_analysis) > 0 else 'normal_vol',
                    'current_trend_regime': regime_analysis['trend_regime'].iloc[-1] if len(regime_analysis) > 0 else 'uptrend',
                    'current_rvi_regime': regime_analysis['rvi_regime'].iloc[-1] if len(regime_analysis) > 0 else 'bullish'
                },
                'ml_info': {
                    'model_fitted': self.is_fitted,
                    'regime_model_fitted': self.regime_fitted
                }
            }
            
        except Exception as e:
            logger.error(f"Error preparing metadata: {str(e)}")
            return {'indicator_name': 'Relative Vigor Index', 'error': str(e)}
    
    def _get_error_result(self, error_message: str) -> Dict[str, Any]:
        """Return error result structure"""
        return {
            'signal': 0.0,
            'rvi': 0.0,
            'rvi_signal': 0.0,
            'traditional_rvi': pd.DataFrame(),
            'enhanced_rvi': pd.DataFrame(),
            'volume_weighted_rvi': pd.DataFrame(),
            'regime_analysis': pd.DataFrame(),
            'adaptive_thresholds': pd.DataFrame(),
            'divergence_signals': pd.DataFrame(),
            'ml_signals': pd.DataFrame(),
            'trading_signals': pd.DataFrame(),
            'risk_adjusted_signals': pd.DataFrame(),
            'confidence': 0.5,
            'performance_metrics': {},
            'mtf_analysis': pd.DataFrame(),
            'metadata': {
                'indicator_name': 'Relative Vigor Index',
                'error': error_message,
                'status': 'error'
            }
        }