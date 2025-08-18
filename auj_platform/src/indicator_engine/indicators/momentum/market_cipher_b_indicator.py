"""
Market Cipher B Indicator - Advanced Implementation

This indicator implements the Market Cipher B system, which combines multiple momentum
indicators with machine learning integration for sophisticated market analysis.
Based on VuManChu's Market Cipher methodology with advanced enhancements.

Author: Humanitarian Trading Platform
License: MIT
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import logging
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import talib
import warnings
warnings.filterwarnings('ignore')

from ..base.standard_indicator import StandardIndicatorInterface

logger = logging.getLogger(__name__)


@dataclass
class MarketCipherConfig:
    """Configuration for Market Cipher B Indicator"""
    wt_channel_length: int = 9
    wt_average_length: int = 12
    rsi_length: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    mfi_length: int = 14
    stoch_length: int = 14
    stoch_smooth_k: int = 3
    stoch_smooth_d: int = 3
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    volume_threshold: float = 1.5
    ml_lookback: int = 50
    confidence_threshold: float = 0.6
    divergence_periods: int = 20


class MarketCipherB(StandardIndicatorInterface):
    """
    Advanced Market Cipher B Indicator Implementation
    
    Combines Wave Trend, RSI, MFI, Stochastic, MACD, and volume analysis
    with machine learning for comprehensive momentum assessment.
    """
    
    def __init__(self, config: Optional[MarketCipherConfig] = None):
        """Initialize Market Cipher B indicator"""
        self.config = config or MarketCipherConfig()
        self.scaler = StandardScaler()
        self.ml_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.pca = PCA(n_components=5)
        self.is_fitted = False
        self.feature_cache = {}
        
    def calculate(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Calculate Market Cipher B signals with advanced ML integration
        
        Args:
            data: DataFrame with OHLCV data
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing Market Cipher B signals and metadata
        """
        try:
            if len(data) < max(self.config.wt_channel_length, self.config.ml_lookback):
                raise ValueError("Insufficient data for Market Cipher B calculation")
                
            # Extract OHLCV data
            high = data['high'].values
            low = data['low'].values
            close = data['close'].values
            volume = data['volume'].values
            
            # Calculate core components
            wave_trend = self._calculate_wave_trend(high, low, close)
            rsi_signals = self._calculate_rsi_signals(close)
            mfi_signals = self._calculate_mfi_signals(high, low, close, volume)
            stoch_signals = self._calculate_stochastic_signals(high, low, close)
            macd_signals = self._calculate_macd_signals(close)
            volume_signals = self._calculate_volume_signals(volume, close)
            
            # Detect divergences
            divergences = self._detect_divergences(close, wave_trend, rsi_signals)
            
            # Create feature matrix for ML
            features = self._create_feature_matrix(
                wave_trend, rsi_signals, mfi_signals, 
                stoch_signals, macd_signals, volume_signals
            )
            
            # Generate ML-enhanced signals
            ml_signals = self._generate_ml_signals(features, close)
            
            # Combine all signals into composite Market Cipher B
            cipher_signals = self._generate_cipher_signals(
                wave_trend, rsi_signals, mfi_signals, stoch_signals,
                macd_signals, volume_signals, divergences, ml_signals
            )
            
            # Calculate confidence scores
            confidence_scores = self._calculate_confidence_scores(cipher_signals, features)
            
            # Generate final signals with risk assessment
            final_signals = self._generate_final_signals(cipher_signals, confidence_scores)
            
            # Prepare metadata
            metadata = self._prepare_metadata(
                wave_trend, rsi_signals, mfi_signals, stoch_signals,
                macd_signals, volume_signals, divergences, ml_signals,
                confidence_scores
            )
            
            return {
                'signal': final_signals['composite_signal'].iloc[-1] if len(final_signals) > 0 else 0.0,
                'signals': final_signals.to_dict('records'),
                'wave_trend': wave_trend,
                'rsi_signals': rsi_signals,
                'mfi_signals': mfi_signals,
                'stoch_signals': stoch_signals,
                'macd_signals': macd_signals,
                'volume_signals': volume_signals,
                'divergences': divergences,
                'ml_predictions': ml_signals,
                'confidence': confidence_scores.iloc[-1] if len(confidence_scores) > 0 else 0.5,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Error calculating Market Cipher B: {str(e)}")
            return self._get_error_result(str(e))
    
    def _calculate_wave_trend(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> pd.DataFrame:
        """Calculate Wave Trend oscillator"""
        try:
            # Calculate average price
            ap = (high + low + close) / 3
            
            # Exponential moving averages
            esa = pd.Series(ap).ewm(span=self.config.wt_channel_length).mean()
            d = pd.Series(np.abs(ap - esa)).ewm(span=self.config.wt_channel_length).mean()
            
            # Calculate CI (Commodity Index)
            ci = (ap - esa) / (0.015 * d)
            
            # Wave Trend 1 and 2
            wt1 = pd.Series(ci).ewm(span=self.config.wt_average_length).mean()
            wt2 = pd.Series(wt1).rolling(window=4).mean()
            
            # Calculate wave trend momentum
            wt_momentum = wt1 - wt2
            
            # Normalize wave trend values
            wt1_norm = self._normalize_series(wt1, window=50)
            wt2_norm = self._normalize_series(wt2, window=50)
            
            return pd.DataFrame({
                'wt1': wt1_norm,
                'wt2': wt2_norm,
                'wt_momentum': wt_momentum,
                'wt_cross': np.where(wt1 > wt2, 1, -1)
            })
            
        except Exception as e:
            logger.error(f"Error calculating wave trend: {str(e)}")
            return pd.DataFrame({'wt1': [0], 'wt2': [0], 'wt_momentum': [0], 'wt_cross': [0]})
    
    def _calculate_rsi_signals(self, close: np.ndarray) -> pd.DataFrame:
        """Calculate advanced RSI signals"""
        try:
            # Standard RSI
            rsi = talib.RSI(close, timeperiod=self.config.rsi_length)
            
            # RSI smoothed
            rsi_smooth = pd.Series(rsi).rolling(window=3).mean()
            
            # RSI momentum
            rsi_momentum = np.diff(rsi, prepend=rsi[0])
            
            # RSI divergence detection
            rsi_highs = pd.Series(rsi).rolling(window=10).max()
            rsi_lows = pd.Series(rsi).rolling(window=10).min()
            
            # RSI zones
            rsi_overbought = rsi > self.config.rsi_overbought
            rsi_oversold = rsi < self.config.rsi_oversold
            
            # Dynamic RSI thresholds using Bollinger Bands
            rsi_mean = pd.Series(rsi).rolling(window=20).mean()
            rsi_std = pd.Series(rsi).rolling(window=20).std()
            rsi_upper = rsi_mean + 1.5 * rsi_std
            rsi_lower = rsi_mean - 1.5 * rsi_std
            
            return pd.DataFrame({
                'rsi': rsi,
                'rsi_smooth': rsi_smooth,
                'rsi_momentum': rsi_momentum,
                'rsi_overbought': rsi_overbought.astype(int),
                'rsi_oversold': rsi_oversold.astype(int),
                'rsi_upper_band': rsi_upper,
                'rsi_lower_band': rsi_lower,
                'rsi_position': np.where(rsi > rsi_upper, 1, np.where(rsi < rsi_lower, -1, 0))
            })
            
        except Exception as e:
            logger.error(f"Error calculating RSI signals: {str(e)}")
            return pd.DataFrame({'rsi': [50], 'rsi_smooth': [50], 'rsi_momentum': [0], 
                               'rsi_overbought': [0], 'rsi_oversold': [0], 'rsi_position': [0]})
    
    def _calculate_mfi_signals(self, high: np.ndarray, low: np.ndarray, 
                              close: np.ndarray, volume: np.ndarray) -> pd.DataFrame:
        """Calculate Money Flow Index signals"""
        try:
            # Standard MFI
            mfi = talib.MFI(high, low, close, volume, timeperiod=self.config.mfi_length)
            
            # MFI smoothed
            mfi_smooth = pd.Series(mfi).rolling(window=3).mean()
            
            # MFI momentum
            mfi_momentum = np.diff(mfi, prepend=mfi[0])
            
            # MFI zones
            mfi_overbought = mfi > 80
            mfi_oversold = mfi < 20
            
            # Dynamic MFI thresholds
            mfi_mean = pd.Series(mfi).rolling(window=20).mean()
            mfi_std = pd.Series(mfi).rolling(window=20).std()
            
            return pd.DataFrame({
                'mfi': mfi,
                'mfi_smooth': mfi_smooth,
                'mfi_momentum': mfi_momentum,
                'mfi_overbought': mfi_overbought.astype(int),
                'mfi_oversold': mfi_oversold.astype(int),
                'mfi_mean': mfi_mean,
                'mfi_position': np.where(mfi > 80, 1, np.where(mfi < 20, -1, 0))
            })
            
        except Exception as e:
            logger.error(f"Error calculating MFI signals: {str(e)}")
            return pd.DataFrame({'mfi': [50], 'mfi_smooth': [50], 'mfi_momentum': [0],
                               'mfi_overbought': [0], 'mfi_oversold': [0], 'mfi_position': [0]})
    
    def _calculate_stochastic_signals(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> pd.DataFrame:
        """Calculate Stochastic oscillator signals"""
        try:
            # Stochastic %K and %D
            slowk, slowd = talib.STOCH(
                high, low, close,
                fastk_period=self.config.stoch_length,
                slowk_period=self.config.stoch_smooth_k,
                slowd_period=self.config.stoch_smooth_d
            )
            
            # Stochastic momentum
            stoch_momentum = slowk - slowd
            
            # Stochastic zones
            stoch_overbought = slowk > 80
            stoch_oversold = slowk < 20
            
            # Stochastic cross signals
            stoch_cross = np.where(slowk > slowd, 1, -1)
            
            return pd.DataFrame({
                'stoch_k': slowk,
                'stoch_d': slowd,
                'stoch_momentum': stoch_momentum,
                'stoch_overbought': stoch_overbought.astype(int),
                'stoch_oversold': stoch_oversold.astype(int),
                'stoch_cross': stoch_cross
            })
            
        except Exception as e:
            logger.error(f"Error calculating Stochastic signals: {str(e)}")
            return pd.DataFrame({'stoch_k': [50], 'stoch_d': [50], 'stoch_momentum': [0],
                               'stoch_overbought': [0], 'stoch_oversold': [0], 'stoch_cross': [0]})
    
    def _calculate_macd_signals(self, close: np.ndarray) -> pd.DataFrame:
        """Calculate MACD signals"""
        try:
            # MACD components
            macd, macdsignal, macdhist = talib.MACD(
                close,
                fastperiod=self.config.macd_fast,
                slowperiod=self.config.macd_slow,
                signalperiod=self.config.macd_signal
            )
            
            # MACD momentum
            macd_momentum = np.diff(macd, prepend=macd[0])
            
            # MACD cross signals
            macd_cross = np.where(macd > macdsignal, 1, -1)
            
            # MACD histogram analysis
            hist_momentum = np.diff(macdhist, prepend=macdhist[0])
            
            return pd.DataFrame({
                'macd': macd,
                'macd_signal': macdsignal,
                'macd_histogram': macdhist,
                'macd_momentum': macd_momentum,
                'macd_cross': macd_cross,
                'hist_momentum': hist_momentum
            })
            
        except Exception as e:
            logger.error(f"Error calculating MACD signals: {str(e)}")
            return pd.DataFrame({'macd': [0], 'macd_signal': [0], 'macd_histogram': [0],
                               'macd_momentum': [0], 'macd_cross': [0], 'hist_momentum': [0]})
    
    def _calculate_volume_signals(self, volume: np.ndarray, close: np.ndarray) -> pd.DataFrame:
        """Calculate volume-based signals"""
        try:
            # Volume moving average
            volume_ma = pd.Series(volume).rolling(window=20).mean()
            
            # Volume ratio
            volume_ratio = volume / volume_ma
            
            # Volume surge detection
            volume_surge = volume_ratio > self.config.volume_threshold
            
            # Price-Volume correlation
            price_change = np.diff(close, prepend=close[0])
            volume_change = np.diff(volume, prepend=volume[0])
            
            # Rolling correlation
            correlation_window = 20
            pv_correlation = []
            for i in range(len(price_change)):
                start_idx = max(0, i - correlation_window + 1)
                end_idx = i + 1
                if end_idx - start_idx >= 5:  # Minimum points for correlation
                    corr = np.corrcoef(
                        price_change[start_idx:end_idx],
                        volume_change[start_idx:end_idx]
                    )[0, 1]
                    pv_correlation.append(corr if not np.isnan(corr) else 0)
                else:
                    pv_correlation.append(0)
            
            return pd.DataFrame({
                'volume_ratio': volume_ratio,
                'volume_surge': volume_surge.astype(int),
                'volume_ma': volume_ma,
                'pv_correlation': pv_correlation
            })
            
        except Exception as e:
            logger.error(f"Error calculating volume signals: {str(e)}")
            return pd.DataFrame({'volume_ratio': [1], 'volume_surge': [0], 'pv_correlation': [0]})
    
    def _detect_divergences(self, close: np.ndarray, wave_trend: pd.DataFrame, 
                           rsi_signals: pd.DataFrame) -> pd.DataFrame:
        """Detect price-indicator divergences"""
        try:
            periods = self.config.divergence_periods
            
            # Price highs and lows
            price_highs = pd.Series(close).rolling(window=periods).max()
            price_lows = pd.Series(close).rolling(window=periods).min()
            
            # Indicator highs and lows
            wt_highs = wave_trend['wt1'].rolling(window=periods).max()
            wt_lows = wave_trend['wt1'].rolling(window=periods).min()
            rsi_highs = rsi_signals['rsi'].rolling(window=periods).max()
            rsi_lows = rsi_signals['rsi'].rolling(window=periods).min()
            
            # Bullish divergence detection
            bullish_div_wt = (close < price_lows.shift(1)) & (wave_trend['wt1'] > wt_lows.shift(1))
            bullish_div_rsi = (close < price_lows.shift(1)) & (rsi_signals['rsi'] > rsi_lows.shift(1))
            
            # Bearish divergence detection
            bearish_div_wt = (close > price_highs.shift(1)) & (wave_trend['wt1'] < wt_highs.shift(1))
            bearish_div_rsi = (close > price_highs.shift(1)) & (rsi_signals['rsi'] < rsi_highs.shift(1))
            
            return pd.DataFrame({
                'bullish_div_wt': bullish_div_wt.astype(int),
                'bullish_div_rsi': bullish_div_rsi.astype(int),
                'bearish_div_wt': bearish_div_wt.astype(int),
                'bearish_div_rsi': bearish_div_rsi.astype(int),
                'composite_bullish': (bullish_div_wt | bullish_div_rsi).astype(int),
                'composite_bearish': (bearish_div_wt | bearish_div_rsi).astype(int)
            })
            
        except Exception as e:
            logger.error(f"Error detecting divergences: {str(e)}")
            return pd.DataFrame({
                'bullish_div_wt': [0], 'bullish_div_rsi': [0],
                'bearish_div_wt': [0], 'bearish_div_rsi': [0],
                'composite_bullish': [0], 'composite_bearish': [0]
            })
    
    def _create_feature_matrix(self, wave_trend: pd.DataFrame, rsi_signals: pd.DataFrame,
                              mfi_signals: pd.DataFrame, stoch_signals: pd.DataFrame,
                              macd_signals: pd.DataFrame, volume_signals: pd.DataFrame) -> np.ndarray:
        """Create feature matrix for ML models"""
        try:
            features = []
            
            # Wave Trend features
            features.extend([
                wave_trend['wt1'].values,
                wave_trend['wt2'].values,
                wave_trend['wt_momentum'].values,
                wave_trend['wt_cross'].values
            ])
            
            # RSI features
            features.extend([
                rsi_signals['rsi'].values,
                rsi_signals['rsi_momentum'].values,
                rsi_signals['rsi_position'].values
            ])
            
            # MFI features
            features.extend([
                mfi_signals['mfi'].values,
                mfi_signals['mfi_momentum'].values,
                mfi_signals['mfi_position'].values
            ])
            
            # Stochastic features
            features.extend([
                stoch_signals['stoch_k'].values,
                stoch_signals['stoch_momentum'].values,
                stoch_signals['stoch_cross'].values
            ])
            
            # MACD features
            features.extend([
                macd_signals['macd'].values,
                macd_signals['macd_histogram'].values,
                macd_signals['macd_cross'].values
            ])
            
            # Volume features
            features.extend([
                volume_signals['volume_ratio'].values,
                volume_signals['pv_correlation'].values
            ])
            
            return np.column_stack(features)
            
        except Exception as e:
            logger.error(f"Error creating feature matrix: {str(e)}")
            return np.zeros((len(wave_trend), 16))
    
    def _generate_ml_signals(self, features: np.ndarray, close: np.ndarray) -> pd.DataFrame:
        """Generate ML-enhanced signals"""
        try:
            if len(features) < self.config.ml_lookback:
                return pd.DataFrame({'ml_prediction': np.zeros(len(features)), 'anomaly_score': np.zeros(len(features))})
            
            # Prepare target variable (future returns)
            returns = np.diff(close, prepend=close[0]) / close[:-1] if len(close) > 1 else np.array([0])
            
            # Handle edge cases
            if len(returns) != len(features):
                min_len = min(len(returns), len(features))
                returns = returns[:min_len]
                features = features[:min_len]
            
            if len(features) < 10:  # Minimum data points
                return pd.DataFrame({'ml_prediction': np.zeros(len(features)), 'anomaly_score': np.zeros(len(features))})
            
            # Clean features (handle NaN and infinite values)
            features_clean = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Train models if enough data
            if not self.is_fitted and len(features_clean) >= self.config.ml_lookback:
                try:
                    # Scale features
                    features_scaled = self.scaler.fit_transform(features_clean)
                    
                    # Apply PCA
                    features_pca = self.pca.fit_transform(features_scaled)
                    
                    # Train prediction model
                    lookback = self.config.ml_lookback
                    X_train = features_pca[:-1]  # All but last
                    y_train = returns[1:]        # Future returns
                    
                    if len(X_train) > 0 and len(y_train) > 0:
                        self.ml_model.fit(X_train, y_train)
                        
                        # Train anomaly detector
                        self.anomaly_detector.fit(features_scaled)
                        
                        self.is_fitted = True
                except Exception as e:
                    logger.warning(f"ML model fitting failed: {str(e)}")
                    self.is_fitted = False
            
            # Generate predictions
            if self.is_fitted:
                try:
                    features_scaled = self.scaler.transform(features_clean)
                    features_pca = self.pca.transform(features_scaled)
                    
                    predictions = self.ml_model.predict(features_pca)
                    anomaly_scores = self.anomaly_detector.decision_function(features_scaled)
                    
                    return pd.DataFrame({
                        'ml_prediction': predictions,
                        'anomaly_score': anomaly_scores
                    })
                except Exception as e:
                    logger.warning(f"ML prediction failed: {str(e)}")
            
            # Fallback to zeros
            return pd.DataFrame({
                'ml_prediction': np.zeros(len(features)),
                'anomaly_score': np.zeros(len(features))
            })
            
        except Exception as e:
            logger.error(f"Error generating ML signals: {str(e)}")
            return pd.DataFrame({'ml_prediction': [0], 'anomaly_score': [0]})
    
    def _generate_cipher_signals(self, wave_trend: pd.DataFrame, rsi_signals: pd.DataFrame,
                                mfi_signals: pd.DataFrame, stoch_signals: pd.DataFrame,
                                macd_signals: pd.DataFrame, volume_signals: pd.DataFrame,
                                divergences: pd.DataFrame, ml_signals: pd.DataFrame) -> pd.DataFrame:
        """Generate composite Market Cipher B signals"""
        try:
            # Initialize signal components
            signals = pd.DataFrame(index=range(len(wave_trend)))
            
            # Wave Trend signals (40% weight)
            wt_signal = (wave_trend['wt_cross'] * 0.4 + 
                        np.where(wave_trend['wt_momentum'] > 0, 0.3, -0.3))
            
            # RSI signals (20% weight)
            rsi_signal = (rsi_signals['rsi_position'] * 0.2)
            
            # MFI signals (15% weight)
            mfi_signal = (mfi_signals['mfi_position'] * 0.15)
            
            # Stochastic signals (10% weight)
            stoch_signal = (stoch_signals['stoch_cross'] * 0.1)
            
            # MACD signals (10% weight)
            macd_signal = (macd_signals['macd_cross'] * 0.1)
            
            # Volume confirmation (5% weight)
            volume_signal = np.where(volume_signals['volume_surge'] == 1, 0.05, -0.05)
            
            # Combine base signals
            base_signal = (wt_signal + rsi_signal + mfi_signal + 
                          stoch_signal + macd_signal + volume_signal)
            
            # Apply divergence adjustments
            divergence_adjustment = (divergences['composite_bullish'] * 0.2 - 
                                   divergences['composite_bearish'] * 0.2)
            
            # Apply ML enhancement
            ml_adjustment = np.tanh(ml_signals['ml_prediction'] * 0.1)  # Bounded adjustment
            
            # Calculate composite signal
            composite_signal = base_signal + divergence_adjustment + ml_adjustment
            
            # Apply smoothing
            composite_signal_smooth = pd.Series(composite_signal).rolling(window=3).mean()
            
            # Generate trade signals
            buy_signal = (composite_signal_smooth > 0.3) & (composite_signal_smooth.shift(1) <= 0.3)
            sell_signal = (composite_signal_smooth < -0.3) & (composite_signal_smooth.shift(1) >= -0.3)
            
            signals['composite_signal'] = composite_signal_smooth
            signals['buy_signal'] = buy_signal.astype(int)
            signals['sell_signal'] = sell_signal.astype(int)
            signals['signal_strength'] = np.abs(composite_signal_smooth)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating cipher signals: {str(e)}")
            return pd.DataFrame({'composite_signal': [0], 'buy_signal': [0], 'sell_signal': [0], 'signal_strength': [0]})
    
    def _calculate_confidence_scores(self, signals: pd.DataFrame, features: np.ndarray) -> pd.Series:
        """Calculate confidence scores for signals"""
        try:
            # Base confidence from signal strength
            base_confidence = np.clip(signals['signal_strength'] / 2.0, 0, 1)
            
            # Feature consistency check
            feature_consistency = np.std(features, axis=1) if len(features.shape) > 1 else np.array([0.5] * len(signals))
            feature_confidence = 1 / (1 + feature_consistency)
            
            # Combine confidences
            combined_confidence = (base_confidence * 0.7 + feature_confidence * 0.3)
            
            return pd.Series(combined_confidence, index=signals.index)
            
        except Exception as e:
            logger.error(f"Error calculating confidence scores: {str(e)}")
            return pd.Series([0.5] * len(signals), index=signals.index)
    
    def _generate_final_signals(self, cipher_signals: pd.DataFrame, confidence_scores: pd.Series) -> pd.DataFrame:
        """Generate final trading signals with confidence filtering"""
        try:
            final_signals = cipher_signals.copy()
            
            # Apply confidence threshold
            high_confidence_mask = confidence_scores >= self.config.confidence_threshold
            
            final_signals['buy_signal'] = (cipher_signals['buy_signal'] & high_confidence_mask).astype(int)
            final_signals['sell_signal'] = (cipher_signals['sell_signal'] & high_confidence_mask).astype(int)
            final_signals['confidence'] = confidence_scores
            
            # Calculate signal quality
            final_signals['signal_quality'] = final_signals['signal_strength'] * confidence_scores
            
            return final_signals
            
        except Exception as e:
            logger.error(f"Error generating final signals: {str(e)}")
            return cipher_signals
    
    def _normalize_series(self, series: pd.Series, window: int = 50) -> pd.Series:
        """Normalize series using rolling statistics"""
        try:
            rolling_mean = series.rolling(window=window, min_periods=1).mean()
            rolling_std = series.rolling(window=window, min_periods=1).std()
            rolling_std = rolling_std.replace(0, 1)  # Avoid division by zero
            
            return (series - rolling_mean) / rolling_std
            
        except Exception as e:
            logger.error(f"Error normalizing series: {str(e)}")
            return series
    
    def _prepare_metadata(self, wave_trend: pd.DataFrame, rsi_signals: pd.DataFrame,
                         mfi_signals: pd.DataFrame, stoch_signals: pd.DataFrame,
                         macd_signals: pd.DataFrame, volume_signals: pd.DataFrame,
                         divergences: pd.DataFrame, ml_signals: pd.DataFrame,
                         confidence_scores: pd.Series) -> Dict[str, Any]:
        """Prepare comprehensive metadata"""
        try:
            return {
                'indicator_name': 'Market Cipher B',
                'version': '1.0.0',
                'data_points': len(wave_trend),
                'config': {
                    'wt_channel_length': self.config.wt_channel_length,
                    'wt_average_length': self.config.wt_average_length,
                    'rsi_length': self.config.rsi_length,
                    'mfi_length': self.config.mfi_length,
                    'confidence_threshold': self.config.confidence_threshold
                },
                'statistics': {
                    'avg_wt1': float(wave_trend['wt1'].mean()) if len(wave_trend) > 0 else 0.0,
                    'avg_rsi': float(rsi_signals['rsi'].mean()) if len(rsi_signals) > 0 else 50.0,
                    'avg_mfi': float(mfi_signals['mfi'].mean()) if len(mfi_signals) > 0 else 50.0,
                    'avg_confidence': float(confidence_scores.mean()) if len(confidence_scores) > 0 else 0.5,
                    'bullish_divergences': int(divergences['composite_bullish'].sum()) if len(divergences) > 0 else 0,
                    'bearish_divergences': int(divergences['composite_bearish'].sum()) if len(divergences) > 0 else 0
                },
                'ml_info': {
                    'model_fitted': self.is_fitted,
                    'feature_count': len(ml_signals.columns) if len(ml_signals) > 0 else 0,
                    'avg_ml_prediction': float(ml_signals['ml_prediction'].mean()) if len(ml_signals) > 0 else 0.0
                }
            }
            
        except Exception as e:
            logger.error(f"Error preparing metadata: {str(e)}")
            return {'indicator_name': 'Market Cipher B', 'error': str(e)}
    
    def _get_error_result(self, error_message: str) -> Dict[str, Any]:
        """Return error result structure"""
        return {
            'signal': 0.0,
            'signals': [],
            'wave_trend': pd.DataFrame(),
            'rsi_signals': pd.DataFrame(),
            'mfi_signals': pd.DataFrame(),
            'stoch_signals': pd.DataFrame(),
            'macd_signals': pd.DataFrame(),
            'volume_signals': pd.DataFrame(),
            'divergences': pd.DataFrame(),
            'ml_predictions': pd.DataFrame(),
            'confidence': 0.5,
            'metadata': {
                'indicator_name': 'Market Cipher B',
                'error': error_message,
                'status': 'error'
            }
        }