"""
Relative Strength Mansfield Indicator - Advanced Implementation

This indicator implements Stan Weinstein's Mansfield Relative Strength methodology
with advanced machine learning enhancements for superior market analysis.
Based on "Secrets for Profiting in Bull and Bear Markets" principles.

Author: Humanitarian Trading Platform
License: MIT
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import talib
import warnings
warnings.filterwarnings('ignore')

from ..base.standard_indicator import StandardIndicatorInterface

logger = logging.getLogger(__name__)


@dataclass
class MansfieldRSConfig:
    """Configuration for Mansfield Relative Strength Indicator"""
    primary_period: int = 52  # 52-week period for relative strength
    smooth_period: int = 10   # Smoothing period
    benchmark_period: int = 250  # Benchmark lookback period
    trend_period: int = 20    # Trend determination period
    volatility_period: int = 20  # Volatility calculation period
    ml_lookback: int = 100    # ML training lookback
    strength_threshold: float = 1.0  # Relative strength threshold
    confidence_threshold: float = 0.7  # Confidence threshold
    regime_periods: List[int] = None  # Multiple timeframe analysis
    adaptive_smoothing: bool = True   # Adaptive smoothing based on volatility


class RelativeStrengthMansfield(StandardIndicatorInterface):
    """
    Advanced Mansfield Relative Strength Indicator Implementation
    
    Calculates relative strength using Stan Weinstein's methodology enhanced
    with machine learning for market regime detection and adaptive parameters.
    """
    
    def __init__(self, config: Optional[MansfieldRSConfig] = None):
        """Initialize Mansfield Relative Strength indicator"""
        self.config = config or MansfieldRSConfig()
        if self.config.regime_periods is None:
            self.config.regime_periods = [10, 20, 50, 100]
            
        self.scaler = RobustScaler()
        self.ml_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.regime_classifier = KMeans(n_clusters=4, random_state=42)
        self.pca = PCA(n_components=5)
        self.is_fitted = False
        
        # Cache for benchmark data
        self.benchmark_cache = {}
        
    def calculate(self, data: pd.DataFrame, benchmark_data: Optional[pd.DataFrame] = None, **kwargs) -> Dict[str, Any]:
        """
        Calculate Mansfield Relative Strength with advanced ML integration
        
        Args:
            data: DataFrame with OHLCV data for the security
            benchmark_data: DataFrame with benchmark data (e.g., S&P 500)
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing Mansfield RS signals and metadata
        """
        try:
            if len(data) < self.config.primary_period:
                raise ValueError("Insufficient data for Mansfield RS calculation")
                
            # Extract price data
            close = data['close'].values
            volume = data['volume'].values if 'volume' in data else None
            
            # Use S&P 500 proxy if no benchmark provided
            if benchmark_data is None:
                benchmark_data = self._create_market_proxy(data)
            
            benchmark_close = benchmark_data['close'].values
            
            # Calculate core relative strength
            relative_strength = self._calculate_relative_strength(close, benchmark_close)
            
            # Calculate Mansfield RS components
            mansfield_rs = self._calculate_mansfield_rs(relative_strength)
            
            # Trend analysis
            trend_analysis = self._analyze_trends(close, mansfield_rs)
            
            # Volume confirmation
            volume_confirmation = self._calculate_volume_confirmation(close, volume, mansfield_rs)
            
            # Market regime detection
            regime_analysis = self._detect_market_regimes(close, benchmark_close, mansfield_rs)
            
            # Volatility-based adaptive parameters
            adaptive_params = self._calculate_adaptive_parameters(close, mansfield_rs)
            
            # Create feature matrix for ML
            features = self._create_feature_matrix(
                mansfield_rs, trend_analysis, volume_confirmation,
                regime_analysis, adaptive_params, close
            )
            
            # Generate ML-enhanced signals
            ml_signals = self._generate_ml_signals(features, mansfield_rs)
            
            # Generate final trading signals
            trading_signals = self._generate_trading_signals(
                mansfield_rs, trend_analysis, volume_confirmation,
                regime_analysis, ml_signals, adaptive_params
            )
            
            # Calculate confidence scores
            confidence_scores = self._calculate_confidence_scores(
                trading_signals, features, mansfield_rs
            )
            
            # Performance analytics
            performance_metrics = self._calculate_performance_metrics(
                close, mansfield_rs, trading_signals
            )
            
            # Prepare metadata
            metadata = self._prepare_metadata(
                mansfield_rs, trend_analysis, regime_analysis,
                performance_metrics, confidence_scores
            )
            
            return {
                'signal': trading_signals['composite_signal'].iloc[-1] if len(trading_signals) > 0 else 0.0,
                'mansfield_rs': mansfield_rs,
                'relative_strength': relative_strength,
                'trend_analysis': trend_analysis,
                'volume_confirmation': volume_confirmation,
                'regime_analysis': regime_analysis,
                'ml_signals': ml_signals,
                'trading_signals': trading_signals,
                'confidence': confidence_scores.iloc[-1] if len(confidence_scores) > 0 else 0.5,
                'performance_metrics': performance_metrics,
                'adaptive_params': adaptive_params,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Error calculating Mansfield RS: {str(e)}")
            return self._get_error_result(str(e))
    
    def _calculate_relative_strength(self, security_prices: np.ndarray, benchmark_prices: np.ndarray) -> pd.DataFrame:
        """Calculate basic relative strength ratio"""
        try:
            # Ensure same length
            min_length = min(len(security_prices), len(benchmark_prices))
            security_prices = security_prices[-min_length:]
            benchmark_prices = benchmark_prices[-min_length:]
            
            # Calculate relative strength ratio
            rs_ratio = security_prices / benchmark_prices
            
            # Calculate rate of change over different periods
            rs_roc_fast = pd.Series(rs_ratio).pct_change(periods=self.config.trend_period)
            rs_roc_medium = pd.Series(rs_ratio).pct_change(periods=self.config.primary_period // 2)
            rs_roc_slow = pd.Series(rs_ratio).pct_change(periods=self.config.primary_period)
            
            # Normalize relative strength
            rs_normalized = self._normalize_series(pd.Series(rs_ratio), window=self.config.primary_period)
            
            # Calculate relative strength momentum
            rs_momentum = pd.Series(rs_ratio).diff(periods=self.config.trend_period)
            
            return pd.DataFrame({
                'rs_ratio': rs_ratio,
                'rs_normalized': rs_normalized,
                'rs_roc_fast': rs_roc_fast,
                'rs_roc_medium': rs_roc_medium,
                'rs_roc_slow': rs_roc_slow,
                'rs_momentum': rs_momentum
            })
            
        except Exception as e:
            logger.error(f"Error calculating relative strength: {str(e)}")
            return pd.DataFrame({'rs_ratio': [1], 'rs_normalized': [0], 'rs_momentum': [0]})
    
    def _calculate_mansfield_rs(self, relative_strength: pd.DataFrame) -> pd.DataFrame:
        """Calculate Mansfield Relative Strength indicator"""
        try:
            rs_ratio = relative_strength['rs_ratio']
            
            # Mansfield RS calculation
            # RS = (Current Price / Price n periods ago) / (Index Current / Index n periods ago)
            security_change = rs_ratio / rs_ratio.shift(self.config.primary_period)
            
            # Convert to percentile ranking
            mansfield_raw = security_change.rolling(window=self.config.primary_period).rank(pct=True)
            
            # Smooth the indicator
            if self.config.adaptive_smoothing:
                # Calculate volatility for adaptive smoothing
                volatility = relative_strength['rs_ratio'].rolling(window=self.config.volatility_period).std()
                smoothing_factor = np.clip(volatility * 10, 3, 20).astype(int)
                
                mansfield_smooth = pd.Series(index=mansfield_raw.index)
                for i in range(len(mansfield_raw)):
                    if i >= smoothing_factor.iloc[i] if not pd.isna(smoothing_factor.iloc[i]) else 10:
                        window = int(smoothing_factor.iloc[i]) if not pd.isna(smoothing_factor.iloc[i]) else 10
                        mansfield_smooth.iloc[i] = mansfield_raw.iloc[max(0, i-window):i+1].mean()
                    else:
                        mansfield_smooth.iloc[i] = mansfield_raw.iloc[i]
            else:
                mansfield_smooth = mansfield_raw.rolling(window=self.config.smooth_period).mean()
            
            # Calculate Mansfield RS line (0-100 scale)
            mansfield_rs_line = (mansfield_smooth - 0.5) * 100
            
            # Calculate RS ranking percentile
            rs_percentile = mansfield_raw * 100
            
            # Calculate trend strength
            trend_strength = mansfield_rs_line.diff(periods=self.config.trend_period)
            
            # Stage analysis (Weinstein's 4-stage cycle)
            stage_analysis = self._calculate_stage_analysis(mansfield_rs_line, trend_strength)
            
            return pd.DataFrame({
                'mansfield_rs': mansfield_rs_line,
                'rs_percentile': rs_percentile,
                'mansfield_smooth': mansfield_smooth,
                'trend_strength': trend_strength,
                'stage': stage_analysis['stage'],
                'stage_confidence': stage_analysis['confidence']
            })
            
        except Exception as e:
            logger.error(f"Error calculating Mansfield RS: {str(e)}")
            return pd.DataFrame({'mansfield_rs': [0], 'rs_percentile': [50], 'stage': [2]})
    
    def _calculate_stage_analysis(self, mansfield_rs: pd.Series, trend_strength: pd.Series) -> Dict[str, pd.Series]:
        """Calculate Weinstein's 4-stage market cycle"""
        try:
            # Stage 1: Accumulation (RS flat, beginning to turn up)
            # Stage 2: Advancing (RS rising, strong uptrend)
            # Stage 3: Distribution (RS peaks, beginning to weaken)
            # Stage 4: Declining (RS falling, downtrend)
            
            # Calculate moving averages for trend determination
            ma_short = mansfield_rs.rolling(window=10).mean()
            ma_long = mansfield_rs.rolling(window=30).mean()
            
            # Stage determination logic
            stage = pd.Series(index=mansfield_rs.index, dtype=int)
            confidence = pd.Series(index=mansfield_rs.index, dtype=float)
            
            for i in range(len(mansfield_rs)):
                if i < 30:  # Not enough data
                    stage.iloc[i] = 2
                    confidence.iloc[i] = 0.5
                    continue
                
                current_rs = mansfield_rs.iloc[i]
                ma_short_val = ma_short.iloc[i]
                ma_long_val = ma_long.iloc[i]
                trend_val = trend_strength.iloc[i] if not pd.isna(trend_strength.iloc[i]) else 0
                
                # Stage 1: Accumulation
                if (current_rs < 0 and trend_val > 0 and ma_short_val > ma_long_val):
                    stage.iloc[i] = 1
                    confidence.iloc[i] = min(0.8, abs(trend_val) / 10)
                    
                # Stage 2: Advancing
                elif (current_rs > 0 and trend_val > 0 and ma_short_val > ma_long_val):
                    stage.iloc[i] = 2
                    confidence.iloc[i] = min(0.9, abs(trend_val) / 5)
                    
                # Stage 3: Distribution
                elif (current_rs > 0 and trend_val < 0 and ma_short_val < ma_long_val):
                    stage.iloc[i] = 3
                    confidence.iloc[i] = min(0.8, abs(trend_val) / 10)
                    
                # Stage 4: Declining
                elif (current_rs < 0 and trend_val < 0 and ma_short_val < ma_long_val):
                    stage.iloc[i] = 4
                    confidence.iloc[i] = min(0.9, abs(trend_val) / 5)
                    
                else:
                    # Transition or unclear stage
                    stage.iloc[i] = stage.iloc[i-1] if i > 0 else 2
                    confidence.iloc[i] = 0.3
            
            return {'stage': stage, 'confidence': confidence}
            
        except Exception as e:
            logger.error(f"Error calculating stage analysis: {str(e)}")
            return {'stage': pd.Series([2] * len(mansfield_rs)), 'confidence': pd.Series([0.5] * len(mansfield_rs))}
    
    def _analyze_trends(self, prices: np.ndarray, mansfield_rs: pd.DataFrame) -> pd.DataFrame:
        """Analyze price and RS trends"""
        try:
            # Price trend analysis
            price_ma_short = pd.Series(prices).rolling(window=20).mean()
            price_ma_long = pd.Series(prices).rolling(window=50).mean()
            price_trend = np.where(price_ma_short > price_ma_long, 1, -1)
            
            # RS trend analysis
            rs_ma_short = mansfield_rs['mansfield_rs'].rolling(window=10).mean()
            rs_ma_long = mansfield_rs['mansfield_rs'].rolling(window=30).mean()
            rs_trend = np.where(rs_ma_short > rs_ma_long, 1, -1)
            
            # Trend alignment
            trend_alignment = price_trend * rs_trend  # +1 aligned bullish, -1 aligned bearish, 0 divergent
            
            # Trend strength
            price_trend_strength = abs(price_ma_short - price_ma_long) / price_ma_long
            rs_trend_strength = abs(rs_ma_short - rs_ma_long)
            
            # Trend acceleration
            price_acceleration = pd.Series(prices).diff().diff()
            rs_acceleration = mansfield_rs['mansfield_rs'].diff().diff()
            
            return pd.DataFrame({
                'price_trend': price_trend,
                'rs_trend': rs_trend,
                'trend_alignment': trend_alignment,
                'price_trend_strength': price_trend_strength,
                'rs_trend_strength': rs_trend_strength,
                'price_acceleration': price_acceleration,
                'rs_acceleration': rs_acceleration
            })
            
        except Exception as e:
            logger.error(f"Error analyzing trends: {str(e)}")
            return pd.DataFrame({'price_trend': [1], 'rs_trend': [1], 'trend_alignment': [1]})
    
    def _calculate_volume_confirmation(self, prices: np.ndarray, volume: Optional[np.ndarray], 
                                     mansfield_rs: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume confirmation signals"""
        try:
            if volume is None:
                # Create dummy volume data
                volume = np.ones(len(prices))
            
            # Volume moving average
            volume_ma = pd.Series(volume).rolling(window=20).mean()
            volume_ratio = volume / volume_ma
            
            # Volume trend
            volume_trend = pd.Series(volume).rolling(window=10).mean() > pd.Series(volume).rolling(window=30).mean()
            
            # Price-volume relationship
            price_change = pd.Series(prices).pct_change()
            volume_confirmation = np.where(
                (price_change > 0) & (volume_ratio > 1.2), 1,  # Strong buying
                np.where((price_change < 0) & (volume_ratio > 1.2), -1, 0)  # Strong selling
            )
            
            # RS-volume correlation
            rs_change = mansfield_rs['mansfield_rs'].pct_change()
            rs_volume_correlation = pd.Series(rs_change).rolling(window=20).corr(pd.Series(volume_ratio))
            
            return pd.DataFrame({
                'volume_ratio': volume_ratio,
                'volume_trend': volume_trend.astype(int),
                'volume_confirmation': volume_confirmation,
                'rs_volume_correlation': rs_volume_correlation
            })
            
        except Exception as e:
            logger.error(f"Error calculating volume confirmation: {str(e)}")
            return pd.DataFrame({'volume_ratio': [1], 'volume_confirmation': [0]})
    
    def _detect_market_regimes(self, security_prices: np.ndarray, benchmark_prices: np.ndarray,
                              mansfield_rs: pd.DataFrame) -> pd.DataFrame:
        """Detect market regimes using multiple timeframes"""
        try:
            regimes = pd.DataFrame(index=range(len(security_prices)))
            
            # Calculate volatility regimes
            security_vol = pd.Series(security_prices).pct_change().rolling(window=20).std()
            benchmark_vol = pd.Series(benchmark_prices).pct_change().rolling(window=20).std()
            
            vol_regime = np.where(
                (security_vol > security_vol.quantile(0.75)) | (benchmark_vol > benchmark_vol.quantile(0.75)),
                'high_vol',
                np.where(
                    (security_vol < security_vol.quantile(0.25)) & (benchmark_vol < benchmark_vol.quantile(0.25)),
                    'low_vol', 'normal_vol'
                )
            )
            
            # Calculate trend regimes
            security_returns = pd.Series(security_prices).pct_change(periods=self.config.trend_period)
            benchmark_returns = pd.Series(benchmark_prices).pct_change(periods=self.config.trend_period)
            
            trend_regime = np.where(
                (security_returns > 0) & (benchmark_returns > 0), 'bull_market',
                np.where(
                    (security_returns < 0) & (benchmark_returns < 0), 'bear_market', 'mixed_market'
                )
            )
            
            # RS regime
            rs_regime = np.where(
                mansfield_rs['mansfield_rs'] > mansfield_rs['mansfield_rs'].quantile(0.75), 'outperforming',
                np.where(
                    mansfield_rs['mansfield_rs'] < mansfield_rs['mansfield_rs'].quantile(0.25), 'underperforming',
                    'neutral'
                )
            )
            
            # Market cycle phase
            market_cycle = self._determine_market_cycle(security_prices, benchmark_prices)
            
            regimes['volatility_regime'] = vol_regime
            regimes['trend_regime'] = trend_regime
            regimes['rs_regime'] = rs_regime
            regimes['market_cycle'] = market_cycle
            
            return regimes
            
        except Exception as e:
            logger.error(f"Error detecting market regimes: {str(e)}")
            return pd.DataFrame({'volatility_regime': ['normal_vol'], 'trend_regime': ['mixed_market'], 'rs_regime': ['neutral']})
    
    def _determine_market_cycle(self, security_prices: np.ndarray, benchmark_prices: np.ndarray) -> List[str]:
        """Determine market cycle phase"""
        try:
            # Calculate long-term moving averages
            security_ma_long = pd.Series(security_prices).rolling(window=200).mean()
            benchmark_ma_long = pd.Series(benchmark_prices).rolling(window=200).mean()
            
            # Current position relative to long-term MA
            security_position = security_prices > security_ma_long
            benchmark_position = benchmark_prices > benchmark_ma_long
            
            # Slope of long-term MA
            security_slope = security_ma_long.diff(periods=20) > 0
            benchmark_slope = benchmark_ma_long.diff(periods=20) > 0
            
            cycle_phase = []
            for i in range(len(security_prices)):
                if security_position.iloc[i] and benchmark_position.iloc[i] and security_slope.iloc[i]:
                    cycle_phase.append('expansion')
                elif not security_position.iloc[i] and not benchmark_position.iloc[i] and not security_slope.iloc[i]:
                    cycle_phase.append('contraction')
                elif security_position.iloc[i] and not security_slope.iloc[i]:
                    cycle_phase.append('peak')
                elif not security_position.iloc[i] and security_slope.iloc[i]:
                    cycle_phase.append('trough')
                else:
                    cycle_phase.append('transition')
            
            return cycle_phase
            
        except Exception as e:
            logger.error(f"Error determining market cycle: {str(e)}")
            return ['transition'] * len(security_prices)
    
    def _calculate_adaptive_parameters(self, prices: np.ndarray, mansfield_rs: pd.DataFrame) -> pd.DataFrame:
        """Calculate adaptive parameters based on market conditions"""
        try:
            # Volatility-based adaptive smoothing
            volatility = pd.Series(prices).pct_change().rolling(window=20).std()
            
            # Adaptive thresholds
            rs_volatility = mansfield_rs['mansfield_rs'].rolling(window=20).std()
            
            adaptive_upper_threshold = self.config.strength_threshold + rs_volatility
            adaptive_lower_threshold = -self.config.strength_threshold - rs_volatility
            
            # Adaptive lookback periods
            adaptive_short_period = np.clip(10 / (1 + volatility * 100), 5, 20).astype(int)
            adaptive_long_period = np.clip(50 / (1 + volatility * 50), 20, 100).astype(int)
            
            # Market stress indicator
            stress_indicator = (volatility / volatility.rolling(window=60).mean()) - 1
            
            return pd.DataFrame({
                'volatility': volatility,
                'rs_volatility': rs_volatility,
                'adaptive_upper_threshold': adaptive_upper_threshold,
                'adaptive_lower_threshold': adaptive_lower_threshold,
                'adaptive_short_period': adaptive_short_period,
                'adaptive_long_period': adaptive_long_period,
                'stress_indicator': stress_indicator
            })
            
        except Exception as e:
            logger.error(f"Error calculating adaptive parameters: {str(e)}")
            return pd.DataFrame({'volatility': [0.01], 'adaptive_upper_threshold': [1], 'adaptive_lower_threshold': [-1]})
    
    def _create_feature_matrix(self, mansfield_rs: pd.DataFrame, trend_analysis: pd.DataFrame,
                              volume_confirmation: pd.DataFrame, regime_analysis: pd.DataFrame,
                              adaptive_params: pd.DataFrame, prices: np.ndarray) -> np.ndarray:
        """Create feature matrix for ML models"""
        try:
            features = []
            
            # Mansfield RS features
            features.extend([
                mansfield_rs['mansfield_rs'].values,
                mansfield_rs['rs_percentile'].values,
                mansfield_rs['trend_strength'].values,
                mansfield_rs['stage'].values
            ])
            
            # Trend features
            features.extend([
                trend_analysis['price_trend'].values,
                trend_analysis['rs_trend'].values,
                trend_analysis['trend_alignment'].values,
                trend_analysis['price_trend_strength'].values
            ])
            
            # Volume features
            features.extend([
                volume_confirmation['volume_ratio'].values,
                volume_confirmation['volume_confirmation'].values
            ])
            
            # Price momentum features
            price_roc = pd.Series(prices).pct_change(periods=10).values
            features.append(price_roc)
            
            # Adaptive parameter features
            features.extend([
                adaptive_params['volatility'].values,
                adaptive_params['stress_indicator'].values
            ])
            
            return np.column_stack(features)
            
        except Exception as e:
            logger.error(f"Error creating feature matrix: {str(e)}")
            return np.zeros((len(mansfield_rs), 12))
    
    def _generate_ml_signals(self, features: np.ndarray, mansfield_rs: pd.DataFrame) -> pd.DataFrame:
        """Generate ML-enhanced signals"""
        try:
            if len(features) < self.config.ml_lookback:
                return pd.DataFrame({'ml_signal': np.zeros(len(features)), 'ml_confidence': np.zeros(len(features))})
            
            # Prepare target variable (future RS change)
            target = mansfield_rs['mansfield_rs'].shift(-5) - mansfield_rs['mansfield_rs']
            
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
                    
                    # Train model
                    split_point = int(len(features_scaled) * 0.8)
                    X_train, y_train = features_scaled[:split_point], target_clean[:split_point]
                    
                    if len(X_train) > 10:
                        self.ml_model.fit(X_train, y_train)
                        self.is_fitted = True
                        
                        # Calculate feature importance
                        self.feature_importance = self.ml_model.feature_importances_
                        
                except Exception as e:
                    logger.warning(f"ML model training failed: {str(e)}")
                    self.is_fitted = False
            
            # Generate predictions
            if self.is_fitted:
                try:
                    features_scaled = self.scaler.transform(features_clean)
                    predictions = self.ml_model.predict(features_scaled)
                    
                    # Calculate prediction confidence based on consistency
                    prediction_std = np.std(predictions[-20:]) if len(predictions) > 20 else np.std(predictions)
                    confidence = 1 / (1 + prediction_std) if prediction_std > 0 else 0.5
                    
                    ml_signals = np.tanh(predictions)  # Normalize predictions
                    ml_confidence = np.full(len(predictions), confidence)
                    
                    return pd.DataFrame({
                        'ml_signal': ml_signals,
                        'ml_confidence': ml_confidence
                    })
                except Exception as e:
                    logger.warning(f"ML prediction failed: {str(e)}")
            
            # Fallback
            return pd.DataFrame({
                'ml_signal': np.zeros(len(features)),
                'ml_confidence': np.full(len(features), 0.5)
            })
            
        except Exception as e:
            logger.error(f"Error generating ML signals: {str(e)}")
            return pd.DataFrame({'ml_signal': [0], 'ml_confidence': [0.5]})
    
    def _generate_trading_signals(self, mansfield_rs: pd.DataFrame, trend_analysis: pd.DataFrame,
                                 volume_confirmation: pd.DataFrame, regime_analysis: pd.DataFrame,
                                 ml_signals: pd.DataFrame, adaptive_params: pd.DataFrame) -> pd.DataFrame:
        """Generate comprehensive trading signals"""
        try:
            signals = pd.DataFrame(index=mansfield_rs.index)
            
            # Base Mansfield RS signals
            rs_signal = np.where(
                mansfield_rs['mansfield_rs'] > adaptive_params['adaptive_upper_threshold'], 1,
                np.where(mansfield_rs['mansfield_rs'] < adaptive_params['adaptive_lower_threshold'], -1, 0)
            )
            
            # Stage-based signals (Weinstein methodology)
            stage_signal = np.where(
                mansfield_rs['stage'] == 2, 1,  # Stage 2: Buy
                np.where(mansfield_rs['stage'] == 4, -1, 0)  # Stage 4: Sell
            )
            
            # Trend alignment signals
            trend_signal = trend_analysis['trend_alignment'] * 0.5
            
            # Volume confirmation
            volume_signal = volume_confirmation['volume_confirmation'] * 0.3
            
            # ML enhancement
            ml_enhancement = ml_signals['ml_signal'] * ml_signals['ml_confidence']
            
            # Composite signal
            composite_signal = (
                rs_signal * 0.4 +
                stage_signal * 0.3 +
                trend_signal * 0.15 +
                volume_signal * 0.1 +
                ml_enhancement * 0.05
            )
            
            # Generate buy/sell signals
            buy_signal = (composite_signal > 0.5) & (composite_signal.shift(1) <= 0.5)
            sell_signal = (composite_signal < -0.5) & (composite_signal.shift(1) >= -0.5)
            
            signals['composite_signal'] = composite_signal
            signals['rs_signal'] = rs_signal
            signals['stage_signal'] = stage_signal
            signals['buy_signal'] = buy_signal.astype(int)
            signals['sell_signal'] = sell_signal.astype(int)
            signals['signal_strength'] = np.abs(composite_signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating trading signals: {str(e)}")
            return pd.DataFrame({'composite_signal': [0], 'buy_signal': [0], 'sell_signal': [0]})
    
    def _calculate_confidence_scores(self, signals: pd.DataFrame, features: np.ndarray, 
                                    mansfield_rs: pd.DataFrame) -> pd.Series:
        """Calculate confidence scores for signals"""
        try:
            # Signal strength confidence
            signal_confidence = signals['signal_strength']
            
            # Stage confidence
            stage_confidence = mansfield_rs['stage_confidence']
            
            # Feature stability
            feature_std = np.std(features, axis=1) if len(features.shape) > 1 else np.array([0.5] * len(signals))
            stability_confidence = 1 / (1 + feature_std)
            
            # Combined confidence
            combined_confidence = (
                signal_confidence * 0.4 +
                stage_confidence * 0.4 +
                stability_confidence * 0.2
            )
            
            return pd.Series(combined_confidence, index=signals.index)
            
        except Exception as e:
            logger.error(f"Error calculating confidence scores: {str(e)}")
            return pd.Series([0.5] * len(signals), index=signals.index)
    
    def _calculate_performance_metrics(self, prices: np.ndarray, mansfield_rs: pd.DataFrame,
                                     signals: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance metrics"""
        try:
            # Price returns
            price_returns = pd.Series(prices).pct_change()
            
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
            
            # Win rate
            positive_returns = signal_returns > 0
            win_rate = positive_returns.sum() / len(positive_returns.dropna()) if len(positive_returns.dropna()) > 0 else 0
            
            return {
                'total_return': float(total_return),
                'volatility': float(volatility),
                'sharpe_ratio': float(sharpe_ratio),
                'max_drawdown': float(max_drawdown),
                'win_rate': float(win_rate),
                'avg_rs': float(mansfield_rs['mansfield_rs'].mean()),
                'rs_volatility': float(mansfield_rs['mansfield_rs'].std())
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            return {'total_return': 0.0, 'sharpe_ratio': 0.0, 'max_drawdown': 0.0}
    
    def _create_market_proxy(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create a market proxy when no benchmark is provided"""
        try:
            # Simple market proxy using smoothed price data
            close = data['close']
            market_proxy = close.rolling(window=10).mean()
            
            return pd.DataFrame({'close': market_proxy})
            
        except Exception as e:
            logger.error(f"Error creating market proxy: {str(e)}")
            return data[['close']]
    
    def _normalize_series(self, series: pd.Series, window: int) -> pd.Series:
        """Normalize series using rolling statistics"""
        try:
            rolling_mean = series.rolling(window=window, min_periods=1).mean()
            rolling_std = series.rolling(window=window, min_periods=1).std()
            rolling_std = rolling_std.replace(0, 1)
            
            return (series - rolling_mean) / rolling_std
            
        except Exception as e:
            logger.error(f"Error normalizing series: {str(e)}")
            return series
    
    def _prepare_metadata(self, mansfield_rs: pd.DataFrame, trend_analysis: pd.DataFrame,
                         regime_analysis: pd.DataFrame, performance_metrics: Dict[str, float],
                         confidence_scores: pd.Series) -> Dict[str, Any]:
        """Prepare comprehensive metadata"""
        try:
            return {
                'indicator_name': 'Mansfield Relative Strength',
                'version': '1.0.0',
                'data_points': len(mansfield_rs),
                'config': {
                    'primary_period': self.config.primary_period,
                    'smooth_period': self.config.smooth_period,
                    'strength_threshold': self.config.strength_threshold,
                    'adaptive_smoothing': self.config.adaptive_smoothing
                },
                'current_metrics': {
                    'current_rs': float(mansfield_rs['mansfield_rs'].iloc[-1]) if len(mansfield_rs) > 0 else 0.0,
                    'current_stage': int(mansfield_rs['stage'].iloc[-1]) if len(mansfield_rs) > 0 else 2,
                    'trend_alignment': int(trend_analysis['trend_alignment'].iloc[-1]) if len(trend_analysis) > 0 else 0,
                    'avg_confidence': float(confidence_scores.mean()) if len(confidence_scores) > 0 else 0.5
                },
                'performance_metrics': performance_metrics,
                'ml_info': {
                    'model_fitted': self.is_fitted,
                    'feature_importance': getattr(self, 'feature_importance', None)
                },
                'regime_analysis': {
                    'current_volatility_regime': regime_analysis['volatility_regime'].iloc[-1] if len(regime_analysis) > 0 else 'normal_vol',
                    'current_trend_regime': regime_analysis['trend_regime'].iloc[-1] if len(regime_analysis) > 0 else 'mixed_market'
                }
            }
            
        except Exception as e:
            logger.error(f"Error preparing metadata: {str(e)}")
            return {'indicator_name': 'Mansfield Relative Strength', 'error': str(e)}
    
    def _get_error_result(self, error_message: str) -> Dict[str, Any]:
        """Return error result structure"""
        return {
            'signal': 0.0,
            'mansfield_rs': pd.DataFrame(),
            'relative_strength': pd.DataFrame(),
            'trend_analysis': pd.DataFrame(),
            'volume_confirmation': pd.DataFrame(),
            'regime_analysis': pd.DataFrame(),
            'ml_signals': pd.DataFrame(),
            'trading_signals': pd.DataFrame(),
            'confidence': 0.5,
            'performance_metrics': {},
            'adaptive_params': pd.DataFrame(),
            'metadata': {
                'indicator_name': 'Mansfield Relative Strength',
                'error': error_message,
                'status': 'error'
            }
        }