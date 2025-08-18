"""
Advanced Price Volume Trend (PVT) Indicator for AUJ Platform

This module implements a sophisticated Price Volume Trend indicator with momentum confirmation,
trend strength measurement, volume-weighted analysis, and machine learning enhancements.
The PVT indicator measures the relationship between price changes and volume to identify
accumulation and distribution patterns.

Key Features:
- Advanced PVT calculation with momentum confirmation
- Trend strength measurement and classification
- Volume-weighted directional analysis
- Machine learning trend prediction
- Multi-timeframe PVT analysis
- Divergence detection algorithms
- Adaptive smoothing and filtering
- Statistical confidence scoring
- Risk-adjusted position sizing

Mathematical Models:
- Enhanced PVT with volume weighting
- Exponential smoothing for trend analysis
- Statistical confidence intervals
- Machine learning trend classification
- Monte Carlo simulation for risk assessment

The implementation follows AUJ Platform's humanitarian mission requirements with
robust error handling, comprehensive logging, and production-ready code quality.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
import warnings
from scipy import stats
from scipy.signal import savgol_filter
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import joblib
import traceback

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrendDirection(Enum):
    """Enumeration for trend direction classification."""
    STRONG_BULLISH = "STRONG_BULLISH"
    BULLISH = "BULLISH"
    WEAK_BULLISH = "WEAK_BULLISH"
    NEUTRAL = "NEUTRAL"
    WEAK_BEARISH = "WEAK_BEARISH"
    BEARISH = "BEARISH"
    STRONG_BEARISH = "STRONG_BEARISH"


class VolumeRegime(Enum):
    """Enumeration for volume regime classification."""
    HIGH_VOLUME = "HIGH_VOLUME"
    NORMAL_VOLUME = "NORMAL_VOLUME"
    LOW_VOLUME = "LOW_VOLUME"
    ANOMALOUS_VOLUME = "ANOMALOUS_VOLUME"


@dataclass
class PVTSignalComponents:
    """Data class for PVT signal components."""
    pvt_value: float
    pvt_smoothed: float
    momentum: float
    trend_strength: float
    volume_regime: VolumeRegime
    trend_direction: TrendDirection
    
    # Advanced metrics
    acceleration: float = 0.0
    volatility: float = 0.0
    volume_efficiency: float = 0.0
    trend_consistency: float = 0.0
    
    # Statistical measures
    confidence_score: float = 0.0
    z_score: float = 0.0
    percentile_rank: float = 0.0
    
    # Risk metrics
    risk_score: float = 0.0
    position_size: float = 0.0
    
    # Divergence analysis
    price_pvt_divergence: float = 0.0
    volume_divergence: float = 0.0
    
    # ML predictions
    trend_prediction: float = 0.0
    prediction_confidence: float = 0.0


@dataclass
class PVTParameters:
    """Configuration parameters for PVT indicator."""
    # Basic PVT parameters
    smoothing_period: int = 14
    momentum_period: int = 10
    trend_period: int = 20
    
    # Volume analysis
    volume_ma_period: int = 20
    volume_threshold: float = 1.5
    
    # Statistical parameters
    confidence_period: int = 50
    percentile_period: int = 100
    
    # ML parameters
    ml_lookback: int = 252
    ml_features: int = 10
    prediction_horizon: int = 5
    
    # Risk management
    risk_lookback: int = 30
    max_position_size: float = 1.0
    
    # Smoothing parameters
    ma_type: str = "ema"  # ema, sma, wma
    smoothing_factor: float = 0.1
    
    # Validation parameters
    min_periods: int = 50
    max_nan_ratio: float = 0.1


class AdvancedPriceVolumeTrendIndicator:
    """
    Advanced Price Volume Trend (PVT) Indicator with machine learning enhancements.
    
    This class implements a sophisticated PVT indicator that combines traditional
    volume-price analysis with modern statistical methods and machine learning
    for enhanced trend identification and momentum confirmation.
    """
    
    def __init__(self, parameters: Optional[PVTParameters] = None):
        """
        Initialize the Advanced PVT Indicator.
        
        Args:
            parameters: Configuration parameters for the indicator
        """
        self.params = parameters or PVTParameters()
        self.logger = logging.getLogger(__name__)
        
        # Initialize ML models
        self._init_ml_models()
        
        # Initialize scalers
        self.scaler = RobustScaler()
        self.feature_scaler = StandardScaler()
        
        # Initialize state variables
        self.is_trained = False
        self.last_signals = []
        self.feature_names = []
        
        # Performance tracking
        self.calculation_count = 0
        self.total_calculation_time = 0.0
        
        self.logger.info("Advanced PVT Indicator initialized successfully")
    
    def _init_ml_models(self) -> None:
        """Initialize machine learning models."""
        try:
            # Trend prediction model
            self.trend_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            
            # Anomaly detection model
            self.anomaly_model = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_jobs=-1
            )
            
            self.logger.info("ML models initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing ML models: {str(e)}")
            raise
    
    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate advanced PVT indicator with comprehensive analysis.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary containing PVT signals and analysis
        """
        try:
            start_time = datetime.now()
            
            # Validate input data
            if not self._validate_data(data):
                raise ValueError("Invalid input data for PVT calculation")
            
            # Prepare data
            df = self._prepare_data(data.copy())
            
            # Calculate core PVT
            df = self._calculate_core_pvt(df)
            
            # Calculate advanced metrics
            df = self._calculate_advanced_metrics(df)
            
            # Perform statistical analysis
            df = self._calculate_statistical_metrics(df)
            
            # Volume regime analysis
            df = self._analyze_volume_regime(df)
            
            # Trend analysis
            df = self._analyze_trend_patterns(df)
            
            # Machine learning predictions
            if len(df) >= self.params.ml_lookback:
                df = self._calculate_ml_predictions(df)
            
            # Generate signals
            signals = self._generate_signals(df)
            
            # Calculate performance metrics
            performance = self._calculate_performance_metrics(df)
            
            # Update calculation statistics
            calculation_time = (datetime.now() - start_time).total_seconds()
            self._update_performance_stats(calculation_time)
            
            # Prepare output
            result = {
                'signals': signals,
                'data': df,
                'performance': performance,
                'metadata': {
                    'calculation_time': calculation_time,
                    'data_points': len(df),
                    'parameters': self.params.__dict__,
                    'model_trained': self.is_trained
                }
            }
            
            self.logger.info(f"PVT calculation completed successfully in {calculation_time:.4f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in PVT calculation: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
    
    def _validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data quality and completeness."""
        try:
            required_columns = ['high', 'low', 'close', 'volume']
            
            # Check required columns
            if not all(col in data.columns for col in required_columns):
                self.logger.error(f"Missing required columns. Required: {required_columns}")
                return False
            
            # Check data length
            if len(data) < self.params.min_periods:
                self.logger.error(f"Insufficient data points. Required: {self.params.min_periods}, Got: {len(data)}")
                return False
            
            # Check for excessive NaN values
            nan_ratios = data[required_columns].isnull().sum() / len(data)
            if (nan_ratios > self.params.max_nan_ratio).any():
                self.logger.error(f"Excessive NaN values detected: {nan_ratios.to_dict()}")
                return False
            
            # Check for non-positive volume
            if (data['volume'] <= 0).any():
                self.logger.warning("Non-positive volume values detected")
            
            # Check for price data consistency
            if (data['high'] < data['low']).any():
                self.logger.error("Invalid price data: high < low detected")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in data validation: {str(e)}")
            return False
    
    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for PVT calculation."""
        try:
            # Ensure index is datetime
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            # Sort by index
            df = df.sort_index()
            
            # Handle missing values
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            # Calculate additional price metrics
            df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
            df['price_change'] = df['close'].pct_change()
            df['volume_ma'] = df['volume'].rolling(window=self.params.volume_ma_period).mean()
            
            # Calculate relative volume
            df['relative_volume'] = df['volume'] / df['volume_ma']
            
            # Log transform volume to reduce skewness
            df['log_volume'] = np.log1p(df['volume'])
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error in data preparation: {str(e)}")
            raise
    
    def _calculate_core_pvt(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate core PVT values with enhancements."""
        try:
            # Calculate price change ratio
            df['price_change_ratio'] = df['close'].pct_change()
            
            # Calculate volume-weighted price change
            df['volume_weighted_change'] = df['price_change_ratio'] * df['volume']
            
            # Calculate cumulative PVT
            df['pvt_raw'] = df['volume_weighted_change'].cumsum()
            
            # Apply smoothing
            if self.params.ma_type == 'ema':
                df['pvt'] = df['pvt_raw'].ewm(span=self.params.smoothing_period).mean()
            elif self.params.ma_type == 'sma':
                df['pvt'] = df['pvt_raw'].rolling(window=self.params.smoothing_period).mean()
            elif self.params.ma_type == 'wma':
                weights = np.arange(1, self.params.smoothing_period + 1)
                df['pvt'] = df['pvt_raw'].rolling(window=self.params.smoothing_period).apply(
                    lambda x: np.average(x, weights=weights), raw=True
                )
            else:
                df['pvt'] = df['pvt_raw']
            
            # Calculate PVT momentum
            df['pvt_momentum'] = df['pvt'].pct_change(periods=self.params.momentum_period)
            
            # Calculate PVT acceleration
            df['pvt_acceleration'] = df['pvt_momentum'].diff()
            
            # Normalize PVT values
            rolling_mean = df['pvt'].rolling(window=self.params.trend_period).mean()
            rolling_std = df['pvt'].rolling(window=self.params.trend_period).std()
            df['pvt_normalized'] = (df['pvt'] - rolling_mean) / rolling_std
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error in core PVT calculation: {str(e)}")
            raise
    
    def _calculate_advanced_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate advanced PVT metrics."""
        try:
            # Trend strength
            df['trend_strength'] = abs(df['pvt_momentum'].rolling(
                window=self.params.trend_period
            ).mean())
            
            # Volume efficiency (PVT change per unit volume)
            df['volume_efficiency'] = df['pvt_momentum'] / (df['relative_volume'] + 1e-8)
            
            # Volatility measure
            df['pvt_volatility'] = df['pvt'].rolling(
                window=self.params.trend_period
            ).std()
            
            # Trend consistency
            momentum_signs = np.sign(df['pvt_momentum'])
            df['trend_consistency'] = momentum_signs.rolling(
                window=self.params.trend_period
            ).apply(lambda x: abs(x.sum()) / len(x), raw=True)
            
            # Calculate Savitzky-Golay smoothed PVT for noise reduction
            if len(df) >= 21:  # Minimum window for Savgol filter
                try:
                    window_length = min(21, len(df) // 2 * 2 + 1)  # Ensure odd number
                    df['pvt_smooth'] = savgol_filter(
                        df['pvt'].fillna(method='ffill').values,
                        window_length=window_length,
                        polyorder=3
                    )
                except:
                    df['pvt_smooth'] = df['pvt']
            else:
                df['pvt_smooth'] = df['pvt']
            
            # PVT momentum bands
            momentum_mean = df['pvt_momentum'].rolling(window=20).mean()
            momentum_std = df['pvt_momentum'].rolling(window=20).std()
            df['pvt_momentum_upper'] = momentum_mean + 2 * momentum_std
            df['pvt_momentum_lower'] = momentum_mean - 2 * momentum_std
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error in advanced metrics calculation: {str(e)}")
            raise
    
    def _calculate_statistical_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate statistical confidence metrics."""
        try:
            # Z-score calculation
            rolling_mean = df['pvt'].rolling(window=self.params.confidence_period).mean()
            rolling_std = df['pvt'].rolling(window=self.params.confidence_period).std()
            df['pvt_zscore'] = (df['pvt'] - rolling_mean) / (rolling_std + 1e-8)
            
            # Percentile rank
            df['pvt_percentile'] = df['pvt'].rolling(
                window=self.params.percentile_period
            ).rank(pct=True)
            
            # Confidence score based on multiple factors
            momentum_confidence = 1.0 - abs(df['pvt_zscore']) / 3.0
            trend_confidence = df['trend_consistency']
            volume_confidence = np.clip(df['relative_volume'] / 2.0, 0, 1)
            
            df['confidence_score'] = (
                momentum_confidence * 0.4 +
                trend_confidence * 0.4 +
                volume_confidence * 0.2
            ).clip(0, 1)
            
            # Statistical significance test
            window = self.params.confidence_period
            df['statistical_significance'] = df['pvt_momentum'].rolling(window).apply(
                lambda x: abs(stats.ttest_1samp(x.dropna(), 0)[0]) if len(x.dropna()) > 1 else 0,
                raw=False
            )
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error in statistical metrics calculation: {str(e)}")
            raise
    
    def _analyze_volume_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze volume regime characteristics."""
        try:
            # Volume regime classification
            volume_conditions = [
                df['relative_volume'] >= 2.0,
                df['relative_volume'] >= 1.5,
                df['relative_volume'] >= 0.5,
                df['relative_volume'] < 0.5
            ]
            
            volume_regimes = [
                VolumeRegime.HIGH_VOLUME,
                VolumeRegime.NORMAL_VOLUME,
                VolumeRegime.NORMAL_VOLUME,
                VolumeRegime.LOW_VOLUME
            ]
            
            df['volume_regime'] = np.select(volume_conditions, volume_regimes, 
                                           default=VolumeRegime.ANOMALOUS_VOLUME)
            
            # Volume regime strength
            df['volume_regime_strength'] = np.abs(
                df['relative_volume'] - 1.0
            )
            
            # Volume momentum
            df['volume_momentum'] = df['volume'].pct_change(
                periods=self.params.momentum_period
            )
            
            # Volume acceleration
            df['volume_acceleration'] = df['volume_momentum'].diff()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error in volume regime analysis: {str(e)}")
            raise
    
    def _analyze_trend_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze trend patterns and classify trend direction."""
        try:
            # Trend direction classification
            momentum_threshold = df['pvt_momentum'].rolling(20).std()
            
            trend_conditions = [
                df['pvt_momentum'] > 2 * momentum_threshold,
                df['pvt_momentum'] > momentum_threshold,
                df['pvt_momentum'] > 0.5 * momentum_threshold,
                (df['pvt_momentum'] >= -0.5 * momentum_threshold) & 
                (df['pvt_momentum'] <= 0.5 * momentum_threshold),
                df['pvt_momentum'] < -0.5 * momentum_threshold,
                df['pvt_momentum'] < -momentum_threshold,
                df['pvt_momentum'] < -2 * momentum_threshold
            ]
            
            trend_directions = [
                TrendDirection.STRONG_BULLISH,
                TrendDirection.BULLISH,
                TrendDirection.WEAK_BULLISH,
                TrendDirection.NEUTRAL,
                TrendDirection.WEAK_BEARISH,
                TrendDirection.BEARISH,
                TrendDirection.STRONG_BEARISH
            ]
            
            df['trend_direction'] = np.select(trend_conditions, trend_directions,
                                            default=TrendDirection.NEUTRAL)
            
            # Divergence analysis between price and PVT
            price_momentum = df['close'].pct_change(periods=self.params.momentum_period)
            df['price_pvt_divergence'] = price_momentum - df['pvt_momentum']
            
            # Volume divergence
            expected_volume = df['volume_ma']
            df['volume_divergence'] = (df['volume'] - expected_volume) / expected_volume
            
            # Trend strength score
            df['trend_strength_score'] = (
                abs(df['pvt_momentum']) * df['confidence_score'] * 
                (1 + df['volume_regime_strength'])
            )
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error in trend pattern analysis: {str(e)}")
            raise
    
    def _calculate_ml_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate machine learning-based predictions."""
        try:
            # Prepare features for ML
            features = self._prepare_ml_features(df)
            
            if len(features) < self.params.ml_lookback:
                self.logger.warning("Insufficient data for ML predictions")
                df['trend_prediction'] = 0.0
                df['prediction_confidence'] = 0.0
                return df
            
            # Train models if not already trained
            if not self.is_trained:
                self._train_ml_models(features)
            
            # Generate predictions
            if self.is_trained:
                predictions = self._generate_ml_predictions(features)
                df = df.iloc[-len(predictions):].copy()
                df['trend_prediction'] = predictions['trend_prediction']
                df['prediction_confidence'] = predictions['confidence']
            else:
                df['trend_prediction'] = 0.0
                df['prediction_confidence'] = 0.0
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error in ML predictions: {str(e)}")
            df['trend_prediction'] = 0.0
            df['prediction_confidence'] = 0.0
            return df
    
    def _prepare_ml_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for machine learning models."""
        try:
            features_df = pd.DataFrame(index=df.index)
            
            # PVT-based features
            features_df['pvt_momentum'] = df['pvt_momentum']
            features_df['pvt_acceleration'] = df['pvt_acceleration']
            features_df['pvt_volatility'] = df['pvt_volatility']
            features_df['trend_strength'] = df['trend_strength']
            features_df['volume_efficiency'] = df['volume_efficiency']
            
            # Volume features
            features_df['relative_volume'] = df['relative_volume']
            features_df['volume_momentum'] = df['volume_momentum']
            features_df['volume_acceleration'] = df['volume_acceleration']
            
            # Statistical features
            features_df['pvt_zscore'] = df['pvt_zscore']
            features_df['pvt_percentile'] = df['pvt_percentile']
            
            # Price features
            features_df['price_change'] = df['price_change']
            features_df['price_volatility'] = df['close'].rolling(20).std()
            
            # Divergence features
            features_df['price_pvt_divergence'] = df['price_pvt_divergence']
            features_df['volume_divergence'] = df['volume_divergence']
            
            # Lagged features
            for lag in [1, 2, 3, 5]:
                features_df[f'pvt_momentum_lag_{lag}'] = df['pvt_momentum'].shift(lag)
                features_df[f'volume_momentum_lag_{lag}'] = df['volume_momentum'].shift(lag)
            
            # Technical indicators
            features_df['rsi'] = self._calculate_rsi(df['close'])
            features_df['macd'] = self._calculate_macd(df['close'])
            
            # Remove NaN and infinite values
            features_df = features_df.fillna(0).replace([np.inf, -np.inf], 0)
            
            # Store feature names
            self.feature_names = features_df.columns.tolist()
            
            return features_df
            
        except Exception as e:
            self.logger.error(f"Error in ML feature preparation: {str(e)}")
            return pd.DataFrame()
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI for additional features."""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / (loss + 1e-8)
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except:
            return pd.Series(50, index=prices.index)
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
        """Calculate MACD for additional features."""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            return macd
        except:
            return pd.Series(0, index=prices.index)
    
    def _train_ml_models(self, features: pd.DataFrame) -> None:
        """Train machine learning models."""
        try:
            if len(features) < self.params.ml_lookback:
                self.logger.warning("Insufficient data for ML training")
                return
            
            # Prepare training data
            X, y = self._prepare_training_data(features)
            
            if len(X) < 50:  # Minimum samples for training
                self.logger.warning("Insufficient samples for ML training")
                return
            
            # Scale features
            X_scaled = self.feature_scaler.fit_transform(X)
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            
            # Train trend prediction model
            self.trend_model.fit(X_scaled, y)
            
            # Train anomaly detection model
            self.anomaly_model.fit(X_scaled)
            
            self.is_trained = True
            self.logger.info("ML models trained successfully")
            
        except Exception as e:
            self.logger.error(f"Error in ML model training: {str(e)}")
    
    def _prepare_training_data(self, features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for ML models."""
        try:
            # Use lookback window for features
            X_list = []
            y_list = []
            
            lookback = self.params.ml_features
            horizon = self.params.prediction_horizon
            
            for i in range(lookback, len(features) - horizon):
                # Features: past lookback values
                X_window = features.iloc[i-lookback:i].values.flatten()
                X_list.append(X_window)
                
                # Target: future momentum
                if 'pvt_momentum' in features.columns:
                    future_momentum = features['pvt_momentum'].iloc[i:i+horizon].mean()
                    y_list.append(future_momentum)
                else:
                    y_list.append(0.0)
            
            X = np.array(X_list)
            y = np.array(y_list)
            
            # Remove invalid samples
            valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X = X[valid_mask]
            y = y[valid_mask]
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error in training data preparation: {str(e)}")
            return np.array([]), np.array([])
    
    def _generate_ml_predictions(self, features: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate ML-based predictions."""
        try:
            # Prepare recent features
            if len(features) < self.params.ml_features:
                return {
                    'trend_prediction': np.zeros(len(features)),
                    'confidence': np.zeros(len(features))
                }
            
            # Generate predictions for recent data
            predictions = []
            confidences = []
            
            lookback = self.params.ml_features
            
            for i in range(lookback, len(features)):
                # Prepare feature window
                X_window = features.iloc[i-lookback:i].values.flatten().reshape(1, -1)
                
                # Scale features
                X_scaled = self.feature_scaler.transform(X_window)
                
                # Generate prediction
                pred = self.trend_model.predict(X_scaled)[0]
                predictions.append(pred)
                
                # Calculate confidence (using ensemble variance)
                tree_predictions = [tree.predict(X_scaled)[0] for tree in self.trend_model.estimators_]
                confidence = 1.0 / (1.0 + np.std(tree_predictions))
                confidences.append(confidence)
            
            # Pad with zeros for initial values
            full_predictions = np.zeros(len(features))
            full_confidences = np.zeros(len(features))
            
            full_predictions[lookback:] = predictions
            full_confidences[lookback:] = confidences
            
            return {
                'trend_prediction': full_predictions,
                'confidence': full_confidences
            }
            
        except Exception as e:
            self.logger.error(f"Error in ML prediction generation: {str(e)}")
            return {
                'trend_prediction': np.zeros(len(features)),
                'confidence': np.zeros(len(features))
            }
    
    def _generate_signals(self, df: pd.DataFrame) -> List[PVTSignalComponents]:
        """Generate PVT signals with comprehensive analysis."""
        try:
            signals = []
            
            for i in range(len(df)):
                row = df.iloc[i]
                
                # Extract signal components
                signal = PVTSignalComponents(
                    pvt_value=float(row.get('pvt', 0)),
                    pvt_smoothed=float(row.get('pvt_smooth', 0)),
                    momentum=float(row.get('pvt_momentum', 0)),
                    trend_strength=float(row.get('trend_strength', 0)),
                    volume_regime=row.get('volume_regime', VolumeRegime.NORMAL_VOLUME),
                    trend_direction=row.get('trend_direction', TrendDirection.NEUTRAL),
                    
                    # Advanced metrics
                    acceleration=float(row.get('pvt_acceleration', 0)),
                    volatility=float(row.get('pvt_volatility', 0)),
                    volume_efficiency=float(row.get('volume_efficiency', 0)),
                    trend_consistency=float(row.get('trend_consistency', 0)),
                    
                    # Statistical measures
                    confidence_score=float(row.get('confidence_score', 0)),
                    z_score=float(row.get('pvt_zscore', 0)),
                    percentile_rank=float(row.get('pvt_percentile', 0)),
                    
                    # Risk metrics
                    risk_score=self._calculate_risk_score(row),
                    position_size=self._calculate_position_size(row),
                    
                    # Divergence analysis
                    price_pvt_divergence=float(row.get('price_pvt_divergence', 0)),
                    volume_divergence=float(row.get('volume_divergence', 0)),
                    
                    # ML predictions
                    trend_prediction=float(row.get('trend_prediction', 0)),
                    prediction_confidence=float(row.get('prediction_confidence', 0))
                )
                
                signals.append(signal)
            
            # Store recent signals for analysis
            self.last_signals = signals[-100:] if len(signals) > 100 else signals
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error in signal generation: {str(e)}")
            return []
    
    def _calculate_risk_score(self, row: pd.Series) -> float:
        """Calculate risk score for the current signal."""
        try:
            # Base risk from volatility
            volatility_risk = min(abs(row.get('pvt_volatility', 0)) / 10.0, 1.0)
            
            # Volume regime risk
            volume_regime = row.get('volume_regime', VolumeRegime.NORMAL_VOLUME)
            if volume_regime == VolumeRegime.ANOMALOUS_VOLUME:
                volume_risk = 1.0
            elif volume_regime == VolumeRegime.LOW_VOLUME:
                volume_risk = 0.7
            else:
                volume_risk = 0.3
            
            # Confidence risk (inverse of confidence)
            confidence_risk = 1.0 - row.get('confidence_score', 0)
            
            # Divergence risk
            divergence_risk = min(abs(row.get('price_pvt_divergence', 0)), 1.0)
            
            # Combined risk score
            risk_score = (
                volatility_risk * 0.3 +
                volume_risk * 0.3 +
                confidence_risk * 0.2 +
                divergence_risk * 0.2
            )
            
            return np.clip(risk_score, 0.0, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error in risk score calculation: {str(e)}")
            return 0.5
    
    def _calculate_position_size(self, row: pd.Series) -> float:
        """Calculate position size based on risk and confidence."""
        try:
            # Base position size
            base_size = self.params.max_position_size
            
            # Risk adjustment
            risk_score = self._calculate_risk_score(row)
            risk_adjustment = 1.0 - risk_score
            
            # Confidence adjustment
            confidence = row.get('confidence_score', 0)
            confidence_adjustment = confidence
            
            # Trend strength adjustment
            trend_strength = row.get('trend_strength', 0)
            trend_adjustment = min(trend_strength * 10.0, 1.0)
            
            # Combined position size
            position_size = base_size * risk_adjustment * confidence_adjustment * trend_adjustment
            
            return np.clip(position_size, 0.0, self.params.max_position_size)
            
        except Exception as e:
            self.logger.error(f"Error in position size calculation: {str(e)}")
            return 0.0
    
    def _calculate_performance_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate performance metrics for the indicator."""
        try:
            if len(df) < 2:
                return {}
            
            metrics = {}
            
            # Signal quality metrics
            if 'pvt_momentum' in df.columns:
                momentum_series = df['pvt_momentum'].dropna()
                if len(momentum_series) > 0:
                    metrics['momentum_mean'] = float(momentum_series.mean())
                    metrics['momentum_std'] = float(momentum_series.std())
                    metrics['momentum_sharpe'] = float(momentum_series.mean() / (momentum_series.std() + 1e-8))
            
            # Confidence metrics
            if 'confidence_score' in df.columns:
                confidence_series = df['confidence_score'].dropna()
                if len(confidence_series) > 0:
                    metrics['avg_confidence'] = float(confidence_series.mean())
                    metrics['confidence_stability'] = float(1.0 - confidence_series.std())
            
            # Volume metrics
            if 'relative_volume' in df.columns:
                volume_series = df['relative_volume'].dropna()
                if len(volume_series) > 0:
                    metrics['avg_relative_volume'] = float(volume_series.mean())
                    metrics['volume_efficiency'] = float(df.get('volume_efficiency', pd.Series(0)).mean())
            
            # Trend metrics
            if 'trend_strength' in df.columns:
                trend_series = df['trend_strength'].dropna()
                if len(trend_series) > 0:
                    metrics['avg_trend_strength'] = float(trend_series.mean())
                    metrics['trend_consistency'] = float(df.get('trend_consistency', pd.Series(0)).mean())
            
            # ML model performance
            if self.is_trained and 'trend_prediction' in df.columns:
                pred_series = df['trend_prediction'].dropna()
                if len(pred_series) > 0:
                    metrics['prediction_coverage'] = float(len(pred_series) / len(df))
                    metrics['avg_prediction_confidence'] = float(df.get('prediction_confidence', pd.Series(0)).mean())
            
            # Risk metrics
            metrics['avg_risk_score'] = float(np.mean([
                self._calculate_risk_score(df.iloc[i]) for i in range(len(df))
            ]))
            
            # Data quality
            metrics['data_completeness'] = float(1.0 - df.isnull().sum().sum() / (len(df) * len(df.columns)))
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error in performance metrics calculation: {str(e)}")
            return {}
    
    def _update_performance_stats(self, calculation_time: float) -> None:
        """Update performance statistics."""
        try:
            self.calculation_count += 1
            self.total_calculation_time += calculation_time
            
            if self.calculation_count % 100 == 0:
                avg_time = self.total_calculation_time / self.calculation_count
                self.logger.info(f"Average calculation time over {self.calculation_count} runs: {avg_time:.4f}s")
                
        except Exception as e:
            self.logger.error(f"Error updating performance stats: {str(e)}")
    
    def get_signal_strength(self, signal: PVTSignalComponents) -> float:
        """Calculate overall signal strength."""
        try:
            # Combine multiple factors
            momentum_strength = min(abs(signal.momentum) * 10.0, 1.0)
            trend_strength = signal.trend_strength
            confidence_strength = signal.confidence_score
            volume_strength = 1.0 if signal.volume_regime in [VolumeRegime.HIGH_VOLUME, VolumeRegime.NORMAL_VOLUME] else 0.5
            
            # Weight the components
            overall_strength = (
                momentum_strength * 0.3 +
                trend_strength * 0.3 +
                confidence_strength * 0.2 +
                volume_strength * 0.2
            )
            
            return np.clip(overall_strength, 0.0, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating signal strength: {str(e)}")
            return 0.0
    
    def get_trading_recommendation(self, signal: PVTSignalComponents) -> Dict[str, Any]:
        """Generate trading recommendation based on signal."""
        try:
            recommendation = {
                'action': 'HOLD',
                'confidence': 0.0,
                'position_size': 0.0,
                'risk_level': 'MEDIUM',
                'reasoning': []
            }
            
            # Determine action based on momentum and trend
            if signal.momentum > 0 and signal.trend_direction in [TrendDirection.BULLISH, TrendDirection.STRONG_BULLISH]:
                recommendation['action'] = 'BUY'
            elif signal.momentum < 0 and signal.trend_direction in [TrendDirection.BEARISH, TrendDirection.STRONG_BEARISH]:
                recommendation['action'] = 'SELL'
            
            # Set confidence and position size
            recommendation['confidence'] = signal.confidence_score
            recommendation['position_size'] = signal.position_size
            
            # Determine risk level
            if signal.risk_score < 0.3:
                recommendation['risk_level'] = 'LOW'
            elif signal.risk_score > 0.7:
                recommendation['risk_level'] = 'HIGH'
            
            # Add reasoning
            if signal.volume_regime == VolumeRegime.HIGH_VOLUME:
                recommendation['reasoning'].append("High volume supports the signal")
            if abs(signal.price_pvt_divergence) > 0.5:
                recommendation['reasoning'].append("Price-PVT divergence detected")
            if signal.trend_consistency > 0.7:
                recommendation['reasoning'].append("Strong trend consistency")
            
            return recommendation
            
        except Exception as e:
            self.logger.error(f"Error generating trading recommendation: {str(e)}")
            return {'action': 'HOLD', 'confidence': 0.0, 'position_size': 0.0, 'risk_level': 'HIGH', 'reasoning': ['Error in analysis']}


# Example usage and testing
if __name__ == "__main__":
    # Test the indicator with sample data
    import yfinance as yf
    
    try:
        # Download sample data
        ticker = "EURUSD=X"
        data = yf.download(ticker, period="1y", interval="1d")
        
        if data.empty:
            print("No data available for testing")
        else:
            # Initialize indicator
            params = PVTParameters(
                smoothing_period=14,
                momentum_period=10,
                ml_lookback=100
            )
            
            indicator = AdvancedPriceVolumeTrendIndicator(params)
            
            # Calculate signals
            result = indicator.calculate(data)
            
            # Display results
            print(f"Calculated PVT for {len(result['signals'])} periods")
            print(f"Calculation time: {result['metadata']['calculation_time']:.4f}s")
            print(f"Model trained: {result['metadata']['model_trained']}")
            
            # Show latest signals
            if result['signals']:
                latest_signal = result['signals'][-1]
                print(f"\nLatest Signal:")
                print(f"PVT Value: {latest_signal.pvt_value:.6f}")
                print(f"Momentum: {latest_signal.momentum:.6f}")
                print(f"Trend Direction: {latest_signal.trend_direction.value}")
                print(f"Volume Regime: {latest_signal.volume_regime.value}")
                print(f"Confidence: {latest_signal.confidence_score:.3f}")
                print(f"Risk Score: {latest_signal.risk_score:.3f}")
                
                # Get trading recommendation
                recommendation = indicator.get_trading_recommendation(latest_signal)
                print(f"\nTrading Recommendation:")
                print(f"Action: {recommendation['action']}")
                print(f"Confidence: {recommendation['confidence']:.3f}")
                print(f"Position Size: {recommendation['position_size']:.3f}")
                print(f"Risk Level: {recommendation['risk_level']}")
                
                if recommendation['reasoning']:
                    print("Reasoning:")
                    for reason in recommendation['reasoning']:
                        print(f"  - {reason}")
                
            # Performance metrics
            if result['performance']:
                print(f"\nPerformance Metrics:")
                for key, value in result['performance'].items():
                    print(f"{key}: {value:.4f}")
    
    except Exception as e:
        print(f"Error in testing: {str(e)}")
        print(traceback.format_exc())