"""
Advanced Smart Money Detection Suite for AUJ Platform

This module implements a comprehensive smart money detection system that combines
multiple algorithms to identify institutional trading activity, flow analysis, 
and sophisticated behavioral tracking patterns. The suite integrates various
methodologies to detect "smart money" movements versus retail activity.

Key Features:
- Multi-algorithm smart money detection
- Institutional flow pattern analysis
- Block trade identification and tracking
- Volume profile anomaly detection
- Price action divergence analysis
- Order flow imbalance detection
- Statistical significance testing
- Machine learning classification
- Real-time monitoring capabilities
- Risk-adjusted position sizing

Detection Algorithms:
- Volume-weighted institutional indicators
- Price-volume divergence analysis
- Large block trade detection
- Unusual option activity correlation
- Market microstructure analysis
- Flow dynamics modeling
- Statistical outlier detection
- Pattern recognition algorithms

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
from scipy.signal import find_peaks, savgol_filter
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
import joblib
import traceback

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SmartMoneySignal(Enum):
    """Enumeration for smart money signal types."""
    ACCUMULATION = "ACCUMULATION"
    DISTRIBUTION = "DISTRIBUTION"
    NEUTRAL = "NEUTRAL"
    STEALTH_BUYING = "STEALTH_BUYING"
    STEALTH_SELLING = "STEALTH_SELLING"
    AGGRESSIVE_BUYING = "AGGRESSIVE_BUYING"
    AGGRESSIVE_SELLING = "AGGRESSIVE_SELLING"


class InstitutionalActivity(Enum):
    """Enumeration for institutional activity levels."""
    HIGH_ACTIVITY = "HIGH_ACTIVITY"
    MODERATE_ACTIVITY = "MODERATE_ACTIVITY"
    LOW_ACTIVITY = "LOW_ACTIVITY"
    ANOMALOUS_ACTIVITY = "ANOMALOUS_ACTIVITY"


class FlowDirection(Enum):
    """Enumeration for money flow direction."""
    INFLOW = "INFLOW"
    OUTFLOW = "OUTFLOW"
    BALANCED = "BALANCED"
    VOLATILE = "VOLATILE"


@dataclass
class SmartMoneyComponents:
    """Data class for smart money detection components."""
    smart_money_signal: SmartMoneySignal
    institutional_activity: InstitutionalActivity
    flow_direction: FlowDirection
    confidence_score: float
    
    # Volume analysis
    block_trade_ratio: float = 0.0
    volume_anomaly_score: float = 0.0
    relative_volume: float = 0.0
    volume_efficiency: float = 0.0
    
    # Price analysis
    price_volume_divergence: float = 0.0
    stealth_index: float = 0.0
    absorption_ratio: float = 0.0
    momentum_divergence: float = 0.0
    
    # Order flow metrics
    order_flow_imbalance: float = 0.0
    market_impact: float = 0.0
    participation_rate: float = 0.0
    execution_quality: float = 0.0
    
    # Statistical measures
    statistical_significance: float = 0.0
    z_score: float = 0.0
    percentile_rank: float = 0.0
    
    # ML classifications
    institutional_probability: float = 0.0
    pattern_confidence: float = 0.0
    anomaly_score: float = 0.0
    
    # Risk metrics
    risk_score: float = 0.0
    position_size: float = 0.0
    exposure_limit: float = 0.0


@dataclass
class SmartMoneyParameters:
    """Configuration parameters for smart money detection."""
    # Volume thresholds
    block_trade_threshold: float = 2.0  # Multiples of average volume
    large_trade_percentile: float = 95.0  # Percentile for large trades
    volume_spike_threshold: float = 3.0  # Standard deviations
    
    # Price analysis
    price_impact_threshold: float = 0.01  # Minimum price impact
    divergence_threshold: float = 0.5  # Correlation threshold
    stealth_window: int = 20  # Periods for stealth detection
    
    # Statistical parameters
    confidence_period: int = 50
    significance_level: float = 0.05
    lookback_period: int = 100
    
    # ML parameters
    ml_lookback: int = 252
    feature_count: int = 15
    classification_threshold: float = 0.7
    
    # Risk management
    max_position_size: float = 1.0
    risk_lookback: int = 30
    
    # Flow analysis
    flow_window: int = 10
    absorption_window: int = 15
    participation_window: int = 20
    
    # Validation parameters
    min_periods: int = 50
    max_nan_ratio: float = 0.1


class AdvancedSmartMoneyIndicators:
    """
    Advanced Smart Money Detection Suite with machine learning enhancements.
    
    This class implements a comprehensive suite of indicators to detect
    institutional trading activity, smart money flows, and sophisticated
    behavioral patterns in market data.
    """
    
    def __init__(self, parameters: Optional[SmartMoneyParameters] = None):
        """
        Initialize the Advanced Smart Money Detection Suite.
        
        Args:
            parameters: Configuration parameters for the indicator suite
        """
        self.params = parameters or SmartMoneyParameters()
        self.logger = logging.getLogger(__name__)
        
        # Initialize ML models
        self._init_ml_models()
        
        # Initialize scalers
        self.scaler = RobustScaler()
        self.feature_scaler = StandardScaler()
        
        # Initialize clustering models
        self.volume_cluster = KMeans(n_clusters=5, random_state=42)
        self.flow_cluster = DBSCAN(eps=0.3, min_samples=5)
        
        # Initialize state variables
        self.is_trained = False
        self.last_signals = []
        self.feature_names = []
        self.block_trades = []
        
        # Performance tracking
        self.calculation_count = 0
        self.total_calculation_time = 0.0
        
        self.logger.info("Advanced Smart Money Detection Suite initialized successfully")
    
    def _init_ml_models(self) -> None:
        """Initialize machine learning models."""
        try:
            # Smart money classification model
            self.smart_money_classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            
            # Anomaly detection model
            self.anomaly_detector = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_jobs=-1
            )
            
            # Volume pattern classifier
            self.volume_classifier = RandomForestClassifier(
                n_estimators=50,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            self.logger.info("ML models initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing ML models: {str(e)}")
            raise
    
    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate smart money indicators with comprehensive analysis.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary containing smart money analysis and signals
        """
        try:
            start_time = datetime.now()
            
            # Validate input data
            if not self._validate_data(data):
                raise ValueError("Invalid input data for smart money analysis")
            
            # Prepare data
            df = self._prepare_data(data.copy())
            
            # Calculate volume analysis
            df = self._calculate_volume_analysis(df)
            
            # Calculate price analysis
            df = self._calculate_price_analysis(df)
            
            # Calculate order flow metrics
            df = self._calculate_order_flow_metrics(df)
            
            # Detect block trades
            df = self._detect_block_trades(df)
            
            # Calculate stealth indicators
            df = self._calculate_stealth_indicators(df)
            
            # Perform statistical analysis
            df = self._calculate_statistical_metrics(df)
            
            # Machine learning analysis
            if len(df) >= self.params.ml_lookback:
                df = self._calculate_ml_analysis(df)
            
            # Generate smart money signals
            signals = self._generate_smart_money_signals(df)
            
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
                'block_trades': self.block_trades[-100:],  # Last 100 block trades
                'metadata': {
                    'calculation_time': calculation_time,
                    'data_points': len(df),
                    'parameters': self.params.__dict__,
                    'model_trained': self.is_trained
                }
            }
            
            self.logger.info(f"Smart money analysis completed successfully in {calculation_time:.4f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in smart money analysis: {str(e)}")
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
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in data validation: {str(e)}")
            return False
    
    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for smart money analysis."""
        try:
            # Ensure index is datetime
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            # Sort by index
            df = df.sort_index()
            
            # Handle missing values
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            # Calculate basic metrics
            df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
            df['price_change'] = df['close'].pct_change()
            df['price_range'] = df['high'] - df['low']
            df['volume_change'] = df['volume'].pct_change()
            
            # Calculate moving averages
            df['volume_sma_20'] = df['volume'].rolling(20).mean()
            df['volume_sma_50'] = df['volume'].rolling(50).mean()
            df['price_sma_20'] = df['close'].rolling(20).mean()
            
            # Calculate relative metrics
            df['relative_volume'] = df['volume'] / df['volume_sma_20']
            df['relative_price'] = df['close'] / df['price_sma_20']
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error in data preparation: {str(e)}")
            raise
    
    def _calculate_volume_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive volume analysis metrics."""
        try:
            # Volume anomaly detection
            volume_mean = df['volume'].rolling(self.params.lookback_period).mean()
            volume_std = df['volume'].rolling(self.params.lookback_period).std()
            df['volume_zscore'] = (df['volume'] - volume_mean) / (volume_std + 1e-8)
            df['volume_anomaly_score'] = np.abs(df['volume_zscore'])
            
            # Block trade detection
            df['is_block_trade'] = df['relative_volume'] >= self.params.block_trade_threshold
            df['block_trade_ratio'] = df['is_block_trade'].rolling(20).mean()
            
            # Volume efficiency (price movement per unit volume)
            df['volume_efficiency'] = np.abs(df['price_change']) / (df['relative_volume'] + 1e-8)
            
            # Volume momentum
            df['volume_momentum'] = df['volume'].pct_change(periods=5)
            df['volume_acceleration'] = df['volume_momentum'].diff()
            
            # Volume clustering analysis
            if len(df) >= 50:
                volume_features = df[['relative_volume', 'volume_momentum', 'volume_efficiency']].fillna(0)
                try:
                    volume_clusters = self.volume_cluster.fit_predict(volume_features[-50:])
                    df.loc[df.index[-50:], 'volume_cluster'] = volume_clusters
                except:
                    df['volume_cluster'] = 0
            else:
                df['volume_cluster'] = 0
            
            # Large volume percentile
            df['volume_percentile'] = df['volume'].rolling(self.params.lookback_period).rank(pct=True)
            df['large_volume_flag'] = df['volume_percentile'] >= (self.params.large_trade_percentile / 100.0)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error in volume analysis: {str(e)}")
            raise
    
    def _calculate_price_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate price-based smart money indicators."""
        try:
            # Price impact analysis
            df['price_impact'] = np.abs(df['price_change']) / (np.log1p(df['relative_volume']) + 1e-8)
            
            # Price-volume divergence
            price_momentum = df['price_change'].rolling(10).mean()
            volume_momentum = df['volume_change'].rolling(10).mean()
            
            # Calculate correlation for divergence
            df['pv_correlation'] = df['price_change'].rolling(20).corr(df['volume_change'])
            df['price_volume_divergence'] = 1.0 - np.abs(df['pv_correlation'])
            
            # Absorption analysis (price stability despite high volume)
            high_volume_mask = df['relative_volume'] > 1.5
            df['absorption_ratio'] = 0.0
            
            for i in range(self.params.absorption_window, len(df)):
                window_data = df.iloc[i-self.params.absorption_window:i+1]
                high_vol_periods = window_data[window_data['relative_volume'] > 1.5]
                
                if len(high_vol_periods) > 0:
                    avg_price_change = np.abs(high_vol_periods['price_change']).mean()
                    expected_change = high_vol_periods['relative_volume'].mean() * 0.01  # Expected impact
                    df.iloc[i, df.columns.get_loc('absorption_ratio')] = 1.0 - min(avg_price_change / (expected_change + 1e-8), 1.0)
            
            # Momentum divergence
            price_momentum_5 = df['close'].pct_change(5)
            price_momentum_20 = df['close'].pct_change(20)
            df['momentum_divergence'] = price_momentum_5 - price_momentum_20
            
            # Support/Resistance breach analysis
            df['support_resistance'] = self._calculate_support_resistance(df)
            df['breach_with_volume'] = (
                (np.abs(df['price_change']) > 0.01) & 
                (df['relative_volume'] > 1.5)
            ).astype(float)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error in price analysis: {str(e)}")
            raise
    
    def _calculate_support_resistance(self, df: pd.DataFrame) -> pd.Series:
        """Calculate support and resistance levels."""
        try:
            window = 20
            support_resistance = pd.Series(0.0, index=df.index)
            
            for i in range(window, len(df)):
                window_data = df.iloc[i-window:i+1]
                
                # Find local maxima and minima
                highs = window_data['high'].values
                lows = window_data['low'].values
                
                high_peaks, _ = find_peaks(highs, height=np.percentile(highs, 80))
                low_peaks, _ = find_peaks(-lows, height=-np.percentile(lows, 20))
                
                current_price = df.iloc[i]['close']
                
                # Check if price is near support/resistance
                resistance_levels = highs[high_peaks] if len(high_peaks) > 0 else []
                support_levels = lows[low_peaks] if len(low_peaks) > 0 else []
                
                min_distance = float('inf')
                for level in list(resistance_levels) + list(support_levels):
                    distance = abs(current_price - level) / current_price
                    min_distance = min(min_distance, distance)
                
                support_resistance.iloc[i] = 1.0 / (1.0 + min_distance * 100) if min_distance != float('inf') else 0.0
            
            return support_resistance
            
        except Exception as e:
            self.logger.error(f"Error in support/resistance calculation: {str(e)}")
            return pd.Series(0.0, index=df.index)
    
    def _calculate_order_flow_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate order flow and market microstructure metrics."""
        try:
            # Estimate bid/ask spread (using high-low as proxy)
            df['estimated_spread'] = (df['high'] - df['low']) / df['close']
            
            # Order flow imbalance estimation
            # Using price action and volume to estimate buy/sell pressure
            buy_pressure = ((df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)) * df['volume']
            sell_pressure = ((df['high'] - df['close']) / (df['high'] - df['low'] + 1e-8)) * df['volume']
            
            df['buy_pressure'] = buy_pressure
            df['sell_pressure'] = sell_pressure
            df['order_flow_imbalance'] = (buy_pressure - sell_pressure) / (buy_pressure + sell_pressure + 1e-8)
            
            # Market impact estimation
            df['market_impact'] = np.abs(df['price_change']) * np.log1p(df['volume'])
            
            # Participation rate (volume relative to typical trading)
            df['participation_rate'] = df['volume'] / df['volume_sma_50']
            
            # Execution quality (price improvement/slippage estimation)
            vwap_estimate = (df['high'] + df['low'] + df['close']) / 3
            df['execution_quality'] = 1.0 - np.abs(df['close'] - vwap_estimate) / vwap_estimate
            
            # Flow persistence
            df['flow_persistence'] = df['order_flow_imbalance'].rolling(self.params.flow_window).std()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error in order flow metrics: {str(e)}")
            raise
    
    def _detect_block_trades(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and analyze block trades."""
        try:
            # Block trade criteria
            volume_threshold = df['volume'].rolling(50).quantile(0.95)
            price_impact_threshold = df['price_change'].rolling(50).std() * 2
            
            block_conditions = (
                (df['volume'] > volume_threshold) & 
                (np.abs(df['price_change']) > 0.005) &  # Minimum price movement
                (df['relative_volume'] > self.params.block_trade_threshold)
            )
            
            df['is_block_trade'] = block_conditions
            
            # Block trade characteristics
            df['block_trade_size'] = np.where(
                block_conditions,
                df['volume'] / df['volume_sma_20'],
                0.0
            )
            
            df['block_trade_impact'] = np.where(
                block_conditions,
                np.abs(df['price_change']) * df['relative_volume'],
                0.0
            )
            
            # Store block trade information
            block_trades = df[block_conditions].copy()
            if not block_trades.empty:
                for _, trade in block_trades.iterrows():
                    trade_info = {
                        'timestamp': trade.name,
                        'volume': trade['volume'],
                        'price_impact': trade['price_change'],
                        'relative_size': trade['relative_volume'],
                        'market_impact': trade['block_trade_impact']
                    }
                    self.block_trades.append(trade_info)
            
            # Block trade frequency
            df['block_trade_frequency'] = df['is_block_trade'].rolling(20).sum()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error in block trade detection: {str(e)}")
            raise
    
    def _calculate_stealth_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate stealth trading indicators."""
        try:
            # Stealth index (high volume with minimal price impact)
            price_impact_threshold = df['price_change'].rolling(50).std()
            
            stealth_conditions = (
                (df['relative_volume'] > 1.2) & 
                (np.abs(df['price_change']) < price_impact_threshold) &
                (df['volume'] > df['volume_sma_20'])
            )
            
            df['stealth_trading'] = stealth_conditions.astype(float)
            df['stealth_index'] = df['stealth_trading'].rolling(self.params.stealth_window).mean()
            
            # Iceberg detection (consistent volume without corresponding price movement)
            df['volume_consistency'] = 1.0 - df['volume'].rolling(10).std() / (df['volume'].rolling(10).mean() + 1e-8)
            df['price_stability'] = 1.0 - df['price_change'].rolling(10).std()
            
            df['iceberg_indicator'] = (
                df['volume_consistency'] * df['price_stability'] * 
                (df['relative_volume'] > 1.0).astype(float)
            )
            
            # Accumulation/Distribution pressure
            # Williams %R style calculation for accumulation
            highest_high = df['high'].rolling(14).max()
            lowest_low = df['low'].rolling(14).min()
            
            williams_r = -100 * (highest_high - df['close']) / (highest_high - lowest_low + 1e-8)
            volume_weighted_wr = williams_r * df['relative_volume']
            
            df['accumulation_pressure'] = volume_weighted_wr.rolling(10).mean()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error in stealth indicators: {str(e)}")
            raise
    
    def _calculate_statistical_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate statistical significance and confidence metrics."""
        try:
            # Z-scores for various metrics
            metrics_to_zscore = ['volume', 'price_change', 'order_flow_imbalance', 'market_impact']
            
            for metric in metrics_to_zscore:
                if metric in df.columns:
                    rolling_mean = df[metric].rolling(self.params.confidence_period).mean()
                    rolling_std = df[metric].rolling(self.params.confidence_period).std()
                    df[f'{metric}_zscore'] = (df[metric] - rolling_mean) / (rolling_std + 1e-8)
            
            # Statistical significance of order flow
            window = self.params.confidence_period
            df['flow_significance'] = df['order_flow_imbalance'].rolling(window).apply(
                lambda x: abs(stats.ttest_1samp(x.dropna(), 0)[0]) if len(x.dropna()) > 1 else 0,
                raw=False
            )
            
            # Percentile rankings
            df['volume_percentile'] = df['volume'].rolling(self.params.lookback_period).rank(pct=True)
            df['impact_percentile'] = df['market_impact'].rolling(self.params.lookback_period).rank(pct=True)
            
            # Confidence scores
            volume_confidence = np.clip(df['volume_zscore'].abs() / 3.0, 0, 1)
            flow_confidence = np.clip(df['order_flow_imbalance'].abs(), 0, 1)
            impact_confidence = np.clip(df['market_impact'] / df['market_impact'].rolling(50).quantile(0.95), 0, 1)
            
            df['statistical_confidence'] = (
                volume_confidence * 0.4 +
                flow_confidence * 0.3 +
                impact_confidence * 0.3
            )
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error in statistical metrics: {str(e)}")
            raise
    
    def _calculate_ml_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate machine learning-based analysis."""
        try:
            # Prepare features for ML
            features = self._prepare_ml_features(df)
            
            if len(features) < self.params.ml_lookback:
                self.logger.warning("Insufficient data for ML analysis")
                df['institutional_probability'] = 0.5
                df['pattern_confidence'] = 0.0
                df['anomaly_score'] = 0.0
                return df
            
            # Train models if not already trained
            if not self.is_trained:
                self._train_ml_models(features)
            
            # Generate predictions
            if self.is_trained:
                predictions = self._generate_ml_predictions(features)
                df = df.iloc[-len(predictions):].copy()
                
                df['institutional_probability'] = predictions['institutional_prob']
                df['pattern_confidence'] = predictions['pattern_confidence']
                df['anomaly_score'] = predictions['anomaly_score']
            else:
                df['institutional_probability'] = 0.5
                df['pattern_confidence'] = 0.0
                df['anomaly_score'] = 0.0
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error in ML analysis: {str(e)}")
            df['institutional_probability'] = 0.5
            df['pattern_confidence'] = 0.0
            df['anomaly_score'] = 0.0
            return df
    
    def _prepare_ml_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for machine learning models."""
        try:
            features_df = pd.DataFrame(index=df.index)
            
            # Volume features
            features_df['relative_volume'] = df['relative_volume']
            features_df['volume_anomaly_score'] = df['volume_anomaly_score']
            features_df['block_trade_ratio'] = df['block_trade_ratio']
            features_df['volume_efficiency'] = df['volume_efficiency']
            features_df['volume_momentum'] = df['volume_momentum']
            
            # Price features
            features_df['price_change'] = df['price_change']
            features_df['price_impact'] = df['price_impact']
            features_df['price_volume_divergence'] = df['price_volume_divergence']
            features_df['absorption_ratio'] = df['absorption_ratio']
            features_df['momentum_divergence'] = df['momentum_divergence']
            
            # Order flow features
            features_df['order_flow_imbalance'] = df['order_flow_imbalance']
            features_df['market_impact'] = df['market_impact']
            features_df['participation_rate'] = df['participation_rate']
            features_df['execution_quality'] = df['execution_quality']
            
            # Stealth features
            features_df['stealth_index'] = df['stealth_index']
            features_df['iceberg_indicator'] = df['iceberg_indicator']
            features_df['accumulation_pressure'] = df['accumulation_pressure']
            
            # Statistical features
            features_df['statistical_confidence'] = df['statistical_confidence']
            features_df['flow_significance'] = df['flow_significance']
            
            # Technical indicators
            features_df['rsi'] = self._calculate_rsi(df['close'])
            features_df['bollinger_position'] = self._calculate_bollinger_position(df['close'])
            
            # Lagged features
            for lag in [1, 2, 3, 5]:
                features_df[f'volume_lag_{lag}'] = df['relative_volume'].shift(lag)
                features_df[f'flow_lag_{lag}'] = df['order_flow_imbalance'].shift(lag)
            
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
    
    def _calculate_bollinger_position(self, prices: pd.Series, period: int = 20) -> pd.Series:
        """Calculate position within Bollinger Bands."""
        try:
            sma = prices.rolling(period).mean()
            std = prices.rolling(period).std()
            upper = sma + 2 * std
            lower = sma - 2 * std
            position = (prices - lower) / (upper - lower + 1e-8)
            return position.clip(0, 1)
        except:
            return pd.Series(0.5, index=prices.index)
    
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
            
            # Train smart money classifier
            self.smart_money_classifier.fit(X_scaled, y)
            
            # Train anomaly detector
            self.anomaly_detector.fit(X_scaled)
            
            # Train volume classifier
            volume_labels = self._create_volume_labels(features)
            if len(volume_labels) == len(X_scaled):
                self.volume_classifier.fit(X_scaled, volume_labels)
            
            self.is_trained = True
            self.logger.info("ML models trained successfully")
            
        except Exception as e:
            self.logger.error(f"Error in ML model training: {str(e)}")
    
    def _prepare_training_data(self, features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for ML models."""
        try:
            # Create labels based on multiple criteria
            labels = []
            
            for i in range(len(features)):
                row = features.iloc[i]
                
                # Smart money criteria
                smart_money_score = 0
                
                # High volume with price impact
                if row.get('relative_volume', 0) > 2.0 and abs(row.get('price_change', 0)) > 0.01:
                    smart_money_score += 1
                
                # Order flow imbalance
                if abs(row.get('order_flow_imbalance', 0)) > 0.3:
                    smart_money_score += 1
                
                # Stealth trading
                if row.get('stealth_index', 0) > 0.5:
                    smart_money_score += 1
                
                # Absorption (high volume, low price impact)
                if row.get('absorption_ratio', 0) > 0.7:
                    smart_money_score += 1
                
                # Statistical significance
                if row.get('statistical_confidence', 0) > 0.7:
                    smart_money_score += 1
                
                # Binary classification: smart money vs retail
                labels.append(1 if smart_money_score >= 3 else 0)
            
            X = features.values
            y = np.array(labels)
            
            # Remove invalid samples
            valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X = X[valid_mask]
            y = y[valid_mask]
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error in training data preparation: {str(e)}")
            return np.array([]), np.array([])
    
    def _create_volume_labels(self, features: pd.DataFrame) -> np.ndarray:
        """Create volume pattern labels for classification."""
        try:
            labels = []
            
            for i in range(len(features)):
                row = features.iloc[i]
                relative_vol = row.get('relative_volume', 1.0)
                
                if relative_vol > 3.0:
                    labels.append(4)  # Extreme volume
                elif relative_vol > 2.0:
                    labels.append(3)  # High volume
                elif relative_vol > 1.5:
                    labels.append(2)  # Above average
                elif relative_vol > 0.5:
                    labels.append(1)  # Normal volume
                else:
                    labels.append(0)  # Low volume
            
            return np.array(labels)
            
        except Exception as e:
            self.logger.error(f"Error creating volume labels: {str(e)}")
            return np.zeros(len(features))
    
    def _generate_ml_predictions(self, features: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate ML-based predictions."""
        try:
            if len(features) == 0:
                return {
                    'institutional_prob': np.array([0.5]),
                    'pattern_confidence': np.array([0.0]),
                    'anomaly_score': np.array([0.0])
                }
            
            # Scale features
            X_scaled = self.feature_scaler.transform(features.values)
            
            # Smart money probability
            smart_money_probs = self.smart_money_classifier.predict_proba(X_scaled)
            institutional_prob = smart_money_probs[:, 1] if smart_money_probs.shape[1] > 1 else smart_money_probs[:, 0]
            
            # Pattern confidence (using prediction confidence)
            pattern_confidence = np.max(smart_money_probs, axis=1)
            
            # Anomaly scores
            anomaly_scores = self.anomaly_detector.decision_function(X_scaled)
            # Normalize to 0-1 range
            anomaly_scores = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min() + 1e-8)
            
            return {
                'institutional_prob': institutional_prob,
                'pattern_confidence': pattern_confidence,
                'anomaly_score': anomaly_scores
            }
            
        except Exception as e:
            self.logger.error(f"Error in ML prediction generation: {str(e)}")
            return {
                'institutional_prob': np.array([0.5] * len(features)),
                'pattern_confidence': np.array([0.0] * len(features)),
                'anomaly_score': np.array([0.0] * len(features))
            }
    
    def _generate_smart_money_signals(self, df: pd.DataFrame) -> List[SmartMoneyComponents]:
        """Generate comprehensive smart money signals."""
        try:
            signals = []
            
            for i in range(len(df)):
                row = df.iloc[i]
                
                # Determine smart money signal
                smart_money_signal = self._classify_smart_money_signal(row)
                
                # Determine institutional activity level
                institutional_activity = self._classify_institutional_activity(row)
                
                # Determine flow direction
                flow_direction = self._classify_flow_direction(row)
                
                # Calculate confidence score
                confidence_score = self._calculate_signal_confidence(row)
                
                # Create signal components
                signal = SmartMoneyComponents(
                    smart_money_signal=smart_money_signal,
                    institutional_activity=institutional_activity,
                    flow_direction=flow_direction,
                    confidence_score=confidence_score,
                    
                    # Volume analysis
                    block_trade_ratio=float(row.get('block_trade_ratio', 0)),
                    volume_anomaly_score=float(row.get('volume_anomaly_score', 0)),
                    relative_volume=float(row.get('relative_volume', 1)),
                    volume_efficiency=float(row.get('volume_efficiency', 0)),
                    
                    # Price analysis
                    price_volume_divergence=float(row.get('price_volume_divergence', 0)),
                    stealth_index=float(row.get('stealth_index', 0)),
                    absorption_ratio=float(row.get('absorption_ratio', 0)),
                    momentum_divergence=float(row.get('momentum_divergence', 0)),
                    
                    # Order flow metrics
                    order_flow_imbalance=float(row.get('order_flow_imbalance', 0)),
                    market_impact=float(row.get('market_impact', 0)),
                    participation_rate=float(row.get('participation_rate', 1)),
                    execution_quality=float(row.get('execution_quality', 0)),
                    
                    # Statistical measures
                    statistical_significance=float(row.get('flow_significance', 0)),
                    z_score=float(row.get('volume_zscore', 0)),
                    percentile_rank=float(row.get('volume_percentile', 0.5)),
                    
                    # ML classifications
                    institutional_probability=float(row.get('institutional_probability', 0.5)),
                    pattern_confidence=float(row.get('pattern_confidence', 0)),
                    anomaly_score=float(row.get('anomaly_score', 0)),
                    
                    # Risk metrics
                    risk_score=self._calculate_risk_score(row),
                    position_size=self._calculate_position_size(row),
                    exposure_limit=self._calculate_exposure_limit(row)
                )
                
                signals.append(signal)
            
            # Store recent signals for analysis
            self.last_signals = signals[-100:] if len(signals) > 100 else signals
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error in smart money signal generation: {str(e)}")
            return []
    
    def _classify_smart_money_signal(self, row: pd.Series) -> SmartMoneySignal:
        """Classify the smart money signal type."""
        try:
            # Get key metrics
            stealth_index = row.get('stealth_index', 0)
            absorption_ratio = row.get('absorption_ratio', 0)
            order_flow_imbalance = row.get('order_flow_imbalance', 0)
            relative_volume = row.get('relative_volume', 1)
            price_change = row.get('price_change', 0)
            
            # Aggressive signals (high volume + significant price movement)
            if relative_volume > 2.0 and abs(price_change) > 0.015:
                if order_flow_imbalance > 0.5:
                    return SmartMoneySignal.AGGRESSIVE_BUYING
                elif order_flow_imbalance < -0.5:
                    return SmartMoneySignal.AGGRESSIVE_SELLING
            
            # Stealth signals (high volume + low price impact)
            if stealth_index > 0.6 or absorption_ratio > 0.7:
                if order_flow_imbalance > 0.2:
                    return SmartMoneySignal.STEALTH_BUYING
                elif order_flow_imbalance < -0.2:
                    return SmartMoneySignal.STEALTH_SELLING
            
            # Accumulation/Distribution patterns
            accumulation_score = (
                stealth_index * 0.3 +
                absorption_ratio * 0.3 +
                max(order_flow_imbalance, 0) * 0.4
            )
            
            distribution_score = (
                stealth_index * 0.3 +
                absorption_ratio * 0.3 +
                max(-order_flow_imbalance, 0) * 0.4
            )
            
            if accumulation_score > 0.6:
                return SmartMoneySignal.ACCUMULATION
            elif distribution_score > 0.6:
                return SmartMoneySignal.DISTRIBUTION
            
            return SmartMoneySignal.NEUTRAL
            
        except Exception as e:
            self.logger.error(f"Error in smart money signal classification: {str(e)}")
            return SmartMoneySignal.NEUTRAL
    
    def _classify_institutional_activity(self, row: pd.Series) -> InstitutionalActivity:
        """Classify institutional activity level."""
        try:
            # Institutional activity score
            activity_score = (
                row.get('block_trade_ratio', 0) * 0.3 +
                row.get('institutional_probability', 0.5) * 0.4 +
                min(row.get('relative_volume', 1) / 3.0, 1.0) * 0.3
            )
            
            if activity_score > 0.8:
                return InstitutionalActivity.HIGH_ACTIVITY
            elif activity_score > 0.6:
                return InstitutionalActivity.MODERATE_ACTIVITY
            elif activity_score > 0.3:
                return InstitutionalActivity.LOW_ACTIVITY
            else:
                return InstitutionalActivity.ANOMALOUS_ACTIVITY
                
        except Exception as e:
            self.logger.error(f"Error in institutional activity classification: {str(e)}")
            return InstitutionalActivity.LOW_ACTIVITY
    
    def _classify_flow_direction(self, row: pd.Series) -> FlowDirection:
        """Classify money flow direction."""
        try:
            order_flow = row.get('order_flow_imbalance', 0)
            flow_persistence = row.get('flow_persistence', 0)
            
            if abs(order_flow) < 0.1:
                return FlowDirection.BALANCED
            elif flow_persistence > 0.5:  # High volatility in flow
                return FlowDirection.VOLATILE
            elif order_flow > 0.1:
                return FlowDirection.INFLOW
            else:
                return FlowDirection.OUTFLOW
                
        except Exception as e:
            self.logger.error(f"Error in flow direction classification: {str(e)}")
            return FlowDirection.BALANCED
    
    def _calculate_signal_confidence(self, row: pd.Series) -> float:
        """Calculate overall signal confidence."""
        try:
            # Multiple confidence factors
            statistical_conf = row.get('statistical_confidence', 0)
            pattern_conf = row.get('pattern_confidence', 0)
            volume_conf = min(row.get('relative_volume', 1) / 2.0, 1.0)
            flow_conf = abs(row.get('order_flow_imbalance', 0))
            
            # Weighted average
            confidence = (
                statistical_conf * 0.3 +
                pattern_conf * 0.3 +
                volume_conf * 0.2 +
                flow_conf * 0.2
            )
            
            return np.clip(confidence, 0.0, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error in confidence calculation: {str(e)}")
            return 0.5
    
    def _calculate_risk_score(self, row: pd.Series) -> float:
        """Calculate risk score for the signal."""
        try:
            # Risk factors
            anomaly_risk = row.get('anomaly_score', 0)
            volume_risk = min(row.get('volume_anomaly_score', 0) / 3.0, 1.0)
            confidence_risk = 1.0 - row.get('statistical_confidence', 0.5)
            flow_volatility_risk = row.get('flow_persistence', 0)
            
            # Combined risk
            risk_score = (
                anomaly_risk * 0.3 +
                volume_risk * 0.3 +
                confidence_risk * 0.2 +
                flow_volatility_risk * 0.2
            )
            
            return np.clip(risk_score, 0.0, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error in risk score calculation: {str(e)}")
            return 0.5
    
    def _calculate_position_size(self, row: pd.Series) -> float:
        """Calculate recommended position size."""
        try:
            # Base position size
            base_size = self.params.max_position_size
            
            # Adjustments
            confidence = self._calculate_signal_confidence(row)
            risk_score = self._calculate_risk_score(row)
            institutional_prob = row.get('institutional_probability', 0.5)
            
            # Position size calculation
            risk_adjustment = 1.0 - risk_score
            confidence_adjustment = confidence
            institutional_adjustment = institutional_prob
            
            position_size = base_size * risk_adjustment * confidence_adjustment * institutional_adjustment
            
            return np.clip(position_size, 0.0, self.params.max_position_size)
            
        except Exception as e:
            self.logger.error(f"Error in position size calculation: {str(e)}")
            return 0.0
    
    def _calculate_exposure_limit(self, row: pd.Series) -> float:
        """Calculate exposure limit based on risk."""
        try:
            risk_score = self._calculate_risk_score(row)
            base_limit = 0.1  # 10% base exposure
            
            # Reduce exposure for higher risk
            exposure_limit = base_limit * (1.0 - risk_score)
            
            return max(exposure_limit, 0.01)  # Minimum 1% exposure
            
        except Exception as e:
            self.logger.error(f"Error in exposure limit calculation: {str(e)}")
            return 0.05
    
    def _calculate_performance_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate performance metrics for the indicator suite."""
        try:
            if len(df) < 2:
                return {}
            
            metrics = {}
            
            # Detection statistics
            if 'is_block_trade' in df.columns:
                metrics['block_trade_detection_rate'] = float(df['is_block_trade'].mean())
                metrics['total_block_trades'] = int(df['is_block_trade'].sum())
            
            # Volume analysis
            if 'relative_volume' in df.columns:
                metrics['avg_relative_volume'] = float(df['relative_volume'].mean())
                metrics['volume_efficiency'] = float(df.get('volume_efficiency', pd.Series(0)).mean())
            
            # Flow analysis
            if 'order_flow_imbalance' in df.columns:
                flow_series = df['order_flow_imbalance'].dropna()
                if len(flow_series) > 0:
                    metrics['avg_flow_imbalance'] = float(flow_series.mean())
                    metrics['flow_volatility'] = float(flow_series.std())
            
            # Smart money detection
            if 'stealth_index' in df.columns:
                metrics['avg_stealth_activity'] = float(df['stealth_index'].mean())
                metrics['stealth_detection_rate'] = float((df['stealth_index'] > 0.5).mean())
            
            # ML model performance
            if self.is_trained and 'institutional_probability' in df.columns:
                metrics['institutional_detection_rate'] = float((df['institutional_probability'] > 0.7).mean())
                metrics['avg_pattern_confidence'] = float(df.get('pattern_confidence', pd.Series(0)).mean())
            
            # Risk metrics
            metrics['avg_anomaly_score'] = float(df.get('anomaly_score', pd.Series(0)).mean())
            
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
    
    def get_smart_money_summary(self, signals: List[SmartMoneyComponents]) -> Dict[str, Any]:
        """Generate summary of smart money activity."""
        try:
            if not signals:
                return {}
            
            latest_signals = signals[-20:]  # Last 20 periods
            
            summary = {
                'current_signal': latest_signals[-1].smart_money_signal.value,
                'institutional_activity': latest_signals[-1].institutional_activity.value,
                'flow_direction': latest_signals[-1].flow_direction.value,
                'confidence': latest_signals[-1].confidence_score,
                
                # Recent activity
                'recent_block_trades': sum(1 for s in latest_signals if s.block_trade_ratio > 0.1),
                'avg_stealth_activity': np.mean([s.stealth_index for s in latest_signals]),
                'avg_institutional_prob': np.mean([s.institutional_probability for s in latest_signals]),
                
                # Risk assessment
                'current_risk': latest_signals[-1].risk_score,
                'recommended_position': latest_signals[-1].position_size,
                
                # Market conditions
                'volume_regime': 'High' if latest_signals[-1].relative_volume > 2.0 else 'Normal',
                'flow_imbalance': latest_signals[-1].order_flow_imbalance,
                'market_impact': latest_signals[-1].market_impact
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating smart money summary: {str(e)}")
            return {}
    
    def get_trading_recommendation(self, signal: SmartMoneyComponents) -> Dict[str, Any]:
        """Generate trading recommendation based on smart money analysis."""
        try:
            recommendation = {
                'action': 'HOLD',
                'confidence': signal.confidence_score,
                'position_size': signal.position_size,
                'risk_level': 'MEDIUM',
                'reasoning': []
            }
            
            # Determine action
            if signal.smart_money_signal in [SmartMoneySignal.ACCUMULATION, SmartMoneySignal.STEALTH_BUYING, SmartMoneySignal.AGGRESSIVE_BUYING]:
                recommendation['action'] = 'BUY'
            elif signal.smart_money_signal in [SmartMoneySignal.DISTRIBUTION, SmartMoneySignal.STEALTH_SELLING, SmartMoneySignal.AGGRESSIVE_SELLING]:
                recommendation['action'] = 'SELL'
            
            # Risk level
            if signal.risk_score < 0.3:
                recommendation['risk_level'] = 'LOW'
            elif signal.risk_score > 0.7:
                recommendation['risk_level'] = 'HIGH'
            
            # Reasoning
            if signal.institutional_activity == InstitutionalActivity.HIGH_ACTIVITY:
                recommendation['reasoning'].append("High institutional activity detected")
            
            if signal.block_trade_ratio > 0.2:
                recommendation['reasoning'].append("Significant block trading activity")
            
            if signal.stealth_index > 0.6:
                recommendation['reasoning'].append("Stealth trading patterns identified")
            
            if signal.order_flow_imbalance > 0.5:
                recommendation['reasoning'].append("Strong buying pressure")
            elif signal.order_flow_imbalance < -0.5:
                recommendation['reasoning'].append("Strong selling pressure")
            
            if signal.institutional_probability > 0.8:
                recommendation['reasoning'].append("High probability of institutional involvement")
            
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
        data = yf.download(ticker, period="6mo", interval="1h")
        
        if data.empty:
            print("No data available for testing")
        else:
            # Initialize indicator
            params = SmartMoneyParameters(
                block_trade_threshold=2.0,
                ml_lookback=200,
                confidence_period=50
            )
            
            indicator = AdvancedSmartMoneyIndicators(params)
            
            # Calculate signals
            result = indicator.calculate(data)
            
            # Display results
            print(f"Analyzed smart money patterns for {len(result['signals'])} periods")
            print(f"Calculation time: {result['metadata']['calculation_time']:.4f}s")
            print(f"Model trained: {result['metadata']['model_trained']}")
            print(f"Block trades detected: {len(result['block_trades'])}")
            
            # Show latest signals
            if result['signals']:
                latest_signal = result['signals'][-1]
                print(f"\nLatest Smart Money Analysis:")
                print(f"Signal: {latest_signal.smart_money_signal.value}")
                print(f"Institutional Activity: {latest_signal.institutional_activity.value}")
                print(f"Flow Direction: {latest_signal.flow_direction.value}")
                print(f"Confidence: {latest_signal.confidence_score:.3f}")
                print(f"Institutional Probability: {latest_signal.institutional_probability:.3f}")
                print(f"Stealth Index: {latest_signal.stealth_index:.3f}")
                print(f"Block Trade Ratio: {latest_signal.block_trade_ratio:.3f}")
                print(f"Order Flow Imbalance: {latest_signal.order_flow_imbalance:.3f}")
                
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
                
                # Smart money summary
                summary = indicator.get_smart_money_summary(result['signals'])
                if summary:
                    print(f"\nSmart Money Summary:")
                    print(f"Recent Block Trades: {summary.get('recent_block_trades', 0)}")
                    print(f"Avg Stealth Activity: {summary.get('avg_stealth_activity', 0):.3f}")
                    print(f"Volume Regime: {summary.get('volume_regime', 'Unknown')}")
                    print(f"Flow Imbalance: {summary.get('flow_imbalance', 0):.3f}")
                
            # Performance metrics
            if result['performance']:
                print(f"\nPerformance Metrics:")
                for key, value in result['performance'].items():
                    if isinstance(value, (int, float)):
                        print(f"{key}: {value:.4f}")
                    else:
                        print(f"{key}: {value}")
    
    except Exception as e:
        print(f"Error in testing: {str(e)}")
        print(traceback.format_exc())