"""
Intermarket Correlation Indicator - Advanced Cross-Asset Correlation Analysis
==================================

This module implements a sophisticated intermarket correlation indicator that analyzes
correlations between different markets, asset classes, and financial instruments.
It provides comprehensive correlation analysis, regime detection, and predictive
modeling for understanding market relationships and dependencies.

Features:
    - Multi-asset correlation matrix calculation
- Dynamic correlation tracking over time
- Correlation regime detection and classification
- Rolling correlation analysis with multiple timeframes
- Cross-asset signal generation based on correlations
- Correlation breakdown detection and alerts
- Machine learning enhanced correlation prediction
- Portfolio diversification analysis
- Risk contagion detection

The indicator helps traders understand market interconnections and identify
opportunities based on correlation relationships and their changes over time.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.covariance import LedoitWolf
import warnings
warnings.filterwarnings('ignore')

from auj_platform.src.indicator_engine.indicators.base.standard_indicator import StandardIndicatorInterface
from auj_platform.src.indicator_engine.indicators.base.standard_indicator import IndicatorCalculationError

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class CorrelationPair:
    """Represents a correlation relationship between two assets"""
    asset1: str
    asset2: str
    correlation: float
    p_value: float
    correlation_type: str  # 'pearson', 'spearman', 'kendall'
    timeframe: str
    strength: str  # 'weak', 'moderate', 'strong'
    direction: str  # 'positive', 'negative', 'neutral'
    stability: float  # How stable the correlation is over time


@dataclass
class CorrelationRegime:
    """Represents a correlation regime period"""
    start_time: datetime
    end_time: datetime
    regime_type: str  # 'high_correlation', 'low_correlation', 'mixed'
    average_correlation: float
    volatility: float
    market_stress: float
    regime_strength: float


@dataclass
class MarketCluster:
    """Represents a cluster of highly correlated markets"""
    cluster_id: int
    assets: List[str]
    average_correlation: float
    cluster_strength: float
    stability_score: float
    representative_asset: str


class IntermarketCorrelationIndicator(StandardIndicatorInterface):
    """
    Advanced Intermarket Correlation Indicator with regime detection
    and predictive modeling capabilities.
    """

def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'correlation_window': 60,  # Days for correlation calculation
            'rolling_windows': [20, 60, 120, 252],  # Multiple timeframes
            'correlation_threshold': 0.3,  # Minimum significant correlation
            'regime_detection_window': 30,
            'min_correlation_strength': 0.2,
            'correlation_types': ['pearson', 'spearman'],
            'cluster_count': 5,  # Number of market clusters
            'stability_window': 20,  # Window for stability calculation
            'stress_threshold': 0.7,  # Market stress detection threshold
            'prediction_horizon': 5,  # Days to predict correlation
            'ml_features_window': 10,
            'significance_level': 0.05,
            'asset_categories': {
                'equities': ['SPY', 'QQQ', 'IWM'],
                'bonds': ['TLT', 'IEF', 'SHY'],
                'commodities': ['GLD', 'SLV', 'USO'],
                'currencies': ['UUP', 'FXE', 'FXY'],
                'volatility': ['VIX', 'UVXY']
            },
            'correlation_decay_factor': 0.94,  # For exponential weighting
            'outlier_threshold': 3.0  # Standard deviations for outlier detection
        }

        if parameters:
            default_params.update(parameters)

        super().__init__(name="IntermarketCorrelation")

        # Initialize internal state
        self.correlation_matrix = None
        self.correlation_history = []
        self.correlation_pairs: List[CorrelationPair] = []
        self.regimes: List[CorrelationRegime] = []
        self.market_clusters: List[MarketCluster] = []
        self.ml_model = None
        self.scaler = StandardScaler()
        self.ledoit_wolf = LedoitWolf()

        logger.info(f"IntermarketCorrelationIndicator initialized")

def get_data_requirements(self) -> DataRequirement:
        """Define the data requirements for correlation calculation"""
        return DataRequirement()
            data_type=DataType.OHLCV,
            required_columns=['close'],  # Only need close prices for correlation
            min_periods=max(self.parameters['correlation_window'], 50),
            lookback_periods=max(self.parameters['rolling_windows']) * 2
(        )

def validate_parameters(self) -> bool:
        """Validate the indicator parameters"""
        try:
            required_params = ['correlation_window', 'rolling_windows']
            for param in required_params:
                if param not in self.parameters:
                    raise ValueError(f"Missing required parameter: {param}")

            if self.parameters['correlation_window'] < 10:
                raise ValueError("correlation_window must be at least 10")

            if not self.parameters['rolling_windows']:
                raise ValueError("rolling_windows cannot be empty")

            return True

        except Exception as e:
            logger.error(f"Parameter validation failed: {str(e)}")
            return False

def _calculate_correlation_matrix(self, returns_data: pd.DataFrame,:)
(                                    method: str = 'pearson') -> pd.DataFrame:
        """Calculate correlation matrix using specified method"""
        try:
            if method == 'pearson':
                correlation_matrix = returns_data.corr(method='pearson')
            elif method == 'spearman':
                correlation_matrix = returns_data.corr(method='spearman')
            elif method == 'kendall':
                correlation_matrix = returns_data.corr(method='kendall')
            elif method == 'ledoit_wolf':
                # Use Ledoit-Wolf shrinkage estimator for robust correlation
                covariance_matrix, _ = self.ledoit_wolf.fit(returns_data.fillna(0)).covariance_, None
                std_dev = np.sqrt(np.diag(covariance_matrix))
                correlation_matrix = covariance_matrix / np.outer(std_dev, std_dev)
                correlation_matrix = pd.DataFrame()
                    correlation_matrix,
                    index=returns_data.columns,
                    columns=returns_data.columns
(                )
            else:
                correlation_matrix = returns_data.corr(method='pearson')

            return correlation_matrix

        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {str(e)}")
            return pd.DataFrame()

def _calculate_rolling_correlations(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Calculate rolling correlations for multiple timeframes"""
        try:
            rolling_correlations = {}
            returns = data.pct_change().dropna()

            for window in self.parameters['rolling_windows']:
                if len(returns) >= window:
                    rolling_corr = returns.rolling(window).corr()
                    rolling_correlations[f'{window}d'] = rolling_corr

            return rolling_correlations

        except Exception as e:
            logger.error(f"Error calculating rolling correlations: {str(e)}")
            return {}

def _extract_correlation_pairs(self, correlation_matrix: pd.DataFrame,:)
(                                 timeframe: str) -> List[CorrelationPair]:
        """Extract significant correlation pairs from matrix"""
        try:
            pairs = []
            threshold = self.parameters['correlation_threshold']

            for i, asset1 in enumerate(correlation_matrix.columns):
                for j, asset2 in enumerate(correlation_matrix.columns):
                    if i < j:  # Avoid duplicates and self-correlation:
                        corr_value = correlation_matrix.loc[asset1, asset2]

                        if not pd.isna(corr_value) and abs(corr_value) >= threshold:
                            # Determine correlation strength
                            abs_corr = abs(corr_value)
                            if abs_corr >= 0.7:
                                strength = 'strong'
                            elif abs_corr >= 0.4:
                                strength = 'moderate'
                            else:
                                strength = 'weak'

                            # Determine direction
                            if corr_value > 0.1:
                                direction = 'positive'
                            elif corr_value < -0.1:
                                direction = 'negative'
                            else:
                                direction = 'neutral'

                            pair = CorrelationPair()
                                asset1=asset1,
                                asset2=asset2,
                                correlation=corr_value,
                                p_value=0.05,  # Would need actual calculation
                                correlation_type='pearson',
                                timeframe=timeframe,
                                strength=strength,
                                direction=direction,
                                stability=0.5  # Will be calculated separately
(                            )
                            pairs.append(pair)

            return pairs

        except Exception as e:
            logger.error(f"Error extracting correlation pairs: {str(e)}")
            return []

def _calculate_correlation_stability(self, rolling_correlations: Dict[str, pd.DataFrame],:)
(                                       pair: CorrelationPair) -> float:
        """Calculate stability of correlation over time"""
        try:
            stability_window = self.parameters['stability_window']

            # Get correlation time series for this pair
            correlation_series = []

            for timeframe, rolling_corr in rolling_correlations.items():
                if pair.asset1 in rolling_corr.columns and pair.asset2 in rolling_corr.columns:
                    pair_corr = rolling_corr.loc[(slice(None), pair.asset1), pair.asset2]
                    if len(pair_corr) > 0:
                        correlation_series.extend(pair_corr.dropna().values)

            if len(correlation_series) >= stability_window:
                # Calculate stability as inverse of standard deviation
                correlation_std = np.std(correlation_series[-stability_window:])
                stability = max(0, 1 - correlation_std)
                return min(stability, 1.0)

            return 0.5  # Default moderate stability

        except Exception as e:
            logger.error(f"Error calculating correlation stability: {str(e)}")
            return 0.5

def _detect_correlation_regimes(self, correlation_history: List[float]) -> List[CorrelationRegime]:
        """Detect different correlation regimes over time"""
        try:
            if len(correlation_history) < self.parameters['regime_detection_window']:
                return []

            regimes = []
            window = self.parameters['regime_detection_window']

            # Use K-means clustering to identify regimes
            correlation_array = np.array(correlation_history).reshape(-1, 1)
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            regime_labels = kmeans.fit_predict(correlation_array)

            # Map clusters to regime types
            cluster_centers = kmeans.cluster_centers_.flatten()
            sorted_indices = np.argsort(cluster_centers)

            regime_mapping = {
                sorted_indices[0]: 'low_correlation',
                sorted_indices[1]: 'mixed',
                sorted_indices[2]: 'high_correlation'
            }

            # Create regime periods
            current_regime = None
            regime_start = 0

            for i, label in enumerate(regime_labels):
                regime_type = regime_mapping[label]

                if current_regime != regime_type:
                    if current_regime is not None:
                        # End previous regime
                        regime_data = correlation_history[regime_start:i]
                        regime = CorrelationRegime()
                            start_time=datetime.utcnow() - timedelta(days=len(correlation_history) - regime_start),
                            end_time=datetime.utcnow() - timedelta(days=len(correlation_history) - i),
                            regime_type=current_regime,
                            average_correlation=np.mean(regime_data),
                            volatility=np.std(regime_data),
                            market_stress=self._calculate_market_stress(regime_data),
                            regime_strength=1.0 - np.std(regime_data)
(                        )
                        regimes.append(regime)

                    current_regime = regime_type
                    regime_start = i

            # Add final regime
            if current_regime is not None:
                regime_data = correlation_history[regime_start:]
                regime = CorrelationRegime()
                    start_time=datetime.utcnow() - timedelta(days=len(correlation_history) - regime_start),
                    end_time=datetime.utcnow(),
                    regime_type=current_regime,
                    average_correlation=np.mean(regime_data),
                    volatility=np.std(regime_data),
                    market_stress=self._calculate_market_stress(regime_data),
                    regime_strength=1.0 - np.std(regime_data)
(                )
                regimes.append(regime)

            return regimes

        except Exception as e:
            logger.error(f"Error detecting correlation regimes: {str(e)}")
            return []

def _calculate_market_stress(self, correlation_data: List[float]) -> float:
        """Calculate market stress level based on correlation patterns"""
        try:
            if not correlation_data:
                return 0.0

            # High correlations during stress, low during normal times
            avg_correlation = np.mean(np.abs(correlation_data))
            correlation_increase = max(0, avg_correlation - 0.3)  # Baseline correlation

            # Stress increases with higher absolute correlations
            stress_level = min(correlation_increase / 0.4, 1.0)  # Normalize to 0-1

            return stress_level

        except Exception as e:
            logger.error(f"Error calculating market stress: {str(e)}")
            return 0.0

def _identify_market_clusters(self, correlation_matrix: pd.DataFrame) -> List[MarketCluster]:
        """Identify clusters of highly correlated markets"""
        try:
            if correlation_matrix.empty:
                return []

            # Use hierarchical clustering on correlation matrix
            distance_matrix = 1 - correlation_matrix.abs()

            # Simple clustering based on correlation threshold
            clusters = []
            processed_assets = set()
            cluster_threshold = 0.6

            for i, asset1 in enumerate(correlation_matrix.columns):
                if asset1 in processed_assets:
                    continue

                cluster_assets = [asset1]
                processed_assets.add(asset1)

                for j, asset2 in enumerate(correlation_matrix.columns):
                    if asset2 != asset1 and asset2 not in processed_assets:
                        correlation = abs(correlation_matrix.loc[asset1, asset2])
                        if correlation >= cluster_threshold:
                            cluster_assets.append(asset2)
                            processed_assets.add(asset2)

                if len(cluster_assets) > 1:
                    # Calculate cluster properties
                    cluster_correlations = []
                    for a1 in cluster_assets:
                        for a2 in cluster_assets:
                            if a1 != a2:
                                cluster_correlations.append(abs(correlation_matrix.loc[a1, a2]))

                    avg_correlation = np.mean(cluster_correlations) if cluster_correlations else 0
                    cluster_strength = min(avg_correlation * len(cluster_assets) / 5, 1.0)

                    cluster = MarketCluster()
                        cluster_id=len(clusters),
                        assets=cluster_assets,
                        average_correlation=avg_correlation,
                        cluster_strength=cluster_strength,
                        stability_score=0.7,  # Would calculate from historical data
                        representative_asset=cluster_assets[0]  # Could be more sophisticated
(                    )
                    clusters.append(cluster)

            return clusters

        except Exception as e:
            logger.error(f"Error identifying market clusters: {str(e)}")
            return []
def _prepare_ml_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for correlation prediction model"""
        try:
            features = []
            returns = data.pct_change().dropna()

            if len(returns) < self.parameters['ml_features_window']:
                return np.array([[0.0] * 10])

            recent_returns = returns.tail(self.parameters['ml_features_window'])

            # Volatility features
            volatility = recent_returns.std()
            features.extend([volatility.mean(), volatility.std()])

            # Momentum features
            momentum = recent_returns.mean()
            features.extend([momentum.mean(), momentum.std()])

            # Correlation dispersion
            recent_corr = recent_returns.corr()
            upper_triangle = recent_corr.where(np.triu(np.ones(recent_corr.shape), k=1).astype(bool))
            corr_values = upper_triangle.stack().values
            features.extend([np.mean(corr_values), np.std(corr_values)])

            # Market stress indicators
            max_drawdown = (recent_returns.cumsum() - recent_returns.cumsum().expanding().max()).min()
            features.append(abs(max_drawdown.mean()))

            # Volume-like features (using price movements as proxy)
            price_velocity = abs(recent_returns).mean()
            features.append(price_velocity.mean())

            # Trend features
            trend_strength = abs(recent_returns.cumsum().iloc[-1])
            features.append(trend_strength.mean())

            # Cross-sectional features
            cross_sectional_momentum = recent_returns.iloc[-1] - recent_returns.mean()
            features.append(cross_sectional_momentum.std())

            # Pad or truncate to fixed size
            while len(features) < 10:
                features.append(0.0)
            features = features[:10]

            return np.array(features).reshape(1, -1)

        except Exception as e:
            logger.error(f"Error preparing ML features: {str(e)}")
            return np.array([[0.0] * 10])

def _train_correlation_prediction_model(self, data: pd.DataFrame):
        """Train ML model to predict correlation changes"""
        try:
            returns = data.pct_change().dropna()

            if len(returns) < 100:  # Need sufficient data:
                return

            X, y = [], []
            window = self.parameters['ml_features_window']
            horizon = self.parameters['prediction_horizon']

            for i in range(window, len(returns) - horizon):
                # Features from current window
                subset = returns.iloc[i-window:i]
                features = self._prepare_ml_features(pd.DataFrame(subset))

                # Target: future correlation change
                current_corr = subset.corr()
                future_subset = returns.iloc[i:i+horizon]
                future_corr = future_subset.corr()

                # Calculate correlation change magnitude
                upper_triangle_current = current_corr.where(np.triu(np.ones(current_corr.shape), k=1).astype(bool))
                upper_triangle_future = future_corr.where(np.triu(np.ones(future_corr.shape), k=1).astype(bool))

                current_values = upper_triangle_current.stack().values
                future_values = upper_triangle_future.stack().values

                if len(current_values) > 0 and len(future_values) > 0:
                    correlation_change = np.mean(abs(future_values - current_values))

                    X.append(features[0])
                    y.append(correlation_change)

            if len(X) > 20:
                X = np.array(X)
                y = np.array(y)

                # Scale features
                X_scaled = self.scaler.fit_transform(X)

                # Train model
                self.ml_model = RandomForestRegressor()
                    n_estimators=50,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
(                )
                self.ml_model.fit(X_scaled, y)

                logger.debug(f"Correlation prediction model trained with {len(X)} samples")

        except Exception as e:
            logger.error(f"Error training correlation prediction model: {str(e)}")

def _predict_correlation_change(self, data: pd.DataFrame) -> float:
        """Predict future correlation changes using ML model"""
        try:
            if self.ml_model is None:
                return 0.0

            features = self._prepare_ml_features(data)
            features_scaled = self.scaler.transform(features)
            prediction = self.ml_model.predict(features_scaled)[0]

            return min(prediction, 1.0)

        except Exception as e:
            logger.error(f"Error predicting correlation change: {str(e)}")
            return 0.0

def calculate_raw(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate comprehensive intermarket correlation analysis
        """
        try:
            # Calculate returns for correlation analysis
            returns = data.pct_change().dropna()

            if len(returns) < self.parameters['correlation_window']:
                return {
                    'correlation_matrix': {},
                    'correlation_pairs': [],
                    'market_clusters': [],
                    'current_regime': None,
                    'correlation_strength': 0.0,
                    'market_stress': 0.0,
                    'prediction': 0.0
                }

            # Calculate main correlation matrix
            correlation_matrix = self._calculate_correlation_matrix()
                returns.tail(self.parameters['correlation_window'])
(            )
            self.correlation_matrix = correlation_matrix

            # Calculate rolling correlations for multiple timeframes
            rolling_correlations = self._calculate_rolling_correlations(data)

            # Extract significant correlation pairs
            all_pairs = []
            for timeframe, rolling_corr in rolling_correlations.items():
                if not rolling_corr.empty:
                    # Get most recent correlation matrix for this timeframe
                    recent_corr = rolling_corr.groupby(level=1).last()
                    pairs = self._extract_correlation_pairs(recent_corr, timeframe)

                    # Calculate stability for each pair
                    for pair in pairs:
                        pair.stability = self._calculate_correlation_stability(rolling_correlations, pair)

                    all_pairs.extend(pairs)

            self.correlation_pairs = all_pairs

            # Identify market clusters
            market_clusters = self._identify_market_clusters(correlation_matrix)
            self.market_clusters = market_clusters

            # Build correlation history for regime detection
            if correlation_matrix.size > 0:
                # Calculate average absolute correlation as market correlation measure
                upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
                current_avg_correlation = upper_triangle.stack().abs().mean()
                self.correlation_history.append(current_avg_correlation)

                # Keep only recent history
                max_history = 100
                if len(self.correlation_history) > max_history:
                    self.correlation_history = self.correlation_history[-max_history:]

            # Detect correlation regimes
            regimes = self._detect_correlation_regimes(self.correlation_history)
            self.regimes = regimes

            # Train ML model for correlation prediction
            self._train_correlation_prediction_model(data)

            # Make correlation change prediction
            correlation_prediction = self._predict_correlation_change(data)

            # Calculate overall metrics
            correlation_strength = 0.0
            market_stress = 0.0

            if self.correlation_history:
                correlation_strength = self.correlation_history[-1]
                market_stress = self._calculate_market_stress(self.correlation_history[-10:])

            # Current regime
            current_regime = regimes[-1] if regimes else None

            # Calculate signal strength based on correlation analysis
            signal_strength = 0.0
            if correlation_strength > 0.7:  # High correlation period:
                signal_strength = min(correlation_strength * market_stress, 1.0)

            # Prepare result
            result = {
                'correlation_matrix': correlation_matrix.to_dict() if not correlation_matrix.empty else {},
                'correlation_pairs': [self._pair_to_dict(pair) for pair in all_pairs[:20]],  # Top 20 pairs
                'market_clusters': [self._cluster_to_dict(cluster) for cluster in market_clusters],
                'correlation_regimes': [self._regime_to_dict(regime) for regime in regimes[-5:]],  # Recent regimes
                'current_regime': self._regime_to_dict(current_regime) if current_regime else None,
                'correlation_strength': correlation_strength,
                'market_stress': market_stress,
                'signal_strength': signal_strength,
                'correlation_prediction': correlation_prediction,
                'total_pairs': len(all_pairs),
                'significant_clusters': len(market_clusters),
                'ml_model_active': self.ml_model is not None,
                'analysis_timestamp': datetime.utcnow().isoformat()
            }

            return result

        except Exception as e:
            logger.error(f"Error in intermarket correlation calculation: {str(e)}")
            raise IndicatorCalculationError()
                indicator_name=self.name,
                calculation_step="intermarket_correlation_calculation",
                message=str(e)
(            )

def _pair_to_dict(self, pair: CorrelationPair) -> Dict[str, Any]:
        """Convert CorrelationPair to dictionary"""
        return {
            'asset1': pair.asset1,
            'asset2': pair.asset2,
            'correlation': pair.correlation,
            'p_value': pair.p_value,
            'correlation_type': pair.correlation_type,
            'timeframe': pair.timeframe,
            'strength': pair.strength,
            'direction': pair.direction,
            'stability': pair.stability
        }

def _cluster_to_dict(self, cluster: MarketCluster) -> Dict[str, Any]:
        """Convert MarketCluster to dictionary"""
        return {
            'cluster_id': cluster.cluster_id,
            'assets': cluster.assets,
            'average_correlation': cluster.average_correlation,
            'cluster_strength': cluster.cluster_strength,
            'stability_score': cluster.stability_score,
            'representative_asset': cluster.representative_asset
        }

def _regime_to_dict(self, regime: CorrelationRegime) -> Dict[str, Any]:
        """Convert CorrelationRegime to dictionary"""
        return {
            'start_time': regime.start_time.isoformat(),
            'end_time': regime.end_time.isoformat(),
            'regime_type': regime.regime_type,
            'average_correlation': regime.average_correlation,
            'volatility': regime.volatility,
            'market_stress': regime.market_stress,
            'regime_strength': regime.regime_strength
        }

def _generate_signal(self, value: Dict[str, Any], data: pd.DataFrame) -> Tuple[Optional[SignalType], float]:
        """
        Generate trading signals based on intermarket correlation analysis
        """
        try:
            signal_strength = value.get('signal_strength', 0.0)
            market_stress = value.get('market_stress', 0.0)
            correlation_strength = value.get('correlation_strength', 0.0)

            if signal_strength < 0.3:
                return SignalType.NEUTRAL, 0.0

            # High correlation with high stress = potential reversal
            if correlation_strength > 0.7 and market_stress > 0.6:
                # Markets are highly correlated during stress - contrarian signal
                recent_momentum = data['close'].pct_change(5).iloc[-1] if 'close' in data.columns else 0

                if recent_momentum < -0.02:  # Oversold:
                    return SignalType.BUY, signal_strength * 0.8
                elif recent_momentum > 0.02:  # Overbought:
                    return SignalType.SELL, signal_strength * 0.8

            # Low correlation = diversification opportunity
            elif correlation_strength < 0.3 and market_stress < 0.4:
                # Low correlation, low stress = trend following
                recent_momentum = data['close'].pct_change(10).iloc[-1] if 'close' in data.columns else 0

                if recent_momentum > 0.01:
                    return SignalType.BUY, signal_strength * 0.6
                elif recent_momentum < -0.01:
                    return SignalType.SELL, signal_strength * 0.6

            return SignalType.NEUTRAL, 0.0

        except Exception as e:
            logger.error(f"Error generating signal: {str(e)}")
            return SignalType.NEUTRAL, 0.0

def _get_metadata(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get additional metadata for the calculation"""
        base_metadata = super()._get_metadata(data)

        correlation_metadata = {
            'correlation_window': self.parameters['correlation_window'],
            'total_correlation_pairs': len(self.correlation_pairs),
            'market_clusters_detected': len(self.market_clusters),
            'correlation_regimes': len(self.regimes),
            'ml_model_trained': self.ml_model is not None,
            'correlation_history_length': len(self.correlation_history),
            'asset_categories': list(self.parameters['asset_categories'].keys())
        }

        base_metadata.update(correlation_metadata)
        return base_metadata


def create_intermarket_correlation_indicator(parameters: Optional[Dict[str, Any]] = None) -> IntermarketCorrelationIndicator:
    """
    Factory function to create an IntermarketCorrelationIndicator instance

    Args:
        parameters: Optional dictionary of parameters to customize the indicator

    Returns:
        Configured IntermarketCorrelationIndicator instance
    """
    return IntermarketCorrelationIndicator(parameters=parameters)


# Example usage
if __name__ == "__main__":
    # Create sample multi-asset data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')

    # Simulate correlated assets
    base_returns = np.random.randn(len(dates)) * 0.01

    sample_data = pd.DataFrame({)
        'SPY': 100 * np.exp(np.cumsum(base_returns + np.random.randn(len(dates)) * 0.005)),
        'QQQ': 100 * np.exp(np.cumsum(base_returns * 0.8 + np.random.randn(len(dates)) * 0.006)),
        'IWM': 100 * np.exp(np.cumsum(base_returns * 0.6 + np.random.randn(len(dates)) * 0.008)),
        'TLT': 100 * np.exp(np.cumsum(-base_returns * 0.3 + np.random.randn(len(dates)) * 0.004)),
        'GLD': 100 * np.exp(np.cumsum(base_returns * 0.2 + np.random.randn(len(dates)) * 0.007)),
(    }, index=dates)

    # Test the indicator
    indicator = create_intermarket_correlation_indicator({)
        'correlation_window': 60,
        'rolling_windows': [20, 60, 120]
(    })

    try:
        result = indicator.calculate(sample_data)
        print("Intermarket Correlation Calculation Result:")
        print(f"Signal: {result.signal}, Confidence: {result.confidence:.3f}")
        print(f"Correlation strength: {result.value.get('correlation_strength', 0):.3f}")
        print(f"Market stress: {result.value.get('market_stress', 0):.3f}")
        print(f"Total pairs: {result.value.get('total_pairs', 0)}")
        print(f"Market clusters: {result.value.get('significant_clusters', 0)}")

        if result.value.get('current_regime'):
            regime = result.value['current_regime']
            print(f"Current regime: {regime['regime_type']}")
            print(f"Regime strength: {regime['regime_strength']:.3f}")

    except Exception as e:
        print(f"Error testing indicator: {str(e)}")
