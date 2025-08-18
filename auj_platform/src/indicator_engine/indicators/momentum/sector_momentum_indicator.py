"""
Sector Momentum Indicator - Advanced Sector-Based Momentum Analysis
==============================

This module implements a sophisticated sector momentum indicator that analyzes
momentum patterns across different market sectors. It provides comprehensive
sector rotation analysis, relative strength measurement, and momentum-based
sector allocation recommendations.

Features:
    - Multi-sector momentum calculation and comparison
- Sector rotation detection and classification
- Relative strength ranking and analysis
- Momentum persistence and reversal detection
- Cross-sector correlation and leadership analysis
- Risk-adjusted momentum measurement
- Machine learning enhanced sector prediction
- Sector allocation optimization
- Economic cycle sector analysis

The indicator helps traders identify sector rotation opportunities and
momentum-based investment strategies across different market sectors.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from scipy import stats
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

from auj_platform.src.indicator_engine.indicators.base.standard_indicator import StandardIndicatorInterface
from auj_platform.src.indicator_engine.indicators.base.standard_indicator import IndicatorCalculationError

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class SectorMomentum:
    """Represents momentum data for a specific sector"""
    sector: str
    momentum_1m: float  # 1-month momentum
    momentum_3m: float  # 3-month momentum
    momentum_6m: float  # 6-month momentum
    momentum_12m: float  # 12-month momentum
    relative_strength: float  # Relative to market
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    momentum_rank: int
    momentum_score: float  # Composite momentum score
    trend_strength: float
    reversal_probability: float


@dataclass
class SectorRotation:
    """Represents a sector rotation pattern"""
    from_sector: str
    to_sector: str
    rotation_strength: float
    rotation_type: str  # 'cyclical', 'defensive', 'growth', 'value'
    confidence: float
    duration_estimate: int  # Estimated days
    economic_phase: str  # 'early_cycle', 'mid_cycle', 'late_cycle', 'recession'


@dataclass
class SectorAllocation:
    """Represents recommended sector allocation"""
    sector: str
    current_weight: float
    recommended_weight: float
    weight_change: float
    allocation_reason: str
    expected_return: float
    risk_level: str  # 'low', 'medium', 'high'


class SectorMomentumIndicator(StandardIndicatorInterface):
    """
    Advanced Sector Momentum Indicator with rotation detection
    and allocation optimization capabilities.
    """

def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'momentum_windows': [21, 63, 126, 252],  # 1m, 3m, 6m, 12m
            'sector_etfs': {
                'Technology': 'XLK',
                'Healthcare': 'XLV',
                'Financials': 'XLF',
                'Consumer Discretionary': 'XLY',
                'Communication Services': 'XLC',
                'Industrials': 'XLI',
                'Consumer Staples': 'XLP',
                'Energy': 'XLE',
                'Utilities': 'XLU',
                'Real Estate': 'XLRE',
                'Materials': 'XLB'
            },
            'benchmark': 'SPY',  # Market benchmark
            'relative_strength_window': 63,
            'volatility_window': 21,
            'momentum_weight': 0.4,
            'relative_strength_weight': 0.3,
            'risk_weight': 0.3,
            'rotation_threshold': 0.15,  # 15% momentum difference for rotation
            'min_allocation': 0.05,  # 5% minimum allocation
            'max_allocation': 0.25,  # 25% maximum allocation
            'rebalance_threshold': 0.02,  # 2% threshold for rebalancing
            'risk_free_rate': 0.02,  # 2% annual risk-free rate
            'lookback_periods': 252,
            'prediction_horizon': 21,  # 21 days prediction
            'ml_features_window': 30,
            'economic_cycle_indicators': ['yield_curve', 'credit_spreads', 'volatility'],
            'sector_correlations_window': 126,
            'momentum_persistence_factor': 0.7
        }

        if parameters:
            default_params.update(parameters)

        super().__init__(name="SectorMomentum")

        # Initialize internal state
        self.sector_momentum_data: Dict[str, SectorMomentum] = {}
        self.sector_rotations: List[SectorRotation] = []
        self.sector_allocations: List[SectorAllocation] = []
        self.momentum_history: Dict[str, List[float]] = {}
        self.ml_model = None
        self.scaler = StandardScaler()
        self.current_economic_phase = 'mid_cycle'

        # Initialize momentum history for each sector
        for sector in self.parameters['sector_etfs'].keys():
            self.momentum_history[sector] = []

        logger.info(f"SectorMomentumIndicator initialized with {len(self.parameters['sector_etfs'])} sectors")

def get_data_requirements(self) -> DataRequirement:
        """Define the data requirements for sector momentum calculation"""
        return DataRequirement()
            data_type=DataType.OHLCV,
            required_columns=['close'],
            min_periods=max(self.parameters['momentum_windows']) + 50,
            lookback_periods=self.parameters['lookback_periods']
(        )

def validate_parameters(self) -> bool:
        """Validate the indicator parameters"""
        try:
            required_params = ['momentum_windows', 'sector_etfs']
            for param in required_params:
                if param not in self.parameters:
                    raise ValueError(f"Missing required parameter: {param}")

            if not self.parameters['momentum_windows']:
                raise ValueError("momentum_windows cannot be empty")

            if not self.parameters['sector_etfs']:
                raise ValueError("sector_etfs cannot be empty")

            return True

        except Exception as e:
            logger.error(f"Parameter validation failed: {str(e)}")
            return False

def _simulate_sector_data(self, market_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Simulate sector ETF data based on market data
        In real implementation, this would use actual sector ETF data
        """
        try:
            np.random.seed(42)  # For reproducible results
            sector_data = {}

            market_returns = market_data['close'].pct_change().dropna()

            # Sector characteristics (beta, alpha, correlation)
            sector_characteristics = {
                'Technology': {'beta': 1.2, 'alpha': 0.0005, 'correlation': 0.85},
                'Healthcare': {'beta': 0.9, 'alpha': 0.0002, 'correlation': 0.75},
                'Financials': {'beta': 1.3, 'alpha': -0.0001, 'correlation': 0.80},
                'Consumer Discretionary': {'beta': 1.1, 'alpha': 0.0003, 'correlation': 0.82},
                'Communication Services': {'beta': 1.0, 'alpha': 0.0001, 'correlation': 0.78},
                'Industrials': {'beta': 1.1, 'alpha': 0.0002, 'correlation': 0.83},
                'Consumer Staples': {'beta': 0.7, 'alpha': 0.0001, 'correlation': 0.65},
                'Energy': {'beta': 1.4, 'alpha': -0.0002, 'correlation': 0.70},
                'Utilities': {'beta': 0.6, 'alpha': 0.0000, 'correlation': 0.55},
                'Real Estate': {'beta': 0.8, 'alpha': 0.0001, 'correlation': 0.60},
                'Materials': {'beta': 1.2, 'alpha': 0.0000, 'correlation': 0.85}
            }

            for sector, etf in self.parameters['sector_etfs'].items():
                if sector in sector_characteristics:
                    char = sector_characteristics[sector]

                    # Generate sector returns
                    sector_returns = (char['alpha'] +)
                                    char['beta'] * market_returns * char['correlation'] +
(                                    np.random.normal(0, market_returns.std() * 0.5, len(market_returns)))

                    # Calculate sector prices
                    sector_prices = 100 * (1 + sector_returns).cumprod()

                    sector_df = pd.DataFrame({)
                        'close': sector_prices
(                    }, index=market_data.index[1:])  # Skip first row due to returns calculation

                    sector_data[sector] = sector_df

            return sector_data

        except Exception as e:
            logger.error(f"Error simulating sector data: {str(e)}")
            return {}

def _calculate_sector_momentum(self, sector_data: pd.DataFrame,:)
(                                 benchmark_data: pd.DataFrame, sector_name: str) -> SectorMomentum:
        """Calculate comprehensive momentum metrics for a sector"""
        try:
            if len(sector_data) < max(self.parameters['momentum_windows']):
                return self._create_default_sector_momentum(sector_name)

            sector_prices = sector_data['close']
            benchmark_prices = benchmark_data['close']

            # Calculate momentum for different periods
            momentum_1m = sector_prices.pct_change(self.parameters['momentum_windows'][0]).iloc[-1]
            momentum_3m = sector_prices.pct_change(self.parameters['momentum_windows'][1]).iloc[-1]
            momentum_6m = sector_prices.pct_change(self.parameters['momentum_windows'][2]).iloc[-1]
            momentum_12m = sector_prices.pct_change(self.parameters['momentum_windows'][3]).iloc[-1]

            # Calculate relative strength vs benchmark
            sector_relative = sector_prices / benchmark_prices
            relative_strength = sector_relative.pct_change(self.parameters['relative_strength_window']).iloc[-1]

            # Calculate volatility
            returns = sector_prices.pct_change().dropna()
            volatility = returns.rolling(self.parameters['volatility_window']).std().iloc[-1] * np.sqrt(252)

            # Calculate Sharpe ratio
            excess_returns = returns - (self.parameters['risk_free_rate'] / 252)
            sharpe_ratio = excess_returns.mean() / max(returns.std(), 1e-10) * np.sqrt(252)

            # Calculate maximum drawdown
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()

            # Calculate trend strength using ADX-like calculation
            high_low = abs(sector_prices.diff())
            trend_strength = high_low.rolling(14).mean().iloc[-1] / sector_prices.iloc[-1]

            # Calculate composite momentum score
            momentum_scores = [momentum_1m, momentum_3m, momentum_6m, momentum_12m]
            weights = [0.4, 0.3, 0.2, 0.1]  # Recent momentum weighted more
            momentum_score = sum(score * weight for score, weight in zip(momentum_scores, weights) if pd.notna(score))

            # Calculate reversal probability based on momentum extremes
            momentum_percentile = stats.percentileofscore([momentum_1m, momentum_3m], momentum_score) / 100
            reversal_probability = max(0, momentum_percentile - 0.8) * 5 if momentum_percentile > 0.8 else 0

            # Store momentum history
            if sector_name not in self.momentum_history:
                self.momentum_history[sector_name] = []
            self.momentum_history[sector_name].append(momentum_score)

            # Keep only recent history
            max_history = 100
            if len(self.momentum_history[sector_name]) > max_history:
                self.momentum_history[sector_name] = self.momentum_history[sector_name][-max_history:]

            return SectorMomentum()
                sector=sector_name,
                momentum_1m=momentum_1m if pd.notna(momentum_1m) else 0.0,
                momentum_3m=momentum_3m if pd.notna(momentum_3m) else 0.0,
                momentum_6m=momentum_6m if pd.notna(momentum_6m) else 0.0,
                momentum_12m=momentum_12m if pd.notna(momentum_12m) else 0.0,
                relative_strength=relative_strength if pd.notna(relative_strength) else 0.0,
                volatility=volatility if pd.notna(volatility) else 0.2,
                sharpe_ratio=sharpe_ratio if pd.notna(sharpe_ratio) else 0.0,
                max_drawdown=max_drawdown if pd.notna(max_drawdown) else 0.0,
                momentum_rank=0,  # Will be calculated after all sectors
                momentum_score=momentum_score,
                trend_strength=trend_strength if pd.notna(trend_strength) else 0.0,
                reversal_probability=reversal_probability
(            )

        except Exception as e:
            logger.error(f"Error calculating sector momentum for {sector_name}: {str(e)}")
            return self._create_default_sector_momentum(sector_name)

def _create_default_sector_momentum(self, sector_name: str) -> SectorMomentum:
        """Create default sector momentum data"""
        return SectorMomentum()
            sector=sector_name,
            momentum_1m=0.0,
            momentum_3m=0.0,
            momentum_6m=0.0,
            momentum_12m=0.0,
            relative_strength=0.0,
            volatility=0.2,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            momentum_rank=0,
            momentum_score=0.0,
            trend_strength=0.0,
            reversal_probability=0.0
(        )

def _rank_sectors_by_momentum(self, sector_momentum_list: List[SectorMomentum]) -> List[SectorMomentum]:
        """Rank sectors by their momentum scores"""
        try:
            # Sort by momentum score (descending)
            sorted_sectors = sorted(sector_momentum_list, key=lambda x: x.momentum_score, reverse=True)

            # Assign ranks
            for i, sector in enumerate(sorted_sectors):
                sector.momentum_rank = i + 1

            return sorted_sectors

        except Exception as e:
            logger.error(f"Error ranking sectors: {str(e)}")
            return sector_momentum_list

def _detect_sector_rotations(self, ranked_sectors: List[SectorMomentum]) -> List[SectorRotation]:
        """Detect sector rotation patterns"""
        try:
            rotations = []

            if len(ranked_sectors) < 2:
                return rotations

            # Get top and bottom performers
            top_sectors = ranked_sectors[:3]  # Top 3
            bottom_sectors = ranked_sectors[-3:]  # Bottom 3

            rotation_threshold = self.parameters['rotation_threshold']

            # Detect strong momentum shifts
            for top_sector in top_sectors:
                for bottom_sector in bottom_sectors:
                    momentum_diff = top_sector.momentum_score - bottom_sector.momentum_score

                    if momentum_diff >= rotation_threshold:
                        # Determine rotation type based on sector characteristics
                        rotation_type = self._classify_rotation_type(top_sector.sector, bottom_sector.sector)

                        # Estimate rotation strength and confidence
                        rotation_strength = min(momentum_diff / 0.5, 1.0)  # Normalize to 0-1
                        confidence = min(rotation_strength * 0.8, 0.9)

                        # Estimate duration based on historical patterns
                        duration_estimate = int(30 + (1 - rotation_strength) * 60)  # 30-90 days

                        rotation = SectorRotation()
                            from_sector=bottom_sector.sector,
                            to_sector=top_sector.sector,
                            rotation_strength=rotation_strength,
                            rotation_type=rotation_type,
                            confidence=confidence,
                            duration_estimate=duration_estimate,
                            economic_phase=self.current_economic_phase
(                        )
                        rotations.append(rotation)

            # Sort by rotation strength
            rotations.sort(key=lambda x: x.rotation_strength, reverse=True)
            return rotations[:5]  # Return top 5 rotations

        except Exception as e:
            logger.error(f"Error detecting sector rotations: {str(e)}")
            return []

def _classify_rotation_type(self, strong_sector: str, weak_sector: str) -> str:
        """Classify the type of sector rotation"""
        try:
            # Sector classifications
            cyclical_sectors = ['Energy', 'Materials', 'Industrials', 'Financials']
            defensive_sectors = ['Utilities', 'Consumer Staples', 'Healthcare']
            growth_sectors = ['Technology', 'Consumer Discretionary', 'Communication Services']

            if strong_sector in cyclical_sectors and weak_sector in defensive_sectors:
                return 'cyclical'
            elif strong_sector in defensive_sectors and weak_sector in cyclical_sectors:
                return 'defensive'
            elif strong_sector in growth_sectors:
                return 'growth'
            elif strong_sector in defensive_sectors or strong_sector in ['Real Estate']:
                return 'value'
            else:
                return 'mixed'

        except Exception as e:
            logger.error(f"Error classifying rotation type: {str(e)}")
            return 'mixed'

def _optimize_sector_allocation(self, ranked_sectors: List[SectorMomentum]) -> List[SectorAllocation]:
        """Optimize sector allocation based on momentum and risk"""
        try:
            allocations = []

            if not ranked_sectors:
                return allocations

            # Calculate risk-adjusted momentum scores
            risk_adjusted_scores = []
            for sector in ranked_sectors:
                # Risk-adjusted score = momentum / volatility
                risk_adj_score = sector.momentum_score / max(sector.volatility, 0.05)
                risk_adjusted_scores.append(max(risk_adj_score, 0))

            # Normalize scores to sum to 1
            total_score = sum(risk_adjusted_scores)
            if total_score <= 0:
                equal_weight = 1.0 / len(ranked_sectors)
                normalized_scores = [equal_weight] * len(ranked_sectors)
            else:
                normalized_scores = [score / total_score for score in risk_adjusted_scores]

            # Apply constraints
            min_alloc = self.parameters['min_allocation']
            max_alloc = self.parameters['max_allocation']

            for i, sector in enumerate(ranked_sectors):
                raw_weight = normalized_scores[i]

                # Apply min/max constraints
                recommended_weight = max(min_alloc, min(raw_weight, max_alloc))

                # Current weight (assume equal weighting for baseline)
                current_weight = 1.0 / len(ranked_sectors)
                weight_change = recommended_weight - current_weight

                # Determine allocation reason
                if sector.momentum_rank <= 3:
                    allocation_reason = f"Top momentum performer (rank {sector.momentum_rank})"
                elif sector.momentum_score > 0.1:
                    allocation_reason = "Positive momentum with good risk-adjusted returns"
                elif sector.volatility < 0.15:
                    allocation_reason = "Low volatility defensive play"
                else:
                    allocation_reason = "Minimum allocation for diversification"

                # Expected return based on momentum persistence
                expected_return = sector.momentum_score * self.parameters['momentum_persistence_factor']

                # Risk level
                if sector.volatility > 0.25:
                    risk_level = 'high'
                elif sector.volatility > 0.18:
                    risk_level = 'medium'
                else:
                    risk_level = 'low'

                allocation = SectorAllocation()
                    sector=sector.sector,
                    current_weight=current_weight,
                    recommended_weight=recommended_weight,
                    weight_change=weight_change,
                    allocation_reason=allocation_reason,
                    expected_return=expected_return,
                    risk_level=risk_level
(                )
                allocations.append(allocation)

            # Normalize weights to sum to 1
            total_recommended = sum(alloc.recommended_weight for alloc in allocations)
            if total_recommended > 0:
                for alloc in allocations:
                    alloc.recommended_weight /= total_recommended
                    alloc.weight_change = alloc.recommended_weight - alloc.current_weight

            return allocations

        except Exception as e:
            logger.error(f"Error optimizing sector allocation: {str(e)}")
            return []
def _detect_economic_phase(self, market_data: pd.DataFrame) -> str:
        """Detect current economic phase based on market indicators"""
        try:
            # Simplified economic phase detection
            # In real implementation, would use yield curve, credit spreads, etc.

            returns = market_data['close'].pct_change().dropna()

            # Recent momentum (3-month)
            recent_momentum = returns.tail(63).mean() * 252  # Annualized

            # Volatility regime
            volatility = returns.tail(21).std() * np.sqrt(252)

            # Market trend
            sma_short = market_data['close'].rolling(50).mean().iloc[-1]
            sma_long = market_data['close'].rolling(200).mean().iloc[-1]
            trend_strength = (sma_short - sma_long) / sma_long

            # Phase classification logic
            if recent_momentum > 0.1 and volatility < 0.2 and trend_strength > 0.05:
                return 'early_cycle'
            elif recent_momentum > 0.05 and trend_strength > 0.02:
                return 'mid_cycle'
            elif recent_momentum > 0 and volatility > 0.25:
                return 'late_cycle'
            else:
                return 'recession'

        except Exception as e:
            logger.error(f"Error detecting economic phase: {str(e)}")
            return 'mid_cycle'

def _prepare_ml_features(self, sector_data: Dict[str, pd.DataFrame],:)
(                           market_data: pd.DataFrame) -> np.ndarray:
        """Prepare features for ML prediction model"""
        try:
            features = []

            if not sector_data:
                return np.array([[0.0] * 15])

            # Market features
            market_returns = market_data['close'].pct_change().dropna()
            if len(market_returns) >= 30:
                market_momentum = market_returns.tail(21).mean()
                market_volatility = market_returns.tail(21).std()
                features.extend([market_momentum, market_volatility])
            else:
                features.extend([0.0, 0.2])

            # Sector dispersion
            sector_returns = []
            for sector, data in sector_data.items():
                if len(data) > 1:
                    sector_return = data['close'].pct_change().iloc[-1]
                    if pd.notna(sector_return):
                        sector_returns.append(sector_return)

            if sector_returns:
                sector_dispersion = np.std(sector_returns)
                features.append(sector_dispersion)
            else:
                features.append(0.02)

            # Cross-sectional momentum
            if len(sector_returns) > 1:
                momentum_spread = max(sector_returns) - min(sector_returns)
                features.append(momentum_spread)
            else:
                features.append(0.0)

            # Correlation features
            if len(sector_data) >= 2:
                correlations = []
                sector_list = list(sector_data.keys())
                for i in range(min(3, len(sector_list))):
                    for j in range(i+1, min(3, len(sector_list))):
                        sector1_data = sector_data[sector_list[i]]
                        sector2_data = sector_data[sector_list[j]]

                        if len(sector1_data) > 20 and len(sector2_data) > 20:
                            returns1 = sector1_data['close'].pct_change().tail(20)
                            returns2 = sector2_data['close'].pct_change().tail(20)
                            corr = returns1.corr(returns2)
                            if pd.notna(corr):
                                correlations.append(corr)

                if correlations:
                    avg_correlation = np.mean(correlations)
                    features.append(avg_correlation)
                else:
                    features.append(0.5)
            else:
                features.append(0.5)

            # Technical features
            if 'Technology' in sector_data and len(sector_data['Technology']) > 50:
                tech_data = sector_data['Technology']['close']
                tech_sma_20 = tech_data.rolling(20).mean().iloc[-1]
                tech_sma_50 = tech_data.rolling(50).mean().iloc[-1]
                tech_trend = (tech_sma_20 - tech_sma_50) / tech_sma_50
                features.append(tech_trend)
            else:
                features.append(0.0)

            # Pad or truncate to fixed size
            target_size = 15
            while len(features) < target_size:
                features.append(0.0)
            features = features[:target_size]

            return np.array(features).reshape(1, -1)

        except Exception as e:
            logger.error(f"Error preparing ML features: {str(e)}")
            return np.array([[0.0] * 15])

def _train_sector_prediction_model(self, sector_data: Dict[str, pd.DataFrame], market_data: pd.DataFrame):
        """Train ML model to predict sector momentum"""
        try:
            if len(market_data) < 100:
                return

            X, y = [], []
            window = self.parameters['ml_features_window']
            horizon = self.parameters['prediction_horizon']

            # Prepare training data
            for i in range(window, len(market_data) - horizon):
                # Get historical subset
                subset_market = market_data.iloc[i-window:i]
                subset_sectors = {}

                for sector, data in sector_data.items():
                    if i < len(data):
                        subset_sectors[sector] = data.iloc[max(0, i-window):i]

                if subset_sectors:
                    features = self._prepare_ml_features(subset_sectors, subset_market)

                    # Target: future best performing sector
                    future_returns = {}
                    for sector, data in sector_data.items():
                        if i + horizon < len(data):
                            future_return = (data['close'].iloc[i + horizon] - data['close'].iloc[i]) / data['close'].iloc[i]
                            future_returns[sector] = future_return

                    if future_returns:
                        best_sector = max(future_returns, key=future_returns.get)
                        sectors_list = list(self.parameters['sector_etfs'].keys())
                        target = sectors_list.index(best_sector) if best_sector in sectors_list else 0

                        X.append(features[0])
                        y.append(target)

            if len(X) > 20:
                X = np.array(X)
                y = np.array(y)

                # Scale features
                X_scaled = self.scaler.fit_transform(X)

                # Train classifier
                self.ml_model = RandomForestClassifier()
                    n_estimators=50,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
(                )
                self.ml_model.fit(X_scaled, y)

                logger.debug(f"Sector prediction model trained with {len(X)} samples")

        except Exception as e:
            logger.error(f"Error training sector prediction model: {str(e)}")

def _predict_sector_performance(self, sector_data: Dict[str, pd.DataFrame],:)
(                                  market_data: pd.DataFrame) -> Dict[str, float]:
        """Predict sector performance using ML model"""
        try:
            if self.ml_model is None:
                return {}

            features = self._prepare_ml_features(sector_data, market_data)
            features_scaled = self.scaler.transform(features)

            # Get prediction probabilities
            probabilities = self.ml_model.predict_proba(features_scaled)[0]
            sectors_list = list(self.parameters['sector_etfs'].keys())

            # Create prediction dictionary
            predictions = {}
            for i, sector in enumerate(sectors_list):
                if i < len(probabilities):
                    predictions[sector] = probabilities[i]
                else:
                    predictions[sector] = 0.0

            return predictions

        except Exception as e:
            logger.error(f"Error predicting sector performance: {str(e)}")
            return {}

def calculate_raw(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate comprehensive sector momentum analysis
        """
        try:
            # Simulate sector data (in real implementation, use actual sector ETF data)
            sector_data = self._simulate_sector_data(data)

            if not sector_data:
                return {
                    'sector_momentum': [],
                    'sector_rotations': [],
                    'sector_allocations': [],
                    'economic_phase': 'unknown',
                    'momentum_leader': None,
                    'momentum_laggard': None,
                    'rotation_signals': 0
                }

            # Detect economic phase
            self.current_economic_phase = self._detect_economic_phase(data)

            # Calculate momentum for each sector
            sector_momentum_list = []
            for sector_name, sector_df in sector_data.items():
                momentum = self._calculate_sector_momentum(sector_df, data, sector_name)
                sector_momentum_list.append(momentum)

            # Rank sectors by momentum
            ranked_sectors = self._rank_sectors_by_momentum(sector_momentum_list)

            # Update internal state
            self.sector_momentum_data = {sector.sector: sector for sector in ranked_sectors}

            # Detect sector rotations
            sector_rotations = self._detect_sector_rotations(ranked_sectors)
            self.sector_rotations = sector_rotations

            # Optimize sector allocation
            sector_allocations = self._optimize_sector_allocation(ranked_sectors)
            self.sector_allocations = sector_allocations

            # Train ML model for predictions
            self._train_sector_prediction_model(sector_data, data)

            # Get ML predictions
            ml_predictions = self._predict_sector_performance(sector_data, data)

            # Calculate overall metrics
            momentum_leader = ranked_sectors[0] if ranked_sectors else None
            momentum_laggard = ranked_sectors[-1] if ranked_sectors else None

            # Calculate momentum spread
            momentum_spread = 0.0
            if momentum_leader and momentum_laggard:
                momentum_spread = momentum_leader.momentum_score - momentum_laggard.momentum_score

            # Count significant rotation signals
            significant_rotations = len([r for r in sector_rotations if r.confidence >= 0.7])

            # Calculate portfolio momentum score
            portfolio_momentum = np.mean([sector.momentum_score for sector in ranked_sectors]) if ranked_sectors else 0.0

            # Prepare result
            result = {
                'sector_momentum': [self._sector_momentum_to_dict(sector) for sector in ranked_sectors],
                'sector_rotations': [self._sector_rotation_to_dict(rotation) for rotation in sector_rotations],
                'sector_allocations': [self._sector_allocation_to_dict(allocation) for allocation in sector_allocations],
                'economic_phase': self.current_economic_phase,
                'momentum_leader': self._sector_momentum_to_dict(momentum_leader) if momentum_leader else None,
                'momentum_laggard': self._sector_momentum_to_dict(momentum_laggard) if momentum_laggard else None,
                'momentum_spread': momentum_spread,
                'portfolio_momentum_score': portfolio_momentum,
                'rotation_signals': significant_rotations,
                'ml_predictions': ml_predictions,
                'total_sectors': len(ranked_sectors),
                'ml_model_active': self.ml_model is not None,
                'analysis_timestamp': datetime.utcnow().isoformat()
            }

            return result

        except Exception as e:
            logger.error(f"Error in sector momentum calculation: {str(e)}")
            raise IndicatorCalculationError()
                indicator_name=self.name,
                calculation_step="sector_momentum_calculation",
                message=str(e)
(            )

def _sector_momentum_to_dict(self, sector_momentum: SectorMomentum) -> Dict[str, Any]:
        """Convert SectorMomentum to dictionary"""
        return {
            'sector': sector_momentum.sector,
            'momentum_1m': sector_momentum.momentum_1m,
            'momentum_3m': sector_momentum.momentum_3m,
            'momentum_6m': sector_momentum.momentum_6m,
            'momentum_12m': sector_momentum.momentum_12m,
            'relative_strength': sector_momentum.relative_strength,
            'volatility': sector_momentum.volatility,
            'sharpe_ratio': sector_momentum.sharpe_ratio,
            'max_drawdown': sector_momentum.max_drawdown,
            'momentum_rank': sector_momentum.momentum_rank,
            'momentum_score': sector_momentum.momentum_score,
            'trend_strength': sector_momentum.trend_strength,
            'reversal_probability': sector_momentum.reversal_probability
        }

def _sector_rotation_to_dict(self, rotation: SectorRotation) -> Dict[str, Any]:
        """Convert SectorRotation to dictionary"""
        return {
            'from_sector': rotation.from_sector,
            'to_sector': rotation.to_sector,
            'rotation_strength': rotation.rotation_strength,
            'rotation_type': rotation.rotation_type,
            'confidence': rotation.confidence,
            'duration_estimate': rotation.duration_estimate,
            'economic_phase': rotation.economic_phase
        }

def _sector_allocation_to_dict(self, allocation: SectorAllocation) -> Dict[str, Any]:
        """Convert SectorAllocation to dictionary"""
        return {
            'sector': allocation.sector,
            'current_weight': allocation.current_weight,
            'recommended_weight': allocation.recommended_weight,
            'weight_change': allocation.weight_change,
            'allocation_reason': allocation.allocation_reason,
            'expected_return': allocation.expected_return,
            'risk_level': allocation.risk_level
        }

def _generate_signal(self, value: Dict[str, Any], data: pd.DataFrame) -> Tuple[Optional[SignalType], float]:
        """
        Generate trading signals based on sector momentum analysis
        """
        try:
            portfolio_momentum = value.get('portfolio_momentum_score', 0.0)
            momentum_spread = value.get('momentum_spread', 0.0)
            rotation_signals = value.get('rotation_signals', 0)
            momentum_leader = value.get('momentum_leader')

            if not momentum_leader or rotation_signals == 0:
                return SignalType.NEUTRAL, 0.0

            # Strong portfolio momentum with clear leader = bullish
            if portfolio_momentum > 0.1 and momentum_spread > 0.2:
                confidence = min(portfolio_momentum * 2 + momentum_spread, 1.0)
                return SignalType.BUY, confidence

            # Weak portfolio momentum with defensive rotation = bearish
            elif portfolio_momentum < -0.05 and momentum_spread > 0.15:
                # Check if rotation is defensive
                rotations = value.get('sector_rotations', [])
                defensive_rotations = [r for r in rotations if r.get('rotation_type') == 'defensive']

                if defensive_rotations:
                    confidence = min(abs(portfolio_momentum) * 2 + momentum_spread, 1.0)
                    return SignalType.SELL, confidence * 0.8

            # Strong sector rotation signal
            if rotation_signals >= 2:
                # Get strongest rotation
                rotations = value.get('sector_rotations', [])
                if rotations:
                    strongest_rotation = max(rotations, key=lambda x: x.get('confidence', 0))

                    if strongest_rotation.get('rotation_type') in ['cyclical', 'growth']:
                        return SignalType.BUY, strongest_rotation.get('confidence', 0) * 0.7
                    elif strongest_rotation.get('rotation_type') == 'defensive':
                        return SignalType.SELL, strongest_rotation.get('confidence', 0) * 0.6

            return SignalType.NEUTRAL, 0.0

        except Exception as e:
            logger.error(f"Error generating signal: {str(e)}")
            return SignalType.NEUTRAL, 0.0

def _get_metadata(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get additional metadata for the calculation"""
        base_metadata = super()._get_metadata(data)

        sector_metadata = {
            'total_sectors_analyzed': len(self.parameters['sector_etfs']),
            'momentum_windows': self.parameters['momentum_windows'],
            'sector_rotations_detected': len(self.sector_rotations),
            'current_economic_phase': self.current_economic_phase,
            'ml_model_trained': self.ml_model is not None,
            'momentum_history_length': len(self.momentum_history.get('Technology', [])),
            'benchmark': self.parameters['benchmark']
        }

        base_metadata.update(sector_metadata)
        return base_metadata


def create_sector_momentum_indicator(parameters: Optional[Dict[str, Any]] = None) -> SectorMomentumIndicator:
    """
    Factory function to create a SectorMomentumIndicator instance

    Args:
        parameters: Optional dictionary of parameters to customize the indicator

    Returns:
        Configured SectorMomentumIndicator instance
    """
    return SectorMomentumIndicator(parameters=parameters)


# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')

    # Simulate market data with sector rotation patterns
    base_returns = np.random.randn(len(dates)) * 0.01
    trend = np.linspace(0, 0.3, len(dates))  # Upward trend

    sample_data = pd.DataFrame({)
        'close': 100 * np.exp(np.cumsum(base_returns + trend/252))
(    }, index=dates)

    # Test the indicator
    indicator = create_sector_momentum_indicator({)
        'momentum_windows': [21, 63, 126, 252],
        'rotation_threshold': 0.10
(    })

    try:
        result = indicator.calculate(sample_data)
        print("Sector Momentum Calculation Result:")
        print(f"Signal: {result.signal}, Confidence: {result.confidence:.3f}")
        print(f"Portfolio momentum score: {result.value.get('portfolio_momentum_score', 0):.3f}")
        print(f"Economic phase: {result.value.get('economic_phase', 'unknown')}")
        print(f"Rotation signals: {result.value.get('rotation_signals', 0)}")

        momentum_leader = result.value.get('momentum_leader')
        if momentum_leader:
            print(f"Momentum leader: {momentum_leader['sector']} (score: {momentum_leader['momentum_score']:.3f})")

        momentum_laggard = result.value.get('momentum_laggard')
        if momentum_laggard:
            print(f"Momentum laggard: {momentum_laggard['sector']} (score: {momentum_laggard['momentum_score']:.3f})")

        print(f"Momentum spread: {result.value.get('momentum_spread', 0):.3f}")

    except Exception as e:
        print(f"Error testing indicator: {str(e)}")
