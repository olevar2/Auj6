"""
Parabolic SAR Indicator - Advanced Trend Following with Machine Learning Optimization

This module implements a sophisticated Parabolic SAR (Stop and Reverse) indicator with:
- Adaptive acceleration factors based on market volatility
- Machine learning optimization for parameter selection
- Trend strength analysis and momentum calculation
- Multi-timeframe SAR analysis and aggregation
- Volatility-adjusted SAR calculations
- Advanced signal filtering and noise reduction
- Risk-adjusted position sizing recommendations
- Regime-based parameter adaptation
- Production-grade error handling and logging

Author: AI Enhancement Team
Version: 7.0
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from scipy import optimize
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

@dataclass
class SARState:
    """Current state of the Parabolic SAR calculation."""
    sar_value: float
    trend_direction: int  # 1 for uptrend, -1 for downtrend
    acceleration_factor: float
    extreme_point: float
    periods_in_trend: int
    last_reversal_price: float
    trend_strength: float

@dataclass
class SARSignal:
    """Parabolic SAR signal structure."""
    timestamp: datetime
    sar_value: float
    price: float
    trend_direction: int
    acceleration_factor: float
    trend_strength: float
    signal_strength: float
    confidence: float
    volatility_adjustment: float
    risk_score: float
    position_size_factor: float

class ParabolicSARIndicator:
    """
    Advanced Parabolic SAR Indicator with machine learning optimization and adaptive parameters.

    This indicator provides sophisticated trend following analysis including:
    - Adaptive acceleration factors based on market conditions
    - Machine learning optimization for parameter selection
    - Volatility-adjusted SAR calculations
    - Multi-timeframe analysis and signal aggregation
    - Trend strength analysis with momentum indicators
    - Advanced signal filtering and noise reduction
    - Risk assessment and position sizing recommendations
    - Regime detection for parameter adaptation
    """

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize the Parabolic SAR Indicator.

        Args:
            parameters: Configuration parameters for the indicator
        """
        self.parameters = self._set_default_parameters(parameters or {})
        self.logger = self._setup_logger()

        # Core components
        self.volatility_analyzer = VolatilityAnalyzer(self.parameters)
        self.trend_analyzer = TrendStrengthAnalyzer(self.parameters)
        self.ml_optimizer = MLParameterOptimizer(self.parameters)
        self.signal_filter = SignalFilter(self.parameters)
        self.risk_manager = SARRiskManager(self.parameters)

        # SAR calculation state
        self.sar_state: Optional[SARState] = None
        self.sar_history: List[SARSignal] = []
        self.price_history: List[float] = []
        self.volatility_history: List[float] = []

        # Machine learning models
        self.parameter_predictor = None
        self.trend_classifier = None
        self.scaler = StandardScaler()
        self.is_trained = False

        # Multi-timeframe data
        self.timeframe_data = {}
        self.adaptive_parameters = {}

        # Performance tracking
        self.performance_metrics = {
            'total_signals': 0,
            'profitable_signals': 0,
            'avg_trend_length': 0,
            'max_drawdown': 0
        }

        self.logger.info("Parabolic SAR Indicator initialized with advanced ML optimization")

    def _set_default_parameters(self, user_params: Dict[str, Any]) -> Dict[str, Any]:
        """Set default parameters for the indicator."""
        defaults = {
            # Basic SAR parameters
            'initial_af': 0.02,
            'max_af': 0.20,
            'af_increment': 0.02,
            'lookback_period': 100,

            # Adaptive parameters
            'adaptive_af': True,
            'volatility_adjustment': True,
            'trend_strength_adjustment': True,
            'regime_adaptation': True,

            # Volatility analysis
            'volatility_window': 14,
            'volatility_multiplier': 2.0,
            'atr_period': 14,
            'min_af': 0.005,
            'max_af_adaptive': 0.50,

            # Trend analysis
            'trend_strength_window': 20,
            'momentum_window': 10,
            'trend_confirmation_periods': 3,
            'min_trend_strength': 0.3,

            # Machine learning
            'ml_enabled': True,
            'ml_training_period': 200,
            'ml_retrain_frequency': 50,
            'feature_count': 15,
            'prediction_horizon': 5,

            # Signal filtering
            'signal_smoothing': True,
            'smoothing_factor': 0.3,
            'noise_threshold': 0.001,
            'confirmation_required': True,
            'min_signal_strength': 0.4,

            # Risk management
            'max_position_size': 0.1,
            'risk_adjustment_factor': 0.5,
            'stop_loss_multiplier': 2.0,
            'profit_target_multiplier': 3.0,

            # Multi-timeframe
            'timeframes': ['1h', '4h', '1d'],
            'timeframe_weights': [0.3, 0.5, 0.2],
            'consensus_threshold': 0.6,

            # Performance optimization
            'dynamic_optimization': True,
            'performance_tracking': True,
            'adaptive_learning': True
        }

        # Update with user parameters
        defaults.update(user_params)
        return defaults

    def _setup_logger(self) -> logging.Logger:
        """Setup logging for the indicator."""
        logger = logging.getLogger(f"{self.__class__.__name__}")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def calculate(self, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                 volume: np.ndarray = None, timestamp: np.ndarray = None) -> Dict[str, Any]:
        """
        Calculate advanced Parabolic SAR with machine learning optimization.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            volume: Optional volume data
            timestamp: Optional timestamp data

        Returns:
            Dict containing comprehensive SAR analysis results
        """
        try:
            if len(close) < self.parameters['lookback_period']:
                return self._generate_default_result()

            # Store price history
            self.price_history = close.tolist()

            # Calculate volatility
            volatility_analysis = self._analyze_volatility(high, low, close)

            # Calculate trend strength
            trend_analysis = self._analyze_trend_strength(high, low, close, volume)

            # Optimize parameters using ML if enabled
            if self.parameters['ml_enabled']:
                optimized_params = self._optimize_parameters_ml(high, low, close, volatility_analysis)
            else:
                optimized_params = self._get_base_parameters()

            # Calculate adaptive SAR
            sar_values, sar_signals = self._calculate_adaptive_sar(
                high, low, close, optimized_params, volatility_analysis, trend_analysis
            )

            # Apply signal filtering
            filtered_signals = self._filter_signals(sar_signals)

            # Generate risk assessment
            risk_analysis = self._assess_risks(filtered_signals, volatility_analysis)

            # Calculate multi-timeframe consensus if enabled
            if len(self.parameters['timeframes']) > 1:
                consensus_analysis = self._calculate_timeframe_consensus(high, low, close)
            else:
                consensus_analysis = {'consensus_strength': 1.0, 'consensus_direction': 0}

            # Generate comprehensive results
            current_signal = filtered_signals[-1] if filtered_signals else None

            result = {
                'sar_value': current_signal.sar_value if current_signal else 0.0,
                'trend_direction': current_signal.trend_direction if current_signal else 0,
                'acceleration_factor': current_signal.acceleration_factor if current_signal else optimized_params['current_af'],
                'trend_strength': current_signal.trend_strength if current_signal else 0.0,
                'signal_strength': current_signal.signal_strength if current_signal else 0.0,
                'confidence': current_signal.confidence if current_signal else 0.0,
                'volatility_adjustment': current_signal.volatility_adjustment if current_signal else 1.0,
                'risk_score': current_signal.risk_score if current_signal else 0.0,
                'position_size_factor': current_signal.position_size_factor if current_signal else 0.0,

                # Advanced metrics
                'optimized_initial_af': optimized_params['initial_af'],
                'optimized_max_af': optimized_params['max_af'],
                'current_volatility': volatility_analysis.get('current_volatility', 0.0),
                'volatility_regime': volatility_analysis.get('regime', 'normal'),
                'trend_momentum': trend_analysis.get('momentum', 0.0),
                'trend_consistency': trend_analysis.get('consistency', 0.0),

                # Multi-timeframe analysis
                'consensus_strength': consensus_analysis.get('consensus_strength', 1.0),
                'consensus_direction': consensus_analysis.get('consensus_direction', 0),
                'timeframe_agreement': consensus_analysis.get('agreement_ratio', 1.0),

                # Signal quality metrics
                'signal_quality': self._assess_signal_quality(current_signal, risk_analysis),
                'noise_level': volatility_analysis.get('noise_level', 0.0),
                'trend_maturity': self._calculate_trend_maturity(),

                # Risk metrics
                'max_risk_exposure': risk_analysis.get('max_exposure', 0.0),
                'recommended_stop_loss': risk_analysis.get('stop_loss', 0.0),
                'profit_target': risk_analysis.get('profit_target', 0.0),

                # Performance metrics
                'signal_history_count': len(self.sar_history),
                'recent_performance': self._calculate_recent_performance(),
                'parameter_adaptation_score': self._calculate_adaptation_score(),

                # Detailed analysis
                'volatility_analysis': volatility_analysis,
                'trend_analysis': trend_analysis,
                'optimized_parameters': optimized_params,
                'risk_analysis': risk_analysis,
                'consensus_analysis': consensus_analysis,

                # Raw SAR data
                'sar_values': sar_values.tolist() if isinstance(sar_values, np.ndarray) else sar_values,
                'all_signals': [self._signal_to_dict(sig) for sig in filtered_signals[-10:]],  # Last 10 signals

                # Metadata
                'calculation_timestamp': datetime.now().isoformat(),
                'parameters_used': optimized_params,
                'model_trained': self.is_trained
            }

            return result

        except Exception as e:
            self.logger.error(f"Error calculating Parabolic SAR: {e}")
            return self._generate_error_result(str(e))

    def _analyze_volatility(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, Any]:
        """Analyze market volatility for adaptive parameter adjustment."""
        try:
            # Calculate ATR
            atr_period = self.parameters['atr_period']
            true_range = np.maximum(
                high[1:] - low[1:],
                np.maximum(
                    np.abs(high[1:] - close[:-1]),
                    np.abs(low[1:] - close[:-1])
                )
            )

            atr = pd.Series(true_range).rolling(window=atr_period).mean().fillna(0)
            current_atr = atr.iloc[-1] if len(atr) > 0 else 0

            # Calculate price volatility
            returns = np.diff(np.log(close))
            volatility_window = self.parameters['volatility_window']
            rolling_vol = pd.Series(returns).rolling(window=volatility_window).std().fillna(0)
            current_volatility = rolling_vol.iloc[-1] if len(rolling_vol) > 0 else 0

            # Volatility regime classification
            vol_percentiles = np.percentile(rolling_vol.dropna(), [25, 75])
            if current_volatility > vol_percentiles[1]:
                regime = 'high'
            elif current_volatility < vol_percentiles[0]:
                regime = 'low'
            else:
                regime = 'normal'

            # Calculate noise level
            price_changes = np.diff(close)
            noise_level = np.std(price_changes) / (np.mean(np.abs(price_changes)) + 1e-10)

            # Volatility trend
            if len(rolling_vol) > 5:
                vol_trend = np.polyfit(range(5), rolling_vol.iloc[-5:].values, 1)[0]
            else:
                vol_trend = 0

            self.volatility_history.append(current_volatility)
            if len(self.volatility_history) > 100:
                self.volatility_history = self.volatility_history[-100:]

            return {
                'current_atr': current_atr,
                'current_volatility': current_volatility,
                'regime': regime,
                'noise_level': noise_level,
                'volatility_trend': vol_trend,
                'atr_values': atr.values,
                'volatility_values': rolling_vol.values,
                'volatility_percentiles': vol_percentiles
            }

        except Exception as e:
            self.logger.error(f"Error analyzing volatility: {e}")
            return {'current_atr': 0, 'current_volatility': 0, 'regime': 'normal', 'noise_level': 0}

    def _analyze_trend_strength(self, high: np.ndarray, low: np.ndarray,
                               close: np.ndarray, volume: np.ndarray = None) -> Dict[str, Any]:
        """Analyze trend strength and momentum."""
        try:
            window = self.parameters['trend_strength_window']
            momentum_window = self.parameters['momentum_window']

            # Calculate trend strength using various methods

            # 1. Price momentum
            if len(close) > momentum_window:
                momentum = (close[-1] - close[-momentum_window]) / close[-momentum_window]
            else:
                momentum = 0

            # 2. Directional movement
            if len(high) > window:
                up_moves = np.maximum(high[1:] - high[:-1], 0)
                down_moves = np.maximum(low[:-1] - low[1:], 0)

                dm_plus = pd.Series(up_moves).rolling(window=window).sum()
                dm_minus = pd.Series(down_moves).rolling(window=window).sum()

                trend_strength = abs(dm_plus.iloc[-1] - dm_minus.iloc[-1]) / (dm_plus.iloc[-1] + dm_minus.iloc[-1] + 1e-10)
            else:
                trend_strength = 0

            # 3. Price consistency
            if len(close) > window:
                price_changes = np.diff(close[-window:])
                positive_changes = np.sum(price_changes > 0)
                consistency = abs(positive_changes - (len(price_changes) - positive_changes)) / len(price_changes)
            else:
                consistency = 0

            # 4. Volume-weighted trend (if volume available)
            volume_strength = 0
            if volume is not None and len(volume) > window:
                volume_trend = np.corrcoef(close[-window:], volume[-window:])[0, 1]
                volume_strength = abs(volume_trend) if not np.isnan(volume_trend) else 0

            # 5. Trend acceleration
            if len(close) > window * 2:
                recent_trend = np.polyfit(range(window), close[-window:], 1)[0]
                older_trend = np.polyfit(range(window), close[-window*2:-window], 1)[0]
                acceleration = (recent_trend - older_trend) / (abs(older_trend) + 1e-10)
            else:
                acceleration = 0

            # Combine metrics
            combined_strength = (trend_strength + consistency + volume_strength) / 3

            return {
                'momentum': momentum,
                'trend_strength': trend_strength,
                'consistency': consistency,
                'volume_strength': volume_strength,
                'acceleration': acceleration,
                'combined_strength': combined_strength,
                'trend_direction': 1 if momentum > 0 else -1,
                'strength_category': self._categorize_strength(combined_strength)
            }

        except Exception as e:
            self.logger.error(f"Error analyzing trend strength: {e}")
            return {'momentum': 0, 'trend_strength': 0, 'consistency': 0, 'combined_strength': 0}

    def _categorize_strength(self, strength: float) -> str:
        """Categorize trend strength."""
        if strength > 0.7:
            return 'very_strong'
        elif strength > 0.5:
            return 'strong'
        elif strength > 0.3:
            return 'moderate'
        elif strength > 0.1:
            return 'weak'
        else:
            return 'very_weak'

    def _optimize_parameters_ml(self, high: np.ndarray, low: np.ndarray,
                               close: np.ndarray, volatility_analysis: Dict) -> Dict[str, Any]:
        """Optimize SAR parameters using machine learning."""
        try:
            if not self.ml_optimizer:
                return self._get_base_parameters()

            # Train ML model if needed
            if not self.is_trained and len(close) >= self.parameters['ml_training_period']:
                self._train_ml_models(high, low, close)

            if not self.is_trained:
                return self._get_base_parameters()

            # Generate features for parameter prediction
            features = self._extract_ml_features(high, low, close, volatility_analysis)

            if features is None:
                return self._get_base_parameters()

            # Predict optimal parameters
            optimized_params = self.ml_optimizer.predict_parameters(features)

            # Apply constraints and validation
            optimized_params = self._validate_parameters(optimized_params)

            return optimized_params

        except Exception as e:
            self.logger.error(f"Error optimizing parameters with ML: {e}")
            return self._get_base_parameters()

    def _get_base_parameters(self) -> Dict[str, Any]:
        """Get base SAR parameters."""
        return {
            'initial_af': self.parameters['initial_af'],
            'max_af': self.parameters['max_af'],
            'af_increment': self.parameters['af_increment'],
            'current_af': self.parameters['initial_af'],
            'optimization_method': 'base'
        }

    def _extract_ml_features(self, high: np.ndarray, low: np.ndarray,
                            close: np.ndarray, volatility_analysis: Dict) -> Optional[np.ndarray]:
        """Extract features for machine learning models."""
        try:
            features = []

            # Price-based features
            if len(close) >= 20:
                features.extend([
                    np.mean(close[-20:]),
                    np.std(close[-20:]),
                    (close[-1] - close[-20]) / close[-20],  # 20-period return
                    (close[-1] - close[-5]) / close[-5],   # 5-period return
                ])
            else:
                features.extend([close[-1], 0, 0, 0])

            # Volatility features
            features.extend([
                volatility_analysis.get('current_atr', 0),
                volatility_analysis.get('current_volatility', 0),
                volatility_analysis.get('noise_level', 0),
                volatility_analysis.get('volatility_trend', 0)
            ])

            # Trend features
            if len(close) >= 10:
                trend_slope = np.polyfit(range(10), close[-10:], 1)[0]
                trend_r2 = pearsonr(range(10), close[-10:])[0] ** 2 if len(close) >= 10 else 0
                features.extend([trend_slope, trend_r2])
            else:
                features.extend([0, 0])

            # Range features
            if len(high) >= 5:
                avg_range = np.mean(high[-5:] - low[-5:])
                max_range = np.max(high[-5:] - low[-5:])
                features.extend([avg_range, max_range])
            else:
                features.extend([0, 0])

            # Market structure features
            if len(close) >= 10:
                higher_highs = np.sum(np.diff(close[-10:]) > 0)
                momentum_consistency = higher_highs / 9  # 9 price changes
                features.append(momentum_consistency)
            else:
                features.append(0.5)

            # Ensure exact feature count
            while len(features) < self.parameters['feature_count']:
                features.append(0.0)

            features = features[:self.parameters['feature_count']]

            return np.array(features).reshape(1, -1)

        except Exception as e:
            self.logger.error(f"Error extracting ML features: {e}")
            return None
    def _train_ml_models(self, high: np.ndarray, low: np.ndarray, close: np.ndarray):
        """Train machine learning models for parameter optimization."""
        try:
            if len(close) < self.parameters['ml_training_period']:
                return

            # Generate training data
            X, y = self._generate_training_data(high, low, close)

            if len(X) < 50:  # Minimum samples needed
                return

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Train parameter predictor
            self.parameter_predictor = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            self.parameter_predictor.fit(X_train_scaled, y_train)

            # Evaluate model
            y_pred = self.parameter_predictor.predict(X_test_scaled)
            r2 = r2_score(y_test, y_pred)

            self.logger.info(f"ML model trained with R2 score: {r2:.4f}")

            # Initialize ML optimizer with trained model
            if self.ml_optimizer:
                self.ml_optimizer.set_model(self.parameter_predictor, self.scaler)

            self.is_trained = True

        except Exception as e:
            self.logger.error(f"Error training ML models: {e}")

    def _generate_training_data(self, high: np.ndarray, low: np.ndarray,
                               close: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate training data for ML models."""
        try:
            X, y = [], []

            window = 50

            for i in range(window, len(close) - 10):
                # Extract features for this window
                features = []

                # Price features
                features.extend([
                    np.mean(close[i-20:i]),
                    np.std(close[i-20:i]),
                    (close[i] - close[i-20]) / close[i-20],
                    (close[i] - close[i-5]) / close[i-5]
                ])

                # Volatility features
                true_range = np.maximum(
                    high[i-19:i] - low[i-19:i],
                    np.maximum(
                        np.abs(high[i-19:i] - close[i-20:i-1]),
                        np.abs(low[i-19:i] - close[i-20:i-1])
                    )
                )
                atr = np.mean(true_range)
                vol = np.std(np.diff(np.log(close[i-20:i])))

                features.extend([atr, vol])

                # Trend features
                if i >= 30:
                    trend_slope = np.polyfit(range(10), close[i-10:i], 1)[0]
                    features.append(trend_slope)
                else:
                    features.append(0)

                # Range features
                avg_range = np.mean(high[i-5:i] - low[i-5:i])
                features.append(avg_range)

                # Pad features to required count
                while len(features) < self.parameters['feature_count']:
                    features.append(0.0)
                features = features[:self.parameters['feature_count']]

                # Calculate optimal parameters for this period
                optimal_params = self._calculate_optimal_parameters_historical(
                    high[i:i+10], low[i:i+10], close[i:i+10]
                )

                X.append(features)
                y.append([optimal_params['initial_af'], optimal_params['max_af']])

            return np.array(X), np.array(y)

        except Exception as e:
            self.logger.error(f"Error generating training data: {e}")
            return np.array([]), np.array([])

    def _calculate_optimal_parameters_historical(self, high: np.ndarray,
                                               low: np.ndarray, close: np.ndarray) -> Dict[str, float]:
        """Calculate optimal parameters for historical data."""
        try:
            best_params = {'initial_af': 0.02, 'max_af': 0.20}
            best_score = float('-inf')

            # Grid search over parameter space
            initial_af_range = np.arange(0.01, 0.05, 0.01)
            max_af_range = np.arange(0.10, 0.30, 0.05)

            for initial_af in initial_af_range:
                for max_af in max_af_range:
                    if max_af <= initial_af:
                        continue

                    # Calculate SAR with these parameters
                    score = self._evaluate_parameters(high, low, close, initial_af, max_af)

                    if score > best_score:
                        best_score = score
                        best_params = {'initial_af': initial_af, 'max_af': max_af}

            return best_params

        except Exception as e:
            self.logger.error(f"Error calculating optimal parameters: {e}")
            return {'initial_af': 0.02, 'max_af': 0.20}

    def _evaluate_parameters(self, high: np.ndarray, low: np.ndarray,
                           close: np.ndarray, initial_af: float, max_af: float) -> float:
        """Evaluate SAR parameters on historical data."""
        try:
            if len(close) < 2:
                return 0

            # Simple SAR calculation for evaluation
            sar = np.zeros(len(close))
            trend = np.ones(len(close))
            af = initial_af

            # Initialize
            sar[0] = low[0] if close[1] > close[0] else high[0]
            trend[0] = 1 if close[1] > close[0] else -1
            ep = high[0] if trend[0] == 1 else low[0]

            for i in range(1, len(close)):
                # Calculate SAR
                sar[i] = sar[i-1] + af * (ep - sar[i-1])

                # Check for trend reversal
                if trend[i-1] == 1:  # Uptrend
                    if low[i] <= sar[i]:
                        # Trend reversal to down
                        trend[i] = -1
                        sar[i] = ep
                        af = initial_af
                        ep = low[i]
                    else:
                        trend[i] = 1
                        if high[i] > ep:
                            ep = high[i]
                            af = min(af + 0.02, max_af)

                        # Adjust SAR to not exceed previous lows
                        sar[i] = min(sar[i], min(low[i-1], low[i-2] if i > 1 else low[i-1]))
                else:  # Downtrend
                    if high[i] >= sar[i]:
                        # Trend reversal to up
                        trend[i] = 1
                        sar[i] = ep
                        af = initial_af
                        ep = high[i]
                    else:
                        trend[i] = -1
                        if low[i] < ep:
                            ep = low[i]
                            af = min(af + 0.02, max_af)

                        # Adjust SAR to not exceed previous highs
                        sar[i] = max(sar[i], max(high[i-1], high[i-2] if i > 1 else high[i-1]))

            # Calculate performance score
            signals = np.diff(trend)
            entries = np.where(signals != 0)[0] + 1

            if len(entries) < 2:
                return 0

            total_return = 0
            for j in range(len(entries) - 1):
                entry_idx = entries[j]
                exit_idx = entries[j + 1]

                if trend[entry_idx] == 1:  # Long position
                    trade_return = (close[exit_idx] - close[entry_idx]) / close[entry_idx]
                else:  # Short position
                    trade_return = (close[entry_idx] - close[exit_idx]) / close[entry_idx]

                total_return += trade_return

            return total_return

        except Exception as e:
            self.logger.error(f"Error evaluating parameters: {e}")
            return 0

    def _validate_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and constrain optimized parameters."""
        try:
            validated = {
                'initial_af': np.clip(
                    params.get('initial_af', self.parameters['initial_af']),
                    self.parameters['min_af'],
                    self.parameters['max_af']
                ),
                'max_af': np.clip(
                    params.get('max_af', self.parameters['max_af']),
                    self.parameters['initial_af'],
                    self.parameters['max_af_adaptive']
                ),
                'af_increment': self.parameters['af_increment'],
                'optimization_method': 'ml'
            }

            # Ensure max_af > initial_af
            if validated['max_af'] <= validated['initial_af']:
                validated['max_af'] = validated['initial_af'] + 0.05

            validated['current_af'] = validated['initial_af']

            return validated

        except Exception as e:
            self.logger.error(f"Error validating parameters: {e}")
            return self._get_base_parameters()

    def _calculate_adaptive_sar(self, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                               optimized_params: Dict, volatility_analysis: Dict,
                               trend_analysis: Dict) -> Tuple[np.ndarray, List[SARSignal]]:
        """Calculate adaptive Parabolic SAR with optimized parameters."""
        try:
            n = len(close)
            sar_values = np.zeros(n)
            signals = []

            # Initialize SAR state
            if self.sar_state is None:
                self.sar_state = SARState(
                    sar_value=low[0] if close[1] > close[0] else high[0],
                    trend_direction=1 if close[1] > close[0] else -1,
                    acceleration_factor=optimized_params['initial_af'],
                    extreme_point=high[0] if close[1] > close[0] else low[0],
                    periods_in_trend=1,
                    last_reversal_price=close[0],
                    trend_strength=0.5
                )

            # Copy state for calculation
            current_sar = self.sar_state.sar_value
            current_trend = self.sar_state.trend_direction
            current_af = self.sar_state.acceleration_factor
            current_ep = self.sar_state.extreme_point
            periods_in_trend = self.sar_state.periods_in_trend

            for i in range(n):
                # Calculate base SAR
                next_sar = current_sar + current_af * (current_ep - current_sar)

                # Apply volatility adjustment
                if self.parameters['volatility_adjustment']:
                    vol_multiplier = self._get_volatility_multiplier(volatility_analysis)
                    current_af = np.clip(
                        optimized_params['initial_af'] * vol_multiplier,
                        optimized_params['initial_af'],
                        optimized_params['max_af']
                    )

                # Apply trend strength adjustment
                if self.parameters['trend_strength_adjustment']:
                    strength_multiplier = self._get_strength_multiplier(trend_analysis)
                    current_af = current_af * strength_multiplier

                # Check for trend reversal
                reversal_occurred = False

                if current_trend == 1:  # Uptrend
                    if low[i] <= next_sar:
                        # Trend reversal to downtrend
                        reversal_occurred = True
                        current_trend = -1
                        current_sar = current_ep
                        current_af = optimized_params['initial_af']
                        current_ep = low[i]
                        periods_in_trend = 1
                    else:
                        # Continue uptrend
                        if high[i] > current_ep:
                            current_ep = high[i]
                            current_af = min(current_af + optimized_params['af_increment'],
                                           optimized_params['max_af'])

                        # Prevent SAR from exceeding recent lows
                        if i > 0:
                            next_sar = min(next_sar, low[i-1])
                        if i > 1:
                            next_sar = min(next_sar, low[i-2])

                        current_sar = next_sar
                        periods_in_trend += 1

                else:  # Downtrend
                    if high[i] >= next_sar:
                        # Trend reversal to uptrend
                        reversal_occurred = True
                        current_trend = 1
                        current_sar = current_ep
                        current_af = optimized_params['initial_af']
                        current_ep = high[i]
                        periods_in_trend = 1
                    else:
                        # Continue downtrend
                        if low[i] < current_ep:
                            current_ep = low[i]
                            current_af = min(current_af + optimized_params['af_increment'],
                                           optimized_params['max_af'])

                        # Prevent SAR from going below recent highs
                        if i > 0:
                            next_sar = max(next_sar, high[i-1])
                        if i > 1:
                            next_sar = max(next_sar, high[i-2])

                        current_sar = next_sar
                        periods_in_trend += 1

                sar_values[i] = current_sar

                # Calculate signal metrics
                signal_strength = self._calculate_signal_strength(
                    current_sar, close[i], current_trend, periods_in_trend
                )

                confidence = self._calculate_confidence(
                    volatility_analysis, trend_analysis, periods_in_trend
                )

                # Create signal
                signal = SARSignal(
                    timestamp=datetime.now(),
                    sar_value=current_sar,
                    price=close[i],
                    trend_direction=current_trend,
                    acceleration_factor=current_af,
                    trend_strength=trend_analysis.get('combined_strength', 0.5),
                    signal_strength=signal_strength,
                    confidence=confidence,
                    volatility_adjustment=self._get_volatility_multiplier(volatility_analysis),
                    risk_score=self._calculate_risk_score(current_sar, close[i], current_trend),
                    position_size_factor=self._calculate_position_size(signal_strength, confidence)
                )

                signals.append(signal)

            # Update SAR state
            self.sar_state = SARState(
                sar_value=current_sar,
                trend_direction=current_trend,
                acceleration_factor=current_af,
                extreme_point=current_ep,
                periods_in_trend=periods_in_trend,
                last_reversal_price=close[-1] if len(signals) > 0 and reversal_occurred else self.sar_state.last_reversal_price,
                trend_strength=trend_analysis.get('combined_strength', 0.5)
            )

            return sar_values, signals

        except Exception as e:
            self.logger.error(f"Error calculating adaptive SAR: {e}")
            return np.zeros(len(close)), []

    def _get_volatility_multiplier(self, volatility_analysis: Dict) -> float:
        """Get volatility-based multiplier for acceleration factor."""
        try:
            regime = volatility_analysis.get('regime', 'normal')
            current_vol = volatility_analysis.get('current_volatility', 0)

            if regime == 'high':
                return 0.5  # Reduce AF in high volatility
            elif regime == 'low':
                return 1.5  # Increase AF in low volatility
            else:
                # Gradual adjustment based on volatility level
                vol_percentiles = volatility_analysis.get('volatility_percentiles', [0, 1])
                if len(vol_percentiles) >= 2:
                    vol_range = vol_percentiles[1] - vol_percentiles[0]
                    if vol_range > 0:
                        normalized_vol = (current_vol - vol_percentiles[0]) / vol_range
                        return 1.5 - normalized_vol  # Higher vol = lower multiplier

                return 1.0

        except Exception as e:
            self.logger.error(f"Error calculating volatility multiplier: {e}")
            return 1.0

    def _get_strength_multiplier(self, trend_analysis: Dict) -> float:
        """Get trend strength-based multiplier for acceleration factor."""
        try:
            strength = trend_analysis.get('combined_strength', 0.5)

            # Strong trends get higher acceleration
            if strength > 0.7:
                return 1.3
            elif strength > 0.5:
                return 1.1
            elif strength < 0.2:
                return 0.7
            else:
                return 1.0

        except Exception as e:
            self.logger.error(f"Error calculating strength multiplier: {e}")
            return 1.0
    def _calculate_signal_strength(self, sar_value: float, price: float,
                                  trend_direction: int, periods_in_trend: int) -> float:
        """Calculate signal strength based on SAR and price relationship."""
        try:
            # Distance between SAR and price
            distance = abs(price - sar_value) / price

            # Normalize distance (closer = stronger signal)
            distance_strength = np.exp(-distance * 10)

            # Trend maturity factor
            maturity_factor = min(periods_in_trend / 10, 1.0)

            # Combine factors
            signal_strength = (distance_strength + maturity_factor) / 2

            return np.clip(signal_strength, 0, 1)

        except Exception as e:
            self.logger.error(f"Error calculating signal strength: {e}")
            return 0.5

    def _calculate_confidence(self, volatility_analysis: Dict,
                             trend_analysis: Dict, periods_in_trend: int) -> float:
        """Calculate confidence score for the signal."""
        try:
            # Volatility confidence (lower volatility = higher confidence)
            vol_confidence = 1.0 - min(volatility_analysis.get('current_volatility', 0), 1.0)

            # Trend strength confidence
            trend_confidence = trend_analysis.get('combined_strength', 0.5)

            # Trend maturity confidence
            maturity_confidence = min(periods_in_trend / 5, 1.0)

            # Noise level confidence
            noise_level = volatility_analysis.get('noise_level', 0)
            noise_confidence = 1.0 - min(noise_level, 1.0)

            # Combined confidence
            confidence = (vol_confidence + trend_confidence + maturity_confidence + noise_confidence) / 4

            return np.clip(confidence, 0, 1)

        except Exception as e:
            self.logger.error(f"Error calculating confidence: {e}")
            return 0.5

    def _calculate_risk_score(self, sar_value: float, price: float, trend_direction: int) -> float:
        """Calculate risk score for the current position."""
        try:
            # Distance-based risk
            distance_risk = abs(price - sar_value) / price

            # Trend direction risk (against trend is riskier)
            if trend_direction == 1 and price < sar_value:
                direction_risk = 0.8
            elif trend_direction == -1 and price > sar_value:
                direction_risk = 0.8
            else:
                direction_risk = 0.2

            # Combined risk
            risk_score = (distance_risk + direction_risk) / 2

            return np.clip(risk_score, 0, 1)

        except Exception as e:
            self.logger.error(f"Error calculating risk score: {e}")
            return 0.5

    def _calculate_position_size(self, signal_strength: float, confidence: float) -> float:
        """Calculate recommended position size factor."""
        try:
            # Base position size on signal strength and confidence
            base_size = (signal_strength + confidence) / 2

            # Apply risk adjustment
            risk_adjusted_size = base_size * self.parameters['risk_adjustment_factor']

            # Cap at maximum position size
            final_size = min(risk_adjusted_size, self.parameters['max_position_size'])

            return final_size

        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0.0

    def _filter_signals(self, signals: List[SARSignal]) -> List[SARSignal]:
        """Apply signal filtering and noise reduction."""
        try:
            if not signals:
                return []

            filtered_signals = []

            for i, signal in enumerate(signals):
                # Skip if signal strength too low
                if signal.signal_strength < self.parameters['min_signal_strength']:
                    continue

                # Skip if confidence too low
                if signal.confidence < 0.3:
                    continue

                # Apply noise filtering
                if self.parameters['signal_smoothing'] and filtered_signals:
                    # Smooth with previous signal
                    prev_signal = filtered_signals[-1]
                    smoothing_factor = self.parameters['smoothing_factor']

                    smoothed_sar = (
                        signal.sar_value * smoothing_factor +
                        prev_signal.sar_value * (1 - smoothing_factor)
                    )

                    # Create smoothed signal
                    smoothed_signal = SARSignal(
                        timestamp=signal.timestamp,
                        sar_value=smoothed_sar,
                        price=signal.price,
                        trend_direction=signal.trend_direction,
                        acceleration_factor=signal.acceleration_factor,
                        trend_strength=signal.trend_strength,
                        signal_strength=signal.signal_strength,
                        confidence=signal.confidence,
                        volatility_adjustment=signal.volatility_adjustment,
                        risk_score=signal.risk_score,
                        position_size_factor=signal.position_size_factor
                    )

                    filtered_signals.append(smoothed_signal)
                else:
                    filtered_signals.append(signal)

            # Apply confirmation requirement
            if self.parameters['confirmation_required'] and len(filtered_signals) >= 2:
                confirmed_signals = []

                for i in range(1, len(filtered_signals)):
                    current = filtered_signals[i]
                    previous = filtered_signals[i-1]

                    # Confirm if trend direction is consistent
                    if current.trend_direction == previous.trend_direction:
                        confirmed_signals.append(current)
                    elif current.signal_strength > 0.7:  # Strong signals don't need confirmation
                        confirmed_signals.append(current)

                return confirmed_signals

            return filtered_signals

        except Exception as e:
            self.logger.error(f"Error filtering signals: {e}")
            return signals

    def _assess_risks(self, signals: List[SARSignal], volatility_analysis: Dict) -> Dict[str, Any]:
        """Assess risks associated with current signals."""
        try:
            if not signals:
                return {'max_exposure': 0.0, 'stop_loss': 0.0, 'profit_target': 0.0}

            current_signal = signals[-1]

            # Calculate stop loss
            atr = volatility_analysis.get('current_atr', 0.01)
            if current_signal.trend_direction == 1:
                stop_loss = current_signal.price - (atr * self.parameters['stop_loss_multiplier'])
            else:
                stop_loss = current_signal.price + (atr * self.parameters['stop_loss_multiplier'])

            # Calculate profit target
            if current_signal.trend_direction == 1:
                profit_target = current_signal.price + (atr * self.parameters['profit_target_multiplier'])
            else:
                profit_target = current_signal.price - (atr * self.parameters['profit_target_multiplier'])

            # Calculate maximum exposure
            max_exposure = current_signal.position_size_factor * current_signal.confidence

            # Risk-reward ratio
            risk_amount = abs(current_signal.price - stop_loss)
            reward_amount = abs(profit_target - current_signal.price)
            risk_reward_ratio = reward_amount / (risk_amount + 1e-10)

            return {
                'max_exposure': max_exposure,
                'stop_loss': stop_loss,
                'profit_target': profit_target,
                'risk_reward_ratio': risk_reward_ratio,
                'current_risk_score': current_signal.risk_score,
                'volatility_risk': volatility_analysis.get('regime', 'normal'),
                'trend_risk': 1.0 - current_signal.trend_strength
            }

        except Exception as e:
            self.logger.error(f"Error assessing risks: {e}")
            return {'max_exposure': 0.0, 'stop_loss': 0.0, 'profit_target': 0.0}

    def _calculate_timeframe_consensus(self, high: np.ndarray,
                                     low: np.ndarray, close: np.ndarray) -> Dict[str, Any]:
        """Calculate consensus across multiple timeframes."""
        try:
            if len(self.parameters['timeframes']) <= 1:
                return {'consensus_strength': 1.0, 'consensus_direction': 0, 'agreement_ratio': 1.0}

            timeframe_signals = []
            timeframe_weights = self.parameters['timeframe_weights']

            # Simulate different timeframes by resampling data
            for i, timeframe in enumerate(self.parameters['timeframes']):
                weight = timeframe_weights[i] if i < len(timeframe_weights) else 1.0

                # Simple resampling simulation
                if timeframe == '1h':
                    step = 1
                elif timeframe == '4h':
                    step = 4
                elif timeframe == '1d':
                    step = 24
                else:
                    step = 1

                if step < len(close):
                    resampled_high = high[::step]
                    resampled_low = low[::step]
                    resampled_close = close[::step]

                    if len(resampled_close) > 10:
                        # Calculate simplified SAR for this timeframe
                        tf_result = self._calculate_simple_sar(resampled_high, resampled_low, resampled_close)
                        timeframe_signals.append({
                            'direction': tf_result['direction'],
                            'strength': tf_result['strength'],
                            'weight': weight
                        })

            if not timeframe_signals:
                return {'consensus_strength': 1.0, 'consensus_direction': 0, 'agreement_ratio': 1.0}

            # Calculate weighted consensus
            total_weight = sum(sig['weight'] for sig in timeframe_signals)
            weighted_direction = sum(sig['direction'] * sig['weight'] for sig in timeframe_signals) / total_weight
            weighted_strength = sum(sig['strength'] * sig['weight'] for sig in timeframe_signals) / total_weight

            # Calculate agreement ratio
            positive_directions = sum(1 for sig in timeframe_signals if sig['direction'] > 0)
            negative_directions = sum(1 for sig in timeframe_signals if sig['direction'] < 0)
            agreement_ratio = max(positive_directions, negative_directions) / len(timeframe_signals)

            consensus_direction = 1 if weighted_direction > 0 else -1 if weighted_direction < 0 else 0

            return {
                'consensus_strength': weighted_strength,
                'consensus_direction': consensus_direction,
                'agreement_ratio': agreement_ratio,
                'timeframe_count': len(timeframe_signals),
                'individual_signals': timeframe_signals
            }

        except Exception as e:
            self.logger.error(f"Error calculating timeframe consensus: {e}")
            return {'consensus_strength': 1.0, 'consensus_direction': 0, 'agreement_ratio': 1.0}

    def _calculate_simple_sar(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, Any]:
        """Calculate simplified SAR for timeframe analysis."""
        try:
            if len(close) < 5:
                return {'direction': 0, 'strength': 0.0}

            # Simple trend detection
            recent_trend = np.polyfit(range(min(10, len(close))), close[-min(10, len(close)):], 1)[0]

            # Trend strength based on price consistency
            if len(close) >= 5:
                price_changes = np.diff(close[-5:])
                positive_changes = np.sum(price_changes > 0)
                strength = abs(positive_changes - 2) / 2  # Normalize around 2.5
            else:
                strength = 0.5

            direction = 1 if recent_trend > 0 else -1 if recent_trend < 0 else 0

            return {'direction': direction, 'strength': strength}

        except Exception as e:
            self.logger.error(f"Error calculating simple SAR: {e}")
            return {'direction': 0, 'strength': 0.0}

    def _assess_signal_quality(self, signal: Optional[SARSignal], risk_analysis: Dict) -> float:
        """Assess overall signal quality."""
        try:
            if signal is None:
                return 0.0

            # Component scores
            strength_score = signal.signal_strength
            confidence_score = signal.confidence
            trend_score = signal.trend_strength
            risk_score = 1.0 - signal.risk_score  # Invert risk (lower risk = higher quality)

            # Risk-reward consideration
            rr_ratio = risk_analysis.get('risk_reward_ratio', 1.0)
            rr_score = min(rr_ratio / 2.0, 1.0)  # Normalize around 2:1 ratio

            # Combined quality score
            quality = (strength_score + confidence_score + trend_score + risk_score + rr_score) / 5

            return np.clip(quality, 0, 1)

        except Exception as e:
            self.logger.error(f"Error assessing signal quality: {e}")
            return 0.0

    def _calculate_trend_maturity(self) -> float:
        """Calculate trend maturity score."""
        try:
            if self.sar_state is None:
                return 0.0

            periods = self.sar_state.periods_in_trend

            # Maturity curve: starts low, peaks around 10-15 periods, then declines
            if periods <= 5:
                return periods / 5
            elif periods <= 15:
                return 1.0
            else:
                return max(0.3, 1.0 - (periods - 15) / 20)

        except Exception as e:
            self.logger.error(f"Error calculating trend maturity: {e}")
            return 0.0

    def _calculate_recent_performance(self) -> float:
        """Calculate recent performance score."""
        try:
            if len(self.sar_history) < 10:
                return 0.5

            recent_signals = self.sar_history[-10:]

            # Calculate average signal strength
            avg_strength = np.mean([sig.signal_strength for sig in recent_signals])

            # Calculate average confidence
            avg_confidence = np.mean([sig.confidence for sig in recent_signals])

            # Combined performance
            performance = (avg_strength + avg_confidence) / 2

            return performance

        except Exception as e:
            self.logger.error(f"Error calculating recent performance: {e}")
            return 0.5

    def _calculate_adaptation_score(self) -> float:
        """Calculate parameter adaptation effectiveness score."""
        try:
            if not self.is_trained:
                return 0.0

            # Score based on ML model performance and parameter optimization
            base_score = 0.7 if self.is_trained else 0.3

            # Adjust based on recent volatility adaptation
            if self.parameters['adaptive_af']:
                base_score += 0.1

            if self.parameters['regime_adaptation']:
                base_score += 0.1

            if self.parameters['volatility_adjustment']:
                base_score += 0.1

            return min(base_score, 1.0)

        except Exception as e:
            self.logger.error(f"Error calculating adaptation score: {e}")
            return 0.0

    def _signal_to_dict(self, signal: SARSignal) -> Dict[str, Any]:
        """Convert SARSignal to dictionary."""
        return {
            'timestamp': signal.timestamp.isoformat(),
            'sar_value': signal.sar_value,
            'price': signal.price,
            'trend_direction': signal.trend_direction,
            'acceleration_factor': signal.acceleration_factor,
            'trend_strength': signal.trend_strength,
            'signal_strength': signal.signal_strength,
            'confidence': signal.confidence,
            'volatility_adjustment': signal.volatility_adjustment,
            'risk_score': signal.risk_score,
            'position_size_factor': signal.position_size_factor
        }

    def _generate_default_result(self) -> Dict[str, Any]:
        """Generate default result when insufficient data."""
        return {
            'sar_value': 0.0,
            'trend_direction': 0,
            'acceleration_factor': self.parameters['initial_af'],
            'trend_strength': 0.0,
            'signal_strength': 0.0,
            'confidence': 0.0,
            'error': 'Insufficient data for calculation',
            'calculation_timestamp': datetime.now().isoformat()
        }

    def _generate_error_result(self, error_msg: str) -> Dict[str, Any]:
        """Generate error result."""
        return {
            'sar_value': 0.0,
            'trend_direction': 0,
            'acceleration_factor': self.parameters['initial_af'],
            'trend_strength': 0.0,
            'signal_strength': 0.0,
            'confidence': 0.0,
            'error': error_msg,
            'calculation_timestamp': datetime.now().isoformat()
        }


class VolatilityAnalyzer:
    """Specialized volatility analysis for SAR optimization."""

    def __init__(self, parameters: Dict[str, Any]):
        self.parameters = parameters
        self.logger = logging.getLogger(f"{self.__class__.__name__}")

    def analyze(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, Any]:
        """Analyze volatility patterns for SAR optimization."""
        try:
            if len(high) < 20:
                return {'current_volatility': 0.02, 'regime': 'normal'}

            # Calculate True Range for volatility measurement
            tr1 = high - low
            tr2 = np.abs(high - np.roll(close, 1))
            tr3 = np.abs(low - np.roll(close, 1))
            true_range = np.maximum(tr1, np.maximum(tr2, tr3))

            # Average True Range (ATR) for current volatility
            atr_period = min(14, len(true_range) // 2)
            current_atr = np.mean(true_range[-atr_period:])
            current_volatility = current_atr / np.mean(close[-atr_period:])

            # Volatility regime analysis
            long_atr = np.mean(true_range[-min(50, len(true_range)):])
            volatility_ratio = current_atr / long_atr if long_atr > 0 else 1.0

            if volatility_ratio > 1.5:
                regime = 'high_volatility'
            elif volatility_ratio > 1.2:
                regime = 'elevated'
            elif volatility_ratio < 0.7:
                regime = 'low_volatility'
            elif volatility_ratio < 0.8:
                regime = 'quiet'
            else:
                regime = 'normal'

            # Volatility trend (increasing/decreasing)
            recent_vol = np.mean(true_range[-10:])
            older_vol = np.mean(true_range[-20:-10])
            vol_trend = 'increasing' if recent_vol > older_vol * 1.1 else ('decreasing' if recent_vol < older_vol * 0.9 else 'stable')

            return {
                'current_volatility': float(current_volatility),
                'regime': regime,
                'volatility_ratio': float(volatility_ratio),
                'volatility_trend': vol_trend,
                'atr_normalized': float(current_atr / np.mean(close[-atr_period:]))
            }

        except Exception as e:
            self.logger.error(f"Volatility analysis failed: {e}")
            return {'current_volatility': 0.02, 'regime': 'normal'}


class TrendStrengthAnalyzer:
    """Specialized trend strength analysis for SAR optimization."""

    def __init__(self, parameters: Dict[str, Any]):
        self.parameters = parameters
        self.logger = logging.getLogger(f"{self.__class__.__name__}")

    def analyze(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, Any]:
        """Analyze trend strength for SAR optimization."""
        try:
            if len(close) < 20:
                return {'combined_strength': 0.5, 'momentum': 0.0}

            # 1. Price momentum analysis
            momentum_period = min(14, len(close) // 2)
            price_momentum = (close[-1] - close[-momentum_period]) / close[-momentum_period]

            # 2. Directional movement strength
            high_diff = np.diff(high)
            low_diff = -np.diff(low)

            # Calculate positive and negative directional movement
            pos_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
            neg_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)

            # Smooth directional movements
            smoothing_period = min(14, len(pos_dm))
            pos_dm_smoothed = np.mean(pos_dm[-smoothing_period:])
            neg_dm_smoothed = np.mean(neg_dm[-smoothing_period:])

            # Calculate directional index
            total_dm = pos_dm_smoothed + neg_dm_smoothed
            if total_dm > 0:
                dx = abs(pos_dm_smoothed - neg_dm_smoothed) / total_dm
            else:
                dx = 0

            # 3. Price consistency (trend reliability)
            price_changes = np.diff(close[-min(20, len(close)):])
            positive_moves = np.sum(price_changes > 0)
            negative_moves = np.sum(price_changes < 0)
            total_moves = len(price_changes)

            if total_moves > 0:
                consistency = max(positive_moves, negative_moves) / total_moves
            else:
                consistency = 0.5

            # 4. Trend acceleration analysis
            if len(close) >= 30:
                short_momentum = (close[-1] - close[-10]) / close[-10]
                long_momentum = (close[-1] - close[-20]) / close[-20]
                acceleration = short_momentum - long_momentum
            else:
                acceleration = 0

            # 5. Volatility-adjusted trend strength
            if len(high) >= 14:
                true_range = np.maximum(
                    high[-14:] - low[-14:],
                    np.maximum(
                        np.abs(high[-14:] - np.roll(close[-15:], 1)[1:]),
                        np.abs(low[-14:] - np.roll(close[-15:], 1)[1:])
                    )
                )
                atr = np.mean(true_range)
                volatility_adjusted_momentum = price_momentum / (atr / close[-1]) if atr > 0 else 0
            else:
                volatility_adjusted_momentum = price_momentum

            # 6. Combine all strength metrics
            momentum_strength = np.tanh(abs(price_momentum) * 10)  # Normalize to 0-1
            directional_strength = dx
            trend_acceleration_strength = np.tanh(abs(acceleration) * 20)
            volatility_strength = np.tanh(abs(volatility_adjusted_momentum) * 5)

            # Weighted combination
            combined_strength = (
                momentum_strength * 0.3 +
                directional_strength * 0.25 +
                consistency * 0.2 +
                trend_acceleration_strength * 0.15 +
                volatility_strength * 0.1
            )

            # Determine trend direction
            trend_direction = 1 if price_momentum > 0 else -1 if price_momentum < 0 else 0

            return {
                'combined_strength': float(np.clip(combined_strength, 0, 1)),
                'momentum': float(price_momentum),
                'directional_index': float(dx),
                'consistency': float(consistency),
                'acceleration': float(acceleration),
                'volatility_adjusted_momentum': float(volatility_adjusted_momentum),
                'trend_direction': trend_direction,
                'strength_category': self._categorize_trend_strength(combined_strength)
            }

        except Exception as e:
            self.logger.error(f"Trend strength analysis failed: {e}")
            return {'combined_strength': 0.5, 'momentum': 0.0}

    def _categorize_trend_strength(self, strength: float) -> str:
        """Categorize trend strength level."""
        if strength > 0.8:
            return 'very_strong'
        elif strength > 0.6:
            return 'strong'
        elif strength > 0.4:
            return 'moderate'
        elif strength > 0.2:
            return 'weak'
        else:
            return 'very_weak'


class MLParameterOptimizer:
    """Machine learning-based parameter optimizer for SAR."""

    def __init__(self, parameters: Dict[str, Any]):
        self.parameters = parameters
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.model = None
        self.scaler = None

    def set_model(self, model, scaler):
        """Set the trained model and scaler."""
        self.model = model
        self.scaler = scaler

    def predict_parameters(self, features: np.ndarray) -> Dict[str, Any]:
        """Predict optimal parameters using ML model."""
        try:
            if self.model is None or self.scaler is None:
                return {'initial_af': 0.02, 'max_af': 0.20}

            features_scaled = self.scaler.transform(features)
            prediction = self.model.predict(features_scaled)[0]

            return {
                'initial_af': np.clip(prediction[0], 0.005, 0.05),
                'max_af': np.clip(prediction[1], 0.10, 0.50)
            }

        except Exception as e:
            self.logger.error(f"Error predicting parameters: {e}")
            return {'initial_af': 0.02, 'max_af': 0.20}


class SignalFilter:
    """Advanced signal filtering and noise reduction."""

    def __init__(self, parameters: Dict[str, Any]):
        self.parameters = parameters
        self.logger = logging.getLogger(f"{self.__class__.__name__}")

    def filter(self, signals: List[SARSignal]) -> List[SARSignal]:
        """Apply advanced filtering to SAR signals."""
        try:
            if not signals:
                return []

            filtered_signals = []

            # Apply filtering criteria
            for i, signal in enumerate(signals):
                should_include = True

                # 1. Minimum signal strength filter
                min_strength = self.parameters.get('min_signal_strength', 0.3)
                if signal.signal_strength < min_strength:
                    should_include = False
                    continue

                # 2. Minimum confidence filter
                min_confidence = self.parameters.get('min_confidence', 0.4)
                if signal.confidence < min_confidence:
                    should_include = False
                    continue

                # 3. Risk score filter (avoid high-risk signals)
                max_risk = self.parameters.get('max_risk_score', 0.8)
                if signal.risk_score > max_risk:
                    should_include = False
                    continue

                # 4. Trend consistency filter
                if len(filtered_signals) >= 2:
                    recent_trends = [s.trend_direction for s in filtered_signals[-2:]]
                    if all(trend == signal.trend_direction for trend in recent_trends):
                        # Consistent trend - increase confidence
                        signal.confidence = min(1.0, signal.confidence * 1.1)
                    elif all(trend != signal.trend_direction for trend in recent_trends):
                        # Trend reversal - requires higher strength
                        reversal_threshold = self.parameters.get('reversal_strength_threshold', 0.7)
                        if signal.signal_strength < reversal_threshold:
                            should_include = False
                            continue

                # 5. Noise reduction filter
                if len(filtered_signals) >= 1:
                    prev_signal = filtered_signals[-1]

                    # Calculate signal similarity
                    price_change = abs(signal.price - prev_signal.price) / prev_signal.price
                    sar_change = abs(signal.sar_value - prev_signal.sar_value) / prev_signal.sar_value

                    # Filter out very similar signals (noise)
                    noise_threshold = self.parameters.get('noise_threshold', 0.005)
                    if price_change < noise_threshold and sar_change < noise_threshold:
                        if signal.signal_strength <= prev_signal.signal_strength:
                            should_include = False
                            continue

                # 6. Whipsaw protection
                if len(filtered_signals) >= 3:
                    recent_directions = [s.trend_direction for s in filtered_signals[-3:]]
                    if len(set(recent_directions)) >= 2:  # Direction changes
                        # Potential whipsaw - require higher confidence
                        whipsaw_threshold = self.parameters.get('whipsaw_confidence_threshold', 0.6)
                        if signal.confidence < whipsaw_threshold:
                            should_include = False
                            continue

                # 7. Volume confirmation (if available)
                volume_weight = self.parameters.get('volume_weight', 0.1)
                if hasattr(signal, 'volume') and signal.volume:
                    # Adjust signal strength based on volume
                    avg_volume = self.parameters.get('average_volume', 1000)
                    volume_ratio = signal.volume / avg_volume
                    volume_adjustment = 1.0 + (volume_ratio - 1.0) * volume_weight
                    signal.signal_strength *= volume_adjustment
                    signal.signal_strength = min(1.0, signal.signal_strength)

                if should_include:
                    filtered_signals.append(signal)

            # 8. Apply signal smoothing if enabled
            if self.parameters.get('apply_smoothing', False) and len(filtered_signals) > 1:
                filtered_signals = self._apply_signal_smoothing(filtered_signals)

            # 9. Limit number of signals to prevent overtrading
            max_signals = self.parameters.get('max_signals_per_batch', 10)
            if len(filtered_signals) > max_signals:
                # Keep the highest quality signals
                filtered_signals.sort(key=lambda s: s.signal_strength * s.confidence, reverse=True)
                filtered_signals = filtered_signals[:max_signals]

            return filtered_signals

        except Exception as e:
            self.logger.error(f"Signal filtering failed: {e}")
            return signals  # Return original signals if filtering fails

    def _apply_signal_smoothing(self, signals: List[SARSignal]) -> List[SARSignal]:
        """Apply smoothing to signal values."""
        try:
            if len(signals) < 2:
                return signals

            smoothed_signals = [signals[0]]  # Keep first signal unchanged
            smoothing_factor = self.parameters.get('smoothing_factor', 0.3)

            for i in range(1, len(signals)):
                current = signals[i]
                previous = smoothed_signals[-1]

                # Smooth SAR value
                smoothed_sar = (
                    current.sar_value * smoothing_factor +
                    previous.sar_value * (1 - smoothing_factor)
                )

                # Smooth signal strength
                smoothed_strength = (
                    current.signal_strength * smoothing_factor +
                    previous.signal_strength * (1 - smoothing_factor)
                )

                # Create smoothed signal
                smoothed_signal = SARSignal(
                    timestamp=current.timestamp,
                    sar_value=smoothed_sar,
                    price=current.price,
                    trend_direction=current.trend_direction,
                    acceleration_factor=current.acceleration_factor,
                    trend_strength=current.trend_strength,
                    signal_strength=smoothed_strength,
                    confidence=current.confidence,
                    volatility_adjustment=current.volatility_adjustment,
                    risk_score=current.risk_score,
                    position_size_factor=current.position_size_factor
                )

                smoothed_signals.append(smoothed_signal)

            return smoothed_signals

        except Exception as e:
            self.logger.error(f"Signal smoothing failed: {e}")
            return signals


class SARRiskManager:
    """Risk management for SAR signals."""

    def __init__(self, parameters: Dict[str, Any]):
        self.parameters = parameters
        self.logger = logging.getLogger(f"{self.__class__.__name__}")

    def assess_risk(self, signal: SARSignal, market_data: Dict) -> Dict[str, Any]:
        """Assess risk for SAR signal."""
        try:
            # 1. Price distance risk
            price_distance = abs(signal.price - signal.sar_value) / signal.price
            distance_risk = 1.0 - np.exp(-price_distance * 10)  # Higher distance = higher risk

            # 2. Volatility risk
            volatility = market_data.get('volatility', 0.02)
            volatility_risk = np.clip(volatility / 0.05, 0, 1)  # Normalize to 5% baseline

            # 3. Trend maturity risk
            af = signal.acceleration_factor
            max_af = self.parameters.get('max_af', 0.20)
            maturity_risk = af / max_af  # Higher AF suggests mature trend

            # 4. Market regime risk
            regime = market_data.get('regime', 'normal')
            regime_risk_map = {
                'high_volatility': 0.8,
                'elevated': 0.6,
                'normal': 0.4,
                'quiet': 0.2,
                'low_volatility': 0.1
            }
            regime_risk = regime_risk_map.get(regime, 0.4)

            # 5. Signal confidence risk (inverse)
            confidence_risk = 1.0 - signal.confidence

            # 6. Trend strength risk (inverse)
            strength_risk = 1.0 - signal.trend_strength

            # 7. Position size risk
            position_risk = signal.position_size_factor  # Larger positions = higher risk

            # 8. Market session risk (if available)
            session_risk = 0.3  # Default
            if 'session' in market_data:
                session_risk_map = {
                    'asian': 0.4,    # Lower liquidity
                    'european': 0.2, # High liquidity
                    'american': 0.3, # High liquidity
                    'overlap': 0.1   # Highest liquidity
                }
                session_risk = session_risk_map.get(market_data['session'], 0.3)

            # Weighted risk combination
            weighted_risk = (
                distance_risk * 0.20 +
                volatility_risk * 0.20 +
                maturity_risk * 0.15 +
                regime_risk * 0.15 +
                confidence_risk * 0.10 +
                strength_risk * 0.10 +
                position_risk * 0.05 +
                session_risk * 0.05
            )

            # Risk categorization
            if weighted_risk > 0.8:
                risk_category = 'very_high'
            elif weighted_risk > 0.6:
                risk_category = 'high'
            elif weighted_risk > 0.4:
                risk_category = 'moderate'
            elif weighted_risk > 0.2:
                risk_category = 'low'
            else:
                risk_category = 'very_low'

            # Calculate stop loss and position sizing recommendations
            atr = market_data.get('atr', volatility * signal.price)

            # Dynamic stop loss based on risk level
            stop_loss_multiplier = 1.0 + weighted_risk  # Higher risk = wider stop
            stop_loss_distance = atr * stop_loss_multiplier

            if signal.trend_direction == 1:  # Long position
                stop_loss = signal.price - stop_loss_distance
            else:  # Short position
                stop_loss = signal.price + stop_loss_distance

            # Position size recommendation (inverse of risk)
            max_position_size = self.parameters.get('max_position_size', 0.1)
            recommended_position_size = max_position_size * (1.0 - weighted_risk) * signal.confidence

            # Risk-adjusted profit target
            profit_target_multiplier = 2.0 + (1.0 - weighted_risk)  # Lower risk = higher target
            profit_target_distance = atr * profit_target_multiplier

            if signal.trend_direction == 1:  # Long position
                profit_target = signal.price + profit_target_distance
            else:  # Short position
                profit_target = signal.price - profit_target_distance

            return {
                'risk_score': float(np.clip(weighted_risk, 0, 1)),
                'risk_category': risk_category,
                'component_risks': {
                    'distance_risk': float(distance_risk),
                    'volatility_risk': float(volatility_risk),
                    'maturity_risk': float(maturity_risk),
                    'regime_risk': float(regime_risk),
                    'confidence_risk': float(confidence_risk),
                    'strength_risk': float(strength_risk),
                    'position_risk': float(position_risk),
                    'session_risk': float(session_risk)
                },
                'recommendations': {
                    'stop_loss': float(stop_loss),
                    'profit_target': float(profit_target),
                    'position_size': float(recommended_position_size),
                    'risk_reward_ratio': float(profit_target_distance / stop_loss_distance)
                },
                'risk_management': {
                    'max_drawdown_threshold': 0.05 * (1.0 - weighted_risk),
                    'correlation_limit': 0.7,
                    'position_concentration_limit': 0.2,
                    'daily_var_limit': 0.02 * (1.0 - weighted_risk)
                }
            }

        except Exception as e:
            self.logger.error(f"Risk assessment failed: {e}")
            return {'risk_score': 0.5, 'risk_category': 'moderate'}


# Factory function for easy instantiation
def create_parabolic_sar_indicator(parameters: Optional[Dict[str, Any]] = None) -> ParabolicSARIndicator:
    """
    Factory function to create a Parabolic SAR Indicator instance.

    Args:
        parameters: Optional configuration parameters

    Returns:
        Configured ParabolicSARIndicator instance
    """
    return ParabolicSARIndicator(parameters)


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    import yfinance as yf

    # Create indicator
    sar_indicator = create_parabolic_sar_indicator({
        'ml_enabled': True,
        'adaptive_af': True,
        'volatility_adjustment': True
    })

    # Generate sample data
    np.random.seed(42)
    n_points = 200

    # Simulate price data with trend
    base_price = 100
    price_data = [base_price]

    for i in range(n_points - 1):
        # Add trend and noise
        trend = 0.001 * np.sin(i / 20)  # Cyclical trend
        noise = np.random.normal(0, 0.02)
        price_change = trend + noise
        new_price = price_data[-1] * (1 + price_change)
        price_data.append(new_price)

    prices = np.array(price_data)

    # Create OHLC data
    high = prices * (1 + np.random.uniform(0, 0.02, len(prices)))
    low = prices * (1 - np.random.uniform(0, 0.02, len(prices)))
    close = prices
    volume = np.random.uniform(1000, 10000, len(prices))

    # Calculate SAR
    result = sar_indicator.calculate(high, low, close, volume)

    print("Advanced Parabolic SAR Indicator Results:")
    print(f"SAR Value: {result['sar_value']:.6f}")
    print(f"Trend Direction: {result['trend_direction']}")
    print(f"Signal Strength: {result['signal_strength']:.4f}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Risk Score: {result['risk_score']:.4f}")
    print(f"Position Size Factor: {result['position_size_factor']:.4f}")
    print(f"Optimized Initial AF: {result['optimized_initial_af']:.4f}")
    print(f"Optimized Max AF: {result['optimized_max_af']:.4f}")
    print(f"Volatility Regime: {result['volatility_regime']}")
    print(f"Signal Quality: {result['signal_quality']:.4f}")
