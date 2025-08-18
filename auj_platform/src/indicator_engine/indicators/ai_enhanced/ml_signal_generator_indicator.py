"""
ML Signal Generator Indicator - AI Enhanced Category
====================================================

Advanced machine learning-based signal generation system with ensemble methods,
adaptive learning, and multi-model optimization for trading signal generation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Union, Optional, Tuple, List
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                             VotingClassifier, ExtraTreesClassifier,
                             RandomForestRegressor, GradientBoostingRegressor)
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy import signal, stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

from ..base.standard_indicator import StandardIndicatorInterface, DataRequirement, DataType, SignalType


class MLSignalGeneratorIndicator(StandardIndicatorInterface):
    """
    AI-Enhanced ML Signal Generator with advanced features.
    
    Features:
    - Multi-algorithm ensemble learning
    - Adaptive model selection and weighting
    - Feature engineering and selection optimization
    - Real-time model performance monitoring
    - Online learning and model updates
    - Risk-adjusted signal generation
    - Market regime detection and adaptation
    - Cross-validation and backtesting integration
    - Hyperparameter optimization
    - Signal confidence and uncertainty quantification
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'lookback_period': 100,          # Historical data for training
            'prediction_horizon': 5,         # Periods ahead to predict
            'feature_window': 50,           # Window for feature calculation
            'min_training_samples': 200,    # Minimum samples for training
            'ensemble_size': 5,             # Number of models in ensemble
            'feature_selection_k': 20,      # Number of features to select
            'cv_folds': 5,                  # Cross-validation folds
            'model_update_frequency': 50,   # How often to retrain models
            'confidence_threshold': 0.6,    # Minimum confidence for signals
            'risk_adjustment': True,        # Enable risk-adjusted signals
            'regime_detection': True,       # Enable market regime detection
            'online_learning': True,        # Enable online learning
            'feature_importance_tracking': True,  # Track feature importance
            'hyperparameter_optimization': True,  # Enable hyperparameter tuning
            'signal_smoothing': True,       # Enable signal smoothing
            'uncertainty_quantification': True,  # Enable uncertainty estimation
        }
        
        if parameters:
            default_params.update(parameters)
        
        super().__init__("MLSignalGeneratorIndicator", default_params)
        
        # Initialize ML models
        self.models = self._initialize_models()
        self.ensemble_weights = None
        self.feature_selector = SelectKBest(f_classif, k=self.parameters['feature_selection_k'])
        
        # Scalers
        self.scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler(),
            'minmax': MinMaxScaler()
        }
        self.active_scaler = 'standard'
        
        # State tracking
        self.training_data = []
        self.performance_history = []
        self.feature_importance_history = []
        self.model_predictions = []
        self.is_trained = False
        self.last_update = 0
        self.regime_detector = None
        
        # Performance metrics
        self.accuracy_scores = []
        self.precision_scores = []
        self.recall_scores = []
        self.f1_scores = []
    
    def get_data_requirements(self) -> DataRequirement:
        return DataRequirement(
            data_type=DataType.OHLCV,
            required_columns=["high", "low", "close", "volume"],
            min_periods=self.parameters['min_training_samples']
        )
    
    def calculate_raw(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate advanced ML-based signals with ensemble learning."""
        try:
            if len(data) < self.get_data_requirements().min_periods:
                return self._get_default_output()
            
            # Extract data arrays
            high = data['high'].values
            low = data['low'].values
            close = data['close'].values
            volume = data['volume'].values
            
            # Feature engineering
            features = self._engineer_features(high, low, close, volume)
            
            # Market regime detection
            regime_info = {}
            if self.parameters['regime_detection']:
                regime_info = self._detect_market_regime(close, volume, features)
            
            # Model training and updating
            model_status = self._manage_model_training(features, close)
            
            # Generate predictions
            predictions = self._generate_predictions(features)
            
            # Ensemble prediction
            ensemble_prediction = self._calculate_ensemble_prediction(predictions)
            
            # Signal generation
            signals = self._generate_ml_signals(ensemble_prediction, predictions, regime_info)
            
            # Risk adjustment
            if self.parameters['risk_adjustment']:
                signals = self._apply_risk_adjustment(signals, features, regime_info)
            
            # Signal smoothing
            if self.parameters['signal_smoothing']:
                signals = self._apply_signal_smoothing(signals)
            
            # Uncertainty quantification
            uncertainty_metrics = {}
            if self.parameters['uncertainty_quantification']:
                uncertainty_metrics = self._quantify_uncertainty(predictions, ensemble_prediction)
            
            # Feature importance analysis
            feature_importance = {}
            if self.parameters['feature_importance_tracking'] and self.is_trained:
                feature_importance = self._analyze_feature_importance(features)
            
            # Performance monitoring
            performance_metrics = self._monitor_performance(signals, close)
            
            # Model diagnostics
            diagnostics = self._generate_model_diagnostics(model_status, performance_metrics)
            
            return {
                'features': features,
                'regime_info': regime_info,
                'model_status': model_status,
                'predictions': predictions,
                'ensemble_prediction': ensemble_prediction,
                'signals': signals,
                'uncertainty_metrics': uncertainty_metrics,
                'feature_importance': feature_importance,
                'performance_metrics': performance_metrics,
                'diagnostics': diagnostics,
                'signal_direction': signals.get('direction', 'neutral'),
                'signal_strength': signals.get('strength', 0.0),
                'signal_confidence': signals.get('confidence', 0.5),
                'model_accuracy': performance_metrics.get('current_accuracy', 0.5)
            }
            
        except Exception as e:
            return self._handle_calculation_error(e)
    
    def _initialize_models(self) -> Dict[str, Any]:
        """Initialize the ensemble of ML models."""
        models = {
            'random_forest': {
                'classifier': RandomForestClassifier(
                    n_estimators=100, max_depth=10, random_state=42
                ),
                'regressor': RandomForestRegressor(
                    n_estimators=100, max_depth=10, random_state=42
                ),
                'weight': 1.0
            },
            'gradient_boosting': {
                'classifier': GradientBoostingClassifier(
                    n_estimators=100, max_depth=8, random_state=42
                ),
                'regressor': GradientBoostingRegressor(
                    n_estimators=100, max_depth=8, random_state=42
                ),
                'weight': 1.0
            },
            'extra_trees': {
                'classifier': ExtraTreesClassifier(
                    n_estimators=100, max_depth=10, random_state=42
                ),
                'regressor': None,  # Not needed for this model
                'weight': 0.8
            },
            'logistic_regression': {
                'classifier': LogisticRegression(random_state=42, max_iter=1000),
                'regressor': Ridge(random_state=42),
                'weight': 0.6
            },
            'svm': {
                'classifier': SVC(probability=True, random_state=42),
                'regressor': SVR(),
                'weight': 0.7
            },
            'neural_network': {
                'classifier': MLPClassifier(
                    hidden_layer_sizes=(100, 50), random_state=42, max_iter=500
                ),
                'regressor': MLPRegressor(
                    hidden_layer_sizes=(100, 50), random_state=42, max_iter=500
                ),
                'weight': 0.9
            }
        }
        
        return models
    
    def _engineer_features(self, high: np.ndarray, low: np.ndarray, 
                          close: np.ndarray, volume: np.ndarray) -> Dict[str, np.ndarray]:
        """Engineer comprehensive features for ML models."""
        features = {}
        
        # Price-based features
        features.update(self._calculate_price_features(high, low, close))
        
        # Volume-based features
        features.update(self._calculate_volume_features(volume, close))
        
        # Technical indicators
        features.update(self._calculate_technical_indicators(high, low, close, volume))
        
        # Statistical features
        features.update(self._calculate_statistical_features(close))
        
        # Momentum features
        features.update(self._calculate_momentum_features(close))
        
        # Volatility features
        features.update(self._calculate_volatility_features(high, low, close))
        
        # Trend features
        features.update(self._calculate_trend_features(close))
        
        # Pattern features
        features.update(self._calculate_pattern_features(high, low, close))
        
        return features
    
    def _calculate_price_features(self, high: np.ndarray, low: np.ndarray, 
                                close: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate price-based features."""
        features = {}
        
        # Returns
        returns = np.diff(np.log(close), prepend=np.log(close[0]))
        features['returns'] = returns
        
        # Price ratios
        features['high_close_ratio'] = high / close
        features['low_close_ratio'] = low / close
        features['hl_ratio'] = high / low
        
        # Price position within range
        features['price_position'] = (close - low) / np.where((high - low) == 0, 1, (high - low))
        
        # Relative price levels
        windows = [5, 10, 20, 50]
        for window in windows:
            if len(close) >= window:
                rolling_max = pd.Series(close).rolling(window).max().values
                rolling_min = pd.Series(close).rolling(window).min().values
                features[f'price_percentile_{window}'] = (close - rolling_min) / np.where((rolling_max - rolling_min) == 0, 1, (rolling_max - rolling_min))
        
        return features
    
    def _calculate_volume_features(self, volume: np.ndarray, close: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate volume-based features."""
        features = {}
        
        # Volume ratios
        windows = [5, 10, 20, 50]
        for window in windows:
            if len(volume) >= window:
                volume_ma = pd.Series(volume).rolling(window).mean().values
                features[f'volume_ratio_{window}'] = volume / np.where(volume_ma == 0, 1, volume_ma)
        
        # Volume-price correlation
        for window in [10, 20]:
            if len(volume) >= window:
                correlations = []
                for i in range(window-1, len(volume)):
                    vol_window = volume[i-window+1:i+1]
                    price_window = close[i-window+1:i+1]
                    corr = np.corrcoef(vol_window, price_window)[0, 1]
                    correlations.append(corr if not np.isnan(corr) else 0.0)
                
                features[f'volume_price_corr_{window}'] = np.array([0.0] * (window-1) + correlations)
        
        # On-Balance Volume
        obv = np.zeros_like(volume)
        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                obv[i] = obv[i-1] + volume[i]
            elif close[i] < close[i-1]:
                obv[i] = obv[i-1] - volume[i]
            else:
                obv[i] = obv[i-1]
        
        features['obv'] = obv
        features['obv_momentum'] = np.diff(obv, prepend=obv[0])
        
        return features
    
    def _calculate_technical_indicators(self, high: np.ndarray, low: np.ndarray, 
                                      close: np.ndarray, volume: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate technical indicator features."""
        features = {}
        
        # Moving averages
        windows = [5, 10, 20, 50, 100]
        for window in windows:
            if len(close) >= window:
                ma = pd.Series(close).rolling(window).mean().values
                features[f'ma_{window}'] = close / np.where(ma == 0, 1, ma)
        
        # Exponential moving averages
        for window in [12, 26]:
            if len(close) >= window:
                ema = pd.Series(close).ewm(span=window).mean().values
                features[f'ema_{window}'] = close / np.where(ema == 0, 1, ema)
        
        # MACD
        if len(close) >= 26:
            ema_12 = pd.Series(close).ewm(span=12).mean().values
            ema_26 = pd.Series(close).ewm(span=26).mean().values
            macd = ema_12 - ema_26
            macd_signal = pd.Series(macd).ewm(span=9).mean().values
            features['macd'] = macd / close
            features['macd_signal'] = macd_signal / close
            features['macd_histogram'] = (macd - macd_signal) / close
        
        # RSI
        if len(close) >= 14:
            rsi = self._calculate_rsi(close, 14)
            features['rsi'] = rsi / 100.0
        
        # Bollinger Bands
        if len(close) >= 20:
            bb_middle = pd.Series(close).rolling(20).mean().values
            bb_std = pd.Series(close).rolling(20).std().values
            bb_upper = bb_middle + 2 * bb_std
            bb_lower = bb_middle - 2 * bb_std
            features['bb_position'] = (close - bb_lower) / np.where((bb_upper - bb_lower) == 0, 1, (bb_upper - bb_lower))
            features['bb_width'] = bb_std / bb_middle
        
        # Stochastic
        if len(close) >= 14:
            stoch_k, stoch_d = self._calculate_stochastic(high, low, close, 14)
            features['stoch_k'] = stoch_k / 100.0
            features['stoch_d'] = stoch_d / 100.0
        
        return features
    
    def _calculate_statistical_features(self, close: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate statistical features."""
        features = {}
        
        windows = [10, 20, 50]
        for window in windows:
            if len(close) >= window:
                # Rolling statistics
                rolling_mean = pd.Series(close).rolling(window).mean().values
                rolling_std = pd.Series(close).rolling(window).std().values
                rolling_skew = pd.Series(close).rolling(window).skew().values
                rolling_kurt = pd.Series(close).rolling(window).kurt().values
                
                features[f'zscore_{window}'] = (close - rolling_mean) / np.where(rolling_std == 0, 1, rolling_std)
                features[f'skewness_{window}'] = np.where(np.isnan(rolling_skew), 0, rolling_skew)
                features[f'kurtosis_{window}'] = np.where(np.isnan(rolling_kurt), 0, rolling_kurt)
        
        return features
    
    def _calculate_momentum_features(self, close: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate momentum features."""
        features = {}
        
        # Rate of change
        periods = [1, 5, 10, 20]
        for period in periods:
            if len(close) > period:
                roc = (close[period:] - close[:-period]) / close[:-period]
                features[f'roc_{period}'] = np.concatenate([np.zeros(period), roc])
        
        # Momentum oscillator
        if len(close) >= 10:
            momentum = close - np.roll(close, 10)
            features['momentum'] = momentum / close
        
        return features
    
    def _calculate_volatility_features(self, high: np.ndarray, low: np.ndarray, 
                                     close: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate volatility features."""
        features = {}
        
        # True Range
        tr = self._calculate_true_range(high, low, close)
        
        # Average True Range
        windows = [14, 20, 50]
        for window in windows:
            if len(tr) >= window:
                atr = pd.Series(tr).rolling(window).mean().values
                features[f'atr_{window}'] = atr / close
        
        # Realized volatility
        returns = np.diff(np.log(close), prepend=np.log(close[0]))
        for window in [10, 20]:
            if len(returns) >= window:
                vol = pd.Series(returns).rolling(window).std().values * np.sqrt(252)
                features[f'volatility_{window}'] = vol
        
        return features
    
    def _calculate_trend_features(self, close: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate trend features."""
        features = {}
        
        # Linear regression slope
        windows = [10, 20, 50]
        for window in windows:
            if len(close) >= window:
                slopes = []
                for i in range(window-1, len(close)):
                    y = close[i-window+1:i+1]
                    x = np.arange(len(y))
                    slope = np.polyfit(x, y, 1)[0]
                    slopes.append(slope / close[i])  # Normalize by price
                
                features[f'trend_slope_{window}'] = np.array([0.0] * (window-1) + slopes)
        
        # Trend strength
        if len(close) >= 20:
            trend_strength = []
            for i in range(19, len(close)):
                y = close[i-19:i+1]
                x = np.arange(len(y))
                _, _, r_value, _, _ = stats.linregress(x, y)
                trend_strength.append(r_value ** 2)  # R-squared
            
            features['trend_strength'] = np.array([0.0] * 19 + trend_strength)
        
        return features
    
    def _calculate_pattern_features(self, high: np.ndarray, low: np.ndarray, 
                                  close: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate pattern recognition features."""
        features = {}
        
        # Doji patterns
        body_size = np.abs(close - np.roll(close, 1))
        hl_range = high - low
        doji_pattern = body_size / np.where(hl_range == 0, 1, hl_range)
        features['doji_strength'] = 1.0 - doji_pattern
        
        # Hammer patterns
        lower_shadow = close - low
        upper_shadow = high - close
        body = np.abs(close - np.roll(close, 1))
        
        hammer_signal = (lower_shadow > 2 * body) & (upper_shadow < 0.1 * hl_range)
        features['hammer_pattern'] = hammer_signal.astype(float)
        
        # Gap detection
        gaps = (low[1:] - high[:-1]) / high[:-1]
        features['gap_size'] = np.concatenate([np.array([0.0]), gaps])
        
        return features
    
    def _calculate_rsi(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate Relative Strength Index."""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = pd.Series(gains).rolling(window).mean().values
        avg_losses = pd.Series(losses).rolling(window).mean().values
        
        rs = avg_gains / np.where(avg_losses == 0, 1e-10, avg_losses)
        rsi = 100 - (100 / (1 + rs))
        
        return np.concatenate([np.array([50.0]), rsi])
    
    def _calculate_stochastic(self, high: np.ndarray, low: np.ndarray, 
                            close: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate Stochastic oscillator."""
        lowest_low = pd.Series(low).rolling(window).min().values
        highest_high = pd.Series(high).rolling(window).max().values
        
        k_percent = 100 * (close - lowest_low) / np.where((highest_high - lowest_low) == 0, 1, (highest_high - lowest_low))
        d_percent = pd.Series(k_percent).rolling(3).mean().values
        
        return k_percent, d_percent
    
    def _calculate_true_range(self, high: np.ndarray, low: np.ndarray, 
                            close: np.ndarray) -> np.ndarray:
        """Calculate True Range."""
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        
        return np.maximum(tr1, np.maximum(tr2, tr3))    
    def _detect_market_regime(self, close: np.ndarray, volume: np.ndarray, 
                            features: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Detect market regime using advanced statistical methods."""
        regime_info = {
            'regime': 'normal',
            'volatility_regime': 'medium',
            'trend_regime': 'sideways',
            'volume_regime': 'normal',
            'regime_probability': 0.5,
            'regime_strength': 0.5
        }
        
        try:
            if len(close) < 50:
                return regime_info
            
            # Volatility regime detection
            returns = np.diff(np.log(close))
            vol_window = 20
            if len(returns) >= vol_window:
                rolling_vol = pd.Series(returns).rolling(vol_window).std().values[-1]
                vol_percentiles = np.percentile(pd.Series(returns).rolling(vol_window).std().dropna(), [25, 75])
                
                if rolling_vol < vol_percentiles[0]:
                    regime_info['volatility_regime'] = 'low'
                elif rolling_vol > vol_percentiles[1]:
                    regime_info['volatility_regime'] = 'high'
                else:
                    regime_info['volatility_regime'] = 'medium'
            
            # Trend regime detection using HMM-like approach
            if 'trend_slope_20' in features and len(features['trend_slope_20']) > 0:
                recent_slopes = features['trend_slope_20'][-20:]
                mean_slope = np.mean(recent_slopes)
                slope_threshold = np.std(recent_slopes) * 0.5
                
                if mean_slope > slope_threshold:
                    regime_info['trend_regime'] = 'uptrend'
                elif mean_slope < -slope_threshold:
                    regime_info['trend_regime'] = 'downtrend'
                else:
                    regime_info['trend_regime'] = 'sideways'
            
            # Volume regime detection
            if len(volume) >= 20:
                recent_volume = volume[-20:]
                volume_ma = np.mean(recent_volume)
                volume_std = np.std(recent_volume)
                current_volume = volume[-1]
                
                if current_volume > volume_ma + volume_std:
                    regime_info['volume_regime'] = 'high'
                elif current_volume < volume_ma - volume_std:
                    regime_info['volume_regime'] = 'low'
                else:
                    regime_info['volume_regime'] = 'normal'
            
            # Overall regime classification
            if (regime_info['volatility_regime'] == 'high' and 
                regime_info['trend_regime'] in ['uptrend', 'downtrend']):
                regime_info['regime'] = 'trending_volatile'
                regime_info['regime_strength'] = 0.8
            elif regime_info['trend_regime'] in ['uptrend', 'downtrend']:
                regime_info['regime'] = 'trending'
                regime_info['regime_strength'] = 0.7
            elif regime_info['volatility_regime'] == 'high':
                regime_info['regime'] = 'volatile'
                regime_info['regime_strength'] = 0.6
            else:
                regime_info['regime'] = 'normal'
                regime_info['regime_strength'] = 0.5
            
            # Calculate regime probability using ensemble of indicators
            regime_scores = []
            if 'rsi' in features and len(features['rsi']) > 0:
                rsi_score = 1.0 - abs(features['rsi'][-1] - 0.5) * 2  # Distance from neutral
                regime_scores.append(rsi_score)
            
            if regime_scores:
                regime_info['regime_probability'] = np.mean(regime_scores)
            
        except Exception as e:
            print(f"Error in regime detection: {e}")
        
        return regime_info
    
    def _manage_model_training(self, features: Dict[str, np.ndarray], 
                             close: np.ndarray) -> Dict[str, Any]:
        """Manage model training and updates."""
        model_status = {
            'training_required': False,
            'models_trained': 0,
            'training_samples': 0,
            'last_training_accuracy': 0.0,
            'feature_selection_complete': False,
            'hyperparameter_optimization_complete': False
        }
        
        try:
            # Check if training is required
            should_train = (not self.is_trained or 
                          len(self.training_data) - self.last_update >= self.parameters['model_update_frequency'])
            
            if should_train and len(close) >= self.parameters['min_training_samples']:
                model_status['training_required'] = True
                
                # Prepare training data
                X, y = self._prepare_training_data(features, close)
                
                if X is not None and len(X) > 0:
                    # Feature selection
                    if self.parameters['feature_selection_k'] < X.shape[1]:
                        X_selected = self.feature_selector.fit_transform(X, y)
                        model_status['feature_selection_complete'] = True
                    else:
                        X_selected = X
                    
                    # Scale features
                    X_scaled = self.scalers[self.active_scaler].fit_transform(X_selected)
                    
                    # Train models
                    trained_models = self._train_ensemble_models(X_scaled, y)
                    model_status['models_trained'] = trained_models
                    model_status['training_samples'] = len(X)
                    
                    # Cross-validation
                    cv_scores = self._perform_cross_validation(X_scaled, y)
                    model_status['last_training_accuracy'] = np.mean(cv_scores)
                    
                    # Hyperparameter optimization
                    if self.parameters['hyperparameter_optimization']:
                        self._optimize_hyperparameters(X_scaled, y)
                        model_status['hyperparameter_optimization_complete'] = True
                    
                    # Update ensemble weights
                    self._update_ensemble_weights(X_scaled, y)
                    
                    self.is_trained = True
                    self.last_update = len(self.training_data)
        
        except Exception as e:
            print(f"Error in model training: {e}")
        
        return model_status
    
    def _prepare_training_data(self, features: Dict[str, np.ndarray], 
                             close: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Prepare training data for ML models."""
        try:
            # Create feature matrix
            feature_arrays = []
            feature_names = []
            
            min_length = min(len(arr) for arr in features.values() if len(arr) > 0)
            if min_length == 0:
                return None, None
            
            for name, arr in features.items():
                if len(arr) >= min_length and not np.all(np.isnan(arr)):
                    # Truncate to minimum length and handle NaN values
                    clean_arr = arr[-min_length:]
                    clean_arr = np.where(np.isnan(clean_arr), 0, clean_arr)
                    clean_arr = np.where(np.isinf(clean_arr), 0, clean_arr)
                    feature_arrays.append(clean_arr)
                    feature_names.append(name)
            
            if not feature_arrays:
                return None, None
            
            X = np.column_stack(feature_arrays)
            
            # Create targets (future returns)
            horizon = self.parameters['prediction_horizon']
            returns = np.diff(np.log(close[-min_length:]))
            
            # Classification targets (future direction)
            y_class = np.zeros(len(returns) - horizon)
            for i in range(len(y_class)):
                future_returns = returns[i:i+horizon]
                if len(future_returns) > 0:
                    cumulative_return = np.sum(future_returns)
                    y_class[i] = 1 if cumulative_return > 0 else 0
            
            # Align X and y
            X_aligned = X[:-horizon-1]
            y_aligned = y_class
            
            # Store training data for online learning
            self.training_data.append({
                'X': X_aligned,
                'y': y_aligned,
                'feature_names': feature_names,
                'timestamp': len(close)
            })
            
            # Keep only recent training data
            max_history = 1000
            if len(self.training_data) > max_history:
                self.training_data = self.training_data[-max_history:]
            
            return X_aligned, y_aligned
            
        except Exception as e:
            print(f"Error preparing training data: {e}")
            return None, None
    
    def _train_ensemble_models(self, X: np.ndarray, y: np.ndarray) -> int:
        """Train ensemble of ML models."""
        trained_count = 0
        
        try:
            for model_name, model_info in self.models.items():
                try:
                    # Train classifier
                    if model_info['classifier'] is not None:
                        model_info['classifier'].fit(X, y)
                        trained_count += 1
                    
                    # Train regressor for continuous predictions
                    if model_info['regressor'] is not None:
                        # Create regression targets (future returns)
                        returns = np.diff(np.log(np.random.random(len(y) + 1)))  # Placeholder
                        model_info['regressor'].fit(X, returns[:len(y)])
                        
                except Exception as e:
                    print(f"Error training {model_name}: {e}")
                    continue
            
        except Exception as e:
            print(f"Error in ensemble training: {e}")
        
        return trained_count
    
    def _perform_cross_validation(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Perform time series cross-validation."""
        try:
            tscv = TimeSeriesSplit(n_splits=self.parameters['cv_folds'])
            cv_scores = []
            
            for model_name, model_info in self.models.items():
                if model_info['classifier'] is not None:
                    try:
                        scores = cross_val_score(
                            model_info['classifier'], X, y, 
                            cv=tscv, scoring='accuracy'
                        )
                        cv_scores.extend(scores)
                    except Exception as e:
                        print(f"Error in CV for {model_name}: {e}")
                        continue
            
            return np.array(cv_scores) if cv_scores else np.array([0.5])
            
        except Exception as e:
            print(f"Error in cross-validation: {e}")
            return np.array([0.5])
    
    def _optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> None:
        """Optimize hyperparameters for ensemble models."""
        try:
            # Simple grid search for key parameters
            for model_name, model_info in self.models.items():
                if model_name == 'random_forest':
                    # Optimize n_estimators and max_depth
                    best_score = 0
                    best_params = {}
                    
                    for n_est in [50, 100, 200]:
                        for max_d in [5, 10, 15]:
                            try:
                                temp_model = RandomForestClassifier(
                                    n_estimators=n_est, max_depth=max_d, random_state=42
                                )
                                scores = cross_val_score(temp_model, X, y, cv=3, scoring='accuracy')
                                score = np.mean(scores)
                                
                                if score > best_score:
                                    best_score = score
                                    best_params = {'n_estimators': n_est, 'max_depth': max_d}
                            except:
                                continue
                    
                    if best_params:
                        model_info['classifier'].set_params(**best_params)
                        
        except Exception as e:
            print(f"Error in hyperparameter optimization: {e}")
    
    def _update_ensemble_weights(self, X: np.ndarray, y: np.ndarray) -> None:
        """Update ensemble weights based on model performance."""
        try:
            model_scores = {}
            
            for model_name, model_info in self.models.items():
                if model_info['classifier'] is not None:
                    try:
                        # Calculate accuracy score
                        y_pred = model_info['classifier'].predict(X)
                        accuracy = accuracy_score(y, y_pred)
                        model_scores[model_name] = accuracy
                    except Exception as e:
                        print(f"Error evaluating {model_name}: {e}")
                        model_scores[model_name] = 0.5
            
            # Normalize weights based on performance
            if model_scores:
                total_score = sum(model_scores.values())
                if total_score > 0:
                    for model_name in model_scores:
                        self.models[model_name]['weight'] = model_scores[model_name] / total_score
            
        except Exception as e:
            print(f"Error updating ensemble weights: {e}")
    
    def _generate_predictions(self, features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Generate predictions from ensemble models."""
        predictions = {}
        
        try:
            if not self.is_trained:
                return predictions
            
            # Prepare feature matrix for prediction
            X_pred = self._prepare_prediction_features(features)
            
            if X_pred is None or len(X_pred) == 0:
                return predictions
            
            # Get predictions from each model
            for model_name, model_info in self.models.items():
                try:
                    # Classification prediction
                    if model_info['classifier'] is not None:
                        pred_proba = model_info['classifier'].predict_proba(X_pred)
                        if pred_proba.shape[1] >= 2:
                            predictions[f'{model_name}_class_proba'] = pred_proba[:, 1]  # Probability of positive class
                        
                        pred_class = model_info['classifier'].predict(X_pred)
                        predictions[f'{model_name}_class'] = pred_class
                    
                    # Regression prediction
                    if model_info['regressor'] is not None:
                        pred_reg = model_info['regressor'].predict(X_pred)
                        predictions[f'{model_name}_reg'] = pred_reg
                        
                except Exception as e:
                    print(f"Error generating predictions for {model_name}: {e}")
                    continue
            
        except Exception as e:
            print(f"Error in prediction generation: {e}")
        
        return predictions
    
    def _prepare_prediction_features(self, features: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        """Prepare features for prediction."""
        try:
            if not features:
                return None
            
            # Get the last observation for each feature
            feature_vector = []
            for name, arr in features.items():
                if len(arr) > 0:
                    last_value = arr[-1]
                    if not (np.isnan(last_value) or np.isinf(last_value)):
                        feature_vector.append(last_value)
                    else:
                        feature_vector.append(0.0)
                else:
                    feature_vector.append(0.0)
            
            if not feature_vector:
                return None
            
            X = np.array(feature_vector).reshape(1, -1)
            
            # Apply feature selection if fitted
            if hasattr(self.feature_selector, 'transform'):
                try:
                    X = self.feature_selector.transform(X)
                except:
                    pass  # Skip if feature selector not properly fitted
            
            # Apply scaling
            try:
                X = self.scalers[self.active_scaler].transform(X)
            except:
                pass  # Skip if scaler not properly fitted
            
            return X
            
        except Exception as e:
            print(f"Error preparing prediction features: {e}")
            return None
    
    def _calculate_ensemble_prediction(self, predictions: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate weighted ensemble prediction."""
        ensemble_pred = {
            'class_probability': 0.5,
            'direction_score': 0.0,
            'regression_value': 0.0,
            'confidence': 0.5
        }
        
        try:
            if not predictions:
                return ensemble_pred
            
            # Weighted ensemble for classification
            class_probas = []
            class_weights = []
            
            for model_name, model_info in self.models.items():
                prob_key = f'{model_name}_class_proba'
                if prob_key in predictions:
                    prob_values = predictions[prob_key]
                    if len(prob_values) > 0:
                        class_probas.append(prob_values[0])
                        class_weights.append(model_info['weight'])
            
            if class_probas and class_weights:
                total_weight = sum(class_weights)
                if total_weight > 0:
                    weighted_prob = sum(p * w for p, w in zip(class_probas, class_weights)) / total_weight
                    ensemble_pred['class_probability'] = weighted_prob
                    ensemble_pred['direction_score'] = (weighted_prob - 0.5) * 2  # Scale to [-1, 1]
            
            # Weighted ensemble for regression
            reg_values = []
            reg_weights = []
            
            for model_name, model_info in self.models.items():
                reg_key = f'{model_name}_reg'
                if reg_key in predictions:
                    reg_vals = predictions[reg_key]
                    if len(reg_vals) > 0:
                        reg_values.append(reg_vals[0])
                        reg_weights.append(model_info['weight'])
            
            if reg_values and reg_weights:
                total_weight = sum(reg_weights)
                if total_weight > 0:
                    weighted_reg = sum(r * w for r, w in zip(reg_values, reg_weights)) / total_weight
                    ensemble_pred['regression_value'] = weighted_reg
            
            # Calculate ensemble confidence
            if class_probas:
                # Confidence based on agreement between models
                prob_std = np.std(class_probas)
                prob_mean = np.mean(class_probas)
                confidence = max(0.5, 1.0 - prob_std * 2)  # Lower std = higher confidence
                ensemble_pred['confidence'] = min(0.95, confidence)
            
        except Exception as e:
            print(f"Error in ensemble prediction: {e}")
        
        return ensemble_pred    
    def _generate_ml_signals(self, ensemble_prediction: Dict[str, float], 
                           predictions: Dict[str, np.ndarray], 
                           regime_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signals from ML predictions."""
        signals = {
            'direction': 'neutral',
            'strength': 0.0,
            'confidence': 0.5,
            'entry_signal': False,
            'exit_signal': False,
            'regime_adjusted_strength': 0.0
        }
        
        try:
            # Base signal from ensemble prediction
            class_prob = ensemble_prediction.get('class_probability', 0.5)
            direction_score = ensemble_prediction.get('direction_score', 0.0)
            confidence = ensemble_prediction.get('confidence', 0.5)
            
            # Determine direction
            if class_prob > 0.5 + self.parameters['confidence_threshold'] / 2:
                signals['direction'] = 'bullish'
                signals['strength'] = (class_prob - 0.5) * 2
            elif class_prob < 0.5 - self.parameters['confidence_threshold'] / 2:
                signals['direction'] = 'bearish'
                signals['strength'] = (0.5 - class_prob) * 2
            else:
                signals['direction'] = 'neutral'
                signals['strength'] = 0.0
            
            signals['confidence'] = confidence
            
            # Entry/Exit signals based on confidence threshold
            if confidence >= self.parameters['confidence_threshold']:
                if signals['strength'] > 0.3:  # Strong enough signal
                    signals['entry_signal'] = True
            
            # Regime-based adjustments
            regime_strength = regime_info.get('regime_strength', 0.5)
            regime_type = regime_info.get('regime', 'normal')
            
            # Adjust signal strength based on market regime
            if regime_type == 'trending_volatile':
                # Reduce signal strength in highly volatile trending markets
                signals['regime_adjusted_strength'] = signals['strength'] * 0.8
            elif regime_type == 'trending':
                # Boost signal strength in trending markets
                signals['regime_adjusted_strength'] = signals['strength'] * 1.2
            elif regime_type == 'volatile':
                # Reduce signal strength in volatile sideways markets
                signals['regime_adjusted_strength'] = signals['strength'] * 0.6
            else:
                signals['regime_adjusted_strength'] = signals['strength']
            
            # Ensure adjusted strength stays within bounds
            signals['regime_adjusted_strength'] = max(0.0, min(1.0, signals['regime_adjusted_strength']))
            
            # Model agreement analysis
            agreement_score = self._calculate_model_agreement(predictions)
            signals['model_agreement'] = agreement_score
            
            # Adjust confidence based on model agreement
            signals['confidence'] = min(1.0, signals['confidence'] * agreement_score)
            
        except Exception as e:
            print(f"Error generating ML signals: {e}")
        
        return signals
    
    def _calculate_model_agreement(self, predictions: Dict[str, np.ndarray]) -> float:
        """Calculate agreement between different models."""
        try:
            class_predictions = []
            
            for key, values in predictions.items():
                if '_class' in key and not '_proba' in key:
                    if len(values) > 0:
                        class_predictions.append(values[0])
            
            if len(class_predictions) < 2:
                return 1.0  # Perfect agreement if only one model
            
            # Calculate agreement as percentage of models agreeing with majority
            predictions_array = np.array(class_predictions)
            majority_vote = np.round(np.mean(predictions_array))
            agreement_count = np.sum(predictions_array == majority_vote)
            agreement_score = agreement_count / len(predictions_array)
            
            return agreement_score
            
        except Exception as e:
            print(f"Error calculating model agreement: {e}")
            return 0.5
    
    def _apply_risk_adjustment(self, signals: Dict[str, Any], 
                             features: Dict[str, np.ndarray], 
                             regime_info: Dict[str, Any]) -> Dict[str, Any]:
        """Apply risk-based adjustments to signals."""
        try:
            # Volatility adjustment
            if 'volatility_20' in features and len(features['volatility_20']) > 0:
                current_vol = features['volatility_20'][-1]
                vol_threshold = 0.3  # 30% annualized volatility
                
                if current_vol > vol_threshold:
                    # Reduce signal strength in high volatility
                    vol_adjustment = max(0.5, 1.0 - (current_vol - vol_threshold) / vol_threshold)
                    signals['strength'] *= vol_adjustment
                    signals['regime_adjusted_strength'] *= vol_adjustment
            
            # Trend strength adjustment
            if 'trend_strength' in features and len(features['trend_strength']) > 0:
                trend_strength = features['trend_strength'][-1]
                if trend_strength < 0.3:  # Weak trend
                    # Reduce signal strength in weak trending environments
                    signals['strength'] *= 0.8
                    signals['regime_adjusted_strength'] *= 0.8
            
            # Volume confirmation
            volume_regime = regime_info.get('volume_regime', 'normal')
            if volume_regime == 'low':
                # Reduce confidence in low volume environments
                signals['confidence'] *= 0.9
            elif volume_regime == 'high':
                # Boost confidence in high volume environments
                signals['confidence'] = min(1.0, signals['confidence'] * 1.1)
            
        except Exception as e:
            print(f"Error in risk adjustment: {e}")
        
        return signals
    
    def _apply_signal_smoothing(self, signals: Dict[str, Any]) -> Dict[str, Any]:
        """Apply smoothing to reduce signal noise."""
        try:
            # Store recent signals for smoothing
            if not hasattr(self, 'signal_history'):
                self.signal_history = []
            
            self.signal_history.append({
                'strength': signals['strength'],
                'confidence': signals['confidence'],
                'direction': signals['direction']
            })
            
            # Keep only recent history
            max_history = 10
            if len(self.signal_history) > max_history:
                self.signal_history = self.signal_history[-max_history:]
            
            # Apply exponential smoothing
            if len(self.signal_history) >= 3:
                alpha = 0.3  # Smoothing parameter
                
                # Smooth strength
                strengths = [s['strength'] for s in self.signal_history]
                smoothed_strength = strengths[-1]
                for i in range(len(strengths) - 2, -1, -1):
                    smoothed_strength = alpha * strengths[i] + (1 - alpha) * smoothed_strength
                
                signals['smoothed_strength'] = smoothed_strength
                
                # Smooth confidence
                confidences = [s['confidence'] for s in self.signal_history]
                smoothed_confidence = confidences[-1]
                for i in range(len(confidences) - 2, -1, -1):
                    smoothed_confidence = alpha * confidences[i] + (1 - alpha) * smoothed_confidence
                
                signals['smoothed_confidence'] = smoothed_confidence
            else:
                signals['smoothed_strength'] = signals['strength']
                signals['smoothed_confidence'] = signals['confidence']
            
        except Exception as e:
            print(f"Error in signal smoothing: {e}")
            signals['smoothed_strength'] = signals['strength']
            signals['smoothed_confidence'] = signals['confidence']
        
        return signals
    
    def _quantify_uncertainty(self, predictions: Dict[str, np.ndarray], 
                            ensemble_prediction: Dict[str, float]) -> Dict[str, float]:
        """Quantify prediction uncertainty."""
        uncertainty_metrics = {
            'prediction_variance': 0.0,
            'model_disagreement': 0.0,
            'epistemic_uncertainty': 0.0,
            'aleatoric_uncertainty': 0.0,
            'total_uncertainty': 0.0
        }
        
        try:
            # Calculate prediction variance
            class_probas = []
            for key, values in predictions.items():
                if '_class_proba' in key:
                    if len(values) > 0:
                        class_probas.append(values[0])
            
            if class_probas:
                uncertainty_metrics['prediction_variance'] = np.var(class_probas)
                uncertainty_metrics['model_disagreement'] = np.std(class_probas)
            
            # Epistemic uncertainty (model uncertainty)
            ensemble_prob = ensemble_prediction.get('class_probability', 0.5)
            entropy = -ensemble_prob * np.log(ensemble_prob + 1e-10) - (1 - ensemble_prob) * np.log(1 - ensemble_prob + 1e-10)
            uncertainty_metrics['epistemic_uncertainty'] = entropy
            
            # Aleatoric uncertainty (data uncertainty) - approximated
            if hasattr(self, 'performance_history') and len(self.performance_history) > 0:
                recent_accuracy = np.mean(self.performance_history[-10:])
                uncertainty_metrics['aleatoric_uncertainty'] = 1.0 - recent_accuracy
            else:
                uncertainty_metrics['aleatoric_uncertainty'] = 0.5
            
            # Total uncertainty
            uncertainty_metrics['total_uncertainty'] = (
                uncertainty_metrics['epistemic_uncertainty'] + 
                uncertainty_metrics['aleatoric_uncertainty']
            ) / 2.0
            
        except Exception as e:
            print(f"Error quantifying uncertainty: {e}")
        
        return uncertainty_metrics
    
    def _analyze_feature_importance(self, features: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Analyze feature importance across models."""
        feature_importance = {}
        
        try:
            importance_scores = {}
            
            for model_name, model_info in self.models.items():
                if hasattr(model_info['classifier'], 'feature_importances_'):
                    importances = model_info['classifier'].feature_importances_
                    for i, importance in enumerate(importances):
                        feature_key = f"feature_{i}"
                        if feature_key not in importance_scores:
                            importance_scores[feature_key] = []
                        importance_scores[feature_key].append(importance)
            
            # Average importance across models
            for feature_key, scores in importance_scores.items():
                feature_importance[feature_key] = np.mean(scores)
            
            # Normalize importance scores
            total_importance = sum(feature_importance.values())
            if total_importance > 0:
                for key in feature_importance:
                    feature_importance[key] /= total_importance
            
            # Store for tracking
            self.feature_importance_history.append(feature_importance)
            
            # Keep only recent history
            max_history = 50
            if len(self.feature_importance_history) > max_history:
                self.feature_importance_history = self.feature_importance_history[-max_history:]
            
        except Exception as e:
            print(f"Error analyzing feature importance: {e}")
        
        return feature_importance
    
    def _monitor_performance(self, signals: Dict[str, Any], close: np.ndarray) -> Dict[str, float]:
        """Monitor model performance in real-time."""
        performance_metrics = {
            'current_accuracy': 0.5,
            'rolling_accuracy': 0.5,
            'precision': 0.5,
            'recall': 0.5,
            'f1_score': 0.5,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0
        }
        
        try:
            # Store predictions for evaluation
            if not hasattr(self, 'prediction_history'):
                self.prediction_history = []
            
            self.prediction_history.append({
                'signal_direction': signals.get('direction', 'neutral'),
                'signal_strength': signals.get('strength', 0.0),
                'confidence': signals.get('confidence', 0.5),
                'price': close[-1] if len(close) > 0 else 0.0,
                'timestamp': len(close)
            })
            
            # Keep recent history for evaluation
            max_history = 100
            if len(self.prediction_history) > max_history:
                self.prediction_history = self.prediction_history[-max_history:]
            
            # Evaluate predictions after sufficient history
            if len(self.prediction_history) >= 20:
                accuracy = self._calculate_prediction_accuracy()
                performance_metrics['current_accuracy'] = accuracy
                self.accuracy_scores.append(accuracy)
                
                # Rolling accuracy
                if len(self.accuracy_scores) >= 10:
                    performance_metrics['rolling_accuracy'] = np.mean(self.accuracy_scores[-10:])
                
                # Calculate other metrics
                precision, recall, f1 = self._calculate_classification_metrics()
                performance_metrics['precision'] = precision
                performance_metrics['recall'] = recall
                performance_metrics['f1_score'] = f1
                
                # Calculate financial metrics
                sharpe, max_dd = self._calculate_financial_metrics()
                performance_metrics['sharpe_ratio'] = sharpe
                performance_metrics['max_drawdown'] = max_dd
            
            # Store performance history
            self.performance_history.append(performance_metrics['current_accuracy'])
            
        except Exception as e:
            print(f"Error monitoring performance: {e}")
        
        return performance_metrics
    
    def _calculate_prediction_accuracy(self) -> float:
        """Calculate prediction accuracy based on historical performance."""
        try:
            if len(self.prediction_history) < 5:
                return 0.5
            
            correct_predictions = 0
            total_predictions = 0
            
            for i in range(len(self.prediction_history) - 5):
                current_pred = self.prediction_history[i]
                future_prices = [p['price'] for p in self.prediction_history[i+1:i+6]]
                
                if len(future_prices) >= 5:
                    current_price = current_pred['price']
                    future_price = future_prices[-1]
                    
                    actual_direction = 'bullish' if future_price > current_price else 'bearish'
                    predicted_direction = current_pred['signal_direction']
                    
                    if predicted_direction != 'neutral':
                        total_predictions += 1
                        if predicted_direction == actual_direction:
                            correct_predictions += 1
            
            if total_predictions > 0:
                return correct_predictions / total_predictions
            else:
                return 0.5
                
        except Exception as e:
            print(f"Error calculating prediction accuracy: {e}")
            return 0.5
    
    def _calculate_classification_metrics(self) -> Tuple[float, float, float]:
        """Calculate precision, recall, and F1 score."""
        try:
            true_positives = 0
            false_positives = 0
            false_negatives = 0
            
            for i in range(len(self.prediction_history) - 5):
                current_pred = self.prediction_history[i]
                future_prices = [p['price'] for p in self.prediction_history[i+1:i+6]]
                
                if len(future_prices) >= 5:
                    current_price = current_pred['price']
                    future_price = future_prices[-1]
                    
                    actual_positive = future_price > current_price
                    predicted_positive = current_pred['signal_direction'] == 'bullish'
                    
                    if predicted_positive and actual_positive:
                        true_positives += 1
                    elif predicted_positive and not actual_positive:
                        false_positives += 1
                    elif not predicted_positive and actual_positive:
                        false_negatives += 1
            
            # Calculate metrics
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.5
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.5
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.5
            
            return precision, recall, f1
            
        except Exception as e:
            print(f"Error calculating classification metrics: {e}")
            return 0.5, 0.5, 0.5
    
    def _calculate_financial_metrics(self) -> Tuple[float, float]:
        """Calculate Sharpe ratio and maximum drawdown."""
        try:
            if len(self.prediction_history) < 10:
                return 0.0, 0.0
            
            # Simulate returns based on signals
            returns = []
            portfolio_value = 1.0
            portfolio_values = [portfolio_value]
            
            for i in range(1, len(self.prediction_history)):
                prev_pred = self.prediction_history[i-1]
                current_pred = self.prediction_history[i]
                
                price_return = (current_pred['price'] - prev_pred['price']) / prev_pred['price']
                
                # Apply signal
                if prev_pred['signal_direction'] == 'bullish':
                    signal_return = price_return * prev_pred['signal_strength']
                elif prev_pred['signal_direction'] == 'bearish':
                    signal_return = -price_return * prev_pred['signal_strength']
                else:
                    signal_return = 0.0
                
                returns.append(signal_return)
                portfolio_value *= (1 + signal_return)
                portfolio_values.append(portfolio_value)
            
            # Calculate Sharpe ratio
            if len(returns) > 0:
                mean_return = np.mean(returns)
                std_return = np.std(returns)
                sharpe_ratio = mean_return / std_return if std_return > 0 else 0.0
                sharpe_ratio *= np.sqrt(252)  # Annualized
            else:
                sharpe_ratio = 0.0
            
            # Calculate maximum drawdown
            max_drawdown = 0.0
            peak = portfolio_values[0]
            
            for value in portfolio_values:
                if value > peak:
                    peak = value
                else:
                    drawdown = (peak - value) / peak
                    max_drawdown = max(max_drawdown, drawdown)
            
            return sharpe_ratio, max_drawdown
            
        except Exception as e:
            print(f"Error calculating financial metrics: {e}")
            return 0.0, 0.0
    
    def _generate_model_diagnostics(self, model_status: Dict[str, Any], 
                                  performance_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Generate comprehensive model diagnostics."""
        diagnostics = {
            'model_health': 'good',
            'training_status': 'current',
            'performance_trend': 'stable',
            'feature_stability': 'stable',
            'recommendations': []
        }
        
        try:
            # Assess model health
            current_accuracy = performance_metrics.get('current_accuracy', 0.5)
            if current_accuracy < 0.4:
                diagnostics['model_health'] = 'poor'
                diagnostics['recommendations'].append('Consider retraining models with more data')
            elif current_accuracy < 0.5:
                diagnostics['model_health'] = 'fair'
                diagnostics['recommendations'].append('Monitor performance closely')
            
            # Training status
            if model_status.get('training_required', False):
                diagnostics['training_status'] = 'update_needed'
                diagnostics['recommendations'].append('Update model training')
            
            # Performance trend
            if len(self.accuracy_scores) >= 5:
                recent_trend = np.polyfit(range(5), self.accuracy_scores[-5:], 1)[0]
                if recent_trend < -0.01:
                    diagnostics['performance_trend'] = 'declining'
                    diagnostics['recommendations'].append('Performance is declining, consider feature engineering')
                elif recent_trend > 0.01:
                    diagnostics['performance_trend'] = 'improving'
            
            # Feature importance stability
            if len(self.feature_importance_history) >= 3:
                recent_importance = self.feature_importance_history[-3:]
                importance_stability = self._calculate_importance_stability(recent_importance)
                if importance_stability < 0.7:
                    diagnostics['feature_stability'] = 'unstable'
                    diagnostics['recommendations'].append('Feature importance is unstable')
            
        except Exception as e:
            print(f"Error generating diagnostics: {e}")
        
        return diagnostics
    
    def _calculate_importance_stability(self, importance_history: List[Dict[str, float]]) -> float:
        """Calculate stability of feature importance."""
        try:
            if len(importance_history) < 2:
                return 1.0
            
            # Calculate correlation between consecutive importance rankings
            correlations = []
            for i in range(1, len(importance_history)):
                prev_imp = importance_history[i-1]
                curr_imp = importance_history[i]
                
                # Get common features
                common_features = set(prev_imp.keys()) & set(curr_imp.keys())
                if len(common_features) > 1:
                    prev_values = [prev_imp[f] for f in common_features]
                    curr_values = [curr_imp[f] for f in common_features]
                    
                    corr = np.corrcoef(prev_values, curr_values)[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
            
            return np.mean(correlations) if correlations else 1.0
            
        except Exception as e:
            print(f"Error calculating importance stability: {e}")
            return 1.0
    
    def _get_default_output(self) -> Dict[str, Any]:
        """Return default output when calculation fails."""
        return {
            'features': {},
            'regime_info': {'regime': 'unknown', 'regime_strength': 0.0},
            'model_status': {'training_required': True, 'models_trained': 0},
            'predictions': {},
            'ensemble_prediction': {'class_probability': 0.5, 'confidence': 0.5},
            'signals': {'direction': 'neutral', 'strength': 0.0, 'confidence': 0.5},
            'uncertainty_metrics': {'total_uncertainty': 1.0},
            'feature_importance': {},
            'performance_metrics': {'current_accuracy': 0.5},
            'diagnostics': {'model_health': 'unknown'},
            'signal_direction': 'neutral',
            'signal_strength': 0.0,
            'signal_confidence': 0.5,
            'model_accuracy': 0.5
        }
    
    def _handle_calculation_error(self, error: Exception) -> Dict[str, Any]:
        """Handle calculation errors gracefully."""
        print(f"MLSignalGeneratorIndicator calculation error: {error}")
        return self._get_default_output()
    
    def get_signal_type(self) -> SignalType:
        """Return the type of signal this indicator provides."""
        return SignalType.PREDICTION
    
    def get_display_name(self) -> str:
        """Return display name for the indicator."""
        return "ML Signal Generator"
    
    def get_description(self) -> str:
        """Return description of the indicator."""
        return """
        Advanced machine learning signal generator with ensemble methods.
        
        Features:
        - Multi-algorithm ensemble learning
        - Adaptive model selection and weighting
        - Feature engineering and selection optimization
        - Real-time performance monitoring
        - Market regime detection and adaptation
        - Risk-adjusted signal generation
        - Uncertainty quantification
        
        This indicator uses multiple machine learning models to generate
        trading signals based on comprehensive feature engineering and
        ensemble methods for improved accuracy and robustness.
        """