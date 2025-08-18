"""
Advanced ML Engine Indicator - Multi-Model Machine Learning Ensemble
==================================================================

Sophisticated machine learning ensemble for market prediction using multiple algorithms.
Combines deep learning, ensemble methods, and time series analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

from ..base.standard_indicator import StandardIndicatorInterface, DataRequirement, DataType, SignalType


class AdvancedMLEngineIndicator(StandardIndicatorInterface):
    """
    Advanced machine learning ensemble for market prediction and signal generation.
    
    Features:
    - Multi-algorithm ensemble (Random Forest, Gradient Boosting, Neural Network)
    - Advanced feature engineering with technical indicators
    - Time series cross-validation for robust model selection
    - Dynamic model updating based on performance
    - Uncertainty quantification for risk assessment
    - Market regime-aware predictions
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'prediction_horizon': 5,  # Periods ahead to predict
            'feature_window': 50,     # Historical periods for features
            'retrain_frequency': 200, # Retrain every N periods
            'ensemble_methods': ['rf', 'gb', 'mlp'],
            'n_estimators': 75,       # Reduced from 100 for conservative setting
            'max_depth': 8,           # Reduced from 10 for conservative setting
            'learning_rate': 0.01,
            'hidden_layer_sizes': (75, 40),  # Reduced from (100, 50) for conservative setting
            'cv_folds': 5,
            'min_samples_for_training': 500,
            'uncertainty_threshold': 0.3,
            'feature_selection': True,
            'regime_detection': True,
            'ml_complexity': 'medium', # Conservative ML complexity for production
        }
        if parameters:
            default_params.update(parameters)
        
        super().__init__("AdvancedMLEngineIndicator", default_params)
        
        # Model components
        self.models = {}
        self.scalers = {}
        self.feature_selector = None
        self.is_trained = False
        self.last_training_size = 0
        self.performance_history = []
        self.feature_importance = {}
        
        # Initialize models
        self._initialize_models()
    
    def get_data_requirements(self) -> DataRequirement:
        return DataRequirement(
            data_type=DataType.OHLCV,
            required_columns=['open', 'high', 'low', 'close', 'volume'],
            min_periods=self.parameters['min_samples_for_training'],
            lookback_periods=1000
        )
    
    def calculate_raw(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate ML-based predictions and signals.
        """
        try:
            # Extract features and targets
            features, targets = self._extract_features_and_targets(data)
            
            if len(features) < self.parameters['min_samples_for_training']:
                return self._default_output()
            
            # Check if retraining is needed
            if self._should_retrain(len(features)):
                self._train_models(features, targets)
            
            # Generate predictions if models are trained
            if self.is_trained:
                predictions = self._generate_predictions(features)
                uncertainty = self._calculate_prediction_uncertainty(features, predictions)
                
                # Model performance metrics
                performance = self._calculate_model_performance(features, targets)
                
                # Feature analysis
                feature_analysis = self._analyze_features(features)
                
                # Generate trading signals
                signal_strength = self._calculate_signal_strength(predictions, uncertainty)
                
                # Risk assessment
                risk_metrics = self._calculate_risk_metrics(predictions, uncertainty, data)
                
                return {
                    'predictions': predictions,
                    'prediction_mean': np.mean(predictions['ensemble']),
                    'prediction_std': np.std(predictions['individual']),
                    'uncertainty': uncertainty,
                    'signal_strength': signal_strength,
                    'model_performance': performance,
                    'feature_importance': self.feature_importance,
                    'feature_analysis': feature_analysis,
                    'risk_metrics': risk_metrics,
                    'model_confidence': 1.0 - uncertainty,
                    'prediction_horizon': self.parameters['prediction_horizon'],
                    'training_samples': len(features)
                }
            else:
                return self._default_output()
                
        except Exception as e:
            raise Exception(f"AdvancedMLEngineIndicator calculation failed: {str(e)}")
    
    def _initialize_models(self):
        """Initialize the ensemble of ML models."""
        # Random Forest
        if 'rf' in self.parameters['ensemble_methods']:
            self.models['rf'] = RandomForestRegressor(
                n_estimators=self.parameters['n_estimators'],
                max_depth=self.parameters['max_depth'],
                random_state=42,
                n_jobs=-1
            )
            self.scalers['rf'] = RobustScaler()
        
        # Gradient Boosting
        if 'gb' in self.parameters['ensemble_methods']:
            self.models['gb'] = GradientBoostingRegressor(
                n_estimators=self.parameters['n_estimators'],
                max_depth=self.parameters['max_depth'],
                learning_rate=self.parameters['learning_rate'],
                random_state=42
            )
            self.scalers['gb'] = StandardScaler()
        
        # Multi-layer Perceptron
        if 'mlp' in self.parameters['ensemble_methods']:
            self.models['mlp'] = MLPRegressor(
                hidden_layer_sizes=self.parameters['hidden_layer_sizes'],
                learning_rate_init=self.parameters['learning_rate'],
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            )
            self.scalers['mlp'] = StandardScaler()
    
    def _extract_features_and_targets(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features and targets for ML training."""
        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        volume = data['volume'].values
        open_prices = data['open'].values
        
        features = []
        targets = []
        
        feature_window = self.parameters['feature_window']
        pred_horizon = self.parameters['prediction_horizon']
        
        for i in range(feature_window, len(close) - pred_horizon):
            # Price-based features
            window_close = close[i-feature_window:i]
            window_high = high[i-feature_window:i]
            window_low = low[i-feature_window:i]
            window_volume = volume[i-feature_window:i]
            window_open = open_prices[i-feature_window:i]
            
            feature_vector = self._calculate_technical_features(
                window_open, window_high, window_low, window_close, window_volume
            )
            
            # Target: Future return
            future_price = close[i + pred_horizon]
            current_price = close[i]
            target = (future_price - current_price) / current_price
            
            features.append(feature_vector)
            targets.append(target)
        
        return np.array(features), np.array(targets)
    
    def _calculate_technical_features(self, open_prices: np.ndarray, high: np.ndarray, 
                                    low: np.ndarray, close: np.ndarray, 
                                    volume: np.ndarray) -> List[float]:
        """Calculate comprehensive technical indicator features."""
        features = []
        
        # Price features
        returns = np.diff(close) / close[:-1]
        features.extend([
            np.mean(returns),           # Average return
            np.std(returns),            # Volatility
            np.skew(returns) if len(returns) > 2 else 0,  # Skewness
            np.kurtosis(returns) if len(returns) > 2 else 0,  # Kurtosis
            (close[-1] - close[0]) / close[0],  # Total return
        ])
        
        # Moving averages
        for period in [5, 10, 20]:
            if len(close) >= period:
                ma = np.mean(close[-period:])
                features.append(close[-1] / ma - 1)  # Price vs MA ratio
        
        # Momentum indicators
        if len(close) >= 14:
            # RSI
            deltas = np.diff(close)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            avg_gain = np.mean(gains[-14:])
            avg_loss = np.mean(losses[-14:])
            rsi = 100 - (100 / (1 + avg_gain / (avg_loss + 1e-8)))
            features.append(rsi / 100 - 0.5)  # Normalized RSI
        else:
            features.append(0)
        
        # Volume features
        features.extend([
            np.log(volume[-1] / (np.mean(volume) + 1e-8)),  # Volume ratio (log)
            np.corrcoef(close, volume)[0, 1] if len(close) > 1 else 0,  # Price-volume correlation
        ])
        
        # Volatility features
        if len(high) >= 20:
            true_range = np.maximum(
                high[1:] - low[1:],
                np.maximum(
                    np.abs(high[1:] - close[:-1]),
                    np.abs(low[1:] - close[:-1])
                )
            )
            atr = np.mean(true_range[-14:])
            features.append(atr / close[-1])  # Normalized ATR
        else:
            features.append(0)
        
        # Pattern features
        if len(close) >= 5:
            # Higher highs, lower lows
            recent_highs = high[-5:]
            recent_lows = low[-5:]
            higher_highs = np.sum(np.diff(recent_highs) > 0) / 4
            lower_lows = np.sum(np.diff(recent_lows) < 0) / 4
            features.extend([higher_highs, lower_lows])
        else:
            features.extend([0, 0])
        
        # Trend features
        if len(close) >= 10:
            # Linear regression slope
            x = np.arange(len(close))
            slope = np.polyfit(x, close, 1)[0]
            features.append(slope / (np.mean(close) + 1e-8))  # Normalized slope
        else:
            features.append(0)
        
        # Seasonality features (day of week, hour if available)
        # For now, using simple cyclical features
        features.extend([
            np.sin(2 * np.pi * len(close) / 7),   # Weekly cycle
            np.cos(2 * np.pi * len(close) / 7),   # Weekly cycle
        ])
        
        return features
    
    def _should_retrain(self, current_size: int) -> bool:
        """Determine if models should be retrained."""
        if not self.is_trained:
            return True
        
        # Retrain based on frequency
        if current_size - self.last_training_size >= self.parameters['retrain_frequency']:
            return True
        
        # Retrain if performance has degraded
        if len(self.performance_history) >= 5:
            recent_performance = np.mean(self.performance_history[-3:])
            historical_performance = np.mean(self.performance_history[:-3])
            if recent_performance < historical_performance * 0.8:  # 20% degradation
                return True
        
        return False
    
    def _train_models(self, features: np.ndarray, targets: np.ndarray):
        """Train the ensemble of ML models."""
        try:
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=self.parameters['cv_folds'])
            
            model_scores = {}
            
            for model_name, model in self.models.items():
                scaler = self.scalers[model_name]
                scores = []
                
                for train_idx, val_idx in tscv.split(features):
                    X_train, X_val = features[train_idx], features[val_idx]
                    y_train, y_val = targets[train_idx], targets[val_idx]
                    
                    # Scale features
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_val_scaled = scaler.transform(X_val)
                    
                    # Train model
                    model.fit(X_train_scaled, y_train)
                    
                    # Validate
                    y_pred = model.predict(X_val_scaled)
                    score = r2_score(y_val, y_pred)
                    scores.append(score)
                
                model_scores[model_name] = np.mean(scores)
            
            # Final training on all data
            for model_name, model in self.models.items():
                scaler = self.scalers[model_name]
                X_scaled = scaler.fit_transform(features)
                model.fit(X_scaled, targets)
                
                # Store feature importance if available
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[model_name] = model.feature_importances_
            
            self.is_trained = True
            self.last_training_size = len(features)
            self.performance_history.append(np.mean(list(model_scores.values())))
            
        except Exception as e:
            print(f"Training failed: {str(e)}")
    
    def _generate_predictions(self, features: np.ndarray) -> Dict[str, Any]:
        """Generate predictions from the ensemble."""
        if len(features) == 0:
            return {'ensemble': 0.0, 'individual': {}}
        
        latest_features = features[-1:] # Get latest observation
        predictions = {}
        
        for model_name, model in self.models.items():
            if self.is_trained:
                scaler = self.scalers[model_name]
                X_scaled = scaler.transform(latest_features)
                pred = model.predict(X_scaled)[0]
                predictions[model_name] = pred
        
        # Ensemble prediction (weighted average)
        if predictions:
            individual_preds = list(predictions.values())
            ensemble_pred = np.mean(individual_preds)
            
            return {
                'ensemble': ensemble_pred,
                'individual': predictions,
                'individual_list': individual_preds
            }
        
        return {'ensemble': 0.0, 'individual': {}}
    
    def _calculate_prediction_uncertainty(self, features: np.ndarray, 
                                        predictions: Dict[str, Any]) -> float:
        """Calculate uncertainty in predictions."""
        if 'individual_list' not in predictions or len(predictions['individual_list']) < 2:
            return 1.0  # Maximum uncertainty
        
        # Standard deviation of individual predictions
        pred_std = np.std(predictions['individual_list'])
        
        # Normalize by mean absolute prediction
        mean_abs_pred = np.mean(np.abs(predictions['individual_list']))
        uncertainty = pred_std / (mean_abs_pred + 1e-8)
        
        return float(np.clip(uncertainty, 0, 1))
    
    def _calculate_model_performance(self, features: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Calculate current model performance metrics."""
        if not self.is_trained or len(features) == 0:
            return {'r2_score': 0.0, 'mse': 1.0, 'mae': 1.0}
        
        try:
            # Use recent data for performance evaluation
            recent_size = min(100, len(features))
            X_recent = features[-recent_size:]
            y_recent = targets[-recent_size:]
            
            predictions = []
            for model_name, model in self.models.items():
                scaler = self.scalers[model_name]
                X_scaled = scaler.transform(X_recent)
                pred = model.predict(X_scaled)
                predictions.append(pred)
            
            # Ensemble prediction
            ensemble_pred = np.mean(predictions, axis=0)
            
            # Calculate metrics
            r2 = r2_score(y_recent, ensemble_pred)
            mse = mean_squared_error(y_recent, ensemble_pred)
            mae = np.mean(np.abs(y_recent - ensemble_pred))
            
            return {
                'r2_score': float(r2),
                'mse': float(mse),
                'mae': float(mae)
            }
            
        except Exception:
            return {'r2_score': 0.0, 'mse': 1.0, 'mae': 1.0}
    
    def _analyze_features(self, features: np.ndarray) -> Dict[str, Any]:
        """Analyze feature importance and statistics."""
        if len(features) == 0:
            return {}
        
        latest_features = features[-1]
        feature_stats = {
            'feature_count': len(latest_features),
            'feature_mean': float(np.mean(latest_features)),
            'feature_std': float(np.std(latest_features)),
            'feature_range': float(np.max(latest_features) - np.min(latest_features))
        }
        
        # Add feature importance if available
        if self.feature_importance:
            avg_importance = np.mean([imp for imp in self.feature_importance.values()], axis=0)
            feature_stats['top_features'] = np.argsort(avg_importance)[-5:].tolist()
            feature_stats['avg_importance'] = avg_importance.tolist()
        
        return feature_stats
    
    def _calculate_signal_strength(self, predictions: Dict[str, Any], uncertainty: float) -> float:
        """Calculate trading signal strength from predictions."""
        if 'ensemble' not in predictions:
            return 0.0
        
        ensemble_pred = predictions['ensemble']
        
        # Adjust signal by uncertainty
        confidence = 1.0 - uncertainty
        signal_strength = ensemble_pred * confidence
        
        # Apply threshold
        if uncertainty > self.parameters['uncertainty_threshold']:
            signal_strength *= 0.5  # Reduce signal when uncertain
        
        return float(np.clip(signal_strength, -1, 1))
    
    def _calculate_risk_metrics(self, predictions: Dict[str, Any], uncertainty: float, 
                              data: pd.DataFrame) -> Dict[str, float]:
        """Calculate risk metrics for the predictions."""
        close = data['close'].values
        
        # Volatility-based risk
        recent_volatility = np.std(np.diff(close[-20:]) / close[-20:-1]) if len(close) >= 21 else 0.02
        
        # Prediction-based risk
        prediction_risk = abs(predictions.get('ensemble', 0)) * uncertainty
        
        # Combined risk score
        total_risk = (recent_volatility + prediction_risk) / 2
        
        return {
            'volatility_risk': float(recent_volatility),
            'prediction_risk': float(prediction_risk),
            'total_risk': float(total_risk),
            'uncertainty': uncertainty,
            'risk_adjusted_signal': predictions.get('ensemble', 0) / (1 + total_risk)
        }
    
    def _default_output(self) -> Dict[str, Any]:
        """Return default output when models are not trained."""
        return {
            'predictions': {'ensemble': 0.0, 'individual': {}},
            'prediction_mean': 0.0,
            'prediction_std': 0.0,
            'uncertainty': 1.0,
            'signal_strength': 0.0,
            'model_performance': {'r2_score': 0.0, 'mse': 1.0, 'mae': 1.0},
            'feature_importance': {},
            'feature_analysis': {},
            'risk_metrics': {'total_risk': 1.0, 'uncertainty': 1.0},
            'model_confidence': 0.0,
            'prediction_horizon': self.parameters['prediction_horizon'],
            'training_samples': 0
        }
    
    def _generate_signal(self, value: Dict[str, Any], data: pd.DataFrame) -> Tuple[Optional[SignalType], float]:
        """Generate trading signal based on ML predictions."""
        signal_strength = value['signal_strength']
        model_confidence = value['model_confidence']
        
        # Require minimum confidence for signals
        if model_confidence < 0.3:
            return SignalType.NEUTRAL, model_confidence
        
        # Generate signal based on strength
        confidence = min(abs(signal_strength) * model_confidence, 1.0)
        
        if signal_strength > 0.6:
            return SignalType.STRONG_BUY, confidence
        elif signal_strength > 0.2:
            return SignalType.BUY, confidence
        elif signal_strength < -0.6:
            return SignalType.STRONG_SELL, confidence
        elif signal_strength < -0.2:
            return SignalType.SELL, confidence
        else:
            return SignalType.NEUTRAL, confidence
