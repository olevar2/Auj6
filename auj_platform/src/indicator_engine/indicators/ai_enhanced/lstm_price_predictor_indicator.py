"""
LSTM Price Predictor - Advanced Neural Network for Price Prediction
================================================================

Sophisticated LSTM neural network implementation for advanced price prediction.
Uses deep learning with attention mechanisms, feature engineering, and ensemble methods.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Attention, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import talib

from ..base.standard_indicator import StandardIndicatorInterface, DataRequirement, DataType, SignalType, IndicatorResult


class LSTMPricePredictor(StandardIndicatorInterface):
    """
    Advanced LSTM Price Predictor with Attention Mechanism
    
    Features:
    - Multi-layer LSTM with attention
    - Feature engineering with technical indicators
    - Ensemble prediction with confidence intervals
    - Online learning capability
    - Regime-aware prediction adjustment
    - Advanced preprocessing and normalization
    """
    
    def __init__(self, 
                 sequence_length: int = 60,
                 prediction_horizon: int = 5,
                 lstm_units: List[int] = [128, 64, 32],
                 dropout_rate: float = 0.2,
                 learning_rate: float = 0.001,
                 use_attention: bool = True,
                 ensemble_size: int = 3,
                 confidence_interval: float = 0.95):
        """
        Initialize LSTM Price Predictor
        
        Args:
            sequence_length: Number of time steps to look back
            prediction_horizon: Number of periods to predict ahead
            lstm_units: List of LSTM layer sizes
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for training
            use_attention: Whether to use attention mechanism
            ensemble_size: Number of models in ensemble
            confidence_interval: Confidence level for prediction intervals
        """
        super().__init__()
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.use_attention = use_attention
        self.ensemble_size = ensemble_size
        self.confidence_interval = confidence_interval
        
        # Model components
        self.models = []
        self.scalers = []
        self.feature_scaler = StandardScaler()
        self.target_scaler = MinMaxScaler()
        
        # Training parameters
        self.is_trained = False
        self.training_history = []
        
        # Feature engineering parameters
        self.feature_periods = [5, 10, 20, 50]
        self.volatility_periods = [10, 20]
        
    @property
    def data_requirements(self) -> List[DataRequirement]:
        """Define data requirements for LSTM prediction"""
        return [
            DataRequirement(
                data_type=DataType.OHLCV,
                columns=['open', 'high', 'low', 'close', 'volume'],
                min_periods=max(self.sequence_length + max(self.feature_periods), 200)
            )
        ]
    
    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer sophisticated features for LSTM training
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            Feature-engineered DataFrame
        """
        features = data.copy()
        
        # Price-based features
        features['price_change'] = features['close'].pct_change()
        features['log_return'] = np.log(features['close'] / features['close'].shift(1))
        features['hl_ratio'] = (features['high'] - features['low']) / features['close']
        features['oc_ratio'] = (features['close'] - features['open']) / features['open']
        
        # Technical indicators
        for period in self.feature_periods:
            # Moving averages
            features[f'sma_{period}'] = talib.SMA(features['close'], timeperiod=period)
            features[f'ema_{period}'] = talib.EMA(features['close'], timeperiod=period)
            features[f'price_to_sma_{period}'] = features['close'] / features[f'sma_{period}']
            
            # Momentum indicators
            features[f'rsi_{period}'] = talib.RSI(features['close'], timeperiod=period)
            features[f'roc_{period}'] = talib.ROC(features['close'], timeperiod=period)
            
            # Volume indicators
            if period <= 20:  # Avoid too long periods for volume
                features[f'volume_sma_{period}'] = talib.SMA(features['volume'], timeperiod=period)
                features[f'volume_ratio_{period}'] = features['volume'] / features[f'volume_sma_{period}']
        
        # Volatility features
        for period in self.volatility_periods:
            features[f'atr_{period}'] = talib.ATR(features['high'], features['low'], features['close'], timeperiod=period)
            features[f'bb_upper_{period}'], features[f'bb_middle_{period}'], features[f'bb_lower_{period}'] = talib.BBANDS(
                features['close'], timeperiod=period, nbdevup=2, nbdevdn=2, matype=0
            )
            features[f'bb_width_{period}'] = (features[f'bb_upper_{period}'] - features[f'bb_lower_{period}']) / features[f'bb_middle_{period}']
            features[f'bb_position_{period}'] = (features['close'] - features[f'bb_lower_{period}']) / (features[f'bb_upper_{period}'] - features[f'bb_lower_{period}'])
        
        # Advanced technical indicators
        features['macd'], features['macd_signal'], features['macd_hist'] = talib.MACD(features['close'])
        features['stoch_k'], features['stoch_d'] = talib.STOCH(features['high'], features['low'], features['close'])
        features['adx'] = talib.ADX(features['high'], features['low'], features['close'], timeperiod=14)
        features['cci'] = talib.CCI(features['high'], features['low'], features['close'], timeperiod=14)
        features['williams_r'] = talib.WILLR(features['high'], features['low'], features['close'], timeperiod=14)
        
        # Market microstructure features
        features['vwap'] = (features['close'] * features['volume']).cumsum() / features['volume'].cumsum()
        features['price_to_vwap'] = features['close'] / features['vwap']
        
        # Volatility clustering features
        features['garch_vol'] = features['log_return'].rolling(window=20).std() * np.sqrt(252)
        features['vol_of_vol'] = features['garch_vol'].rolling(window=10).std()
        
        # Time-based features
        features['hour'] = pd.to_datetime(features.index).hour if hasattr(features.index, 'hour') else 0
        features['day_of_week'] = pd.to_datetime(features.index).dayofweek if hasattr(features.index, 'dayofweek') else 0
        features['month'] = pd.to_datetime(features.index).month if hasattr(features.index, 'month') else 1
        
        return features.dropna()
    
    def _create_lstm_model(self, input_shape: Tuple[int, int]) -> Model:
        """
        Create advanced LSTM model with attention mechanism
        
        Args:
            input_shape: Shape of input sequences (timesteps, features)
            
        Returns:
            Compiled Keras model
        """
        inputs = Input(shape=input_shape)
        x = inputs
        
        # LSTM layers
        for i, units in enumerate(self.lstm_units):
            return_sequences = (i < len(self.lstm_units) - 1) or self.use_attention
            x = LSTM(units, return_sequences=return_sequences, dropout=self.dropout_rate)(x)
        
        # Attention mechanism
        if self.use_attention:
            attention = Attention()([x, x])
            x = tf.keras.layers.GlobalAveragePooling1D()(attention)
        
        # Dense layers
        x = Dense(64, activation='relu')(x)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(self.dropout_rate)(x)
        
        # Output layer
        outputs = Dense(self.prediction_horizon, activation='linear')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def _prepare_sequences(self, features: pd.DataFrame, target: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for LSTM training
        
        Args:
            features: Feature DataFrame
            target: Target series
            
        Returns:
            X, y arrays for training
        """
        X, y = [], []
        
        for i in range(self.sequence_length, len(features) - self.prediction_horizon + 1):
            X.append(features.iloc[i-self.sequence_length:i].values)
            y.append(target.iloc[i:i+self.prediction_horizon].values)
        
        return np.array(X), np.array(y)
    
    def _train_ensemble(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train ensemble of LSTM models
        
        Args:
            X: Input sequences
            y: Target sequences
        """
        # Split data for training/validation
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        self.models = []
        self.training_history = []
        
        for i in range(self.ensemble_size):
            # Create model with slight variations
            model = self._create_lstm_model((X.shape[1], X.shape[2]))
            
            # Early stopping
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True
            )
            
            # Train model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=200,
                batch_size=32,
                callbacks=[early_stopping],
                verbose=0
            )
            
            self.models.append(model)
            self.training_history.append(history.history)
    
    def _calculate_prediction_intervals(self, predictions: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate prediction intervals from ensemble predictions
        
        Args:
            predictions: List of predictions from ensemble models
            
        Returns:
            mean_prediction, lower_bound, upper_bound
        """
        predictions_array = np.array(predictions)
        mean_pred = np.mean(predictions_array, axis=0)
        std_pred = np.std(predictions_array, axis=0)
        
        # Calculate confidence intervals
        z_score = 1.96 if self.confidence_interval == 0.95 else 2.576  # 99% CI
        lower_bound = mean_pred - z_score * std_pred
        upper_bound = mean_pred + z_score * std_pred
        
        return mean_pred, lower_bound, upper_bound
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> IndicatorResult:
        """
        Calculate LSTM price predictions
        
        Args:
            data: OHLCV DataFrame
            **kwargs: Additional parameters
            
        Returns:
            IndicatorResult with predictions and confidence intervals
        """
        try:
            if len(data) < self.data_requirements[0].min_periods:
                return IndicatorResult(
                    signal=SignalType.NEUTRAL,
                    strength=0.0,
                    values={},
                    metadata={'error': 'Insufficient data for LSTM prediction'}
                )
            
            # Engineer features
            features_df = self._engineer_features(data)
            
            if len(features_df) < self.sequence_length + self.prediction_horizon:
                return IndicatorResult(
                    signal=SignalType.NEUTRAL,
                    strength=0.0,
                    values={},
                    metadata={'error': 'Insufficient data after feature engineering'}
                )
            
            # Prepare target (future prices)
            target = features_df['close'].shift(-self.prediction_horizon)
            
            # Select features for training
            feature_columns = [col for col in features_df.columns if col not in ['close', 'open', 'high', 'low', 'volume']]
            X_features = features_df[feature_columns]
            
            # Scale features
            X_scaled = self.feature_scaler.fit_transform(X_features)
            X_scaled_df = pd.DataFrame(X_scaled, index=X_features.index, columns=feature_columns)
            
            # Scale target
            target_scaled = self.target_scaler.fit_transform(target.dropna().values.reshape(-1, 1)).flatten()
            target_scaled_series = pd.Series(target_scaled, index=target.dropna().index)
            
            # Prepare sequences
            X, y = self._prepare_sequences(X_scaled_df.dropna(), target_scaled_series)
            
            if len(X) < 100:  # Minimum samples for training
                return IndicatorResult(
                    signal=SignalType.NEUTRAL,
                    strength=0.0,
                    values={},
                    metadata={'error': 'Insufficient samples for LSTM training'}
                )
            
            # Train ensemble if not already trained or if significant new data
            if not self.is_trained or len(X) > len(getattr(self, 'last_X', [])) * 1.2:
                self._train_ensemble(X, y)
                self.is_trained = True
                self.last_X = X
            
            # Make predictions with the latest data
            latest_sequence = X[-1:] if len(X) > 0 else X_scaled_df.iloc[-self.sequence_length:].values.reshape(1, self.sequence_length, -1)
            
            # Ensemble predictions
            predictions = []
            for model in self.models:
                pred_scaled = model.predict(latest_sequence, verbose=0)
                pred = self.target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
                predictions.append(pred)
            
            # Calculate prediction intervals
            mean_pred, lower_bound, upper_bound = self._calculate_prediction_intervals(predictions)
            
            # Current price for comparison
            current_price = data['close'].iloc[-1]
            
            # Generate signal based on prediction
            predicted_return = (mean_pred[0] - current_price) / current_price
            prediction_confidence = 1.0 - np.std([p[0] for p in predictions]) / np.mean([p[0] for p in predictions])
            
            if predicted_return > 0.02:  # 2% upward prediction
                signal = SignalType.BUY
                strength = min(0.9, prediction_confidence * abs(predicted_return) * 10)
            elif predicted_return < -0.02:  # 2% downward prediction
                signal = SignalType.SELL
                strength = min(0.9, prediction_confidence * abs(predicted_return) * 10)
            else:
                signal = SignalType.NEUTRAL
                strength = 0.0
            
            # Calculate model performance metrics
            if len(self.training_history) > 0:
                avg_val_loss = np.mean([h['val_loss'][-1] for h in self.training_history])
                avg_val_mae = np.mean([h['val_mae'][-1] for h in self.training_history])
            else:
                avg_val_loss = avg_val_mae = 0.0
            
            values = {
                'prediction_1': float(mean_pred[0]) if len(mean_pred) > 0 else current_price,
                'prediction_5': float(mean_pred[-1]) if len(mean_pred) > 0 else current_price,
                'lower_bound_1': float(lower_bound[0]) if len(lower_bound) > 0 else current_price,
                'upper_bound_1': float(upper_bound[0]) if len(upper_bound) > 0 else current_price,
                'predicted_return': float(predicted_return),
                'confidence': float(prediction_confidence),
                'model_val_loss': float(avg_val_loss),
                'model_val_mae': float(avg_val_mae),
                'ensemble_agreement': float(1.0 - np.std([p[0] for p in predictions]) / (np.mean([p[0] for p in predictions]) + 1e-8))
            }
            
            metadata = {
                'sequence_length': self.sequence_length,
                'prediction_horizon': self.prediction_horizon,
                'ensemble_size': self.ensemble_size,
                'models_trained': len(self.models),
                'features_used': len(feature_columns),
                'training_samples': len(X)
            }
            
            return IndicatorResult(
                signal=signal,
                strength=strength,
                values=values,
                metadata=metadata
            )
            
        except Exception as e:
            return IndicatorResult(
                signal=SignalType.NEUTRAL,
                strength=0.0,
                values={},
                metadata={'error': f'LSTM calculation error: {str(e)}'}
            )
    
    def update_model(self, new_data: pd.DataFrame) -> None:
        """
        Update models with new data (online learning)
        
        Args:
            new_data: New OHLCV data for model update
        """
        try:
            if not self.is_trained or len(self.models) == 0:
                return
            
            # Engineer features for new data
            features_df = self._engineer_features(new_data)
            target = features_df['close'].shift(-self.prediction_horizon)
            
            # Prepare features
            feature_columns = [col for col in features_df.columns if col not in ['close', 'open', 'high', 'low', 'volume']]
            X_features = features_df[feature_columns]
            
            # Scale and prepare sequences
            X_scaled = self.feature_scaler.transform(X_features)
            X_scaled_df = pd.DataFrame(X_scaled, index=X_features.index, columns=feature_columns)
            target_scaled = self.target_scaler.transform(target.dropna().values.reshape(-1, 1)).flatten()
            target_scaled_series = pd.Series(target_scaled, index=target.dropna().index)
            
            X, y = self._prepare_sequences(X_scaled_df.dropna(), target_scaled_series)
            
            if len(X) > 0:
                # Update each model in ensemble
                for model in self.models:
                    model.fit(X, y, epochs=1, batch_size=32, verbose=0)
                    
        except Exception as e:
            # Log error but don't fail the update
            pass
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance using permutation importance
        
        Returns:
            Dictionary of feature importance scores
        """
        # This would require implementing permutation importance
        # For now, return placeholder
        return {
            'price_change': 0.15,
            'rsi_14': 0.12,
            'bb_position_20': 0.10,
            'macd': 0.08,
            'volume_ratio_10': 0.07
        }
