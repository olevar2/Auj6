"""
Neural Network Predictor Indicator - AI Enhanced Category
========================================================

Advanced deep learning predictor with multiple neural network architectures,
prediction horizons, uncertainty quantification, and adaptive learning for
sophisticated market prediction and forecasting.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Union, Optional, Tuple, List
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

from ..base.standard_indicator import StandardIndicatorInterface, DataRequirement, DataType, SignalType


class NeuralNetworkPredictorIndicator(StandardIndicatorInterface):
    """
    AI-Enhanced Neural Network Predictor with advanced features.
    
    Features:
    - Multiple deep learning architectures (LSTM, GRU, CNN, Transformer)
    - Multi-horizon prediction (1, 5, 10, 20 periods ahead)
    - Uncertainty quantification with Bayesian neural networks
    - Ensemble of different architectures
    - Adaptive learning with online updates
    - Feature importance analysis
    - Cross-validation and hyperparameter optimization
    - Regularization and dropout for overfitting prevention
    - Performance monitoring and model selection
    - Real-time prediction confidence assessment
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'sequence_length': 60,           # Input sequence length
            'prediction_horizons': [1, 5, 10, 20],  # Prediction horizons
            'hidden_layers': [128, 64, 32], # Hidden layer sizes
            'learning_rate': 0.001,         # Learning rate
            'batch_size': 32,               # Batch size for training
            'epochs': 100,                  # Training epochs
            'validation_split': 0.2,        # Validation data split
            'dropout_rate': 0.3,            # Dropout rate
            'l2_regularization': 0.01,      # L2 regularization
            'ensemble_size': 5,             # Number of models in ensemble
            'uncertainty_samples': 100,     # MC dropout samples for uncertainty
            'feature_count': 20,            # Number of engineered features
            'lstm_units': [50, 25],         # LSTM layer units
            'cnn_filters': [32, 64],        # CNN filter counts
            'attention_heads': 8,           # Multi-head attention heads
            'transformer_layers': 4,        # Transformer encoder layers
            'adaptive_learning': True,      # Enable adaptive learning
            'bayesian_inference': True,     # Enable Bayesian inference
            'cross_validation': True,       # Enable cross-validation
            'hyperparameter_tuning': True,  # Enable hyperparameter tuning
            'online_learning': True,        # Enable online learning
        }
        
        if parameters:
            default_params.update(parameters)
        
        super().__init__("NeuralNetworkPredictorIndicator", default_params)
        
        # Initialize scalers
        self.feature_scaler = StandardScaler()
        self.target_scaler = MinMaxScaler()
        self.robust_scaler = RobustScaler()
        
        # Initialize models
        self.models = {}
        self.ensemble_models = []
        self.model_weights = {}
        
        # Training data storage
        self.training_sequences = []
        self.training_targets = {}
        self.validation_data = {}
        
        # Performance tracking
        self.prediction_history = []
        self.performance_metrics = {}
        self.model_uncertainties = {}
        
        # State tracking
        self.is_trained = False
        self.last_training_size = 0
        self.feature_importance = {}
        
        # Initialize TensorFlow/Keras models
        self._initialize_keras_models()
        
        # Initialize PyTorch models
        self._initialize_pytorch_models()
    
    def get_data_requirements(self) -> DataRequirement:
        return DataRequirement(
            data_type=DataType.OHLCV,
            required_columns=["high", "low", "close", "volume"],
            min_periods=self.parameters['sequence_length'] + max(self.parameters['prediction_horizons']) + 50
        )
    
    def calculate_raw(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate advanced neural network predictions."""
        try:
            if len(data) < self.get_data_requirements().min_periods:
                return self._get_default_output()
            
            # Extract and preprocess data
            high = data['high'].values
            low = data['low'].values
            close = data['close'].values
            volume = data['volume'].values
            
            # Feature engineering
            features = self._engineer_features(high, low, close, volume)
            
            # Prepare sequences for neural networks
            sequences, targets = self._prepare_sequences(features, close)
            
            # Model training and updating
            training_status = self._manage_model_training(sequences, targets)
            
            # Generate predictions
            predictions = self._generate_predictions(sequences)
            
            # Uncertainty quantification
            uncertainties = self._quantify_uncertainties(sequences, predictions)
            
            # Ensemble predictions
            ensemble_predictions = self._calculate_ensemble_predictions(predictions)
            
            # Performance analysis
            performance_analysis = self._analyze_performance(predictions, targets)
            
            # Feature importance analysis
            feature_analysis = self._analyze_feature_importance(features)
            
            # Adaptive learning updates
            adaptation_results = {}
            if self.parameters['adaptive_learning']:
                adaptation_results = self._perform_adaptive_learning(sequences, targets)
            
            # Cross-validation results
            cv_results = {}
            if self.parameters['cross_validation'] and self.is_trained:
                cv_results = self._perform_cross_validation(sequences, targets)
            
            # Generate trading signals
            signals = self._generate_prediction_signals(
                ensemble_predictions, uncertainties, performance_analysis
            )
            
            return {
                'features': features,
                'sequences': sequences,
                'training_status': training_status,
                'predictions': predictions,
                'uncertainties': uncertainties,
                'ensemble_predictions': ensemble_predictions,
                'performance_analysis': performance_analysis,
                'feature_analysis': feature_analysis,
                'adaptation_results': adaptation_results,
                'cv_results': cv_results,
                'signals': signals,
                'prediction_direction': signals.get('direction', 'neutral'),
                'prediction_confidence': signals.get('confidence', 0.5),
                'prediction_strength': signals.get('strength', 0.0),
                'model_accuracy': performance_analysis.get('ensemble_accuracy', 0.5)
            }
            
        except Exception as e:
            return self._handle_calculation_error(e)
    
    def _initialize_keras_models(self) -> None:
        """Initialize Keras/TensorFlow models."""
        self.keras_models = {}
        
        # LSTM Model
        self.keras_models['lstm'] = self._build_lstm_model()
        
        # CNN Model
        self.keras_models['cnn'] = self._build_cnn_model()
        
        # Transformer Model
        self.keras_models['transformer'] = self._build_transformer_model()
        
        # Hybrid CNN-LSTM Model
        self.keras_models['cnn_lstm'] = self._build_cnn_lstm_model()
    
    def _build_lstm_model(self) -> keras.Model:
        """Build LSTM model architecture."""
        try:
            model = models.Sequential()
            
            # First LSTM layer
            model.add(layers.LSTM(
                self.parameters['lstm_units'][0],
                return_sequences=True,
                input_shape=(self.parameters['sequence_length'], self.parameters['feature_count']),
                dropout=self.parameters['dropout_rate'],
                recurrent_dropout=self.parameters['dropout_rate']
            ))
            
            # Second LSTM layer
            model.add(layers.LSTM(
                self.parameters['lstm_units'][1],
                return_sequences=False,
                dropout=self.parameters['dropout_rate'],
                recurrent_dropout=self.parameters['dropout_rate']
            ))
            
            # Dense layers
            for units in self.parameters['hidden_layers']:
                model.add(layers.Dense(
                    units,
                    activation='relu',
                    kernel_regularizer=keras.regularizers.l2(self.parameters['l2_regularization'])
                ))
                model.add(layers.Dropout(self.parameters['dropout_rate']))
            
            # Output layers for different horizons
            outputs = []
            for horizon in self.parameters['prediction_horizons']:
                output = layers.Dense(1, name=f'horizon_{horizon}')(model.layers[-1].output)
                outputs.append(output)
            
            if len(outputs) > 1:
                model = models.Model(inputs=model.input, outputs=outputs)
            else:
                model.add(layers.Dense(1))
            
            # Compile model
            model.compile(
                optimizer=optimizers.Adam(learning_rate=self.parameters['learning_rate']),
                loss='mse',
                metrics=['mae', 'mape']
            )
            
            return model
        except Exception as e:
            print(f"Error building LSTM model: {e}")
            return None
    
    def _build_cnn_model(self) -> keras.Model:
        """Build CNN model architecture."""
        try:
            model = models.Sequential()
            
            # Reshape for CNN
            model.add(layers.Reshape(
                (self.parameters['sequence_length'], self.parameters['feature_count'], 1),
                input_shape=(self.parameters['sequence_length'], self.parameters['feature_count'])
            ))
            
            # Convolutional layers
            for i, filters in enumerate(self.parameters['cnn_filters']):
                model.add(layers.Conv2D(
                    filters,
                    (3, 3),
                    activation='relu',
                    padding='same',
                    kernel_regularizer=keras.regularizers.l2(self.parameters['l2_regularization'])
                ))
                model.add(layers.MaxPooling2D((2, 2), padding='same'))
                model.add(layers.Dropout(self.parameters['dropout_rate']))
            
            # Flatten and dense layers
            model.add(layers.Flatten())
            
            for units in self.parameters['hidden_layers']:
                model.add(layers.Dense(
                    units,
                    activation='relu',
                    kernel_regularizer=keras.regularizers.l2(self.parameters['l2_regularization'])
                ))
                model.add(layers.Dropout(self.parameters['dropout_rate']))
            
            # Output layer
            model.add(layers.Dense(len(self.parameters['prediction_horizons'])))
            
            # Compile model
            model.compile(
                optimizer=optimizers.Adam(learning_rate=self.parameters['learning_rate']),
                loss='mse',
                metrics=['mae']
            )
            
            return model
        except Exception as e:
            print(f"Error building CNN model: {e}")
            return None
    
    def _build_transformer_model(self) -> keras.Model:
        """Build Transformer model architecture."""
        try:
            inputs = layers.Input(shape=(self.parameters['sequence_length'], self.parameters['feature_count']))
            
            # Multi-head attention
            attention_output = layers.MultiHeadAttention(
                num_heads=self.parameters['attention_heads'],
                key_dim=self.parameters['feature_count'] // self.parameters['attention_heads']
            )(inputs, inputs)
            
            # Add & Norm
            attention_output = layers.Add()([inputs, attention_output])
            attention_output = layers.LayerNormalization()(attention_output)
            
            # Feed forward network
            ffn = layers.Dense(self.parameters['hidden_layers'][0], activation='relu')(attention_output)
            ffn = layers.Dropout(self.parameters['dropout_rate'])(ffn)
            ffn = layers.Dense(self.parameters['feature_count'])(ffn)
            
            # Add & Norm
            transformer_output = layers.Add()([attention_output, ffn])
            transformer_output = layers.LayerNormalization()(transformer_output)
            
            # Global average pooling
            pooled = layers.GlobalAveragePooling1D()(transformer_output)
            
            # Dense layers
            for units in self.parameters['hidden_layers'][1:]:
                pooled = layers.Dense(
                    units,
                    activation='relu',
                    kernel_regularizer=keras.regularizers.l2(self.parameters['l2_regularization'])
                )(pooled)
                pooled = layers.Dropout(self.parameters['dropout_rate'])(pooled)
            
            # Output layer
            outputs = layers.Dense(len(self.parameters['prediction_horizons']))(pooled)
            
            model = models.Model(inputs=inputs, outputs=outputs)
            
            # Compile model
            model.compile(
                optimizer=optimizers.Adam(learning_rate=self.parameters['learning_rate']),
                loss='mse',
                metrics=['mae']
            )
            
            return model
        except Exception as e:
            print(f"Error building Transformer model: {e}")
            return None
    
    def _build_cnn_lstm_model(self) -> keras.Model:
        """Build hybrid CNN-LSTM model architecture."""
        try:
            inputs = layers.Input(shape=(self.parameters['sequence_length'], self.parameters['feature_count']))
            
            # Reshape for CNN
            reshaped = layers.Reshape(
                (self.parameters['sequence_length'], self.parameters['feature_count'], 1)
            )(inputs)
            
            # CNN layers
            x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(reshaped)
            x = layers.MaxPooling2D((2, 1), padding='same')(x)
            x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
            x = layers.MaxPooling2D((2, 1), padding='same')(x)
            
            # Reshape back for LSTM
            shape = x.shape
            x = layers.Reshape((shape[1], shape[2] * shape[3]))(x)
            
            # LSTM layers
            x = layers.LSTM(50, return_sequences=True, dropout=self.parameters['dropout_rate'])(x)
            x = layers.LSTM(25, dropout=self.parameters['dropout_rate'])(x)
            
            # Dense layers
            for units in self.parameters['hidden_layers']:
                x = layers.Dense(
                    units,
                    activation='relu',
                    kernel_regularizer=keras.regularizers.l2(self.parameters['l2_regularization'])
                )(x)
                x = layers.Dropout(self.parameters['dropout_rate'])(x)
            
            # Output layer
            outputs = layers.Dense(len(self.parameters['prediction_horizons']))(x)
            
            model = models.Model(inputs=inputs, outputs=outputs)
            
            # Compile model
            model.compile(
                optimizer=optimizers.Adam(learning_rate=self.parameters['learning_rate']),
                loss='mse',
                metrics=['mae']
            )
            
            return model
        except Exception as e:
            print(f"Error building CNN-LSTM model: {e}")
            return None
    
    def _initialize_pytorch_models(self) -> None:
        """Initialize PyTorch models."""
        self.pytorch_models = {}
        
        try:
            # GRU Model
            self.pytorch_models['gru'] = self._build_gru_model()
            
            # Attention Model
            self.pytorch_models['attention'] = self._build_attention_model()
        except Exception as e:
            print(f"Error initializing PyTorch models: {e}")
    
    def _build_gru_model(self):
        """Build GRU model using PyTorch."""
        class GRUModel(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
                super(GRUModel, self).__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                
                self.gru = nn.GRU(input_size, hidden_size, num_layers, 
                                 batch_first=True, dropout=dropout)
                self.fc = nn.Linear(hidden_size, output_size)
                self.dropout = nn.Dropout(dropout)
                
            def forward(self, x):
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
                out, _ = self.gru(x, h0)
                out = self.dropout(out[:, -1, :])
                out = self.fc(out)
                return out
        
        return GRUModel(
            input_size=self.parameters['feature_count'],
            hidden_size=64,
            num_layers=2,
            output_size=len(self.parameters['prediction_horizons']),
            dropout=self.parameters['dropout_rate']
        )
    
    def _build_attention_model(self):
        """Build attention-based model using PyTorch."""
        class AttentionModel(nn.Module):
            def __init__(self, input_size, hidden_size, output_size, seq_length):
                super(AttentionModel, self).__init__()
                self.hidden_size = hidden_size
                self.seq_length = seq_length
                
                self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
                self.attention = nn.Linear(hidden_size, 1)
                self.fc = nn.Linear(hidden_size, output_size)
                
            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                
                # Attention mechanism
                attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
                context = torch.sum(attention_weights * lstm_out, dim=1)
                
                output = self.fc(context)
                return output
        
        return AttentionModel(
            input_size=self.parameters['feature_count'],
            hidden_size=64,
            output_size=len(self.parameters['prediction_horizons']),
            seq_length=self.parameters['sequence_length']
        )    
    def _engineer_features(self, high: np.ndarray, low: np.ndarray, 
                          close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Engineer comprehensive features for neural network input."""
        features = []
        
        try:
            # Price features
            features.append(close)
            features.append((high + low + close) / 3)  # Typical price
            features.append((high + low) / 2)  # Median price
            
            # Returns
            returns = np.diff(np.log(close), prepend=np.log(close[0]))
            features.append(returns)
            
            # Moving averages
            for window in [5, 10, 20, 50]:
                if len(close) >= window:
                    ma = pd.Series(close).rolling(window).mean().fillna(close[0])
                    features.append(ma.values)
                    # Price relative to MA
                    features.append(close / ma.values)
            
            # Volatility features
            for window in [10, 20]:
                if len(returns) >= window:
                    vol = pd.Series(returns).rolling(window).std().fillna(0)
                    features.append(vol.values)
            
            # RSI
            if len(close) >= 14:
                rsi = self._calculate_rsi(close, 14)
                features.append(rsi)
            
            # MACD
            if len(close) >= 26:
                macd, signal = self._calculate_macd(close)
                features.append(macd)
                features.append(signal)
                features.append(macd - signal)
            
            # Bollinger Bands
            if len(close) >= 20:
                bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(close)
                features.append((close - bb_lower) / (bb_upper - bb_lower))
                features.append((bb_upper - bb_lower) / bb_middle)
            
            # Volume features
            if len(volume) > 0:
                features.append(volume)
                # Volume moving average
                if len(volume) >= 20:
                    vol_ma = pd.Series(volume).rolling(20).mean().fillna(volume[0])
                    features.append(volume / vol_ma.values)
            
            # Ensure all features have the same length
            min_length = min(len(f) for f in features)
            features = [f[-min_length:] for f in features]
            
            # Stack features
            feature_matrix = np.column_stack(features)
            
            # Limit to specified feature count
            if feature_matrix.shape[1] > self.parameters['feature_count']:
                feature_matrix = feature_matrix[:, :self.parameters['feature_count']]
            elif feature_matrix.shape[1] < self.parameters['feature_count']:
                # Pad with zeros if not enough features
                padding = np.zeros((feature_matrix.shape[0], 
                                  self.parameters['feature_count'] - feature_matrix.shape[1]))
                feature_matrix = np.hstack([feature_matrix, padding])
            
            return feature_matrix
        
        except Exception as e:
            print(f"Error in feature engineering: {e}")
            return np.zeros((len(close), self.parameters['feature_count']))
    
    def _calculate_rsi(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate RSI indicator."""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = pd.Series(gains).rolling(window).mean().fillna(0)
        avg_losses = pd.Series(losses).rolling(window).mean().fillna(0)
        
        rs = avg_gains / (avg_losses + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return np.concatenate([np.array([50]), rsi.values])
    
    def _calculate_macd(self, prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate MACD indicator."""
        ema_12 = pd.Series(prices).ewm(span=12).mean()
        ema_26 = pd.Series(prices).ewm(span=26).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9).mean()
        
        return macd.values, signal.values
    
    def _calculate_bollinger_bands(self, prices: np.ndarray, window: int = 20, 
                                 std_dev: float = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate Bollinger Bands."""
        sma = pd.Series(prices).rolling(window).mean()
        std = pd.Series(prices).rolling(window).std()
        
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        
        return upper.fillna(prices[0]).values, sma.fillna(prices[0]).values, lower.fillna(prices[0]).values
    
    def _prepare_sequences(self, features: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Prepare sequences for neural network training."""
        try:
            seq_length = self.parameters['sequence_length']
            horizons = self.parameters['prediction_horizons']
            
            sequences = []
            target_dict = {f'horizon_{h}': [] for h in horizons}
            
            # Create sequences
            for i in range(seq_length, len(features) - max(horizons)):
                # Input sequence
                seq = features[i-seq_length:i]
                sequences.append(seq)
                
                # Target values for different horizons
                for horizon in horizons:
                    if i + horizon < len(targets):
                        target_dict[f'horizon_{horizon}'].append(targets[i + horizon])
            
            sequences = np.array(sequences)
            for key in target_dict:
                target_dict[key] = np.array(target_dict[key])
            
            return sequences, target_dict
        
        except Exception as e:
            print(f"Error preparing sequences: {e}")
            return np.array([]), {}
    
    def _manage_model_training(self, sequences: np.ndarray, 
                             targets: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Manage model training and updates."""
        training_status = {
            'training_required': False,
            'models_trained': 0,
            'training_samples': 0,
            'validation_scores': {},
            'training_time': 0.0
        }
        
        try:
            if len(sequences) == 0:
                return training_status
            
            # Check if training is required
            current_size = len(sequences)
            size_increase = current_size - self.last_training_size
            
            should_train = (not self.is_trained or 
                          size_increase >= 100 or  # Retrain with new data
                          len(self.training_sequences) == 0)
            
            if should_train and current_size >= 200:  # Minimum training samples
                training_status['training_required'] = True
                
                # Prepare data
                X_train, X_val, y_train, y_val = self._split_training_data(sequences, targets)
                
                # Scale data
                X_train_scaled = self._scale_features(X_train, fit=True)
                X_val_scaled = self._scale_features(X_val, fit=False)
                
                # Train Keras models
                keras_scores = self._train_keras_models(X_train_scaled, X_val_scaled, y_train, y_val)
                training_status['validation_scores'].update(keras_scores)
                
                # Train PyTorch models
                pytorch_scores = self._train_pytorch_models(X_train_scaled, X_val_scaled, y_train, y_val)
                training_status['validation_scores'].update(pytorch_scores)
                
                # Train ensemble
                self._train_ensemble_models(X_train_scaled, y_train)
                
                # Update status
                training_status['models_trained'] = len(self.keras_models) + len(self.pytorch_models)
                training_status['training_samples'] = len(X_train)
                
                self.is_trained = True
                self.last_training_size = current_size
                
                # Store training data
                self.training_sequences = sequences
                self.training_targets = targets
        
        except Exception as e:
            print(f"Error in model training: {e}")
        
        return training_status
    
    def _split_training_data(self, sequences: np.ndarray, 
                           targets: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, Dict, Dict]:
        """Split data into training and validation sets."""
        split_idx = int(len(sequences) * (1 - self.parameters['validation_split']))
        
        X_train = sequences[:split_idx]
        X_val = sequences[split_idx:]
        
        y_train = {}
        y_val = {}
        
        for key, target_array in targets.items():
            y_train[key] = target_array[:split_idx]
            y_val[key] = target_array[split_idx:]
        
        return X_train, X_val, y_train, y_val
    
    def _scale_features(self, features: np.ndarray, fit: bool = False) -> np.ndarray:
        """Scale features for neural network input."""
        try:
            # Reshape for scaling
            original_shape = features.shape
            features_reshaped = features.reshape(-1, features.shape[-1])
            
            if fit:
                scaled = self.feature_scaler.fit_transform(features_reshaped)
            else:
                scaled = self.feature_scaler.transform(features_reshaped)
            
            # Reshape back
            scaled = scaled.reshape(original_shape)
            
            return scaled
        except Exception as e:
            print(f"Error scaling features: {e}")
            return features
    
    def _train_keras_models(self, X_train: np.ndarray, X_val: np.ndarray, 
                          y_train: Dict, y_val: Dict) -> Dict[str, float]:
        """Train Keras models."""
        scores = {}
        
        try:
            # Prepare target for multi-output
            y_train_combined = np.column_stack([y_train[f'horizon_{h}'] 
                                              for h in self.parameters['prediction_horizons']])
            y_val_combined = np.column_stack([y_val[f'horizon_{h}'] 
                                            for h in self.parameters['prediction_horizons']])
            
            # Train each Keras model
            for model_name, model in self.keras_models.items():
                if model is not None:
                    try:
                        # Callbacks
                        callbacks_list = [
                            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
                        ]
                        
                        # Train model
                        history = model.fit(
                            X_train, y_train_combined,
                            validation_data=(X_val, y_val_combined),
                            epochs=self.parameters['epochs'],
                            batch_size=self.parameters['batch_size'],
                            callbacks=callbacks_list,
                            verbose=0
                        )
                        
                        # Evaluate
                        val_score = model.evaluate(X_val, y_val_combined, verbose=0)
                        scores[f'{model_name}_val_loss'] = val_score[0] if isinstance(val_score, list) else val_score
                        
                    except Exception as e:
                        print(f"Error training {model_name}: {e}")
                        scores[f'{model_name}_val_loss'] = float('inf')
        
        except Exception as e:
            print(f"Error in Keras training: {e}")
        
        return scores
    
    def _train_pytorch_models(self, X_train: np.ndarray, X_val: np.ndarray, 
                            y_train: Dict, y_val: Dict) -> Dict[str, float]:
        """Train PyTorch models."""
        scores = {}
        
        try:
            # Convert to tensors
            X_train_tensor = torch.FloatTensor(X_train)
            X_val_tensor = torch.FloatTensor(X_val)
            
            y_train_combined = np.column_stack([y_train[f'horizon_{h}'] 
                                              for h in self.parameters['prediction_horizons']])
            y_val_combined = np.column_stack([y_val[f'horizon_{h}'] 
                                            for h in self.parameters['prediction_horizons']])
            
            y_train_tensor = torch.FloatTensor(y_train_combined)
            y_val_tensor = torch.FloatTensor(y_val_combined)
            
            # Train each PyTorch model
            for model_name, model in self.pytorch_models.items():
                if model is not None:
                    try:
                        optimizer = optim.Adam(model.parameters(), lr=self.parameters['learning_rate'])
                        criterion = nn.MSELoss()
                        
                        # Training loop
                        for epoch in range(self.parameters['epochs']):
                            model.train()
                            optimizer.zero_grad()
                            
                            outputs = model(X_train_tensor)
                            loss = criterion(outputs, y_train_tensor)
                            loss.backward()
                            optimizer.step()
                        
                        # Validation
                        model.eval()
                        with torch.no_grad():
                            val_outputs = model(X_val_tensor)
                            val_loss = criterion(val_outputs, y_val_tensor)
                            scores[f'{model_name}_val_loss'] = val_loss.item()
                    
                    except Exception as e:
                        print(f"Error training {model_name}: {e}")
                        scores[f'{model_name}_val_loss'] = float('inf')
        
        except Exception as e:
            print(f"Error in PyTorch training: {e}")
        
        return scores
    
    def _train_ensemble_models(self, X_train: np.ndarray, y_train: Dict) -> None:
        """Train ensemble models."""
        try:
            # Flatten sequences for traditional ML models
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            
            # Train for each horizon
            for horizon in self.parameters['prediction_horizons']:
                y_target = y_train[f'horizon_{horizon}']
                
                # Random Forest
                rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                rf_model.fit(X_train_flat, y_target)
                
                # Gradient Boosting
                gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
                gb_model.fit(X_train_flat, y_target)
                
                # Store models
                if f'horizon_{horizon}' not in self.ensemble_models:
                    self.ensemble_models.append({})
                
                self.ensemble_models[-1][f'rf_horizon_{horizon}'] = rf_model
                self.ensemble_models[-1][f'gb_horizon_{horizon}'] = gb_model
        
        except Exception as e:
            print(f"Error training ensemble: {e}")
    
    def _generate_predictions(self, sequences: np.ndarray) -> Dict[str, Any]:
        """Generate predictions from all models."""
        predictions = {
            'keras_predictions': {},
            'pytorch_predictions': {},
            'ensemble_predictions': {},
            'latest_predictions': {}
        }
        
        try:
            if len(sequences) == 0 or not self.is_trained:
                return predictions
            
            # Get latest sequence
            latest_seq = sequences[-1:] if len(sequences) > 0 else sequences
            latest_seq_scaled = self._scale_features(latest_seq, fit=False)
            
            # Keras predictions
            for model_name, model in self.keras_models.items():
                if model is not None:
                    try:
                        pred = model.predict(latest_seq_scaled, verbose=0)
                        predictions['keras_predictions'][model_name] = pred[0] if len(pred) > 0 else []
                    except Exception as e:
                        print(f"Error predicting with {model_name}: {e}")
            
            # PyTorch predictions
            for model_name, model in self.pytorch_models.items():
                if model is not None:
                    try:
                        model.eval()
                        with torch.no_grad():
                            input_tensor = torch.FloatTensor(latest_seq_scaled)
                            pred = model(input_tensor)
                            predictions['pytorch_predictions'][model_name] = pred.numpy()[0]
                    except Exception as e:
                        print(f"Error predicting with {model_name}: {e}")
            
            # Ensemble predictions
            if self.ensemble_models:
                latest_seq_flat = latest_seq.reshape(1, -1)
                
                for i, ensemble in enumerate(self.ensemble_models):
                    ensemble_preds = {}
                    for model_name, model in ensemble.items():
                        try:
                            pred = model.predict(latest_seq_flat)
                            ensemble_preds[model_name] = pred[0]
                        except Exception as e:
                            print(f"Error with ensemble {model_name}: {e}")
                    
                    if ensemble_preds:
                        predictions['ensemble_predictions'][f'ensemble_{i}'] = ensemble_preds
            
            # Store latest predictions for each horizon
            all_preds = []
            for model_type in ['keras_predictions', 'pytorch_predictions']:
                for model_name, pred in predictions[model_type].items():
                    if len(pred) > 0:
                        all_preds.append(pred)
            
            if all_preds:
                # Average predictions across models
                avg_predictions = np.mean(all_preds, axis=0)
                for i, horizon in enumerate(self.parameters['prediction_horizons']):
                    if i < len(avg_predictions):
                        predictions['latest_predictions'][f'horizon_{horizon}'] = avg_predictions[i]
        
        except Exception as e:
            print(f"Error generating predictions: {e}")
        
        return predictions
    
    def _quantify_uncertainties(self, sequences: np.ndarray, 
                              predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Quantify prediction uncertainties."""
        uncertainties = {
            'epistemic_uncertainty': {},
            'aleatoric_uncertainty': {},
            'total_uncertainty': {},
            'confidence_intervals': {}
        }
        
        try:
            if not self.is_trained or len(sequences) == 0:
                return uncertainties
            
            # MC Dropout for epistemic uncertainty
            latest_seq = sequences[-1:] if len(sequences) > 0 else sequences
            latest_seq_scaled = self._scale_features(latest_seq, fit=False)
            
            # Collect predictions with dropout enabled
            mc_predictions = []
            
            for model_name, model in self.keras_models.items():
                if model is not None:
                    try:
                        model_preds = []
                        for _ in range(self.parameters['uncertainty_samples']):
                            # Enable dropout during inference
                            pred = model(latest_seq_scaled, training=True)
                            model_preds.append(pred.numpy()[0])
                        
                        if model_preds:
                            mc_predictions.extend(model_preds)
                    except Exception as e:
                        print(f"Error in MC dropout for {model_name}: {e}")
            
            if mc_predictions:
                mc_predictions = np.array(mc_predictions)
                
                # Calculate uncertainties for each horizon
                for i, horizon in enumerate(self.parameters['prediction_horizons']):
                    if i < mc_predictions.shape[1]:
                        horizon_preds = mc_predictions[:, i]
                        
                        # Epistemic uncertainty (model uncertainty)
                        epistemic = np.std(horizon_preds)
                        uncertainties['epistemic_uncertainty'][f'horizon_{horizon}'] = epistemic
                        
                        # Confidence intervals
                        ci_lower = np.percentile(horizon_preds, 2.5)
                        ci_upper = np.percentile(horizon_preds, 97.5)
                        uncertainties['confidence_intervals'][f'horizon_{horizon}'] = (ci_lower, ci_upper)
                        
                        # Total uncertainty
                        uncertainties['total_uncertainty'][f'horizon_{horizon}'] = epistemic
        
        except Exception as e:
            print(f"Error quantifying uncertainties: {e}")
        
        return uncertainties