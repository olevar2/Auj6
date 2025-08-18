"""
AI Ensemble Modeling System for AUJ Platform.

This module implements anti-overfitting techniques through ensemble modeling,
providing robust decision-making capabilities for AI agents.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
from enum import Enum
import joblib
import os
from pathlib import Path

# Scikit-learn imports
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report

from ..core.logging_setup import get_logger

logger = get_logger(__name__)


class ModelType(Enum):
    """Types of models in the ensemble."""
    DECISION_TREE = "decision_tree"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOST = "gradient_boost"
    NEURAL_NETWORK = "neural_network"
    SVM = "svm"
    LOGISTIC_REGRESSION = "logistic_regression"


class VotingMethod(Enum):
    """Methods for ensemble voting."""
    MAJORITY = "majority"
    WEIGHTED = "weighted"
    SOFT_VOTING = "soft_voting"
    ADAPTIVE = "adaptive"


@dataclass
class ModelPrediction:
    """Individual model prediction result."""
    model_id: str
    model_type: ModelType
    prediction: str  # BUY, SELL, HOLD, NO_SIGNAL
    confidence: float
    probability_distribution: Dict[str, float]
    feature_importance: Dict[str, float]
    timestamp: datetime
    data_subset_hash: str


@dataclass
class EnsemblePrediction:
    """Ensemble prediction result."""
    final_prediction: str
    ensemble_confidence: float
    individual_predictions: List[ModelPrediction]
    voting_method: VotingMethod
    consensus_strength: float
    disagreement_score: float
    overfitting_risk: float
    timestamp: datetime


class BaseEnsembleModel(ABC):
    """Base class for ensemble models."""
    
    def __init__(self, model_id: str, model_type: ModelType, config: Optional[Dict[str, Any]] = None):
        # Note: This class now requires config_manager parameter in __init__
        # self.config_manager = config_manager or UnifiedConfigManager()
        self.model_id = model_id
        self.model_type = model_type
        self.config = config or {}
        self.is_trained = False
        self.training_data_hash = ""
        self.performance_metrics = {}
        self.model_path = None
        self._setup_model_path()
        
    def _setup_model_path(self):
        """Setup the path for saving/loading the model."""
        models_dir = Path("models") / "ensemble"
        models_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = models_dir / f"{self.model_id}_{self.model_type.value}.joblib"
        
    @abstractmethod
    def train(self, features: pd.DataFrame, targets: pd.Series, sample_weights: Optional[np.ndarray] = None) -> None:
        """Train the model on provided data."""
        pass
    
    @abstractmethod
    def predict(self, features: pd.DataFrame) -> ModelPrediction:
        """Make prediction using the trained model."""
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        pass
    
    def save_model(self) -> bool:
        """Save the trained model to disk."""
        try:
            if not self.is_trained:
                logger.warning(f"Model {self.model_id} is not trained, cannot save")
                return False
                
            model_data = {
                'model': self.model,
                'label_encoder': getattr(self, 'label_encoder', None),
                'scaler': getattr(self, 'scaler', None),
                'training_data_hash': self.training_data_hash,
                'performance_metrics': self.performance_metrics,
                'config': self.config
            }
            
            joblib.dump(model_data, self.model_path)
            logger.info(f"Model {self.model_id} saved to {self.model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model {self.model_id}: {str(e)}")
            return False
    
    def load_model(self) -> bool:
        """Load a trained model from disk."""
        try:
            if not self.model_path.exists():
                logger.warning(f"Model file {self.model_path} does not exist")
                return False
                
            model_data = joblib.load(self.model_path)
            
            self.model = model_data['model']
            self.label_encoder = model_data.get('label_encoder')
            self.scaler = model_data.get('scaler')
            self.training_data_hash = model_data.get('training_data_hash', '')
            self.performance_metrics = model_data.get('performance_metrics', {})
            self.config.update(model_data.get('config', {}))
            self.is_trained = True
            
            logger.info(f"Model {self.model_id} loaded from {self.model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_id}: {str(e)}")
            return False
class SimpleDecisionTreeModel(BaseEnsembleModel):
    """Decision tree model using scikit-learn."""
    
    def __init__(self, model_id: str, config: Optional[Dict[str, Any]] = None):
        # Note: This class now requires config_manager parameter in __init__
        # self.config_manager = config_manager or UnifiedConfigManager()
        super().__init__(model_id, ModelType.DECISION_TREE, config)
        
        # Initialize scikit-learn model with config parameters
        self.model = DecisionTreeClassifier(
            max_depth=self.config_manager.get_int('max_depth', 10) if config else 10,
            min_samples_split=self.config_manager.get_int('min_samples_split', 5) if config else 5,
            min_samples_leaf=self.config_manager.get_int('min_samples_leaf', 2) if config else 2,
            random_state=self.config_manager.get_int('random_state', 42) if config else 42,
            class_weight='balanced'
        )
        
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
    def train(self, features: pd.DataFrame, targets: pd.Series, sample_weights: Optional[np.ndarray] = None) -> None:
        """Train decision tree using scikit-learn."""
        try:
            # Prepare data
            X = features.fillna(0)  # Handle missing values
            y = targets.fillna('HOLD')  # Handle missing targets
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Encode labels
            y_encoded = self.label_encoder.fit_transform(y)
            
            # Train the model
            if sample_weights is not None:
                self.model.fit(X_scaled, y_encoded, sample_weight=sample_weights)
            else:
                self.model.fit(X_scaled, y_encoded)
            
            # Calculate performance metrics
            predictions = self.model.predict(X_scaled)
            accuracy = accuracy_score(y_encoded, predictions)
            
            # Cross-validation score
            cv_scores = cross_val_score(self.model, X_scaled, y_encoded, cv=3)
            
            self.performance_metrics = {
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            self.is_trained = True
            self.training_data_hash = str(hash(str(features.values.tobytes())))
            
            logger.info(f"DecisionTree {self.model_id} trained with accuracy: {accuracy:.3f}")
            
        except Exception as e:
            logger.error(f"Decision tree training failed: {str(e)}")
            raise
    
    def predict(self, features: pd.DataFrame) -> ModelPrediction:
        """Make prediction using trained decision tree."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        try:
            # Prepare data
            X = features.fillna(0)
            X_scaled = self.scaler.transform(X)
            
            # Make prediction
            prediction_encoded = self.model.predict(X_scaled)[0]
            prediction_proba = self.model.predict_proba(X_scaled)[0]
            
            # Decode prediction
            prediction = self.label_encoder.inverse_transform([prediction_encoded])[0]
            
            # Get confidence (max probability)
            confidence = np.max(prediction_proba)
            
            # Create probability distribution
            classes = self.label_encoder.classes_
            prob_dist = dict(zip(classes, prediction_proba))
            
            # Ensure all required classes are present
            for class_name in ['BUY', 'SELL', 'HOLD']:
                if class_name not in prob_dist:
                    prob_dist[class_name] = 0.0
            
            return ModelPrediction(
                model_id=self.model_id,
                model_type=self.model_type,
                prediction=prediction,
                confidence=confidence,
                probability_distribution=prob_dist,
                feature_importance=self.get_feature_importance(),
                timestamp=datetime.now(),
                data_subset_hash=self.training_data_hash
            )
            
        except Exception as e:
            logger.error(f"Decision tree prediction failed: {str(e)}")
            raise
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the trained model."""
        if not self.is_trained:
            return {}
        
        try:
            importances = self.model.feature_importances_
            feature_names = list(range(len(importances)))  # Use indices if no names available
            return dict(zip(feature_names, importances))
        except Exception as e:
            logger.error(f"Failed to get feature importance: {str(e)}")
            return {}


class SimpleRandomForestModel(BaseEnsembleModel):
    """Random forest model using scikit-learn."""
    
    def __init__(self, model_id: str, config: Optional[Dict[str, Any]] = None):
        # Note: This class now requires config_manager parameter in __init__
        # self.config_manager = config_manager or UnifiedConfigManager()
        super().__init__(model_id, ModelType.RANDOM_FOREST, config)
        
        # Initialize scikit-learn model with config parameters
        self.model = RandomForestClassifier(
            n_estimators=self.config_manager.get_int('n_estimators', 100) if config else 100,
            max_depth=self.config_manager.get_int('max_depth', 10) if config else 10,
            min_samples_split=self.config_manager.get_int('min_samples_split', 5) if config else 5,
            min_samples_leaf=self.config_manager.get_int('min_samples_leaf', 2) if config else 2,
            random_state=self.config_manager.get_int('random_state', 42) if config else 42,
            class_weight='balanced',
            n_jobs=-1  # Use all available cores
        )
        
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
    def train(self, features: pd.DataFrame, targets: pd.Series, sample_weights: Optional[np.ndarray] = None) -> None:
        """Train random forest using scikit-learn."""
        try:
            # Prepare data
            X = features.fillna(0)
            y = targets.fillna('HOLD')
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Encode labels
            y_encoded = self.label_encoder.fit_transform(y)
            
            # Train the model
            if sample_weights is not None:
                self.model.fit(X_scaled, y_encoded, sample_weight=sample_weights)
            else:
                self.model.fit(X_scaled, y_encoded)
            
            # Calculate performance metrics
            predictions = self.model.predict(X_scaled)
            accuracy = accuracy_score(y_encoded, predictions)
            
            # Cross-validation score
            cv_scores = cross_val_score(self.model, X_scaled, y_encoded, cv=3)
            
            self.performance_metrics = {
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'oob_score': getattr(self.model, 'oob_score_', None)
            }
            
            self.is_trained = True
            self.training_data_hash = str(hash(str(features.values.tobytes())))
            
            logger.info(f"RandomForest {self.model_id} trained with accuracy: {accuracy:.3f}")
            
        except Exception as e:
            logger.error(f"Random forest training failed: {str(e)}")
            raise
    
    def predict(self, features: pd.DataFrame) -> ModelPrediction:
        """Make prediction using trained random forest."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        try:
            # Prepare data
            X = features.fillna(0)
            X_scaled = self.scaler.transform(X)
            
            # Make prediction
            prediction_encoded = self.model.predict(X_scaled)[0]
            prediction_proba = self.model.predict_proba(X_scaled)[0]
            
            # Decode prediction
            prediction = self.label_encoder.inverse_transform([prediction_encoded])[0]
            
            # Get confidence (max probability)
            confidence = np.max(prediction_proba)
            
            # Create probability distribution
            classes = self.label_encoder.classes_
            prob_dist = dict(zip(classes, prediction_proba))
            
            # Ensure all required classes are present
            for class_name in ['BUY', 'SELL', 'HOLD']:
                if class_name not in prob_dist:
                    prob_dist[class_name] = 0.0
            
            return ModelPrediction(
                model_id=self.model_id,
                model_type=self.model_type,
                prediction=prediction,
                confidence=confidence,
                probability_distribution=prob_dist,
                feature_importance=self.get_feature_importance(),
                timestamp=datetime.now(),
                data_subset_hash=self.training_data_hash
            )
            
        except Exception as e:
            logger.error(f"Random forest prediction failed: {str(e)}")
            raise
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the trained model."""
        if not self.is_trained:
            return {}
        
        try:
            importances = self.model.feature_importances_
            feature_names = list(range(len(importances)))
            return dict(zip(feature_names, importances))
        except Exception as e:
            logger.error(f"Failed to get feature importance: {str(e)}")
            return {}


class SimpleLogisticRegressionModel(BaseEnsembleModel):
    """Logistic regression model using scikit-learn."""
    
    def __init__(self, model_id: str, config: Optional[Dict[str, Any]] = None):
        # Note: This class now requires config_manager parameter in __init__
        # self.config_manager = config_manager or UnifiedConfigManager()
        super().__init__(model_id, ModelType.LOGISTIC_REGRESSION, config)
        
        # Initialize scikit-learn model with config parameters
        self.model = LogisticRegression(
            C=self.config_manager.get_float('C', 1.0) if config else 1.0,
            penalty=self.config_manager.get_str('penalty', 'l2') if config else 'l2',
            solver=self.config_manager.get_str('solver', 'lbfgs') if config else 'lbfgs',
            max_iter=self.config_manager.get_int('max_iter', 1000) if config else 1000,
            random_state=self.config_manager.get_int('random_state', 42) if config else 42,
            class_weight='balanced'
        )
        
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
    def train(self, features: pd.DataFrame, targets: pd.Series, sample_weights: Optional[np.ndarray] = None) -> None:
        """Train logistic regression using scikit-learn."""
        try:
            # Prepare data
            X = features.fillna(0)
            y = targets.fillna('HOLD')
            
            # Scale features (important for logistic regression)
            X_scaled = self.scaler.fit_transform(X)
            
            # Encode labels
            y_encoded = self.label_encoder.fit_transform(y)
            
            # Train the model
            if sample_weights is not None:
                self.model.fit(X_scaled, y_encoded, sample_weight=sample_weights)
            else:
                self.model.fit(X_scaled, y_encoded)
            
            # Calculate performance metrics
            predictions = self.model.predict(X_scaled)
            accuracy = accuracy_score(y_encoded, predictions)
            
            # Cross-validation score
            cv_scores = cross_val_score(self.model, X_scaled, y_encoded, cv=3)
            
            self.performance_metrics = {
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            self.is_trained = True
            self.training_data_hash = str(hash(str(features.values.tobytes())))
            
            logger.info(f"LogisticRegression {self.model_id} trained with accuracy: {accuracy:.3f}")
            
        except Exception as e:
            logger.error(f"Logistic regression training failed: {str(e)}")
            raise
    
    def predict(self, features: pd.DataFrame) -> ModelPrediction:
        """Make prediction using trained logistic regression."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        try:
            # Prepare data
            X = features.fillna(0)
            X_scaled = self.scaler.transform(X)
            
            # Make prediction
            prediction_encoded = self.model.predict(X_scaled)[0]
            prediction_proba = self.model.predict_proba(X_scaled)[0]
            
            # Decode prediction
            prediction = self.label_encoder.inverse_transform([prediction_encoded])[0]
            
            # Get confidence (max probability)
            confidence = np.max(prediction_proba)
            
            # Create probability distribution
            classes = self.label_encoder.classes_
            prob_dist = dict(zip(classes, prediction_proba))
            
            # Ensure all required classes are present
            for class_name in ['BUY', 'SELL', 'HOLD']:
                if class_name not in prob_dist:
                    prob_dist[class_name] = 0.0
            
            return ModelPrediction(
                model_id=self.model_id,
                model_type=self.model_type,
                prediction=prediction,
                confidence=confidence,
                probability_distribution=prob_dist,
                feature_importance=self.get_feature_importance(),
                timestamp=datetime.now(),
                data_subset_hash=self.training_data_hash
            )
            
        except Exception as e:
            logger.error(f"Logistic regression prediction failed: {str(e)}")
            raise
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from coefficients."""
        if not self.is_trained:
            return {}
        
        try:
            # For multi-class, take the average of absolute coefficients
            if len(self.model.coef_.shape) > 1:
                importances = np.mean(np.abs(self.model.coef_), axis=0)
            else:
                importances = np.abs(self.model.coef_[0])
            
            # Normalize to sum to 1
            if importances.sum() > 0:
                importances = importances / importances.sum()
            
            feature_names = list(range(len(importances)))
            return dict(zip(feature_names, importances))
        except Exception as e:
            logger.error(f"Failed to get feature importance: {str(e)}")
            return {}


class EnsembleModelSystem:
    """
    Unified ensemble model system that combines multiple ML models.
    Provides a high-level interface for the ensemble functionality.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Note: This class now requires config_manager parameter in __init__
        # self.config_manager = config_manager or UnifiedConfigManager()
        """Initialize the ensemble model system."""
        self.config = config or {}
        self.models = []
        self.is_trained = False
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize the ensemble models."""
        try:
            # Create different model types for diversity
            models_config = [
                {"model_id": "tree_1", "model_type": ModelType.DECISION_TREE},
                {"model_id": "forest_1", "model_type": ModelType.RANDOM_FOREST},
                {"model_id": "logistic_1", "model_type": ModelType.LOGISTIC_REGRESSION}
            ]
            
            for model_config in models_config:
                if model_config["model_type"] == ModelType.DECISION_TREE:
                    model = SimpleDecisionTreeModel(
                        model_config["model_id"],
                        model_config["model_type"],
                        self.config
                    )
                elif model_config["model_type"] == ModelType.RANDOM_FOREST:
                    model = SimpleRandomForestModel(
                        model_config["model_id"],
                        model_config["model_type"],
                        self.config
                    )
                elif model_config["model_type"] == ModelType.LOGISTIC_REGRESSION:
                    model = SimpleLogisticRegressionModel(
                        model_config["model_id"],
                        model_config["model_type"],
                        self.config
                    )
                else:
                    continue
                    
                self.models.append(model)
                
        except Exception as e:
            logger.error(f"Failed to initialize ensemble models: {str(e)}")
    
    def train(self, X, y):
        """Train all models in the ensemble."""
        try:
            for model in self.models:
                model.train(X, y)
            self.is_trained = True
            return True
        except Exception as e:
            logger.error(f"Failed to train ensemble: {str(e)}")
            return False
    
    def predict(self, X):
        """Make predictions using the ensemble."""
        if not self.is_trained:
            raise ValueError("Models must be trained before making predictions")
        
        try:
            predictions = []
            for model in self.models:
                pred = model.predict(X)
                predictions.append(pred)
            
            # Simple voting - return most common prediction
            if predictions:
                return max(set(predictions), key=predictions.count)
            else:
                return "HOLD"  # Default prediction
                
        except Exception as e:
            logger.error(f"Failed to make ensemble prediction: {str(e)}")
            return "HOLD"