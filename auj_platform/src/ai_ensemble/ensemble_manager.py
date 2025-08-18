"""
Ensemble Manager for Anti-Overfitting AI System.

This module manages multiple models and implements voting mechanisms
to prevent overfitting and improve decision robustness.
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import hashlib
import random

from .ensemble_models import (
    BaseEnsembleModel, SimpleDecisionTreeModel, SimpleRandomForestModel, 
    SimpleLogisticRegressionModel, ModelPrediction, EnsemblePrediction,
    VotingMethod, ModelType
)
from ..core.logging_setup import get_logger

logger = get_logger(__name__)


class EnsembleManager:
    """
    Manages ensemble of models for robust decision making.
    
    Key features:
    - Multiple model types to avoid single-model bias
    - Data subset training to prevent overfitting
    - Adaptive voting based on model performance
    - Consensus strength measurement
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Note: This class now requires config_manager parameter in __init__
        # self.config_manager = config_manager or UnifiedConfigManager()
        self.config = config or {}
        self.models: List[BaseEnsembleModel] = []
        self.model_performance_history = {}
        self.training_history = []
        self.prediction_history = []
        
        # Configuration parameters
        self.min_models = self.config_manager.get_int('min_models', 3)
        self.max_models = self.config_manager.get_int('max_models', 7)
        self.data_subset_ratio = self.config_manager.get_float('data_subset_ratio', 0.8)
        self.performance_window = self.config_manager.get_int('performance_window', 50)
        self.consensus_threshold = self.config_manager.get_float('consensus_threshold', 0.6)
        
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize diverse set of models for ensemble."""
        try:
            # Create multiple models of different types
            model_configs = [
                (SimpleDecisionTreeModel, {"max_depth": 5}),
                (SimpleDecisionTreeModel, {"max_depth": 3}),
                (SimpleRandomForestModel, {"n_trees": 5}),
                (SimpleRandomForestModel, {"n_trees": 3}),
                (SimpleLogisticRegressionModel, {"regularization": 0.01}),
            ]
            
            for i, (model_class, config) in enumerate(model_configs):
                model_id = f"{model_class.__name__}_{i}"
                model = model_class(model_id, config)
                self.models.append(model)
                
                # Initialize performance tracking
                self.model_performance_history[model_id] = {
                    "accuracy": [],
                    "confidence": [],
                    "predictions": [],
                    "last_updated": datetime.now()
                }
            
            logger.info(f"Initialized ensemble with {len(self.models)} models")
            
        except Exception as e:
            logger.error(f"Failed to initialize ensemble models: {str(e)}")
            raise
    
    def train_ensemble(self, features: pd.DataFrame, targets: pd.Series) -> Dict[str, Any]:
        """
        Train all models in ensemble with different data subsets.
        
        This prevents overfitting by ensuring each model sees only a subset
        of the data and has different perspectives.
        """
        try:
            training_results = {
                "models_trained": 0,
                "training_errors": [],
                "data_subsets": [],
                "training_timestamp": datetime.now()
            }
            
            n_samples = len(features)
            subset_size = int(n_samples * self.data_subset_ratio)
            
            for model in self.models:
                try:
                    # Create unique data subset for this model
                    subset_indices = self._create_data_subset(n_samples, subset_size, model.model_id)
                    subset_features = features.iloc[subset_indices]
                    subset_targets = targets.iloc[subset_indices]
                    
                    # Train model on subset
                    model.train(subset_features, subset_targets)
                    
                    # Record subset information
                    subset_hash = hashlib.md5(str(subset_indices).encode()).hexdigest()
                    training_results["data_subsets"].append({
                        "model_id": model.model_id,
                        "subset_size": len(subset_indices),
                        "subset_hash": subset_hash
                    })
                    
                    training_results["models_trained"] += 1
                    
                except Exception as e:
                    error_msg = f"Model {model.model_id} training failed: {str(e)}"
                    training_results["training_errors"].append(error_msg)
                    logger.warning(error_msg)
            
            self.training_history.append(training_results)
            
            logger.info(f"Ensemble training completed: {training_results['models_trained']}/{len(self.models)} models trained")
            
            return training_results
            
        except Exception as e:
            logger.error(f"Ensemble training failed: {str(e)}")
            raise
    
    def _create_data_subset(self, n_samples: int, subset_size: int, model_id: str) -> List[int]:
        """Create reproducible but unique data subset for each model."""
        # Use model_id as seed for reproducibility
        random.seed(hash(model_id) % (2**32))
        
        # Different sampling strategies for diversity
        if "tree" in model_id.lower():
            # Random sampling for tree models
            indices = random.sample(range(n_samples), subset_size)
        elif "forest" in model_id.lower():
            # Bootstrap sampling for forest models
            indices = [random.randint(0, n_samples - 1) for _ in range(subset_size)]
        else:
            # Stratified sampling for other models
            step = n_samples / subset_size
            indices = [int(i * step) for i in range(subset_size)]
        
        return sorted(indices)
    
    def predict_ensemble(self, features: pd.DataFrame, voting_method: VotingMethod = VotingMethod.ADAPTIVE) -> EnsemblePrediction:
        """
        Generate ensemble prediction using specified voting method.
        """
        try:
            # Get predictions from all trained models
            individual_predictions = []
            
            for model in self.models:
                if model.is_trained:
                    try:
                        prediction = model.predict(features)
                        individual_predictions.append(prediction)
                    except Exception as e:
                        logger.warning(f"Model {model.model_id} prediction failed: {str(e)}")
            
            if not individual_predictions:
                raise ValueError("No trained models available for prediction")
            
            # Apply voting method
            final_prediction, ensemble_confidence = self._apply_voting_method(
                individual_predictions, voting_method
            )
            
            # Calculate ensemble metrics
            consensus_strength = self._calculate_consensus_strength(individual_predictions)
            disagreement_score = self._calculate_disagreement_score(individual_predictions)
            overfitting_risk = self._assess_overfitting_risk(individual_predictions)
            
            ensemble_result = EnsemblePrediction(
                final_prediction=final_prediction,
                ensemble_confidence=ensemble_confidence,
                individual_predictions=individual_predictions,
                voting_method=voting_method,
                consensus_strength=consensus_strength,
                disagreement_score=disagreement_score,
                overfitting_risk=overfitting_risk,
                timestamp=datetime.now()
            )
            
            # Record prediction for performance tracking
            self.prediction_history.append(ensemble_result)
            
            return ensemble_result
            
        except Exception as e:
            logger.error(f"Ensemble prediction failed: {str(e)}")
            raise
    
    def _apply_voting_method(self, predictions: List[ModelPrediction], method: VotingMethod) -> Tuple[str, float]:
        """Apply specified voting method to combine predictions."""
        
        if method == VotingMethod.MAJORITY:
            return self._majority_vote(predictions)
        elif method == VotingMethod.WEIGHTED:
            return self._weighted_vote(predictions)
        elif method == VotingMethod.SOFT_VOTING:
            return self._soft_vote(predictions)
        elif method == VotingMethod.ADAPTIVE:
            return self._adaptive_vote(predictions)
        else:
            return self._majority_vote(predictions)
    
    def _majority_vote(self, predictions: List[ModelPrediction]) -> Tuple[str, float]:
        """Simple majority voting."""
        votes = [pred.prediction for pred in predictions]
        
        # Count votes
        vote_counts = {}
        for vote in votes:
            vote_counts[vote] = vote_counts.get(vote, 0) + 1
        
        # Find winner
        winner = max(vote_counts, key=vote_counts.get)
        confidence = vote_counts[winner] / len(votes)
        
        return winner, confidence
    
    def _weighted_vote(self, predictions: List[ModelPrediction]) -> Tuple[str, float]:
        """Weighted voting based on model confidence."""
        weighted_votes = {}
        total_weight = 0
        
        for pred in predictions:
            weight = pred.confidence
            vote = pred.prediction
            
            if vote not in weighted_votes:
                weighted_votes[vote] = 0
            weighted_votes[vote] += weight
            total_weight += weight
        
        if total_weight == 0:
            return self._majority_vote(predictions)
        
        # Normalize weights
        for vote in weighted_votes:
            weighted_votes[vote] /= total_weight
        
        winner = max(weighted_votes, key=weighted_votes.get)
        confidence = weighted_votes[winner]
        
        return winner, confidence
    
    def _soft_vote(self, predictions: List[ModelPrediction]) -> Tuple[str, float]:
        """Soft voting using probability distributions."""
        combined_probabilities = {"BUY": 0, "SELL": 0, "HOLD": 0, "NO_SIGNAL": 0}
        
        for pred in predictions:
            for decision, prob in pred.probability_distribution.items():
                if decision in combined_probabilities:
                    combined_probabilities[decision] += prob
        
        # Normalize
        total_prob = sum(combined_probabilities.values())
        if total_prob > 0:
            for decision in combined_probabilities:
                combined_probabilities[decision] /= total_prob
        
        winner = max(combined_probabilities, key=combined_probabilities.get)
        confidence = combined_probabilities[winner]
        
        return winner, confidence
    
    def _adaptive_vote(self, predictions: List[ModelPrediction]) -> Tuple[str, float]:
        """Adaptive voting considering model performance history."""
        # Use soft voting as baseline for adaptive approach
        soft_winner, soft_confidence = self._soft_vote(predictions)
        
        # Adjust based on consensus strength
        consensus = self._calculate_consensus_strength(predictions)
        
        if consensus >= self.consensus_threshold:
            # High consensus - trust the soft vote
            return soft_winner, soft_confidence * consensus
        else:
            # Low consensus - be more conservative
            if soft_confidence < 0.6:
                return "HOLD", soft_confidence * 0.5
            else:
                return soft_winner, soft_confidence * 0.8
    
    def _calculate_consensus_strength(self, predictions: List[ModelPrediction]) -> float:
        """Calculate how much the models agree."""
        if len(predictions) <= 1:
            return 1.0
        
        votes = [pred.prediction for pred in predictions]
        
        # Most common vote
        vote_counts = {}
        for vote in votes:
            vote_counts[vote] = vote_counts.get(vote, 0) + 1
        
        max_agreement = max(vote_counts.values())
        consensus_strength = max_agreement / len(votes)
        
        return consensus_strength
    
    def _calculate_disagreement_score(self, predictions: List[ModelPrediction]) -> float:
        """Calculate level of disagreement among models."""
        if len(predictions) <= 1:
            return 0.0
        
        votes = [pred.prediction for pred in predictions]
        unique_votes = len(set(votes))
        max_possible_disagreement = min(len(votes), 4)  # BUY, SELL, HOLD, NO_SIGNAL
        
        disagreement = unique_votes / max_possible_disagreement
        return disagreement
    
    def _assess_overfitting_risk(self, predictions: List[ModelPrediction]) -> float:
        """Assess risk of overfitting based on model behavior."""
        if len(predictions) <= 1:
            return 0.5
        
        # Check confidence variance
        confidences = [pred.confidence for pred in predictions]
        confidence_variance = np.var(confidences)
        
        # Check if models trained on different subsets agree too much
        data_hashes = [pred.data_subset_hash for pred in predictions]
        unique_hashes = len(set(data_hashes))
        
        if unique_hashes == len(predictions):
            # All models trained on different data
            hash_diversity = 1.0
        else:
            hash_diversity = unique_hashes / len(predictions)
        
        # High agreement + low hash diversity = potential overfitting
        consensus = self._calculate_consensus_strength(predictions)
        
        if consensus > 0.9 and hash_diversity < 0.5:
            overfitting_risk = 0.8
        elif confidence_variance < 0.01:  # Very similar confidences
            overfitting_risk = 0.6
        else:
            overfitting_risk = max(0.1, 1.0 - hash_diversity) * consensus
        
        return min(overfitting_risk, 1.0)    
    def update_model_performance(self, actual_outcome: str, prediction_timestamp: datetime) -> Dict[str, Any]:
        """
        Update model performance metrics based on actual outcomes.
        This helps with adaptive weighting and model selection.
        """
        try:
            # Find prediction in history
            matching_prediction = None
            for pred_result in reversed(self.prediction_history):
                if abs((pred_result.timestamp - prediction_timestamp).total_seconds()) < 60:
                    matching_prediction = pred_result
                    break
            
            if not matching_prediction:
                logger.warning("No matching prediction found for performance update")
                return {}
            
            performance_update = {
                "timestamp": datetime.now(),
                "actual_outcome": actual_outcome,
                "predicted_outcome": matching_prediction.final_prediction,
                "ensemble_correct": actual_outcome == matching_prediction.final_prediction,
                "model_performances": {}
            }
            
            # Update individual model performances
            for pred in matching_prediction.individual_predictions:
                model_id = pred.model_id
                is_correct = actual_outcome == pred.prediction
                
                if model_id in self.model_performance_history:
                    history = self.model_performance_history[model_id]
                    history["predictions"].append({
                        "predicted": pred.prediction,
                        "actual": actual_outcome,
                        "correct": is_correct,
                        "confidence": pred.confidence,
                        "timestamp": prediction_timestamp
                    })
                    
                    # Maintain window size
                    if len(history["predictions"]) > self.performance_window:
                        history["predictions"] = history["predictions"][-self.performance_window:]
                    
                    # Update accuracy
                    recent_predictions = history["predictions"]
                    correct_predictions = sum(1 for p in recent_predictions if p["correct"])
                    accuracy = correct_predictions / len(recent_predictions) if recent_predictions else 0.0
                    history["accuracy"].append(accuracy)
                    
                    # Update confidence
                    avg_confidence = np.mean([p["confidence"] for p in recent_predictions])
                    history["confidence"].append(avg_confidence)
                    
                    history["last_updated"] = datetime.now()
                    
                    performance_update["model_performances"][model_id] = {
                        "accuracy": accuracy,
                        "confidence": avg_confidence,
                        "correct": is_correct
                    }
            
            logger.info(f"Performance updated: Ensemble {'correct' if performance_update['ensemble_correct'] else 'incorrect'}")
            
            return performance_update
            
        except Exception as e:
            logger.error(f"Failed to update model performance: {str(e)}")
            return {}
    
    def get_model_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary for all models."""
        summary = {
            "ensemble_summary": {
                "total_predictions": len(self.prediction_history),
                "models_count": len(self.models),
                "trained_models": sum(1 for model in self.models if model.is_trained)
            },
            "model_performances": {},
            "overfitting_analysis": self._analyze_overfitting_trends()
        }
        
        for model_id, history in self.model_performance_history.items():
            if history["predictions"]:
                recent_accuracy = history["accuracy"][-1] if history["accuracy"] else 0.0
                recent_confidence = history["confidence"][-1] if history["confidence"] else 0.0
                
                summary["model_performances"][model_id] = {
                    "current_accuracy": recent_accuracy,
                    "average_confidence": recent_confidence,
                    "prediction_count": len(history["predictions"]),
                    "last_updated": history["last_updated"],
                    "accuracy_trend": self._calculate_trend(history["accuracy"]),
                    "confidence_trend": self._calculate_trend(history["confidence"])
                }
        
        return summary
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for a list of values."""
        if len(values) < 2:
            return "insufficient_data"
        
        recent = values[-min(5, len(values)):]
        if len(recent) < 2:
            return "stable"
        
        slope = (recent[-1] - recent[0]) / len(recent)
        
        if slope > 0.02:
            return "improving"
        elif slope < -0.02:
            return "declining"
        else:
            return "stable"
    
    def _analyze_overfitting_trends(self) -> Dict[str, Any]:
        """Analyze overfitting trends across recent predictions."""
        if len(self.prediction_history) < 10:
            return {"status": "insufficient_data"}
        
        recent_predictions = self.prediction_history[-20:]
        
        # Analyze overfitting risk trends
        overfitting_risks = [pred.overfitting_risk for pred in recent_predictions]
        avg_overfitting_risk = np.mean(overfitting_risks)
        
        # Analyze consensus trends
        consensus_strengths = [pred.consensus_strength for pred in recent_predictions]
        avg_consensus = np.mean(consensus_strengths)
        
        # Analyze disagreement trends
        disagreement_scores = [pred.disagreement_score for pred in recent_predictions]
        avg_disagreement = np.mean(disagreement_scores)
        
        analysis = {
            "average_overfitting_risk": avg_overfitting_risk,
            "average_consensus_strength": avg_consensus,
            "average_disagreement": avg_disagreement,
            "overfitting_trend": self._calculate_trend(overfitting_risks),
            "consensus_trend": self._calculate_trend(consensus_strengths),
            "status": "healthy"
        }
        
        # Determine overall status
        if avg_overfitting_risk > 0.7:
            analysis["status"] = "high_overfitting_risk"
        elif avg_consensus > 0.95 and avg_disagreement < 0.1:
            analysis["status"] = "potential_overfitting"
        elif avg_disagreement > 0.8:
            analysis["status"] = "high_disagreement"
        
        return analysis
    
    def retrain_if_needed(self, features: pd.DataFrame, targets: pd.Series) -> bool:
        """
        Determine if ensemble needs retraining based on performance metrics.
        """
        try:
            performance_summary = self.get_model_performance_summary()
            overfitting_analysis = performance_summary["overfitting_analysis"]
            
            should_retrain = False
            retrain_reasons = []
            
            # Check for performance decline
            declining_models = 0
            for model_id, perf in performance_summary["model_performances"].items():
                if perf["accuracy_trend"] == "declining":
                    declining_models += 1
            
            if declining_models > len(self.models) / 2:
                should_retrain = True
                retrain_reasons.append("majority_models_declining")
            
            # Check for overfitting
            if overfitting_analysis["status"] in ["high_overfitting_risk", "potential_overfitting"]:
                should_retrain = True
                retrain_reasons.append(f"overfitting_detected: {overfitting_analysis['status']}")
            
            # Check for insufficient training
            trained_models = performance_summary["ensemble_summary"]["trained_models"]
            if trained_models < self.min_models:
                should_retrain = True
                retrain_reasons.append("insufficient_trained_models")
            
            if should_retrain:
                logger.info(f"Retraining ensemble due to: {', '.join(retrain_reasons)}")
                training_result = self.train_ensemble(features, targets)
                return training_result["models_trained"] >= self.min_models
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to check retrain status: {str(e)}")
            return False
    
    def get_ensemble_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the ensemble system."""
        status = {
            "timestamp": datetime.now(),
            "models": {
                "total": len(self.models),
                "trained": sum(1 for model in self.models if model.is_trained),
                "model_types": [model.model_type.value for model in self.models]
            },
            "predictions": {
                "total": len(self.prediction_history),
                "recent_consensus": 0.0,
                "recent_overfitting_risk": 0.0
            },
            "performance": self.get_model_performance_summary(),
            "health_status": "unknown"
        }
        
        # Calculate recent metrics
        if self.prediction_history:
            recent_predictions = self.prediction_history[-10:]
            status["predictions"]["recent_consensus"] = np.mean([p.consensus_strength for p in recent_predictions])
            status["predictions"]["recent_overfitting_risk"] = np.mean([p.overfitting_risk for p in recent_predictions])
        
        # Determine health status
        trained_ratio = status["models"]["trained"] / status["models"]["total"]
        overfitting_status = status["performance"]["overfitting_analysis"]["status"]
        
        if trained_ratio < 0.5:
            status["health_status"] = "unhealthy_insufficient_training"
        elif overfitting_status in ["high_overfitting_risk", "potential_overfitting"]:
            status["health_status"] = f"warning_{overfitting_status}"
        elif trained_ratio >= 0.8 and overfitting_status == "healthy":
            status["health_status"] = "healthy"
        else:
            status["health_status"] = "moderate"
        
        return status