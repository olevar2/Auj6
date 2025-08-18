"""
AI Ensemble Package for AUJ Platform.

This package provides anti-overfitting ensemble modeling capabilities
for robust AI-enhanced decision making.
"""

from .ensemble_models import (
    BaseEnsembleModel,
    SimpleDecisionTreeModel,
    SimpleRandomForestModel, 
    SimpleLogisticRegressionModel,
    ModelPrediction,
    EnsemblePrediction,
    ModelType,
    VotingMethod
)

from .ensemble_manager import EnsembleManager

__all__ = [
    "BaseEnsembleModel",
    "SimpleDecisionTreeModel",
    "SimpleRandomForestModel",
    "SimpleLogisticRegressionModel", 
    "ModelPrediction",
    "EnsemblePrediction",
    "ModelType",
    "VotingMethod",
    "EnsembleManager"
]