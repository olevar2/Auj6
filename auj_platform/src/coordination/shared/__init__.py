"""
Shared coordination components.

This package contains common types and utilities used across the coordination system.
"""

from .coordination_types import (
    AnalysisCyclePhase,
    AnalysisCycleState,
    EliteIndicatorSet
)
from .coordination_exceptions import (
    CoordinationError,
    ValidationError,
    AgentError
)

__all__ = [
    'AnalysisCyclePhase',
    'AnalysisCycleState',
    'EliteIndicatorSet',
    'CoordinationError',
    'ValidationError',
    'AgentError',
]
