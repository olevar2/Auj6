"""
Shared data structures for coordination system.

This module contains common types used across the coordination components
to avoid circular dependencies and improve cohesion.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any

from ...core.data_contracts import (
    TradeSignal, MarketConditions, MarketRegime
)
from ...core.event_bus import Event
from ...agents.base_agent import AnalysisResult


class AnalysisCyclePhase(str, Enum):
    """Phases of the analysis cycle."""
    INITIALIZATION = "INITIALIZATION"
    DATA_COLLECTION = "DATA_COLLECTION"
    INDICATOR_CALCULATION = "INDICATOR_CALCULATION"
    AGENT_ANALYSIS = "AGENT_ANALYSIS"
    DECISION_SYNTHESIS = "DECISION_SYNTHESIS"
    VALIDATION = "VALIDATION"
    EXECUTION_PREPARATION = "EXECUTION_PREPARATION"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"


@dataclass
class AnalysisCycleState:
    """State tracking for analysis cycle."""
    cycle_id: str
    phase: AnalysisCyclePhase
    start_time: datetime
    symbol: str
    alpha_agent: Optional[str] = None
    beta_agents: List[str] = field(default_factory=list)
    gamma_agents: List[str] = field(default_factory=list)
    agent_decisions: Dict[str, AnalysisResult] = field(default_factory=dict)
    final_signal: Optional[TradeSignal] = None
    performance_data: Dict[str, Any] = field(default_factory=dict)
    error_log: List[str] = field(default_factory=list)
    market_conditions: Optional[MarketConditions] = None
    trigger_event: Optional[Event] = None

    @property
    def duration_seconds(self) -> float:
        """Get cycle duration in seconds."""
        return (datetime.utcnow() - self.start_time).total_seconds()

    @property
    def is_successful(self) -> bool:
        """Check if cycle completed successfully."""
        return self.phase == AnalysisCyclePhase.COMPLETED and not self.error_log


@dataclass
class EliteIndicatorSet:
    """Elite indicator set for specific market regime."""
    regime: MarketRegime
    indicators: List[str]
    performance_score: float
    validation_score: float
    last_updated: datetime
    out_of_sample_performance: Dict[str, float]
    robustness_metrics: Dict[str, float]

    @property
    def composite_score(self) -> float:
        """Calculate composite score favoring out-of-sample performance."""
        return (self.validation_score * 0.7) + (self.performance_score * 0.3)
