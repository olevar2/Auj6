"""
Economic Calendar Models and Data Structures
============================================

This module contains specialized data structures for economic calendar functionality,
including market impact modeling, event correlation analysis, and trading signal generation.
"""

from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Tuple
from pydantic import BaseModel, Field, field_validator, model_validator
import uuid
import numpy as np

from .data_contracts import (
    EconomicEvent, EconomicEventImpact, EconomicEventCategory, 
    CurrencyCode, EconomicEventTradingSignal
)


class MarketSession(str, Enum):
    """Trading session enumeration."""
    SYDNEY = "SYDNEY"
    TOKYO = "TOKYO"
    LONDON = "LONDON"
    NEW_YORK = "NEW_YORK"
    OVERLAP_ASIAN_EUROPEAN = "OVERLAP_ASIAN_EUROPEAN"
    OVERLAP_EUROPEAN_AMERICAN = "OVERLAP_EUROPEAN_AMERICAN"


class EventCorrelationType(str, Enum):
    """Types of correlations between economic events."""
    LEADING = "LEADING"      # Event A leads to Event B
    LAGGING = "LAGGING"      # Event A follows Event B
    CONCURRENT = "CONCURRENT" # Events happen simultaneously
    INVERSE = "INVERSE"      # Events move in opposite directions
    REINFORCING = "REINFORCING" # Events strengthen each other's impact


class TradingStrategy(str, Enum):
    """Economic event-based trading strategies."""
    PRE_EVENT_POSITIONING = "PRE_EVENT_POSITIONING"
    NEWS_TRADING = "NEWS_TRADING"
    FADE_THE_MOVE = "FADE_THE_MOVE"
    TREND_CONTINUATION = "TREND_CONTINUATION"
    VOLATILITY_BREAKOUT = "VOLATILITY_BREAKOUT"
    MEAN_REVERSION = "MEAN_REVERSION"
    CORRELATION_PLAY = "CORRELATION_PLAY"


class EventCluster(BaseModel):
    """Cluster of related economic events occurring within a time window."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description="Cluster name (e.g., 'US Employment Week')")
    start_time: datetime = Field(..., description="Cluster start time")
    end_time: datetime = Field(..., description="Cluster end time")
    
    # Events in cluster
    primary_events: List[str] = Field(default_factory=list, description="Main events driving the cluster")
    secondary_events: List[str] = Field(default_factory=list, description="Supporting events")
    
    # Cluster characteristics
    dominant_currency: CurrencyCode = Field(..., description="Primary currency affected")
    overall_impact: EconomicEventImpact = Field(..., description="Combined impact level")
    theme: str = Field(..., description="Market theme (e.g., 'Inflation fears', 'Growth concerns')")
    
    # Impact predictions
    expected_volatility_spike: float = Field(default=0.0, ge=0.0, description="Expected volatility increase %")
    affected_instruments: List[str] = Field(default_factory=list)
    trading_opportunities: List[TradingStrategy] = Field(default_factory=list)
    
    # Risk assessment
    uncertainty_level: float = Field(default=0.5, ge=0.0, le=1.0, description="Uncertainty about outcomes")
    conflicting_signals_risk: float = Field(default=0.0, ge=0.0, le=1.0, description="Risk of conflicting signals")


class EconomicEventCorrelation(BaseModel):
    """Correlation analysis between two economic events."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
    # Event relationships
    primary_event_id: str = Field(..., description="Primary event ID")
    secondary_event_id: str = Field(..., description="Secondary event ID")
    correlation_type: EventCorrelationType = Field(..., description="Type of correlation")
    
    # Statistical measures
    correlation_coefficient: float = Field(..., ge=-1.0, le=1.0, description="Statistical correlation")
    confidence_level: float = Field(..., ge=0.0, le=1.0, description="Confidence in correlation")
    sample_size: int = Field(..., gt=0, description="Number of observations")
    
    # Timing relationships
    typical_lag_minutes: Optional[int] = Field(None, description="Typical lag between events")
    impact_duration_minutes: int = Field(default=60, description="How long correlation impact lasts")
    
    # Market impact
    combined_impact_multiplier: float = Field(default=1.0, ge=0.0, description="Impact multiplier when both events occur")
    instruments_affected: List[str] = Field(default_factory=list)
    
    # Historical analysis
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    historical_accuracy: float = Field(default=0.5, ge=0.0, le=1.0, description="Historical prediction accuracy")


class MarketVolatilityModel(BaseModel):
    """Model for predicting market volatility around economic events."""
    event_category: EconomicEventCategory = Field(..., description="Category of economic event")
    currency: CurrencyCode = Field(..., description="Currency affected")
    
    # Volatility patterns
    pre_event_volatility_change: float = Field(default=0.0, description="Volatility change % before event")
    event_volatility_spike: float = Field(default=0.0, ge=0.0, description="Volatility spike % during event")
    post_event_volatility_decay: float = Field(default=0.0, description="Volatility decay % after event")
    
    # Timing windows
    pre_event_window_minutes: int = Field(default=30, description="Pre-event analysis window")
    peak_volatility_delay_minutes: int = Field(default=5, description="Time to peak volatility")
    volatility_normalization_minutes: int = Field(default=120, description="Time to return to normal")
    
    # Model parameters
    surprise_sensitivity: float = Field(default=1.0, ge=0.0, description="Sensitivity to data surprises")
    market_session_modifier: Dict[MarketSession, float] = Field(default_factory=dict)
    
    # Historical validation
    model_accuracy: float = Field(default=0.0, ge=0.0, le=1.0, description="Model prediction accuracy")
    confidence_intervals: Dict[str, Tuple[float, float]] = Field(default_factory=dict)
    last_updated: datetime = Field(default_factory=datetime.utcnow)


class EconomicSentimentIndex(BaseModel):
    """Economic sentiment index based on recent economic events."""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    currency: CurrencyCode = Field(..., description="Currency being analyzed")
    
    # Sentiment scores
    overall_sentiment: float = Field(..., ge=-1.0, le=1.0, description="Overall economic sentiment")
    growth_sentiment: float = Field(..., ge=-1.0, le=1.0, description="Economic growth sentiment")
    inflation_sentiment: float = Field(..., ge=-1.0, le=1.0, description="Inflation concerns sentiment")
    employment_sentiment: float = Field(..., ge=-1.0, le=1.0, description="Employment sentiment")
    monetary_policy_sentiment: float = Field(..., ge=-1.0, le=1.0, description="Monetary policy sentiment")
    
    # Sentiment drivers
    recent_events_impact: List[Dict[str, Any]] = Field(default_factory=list)
    sentiment_momentum: float = Field(..., ge=-1.0, le=1.0, description="Rate of sentiment change")
    consensus_strength: float = Field(..., ge=0.0, le=1.0, description="Consensus among events")
    
    # Time-based analysis
    sentiment_history_7d: List[float] = Field(default_factory=list)
    sentiment_volatility: float = Field(default=0.0, ge=0.0, description="Sentiment volatility")
    
    # Market implications
    risk_appetite_indication: str = Field(default="NEUTRAL", description="RISK_ON, RISK_OFF, NEUTRAL")
    currency_strength_bias: float = Field(..., ge=-1.0, le=1.0, description="Currency strength bias")


class EconomicEventFilter(BaseModel):
    """Filter configuration for economic events."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description="Filter name")
    
    # Basic filters
    currencies: List[CurrencyCode] = Field(default_factory=list)
    categories: List[EconomicEventCategory] = Field(default_factory=list)
    impact_levels: List[EconomicEventImpact] = Field(default_factory=list)
    
    # Time filters
    time_range_start: Optional[datetime] = Field(None)
    time_range_end: Optional[datetime] = Field(None)
    trading_sessions: List[MarketSession] = Field(default_factory=list)
    exclude_weekends: bool = Field(default=True)
    exclude_holidays: bool = Field(default=True)
    
    # Advanced filters
    min_surprise_threshold: Optional[float] = Field(None, description="Minimum surprise index")
    keywords_include: List[str] = Field(default_factory=list)
    keywords_exclude: List[str] = Field(default_factory=list)
    
    # Market condition filters
    min_volatility_context: Optional[float] = Field(None, description="Minimum market volatility")
    max_volatility_context: Optional[float] = Field(None, description="Maximum market volatility")
    trending_markets_only: bool = Field(default=False)
    
    # Output configuration
    max_results: int = Field(default=100, gt=0)
    sort_by: str = Field(default="impact_level", description="Sorting criteria")
    include_related_events: bool = Field(default=True)


class TradingSessionEconomicProfile(BaseModel):
    """Economic event profile for different trading sessions."""
    session: MarketSession = Field(..., description="Trading session")
    
    # Session characteristics
    typical_event_count: int = Field(default=0, description="Typical events per session")
    high_impact_frequency: float = Field(default=0.0, ge=0.0, description="High impact events per session")
    dominant_currencies: List[CurrencyCode] = Field(default_factory=list)
    
    # Volume and volatility patterns
    typical_volume_profile: Dict[str, float] = Field(default_factory=dict)
    volatility_patterns: Dict[str, float] = Field(default_factory=dict)
    liquidity_characteristics: Dict[str, str] = Field(default_factory=dict)
    
    # Trading considerations
    recommended_strategies: List[TradingStrategy] = Field(default_factory=list)
    risk_factors: List[str] = Field(default_factory=list)
    optimal_instruments: List[str] = Field(default_factory=list)
    
    # Historical analysis
    success_rate_by_strategy: Dict[TradingStrategy, float] = Field(default_factory=dict)
    average_pip_movement: Dict[str, float] = Field(default_factory=dict)
    typical_reaction_time_minutes: int = Field(default=5)


class EconomicEventPreprocessor(BaseModel):
    """Configuration for preprocessing economic events."""
    # Data cleaning rules
    remove_duplicates: bool = Field(default=True)
    normalize_event_names: bool = Field(default=True)
    standardize_currencies: bool = Field(default=True)
    validate_timestamps: bool = Field(default=True)
    
    # Data enrichment
    calculate_surprise_index: bool = Field(default=True)
    add_historical_context: bool = Field(default=True)
    identify_event_clusters: bool = Field(default=True)
    calculate_correlations: bool = Field(default=True)
    
    # Quality controls
    min_data_quality_score: float = Field(default=0.7, ge=0.0, le=1.0)
    require_forecast_data: bool = Field(default=False)
    exclude_unconfirmed_events: bool = Field(default=True)
    
    # Processing options
    cache_results: bool = Field(default=True)
    cache_duration_hours: int = Field(default=24)
    batch_processing: bool = Field(default=True)
    max_batch_size: int = Field(default=1000)


class EconomicEventMetrics(BaseModel):
    """Metrics and analytics for economic event performance."""
    event_id: str = Field(..., description="Economic event ID")
    analysis_period_start: datetime = Field(...)
    analysis_period_end: datetime = Field(...)
    
    # Prediction accuracy metrics
    forecast_accuracy: Optional[float] = Field(None, ge=0.0, le=1.0)
    direction_accuracy: Optional[float] = Field(None, ge=0.0, le=1.0)
    magnitude_accuracy: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Market impact metrics
    actual_volatility_spike: float = Field(default=0.0, ge=0.0)
    price_movement_accuracy: Dict[str, float] = Field(default_factory=dict)
    volume_spike_ratio: float = Field(default=1.0, ge=0.0)
    
    # Trading performance
    signals_generated: int = Field(default=0, ge=0)
    successful_signals: int = Field(default=0, ge=0)
    signal_accuracy: float = Field(default=0.0, ge=0.0, le=1.0)
    average_pnl_per_signal: float = Field(default=0.0)
    
    # Learning metrics
    model_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    prediction_improvement: float = Field(default=0.0, ge=-1.0, le=1.0)
    correlation_strength: float = Field(default=0.0, ge=0.0, le=1.0)
    
    # Update tracking
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    update_count: int = Field(default=0, ge=0)


# Utility functions for economic calendar data processing

def calculate_event_importance_score(event: EconomicEvent) -> float:
    """Calculate a numerical importance score for an economic event."""
    base_scores = {
        EconomicEventImpact.CRITICAL: 10.0,
        EconomicEventImpact.HIGH: 7.5,
        EconomicEventImpact.MEDIUM: 5.0,
        EconomicEventImpact.LOW: 2.5,
        EconomicEventImpact.NEGLIGIBLE: 1.0
    }
    
    category_multipliers = {
        EconomicEventCategory.MONETARY_POLICY: 1.5,
        EconomicEventCategory.EMPLOYMENT: 1.3,
        EconomicEventCategory.INFLATION: 1.3,
        EconomicEventCategory.GDP_GROWTH: 1.2,
        EconomicEventCategory.MANUFACTURING: 1.0,
        EconomicEventCategory.CONSUMER_SPENDING: 1.0,
        EconomicEventCategory.HOUSING: 0.8,
        EconomicEventCategory.TRADE_BALANCE: 0.8,
        EconomicEventCategory.GOVERNMENT_FISCAL: 0.7,
        EconomicEventCategory.BUSINESS_INVESTMENT: 0.7,
        EconomicEventCategory.FINANCIAL_STABILITY: 1.1,
        EconomicEventCategory.GEOPOLITICAL: 1.4,
        EconomicEventCategory.COMMODITY: 0.9,
        EconomicEventCategory.OTHER: 0.5
    }
    
    base_score = base_scores.get(event.impact_level, 1.0)
    category_multiplier = category_multipliers.get(event.category, 1.0)
    
    # Add surprise factor if available
    surprise_multiplier = 1.0
    if event.surprise_index is not None:
        surprise_multiplier = 1.0 + (abs(event.surprise_index) / 100.0)
    
    return base_score * category_multiplier * surprise_multiplier


def determine_optimal_trading_session(event: EconomicEvent) -> MarketSession:
    """Determine the optimal trading session for an economic event."""
    # Map currencies to their primary trading sessions
    currency_session_map = {
        CurrencyCode.USD: MarketSession.NEW_YORK,
        CurrencyCode.EUR: MarketSession.LONDON,
        CurrencyCode.GBP: MarketSession.LONDON,
        CurrencyCode.JPY: MarketSession.TOKYO,
        CurrencyCode.CHF: MarketSession.LONDON,
        CurrencyCode.CAD: MarketSession.NEW_YORK,
        CurrencyCode.AUD: MarketSession.SYDNEY,
        CurrencyCode.NZD: MarketSession.SYDNEY,
        CurrencyCode.CNY: MarketSession.TOKYO,
    }
    
    # Check if event time aligns with session overlap periods
    event_hour = event.scheduled_time.hour
    
    # Session overlap periods (UTC)
    if 8 <= event_hour <= 9:  # London-Tokyo overlap
        return MarketSession.OVERLAP_ASIAN_EUROPEAN
    elif 13 <= event_hour <= 17:  # London-New York overlap
        return MarketSession.OVERLAP_EUROPEAN_AMERICAN
    else:
        return currency_session_map.get(event.currency, MarketSession.LONDON)


def group_events_by_cluster(events: List[EconomicEvent], 
                          time_window_hours: int = 24) -> List[EventCluster]:
    """Group related economic events into clusters."""
    clusters = []
    processed_events = set()
    
    for event in events:
        if event.id in processed_events:
            continue
            
        # Find related events within time window
        related_events = []
        cluster_start = event.scheduled_time
        cluster_end = event.scheduled_time
        
        for other_event in events:
            if (other_event.id != event.id and 
                other_event.currency == event.currency and
                abs((other_event.scheduled_time - event.scheduled_time).total_seconds()) <= time_window_hours * 3600):
                
                related_events.append(other_event)
                processed_events.add(other_event.id)
                
                # Update cluster time bounds
                if other_event.scheduled_time < cluster_start:
                    cluster_start = other_event.scheduled_time
                if other_event.scheduled_time > cluster_end:
                    cluster_end = other_event.scheduled_time
        
        if related_events:
            # Create cluster
            all_events = [event] + related_events
            primary_events = [e.id for e in all_events if e.impact_level in [EconomicEventImpact.HIGH, EconomicEventImpact.CRITICAL]]
            secondary_events = [e.id for e in all_events if e.id not in primary_events]
            
            # Determine overall impact
            max_impact = max(e.impact_level for e in all_events)
            
            cluster = EventCluster(
                name=f"{event.currency.value} Economic Events",
                start_time=cluster_start,
                end_time=cluster_end,
                primary_events=primary_events,
                secondary_events=secondary_events,
                dominant_currency=event.currency,
                overall_impact=max_impact,
                theme=f"{event.category.value.replace('_', ' ').title()} Focus"
            )
            
            clusters.append(cluster)
        
        processed_events.add(event.id)
    
    return clusters