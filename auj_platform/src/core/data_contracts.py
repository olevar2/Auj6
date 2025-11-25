"""
Data contracts and structures for the AUJ Platform.

This module defines all core data structures using Pydantic for validation
and type safety throughout the platform.
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, field_validator, model_validator
import uuid


class TradeDirection(str, Enum):
    """Trade direction enumeration."""
    BUY = "BUY"
    SELL = "SELL"


class TradeStatus(str, Enum):
    """Trade status enumeration."""
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"
    ERROR = "ERROR"


class DealGrade(str, Enum):
    """Deal grade enumeration based on quality assessment."""
    A_PLUS = "A+"
    A = "A"
    A_MINUS = "A-"
    B_PLUS = "B+"
    B = "B"
    B_MINUS = "B-"
    C_PLUS = "C+"
    C = "C"
    C_MINUS = "C-"
    D = "D"
    F = "F"


class MarketRegime(str, Enum):
    """Market regime enumeration."""
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    SIDEWAYS = "SIDEWAYS"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    BREAKOUT = "BREAKOUT"
    REVERSAL = "REVERSAL"
    ACCUMULATION = "ACCUMULATION"
    DISTRIBUTION = "DISTRIBUTION"


class AgentRank(str, Enum):
    """Agent hierarchy rank enumeration."""
    ALPHA = "ALPHA"
    BETA = "BETA"
    GAMMA = "GAMMA"
    INACTIVE = "INACTIVE"


class ConfidenceLevel(str, Enum):
    """Confidence level enumeration."""
    VERY_HIGH = "VERY_HIGH"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    VERY_LOW = "VERY_LOW"


class RiskLevel(str, Enum):
    """Risk level enumeration."""
    CRITICAL = "CRITICAL"
    VERY_HIGH = "VERY_HIGH"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    VERY_LOW = "VERY_LOW"


class PositionSizeAdjustment(str, Enum):
    """Types of position size adjustments applied."""
    ACCOUNT_PROTECTION = "ACCOUNT_PROTECTION"
    PORTFOLIO_HEAT = "PORTFOLIO_HEAT"
    CORRELATION_REDUCTION = "CORRELATION_REDUCTION"
    VOLATILITY_SCALING = "VOLATILITY_SCALING"
    CONFIDENCE_SCALING = "CONFIDENCE_SCALING"
    REGIME_ADJUSTMENT = "REGIME_ADJUSTMENT"


class RiskMetrics(BaseModel):
    """Comprehensive risk metrics for a trade."""
    position_risk_percent: float
    portfolio_heat: float
    volatility_adjustment: float
    correlation_penalty: float
    confidence_multiplier: float
    final_position_size: Decimal
    max_loss_amount: Decimal
    risk_reward_ratio: float
    risk_level: RiskLevel
    adjustments_applied: List[PositionSizeAdjustment]
    warnings: List[str]


class OHLCVData(BaseModel):
    """OHLCV candlestick data structure."""
    timestamp: datetime
    open: Decimal = Field(..., gt=0)
    high: Decimal = Field(..., gt=0)
    low: Decimal = Field(..., gt=0)
    close: Decimal = Field(..., gt=0)
    volume: Decimal = Field(..., ge=0)

    @field_validator('high')
    @classmethod
    def high_must_be_highest(cls, v, info):
        if info.data and 'low' in info.data and v < info.data['low']:
            raise ValueError('High must be >= Low')
        return v


class TickData(BaseModel):
    """Tick data structure."""
    timestamp: datetime
    symbol: str
    bid: Decimal = Field(..., gt=0)
    ask: Decimal = Field(..., gt=0)
    volume: Decimal = Field(..., ge=0)

    @field_validator('ask')
    @classmethod
    def ask_must_be_greater_than_bid(cls, v, info):
        if info.data and 'bid' in info.data and v <= info.data['bid']:
            raise ValueError('Ask must be > Bid')
        return v


class MarketDataPoint(BaseModel):
    """Single market data point for analysis."""
    timestamp: datetime
    symbol: str
    price: Decimal = Field(..., gt=0)
    volume: Optional[Decimal] = Field(None, ge=0)
    ohlcv: Optional[OHLCVData] = None
    tick: Optional[TickData] = None
    indicators: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
class TradeSignal(BaseModel):
    """Core trade signal structure."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    symbol: str
    direction: TradeDirection
    confidence: float = Field(..., ge=0.0, le=1.0)
    confidence_level: ConfidenceLevel
    entry_price: Optional[Decimal] = Field(None, gt=0)
    stop_loss: Optional[Decimal] = Field(None, gt=0)
    take_profit: Optional[Decimal] = Field(None, gt=0)
    position_size: Optional[Decimal] = Field(None, gt=0)
    timeframe: str
    strategy: str
    generating_agent: str
    supporting_agents: List[str] = Field(default_factory=list)
    indicators_used: List[str] = Field(default_factory=list)
    market_regime: Optional[MarketRegime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentDecision(BaseModel):
    """Agent decision structure."""
    agent_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    symbol: str
    decision: str  # "BUY", "SELL", "HOLD", "NO_SIGNAL"
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str
    indicators_analyzed: List[str] = Field(default_factory=list)
    technical_analysis: Dict[str, Any] = Field(default_factory=dict)
    risk_assessment: Dict[str, Any] = Field(default_factory=dict)
    supporting_data: Dict[str, Any] = Field(default_factory=dict)


class GradedDeal(BaseModel):
    """Graded deal structure for quality assessment."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    original_signal: TradeSignal
    entry_time: datetime
    exit_time: Optional[datetime] = None
    entry_price: Decimal = Field(..., gt=0)
    exit_price: Optional[Decimal] = Field(None, gt=0)
    position_size: Decimal = Field(..., gt=0)
    pnl: Optional[Decimal] = None
    pnl_percentage: Optional[float] = None
    status: TradeStatus
    grade: Optional[DealGrade] = None
    grade_factors: Dict[str, float] = Field(default_factory=dict)
    execution_quality: Optional[float] = Field(None, ge=0.0, le=1.0)
    slippage: Optional[Decimal] = None
    duration_minutes: Optional[int] = None
    max_drawdown: Optional[Decimal] = None
    max_profit: Optional[Decimal] = None
    risk_adjusted_return: Optional[float] = None


class IndicatorResult(BaseModel):
    """Indicator calculation result."""
    indicator_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    symbol: str
    timeframe: str
    value: Union[float, Dict[str, float], List[float]]
    signal: Optional[str] = None  # "BUY", "SELL", "NEUTRAL"
    strength: Optional[float] = Field(None, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentPerformanceMetrics(BaseModel):
    """Agent performance tracking structure."""
    agent_name: str
    current_rank: AgentRank
    total_signals: int = 0
    successful_signals: int = 0
    win_rate: float = Field(0.0, ge=0.0, le=1.0)
    total_pnl: Decimal = Field(default=Decimal('0'))
    avg_pnl_per_trade: Decimal = Field(default=Decimal('0'))
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[Decimal] = None
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    performance_history: List[Dict[str, Any]] = Field(default_factory=list)


class MarketConditions(BaseModel):
    """Current market conditions assessment."""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    symbol: str
    regime: MarketRegime
    volatility: float = Field(..., ge=0.0)
    trend_strength: float = Field(..., ge=0.0, le=1.0)
    volume_profile: Dict[str, float] = Field(default_factory=dict)
    support_levels: List[Decimal] = Field(default_factory=list)
    resistance_levels: List[Decimal] = Field(default_factory=list)
    key_indicators: Dict[str, Any] = Field(default_factory=dict)
class PlatformStatus(BaseModel):
    """Overall platform status structure."""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = True
    active_agents: List[str] = Field(default_factory=list)
    alpha_agent: Optional[str] = None
    daily_pnl: Decimal = Field(default=Decimal('0'))
    total_equity: Decimal = Field(default=Decimal('0'))
    active_positions: int = 0
    win_rate: float = Field(0.0, ge=0.0, le=1.0)
    current_market_regime: Optional[MarketRegime] = None
    system_health: Dict[str, Any] = Field(default_factory=dict)


class ValidationResult(BaseModel):
    """Walk-forward validation result structure."""
    validation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    strategy_name: str
    indicator_set: List[str]
    in_sample_period: Dict[str, datetime]
    out_of_sample_period: Dict[str, datetime]
    in_sample_performance: Dict[str, float]
    out_of_sample_performance: Dict[str, float]
    overfitting_score: float = Field(..., ge=0.0, le=1.0)
    robustness_score: float = Field(..., ge=0.0, le=1.0)
    recommended_for_live: bool = False
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class OptimizationConfig(BaseModel):
    """Optimization configuration structure."""
    feature_flags: Dict[str, bool] = Field(default_factory=dict)
    selective_indicators: Dict[str, List[str]] = Field(default_factory=dict)
    risk_parameters: Dict[str, float] = Field(default_factory=dict)
    performance_thresholds: Dict[str, float] = Field(default_factory=dict)
    learning_rate: float = Field(0.01, ge=0.001, le=0.1)
    regularization_factor: float = Field(0.1, ge=0.0, le=1.0)


class AccountInfo(BaseModel):
    """Trading account information."""
    account_id: str
    broker: str
    account_type: str  # "DEMO", "LIVE"
    currency: str
    balance: Decimal = Field(..., ge=0)
    equity: Decimal = Field(..., ge=0)
    margin_available: Decimal = Field(..., ge=0)
    margin_used: Decimal = Field(default=Decimal('0'), ge=0)
    is_active: bool = True
    last_updated: datetime = Field(default_factory=datetime.utcnow)


class NewsEvent(BaseModel):
    """News event structure (for future use when news data becomes available)."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime
    headline: str
    content: str
    sentiment: Optional[float] = Field(None, ge=-1.0, le=1.0)
    impact_level: Optional[str] = None  # "HIGH", "MEDIUM", "LOW"
    affected_symbols: List[str] = Field(default_factory=list)
    source: str


# ============================================================================
# ECONOMIC CALENDAR DATA STRUCTURES
# ============================================================================

class EconomicEventImpact(str, Enum):
    """Economic event impact levels on financial markets."""
    CRITICAL = "CRITICAL"      # Market-moving events (NFP, FOMC, GDP)
    HIGH = "HIGH"              # Significant impact (CPI, Retail Sales)
    MEDIUM = "MEDIUM"          # Moderate impact (Housing data, Regional indices)
    LOW = "LOW"                # Minor impact (Preliminary readings)
    NEGLIGIBLE = "NEGLIGIBLE"  # Minimal market reaction expected


class EconomicEventCategory(str, Enum):
    """Categories of economic events."""
    MONETARY_POLICY = "MONETARY_POLICY"        # Central bank decisions, rates
    EMPLOYMENT = "EMPLOYMENT"                  # Jobs data, unemployment
    INFLATION = "INFLATION"                    # CPI, PPI, core inflation
    GDP_GROWTH = "GDP_GROWTH"                  # GDP, growth indicators
    MANUFACTURING = "MANUFACTURING"            # PMI, industrial production
    CONSUMER_SPENDING = "CONSUMER_SPENDING"    # Retail sales, consumer confidence
    HOUSING = "HOUSING"                        # Housing starts, sales
    TRADE_BALANCE = "TRADE_BALANCE"            # Import/export data
    GOVERNMENT_FISCAL = "GOVERNMENT_FISCAL"    # Budget, debt, spending
    BUSINESS_INVESTMENT = "BUSINESS_INVESTMENT" # Capex, business confidence
    FINANCIAL_STABILITY = "FINANCIAL_STABILITY" # Bank stress tests, stability reports
    GEOPOLITICAL = "GEOPOLITICAL"              # Elections, political events
    COMMODITY = "COMMODITY"                    # Oil inventories, agricultural data
    OTHER = "OTHER"                            # Miscellaneous economic data


class EconomicEventStatus(str, Enum):
    """Status of economic events."""
    SCHEDULED = "SCHEDULED"    # Event is scheduled for future
    RELEASED = "RELEASED"      # Data has been released
    REVISED = "REVISED"        # Data has been revised
    CANCELLED = "CANCELLED"    # Event was cancelled
    DELAYED = "DELAYED"        # Event release delayed


class EconomicIndicatorFrequency(str, Enum):
    """Frequency of economic indicator releases."""
    DAILY = "DAILY"
    WEEKLY = "WEEKLY"
    MONTHLY = "MONTHLY"
    QUARTERLY = "QUARTERLY"
    ANNUALLY = "ANNUALLY"
    IRREGULAR = "IRREGULAR"


class CurrencyCode(str, Enum):
    """Major currency codes affected by economic events."""
    USD = "USD"  # US Dollar
    EUR = "EUR"  # Euro
    GBP = "GBP"  # British Pound
    JPY = "JPY"  # Japanese Yen
    CHF = "CHF"  # Swiss Franc
    CAD = "CAD"  # Canadian Dollar
    AUD = "AUD"  # Australian Dollar
    NZD = "NZD"  # New Zealand Dollar
    CNY = "CNY"  # Chinese Yuan
    ALL = "ALL"  # Global impact


class EconomicEvent(BaseModel):
    """Comprehensive economic event data structure."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
    # Basic Event Information
    name: str = Field(..., description="Event name (e.g., 'Non-Farm Payrolls')")
    country: str = Field(..., description="Country code (e.g., 'US', 'EU', 'UK')")
    currency: CurrencyCode = Field(..., description="Primary currency affected")
    category: EconomicEventCategory = Field(..., description="Event category")
    
    # Timing Information
    scheduled_time: datetime = Field(..., description="Scheduled release time (UTC)")
    actual_release_time: Optional[datetime] = Field(None, description="Actual release time if different")
    frequency: EconomicIndicatorFrequency = Field(..., description="How often this indicator is released")
    
    # Impact Assessment
    impact_level: EconomicEventImpact = Field(..., description="Expected market impact")
    historical_volatility_minutes: int = Field(default=60, description="Historical volatility window in minutes")
    
    # Event Status and Data
    status: EconomicEventStatus = Field(default=EconomicEventStatus.SCHEDULED)
    forecast_value: Optional[str] = Field(None, description="Forecasted value")
    actual_value: Optional[str] = Field(None, description="Actual released value")
    previous_value: Optional[str] = Field(None, description="Previous period value")
    revised_value: Optional[str] = Field(None, description="Revised value if applicable")
    
    # Market Impact Analysis
    surprise_index: Optional[float] = Field(None, ge=-10.0, le=10.0, description="Surprise factor (actual vs forecast)")
    market_reaction_score: Optional[float] = Field(None, ge=-1.0, le=1.0, description="Market reaction magnitude")
    volatility_spike: Optional[float] = Field(None, ge=0.0, description="Volatility increase percentage")
    
    # Related Information
    affected_instruments: List[str] = Field(default_factory=list, description="Trading instruments affected")
    related_events: List[str] = Field(default_factory=list, description="IDs of related economic events")
    source: str = Field(default="ForexFactory", description="Data source")
    
    # Processing Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    is_processed: bool = Field(default=False, description="Whether event has been processed for trading signals")
    
    # Additional Context
    description: Optional[str] = Field(None, description="Detailed description of the economic indicator")
    importance_note: Optional[str] = Field(None, description="Why this event is important for markets")
    typical_market_reaction: Optional[str] = Field(None, description="Typical market reaction pattern")
    
    @field_validator('surprise_index')
    @classmethod
    def calculate_surprise_index(cls, v, info):
        """Calculate surprise index when actual and forecast values are available."""
        if v is not None:
            return v
        
        data = info.data
        if data and 'actual_value' in data and 'forecast_value' in data:
            actual = data.get('actual_value')
            forecast = data.get('forecast_value')
            
            if actual and forecast:
                try:
                    # Try to convert to float for calculation
                    actual_num = float(actual.replace('%', '').replace('K', '000').replace('M', '000000'))
                    forecast_num = float(forecast.replace('%', '').replace('K', '000').replace('M', '000000'))
                    
                    if forecast_num != 0:
                        return round(((actual_num - forecast_num) / abs(forecast_num)) * 100, 2)
                except (ValueError, AttributeError):
                    pass
        
        return v


class EconomicCalendar(BaseModel):
    """Economic calendar containing multiple events."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    date_range_start: datetime = Field(..., description="Calendar start date")
    date_range_end: datetime = Field(..., description="Calendar end date")
    events: List[EconomicEvent] = Field(default_factory=list)
    total_events: int = Field(default=0)
    high_impact_count: int = Field(default=0)
    medium_impact_count: int = Field(default=0)
    low_impact_count: int = Field(default=0)
    source: str = Field(default="ForexFactory")
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    
    @model_validator(mode='after')
    def calculate_event_counts(self):
        """Calculate impact level counts from events."""
        if self.events:
            self.total_events = len(self.events)
            self.high_impact_count = len([e for e in self.events if e.impact_level in [EconomicEventImpact.HIGH, EconomicEventImpact.CRITICAL]])
            self.medium_impact_count = len([e for e in self.events if e.impact_level == EconomicEventImpact.MEDIUM])
            self.low_impact_count = len([e for e in self.events if e.impact_level in [EconomicEventImpact.LOW, EconomicEventImpact.NEGLIGIBLE]])
        return self


class EconomicEventImpactAnalysis(BaseModel):
    """Analysis of economic event impact on trading instruments."""
    event_id: str = Field(..., description="Reference to economic event")
    analysis_timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Pre-event Analysis
    pre_event_volatility: Dict[str, float] = Field(default_factory=dict, description="Volatility before event by instrument")
    pre_event_trend: Dict[str, str] = Field(default_factory=dict, description="Trend before event by instrument") 
    pre_event_volume: Dict[str, float] = Field(default_factory=dict, description="Volume before event by instrument")
    
    # Post-event Analysis
    post_event_volatility: Dict[str, float] = Field(default_factory=dict, description="Volatility after event by instrument")
    price_movement_pips: Dict[str, float] = Field(default_factory=dict, description="Price movement in pips by instrument")
    volume_spike_ratio: Dict[str, float] = Field(default_factory=dict, description="Volume spike ratio by instrument")
    trend_reversal_probability: Dict[str, float] = Field(default_factory=dict, description="Trend reversal probability by instrument")
    
    # Impact Metrics
    overall_impact_score: float = Field(default=0.0, ge=0.0, le=10.0, description="Overall market impact score")
    primary_affected_instruments: List[str] = Field(default_factory=list)
    secondary_affected_instruments: List[str] = Field(default_factory=list)
    
    # Timing Analysis
    reaction_start_time: Optional[datetime] = Field(None, description="When market reaction started")
    peak_reaction_time: Optional[datetime] = Field(None, description="When reaction peaked")
    reaction_duration_minutes: Optional[int] = Field(None, description="How long the reaction lasted")
    
    # Pattern Recognition
    reaction_pattern: Optional[str] = Field(None, description="Type of market reaction pattern")
    historical_similarity_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Similarity to historical events")


class EconomicEventTradingSignal(BaseModel):
    """Trading signal generated from economic event analysis."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_id: str = Field(..., description="Reference to triggering economic event")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Signal Details
    instrument: str = Field(..., description="Trading instrument")
    signal_type: str = Field(..., description="BUY, SELL, or HOLD")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Signal confidence")
    strength: float = Field(..., ge=0.0, le=1.0, description="Signal strength")
    
    # Timing Information
    signal_validity_start: datetime = Field(..., description="When signal becomes valid")
    signal_validity_end: datetime = Field(..., description="When signal expires")
    recommended_entry_time: Optional[datetime] = Field(None, description="Recommended entry timing")
    
    # Risk Management
    suggested_stop_loss: Optional[float] = Field(None, description="Suggested stop loss level")
    suggested_take_profit: Optional[float] = Field(None, description="Suggested take profit level")
    position_size_recommendation: Optional[float] = Field(None, ge=0.0, le=1.0, description="Recommended position size")
    risk_level: str = Field(default="MEDIUM", description="LOW, MEDIUM, HIGH")
    
    # Supporting Analysis
    event_surprise_factor: Optional[float] = Field(None, description="How much event surprised market")
    historical_pattern_match: Optional[float] = Field(None, ge=0.0, le=1.0, description="Match with historical patterns")
    market_sentiment_alignment: Optional[float] = Field(None, ge=-1.0, le=1.0, description="Alignment with market sentiment")
    
    # Metadata
    generating_agent: str = Field(default="EconomicCalendarAgent")
    supporting_indicators: List[str] = Field(default_factory=list)
    signal_rationale: str = Field(default="", description="Explanation of why signal was generated")


class SystemConfig(BaseModel):
    """System configuration structure."""
    database_url: str
    log_level: str = "INFO"
    max_position_size: Decimal = Field(..., gt=0)
    max_daily_loss: Decimal = Field(..., gt=0)
    trading_hours: Dict[str, str] = Field(default_factory=dict)
    data_providers: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    api_settings: Dict[str, Any] = Field(default_factory=dict)
