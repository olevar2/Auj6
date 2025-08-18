"""
Economic Calendar Database Schema Extensions for AUJ Platform.

This module extends the existing database schema with economic calendar
specific tables for storing events, impacts, and performance metrics.
"""

from sqlalchemy import (
    Table, Column, String, DateTime, Text, JSON, Boolean, 
    DECIMAL, Integer, ForeignKey, Index, UniqueConstraint
)
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime


def add_economic_calendar_tables(metadata):
    """
    Add economic calendar related tables to the database metadata.
    
    Args:
        metadata: SQLAlchemy MetaData object to add tables to
    """
    
    # Economic Events table - stores all economic calendar events
    economic_events_table = Table(
        'economic_events', metadata,
        Column('id', String(36), primary_key=True, default=lambda: str(uuid.uuid4())),
        Column('event_name', String(200), nullable=False),
        Column('event_time', DateTime, nullable=False),
        Column('currency', String(3), nullable=False),
        Column('country', String(2), nullable=False),
        
        # Event Classification
        Column('category', String(50), nullable=False),  # Employment, Inflation, GDP, etc.
        Column('importance', String(10), nullable=False),  # LOW, MEDIUM, HIGH, CRITICAL
        Column('event_type', String(20), nullable=False),  # RELEASE, SPEECH, DECISION
        
        # Event Values
        Column('actual_value', DECIMAL(20, 6)),
        Column('forecast_value', DECIMAL(20, 6)),
        Column('previous_value', DECIMAL(20, 6)),
        Column('revised_value', DECIMAL(20, 6)),
        
        # Event Details
        Column('description', Text),
        Column('source', String(100)),  # ForexFactory, Investing.com, etc.
        Column('url', String(500)),
        Column('time_until_event', Integer),  # Minutes until event
        
        # Data Provider Information
        Column('provider', String(50), nullable=False),
        Column('provider_event_id', String(100)),
        Column('created_at', DateTime, default=datetime.utcnow),
        Column('updated_at', DateTime, default=datetime.utcnow, onupdate=datetime.utcnow),
        
        # Event Status
        Column('is_processed', Boolean, default=False),
        Column('is_active', Boolean, default=True),
        Column('processing_notes', Text),
        
        # Metadata
        Column('metadata', JSON),
        
        # Indexes for performance
        Index('idx_economic_events_time', 'event_time'),
        Index('idx_economic_events_currency', 'currency'),
        Index('idx_economic_events_importance', 'importance'),
        Index('idx_economic_events_category', 'category'),
        Index('idx_economic_events_provider', 'provider', 'provider_event_id'),
        
        # Unique constraint to prevent duplicates
        UniqueConstraint('provider', 'provider_event_id', 'event_time', 
                        name='uq_economic_events_provider_id_time')
    )
    
    # Economic Event Impacts table - stores historical market reactions
    economic_event_impacts_table = Table(
        'economic_event_impacts', metadata,
        Column('id', String(36), primary_key=True, default=lambda: str(uuid.uuid4())),
        Column('event_id', String(36), ForeignKey('economic_events.id'), nullable=False),
        Column('symbol', String(20), nullable=False),
        Column('impact_timestamp', DateTime, nullable=False),
        
        # Price Impact
        Column('price_before', DECIMAL(15, 6), nullable=False),
        Column('price_after_1min', DECIMAL(15, 6)),
        Column('price_after_5min', DECIMAL(15, 6)),
        Column('price_after_15min', DECIMAL(15, 6)),
        Column('price_after_30min', DECIMAL(15, 6)),
        Column('price_after_1hour', DECIMAL(15, 6)),
        Column('price_after_4hour', DECIMAL(15, 6)),
        Column('price_after_1day', DECIMAL(15, 6)),
        
        # Volatility Impact
        Column('volatility_before', DECIMAL(8, 6)),
        Column('volatility_after_1min', DECIMAL(8, 6)),
        Column('volatility_after_5min', DECIMAL(8, 6)),
        Column('volatility_after_15min', DECIMAL(8, 6)),
        Column('volatility_after_30min', DECIMAL(8, 6)),
        Column('volatility_after_1hour', DECIMAL(8, 6)),
        Column('volatility_after_4hour', DECIMAL(8, 6)),
        Column('volatility_after_1day', DECIMAL(8, 6)),
        
        # Volume Impact
        Column('volume_before', DECIMAL(20, 2)),
        Column('volume_after_1min', DECIMAL(20, 2)),
        Column('volume_after_5min', DECIMAL(20, 2)),
        Column('volume_after_15min', DECIMAL(20, 2)),
        Column('volume_after_30min', DECIMAL(20, 2)),
        Column('volume_after_1hour', DECIMAL(20, 2)),
        
        # Impact Metrics
        Column('max_price_move_pips', DECIMAL(8, 2)),
        Column('max_price_move_percentage', DECIMAL(8, 4)),
        Column('reaction_speed_seconds', Integer),  # Time to first significant move
        Column('reaction_direction', String(10)),  # UP, DOWN, NEUTRAL
        Column('reaction_strength', DECIMAL(5, 4)),  # 0.0 to 1.0
        Column('reaction_duration_minutes', Integer),
        
        # Pattern Recognition
        Column('reaction_pattern', String(50)),  # SPIKE, TREND, REVERSAL, etc.
        Column('pattern_confidence', DECIMAL(5, 4)),
        Column('historical_similarity_score', DECIMAL(5, 4)),
        
        # Analysis Results
        Column('surprise_factor', DECIMAL(8, 4)),  # How much event surprised market
        Column('correlation_with_forecast', DECIMAL(8, 4)),
        Column('market_sentiment_alignment', DECIMAL(8, 4)),
        
        # Metadata
        Column('analysis_timestamp', DateTime, default=datetime.utcnow),
        Column('analyst_notes', Text),
        Column('metadata', JSON),
        
        # Indexes
        Index('idx_economic_impacts_event', 'event_id'),
        Index('idx_economic_impacts_symbol', 'symbol'),
        Index('idx_economic_impacts_timestamp', 'impact_timestamp'),
        Index('idx_economic_impacts_symbol_time', 'symbol', 'impact_timestamp')
    )
    
    # Economic Trading Signals table - signals generated from economic events
    economic_trading_signals_table = Table(
        'economic_trading_signals', metadata,
        Column('id', String(36), primary_key=True, default=lambda: str(uuid.uuid4())),
        Column('event_id', String(36), ForeignKey('economic_events.id'), nullable=False),
        Column('impact_id', String(36), ForeignKey('economic_event_impacts.id')),
        Column('signal_timestamp', DateTime, nullable=False),
        
        # Signal Details
        Column('symbol', String(20), nullable=False),
        Column('signal_type', String(10), nullable=False),  # BUY, SELL, HOLD
        Column('confidence', DECIMAL(5, 4), nullable=False),
        Column('strength', DECIMAL(5, 4), nullable=False),
        Column('timeframe', String(10), nullable=False),
        
        # Timing Information
        Column('signal_validity_start', DateTime, nullable=False),
        Column('signal_validity_end', DateTime, nullable=False),
        Column('recommended_entry_time', DateTime),
        Column('time_to_event_minutes', Integer),
        
        # Risk Management
        Column('suggested_stop_loss', DECIMAL(15, 6)),
        Column('suggested_take_profit', DECIMAL(15, 6)),
        Column('position_size_recommendation', DECIMAL(5, 4)),
        Column('risk_level', String(10), nullable=False),  # LOW, MEDIUM, HIGH
        Column('max_risk_percentage', DECIMAL(5, 4)),
        
        # Signal Analysis
        Column('event_surprise_factor', DECIMAL(8, 4)),
        Column('historical_pattern_match', DECIMAL(5, 4)),
        Column('market_sentiment_alignment', DECIMAL(8, 4)),
        Column('volatility_expectation', String(10)),  # LOW, MEDIUM, HIGH
        
        # Signal Generation
        Column('generating_agent', String(50), nullable=False),
        Column('supporting_indicators', JSON),
        Column('signal_rationale', Text),
        Column('technical_factors', JSON),
        Column('fundamental_factors', JSON),
        
        # Signal Performance (filled after execution)
        Column('was_executed', Boolean, default=False),
        Column('execution_price', DECIMAL(15, 6)),
        Column('execution_timestamp', DateTime),
        Column('signal_performance_pnl', DECIMAL(15, 2)),
        Column('signal_performance_percentage', DECIMAL(8, 4)),
        Column('signal_duration_minutes', Integer),
        Column('signal_success', Boolean),
        Column('performance_notes', Text),
        
        # Metadata
        Column('created_at', DateTime, default=datetime.utcnow),
        Column('metadata', JSON),
        
        # Indexes
        Index('idx_economic_signals_event', 'event_id'),
        Index('idx_economic_signals_symbol', 'symbol'),
        Index('idx_economic_signals_timestamp', 'signal_timestamp'),
        Index('idx_economic_signals_type', 'signal_type'),
        Index('idx_economic_signals_agent', 'generating_agent'),
        Index('idx_economic_signals_performance', 'was_executed', 'signal_success')
    )
    
    # Economic Calendar Performance table - tracks overall economic calendar system performance
    economic_calendar_performance_table = Table(
        'economic_calendar_performance', metadata,
        Column('id', String(36), primary_key=True, default=lambda: str(uuid.uuid4())),
        Column('date', DateTime, nullable=False),
        Column('period_type', String(10), nullable=False),  # DAILY, WEEKLY, MONTHLY
        
        # Event Statistics
        Column('total_events_processed', Integer, default=0),
        Column('high_impact_events', Integer, default=0),
        Column('events_with_surprise', Integer, default=0),
        Column('events_generating_signals', Integer, default=0),
        
        # Signal Performance
        Column('total_signals_generated', Integer, default=0),
        Column('signals_executed', Integer, default=0),
        Column('successful_signals', Integer, default=0),
        Column('economic_signal_win_rate', DECIMAL(5, 4), default=0),
        Column('economic_signal_pnl', DECIMAL(15, 2), default=0),
        Column('average_signal_duration_minutes', DECIMAL(8, 2)),
        
        # Risk Metrics
        Column('total_risk_adjusted', DECIMAL(15, 2), default=0),
        Column('max_single_signal_loss', DECIMAL(15, 2)),
        Column('max_single_signal_profit', DECIMAL(15, 2)),
        Column('volatility_prediction_accuracy', DECIMAL(5, 4)),
        
        # Economic Analysis Performance
        Column('forecast_accuracy_rate', DECIMAL(5, 4)),
        Column('surprise_prediction_rate', DECIMAL(5, 4)),
        Column('impact_prediction_accuracy', DECIMAL(5, 4)),
        Column('sentiment_analysis_accuracy', DECIMAL(5, 4)),
        
        # Currency Performance
        Column('currency_performance', JSON),  # Performance by currency
        Column('category_performance', JSON),  # Performance by economic category
        Column('agent_performance', JSON),     # Performance by generating agent
        
        # System Health
        Column('data_provider_uptime', DECIMAL(5, 4)),
        Column('data_latency_avg_seconds', DECIMAL(8, 2)),
        Column('processing_errors', Integer, default=0),
        Column('data_quality_score', DECIMAL(5, 4)),
        
        # Metadata
        Column('calculated_at', DateTime, default=datetime.utcnow),
        Column('notes', Text),
        Column('metadata', JSON),
        
        # Indexes
        Index('idx_economic_performance_date', 'date'),
        Index('idx_economic_performance_period', 'period_type', 'date'),
        
        # Unique constraint for period
        UniqueConstraint('date', 'period_type', name='uq_economic_performance_date_period')
    )
    
    # Economic Event Correlations table - tracks correlations between events and market movements
    economic_event_correlations_table = Table(
        'economic_event_correlations', metadata,
        Column('id', String(36), primary_key=True, default=lambda: str(uuid.uuid4())),
        Column('event_category', String(50), nullable=False),
        Column('symbol', String(20), nullable=False),
        Column('analysis_period_start', DateTime, nullable=False),
        Column('analysis_period_end', DateTime, nullable=False),
        
        # Correlation Statistics
        Column('total_events_analyzed', Integer, nullable=False),
        Column('correlation_coefficient', DECIMAL(8, 6)),  # -1.0 to 1.0
        Column('correlation_strength', String(10)),  # WEAK, MODERATE, STRONG
        Column('correlation_confidence', DECIMAL(5, 4)),
        Column('statistical_significance', DECIMAL(8, 6)),  # p-value
        
        # Directional Analysis
        Column('positive_reaction_percentage', DECIMAL(5, 4)),
        Column('negative_reaction_percentage', DECIMAL(5, 4)),
        Column('neutral_reaction_percentage', DECIMAL(5, 4)),
        Column('average_reaction_magnitude', DECIMAL(8, 4)),
        Column('reaction_consistency_score', DECIMAL(5, 4)),
        
        # Timing Analysis
        Column('average_reaction_delay_seconds', DECIMAL(8, 2)),
        Column('reaction_duration_avg_minutes', DECIMAL(8, 2)),
        Column('peak_impact_time_minutes', DECIMAL(8, 2)),
        
        # Performance Metrics
        Column('trading_opportunity_rate', DECIMAL(5, 4)),
        Column('historical_profitability', DECIMAL(8, 4)),
        Column('risk_adjusted_return', DECIMAL(8, 4)),
        Column('sharpe_ratio', DECIMAL(8, 4)),
        
        # Analysis Metadata
        Column('analysis_method', String(50)),
        Column('sample_size', Integer),
        Column('analysis_quality_score', DECIMAL(5, 4)),
        Column('last_updated', DateTime, default=datetime.utcnow),
        Column('notes', Text),
        Column('metadata', JSON),
        
        # Indexes
        Index('idx_economic_correlations_category', 'event_category'),
        Index('idx_economic_correlations_symbol', 'symbol'),
        Index('idx_economic_correlations_period', 'analysis_period_start', 'analysis_period_end'),
        
        # Unique constraint
        UniqueConstraint('event_category', 'symbol', 'analysis_period_start', 
                        name='uq_economic_correlations_category_symbol_period')
    )
    
    # Economic News Sentiment table - stores news sentiment analysis
    economic_news_sentiment_table = Table(
        'economic_news_sentiment', metadata,
        Column('id', String(36), primary_key=True, default=lambda: str(uuid.uuid4())),
        Column('event_id', String(36), ForeignKey('economic_events.id')),
        Column('news_timestamp', DateTime, nullable=False),
        
        # News Details
        Column('headline', Text, nullable=False),
        Column('content', Text),
        Column('source', String(100)),
        Column('author', String(100)),
        Column('url', String(500)),
        
        # Sentiment Analysis
        Column('sentiment_score', DECIMAL(8, 4)),  # -1.0 to 1.0
        Column('sentiment_label', String(20)),  # POSITIVE, NEGATIVE, NEUTRAL
        Column('sentiment_confidence', DECIMAL(5, 4)),
        Column('emotion_scores', JSON),  # Fear, greed, optimism, etc.
        
        # Currency/Market Impact
        Column('currencies_mentioned', JSON),
        Column('economic_indicators_mentioned', JSON),
        Column('central_banks_mentioned', JSON),
        Column('market_impact_prediction', String(10)),  # HIGH, MEDIUM, LOW
        
        # NLP Analysis
        Column('key_phrases', JSON),
        Column('named_entities', JSON),
        Column('topic_classification', JSON),
        Column('language', String(5), default='en'),
        
        # Sentiment Performance
        Column('sentiment_accuracy', DECIMAL(5, 4)),
        Column('prediction_verified', Boolean),
        Column('actual_market_reaction', String(20)),
        
        # Metadata
        Column('analysis_model', String(50)),
        Column('processed_at', DateTime, default=datetime.utcnow),
        Column('metadata', JSON),
        
        # Indexes
        Index('idx_economic_news_timestamp', 'news_timestamp'),
        Index('idx_economic_news_sentiment', 'sentiment_score'),
        Index('idx_economic_news_event', 'event_id'),
        Index('idx_economic_news_source', 'source')
    )
    
    return {
        'economic_events': economic_events_table,
        'economic_event_impacts': economic_event_impacts_table,
        'economic_trading_signals': economic_trading_signals_table,
        'economic_calendar_performance': economic_calendar_performance_table,
        'economic_event_correlations': economic_event_correlations_table,
        'economic_news_sentiment': economic_news_sentiment_table
    }


# Export the function
__all__ = ['add_economic_calendar_tables']