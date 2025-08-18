# Economic Calendar Database Schema Documentation

## Overview

The AUJ Platform economic calendar database schema provides comprehensive storage and tracking for economic events, market impacts, trading signals, and performance metrics. This document describes the database tables, their relationships, and usage patterns.

## Database Tables

### 1. `economic_events`
**Primary storage for economic calendar events**

| Column | Type | Description |
|--------|------|-------------|
| `id` | String(36) | Primary key (UUID) |
| `event_name` | String(200) | Name of the economic event |
| `event_time` | DateTime | When the event occurs/occurred |
| `currency` | String(3) | Currency code (USD, EUR, etc.) |
| `country` | String(2) | Country code (US, GB, etc.) |
| `category` | String(50) | Event category (Employment, Inflation, GDP, etc.) |
| `importance` | String(10) | Impact level (LOW, MEDIUM, HIGH, CRITICAL) |
| `event_type` | String(20) | Type (RELEASE, SPEECH, DECISION) |
| `actual_value` | Decimal(20,6) | Actual reported value |
| `forecast_value` | Decimal(20,6) | Forecasted value |
| `previous_value` | Decimal(20,6) | Previous period value |
| `revised_value` | Decimal(20,6) | Revised previous value |
| `description` | Text | Event description |
| `source` | String(100) | Data source |
| `url` | String(500) | Reference URL |
| `provider` | String(50) | Data provider (ForexFactory, etc.) |
| `provider_event_id` | String(100) | Provider's event ID |
| `is_processed` | Boolean | Processing status |
| `metadata` | JSON | Additional metadata |

**Indexes:**
- `idx_economic_events_time` (event_time)
- `idx_economic_events_currency` (currency)
- `idx_economic_events_importance` (importance)
- `idx_economic_events_category` (category)

### 2. `economic_event_impacts`
**Historical market reactions to economic events**

| Column | Type | Description |
|--------|------|-------------|
| `id` | String(36) | Primary key (UUID) |
| `event_id` | String(36) | Foreign key to economic_events |
| `symbol` | String(20) | Trading symbol (EURUSD, etc.) |
| `impact_timestamp` | DateTime | When impact was measured |
| `price_before` | Decimal(15,6) | Price before event |
| `price_after_1min` | Decimal(15,6) | Price 1 minute after |
| `price_after_5min` | Decimal(15,6) | Price 5 minutes after |
| `price_after_15min` | Decimal(15,6) | Price 15 minutes after |
| `price_after_30min` | Decimal(15,6) | Price 30 minutes after |
| `price_after_1hour` | Decimal(15,6) | Price 1 hour after |
| `price_after_4hour` | Decimal(15,6) | Price 4 hours after |
| `price_after_1day` | Decimal(15,6) | Price 1 day after |
| `volatility_before` | Decimal(8,6) | Volatility before event |
| `volatility_after_*` | Decimal(8,6) | Volatility at various intervals |
| `max_price_move_pips` | Decimal(8,2) | Maximum price movement in pips |
| `max_price_move_percentage` | Decimal(8,4) | Maximum price movement percentage |
| `reaction_speed_seconds` | Integer | Time to first significant move |
| `reaction_direction` | String(10) | UP, DOWN, NEUTRAL |
| `reaction_strength` | Decimal(5,4) | Reaction strength (0.0-1.0) |
| `surprise_factor` | Decimal(8,4) | How much event surprised market |
| `metadata` | JSON | Additional analysis data |

### 3. `economic_trading_signals`
**Trading signals generated from economic events**

| Column | Type | Description |
|--------|------|-------------|
| `id` | String(36) | Primary key (UUID) |
| `event_id` | String(36) | Foreign key to economic_events |
| `impact_id` | String(36) | Foreign key to economic_event_impacts |
| `signal_timestamp` | DateTime | When signal was generated |
| `symbol` | String(20) | Trading symbol |
| `signal_type` | String(10) | BUY, SELL, HOLD |
| `confidence` | Decimal(5,4) | Signal confidence (0.0-1.0) |
| `strength` | Decimal(5,4) | Signal strength (0.0-1.0) |
| `timeframe` | String(10) | Signal timeframe (1M, 5M, 1H, etc.) |
| `signal_validity_start` | DateTime | When signal becomes valid |
| `signal_validity_end` | DateTime | When signal expires |
| `suggested_stop_loss` | Decimal(15,6) | Suggested stop loss level |
| `suggested_take_profit` | Decimal(15,6) | Suggested take profit level |
| `position_size_recommendation` | Decimal(5,4) | Recommended position size |
| `risk_level` | String(10) | LOW, MEDIUM, HIGH |
| `generating_agent` | String(50) | Agent that generated signal |
| `supporting_indicators` | JSON | List of supporting indicators |
| `signal_rationale` | Text | Explanation of signal logic |
| `technical_factors` | JSON | Technical analysis factors |
| `fundamental_factors` | JSON | Fundamental analysis factors |
| `was_executed` | Boolean | Whether signal was executed |
| `execution_price` | Decimal(15,6) | Actual execution price |
| `signal_performance_pnl` | Decimal(15,2) | Signal P&L |
| `signal_success` | Boolean | Whether signal was successful |

### 4. `economic_calendar_performance`
**Overall economic calendar system performance metrics**

| Column | Type | Description |
|--------|------|-------------|
| `id` | String(36) | Primary key (UUID) |
| `date` | DateTime | Performance date |
| `period_type` | String(10) | DAILY, WEEKLY, MONTHLY |
| `total_events_processed` | Integer | Number of events processed |
| `high_impact_events` | Integer | Number of high-impact events |
| `events_generating_signals` | Integer | Events that generated signals |
| `total_signals_generated` | Integer | Total signals created |
| `signals_executed` | Integer | Signals that were executed |
| `successful_signals` | Integer | Successful signals |
| `economic_signal_win_rate` | Decimal(5,4) | Win rate for economic signals |
| `economic_signal_pnl` | Decimal(15,2) | Total P&L from economic signals |
| `volatility_prediction_accuracy` | Decimal(5,4) | Volatility prediction accuracy |
| `forecast_accuracy_rate` | Decimal(5,4) | Economic forecast accuracy |
| `currency_performance` | JSON | Performance by currency |
| `category_performance` | JSON | Performance by event category |
| `agent_performance` | JSON | Performance by generating agent |
| `data_provider_uptime` | Decimal(5,4) | Data provider availability |
| `data_quality_score` | Decimal(5,4) | Overall data quality score |

### 5. `economic_event_correlations`
**Statistical correlations between events and market movements**

| Column | Type | Description |
|--------|------|-------------|
| `id` | String(36) | Primary key (UUID) |
| `event_category` | String(50) | Event category being analyzed |
| `symbol` | String(20) | Trading symbol |
| `analysis_period_start` | DateTime | Analysis start date |
| `analysis_period_end` | DateTime | Analysis end date |
| `total_events_analyzed` | Integer | Number of events in analysis |
| `correlation_coefficient` | Decimal(8,6) | Correlation coefficient (-1.0 to 1.0) |
| `correlation_strength` | String(10) | WEAK, MODERATE, STRONG |
| `statistical_significance` | Decimal(8,6) | P-value |
| `positive_reaction_percentage` | Decimal(5,4) | % of positive reactions |
| `negative_reaction_percentage` | Decimal(5,4) | % of negative reactions |
| `average_reaction_magnitude` | Decimal(8,4) | Average reaction size |
| `historical_profitability` | Decimal(8,4) | Historical profit potential |
| `sharpe_ratio` | Decimal(8,4) | Risk-adjusted return |

### 6. `economic_news_sentiment`
**News sentiment analysis related to economic events**

| Column | Type | Description |
|--------|------|-------------|
| `id` | String(36) | Primary key (UUID) |
| `event_id` | String(36) | Foreign key to economic_events |
| `news_timestamp` | DateTime | When news was published |
| `headline` | Text | News headline |
| `content` | Text | News content |
| `source` | String(100) | News source |
| `sentiment_score` | Decimal(8,4) | Sentiment score (-1.0 to 1.0) |
| `sentiment_label` | String(20) | POSITIVE, NEGATIVE, NEUTRAL |
| `sentiment_confidence` | Decimal(5,4) | Sentiment analysis confidence |
| `currencies_mentioned` | JSON | Currencies mentioned in news |
| `economic_indicators_mentioned` | JSON | Economic indicators mentioned |
| `market_impact_prediction` | String(10) | Predicted market impact |
| `key_phrases` | JSON | Important phrases extracted |
| `named_entities` | JSON | Named entities (people, orgs, etc.) |

## Database Relationships

```
economic_events (1) ──→ (many) economic_event_impacts
economic_events (1) ──→ (many) economic_trading_signals  
economic_events (1) ──→ (many) economic_news_sentiment
economic_event_impacts (1) ──→ (many) economic_trading_signals
```

## Usage Patterns

### 1. Storing Economic Events
```python
event_data = {
    'event_name': 'Non-Farm Payrolls',
    'event_time': datetime.utcnow(),
    'currency': 'USD',
    'importance': 'HIGH',
    # ... other fields
}
event_id = await db_manager.save_economic_event(event_data)
```

### 2. Recording Market Impact
```python
impact_data = {
    'event_id': event_id,
    'symbol': 'EURUSD',
    'price_before': 1.0850,
    'price_after_5min': 1.0865,
    'reaction_direction': 'UP',
    # ... other fields
}
await db_manager.save_economic_event_impact(impact_data)
```

### 3. Generating Trading Signals
```python
signal_data = {
    'event_id': event_id,
    'symbol': 'EURUSD',
    'signal_type': 'BUY',
    'confidence': 0.75,
    'generating_agent': 'EconomicSessionExpert',
    # ... other fields
}
signal_id = await db_manager.save_economic_trading_signal(signal_data)
```

### 4. Querying Economic Events
```python
events = await db_manager.get_economic_events(
    start_time=datetime.utcnow() - timedelta(days=7),
    currencies=['USD', 'EUR'],
    importance_levels=['HIGH', 'CRITICAL']
)
```

## Performance Considerations

### Indexes
- All time-based queries are optimized with datetime indexes
- Currency and importance filtering use dedicated indexes
- Composite indexes for common query patterns

### Data Retention
- Economic events: Permanent retention
- Event impacts: 2 years retention recommended
- Trading signals: 1 year retention recommended
- Performance metrics: Daily (2 years), Weekly (5 years), Monthly (permanent)

### Partitioning (PostgreSQL)
Consider partitioning large tables by date:
- `economic_events` by `event_time`
- `economic_event_impacts` by `impact_timestamp`
- `economic_trading_signals` by `signal_timestamp`

## Migration Commands

### Run Migration
```bash
# Windows
scripts\migrate_economic_calendar_db.bat

# Linux/macOS
./scripts/migrate_economic_calendar_db.sh

# Python direct
python scripts/migrate_economic_calendar_db.py
```

### Verify Tables
```python
from auj_platform.src.core.database import get_database

db = await get_database()
# Tables are automatically created during initialization
```

## Security Considerations

- All economic data should be encrypted at rest
- Limit access to performance tables containing trading results
- Implement audit logging for data modifications
- Use parameterized queries to prevent SQL injection

## Monitoring

Monitor these key metrics:
- Database table sizes and growth rates
- Query performance for time-range queries
- Data insertion rates during market hours
- Index usage statistics

This database schema provides the foundation for comprehensive economic calendar functionality in the AUJ platform.