#!/bin/bash
set -e

# Create database for AUJ Platform
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    -- Create extensions
    CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
    CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
    
    -- Create schemas
    CREATE SCHEMA IF NOT EXISTS auj_platform;
    CREATE SCHEMA IF NOT EXISTS analytics;
    CREATE SCHEMA IF NOT EXISTS monitoring;
    
    -- Grant permissions
    GRANT ALL PRIVILEGES ON SCHEMA auj_platform TO $POSTGRES_USER;
    GRANT ALL PRIVILEGES ON SCHEMA analytics TO $POSTGRES_USER;
    GRANT ALL PRIVILEGES ON SCHEMA monitoring TO $POSTGRES_USER;
    
    -- Create performance tracking table
    CREATE TABLE IF NOT EXISTS auj_platform.agent_performance (
        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
        agent_name VARCHAR(100) NOT NULL,
        timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        win_rate DECIMAL(5,4),
        total_trades INTEGER,
        profit_loss DECIMAL(15,2),
        drawdown DECIMAL(5,4),
        sharpe_ratio DECIMAL(8,4),
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
    );
    
    -- Create indices for performance
    CREATE INDEX IF NOT EXISTS idx_agent_performance_agent_name ON auj_platform.agent_performance(agent_name);
    CREATE INDEX IF NOT EXISTS idx_agent_performance_timestamp ON auj_platform.agent_performance(timestamp);
    
    -- Create monitoring table for system metrics
    CREATE TABLE IF NOT EXISTS monitoring.system_metrics (
        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
        metric_name VARCHAR(100) NOT NULL,
        metric_value DECIMAL(15,4),
        labels JSONB,
        timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
    );
    
    CREATE INDEX IF NOT EXISTS idx_system_metrics_name_time ON monitoring.system_metrics(metric_name, timestamp);
EOSQL

echo "Database initialization completed for AUJ Platform"