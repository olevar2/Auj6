"""
Risk Management Database Schema for AUJ Platform.

This module defines the database tables for persisting risk management state,
including daily loss tracking and open position risk metrics.
"""

from sqlalchemy import (
    Table, Column, String, DateTime, Date, 
    DECIMAL, Integer, JSON, Boolean, Index
)
import uuid
from datetime import datetime

def add_risk_management_tables(metadata):
    """
    Add risk management related tables to the database metadata.
    
    Args:
        metadata: SQLAlchemy MetaData object to add tables to
    """
    
    # Daily Loss Tracking Table
    risk_daily_loss_table = Table(
        'risk_daily_loss', metadata,
        Column('id', String(36), primary_key=True, default=lambda: str(uuid.uuid4())),
        Column('date', Date, nullable=False, unique=True),
        Column('total_loss', DECIMAL(15, 2), default=0),
        Column('trade_count', Integer, default=0),
        Column('updated_at', DateTime, default=datetime.utcnow, onupdate=datetime.utcnow),
        
        # Metadata
        Column('metadata', JSON),
        
        # Indexes
        Index('idx_risk_daily_loss_date', 'date')
    )
    
    # Open Positions Risk Tracking Table
    risk_open_positions_table = Table(
        'risk_open_positions', metadata,
        Column('position_id', String(100), primary_key=True),
        Column('symbol', String(20), nullable=False),
        Column('entry_price', DECIMAL(15, 6)),
        Column('position_size', DECIMAL(15, 6)),
        Column('direction', String(10)),  # BUY/SELL
        
        # Risk Metrics
        Column('initial_equity', DECIMAL(15, 2)),
        Column('current_risk_percent', DECIMAL(5, 4)),
        Column('unrealized_pnl', DECIMAL(15, 2), default=0),
        Column('stop_loss', DECIMAL(15, 6)),
        Column('take_profit', DECIMAL(15, 6)),
        
        # Timing
        Column('opened_at', DateTime, default=datetime.utcnow),
        Column('last_updated', DateTime, default=datetime.utcnow, onupdate=datetime.utcnow),
        
        # Metadata
        Column('strategy_name', String(50)),
        Column('metadata', JSON),
        
        # Indexes
        Index('idx_risk_positions_symbol', 'symbol')
    )
    
    return {
        'risk_daily_loss': risk_daily_loss_table,
        'risk_open_positions': risk_open_positions_table
    }

# Export the function
__all__ = ['add_risk_management_tables']
