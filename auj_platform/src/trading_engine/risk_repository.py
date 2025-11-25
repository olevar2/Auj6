"""
Risk State Repository for AUJ Platform.

This module handles the persistence of risk management state, including
daily loss tracking and open positions, using the UnifiedDatabaseManager.
"""

import logging
from datetime import date, datetime
from decimal import Decimal
from typing import Dict, List, Optional, Any
from sqlalchemy import MetaData, select, delete, update, insert, text

from ..core.unified_database_manager import UnifiedDatabaseManager
from ..core.logging_setup import get_logger
from .risk_schema import add_risk_management_tables

logger = get_logger(__name__)

class RiskStateRepository:
    """
    Repository for persisting risk management state.
    """
    
    def __init__(self, db_manager: UnifiedDatabaseManager):
        self.db_manager = db_manager
        self.metadata = MetaData()
        self.tables = add_risk_management_tables(self.metadata)
        self._initialized = False
        
    async def initialize(self):
        """Initialize the repository and ensure schema exists."""
        if self._initialized:
            return
            
        try:
            # Ensure database manager is initialized
            if not self.db_manager.is_initialized:
                await self.db_manager.initialize()
                
            # Create tables if they don't exist
            # We use the sync engine for schema creation as it's a one-time setup
            # and create_all is synchronous
            if self.db_manager.sync_engine:
                self.metadata.create_all(bind=self.db_manager.sync_engine)
                logger.info("Risk management schema initialized")
            else:
                logger.warning("Sync engine not available for schema creation")
                
            self._initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize RiskStateRepository: {e}")
            raise
            
    async def get_daily_loss(self, day: date) -> float:
        """Get accumulated loss for a specific day."""
        try:
            query = select(self.tables['risk_daily_loss'].c.total_loss).where(
                self.tables['risk_daily_loss'].c.date == day
            )
            
            # Use raw SQL for simplicity with UnifiedDatabaseManager if needed, 
            # but here we try to use the engine directly or construct query string
            # Since UnifiedDatabaseManager takes string queries, we'll compile it.
            
            # However, UnifiedDatabaseManager.execute_async_query expects a string.
            # Let's compile the SQLAlchemy statement.
            compiled_query = str(query.compile(compile_kwargs={"literal_binds": True}))
            
            # But wait, literal_binds might not be safe or supported by all dialects perfectly for dates.
            # Let's use the session directly via db_manager to use SQLAlchemy constructs if possible.
            # UnifiedDatabaseManager exposes get_async_session.
            
            async with self.db_manager.get_async_session() as session:
                result = await session.execute(query)
                row = result.first()
                return float(row[0]) if row else 0.0
                
        except Exception as e:
            logger.error(f"Failed to get daily loss for {day}: {e}")
            return 0.0
            
    async def update_daily_loss(self, day: date, loss_amount: float):
        """Update (accumulate) daily loss."""
        try:
            # Check if record exists
            current_loss = await self.get_daily_loss(day)
            new_loss = current_loss + loss_amount
            
            async with self.db_manager.get_async_session() as session:
                # Upsert logic
                # For simplicity, we'll try update first, then insert if row count is 0
                # Or use a merge/upsert if supported.
                
                # Let's check existence first (we already did get_daily_loss)
                exists_query = select(self.tables['risk_daily_loss']).where(
                    self.tables['risk_daily_loss'].c.date == day
                )
                result = await session.execute(exists_query)
                exists = result.first() is not None
                
                if exists:
                    stmt = update(self.tables['risk_daily_loss']).where(
                        self.tables['risk_daily_loss'].c.date == day
                    ).values(
                        total_loss=new_loss,
                        updated_at=datetime.utcnow()
                    )
                else:
                    stmt = insert(self.tables['risk_daily_loss']).values(
                        date=day,
                        total_loss=loss_amount,
                        trade_count=1, # Initial count
                        updated_at=datetime.utcnow()
                    )
                
                await session.execute(stmt)
                await session.commit()
                
        except Exception as e:
            logger.error(f"Failed to update daily loss for {day}: {e}")
            raise

    async def reset_daily_loss(self, day: date):
        """Reset daily loss for a specific day (or delete old records)."""
        try:
            stmt = delete(self.tables['risk_daily_loss']).where(
                self.tables['risk_daily_loss'].c.date == day
            )
            async with self.db_manager.get_async_session() as session:
                await session.execute(stmt)
                await session.commit()
        except Exception as e:
            logger.error(f"Failed to reset daily loss for {day}: {e}")

    async def get_open_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get all open positions as a dictionary."""
        try:
            query = select(self.tables['risk_open_positions'])
            
            async with self.db_manager.get_async_session() as session:
                result = await session.execute(query)
                rows = result.fetchall()
                
                positions = {}
                for row in rows:
                    # Convert row to dict
                    # row._mapping is available in recent SQLAlchemy
                    pos_data = dict(row._mapping)
                    positions[pos_data['position_id']] = {
                        'symbol': pos_data['symbol'],
                        'unrealized_pnl': float(pos_data['unrealized_pnl'] or 0),
                        'initial_equity': float(pos_data['initial_equity'] or 0),
                        'current_risk_percent': float(pos_data['current_risk_percent'] or 0),
                        # Add other fields as needed by DynamicRiskManager
                    }
                return positions
                
        except Exception as e:
            logger.error(f"Failed to get open positions: {e}")
            return {}

    async def add_open_position(self, position_id: str, data: Dict[str, Any]):
        """Add or update an open position."""
        try:
            async with self.db_manager.get_async_session() as session:
                # Check existence
                exists_query = select(self.tables['risk_open_positions']).where(
                    self.tables['risk_open_positions'].c.position_id == position_id
                )
                result = await session.execute(exists_query)
                exists = result.first() is not None
                
                values = {
                    'position_id': position_id,
                    'symbol': data.get('symbol', 'UNKNOWN'),
                    'entry_price': data.get('entry_price'),
                    'position_size': data.get('size'),
                    'initial_equity': data.get('initial_equity'),
                    'current_risk_percent': data.get('current_risk_percent'),
                    'unrealized_pnl': data.get('unrealized_pnl', 0),
                    'last_updated': datetime.utcnow()
                }
                
                if exists:
                    stmt = update(self.tables['risk_open_positions']).where(
                        self.tables['risk_open_positions'].c.position_id == position_id
                    ).values(**values)
                else:
                    values['opened_at'] = datetime.utcnow()
                    stmt = insert(self.tables['risk_open_positions']).values(**values)
                
                await session.execute(stmt)
                await session.commit()
                
        except Exception as e:
            logger.error(f"Failed to save open position {position_id}: {e}")
            # Don't raise, just log, to avoid interrupting trading flow if DB is down?
            # Ideally we should raise, but Risk Manager might handle it.
            raise

    async def remove_open_position(self, position_id: str):
        """Remove an open position."""
        try:
            stmt = delete(self.tables['risk_open_positions']).where(
                self.tables['risk_open_positions'].c.position_id == position_id
            )
            async with self.db_manager.get_async_session() as session:
                await session.execute(stmt)
                await session.commit()
        except Exception as e:
            logger.error(f"Failed to remove open position {position_id}: {e}")
            raise
