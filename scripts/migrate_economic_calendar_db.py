"""
Database Migration Script for Economic Calendar Tables.

This script creates all the necessary economic calendar tables and indexes
for the AUJ platform.
"""

import asyncio
import sys
import os
from datetime import datetime
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from auj_platform.src.core.database import DatabaseManager
from auj_platform.src.core.logging_setup import get_logger

logger = get_logger(__name__)


async def run_migration():
    """Run database migration to create economic calendar tables."""
    try:
        logger.info("Starting economic calendar database migration...")
        
        # Initialize database manager
        db_manager = DatabaseManager()
        await db_manager.initialize()
        
        logger.info("Database initialization completed successfully")
        
        # Check if tables were created
        async with db_manager.get_session() as session:
            # Test table existence by trying to query them
            try:
                if db_manager.is_sqlite:
                    # For SQLite, check table existence
                    result = session.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'economic_%'"
                    )
                else:
                    # For PostgreSQL, check table existence
                    result = await session.execute(
                        "SELECT table_name FROM information_schema.tables WHERE table_name LIKE 'economic_%'"
                    )
                
                tables = [row[0] for row in result.fetchall()]
                
                expected_tables = [
                    'economic_events',
                    'economic_event_impacts', 
                    'economic_trading_signals',
                    'economic_calendar_performance',
                    'economic_event_correlations',
                    'economic_news_sentiment'
                ]
                
                created_tables = [table for table in expected_tables if table in tables]
                missing_tables = [table for table in expected_tables if table not in tables]
                
                logger.info(f"Created economic calendar tables: {created_tables}")
                
                if missing_tables:
                    logger.warning(f"Missing tables: {missing_tables}")
                else:
                    logger.info("✅ All economic calendar tables created successfully!")
                
                # Create some sample data for testing
                await create_sample_data(db_manager)
                
            except Exception as e:
                logger.error(f"Error checking tables: {e}")
        
        # Close database connections
        await db_manager.close()
        
        logger.info("Migration completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return False


async def create_sample_data(db_manager: DatabaseManager):
    """Create sample economic calendar data for testing."""
    try:
        logger.info("Creating sample economic calendar data...")
        
        # Sample economic event
        sample_event = {
            'event_name': 'Non-Farm Payrolls',
            'event_time': datetime.utcnow(),
            'currency': 'USD',
            'country': 'US',
            'category': 'Employment',
            'importance': 'HIGH',
            'event_type': 'RELEASE',
            'actual_value': 200000.0,
            'forecast_value': 180000.0,
            'previous_value': 175000.0,
            'description': 'Monthly employment report showing job creation',
            'source': 'Bureau of Labor Statistics',
            'provider': 'ForexFactory',
            'provider_event_id': 'ff_nfp_202507',
            'metadata': {
                'created_by': 'migration_script',
                'sample_data': True
            }
        }
        
        event_id = await db_manager.save_economic_event(sample_event)
        logger.info(f"Created sample economic event: {event_id}")
        
        # Sample economic trading signal
        sample_signal = {
            'event_id': event_id,
            'signal_timestamp': datetime.utcnow(),
            'symbol': 'EURUSD',
            'signal_type': 'BUY',
            'confidence': 0.75,
            'strength': 0.8,
            'timeframe': '1H',
            'signal_validity_start': datetime.utcnow(),
            'signal_validity_end': datetime.utcnow(),
            'risk_level': 'MEDIUM',
            'position_size_recommendation': 0.02,
            'generating_agent': 'EconomicSessionExpert',
            'supporting_indicators': ['economic_event_impact_indicator'],
            'signal_rationale': 'Strong NFP beat suggests USD strength',
            'technical_factors': {'trend': 'bullish', 'support': 1.0850},
            'fundamental_factors': {'employment_strength': 'positive'},
            'metadata': {
                'created_by': 'migration_script',
                'sample_data': True
            }
        }
        
        signal_id = await db_manager.save_economic_trading_signal(sample_signal)
        logger.info(f"Created sample economic trading signal: {signal_id}")
        
        # Sample performance metrics
        sample_performance = {
            'date': datetime.utcnow(),
            'period_type': 'DAILY',
            'total_events_processed': 5,
            'high_impact_events': 2,
            'events_generating_signals': 3,
            'total_signals_generated': 8,
            'signals_executed': 6,
            'successful_signals': 4,
            'economic_signal_win_rate': 0.67,
            'economic_signal_pnl': 150.75,
            'data_provider_uptime': 0.98,
            'data_quality_score': 0.95,
            'currency_performance': {
                'USD': {'signals': 4, 'win_rate': 0.75, 'pnl': 120.50},
                'EUR': {'signals': 2, 'win_rate': 0.50, 'pnl': 30.25}
            },
            'category_performance': {
                'Employment': {'signals': 3, 'win_rate': 0.67, 'pnl': 90.00},
                'Inflation': {'signals': 2, 'win_rate': 0.50, 'pnl': 45.00}
            },
            'metadata': {
                'created_by': 'migration_script',
                'sample_data': True
            }
        }
        
        performance_id = await db_manager.save_economic_calendar_performance(sample_performance)
        logger.info(f"Created sample performance metrics: {performance_id}")
        
        logger.info("✅ Sample economic calendar data created successfully!")
        
    except Exception as e:
        logger.error(f"Error creating sample data: {e}")


def main():
    """Main migration entry point."""
    print("🚀 Starting AUJ Platform Economic Calendar Database Migration")
    print("=" * 60)
    
    try:
        success = asyncio.run(run_migration())
        
        if success:
            print("✅ Migration completed successfully!")
            print("\nCreated Tables:")
            print("- economic_events")
            print("- economic_event_impacts") 
            print("- economic_trading_signals")
            print("- economic_calendar_performance")
            print("- economic_event_correlations")
            print("- economic_news_sentiment")
            print("\n🎯 Your AUJ platform is now ready for economic calendar data!")
        else:
            print("❌ Migration failed! Check logs for details.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Migration cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Migration failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()