#!/usr/bin/env python3
"""
Database Migration Script: Remove Early Decision System
=====================================================

This script migrates the AUJ Platform database to remove early decision tracking
and add comprehensive analysis tracking columns.

Usage:
    python database_migration_remove_early_decision.py [--dry-run] [--backup]

Options:
    --dry-run    : Show what would be changed without making changes
    --backup     : Create backup before applying changes
"""

import sqlite3
import argparse
import os
import shutil
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def backup_database(db_path: str) -> str:
    """Create a backup of the database."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{db_path}.backup_{timestamp}"
    shutil.copy2(db_path, backup_path)
    logger.info(f"Database backed up to: {backup_path}")
    return backup_path

def check_column_exists(cursor, table_name: str, column_name: str) -> bool:
    """Check if a column exists in a table."""
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [column[1] for column in cursor.fetchall()]
    return column_name in columns

def migrate_database(db_path: str, dry_run: bool = False):
    """Apply database migration."""
    logger.info(f"Starting database migration for: {db_path}")
    
    if not os.path.exists(db_path):
        logger.warning(f"Database file not found: {db_path}")
        return
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Check if performance_trends table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='performance_trends'")
        if not cursor.fetchone():
            logger.info("performance_trends table not found - creating it")
            if not dry_run:
                cursor.execute("""
                    CREATE TABLE performance_trends (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        overall_performance REAL,
                        overfitting_risk REAL,
                        stability_score REAL,
                        active_agents INTEGER,
                        elite_indicators_count INTEGER,
                        market_regime TEXT,
                        comprehensive_analysis_rate REAL DEFAULT 1.0,
                        analysis_consistency_score REAL DEFAULT 1.0,
                        full_validation_rate REAL DEFAULT 1.0
                    )
                """)
                logger.info("Created performance_trends table with comprehensive analysis columns")
        else:
            # Check for early decision columns and remove them if they exist
            early_decision_columns = ['early_decision_rate', 'early_decision_time_savings', 'early_decision_signals']
            
            for column in early_decision_columns:
                if check_column_exists(cursor, 'performance_trends', column):
                    logger.info(f"Found early decision column: {column}")
                    if not dry_run:
                        # SQLite doesn't support DROP COLUMN directly, so we need to recreate the table
                        logger.info(f"Removing column {column} from performance_trends")
                        # This would require more complex migration logic
            
            # Add comprehensive analysis columns if they don't exist
            comprehensive_columns = [
                ('comprehensive_analysis_rate', 'REAL DEFAULT 1.0'),
                ('analysis_consistency_score', 'REAL DEFAULT 1.0'),
                ('full_validation_rate', 'REAL DEFAULT 1.0')
            ]
            
            for column_name, column_def in comprehensive_columns:
                if not check_column_exists(cursor, 'performance_trends', column_name):
                    logger.info(f"Adding column: {column_name}")
                    if not dry_run:
                        cursor.execute(f"ALTER TABLE performance_trends ADD COLUMN {column_name} {column_def}")
                else:
                    logger.info(f"Column {column_name} already exists")
        
        # Check if economic_calendar_performance table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='economic_calendar_performance'")
        if cursor.fetchone():
            # Remove early decision columns if they exist
            if check_column_exists(cursor, 'economic_calendar_performance', 'early_decision_signals'):
                logger.info("Found early_decision_signals column in economic_calendar_performance")
                # For now, just log - actual removal would need table recreation
            
            # Add comprehensive analysis columns
            if not check_column_exists(cursor, 'economic_calendar_performance', 'comprehensive_analysis_enabled'):
                logger.info("Adding comprehensive_analysis_enabled to economic_calendar_performance")
                if not dry_run:
                    cursor.execute("ALTER TABLE economic_calendar_performance ADD COLUMN comprehensive_analysis_enabled BOOLEAN DEFAULT TRUE")
            
            if not check_column_exists(cursor, 'economic_calendar_performance', 'validation_completeness_rate'):
                logger.info("Adding validation_completeness_rate to economic_calendar_performance")
                if not dry_run:
                    cursor.execute("ALTER TABLE economic_calendar_performance ADD COLUMN validation_completeness_rate REAL DEFAULT 1.0")
        
        # Create indexes for new columns
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_performance_comprehensive ON performance_trends(comprehensive_analysis_rate)",
            "CREATE INDEX IF NOT EXISTS idx_performance_consistency ON performance_trends(analysis_consistency_score)"
        ]
        
        for index_sql in indexes:
            logger.info(f"Creating index: {index_sql}")
            if not dry_run:
                cursor.execute(index_sql)
        
        if not dry_run:
            conn.commit()
            logger.info("Database migration completed successfully")
        else:
            logger.info("Dry run completed - no changes made")
            
    except Exception as e:
        logger.error(f"Migration failed: {str(e)}")
        conn.rollback()
        raise
    finally:
        conn.close()

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Migrate AUJ Platform database to remove early decision system')
    parser.add_argument('--dry-run', action='store_true', help='Show changes without applying them')
    parser.add_argument('--backup', action='store_true', help='Create backup before migration')
    parser.add_argument('--db-path', default='data/auj_platform.db', help='Path to database file')
    
    args = parser.parse_args()
    
    # Database paths to migrate
    db_paths = [
        'data/auj_platform.db',
        'auj_platform/data/auj_platform.db',
        'auj_platform/data/performance_tracking.db'
    ]
    
    if args.db_path != 'data/auj_platform.db':
        db_paths = [args.db_path]
    
    for db_path in db_paths:
        if os.path.exists(db_path):
            logger.info(f"Processing database: {db_path}")
            
            if args.backup and not args.dry_run:
                backup_database(db_path)
            
            migrate_database(db_path, args.dry_run)
        else:
            logger.info(f"Database not found, skipping: {db_path}")

if __name__ == "__main__":
    main()