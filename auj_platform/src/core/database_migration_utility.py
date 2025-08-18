"""
Database Migration Utility for AUJ Platform
==========================================

This utility helps migrate existing database code from the old mixed 
sync/async patterns to the new UnifiedDatabaseManager.

Features:
- Automatic detection of database usage patterns
- Code migration with minimal changes
- Compatibility layer for legacy code
- Performance improvement tracking

Author: AUJ Platform Development Team  
Date: 2025-07-04
Version: 1.0.0
"""

import re
import ast
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging

from ..core.logging_setup import get_logger

logger = get_logger(__name__)


class DatabaseCodeAnalyzer:
    """Analyzes existing code for database usage patterns."""
    
    def __init__(self):
        self.sqlite3_patterns = [
            r'import sqlite3',
            r'sqlite3\.connect\(',
            r'conn\.execute\(',
            r'cursor\.execute\(',
            r'conn\.cursor\(\)',
            r'conn\.commit\(\)',
            r'conn\.rollback\(\)',
            r'conn\.close\(\)'
        ]
        
        self.session_patterns = [
            r'session\(\)',
            r'sessionmaker\(',
            r'get_session\(\)',
            r'session\.execute\(',
            r'session\.commit\(\)',
            r'session\.rollback\(\)',
            r'session\.close\(\)'
        ]
        
        self.asyncpg_patterns = [
            r'import asyncpg',
            r'asyncpg\.connect\(',
            r'await.*\.execute\(',
            r'await.*\.fetch\(',
            r'async with.*pool\.',
            r'await.*\.close\(\)'
        ]
    
    def analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a single file for database usage patterns."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            analysis = {
                'file_path': str(file_path),
                'has_database_code': False,
                'sqlite3_usage': [],
                'session_usage': [], 
                'asyncpg_usage': [],
                'complexity_score': 0,
                'migration_priority': 'LOW',
                'recommendations': []
            }
            
            # Check for SQLite3 patterns
            for pattern in self.sqlite3_patterns:
                matches = re.findall(pattern, content, re.MULTILINE)
                if matches:
                    analysis['sqlite3_usage'].extend(matches)
                    analysis['has_database_code'] = True
            
            # Check for session patterns
            for pattern in self.session_patterns:
                matches = re.findall(pattern, content, re.MULTILINE)
                if matches:
                    analysis['session_usage'].extend(matches)
                    analysis['has_database_code'] = True
            
            # Check for asyncpg patterns
            for pattern in self.asyncpg_patterns:
                matches = re.findall(pattern, content, re.MULTILINE)
                if matches:
                    analysis['asyncpg_usage'].extend(matches)
                    analysis['has_database_code'] = True
            
            # Calculate complexity and priority
            if analysis['has_database_code']:
                analysis['complexity_score'] = (
                    len(analysis['sqlite3_usage']) +
                    len(analysis['session_usage']) +
                    len(analysis['asyncpg_usage'])
                )
                
                if analysis['complexity_score'] > 20:
                    analysis['migration_priority'] = 'HIGH'
                elif analysis['complexity_score'] > 10:
                    analysis['migration_priority'] = 'MEDIUM'
                
                # Generate recommendations
                analysis['recommendations'] = self._generate_recommendations(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze file {file_path}: {e}")
            return {
                'file_path': str(file_path),
                'error': str(e),
                'has_database_code': False
            }
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate migration recommendations based on analysis."""
        recommendations = []
        
        if analysis['sqlite3_usage']:
            recommendations.append(
                "Replace direct SQLite3 usage with UnifiedDatabaseManager.execute_sync_query()"
            )
            recommendations.append(
                "Use UnifiedDatabaseManager.get_sync_session() for transaction management"
            )
        
        if analysis['session_usage'] and analysis['sqlite3_usage']:
            recommendations.append(
                "Consolidate mixed database access patterns - use only UnifiedDatabaseManager"
            )
        
        if analysis['asyncpg_usage']:
            recommendations.append(
                "Replace direct asyncpg usage with UnifiedDatabaseManager.execute_async_query()"
            )
            recommendations.append(
                "Use UnifiedDatabaseManager.get_async_session() for async operations"
            )
        
        if analysis['complexity_score'] > 15:
            recommendations.append(
                "Consider refactoring complex database logic into service classes"
            )
            recommendations.append(
                "Implement database operation caching for frequently executed queries"
            )
        
        return recommendations


class DatabaseMigrator:
    """Migrates existing database code to use UnifiedDatabaseManager."""
    
    def __init__(self):
        self.analyzer = DatabaseCodeAnalyzer()
        self.migration_stats = {
            'files_analyzed': 0,
            'files_migrated': 0,
            'patterns_replaced': 0,
            'errors': []
        }
    
    def migrate_file(self, file_path: Path, backup: bool = True) -> Dict[str, Any]:
        """Migrate a single file to use UnifiedDatabaseManager."""
        try:
            # Analyze first
            analysis = self.analyzer.analyze_file(file_path)
            
            if not analysis.get('has_database_code', False):
                return {
                    'file_path': str(file_path),
                    'migrated': False,
                    'reason': 'No database code found'
                }
            
            # Read original content
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            # Create backup if requested
            if backup:
                backup_path = file_path.with_suffix(f'{file_path.suffix}.backup')
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(original_content)
            
            # Perform migration
            migrated_content = self._migrate_content(original_content, analysis)
            
            # Write migrated content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(migrated_content)
            
            self.migration_stats['files_migrated'] += 1
            
            return {
                'file_path': str(file_path),
                'migrated': True,
                'backup_created': backup,
                'patterns_replaced': self._count_replacements(original_content, migrated_content),
                'recommendations': analysis.get('recommendations', [])
            }
            
        except Exception as e:
            error_msg = f"Failed to migrate file {file_path}: {e}"
            logger.error(error_msg)
            self.migration_stats['errors'].append(error_msg)
            
            return {
                'file_path': str(file_path),
                'migrated': False,
                'error': str(e)
            }
    
    def _migrate_content(self, content: str, analysis: Dict[str, Any]) -> str:
        """Migrate content to use UnifiedDatabaseManager."""
        migrated = content
        
        # Add import if needed
        if not 'from ..core.unified_database_manager import' in migrated:
            import_line = "from ..core.unified_database_manager import get_unified_database, get_unified_database_sync\n"
            
            # Find the last import line to insert after
            lines = migrated.split('\n')
            last_import_index = 0
            for i, line in enumerate(lines):
                if line.strip().startswith(('import ', 'from ')) and not line.strip().startswith('#'):
                    last_import_index = i
            
            lines.insert(last_import_index + 1, import_line)
            migrated = '\n'.join(lines)
        
        # Replace SQLite3 patterns
        if analysis.get('sqlite3_usage'):
            migrated = self._replace_sqlite3_patterns(migrated)
        
        # Replace session patterns
        if analysis.get('session_usage'):
            migrated = self._replace_session_patterns(migrated)
        
        # Replace asyncpg patterns
        if analysis.get('asyncpg_usage'):
            migrated = self._replace_asyncpg_patterns(migrated)
        
        return migrated
    
    def _replace_sqlite3_patterns(self, content: str) -> str:
        """Replace SQLite3 usage patterns."""
        # Replace sqlite3.connect with unified database manager
        content = re.sub(
            r'sqlite3\.connect\([^)]+\)',
            'get_unified_database_sync().get_sync_session()',
            content
        )
        
        # Replace conn.execute with session.execute
        content = re.sub(
            r'conn\.execute\(([^)]+)\)',
            r'session.execute(text(\1))',
            content
        )
        
        # Replace cursor.execute with session.execute
        content = re.sub(
            r'cursor\.execute\(([^)]+)\)',
            r'session.execute(text(\1))',
            content
        )
        
        # Replace conn.commit() with context manager handling
        content = re.sub(
            r'conn\.commit\(\)',
            '# Transaction handled by context manager',
            content
        )
        
        # Replace conn.close() with context manager handling  
        content = re.sub(
            r'conn\.close\(\)',
            '# Connection closed by context manager',
            content
        )
        
        return content
    
    def _replace_session_patterns(self, content: str) -> str:
        """Replace session usage patterns."""
        # Replace get_session() context manager usage
        content = re.sub(
            r'async with.*get_session\(\) as session:',
            'async with get_unified_database().get_async_session() as session:',
            content
        )
        
        content = re.sub(
            r'with.*get_session\(\) as session:',
            'with get_unified_database_sync().get_sync_session() as session:',
            content
        )
        
        return content
    
    def _replace_asyncpg_patterns(self, content: str) -> str:
        """Replace asyncpg usage patterns."""
        # Replace asyncpg.connect with unified database manager
        content = re.sub(
            r'asyncpg\.connect\([^)]+\)',
            'get_unified_database().get_async_session()',
            content
        )
        
        # Replace direct execute calls with query method
        content = re.sub(
            r'await\s+(\w+)\.execute\(([^)]+)\)',
            r'await get_unified_database().execute_async_query(\2)',
            content
        )
        
        return content
    
    def _count_replacements(self, original: str, migrated: str) -> int:
        """Count number of replacements made."""
        # Simple diff-based counting
        original_lines = original.split('\n')
        migrated_lines = migrated.split('\n')
        
        changes = 0
        for orig, migr in zip(original_lines, migrated_lines):
            if orig != migr:
                changes += 1
        
        return changes + abs(len(original_lines) - len(migrated_lines))


class DatabaseMigrationOrchestrator:
    """Orchestrates the complete database migration process."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.analyzer = DatabaseCodeAnalyzer()
        self.migrator = DatabaseMigrator()
        
    def analyze_project(self) -> Dict[str, Any]:
        """Analyze entire project for database usage."""
        logger.info("Starting project-wide database analysis...")
        
        results = {
            'total_files': 0,
            'files_with_db_code': 0,
            'high_priority': [],
            'medium_priority': [],
            'low_priority': [],
            'migration_estimate': {
                'total_effort_hours': 0,
                'complexity_breakdown': {},
                'recommendations': []
            }
        }
        
        # Find all Python files
        python_files = list(self.project_root.rglob('*.py'))
        results['total_files'] = len(python_files)
        
        for file_path in python_files:
            try:
                analysis = self.analyzer.analyze_file(file_path)
                
                if analysis.get('has_database_code', False):
                    results['files_with_db_code'] += 1
                    
                    priority = analysis.get('migration_priority', 'LOW')
                    if priority == 'HIGH':
                        results['high_priority'].append(analysis)
                    elif priority == 'MEDIUM':
                        results['medium_priority'].append(analysis)
                    else:
                        results['low_priority'].append(analysis)
                        
            except Exception as e:
                logger.warning(f"Failed to analyze {file_path}: {e}")
        
        # Calculate migration estimates
        results['migration_estimate'] = self._calculate_migration_estimate(results)
        
        logger.info(f"Analysis complete: {results['files_with_db_code']} files need migration")
        return results
    
    def execute_migration(self, analysis_results: Dict[str, Any], 
                         dry_run: bool = False) -> Dict[str, Any]:
        """Execute the database migration based on analysis results."""
        if dry_run:
            logger.info("Executing migration in DRY RUN mode...")
        else:
            logger.info("Executing database migration...")
        
        migration_results = {
            'files_migrated': 0,
            'files_failed': 0,
            'total_replacements': 0,
            'high_priority_completed': 0,
            'medium_priority_completed': 0,
            'low_priority_completed': 0,
            'errors': [],
            'success_rate': 0.0
        }
        
        # Process high priority files first
        for priority_list, priority_name in [
            (analysis_results['high_priority'], 'high'),
            (analysis_results['medium_priority'], 'medium'),
            (analysis_results['low_priority'], 'low')
        ]:
            for file_analysis in priority_list:
                file_path = Path(file_analysis['file_path'])
                
                if dry_run:
                    # Simulate migration
                    logger.info(f"[DRY RUN] Would migrate {file_path}")
                    migration_results['files_migrated'] += 1
                else:
                    # Actual migration
                    result = self.migrator.migrate_file(file_path)
                    
                    if result.get('migrated', False):
                        migration_results['files_migrated'] += 1
                        migration_results['total_replacements'] += result.get('patterns_replaced', 0)
                        
                        if priority_name == 'high':
                            migration_results['high_priority_completed'] += 1
                        elif priority_name == 'medium':
                            migration_results['medium_priority_completed'] += 1
                        else:
                            migration_results['low_priority_completed'] += 1
                    else:
                        migration_results['files_failed'] += 1
                        if result.get('error'):
                            migration_results['errors'].append(result['error'])
        
        # Calculate success rate
        total_attempted = migration_results['files_migrated'] + migration_results['files_failed']
        if total_attempted > 0:
            migration_results['success_rate'] = migration_results['files_migrated'] / total_attempted
        
        if not dry_run:
            logger.info(f"Migration complete: {migration_results['files_migrated']} files migrated, "
                       f"{migration_results['files_failed']} failed")
        
        return migration_results
    
    def _calculate_migration_estimate(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate migration effort estimates."""
        estimate = {
            'total_effort_hours': 0,
            'complexity_breakdown': {
                'simple': 0,
                'moderate': 0, 
                'complex': 0
            },
            'recommendations': []
        }
        
        # Effort estimation (simplified)
        for file_group, multiplier in [
            (analysis_results['high_priority'], 2.0),
            (analysis_results['medium_priority'], 1.5),
            (analysis_results['low_priority'], 1.0)
        ]:
            for file_analysis in file_group:
                complexity = file_analysis.get('complexity_score', 0)
                
                if complexity > 20:
                    estimate['complexity_breakdown']['complex'] += 1
                    effort_hours = 4 * multiplier
                elif complexity > 10:
                    estimate['complexity_breakdown']['moderate'] += 1
                    effort_hours = 2 * multiplier
                else:
                    estimate['complexity_breakdown']['simple'] += 1
                    effort_hours = 1 * multiplier
                
                estimate['total_effort_hours'] += effort_hours
        
        # General recommendations
        total_files = sum(estimate['complexity_breakdown'].values())
        if total_files > 50:
            estimate['recommendations'].append(
                "Large migration - consider phased approach starting with high priority files"
            )
        
        if estimate['complexity_breakdown']['complex'] > 5:
            estimate['recommendations'].append(
                "Multiple complex files detected - allocate extra time for testing"
            )
        
        if estimate['total_effort_hours'] > 40:
            estimate['recommendations'].append(
                "Consider dedicated migration sprint to minimize disruption"
            )
        
        return estimate


def run_database_migration_analysis(project_root: str) -> Dict[str, Any]:
    """
    Run complete database migration analysis for AUJ Platform.
    
    Args:
        project_root: Root directory of the project
        
    Returns:
        Complete analysis and migration plan
    """
    orchestrator = DatabaseMigrationOrchestrator(Path(project_root))
    
    # Analyze project
    analysis_results = orchestrator.analyze_project()
    
    # Generate migration plan
    migration_plan = {
        'analysis': analysis_results,
        'recommended_approach': _generate_migration_plan(analysis_results),
        'next_steps': _generate_next_steps(analysis_results)
    }
    
    return migration_plan


def _generate_migration_plan(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate recommended migration approach."""
    total_files = analysis_results['files_with_db_code']
    high_priority = len(analysis_results['high_priority'])
    
    if total_files < 10:
        approach = "single_phase"
        description = "Migrate all files in single phase"
    elif high_priority > 10:
        approach = "priority_based"
        description = "Phase 1: High priority files, Phase 2: Medium/Low priority"
    else:
        approach = "gradual"
        description = "Gradual migration with continuous testing"
    
    return {
        'approach': approach,
        'description': description,
        'estimated_duration_days': min(analysis_results['migration_estimate']['total_effort_hours'] / 8, 10),
        'risk_level': 'HIGH' if high_priority > 20 else 'MEDIUM' if high_priority > 10 else 'LOW'
    }


def _generate_next_steps(analysis_results: Dict[str, Any]) -> List[str]:
    """Generate concrete next steps for migration."""
    steps = [
        "1. Review analysis results and migration plan",
        "2. Create feature branch for database migration",
        "3. Run migration in DRY RUN mode first"
    ]
    
    if len(analysis_results['high_priority']) > 0:
        steps.append("4. Start with high priority files:")
        for file_analysis in analysis_results['high_priority'][:5]:  # Top 5
            steps.append(f"   - {file_analysis['file_path']}")
    
    steps.extend([
        "5. Test each migrated component thoroughly", 
        "6. Update unit tests to use UnifiedDatabaseManager",
        "7. Performance test with new database layer",
        "8. Deploy to staging environment for integration testing"
    ])
    
    return steps